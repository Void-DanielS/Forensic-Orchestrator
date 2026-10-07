"""BaseAgent: the contract every specialist agent follows.

``analyze_artifact()`` produces candidate findings (each with citations) for one artifact;
``run()`` passes them, artifact by artifact, through the HeuristicVerificationEngine,
contrasts verified findings with the case context, records the links in the investigation
memory, and emits every step to the run trace (read live by the GUI). The report separates
verified evidence from rejected claims and non-evidentiary model observations.
"""
from __future__ import annotations

import json
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from src.agents.trace import NullTracer
from src.bridge.forensic_bridge import ForensicBridge
from src.config import settings
from src.verification.heuristic_verifier import HeuristicVerificationEngine
from src.verification.models import Citation, Finding, FindingStatus


@dataclass
class AgentReport:
    agent: str
    device: str
    artifacts_analyzed: int
    findings: list[Finding]
    runtime_s: float
    resources: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)
    evidence_id: str | None = None
    run_id: str | None = None

    def by_status(self, status: FindingStatus) -> list[Finding]:
        return [f for f in self.findings if f.status == status]

    def to_dict(self) -> dict[str, Any]:
        counts = {s.value: len(self.by_status(s)) for s in FindingStatus}
        return {
            "agent": self.agent,
            "evidence_id": self.evidence_id,
            "run_id": self.run_id,
            "device": self.device,
            "artifacts_analyzed": self.artifacts_analyzed,
            "runtime_s": round(self.runtime_s, 2),
            "status_counts": counts,
            "resources": self.resources,
            "errors": self.errors,
            "verified_findings": [f.model_dump() for f in self.by_status(FindingStatus.VERIFIED)],
            "rejected_findings": [f.model_dump() for f in self.by_status(FindingStatus.REJECTED)],
            "model_observations_uncorroborated": [f.model_dump() for f in self.by_status(FindingStatus.UNCORROBORATED)],
        }


def _citation_brief(c: Citation) -> dict[str, Any]:
    loc = {"byte_offset": c.byte_offset, "line_number": c.line_number, "sql_query": c.sql_query,
           "regex": c.regex, "exif_tag": c.exif_tag}[c.locator_type]
    return {"file": c.file_path, "locator_type": c.locator_type, "locator": loc, "params": c.sql_params or None,
            "snippet": c.raw_text_snippet[:200], "sha256": c.expected_hash}


class BaseAgent(ABC):
    name: str = "BaseAgent"
    role: settings.Role = "text"

    def __init__(self, bridge: ForensicBridge, verifier: HeuristicVerificationEngine | None = None,
                 tracer=None, context=None, memory=None, evidence_id: str | None = None, run_id: str | None = None):
        self.bridge = bridge
        self.sandbox = bridge.sandbox
        self.verifier = verifier or HeuristicVerificationEngine(bridge.sandbox.root, bridge=bridge)
        self.device = settings.device_for(self.role)
        self.errors: list[str] = []
        self.tracer = tracer or NullTracer()
        self.context = context          # CaseContext (brief only) or None
        self.memory = memory            # InvestigationMemory or None
        self.evidence_id = evidence_id
        self.run_id = run_id

    # -------------------------------------------------------------- to implement
    @abstractmethod
    def analyze_artifact(self, rel: str) -> list[Finding]:
        """Candidate findings for one sandbox-relative artifact."""

    def prepare(self, artifacts: Sequence[str]) -> None:
        """Called once before the first artifact (load models, build indexes)."""

    def release(self) -> None:
        """Free model memory (override when the agent holds GPU models)."""

    def analyze(self, artifacts: Sequence[str]) -> list[Finding]:
        self.prepare(artifacts)
        out: list[Finding] = []
        for rel in artifacts:
            out += self.analyze_artifact(rel)
        return out

    # -------------------------------------------------------------- helpers
    def cite(self, rel: str, locator_type: str, snippet: str, **locator: Any) -> Citation:
        return Citation(file_path=rel, locator_type=locator_type, expected_hash=self.sandbox.hash(rel),
                        raw_text_snippet=snippet, **locator)

    def finding(self, artifact: str, category: str, claim: str, **kw: Any) -> Finding:
        f = Finding(agent=self.name, artifact_path=artifact, category=category, claim=claim,
                    evidence_id=self.evidence_id, **kw)
        self.tracer.emit("finding", artifact=artifact, finding_id=f.finding_id, category=category, claim=claim,
                         observed_text=f.observed_text, citations=[_citation_brief(c) for c in f.citations],
                         model=f.model)
        return f

    def enforce_vram_budget(self) -> None:
        """Cap this process's allocations on the agent's GPU at the configured budget."""
        if not self.device.startswith("cuda"):
            return
        import torch

        idx = torch.device(self.device).index or 0
        total_gb = torch.cuda.get_device_properties(idx).total_memory / 1024**3
        budget = settings.gpu_for(self.role).vram_budget_gb
        if budget < total_gb:
            torch.cuda.set_per_process_memory_fraction(budget / total_gb, idx)

    def gpu_stats(self) -> dict[str, Any]:
        if not self.device.startswith("cuda"):
            return {"device": self.device}
        import torch

        idx = torch.device(self.device).index or 0
        return {
            "device": self.device,
            "gpu_name": torch.cuda.get_device_name(idx),
            "vram_budget_gb": settings.gpu_for(self.role).vram_budget_gb,
            "peak_allocated_gb": round(torch.cuda.max_memory_allocated(idx) / 1024**3, 2),
            "peak_reserved_gb": round(torch.cuda.max_memory_reserved(idx) / 1024**3, 2),
        }

    # -------------------------------------------------------------- verification + context
    def _verify_and_contrast(self, findings: list[Finding]) -> None:
        for f in findings:
            self.verifier.verify_finding(f)
            self.tracer.emit("verification", artifact=f.artifact_path, finding_id=f.finding_id, category=f.category,
                             status=f.status.value,
                             checks=[{"verified": v.get("verified"), "reason": v.get("reason"),
                                      "locator_type": v.get("locator_type"), "file": v.get("file_path"),
                                      "observed": (v.get("observed_snippet") or "")[:160]} for v in f.verification])
            if self.context is None:
                continue
            from src.context.case_context import contrast

            f.context_links = contrast(self.context, f)
            for link in f.context_links:
                self.tracer.emit("context_link", artifact=f.artifact_path, finding_id=f.finding_id,
                                 category=f.category, status=f.status.value, **link)
            if self.memory is not None and f.context_links:
                self.memory.record(self.run_id or "", self.evidence_id or "", f, f.context_links)

    # -------------------------------------------------------------- pipeline
    def run(self, artifacts: Sequence[str], output_dir: Path | None = None) -> AgentReport:
        start = time.perf_counter()
        self.tracer.emit("evidence_start", evidence_id=self.evidence_id, agent=self.name, device=self.device,
                         artifacts=len(artifacts))
        self.prepare(artifacts)
        findings: list[Finding] = []
        for i, rel in enumerate(artifacts, 1):
            t0 = time.perf_counter()
            self.tracer.emit("artifact_start", artifact=rel, index=i, total=len(artifacts))
            try:
                batch = self.analyze_artifact(rel)
            except Exception as exc:  # noqa: BLE001 - one bad artifact must not stop the run
                self.errors.append(f"{rel}: {type(exc).__name__}: {exc}")
                self.tracer.emit("error", artifact=rel, error=self.errors[-1])
                batch = []
            self._verify_and_contrast(batch)
            findings += batch
            counts = {s.value: sum(1 for f in batch if f.status == s) for s in FindingStatus if s != FindingStatus.PENDING}
            self.tracer.emit("artifact_end", artifact=rel, seconds=round(time.perf_counter() - t0, 2), counts=counts)
            if self.memory is not None:
                self.memory.save()
                self.tracer.emit("memory", **self.memory.summary())
        report = AgentReport(self.name, self.device, len(artifacts), findings, time.perf_counter() - start,
                             resources=self.gpu_stats(), errors=list(self.errors), evidence_id=self.evidence_id,
                             run_id=self.run_id)
        self.tracer.emit("evidence_end", evidence_id=self.evidence_id, status_counts=report.to_dict()["status_counts"],
                         runtime_s=round(report.runtime_s, 1), resources=report.resources)
        if output_dir:
            output_dir.mkdir(parents=True, exist_ok=True)
            stem = self.name.lower() if not self.evidence_id else f"{self.name.lower()}__{self.evidence_id}"
            (output_dir / f"{stem}_findings.json").write_text(
                json.dumps(report.to_dict(), indent=2, default=str, ensure_ascii=False), encoding="utf-8")
            self.verifier.dump_audit_log(output_dir / f"{stem}_verification_audit.json")
        return report
