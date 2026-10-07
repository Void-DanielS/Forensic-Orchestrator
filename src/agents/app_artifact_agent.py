"""AppArtifactAgent (Sprint 2): structured records from app and system databases.

For every SQLite database whose path matches a parser spec (``src/agents/parsers``), each row
becomes one finding whose citations are the exact cells it quotes (read-only SQL by rowid). The
HeuristicVerificationEngine re-executes every citation, so a finding is VERIFIED only when every
quoted value is in the evidence at that row. Times are normalised to UTC in ``value.timestamps``
(raw value and epoch kept, FDX §2.3). No model is involved: this agent is deterministic.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Sequence

from src.agents.base_agent import BaseAgent
from src.agents.parsers import ALL_SPECS, TableSpec, detect, parse_table
from src.verification.models import Finding


class AppArtifactAgent(BaseAgent):
    name = "AppArtifactAgent"
    role = "text"

    def __init__(self, bridge, verifier=None, specs: Sequence[TableSpec] | None = None, **kw):
        super().__init__(bridge, verifier, **kw)
        self.specs = list(specs or ALL_SPECS)
        self._matches: dict[str, list[TableSpec]] = {}

    # ------------------------------------------------------------------ discovery
    def discover(self, manifest: dict | None = None) -> list[str]:
        """Sandbox paths of the databases this agent can parse (from MANIFEST.json, else a walk)."""
        if manifest is not None:
            paths = [re.sub(r"^fs/", "", f["path"]) for f in manifest.get("files", [])]
        else:
            paths = [e.path for e in self.bridge.fs.walk()]
        candidates = [p for p in paths if not p.endswith(("-wal", "-shm", "-journal"))]
        self._matches = {p: s for p, s in detect(candidates, self.specs).items()
                         if self.bridge.sqlite.is_sqlite(self.sandbox.resolve(p))}
        return sorted(self._matches)

    def prepare(self, artifacts: Sequence[str]) -> None:
        if not self._matches:
            self._matches = detect(list(artifacts), self.specs)

    # ------------------------------------------------------------------ analysis
    def analyze_artifact(self, rel: str) -> list[Finding]:
        specs = self._matches.get(rel) or [s for s in self.specs if s.matches(rel)]
        try:
            tables = self.bridge.sqlite.tables(rel)
        except Exception as exc:  # noqa: BLE001 - corrupt / encrypted database
            self.tracer.emit("parser_skip", artifact=rel, reason=f"{type(exc).__name__}: {exc}")
            return []
        out: list[Finding] = []
        for spec in specs:
            parsed = parse_table(self.bridge, rel, spec, tables)
            self.tracer.emit("parser", artifact=rel, category=spec.category, parser=spec.description,
                             records=len(parsed), truncated=len(parsed) >= spec.max_rows)
            for p in parsed:
                out.append(self.finding(rel, spec.category, p.claim, value=p.value,
                                        citations=[r.to_citation() for r in p.records]))
        return out


def run_on_evidence(case, evidence_id: str, tracer=None, context=None, memory=None, run_id=None,
                    output_dir: Path | None = None, limit: int = 0):
    """AppArtifactAgent over one evidence item of a case; returns the AgentReport (or None if nothing to parse)."""
    from src.bridge.forensic_bridge import ForensicBridge

    ev_dir = case.dir / "evidence" / evidence_id
    manifest = json.loads((ev_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    bridge = ForensicBridge(ev_dir / "fs")
    try:
        agent = AppArtifactAgent(bridge, tracer=tracer, context=context, memory=memory, evidence_id=evidence_id,
                                 run_id=run_id)
        dbs = agent.discover(manifest)
        if limit:
            dbs = dbs[:limit]
        if not dbs:
            return None
        return agent.run(dbs, output_dir=output_dir)
    finally:
        bridge.close()
