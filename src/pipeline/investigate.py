"""Run the MultimodalVisionAgent over a case: every (or selected) evidence item, one shared model.

    python main.py investigate --case m57_patents --limit 8
    python main.py investigate --case hickman_android_10 --evidence android_10 --images a.jpg,b.png

Per run, in ``data/cases/<case>/workspace/``::

    runs/<run_id>/run.json                                 status, parameters, totals
    runs/<run_id>/trace.jsonl                              every agent step (live in the GUI)
    runs/<run_id>/multimodalvisionagent__<ev>_findings.json
    runs/<run_id>/multimodalvisionagent__<ev>_verification_audit.json
    reports/<run_id>_vision.md                             report: evidence, context contrast, observations
    memory.json                                            investigation memory (context + links), cumulative
"""
from __future__ import annotations

import json
import time
from pathlib import Path

from src.agents.trace import Tracer, new_run_id
from src.bridge.forensic_bridge import ForensicBridge
from src.cases.registry import CASES, Case
from src.config import settings
from src.context.case_context import CaseContext, InvestigationMemory


def fetched_evidence(case: Case) -> list[str]:
    return [e for e in case.evidence_ids() if (case.dir / "evidence" / e / "MANIFEST.json").exists()]


def investigate(case_id: str, evidence: list[str] | None = None, limit: int = 0, images: list[str] | None = None,
                ocr: bool = True, vlm: bool = True, run_id: str | None = None) -> dict:
    from src.agents.vision_agent import MultimodalVisionAgent, rank_images
    from src.bridge.text_index import CaseTextIndex
    from src.reports.case_report import write_vision_report

    case = CASES[case_id]
    items = evidence or fetched_evidence(case)
    if not items:
        raise SystemExit(f"case {case_id} has no fetched evidence; run: python main.py cases build --cases {case_id}")
    cfg = settings.VisionSettings()
    cfg.enable_ocr, cfg.enable_vlm = ocr, vlm
    run_id = run_id or new_run_id("vision")
    ws = case.dir / "workspace"
    run_dir = ws / "runs" / run_id
    tracer = Tracer(run_dir, {"run_id": run_id, "case_id": case_id, "agent": "MultimodalVisionAgent",
                              "evidence": items, "limit": limit, "ocr": ocr, "vlm": vlm,
                              "vlm_model": cfg.vlm_model_id if vlm else None, "device": settings.device_for("vision")})
    ctx = CaseContext.load(case.dir, case.summary)
    memory = InvestigationMemory(case.dir, ctx)
    memory.save()
    tracer.emit("run_start", case_id=case_id, title=case.title, evidence=items,
                context={"brief_docs": ctx.docs, "entity_counts": ctx.to_dict()["entity_counts"],
                         "timeframe": ctx.timeframe})
    reports, totals, t0 = [], {"verified": 0, "rejected": 0, "uncorroborated": 0}, time.perf_counter()
    agent = None
    try:
        for eid in items:
            ev_dir = case.dir / "evidence" / eid
            manifest = json.loads((ev_dir / "MANIFEST.json").read_text(encoding="utf-8"))
            bridge = ForensicBridge(ev_dir / "fs")
            imgs = images or rank_images(bridge, manifest, cfg)
            if limit:
                imgs = imgs[:limit]
            if not imgs:
                tracer.emit("evidence_skipped", evidence_id=eid, reason="no images")
                bridge.close()
                continue
            index = CaseTextIndex(bridge).build() if (ocr or vlm) else None
            tracer.emit("index_built", evidence_id=eid, **(index.stats() if index else {}))
            agent = MultimodalVisionAgent(bridge, text_index=index, cfg=cfg, tracer=tracer, context=ctx, memory=memory,
                                          evidence_id=eid, run_id=run_id)
            rep = agent.run(imgs, output_dir=run_dir)
            agent.release(keep_models=True)
            agent.close()
            bridge.close()
            d = rep.to_dict()
            for k in totals:
                totals[k] += d["status_counts"][k]
            reports.append(d)
            memory.add_run({"run_id": run_id, "evidence_id": eid, "agent": rep.agent, "artifacts": rep.artifacts_analyzed,
                            "status_counts": d["status_counts"]})
            memory.save()
        report_path = write_vision_report(case, ctx, memory, run_id, reports, run_dir)
        tracer.emit("run_end", totals=totals, report=str(report_path.relative_to(case.dir)),
                    seconds=round(time.perf_counter() - t0, 1))
        tracer.finish("done", totals=totals, report=str(report_path.relative_to(case.dir)),
                      seconds=round(time.perf_counter() - t0, 1))
        return {"run_id": run_id, "totals": totals, "report": str(report_path)}
    except BaseException as exc:
        tracer.emit("error", error=f"{type(exc).__name__}: {exc}")
        tracer.finish("failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if agent is not None:
            agent.release()


AGENT_RUNNERS = {
    "apps": ("AppArtifactAgent", "src.agents.app_artifact_agent"),
    "fs": ("FileSystemAgent", "src.agents.filesystem_agent"),
    "win": ("WindowsArtifactAgent", "src.agents.windows_agent"),
    "jev": ("JevAgent", "src.agents.jev_agent"),        # trained model: leads only (UNCORROBORATED)
    "mm": ("ImageTriageAgent", "src.agents.mm_agent"),  # trained multimodal model: leads only (UNCORROBORATED)
}


def investigate_agents(case_id: str, agents: list[str], evidence: list[str] | None = None, limit: int = 0,
                       carve: bool = True, run_id: str | None = None, quiet: bool = False) -> dict:
    """Run the deterministic agents (``apps``, ``fs``) over a case: findings JSON per agent and evidence item,
    a shared trace (live in the GUI), context links in the investigation memory and one Markdown report."""
    import importlib

    from src.reports.case_report import write_agents_report

    case = CASES[case_id]
    items = evidence or fetched_evidence(case)
    if not items:
        raise SystemExit(f"case {case_id} has no fetched evidence; run: python main.py cases build --cases {case_id}")
    run_id = run_id or new_run_id("-".join(agents))
    run_dir = case.dir / "workspace" / "runs" / run_id
    tracer = Tracer(run_dir, {"run_id": run_id, "case_id": case_id, "agents": agents, "evidence": items,
                              "limit": limit, "carve": carve})
    ctx = CaseContext.load(case.dir, case.summary)
    memory = InvestigationMemory(case.dir, ctx)
    tracer.emit("run_start", case_id=case_id, title=case.title, evidence=items, agents=agents)
    reports, totals, t0 = [], {"verified": 0, "rejected": 0, "uncorroborated": 0}, time.perf_counter()
    try:
        for eid in items:
            for key in agents:
                cls_name, module = AGENT_RUNNERS[key]
                runner = importlib.import_module(module).run_on_evidence
                kw = {"carve": carve} if key == "fs" else {}
                rep = runner(case, eid, tracer=tracer, context=ctx, memory=memory, run_id=run_id, output_dir=run_dir,
                             limit=limit, **kw)
                if rep is None:
                    tracer.emit("evidence_skipped", evidence_id=eid, agent=cls_name, reason="nothing to analyse")
                    continue
                d = rep.to_dict()
                for k in totals:
                    totals[k] += d["status_counts"][k]
                reports.append(d)
                memory.add_run({"run_id": run_id, "evidence_id": eid, "agent": rep.agent,
                                "artifacts": rep.artifacts_analyzed, "status_counts": d["status_counts"]})
                memory.save()
                if not quiet:
                    print(f"  [{cls_name}] {eid}: {d['status_counts']} in {d['runtime_s']} s", flush=True)
        report_path = write_agents_report(case, ctx, memory, run_id, reports, run_dir)
        tracer.finish("done", totals=totals, report=str(report_path.relative_to(case.dir)),
                      seconds=round(time.perf_counter() - t0, 1))
        return {"run_id": run_id, "totals": totals, "report": str(report_path), "reports": reports}
    except BaseException as exc:
        tracer.emit("error", error=f"{type(exc).__name__}: {exc}")
        tracer.finish("failed", error=f"{type(exc).__name__}: {exc}")
        raise
