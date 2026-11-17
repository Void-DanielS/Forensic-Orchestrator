"""The deep-learning investigation pipeline: planner -> specialist agents -> verifier -> report.

    python main.py solve --case <id> [--k 100] [--vlm]

1. **Planner** (``src/ml/planner.py``, trained on solved cases): scores every file of the case given the case brief
   and keeps the top-K. For a case that has an answer key, the leave-one-case-out model of that case is used, so the
   planner has never seen the case it investigates.
2. **Dispatch** ("what to give to the other agents"): images -> ImageTriageAgent (Jev-MM: pixels + metadata + OCR
   text) and, with ``--vlm``, the MultimodalVisionAgent (Qwen2.5-VL + OCR + EXIF); every other file -> ReaderAgent.
3. **Safeguard**: every agent statement passes the HeuristicVerificationEngine. Model outputs without citations stay
   UNCORROBORATED leads; cited statements are VERIFIED or REJECTED by re-reading the evidence.
4. **Report** ``workspace/reports/<run>_solve.md``: the plan (ranked files, probability, which modality the planner
   attended to, where each file was sent), verified evidence, model leads and, *after* the run and only for
   evaluation, the share of the answer key's identifiers that the verified findings recovered.
"""
from __future__ import annotations

import json
import os
import time
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np

from src.cases.registry import CASES
from src.ml.mm_dataset import IMAGE_CATEGORIES, IMAGE_EXTS

DEFAULT_K = 100


def planner_for(case_id: str, checkpoint: Path | None = None):
    from src.ml.planner import Planner, default_checkpoint

    if checkpoint:
        return Planner.load(checkpoint), str(checkpoint)
    fold = default_checkpoint().parent / "loco" / f"{case_id}.pt"
    path = fold if fold.exists() else default_checkpoint()
    return Planner.load(path), str(path)


def plan(case, planner, quiet: bool = False) -> tuple[list[dict], np.ndarray, np.ndarray]:
    """Rows of the case (inputs only, no labels), planner probabilities and modality attention."""
    from src.ml.mm_dataset import TextEncoder
    from src.ml.planner import features
    from src.ml.planner_dataset import brief_text, case_rows

    t0 = time.perf_counter()
    rows, head, meta, _ = case_rows(case)
    if not rows:
        raise SystemExit(f"case {case.case_id} has no extracted files")
    enc = TextEncoder()
    snip = enc.encode([r["snippet"] for r in rows], batch=128)
    brief = enc.encode([brief_text(case)[:4000]])
    x = features(rows, head, meta, snip, np.repeat(brief, len(rows), 0))
    sc = planner.score(x, attn=True)
    if not quiet:
        print(f"[solve] planner scored {len(rows):,} files in {time.perf_counter() - t0:.1f} s", flush=True)
    return rows, sc["p"], sc["attention"]


def is_image(r: dict) -> bool:
    return r["category"] in IMAGE_CATEGORIES or r["path"].lower().endswith(IMAGE_EXTS)


def solve(case_id: str, k: int = DEFAULT_K, checkpoint: Path | None = None, vlm: bool = False,
          quiet: bool = False) -> dict:
    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from src.agents.mm_agent import ImageTriageAgent
    from src.agents.reader_agent import ReaderAgent
    from src.agents.trace import Tracer
    from src.bridge.forensic_bridge import ForensicBridge
    from src.ml.planner import MODALITIES

    case = CASES[case_id]
    planner, ck = planner_for(case_id, checkpoint)
    rows, p, att = plan(case, planner, quiet)
    order = np.argsort(-p, kind="stable")[:k]
    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + "-solve"
    run_dir = case.dir / "workspace" / "runs" / run_id
    tracer = Tracer(run_dir, {"run_id": run_id, "case_id": case_id, "k": k, "planner": ck})
    the_plan = []
    for rank, i in enumerate(order, 1):
        r = rows[i]
        the_plan.append({"rank": rank, "evidence_id": r["evidence_id"], "path": r["path"], "name": r["name"],
                         "category": r["category"], "p_relevant": round(float(p[i]), 4),
                         "attention": {m: round(float(a), 3) for m, a in zip(MODALITIES, att[i])},
                         "dispatch": "image" if is_image(r) else "reader"})
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plan.json").write_text(json.dumps(the_plan, indent=1, ensure_ascii=False), encoding="utf-8")
    tracer.emit("plan", files=len(rows), k=k, dispatch=dict(Counter(x["dispatch"] for x in the_plan)))

    by_ev: dict[str, dict[str, list[str]]] = defaultdict(lambda: {"image": [], "reader": []})
    for x in the_plan:
        rel = x["path"][3:] if x["path"].startswith("fs/") else x["path"]
        by_ev[x["evidence_id"]][x["dispatch"]].append(rel)
    reports = []
    for eid, work in by_ev.items():
        ev_dir = case.dir / "evidence" / eid
        manifest = json.loads((ev_dir / "MANIFEST.json").read_text(encoding="utf-8"))
        names = {f["path"][3:]: (f.get("original_path") or f["path"]) for f in manifest["files"]
                 if f["path"].startswith("fs/")}
        bridge = ForensicBridge(ev_dir / "fs")
        try:
            if work["reader"]:
                ag = ReaderAgent(bridge, names=names, tracer=tracer, evidence_id=eid, run_id=run_id)
                try:
                    reports.append(ag.run(work["reader"], output_dir=run_dir))
                finally:
                    ag.close()
            if work["image"]:
                ag = ImageTriageAgent(bridge, manifest=manifest, tracer=tracer, evidence_id=eid, run_id=run_id)
                try:
                    reports.append(ag.run(work["image"], output_dir=run_dir))
                finally:
                    ag.release()
                    ag.close()
                if vlm:
                    from src.agents.vision_agent import MultimodalVisionAgent

                    va = MultimodalVisionAgent(bridge, tracer=tracer, evidence_id=eid, run_id=run_id)
                    try:
                        reports.append(va.run(work["image"], output_dir=run_dir))
                    finally:
                        va.release()
                        va.close()
        finally:
            bridge.close()
    findings = [f for rep in reports for f in rep.findings]
    status = Counter(f.status.value for f in findings)
    verified = [f.model_dump(mode="json") for f in findings if f.status.value == "verified"]
    evaluation = evaluate_against_key(case, verified)
    tracer.finish("done", counts=dict(status), evaluation=evaluation)
    report = write_report(case, run_id, ck, len(rows), the_plan, findings, status, evaluation)
    out = {"run_id": run_id, "files": len(rows), "k": k, "status": dict(status), "evaluation": evaluation,
           "report": str(report), "planner": ck}
    (run_dir / "solve.json").write_text(json.dumps(out, indent=1, default=str), encoding="utf-8")
    if not quiet:
        print(f"[solve] {case_id}: {dict(status)}; evaluation {evaluation}; report {report}", flush=True)
    return out


def evaluate_against_key(case, verified: list[dict]) -> dict | None:
    """Evaluation only (after the run): share of the answer key's identifiers present in verified findings."""
    from src.eval.scoring import entity_recall, key_entities

    ents = key_entities(case.dir, case.summary)
    if not ents:
        return None
    r = entity_recall(ents, verified)
    return {"key_identifiers": r["entities"], "found": r.get("found", 0), "recall_pct": r.get("recall_pct", 0.0),
            "found_examples": r.get("found_examples", [])[:10]}


def write_report(case, run_id, ck, n_files, the_plan, findings, status, evaluation) -> Path:
    from src.ml.planner import MODALITIES

    ver = [f for f in findings if f.status.value == "verified"]
    leads = [f for f in findings if f.status.value == "uncorroborated"]
    rej = [f for f in findings if f.status.value == "rejected"]
    plan_rows = "\n".join(
        f"| {x['rank']} | `{x['name'][-70:]}` | {x['category']} | {x['p_relevant']:.3f} | "
        f"{max(x['attention'], key=x['attention'].get)} | {x['dispatch']} |" for x in the_plan[:40])
    ev_rows = "\n".join(f"| {f.finding_id[:8]} | {f.claim[:140]} | `{f.citations[0].file_path[-50:]}` @ "
                        f"{f.citations[0].byte_offset} |" for f in ver[:60])
    lead_rows = "\n".join(f"* {f.claim[:160]}" for f in leads[:20])
    ev_line = ("no answer key for this case" if not evaluation else
               f"{evaluation['found']} of {evaluation['key_identifiers']} answer-key identifiers "
               f"({evaluation['recall_pct']} %) appear in VERIFIED findings")
    md = f"""# Deep-learning investigation: {case.case_id}

Run `{run_id}` · planner `{Path(ck).name}` · {n_files:,} files scored, top {len(the_plan)} dispatched
({dict(Counter(x['dispatch'] for x in the_plan))}) · findings {dict(status)}.

**Evaluation (computed after the run; the answer key is never an input):** {ev_line}.

## 1. The plan: where the planner chose to look

| rank | file | category | P(relevant) | most-attended modality | sent to |
|---|---|---|---|---|---|
{plan_rows}

Modalities the planner reads: {', '.join(MODALITIES)}.

## 2. Verified evidence (re-read from the evidence files by the HeuristicVerificationEngine)

| finding | statement | citation (file @ byte offset) |
|---|---|---|
{ev_rows or '| | none | |'}

## 3. Model leads (not evidence)

{lead_rows or 'none'}

## 4. Rejected statements

{len(rej)} (a rejected statement is a hallucination or tampering signal and is excluded from §2).
"""
    path = case.dir / "workspace" / "reports" / f"{run_id}_solve.md"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(md, encoding="utf-8")
    return path
