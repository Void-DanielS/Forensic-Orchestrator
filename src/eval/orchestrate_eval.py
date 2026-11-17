"""End-to-end evaluation of the orchestrated multi-agent system on the solved cases (pre-registered protocol:
``docs/PREREGISTRATION.md`` section 3).

    python -m src.eval.orchestrate_eval [--k 100] [--batch 25] [--modes full,nofeedback,notool,lr] [--cases a,b]

Every solved case with automatic answer-key identifiers is investigated with the planner fold that never saw it
(mvs_2024 and ngdc_2012 have none and are excluded). Systems, same K:

* ``full``        planner ranking, rounds + blackboard of verified leads + re-ranking, analyst with its tool;
* ``nofeedback``  planner ranking chosen once (no blackboard, no re-ranking);
* ``notool``      full system with the analyst's deterministic identifier-extraction tool off;
* ``lr``          control: top K of the logistic-regression baseline, same agents, no feedback.

Metric: answer-key identifiers in VERIFIED findings, split by source (LLM quotes / tool / image agent / union /
found only by an LLM quote). The answer key is used only afterwards.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

from src.config import settings

MODES = {"full": {}, "nofeedback": {"feedback": False}, "notool": {"tools": False}, "lr": {"ranker": "lr"}}
NO_KEY_IDENTIFIERS = ("mvs_2024", "ngdc_2012")        # labelled through curated tokens only: excluded
SOURCES = ("llm", "tool", "image", "union", "llm_only")


def run(k: int = 100, batch: int = 25, modes: tuple[str, ...] = tuple(MODES), vlm: bool = True,
        cases: list[str] | None = None, out_dir: Path | None = None, planners: Path | None = None) -> Path:
    """``planners``: directory of per-case fold checkpoints (``<case>.pt``, e.g. the nested final models); a case
    without one uses the default LOCO fold."""
    from src.ml.planner_dataset import load
    from src.pipeline.orchestrate import investigate

    solved = cases or sorted({r["case_id"] for r in load()["rows"]} - set(NO_KEY_IDENTIFIERS))
    out_dir = out_dir or settings.OUTPUTS_DIR / "ml" / "orchestrate" / f"eval_k{k}_{datetime.now():%Y%m%d_%H%M%S}"
    out_dir.mkdir(parents=True, exist_ok=True)
    state = out_dir / "results.json"
    rows = json.loads(state.read_text(encoding="utf-8")) if state.exists() else {}      # resumable
    for c in solved:
        for m in modes:
            if f"{c}|{m}" in rows:
                continue
            t0 = time.perf_counter()
            ck = planners / f"{c}.pt" if planners and (planners / f"{c}.pt").exists() else None
            o = investigate(c, k=k, batch=batch, vlm=vlm, quiet=True, checkpoint=ck, **MODES[m])
            ev = o["evaluation"] or {}
            rows[f"{c}|{m}"] = {"case": c, "mode": m, "identifiers": ev.get("key_identifiers", 0),
                                "by_source": ev.get("by_source") or {}, "status": o["status"],
                                "hallucinations": o["hallucinations_caught"], "rounds": len(o["rounds"]),
                                "seconds": round(time.perf_counter() - t0, 1), "run": o["run_id"]}
            state.write_text(json.dumps(rows, indent=1), encoding="utf-8")
            r = rows[f"{c}|{m}"]
            print(f"[orchestrate-eval] {c} {m}: {r['by_source']} ({r['seconds']} s)", flush=True)
    return report(out_dir, modes)


def report(out_dir: Path, modes=tuple(MODES)) -> Path:
    rows = json.loads((out_dir / "results.json").read_text(encoding="utf-8"))
    cases = sorted({r["case"] for r in rows.values()})
    have = [m for m in modes if any(r["mode"] == m for r in rows.values())]
    ids = {r["case"]: r["identifiers"] for r in rows.values()}
    L = ["# Orchestrated multi-agent system: answer-key identifiers in VERIFIED findings", "",
         f"{len(cases)} cases ({', '.join(NO_KEY_IDENTIFIERS)} excluded: no automatic key identifiers). "
         "Columns per system: LLM quotes / tool / image / union / found only by an LLM quote.", "",
         "| case | identifiers | " + " | ".join(have) + " |", "|---|---|" + "---|" * len(have)]
    for c in cases:
        cells = []
        for m in have:
            r = rows.get(f"{c}|{m}")
            cells.append("-" if r is None else " / ".join(str(r["by_source"].get(s, 0)) for s in SOURCES))
        L.append(f"| {c} | {ids[c]} | " + " | ".join(cells) + " |")
    total = sum(ids.values())
    L += ["", "| system | " + " | ".join(SOURCES) + " | cases with a hit (union) | LLM quotes rejected / total | "
          "mean s per case |", "|---|" + "---|" * (len(SOURCES) + 3)]
    for m in have:
        rs = [r for r in rows.values() if r["mode"] == m]
        sums = {s: sum(r["by_source"].get(s, 0) for r in rs) for s in SOURCES}
        q = sum(r["hallucinations"]["llm_quotes"] for r in rs)
        rj = sum(r["hallucinations"]["rejected"] for r in rs)
        L.append(f"| {m} | " + " | ".join(f"{sums[s]} ({100 * sums[s] / max(total, 1):.1f} %)" for s in SOURCES)
                 + f" | {sum(1 for r in rs if r['by_source'].get('union', 0) > 0)} / {len(rs)} | {rj} / {q} | "
                   f"{sum(r['seconds'] for r in rs) / max(len(rs), 1):.0f} |")
    L += ["", f"Total answer-key identifiers: {total}. Rejected LLM quotes are hallucinations caught by the "
              "verifier; they never become evidence."]
    p = out_dir / "REPORT.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=100)
    ap.add_argument("--batch", type=int, default=25)
    ap.add_argument("--modes", default=",".join(MODES))
    ap.add_argument("--cases")
    ap.add_argument("--no-vlm", action="store_true")
    ap.add_argument("--out", type=Path)
    ap.add_argument("--planners", type=Path, help="per-case planner checkpoints (e.g. <nested run>/models)")
    a = ap.parse_args()
    print(run(a.k, a.batch, tuple(a.modes.split(",")), vlm=not a.no_vlm,
              cases=a.cases.split(",") if a.cases else None, out_dir=a.out,
              planners=a.planners).read_text(encoding="utf-8"))
