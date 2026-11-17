"""End-to-end evaluation of the deep-learning pipeline on the solved cases (leave-one-case-out planners).

    python -m src.eval.solve_eval [--k 100]

For every solved case, ``solve`` runs with the planner fold that never saw the case; the answer key is used only
afterwards, to measure the share of its identifiers that the VERIFIED findings recovered. The reference is the
earlier deterministic agents (apps + fs + win over **every** file, ``data/outputs/scores/scores_all.json``).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

from src.config import settings


def run(k: int = 100, cases: list[str] | None = None) -> Path:
    from src.ml.planner_dataset import load
    from src.pipeline.solve import solve

    solved = cases or sorted({r["case_id"] for r in load()["rows"]})
    old = {}
    ref = settings.OUTPUTS_DIR / "scores" / "scores_all.json"
    if ref.exists():
        for c in json.loads(ref.read_text(encoding="utf-8")):
            er = c.get("entity_recall") or {}
            if er.get("entities"):
                old[c["case_id"]] = (er.get("found", 0), er["entities"])
    rows = []
    for c in solved:
        out = solve(c, k=k, quiet=True)
        ev = out["evaluation"] or {}
        rows.append({"case": c, "files": out["files"], "k": k, "status": out["status"],
                     "found": ev.get("found", 0), "identifiers": ev.get("key_identifiers", 0),
                     "old_found": old.get(c, (None, None))[0], "planner": Path(out["planner"]).name})
        print(f"[solve-eval] {c}: {rows[-1]}", flush=True)
    tot_new = sum(r["found"] for r in rows)
    tot_id = sum(r["identifiers"] for r in rows)
    tot_old = sum(r["old_found"] or 0 for r in rows)
    lines = [f"# Deep-learning pipeline vs deterministic agents (answer-key identifiers recovered, K = {k})", "",
             "| case | files | read by DL pipeline | identifiers in key | DL pipeline (verified) | deterministic agents "
             "(all files) | status counts | planner |", "|---|---|---|---|---|---|---|---|"]
    lines += [f"| {r['case']} | {r['files']:,} | {min(k, r['files'])} | {r['identifiers']} | {r['found']} | "
              f"{r['old_found'] if r['old_found'] is not None else '-'} | {r['status']} | {r['planner']} |"
              for r in rows]
    lines += ["", f"Pooled: DL pipeline {tot_new} / {tot_id} ({100 * tot_new / max(tot_id, 1):.1f} %), deterministic "
                  f"agents {tot_old} / {tot_id} ({100 * tot_old / max(tot_id, 1):.1f} %)."]
    out_dir = settings.OUTPUTS_DIR / "ml" / "planner"
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"solve_eval_k{k}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    (path.with_suffix(".json")).write_text(json.dumps(rows, indent=1), encoding="utf-8")
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=100)
    a = ap.parse_args()
    print(run(a.k).read_text(encoding="utf-8"))
