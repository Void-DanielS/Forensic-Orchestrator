"""Seed study for the planner's key variants (LOCO repeated with several seeds; nothing deployed or overwritten).

    python -m src.ml.planner_seeds --seeds 13,14,15

Single LOCO runs vary noticeably between seeds (and between GPU runs of one seed), so claims about pretraining are
made on the mean over seeds of the per-run case means."""
from __future__ import annotations

import argparse
import json
from datetime import datetime

import numpy as np

from src.config import settings
from src.ml import planner_train as pt
from src.ml.planner_pretrain import default_encoders

VARIANTS = ("metadata", "all - context", "all - context + SSL", "all - context + SSL frozen", "metadata + SSL",
            "anchored", "anchored + SSL frozen")
KEYS = ("auroc", "ap", "coverage@10", "coverage@50", "coverage@100")


def ensemble(data: "pt.PlanData", scores: dict, variants) -> dict:
    """Per fold, average the seed models' probabilities (a seed ensemble), then the usual case metrics."""
    out = {}
    first = next(iter(scores.values()))
    for v in variants:
        res = {}
        for (name, case) in first:
            if name != v:
                continue
            s = np.mean([scores[k][(name, case)] for k in scores], 0)
            y = data.y[data.case == case]
            res[case] = pt.case_metrics(y, s, data.items[case])
        out[f"{v} (seed ensemble)"] = pt.summarize({v: res})[v]
    return out


def main(seeds: list[int], epochs: int = 10, variants: tuple[str, ...] = VARIANTS) -> None:
    data = pt.PlanData()
    known = {**pt.ABLATIONS, **pt.EXPERIMENTAL}
    abl = {k: known[k] for k in variants}
    runs, scores = {}, {}
    for s in seeds:
        scores[s] = {}
        res = pt.loco(data, abl, epochs, s, progress=True, encoders=default_encoders(), scores=scores[s])
        runs[s] = pt.summarize(res)
    ens = ensemble(data, scores, variants)
    out = settings.OUTPUTS_DIR / "ml" / "planner" / f"seeds_{datetime.now():%Y%m%d_%H%M%S}"
    out.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out / "scores.npz", **{f"{s}|{n}|{c}": v for s in scores for (n, c), v in scores[s].items()})
    names = ["random", "category prior", "logistic regression (metadata)", *variants]
    table = {n: {k: [runs[s][n][k]["mean"] for s in seeds] for k in KEYS} for n in names}
    (out / "seeds.json").write_text(json.dumps({"seeds": seeds, "runs": runs, "table": table, "ensemble": ens},
                                               indent=1),
                                    encoding="utf-8")
    lines = [f"# Planner seed study (seeds {seeds}, variants {list(variants)}; mean ± std over seeds of the LOCO case means)", "",
             "| model | " + " | ".join(KEYS) + " |", "|---|" + "---|" * len(KEYS)]
    for n in names:
        lines.append(f"| {n} | " + " | ".join(f"{np.mean(v):.3f} ± {np.std(v):.3f}" for v in table[n].values()) + " |")
    lines += ["", f"Ensembles: probabilities averaged over the {len(seeds)} seed models of each fold (one number each).",
              "", "| model | " + " | ".join(KEYS) + " |", "|---|" + "---|" * len(KEYS)]
    for n, m in ens.items():
        lines.append(f"| {n} | " + " | ".join(f"{m[k]['mean']:.3f}" if m[k]["mean"] is not None else "-"
                                               for k in KEYS) + " |")
    (out / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="13,14,15")
    ap.add_argument("--variants", default=",".join(VARIANTS),
                    help="comma-separated names from ABLATIONS or EXPERIMENTAL (e.g. 'anchored,anchored + aug')")
    a = ap.parse_args()
    main([int(x) for x in a.seeds.split(",")], variants=tuple(v.strip() for v in a.variants.split(",")))
