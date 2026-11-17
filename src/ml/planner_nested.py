"""Nested leave-one-case-out evaluation of the planner: the pre-registered final model vs logistic regression.

    python -m src.ml.planner_nested run [--candidates a,b] [--seeds 13,14,15] [--out DIR]   # resumable
    python -m src.ml.planner_nested report DIR

Protocol (``docs/PREREGISTRATION.md``, ``src/ml/planner_protocol.py``): for every rankable held-out case, every
candidate is trained once per inner-validation case (the other training cases with >= 10 relevant files, each left out
in turn) and per seed. The (variant, epoch) with the best mean inner-validation AP is selected; the fold's final model
is the mean logit of its inner models at that epoch. The held-out case is never used to choose anything: its scores
are recorded at every epoch only so the selected one can be evaluated afterwards without retraining.

Outputs per held-out case ``<out>/<case>.npz`` (scores: candidate x inner case x seed x epoch) + ``<case>.json``
(inner-validation AP of each); ``report`` applies the selection, the verdict and the user-level secondary metric.
"""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

from src.config import settings
from src.ml import planner_protocol as proto
from src.ml import planner_train as pt

KEYS = ("auroc", "ap", "coverage@10", "coverage@50", "coverage@100")


def eval_cases(data: pt.PlanData) -> list[str]:
    """Held-out cases where ranking is meaningful (the 13 rankable cases)."""
    out = []
    for c in data.cases:
        y = data.y[data.case == c]
        if pt.rankable({"files": len(y), "relevant": int(y.sum())}):
            out.append(c)
    return out


def inner_cases(data: pt.PlanData, pool: list[str]) -> list[str]:
    return [c for c in pool if data.y[data.case == c].sum() >= proto.MIN_VAL_POSITIVES]


def prior_views(data: pt.PlanData, candidates, device: str) -> dict:
    """variant -> PlanData with that variant's LLM prior columns, for the prior variants among ``candidates``."""
    files = {pt.PRIOR_VARIANTS[c] for c in candidates if c in pt.PRIOR_VARIANTS}
    views = {f: data.with_prior(f).cache(device) for f in files}
    return {c: views[pt.PRIOR_VARIANTS[c]] for c in candidates if c in pt.PRIOR_VARIANTS}


def run_fold(data: pt.PlanData, test: str, candidates, seeds, epochs: int, device: str,
             priors: dict | None = None) -> tuple[dict, dict]:
    base = data
    pool = [c for c in data.cases if c != test]
    te = np.flatnonzero(data.case == test)
    scores, val = {}, {}
    known = {**pt.ABLATIONS, **pt.EXPERIMENTAL}
    for name in candidates:
        ex = pt.EXTRA.get(name, {})
        data = (priors or {}).get(name) if name in pt.PRIOR_VARIANTS else base
        if data is None:
            raise SystemExit(f"{name} needs the LLM prior: run `python -m src.ml.llm_prior` and pass --prior")
        for v in inner_cases(data, pool):
            tr = [c for c in pool if c != v]
            va = np.flatnonzero(data.case == v)
            for seed in seeds:
                def record(ep, net, _k=(name, v, seed)):
                    s_va = pt.predict(net, data, va, device)
                    val[(*_k, ep)] = pt.average_precision(data.y[va], s_va)
                    scores[(*_k, ep)] = pt.predict(net, data, te, device).astype(np.float32)

                pt.train_model(data, tr, [v], known[name], epochs=epochs, seed=seed, device=device,
                               anchor=name in pt.ANCHOR_VARIANTS, augment_=ex.get("augment", False),
                               wide=ex.get("wide", False), on_epoch=record)
    return scores, val


def _key(k) -> str:
    return "|".join(map(str, k))


def run(out: Path, candidates=proto.CANDIDATES, seeds=proto.SEEDS, epochs: int = proto.EPOCHS,
        prior: bool = False) -> Path:
    data = pt.PlanData()
    device = settings.device_for("orchestrator")
    data.cache(device)
    if prior:
        candidates = tuple(dict.fromkeys([*candidates, *proto.PRIOR_CANDIDATES]))
    priors = prior_views(data, candidates, device)
    out.mkdir(parents=True, exist_ok=True)
    (out / "config.json").write_text(json.dumps({"candidates": list(candidates), "seeds": list(seeds),
                                                 "epochs": epochs, "dataset": str(data.dir), "prior": prior}, indent=1),
                                     encoding="utf-8")
    t0 = time.perf_counter()
    cases = eval_cases(data)
    for i, test in enumerate(cases, 1):
        if (out / f"{test}.json").exists():
            continue
        scores, val = run_fold(data, test, candidates, seeds, epochs, device, priors)
        np.savez_compressed(out / f"{test}.npz", **{_key(k): s for k, s in scores.items()})
        (out / f"{test}.json").write_text(json.dumps({_key(k): a for k, a in val.items()}, indent=0),
                                          encoding="utf-8")
        print(f"[nested] {i}/{len(cases)} {test}: {len(val)} inner evaluations "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
    return out


def extend(out: Path, candidates=proto.TUNED_CANDIDATES) -> Path:
    """Adds ``candidates`` to a finished (or partly finished) nested run with the run's own seeds and epochs: their
    inner folds are trained and merged into each case's recorded scores / inner AP. Selection then runs over all
    candidates. Resumable (a case that already holds a candidate's keys is skipped for it)."""
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    data = pt.PlanData()
    device = settings.device_for("orchestrator")
    data.cache(device)
    priors = prior_views(data, candidates, device)
    cfg["candidates"] = list(dict.fromkeys([*cfg["candidates"], *candidates]))
    (out / "config.json").write_text(json.dumps(cfg, indent=1), encoding="utf-8")
    t0 = time.perf_counter()
    cases = eval_cases(data)
    for i, test in enumerate(cases, 1):
        jf, zf = out / f"{test}.json", out / f"{test}.npz"
        if not jf.exists():
            continue                                # `run` will train all candidates of config.json for it
        val = json.loads(jf.read_text(encoding="utf-8"))
        todo = [c for c in candidates if not any(k.startswith(f"{c}|") for k in val)]
        if not todo:
            continue
        s_new, v_new = run_fold(data, test, todo, cfg["seeds"], cfg["epochs"], device, priors)
        z = dict(np.load(zf))
        z.update({_key(k): s for k, s in s_new.items()})
        val.update({_key(k): a for k, a in v_new.items()})
        np.savez_compressed(zf, **z)
        jf.write_text(json.dumps(val, indent=0), encoding="utf-8")
        print(f"[nested extend] {i}/{len(cases)} {test}: +{len(v_new)} inner evaluations "
              f"({time.perf_counter() - t0:.0f} s)", flush=True)
    return out


def select(val: dict[str, float]) -> tuple[str, int, list[str]]:
    """(variant, epoch, keys of the inner models) with the best mean inner-validation AP."""
    by: dict[tuple[str, int], list] = {}
    for k, a in val.items():
        name, v, seed, ep = k.split("|")
        if a is not None:
            by.setdefault((name, int(ep)), []).append((a, k))
    (name, ep), lst = max(by.items(), key=lambda kv: (np.mean([a for a, _ in kv[1]]), -kv[0][1]))
    return name, ep, [k for _, k in lst]


def deploy(out: Path, dest: Path | None = None) -> Path:
    """Rebuild every fold's final model (the selected variant's inner models at the selected epoch; deterministic
    retraining stops at that epoch) and save it as an ensemble checkpoint ``<dest>/<case>.pt``. Checks that the
    rebuilt ensemble reproduces the recorded held-out scores."""
    from src.ml.planner import EnsemblePlanner

    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    base = pt.PlanData()
    device = settings.device_for("orchestrator")
    base.cache(device)
    priors = prior_views(base, cfg["candidates"], device)
    dest = dest or out / "models"
    known = {**pt.ABLATIONS, **pt.EXPERIMENTAL}
    for test in eval_cases(base):
        f = out / f"{test}.json"
        if not f.exists() or (dest / f"{test}.pt").exists():
            continue
        name, ep, keys = select(json.loads(f.read_text(encoding="utf-8")))
        data = priors[name] if name in pt.PRIOR_VARIANTS else base
        ex = pt.EXTRA.get(name, {})
        pool = [c for c in base.cases if c != test]
        nets = []
        for k in keys:
            _, v, seed, _ = k.split("|")
            net, _ = pt.train_model(data, [c for c in pool if c != v], [v], known[name], epochs=cfg["epochs"],
                                    seed=int(seed), device=device, anchor=name in pt.ANCHOR_VARIANTS,
                                    augment_=ex.get("augment", False), wide=ex.get("wide", False), stop_at=ep)
            nets.append(net)
        te = np.flatnonzero(base.case == test)
        s = np.mean([pt.predict(n, data, te, device) for n in nets], 0)
        z = np.load(out / f"{test}.npz")
        diff = float(np.abs(s - np.mean([z[k] for k in keys], 0)).max())
        EnsemblePlanner.save_ensemble(nets, dest / f"{test}.pt",
                                      {"variant": name, "epoch": ep, "inner_models": keys, "run": str(out),
                                       "llm_prior": name in pt.PRIOR_VARIANTS,
                                       "prior_file": pt.PRIOR_VARIANTS.get(name), "reproduction_max_abs_diff": diff})
        print(f"[nested deploy] {test}: {name} epoch {ep}, {len(nets)} models, max |score diff| {diff:.2e}", flush=True)
    return dest


def _metrics(data, test, s, mask=None):
    m = data.case == test
    y, items = data.y[m], data.items[test]
    if mask is not None:
        y, s, items = y[mask], s[mask], [it for it, keep in zip(items, mask) if keep]
    return pt.case_metrics(y, s, items)


def report(out: Path) -> Path:
    cfg = json.loads((out / "config.json").read_text(encoding="utf-8"))
    data = pt.PlanData()                     # LR baseline and metrics: metadata only, as pre-registered
    rows = [r for r in data.rows]
    user = np.array([not proto.is_container(r) for r in rows])
    per = {"final": {}, "LR": {}}
    per_u = {"final": {}, "LR": {}}
    chosen = {}
    desc: dict[str, dict] = {}
    for test in eval_cases(data):
        f = out / f"{test}.json"
        if not f.exists():
            continue
        val = json.loads(f.read_text(encoding="utf-8"))
        z = np.load(out / f"{test}.npz")
        name, ep, keys = select(val)
        chosen[test] = {"variant": name, "epoch": ep, "inner_models": len(keys),
                        "inner_val_ap": round(float(np.mean([val[k] for k in keys])), 4)}
        s = np.mean([z[k] for k in keys], 0)
        pool = [c for c in data.cases if c != test]
        te = np.flatnonzero(data.case == test)
        import torch
        lin = pt.fit_logreg(data, pool)
        with torch.no_grad():
            lr = lin(torch.from_numpy(data.x["meta"][te])).squeeze(-1).numpy()
        um = user[te]
        per["final"][test], per["LR"][test] = _metrics(data, test, s), _metrics(data, test, lr)
        per_u["final"][test], per_u["LR"][test] = _metrics(data, test, s, um), _metrics(data, test, lr, um)
        # descriptive only: every candidate at ITS inner-validation-best epoch (not used for any claim)
        for cand in cfg["candidates"]:
            sub = {k: a for k, a in val.items() if k.startswith(cand + "|")}
            if sub:
                _, e2, k2 = select(sub)
                desc.setdefault(cand, {})[test] = _metrics(data, test, np.mean([z[k] for k in k2], 0))
    done = list(per["final"])
    ap = {k: {c: per[k][c]["ap"] for c in done} for k in per}
    v = proto.verdict(ap["final"], ap["LR"])
    ok_u = [c for c in done if pt.rankable(per_u["LR"][c])]
    vu = proto.verdict({c: per_u["final"][c]["ap"] for c in ok_u}, {c: per_u["LR"][c]["ap"] for c in ok_u}) \
        if ok_u else None

    def mean(d, k):
        x = [m[k] for m in d.values() if m.get(k) is not None]
        return f"{np.mean(x):.3f}" if x else "-"

    L = [f"# Nested LOCO: pre-registered final model vs logistic regression", "",
         f"Run `{out.name}`: candidates {cfg['candidates']}, seeds {cfg['seeds']}, {cfg['epochs']} epochs; "
         f"{len(done)} held-out cases. Protocol: docs/PREREGISTRATION.md.", "",
         "## Primary (original labels)", "", "| model | " + " | ".join(KEYS) + " |", "|---|" + "---|" * len(KEYS)]
    for k in ("LR", "final"):
        L.append(f"| {'logistic regression (metadata)' if k == 'LR' else 'final model (nested selection)'} | "
                 + " | ".join(mean(per[k], x) for x in KEYS) + " |")
    L += ["", f"**Verdict:** mean AP gain {v['mean_gain']:+.4f}, 95 % CI {v['ci95']}, wins {v['wins']} / {v['cases']} "
              f"-> criteria {v['criteria']} -> **clearly beats LR: {v['clearly_beats_lr']}**", "",
          "| held-out case | files | relevant | chosen variant | epoch | inner-val AP | AP final | AP LR | delta |",
          "|---|---|---|---|---|---|---|---|---|"]
    for c in done:
        ch = chosen[c]
        L.append(f"| {c} | {per['LR'][c]['files']:,} | {per['LR'][c]['relevant']} | {ch['variant']} | {ch['epoch']} | "
                 f"{ch['inner_val_ap']:.3f} | {ap['final'][c]:.3f} | {ap['LR'][c]:.3f} | {v['per_case_delta'][c]:+.3f} |")
    L += ["", "## Secondary: user-level files only (container files excluded; same scores, no retraining)", ""]
    if vu:
        L += ["| model | " + " | ".join(KEYS) + " |", "|---|" + "---|" * len(KEYS)]
        for k in ("LR", "final"):
            sub = {c: per_u[k][c] for c in ok_u}
            L.append(f"| {'logistic regression (metadata)' if k == 'LR' else 'final model'} | "
                     + " | ".join(mean(sub, x) for x in KEYS) + " |")
        L += ["", f"Same statistics (reported, not used for the claim): mean AP gain {vu['mean_gain']:+.4f}, "
                  f"95 % CI {vu['ci95']}, wins {vu['wins']} / {vu['cases']}."]
    L += ["", "## Descriptive only: each candidate at its own inner-validation-best epoch (NOT a claim)", "",
          "| candidate | " + " | ".join(KEYS) + " |", "|---|" + "---|" * len(KEYS)]
    for cand, d in desc.items():
        L.append(f"| {cand} | " + " | ".join(mean(d, x) for x in KEYS) + " |")
    (out / "verdict.json").write_text(json.dumps({"primary": v, "user_level": vu, "chosen": chosen,
                                                  "per_case": per, "per_case_user": per_u}, indent=1, default=str),
                                      encoding="utf-8")
    p = out / "REPORT.md"
    p.write_text("\n".join(L) + "\n", encoding="utf-8")
    return p


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--candidates", default=",".join(proto.CANDIDATES))
    r.add_argument("--seeds", default=",".join(map(str, proto.SEEDS)))
    r.add_argument("--epochs", type=int, default=proto.EPOCHS)
    r.add_argument("--prior", action="store_true")
    r.add_argument("--out", type=Path)
    q = sub.add_parser("report")
    q.add_argument("out", type=Path)
    dp = sub.add_parser("deploy")
    dp.add_argument("out", type=Path)
    xt = sub.add_parser("extend", help="add candidates (default: the tuned-prior variant) to an existing run")
    xt.add_argument("out", type=Path)
    xt.add_argument("--candidates", default=",".join(proto.TUNED_CANDIDATES))
    a = ap.parse_args()
    if a.cmd == "run":
        o = a.out or settings.OUTPUTS_DIR / "ml" / "planner" / f"nested_{datetime.now():%Y%m%d_%H%M%S}"
        run(o, tuple(x.strip() for x in a.candidates.split(",")), tuple(int(x) for x in a.seeds.split(",")),
            a.epochs, a.prior)
        print(report(o).read_text(encoding="utf-8"))
    elif a.cmd == "extend":
        print(report(extend(a.out, tuple(x.strip() for x in a.candidates.split(",")))).read_text(encoding="utf-8"))
    elif a.cmd == "deploy":
        print(deploy(a.out))
    else:
        print(report(a.out).read_text(encoding="utf-8"))
