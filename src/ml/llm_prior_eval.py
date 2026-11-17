"""LOCO evaluation of the zero-shot LLM relevance prior (run ``python -m src.ml.llm_prior`` first).

    python -m src.ml.llm_prior_eval

Compares logistic regression on metadata (the baseline), the LLM prior alone (zero-shot, no training) and logistic
regression on metadata + LLM prior (raw and per-case z-scored; the weights are fitted on the training cases of each
fold only, so the held-out case is never used). Same 13 rankable held-out cases and metrics as the seed study.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F

from src.ml import planner_train as pt
from src.ml.llm_prior import prior_path

KEYS = ("auroc", "ap", "coverage@10", "coverage@50", "coverage@100")


def fit_lr(data: pt.PlanData, X: np.ndarray, train_cases: list[str], l2: float = 1e-3) -> np.ndarray:
    """Same estimator as ``pt.fit_logreg`` (case-balanced, LBFGS, L2) on an arbitrary feature matrix."""
    X = torch.from_numpy(X.astype(np.float32))
    tri = np.flatnonzero(np.isin(data.case, train_cases))
    w = pt._sample_weights(data, tri).float() * len(tri)
    lin = torch.nn.Linear(X.shape[1], 1)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=200)
    xt, yt = X[tri], torch.from_numpy(data.y[tri])

    def closure():
        opt.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(lin(xt).squeeze(-1), yt, reduction="none") * w).mean() \
            + l2 * lin.weight.pow(2).sum()
        loss.backward()
        return loss

    with torch.enable_grad():
        opt.step(closure)
    with torch.no_grad():
        return lin(X).squeeze(-1).numpy()


def main() -> None:
    d = pt.PlanData()
    z = np.load(prior_path(d.dir))
    L = z["score"].astype(np.float32)
    assert len(L) == len(d.case) and (z["case"] == d.case).all(), "llm_prior.npz does not match the dataset"
    Lz = np.zeros_like(L)
    for c in d.cases:
        m = d.case == c
        Lz[m] = (L[m] - L[m].mean()) / (L[m].std() + 1e-6)
    M = d.x["meta"]
    models = {"logistic regression (metadata)": lambda o: fit_lr(d, M, o),
              "LLM prior (zero-shot)": lambda o: L,
              "LR (metadata + LLM prior)": lambda o: fit_lr(d, np.c_[M, L / 10, Lz], o)}
    ev = [k for k in d.cases if d.y[d.case == k].sum() > 0]
    per = {}
    for name, f in models.items():
        per[name] = {t: pt.case_metrics(d.y[d.case == t], f([k for k in d.cases if k != t])[d.case == t], d.items[t])
                     for t in ev}
        s = pt.summarize({name: per[name]})[name]
        print(f"| {name} | " + " | ".join(f"{s[k]['mean']:.3f}" for k in KEYS) + " |", flush=True)
    print("\nper-case AP: " + " / ".join(models))
    for t in ev:
        if pt.rankable(per["logistic regression (metadata)"][t]):
            print(f"  {t:24s}" + "  ".join(f"{per[n][t]['ap']:.3f}" for n in models))


if __name__ == "__main__":
    main()
