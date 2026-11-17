"""Nested selection of the planner: inner-validation cases, no leak of the validation case into the wide part, the
recorded scores and the (variant, epoch) choice. Synthetic data on CPU."""
import functools

import numpy as np
import torch

from src.ml import planner_dataset as pd_
from src.ml import planner_nested as pn
from src.ml import planner_train as pt
from src.ml.planner import features


def _data(n_cases=4, n=60):
    rng = np.random.default_rng(0)
    N = n_cases * n
    rows = [{"path_text": f"Users/a/f{i}.txt", "snippet": "x", "case_id": f"c{i // n}", "relevant": int(i % 4 == 0),
             "entities": [f"e{i}"] if i % 4 == 0 else [], "by_name": 0, "path": f"fs/f{i}"} for i in range(N)]
    d = object.__new__(pt.PlanData)
    d.dir, d.rows = None, rows
    d.case = np.array([r["case_id"] for r in rows])
    d.x = features(rows, rng.integers(0, 255, (N, 256)).astype(np.uint8),
                   rng.random((N, pd_.N_META), dtype=np.float32), rng.normal(size=(N, 768)).astype(np.float32),
                   np.zeros((N, 768), np.float32))
    d.y = np.array([r["relevant"] for r in rows], np.float32)
    d.x["meta"][:, 0] += 3 * d.y                          # learnable signal in the metadata
    d.cases = sorted(set(d.case))
    d.items = {c: pt.answer_items([r for r in rows if r["case_id"] == c]) for c in d.cases}
    return d


def test_run_fold_and_select(monkeypatch):
    d = _data()
    seen = []
    real_fit = pt.fit_logreg

    def spy_fit(data, cases):
        seen.append(sorted(cases))
        return real_fit(data, cases)

    monkeypatch.setattr(pt, "fit_logreg", spy_fit)
    monkeypatch.setattr(pt, "train_model", functools.partial(pt.train_model, steps_per_epoch=2, batch=16))
    scores, val = pn.run_fold(d, "c0", ("metadata", "wide & deep"), (1,), 2, "cpu")
    inner = pn.inner_cases(d, ["c1", "c2", "c3"])
    assert inner == ["c1", "c2", "c3"]
    # wide part: fitted on the inner training cases only, never on the inner-validation case
    assert seen == [sorted(c for c in ["c1", "c2", "c3"] if c != v) for v in inner]
    # metadata: epochs 1..2; wide & deep: epochs 0..2 (epoch 0 = the logistic regression)
    assert {k for k in val if k[0] == "metadata"} == {("metadata", v, 1, e) for v in inner for e in (1, 2)}
    assert {k for k in val if k[0] == "wide & deep"} == {("wide & deep", v, 1, e) for v in inner for e in (0, 1, 2)}
    assert all(s.shape == (60,) for s in scores.values())
    name, ep, keys = pn.select({pn._key(k): a for k, a in val.items()})
    assert len(keys) == 3 and all(k.startswith(f"{name}|") and k.endswith(f"|{ep}") for k in keys)


def test_select_prefers_best_mean_then_earliest_epoch():
    val = {"a|v1|1|1": 0.5, "a|v2|1|1": 0.1, "b|v1|1|0": 0.3, "b|v2|1|0": 0.4, "b|v1|1|2": 0.3, "b|v2|1|2": 0.4}
    assert pn.select(val)[:2] == ("b", 0)


def test_gpu_cache_gives_same_tensors():
    d = _data(2, 20)
    idx = np.array([3, 1, 7])
    a = d.tensors(idx, "cpu")
    d.cache("cpu")
    b = d.tensors(idx, "cpu")
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_ensemble_checkpoint_round_trip(tmp_path):
    from src.ml.planner import EnsemblePlanner, Planner, PlannerConfig, PlannerNet

    d = _data(1, 30)
    torch.manual_seed(0)
    nets = [PlannerNet(PlannerConfig(modalities=("metadata",), dim=32, heads=2, layers=1)) for _ in range(3)]
    path = EnsemblePlanner.save_ensemble(nets, tmp_path / "e.pt", {"variant": "metadata"})
    pl = Planner.load(path, "cpu")
    assert isinstance(pl, EnsemblePlanner) and len(pl.nets) == 3 and pl.meta["variant"] == "metadata"
    p = pl.score(d.x)["p"]
    logits = np.mean([pt.predict(n, d, np.arange(30), "cpu") for n in nets], 0)
    assert np.allclose(p, 1 / (1 + np.exp(-logits)), atol=1e-5)
