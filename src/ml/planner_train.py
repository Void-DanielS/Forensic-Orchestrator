"""Train and evaluate the investigation planner leave-one-case-out (LOCO).

    python main.py planner train          # LOCO evaluation (ablations + baselines) + final model on all cases

Few cases have published answer keys, so every solved case is held out once: the model is trained on the other
cases (two of them, drawn with a fixed seed, act as validation for the epoch choice and the temperature) and scored
on the held-out case, which it never saw. Metrics per held-out case, then averaged over cases (each case counts once):

  AUROC, average precision (AP)          ranking quality of "relevant" files
  coverage@K                             share of the case's *answer items* (key identifiers present in the evidence
                                         + files the key names) inside the top-K ranked files: how much of the
                                         solution an examiner reaches reading only K files

Baselines on the same folds: random order; a category prior (positive rate per file category in the training
cases); logistic regression on the metadata vector. Ablations: the same network with one modality, without the case
context, and with everything.
"""
from __future__ import annotations

import os

os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")      # deterministic cuBLAS (set before CUDA starts)

import json
import math
import re
import random
import shutil
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.config import settings
from src.ml import planner_dataset as pd_
from src.ml.planner import MODALITIES, Planner, PlannerConfig, PlannerNet, default_checkpoint, features
from src.ml.train import auroc, fit_temperature

ABLATIONS = {"path": ("path",), "header": ("header",), "metadata": ("metadata",), "content": ("content",),
             "all - context": ("path", "header", "metadata", "content"), "all": MODALITIES,
             "all + aux heads": MODALITIES,
             "all - context + SSL": ("path", "header", "metadata", "content"),
             "all - context + SSL frozen": ("path", "header", "metadata", "content"),
             "metadata + SSL": ("metadata",),
             "anchored": ("path", "header", "metadata", "content"),
             "anchored + SSL frozen": ("path", "header", "metadata", "content")}
# DRAFT, NOT YET RUN (2026-11-16, see HANDOVER next steps): data-efficiency variants (augmentation / synthetic
# positives, transfer learning from Jev, Noisy-Student self-training). Kept out of ABLATIONS / CANDIDATES so default
# runs are unchanged; evaluate with planner_seeds before adding them there.
EXPERIMENTAL = {"anchored + aug": ("path", "header", "metadata", "content"),
                "anchored + Jev header": ("path", "header", "metadata", "content"),
                "anchored + noisy student": ("path", "header", "metadata", "content")}
EXTRA = {"anchored + aug": {"augment": True},
         "anchored + Jev header": {"jev": True},
         "anchored + noisy student": {"augment": True, "student": True}}
ANCHOR_VARIANTS = {"anchored", "anchored + SSL frozen", "anchored + aug", "anchored + Jev header",
                   "anchored + noisy student"}
AUX_VARIANTS = {"all + aux heads"}
# encoders initialised from the self-supervised checkpoint (src/ml/planner_pretrain.py): name -> frozen?
SSL_VARIANTS = {"all - context + SSL": False, "all - context + SSL frozen": True, "metadata + SSL": False,
                "anchored + SSL frozen": True}
SSL_ENCODER_LR = 0.1          # fine-tuned pretrained encoders move at 0.1 x the fusion learning rate
# deep-learning variants that may be deployed; chosen by mean INNER-validation AP (never by held-out results)
CANDIDATES = ("all", "all + aux heads", "all - context", "metadata", "all - context + SSL",
              "all - context + SSL frozen", "metadata + SSL", "anchored", "anchored + SSL frozen")


def slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
AUX_WEIGHT = 0.5
AUX_MODALITY_DROPOUT = 0.4
KS = (10, 50, 100)


# ---------------------------------------------------------------------------------------------- data
class PlanData:
    def __init__(self, ds: Path | None = None):
        d = pd_.load(ds)
        if "snip_emb" not in d:
            raise SystemExit("text features missing: run `python main.py planner encode`")
        self.dir, self.rows = d["dir"], d["rows"]
        self.case = np.array([r["case_id"] for r in self.rows])
        bidx = {c: i for i, c in enumerate(d["brief_cases"])}
        brief = d["brief_emb"][[bidx[c] for c in self.case]]
        self.x = features(self.rows, d["head"], d["meta"], d["snip_emb"], brief)
        self.y = np.array([r["relevant"] for r in self.rows], dtype=np.float32)
        self.cases = sorted(set(self.case))
        self.items = {c: answer_items([r for r in self.rows if r["case_id"] == c]) for c in self.cases}

    def tensors(self, idx: np.ndarray, device: str) -> dict[str, torch.Tensor]:
        return {k: torch.from_numpy(np.ascontiguousarray(v[idx])).to(device) for k, v in self.x.items()}


def answer_items(rows: list[dict]) -> list[set]:
    """Per row, the answer items it reaches: key identifiers it contains + itself when the key names it."""
    return [set(r["entities"]) | ({f"file:{r['path']}"} if r["by_name"] else set()) for r in rows]


def coverage_at(order: np.ndarray, items: list[set], k: int) -> float | None:
    total = set().union(*items) if items else set()
    if not total:
        return None
    got = set().union(*(items[i] for i in order[:k])) if k else set()
    return len(got) / len(total)


def average_precision(y: np.ndarray, s: np.ndarray) -> float | None:
    if y.sum() == 0:
        return None
    o = np.argsort(-s, kind="stable")
    yy = y[o]
    hits = np.cumsum(yy)
    return float((hits[yy == 1] / (np.flatnonzero(yy == 1) + 1)).mean())


def case_metrics(y: np.ndarray, s: np.ndarray, items: list[set]) -> dict:
    o = np.argsort(-s, kind="stable")
    n = len(y)
    out = {"files": int(n), "relevant": int(y.sum()), "auroc": auroc(y.astype(int), s),
           "ap": average_precision(y, s), "prevalence": float(y.mean()) if n else 0.0,
           "answer_items": len(set().union(*items)) if items else 0}
    for k in KS:
        out[f"coverage@{k}"] = coverage_at(o, items, k)
    out["coverage@5%"] = coverage_at(o, items, max(1, math.ceil(0.05 * n)))
    pos = np.flatnonzero(y[o] == 1)
    out["recall@5%"] = float((pos < max(1, math.ceil(0.05 * n))).mean()) if len(pos) else None
    return out


# ---------------------------------------------------------------------------------------------- training
def _sample_weights(data: PlanData, idx: np.ndarray, pos_share: float = 0.25) -> torch.Tensor:
    """Each case counts equally; inside a case positives make up ``pos_share`` of the draws."""
    w = np.zeros(len(idx))
    cases = data.case[idx]
    y = data.y[idx]
    for c in set(cases):
        m = cases == c
        npos, nneg = y[m].sum(), (1 - y[m]).sum()
        if npos and nneg:
            w[m & (y == 1)] = pos_share / npos
            w[m & (y == 0)] = (1 - pos_share) / nneg
        else:
            w[m] = 1.0 / m.sum()
    return torch.tensor(w / w.sum())


# ---------------------------------------------------------------------------------------------- augmentation
META_CONT = [pd_.META_FEATURES.index(k) for k in ("log_size", "depth", "t_m", "t_a", "t_c", "t_b")]
NAME_CHARS = 24          # the right-aligned tail of the path (file name) is never perturbed
MAGIC = 16               # nor are the first header bytes (file signature)


def augment(t: dict, y: torch.Tensor, synth: float = 0.5) -> dict:
    """Label-preserving perturbations + synthetic positives, on the GPU, per batch.

    * path: random prefix crop (keep the last 24..160 chars) + 3 % character substitutions outside the file name:
      the same artefact lives under other user / volume / mount names in other cases;
    * header: 2 % byte noise after the 16-byte signature (slack, versions, timestamps differ between files);
    * metadata: Gaussian jitter (sigma 0.02) on the continuous columns (size, depth, times);
    * content: dropout 0.1 + Gaussian noise on the sentence embedding;
    * synthetic positives (SMOTE in feature space): a share ``synth`` of the positives get metadata (continuous
      columns) and content interpolated towards another positive of the batch, lambda ~ U(0.5, 1); label stays 1."""
    t = dict(t)
    p = t["path"]
    b, n = p.shape
    pos = torch.arange(n, device=p.device).expand(b, n)
    keep = torch.randint(NAME_CHARS, n + 1, (b, 1), device=p.device)
    p = torch.where(pos < n - keep, torch.zeros_like(p), p)
    sub = (torch.rand(b, n, device=p.device) < 0.03) & (pos < n - NAME_CHARS) & (p > 0)
    t["path"] = torch.where(sub, torch.randint(1, 128, (b, n), device=p.device), p)
    h = t["header"]
    hp = torch.arange(h.shape[1], device=h.device).expand_as(h)
    noise = (torch.rand(h.shape, device=h.device) < 0.02) & (hp >= MAGIC) & (h >= 0)
    t["header"] = torch.where(noise, torch.randint(0, 256, h.shape, device=h.device), h)
    m = t["meta"].clone()
    m[:, META_CONT] += 0.02 * torch.randn(b, len(META_CONT), device=m.device)
    c = F.dropout(t["content"], 0.1) + 0.01 * torch.randn_like(t["content"])
    pi = torch.nonzero(y > 0.5).flatten()
    if len(pi) > 1 and synth > 0:
        sel = pi[torch.rand(len(pi), device=pi.device) < synth]
        if len(sel):
            other = pi[torch.randint(0, len(pi), (len(sel),), device=pi.device)]
            lam = 0.5 + 0.5 * torch.rand(len(sel), 1, device=m.device)
            cols = torch.tensor(META_CONT, device=m.device)
            m[sel[:, None], cols] = lam * m[sel[:, None], cols] + (1 - lam) * m[other[:, None], cols]
            c[sel] = lam * c[sel] + (1 - lam) * c[other]
    t["meta"], t["content"] = m, c
    return t


class Pseudo:
    """Self-training data: the unlabelled READY cases (the SSL pretraining set, ``planner_pretrain.PreData``; no
    labelled case is in it) with soft labels from a teacher planner (Noisy Student, Xie et al. 2020)."""

    WEIGHT = 0.5         # loss weight of the pseudo-labelled half batch

    def __init__(self, pre, q: np.ndarray):
        self.pre, self.q = pre, q.astype(np.float32)
        w = np.zeros(len(q))
        for c in set(pre.case):                      # each case equal; inside it, 25 % of draws follow q
            m = pre.case == c
            w[m] = 0.75 / m.sum() + 0.25 * q[m] / max(q[m].sum(), 1e-9)
        self.w = torch.tensor(w / w.sum())

    def tensors(self, idx: np.ndarray, device: str) -> dict:
        t = self.pre.tensors(idx, device)
        b = len(idx)
        present = torch.ones(b, 5, dtype=torch.bool, device=device)
        present[:, 3] = t.pop("has_content")
        present[:, 4] = False
        t["context"] = torch.zeros(b, 768, device=device)
        t["present"] = present
        return t


@torch.no_grad()
def pseudo_labels(net: PlannerNet, pre, device: str, batch: int = 4096) -> np.ndarray:
    net.eval()
    ps = Pseudo(pre, np.zeros(len(pre)))
    out = []
    for i in range(0, len(pre), batch):
        t = ps.tensors(np.arange(i, min(i + batch, len(pre))), device)
        out.append(net(t["path"], t["header"], t["meta"], t["content"], t["context"], t["present"])["logit"]
                   .float().cpu().numpy())
    return 1 / (1 + np.exp(-np.concatenate(out) / net.cfg.temperature))


_PRE = None


def pretrain_data():
    global _PRE
    if _PRE is None:
        from src.ml.planner_pretrain import PreData

        _PRE = PreData()
    return _PRE


@torch.no_grad()
def predict(net: PlannerNet, data: PlanData, idx: np.ndarray, device: str, batch: int = 4096) -> np.ndarray:
    net.eval()
    out = []
    for i in range(0, len(idx), batch):
        t = data.tensors(idx[i:i + batch], device)
        out.append(net(t["path"], t["header"], t["meta"], t["content"], t["context"], t["present"])["logit"]
                   .float().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def train_model(data: PlanData, train_cases: list[str], val_cases: list[str], modalities=MODALITIES, epochs: int = 10,
                steps_per_epoch: int = 150, batch: int = 256, lr: float = 2e-3, seed: int = 13,
                device: str | None = None, aux: bool = False, init: Path | None = None,
                freeze: bool = False, anchor: bool = False, augment_: bool = False, jev: bool = False,
                pseudo: "Pseudo | None" = None) -> tuple[PlannerNet, dict]:
    device = device or settings.device_for("orchestrator")
    deterministic(seed)
    tr = np.flatnonzero(np.isin(data.case, train_cases))
    va = np.flatnonzero(np.isin(data.case, val_cases))
    cfg = PlannerConfig(modalities=tuple(modalities), aux_heads=aux, anchor=anchor, header_jev=jev,
                        modality_dropout=AUX_MODALITY_DROPOUT if aux else 0.15)
    net = PlannerNet(cfg)
    loaded: list[str] = []
    slow: set[int] = set()
    if jev:                                   # transfer learning: Jev trunk fine-tuned at 0.1 x lr
        from src.ml.planner import load_jev

        load_jev(net)
        slow = {id(p) for p in net.enc["header"].trunk.parameters()}
        cfg.meta.update(transfer="jev byte CNN -> header encoder")
    cfg.meta.update(augment=augment_, self_training=pseudo is not None)
    if init is not None:
        from src.ml.planner_pretrain import load_into

        loaded = load_into(net, init, freeze=freeze)
        cfg.meta.update(pretrained=str(init), pretrained_frozen=freeze)
    net = net.to(device)
    pre = {id(p) for m in loaded for p in net.enc[m].parameters()} if not freeze else set()
    pre |= slow
    groups = [{"params": [p for p in net.parameters() if p.requires_grad and id(p) not in pre], "lr": lr}]
    if pre:
        groups.append({"params": [p for p in net.parameters() if id(p) in pre], "lr": lr * SSL_ENCODER_LR})
    opt = torch.optim.AdamW(groups, lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups],
                                                total_steps=epochs * steps_per_epoch, pct_start=0.1)
    w = _sample_weights(data, tr)
    best, best_state, best_ep = -1.0, None, 0
    for ep in range(1, epochs + 1):
        net.train()
        draw = tr[torch.multinomial(w, steps_per_epoch * batch, replacement=True).numpy()]
        pdraw = (torch.multinomial(pseudo.w, steps_per_epoch * (batch // 2), replacement=True).numpy()
                 if pseudo is not None else None)
        for s in range(steps_per_epoch):
            ii = draw[s * batch:(s + 1) * batch]
            t = data.tensors(ii, device)
            yt = torch.from_numpy(data.y[ii]).to(device)
            if augment_:
                t = augment(t, yt)
            present = t["present"] & (torch.rand_like(t["present"], dtype=torch.float) > cfg.modality_dropout)
            o = net(t["path"], t["header"], t["meta"], t["content"], t["context"], present)
            loss = F.binary_cross_entropy_with_logits(o["logit"], yt)
            if "anchor" in o:          # the anchor must stand alone as a metadata model
                loss = loss + F.binary_cross_entropy_with_logits(o["anchor"], yt)
            if pdraw is not None:      # Noisy Student: soft teacher labels on unlabelled cases, noised input
                jj = pdraw[s * (batch // 2):(s + 1) * (batch // 2)]
                tp = pseudo.tensors(jj, device)
                qt = torch.from_numpy(pseudo.q[jj]).to(device)
                tp = augment(tp, qt, synth=0.0)
                pp = tp["present"] & (torch.rand_like(tp["present"], dtype=torch.float) > cfg.modality_dropout)
                op = net(tp["path"], tp["header"], tp["meta"], tp["content"], tp["context"], pp)
                lp = F.binary_cross_entropy_with_logits(op["logit"], qt)
                if "anchor" in op:
                    lp = lp + F.binary_cross_entropy_with_logits(op["anchor"], qt)
                loss = loss + Pseudo.WEIGHT * lp
            if "aux" in o:
                loss = loss + AUX_WEIGHT * torch.stack([F.binary_cross_entropy_with_logits(v, yt)
                                                        for v in o["aux"].values()]).mean()
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
        s_va = predict(net, data, va, device)
        aps = [average_precision(data.y[va][data.case[va] == c], s_va[data.case[va] == c]) for c in val_cases]
        sel = float(np.mean([a for a in aps if a is not None])) if any(a is not None for a in aps) else 0.0
        if sel > best:
            best, best_ep = sel, ep
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    cfg.temperature = fit_temperature(predict(net, data, va, device), data.y[va], binary=True) if len(va) else 1.0
    cfg.meta.update(best_epoch=best_ep, val_ap=round(best, 4), train_cases=list(train_cases),
                    val_cases=list(val_cases))
    net.cfg = cfg
    return net, {"best_epoch": best_ep, "val_ap": best}


def train_variant(data: PlanData, name: str, tr_cases: list[str], va_cases: list[str], epochs: int, seed: int,
                  device: str | None = None, encoders: Path | None = None) -> tuple[PlannerNet, dict]:
    """Train ABLATIONS[name] with its options; the noisy student first trains a plain anchored teacher on the same
    labelled cases, pseudo-labels the unlabelled cases with it, then trains the student on both."""
    ex = EXTRA.get(name, {})
    kw = dict(epochs=epochs, seed=seed, device=device, aux=name in AUX_VARIANTS,
              init=encoders if name in SSL_VARIANTS else None, freeze=SSL_VARIANTS.get(name, False),
              anchor=name in ANCHOR_VARIANTS, augment_=ex.get("augment", False), jev=ex.get("jev", False))
    pseudo = None
    if ex.get("student"):
        teacher, _ = train_model(data, tr_cases, va_cases, {**ABLATIONS, **EXPERIMENTAL}[name], **{**kw, "augment_": False})
        pre = pretrain_data()
        pseudo = Pseudo(pre, pseudo_labels(teacher, pre, device or settings.device_for("orchestrator")))
        del teacher
    return train_model(data, tr_cases, va_cases, {**ABLATIONS, **EXPERIMENTAL}[name], pseudo=pseudo, **kw)


MIN_VAL_POSITIVES = 10


def deterministic(seed: int) -> None:
    """Same seed -> same model on one machine (a rerun used to move coverage@50 by 0.1 through GPU non-determinism)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.use_deterministic_algorithms(True, warn_only=True)


def _val_split(cases: list[str], seed: int, data: "PlanData | None" = None, n: int = 1
               ) -> tuple[list[str], list[str]]:
    """One validation case (epoch choice + temperature), drawn among cases with enough relevant files for the
    validation AP to mean something (tiny cases give AP = 1.0 after one epoch)."""
    rng = random.Random(seed)
    pool = [c for c in cases if data is None or data.y[data.case == c].sum() >= MIN_VAL_POSITIVES] or list(cases)
    val = rng.sample(pool, min(n, max(len(pool) - 1, 1)))
    return [c for c in cases if c not in val], val


# ---------------------------------------------------------------------------------------------- baselines
def baseline_scores(data: PlanData, train_cases: list[str], test_idx: np.ndarray, seed: int) -> dict[str, np.ndarray]:
    tr = np.isin(data.case, train_cases)
    cats = np.array([r["category"] for r in data.rows])
    rate = {}
    for c in set(cats):
        m = tr & (cats == c)
        # per-case average so one big case does not decide the prior
        per_case = [data.y[m & (data.case == k)].mean() for k in train_cases if (m & (data.case == k)).any()]
        rate[c] = float(np.mean(per_case)) if per_case else 0.0
    size = data.x["meta"][:, 0]
    prior = np.array([rate.get(c, 0.0) for c in cats[test_idx]]) + 1e-6 * size[test_idx]
    # logistic regression on metadata (same case-balanced weights)
    xm = torch.from_numpy(data.x["meta"])
    tri = np.flatnonzero(tr)
    w = _sample_weights(data, tri).float() * len(tri)
    lin = torch.nn.Linear(xm.shape[1], 1)
    opt = torch.optim.LBFGS(lin.parameters(), max_iter=200)
    xt, yt = xm[tri], torch.from_numpy(data.y[tri])

    def closure():
        opt.zero_grad()
        loss = (F.binary_cross_entropy_with_logits(lin(xt).squeeze(-1), yt, reduction="none") * w).mean() \
            + 1e-3 * lin.weight.pow(2).sum()
        loss.backward()
        return loss

    opt.step(closure)
    with torch.no_grad():
        lr_s = lin(xm[test_idx]).squeeze(-1).numpy()
    rng = np.random.default_rng(seed)
    return {"random": rng.random(len(test_idx)), "category prior": prior, "logistic regression (metadata)": lr_s}


# ---------------------------------------------------------------------------------------------- LOCO
def loco(data: PlanData, ablations: dict = ABLATIONS, epochs: int = 10, seed: int = 13, progress: bool = True,
         device: str | None = None, save_dir: Path | None = None, encoders: Path | None = None) -> dict:
    """Leave-one-case-out; with ``save_dir`` the full model of every fold is kept (``<case>.pt``) so a solved case
    can be investigated end to end by a model that never saw it."""
    res: dict = defaultdict(dict)
    eval_cases = [c for c in data.cases if data.y[data.case == c].sum() > 0]
    t0 = time.perf_counter()
    for i, test in enumerate(eval_cases, 1):
        others = [c for c in data.cases if c != test]
        tr_cases, va_cases = _val_split(others, seed + i, data)
        te = np.flatnonzero(data.case == test)
        items = data.items[test]
        y = data.y[te]
        for name, s in baseline_scores(data, others, te, seed + i).items():
            res[name][test] = case_metrics(y, s, items)
        for name, mods in ablations.items():
            net, info = train_variant(data, name, tr_cases, va_cases, epochs, seed, device, encoders)
            s = predict(net, data, te, device or settings.device_for("orchestrator"))
            res[name][test] = {**case_metrics(y, s, items), "best_epoch": info["best_epoch"],
                               "inner_val_ap": round(info["val_ap"], 4)}
            if save_dir is not None and name in CANDIDATES:
                Planner.save(net, save_dir / slug(name) / f"{test}.pt")
            del net
        if progress:
            shown = "all" if "all" in ablations else next(iter(ablations))
            a = res[shown][test]
            print(f"[planner loco] {i}/{len(eval_cases)} {test}: files {a['files']:,}, relevant {a['relevant']:,}, "
                  f"AP {shown} {a['ap']:.3f} vs prior {res['category prior'][test]['ap']:.3f} vs random "
                  f"{res['random'][test]['ap']:.3f}; coverage@50 {a['coverage@50']} "
                  f"({time.perf_counter() - t0:.0f} s)", flush=True)
    return dict(res)


MIN_RANKED_FILES = 50


def rankable(m: dict) -> bool:
    """A held-out case where ranking means something: enough files and not every file relevant."""
    return m["files"] >= MIN_RANKED_FILES and 0 < m["relevant"] < m["files"]


def summarize(res: dict) -> dict:
    keys = ["auroc", "ap", "coverage@10", "coverage@50", "coverage@100", "coverage@5%", "recall@5%"]
    out = {}
    for name, per in res.items():
        out[name] = {}
        for k in keys:
            v = [m[k] for m in per.values() if m.get(k) is not None and rankable(m)]
            out[name][k] = {"mean": round(float(np.mean(v)), 4) if v else None,
                            "sem": round(float(np.std(v) / math.sqrt(len(v))), 4) if len(v) > 1 else None,
                            "n_cases": len(v)}
    return out


def train(dataset: Path | None = None, epochs: int = 10, seed: int = 13, ablations: bool = True,
          progress: bool = True, out_dir: Path | None = None, checkpoint: Path | None = None) -> dict:
    data = PlanData(dataset)
    run = out_dir or settings.OUTPUTS_DIR / "ml" / "planner" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    from src.ml.planner_pretrain import default_encoders

    encoders = default_encoders() if default_encoders().exists() else None
    abl = dict(ABLATIONS) if ablations else {"all": MODALITIES}
    if encoders is None:
        for k in SSL_VARIANTS:
            abl.pop(k, None)
        if progress:
            print("[planner] no pretrained encoders: SSL variants skipped (`python main.py planner pretrain`)")
    res = loco(data, abl, epochs, seed, progress, save_dir=default_checkpoint().parent / "loco", encoders=encoders)
    summ = summarize(res)
    # model selection on inner validation only
    sel = {n: round(float(np.mean([v["inner_val_ap"] for v in res[n].values()])), 4) for n in CANDIDATES if n in res}
    deploy = max(sel, key=sel.get)
    loco_dir = default_checkpoint().parent / "loco"
    for f in (loco_dir / slug(deploy)).glob("*.pt"):
        shutil.copyfile(f, loco_dir / f.name)          # loco/<case>.pt = the selected variant's fold model
    # deployed model: all solved cases (one held for validation / temperature)
    tr_cases, va_cases = _val_split(data.cases, seed, data)
    net, info = train_variant(data, deploy, tr_cases, va_cases, epochs, seed, encoders=encoders)
    ck = Planner.save(net, run / "planner.pt")
    dest = checkpoint or default_checkpoint()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ck, dest)
    # what the final model attends to, per category (diagnostic)
    pl = Planner(net, settings.device_for("orchestrator"))
    sc = pl.score(data.x, attn=True)
    cats = np.array([r["category"] for r in data.rows])
    att = {c: dict(zip(MODALITIES, np.round(sc["attention"][cats == c].mean(0), 3).tolist()))
           for c in sorted(set(cats)) if (cats == c).sum() >= 50}
    metrics = {"dataset": str(data.dir), "cases": data.cases, "files": len(data.rows),
               "relevant": int(data.y.sum()), "per_case": res, "summary": summ, "final": info,
               "selection": {"inner_val_ap": sel, "deployed": deploy},
               "checkpoint": str(dest), "attention_by_category": att}
    (run / "metrics.json").write_text(json.dumps(metrics, indent=1, default=str), encoding="utf-8")
    write_report(run, metrics)
    return {"run": str(run), "checkpoint": str(dest), "metrics": metrics}


def rereport(run: Path) -> Path:
    """Recompute the summary from a run's saved per-case metrics and rewrite its REPORT.md."""
    m = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    m["summary"] = summarize(m["per_case"])
    (run / "metrics.json").write_text(json.dumps(m, indent=1, default=str), encoding="utf-8")
    return write_report(run, m)


def write_report(run: Path, m: dict) -> Path:
    s = m["summary"]

    def f(name, k):
        v = s[name][k]
        return "" if v["mean"] is None else f"{v['mean']:.3f}" + (f" ± {v['sem']:.3f}" if v["sem"] is not None else "")

    order = ["random", "category prior", "logistic regression (metadata)"] + [k for k in ABLATIONS if k in s]
    cols = ["auroc", "ap", "coverage@10", "coverage@50", "coverage@100", "coverage@5%"]
    rows = "\n".join(f"| {n} | " + " | ".join(f(n, k) for k in cols) + " |" for n in order if n in s)
    dep = m.get("selection", {}).get("deployed", "all")
    per = m["per_case"][dep]
    pc = "\n".join(f"| {c} | {v['files']:,} | {v['relevant']:,} | {v['answer_items']} | {v['auroc']} | "
                   f"{v['ap']:.3f} | {m['per_case']['category prior'][c]['ap']:.3f} | {v['coverage@50']} | "
                   f"{m['per_case']['random'][c]['coverage@50']} |" for c, v in per.items())
    att = "\n".join(f"| {c} | " + " | ".join(f"{a[x]:.2f}" for x in MODALITIES) + " |"
                    for c, a in m["attention_by_category"].items())
    md = f"""# Investigation planner: leave-one-case-out report

Run `{run.name}` · dataset `{m['dataset']}` · {m['files']:,} files from {len(m['cases'])} solved cases,
{m['relevant']:,} relevant (named by the answer key or containing one of its identifiers).

Each solved case is held out once; the model never sees it (nor its answer key) during training. Values are means
over held-out cases ± standard error, over the {sum(rankable(v) for v in m['per_case'][dep].values())} cases where
ranking is meaningful (≥ {MIN_RANKED_FILES} files, not every file relevant); every case is listed below.

## Held-out cases: baselines and modality ablation

| model | AUROC | AP | coverage@10 | coverage@50 | coverage@100 | coverage@5% |
|---|---|---|---|---|---|---|
{rows}

**Model selection** (mean AP on the inner-validation case of each fold; held-out cases play no part): {m.get('selection', {}).get('inner_val_ap')} → deployed **{dep}**.

coverage@K = share of the case's answer items (key identifiers present in the evidence + files the key names) inside
the K top-ranked files.

## Per held-out case (deployed variant: {dep})

| case | files | relevant | answer items | AUROC | AP | AP category prior | coverage@50 | coverage@50 random |
|---|---|---|---|---|---|---|---|---|
{pc}

## Deployed model (trained on all solved cases): fusion attention by file category

| category | {' | '.join(MODALITIES)} |
|---|{'---|' * len(MODALITIES)}
{att}

The planner's scores are **leads**: they decide which files the vision agent (images) and the reader (everything
else) open first. Nothing it says is evidence; every reported statement still passes the HeuristicVerificationEngine.
"""
    p = run / "REPORT.md"
    p.write_text(md, encoding="utf-8")
    return p
