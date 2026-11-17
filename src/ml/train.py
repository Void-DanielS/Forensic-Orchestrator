"""Train, calibrate and evaluate the Jev typed-answer model on the fragment dataset.

    python main.py jev train [--epochs 20] [--dataset data/datasets/fragments/<date>]
    python main.py jev eval  [--checkpoint data/models/jev/jev.pt]

Protocol: train on the train cases, select the epoch and fit the calibration temperatures on the validation cases,
report once on the test cases (case-level splits, ``src/datasets/export.SPLITS``). Baselines on the same splits:

  choice   multinomial logistic regression on the 256-bin byte histogram (no deep features)
  yes_no   that baseline's predicted type + the FileSystemAgent extension rule
  score    constant 0.5 (centre of the file)

Outputs: ``data/outputs/ml/jev/<run>/{metrics.json, history.json, REPORT.md}`` and the checkpoint
``data/models/jev/jev.pt`` (copied from the run folder).
"""
from __future__ import annotations

import json
import math
import random
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.config import settings
from src.ml import fragments as fr
from src.ml.jev import (BIN_CENTERS, SCORE_BINS, Jev, JevConfig, JevNet, default_checkpoint, encode_ext,
                        score_targets)

MIN_TRAIN_FRAGMENTS = 200     # a type with fewer training fragments is left out (reported, not merged)
DISGUISE_AUG = 0.5            # share of signature-type samples whose extension is re-drawn from the pool
LOSS_W = {"choice": 1.0, "yes_no": 0.5, "score": 0.3}


# ---------------------------------------------------------------------------------------------- data
# Extension pool per signature type (generic / temp names excluded: the rule never flags them).
EXT_POOL = {k: sorted(e for e in exts if e and e not in fr.TOLERATED_EXTS and len(e) <= 6)
            for k, exts in fr.ALLOWED_EXTS.items()}
EXT_POOL = {k: v for k, v in EXT_POOL.items() if v}


def augment(labels: np.ndarray, exts: np.ndarray, disg: np.ndarray, rng: random.Random,
            p: float = DISGUISE_AUG) -> tuple[np.ndarray, np.ndarray]:
    """Re-draw the extension of a share of signature-type samples from one shared pool and relabel with the rule.

    The pool type is drawn uniformly, so every extension is paired with every content type: the extension alone
    cannot tell disguised from consistent; the content has to (checked by the extension-only baseline).
    """
    exts, disg = exts.copy(), disg.copy()
    kinds = sorted(EXT_POOL)
    for i, lab in enumerate(labels):
        if lab in fr.ALLOWED_EXTS and rng.random() < p:
            exts[i] = rng.choice(EXT_POOL[rng.choice(kinds)])
            disg[i] = fr.is_disguised(str(lab), "x" + exts[i])
    return exts, disg


class Split:
    def __init__(self, d: dict, mask: np.ndarray, classes: list[str], device: str):
        self.labels_txt = d["label"][mask]
        self.x = torch.from_numpy(d["x"][mask]).to(device)
        self.y = torch.tensor([classes.index(c) for c in self.labels_txt], device=device)
        self.pos = torch.from_numpy(d["position"][mask]).to(device)
        self.ext_raw = d["ext"][mask]
        self.disg_raw = d["disguised"][mask].astype(np.int64)
        self.case_id = d["case_id"][mask]
        self.device = device
        self.set_ext(self.ext_raw, self.disg_raw)

    def set_ext(self, exts: np.ndarray, disg: np.ndarray) -> None:
        self.ext_txt, self.disg_np = exts, disg
        self.ext = torch.tensor(np.array([encode_ext(e) for e in exts], dtype=np.int64), device=self.device)
        self.disg = torch.from_numpy(disg).to(self.device)

    def __len__(self) -> int:
        return len(self.y)


def prepare(d: dict, device: str, seed: int = 13) -> tuple[list[str], dict[str, Split], dict]:
    train_counts = Counter(d["label"][d["split"] == "train"])
    classes = sorted(k for k, n in train_counts.items() if n >= MIN_TRAIN_FRAGMENTS)
    dropped = {str(k): n for k, n in train_counts.items() if k not in classes}
    train_counts = {str(k): n for k, n in train_counts.items()}
    keep = np.isin(d["label"], classes)
    splits = {s: Split(d, keep & (d["split"] == s), classes, device) for s in ("train", "val", "test")}
    for i, s in enumerate(("val", "test")):     # fixed, seeded disguise augmentation for the evaluation splits
        sp = splits[s]
        sp.set_ext(*augment(sp.labels_txt, sp.ext_raw, sp.disg_raw, random.Random(seed + 1 + i)))
    return classes, splits, {"dropped_classes": dropped, "train_counts": dict(train_counts)}


# ---------------------------------------------------------------------------------------------- training
def _losses(out: dict, y, disg, pos) -> dict[str, torch.Tensor]:
    m = disg >= 0
    yn = F.binary_cross_entropy_with_logits(out["yes_no"][m], disg[m].float()) if m.any() else out["yes_no"].sum() * 0
    return {"choice": F.cross_entropy(out["choice"], y, label_smoothing=0.05), "yes_no": yn,
            "score": F.cross_entropy(out["score"], score_targets(pos))}


@torch.no_grad()
def predict(net: JevNet, sp: Split, batch: int = 4096) -> dict[str, np.ndarray]:
    net.eval()
    outs = {"choice": [], "score": [], "yes_no": []}
    for i in range(0, len(sp), batch):
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=sp.device.startswith("cuda")):
            o = net(sp.x[i:i + batch].long(), sp.ext[i:i + batch])
        for k in outs:
            outs[k].append(o[k].float().cpu().numpy())
    return {k: np.concatenate(v) for k, v in outs.items()}


def train(dataset: Path | None = None, epochs: int = 20, batch: int = 512, lr: float = 2e-3, seed: int = 13,
          out_dir: Path | None = None, device: str | None = None, progress: bool = True,
          checkpoint: Path | None = None) -> dict:
    device = device or settings.device_for("text")
    torch.manual_seed(seed)
    np.random.seed(seed)
    rng = random.Random(seed)
    ds_path = dataset or fr.latest()
    d = fr.load(ds_path)
    classes, splits, info = prepare(d, device, seed)
    tr, va, te = splits["train"], splits["val"], splits["test"]
    if not len(tr) or not len(va) or not len(te):
        raise SystemExit(f"empty split (train={len(tr)}, val={len(va)}, test={len(te)}); rebuild the dataset")
    cfg = JevConfig(classes=classes, meta={"dataset": str(ds_path), "seed": seed, "epochs": epochs,
                                           "trained_utc": datetime.now().isoformat(timespec="seconds")})
    net = JevNet(cfg).to(device)
    n_params = sum(p.numel() for p in net.parameters())
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
    steps = epochs * math.ceil(len(tr) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=steps, pct_start=0.1)
    # class-balanced sampling, softened (weight ~ n^-0.5): rare types are seen more, without pure oversampling
    cnt = torch.bincount(tr.y, minlength=len(classes)).float()
    w = (cnt[tr.y] ** -0.5)
    run = out_dir or settings.OUTPUTS_DIR / "ml" / "jev" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    history, best, best_state, t0 = [], -1.0, None, time.perf_counter()
    if progress:
        print(f"[jev train] {len(classes)} classes, train/val/test = {len(tr):,}/{len(va):,}/{len(te):,} fragments, "
              f"{n_params / 1e6:.2f} M parameters, device {device}", flush=True)
    for ep in range(1, epochs + 1):
        net.train()
        tr.set_ext(*augment(tr.labels_txt, tr.ext_raw, tr.disg_raw, rng))
        order = torch.multinomial(w, len(tr), replacement=True)
        sums: Counter = Counter()
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                out = net(tr.x[idx].long(), tr.ext[idx])
            ls = _losses({k: v.float() for k, v in out.items()}, tr.y[idx], tr.disg[idx], tr.pos[idx])
            loss = sum(LOSS_W[k] * v for k, v in ls.items())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            for k, v in ls.items():
                sums[k] += float(v.detach()) * len(idx)
        vm = evaluate_logits(predict(net, va), va, classes)
        row = {"epoch": ep, **{f"train_loss_{k}": round(v / len(tr), 4) for k, v in sums.items()},
               "val_choice_acc": vm["choice"]["accuracy"], "val_choice_macro_f1": vm["choice"]["macro_f1"],
               "val_yes_no_auroc": vm["yes_no"]["auroc"], "val_score_mae": vm["score"]["mae"],
               "seconds": round(time.perf_counter() - t0, 1)}
        history.append(row)
        if progress:
            print(f"[jev train] epoch {ep}/{epochs} " + " ".join(f"{k}={v}" for k, v in row.items() if k != "epoch"),
                  flush=True)
        # selection on the validation cases only
        sel = vm["choice"]["macro_f1"] + 0.25 * (vm["yes_no"]["auroc"] or 0)
        if sel > best:
            best, best_state = sel, {k: v.detach().clone() for k, v in net.state_dict().items()}
            cfg.meta["best_epoch"] = ep
    net.load_state_dict(best_state)
    lv = predict(net, va)
    cfg.t_choice = fit_temperature(lv["choice"], va.y.cpu().numpy())
    m = va.disg_np >= 0
    cfg.t_yes_no = fit_temperature(lv["yes_no"][m], va.disg_np[m], binary=True)
    cfg.t_score = fit_temperature(lv["score"], score_targets(va.pos).cpu().numpy())
    cfg.meta.update(n_params=n_params, train_seconds=round(time.perf_counter() - t0, 1), **info)
    ckpt = Jev.save(net, run / "jev.pt")
    metrics = {"classes": classes, "config": {k: v for k, v in vars(cfg).items() if k != "meta"}, "meta": cfg.meta,
               "sizes": {s: len(sp) for s, sp in splits.items()},
               "val": evaluate_logits(lv, va, classes, cfg), "test": evaluate_logits(predict(net, te), te, classes, cfg),
               "test_natural_yes_no": natural_yes_no(net, te, cfg), "baselines": baselines(splits, classes, device)}
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (run / "history.json").write_text(json.dumps(history, indent=2), encoding="utf-8")
    write_report(run, metrics, history)
    if progress:
        print(f"[jev train] natural disguise labels (test): {metrics['test_natural_yes_no']}", flush=True)
    dest = checkpoint or default_checkpoint()
    dest.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(ckpt, dest)
    return {"run": str(run), "checkpoint": str(dest), "metrics": metrics}


def natural_yes_no(net: JevNet, sp: Split, cfg: JevConfig) -> dict:
    """Disguise query on the files' real extensions only (no augmentation): the deployment condition."""
    aug_ext, aug_disg = sp.ext_txt, sp.disg_np
    sp.set_ext(sp.ext_raw, sp.disg_raw)
    try:
        r = evaluate_logits(predict(net, sp), sp, cfg.classes, cfg)["yes_no"]
    finally:
        sp.set_ext(aug_ext, aug_disg)
    return r


def fit_temperature(logits: np.ndarray, y: np.ndarray, binary: bool = False) -> float:
    """Temperature scaling (Guo et al., 2017): one scalar T minimising validation NLL."""
    if not len(y):
        return 1.0
    z, t = torch.tensor(logits, dtype=torch.float64), torch.tensor(y)
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=200)

    def closure():
        opt.zero_grad()
        s = z / log_t.exp()
        loss = F.binary_cross_entropy_with_logits(s, t.double()) if binary else F.cross_entropy(s, t.long())
        loss.backward()
        return loss

    opt.step(closure)
    # perfectly separable validation logits drive T -> 0 (or diverge): keep it in a sane range
    return round(min(max(float(log_t.detach().exp()), 0.05), 20.0), 4)


# ---------------------------------------------------------------------------------------------- metrics
def classification_metrics(y: np.ndarray, pred: np.ndarray, classes: list[str]) -> dict:
    k = len(classes)
    cm = np.zeros((k, k), dtype=np.int64)
    np.add.at(cm, (y, pred), 1)
    tp = np.diag(cm).astype(float)
    prec = np.divide(tp, cm.sum(0), out=np.zeros(k), where=cm.sum(0) > 0)
    rec = np.divide(tp, cm.sum(1), out=np.zeros(k), where=cm.sum(1) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(k), where=(prec + rec) > 0)
    present = cm.sum(1) > 0
    return {"accuracy": round(float(tp.sum() / max(cm.sum(), 1)), 4),
            "macro_f1": round(float(f1[present].mean()) if present.any() else 0.0, 4),
            "per_class": {c: {"precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4),
                              "f1": round(float(f1[i]), 4), "support": int(cm[i].sum())} for i, c in enumerate(classes)},
            "confusion": cm.tolist()}


def auroc(y: np.ndarray, s: np.ndarray) -> float | None:
    """Mann-Whitney AUROC with tie handling."""
    pos, neg = int((y == 1).sum()), int((y == 0).sum())
    if not pos or not neg:
        return None
    order = np.argsort(s, kind="mergesort")
    ranks = np.empty(len(s))
    sorted_s = s[order]
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and sorted_s[j + 1] == sorted_s[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2 + 1
        i = j + 1
    return round(float((ranks[y == 1].sum() - pos * (pos + 1) / 2) / (pos * neg)), 4)


def ece(conf: np.ndarray, correct: np.ndarray, bins: int = 15) -> float:
    """Expected calibration error (equal-width confidence bins)."""
    edges = np.linspace(0, 1, bins + 1)
    total = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (conf > lo) & (conf <= hi)
        if m.any():
            total += m.mean() * abs(correct[m].mean() - conf[m].mean())
    return round(float(total), 4)


def _sm(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def evaluate_logits(lg: dict, sp: Split, classes: list[str], cfg: JevConfig | None = None) -> dict:
    t = cfg or JevConfig(classes=classes)
    y = sp.y.cpu().numpy()
    p = _sm(lg["choice"] / t.t_choice)
    pred, conf = p.argmax(1), p.max(1)
    ch = classification_metrics(y, pred, classes)
    ch["ece_uncalibrated"] = ece(_sm(lg["choice"]).max(1), _sm(lg["choice"]).argmax(1) == y)
    ch["ece"] = ece(conf, pred == y)
    keep = conf >= t.abstain_below
    ch["coverage_at_threshold"] = round(float(keep.mean()), 4)
    ch["selective_accuracy"] = round(float((pred == y)[keep].mean()) if keep.any() else 0.0, 4)
    m = sp.disg_np >= 0
    yn_y, yn_p = sp.disg_np[m], 1 / (1 + np.exp(-lg["yes_no"][m] / t.t_yes_no))
    yn_pred = (yn_p >= 0.5).astype(int)
    tp, fp, fn = int(((yn_pred == 1) & (yn_y == 1)).sum()), int(((yn_pred == 1) & (yn_y == 0)).sum()), \
        int(((yn_pred == 0) & (yn_y == 1)).sum())
    yn = {"n": int(m.sum()), "positives": int(yn_y.sum()),
          "accuracy": round(float((yn_pred == yn_y).mean()) if m.any() else 0.0, 4), "auroc": auroc(yn_y, yn_p),
          "precision": round(tp / max(tp + fp, 1), 4), "recall": round(tp / max(tp + fn, 1), 4),
          "ece": ece(np.maximum(yn_p, 1 - yn_p), yn_pred == yn_y) if m.any() else None}
    ps = _sm(lg["score"] / t.t_score)
    pos = sp.pos.cpu().numpy()
    sc = {"mae": round(float(np.abs(ps @ BIN_CENTERS - pos).mean()), 4),
          "bin_accuracy": round(float((ps.argmax(1) == score_targets(sp.pos).cpu().numpy()).mean()), 4),
          "bins": SCORE_BINS}
    return {"choice": ch, "yes_no": yn, "score": sc}


# ---------------------------------------------------------------------------------------------- baselines
def _hist(x: torch.Tensor) -> torch.Tensor:
    h = torch.zeros(len(x), 256, device=x.device)
    h.scatter_add_(1, x.long(), torch.ones_like(x, dtype=torch.float))
    return h / x.shape[1]


def baselines(splits: dict[str, Split], classes: list[str], device: str, steps: int = 600) -> dict:
    tr, te = splits["train"], splits["test"]
    xtr, xte = _hist(tr.x), _hist(te.x)
    mu, sd = xtr.mean(0), xtr.std(0) + 1e-6
    lin = torch.nn.Linear(256, len(classes)).to(device)
    opt = torch.optim.Adam(lin.parameters(), lr=0.05, weight_decay=1e-5)
    cnt = torch.bincount(tr.y, minlength=len(classes)).float()
    cw = (cnt.sum() / (len(classes) * cnt.clamp(min=1))) ** 0.5
    for _ in range(steps):
        opt.zero_grad()
        F.cross_entropy(lin((xtr - mu) / sd), tr.y, weight=cw).backward()
        opt.step()
    with torch.no_grad():
        pred = lin((xte - mu) / sd).argmax(1).cpu().numpy()
    y = te.y.cpu().numpy()
    hist = classification_metrics(y, pred, classes)
    hist.pop("confusion")
    m = te.disg_np >= 0
    rule = np.array([fr.is_disguised(classes[p], "x" + e) for p, e in zip(pred, te.ext_txt)])
    rule = np.where(rule < 0, 0, rule)[m]
    yy = te.disg_np[m]
    # extension alone: disguise rate per extension on one augmented pass over train (shortcut check)
    ext_tr, disg_tr = augment(tr.labels_txt, tr.ext_raw, tr.disg_raw, random.Random(99))
    rate: dict[str, list[int]] = {}
    for e, v in zip(ext_tr, disg_tr):
        if v >= 0:
            rate.setdefault(str(e), []).append(int(v))
    prior = float(np.mean([v for vs in rate.values() for v in vs]))
    ext_only = np.array([np.mean(rate[str(e)]) if str(e) in rate else prior for e in te.ext_txt])[m]
    pos = te.pos.cpu().numpy()
    majority = Counter(y.tolist()).most_common(1)[0][0]
    return {"choice_byte_histogram_logreg": {"accuracy": hist["accuracy"], "macro_f1": hist["macro_f1"],
                                             "per_class_f1": {c: v["f1"] for c, v in hist["per_class"].items()}},
            "choice_majority_class": {"class": classes[majority], "accuracy": round(float((y == majority).mean()), 4)},
            "yes_no_histogram_type_plus_rule": {"accuracy": round(float((rule == yy).mean()), 4) if m.any() else None,
                                                "auroc": auroc(yy, rule.astype(float))},
            "yes_no_extension_only": {"accuracy": round(float(((ext_only >= 0.5) == yy).mean()), 4) if m.any() else None,
                                      "auroc": auroc(yy, ext_only)},
            "score_constant_0.5": {"mae": round(float(np.abs(0.5 - pos).mean()), 4)}}


# ---------------------------------------------------------------------------------------------- report
def write_report(run: Path, m: dict, history: list[dict]) -> Path:
    te, va, b = m["test"], m["val"], m["baselines"]
    classes = m["classes"]
    pc = "\n".join(f"| {c} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | "
                   f"{b['choice_byte_histogram_logreg']['per_class_f1'].get(c, 0):.3f} | {v['support']:,} |"
                   for c, v in te["choice"]["per_class"].items())
    cm = te["choice"]["confusion"]
    head = "| true \\ pred | " + " | ".join(c[:6] for c in classes) + " |\n|---|" + "---|" * len(classes)
    cm_rows = "\n".join(f"| {c} | " + " | ".join(str(n) for n in row) + " |" for c, row in zip(classes, cm))
    curve = "\n".join(f"| {h['epoch']} | {h['train_loss_choice']} | {h['train_loss_yes_no']} | {h['train_loss_score']} "
                      f"| {h['val_choice_acc']} | {h['val_choice_macro_f1']} | {h['val_yes_no_auroc']} | "
                      f"{h['val_score_mae']} |" for h in history)
    cfg, meta = m["config"], m["meta"]
    md = f"""# Jev typed-answer model: training report

Run `{run.name}` · dataset `{meta['dataset']}` · {meta['n_params'] / 1e6:.2f} M parameters · best epoch
{meta.get('best_epoch')} of {meta['epochs']} · {meta['train_seconds']} s · fragments train / val / test:
{m['sizes']['train']:,} / {m['sizes']['val']:,} / {m['sizes']['test']:,} (case-level splits).
Types left out (fewer than {MIN_TRAIN_FRAGMENTS} training fragments): {meta['dropped_classes'] or 'none'}.

Calibration temperatures (validation cases): choice {cfg['t_choice']}, yes/no {cfg['t_yes_no']}, score
{cfg['t_score']}; answers below confidence {cfg['abstain_below']} abstain.

## Test cases: summary

| Query | Metric | Jev | Baseline |
|---|---|---|---|
| choice (file type) | accuracy | **{te['choice']['accuracy']:.3f}** | {b['choice_byte_histogram_logreg']['accuracy']:.3f} (byte-histogram log. reg.); {b['choice_majority_class']['accuracy']:.3f} (majority) |
| choice (file type) | macro-F1 | **{te['choice']['macro_f1']:.3f}** | {b['choice_byte_histogram_logreg']['macro_f1']:.3f} |
| choice | ECE before / after temperature | {te['choice']['ece_uncalibrated']} / **{te['choice']['ece']}** | |
| choice | coverage / accuracy when answering | {te['choice']['coverage_at_threshold']:.3f} / {te['choice']['selective_accuracy']:.3f} | |
| yes/no (disguised) | AUROC | **{te['yes_no']['auroc']}** | {b['yes_no_histogram_type_plus_rule']['auroc']} (histogram type + rule) |
| yes/no (disguised) | AUROC, extension only (no content) | | {b['yes_no_extension_only']['auroc']} |
| yes/no, real extensions only ({m['test_natural_yes_no']['positives']} disguised of {m['test_natural_yes_no']['n']:,}) | AUROC (precision / recall) | **{m['test_natural_yes_no']['auroc']}** ({m['test_natural_yes_no']['precision']:.3f} / {m['test_natural_yes_no']['recall']:.3f}) | |
| yes/no (disguised) | accuracy (precision / recall) | {te['yes_no']['accuracy']:.3f} ({te['yes_no']['precision']:.3f} / {te['yes_no']['recall']:.3f}) | {b['yes_no_histogram_type_plus_rule']['accuracy']} |
| score (position in file) | MAE | **{te['score']['mae']:.3f}** | {b['score_constant_0.5']['mae']:.3f} (constant 0.5) |

Validation: choice accuracy {va['choice']['accuracy']:.3f}, macro-F1 {va['choice']['macro_f1']:.3f}; yes/no AUROC
{va['yes_no']['auroc']}; score MAE {va['score']['mae']:.3f}.

## Test cases: file type per class

| type | precision | recall | F1 | F1 baseline | support |
|---|---|---|---|---|---|
{pc}

## Test confusion matrix (fragments)

{head}
{cm_rows}

## Training curve

| epoch | loss choice | loss yes/no | loss score | val acc | val macro-F1 | val AUROC | val MAE |
|---|---|---|---|---|---|---|---|
{curve}

## Reading these numbers

* Fragments come from the **body** of each file (first sector excluded), so the header signature that produced the
  label is never an input. Compressed containers (zip, gzip, 7z, rar, png, jpeg bodies) are near-uniform random bytes
  and are expected to be confused with one another; the confusion matrix shows where.
* The "disguised" evaluation labels include a seeded share of re-drawn extensions (same procedure as training,
  different seed); natural disguised files are rare in the corpus.
* Jev answers are **model observations**. The JevAgent reports them as UNCORROBORATED leads; the
  HeuristicVerificationEngine never accepts them as evidence.
"""
    path = run / "REPORT.md"
    path.write_text(md, encoding="utf-8")
    return path


def evaluate(checkpoint: Path | None = None, dataset: Path | None = None, device: str | None = None) -> dict:
    """Re-evaluate a saved checkpoint on the val / test splits of a fragment dataset."""
    jev = Jev.load(checkpoint, device)
    d = fr.load(dataset)
    keep = np.isin(d["label"], jev.cfg.classes)
    out = {}
    for i, s in enumerate(("val", "test")):
        sp = Split(d, keep & (d["split"] == s), jev.cfg.classes, jev.device)
        sp.set_ext(*augment(sp.labels_txt, sp.ext_raw, sp.disg_raw, random.Random(jev.cfg.meta.get("seed", 13) + 1 + i)))
        out[s] = evaluate_logits(predict(jev.net, sp), sp, jev.cfg.classes, jev.cfg)
        out[s]["choice"].pop("confusion")
    return out
