"""Train, calibrate and evaluate Jev-MM (``src/ml/multimodal.py``), with modality ablations.

    python main.py mm train [--epochs 12] [--ablations]      # full model (+ one model per modality subset)
    python main.py mm eval  [--checkpoint ...]

Protocol (same as Jev): train on the train cases, choose the epoch and fit the calibration temperatures on the
validation cases, report once on the test cases (case-level splits). Each ablation is the same network trained
with only some modalities. Robustness: the full model is also tested with modalities withheld at inference
(e.g. EXIF stripped, no readable text), which modality dropout during training prepares it for.

Outputs: ``data/outputs/ml/multimodal/<run>/{metrics.json, history.json, REPORT.md}`` and the checkpoint
``data/models/multimodal/jev_mm.pt`` (trained parameters only; the frozen CLIP weights come from the HF cache).
"""
from __future__ import annotations

import json
import math
import shutil
import time
from collections import Counter
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from src.config import settings
from src.ml import mm_dataset as mmd
from src.ml.multimodal import MODALITIES, JevMM, JevMMNet, MMConfig, default_checkpoint
from src.ml.train import auroc, classification_metrics, ece, fit_temperature

ABLATIONS = {"vision": ("vision",), "metadata": ("metadata",), "text": ("text",),
             "vision+metadata": ("vision", "metadata"), "all": MODALITIES}
MODALITY_DROPOUT = 0.25
LOSS_W = {"choice": 1.0, "yes_no": 0.5}
WITHHELD = {"all modalities": (), "no text": ("text",), "no metadata (EXIF stripped)": ("metadata",),
            "vision only": ("metadata", "text"), "no vision": ("vision",)}


# ---------------------------------------------------------------------------------------------- data
class MMData:
    """All splits in memory: thumbnails as one uint8 CPU tensor, the rest as numpy arrays."""

    def __init__(self, ds: Path | None = None, classes: list[str] | None = None):
        d = mmd.load(ds)
        if "text_emb" not in d:
            raise SystemExit("text features missing: run `python main.py mm ocr` and `python main.py mm encode`")
        self.dir, self.rows = d["dir"], d["rows"]
        self.classes = classes or list(mmd.ORIGINS)
        self.thumbs = torch.from_numpy(np.array(d["thumbs"]))          # copy the memmap into RAM
        self.meta = d["meta"].astype(np.float32)
        self.text = np.concatenate([d["text_emb"], d["text_stats"]], 1).astype(np.float32)
        self.present = np.stack([np.ones(len(self.rows), bool), np.ones(len(self.rows), bool),
                                 d["text_stats"][:, 0] > 0], 1)
        self.y = np.array([self.classes.index(r["origin"]) for r in self.rows])
        self.ug = np.array([r["user_generated"] for r in self.rows], dtype=np.float32)
        self.split = np.array([r["split"] for r in self.rows])
        self.platform = np.array([r["platform"] for r in self.rows])

    def idx(self, split: str) -> np.ndarray:
        return np.flatnonzero(self.split == split)

    def batch(self, ii: np.ndarray, device: str, withhold: tuple[str, ...] = ()):
        present = self.present[ii].copy()
        for m in withhold:
            present[:, MODALITIES.index(m)] = False
        return (self.thumbs[ii].to(device, non_blocking=True), torch.from_numpy(self.meta[ii]).to(device),
                torch.from_numpy(self.text[ii]).to(device), torch.from_numpy(present).to(device))


def _modality_dropout(present: torch.Tensor, p: float) -> torch.Tensor:
    drop = torch.rand_like(present, dtype=torch.float) < p
    out = present & ~drop
    empty = out.sum(1) == 0
    out[empty] = present[empty]                      # never drop every available modality
    return out


@torch.no_grad()
def predict(net: JevMMNet, data: MMData, ii: np.ndarray, device: str, withhold=(), batch: int = 256):
    net.eval()
    ch, yn = [], []
    for s in range(0, len(ii), batch):
        th, me, tx, pr = data.batch(ii[s:s + batch], device, withhold)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
            o = net(th, me, tx, pr)
        ch.append(o["choice"].float().cpu().numpy())
        yn.append(o["yes_no"].float().cpu().numpy())
    return {"choice": np.concatenate(ch), "yes_no": np.concatenate(yn)}


def _sm(z):
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


def evaluate_logits(lg: dict, y: np.ndarray, ug: np.ndarray, classes: list[str], cfg: MMConfig | None = None) -> dict:
    t = cfg or MMConfig(classes=classes)
    p = _sm(lg["choice"] / t.t_choice)
    pred, conf = p.argmax(1), p.max(1)
    ch = classification_metrics(y, pred, classes)
    ch["ece_uncalibrated"] = ece(_sm(lg["choice"]).max(1), _sm(lg["choice"]).argmax(1) == y)
    ch["ece"] = ece(conf, pred == y)
    keep = conf >= t.abstain_below
    ch["coverage_at_threshold"] = round(float(keep.mean()), 4)
    ch["selective_accuracy"] = round(float((pred == y)[keep].mean()) if keep.any() else 0.0, 4)
    q = 1 / (1 + np.exp(-lg["yes_no"] / t.t_yes_no))
    qp = (q >= 0.5).astype(int)
    tp, fp, fn = int(((qp == 1) & (ug == 1)).sum()), int(((qp == 1) & (ug == 0)).sum()), int(((qp == 0) & (ug == 1)).sum())
    yn = {"n": int(len(ug)), "positives": int(ug.sum()), "accuracy": round(float((qp == ug).mean()), 4),
          "auroc": auroc(ug.astype(int), q), "precision": round(tp / max(tp + fp, 1), 4),
          "recall": round(tp / max(tp + fn, 1), 4), "f1": round(2 * tp / max(2 * tp + fp + fn, 1), 4),
          "ece": ece(np.maximum(q, 1 - q), qp == ug)}
    return {"choice": ch, "yes_no": yn}


# ---------------------------------------------------------------------------------------------- training
def train_one(data: MMData, modalities: tuple[str, ...], epochs: int = 12, batch: int = 128, lr: float = 1e-3,
              vit_lr: float = 1e-5, seed: int = 13, device: str | None = None, progress: bool = True,
              label: str = "all", unfreeze_blocks: int = 1, pretrained: bool = True, balance: float = 0.5
              ) -> tuple[JevMMNet, MMConfig, dict, list]:
    """``balance``: class-balanced sampling exponent (weight = class count ** -balance; 0.5 is the original
    recipe, 1.0 samples every class equally often)."""
    device = device or settings.device_for("vision")
    torch.manual_seed(seed)
    np.random.seed(seed)
    tr, va, te = data.idx("train"), data.idx("val"), data.idx("test")
    if not len(tr) or not len(va) or not len(te):
        raise SystemExit(f"empty split (train={len(tr)}, val={len(va)}, test={len(te)})")
    cfg = MMConfig(classes=data.classes, modalities=tuple(modalities), unfreeze_blocks=unfreeze_blocks,
                   meta={"dataset": str(data.dir), "seed": seed, "epochs": epochs, "ablation": label,
                         "trained_utc": datetime.now().isoformat(timespec="seconds")})
    net = JevMMNet(cfg, pretrained=pretrained).to(device)
    vit_params = [p for n, p in net.named_parameters() if n.startswith("vit.") and p.requires_grad]
    other = [p for n, p in net.named_parameters() if not n.startswith("vit.") and p.requires_grad]
    groups = [{"params": other, "lr": lr}] + ([{"params": vit_params, "lr": vit_lr}] if vit_params else [])
    opt = torch.optim.AdamW(groups, weight_decay=0.05)
    steps = epochs * math.ceil(len(tr) / batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in groups], total_steps=steps,
                                                pct_start=0.1)
    cnt = np.bincount(data.y[tr], minlength=len(data.classes)).astype(float)
    w = torch.tensor(cnt[data.y[tr]] ** -balance)
    n_train = sum(p.numel() for p in net.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in net.parameters())
    if progress:
        print(f"[mm train:{label}] modalities={'+'.join(modalities)} train/val/test = {len(tr):,}/{len(va):,}/"
              f"{len(te):,}; {n_train / 1e6:.2f} M trainable of {n_all / 1e6:.1f} M parameters; device {device}",
              flush=True)
    history, best, best_state, t0 = [], -1.0, None, time.perf_counter()
    for ep in range(1, epochs + 1):
        net.train()
        order = tr[torch.multinomial(w, len(tr), replacement=True).numpy()]
        sums: Counter = Counter()
        for s in range(0, len(order), batch):
            ii = order[s:s + batch]
            th, me, tx, pr = data.batch(ii, device)
            if torch.rand(()) < 0.5:
                th = th.flip(2)                       # horizontal flip (whole batch)
            pr = _modality_dropout(pr, MODALITY_DROPOUT)
            y = torch.from_numpy(data.y[ii]).to(device)
            ug = torch.from_numpy(data.ug[ii]).to(device)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                out = net(th, me, tx, pr)
            ls = {"choice": F.cross_entropy(out["choice"].float(), y, label_smoothing=0.05),
                  "yes_no": F.binary_cross_entropy_with_logits(out["yes_no"].float(), ug)}
            loss = sum(LOSS_W[k] * v for k, v in ls.items())
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
            opt.step()
            sched.step()
            for k, v in ls.items():
                sums[k] += float(v.detach()) * len(ii)
        vm = evaluate_logits(predict(net, data, va, device), data.y[va], data.ug[va], data.classes)
        row = {"epoch": ep, **{f"train_loss_{k}": round(v / len(order), 4) for k, v in sums.items()},
               "val_acc": vm["choice"]["accuracy"], "val_macro_f1": vm["choice"]["macro_f1"],
               "val_ug_auroc": vm["yes_no"]["auroc"], "seconds": round(time.perf_counter() - t0, 1)}
        history.append(row)
        if progress:
            print(f"[mm train:{label}] epoch {ep}/{epochs} " + " ".join(f"{k}={v}" for k, v in row.items()
                                                                       if k != "epoch"), flush=True)
        sel = vm["choice"]["macro_f1"] + 0.25 * (vm["yes_no"]["auroc"] or 0)
        if sel > best:
            best = sel
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
            cfg.meta["best_epoch"] = ep
    net.load_state_dict(best_state)
    lv = predict(net, data, va, device)
    cfg.t_choice = fit_temperature(lv["choice"], data.y[va])
    cfg.t_yes_no = fit_temperature(lv["yes_no"], data.ug[va], binary=True)
    cfg.meta.update(n_params_trainable=n_train, n_params_total=n_all,
                    train_seconds=round(time.perf_counter() - t0, 1))
    net.cfg = cfg
    return net, cfg, {"val": evaluate_logits(lv, data.y[va], data.ug[va], data.classes, cfg)}, history


def test_metrics(net: JevMMNet, cfg: MMConfig, data: MMData, device: str, robustness: bool = False) -> dict:
    te = data.idx("test")
    lg = predict(net, data, te, device)
    out = {"test": evaluate_logits(lg, data.y[te], data.ug[te], data.classes, cfg)}
    pred = _sm(lg["choice"]).argmax(1)
    out["test_by_platform"] = {p: {"n": int(m.sum()), "accuracy": round(float((pred[m] == data.y[te][m]).mean()), 4)}
                               for p in sorted(set(data.platform[te])) if (m := data.platform[te] == p).sum() >= 20}
    has_text = data.present[te, 2]
    out["test_text_bearing"] = {"n": int(has_text.sum()),
                                "accuracy": round(float((pred[has_text] == data.y[te][has_text]).mean()), 4)
                                if has_text.any() else None}
    if robustness:
        out["robustness"] = {}
        for name, wh in WITHHELD.items():
            wh = tuple(m for m in wh if m in cfg.modalities)
            r = evaluate_logits(predict(net, data, te, device, withhold=wh), data.y[te], data.ug[te], data.classes, cfg)
            out["robustness"][name] = {"accuracy": r["choice"]["accuracy"], "macro_f1": r["choice"]["macro_f1"],
                                       "ug_auroc": r["yes_no"]["auroc"]}
        with torch.no_grad():
            att = []
            for s in range(0, len(te), 256):
                th, me, tx, pr = data.batch(te[s:s + 256], device)
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
                    att.append(net(th, me, tx, pr, attn=True)["attention"].float().cpu().numpy())
        att = np.concatenate(att)
        out["attention_by_class"] = {c: {m: round(float(att[data.y[te] == i, j].mean()), 3)
                                         for j, m in enumerate(MODALITIES)}
                                     for i, c in enumerate(data.classes) if (data.y[te] == i).any()}
    return out


# chosen on the VALIDATION cases (sweep over unfrozen ViT blocks 0 / 1 / 2 and the ViT learning rate, README §6.6)
DEFAULT_UNFREEZE, DEFAULT_VIT_LR = 1, 1e-5
SEEDS = (13, 14, 15)


def _summary(r: dict) -> dict:
    return {"accuracy": r["test"]["choice"]["accuracy"], "macro_f1": r["test"]["choice"]["macro_f1"],
            "ug_auroc": r["test"]["yes_no"]["auroc"], "ug_f1": r["test"]["yes_no"]["f1"],
            "ece": r["test"]["choice"]["ece"], "val_macro_f1": r["val"]["choice"]["macro_f1"]}


def train(dataset: Path | None = None, epochs: int = 12, batch: int = 128, lr: float = 1e-3,
          vit_lr: float = DEFAULT_VIT_LR, seed: int = 13, ablations: bool = True, out_dir: Path | None = None,
          device: str | None = None, checkpoint: Path | None = None, progress: bool = True,
          unfreeze_blocks: int = DEFAULT_UNFREEZE, seeds: tuple[int, ...] | None = None) -> dict:
    """Every ablation is trained once per seed; the report gives mean +- std over seeds and the full detail of the
    first seed. The deployed checkpoint is the full model of the first seed (nothing is selected on test)."""
    device = device or settings.device_for("vision")
    seeds = tuple(seeds or (seed,))
    data = MMData(dataset)
    run = out_dir or settings.OUTPUTS_DIR / "ml" / "multimodal" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    te = data.idx("test")
    majority = Counter(data.y[data.idx("train")].tolist()).most_common(1)[0][0]
    metrics: dict = {"classes": data.classes, "dataset": str(data.dir), "seeds": list(seeds),
                     "sizes": {s: int((data.split == s).sum()) for s in ("train", "val", "test")},
                     "text_bearing": {s: int(data.present[data.split == s, 2].sum()) for s in ("train", "val", "test")},
                     "baselines": {"majority_class": {"class": data.classes[majority],
                                                      "accuracy": round(float((data.y[te] == majority).mean()), 4)},
                                   "ug_prior": {"positive_rate_test": round(float(data.ug[te].mean()), 4)}},
                     "ablations": {}}
    histories = {}
    for label, mods in (ABLATIONS.items() if ablations else [("all", MODALITIES)]):
        per_seed = []
        for k, sd in enumerate(seeds):
            net, cfg, val, hist = train_one(data, mods, epochs, batch, lr, vit_lr, sd, device, progress,
                                            f"{label}/seed{sd}", unfreeze_blocks)
            res = test_metrics(net, cfg, data, device, robustness=(label == "all" and k == 0))
            per_seed.append({"seed": sd, **_summary({**val, **res})})
            if k == 0:
                metrics["ablations"][label] = {"modalities": list(mods), "best_epoch": cfg.meta["best_epoch"],
                                               "train_seconds": cfg.meta["train_seconds"],
                                               "temperatures": {"choice": cfg.t_choice, "yes_no": cfg.t_yes_no},
                                               **val, **res}
                histories[label] = hist
                if label == "all":
                    metrics["config"] = {k2: v for k2, v in vars(cfg).items() if k2 != "meta"}
                    metrics["meta"] = cfg.meta
                    ck = JevMM.save(net, run / "jev_mm.pt")
                    dest = checkpoint or default_checkpoint()
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copyfile(ck, dest)
                    metrics["checkpoint"] = str(dest)
            del net
            if device.startswith("cuda"):
                torch.cuda.empty_cache()
        keys = [k2 for k2 in per_seed[0] if k2 != "seed"]
        metrics["ablations"][label]["seeds"] = per_seed
        metrics["ablations"][label]["mean"] = {k2: round(float(np.mean([r[k2] for r in per_seed])), 4) for k2 in keys}
        metrics["ablations"][label]["std"] = {k2: round(float(np.std([r[k2] for r in per_seed])), 4) for k2 in keys}
    (run / "metrics.json").write_text(json.dumps(metrics, indent=2, default=str), encoding="utf-8")
    (run / "history.json").write_text(json.dumps(histories, indent=2), encoding="utf-8")
    write_report(run, metrics, histories)
    return {"run": str(run), "checkpoint": metrics.get("checkpoint"), "metrics": metrics}


def evaluate(checkpoint: Path | None = None, dataset: Path | None = None, device: str | None = None) -> dict:
    jev = JevMM.load(checkpoint, device)
    data = MMData(dataset, jev.cfg.classes)
    res = test_metrics(jev.net, jev.cfg, data, jev.device, robustness=True)
    res["test"]["choice"].pop("confusion", None)
    return res


# ---------------------------------------------------------------------------------------------- report
def write_report(run: Path, m: dict, histories: dict) -> Path:
    ab = m["ablations"]
    full = ab.get("all")
    classes = m["classes"]
    rows = "\n".join(
        f"| {k} | {'+'.join(v['modalities'])} | {v['test']['choice']['accuracy']:.3f} | "
        f"{v['test']['choice']['macro_f1']:.3f} | {v['test']['yes_no']['auroc']} | {v['test']['yes_no']['f1']:.3f} | "
        f"{v['test']['choice']['ece']} | {v['best_epoch']} |" for k, v in ab.items())
    md = [f"# Jev-MM multimodal image-triage model: training report\n",
          f"Run `{run.name}` · dataset `{m['dataset']}` · images train / val / test: {m['sizes']['train']:,} / "
          f"{m['sizes']['val']:,} / {m['sizes']['test']:,} (case-level splits) · with OCR text: "
          f"{m['text_bearing']['train']:,} / {m['text_bearing']['val']:,} / {m['text_bearing']['test']:,}.\n",
          "Queries: **choice** = origin of the image (7 classes), **yes/no** = user-generated content. The storage "
          "path that defines the label is never an input.\n",
          "## Test cases: modality ablation, mean +- std over seeds " + ", ".join(map(str, m.get("seeds", []))) + "\n",
          "| model | modalities | accuracy | macro-F1 | user-gen. AUROC | user-gen. F1 | val macro-F1 |",
          "|---|---|---|---|---|---|---|",
          "\n".join(f"| {k} | {'+'.join(v['modalities'])} | " + " | ".join(
              f"{v['mean'][x]:.3f} +- {v['std'][x]:.3f}" for x in ("accuracy", "macro_f1", "ug_auroc", "ug_f1",
                                                                  "val_macro_f1")) + " |"
                    for k, v in ab.items() if "mean" in v),
          f"| majority class | none | {m['baselines']['majority_class']['accuracy']:.3f} | | 0.5 | | |\n",
          "## Test cases: modality ablation, first seed (detail)\n",
          "| model | modalities | accuracy | macro-F1 | user-gen. AUROC | user-gen. F1 | ECE (calibrated) | best epoch |",
          "|---|---|---|---|---|---|---|---|", rows,
          f"| majority class | none | {m['baselines']['majority_class']['accuracy']:.3f} | | 0.5 | | | |\n"]
    if full:
        meta = m["meta"]
        cfg = m["config"]
        pc = "\n".join(f"| {c} | {v['precision']:.3f} | {v['recall']:.3f} | {v['f1']:.3f} | "
                       + " | ".join(f"{ab[k]['test']['choice']['per_class'][c]['f1']:.3f}" for k in ab if k != "all")
                       + f" | {v['support']:,} |" for c, v in full["test"]["choice"]["per_class"].items())
        others = [k for k in ab if k != "all"]
        cm = full["test"]["choice"]["confusion"]
        head = "| true \\ pred | " + " | ".join(c[:9] for c in classes) + " |\n|---|" + "---|" * len(classes)
        cm_rows = "\n".join(f"| {c} | " + " | ".join(str(n) for n in row) + " |" for c, row in zip(classes, cm))
        rob = "\n".join(f"| {k} | {v['accuracy']:.3f} | {v['macro_f1']:.3f} | {v['ug_auroc']} |"
                        for k, v in full["robustness"].items())
        att = "\n".join(f"| {c} | " + " | ".join(f"{a[x]:.2f}" for x in MODALITIES) + " |"
                        for c, a in full["attention_by_class"].items())
        plat = ", ".join(f"{p} {v['accuracy']:.3f} (n={v['n']})" for p, v in full["test_by_platform"].items())
        curve = "\n".join(f"| {h['epoch']} | {h['train_loss_choice']} | {h['train_loss_yes_no']} | {h['val_acc']} | "
                          f"{h['val_macro_f1']} | {h['val_ug_auroc']} |" for h in histories["all"])
        md += [
            "## Full model (vision + metadata + text)\n",
            f"{meta['n_params_trainable'] / 1e6:.2f} M trainable of {meta['n_params_total'] / 1e6:.1f} M parameters "
            f"(CLIP ViT-B/32, last {cfg['unfreeze_blocks']} blocks fine-tuned; fusion transformer {cfg['layers']} "
            f"layers x {cfg['heads']} heads, d={cfg['dim']}); best epoch {full['best_epoch']} of {meta['epochs']}; "
            f"{full['train_seconds']} s. Temperatures (validation): choice {cfg['t_choice']}, yes/no "
            f"{cfg['t_yes_no']}; answers below {cfg['abstain_below']} abstain (coverage "
            f"{full['test']['choice']['coverage_at_threshold']:.3f}, accuracy when answering "
            f"{full['test']['choice']['selective_accuracy']:.3f}). ECE before / after calibration: "
            f"{full['test']['choice']['ece_uncalibrated']} / {full['test']['choice']['ece']}.\n",
            f"User-generated (test): AUROC {full['test']['yes_no']['auroc']}, precision "
            f"{full['test']['yes_no']['precision']:.3f}, recall {full['test']['yes_no']['recall']:.3f} "
            f"({full['test']['yes_no']['positives']} positives of {full['test']['yes_no']['n']:,}).\n",
            f"Accuracy by platform (test): {plat}. Images with OCR text: accuracy "
            f"{full['test_text_bearing']['accuracy']} (n={full['test_text_bearing']['n']}).\n",
            "### Per class (test), F1 of every ablation\n",
            "| origin | precision | recall | F1 (all) | " + " | ".join(f"F1 ({k})" for k in others) + " | support |",
            "|---|---|---|---|" + "---|" * len(others) + "---|", pc, "",
            "### Confusion matrix (test)\n", head, cm_rows, "",
            "### Robustness: modalities withheld at inference (same full model)\n",
            "| withheld | accuracy | macro-F1 | user-gen. AUROC |", "|---|---|---|---|", rob, "",
            "### Attention of the fusion CLS token per modality (test mean, by true class)\n",
            "| origin | " + " | ".join(MODALITIES) + " |", "|---|---|---|---|", att, "",
            "### Training curve\n", "| epoch | loss choice | loss yes/no | val acc | val macro-F1 | val AUROC |",
            "|---|---|---|---|---|---|", curve, ""]
    md += ["## Reading these numbers\n",
           "* The label is the storage location; content and metadata are only evidence of it, so some confusions "
           "are expected (a camera photo copied to Downloads is a `user_file`; a web image saved by an app is an "
           "`app_asset`).",
           "* Messaging media and screenshots are small classes in the corpus (see the datasheet); their test F1 "
           "carries wide uncertainty.",
           "* Jev-MM answers are **model observations**. The ImageTriageAgent reports them as UNCORROBORATED leads "
           "and uses them to order images for the vision agent; the HeuristicVerificationEngine never accepts "
           "them as evidence."]
    path = run / "REPORT.md"
    path.write_text("\n".join(md) + "\n", encoding="utf-8")
    return path


# ---------------------------------------------------------------------------------------------- improvement
# Fixed BEFORE running (2026-11-17): recipes compared on the VALIDATION cases only; the chosen recipe's seed
# ensemble is tested once. Nothing is chosen on the test cases.
RECIPES = {"original (12 ep, sqrt balance)": {"epochs": 12, "balance": 0.5},
           "full class balance (12 ep)": {"epochs": 12, "balance": 1.0},
           "longer (20 ep, sqrt balance)": {"epochs": 20, "balance": 0.5}}


def _val_score(m: dict) -> float:
    """The epoch-selection criterion of ``train_one`` (val macro-F1 + 0.25 user-generated AUROC)."""
    return m["choice"]["macro_f1"] + 0.25 * (m["yes_no"]["auroc"] or 0)


def logreg_baseline(data: MMData, device: str = "cpu") -> dict:
    """Multinomial / binary L2 logistic regression on metadata and on metadata + text features (class-balanced,
    standardised on train; the L2 strength chosen on validation macro-F1). Vision has no fixed feature vector, so
    it is not included."""
    tr, va, te = data.idx("train"), data.idx("val"), data.idx("test")
    k = len(data.classes)

    def fit(X, t, n_out, wd):
        X = torch.tensor(X, dtype=torch.float32)
        t = torch.tensor(t)
        W = torch.zeros(X.shape[1], n_out, requires_grad=True)
        b = torch.zeros(n_out, requires_grad=True)
        cnt = torch.bincount(t[tr].long(), minlength=n_out).float() if n_out > 1 else None
        cw = cnt.sum() / (n_out * cnt.clamp(min=1)) if n_out > 1 else None
        opt = torch.optim.LBFGS([W, b], max_iter=300, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            z = X[tr] @ W + b
            loss = (F.cross_entropy(z, t[tr].long(), weight=cw) if n_out > 1
                    else F.binary_cross_entropy_with_logits(z[:, 0], t[tr].float()))
            loss = loss + wd * (W ** 2).sum()
            loss.backward()
            return loss

        opt.step(closure)
        with torch.no_grad():
            return (X @ W + b).numpy()

    out = {}
    for name, X in {"metadata": data.meta, "metadata+text": np.concatenate([data.meta, data.text], 1)}.items():
        X = np.nan_to_num(X.astype(np.float64))
        X = (X - X[tr].mean(0)) / (X[tr].std(0) + 1e-6)
        best = None
        for wd in (1e-4, 1e-3, 1e-2):
            z = fit(X, data.y, k, wd)
            f = classification_metrics(data.y[va], z[va].argmax(1), data.classes)["macro_f1"]
            if best is None or f > best[0]:
                best = (f, wd, z)
        f, wd, z = best
        u = fit(X, data.ug, 1, wd)[:, 0]
        m = classification_metrics(data.y[te], z[te].argmax(1), data.classes)
        out[name] = {"l2": wd, "val_macro_f1": round(f, 4), "accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                     "ug_auroc": auroc(data.ug[te].astype(int), u[te])}
    return out


def improve(dataset: Path | None = None, seeds: tuple[int, ...] = SEEDS, deploy: bool = True,
            out_dir: Path | None = None) -> Path:
    device = settings.device_for("vision")
    data = MMData(dataset)
    va, te = data.idx("val"), data.idx("test")
    run = out_dir or settings.OUTPUTS_DIR / "ml" / "multimodal" / ("improve_" + datetime.now().strftime("%Y%m%d_%H%M%S"))
    run.mkdir(parents=True, exist_ok=True)
    res: dict = {"recipes": {}, "seeds": list(seeds)}
    kept = {}
    for name, r in RECIPES.items():
        nets, cfgs, vlogs, singles = [], [], [], []
        for sd in seeds:
            net, cfg, val, _ = train_one(data, MODALITIES, r["epochs"], seed=sd, device=device, label=f"{name}/{sd}",
                                         unfreeze_blocks=DEFAULT_UNFREEZE, balance=r["balance"])
            vlogs.append(predict(net, data, va, device))
            singles.append(_val_score(evaluate_logits(vlogs[-1], data.y[va], data.ug[va], data.classes)))
            nets.append(net.cpu())
            cfgs.append(cfg)
            torch.cuda.empty_cache()
        ens = {k: np.mean([v[k] for v in vlogs], 0) for k in vlogs[0]}
        ev = evaluate_logits(ens, data.y[va], data.ug[va], data.classes)
        res["recipes"][name] = {**r, "val_single_mean": round(float(np.mean(singles)), 4),
                                "val_ensemble": round(_val_score(ev), 4),
                                "val_ensemble_macro_f1": ev["choice"]["macro_f1"],
                                "val_ensemble_acc": ev["choice"]["accuracy"]}
        kept[name] = (nets, cfgs, ens)
        print(f"[mm improve] {name}: val score single {np.mean(singles):.4f} ensemble {_val_score(ev):.4f}", flush=True)
    chosen = max(res["recipes"], key=lambda n: res["recipes"][n]["val_ensemble"])
    nets, cfgs, ens = kept[chosen]
    cfg = cfgs[0]
    cfg.t_choice = fit_temperature(ens["choice"], data.y[va])
    cfg.t_yes_no = fit_temperature(ens["yes_no"], data.ug[va], binary=True)
    cfg.meta.update(ensemble=len(nets), recipe=chosen, seeds=list(seeds))
    from src.ml.multimodal import JevMMEnsemble

    model = JevMMEnsemble([n.to(device) for n in nets], cfg)
    res["chosen"] = chosen
    res["test"] = test_metrics(model, cfg, data, device, robustness=False)
    res["test"]["test"]["choice"].pop("confusion", None)
    res["logistic_regression"] = logreg_baseline(data)
    ck = JevMM.save_ensemble(nets, cfg, run / "jev_mm_ensemble.pt")
    if deploy:
        dest = default_checkpoint()
        if dest.exists():
            shutil.copyfile(dest, dest.with_name("jev_mm_single_seed.pt"))
        shutil.copyfile(ck, dest)
        res["deployed"] = str(dest)
    (run / "metrics.json").write_text(json.dumps(res, indent=2, default=str), encoding="utf-8")
    t = res["test"]["test"]
    md = ["# Jev-MM improvement: recipe chosen on validation, seed ensemble tested once\n",
          f"Recipes fixed before running; {len(seeds)} seeds each; selection = validation macro-F1 + 0.25 user-gen. "
          "AUROC of the seed ensemble (same criterion as the epoch choice). Test cases used once.\n",
          "| recipe | val (mean single seed) | val (ensemble) | val ens. acc | val ens. macro-F1 |", "|---|---|---|---|---|"]
    md += [f"| {n}{' **(chosen)**' if n == chosen else ''} | {v['val_single_mean']} | {v['val_ensemble']} | "
           f"{v['val_ensemble_acc']} | {v['val_ensemble_macro_f1']} |" for n, v in res["recipes"].items()]
    md += ["", "## Test cases\n", "| model | accuracy | macro-F1 | user-gen. AUROC |", "|---|---|---|---|",
           f"| **Jev-MM ensemble ({chosen})** | {t['choice']['accuracy']:.3f} | {t['choice']['macro_f1']:.3f} | "
           f"{t['yes_no']['auroc']} |",
           "| Jev-MM single seed, original recipe (run 20261115_191432, mean of 3 seeds) | 0.744 | 0.533 | 0.899 |"]
    md += [f"| logistic regression, {n} (L2 {v['l2']} on val) | {v['accuracy']:.3f} | {v['macro_f1']:.3f} | "
           f"{v['ug_auroc']} |" for n, v in res["logistic_regression"].items()]
    md += ["", "Per class F1 (test): " + ", ".join(f"{c} {v['f1']:.3f}" for c, v in t["choice"]["per_class"].items())]
    (run / "REPORT.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    return run / "REPORT.md"


if __name__ == "__main__":
    import sys

    if sys.argv[1:2] == ["improve"]:
        print(improve().read_text(encoding="utf-8"))
