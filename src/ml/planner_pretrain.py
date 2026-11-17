"""Self-supervised pretraining of the planner's file encoders on the **unlabelled** READY cases.

    python main.py planner pretrain-build   # every READY case WITHOUT an answer key -> model inputs (no labels)
    python main.py planner pretrain         # cross-modal contrastive training of the path / header / metadata /
                                            # content encoders -> data/models/planner/pretrained_encoders.pt

Why: the planner has only 14 labelled cases (13 per leave-one-case-out fold), and its fusion network does not beat a
metadata-only network on that little data. Most of the corpus (READY cases without a published answer key) is
unused. Here every file supplies its own supervision: the four views of one file (where it lives, how its bytes
start, its metadata, its printable strings) must recognise each other among the other files of the batch:

    z_m = normalize(proj_m(enc_m(x_m)))            m in {path, header, metadata, content}
    L   = mean over pairs (a, b) of symmetric InfoNCE(z_a, z_b; learnable temperature)

(CLIP-style alignment, Radford et al. 2021, applied between the modalities of one file). The encoders keep the
exact architecture of ``PlannerNet.enc`` so the fine-tuning step loads them as they are.

**No leakage:** cases with an answer key (the planner's labelled cases, including every held-out LOCO case) are
excluded from pretraining altogether, not only their labels. Files are de-duplicated by SHA-256 so identical system
files (one header, many paths) do not become contradictory positives.
"""
from __future__ import annotations

import itertools
import json
import math
import random
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.config import settings
from src.config.settings import DATA_DIR
from src.ml import planner_dataset as pd_
from src.ml.planner import PlannerConfig, PlannerNet, encode_path

PRE_MODS = ("path", "header", "metadata", "content")
PAIRS = list(itertools.combinations(PRE_MODS, 2))
PROJ = 128


def pretrain_dir() -> Path:
    return DATA_DIR / "datasets" / "planner_pretrain"


def default_encoders() -> Path:
    return DATA_DIR / "models" / "planner" / "pretrained_encoders.pt"


# ---------------------------------------------------------------------------------------------- data
def unlabelled_cases():
    """READY cases with no answer key (the labelled ones stay out of pretraining entirely)."""
    from src.cases.builder import readiness
    from src.cases.registry import CASES

    labelled = set(pd_.curated_tokens())
    return [c for c in CASES.values() if readiness(c)["ready"] and not pd_._key_text(c.dir).strip()
            and c.case_id not in labelled]


def build(out: Path | None = None, progress: bool = True, encode: bool = True) -> Path:
    out = out or pretrain_dir() / datetime.now().strftime("%Y%m%d")
    out.mkdir(parents=True, exist_ok=True)
    rows, heads, metas, seen = [], [], [], set()
    stats = {"cases": 0, "files": 0, "duplicates": 0, "unreadable": 0}
    for case in unlabelled_cases():
        crow, chead, cmeta, unreadable = pd_.case_rows(case)
        stats["unreadable"] += unreadable
        keep = []
        for i, r in enumerate(crow):
            h = r.get("sha256")
            if h and h in seen:
                stats["duplicates"] += 1
                continue
            if h:
                seen.add(h)
            keep.append(i)
        if not keep:
            continue
        stats["cases"] += 1
        rows += [{"case_id": crow[i]["case_id"], "path_text": crow[i]["path_text"], "category": crow[i]["category"],
                  "snippet": crow[i]["snippet"]} for i in keep]
        heads.append(chead[keep])
        metas.append(cmeta[keep])
        if progress:
            print(f"[pretrain build] {case.case_id}: {len(crow):,} files, {len(keep):,} unique", flush=True)
    stats["files"] = len(rows)
    np.savez_compressed(out / "arrays.npz", head=np.concatenate(heads), meta=np.concatenate(metas),
                        path=np.stack([encode_path(r["path_text"]) for r in rows]).astype(np.uint8))
    with open(out / "files.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    info = {"created": datetime.now().isoformat(timespec="seconds"), **stats,
            "case_ids": sorted({r["case_id"] for r in rows})}
    (out / "build.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    if encode:
        from src.ml.mm_dataset import TextEncoder

        snip = TextEncoder().encode([r["snippet"] for r in rows], batch=128)
        np.savez_compressed(out / "text.npz", snip=snip.astype(np.float16))
    if progress:
        print(f"[pretrain build] {len(rows):,} unique files from {stats['cases']} unlabelled cases -> {out}", flush=True)
    return out


def latest() -> Path:
    dirs = sorted(p for p in pretrain_dir().glob("*") if (p / "text.npz").exists())
    if not dirs:
        raise FileNotFoundError("no pretraining dataset; run `python main.py planner pretrain-build` first")
    return dirs[-1]


class PreData:
    def __init__(self, ds: Path | None = None):
        self.dir = ds or latest()
        rows = [json.loads(x) for x in (self.dir / "files.jsonl").read_text(encoding="utf-8").splitlines() if x]
        with np.load(self.dir / "arrays.npz") as z:
            self.x = {"path": z["path"].astype(np.int64), "header": z["head"].astype(np.int64),
                      "meta": z["meta"].astype(np.float32)}
        with np.load(self.dir / "text.npz") as z:
            self.x["content"] = z["snip"].astype(np.float32)
        self.x["has_content"] = np.array([bool(r["snippet"].strip()) for r in rows])
        self.case = np.array([r["case_id"] for r in rows])
        self.cases = sorted(set(self.case))

    def __len__(self) -> int:
        return len(self.case)

    def tensors(self, idx: np.ndarray, device: str) -> dict[str, torch.Tensor]:
        return {k: torch.from_numpy(np.ascontiguousarray(v[idx])).to(device) for k, v in self.x.items()}


# ---------------------------------------------------------------------------------------------- model
class ContrastiveEncoders(nn.Module):
    """``PlannerNet.enc`` (minus the brief) + one projection head per modality for the contrastive loss."""

    def __init__(self, dim: int = PlannerConfig.dim, dropout: float = 0.1):
        super().__init__()
        base = PlannerNet(PlannerConfig(dim=dim, dropout=dropout))
        self.enc = nn.ModuleDict({m: base.enc[m] for m in PRE_MODS})
        self.proj = nn.ModuleDict({m: nn.Sequential(nn.GELU(), nn.Linear(dim, PROJ)) for m in PRE_MODS})
        self.logit_scale = nn.Parameter(torch.tensor(math.log(1 / 0.07)))

    def forward(self, t: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        raw = {"path": self.enc["path"](t["path"]), "header": self.enc["header"](t["header"] + 1),   # as PlannerNet
               "metadata": self.enc["metadata"](t["meta"]), "content": self.enc["content"](t["content"])}
        return {m: F.normalize(self.proj[m](v), dim=-1) for m, v in raw.items()}


def info_nce(a: torch.Tensor, b: torch.Tensor, scale: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Symmetric InfoNCE; returns (loss, top-1 retrieval accuracy a->b and b->a averaged)."""
    logits = scale * a @ b.t()
    tgt = torch.arange(len(a), device=a.device)
    loss = (F.cross_entropy(logits, tgt) + F.cross_entropy(logits.t(), tgt)) / 2
    acc = ((logits.argmax(1) == tgt).float().mean() + (logits.argmax(0) == tgt).float().mean()) / 2
    return loss, acc


def contrastive_loss(net: ContrastiveEncoders, t: dict) -> tuple[torch.Tensor, dict[str, float]]:
    z = net(t)
    scale = net.logit_scale.exp().clamp(max=100)
    has_c = t["has_content"]
    losses, accs = [], {}
    for a, b in PAIRS:
        m = has_c if "content" in (a, b) else torch.ones_like(has_c)
        if m.sum() < 8:
            continue
        lo, ac = info_nce(z[a][m], z[b][m], scale)
        losses.append(lo)
        accs[f"{a}-{b}"] = float(ac)
    return torch.stack(losses).mean(), accs


def _weights(case: np.ndarray, idx: np.ndarray) -> torch.Tensor:
    """Draw probability ∝ 1/sqrt(case size): big phone dumps do not drown the small Windows cases."""
    sizes = {c: (case[idx] == c).sum() for c in set(case[idx])}
    w = np.array([1 / math.sqrt(sizes[c]) for c in case[idx]])
    return torch.tensor(w / w.sum())


@torch.no_grad()
def evaluate(net: ContrastiveEncoders, data: PreData, idx: np.ndarray, device: str, batch: int, seed: int = 0
             ) -> dict:
    """Mean loss and top-1 cross-modal retrieval accuracy on fixed random batches of ``idx`` (chance = 1/batch)."""
    net.eval()
    rng = np.random.default_rng(seed)
    perm = rng.permutation(idx)
    losses, accs = [], {}
    for i in range(0, len(perm) - batch + 1, batch):
        lo, ac = contrastive_loss(net, data.tensors(perm[i:i + batch], device))
        losses.append(float(lo))
        for k, v in ac.items():
            accs.setdefault(k, []).append(v)
    return {"loss": float(np.mean(losses)) if losses else None,
            "retrieval@1": {k: round(float(np.mean(v)), 4) for k, v in accs.items()}, "batch": batch}


def pretrain(ds: Path | None = None, epochs: int = 20, steps_per_epoch: int = 300, batch: int = 512,
             lr: float = 1e-3, val_cases: int = 6, seed: int = 13, device: str | None = None,
             out: Path | None = None, progress: bool = True) -> dict:
    device = device or settings.device_for("orchestrator")
    torch.manual_seed(seed)
    np.random.seed(seed)
    data = PreData(ds)
    va_cases = random.Random(seed).sample(data.cases, min(val_cases, len(data.cases) - 1))
    va = np.flatnonzero(np.isin(data.case, va_cases))
    tr = np.flatnonzero(~np.isin(data.case, va_cases))
    net = ContrastiveEncoders().to(device)
    before = evaluate(net, data, va, device, batch)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=0.05)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=lr, total_steps=epochs * steps_per_epoch, pct_start=0.1)
    w = _weights(data.case, tr)
    best, best_state, best_ep, hist = math.inf, None, 0, []
    t0 = time.perf_counter()
    for ep in range(1, epochs + 1):
        net.train()
        draw = tr[torch.multinomial(w, steps_per_epoch * batch, replacement=True).numpy()]
        tl = []
        for s in range(steps_per_epoch):
            ii = np.unique(draw[s * batch:(s + 1) * batch])       # a file twice in a batch = a false negative
            loss, _ = contrastive_loss(net, data.tensors(ii, device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            tl.append(float(loss))
        ev = evaluate(net, data, va, device, batch)
        hist.append({"epoch": ep, "train_loss": round(float(np.mean(tl)), 4), "val_loss": round(ev["loss"], 4),
                     "val_retrieval@1": ev["retrieval@1"]})
        if progress:
            print(f"[pretrain] epoch {ep}/{epochs} train {np.mean(tl):.3f} val {ev['loss']:.3f} "
                  f"retrieval@1 {ev['retrieval@1']} ({time.perf_counter() - t0:.0f} s)", flush=True)
        if ev["loss"] < best:
            best, best_ep = ev["loss"], ep
            best_state = {k: v.detach().clone() for k, v in net.state_dict().items()}
    net.load_state_dict(best_state)
    after = evaluate(net, data, va, device, batch)
    dest = default_encoders()
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"modalities": PRE_MODS, "dim": PlannerConfig.dim, "dataset": str(data.dir), "best_epoch": best_ep,
                "enc": {m: {k: v.cpu() for k, v in net.enc[m].state_dict().items()} for m in PRE_MODS}}, dest)
    run = out or settings.OUTPUTS_DIR / "ml" / "planner_pretrain" / datetime.now().strftime("%Y%m%d_%H%M%S")
    run.mkdir(parents=True, exist_ok=True)
    metrics = {"dataset": str(data.dir), "files": len(data), "cases": len(data.cases), "train_files": int(len(tr)),
               "val_cases": va_cases, "val_files": int(len(va)), "epochs": epochs, "steps_per_epoch": steps_per_epoch,
               "batch": batch, "lr": lr, "best_epoch": best_ep, "before": before, "after": after, "history": hist,
               "checkpoint": str(dest)}
    (run / "metrics.json").write_text(json.dumps(metrics, indent=1), encoding="utf-8")
    write_report(run, metrics)
    return {"run": str(run), "checkpoint": str(dest), "metrics": metrics}


def write_report(run: Path, m: dict) -> Path:
    pairs = list(m["after"]["retrieval@1"])
    rows = "\n".join(f"| {p} | {m['before']['retrieval@1'].get(p, 0):.3f} | {m['after']['retrieval@1'][p]:.3f} |"
                     for p in pairs)
    md = f"""# Planner encoders: self-supervised cross-modal pretraining

Run `{run.name}` · dataset `{m['dataset']}` · {m['files']:,} unique files from {m['cases']} READY cases **without**
an answer key (the labelled planner cases are excluded entirely). Validation = {len(m['val_cases'])} whole cases
never trained on ({m['val_files']:,} files): {', '.join(m['val_cases'])}.

Objective: symmetric InfoNCE between every pair of the four views of a file (path, header bytes, metadata, content
strings), batch {m['batch']}, {m['epochs']} × {m['steps_per_epoch']} steps, AdamW lr {m['lr']}; best epoch by
validation loss: {m['best_epoch']}. Validation loss {m['before']['loss']:.3f} (random init) → {m['after']['loss']:.3f}.

## Validation cross-modal retrieval (top-1 among {m['batch']} files; chance = {1 / m['batch']:.4f})

| pair | random init | pretrained |
|---|---|---|
{rows}

The encoders (not the projection heads) are saved to `{m['checkpoint']}` and loaded by the planner variants
`… + SSL` in `python main.py planner train`.
"""
    p = run / "REPORT.md"
    p.write_text(md, encoding="utf-8")
    return p


def load_into(net: PlannerNet, path: Path | None = None, freeze: bool = False) -> list[str]:
    """Copy the pretrained encoders into ``net.enc``; returns the modalities loaded."""
    ck = torch.load(path or default_encoders(), map_location="cpu", weights_only=False)
    if ck["dim"] != net.cfg.dim:
        raise ValueError(f"pretrained dim {ck['dim']} != planner dim {net.cfg.dim}")
    for m, sd in ck["enc"].items():
        net.enc[m].load_state_dict(sd)
        if freeze:
            for p in net.enc[m].parameters():
                p.requires_grad_(False)
    return list(ck["enc"])
