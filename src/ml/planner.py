"""Investigation planner: a multimodal network that learns from solved cases **where to look** in a new one.

For every file of an evidence item it predicts P(the file holds part of the case's solution), conditioned on the
case brief. Trained on the answer keys of solved cases (``src/ml/planner_dataset.py``), evaluated leave-one-case-out
(``src/ml/planner_train.py``).

    path text   (160 chars)  -> char Embedding + 3 x Conv1d -> mean||max -> MLP -> token_p
    header      (256 bytes)  -> byte Embedding + 3 x Conv1d -> mean||max -> MLP -> token_h
    metadata    (N_META)     -> MLP                                        -> token_m
    content     (MPNet 768)  -> MLP                                        -> token_c   (masked when no strings)
    brief       (MPNet 768)  -> MLP                                        -> token_q   (the case context)
    [CLS, p, h, m, c, q] + type embeddings -> 2-layer Transformer encoder -> CLS -> relevance logit

**Anchored variant** (``anchor=True``): logit = head_m(token_m) + g * head(CLS), with g a learnable scalar starting
at 0 (ReZero-style) and an auxiliary loss on head_m. The network starts as the metadata-only model, the strongest
single modality on few labelled cases, and the other modalities can only add a learned correction: a remedy for the
modality imbalance where high-capacity encoders overfit the few training cases and drown the metadata signal.

The brief token lets the same file kind score differently in different investigations (an e-mail store matters in a
data-leakage case, a drone flight log in a drone case). Answers are leads: the planner decides what the other agents
read first; it never produces evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import torch
from torch import nn

from src.config.settings import DATA_DIR
from src.ml.planner_dataset import HEAD, N_META, PATH_CHARS

MODEL_VERSION = "planner-1"
MODALITIES = ("path", "header", "metadata", "content", "context")
TEXT_DIM = 768


def default_checkpoint() -> Path:
    return DATA_DIR / "models" / "planner" / "planner.pt"


@dataclass
class PlannerConfig:
    modalities: tuple[str, ...] = MODALITIES
    dim: int = 192
    heads: int = 4
    layers: int = 2
    dropout: float = 0.2
    aux_heads: bool = False          # one auxiliary head per modality (gradient-blending-style supervision)
    anchor: bool = False             # metadata-anchored residual fusion (see module doc)
    header_jev: bool = False         # transfer learning: header encoder = the Jev byte CNN (src/ml/jev.py) + projection
    wide: bool = False               # wide & deep: linear metadata model (init. from logistic regression) + g * fusion
    modality_dropout: float = 0.15
    n_meta: int = N_META
    temperature: float = 1.0
    version: str = MODEL_VERSION
    meta: dict = field(default_factory=dict)


def encode_path(s: str) -> np.ndarray:
    b = s.lower().encode("ascii", "replace")[-PATH_CHARS:]
    a = np.frombuffer(b, dtype=np.uint8).clip(0, 127).astype(np.int64) + 1
    return np.pad(a, (PATH_CHARS - len(a), 0))          # right-aligned: the file name is always at the end


class SeqCNN(nn.Module):
    def __init__(self, vocab: int, emb: int, out: int, p: float):
        super().__init__()
        self.emb = nn.Embedding(vocab, emb, padding_idx=0)
        self.conv = nn.Sequential(nn.Conv1d(emb, 128, 5, padding=2), nn.GELU(), nn.MaxPool1d(2),
                                  nn.Conv1d(128, 128, 5, padding=2), nn.GELU(), nn.MaxPool1d(2),
                                  nn.Conv1d(128, 128, 3, padding=1), nn.GELU())
        self.proj = nn.Sequential(nn.Dropout(p), nn.Linear(256, out))

    def forward(self, x):
        h = self.conv(self.emb(x).transpose(1, 2))
        return self.proj(torch.cat([h.mean(-1), h.amax(-1)], 1))


class JevHeader(nn.Module):
    """Transfer learning for the header modality: the convolutional trunk of the Jev byte model (trained on 140k
    file fragments for file-type recognition) + a new projection. ``load_jev`` copies the trained weights."""

    def __init__(self, out: int):
        super().__init__()
        from src.ml.jev import ByteEncoder, JevConfig

        self.trunk = ByteEncoder(JevConfig(classes=[]))
        self.proj = nn.Linear(self.trunk.proj[1].out_features, out)

    def forward(self, x):                      # x = header + 1 (0 = padding) -> Jev byte ids (padding -> 0x00)
        return self.proj(self.trunk((x - 1).clamp(min=0)))


def load_jev(net: "PlannerNet", path: Path | None = None) -> list[str]:
    """Copy the Jev encoder weights into ``net.enc['header'].trunk``; returns the modalities initialised."""
    from src.ml.jev import default_checkpoint as jev_ckpt

    ck = torch.load(path or jev_ckpt(), map_location="cpu", weights_only=False)
    sd = {k[len("encoder."):]: v for k, v in ck["state_dict"].items() if k.startswith("encoder.")}
    net.enc["header"].trunk.load_state_dict(sd)
    return ["header"]


def _mlp(i, o, p):
    return nn.Sequential(nn.LayerNorm(i), nn.Linear(i, 256), nn.GELU(), nn.Dropout(p), nn.Linear(256, o))


class PlannerNet(nn.Module):
    def __init__(self, cfg: PlannerConfig):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.enc = nn.ModuleDict({"path": SeqCNN(129, 48, d, cfg.dropout), "header": SeqCNN(257, 32, d, cfg.dropout),
                                  "metadata": _mlp(cfg.n_meta, d, cfg.dropout),
                                  "content": _mlp(TEXT_DIM, d, cfg.dropout), "context": _mlp(TEXT_DIM, d, cfg.dropout)})
        if cfg.header_jev:
            self.enc["header"] = JevHeader(d)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.type_emb = nn.Parameter(torch.randn(1, len(MODALITIES), d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.heads, 4 * d, cfg.dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.out = nn.Linear(d, 1)
        self.aux = nn.ModuleDict({m: nn.Linear(d, 1) for m in MODALITIES}) if cfg.aux_heads else None
        if cfg.wide:
            self.wide = nn.Linear(cfg.n_meta, 1)
            self.gate = nn.Parameter(torch.zeros(1))
        elif cfg.anchor:
            self.anchor_head = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, 1))
            self.gate = nn.Parameter(torch.zeros(1))
        self.use = torch.tensor([m in cfg.modalities for m in MODALITIES])

    def forward(self, path, header, meta, content, context, present, attn: bool = False):
        """path (B,160) long; header (B,256) long; meta (B,n); content/context (B,768); present (B,5) bool."""
        b = path.shape[0]
        toks = torch.stack([self.enc["path"](path), self.enc["header"](header + 1), self.enc["metadata"](meta),
                            self.enc["content"](content), self.enc["context"](context)], 1) + self.type_emb
        keep = present & self.use.to(present.device)
        keep[keep.sum(1) == 0, 2] = True
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=path.device), ~keep], 1)
        seq = torch.cat([self.cls.expand(b, -1, -1), toks], 1)
        h = self.norm(self.fusion(seq, src_key_padding_mask=pad))[:, 0]
        out = {"logit": self.out(h).squeeze(-1)}
        if getattr(self.cfg, "wide", False):
            a = self.wide(meta).squeeze(-1)                                         # raw metadata, linear
            out["anchor"] = a
            out["logit"] = a + self.gate * out["logit"]
        elif self.cfg.anchor:
            a = self.anchor_head(toks[:, 2] - self.type_emb[:, 2]).squeeze(-1)     # metadata token, before fusion
            out["anchor"] = a
            out["logit"] = a + self.gate * out["logit"]
        if self.aux is not None:
            # each modality token must predict relevance on its own (before fusion): a weak modality cannot hide
            # behind the strongest one, and its encoder is pushed towards features that generalise
            out["aux"] = {m: self.aux[m](toks[:, j]).squeeze(-1) for j, m in enumerate(MODALITIES)
                          if m in self.cfg.modalities}
        if attn:
            x, maps = seq, []
            with torch.no_grad():
                for layer in self.fusion.layers:
                    q = layer.norm1(x)
                    _, w = layer.self_attn(q, q, q, key_padding_mask=pad, need_weights=True)
                    maps.append(w[:, 0, 1:])
                    x = layer(x, src_key_padding_mask=pad)
                a = torch.stack(maps).mean(0)
                out["attention"] = a / a.sum(1, keepdim=True).clamp(min=1e-9)
        return out


class Planner:
    """Ranks the files of a case; the deployed checkpoint is trained on every solved case."""

    def __init__(self, net: PlannerNet, device: str = "cpu"):
        self.net = net.to(device).eval()
        self.cfg = net.cfg
        self.device = device

    @staticmethod
    def save(net: PlannerNet, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(net.cfg), "state_dict": {k: v.cpu() for k, v in net.state_dict().items()}}, path)
        return path

    @classmethod
    def load(cls, path: Path | None = None, device: str | None = None) -> "Planner":
        from src.config import settings

        path = path or default_checkpoint()
        if not path.exists():
            raise FileNotFoundError(f"no planner checkpoint at {path}; run `python main.py planner train` first")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        if "ensemble" in ck:
            return EnsemblePlanner([_net(m) for m in ck["ensemble"]], device or settings.device_for("orchestrator"),
                                   ck.get("meta", {}))
        return cls(_net(ck), device or settings.device_for("orchestrator"))

    @torch.no_grad()
    def score(self, x: dict, batch: int = 2048, attn: bool = False) -> dict[str, np.ndarray]:
        """``x`` = arrays path/header/meta/content/context/present (see ``features``); returns probabilities."""
        out, att = [], []
        n = len(x["path"])
        for i in range(0, n, batch):
            sl = slice(i, i + batch)
            t = {k: torch.from_numpy(np.ascontiguousarray(v[sl])).to(self.device) for k, v in x.items()}
            o = self.net(t["path"], t["header"], t["meta"], t["content"], t["context"], t["present"], attn=attn)
            out.append(torch.sigmoid(o["logit"] / self.cfg.temperature).float().cpu().numpy())
            if attn:
                att.append(o["attention"].float().cpu().numpy())
        res = {"p": np.concatenate(out) if out else np.zeros(0)}
        if attn:
            res["attention"] = np.concatenate(att) if att else np.zeros((0, len(MODALITIES)))
        return res


def _net(ck: dict) -> PlannerNet:
    cfg = PlannerConfig(**{**ck["config"], "modalities": tuple(ck["config"]["modalities"])})
    net = PlannerNet(cfg)
    net.load_state_dict(ck["state_dict"])
    return net


class EnsemblePlanner(Planner):
    """The nested-selection final model: the mean logit of several PlannerNets (inner folds x seeds at the selected
    epoch, ``src/ml/planner_nested.py``). Same ``score`` interface; attention is averaged."""

    def __init__(self, nets: list[PlannerNet], device: str = "cpu", meta: dict | None = None):
        super().__init__(nets[0], device)
        self.nets = [n.to(device).eval() for n in nets]
        self.meta = meta or {}

    @staticmethod
    def save_ensemble(nets: list[PlannerNet], path: Path, meta: dict | None = None) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"ensemble": [{"config": asdict(n.cfg), "state_dict": {k: v.cpu() for k, v in
                                                                          n.state_dict().items()}} for n in nets],
                    "meta": meta or {}}, path)
        return path

    @torch.no_grad()
    def score(self, x: dict, batch: int = 2048, attn: bool = False) -> dict[str, np.ndarray]:
        out, att = [], []
        n = len(x["path"])
        for i in range(0, n, batch):
            sl = slice(i, i + batch)
            t = {k: torch.from_numpy(np.ascontiguousarray(v[sl])).to(self.device) for k, v in x.items()}
            os_ = [m(t["path"], t["header"], t["meta"], t["content"], t["context"], t["present"], attn=attn)
                   for m in self.nets]
            out.append(torch.sigmoid(torch.stack([o["logit"] for o in os_]).mean(0)).float().cpu().numpy())
            if attn:
                att.append(torch.stack([o["attention"] for o in os_]).mean(0).float().cpu().numpy())
        res = {"p": np.concatenate(out) if out else np.zeros(0)}
        if attn:
            res["attention"] = np.concatenate(att) if att else np.zeros((0, len(MODALITIES)))
        return res


def features(rows: list[dict], head: np.ndarray, meta: np.ndarray, snip_emb: np.ndarray,
             brief_emb: np.ndarray) -> dict[str, np.ndarray]:
    """Model inputs for ``rows`` (``brief_emb``: one vector per row, the brief of the row's case)."""
    path = np.stack([encode_path(r["path_text"]) for r in rows]) if rows else np.zeros((0, PATH_CHARS), np.int64)
    has_snip = np.array([bool(r.get("snippet", "").strip()) for r in rows], dtype=bool)
    present = np.stack([np.ones(len(rows), bool), np.ones(len(rows), bool), np.ones(len(rows), bool), has_snip,
                        np.ones(len(rows), bool)], 1)
    return {"path": path, "header": head.astype(np.int64), "meta": meta.astype(np.float32),
            "content": snip_emb.astype(np.float32), "context": brief_emb.astype(np.float32), "present": present}


assert HEAD == 256
