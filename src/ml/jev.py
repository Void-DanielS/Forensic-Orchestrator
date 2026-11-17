"""Jev-style typed-answer model: reads a context and answers with probabilities, never with free text.

The context is one 512-byte evidence fragment (plus, for one query, the file's extension). One shared encoder feeds
three heads, one per query primitive:

  choice   "Which file type is this fragment from?"            -> distribution over the trained types
  yes_no   "Is the file disguised (content vs. extension)?"     -> P(yes)
  score    "Where in its file does this fragment sit? (0..1)"   -> expected value over 5 ordinal bins

Architecture (trained from scratch, ``src/ml/train.py``)::

    bytes[512] -> Embedding(256, 64) -> 4 x [Conv1d + BatchNorm + GELU (+ MaxPool)] -> avg||max pool -> z[256]
    choice = Linear(z)                score = Linear(z)              yes_no = MLP(z || ext_encoder(extension))

The choice and score heads read content only, so a file type is never inferred from its name. Every answer carries
a temperature-calibrated confidence (fitted on the validation cases) and abstains below a threshold. Answers are
model observations: an agent may report them as leads, the verifier never accepts them as evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.config.settings import DATA_DIR

FRAG = 512
EXT_LEN = 8
SCORE_BINS = 5
MODEL_VERSION = "jev-bytes-1"


def default_checkpoint() -> Path:
    return DATA_DIR / "models" / "jev" / "jev.pt"


@dataclass
class JevConfig:
    classes: list[str]
    emb: int = 64
    channels: tuple[int, ...] = (128, 192, 256, 256)
    kernels: tuple[int, ...] = (9, 7, 5, 3)
    dim: int = 256
    ext_dim: int = 32
    dropout: float = 0.2
    # calibration (fitted after training on the validation split)
    t_choice: float = 1.0
    t_yes_no: float = 1.0
    t_score: float = 1.0
    abstain_below: float = 0.5
    version: str = MODEL_VERSION
    meta: dict = field(default_factory=dict)


def encode_ext(ext: str) -> list[int]:
    """'.JPG' -> char ids (ASCII + 1, 0 = padding), fixed length."""
    s = ext.lower().lstrip(".")[:EXT_LEN]
    ids = [min(ord(ch), 127) + 1 for ch in s]
    return ids + [0] * (EXT_LEN - len(ids))


class ByteEncoder(nn.Module):
    def __init__(self, cfg: JevConfig):
        super().__init__()
        self.emb = nn.Embedding(256, cfg.emb)
        layers, cin = [], cfg.emb
        for i, (cout, k) in enumerate(zip(cfg.channels, cfg.kernels)):
            layers += [nn.Conv1d(cin, cout, k, padding=k // 2), nn.BatchNorm1d(cout), nn.GELU()]
            if i < len(cfg.channels) - 1:
                layers.append(nn.MaxPool1d(2))
            cin = cout
        self.conv = nn.Sequential(*layers)
        self.proj = nn.Sequential(nn.Dropout(cfg.dropout), nn.Linear(2 * cin, cfg.dim), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:          # x: (B, 512) long
        h = self.conv(self.emb(x).transpose(1, 2))               # (B, C, L)
        return self.proj(torch.cat([h.mean(-1), h.amax(-1)], 1))


class JevNet(nn.Module):
    def __init__(self, cfg: JevConfig):
        super().__init__()
        self.cfg = cfg
        self.encoder = ByteEncoder(cfg)
        self.ext_emb = nn.Embedding(129, cfg.ext_dim, padding_idx=0)
        self.choice = nn.Linear(cfg.dim, len(cfg.classes))
        self.score = nn.Linear(cfg.dim, SCORE_BINS)
        self.yes_no = nn.Sequential(nn.Linear(cfg.dim + cfg.ext_dim + len(cfg.classes), 128), nn.GELU(),
                                    nn.Dropout(cfg.dropout), nn.Linear(128, 1))

    def forward(self, x: torch.Tensor, ext: torch.Tensor) -> dict[str, torch.Tensor]:
        z = self.encoder(x)
        choice = self.choice(z)
        mask = (ext > 0).float().unsqueeze(-1)
        e = (self.ext_emb(ext) * mask).sum(1) / mask.sum(1).clamp(min=1)
        # the disguise head sees the predicted type distribution, not the true label (no leakage at inference)
        yn = self.yes_no(torch.cat([z, e, choice.detach().softmax(-1)], 1)).squeeze(-1)
        return {"choice": choice, "score": self.score(z), "yes_no": yn}


def score_targets(position: torch.Tensor) -> torch.Tensor:
    return (position.clamp(0, 1) * SCORE_BINS).long().clamp(max=SCORE_BINS - 1)


BIN_CENTERS = (np.arange(SCORE_BINS) + 0.5) / SCORE_BINS


class Jev:
    """Query interface over a trained checkpoint: ``choice``, ``yes_no``, ``score`` (single or batched)."""

    def __init__(self, net: JevNet, device: str = "cpu"):
        self.net = net.to(device).eval()
        self.cfg = net.cfg
        self.device = device

    # ------------------------------------------------------------------ persistence
    @classmethod
    def load(cls, path: Path | None = None, device: str | None = None) -> "Jev":
        from src.config import settings

        path = path or default_checkpoint()
        if not path.exists():
            raise FileNotFoundError(f"no Jev checkpoint at {path}; run `python main.py jev train` first")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = JevConfig(**ck["config"])
        net = JevNet(cfg)
        net.load_state_dict(ck["state_dict"])
        return cls(net, device or settings.device_for("text"))

    @staticmethod
    def save(net: JevNet, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"config": asdict(net.cfg), "state_dict": {k: v.cpu() for k, v in net.state_dict().items()}},
                   path)
        return path

    # ------------------------------------------------------------------ inference
    @torch.no_grad()
    def _logits(self, frags: Sequence[bytes] | np.ndarray, exts: Sequence[str] | None = None,
                batch: int = 1024) -> dict[str, np.ndarray]:
        x = frags if isinstance(frags, np.ndarray) else np.stack([_as_frag(f) for f in frags])
        e = np.array([encode_ext(s) for s in (exts or [""] * len(x))], dtype=np.int64)
        outs: dict[str, list] = {"choice": [], "score": [], "yes_no": []}
        for i in range(0, len(x), batch):
            o = self.net(torch.from_numpy(x[i:i + batch].astype(np.int64)).to(self.device),
                         torch.from_numpy(e[i:i + batch]).to(self.device))
            for k in outs:
                outs[k].append(o[k].float().cpu().numpy())
        return {k: np.concatenate(v) for k, v in outs.items()}

    def probabilities(self, frags, exts=None) -> dict[str, np.ndarray]:
        """Calibrated probabilities for all three heads: choice (N, C), yes_no (N,), score (N, bins)."""
        lg = self._logits(frags, exts)
        return {"choice": _softmax(lg["choice"] / self.cfg.t_choice),
                "yes_no": 1 / (1 + np.exp(-lg["yes_no"] / self.cfg.t_yes_no)),
                "score": _softmax(lg["score"] / self.cfg.t_score)}

    def choice(self, frag: bytes, options: Sequence[str] | None = None) -> dict:
        """Pick one type; ``options`` restricts the answer set (probabilities renormalised over it)."""
        p = self.probabilities([frag])["choice"][0]
        return self.choice_answer(p, options)

    def choice_answer(self, p: np.ndarray, options: Sequence[str] | None = None) -> dict:
        names = list(self.cfg.classes)
        if options:
            unknown = set(options) - set(names)
            if unknown:
                raise ValueError(f"options not known to the model: {sorted(unknown)}")
            idx = [names.index(o) for o in options]
            p, names = p[idx] / max(p[idx].sum(), 1e-12), list(options)
        best = int(np.argmax(p))
        conf = float(p[best])
        return {"query": "choice", "answer": None if conf < self.cfg.abstain_below else names[best],
                "best": names[best], "confidence": round(conf, 4), "abstained": conf < self.cfg.abstain_below,
                "probabilities": {n: round(float(v), 4) for n, v in sorted(zip(names, p), key=lambda t: -t[1])},
                "model": self.cfg.version}

    def yes_no(self, frag: bytes, extension: str) -> dict:
        """P(the file's content contradicts ``extension``)."""
        return self.yes_no_answer(float(self.probabilities([frag], [extension])["yes_no"][0]))

    def yes_no_answer(self, p: float) -> dict:
        conf = max(p, 1 - p)
        return {"query": "yes_no", "answer": None if conf < self.cfg.abstain_below else p >= 0.5,
                "probability": round(p, 4), "confidence": round(conf, 4), "abstained": conf < self.cfg.abstain_below,
                "model": self.cfg.version}

    def score(self, frag: bytes) -> dict:
        """Relative position of the fragment inside its file, 0 (start) .. 1 (end)."""
        return self.score_answer(self.probabilities([frag])["score"][0])

    def score_answer(self, p: np.ndarray) -> dict:
        return {"query": "score", "answer": round(float(p @ BIN_CENTERS), 4), "confidence": round(float(p.max()), 4),
                "distribution": [round(float(v), 4) for v in p], "bins": SCORE_BINS, "model": self.cfg.version}


def _as_frag(b: bytes) -> np.ndarray:
    a = np.frombuffer(bytes(b)[:FRAG], dtype=np.uint8)
    return np.pad(a, (0, FRAG - len(a))) if len(a) < FRAG else a


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)
