"""Jev-MM: multimodal image-triage model (vision + metadata + OCR text -> typed answers with probabilities).

Context = one image, seen through three modalities (``src/ml/mm_dataset.py``):

    pixels[224x224x3] -> CLIP ViT-B/32 (pretrained, last blocks fine-tuned) -> 768 -> MLP -> token_v
    metadata[43]      -> MLP                                                  -> token_m
    OCR text          -> frozen multilingual MPNet (768) || text stats[9] -> MLP -> token_t
    [CLS, token_v, token_m, token_t] + modality embeddings -> 2-layer Transformer encoder (key-padding mask for
    absent / dropped modalities) -> CLS -> heads

Queries (all calibrated by temperature scaling on the validation cases, abstain below a confidence threshold):

    choice   "Where does this image come from?"  camera | screenshot | messaging | user_file | web_cache |
                                                  thumbnail | app_asset
    yes_no   "Is it user-generated content?"     (camera, screenshot, messaging media, user file)

The storage path is never an input, so the answers also hold for carved, deleted or renamed images. Training
uses modality dropout, so the model still answers when an image has no EXIF or no readable text; the attention
paid by CLS to each modality token is returned with every answer. Answers are model observations: the
``ImageTriageAgent`` reports them as UNCORROBORATED leads and the verifier never accepts them as evidence.
"""
from __future__ import annotations

import glob
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch import nn

from src.config.settings import DATA_DIR
from src.ml.mm_dataset import N_META, ORIGINS, TEXT_STATS

MODEL_VERSION = "jev-mm-1"
MODALITIES = ("vision", "metadata", "text")
CLIP_NAME = "vit_base_patch32_clip_224.openai"
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)
TEXT_DIM = 768


def default_checkpoint() -> Path:
    return DATA_DIR / "models" / "multimodal" / "jev_mm.pt"


@dataclass
class MMConfig:
    classes: list[str] = field(default_factory=lambda: list(ORIGINS))
    modalities: tuple[str, ...] = MODALITIES
    dim: int = 256
    heads: int = 4
    layers: int = 2
    dropout: float = 0.2
    unfreeze_blocks: int = 1           # last ViT blocks fine-tuned (chosen on validation); the rest stays CLIP
    n_meta: int = N_META
    n_text_stats: int = len(TEXT_STATS)
    t_choice: float = 1.0
    t_yes_no: float = 1.0
    abstain_below: float = 0.5
    version: str = MODEL_VERSION
    meta: dict = field(default_factory=dict)


# ------------------------------------------------------------------------------------------------ encoders
def clip_weights_path() -> str:
    hub = os.environ.get("HF_HUB_CACHE") or os.path.join(
        os.environ.get("HF_HOME") or os.path.expanduser("~/.cache/huggingface"), "hub")
    hits = glob.glob(os.path.join(hub, f"models--timm--{CLIP_NAME}", "snapshots", "*", "open_clip_model.safetensors"))
    if not hits:
        raise FileNotFoundError(f"CLIP weights for {CLIP_NAME} not in the HF cache ({hub})")
    return hits[0]


def build_vit(pretrained: bool = True) -> nn.Module:
    """timm CLIP ViT-B/32 without its projection head (768-d pooled output), weights from the local cache."""
    import timm

    vit = timm.create_model(CLIP_NAME, pretrained=False, num_classes=0)
    if pretrained:
        from safetensors.torch import load_file
        from timm.models.vision_transformer import _convert_openai_clip

        sd = _convert_openai_clip(load_file(clip_weights_path()), vit)
        sd = {k: v for k, v in sd.items() if not k.startswith("head.")}
        vit.load_state_dict(sd, strict=True)
    return vit


def _mlp(i: int, h: int, o: int, p: float) -> nn.Sequential:
    return nn.Sequential(nn.LayerNorm(i), nn.Linear(i, h), nn.GELU(), nn.Dropout(p), nn.Linear(h, o))


class JevMMNet(nn.Module):
    def __init__(self, cfg: MMConfig, vit: nn.Module | None = None, pretrained: bool = True):
        super().__init__()
        self.cfg = cfg
        d = cfg.dim
        self.use = {m: (m in cfg.modalities) for m in MODALITIES}
        if self.use["vision"]:
            self.vit = vit if vit is not None else build_vit(pretrained)
            self._freeze_vit(cfg.unfreeze_blocks)
            self.v_proj = _mlp(768, 512, d, cfg.dropout)
            self.register_buffer("mean", torch.tensor(CLIP_MEAN).view(1, 3, 1, 1), persistent=False)
            self.register_buffer("std", torch.tensor(CLIP_STD).view(1, 3, 1, 1), persistent=False)
        if self.use["metadata"]:
            self.m_proj = _mlp(cfg.n_meta, 256, d, cfg.dropout)
        if self.use["text"]:
            self.t_proj = _mlp(TEXT_DIM + cfg.n_text_stats, 512, d, cfg.dropout)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.mod_emb = nn.Parameter(torch.randn(1, len(MODALITIES), d) * 0.02)
        layer = nn.TransformerEncoderLayer(d, cfg.heads, 4 * d, cfg.dropout, activation="gelu", batch_first=True,
                                           norm_first=True)
        self.fusion = nn.TransformerEncoder(layer, cfg.layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d)
        self.choice = nn.Linear(d, len(cfg.classes))
        self.yes_no = nn.Linear(d, 1)

    def _freeze_vit(self, n: int) -> None:
        for p in self.vit.parameters():
            p.requires_grad = False
        blocks = list(self.vit.blocks)
        for b in blocks[len(blocks) - n:] if n > 0 else []:
            for p in b.parameters():
                p.requires_grad = True
        if n > 0:
            for mod in (self.vit.norm, getattr(self.vit, "fc_norm", None)):
                if isinstance(mod, nn.Module):
                    for p in mod.parameters():
                        p.requires_grad = True

    def frozen_prefix(self) -> set[str]:
        """Parameter names left at the CLIP weights (not stored in checkpoints)."""
        return {f"vit.{k}" for k, p in self.vit.named_parameters() if not p.requires_grad} if self.use["vision"] else set()

    def forward(self, thumbs: torch.Tensor | None, meta: torch.Tensor | None, text: torch.Tensor | None,
                present: torch.Tensor, attn: bool = False) -> dict[str, torch.Tensor]:
        """thumbs (B,224,224,3) uint8; meta (B,n_meta); text (B,768+stats); present (B,3) bool."""
        b = present.shape[0]
        d = self.cfg.dim
        toks = []
        dev = present.device
        if self.use["vision"]:
            x = thumbs.permute(0, 3, 1, 2).float().div(255)
            toks.append(self.v_proj(self.vit((x - self.mean) / self.std)))
        else:
            toks.append(torch.zeros(b, d, device=dev))
        toks.append(self.m_proj(meta) if self.use["metadata"] else torch.zeros(b, d, device=dev))
        toks.append(self.t_proj(text) if self.use["text"] else torch.zeros(b, d, device=dev))
        tok = torch.stack(toks, 1) + self.mod_emb
        use = torch.tensor([self.use[m] for m in MODALITIES], device=dev)
        keep = present & use
        keep[keep.sum(1) == 0, 0] = True          # never mask every token (vision is always computable)
        seq = torch.cat([self.cls.expand(b, -1, -1), tok], 1)
        pad = torch.cat([torch.zeros(b, 1, dtype=torch.bool, device=dev), ~keep], 1)
        h = self.norm(self.fusion(seq, src_key_padding_mask=pad))[:, 0]
        out = {"choice": self.choice(h), "yes_no": self.yes_no(h).squeeze(-1)}
        if attn:
            out["attention"] = self._cls_attention(seq, pad)
        return out

    @torch.no_grad()
    def _cls_attention(self, seq: torch.Tensor, pad: torch.Tensor) -> torch.Tensor:
        """Mean over heads and layers of the attention CLS pays to each modality token (B, 3)."""
        x, maps = seq, []
        for layer in self.fusion.layers:
            q = layer.norm1(x)
            _, w = layer.self_attn(q, q, q, key_padding_mask=pad, need_weights=True, average_attn_weights=True)
            maps.append(w[:, 0, 1:])
            x = layer(x, src_key_padding_mask=pad)
        a = torch.stack(maps).mean(0)
        return a / a.sum(1, keepdim=True).clamp(min=1e-9)


# ------------------------------------------------------------------------------------------------ query API
def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(-1, keepdims=True)


class JevMM:
    """Typed-answer interface over a trained Jev-MM checkpoint."""

    def __init__(self, net: JevMMNet, device: str = "cpu"):
        self.net = net.to(device).eval()
        self.cfg = net.cfg
        self.device = device
        self._ocr = None
        self._text = None

    # -------------------------------------------------------------------------------- persistence
    @staticmethod
    def save(net: JevMMNet, path: Path) -> Path:
        """Stores the trained parameters only; the frozen CLIP weights are re-read from the local cache."""
        path.parent.mkdir(parents=True, exist_ok=True)
        frozen = net.frozen_prefix()
        sd = {k: v.detach().cpu() for k, v in net.state_dict().items() if k not in frozen}
        torch.save({"config": asdict(net.cfg), "state_dict": sd, "clip": CLIP_NAME}, path)
        return path

    @classmethod
    def load(cls, path: Path | None = None, device: str | None = None) -> "JevMM":
        from src.config import settings

        path = path or default_checkpoint()
        if not path.exists():
            raise FileNotFoundError(f"no Jev-MM checkpoint at {path}; run `python main.py mm train` first")
        ck = torch.load(path, map_location="cpu", weights_only=False)
        cfg = MMConfig(**{**ck["config"], "modalities": tuple(ck["config"]["modalities"])})
        net = JevMMNet(cfg)
        missing, unexpected = net.load_state_dict(ck["state_dict"], strict=False)
        bad = [k for k in missing if k not in net.frozen_prefix()]
        if bad or unexpected:
            raise ValueError(f"checkpoint does not match the model: missing {bad[:5]}, unexpected {unexpected[:5]}")
        return cls(net, device or settings.device_for("vision"))

    # -------------------------------------------------------------------------------- inference
    @torch.no_grad()
    def logits(self, thumbs: np.ndarray, meta: np.ndarray, text: np.ndarray, present: np.ndarray,
               batch: int = 256, attn: bool = False) -> dict[str, np.ndarray]:
        outs: dict[str, list] = {"choice": [], "yes_no": [], **({"attention": []} if attn else {})}
        for i in range(0, len(present), batch):
            sl = slice(i, i + batch)
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=str(self.device).startswith("cuda")):
                o = self.net(torch.from_numpy(np.ascontiguousarray(thumbs[sl])).to(self.device),
                             torch.from_numpy(meta[sl]).float().to(self.device),
                             torch.from_numpy(text[sl]).float().to(self.device),
                             torch.from_numpy(present[sl]).to(self.device), attn=attn)
            for k in outs:
                outs[k].append(o[k].float().cpu().numpy())
        return {k: np.concatenate(v) for k, v in outs.items()}

    def probabilities(self, thumbs, meta, text, present, attn: bool = False) -> dict[str, np.ndarray]:
        lg = self.logits(thumbs, meta, text, present, attn=attn)
        out = {"choice": _softmax(lg["choice"] / self.cfg.t_choice),
               "yes_no": 1 / (1 + np.exp(-lg["yes_no"] / self.cfg.t_yes_no))}
        if attn:
            out["attention"] = lg["attention"]
        return out

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
        return {"query": "choice", "question": "origin of the image",
                "answer": None if conf < self.cfg.abstain_below else names[best], "best": names[best],
                "confidence": round(conf, 4), "abstained": conf < self.cfg.abstain_below,
                "probabilities": {n: round(float(v), 4) for n, v in sorted(zip(names, p), key=lambda t: -t[1])},
                "model": self.cfg.version}

    def yes_no_answer(self, p: float) -> dict:
        conf = max(p, 1 - p)
        return {"query": "yes_no", "question": "user-generated content?",
                "answer": None if conf < self.cfg.abstain_below else bool(p >= 0.5), "probability": round(p, 4),
                "confidence": round(conf, 4), "abstained": conf < self.cfg.abstain_below, "model": self.cfg.version}

    # -------------------------------------------------------------------------------- raw images
    def features(self, paths: Sequence[str | Path], ocr: bool = True) -> dict[str, np.ndarray]:
        """The three modalities for image files on disk (the same pipeline as the dataset)."""
        from src.ml import mm_dataset as mmd

        thumbs, metas, texts, present, errors = [], [], [], [], []
        raw_text, stats = [], []
        for p in paths:
            res = mmd.load_image_features(str(p), mmd.THUMB)
            if isinstance(res, str):
                errors.append(res)
                thumbs.append(np.full((mmd.THUMB, mmd.THUMB, 3), 128, np.uint8))
                metas.append(np.zeros(N_META, np.float32))
                raw_text.append("")
                stats.append(np.zeros(len(TEXT_STATS), np.float32))
                present.append([False, False, False])
                continue
            errors.append(None)
            metas.append(res[0])
            thumbs.append(res[1])
            t, s = self._ocr_one(p) if ocr and "text" in self.cfg.modalities else ("", np.zeros(len(TEXT_STATS)))
            raw_text.append(t)
            stats.append(np.asarray(s, np.float32))
            present.append([True, True, bool(t)])
        emb = (self._text_encoder().encode(raw_text) if any(raw_text)
               else np.zeros((len(raw_text), TEXT_DIM), np.float32))
        texts = np.concatenate([emb, np.stack(stats)], 1).astype(np.float32)
        return {"thumbs": np.stack(thumbs), "meta": np.stack(metas), "text": texts,
                "present": np.array(present, dtype=bool), "ocr_text": raw_text, "errors": errors}

    def ask(self, paths: Sequence[str | Path], ocr: bool = True) -> list[dict]:
        f = self.features(paths, ocr)
        p = self.probabilities(f["thumbs"], f["meta"], f["text"], f["present"], attn=True)
        out = []
        for i, path in enumerate(paths):
            if f["errors"][i]:
                out.append({"file": str(path), "error": f["errors"][i]})
                continue
            out.append({"file": str(path), "choice": self.choice_answer(p["choice"][i]),
                        "yes_no": self.yes_no_answer(float(p["yes_no"][i])),
                        "modality_attention": {m: round(float(a), 3) for m, a in zip(MODALITIES, p["attention"][i])},
                        "modalities_present": {m: bool(x) for m, x in zip(MODALITIES, f["present"][i])},
                        "ocr_text": f["ocr_text"][i][:200]})
        return out

    def _ocr_one(self, path) -> tuple[str, np.ndarray]:
        from PIL import Image

        from src.config import settings
        from src.ml import mm_dataset as mmd

        if self._ocr is None:
            import easyocr

            self._ocr = easyocr.Reader(["en", "es"], gpu=settings.device_for("ocr").startswith("cuda"), verbose=False)
        with Image.open(path) as img:
            img.draft("RGB", (mmd.OCR_MAX_SIDE, mmd.OCR_MAX_SIDE))
            img = img.convert("RGB")
            img.thumbnail((mmd.OCR_MAX_SIDE, mmd.OCR_MAX_SIDE))
            arr = np.asarray(img)
        res = [(q, t, float(c)) for q, t, c in self._ocr.readtext(arr, detail=1) if c >= 0.3]
        return " | ".join(t for _, t, _ in res)[:1000], mmd.text_stats(res, arr.shape[1], arr.shape[0])

    def _text_encoder(self):
        if self._text is None:
            from src.ml.mm_dataset import TextEncoder

            self._text = TextEncoder(self.device)
        return self._text
