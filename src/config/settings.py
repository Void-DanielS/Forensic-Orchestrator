"""Central configuration for ForensiAgent-Multimodal.

Device allocation
-----------------
Sprint 1 runs every model on a single GPU (``cuda:0``). The architecture targets a
dual-GPU box (``cuda:0`` = text / orchestration, ``cuda:1`` = vision / VLM).
Switching is a single flag: set ``SINGLE_GPU_MODE = False`` (or the environment
variable ``FORENSI_SINGLE_GPU=0``). Every agent asks :func:`device_for` for its
device instead of hard-coding one, so no agent code changes are needed.

All values can be overridden through environment variables prefixed ``FORENSI_``.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

Role = Literal["vision", "ocr", "text", "embeddings", "orchestrator"]


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _env_float(name: str, default: float) -> float:
    raw = os.environ.get(name)
    return float(raw) if raw else default


# --------------------------------------------------------------------------------------
# Top-level switch. Sprint 1: True (everything on cuda:0).
# Sprint 2+: set to False once the RTX 4060 is installed as cuda:1.
# --------------------------------------------------------------------------------------
SINGLE_GPU_MODE: bool = _env_bool("FORENSI_SINGLE_GPU", True)

PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
DATA_DIR: Path = PROJECT_ROOT / "data"
RAW_IMAGES_DIR: Path = DATA_DIR / "raw_images"
OUTPUTS_DIR: Path = DATA_DIR / "outputs"


@dataclass(frozen=True)
class GPUProfile:
    """A physical GPU and the VRAM budget agents may use on it."""

    index: int
    label: str
    vram_budget_gb: float

    @property
    def device(self) -> str:
        return f"cuda:{self.index}"


# Primary: RTX 5070 (12 GB) per the hardware plan. Budget is a *cap* we design to,
# even when the physical card is larger, so the code stays portable.
PRIMARY_GPU = GPUProfile(0, "RTX 5070 (text/orchestration)", _env_float("FORENSI_GPU0_BUDGET_GB", 12.0))
# Secondary: RTX 4060 (8 GB) – future vision card.
SECONDARY_GPU = GPUProfile(1, "RTX 4060 (vision/VLM)", _env_float("FORENSI_GPU1_BUDGET_GB", 8.0))

# Which GPU each role lands on in dual-GPU mode.
DUAL_GPU_ROLE_MAP: dict[str, GPUProfile] = {
    "vision": SECONDARY_GPU,
    "ocr": SECONDARY_GPU,
    "text": PRIMARY_GPU,
    "embeddings": PRIMARY_GPU,
    "orchestrator": PRIMARY_GPU,
}


def gpu_for(role: Role) -> GPUProfile:
    """Return the GPU profile a role should use under the current mode."""
    if SINGLE_GPU_MODE:
        return PRIMARY_GPU
    return DUAL_GPU_ROLE_MAP[role]


def device_for(role: Role) -> str:
    """Resolve the torch device string for ``role``, degrading to CPU when CUDA is absent."""
    try:
        import torch
    except ImportError:  # pragma: no cover - torch is a hard requirement for agents
        return "cpu"
    if not torch.cuda.is_available():
        return "cpu"
    profile = gpu_for(role)
    if profile.index >= torch.cuda.device_count():
        # Dual-GPU mode requested but the second card is missing: fall back loudly.
        import warnings

        warnings.warn(
            f"Role '{role}' mapped to {profile.device} but only "
            f"{torch.cuda.device_count()} GPU(s) visible; using cuda:0.",
            RuntimeWarning,
            stacklevel=2,
        )
        return "cuda:0"
    return profile.device


def max_memory_for(role: Role) -> dict[int | str, str]:
    """``max_memory`` mapping for HF ``from_pretrained`` that pins a model to the role's GPU."""
    profile = gpu_for(role)
    return {profile.index: f"{profile.vram_budget_gb:.0f}GiB", "cpu": "32GiB"}


@dataclass
class VisionSettings:
    """MultimodalVisionAgent configuration (Sprint 1 deliverable)."""

    # 3B in 4-bit ≈ 3 GB VRAM; 7B in 4-bit ≈ 6 GB. Both fit the 12 GB single-GPU budget
    # alongside EasyOCR (~1 GB).
    vlm_model_id: str = os.environ.get("FORENSI_VLM_MODEL", "Qwen/Qwen2.5-VL-3B-Instruct")
    quantization: Literal["4bit", "8bit", "none"] = os.environ.get("FORENSI_VLM_QUANT", "4bit")  # type: ignore[assignment]
    # Caps the number of visual tokens Qwen2.5-VL produces per image (VRAM control).
    vlm_min_pixels: int = 256 * 28 * 28
    vlm_max_pixels: int = 768 * 28 * 28
    vlm_max_new_tokens: int = 384
    ocr_languages: list[str] = field(default_factory=lambda: ["en", "es"])
    ocr_min_confidence: float = 0.40
    enable_vlm: bool = _env_bool("FORENSI_ENABLE_VLM", True)
    enable_ocr: bool = _env_bool("FORENSI_ENABLE_OCR", True)
    image_extensions: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".heic", ".webp", ".bmp", ".tif", ".tiff")


@dataclass
class VerificationSettings:
    # Minimum normalised similarity for OCR text to count as corroborated by a text artifact.
    ocr_corroboration_threshold: float = 0.85
    # Short model text (< this many normalised chars) only corroborates when it is a
    # substantial part of the cited cell; stops "RADIO SHACK" verifying via a JSON blob.
    corroboration_min_specific_chars: int = 25
    corroboration_min_truth_coverage: float = 0.5
    # Max rows a verification SQL query may return (guards against runaway queries).
    sql_row_limit: int = 5000


VISION = VisionSettings()
VERIFICATION = VerificationSettings()


def describe() -> str:
    lines = [f"SINGLE_GPU_MODE={SINGLE_GPU_MODE}"]
    for role in ("vision", "ocr", "text", "embeddings", "orchestrator"):
        p = gpu_for(role)  # type: ignore[arg-type]
        lines.append(f"  {role:<13}-> {p.device} [{p.label}, budget {p.vram_budget_gb} GB]")
    return "\n".join(lines)
