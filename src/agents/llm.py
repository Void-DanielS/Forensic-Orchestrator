"""Local language model shared by the intelligent agents (nothing leaves the machine).

The model is the same Qwen2.5-VL instance the MultimodalVisionAgent uses (one copy in GPU memory, shared through
``vision_agent._MODEL_CACHE``); with text-only messages it is an ordinary instruction-following LLM. Generation is
greedy (deterministic) and batched with left padding.
"""
from __future__ import annotations

import json
import re
from typing import Any

from src.config import settings


def load(cfg: settings.VisionSettings | None = None, device: str | None = None):
    """(model, processor) for ``cfg.vlm_model_id``, loaded once per process."""
    from src.agents.vision_agent import MultimodalVisionAgent

    class _Loader(MultimodalVisionAgent):          # reuse the VLM loader + cache without building an agent
        def __init__(self):                         # noqa: D401 - no bridge / verifier needed to load a model
            self.cfg = cfg or settings.VISION
            self.device = device or settings.device_for("vision")
            self._vlm = self._processor = None

    return _Loader()._load_vlm()


class LocalLLM:
    def __init__(self, cfg: settings.VisionSettings | None = None, device: str | None = None):
        self.cfg = cfg or settings.VISION
        self.model, self.processor = load(self.cfg, device)
        self.model_id = self.cfg.vlm_model_id
        self.processor.tokenizer.padding_side = "left"

    def chat(self, prompts: list[tuple[str, str]], max_new_tokens: int = 320, batch: int = 8) -> list[str]:
        """``prompts`` = [(system, user)], answers in the same order."""
        import torch

        out: list[str] = []
        for i in range(0, len(prompts), batch):
            msgs = [[{"role": "system", "content": [{"type": "text", "text": s}]},
                     {"role": "user", "content": [{"type": "text", "text": u}]}] for s, u in prompts[i:i + batch]]
            texts = [self.processor.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
            enc = self.processor(text=texts, return_tensors="pt", padding=True).to(self.model.device)
            with torch.inference_mode():
                gen = self.model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False)
            out += self.processor.batch_decode(gen[:, enc["input_ids"].shape[1]:], skip_special_tokens=True)
        return out


def parse_json(raw: str) -> dict[str, Any]:
    """First JSON object in a model answer ({} when there is none or it does not parse)."""
    raw = re.sub(r"^```(?:json)?|```$", "", raw.strip(), flags=re.MULTILINE)
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}
