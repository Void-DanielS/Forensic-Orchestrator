"""Zero-shot relevance prior from a local LLM (transfer learning without any training on the labelled cases).

    python -m src.ml.llm_prior [--batch 24]       # scores every file of the planner dataset

For every file the LLM sees the case brief (shortened), the file's path, category, size and its text excerpt, and is
asked whether the file is likely to hold information that answers the investigation. The prior is the log-odds of the
first answer token, ``log P(Yes) - log P(No)``, read from a single forward pass (no generation, deterministic).

Because the LLM is never trained or prompted with labels, its scores leak nothing from the held-out case and transfer
to unseen cases; the planner variant ``wide & deep + LLM prior`` adds it to the wide part with a weight fitted on the
training cases only.
"""
from __future__ import annotations

import argparse
import re
import time
from pathlib import Path

import numpy as np

from src.config import settings

SYSTEM = ("You are a digital forensic examiner triaging a seized device. You decide which files to read first. "
          "Answer with a single word: Yes or No.")
USER = ("Investigation brief:\n{brief}\n\n"
        "File: {path}\nCategory: {category}; size: {size}\nText excerpt from the file:\n{snippet}\n\n"
        "Is this file likely to contain information that answers the investigation (people, accounts, e-mail "
        "addresses, messages, documents, web activity, locations or events of the scenario)? Answer Yes or No.")
BRIEF_CHARS = 1200
SNIPPET_CHARS = 500
PAD_TO = 64            # padded lengths in steps of 64 tokens: few distinct shapes for the CUDA cache
EMPTY_EVERY = 50


def short_brief(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()[:BRIEF_CHARS]


def _size(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024:
            return f"{n:.0f} {unit}"
        n /= 1024
    return f"{n:.0f} TB"


def prompt(row: dict, brief: str) -> tuple[str, str]:
    snip = re.sub(r"\s+", " ", row.get("snippet") or "")[:SNIPPET_CHARS] or "(no readable text)"
    return SYSTEM, USER.format(brief=brief, path=row.get("path_text") or row.get("path"), category=row.get("category"),
                               size=_size(int(row.get("size") or 0)), snippet=snip)


TUNED_FILE = "llm_prior_tuned.npz"


class LLMPrior:
    def __init__(self, adapter: Path | None = None):
        """``adapter``: a LoRA adapter (``src/ml/finetune_prior.py``) merged into the weights: the tuned prior. The
        merge changes the shared model in place, so it is only done in a process of its own (``build``)."""
        from src.agents.llm import LocalLLM

        self.llm = LocalLLM()
        if adapter is not None:
            from peft import PeftModel

            from src.agents.vision_agent import _MODEL_CACHE

            if self.llm.cfg.quantization not in (None, "", "none"):
                raise SystemExit("the tuned prior needs the unquantised model: set FORENSI_VLM_QUANT=none")
            _MODEL_CACHE.clear()                     # nobody else may pick up the merged weights
            self.llm.model = PeftModel.from_pretrained(self.llm.model, str(adapter)).merge_and_unload().eval()
        tok = self.llm.processor.tokenizer
        self.yes = [tok.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes")]
        self.no = [tok.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no")]
        self.w = self.llm.model.lm_head.weight[self.yes + self.no].float()      # (6, hidden), fp32

    def score(self, prompts: list[tuple[str, str]], batch: int = 24, progress: bool = False) -> np.ndarray:
        import torch

        proc, model = self.llm.processor, self.llm.model
        out = np.zeros(len(prompts), np.float32)
        order = np.argsort([len(u) for _, u in prompts])            # similar lengths per batch: less padding
        t0 = time.perf_counter()
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            msgs = [[{"role": "system", "content": [{"type": "text", "text": prompts[j][0]}]},
                     {"role": "user", "content": [{"type": "text", "text": prompts[j][1]}]}] for j in idx]
            texts = [proc.apply_chat_template(m, tokenize=False, add_generation_prompt=True) for m in msgs]
            enc = proc(text=texts, return_tensors="pt", padding=True, pad_to_multiple_of=PAD_TO).to(model.device)
            with torch.inference_mode():
                # last hidden state (after the final norm) -> logits of the Yes / No tokens only, in fp32. The
                # softmax normaliser cancels in log P(Yes) - log P(No), so the full-vocabulary logits are not
                # needed. (Taking them from the bf16 lm_head output rounded every score to steps of 0.125.)
                h = model.model(**enc, use_cache=False).last_hidden_state[:, -1].float()
                logits = h @ self.w.T
            s = torch.logsumexp(logits[:, :len(self.yes)], -1) - torch.logsumexp(logits[:, len(self.yes):], -1)
            out[idx] = s.cpu().numpy()
            del enc, h, logits
            if (i // batch) % EMPTY_EVERY == EMPTY_EVERY - 1:
                # every padded length leaves its own blocks in the CUDA cache, and on Windows (WDDM) reserved GPU
                # memory counts against the process commit charge: without this the run exhausted system memory
                torch.cuda.empty_cache()
            if progress and (i // batch) % 50 == 0:
                print(f"[llm-prior] {i + len(idx):,}/{len(prompts):,} ({time.perf_counter() - t0:.0f} s)", flush=True)
        return out


def prior_path(ds: Path | None = None) -> Path:
    from src.ml.planner_dataset import latest

    return (ds or latest()) / "llm_prior.npz"


def build(batch: int = 24, limit: int | None = None, adapter: Path | None = None) -> Path:
    """Scores every file of the planner dataset (row order of ``PlanData``); with ``adapter`` the tuned prior."""
    from src.ml.planner_dataset import load

    d = load()
    rows = d["rows"][:limit]
    prompts = [prompt(r, short_brief(d["briefs"].get(r["case_id"], ""))) for r in rows]
    s = LLMPrior(adapter).score(prompts, batch, progress=True)
    path = prior_path(d["dir"]) if adapter is None else prior_path(d["dir"]).with_name(TUNED_FILE)
    if limit is not None:
        path = path.with_suffix(".test.npz")
    np.savez_compressed(path, score=s, case=np.array([r["case_id"] for r in rows]),
                        model=settings.VISION.vlm_model_id, adapter=str(adapter or ""))
    return path


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--batch", type=int, default=24)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--adapter", type=Path, help="LoRA adapter dir (tuned prior), e.g. data/models/llm_prior_lora")
    a = ap.parse_args()
    print(build(a.batch, a.limit, a.adapter))
