"""LoRA fine-tuning of the local LLM on public DFIR write-ups (case context -> solution), for the tuned prior.

    python -m src.ml.finetune_prior                 # data/datasets/writeups/pairs.jsonl -> data/models/llm_prior_lora/

Pre-registered recipe (docs/PREREGISTRATION.md, deviation of 2026-11-17), no search: LoRA r 16, alpha 32,
dropout 0.05 on every attention and MLP projection of the language model (not the vision tower), learning rate 1e-4,
1 epoch, at most 1,024 tokens per example, seed 0. Causal-LM loss on the solution tokens only. Fixed here before
training as well: AdamW, linear decay to 0, effective batch 8 (gradient accumulation), gradient checkpointing.
10 % of the write-ups (seed 0) are held out to report the loss before / after training (descriptive only: nothing
is chosen with it). No dataset case and no label is seen, so one tuned model serves every planner fold.
"""
from __future__ import annotations

import json
import math
import random
import time
from pathlib import Path

from src.config import settings

PAIRS = settings.DATA_DIR / "datasets" / "writeups" / "pairs.jsonl"
OUT = settings.DATA_DIR / "models" / "llm_prior_lora"
R, ALPHA, DROPOUT, LR, EPOCHS, MAX_TOKENS, SEED, ACCUM, HELD_OUT = 16, 32, 0.05, 1e-4, 1, 1024, 0, 8, 0.10
TARGET = r"^(?!.*visual).*\.(q_proj|k_proj|v_proj|o_proj|gate_proj|up_proj|down_proj)$"
SYSTEM = "You are a digital forensic examiner. You plan and explain investigations."
USER = ("Case context:\n{context}\n\nHow do you investigate this case? Say where you look on the evidence, which "
        "tools you use and what you find.")


def encode(proc, pair: dict) -> tuple[list[int], list[int]]:
    """Token ids and labels (-100 on the prompt: the loss is on the solution only), truncated to MAX_TOKENS."""
    msgs = [{"role": "system", "content": [{"type": "text", "text": SYSTEM}]},
            {"role": "user", "content": [{"type": "text", "text": USER.format(context=pair["context"])}]}]
    prompt = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    tok = proc.tokenizer
    p = tok(prompt, add_special_tokens=False)["input_ids"]
    a = tok(pair["solution"] + tok.eos_token, add_special_tokens=False)["input_ids"]
    ids = (p + a)[:MAX_TOKENS]
    return ids, ([-100] * len(p) + a)[:MAX_TOKENS]


def mean_loss(model, examples, device) -> float:
    import torch

    model.eval()
    tot = n = 0
    with torch.inference_mode():
        for ids, lab in examples:
            out = model(input_ids=torch.tensor([ids], device=device), labels=torch.tensor([lab], device=device))
            k = sum(1 for x in lab[1:] if x != -100)
            tot, n = tot + out.loss.item() * k, n + k
    return tot / max(n, 1)


def train(pairs_path: Path = PAIRS, out: Path = OUT, limit: int | None = None) -> Path:
    import torch
    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForImageTextToText, AutoProcessor

    random.seed(SEED)
    torch.manual_seed(SEED)
    pairs = [json.loads(l) for l in pairs_path.read_text(encoding="utf-8").splitlines() if l.strip()][:limit]
    order = list(range(len(pairs)))
    random.Random(SEED).shuffle(order)
    n_ho = max(1, round(HELD_OUT * len(pairs)))
    ho, tr = [pairs[i] for i in order[:n_ho]], [pairs[i] for i in order[n_ho:]]

    device = settings.device_for("vision")
    model = AutoModelForImageTextToText.from_pretrained(settings.VISION.vlm_model_id, dtype=torch.bfloat16,
                                                        device_map={"": device})
    proc = AutoProcessor.from_pretrained(settings.VISION.vlm_model_id)
    tr_ex, ho_ex = [encode(proc, p) for p in tr], [encode(proc, p) for p in ho]
    loss_before = mean_loss(model, ho_ex, device)

    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()
    model = get_peft_model(model, LoraConfig(r=R, lora_alpha=ALPHA, lora_dropout=DROPOUT, target_modules=TARGET,
                                             task_type="CAUSAL_LM"))
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR)
    steps = math.ceil(len(tr_ex) * EPOCHS / ACCUM)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: max(0.0, 1 - s / steps))
    log, t0, run = [], time.perf_counter(), 0.0
    for ep in range(EPOCHS):
        idx = list(range(len(tr_ex)))
        random.Random(SEED + ep).shuffle(idx)
        model.train()
        for i, j in enumerate(idx):
            ids, lab = tr_ex[j]
            loss = model(input_ids=torch.tensor([ids], device=device),
                         labels=torch.tensor([lab], device=device)).loss / ACCUM
            loss.backward()
            run += loss.item()
            if (i + 1) % ACCUM == 0 or i + 1 == len(idx):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                log.append({"step": len(log) + 1, "loss": round(run, 4), "s": round(time.perf_counter() - t0)})
                print(f"[finetune] step {len(log)}/{steps} loss {run:.3f} ({log[-1]['s']} s)", flush=True)
                run = 0.0
    loss_after = mean_loss(model, ho_ex, device)
    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out)
    (out / "training.json").write_text(json.dumps({
        "base_model": settings.VISION.vlm_model_id, "pairs": len(pairs), "train": len(tr), "held_out": len(ho),
        "held_out_sources": [p["source"] for p in ho],
        "recipe": {"r": R, "alpha": ALPHA, "dropout": DROPOUT, "lr": LR, "epochs": EPOCHS, "max_tokens": MAX_TOKENS,
                   "seed": SEED, "accumulation": ACCUM, "target": TARGET},
        "trainable_parameters": trainable, "train_tokens": sum(len(i) for i, _ in tr_ex),
        "held_out_loss_before": round(loss_before, 4), "held_out_loss_after": round(loss_after, 4),
        "held_out_perplexity_before": round(math.exp(loss_before), 2),
        "held_out_perplexity_after": round(math.exp(loss_after), 2), "log": log,
        "seconds": round(time.perf_counter() - t0)}, indent=1), encoding="utf-8")
    print(f"[finetune] held-out loss {loss_before:.3f} -> {loss_after:.3f}")
    return out


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, help="smoke test on the first N pairs")
    ap.add_argument("--out", type=Path, default=OUT)
    a = ap.parse_args()
    print(train(out=a.out, limit=a.limit))
