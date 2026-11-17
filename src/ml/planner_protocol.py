"""Pre-registered evaluation protocol for the investigation planner (fixed 2026-11-17, before any number below was
computed; see ``docs/PREREGISTRATION.md``). Do not edit after results exist; record deviations in the document.

* ``is_container`` - the "user-level files only" secondary metric excludes container-level files: file-system
  metafiles, memory / page / hibernation files, raw disk or partition images, unallocated / slack space and packet
  captures. These hold the bytes of many records at once, so under the byte-match label they are positive in almost
  every case and are found by size and category alone.
* ``verdict`` - the "clearly beats logistic regression" decision: ONE final model against LR on the primary metric
  (AP, original labels, rankable held-out cases); all three conditions must hold.
"""
from __future__ import annotations

import re

import numpy as np

# ---------------------------------------------------------------------------------------------- user-level files
CONTAINER_CATEGORIES = frozenset({"fs_metadata", "memory_image", "network_capture"})
# NTFS / FAT / exFAT metafiles (the recycle bin's $I... / $R... files are user files and are NOT matched)
_METAFILE = re.compile(r"^\$(mft|mftmirr|logfile|boot|attrdef|bitmap|upcase|volume|badclus|secure|extend|usnjrnl|"
                       r"quota|objid|reparse|fat\d*|mbr|alloc_bitmap|upcase_table)$", re.I)
_SYSTEM_BLOBS = frozenset({"pagefile.sys", "hiberfil.sys", "swapfile.sys"})
_IMAGE_EXT = re.compile(r"\.(raw|dd|img|dmp|vmem|mem|e01|vhd|vhdx|vmdk|pcap|pcapng)$", re.I)
_NAND = re.compile(r"^mtd\d+.*\.bin$", re.I)
_UNALLOC = re.compile(r"(^|/)(unalloc|slack)", re.I)          # a path component starting with it (not "free_space_x.pdf")


def is_container(row: dict) -> bool:
    """True for container-level files (excluded from the user-level metric). Uses only the category, the file name
    and the path; never the label."""
    path = str(row.get("path") or "")
    name = path.replace("\\", "/").rsplit("/", 1)[-1]
    return bool(row.get("category") in CONTAINER_CATEGORIES or _METAFILE.match(name)
                or name.lower() in _SYSTEM_BLOBS or _IMAGE_EXT.search(name) or _NAND.match(name)
                or _UNALLOC.search(path))


# ---------------------------------------------------------------------------------------------- "clearly beats LR"
MIN_MEAN_GAIN = 0.05          # mean AP(final) - AP(LR) over rankable held-out cases
MIN_WINS = 9                  # cases with AP(final) > AP(LR) (ties are not wins), out of 13
BOOTSTRAP = 10_000            # paired bootstrap over cases, percentile 95 % CI of the mean gain, lower bound > 0
BOOTSTRAP_SEED = 0
CI = 0.95


def paired_bootstrap(delta: np.ndarray, n: int = BOOTSTRAP, seed: int = BOOTSTRAP_SEED, ci: float = CI
                     ) -> tuple[float, float]:
    """Percentile CI of the mean of per-case differences, resampling cases with replacement."""
    rng = np.random.default_rng(seed)
    d = np.asarray(delta, float)
    means = d[rng.integers(0, len(d), (n, len(d)))].mean(1)
    a = (1 - ci) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def verdict(ap_final: dict[str, float], ap_lr: dict[str, float]) -> dict:
    """Per-case AP of the final model and of LR (same cases) -> the pre-registered decision."""
    cases = sorted(ap_lr)
    d = np.array([ap_final[c] - ap_lr[c] for c in cases])
    lo, hi = paired_bootstrap(d)
    wins = int((d > 0).sum())
    gain = float(d.mean())
    return {"cases": len(cases), "mean_gain": round(gain, 4), "ci95": [round(lo, 4), round(hi, 4)], "wins": wins,
            "criteria": {"mean_gain>=%.2f" % MIN_MEAN_GAIN: gain >= MIN_MEAN_GAIN, "ci_lower>0": lo > 0,
                         "wins>=%d" % MIN_WINS: wins >= MIN_WINS},
            "clearly_beats_lr": bool(gain >= MIN_MEAN_GAIN and lo > 0 and wins >= MIN_WINS),
            "per_case_delta": {c: round(float(x), 4) for c, x in zip(cases, d)}}


# ---------------------------------------------------------------------------------------------- final model
# Nested selection: for each held-out case, every candidate is trained once per inner-validation case (leave one
# validation case out of the 15 training cases; validation cases = training cases with >= 10 relevant files) and per
# seed. (variant, epoch) maximising the mean inner-validation AP is chosen; the final model of the fold is the mean
# logit of all its inner models (inner folds x seeds) at that epoch. Epoch 0 exists only for variants whose initial
# state is a fitted model (wide & deep = logistic regression on the inner training cases).
CANDIDATES = ("metadata", "anchored", "wide & deep", "wide & deep + aug")
PRIOR_CANDIDATES = ("wide & deep + LLM prior",)      # added only if the zero-shot prior has been computed (one fixed
                                                     # prompt, no prompt tuning)
TUNED_CANDIDATES = ("wide & deep + tuned prior",)    # deviation of 2026-11-17 (docs/PREREGISTRATION.md): prior from
                                                     # the LLM LoRA-tuned on public write-ups, same prompt
SEEDS = (13, 14, 15)
EPOCHS = 10
MIN_VAL_POSITIVES = 10
