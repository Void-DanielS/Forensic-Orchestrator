# Pre-registration: planner vs logistic regression, and the orchestrated agents

Fixed on 2026-11-17, **before** any of the numbers it describes were computed. The code is in
`src/ml/planner_protocol.py` (definitions, criteria, bootstrap). Any later change is listed under *Deviations* with
its date and reason; the original text is not edited.

## Context (known before this protocol)

Earlier exploratory runs compared many planner variants on the held-out cases themselves (best seed ensemble AP
0.323 vs logistic regression 0.301; standard error across cases about 0.075). Those numbers are exploratory: the
variants were looked at on held-out cases, so they cannot support a claim. The candidate list below was drawn up
with knowledge of them; this is stated in the paper.

## 1. Planner: primary comparison

* **Data:** planner dataset `data/datasets/planner/20261116` (16 solved cases, 33,047 files, 849 relevant).
* **Labels (primary):** unchanged: a file is relevant when the answer key names it or its bytes contain an
  answer-key identifier.
* **Held-out cases:** leave-one-case-out; metrics are computed on the 13 *rankable* held-out cases (≥ 50 files, not
  every file relevant), each case counting once.
* **Primary metric:** average precision (AP) per held-out case. Secondary, reported but not used for the decision:
  AUROC, coverage@10/50/100.
* **Baseline:** L2 logistic regression on the metadata vector, case-balanced weights, fitted on the 15 non-held-out
  cases (`planner_train.fit_logreg`), unchanged.
* **The one final model** (nested selection, nothing chosen on the held-out case):
  1. For held-out case *t*, the training pool is the other 15 cases. Inner-validation cases are the pool cases with
     ≥ 10 relevant files.
  2. Each candidate in `CANDIDATES` (`metadata`, `anchored`, `wide & deep`, `wide & deep + aug`; plus
     `wide & deep + LLM prior` if the zero-shot prior has been computed with its single fixed prompt) is trained once
     per inner-validation case *v* (trained on pool − *v*) and per seed (13, 14, 15), 10 epochs. In
     `wide & deep` the frozen logistic-regression part is fitted on pool − *v* only (the inner-validation case is
     never inside it).
  3. The (variant, epoch) with the highest mean inner-validation AP over all inner folds and seeds is chosen. Epoch 0
     exists only for wide & deep variants (where it is logistic regression on pool − *v*).
  4. The final model of fold *t* is the mean logit of all its inner models (inner folds × seeds) at the chosen epoch.
* **Decision: "clearly beats logistic regression"** requires all three, on the 13 rankable cases:
  1. mean AP gain (final − LR) ≥ **0.05**;
  2. paired bootstrap over cases (10,000 resamples, seed 0): lower bound of the 95 % percentile CI of the mean gain
     **> 0**;
  3. AP(final) > AP(LR) on **≥ 9 of 13** cases (ties are not wins).

  Only the final model is tested. Per-variant numbers are reported as descriptive, never as the claim.

## 2. Planner: secondary metric "user-level files only"

* **Definition** (`planner_protocol.is_container`, decided before any number): the held-out case's files are
  restricted to non-container files. Container files are those in the categories `fs_metadata`, `memory_image` or
  `network_capture`. Also excluded are NTFS/FAT/exFAT metafiles (`$MFT`, `$LogFile`, `$Boot`, `$FAT1`, ...; but not
  the recycle-bin `$I`/`$R` files), `pagefile.sys` / `hiberfil.sys` / `swapfile.sys`, raw images and captures
  (`.raw .dd .img .dmp .vmem .mem .e01 .vhd .vhdx .vmdk .pcap .pcapng`), NAND dumps `mtd*.bin`, and paths with a
  component starting with `unalloc` or `slack`.
* The models are **not** retrained: the same scores are evaluated on the subset. Rankability uses the same rule on
  the subset.
* It is reported **next to** the primary metric, with the same three statistics, and is **not** used for the claim.
* The paper states openly that the primary labels favour logistic regression, because container files and large
  databases contain answer-key bytes almost everywhere.
* Known before computing any metric (count only, no labels read): only 123 of 33,047 files are container files under
  this definition; the dataset has no pagefile or unallocated files.

## 3. Orchestrated agents (end-to-end)

* Run only after the planner is frozen (the final model above, LOCO folds). The GPU is not used by anything else
  during the run.
* **Cases:** the solved cases that have automatic answer-key identifiers. `mvs_2024` and `ngdc_2012` are excluded
  (none), and the paper says so.
* **Metric:** answer-key identifiers present in VERIFIED findings, split by source:
  * LLM quotes (`analyst.<type>`);
  * the deterministic extraction tool (`analyst.tool.*`);
  * image agent;
  * union.
* **Systems**, all with the same K:
  * **full:** planner ranking, rounds with feedback, analyst with the tool;
  * **no feedback:** planner ranking chosen once;
  * **no tool:** full system with the extraction tool disabled;
  * **LR control:** top-K by logistic regression, same agents, no feedback.
* **K:** start at 100. If timing one case on a free GPU shows the full run cannot finish overnight, K is reduced
  using that timing **only**, before any results are compared, and the reduction is recorded here.

## Deviations

* **2026-11-17: one added planner candidate, `wide & deep + tuned prior`.** Added while the nested run of section 1
  was in progress and before any held-out number from it was read (its output files had not been opened).
  * **Corpus:** public DFIR challenge write-ups (GitHub repositories of CTF/lab solutions). Each write-up becomes one
    plain-text pair, *case context* (the scenario text before the first question/solution section) → *solution*
    (the rest). No other annotation, no hand-built features.
  * **Exclusion (no leakage):** a write-up is dropped if its path, title or text names any of the 16 dataset cases
    (alias list in `src/ml/writeups.py`), or if it contains any answer-key identifier of the 16 cases. The list and
    the filter are fixed before training. (Clarified before the filter was first run: matching is a
    case-insensitive substring test; the identifiers shorter than 4 characters, 3 of 504, are not used because
    they match almost any text.)
  * **Fine-tuning:** LoRA on the same Qwen2.5-VL-3B language model used for the zero-shot prior; causal-LM loss on
    the solution tokens given the context. Fixed recipe, no search: r = 16, alpha = 32, dropout 0.05, all
    attention and MLP projections of the language model, learning rate 1e-4, 1 epoch, max 1,024 tokens, seed 0. No
    dataset case or label is seen in training, so one tuned model serves every fold.
  * **Prior:** identical prompt, scoring and normalisation as `wide & deep + LLM prior`, with the tuned weights.
  * **Selection:** the candidate joins the nested selection of section 1 (same inner folds, seeds, epochs); the
    (variant, epoch) is still chosen by mean inner-validation AP over all six candidates, so there is still ONE
    final model. Decision rule unchanged.
  * **Note for the paper:** the base LLM may have seen write-ups of the dataset cases in pre-training; this applies
    equally to the zero-shot prior and cannot be controlled.

## Decisions recorded under the protocol

* 2026-11-17: K for section 3. One case (nist_data_leakage, full system) took 337 s on a free GPU (about 3.4 s per
  file), so 14 cases × 4 systems take at most about 5 h. **K = 100 is kept.** The decision was made from the timing
  only; no system comparison existed.
* 2026-11-17, before any fine-tuning (at the user's request): **cleaning of the write-up text** so the examiner's
  actions dominate the solution.
  * The solution is laid out as `Approach` / (`Question` → the examiner's own steps → `Answer`).
  * Removed:
    * questions answered without any step;
    * page metadata, links, images, separators, emoji and trailing summary / reference sections;
    * pasted tool output beyond the first 8 lines of a code block.
  * Non-English write-ups (> 5 % non-ASCII letters) are dropped.
  * The leak filter is applied to the raw page, before cleaning, so it is stricter.
  * Result: 284 pairs (108 dropped by the identifier filter, 0 naming a case); 73 % of solution tokens fit in
    1,024 tokens.
