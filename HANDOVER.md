# HANDOVER: ForensiAgent-Multimodal

_Last updated: 2026-11-17 (read §000 first: planner metrics are the open problem; orchestrated LLM agents built). Previous: 2026-11-16 (deep-learning focus: Jev-MM multimodal model, investigation planner + self-supervised encoders, DL pipeline, evidence repair).
Read §0 first, then §1c / §1d. README.md is the full, peer-facing project report (results, metrics, limitations)._

## 00000. 2026-11-17 night HANDOFF: pre-registered RESULTS + what to fix next (READ THIS FIRST)

### Status of the runs
* **Planner (nested LOCO, section 1 of `docs/PREREGISTRATION.md`):** DONE. Report:
  `data/outputs/ml/planner/nested_20261117/REPORT.md` (also `data/outputs/planner_report.log`). Final models deployed
  to `data/outputs/ml/planner/nested_20261117/models/<case>.pt`.
* **Fine-tune (LoRA, tuned prior):** DONE. 284 write-up pairs (256 train / 28 held out), held-out loss
  **2.531 -> 2.320**. Adapter + stats: `data/models/llm_prior_lora/` (`training.json`). Tuned prior:
  `llm_prior_tuned.npz`. Candidate added to the nested run with `planner_nested extend`.
* **Orchestration evaluation (section 3, K = 100, 4 systems):** started 22:13 by `scripts/after_nested.sh`
  (background). Output `data/outputs/ml/orchestrate/eval_k100_final/results.json` (resumable: re-running the same
  command skips finished `case|mode` keys) + `REPORT.md` written at the end; log `data/outputs/orchestrate_eval.log`.
  If it did not finish, resume with:
  `HF_HUB_OFFLINE=1 FORENSI_VLM_QUANT=none .venv/Scripts/python -W ignore -m src.eval.orchestrate_eval --planners
  data/outputs/ml/planner/nested_20261117/models --out data/outputs/ml/orchestrate/eval_k100_final`
  (nothing else on the GPU). Check first: `grep "\[orchestrate-eval\]" data/outputs/orchestrate_eval.log`.

* **2026-11-18 01:4x: the evaluation hit CUDA out-of-memory at `owl|lr` (47 of 56 runs saved).** Resumed with
  `scripts/resume_final.sh` (same command + `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`, appends to the
  same log; finished runs are skipped), which then runs the two steps below. If it stops again: re-run
  `bash scripts/resume_final.sh` (every step is resumable / idempotent). Mention the resume in the paper (no
  configuration change).
* **Queued after it (now run by `scripts/resume_final.sh`; `final_chain.sh` superseded):** (1) `python -m src.ml.mm_train improve`:
  Jev-MM recipes chosen on validation, 3-seed ensemble tested once, LR baseline, deployed to
  `data/models/multimodal/jev_mm.pt` (old single model kept as `jev_mm_single_seed.pt`); report
  `data/outputs/ml/multimodal/improve_*/REPORT.md`, log `data/outputs/mm_improve.log`. (2) final-system run, `full`
  mode only, improved Jev-MM: `data/outputs/ml/orchestrate/final_system_k100/` (log `data/outputs/final_system.log`).
  Both recorded in `docs/PREREGISTRATION.md` before running. If not finished: run the two commands of the script.

### RESULTS (2026-11-18, final)
**Jev-MM improved** (`data/outputs/ml/multimodal/improve_*/REPORT.md`; recipe chosen on validation = the original
recipe; 3-seed ensemble deployed to `data/models/multimodal/jev_mm.pt`):
| model (test cases) | accuracy | macro-F1 | user-gen. AUROC |
|---|---|---|---|
| **Jev-MM 3-seed ensemble** | **0.759** | **0.553** | 0.911 |
| Jev-MM single seed (mean of 3) | 0.744 | 0.533 | 0.899 |
| LR metadata | 0.655 | 0.488 | **0.915** |
| LR metadata + text | 0.692 | 0.489 | 0.893 |
| majority class | 0.450 | | 0.5 |
Per-class F1: camera 0.872, app_asset 0.875, thumbnail 0.784, web_cache 0.658, screenshot 0.340, user_file 0.192,
messaging 0.154 (small classes are the weakness).

**Pre-registered orchestration evaluation (section 3) DONE** (`data/outputs/ml/orchestrate/eval_k100_final/
REPORT.md`; 56 runs, one OOM resume). 869 answer-key identifiers in 14 cases:
| system | LLM quotes | tool | image | union | cases with a hit | LLM quotes rejected / total |
|---|---|---|---|---|---|---|
| full | 21 | 132 | 0 | **136 (15.7 %)** | 7/14 | 545 / 1707 |
| no feedback | 24 | 132 | 0 | **138 (15.9 %)** | 7/14 | 799 / 2572 |
| no tool | 25 | 0 | 0 | **25 (2.9 %)** | 6/14 | 689 / 2262 |
| LR ranking | 19 | 127 | 0 | **132 (15.2 %)** | 7/14 | 680 / 2404 |
Reading: (1) the deterministic extraction tool is what finds identifiers (136 -> 25 without it); (2) planner vs LR
ranking end to end: 136 vs 132, no real difference (consistent with the planner verdict); (3) feedback rounds add
nothing (138 without); (4) the verifier rejects about a third of LLM quotes (hallucinations never become evidence);
(5) the image agent finds 0 identifiers (identifiers are text; its role is triage). 7 of the 14 cases have junk or no
identifiers (dfrws2005/09/16/17/18, dstl, vto_qysea partly): all hits come from lone_wolf, m57, nist_data_leakage,
nist_hacking, nist_rhino, owl, vto_parrot. Post-hoc table on the informative cases: TO DO (benchmark section).

**Final-system run DONE** (`full`, improved Jev-MM, `data/outputs/ml/orchestrate/final_system_k100/REPORT.md`;
one OOM resume at owl): **136 / 869 (15.7 %), 7/14 cases with a hit, 545 / 1707 LLM quotes rejected**, identical
to the pre-registered `full` row case by case (the image model change does not affect text identifiers). This is the
deliverable system's end-to-end number.

**ALL RUNS ARE FINISHED. Nothing is running. Next session = writing only (benchmark tables, PREREGISTRATION notes,
identifier-filter fix as post-hoc, paper).**

### Jev-MM (image model): DL clearly beats LR
Test (case-level split): Jev-MM 0.744 acc / 0.533 macro-F1 / 0.899 user-gen. AUROC (mean of 3 seeds) vs logistic
regression (L2 chosen on val) metadata 0.655 / 0.488 / 0.915, metadata+text 0.692 / 0.489 / 0.893; majority 0.450.
The improved ensemble's numbers go in the `improve_*` report (fill in here and in the paper).

### Planner verdict (pre-registered, final; do NOT re-select on held-out cases)
**Does NOT clearly beat logistic regression.** 13 rankable held-out cases:
| | AUROC | AP | cov@10 | cov@50 | cov@100 |
|---|---|---|---|---|---|
| logistic regression | 0.767 | **0.301** | **0.440** | **0.708** | **0.782** |
| final model (nested) | **0.813** | 0.292 | 0.409 | 0.652 | 0.747 |

* Mean AP gain **-0.009**, 95 % CI [-0.074, +0.060], wins **7/13** -> 0 of 3 criteria met.
* Secondary (user-level files only): gain -0.014, CI [-0.126, +0.087], wins 6/11.
* Selection chose `wide & deep + LLM prior` on 12 folds, `+ tuned prior` on 1 (dfrws2018), **always epoch 0**
  (= frozen LR + prior, no network training wins on inner validation).
* Big wins m57_patents +0.25, mvs_2024 +0.19; big losses dfrws2017 -0.20, ngdc_2012 -0.17, nist_hacking -0.13.
* Descriptive only (NOT a claim; looked at after the fact): `metadata` MLP alone AP 0.350, `anchored` 0.326.
* Reading: the LLM prior improves global ordering (AUROC +0.05) but not the top of the list (AP, cov@10).

### Orchestration: what was seen so far (partial, at 23:30)
* Cases done: dfrws2005, dfrws2009, dfrws2016, dfrws2017, dfrws2018, dstl (partial): **union = 0 for every
  system.** This is NOT a scoring bug (the 11:55 timing run of nist_data_leakage scored 47/250 with the same code).
* **Cause: for these cases the automatic "answer-key identifiers" are junk**: Dropbox / Google Drive download URLs
  and archive names from the case summary (`www dropbox com`, `dfrws2005 timeline txt`, `edge 7z`, ...). No agent
  can or should find them in the evidence. dfrws2017 has **0** identifiers (evaluation `None`) and should have been
  excluded like mvs_2024 / ngdc_2012.
* The informative cases are the rest: lone_wolf, m57_patents, nist_data_leakage, nist_hacking_case,
  nist_rhino_hunt, owl, vto_parrot_bluegrass, vto_qysea_fifish_p3.

### Paper framing decided with the user: BENCHMARK (important for the paper)
Results that are not a win (planner, orchestration if weak, tuned prior) go into the paper as **benchmarks
alongside the other models**, not hidden and not oversold. One benchmark table per task, every model on the same
held-out cases and metrics:
* **Image origin (Jev-MM task):** majority class, LR metadata, LR metadata+text, Jev-MM single modalities (vision /
  metadata / text / vision+metadata), Jev-MM all (single seed, mean +- std), Jev-MM ensemble (improve run).
  Accuracy, macro-F1, user-gen. AUROC. Sources: `multimodal/20261115_191432/REPORT.md`, `multimodal/improve_*`.
* **File type from fragments (Jev):** histogram baseline vs Jev CNN (acc, macro-F1), README §6.5.
* **Evidence ranking (planner), 13 LOCO cases:** LR, metadata MLP, anchored, wide & deep, + aug, + LLM prior,
  + tuned prior (each at its inner-validation-best epoch: the "Descriptive only" table of the nested REPORT), and the
  pre-registered final model. AUROC, AP, cov@10/50/100; plus the user-level-files version. State clearly which row
  is the pre-registered claim (final vs LR: no clear win) and that the others are descriptive (selected on inner
  validation, not on the held-out case, but not pre-registered as the claim).
* **LLM fine-tune:** held-out write-up loss / perplexity, zero-shot vs LoRA.
* **End-to-end agents (K = 100):** full / no feedback / no tool / LR control (+ final system run), by source; main
  table on all cases + post-hoc table on cases with real identifiers.
Each table gets a one-line reading (what wins, by how much, CI where available). The text leads with the
benchmark, then the pre-registered verdicts.

### TO FIX / DO next session (in order)
**THE NEXT SESSION IS THE LAST ONE (user). Scope = items 1-4 + 7 only: writing, benchmark tables, paper,
commit. Items 5-6 are NOT done; mention them in the paper as future work. Step 3 only if time allows (post-hoc).**
Paper must also include the Jev-MM ensemble row (0.759 / 0.553 / 0.911) and the final-system run (136/869).
1. **Read the orchestration results** (`eval_k100_final/REPORT.md`; regenerate with `orchestrate_eval report`
   if missing). Report the by-source table honestly.
2. **Record in `docs/PREREGISTRATION.md` (Decisions / Deviations, dated, results already known so say so):**
   (a) dfrws2017 has no automatic identifiers -> excluded from section 3, same rule as mvs/ngdc; (b) cases whose
   identifiers are only download URLs / archive names (dfrws2005/2009/2016/2018, dstl partly) are uninformative;
   report the main table on all cases as pre-registered AND a clearly labelled post-hoc table on the informative
   cases. Do not drop them silently.
3. **Identifier filter (root cause, affects planner labels too):** `src/eval/scoring.key_entities` takes URLs /
   download links / evidence archive names from the case summary. Add a filter (drop URLs, cloud-storage domains,
   names of the evidence archives themselves). Any re-labelled planner run is a NEW, post-hoc analysis: label it so,
   never replace the pre-registered numbers.
4. **Paper `../dl-report-final.tex`:** fill `\PLANABSTRACT`, `\PLANNERRESULTS`, `\PLANDISCUSSION` with the verdict
   above (negative result, stated plainly: no clear win; AUROC up, AP flat; selection always picks epoch 0),
   the fine-tune held-out loss, the user-level metric, and the orchestration table + identifier-quality caveat.
   Fix overfull hboxes. Compile: `%LOCALAPPDATA%/ForensiAgent/tools/tectonic.exe -X compile`.
5. **Course requirement "one DL planner clearly beats LR" is NOT met under the protocol.** Options that stay
   legitimate (all NEW pre-registrations, written before running): better labels (step 3) with the same rule;
   train for top-of-list ranking (pairwise / listwise loss, since AP is the metric); use the tuned LLM more
   directly (fine-tune it as the ranker on the planner rows, LOCO). Discuss with the user before spending GPU.
6. Then the **agent part** (user: "the agent part is next"): use the fine-tuned model (it knows where to look /
   which tools) inside the analyst agent.
7. Commit: `scripts/after_nested.sh`, this HANDOVER, any report files (code zips for peers exclude HANDOVER.md,
   PROJECT_REPORT.html and Claude/prompt files).

## 0000. 2026-11-17 (later): pre-registered protocol + fixes (read first)

The user approved a protocol after a diagnosis. It is committed in `c91f4d6`: `docs/PREREGISTRATION.md` +
`src/ml/planner_protocol.py`. Do NOT change it after seeing held-out numbers; log deviations in the doc.

- **"Clearly beats LR":** ONE final model chosen by nested inner-LOCO selection. It must meet all three:
  - mean AP gain ≥ 0.05;
  - paired-bootstrap 95 % CI lower bound > 0;
  - wins on ≥ 9 / 13 cases.
- **User-level metric:** secondary, defined by `is_container`. Only 123 / 33k files are containers; there is no
  pagefile or unallocated space in the dataset.
- **Orchestration:** results are split by source (LLM / tool / image / union / LLM-only), and the run includes a
  no-tool ablation and an LR top-K control. It runs only after the planner is frozen; mvs_2024 and ngdc_2012 are
  excluded.

Fixes done (uncommitted):

1. **Wide & deep leak:** the frozen LR part was fitted on train + the validation case, which then chose the epoch.
   It is now fitted on the training cases only (`planner_train.train_model`).
2. **Nested selection:** `src/ml/planner_nested.py`.
   - `run`: resumable, per-fold `.npz`/`.json`.
   - `report`: verdict + user-level metric.
   - `deploy`: rebuilds the ensembles and checks that they reproduce the recorded scores.
   - The ensemble checkpoint is `planner.EnsemblePlanner`.
   - `PlanData.cache(device)` keeps the inputs on the GPU, about 2× faster with identical values.
3. **LLM prior:**
   - **bf16 rounding:** scores came out in 0.125 steps (60 distinct / 600). The Yes/No logits are now computed in fp32
     from the last hidden state; 1,787 distinct / 2,000, corr 0.994 with the old scores.
   - **RAM:** the commit charge grew because each padded length left blocks in the CUDA cache, and on WDDM those count
     toward the process commit (10 → 15 GB in 600 files). Fixed with padding to multiples of 64 + periodic
     `empty_cache` (flat at 11.4 GB).
   - **Speed:** about 90 ms/file, so about 50 min for the whole dataset.
   - **Wiring:** variant `wide & deep + LLM prior` uses `PlanData.with_prior()`; `solve.prior_columns` supplies the
     prior at inference.
4. **Orchestration:**
   - `investigate(tools=, ranker="lr")`, `orchestrate.split_evaluation`, `scoring.found_entities`;
   - `orchestrate_eval` is resumable, with modes full / nofeedback / notool / lr and `--planners <dir>`;
   - CLI `--no-tool --ranker lr`.

Progress 2026-11-17:

- **Prior:** done, `data/datasets/planner/20261116/llm_prior.npz` (33,047 scores, 20,491 distinct, 63 min).
- **Orchestration timing:** 337 s per case at K = 100 on a free GPU, so K = 100 is kept (recorded in the
  pre-registration, `b632621`).
- **Nested run:** STARTED at about 11:50 → `data/outputs/ml/planner/nested_20261117`, log
  `data/outputs/planner_nested.log`. It is resumable: rerun the same command and finished folds are skipped.
- **Paper:**
  - Tectonic is at `%LOCALAPPDATA%/ForensiAgent/tools/tectonic.exe`.
  - With the placeholders stubbed, `../dl-report-final.tex` compiles.
  - The orchestration figure had `\[2pt]` (display math) instead of `\\[2pt]`, and `\st{Verified}` (strikethrough).
    Both are fixed.
  - Compile from the parent folder: `tectonic.exe -X compile dl-report-final.tex`.

**Tuned prior (user request, 2026-11-17; pre-registration deviation `78b39ea` + clarification `742e55f`, written
before any held-out number was read):**

- The user's direction: scrape only plain text (case context → solution) and train only by fine-tuning or deep
  learning. No hand-built features.
- **Corpus:** `src/ml/writeups.py`. It reads the shallow clones in `data/external/writeups/` (git-ignored, untrusted
  data, never executed; one cloned repo ships a CLAUDE.md, which is just data) and writes
  `data/datasets/writeups/pairs.jsonl`:
  - 284 pairs (cleaned so the examiner's actions dominate: Question -> steps -> Answer; ChickenLoner Write_It_UP, Panagiotis-INS Cyber-Defenders, Baniur, dbissell6 DFIR, isaac-do,
    mikehorn-git, stark4n6);
  - leak filter (on the raw page): none named a dataset case; 108 were dropped because they contain an answer-key identifier; 5 non-English dropped.
- **Fine-tune:** `src/ml/finetune_prior.py`.
  - LoRA with the fixed recipe, about 260k training tokens (67 % of the examples are cut at 1,024 tokens).
  - The adapter goes to `data/models/llm_prior_lora/`, and `training.json` records the held-out loss before and
    after (descriptive only).
- **Tuned prior:** `python -m src.ml.llm_prior --adapter data/models/llm_prior_lora` → `llm_prior_tuned.npz`, with the
  same prompt and unquantised as before.
- **Candidate:** `wide & deep + tuned prior`, wired as follows:
  - `PRIOR_VARIANTS` is now variant → npz file;
  - `planner_nested extend <run>` trains only the new candidate's inner folds and merges them;
  - selection is still one final model over all six candidates;
  - `deploy` and `solve.plan` read `prior_file` from the checkpoint.
- **Run all of it:** `bash scripts/tuned_prior_chain.sh`, only after the nested run has finished (GPU). It takes
  about 15 min + 63 min + 1.7 h.

Next, in order:

1. Wait for the nested run to finish (about 38 min per case, so done around 20:00).
2. `bash scripts/tuned_prior_chain.sh`.
3. `planner_nested report`, then `deploy`.
4. `python -m src.eval.orchestrate_eval --planners <nested>/models` overnight.
5. Paper (Tectonic allowed). Mention:
   - the write-up corpus and its leak filter;
   - that the base LLM may have seen the dataset cases in pre-training.

## 000. 2026-11-17 HANDOFF: the planner metrics are bad; plan to fix them

**User requirements (2026-11-16/17, hard):**
1. At least one DL planner variant must **clearly** beat logistic regression on metadata ("No, at least one variant has to clearly beat the logistic regression").
2. At least two **intelligent (model-driven, not deterministic) agents orchestrated together**.
3. Then remake the paper (`../dl-report-final.tex`).

The user, on the current numbers: "improve those horrible metrics".

Improve only legitimately: choose models, weights and epochs on training or inner-validation cases, **never** on held-out cases, and say so in the paper.

### Where the planner stands

LOCO evaluation, 13 rankable held-out cases, 3 seeds (mean ± std over seeds).

| model | AUROC | AP | cov@10 | cov@50 | cov@100 |
|---|---|---|---|---|---|
| random | 0.496 | 0.035 | 0.004 | 0.103 | 0.182 |
| category prior | 0.776 | 0.264 | 0.373 | 0.614 | 0.689 |
| **logistic regression (metadata)** | 0.767 | **0.301** | 0.440 | 0.708 | 0.782 |
| anchored | 0.724 | 0.279 ± 0.026 | 0.433 | 0.632 | 0.691 |
| anchored + aug | 0.740 | 0.274 ± 0.016 | 0.451 | 0.659 | 0.699 |
| anchored + Jev header (transfer) | 0.743 | 0.177 ± 0.029 | 0.410 | 0.619 | 0.672 |
| anchored + noisy student | 0.751 | 0.282 ± 0.038 | 0.506 | 0.643 | 0.703 |
| wide & deep | 0.766 | 0.287 ± 0.027 | 0.429 | 0.684 | 0.774 |
| wide & deep + aug | 0.762 | 0.295 ± 0.019 | 0.472 | 0.674 | 0.740 |
| wide & deep + aug, **seed ensemble** | 0.765 | **0.323** | 0.532 | 0.708 | 0.755 |
| wide & deep + refit | 0.751 | 0.266 ± 0.011 | 0.416 | 0.676 | 0.725 |
| anchored + refit | 0.722 | 0.267 ± 0.002 | 0.428 | 0.635 | 0.688 |

- Reports: `data/outputs/ml/planner/seeds_2026111{6,7}_*/REPORT.md`.
- Logs: `data/outputs/planner_seeds_{wide,refit,experimental}.log`.

Nothing clearly beats LR. The 0.323 seed ensemble is within case-to-case noise.

**CPU probes.** These were single scratch runs with 5-seed ensembles. They are exploratory and NOT reportable as-is, because several settings were compared on held-out cases.

| probe | AP |
|---|---|
| MLP on metadata (more epochs or width made it worse) | 0.307–0.320 |
| MLP + content-embedding PCA (more PCs made it worse) | 0.19–0.29 |
| LR + cosine(content, brief) | 0.300 |
| LR + cross-case file-name commonness (from the 50 unlabelled cases) | 0.308 |
| LR/MLP + directory-sibling aggregates (GraphSAGE-like) | 0.26–0.27, but AUROC **0.79–0.81** |
| ListNet / pairwise ranking losses | 0.22–0.32 |

### Diagnosis: why the metrics are bad

1. **The unit is the case, and there are only 16.** Anything learned from content (path characters, snippet embedding, header bytes) fits case-specific cues. Adding content PCs hurts monotonically. Only metadata transfers, and LR already uses it well.
2. **The metric is noisy.** AP is averaged over 13 cases, and 5 of them have 6 or fewer relevant files. dfrws2017 and vto_qysea_fifish_p3 have exactly 1, so their AP is 1/rank. One case moves the mean by about ±0.03, which is the size of every effect above.
3. **Selection is noisy.** Epoch, gate and variant are chosen on ONE inner-validation case. Wide & deep often picks epoch 0 (pure LR) or a bad deep epoch. Refit on train+val removed the validation case and got worse.
4. **The labels are noisy proxies.** "Relevant" means the file's bytes contain an answer-key identifier. So `$MFT`, `$LogFile`, `pagefile.sys`, unallocated blobs and big SQLite databases are positives in almost every case. Metadata (size, category) finds them easily, which is exactly why LR is strong. User-level files are the hard part.
5. **LR had more training data.** LR was fitted on 15 cases, the nets on 14. This is fixed in the wide part and in refit, and is not the main cause.

### Plan to improve

Work through these in order. For each step: run it, record the result here, and decide by inner validation only.

**1. Zero-shot LLM relevance prior (code ready, never completed).**
- `src/ml/llm_prior.py` asks the local Qwen2.5-VL-3B (text-only) "is this file likely to answer the investigation?" for every file.
- It stores `log P(Yes) - log P(No)` from one forward pass (`logits_to_keep=1`). That is about 0.1 s per file, so roughly 10–15 minutes for 33k files.
- Nothing is trained on the labelled cases, so there is nothing to overfit. This is transfer learning from a foundation model.

```
FORENSI_VLM_QUANT=none HF_HUB_OFFLINE=1 .venv/Scripts/python -m src.ml.llm_prior --batch 8
.venv/Scripts/python -m src.ml.llm_prior_eval        # LR vs LLM alone vs LR(metadata + LLM prior)
```

- The full run was **stopped by Claude Code because system RAM ran low**; it is not a bug. Close other programs and use `--batch 8`.
- If LR + prior wins:
  - Add it to the planner as an extra input of the wide part: `wide & deep + LLM prior`, where wide = LR(meta, prior) fitted on train+val.
  - Compute the prior at inference in `solve.plan()` for new cases.
  - Run `planner_seeds` with it and deploy by inner-validation AP.
- Choose prompt variants (brief length, snippet length) on training cases only.

**2. Fine-tune the LLM prior (LoRA, per LOCO fold).**
- If zero-shot helps, fine-tune Qwen with LoRA on the training cases' files (balanced, about 2k examples per fold, 1 epoch), then score the held-out case.
- This is the "fine-tuning" the user asked for, applied where it can transfer (understanding the brief against the file). It avoids encoders that memorise paths.

**3. Robust selection.**
- Replace the single validation case with inner leave-one-case-out (or 3 validation cases) to pick epoch, gate and variant.
- Average the inner folds' models as an ensemble.
- This is cheap for wide & deep because the deep part is small, and it removes cause 3.

**4. Ensembles declared in advance.**
- Seed ensembles already do better (0.323 vs 0.295).
- `planner_seeds` now saves `scores.npz` per run, so a fixed recipe (e.g. the mean of LR-anchored nets over 5 seeds × 2 variants) can be evaluated without new training.
- Decide the recipe BEFORE looking at test numbers, and write it down here.

**5. Graph / directory context.**
- Directory-sibling aggregation raised AUROC to about 0.80 in the probe, though AP fell.
- Build a proper GNN layer inside PlannerNet: GraphSAGE over file → parent directory → grandparent, with the LR anchor, tuned on inner validation.
- This is promising because directory structure transfers across cases and content does not.

**6. Fix the label proxy.** Decide in advance, apply to every model, and report both versions.
- Also report metrics on "user-level" files only (excluding file-system metafiles, pagefile/hiberfil and unallocated), where ranking is the real problem.
- And/or weight positives by their number of distinct answer items.
- Do NOT pick whichever definition favours the net.

**7. Report uncertainty.**
- Use a paired bootstrap over cases or a Wilcoxon signed-rank test for each model vs LR, plus a per-case AP table.
- A "clear" win must hold case by case (wins on most cases), not just in the mean.

**8. More labelled cases (the biggest lever long term).**
- See §00 "Next steps" 1a–1e.
- Also generate **synthetic labelled cases** by planting key-like tokens (names, e-mails, URLs) into copies of benign files from unlabelled cases; the labels are then exact.

**9. Drop or freeze the content and path encoders** inside the deep part unless inner validation keeps them (so far they hurt).

### Code added 2026-11-16/17 (all uncommitted on `sprint-2`)

**Planner:**
- `src/ml/planner.py`: `PlannerConfig.wide`. The wide part is a frozen copy of the LR, plus a gate g = 0 on the deep fusion.
- `src/ml/planner_train.py`:
  - `fit_logreg`, shared by the baseline and the wide part.
  - Variants `wide & deep`, `wide & deep + aug`, `wide & deep + refit`, `anchored + refit`, and `wide & deep + aug + refit` (never run).
  - `train_model(..., wide=, stop_at=)`: refit on train+val at the chosen epoch.
  - `loco(..., scores=)`.
- `src/ml/planner_seeds.py`: `--variants`, seed ensembles (`ensemble()`), `scores.npz`.
- `src/ml/llm_prior.py` and `src/ml/llm_prior_eval.py`: step 1 above.

**Orchestrated agents (requirement 2):**
- `src/agents/llm.py`: `LocalLLM`, sharing the VLM weights.
- `src/agents/analyst_agent.py`: the LLM reads files given the brief and the current leads. Its quotes are located in the file bytes and verified; invented quotes become REJECTED (counted as hallucinations caught). Leads come only from verified quotes.
- `src/pipeline/orchestrate.py`:
  - The planner is the orchestrator: rounds of 25 files and a blackboard of verified leads.
  - Re-rank score: `logit(p) + 2*log1p(hits)`.
  - Image agent: Jev-MM, plus the VLM on the top 3 images per round.
- `src/eval/orchestrate_eval.py` and `main.py orchestrate`.
- Tests: `tests/test_orchestrate.py` (5 tests, fake LLM).
- A smoke test on nist_hacking_case produced good verified quotes and 11 rejected hallucinations.
- **Not yet evaluated on the 16 cases.** Command: `FORENSI_VLM_QUANT=none .venv/Scripts/python -m src.eval.orchestrate_eval --k 100` (feedback vs no-feedback). It ran at about 14 s per file while the GPU was shared, so expect several hours; run it when the GPU is free.

**Paper (`../dl-report-final.tex`):**
- Placeholders still to fill: `\PLANABSTRACT`, `\PLANNERRESULTS`, `\PLANDISCUSSION`.
- New section "Orchestrated Multi-Agent Investigation" is written; orchestration results are still to add.
- The PDF is not compiled: there is no LaTeX on the machine. A portable Tectonic was downloaded to the session scratchpad, but running it was blocked, so ask the user.

**Delivery rule:** code zips for peers exclude HANDOVER.md, PROJECT_REPORT.html and any Claude / prompt files.

## 00. 2026-11-15 HANDOFF (read first): deep-learning focus, multimodal models, evidence repair

**What the user asked (2026-11-15):** "finish the project, strictly the deep-learning part, with the multimodal
requirement ready; check for data losses and fix everything". Then: "multimodality is non-negotiable"; "you can
eliminate the deterministic agents, keep the hallucination filters / safeguards; gather the solutions of the cases
and train a model that learns where to look, what to investigate and what to give to the other agents; only the
necessary agents"; "this is a Deep Learning course project, not a forensics course". → The deliverable is the DL
methodology; forensics is the application domain. Deterministic agents (apps / fs / win) stay in the repo but are
**not** part of the DL pipeline (do not delete them unless the user says so).

### Done this session
1. **Jev-MM multimodal image model** (vision CLIP ViT-B/32 + 43 metadata features + OCR text → fusion transformer;
   `src/ml/mm_dataset.py`, `multimodal.py`, `mm_train.py`; agent `src/agents/mm_agent.py`, `--agents mm`; CLI
   `python main.py mm build|ocr|encode|prepare|train|eval|ask`). Dataset 12,532 images / 57 cases
   (`data/datasets/multimodal/20261115`, thumbnails in `%LOCALAPPDATA%/ForensiAgent/cache`). Test (3 seeds):
   all modalities **0.744 ± 0.015 acc / 0.533 ± 0.015 macro-F1** vs vision 0.662, metadata 0.655, text 0.502,
   majority 0.450. Report `data/outputs/ml/multimodal/20261115_191432/REPORT.md`; checkpoint
   `data/models/multimodal/jev_mm.pt`. README §6.6. Hyper-parameters chosen on validation only (1 ViT block, lr 1e-5).
2. **Data loss audit + repair** (`src/utils/integrity.py`, `python main.py cases audit|repair`). Full SHA-256 audit
   of 162,793 evidence files found 51 damaged: 29 `Recent/desktop.ini` rewritten by Windows, 17 `-shm` rewritten
   (hickman_android_10), 5 missing. Repair re-reads the original image by range requests and writes only on SHA-256
   match; at-risk files are set read-only. **Result: 39 / 51 restored (SHA-256 match), 22,395 at-risk files made
   read-only; 12 not restorable by range reads** (11 `desktop.ini` in narcos 1-3, magnet 2020 / 2022, ahadi 02 / 03 /
   05, dfirmadness + `Windows PowerShell.evtx` in dfirmadness: images deflated / 7z-packed at the source → need a
   full re-download). Details: `data/outputs/integrity/repair_20261115_200729.json`; log
   `data/outputs/integrity_repair.log`. Restoring a read-only file: `integrity._write_evidence` clears the flag.
3. **Jev byte model data loss fixed**: `fragments.py` silently skipped 6,036 files with paths ≥ 260 chars; rebuilt
   (`data/datasets/fragments/20261115`) and retrained: 0.811 acc / 0.658 macro-F1.
4. `fit_temperature` clamped to [0.05, 20] (perfectly separable validation gave T = 0 → division by zero).
5. Tests: `tests/test_multimodal.py` (20), `tests/test_integrity.py` (3), `tests/test_planner.py` (6); full suite 212 pass.
6. Course report source updated: `../dl-report-ep-corrected.tex` (Jev-MM method + results table; no LaTeX on this
   machine, compile on Overleaf).

### Investigation PLANNER + deep-learning pipeline (done 2026-11-15; README §6.7)
* Dataset `data/datasets/planner/20261115`: 28,523 files / 14 solved cases / 802 relevant (labels: file named by the
  answer key OR its bytes contain a key identifier; Aho-Corasick scan). Inputs: path, 256 header bytes, metadata,
  content strings (MPNet), brief (MPNet). Shared feature code `planner_dataset.case_rows()` (train = inference).
* `src/ml/planner.py` (PlannerNet: char-CNN, byte-CNN, MLPs, fusion transformer, optional per-modality aux heads),
  `src/ml/planner_train.py` (leave-one-case-out; inner validation = one other case with >= 10 relevant files; model
  selection by mean inner-validation AP; `rereport(run)`). Run `data/outputs/ml/planner/20261115_202510/REPORT.md`.
  Fold models `data/models/planner/loco/<case>.pt` (selected variant), all-case model `data/models/planner/planner.pt`.
* Results (10 rankable held-out cases): deployed "4 modalities, no context" coverage@50 0.547 / AP 0.219 vs random
  0.042 / 0.039, category prior 0.535 / 0.269, **metadata-only network 0.659 / 0.350 (best)**, logistic regression
  0.662 / 0.322. Honest finding: fusion does not beat metadata here (13 training cases per fold → modality
  imbalance / overfitting; aux heads raised AUROC 0.75 but not AP). Contrast with Jev-MM where fusion wins.
* Pipeline: `python main.py solve --case X [--k 100] [--vlm]` → planner → `ReaderAgent` (`src/agents/reader_agent.py`:
  `reader.file` + `reader.email` / `reader.url`, byte-offset citations) and `ImageTriageAgent` (Jev-MM) [+ VLM] →
  verifier → `workspace/reports/<run>_solve.md`. End-to-end evaluation `python -m src.eval.solve_eval --k 100` →
  `data/outputs/ml/planner/solve_eval_k100_*.md` (answer-key identifiers recovered by VERIFIED findings vs the old
  deterministic agents that read every file).
  **Result (K=100, LOCO planners):** 126 / 869 key identifiers (14.5 %) vs deterministic agents 137 / 869 (15.8 %)
  reading every file; 8 / 14 cases with a hit (vs 4); ~8,200 VERIFIED, 0 REJECTED.
* Course report `../dl-report-ep-corrected.tex`: Jev-MM + planner sections, tables, limitations (integrity numbers),
  new refs (CLIP, SBERT, Guo 2017, Wang 2020). Not compiled (no LaTeX here): compile on Overleaf.

### 2026-11-16 (done): more labelled cases, metadata-anchored fusion, deterministic training (README §6.7, tex updated)
* **Hand-transcribed labels** `src/eval/planner_tokens.json` (solution tokens per case, with sources);
  `planner_dataset.curated_tokens()` adds them as `token` entities in `solved_cases()`. Result: **mvs_2024 (42
  relevant)** and **ngdc_2012 (5)** become labelled → 16 cases, 33,047 files, 849 relevant
  (`data/datasets/planner/20261116`; the previous dataset was renamed `20261115_v1`, which the 2026-11-15 reports
  cite). Drone GPS tokens (VTO Ryze / Yuneec / Parrot) matched nothing (binary flight logs). Curated cases are
  excluded from SSL pretraining (`unlabelled_cases()`); the encoders did not need retraining (both cases already had
  key text, so they were never in the pretraining set).
* **Anchored fusion** (`PlannerConfig.anchor`): logit = metadata head + g · fused head, g init 0, aux BCE on the
  metadata head → fusion starts as the metadata model and can only add corrections. Variants `anchored`,
  `anchored + SSL frozen` (in CANDIDATES and in `planner_seeds.VARIANTS`).
* **Determinism:** `planner_train.deterministic()` + `CUBLAS_WORKSPACE_CONFIG` → identical predictions on rerun
  (checked: max diff 0.0).
* **Results (16 cases, 3 seeds; `data/outputs/ml/planner/seeds_*` latest, LOCO run `20261116_012239`):** anchored
  fusion AP **0.279 ± 0.026**, cov@10 0.433, cov@50 0.632 vs plain fusion 0.194 / 0.323 / 0.517, metadata-only
  network 0.240 / 0.366 / 0.653, SSL-frozen fusion 0.218 / 0.378 / 0.588; logistic regression on metadata 0.301 /
  0.440 / 0.708 (still marginally best, within 1 std). Anchored + SSL frozen adds nothing (0.219). **Deployed:
  `anchored`** (inner-val AP 0.313) → `planner.pt` + `loco/<case>.pt`.
* End to end K=100, anchored folds, 16 cases: **138 / 869** (same total as SSL folds; per case differs), 7 / 16 cases
  with a hit, 9,353 VERIFIED, 0 REJECTED (`solve_eval_k100_20261116_044831.md`). mvs_2024 / ngdc_2012 have no
  automatic key identifiers, so this metric ignores them.
* Tests 217 pass. The token file lives in `src/eval/` (not `gold/`: `test_gold_files_are_well_formed` globs gold).

### Self-supervised planner encoders (done 2026-11-15/16; README §6.7, course report updated)
* `src/ml/planner_pretrain.py`: unlabelled dataset from the **50 READY cases without an answer key** (labelled cases
  excluded entirely; 80,126 files after SHA-256 dedupe; `data/datasets/planner_pretrain/20261115`), cross-modal
  InfoNCE between path / header / metadata / content encoders (same arch as `PlannerNet.enc`). CLI
  `python main.py planner pretrain-build | pretrain` → `data/models/planner/pretrained_encoders.pt`
  (report `data/outputs/ml/planner_pretrain/20261115_230146/REPORT.md`; held-out retrieval@1 0.10-0.35, chance 0.002).
* `planner_train.py`: variants `all - context + SSL` (encoders at 0.1 × lr), `… + SSL frozen`, `metadata + SSL`
  (skipped automatically when the encoder checkpoint is missing). LOCO run `data/outputs/ml/planner/20261115_230158`
  deployed **`all - context + SSL frozen`** (inner-val AP 0.349); `data/models/planner/loco/<case>.pt` + `planner.pt`
  are now that variant. `planner_dataset.scan_file` reads only 64 KB when no labels are needed.
* **GPU non-determinism:** the same seed gave coverage@50 0.547 vs 0.442 for one variant → compare over seeds.
  `python -m src.ml.planner_seeds` (3 seeds, ~1 h, saves nothing to models): frozen SSL vs no SSL AP 0.169 → 0.198,
  cov@10 0.285 → 0.409, cov@50 0.549 → 0.577; still below metadata-only / logistic regression
  (`data/outputs/ml/planner/seeds_20261116_005920/REPORT.md`).
* End to end K=100 with the SSL folds: **138 / 869** (was 126) vs deterministic 137; 0 REJECTED
  (`data/outputs/ml/planner/solve_eval_k100_20261116_010348.md`). Tests 215 pass (`tests/test_planner_pretrain.py`).
* Nothing committed: all 2026-11-15 work is still uncommitted on branch `sprint-2`.

### Next steps (by value)
1. More labelled planner cases (now 16 of 108 registered / 72 READY). Ways to get more, in order of yield:
   a. **Fetch the evidence of cases that already have a public key:** `msab_ctf_2026` (0 / 2 items) and `mvs_2026`
      (0 / 3) have rich question/answer keys but are not READY: `python main.py cases build --cases <id>
      --groups light,heavy`, check with `cases list`, then transcribe solution tokens into `src/eval/planner_tokens.json`.
   b. **Hand-transcribe tokens for READY cases whose key text yields no labels:** `circl_wiped_disk`, `narcos`,
      `nist_russian_tea_room`, `nitroba`, `vto_ryze_tello`, `vto_yuneec_h520`. Read `context/answer_key`, pick
      distinctive names / file names / strings that should appear in evidence bytes (not GPS numbers: drone logs are
      binary), add them with their source to `planner_tokens.json`, rebuild (`python main.py planner build`, then
      `planner encode`), check the case gets >= 5 relevant files.
   c. **Public write-ups for keyless CTFs:** Cellebrite CTF 2021-2025, Magnet CTF 2019-2022, DFRWS 2021 and the
      Hickman phones have community solution blogs / official walkthroughs. Transcribe answer tokens (with the URL as
      `source`) into `planner_tokens.json`; Cellebrite 2021-2025 also need their evidence fetched.
   d. **Image-creation logs as ground truth:** Hickman / CFTT / VTO / NPS 2011 documentation lists the apps, accounts
      and files the creators planted; those names are labels (same token mechanism). Fetch the non-READY ones
      (`cftt_*`, `nps2011_*`, `vto_dji_*`, ...) first.
   e. **New sources:** Digital Corpora scenarios, CFReDS, DFIR Madness, Ali Hadi's challenges and Josh Hickman's newer
      images (Android 14 / iOS 17) publish evidence + answers; register them in `src/cases/registry.py`. Respect the
      1.5 GB / 1,500-image per-source budget and the >= 100 READY case target.
   Re-run `python -m src.ml.planner_seeds` after each batch: with ~13 training cases per fold, fusion is data-starved,
   so every added labelled case is the most direct way to improve the planner.
1b. **User request (2026-11-16): improve the planner with data augmentation / data generation, transfer learning
   and fine-tuning** (goal: the multimodal anchored fusion, AP 0.279, should clearly beat logistic regression on
   metadata, AP 0.301). A **draft is in the code but has NOT been run or evaluated** (smoke-tested only: forward
   pass, Jev weight load; 217 tests pass). It lives in `src/ml/planner_train.py` and `src/ml/planner.py`:
   * `EXPERIMENTAL` dict + `EXTRA` options (deliberately **not** in `ABLATIONS` / `CANDIDATES`, so default
     `planner train` runs are unchanged); `train_variant(data, name, ...)` trains any variant by name.
   * `anchored + aug`: `augment()`: path prefix crop + 3 % char substitution outside the last 24 chars (file name),
     2 % header byte noise after the 16-byte signature, Gaussian jitter on continuous metadata columns
     (`META_CONT`), dropout + noise on the content embedding, and **synthetic positives** (SMOTE-style
     interpolation of metadata / content between two positives, lambda ~ U(0.5, 1)).
   * `anchored + Jev header`: **transfer learning**: `PlannerConfig.header_jev` swaps the header encoder for
     `JevHeader` (the trained Jev byte CNN trunk from `data/models/jev/jev.pt`, `load_jev()`), **fine-tuned** at
     0.1 x lr. Caveat: Jev was trained on body fragments (first sector excluded), so headers are partly
     out-of-domain; also try it frozen.
   * `anchored + noisy student`: **self-training / data generation**: a plain anchored teacher pseudo-labels the
     80k files of the 50 unlabelled READY cases (`Pseudo`, `pseudo_labels`, SSL dataset `PreData`; no labelled
     case is in it, so no leakage); the student trains on labelled + soft pseudo labels (weight 0.5) with
     augmentation. About 2x the training cost per fold.
   **To run:** add the three names to `planner_seeds.VARIANTS` (and to `ABLATIONS` / `CANDIDATES` if one wins), then
   `.venv/Scripts/python -m src.ml.planner_seeds --seeds 13,14,15` (~1-2 h). Compare with `anchored` and logistic
   regression; deploy only by inner-validation AP. **Use `.venv/Scripts/python`**: the system `python` has no
   CUDA and lacks `piexif`. Add a unit test for `augment` (shapes, file name / signature untouched, label kept).
   Other ideas in the same line: generate synthetic labelled cases by planting key-like tokens (names, e-mails,
   URLs) into copies of benign files of unlabelled cases (labels are then exact); mixup between cases; fine-tune
   the MPNet content encoder (now frozen features) with LoRA; more aggressive modality dropout; an ensemble over
   seeds (cheap, reduces the ±0.026 seed variance).
2. Planner: anchoring works; try anchoring on the logistic-regression score itself (learned residual on top of the
   linear model), and count curated tokens in `solve_eval` so mvs_2024 / ngdc_2012 are measured end to end.
3. Commit: all 2026-11-15/16 work is uncommitted on `sprint-2` (ask the user first).
4. Re-download the 12 unrestorable evidence files' images (narcos 1-3, magnet 2020 / 2022, ahadi 02 / 03 / 05,
   dfirmadness) and run `python main.py cases repair`; move `data/` out of OneDrive (`FORENSI_DATA_DIR`).
5. Jev-MM: messaging / screenshot classes are small; add more mobile cases or weight them.

## 0a. 2026-10-24: Jev typed-answer model (deep learning, course requirement)

**Asked by the user:** the project is for a Deep Learning course and must contain a model that is trained, "a Jev
model" (a classifier that reads a context and returns typed answers with probabilities: choice / score / yes-no).
Built in-house (no external API: evidence never leaves the machine, and the course needs our own training):

* `src/ml/fragments.py`: 512-B body fragments (first sector excluded) from every READY case → `data/datasets/fragments/<date>/`
  (140,522 fragments, 44,939 files, per-case splits).
* `src/ml/jev.py`: 0.89 M-param byte CNN, heads choice (17 types) / yes_no (disguised) / score (position); `Jev` query API.
* `src/ml/train.py`: training, temperature calibration, baselines, `REPORT.md`; checkpoint `data/models/jev/jev.pt`.
* `src/agents/jev_agent.py`: leads for headerless files, always UNCORROBORATED (no citations). `--agents jev`.
* Test cases: choice 0.805 acc / 0.677 macro-F1 (histogram baseline 0.650 / 0.480); yes_no AUROC 0.985 on re-drawn
  extensions but **0.78 on real ones (precision ≈ 5 %)**; score MAE 0.227 vs 0.243 constant. README §6.5.
* An early version of the yes_no augmentation let the extension alone reach 0.97 AUROC (shortcut); fixed with a shared
  extension pool. The extension-only baseline stays in the report as the check.
* Next: more mp4 / zip / pdf / elf support in the val/test cases; train on sliding windows for carving; a stronger
  disguise signal (several fragments per file in the head, not their mean).

## 0. Sprint 2.5 (2026-10-08): what changed, results, how to resume

**Asked by the user:** measure the agents against the answer keys, and close the gaps reported after Sprint 2
(no Windows parsers, noisy carving, verifier sharing the agent's bridge / hash cache, `-wal` not covered by
`expected_hash`, evidence inside OneDrive, stale README).

| Gap | Fix | Where |
|---|---|---|
| No Windows artefacts (40 of 72 READY cases had no app findings) | `WindowsArtifactAgent`: SYSTEM / SOFTWARE / SAM (names only) / NTUSER hives, Prefetch, LNK, `$I` + `INFO2` Recycle Bin, IE `index.dat`, EVTX (selected IDs). New locators `registry` and `evtx_record` | `src/agents/windows_agent.py`, `src/bridge/windows_bridge.py` |
| Verifier shared the agent's bridge and hash cache | `BaseAgent` builds its verifier on a **fresh** `ForensicBridge` (`owns_bridge`, `close()`); scoring re-verifies again with a new engine | `base_agent.py`, `heuristic_verifier.py` |
| `-wal` / `-journal` not covered | `Citation.side_hashes`; filled in `BaseAgent.finding()` for SQL citations; verifier rejects on any side-file difference; SQLite working copies keyed on db + side hashes | `models.py`, `forensic_bridge.py` |
| Carving noise | dedupe per database + `carve_score` triage rank (`value.score`); reports rank carved text | `filesystem_agent.py`, `case_report.py` |
| Mail containers invisible | `fs.mail_store` (PST / OST / DBX / mbox) | `filesystem_agent.py` |
| No measurement vs answer keys | `python main.py score`: fresh audit + gold questions + answer-key entity recall | `src/eval/scoring.py`, `src/eval/gold/*.json` |
| Evidence in OneDrive | `FORENSI_DATA_DIR` env var (data was **not** moved; the user should move `data/` and set it) | `settings.py` |
| EVTX slow (python-evtx, quadratic re-verification) | switched to pyevtx-rs (`evtx==0.13.1`) + per-file summary cache. **python-evtx must not be installed alongside (Evtx/evtx clash on Windows)** | `windows_bridge.py`, `requirements.txt` |

**Results (2026-10-08):**
* Corpus (`python main.py corpus`, apps + fs + win, 72 READY cases, ~28 min): **211,153 VERIFIED, 0 REJECTED**,
  0 case errors (`data/outputs/corpus_runs/corpus_apps-fs-win.json`). Carved strings 158,539 → 128,970 after dedupe.
  READY cases without app / system findings: 40 → 27.
* Fresh audit (`python main.py score --cases all`): 211,203 / 211,203 still VERIFIED (`data/outputs/scores/`).
* Gold: NIST Data Leakage **23 / 38 answered (60.5 %)**, mean credit 64.8 % (23 / 30 = 76.7 % within built
  capabilities); NIST Hacking Case **11 / 23 (47.8 %)**. Per question: `data/cases/<case>/workspace/reports/score.md`.
* Answer-key entity recall: 137 / 907 pooled over 19 cases (15.1 %); Data Leakage 37.2 %.
* M57 has **no** gold set: its public key decks have no answers.
* Tests: 176 pass.

**Gold files** are transcribed by hand from the answer-key PDFs (questions, answers, requirement groups;
`cat:<category>|text` and `same_finding` rules, `needs` for capabilities not built, `excluded` with reasons).
Never load them, or `context/answer_key`, into an agent.

**Next, by expected score gain:** (1) read file *content* (text / HTML / INI / logs, OST / PST / DBX bodies):
most remaining Hacking Case answers and NIST Q21 / Q30 / Q31; (2) ShellBags, jump lists, `$UsnJrnl`, thumbcache,
Sticky Notes, Windows.edb / WebCacheV01 (ESE); (3) gold sets for Lone Wolf, Owl, DFRWS; (4) Sprint 3 agents (§9).
Avoid tuning parsers to the two gold cases: add generic artefacts and confirm they keep 0 REJECTED corpus-wide.

**Resume:** `.venv/Scripts/python -m pytest` (176) → `python main.py corpus` → `python main.py score --cases all`.

## 1. State at a glance

| Component | File | Status |
|---|---|---|
| Device config (single / dual GPU) | `src/config/settings.py` | ✅ done |
| **Public corpus fetcher** (multi-source, nested archives, integrity) | `src/utils/dataset_scraper.py` | ✅ executed, **16 sources, 67,118 files, 12.6 GB, 13,077 SQLite DBs, 12,607 images** |
| Forensic bridge (sandbox, read-only SQLite, EXIF/GPS, text, FS walk) | `src/bridge/forensic_bridge.py` | ✅ done |
| Case text index (proposes citations for OCR/VLM text) | `src/bridge/text_index.py` | ✅ done |
| Citation / Finding contracts | `src/verification/models.py` | ✅ done |
| HeuristicVerificationEngine | `src/verification/heuristic_verifier.py` | ✅ self-test 12/12 on real data |
| BaseAgent | `src/agents/base_agent.py` | ✅ done |
| MultimodalVisionAgent | `src/agents/vision_agent.py` | ✅ PoC on real evidence (`data/outputs/android_10/POC_REPORT.md`) |
| CLI | `main.py` | ✅ `cases / investigate / apps / fs / corpus / dataset / finetune / gui / …` |
| Unit tests | `tests/` | ✅ 176 pass (no GPU or dataset needed; real-corpus tests skip when data is absent) |
| **Case corpus** (generic remote evidence) | `src/cases/registry_ext.py`, `src/utils/bundle.py`, `src/utils/remote.py` | ✅ **108 cases registered, 72 READY**, 100 evidence items, 162,793 files, 43 GB (§1c) |
| **AppArtifactAgent** (27 parser specs) | `src/agents/app_artifact_agent.py`, `src/agents/parsers/` | ✅ corpus run: 0 rejected (§1d) |
| **FileSystemAgent** (signatures, hidden, deleted, WAL, SQLite carving) | `src/agents/filesystem_agent.py` | ✅ corpus run: 0 rejected (§1d) |
| **WindowsArtifactAgent** (registry, Prefetch, LNK, EVTX, Recycle Bin, index.dat) | `src/agents/windows_agent.py`, `src/bridge/windows_bridge.py` | ✅ corpus run: 0 rejected (§0) |
| **Scoring vs answer keys** (audit, gold questions, entity recall) | `src/eval/` | ✅ NIST Data Leakage 60.5 %, Hacking Case 47.8 % (§0) |
| TimelineAgent, OrchestratorAuditorAgent | n/a | ⏳ Sprint 3 |

**Sprint 1 priority (set by the user):** gather a large amount of real data, plus a *small* PoC
of the first agent on one or two images with real results. Both are done.

**Repository:** `https://github.com/Void-DanielS/Forensic-Orchestrator` (`main`); Sprint 1.6 + 2 are on branch
`sprint-2`. Evidence, datasets, outputs, `.venv/` and the bundled 7-Zip binaries (`tools/7zip/`, see
`tools/README.md`) are not in git: rebuild the corpus locally with `main.py cases build`. The `.venv/Scripts/*.exe`
launchers are stale (folder renamed) — use `.venv/Scripts/python -m pip`.

**Where things stand / how to resume (2026-10-07):**
1. `python main.py cases list` → READY count (72). A fetch of `mvs_2023` (iPhone FFS, 338 k members) was still
   running when this was written; re-running the build resumes / skips finished items.
2. Retry the pending items in §1c (Google Drive quota resets in about 24 h):
   `python main.py cases build --cases <ids> --groups light,heavy,extended`.
3. `python main.py corpus` re-runs AppArtifactAgent + FileSystemAgent over every READY case (≈ 10 min).
4. Then Sprint 3 (§9).

## 1c. Sprint 1.6 (2026-10-07): corpus expanded to 108 public cases

The user asked for **at least 100 investigations, each with its images and its context**, before agent work
continued. The registry now holds **108 cases** (`src/cases/registry.py` core 18 + `registry_ext.py` 90);
**72 are READY** (`python main.py cases list`: at least one evidence item extracted plus a brief of at least 300
characters or published questions). The user accepted 72 as enough to proceed (2026-10-07); the rest are queued.

| Source | Cases | Notes |
|---|---|---|
| Digital Corpora | Nitroba, M57-Jean, NGDC 2012 (7 devices), Tuck, NPS-2011 x4, Casper-RW, Domex users, Nexus One/S, Hickman Android 6/7/8/11/12 | NGDC uses the final daily snapshot of each device |
| NIST CFReDS | Hacking Case (31 questions), Rhino Hunt, Russian Tea Room, 10 CFTT chip-off/JTAG phones | CFTT data-population docs are the brief |
| DFRWS | 2005, 2008, 2009, 2010, 2011 (two scenarios), 2016, 2017, 2018, 2023 + CIRCL wiped disk | README results/submissions go to the answer key; scenario text to the brief |
| Ali Hadi (archive.org) | 13 challenges | #6 (archive.org 500 on the file) and #13 (needs re-run) pending |
| CTFs | DFIR Madness 001, MVS 2023/2024/2026, MSAB 2026, Cellebrite 2021/2023/2024/2025, NFI Rodeo 2023 | published zip passwords are in the registry, never in manifests or reports |
| Other | TechHive, Bart the hacker, OU Red-Petya server, OU defaced web server, Cado AWS EKS, DSTL ground truth, UTFPR x2, 18 VTO drone models | |

**How evidence is fetched now.** `RemoteEvidence` (`src/utils/bundle.py`) describes any item as remote files/folders:

* `src/utils/remote.py`: one range-readable source for S3/HTTP, Google Drive files and folders, archive.org (`ia:`),
  GitHub (`gh:`, LFS aware), Digital Corpora prefixes (`dc:`), figshare, Zenodo, Dropbox and the Wayback Machine.
  Drive quota / rate-limit pages are detected and waited out; archive.org gets 3 connections (it throttles 12).
* Dispatch by content: EWF/raw images go to TSK (now also **LVM2** via libvslvm, and **XFS / APFS** via dissect:
  `src/utils/dissect_fs.py`); ZIP is range-read (encrypted / Deflate64 members go to 7-Zip); TAR/TGZ is streamed;
  7z/RAR/bz2/xz use the bundled **7-Zip 24.09 in `tools/7zip`** with *selective* extraction; loose pcap, memory,
  mail, logs and video are copied.
* Images TSK cannot parse are kept byte-for-byte when under the cap (`disk_image_unparsed`); archives whose members
  are too large unpacked (sparse RAM dumps) are kept compressed (`compressed_evidence`). Everything else over a cap
  is listed in `MANIFEST.json -> not_fetched` with its URL and size.
* Documents shipped with evidence are routed to `context/brief`, `answer_key` (solution, write-up, results) or
  `acquisition` (FTK logs, hashes). `publisher_notes` holds scenario text only; plot-revealing text goes to
  `answer_notes` (saved under `answer_key`).
* `case.json -> readiness`, and catalogue columns `questions, brief_chars, context_level, ready`.

**Pending** (re-run with `python main.py cases build --cases <ids> --groups light,heavy,extended`):

* Google Drive quota / rate limit (retry after about 24 h): `cellebrite_ctf_2021`, `cellebrite_ctf_2023`, the
  remaining `vto_*` drones, `utfpr_server_hacked` (XFS now supported), `techhive`, `cellebrite_ctf_2024/2025`.
* Fixed after their first attempt, need a re-run: `hickman_android_6` (Deflate64 zip), `cftt_lg_optimus_hub`
  (raw cap), `ou_defaced_webserver` (root FS on LVM), `ahadi_13_unallocated` (kept compressed), `mvs_2024` (stream
  stopped at a document save; delete `evidence/mvs2024_pixel3` first to fetch it fully).
* Very large, not finished: `tuck_2019` (49 GB, APFS now supported), `hickman_android_12` (42 GB), `mvs_2026`,
  `msab_ctf_2026`; NPS-2011 x4, the remaining CFTT phones and `mvs_2023` were still running in build A.
* Dropped: BelkaCTF (archive passwords no longer public), Magnet 2018 (download files are 1.8 KB stubs), Hickman
  MediaFire mirrors (private now), DFRWS 2006/2007 (pure carving benchmarks, no investigation).

**Warning: evidence inside OneDrive.** Two extracted `desktop.ini` files (m57_terry, ahadi_01) disappeared from
disk after a clean inspection; OneDrive / Explorer treat `desktop.ini` specially. Workspace writes also hit
transient `WinError 5` locks (now retried, `src/utils/fsio.py`). **Move `data/cases` out of the OneDrive folder**
(or exclude it from sync) before relying on the hashes in court-style reports.

## 1d. Sprint 2 (2026-10-07): AppArtifactAgent + FileSystemAgent

Both agents are deterministic (no model, role `text`), subclass `BaseAgent`, and emit findings whose citations the
HeuristicVerificationEngine re-executes. **Corpus run over all 72 READY cases (`python main.py corpus`):
199,093 findings VERIFIED, 0 REJECTED, 0 case failures** (summary: `data/outputs/corpus_runs/corpus_apps-fs.json`;
per case: `workspace/runs/<run>/` and `workspace/reports/<run>_agents.md`).

**AppArtifactAgent.** `src/agents/parsers/` holds declarative `TableSpec`s (27): candidate column names per field
(schemas drift across versions), an epoch per time field (`cocoa_auto` / `unix_auto` detect the unit), and lookups
into related tables (WhatsApp chat to JID, iMessage handle, Safari visit to URL, contact mimetype). Each row becomes
one finding; each quoted cell is a citation `SELECT "<col>" FROM "<table>" WHERE rowid = ?`. BLOB cells are never
quoted. Families: Android SMS (mmssms, Google Messages bugle_db), contacts, call log, calendar, accounts; iOS
Messages, AddressBook, CallHistory, KnowledgeC, Notes, Calendar, Photos (GPS); Chromium URLs / searches / downloads,
Firefox (desktop and Fenix), Safari; WhatsApp (Android old/new schema, iOS), Skype, Facebook Messenger, LINE, Kik,
TikTok. Validation: all 16 distinct SMS documented in the Android 10 `SMS-Messages.xlsx` are recovered and VERIFIED
(text and UTC second): `tests/test_sprint2_agents.py::test_android10_documented_sms_are_recovered_and_verified`.

**FileSystemAgent.** Per extracted file: `fs.magic_mismatch`, `fs.sqlite_unlisted`, `fs.orphan_wal`,
`fs.hidden_path`, `fs.deleted_recovered`, and **SQLite carving** (`sqlite.carved_text`: freelist trunk and leaf
pages, freeblocks, unallocated cell space), each cited by `byte_offset` + hex. Credential stores (Login Data, Web
Data, keychains, key4.db, accounts.db, cookies, anything named *password*) are never carved (FDX B8); leftover
schema text is dropped.

| Category | Corpus count | | Category | Corpus count |
|---|---|---|---|---|
| sqlite.carved_text | 158,539 | | chrome.url | 3,596 |
| knowledgec.event | 10,339 | | fs.sqlite_unlisted | 3,252 |
| fs.deleted_recovered | 7,373 | | sms.message | 3,098 |
| fs.orphan_wal | 2,701 | | firefox.url | 2,388 |
| fs.hidden_path | 1,655 | | fs.magic_mismatch | 1,426 |
| call.log | 1,335 | | calendar.event | 829 |
| chrome.search | 513 | | imessage.message | 351 |

Other Sprint 2 pieces: binary plists in `CaseTextIndex` (strings cited by an escaped-literal `regex`, which the
verifier re-runs over the decoded file); `investigate_agents()` and `write_agents_report()`; CLI `apps`, `fs`,
`corpus`, `investigate --agents apps,fs`; trace event `parser`; memory saves throttled (at most one per 5 s);
FileCategory enum and dictionary §6.2 / §6.4 updated; `docs/schemas/` regenerated.

**Known gaps (next).** Carving is noisy (158 k strings: rank and deduplicate before showing analysts); Windows
browsers in ESE (`WebCacheV01.dat`) and registry / EVTX / LNK / Prefetch parsing are not done (40 of 72 cases have
no app-database findings, mostly Windows images); Telegram / Signal / Snapchat bodies are BLOBs (not parsed); the
verifier still shares the agent's bridge and hash cache; `-wal` side files are not covered by `expected_hash`.

## 1b. Sprint 1.5 (2026-10-06): case database, computer images, context, GUI

**Everything is now organised by case.** `data/raw_images/` was migrated (same-volume rename, re-inspected: 0 hash
mismatches) into `data/cases/<case_id>/evidence/<evidence_id>/`. A case is one investigation with all of its
evidence items and its documents:

```
data/cases/
├── catalog.sqlite                 cases / evidence / files / docs tables  (python main.py cases catalog)
└── <case_id>/
    ├── case.json                  registry entry + fetch status
    ├── context/brief/             what the examiner is told; agents read this (+ <doc>.txt text)
    ├── context/answer_key/        solutions / instructor material; NEVER given to agents (scoring only)
    ├── context/acquisition/       imaging logs (FTK Imager, EWF headers)
    ├── evidence/<evidence_id>/    fs/ (sandbox), MANIFEST.json, INSPECTION.json, FILELIST.jsonl.gz (disk images)
    └── workspace/                 memory.json, runs/<run_id>/{run.json,trace.jsonl,*_findings.json}, reports/*.md
```

| Piece | File | Notes |
|---|---|---|
| Case registry (18 cases, 40 evidence items) | `src/cases/registry.py` | grouping comes from the sources themselves (scenario / CTF year / challenge) |
| Builder + catalogue | `src/cases/builder.py` | `python main.py cases list | build | catalog` |
| Document text (PDF/DOCX/XLSX/HTML) | `src/cases/docs.py` | |
| **Computer / removable images** | `src/utils/disk_image.py` | E01 (libewf) or split raw → The Sleuth Kit; NTFS/FAT/ext; parallel spool (≈30 MB/s) or range-read > 45 GB |
| Investigation context + memory | `src/context/case_context.py` | entities from the brief; deterministic finding↔context links |
| Run tracing | `src/agents/trace.py` | every agent step as JSONL |
| Case runner | `src/pipeline/investigate.py` | `python main.py investigate --case <id> [--items a,b] [--limit N]` |
| Report | `src/reports/case_report.py` | evidence / context contrast / rejected / observations / hand-off |
| **GUI** | `src/gui/server.py`, `src/gui/static/index.html` | `python main.py gui` → http://127.0.0.1:8765 |
| ML dataset export | `src/datasets/export.py` | `python main.py dataset` → `data/datasets/<date>/` (+ DATASHEET.md) |

### Cases

| Case | Evidence (computer ● / mobile ○) | Brief | Answer key |
|---|---|---|---|
| `m57_patents` | ● Charlie, Jo, Pat, Terry workstations (XP); ● 4 USB drives | affidavit + 4 detective reports | 3 scenario decks |
| `lone_wolf` | ● suspect laptop (Win10, 9×E01) | README | forensic reports bundle |
| `narcos` | ● 3 PCs (split raw, heavy) + ● post-CCleaner PC | — | — |
| `nist_data_leakage` | ● PC (Win7), ● USB #1, ● USB #2, ● CD-R | NIST case page | `leakage-answers.pdf` |
| `owl` | ● HD1 desktop (68 GB, range-read), ○ Nexus 5 quick image | — | 4 creation logs |
| `dfrws2021` | ● skimmer microSD, ● Raspberry Pi microSD (heavy), ○ Galaxy S10 | README + challenge details | — |
| `magnet_ctf_2019..2022` | ● Windows (2019 range-read; 2020/2022 heavy) + ○ Android / iOS / Takeout | README | — |
| `hickman_*` (8) | ○ one reference phone each | image-creation PDF | (the PDF itself; no blind key) |

**Fetched (2026-10-06): 39 of 40 evidence items, 102,343 extracted files (20.9 GB), 1,409,882 file-system entries with
MACB times, 2,797 recovered deleted files, 0 hash mismatches.** By platform: Windows 14 items / 24,912 files,
removable 7 / 3,668, Linux 1 / 6,464, Android 9 / 39,923, iOS 7 / 27,194, Takeout 1 / 182. Owl HD1 (68 GB) was
range-read (31.9 GB transferred, 79 min); the rest were spooled in parallel. Logs: `data/outputs/build_logs/*.log`.
`python main.py dataset` → 18,919 images, 42,302 documents, 898 labelled agent findings; splits 10/4/4 cases.

### How the vision agent uses the context

1. `CaseContext` reads only `case.summary` + `context/brief/*.txt` and extracts persons (given-name rule / honorifics),
   e-mails, phones, URLs/domains, file names, devices, GPS, dates (→ timeframe). Each entity keeps doc + line.
2. For every artifact the agent emits EXIF → OCR boxes → index lookups → VLM → findings → verification.
3. After verification, `contrast()` links each finding to the context: `corroborates`, `keyword_match`,
   `outside_timeframe`, `not_in_context`, `new_lead`, `unverified_mention`. Links are saved in `workspace/memory.json`
   after every artifact and appear live in the GUI. They never change a finding's evidentiary status.
4. The report (`workspace/reports/<run>_vision.md`) separates verified evidence (with citations), the context
   contrast, rejected claims, and model observations, and lists the JSON inputs for the consolidation agent.

OCR/VLM text is now corroborated against **SQLite cells and lines of plain-text artefacts** (`.txt/.csv/.eml/.html/
.xml/.json/.log`, cited with `line_number`), which matters on Windows images where SQLite is rare.

### Fixes from the Sprint 1 audit included here
* Windows extended-length paths (`\\?\`) in the sandbox and inspection: evidence paths > 260 chars work without the
  LongPathsEnabled registry switch.
* `TextBridge.read_text` decodes UTF-16 only with a BOM (Latin-1 logs were garbled).
* SQL identifiers are quoted (`"` doubled) in the text index and `tables()`.
* Deleted-file recovery writes flat `fs/_deleted/<inode>_<name>` (names collided with deleted directories).
* Still open: verifier shares the agent's bridge/hash cache; `expected_hash` does not cover `-wal`; `verify-selftest`
  is android_10-specific; EXIF byte-offset citation points at the first occurrence.

### Known gaps
* NIST CD-R (`nist_rm3`): pure UDF 2.x (NSR03, no ISO-9660 bridge); TSK cannot read it → group `future`, needs a UDF reader.
* macOS (Tuck, Magnet 2021 MacBook, APFS), Magnet 2022 Linux (.7z), QNAP NAS (RAID) and memory images are not
  registered/fetched yet.
* Document text inside DOCX/PDF evidence is not yet indexed for corroboration (plain text only).

## 2. Environment (as tested)

* **Hardware found:** NVIDIA **GeForce RTX 5080 Laptop GPU, 16 GB** (the spec assumed an RTX 5070 with 12 GB).
  The 12 GB budget is enforced in software (`BaseAgent.enforce_vram_budget`), so the code stays valid for a
  real 12 GB card. The profile label in `settings.py` still says "RTX 5070".
* Windows 11, Python 3.12.10, venv `forensic_agent/.venv`. Pins: torch 2.11.0+cu128, transformers 5.18.0,
  bitsandbytes 0.50.2, easyocr 1.7.2, pillow-heif 1.8.0 (full list in `requirements.txt`).
* Models are cached: `Qwen/Qwen2.5-VL-3B-Instruct` (HF cache) and EasyOCR `en`+`es`.
* ⚠️ Hub calls fail with `SSL: CERTIFICATE_VERIFY_FAILED (self-signed certificate in chain)`, probably
  antivirus or proxy TLS inspection. Models load from cache. Use `HF_HUB_OFFLINE=1`.
* ⚠️ Windows `LongPathsEnabled = 0`. Paths over 240 characters are stored under `fs/_longpaths/<sha1>_<name>`,
  with the original path in `MANIFEST.json` (`original_path`). Enabling long paths (admin registry change)
  would let the original layout be preserved. The same goes for invalid characters and case collisions.
* ⚠️ PowerShell `*>` redirection writes **UTF-16** logs (`fetch_light.log`, `fetch_heavy.log`, `fetch_extended.log`).
  Read them with `iconv -f UTF-16LE -t UTF-8`. Later logs (`fetch_extended2.log`) are UTF-8.

```bash
cd "C:/Forensic - Orchestrator/forensic_agent"
.venv/Scripts/python main.py config
.venv/Scripts/python -m pytest
.venv/Scripts/python -m src.utils.dataset_scraper --list-sources
HF_HUB_OFFLINE=1 .venv/Scripts/python main.py --source android_10 vision --limit 5
```

## 3. Data corpus (verified)

All sources are public research images hosted by **Digital Corpora** (`https://digitalcorpora.s3.amazonaws.com/corpora/`):
Joshua Hickman reference images (documented device population), Magnet Forensics CTF images
and the DFRWS 2021 challenge. The fetcher never downloads whole images unless it has to:

| Archive shape | Strategy |
|---|---|
| ZIP | HTTP range requests on the central directory; only selected members, each CRC-32 checked |
| ZIP nested in a ZIP, STORED | addressed as a byte sub-range of the parent, so still range-read |
| ZIP nested in a ZIP, DEFLATED | inflated on the fly and spooled to `data/_spool/` (deleted afterwards) |
| TAR / TAR.GZ (also a TAR nested in a ZIP or TAR) | streamed with parallel read-ahead; only selected members are written |
| iTunes backup | `Manifest.db` maps hashed file IDs back to `Domain/relativePath` (all three public ones are **encrypted**) |

**Selection:** SQLite databases (+ WAL/SHM/journal, plus extensionless Chromium DBs such as `History` and `Cookies`, plus
SQLite sniffed by magic in TAR streams); user photos and screenshots; app-private images ≥ 8 KB (chat media
and caches; icons are skipped); app preferences (`shared_prefs`, iOS `Library/Preferences`); system XML/JSON/logs.
The per-source budget is 1.5 GB and 1,500 images. **Credential files** (`*_passwords.txt`, keychains, memory
dumps) are never copied.

Layout per source:
```
data/cases/<case>/evidence/<source_id>/   (was data/raw_images/<source_id>/ in Sprint 1)
├── _case_context/   documentation (image-creation PDFs + extracted .txt, scenario READMEs). NOT evidence, outside the sandbox
├── fs/              SANDBOX ROOT for agents (logical file tree, original member timestamps restored)
├── MANIFEST.json    every file: path, source member, category, size, zip CRC / member mtime, SHA-256, original_path
└── INSPECTION.json  integrity report (SQLite quick_check on working copies, Image.verify, re-hash)
data/raw_images/CORPUS_INDEX.json   per-source + total statistics
```

### Corpus (after light + heavy + extended phases)

| Source | Platform | Files | MB | SQLite DBs | Images | GPS | Screenshots | Case context |
|---|---|---|---|---|---|---|---|---|
| `android_10` | android | 4,197 | 300 | 636 | 1,212 | 3 | 7 | Android10-ImageCreation.pdf, SMS-Messages.xlsx |
| `dfrws2021_galaxy_s10` | android | 5,175 | 497 | 852 | 334 | 17 | 22 | README.md |
| `magnet2019_android` | android | 136 | 53 | 22 | 40 | 1 | 0 | README.md |
| `magnet2021_ios` | ios | 2,653 | 579 | 641 | 393 | 0 | 0 | 518e8d766f9b3e76db216f35fdb6b0604e50f61b.pdf, README.md |
| `magnet2020_takeout` | takeout | 182 | 303 | 0 | 53 | 50 | 0 | README.md |
| `ios13_4_1_ffs` | ios | 4,697 | 919 | 829 | 1,494 | 5 | 22 | iOS13-4-1-ImageCreation.pdf |
| `android_9` | android | 3,825 | 306 | 571 | 936 | 4 | 0 | Android9-ImageCreation.pdf |
| `magnet2020_android` | android | 3,067 | 409 | 489 | 656 | 48 | 10 | README.md |
| `magnet2022_android` | android | 3,693 | 298 | 560 | 1,200 | 4 | 1 | README.md |
| `magnet2022_ios` | ios | 2,956 | 570 | 654 | 592 | 0 | 1 | README.md, fb028ddefa8af7df5b12d3e729f075d150637a31.pdf |
| `android_13` | android | 12,105 | 1,492 | 2,463 | 1,500 | 0 | 57 | android_13_imagecreation_solution.pdf |
| `android_14` | android | 7,544 | 1,389 | 1,322 | 1,481 | 85 | 51 | Android14-ImageCreation.pdf |
| `ios16` | ios | 3,615 | 1,514 | 1,147 | 239 | 0 | 13 | iOS16-ImageCreation.pdf |
| `ios17` | ios | 4,269 | 1,519 | 1,340 | 50 | 0 | 21 | iOS17-ImageCreation.pdf |
| `ios13_3_1_ffs` | ios | 5,413 | 925 | 847 | 1,498 | 5 | 20 | iOS13-3-1-ImageCreation.pdf |
| `magnet2020_ios` | ios | 3,591 | 1,500 | 704 | 929 | 47 | 4 | README.md |
| **Total (16 sources)** | | **67,118** | **12,572** | **13,077** (87,469 tables) | **12,607** | **269** | **229** | |

Every source passed inspection with **0 hash mismatches**: no file changed between acquisition and inspection.
The few SQLite files that fail `quick_check` (for example LINE `as_dic_2` and 5 files in `magnet2022_ios`) are
corrupt **in the source image**, since the archive CRCs matched.

### Registered but not fetched (`--list-sources`, group `future`)
* `android_7`, `android_8`: **raw physical dumps** (`blk0_mmcblk0.bin`, 31 GB, LG Nexus 5X). These need
  partition and ext4 parsing (`pytsk3`/`dfvfs`), which is FileSystemAgent work in Sprint 2. Their image-creation PDFs are
  still useful context.
* `ios15_itunes`, `ios13_4_1_itunes`, `ios13_3_1_itunes`: **encrypted** iTunes backups. Decrypting them needs the
  backup password published in the image documentation, plus a decryption library. That is a deliberate future step.
* Not modelled at all: `android_6` (raw ext4 `data.img`), `android_11` and `android_12` (11–16 GB deflated nested zips
  holding Cellebrite/TAR exports), and the 2011 NexusOne/NexusS YAFFS2 dumps.

### Evidence-handling rules learned in Sprint 1
1. **Never open an evidence SQLite file in place, even with `mode=ro`.** SQLite rewrote 50 `-shm` files during
   the first inspection. The post-inspection re-hash caught it, and the files were restored from the archive by CRC.
   All access goes through `forensic_bridge.working_copy()`. Sprint 2 agents **must** use `bridge.sqlite`.
2. ⚠️ Several `_case_context` documents (image-creation PDFs) list the **published test-account passwords**.
   They are reference material only, and reports must never echo credentials.

## 4. Verification design (what "zero hallucination" means here)

`Citation = {file_path, locator_type ∈ byte_offset|line_number|sql_query|regex|exif_tag, expected_hash (SHA-256), raw_text_snippet, match_mode}`

For each citation the engine checks that the path is inside the sandbox, that the SHA-256 equals `expected_hash`,
and that re-running the locator on the physical file reproduces the snippet. Statuses:
`VERIFIED` (the only evidentiary status), `REJECTED` (a hallucination or tampering signal) and `UNCORROBORATED`
(a model observation: kept and labelled, never evidence).

OCR and VLM text is verified through `corroboration()`. The cited DB value must explain ≥ 85 % of the
model text, and short text (< 25 normalised characters) must also make up ≥ 50 % of the cell. Without that rule,
coincidental matches such as `RADIO SHACK` inside a package-list blob, or the VLM placeholder `not visible text`
inside a Skype JSON blob, were "verified". Both are now regression tests.

## 5. PoC results (MultimodalVisionAgent on real evidence)

Full write-up: **`data/outputs/android_10/POC_REPORT.md`**. Two images from the Pixel 3 took 30 s and 5.2 GB VRAM.
Result: **9 VERIFIED, 0 REJECTED, 39 UNCORROBORATED**.
* Camera photo `IMG_20200209_132353.jpg`: Pixel 3, 2020-02-09 13:23:53 (UTC−5), **GPS 35.660981, −78.879897**.
  Every value was re-checked against the EXIF tags and the file hash.
* Screenshot `Screenshot_20200213-213310.png`: OCR read **"Super Secret Message"**. The verifier re-ran
  `SELECT "title" FROM "favorites" WHERE rowid = 42` on the Pixel launcher DB, which returned `Super Secret Message Apps`.
  This ties the screenshot to a user-created launcher folder.

An earlier full run on all 58 Pixel 3 images gave 40 VERIFIED, 0 REJECTED and 244 UNCORROBORATED in 341 s, with peak
VRAM of 6.5 GB allocated and 10.1 GB reserved.

## 6. Switching to dual GPU (RTX 4060 as `cuda:1`)

1. Install the card and check that `python main.py config` lists `cuda:0` and `cuda:1`. Use `CUDA_DEVICE_ORDER=PCI_BUS_ID` if the order is wrong.
2. Set `SINGLE_GPU_MODE = False` in `src/config/settings.py`, or set the env var `FORENSI_SINGLE_GPU=0`.
3. `DUAL_GPU_ROLE_MAP` sends `vision` and `ocr` to `cuda:1` (8 GB budget), and `text`, `embeddings` and `orchestrator` to `cuda:0` (12 GB).
   Budgets can be changed with `FORENSI_GPU0_BUDGET_GB` / `FORENSI_GPU1_BUDGET_GB`.
4. No agent code changes are needed: agents call `settings.device_for(self.role)`. If `cuda:1` is missing, they warn and fall back to `cuda:0`.
5. On 8 GB, keep the VLM in 4-bit. `FORENSI_VLM_MODEL=Qwen/Qwen2.5-VL-7B-Instruct` in 4-bit (≈ 6 GB) fits.

## 7. Known limitations / backlog

* Vision outputs: `main.py investigate` writes to the case workspace; the legacy `main.py vision` still writes `data/outputs/<source_id>/`.
* Short OCR labels (app names) can't corroborate under the global 10-character minimum. Idea: field-specific lookups.
* The text index covers SQLite text only; XML/JSON/plist/protobuf and UTF-8 BLOBs are not indexed yet.
* EXIF is read with piexif (JPEG). HEIC opens through pillow-heif, but its EXIF is not yet extracted for findings.
* OCR on tall screenshots downscales to 2048 px. Tiling would improve recall.

---

## 8. Sprint 2: done (see §1d)

The original Sprint 2 prompt is fulfilled: parsers and AppArtifactAgent, FileSystemAgent with carving, XML / JSON /
plist indexing, `apps` / `fs` subcommands, per-agent findings, traces and context links, synthetic per-parser tests
and a real-corpus integration test, 0 REJECTED on real data. Raw physical dumps (android_7/8, CFTT chip-offs) are
fetched through the generic bundle (TSK / LVM / dissect) rather than a separate dfvfs path.

## 9. Copy-paste prompt: Sprint 3 (TimelineAgent + OrchestratorAuditorAgent)

```
You are continuing ForensiAgent-Multimodal in C:\Users\danie\OneDrive\Forensic - Orchestrator\forensic_agent.
Read HANDOVER.md (§1c corpus and §1d Sprint 2 first), README.md, docs/DATA_INTERCHANGE_STANDARD.md and all of src/.
Run `python -m pytest` (176 pass) and `python main.py score --cases gold` to confirm the baseline (Data Leakage
23 / 38, Hacking Case 11 / 23).
First, finish the pending fetches listed in §1c (Drive quota permitting) so more cases are READY.

1. src/agents/timeline_agent.py: TimelineAgent(role="text")
   - Input: VERIFIED findings only (data/cases/<case>/workspace/runs/*/*_findings.json: vision, apps, fs) plus
     FILELIST.jsonl.gz MACB entries and MANIFEST macb. Use the ForensicTimestamp already in each finding's
     value.timestamps (raw + epoch + utc); never re-guess time zones. EXIF without an offset stays local time.
   - Unified UTC timeline per case (all evidence items). Cross-source correlation (photo EXIF GPS/time vs
     messages, calls, browser visits, KnowledgeC app usage within a window). A correlation Finding cites the
     union of the underlying citations, so it verifies only if all of them do.
   - Validate against the timelines in the image-creation documents (context/brief/*.txt; Hickman uses EST).
   - Optional: a small local LLM (4-bit, cuda:0) summarises clusters; its output is UNCORROBORATED unless every
     sentence maps to cited findings.
2. src/agents/orchestrator_auditor.py: OrchestratorAuditorAgent(role="orchestrator")
   - Plans and dispatches agents per evidence item (vision for images, apps for databases, fs for everything),
     re-verifies every finding at the end with a FRESH ForensicBridge + HeuristicVerificationEngine (no shared
     hash cache), hashes -wal/-journal side files too, and drops anything not VERIFIED from the evidence sections.
   - Rank sqlite.carved_text (deduplicate; prefer runs near deleted rows or matching brief entities) before reporting.
   - Score with src/eval/scoring.py (gold questions + entity recall); never give the answer key to agents.
   - data/cases/<case>/workspace/reports/case_report.md: case summary, device info (verified only), timeline,
     per-agent evidence tables with citations, a "Model observations (not evidence)" appendix, verification stats.
     Every evidence sentence references a finding_id.
3. Raise the answer-key score without tuning to the gold cases (see §0 "Next"): file-content findings
   (text / HTML / mail stores), ShellBags, jump lists, $UsnJrnl, ESE (WebCacheV01.dat, Windows.edb). Re-run
   `python main.py corpus` and `python main.py score --cases all` after each; REJECTED must stay 0.
4. Dual-GPU readiness check (FORENSI_SINGLE_GPU=0 on one GPU gives the fallback warning). Tests + update HANDOVER.md.
Constraints: single GPU cuda:0, 12 GB budget. Never write into data/cases/*/evidence/*/fs. Never load
context/answer_key into an agent. Never print credentials or archive passwords in reports.
```
