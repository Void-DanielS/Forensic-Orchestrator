# Forensic Orchestrator — ForensiAgent-Multimodal

A multi-agent framework for automated analysis of digital forensic evidence: Android / iOS extractions and
Windows / Linux / removable disk images. Its core rule is that **no finding is reported unless a deterministic
engine re-verifies it against the physical evidence**. Every statement an agent makes carries citations (file +
SHA-256 + an exact locator). The `HeuristicVerificationEngine` re-executes each citation on the file before the
statement can count as evidence.

This document reports the state of the project as of **2026-11-15** (multimodal model, Jev retrain and evidence repair; base results from 2026-10-08, branch `sprint-2`). It covers what is built,
how it works, how it is evaluated, the measured results and the known limitations.

---

## Contents

1. [Summary of results](#1-summary-of-results)
2. [Status by component](#2-status-by-component)
3. [Architecture](#3-architecture)
4. [The zero-hallucination contract](#4-the-zero-hallucination-contract)
5. [Case corpus](#5-case-corpus)
6. [Agents](#6-agents)
7. [Evaluation methodology](#7-evaluation-methodology)
8. [Results in detail](#8-results-in-detail)
9. [Limitations and known gaps](#9-limitations-and-known-gaps)
10. [Installation and usage](#10-installation-and-usage)
11. [Configuration and hardware](#11-configuration-and-hardware)
12. [Repository layout](#12-repository-layout)
13. [Tests](#13-tests)
14. [Roadmap](#14-roadmap)
15. [Data handling and ethics](#15-data-handling-and-ethics)

---

## 1. Summary of results

| Measure | Result |
|---|---|
| Public cases registered / ready for analysis | **108 / 72** (100 evidence items, 162,793 extracted files, ~43 GB) |
| Verified findings over the 72 ready cases (3 deterministic agents) | **211,153 VERIFIED, 0 REJECTED** |
| Fresh re-verification audit (new engine, new bridge, no shared cache) | **211,203 / 211,203 still VERIFIED, 0 failed** |
| NIST CFReDS *Data Leakage* (38 scored answer-key questions) | **23 answered (60.5 %), mean credit 64.8 %**; 76.7 % answered (81.9 % credit) on the 30 questions not needing an unbuilt capability |
| NIST CFReDS *Hacking Case* (23 scored answer-key questions) | **11 answered (47.8 %)** |
| Answer-key entity recall (19 cases whose keys contain identifiers) | 137 / 907 identifiers found (15.1 % pooled); 37.2 % on Data Leakage |
| Vision agent proof of concept (Pixel 3, 58 images) | 40 VERIFIED, 0 REJECTED, 244 model observations (kept, never evidence) |
| **Jev-MM multimodal model** (vision + metadata + OCR text, trained here; held-out test cases, 3 seeds) | image origin (7 classes) **0.744 ± 0.015 accuracy / 0.533 ± 0.015 macro-F1**, vs. 0.662 vision-only, 0.655 metadata-only, 0.502 text-only, 0.450 majority; user-generated AUROC 0.90 |
| **Investigation planner** ("where to look", multimodal network trained on 16 labelled cases; metadata-anchored fusion; encoders pretrained self-supervised on 50 unlabelled cases; leave-one-case-out, 3 seeds) | deployed anchored fusion: top-50 files reach **63 % of an unseen case's answer items** (random 10 %, category prior 61 %), AP 0.279 vs 0.194 for plain fusion and 0.240 for the metadata-only network; still just below logistic regression on metadata (AP 0.301); end to end, reading ≤ 100 files per case recovers **138 / 869** key identifiers vs 137 for the deterministic agents reading every file (§6.7) |
| Jev typed-answer model (byte CNN, trained here; held-out test cases) | file-type choice **0.811 accuracy / 0.658 macro-F1** over 17 types (byte-histogram baseline 0.630 / 0.470); disguise yes/no AUROC 0.76 on real extensions |
| Evidence integrity (full SHA-256 audit of 162,793 files, 2026-11-15) | 51 files damaged by the host (29 `desktop.ini` rewritten by Windows, 17 SQLite `-shm` rewritten, 5 missing); **39 re-acquired** from the original images with SHA-256 match, 22,395 at-risk files made read-only, 12 need a full re-download (§9) |
| Unit and integration tests | 212 pass (no GPU or dataset required; corpus tests skip when data is absent) |

How to read these numbers:

* **0 REJECTED measures faithfulness.** No agent made a statement the evidence does not support.
* **The answer-key score measures usefulness:** how much of what investigators were expected to find the agents
  actually found. It is the honest measure of progress, and it is far from complete (see §8 and §9).

---

## 2. Status by component

| Component | File(s) | Status |
|---|---|---|
| Device / GPU configuration (single GPU now, dual GPU ready) | `src/config/settings.py` | done |
| Case registry (108 cases), builder, catalogue | `src/cases/` | done |
| Remote evidence fetcher (S3/HTTP, Google Drive, archive.org, GitHub LFS, Zenodo, figshare, Dropbox, Wayback) | `src/utils/remote.py`, `src/utils/bundle.py` | done |
| Disk images: EWF / raw → The Sleuth Kit; LVM2; XFS and APFS via dissect; 7-Zip for 7z / RAR / Deflate64 | `src/utils/disk_image.py`, `src/utils/dissect_fs.py` | done |
| Mobile corpus fetcher (range-read ZIP, streamed TAR, nested archives) | `src/utils/dataset_scraper.py` | done |
| Forensic bridge: sandbox, read-only SQLite (WAL-aware), EXIF / GPS, text, FS walk | `src/bridge/forensic_bridge.py` | done |
| Windows bridge: registry hives, EVTX, binary layouts (Prefetch, LNK, Recycle Bin, `index.dat`) | `src/bridge/windows_bridge.py` | done |
| Case text index (proposes citations for OCR / VLM text) | `src/bridge/text_index.py` | done |
| Citation / Finding contracts; FDX interchange schemas | `src/verification/models.py`, `src/schemas/` | done |
| `HeuristicVerificationEngine` (7 locator types, side-file hashing, own bridge) | `src/verification/heuristic_verifier.py` | done |
| `MultimodalVisionAgent` (EXIF + EasyOCR + Qwen2.5-VL 4-bit) | `src/agents/vision_agent.py` | done (PoC scale) |
| `AppArtifactAgent` (27 declarative SQLite parsers) | `src/agents/app_artifact_agent.py`, `src/agents/parsers/` | done |
| `FileSystemAgent` (signatures, hidden / deleted files, mail stores, ranked SQLite carving) | `src/agents/filesystem_agent.py` | done |
| `WindowsArtifactAgent` (registry, Prefetch, LNK, EVTX, Recycle Bin, IE history) | `src/agents/windows_agent.py` | done |
| Scoring against answer keys (audit + gold questions + entity recall) | `src/eval/` | done |
| Case context (brief entities), investigation memory, run traces | `src/context/`, `src/agents/trace.py` | done |
| Markdown reports, local web console (GUI), ML dataset export | `src/reports/`, `src/gui/`, `src/datasets/` | done |
| Jev typed-answer model (fragment dataset, training, calibration, query API) | `src/ml/` | done |
| `JevAgent` (type / disguise leads for headerless files) | `src/agents/jev_agent.py` | done |
| **Jev-MM multimodal model** (image + metadata + OCR text, fusion transformer, ablations, calibration) | `src/ml/mm_dataset.py`, `src/ml/multimodal.py`, `src/ml/mm_train.py` | done |
| `ImageTriageAgent` (image origin / user-generated leads, origin-mismatch leads) | `src/agents/mm_agent.py` | done |
| **Investigation planner** (dataset from answer keys, 5-modality network, LOCO evaluation, model selection) | `src/ml/planner_dataset.py`, `src/ml/planner.py`, `src/ml/planner_train.py` | done |
| `ReaderAgent` (files chosen by the planner: identifiers cited by byte offset) | `src/agents/reader_agent.py` | done |
| **Deep-learning pipeline** `python main.py solve` (planner -> image / reader agents -> verifier -> report) | `src/pipeline/solve.py`, `src/eval/solve_eval.py` | done |
| Evidence integrity audit / repair / write protection | `src/utils/integrity.py` (`main.py cases audit|repair`) | done |
| `TimelineAgent` (cross-source UTC timeline) | — | planned, Sprint 3 |
| `OrchestratorAuditorAgent` (planning, final audit, case report) | — | planned, Sprint 3 |

---

## 3. Architecture

```
                         ┌────────────────────────────── data/cases/<case_id>/ ─────────────────────────────┐
 public sources ──fetch──▶  evidence/<item>/fs/  (read-only sandbox)   context/brief/   context/answer_key/   │
 (NIST, Digital Corpora,  │  MANIFEST.json (hash, MACB per file)      (given to agents) (scoring ONLY)        │
  DFRWS, CTFs, ...)       └──────────┬────────────────────────────────────────┬───────────────┬─────────────┘
                                     │                                        │               │
                          ForensicBridge (+ WindowsBridge)               CaseContext          │
                          sandbox · SQLite working copies · EXIF ·       (brief entities)     │
                          text · registry · EVTX · byte layouts                │               │
                                     │                                        │               │
            ┌────────────────────────┼─────────────────────────┐              │               │
            ▼                        ▼                         ▼              │               │
   MultimodalVisionAgent     AppArtifactAgent          FileSystemAgent        │               │
   WindowsArtifactAgent      (each: analyze → Finding + Citations)            │               │
            │                                                                 │               │
            ▼                                                                 ▼               │
   HeuristicVerificationEngine (own bridge) ──▶ VERIFIED / REJECTED / UNCORROBORATED          │
            │                                   + context links (corroborates, new_lead, ...) │
            ▼                                                                                 ▼
   workspace/runs/<run>/*_findings.json, trace.jsonl, reports/*.md  ──▶  eval: fresh audit + score vs answer key
```

Design principles:

* **Agents propose, the verifier decides.** Agents never set a finding's status. Only the engine promotes a
  finding to VERIFIED, and it uses no model.
* **Evidence is never written.** SQLite is queried on hash-keyed working copies (an evidence database opened even
  `mode=ro` rewrites its `-shm`). Hives, logs and binaries are read as bytes. The sandbox refuses any path that
  escapes the evidence root.
* **The brief and the answer key are separated.** Agents read only the brief (`context/brief`). Answer keys
  (`context/answer_key`) and gold files (`src/eval/gold`) are used only after the run, to score it.
* **Every step is traced** (`trace.jsonl`) and visible live in the GUI.

---

## 4. The zero-hallucination contract

A `Finding` = `{agent, artifact_path, category, claim, value, citations[], observed_text?, status}`.
A `Citation` = `{file_path, locator_type, <locator>, expected_hash, raw_text_snippet, side_hashes?}`.

For each citation the engine:

1. resolves `file_path` inside the sandbox (no absolute paths, no `..`, no symlinks out);
2. recomputes SHA-256 and compares it with `expected_hash`. For SQL citations it also checks the hashes of the
   `-wal` / `-journal` side files the query read (`side_hashes`);
3. re-executes the locator on the physical file;
4. compares the result with `raw_text_snippet`.

| Locator | What the engine re-executes | Used by |
|---|---|---|
| `byte_offset` | reads the bytes at the offset (UTF-8 or hex) | carving, file signatures, Prefetch, LNK, Recycle Bin, `index.dat` |
| `line_number` | reads the text line | OCR corroboration on text files |
| `sql_query` | one read-only `SELECT` (authorizer denies everything else) on a working copy | app databases |
| `regex` | the regex over the decoded file | binary plists |
| `exif_tag` | re-extracts the EXIF tag | photos |
| `registry` | re-opens the hive, reads `"<key>::<value>"` (or the key's last-write time) | Windows registry |
| `evtx_record` | re-renders the event record (EventRecordID) | Windows event logs |

Statuses:

* **VERIFIED**: every citation reproduced. This is the only status that counts as evidence.
* **REJECTED**: a citation failed (altered file, wrong value, bad query). Treated as a hallucination or tampering
  signal.
* **UNCORROBORATED**: a model observation with no ground truth (e.g. a VLM scene description). It is kept and
  labelled, and never presented as evidence.

Model-read text (OCR / VLM) becomes evidence only if the case text index finds the same text in a database cell or
a text-file line, and the cited value explains ≥ 85 % of the model text. Short text (< 25 characters) must also
make up ≥ 50 % of the cell. The second rule stops coincidental matches inside large blobs.

**Independence.** By default each agent's verifier opens its *own* bridge, with its own hash cache and its own
SQLite working copies, so nothing the agent computed or cached is reused to check the agent. The scoring step then
re-verifies all findings once more with a brand-new engine.

---

## 5. Case corpus

A **case** is one investigation: all evidence items seized for a scenario, the brief examiners receive, and (when
published) the answer key.

| Source | Cases (examples) |
|---|---|
| Digital Corpora | M57-Patents, M57-Jean, Nitroba, NGDC 2012, Tuck, NPS-2011 (4), Casper-RW, Domex users, Nexus One / S, Hickman Android 6–14 and iOS 13–17 reference phones |
| NIST CFReDS | Data Leakage, Hacking Case, Rhino Hunt, Russian Tea Room, 10 CFTT chip-off / JTAG phones |
| DFRWS | 2005, 2008, 2009, 2010, 2011 (2), 2016, 2017, 2018, 2021, 2023; CIRCL wiped disk |
| CTFs | Magnet 2019–2022, MVS 2023 / 2024 / 2026, MSAB 2026, Cellebrite 2021–2025, NFI Rodeo 2023, DFIR Madness 001 |
| Other | Lone Wolf, Narcos, Owl, Ali Hadi challenges (13), TechHive, OU Red-Petya / defaced web server, Cado AWS EKS, DSTL ground truth, UTFPR (2), 18 VTO drone models |

* **108 cases registered, 72 READY.** READY means at least one evidence item is extracted and the brief has ≥ 300
  characters or published questions. The remaining cases are waiting on Google Drive quota, very large downloads
  or a re-run after a fix (`HANDOVER.md`).
* **Integrity:** every extracted file has a SHA-256 in `MANIFEST.json` and is re-hashed after inspection
  (0 mismatches). Disk images also produce `FILELIST.jsonl.gz` (every directory entry with MACB times) and recover
  deleted files.
* **Selection:** databases, registry hives, event logs, Prefetch, LNK / jump lists, browser stores, mail, user
  documents and images. **Credential stores (SAM hashes, keychains, `Login Data`, password files) are never
  extracted or carved.**

Layout per case:

```
data/cases/<case_id>/
├── case.json                   registry entry, fetch status, readiness
├── context/brief/              what the examiner is told (+ extracted .txt)
├── context/answer_key/         solutions, used only for scoring
├── context/acquisition/        imaging logs
├── evidence/<item>/            fs/ (sandbox), MANIFEST.json, INSPECTION.json, FILELIST.jsonl.gz
└── workspace/                  memory.json, runs/<run_id>/, reports/ (incl. score.md)
```

---

## 6. Agents

All agents subclass `BaseAgent` (analyze → verify → contrast with the brief → trace → report). The three Sprint 2
agents are deterministic (no model) and run on CPU.

### 6.1 MultimodalVisionAgent (`src/agents/vision_agent.py`)

EXIF / GPS extraction (cited by `exif_tag`), EasyOCR, and Qwen2.5-VL-3B in 4-bit. OCR and VLM text is matched
against the case text index. When the text is backed by a database cell or a text line it becomes a cited,
verifiable finding; otherwise it stays a labelled model observation. Runs on `cuda:0` within a 12 GB budget.

### 6.2 AppArtifactAgent (`src/agents/app_artifact_agent.py`, `src/agents/parsers/`)

27 declarative `TableSpec`s. Each lists candidate column names per field (schemas drift between versions), an
epoch per time field (auto-detected Cocoa / Unix units) and lookups into related tables. Every row becomes one
finding, and every quoted cell is a `SELECT "<col>" FROM "<table>" WHERE rowid = ?` citation. Families covered:
Android SMS / Google Messages, contacts, call log, calendar, accounts; iOS Messages, AddressBook, CallHistory,
KnowledgeC, Notes, Calendar, Photos (GPS); Chromium (URLs, searches, downloads), Firefox, Safari; WhatsApp, Skype,
Facebook Messenger, LINE, Kik, TikTok. All 16 SMS documented for the Hickman Android 10 image are recovered with
exact text and UTC second.

### 6.3 FileSystemAgent (`src/agents/filesystem_agent.py`)

| Category | Meaning |
|---|---|
| `fs.magic_mismatch` | content signature contradicts the extension (renamed / disguised file) |
| `fs.sqlite_unlisted` | SQLite database without a database-like name |
| `fs.orphan_wal` | write-ahead log whose database is missing (uncommitted data) |
| `fs.hidden_path` | user file inside a dot-directory outside app-private trees |
| `fs.deleted_recovered` | file recovered from an unallocated directory entry |
| `fs.mail_store` | PST / OST / DBX / mbox container |
| `sqlite.carved_text` | readable text in SQLite freelist pages, freeblocks and unallocated cell space (deleted rows) |

Carved text is deduplicated per database. Each run gets a triage score (0–1): prose-likeness, identifiers
(e-mail / URL / phone / date) and region (freelist > freeblock > unallocated). Reports list carved text best-first.
The score never changes evidentiary status.

### 6.4 WindowsArtifactAgent (`src/agents/windows_agent.py`)

| Source | Findings |
|---|---|
| SYSTEM hive | computer name, time zone, last shutdown, network interfaces (IP / DHCP / domain), USB storage (USBSTOR, USB VID/PID, serials, key last-write times), MountedDevices |
| SOFTWARE hive | OS product / build / registered owner / install date, installed programs (both views), user profiles, Winlogon last user / domain, network cards, portable devices and ReadyBoost volume labels, network profiles |
| SAM hive (if present) | local account names only, never hashes |
| NTUSER.DAT | UserAssist (ROT13-decoded program, run count, last run), RecentDocs, Office File MRU, TypedURLs, Explorer search terms (WordWheelQuery), RunMRU, MountPoints2 |
| Prefetch (XP / 7 / 8.1 / uncompressed 10) | executable, run count, last run |
| LNK shortcuts | target path, drive type, volume label and serial, network share, target MAC times |
| Recycle Bin | `$I` files (Vista+) and `INFO2` (XP): original path, size, deletion time |
| IE `index.dat` | visited / cached URLs with user and access time |
| EVTX | selected event IDs: logons / logoffs (interactive types), account changes, boot / shutdown, service installs, device installs, MSI installs, RDP sessions, PowerShell, Defender, BITS, task scheduler, Wi-Fi |

The agent reads XP, Vista, 7 and 10 images (validated on NIST, M57, Lone Wolf). EVTX is parsed with pyevtx-rs
(Rust): 28,000 records in under 0.1 s.

### 6.5 Jev typed-answer model and JevAgent (`src/ml/`, `src/agents/jev_agent.py`)

The one model in the project that is **trained here**, from scratch, in PyTorch. Like a Jev-style classifier it
reads a *context* and returns *typed answers with probabilities*, never generated text. The context is one 512-byte
evidence fragment (plus the file extension for one query). There are three query primitives:

| Primitive | Question | Answer |
|---|---|---|
| `choice` | Which file type is this fragment from? (optionally restricted to a given option set) | distribution over 17 types, best type, confidence, abstains below 0.5 |
| `yes_no` | Is the file disguised (content contradicts the extension)? | P(yes), confidence |
| `score` | Where in its file does the fragment sit, 0 (start) to 1 (end)? | expected value over 5 ordinal bins + distribution |

**Data** (`python main.py jev build`, `src/ml/fragments.py`). 140,522 fragments from 44,939 distinct files across the
72 READY cases. The splits are **per case** (`src/datasets/export.SPLITS`): 112,522 train, 12,115 val, 15,813 test
fragments. Files are deduplicated by SHA-256 (test before val before train). Fragments are sector-aligned and come
from the file **body**: the first sector is never used, so the header signature that produced the label is never
an input. Labels are deterministic: the header signature (`filesystem_agent.SIGNATURES`), else `text`, else `other`;
the FileSystemAgent extension rule for "disguised"; and offset / size for position. During training (and with a fixed
seed for val/test), extensions are re-drawn from one shared pool so the extension alone cannot reveal the label.

**Model** (`src/ml/jev.py`, 0.89 M parameters). Each byte goes through Embedding(256, 64), then 4 Conv1d + BatchNorm +
GELU blocks (kernels 9/7/5/3, with max-pooling), then mean‖max pooling into a shared vector z[256]. Three heads sit
on z. `choice` and `score` read the content only. `yes_no` reads z, a character embedding of the extension and the
predicted type distribution. The loss is cross-entropy (label smoothing 0.05) + 0.5·BCE + 0.3·ordinal CE. Training
uses AdamW with a one-cycle learning rate, bf16 autocast and class-balanced sampling (weight ∝ n^-½), runs 20 epochs
(about 2 minutes on the RTX 5080), and keeps the epoch with the best validation score. After training, **temperature
scaling** is fitted on the validation cases, separately for each head.

**Results on the held-out test cases** (`python main.py jev train`; full report with per-class scores, confusion
matrix and training curve in `data/outputs/ml/jev/<run>/REPORT.md`):

| Query | Metric | Jev | Baseline |
|---|---|---|---|
| choice | accuracy / macro-F1 | **0.805 / 0.677** | 0.650 / 0.480 (logistic regression on the byte histogram); 0.241 (majority) |
| choice | ECE before / after temperature scaling | 0.040 / 0.037 | |
| choice | coverage / accuracy when it answers (confidence ≥ 0.5) | 0.911 / 0.846 | |
| yes_no | AUROC, re-drawn extensions | **0.985** | 0.935 (extension only); 0.816 (histogram type + rule) |
| yes_no | AUROC, real extensions only (103 disguised out of 9,591) | 0.780 (precision 0.05, recall 0.43 at 0.5) | |
| score | MAE of the position | 0.227 | 0.243 (constant 0.5) |

Strong types (test F1): evtx 0.94, text 0.95, registry hive 0.93, lnk 0.86, jpeg 0.84, ole2 0.82, bplist 0.80.
Weak types: mp4 / riff / gif / zip bodies (0.03 to 0.38). These are compressed, near-random bytes and are confused
mainly with png and jpeg. A rerun moves accuracy by about ±0.02 (GPU non-determinism).

The **honest weaknesses**:
* On real extensions the disguise query raises many false alarms (precision about 5 % at 0.5).
* The position score barely beats a constant guess.
* elf, gzip and pdf have no test-case support, so they are trained but not measured.

**JevAgent** (`python main.py investigate --case <id> --agents jev`) samples body fragments of each file that has
**no** header signature (wiped headers, unknown containers, recovered files). It asks `choice` and `yes_no` and reports
answers with confidence ≥ 0.8 as `jev.content_type` / `jev.disguised`. These findings have **no citations**, so the
verifier marks them UNCORROBORATED: they are leads that rank files for an examiner, never evidence. The 0
REJECTED contract does not change. Example: on the NIST Data Leakage PC (1,816 files, 3 s) the agent raised 2 leads,
both on the informant's Outlook `.ost`. Its body reads as JPEG-like, which is plausibly Outlook's compressible
encryption and not a real image: exactly the kind of answer that must stay a lead.

Ask the model directly:

```bash
python main.py jev ask --file <any file> --offset 4096 --ext .jpg --options sqlite,jpeg,zip,other
```

**Retrained 2026-11-15** on the repaired fragment dataset (`data/datasets/fragments/20261115`: 145,589 fragments,
46,557 files). The 2026-10-24 build had silently skipped 6,036 files whose paths exceed 260 characters (opened without
the Windows extended-length prefix); fixed in `fragments.py`. Test cases: choice 0.811 accuracy / 0.658 macro-F1
(histogram baseline 0.630 / 0.470); disguise AUROC 0.979 re-drawn, 0.758 on real extensions; position MAE 0.226. The
test set changed (it now includes the long-path files), so the small differences from the table above are not an
improvement or a regression.

### 6.6 Jev-MM: multimodal image-triage model and ImageTriageAgent (`src/ml/mm_*.py`, `src/ml/multimodal.py`, `src/agents/mm_agent.py`)

The project's **multimodal deep-learning model**, trained here. It reads one image through **three modalities** and
fuses them with a learned transformer:

| Modality | Input | Encoder |
|---|---|---|
| vision | the pixels: 224 x 224 letterboxed RGB thumbnail (nothing cropped) | CLIP ViT-B/32 (OpenAI weights, local cache), last block fine-tuned -> 768 -> MLP -> token |
| metadata | 43 features from the container: format, geometry, file size / bits per pixel, colour mode, EXIF tags present (make, model, GPS, capture time, software, UTC offset, exposure), JPEG quality from the quantisation table, chroma subsampling, ICC, PNG text chunks, animation | MLP -> token |
| text | what OCR (EasyOCR, en + es) reads in the image | frozen multilingual MPNet sentence embedding (768) + 9 text statistics (boxes, confidence, text area, status-bar clock, ...) -> MLP -> token |

`[CLS, vision, metadata, text]` + modality embeddings -> 2-layer Transformer encoder (d=256, 4 heads) with a
key-padding mask for absent modalities (no OCR text, undecodable EXIF) -> two heads. **Modality dropout** (p=0.25 per
modality) during training teaches the model to answer with whatever is present. 9.8 M trainable of 90.2 M parameters.

Typed answers, calibrated by temperature scaling on the validation cases (abstain below 0.5):

| Query | Question | Answer |
|---|---|---|
| `choice` | Where does this image come from? | camera / screenshot / messaging media / user file / web cache / thumbnail / app asset |
| `yes_no` | Is it user-generated content? | P(yes) |
| (diagnostic) | Which modality did the fusion rely on? | CLS attention over vision / metadata / text |

**Labels and leakage control.** The label is *where the device stored the file*, from fixed path rules
(`mm_dataset.origin_label`: DCIM -> camera, Screenshots -> screenshot, WhatsApp/Telegram/... media -> messaging,
browser caches -> web cache, thumbnail stores -> thumbnail, other app-private images -> app asset, the rest -> user
file). **The path and file name are never inputs**, so the model must recognise the origin from content and metadata:
exactly the situation of a carved, deleted or renamed image. Splits are per case (same 8 val / test cases as Jev);
files are deduplicated by SHA-256 across splits; at most 400 images per case and origin.

**Data** (`python main.py mm prepare` = `build` + `ocr` + `encode`; datasheet in
`data/datasets/multimodal/<date>/DATASHEET.md`). 12,532 images from 57 READY cases (train / val / test 9,914 / 808 /
1,810); 4,293 carry OCR text. 731 candidates were not decodable (mostly `Thumbs.db` / `thumbcache` containers). The
thumbnail tensor (1.9 GB) is kept outside the synced folder (`settings.CACHE_DIR`) and verified by SHA-256 on load.

**Protocol.** Every model (the fused one and each modality-subset ablation) is the same network trained 12 epochs
with AdamW, one-cycle LR, bf16, class-balanced sampling, label smoothing; the epoch is chosen on the validation cases.
The number of fine-tuned ViT blocks (0 / 1 / 2) and the ViT learning rate were chosen on the validation cases only
(1 block, 1e-5). Each configuration is trained with 3 seeds; test cases are reported once.

**Results on the held-out test cases** (`python main.py mm train`; full report with per-class F1 of every ablation,
confusion matrix, robustness and attention in `data/outputs/ml/multimodal/20261115_191432/REPORT.md`), mean ± std
over seeds 13 / 14 / 15:

| Model | Accuracy | Macro-F1 (7 origins) | User-generated AUROC | User-generated F1 |
|---|---|---|---|---|
| vision only | 0.662 ± 0.010 | 0.423 ± 0.012 | 0.850 ± 0.014 | 0.569 ± 0.005 |
| metadata only | 0.655 ± 0.035 | 0.480 ± 0.056 | **0.909 ± 0.025** | 0.655 ± 0.060 |
| text only | 0.502 ± 0.017 | 0.225 ± 0.021 | 0.611 ± 0.009 | 0.363 ± 0.021 |
| vision + metadata | 0.724 ± 0.006 | 0.520 ± 0.007 | 0.878 ± 0.005 | 0.656 ± 0.016 |
| **vision + metadata + text (Jev-MM)** | **0.744 ± 0.015** | **0.533 ± 0.015** | 0.899 ± 0.003 | **0.681 ± 0.003** |
| majority class | 0.450 | | 0.5 | |

Deployed checkpoint (seed 13): accuracy 0.733, calibrated ECE 0.065 (0.119 before temperature scaling); it answers
90.3 % of images with 0.779 accuracy on those. Per class F1: camera 0.84, app asset 0.88, thumbnail 0.71, web cache
0.63, screenshot 0.32, user file 0.16, messaging 0.11.

**What the multimodality buys** (same deployed model, modalities withheld at inference):

| Withheld | Accuracy | Macro-F1 | User-gen. AUROC |
|---|---|---|---|
| nothing | 0.733 | 0.519 | 0.900 |
| text | 0.735 | 0.543 | 0.888 |
| metadata (EXIF stripped) | 0.647 | 0.396 | 0.847 |
| metadata + text (pixels only) | 0.643 | 0.387 | 0.824 |
| vision | 0.697 | 0.515 | 0.916 |

Reading these honestly:
* **Fusion is the best model on the 7-class task** (+8 accuracy points over the best single modality) and the most
  stable across seeds (metadata alone swings ±0.056 macro-F1). Vision and metadata are complementary: metadata
  identifies camera photos and thumbnails (EXIF, JPEG tables, sizes), pixels identify app assets and web images. The
  fusion's CLS attention reflects this: 0.60 on metadata for camera photos, 0.76 on vision for app assets.
* **Text adds little on average** (+0.02 accuracy, within one std); it helps on screenshots (attention 0.22) but most
  images carry no text. The model is robust to its absence (withholding it costs nothing), which is what modality
  dropout is for.
* **For the binary "user-generated?" query, metadata alone is as good as fusion** (0.909 vs 0.899 AUROC).
* Weak classes: messaging media (106 training images once stickers and emoji are excluded), screenshots (confused
  with web cache / app assets: both are PNG UI renderings), and user files, whose label (a Downloads or Documents
  folder) is not visible in the image. These are the limits of path-derived labels, not hidden successes.

**ImageTriageAgent** (`python main.py investigate --case <id> --agents mm`) asks Jev-MM about every image of an
evidence item and reports `mm.origin` (origin + P(user-generated) + modality attention + OCR text) and
`mm.origin_mismatch` (content confidently contradicts the storage location, or the image was recovered from deleted
space). Findings have **no citations**, so they are UNCORROBORATED leads, never evidence; `rank()` orders images by
P(user-generated) for the vision agent's queue. Check on the held-out case `hickman_android_14` (80 images, 9.6 s):
80 UNCORROBORATED, 0 VERIFIED, 0 REJECTED.

```bash
python main.py mm prepare                      # images -> metadata + thumbnails -> OCR -> text embeddings (~20 min)
python main.py mm train                        # 5 ablations x 3 seeds (~15 min on the RTX 5080) + REPORT.md
python main.py mm ask --file photo.jpg,shot.png
```

Example (`mm ask`, Pixel 3 PoC images, from a training case): the camera photo -> `camera` p=0.91, user-generated
p=0.96, attention metadata 0.61; the screenshot -> `screenshot` p=0.88, OCR "21:33 | LINE | Magisk | Spotify ...",
attention vision 0.40 / metadata 0.40 / text 0.20.


### 6.7 Investigation planner and the deep-learning pipeline (`src/ml/planner*.py`, `src/pipeline/solve.py`)

**Goal (2026-11-15):** learn from solved cases *where to look* in a new one, and hand the right files to the right
agent. The deep-learning pipeline is:

```
case files ──► PLANNER (trained, multimodal) ──► top-K files ──┬─► images ─► ImageTriageAgent (Jev-MM) [+ VLM vision agent]
                                                               └─► others ─► ReaderAgent (identifiers, byte-offset citations)
                                     every statement ──► HeuristicVerificationEngine (safeguard: VERIFIED / REJECTED / lead)
```

The deterministic parsers of §6.2 to §6.4 are **not** part of this pipeline (their code stays in the repository).

**Task and labels.** For every file of a case, predict P(the file holds part of the solution). Labels come from the
published answer keys of the 14 solved cases that have a key and at least one matching file: a file is relevant if
the key names it or its bytes contain one of the key's identifiers (e-mail, URL, domain, phone, file name). The key
is supervision only; the planner sees the brief. Dataset: 28,523 files, 802 relevant (`data/datasets/planner/`).

**Model (multimodal, trained from scratch except the frozen text encoder).** Five modalities, one token each, fused
by a 2-layer transformer: path text (character CNN), first 256 bytes (byte CNN), file-system metadata (size,
category, platform, deleted / hidden, MACB percentiles; MLP), content strings (frozen MPNet + MLP) and the case brief
(frozen MPNet + MLP, the *context* token). BCE loss with case-balanced sampling, modality dropout, temperature
calibration.

**Protocol.** Leave-one-case-out: every case is held out once and never seen in training. Inside each fold, one
other case with ≥ 10 relevant files is the inner validation (epoch choice, temperature, model selection). Averages
use the 10 held-out cases where ranking means something (≥ 50 files, not all relevant).

**Results** (`python main.py planner train`; `data/outputs/ml/planner/20261115_202510/REPORT.md`), mean ± s.e.
over held-out cases:

| model | AUROC | AP | coverage@10 | coverage@50 | coverage@100 |
|---|---|---|---|---|---|
| random | 0.477 ± 0.041 | 0.039 ± 0.017 | 0.012 ± 0.010 | 0.042 ± 0.022 | 0.193 ± 0.078 |
| category prior (positive rate per category) | 0.754 ± 0.049 | 0.269 ± 0.092 | 0.225 ± 0.090 | 0.535 ± 0.102 | 0.624 ± 0.109 |
| logistic regression (metadata) | 0.725 ± 0.074 | 0.322 ± 0.085 | 0.387 ± 0.115 | 0.662 ± 0.092 | 0.746 ± 0.086 |
| network: path only | 0.553 ± 0.051 | 0.092 ± 0.029 | 0.236 ± 0.073 | 0.448 ± 0.107 | 0.537 ± 0.095 |
| network: header bytes only | 0.614 ± 0.058 | 0.081 ± 0.032 | 0.059 ± 0.056 | 0.282 ± 0.079 | 0.430 ± 0.109 |
| network: metadata only | 0.706 ± 0.073 | **0.350 ± 0.093** | **0.552 ± 0.118** | 0.659 ± 0.109 | 0.691 ± 0.107 |
| network: content only | 0.563 ± 0.079 | 0.089 ± 0.037 | 0.154 ± 0.063 | 0.396 ± 0.096 | 0.453 ± 0.111 |
| network: 4 modalities, no context (**deployed**) | 0.710 ± 0.065 | 0.219 ± 0.078 | 0.356 ± 0.122 | 0.547 ± 0.103 | 0.663 ± 0.092 |
| network: 5 modalities | 0.703 ± 0.055 | 0.192 ± 0.061 | 0.307 ± 0.099 | 0.534 ± 0.107 | 0.576 ± 0.107 |
| network: 5 modalities + per-modality auxiliary heads | 0.750 ± 0.045 | 0.118 ± 0.038 | 0.201 ± 0.080 | 0.441 ± 0.115 | 0.668 ± 0.101 |

coverage@K = share of the case's answer items inside the K top-ranked files. In that first run the deployed
variant was chosen by mean inner-validation AP (0.372 vs 0.339 / 0.337 / 0.318), never by held-out results. It has
since been replaced by the self-supervised variant below; a rerun of the same seed gave "4 modalities, no context"
coverage@50 0.442 instead of 0.547 (GPU non-determinism), so single-run differences of that size are noise and the
pretraining comparison is made over 3 seeds.

**Reading it honestly.** The planner learns transferable "where to look" knowledge: reading its top 50 files reaches
55 % of a new case's answer items, against 4 % in random order and 54 % for the category prior, and its top 10 beat
the prior (0.36 vs 0.23). But **multimodal fusion does not beat metadata alone here**: with only 13 training cases
per fold, the high-capacity path and content encoders learn case-specific shortcuts (one case's URLs and folder
names) that do not transfer. This is the *modality imbalance / greedy overfitting* failure of multimodal training
(Wang et al., CVPR 2020); per-modality auxiliary heads, its usual remedy, improved AUROC but not precision. The
contrast with Jev-MM (§6.6), where 9,914 training images let fusion win clearly, is the lesson: fusion pays off
when the number of independent training units (here *cases*, not files) is large enough. More solved cases with
answer keys is the most valuable next step.

**Self-supervised pretraining of the encoders (2026-11-15; `src/ml/planner_pretrain.py`).** The binding constraint
is the number of labelled *cases*, while 50 READY cases have no answer key. Their files supply their own supervision:
the four views of one file (path, header bytes, metadata, content strings) must recognise each other among the other
files of a batch: symmetric InfoNCE over the 6 modality pairs, learnable temperature (CLIP-style alignment between
the modalities of a file). The encoders keep the exact `PlannerNet.enc` architecture.

* Data (`python main.py planner pretrain-build`): 80,126 unique files (43,779 SHA-256 duplicates removed so identical
  system files do not become contradictory positives) from the 50 READY cases **without** an answer key; the 14
  labelled cases (every LOCO held-out case among them) are excluded entirely, so there is no transductive leakage.
* Training (`python main.py planner pretrain`, 2.5 min on the RTX 5080): batch 512, AdamW 1e-3, 20 × 300 steps,
  sampling ∝ 1/√(case size); 6 whole cases held out, best epoch by their loss (5; it overfits afterwards).
  Held-out cross-modal retrieval top-1 among 512 files (chance 0.002): header↔content **0.349**, path↔content 0.264,
  path↔header 0.232, path↔metadata 0.180, header↔metadata 0.108, metadata↔content 0.104
  (`data/outputs/ml/planner_pretrain/20261115_230146/REPORT.md`; encoders `data/models/planner/pretrained_encoders.pt`).
* Fine-tuning variants in `planner train`: `… + SSL` (pretrained encoders at 0.1 × the learning rate),
  `… + SSL frozen` (only the fusion transformer and the brief / head are trained), `metadata + SSL`.

Seed study (`python -m src.ml.planner_seeds`; `data/outputs/ml/planner/seeds_20261116_005920/REPORT.md`), mean ± std
over seeds 13 / 14 / 15 of the LOCO case means:

| model | AUROC | AP | coverage@10 | coverage@50 | coverage@100 |
|---|---|---|---|---|---|
| random | 0.514 ± 0.034 | 0.042 ± 0.002 | 0.014 ± 0.010 | 0.106 ± 0.045 | 0.229 ± 0.054 |
| category prior | 0.754 | 0.269 | 0.225 | 0.535 | 0.624 |
| logistic regression (metadata) | 0.725 | **0.322** | 0.387 | **0.662** | **0.746** |
| network: metadata only | 0.684 ± 0.020 | 0.278 ± 0.056 | **0.414 ± 0.098** | 0.633 ± 0.024 | 0.670 ± 0.015 |
| network: 4 modalities, no context | 0.690 ± 0.020 | 0.169 ± 0.036 | 0.285 ± 0.013 | 0.549 ± 0.046 | 0.599 ± 0.058 |
| + SSL encoders, fine-tuned | **0.730 ± 0.008** | 0.161 ± 0.025 | 0.319 ± 0.037 | 0.487 ± 0.072 | 0.629 ± 0.025 |
| + SSL encoders, frozen (**deployed**) | 0.702 ± 0.024 | 0.198 ± 0.025 | 0.409 ± 0.026 | 0.577 ± 0.035 | 0.651 ± 0.053 |
| network: metadata + SSL encoder | 0.703 ± 0.023 | 0.273 ± 0.043 | 0.381 ± 0.048 | 0.593 ± 0.024 | 0.637 ± 0.006 |

The frozen variant was deployed by inner-validation AP (0.349, the highest of the 7 candidates;
`data/outputs/ml/planner/20261115_230158/REPORT.md`). **Reading it honestly:** pretraining on unlabelled cases
helps the fusion network: frozen pretrained encoders improve AP (+0.03), coverage@10 (+0.12) and coverage@100
(+0.05) and lower the seed variance of AP; fine-tuning them gives the best AUROC of any network (0.73) but a worse
top of the ranking (fine-tuning on 13 cases still overfits). The gains are 1–2 standard deviations, and the
pretrained fusion network still does **not** beat metadata alone or logistic regression on AP / coverage@50:
self-supervision narrows the gap left by the small number of labelled cases but does not close it. Freezing working
better than fine-tuning is consistent with the diagnosis above (high-capacity encoders overfit 13 cases).

**Update 2026-11-16: 16 labelled cases, metadata-anchored fusion, deterministic training.** The tables above use
14 cases. Three changes followed, all evaluated over 3 seeds:

* *More labels.* Several answer keys are password-protected instructor material (M57-Jean, Nitroba, NGDC evidence
  reports, Hickman Android 6, M57 instructor packet) and were not opened. For the readable narrative keys, distinctive
  solution tokens were transcribed by hand (`src/eval/planner_tokens.json`, each with its source): **MVS 2024**
  (42 relevant files: contacts, SMS / RCS, chat and mail databases naming Chadwick / Rocco / "vitamins") and **NGDC
  2012** (5: SMS, mail index and packet captures with "flash mob" / "Krasnovia") become labelled. The drone reports'
  salted GPS points match nothing (binary flight logs). Dataset `data/datasets/planner/20261116`: 16 cases, 33,047
  files, 849 relevant.
* *Metadata-anchored fusion* (`PlannerConfig.anchor`): logit = head_m(metadata token) + g · head(fused CLS), g a
  learnable scalar initialised at 0 (ReZero-style), with an auxiliary loss on head_m. The network starts as the
  metadata model, the strongest modality on few cases, and the other modalities can only add a learned correction.
* *Determinism* (`planner_train.deterministic`): cuDNN / cuBLAS deterministic, so a rerun gives identical
  predictions (checked: max difference 0.0); earlier, one seed moved coverage@50 by 0.1 between runs.

Seed study on 16 cases (`data/outputs/ml/planner/seeds_*` latest; LOCO run with all ablations
`data/outputs/ml/planner/20261116_012239/REPORT.md`), mean ± std over seeds 13 / 14 / 15, 13 rankable held-out cases:

| model | AUROC | AP | coverage@10 | coverage@50 | coverage@100 |
|---|---|---|---|---|---|
| random | 0.496 ± 0.032 | 0.035 ± 0.001 | 0.004 ± 0.004 | 0.103 ± 0.043 | 0.182 ± 0.042 |
| category prior | **0.776** | 0.264 | 0.373 | 0.614 | 0.689 |
| logistic regression (metadata) | 0.767 | **0.301** | **0.440** | **0.708** | **0.782** |
| network: metadata only | 0.720 ± 0.021 | 0.240 ± 0.040 | 0.366 ± 0.117 | 0.653 ± 0.061 | 0.706 ± 0.022 |
| network: 4 modalities, plain fusion | 0.708 ± 0.023 | 0.194 ± 0.005 | 0.323 ± 0.054 | 0.517 ± 0.075 | 0.594 ± 0.084 |
| + SSL encoders, fine-tuned | 0.738 ± 0.031 | 0.172 ± 0.009 | 0.289 ± 0.018 | 0.508 ± 0.020 | 0.620 ± 0.031 |
| + SSL encoders, frozen | 0.740 ± 0.025 | 0.218 ± 0.010 | 0.378 ± 0.036 | 0.588 ± 0.069 | 0.655 ± 0.055 |
| network: metadata + SSL encoder | 0.753 ± 0.011 | 0.270 ± 0.028 | 0.398 ± 0.044 | 0.578 ± 0.016 | 0.662 ± 0.008 |
| **anchored fusion (deployed)** | 0.724 ± 0.017 | **0.279 ± 0.026** | **0.433 ± 0.027** | 0.632 ± 0.030 | 0.691 ± 0.006 |
| anchored fusion + SSL frozen | 0.740 ± 0.008 | 0.219 ± 0.012 | 0.410 ± 0.054 | 0.610 ± 0.017 | 0.672 ± 0.019 |

Inner-validation AP chose **anchored** (0.313; next 0.272 anchored + SSL frozen, 0.267 metadata), so
`data/models/planner/planner.pt` and `loco/<case>.pt` are now anchored-fusion models. **Reading it honestly:**
anchoring is the change that matters for fusion: against plain fusion it raises AP by 0.085 (0.194 → 0.279),
coverage@10 by 0.11 and coverage@100 by 0.10, and it beats the metadata-only network on AP and coverage@10 with a
third of its variance: the fused correction adds information once it can no longer drown the metadata signal. SSL
encoders still help *plain* fusion (AP 0.194 → 0.218) but add nothing on top of anchoring. A logistic regression on
the metadata vector remains marginally better (AP 0.301, within one std of the anchored network): with 15 training
cases per fold, the strongest learnt model still only matches a well-regularised linear model on the same features.

**End to end** (`python -m src.eval.solve_eval --k 100`; `data/outputs/ml/planner/solve_eval_k100_20261116_044831.md`,
anchored folds, 16 cases): **138 / 869** answer-key identifiers (per case: NIST Data Leakage 69, Lone Wolf 60,
M57 4, Owl 2, NIST Hacking 1, Rhino Hunt 1, DFRWS 2005 1), the same total as the SSL-frozen folds below and 137 for
the deterministic agents; 0 REJECTED. (MVS 2024 and NGDC 2012 carry no automatic key identifiers, so they do not
change this measure.)

**End to end, earlier SSL-frozen folds, 14 cases** (`python -m src.eval.solve_eval --k 100`; `data/outputs/ml/planner/solve_eval_k100_20261116_010348.md`):
every solved case investigated by the planner fold that never saw it (deployed variant, SSL frozen), reading **at
most 100 files per case**; the answer key is used only afterwards. The pipeline's VERIFIED findings contain **138 of
869** answer-key identifiers (15.9 %; 126 with the earlier non-pretrained folds), against 137 (15.8 %) for the
earlier deterministic agents that parse **every** file; it recovers something in 8 of 14 cases (the deterministic
agents in 4), e.g. Lone Wolf 60 vs 39 and NIST Hacking Case 2 vs 0, but fewer on NIST Data Leakage (67 vs 93, where
the registry / browser parsers read files beyond the top 100). About 8,900 statements were verified and **0 were
REJECTED**: the safeguard holds for the deep-learning pipeline. (One end-to-end run per variant; not seed-averaged.)

---

## 7. Evaluation methodology

Three questions, three measures:

**(a) Is every reported statement true to the evidence?**
Every finding is re-verified at creation. `python main.py score` then re-verifies the latest findings of every
agent with a fresh engine. Metric: *REJECTED count* and *audit failures*. Target: 0.

**(b) Do the agents find what investigators were expected to find?** Gold questions.
`src/eval/gold/<case>.json` transcribes the scored questions of a published answer key. Each question lists
requirement groups (facts the answer needs). A group is met when any of its alternatives occurs in a VERIFIED
finding (claim + value text; case-insensitive, URL-decoded, `\` normalised to `/`):

* `cat:<category>` matches a finding category, and `cat:<category>|<text>` requires both in one finding. For
  example, "executed" is only satisfied by Prefetch / UserAssist, never by a download URL.
* `same_finding: true` requires all groups to be met by one finding (e.g. IP, DHCP server and domain of one
  interface).
* **Credit** = groups met / groups. **Answered** = all groups met.
* Every met group records the finding id that met it, so each point of the score can be checked against its
  citation in `workspace/reports/score.md`.
* Questions that need a capability not built yet carry `needs` (e.g. `shellbag/jumplist`, `ost`). Scores are
  reported over all questions and over the covered subset.
* Questions that are not checkable facts (acquisition hashes, narratives, diagrams, duplicates) are listed under
  `excluded` with a reason.

Gold sets exist for **NIST Data Leakage** (38 scored / 22 excluded of 60) and **NIST Hacking Case** (23 scored /
8 excluded of 31). M57-Patents could not get one: its instructor answer key is password-protected, and the
remaining M57 documents describe the tasks but not the solutions.

**(c) An automatic, coarse signal for every case with an answer key.** Answer-key entity recall.
Identifiers in the answer key (e-mail, URL, domain, phone, file name, GPS) that are *not* already in the brief are
extracted, and the score is the share that appears in verified findings. *On-target rate* is the share of verified
findings that mention at least one of them. The measure is cheap and automatic, but noisy: answer keys mention
tools, references and URLs that are not evidence. Gold questions remain the measured score.

---

## 8. Results in detail

### 8.1 Corpus run (`python main.py corpus`, 72 READY cases, ~28 min on a laptop CPU)

| Agent | Evidence items with output | Artefacts analysed | VERIFIED | REJECTED |
|---|---|---|---|---|
| FileSystemAgent | 100 | 162,774 | 145,456 | 0 |
| WindowsArtifactAgent | 30 | 12,836 | 41,467 | 0 |
| AppArtifactAgent | 45 | 289 | 24,230 | 0 |
| **Total** | | | **211,153** | **0** |

Largest categories: `sqlite.carved_text` 128,970 (158,539 before deduplication), `win.event` 20,672, `ie.url`
13,497, `knowledgec.event` 10,339, `fs.deleted_recovered` 7,373, `chrome.url` 3,596, `sms.message` 3,098.
The Windows agent cut the number of READY cases without any app- or system-level finding from 40 to 27. The
remaining 27 are mostly memory images, network captures, Linux servers, drone logs and raw phone dumps (§9).

### 8.2 NIST CFReDS Data Leakage (Windows 7 PC + 2 USB sticks; the CD-R is UDF and not yet readable)

| Group | Questions | Result |
|---|---|---|
| System: OS, time zone, computer name, accounts, last logon, shutdown, network interface | Q3–Q9 | all answered |
| Software: installed apps, execution (12 of 13 executables), on/off and logon events, browsers | Q10–Q13 | answered; Q11 92 % |
| User activity: browser searches (12 of 20 terms), Explorer search "secret" | Q16–Q17 | 60 % and 100 % |
| E-mail: client, mail store, account | Q18–Q20 | answered; e-mail *content* (Q21) needs an OST parser |
| Devices: both USB serials and the `IAMAN $_@` volume label; network share 10.11.11.128 and files opened on it | Q22, Q24, Q27, Q28 | answered |
| Cloud: Google Drive / iCloud traces | Q29 | answered; deleted Drive files and account (Q30–Q31) need `sync_log.log` / carved `sync_config.db` reading |
| CD-R / RM#2: files opened from the CD-R, resignation letter and its XPS print, anti-forensics (Eraser, CCleaner) | Q35–Q37, Q52 | answered |
| Not reached | Q25–Q26 (ShellBags / jump lists), Q30–Q34 (Drive logs, burn staging, USN journal), Q39 (thumbcache), Q41 (Sticky Notes), Q51 (Recycle Bin, items not extracted), Q53 / Q56 (FAT / UDF carving) | 0–5 % |

**Total: 23 / 38 answered (60.5 %), mean credit 64.8 %. On the 30 questions within built capabilities: 23 / 30
(76.7 %), mean credit 81.9 %.**

### 8.3 NIST CFReDS Hacking Case (Windows XP laptop)

Answered (11): OS, install date, time zone, registered owner, computer name, last shutdown, main account, network
cards, hacking programs installed (Cain, Ethereal, 123WASP, Anonymizer, CuteFTP, Look@LAN, NetStumbler),
newsgroup / mail clients, subscribed newsgroups (from Outlook Express mail stores).
Not answered (12): primary domain, Look@LAN `irunin.ini` (not extracted), IP / MAC, SMTP / NNTP settings (registry
account keys not parsed), mIRC settings and logs, Ethereal capture contents, victim's browsing, Yahoo mail pages.
Most of these answers sit inside text / HTML / mail files that no agent reads for content yet (§9).
**Total: 11 / 23 (47.8 %).**

### 8.4 Answer-key entity recall

| Case | Found / identifiers | Recall |
|---|---|---|
| nist_data_leakage | 93 / 250 | 37.2 % |
| m57_patents | 3 / 8 | 37.5 % |
| owl | 2 / 8 | 25.0 % |
| lone_wolf | 39 / 502 | 7.8 % |
| 15 other cases with identifiers in the key | 0 / 139 | 0 % |
| **Pooled (19 cases)** | **137 / 907** | **15.1 %** |

Zero recall happens where evidence types are not parsed yet (drone flight logs, memory images, pcaps, Linux
servers) or where the key's identifiers are tool names and references rather than evidence content.

### 8.5 Vision agent (proof of concept)

On the Hickman Android 10 Pixel 3 image, 2 images took 30 s and 5.2 GB VRAM (9 VERIFIED, 0 REJECTED). Examples:
GPS of a camera photo re-checked against EXIF; a screenshot reading "Super Secret Message" tied to a launcher
folder through `SELECT "title" FROM "favorites" WHERE rowid = 42`. All 58 images: 40 VERIFIED, 0 REJECTED, 244
UNCORROBORATED, 341 s, peak 6.5 GB allocated. The vision agent was not part of the corpus run.

---

## 9. Limitations and known gaps

What is measured as missing, in order of expected score gain:

1. **File content is not read for evidence.** Text, HTML, INI and log files and mail bodies (OST / PST / DBX) are
   extracted but not turned into findings. This blocks most of the remaining Hacking Case answers and NIST Q21 / Q30
   / Q31.
2. **Windows artefacts not parsed yet:** ShellBags, jump lists (`automaticDestinations-ms`), `$UsnJrnl` / `$LogFile`,
   thumbcache, Sticky Notes, Windows Search `Windows.edb` and ESE `WebCacheV01.dat`, registry e-mail account keys.
3. **Evidence types without agents:** memory images (Volatility), network captures, Linux logs, drone flight logs,
   raw / chip-off phone dumps without a file system, UDF optical media.
4. **Gold coverage:** only two cases have gold questions. M57's key is encrypted, and other keys (Lone Wolf, Owl,
   DFRWS, VTO drones) still need transcription.
5. **Carving volume:** 128,970 carved strings are ranked but still noisy. A relevance model (Sprint 4) should sort
   them.
6. **Registry transaction logs** (`.LOG1/.LOG2`) are not replayed. What is cited is what the primary hive file
   holds.
7. **Evidence location:** the working copy lives in a OneDrive folder. The 2026-11-15 audit (`python main.py cases
   audit`) found 51 damaged files; `cases repair` restored 39 byte-identical and write-protected 22,395 at-risk
   files. 12 remain (11 `desktop.ini`, 1 `Windows PowerShell.evtx` in dfirmadness) because their images are
   compressed at the source. Set `FORENSI_DATA_DIR` to a local folder before relying on hashes in court-style
   reports; re-run `cases audit` after any move.
8. **Scale:** 36 registered cases are not READY yet (Drive quota, very large images).

---

## 10. Installation and usage

Requires Python 3.12 (Windows 11 tested). CUDA 12.8 wheels are needed for RTX 50xx GPUs (vision agent only).

```bash
python -m venv .venv
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt
```

Note: `requirements.txt` pins `evtx` (pyevtx-rs). Do **not** install `python-evtx` in the same environment: on
Windows the `Evtx` and `evtx` packages collide.

```bash
python main.py cases list                                  # cases, evidence, fetch status, readiness
python main.py cases build --cases all --groups light      # documents + evidence (resumable)
python main.py apps --case hickman_android_10              # AppArtifactAgent
python main.py fs   --case m57_patents                     # FileSystemAgent
python main.py win  --case nist_data_leakage               # WindowsArtifactAgent
python main.py investigate --case m57_patents --limit 5    # vision agent (GPU)
python main.py investigate --case nist_data_leakage --agents apps,fs,win
python main.py corpus                                      # apps + fs + win over every READY case
python main.py score --cases gold                          # gold-question cases; --cases all for every READY case
python main.py gui                                         # http://127.0.0.1:8765
python main.py dataset                                     # ML tables -> data/datasets/<date>/
python main.py jev build                                   # fragment dataset -> data/datasets/fragments/<date>/
python main.py jev train                                   # train + calibrate Jev -> data/models/jev/jev.pt
python main.py jev eval                                    # re-evaluate the checkpoint on val / test cases
python main.py investigate --case nist_data_leakage --agents jev   # JevAgent leads
python main.py schemas                                     # FDX JSON Schemas -> docs/schemas/
python -m pytest                                           # tests
```

Outputs per case: `workspace/runs/<run_id>/` (findings JSON per agent and evidence item, verification audit,
trace), `workspace/reports/<run_id>_agents.md` (report), `workspace/reports/score.md` + `score.json`. Corpus-level:
`data/outputs/corpus_runs/`, `data/outputs/scores/`.

---

## 11. Configuration and hardware

| Setting | Default | Meaning |
|---|---|---|
| `FORENSI_DATA_DIR` | `<repo>/data` | where cases, evidence and outputs live; keep it outside synced folders |
| `FORENSI_SINGLE_GPU` | `1` | `0` enables dual-GPU mode (`vision` / `ocr` on `cuda:1`, text on `cuda:0`) |
| `FORENSI_GPU0_BUDGET_GB` / `FORENSI_GPU1_BUDGET_GB` | 12 / 8 | VRAM budget enforced in software |
| `FORENSI_VLM_MODEL`, `FORENSI_VLM_QUANT` | Qwen2.5-VL-3B-Instruct, 4bit | vision model |
| `HF_HUB_OFFLINE` | — | set to `1` when TLS inspection breaks Hugging Face downloads (models load from cache) |

Tested hardware: NVIDIA RTX 5080 Laptop GPU 16 GB, with the 12 GB budget enforced so the code stays valid for the
target RTX 5070. Target: dual GPU (RTX 5070 + RTX 4060), no agent code changes needed.

---

## 12. Repository layout

```
main.py                             CLI
src/config/settings.py              devices, VRAM budgets, model ids, data directory
src/cases/                          case registry (108), builder, catalogue, document text extraction
src/utils/                          fetchers: remote sources, bundles, disk images, dissect, mobile scraper
src/bridge/forensic_bridge.py       sandbox, SQLite working copies, EXIF, text, FS walk
src/bridge/windows_bridge.py        registry, EVTX, Prefetch / LNK / Recycle Bin / index.dat layouts
src/bridge/text_index.py            index of database text for OCR / VLM corroboration
src/verification/                   Citation / Finding models, HeuristicVerificationEngine
src/agents/base_agent.py            BaseAgent: analyze -> verify -> contrast -> trace -> report
src/agents/vision_agent.py          MultimodalVisionAgent
src/agents/app_artifact_agent.py    AppArtifactAgent; parsers in src/agents/parsers/
src/agents/filesystem_agent.py      FileSystemAgent
src/agents/windows_agent.py         WindowsArtifactAgent
src/agents/jev_agent.py             JevAgent (leads from the trained Jev model)
src/ml/                             Jev: fragment dataset, model + query API, training / calibration / evaluation
src/agents/trace.py                 run tracing (JSONL)
src/context/case_context.py         brief entities, finding <-> context links, investigation memory
src/pipeline/investigate.py         run agents over a case
src/eval/scoring.py                 audit, gold questions, answer-key entity recall
src/eval/gold/                      gold questions per case (from published answer keys)
src/reports/case_report.py          Markdown reports
src/gui/                            local web console
src/datasets/                       ML dataset export, fine-tuning samples
src/schemas/                        FDX contracts (BridgeRecord, ForensicTimestamp, TrainingSample)
docs/DATA_DICTIONARY.md             every table, field and category
docs/DATA_INTERCHANGE_STANDARD.md   FDX: citable records and training samples
docs/schemas/                       JSON Schemas
tests/                              183 tests
tools/README.md                     how to obtain the bundled 7-Zip binaries (not in git)
```

---

## 13. Tests

`python -m pytest` runs 183 tests in about 15 s, with no GPU or dataset needed. They cover:

* **Verifier:** every locator, forgery and tampering (changed byte, changed WAL, missing side-file hash), sandbox
  escapes, OCR corroboration thresholds and regressions.
* **Agents:** every SQLite parser on synthetic databases (round trip, forgery, absence, integrity); FileSystemAgent
  categories; Windows layouts (Prefetch v17 / v23 / v26, `$I` v1 / v2, INFO2, EVTX rendering), long-path
  resolution, verifier independence.
* **Jev model:** fragment labels and disguise rule, header sector never used, a short CPU training run on
  synthetic types (accuracy, calibration, query API), metrics (AUROC, ECE, temperature), and JevAgent answers always
  UNCORROBORATED with no citations.
* **Scoring:** normalisation, category and same-finding rules, partial credit, well-formed gold files.
* **Real-corpus integration** (skipped when data is absent): Android 10 documented SMS, NIST Data Leakage
  registry answers.

---

## 14. Roadmap

* **Sprint 3:**
  * `TimelineAgent`: one UTC timeline per case from VERIFIED findings, cross-source correlation.
  * `OrchestratorAuditorAgent`: plans agents per evidence item, final fresh audit, ranked carving, case report
    where every sentence references a finding id, score against the answer key.
  * Content parsers for text / HTML / mail stores, ShellBags, jump lists, USN journal.
  * More gold sets (Lone Wolf, Owl, DFRWS).
* **Sprint 4:** learning baselines on the exported dataset (artefact classifier, carving relevance, image triage,
  timeline anomalies), evaluated on held-out cases only.
* **Sprint 5:** coverage (UDF, memory images, encrypted iTunes backups, NAS RAID), dual GPU.

---

## 15. Data handling and ethics

* All cases are public training / reference images published for forensic education. **Evidence is not in this
  repository**; `main.py cases build` rebuilds it locally.
* Credential material is never extracted, carved or printed. Reports never echo the test-account passwords that
  some image-creation documents publish, and published archive passwords stay in the registry, never in reports.
* Answer keys are never given to agents and are only used for scoring.

Further documentation: [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md),
[`docs/DATA_INTERCHANGE_STANDARD.md`](docs/DATA_INTERCHANGE_STANDARD.md).

License: MIT (see `LICENSE`).
