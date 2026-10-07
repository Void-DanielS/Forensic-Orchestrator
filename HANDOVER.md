# HANDOVER: ForensiAgent-Multimodal

_Last updated: 2026-10-06, Sprint 1.5 (case database, computer images, context, GUI)._

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
| CLI | `main.py` | ✅ `config / fetch / inspect / verify-selftest / vision / pipeline` |
| Unit tests | `tests/` | ✅ 78 pass (no GPU or dataset needed) |
| FileSystemAgent, AppArtifactAgent | n/a | ⏳ Sprint 2 |
| TimelineAgent, OrchestratorAuditorAgent | n/a | ⏳ Sprint 3 |

**Sprint 1 priority (set by the user):** gather a large amount of real data, plus a *small* PoC
of the first agent on one or two images with real results. Both are done.

The project is **not a git repository** yet (the folder was renamed from `forensi_agent`; `.venv/Scripts/*.exe` launchers are stale — use `.venv/Scripts/python -m pip`). Run `git init` before Sprint 2, and add
`.venv/`, `data/cases/`, `data/datasets/`, `data/_spool/` and `data/outputs/` to `.gitignore`.

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
| `m57_patents` | ● Charlie, Jo, Pat, Terry workstations (XP); ● 4 USB drives | affidavit + 4 detective reports | instructor packet + 3 scenario decks |
| `lone_wolf` | ● suspect laptop (Win10, 9×E01) | README | forensic reports bundle (teacher guide is password-protected) |
| `narcos` | ● 3 PCs (split raw, heavy) + ● post-CCleaner PC | — | teacher guide **password-protected** (see RESTRICTED note) |
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
* Lone Wolf / Narcos teacher guides are password-protected (instructor request to Digital Corpora).
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

## 8. Copy-paste prompt: Sprint 2 (AppArtifactAgent + FileSystemAgent)

```
You are continuing ForensiAgent-Multimodal in C:\Forensic - Orchestrator\forensic_agent.
Read HANDOVER.md and README.md first, then src/agents/base_agent.py, src/agents/vision_agent.py,
src/bridge/forensic_bridge.py, src/verification/heuristic_verifier.py, src/verification/models.py
and src/utils/dataset_scraper.py, src/utils/disk_image.py, src/cases/*, src/context/case_context.py,
src/pipeline/investigate.py. Use the existing .venv and the case database in data/cases/
(data/cases/catalog.sqlite lists cases and evidence; each sandbox is data/cases/<case>/evidence/<id>/fs). Do not re-download.
Run `python -m pytest` and `python main.py verify-selftest` first to confirm a green baseline.

Goal: two agents that subclass BaseAgent and emit Finding objects whose citations the
HeuristicVerificationEngine re-executes. All DB access goes through bridge.sqlite (working copies;
never open evidence in place). Run them across ALL cases, including the Windows / removable / Linux disk images (FILELIST.jsonl.gz has every
MFT/inode entry with MACB times and deleted flags).

1. src/agents/app_artifact_agent.py: AppArtifactAgent(role="text")
   - Parsers in src/agents/parsers/, one module per artifact, each with a `detect(sandbox)` that finds its
     DBs by path pattern (paths differ per Android/iOS version and between Hickman/Magnet/DFRWS images):
     Android SMS (mmssms.db), contacts2.db, calllog, WhatsApp msgstore.db (+wa.db), Chrome History
     (urls/visits/downloads/keyword_search_terms; WebKit epoch), iOS sms.db, AddressBook, CallHistory,
     Safari History.db, KnowledgeC, and at least three third-party messengers present in the corpus
     (Telegram, Signal, Snapchat, Discord, Kik, TikTok...).
   - Each record becomes a Finding with a sql_query citation selecting the row by primary key; the
     raw_text_snippet is the exact stored value. Normalise timestamps to UTC ISO-8601 in Finding.value
     and keep the raw value and epoch type.
   - Validate against case context: android_10/_case_context/SMS-Messages.xlsx and the image-creation
     PDFs (.txt alongside). Write tests that the SMS parser recovers the documented messages.
2. src/agents/filesystem_agent.py: FileSystemAgent(role="text")
   - Walk each sandbox with bridge.fs: MACB timestamps, SHA-256 (reuse MANIFEST.json hashes), extension vs
     magic mismatch, hidden paths, SQLite without a .db extension, orphan -wal files, SQLite freelist /
     unallocated-cell carving for deleted records (cite with byte_offset + hex snippet).
   - Add pytsk3/dfvfs support so the registered physical images android_7/android_8 (blk0_mmcblk0.bin)
     can be fetched: stream the .bin to the spool, open the userdata partition, extract with the same
     selectors. Then move them from group "future" to "extended" in dataset_scraper.SOURCES.
3. Extend CaseTextIndex to XML/JSON/plist text so vision corroboration covers those artifacts.
4. main.py: add `apps` and `fs` subcommands; make `pipeline` run all three agents per source and write
   data/cases/<case>/workspace/runs/<run_id>/<agent>__<evidence>_findings.json, emitting trace events (src/agents/trace.py)
   and context links like the vision agent, so the GUI shows them live.
5. Tests: per-parser unit tests on tiny synthetic DBs in tests/fixtures, plus an integration test on the real
   corpus (skipped when a source is missing). Keep 0 REJECTED on real data; investigate any rejection
   rather than loosening the verifier.
6. Update HANDOVER.md (state, results, next prompt) at the end.
Constraints: single GPU cuda:0 with a 12 GB budget. Never write into data/cases/*/evidence/*/fs. Never load context/answer_key into an agent. Never print
credentials from _case_context in reports.
```

## 9. Copy-paste prompt: Sprint 3 (TimelineAgent + OrchestratorAuditorAgent)

```
You are continuing ForensiAgent-Multimodal in C:\Forensic - Orchestrator\forensic_agent.
Read HANDOVER.md (including the Sprint 2 section), README.md and all of src/. Run `python -m pytest`
and `python main.py --source android_10 pipeline --limit 5` to confirm the baseline.

1. src/agents/timeline_agent.py: TimelineAgent(role="text")
   - Input: VERIFIED findings only (data/cases/<case>/workspace/runs/*/*_findings.json) plus FILELIST MACB entries.
   - Unified UTC timeline per source. Handle unix s/ms, WebKit, Cocoa (s and ns), FILETIME, and EXIF local
     time + OffsetTimeOriginal. The Hickman documentation uses EST (UTC-5).
   - Cross-source correlation (e.g. photo EXIF time/GPS vs messages and browser activity within a window).
     A correlation Finding cites the union of the underlying citations, so it verifies only if all of them do.
   - Validate against the timelines in the image-creation PDFs (_case_context/*.txt).
   - Optional: a small local LLM (4-bit, cuda:0) summarises clusters. Its output is UNCORROBORATED unless
     every sentence maps to cited findings.
2. src/agents/orchestrator_auditor.py: OrchestratorAuditorAgent(role="orchestrator")
   - Plans and dispatches agents per source, re-verifies every finding at the end with a fresh
     HeuristicVerificationEngine and fresh hashes, and drops anything not VERIFIED from the evidence sections.
   - Use a FRESH ForensicBridge/verifier (no shared hash cache) and hash -wal/-journal side files too.
   - Score against context/answer_key (never give it to agents); per case, all evidence items together.
   - data/cases/<case>/workspace/reports/case_report.md: case summary, device info (from verified findings only), timeline,
     per-agent evidence tables with citations (file, locator, SHA-256), a separate "Model observations
     (not evidence)" appendix, and verification statistics. Every evidence sentence references a finding_id.
3. Dual-GPU readiness: run with FORENSI_SINGLE_GPU=0 on one GPU to confirm the fallback warning path works.
   Document the expected VRAM per role.
4. Tests + update HANDOVER.md.
```
