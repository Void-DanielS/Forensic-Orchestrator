# HANDOVER: ForensiAgent-Multimodal

_Last updated: 2026-10-08, Sprint 2.5 (WindowsArtifactAgent, verifier hardening, scoring against answer keys).
Read §0 first, then §1c / §1d. README.md is the full, peer-facing project report (results, metrics, limitations)._

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
* M57 has **no** gold set: `m57-instructor-packet.pdf` is password-protected; the other M57 key decks have no answers.
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
