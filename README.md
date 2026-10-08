# Forensic Orchestrator — ForensiAgent-Multimodal

A multi-agent framework for automated analysis of digital forensic evidence: Android / iOS extractions and
Windows / Linux / removable disk images. Its core rule is that **no finding is reported unless a deterministic
engine re-verifies it against the physical evidence**. Every statement an agent makes carries citations (file +
SHA-256 + an exact locator). The `HeuristicVerificationEngine` re-executes each citation on the file before the
statement can count as evidence.

This document reports the state of the project as of **2026-10-08** (branch `sprint-2`). It covers what is built,
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
| Unit and integration tests | 176 pass (no GPU or dataset required; corpus tests skip when data is absent) |

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
7. **Evidence location:** the working copy currently lives in a OneDrive folder. Sync clients have deleted two
   `desktop.ini` evidence files. Set `FORENSI_DATA_DIR` to a local folder before relying on hashes in court-style
   reports.
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
tests/                              176 tests
tools/README.md                     how to obtain the bundled 7-Zip binaries (not in git)
```

---

## 13. Tests

`python -m pytest` runs 176 tests in about 15 s, with no GPU or dataset needed. They cover:

* **Verifier:** every locator, forgery and tampering (changed byte, changed WAL, missing side-file hash), sandbox
  escapes, OCR corroboration thresholds and regressions.
* **Agents:** every SQLite parser on synthetic databases (round trip, forgery, absence, integrity); FileSystemAgent
  categories; Windows layouts (Prefetch v17 / v23 / v26, `$I` v1 / v2, INFO2, EVTX rendering), long-path
  resolution, verifier independence.
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
