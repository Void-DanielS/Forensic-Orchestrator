# Data Dictionary — ForensiAgent-Multimodal

Field-level reference for every record the project stores or exchanges. The contracts that agents, the bridge and
training code must follow are in [`DATA_INTERCHANGE_STANDARD.md`](DATA_INTERCHANGE_STANDARD.md). Machine-readable JSON
Schemas are in [`schemas/`](schemas/) (regenerate with `python main.py schemas`).

Schema version: **`fdx/1.0`**.

---

## 1. Global conventions

| Concept | Rule | Example |
|---|---|---|
| `case_id` | lowercase snake case, unique, stable | `m57_patents`, `hickman_android_10` |
| `evidence_id` | lowercase snake case, unique across all cases | `m57_charlie`, `android_10` |
| `run_id` | `YYYYMMDD-HHMMSS-<agent>` (local time of the run start) | `20261006-161503-vision` |
| `finding_id` | 12 hex characters (random) | `2307f828637a` |
| `sample_id` | 20 hex characters, SHA-256 of task + messages + target (deterministic) | `84a98ef04c313deb586b` |
| **Sandbox** | the extracted tree of one evidence item: `data/cases/<case_id>/evidence/<evidence_id>/fs/` | |
| Sandbox-relative path | POSIX `/` separators, no leading `/`, no `..`. Used by findings, citations, bridge records | `data/media/0/DCIM/Camera/IMG_1.jpg` |
| `data/cases`-relative path | `<case_id>/evidence/<evidence_id>/fs/<sandbox path>`. Used by dataset tables | `hickman_android_10/evidence/android_10/fs/data/...` |
| Hash | SHA-256 of the whole file, 64 lowercase hex characters | |
| Acquisition hash | as published by the image source (often MD5 only) | `0377b3d4…` |
| Instants | ISO-8601 with offset, normalised to UTC (`+00:00` or `Z`) | `2020-02-09T18:23:53+00:00` |
| Stored times | never overwritten: keep `raw` + `epoch` next to the UTC value (§6.3) | |
| Text encoding | UTF-8 everywhere; binary snippets are lowercase hex with `snippet_encoding: "hex"` | |
| JSONL | one JSON object per line, UTF-8, no BOM, `\n` line ends | |

---

## 2. Storage layout

```
data/cases/
  catalog.sqlite                    index of the whole case database (§3.4)
  <case_id>/
    case.json                       case record (§3.1)
    context/brief/                  documents given to agents (.pdf/.html + extracted .txt)
    context/answer_key/             scoring material ONLY: never given to an agent or used as model input
    evidence/<evidence_id>/
      MANIFEST.json                 evidence record + one entry per extracted file (§3.2, §3.3)
      INSPECTION.json               integrity inspection (SQLite open checks, image decode checks)
      FILELIST.jsonl.gz             every directory entry incl. deleted, with MACB (§3.5) — disk images
      fs/                           the sandbox (read-only evidence)
      _case_context/                acquisition documents shipped with the image (may contain test passwords)
    workspace/
      memory.json                   investigation memory (context entities + links)
      runs/<run_id>/                run.json, trace.jsonl, <agent>__<evidence>_findings.json, *_verification_audit.json
      reports/                      Markdown reports
data/datasets/<YYYYMMDD>/           ML export (§5) + finetune/ (§7)
```

---

## 3. Case database

### 3.1 Case — `case.json`, table `cases`

| Field | Type | Description |
|---|---|---|
| `case_id` | str | identifier (§1) |
| `title` | str | human title |
| `year` | int | year the scenario/image was created |
| `kind` | enum | `scenario` (story with suspects), `ctf` (Magnet CTF), `reference_device` (Hickman phones, no story), `challenge` (DFRWS) |
| `summary` | str | neutral description, no solutions |
| `origin` | str | publisher (Digital Corpora, NIST CFReDS, …) |
| `tags` | list[str] | free tags (`windows`, `usb`, `exfiltration`, …) |
| `questions` | list[str] | investigative questions published with the case |
| `evidence` | list | evidence summaries (`evidence_id`, `kind`, `fetched`, `inspected`, `title`, …) |
| `split` *(export only)* | enum | `train` / `val` / `test`, assigned **per case** (`src/datasets/export.py:SPLITS`) |

### 3.2 Evidence — `MANIFEST.json` (top level), table `evidence`

| Field | Type | Description |
|---|---|---|
| `evidence_id` | str | identifier |
| `title` | str | device and date |
| `platform` | enum | §6.1 |
| `kind` | enum | `mobile_extraction`, `disk_image`, `cloud_export` (dataset export). In raw mobile manifests this field holds the archive format; the export moves it to `container` |
| `container` *(export)* | str | `zip`, `tar`, `tar.gz` (archives) or `ewf`, `raw` (disk images) |
| `spec` | object | the registry entry it was fetched with (URLs, size caps, group `light`/`heavy`/`extended`/`future`) |
| `image` | object | disk images only: `format`, `media_size`, `bytes_per_sector`, `headers` (acquisition metadata), `stored_hashes` (`MD5`, …), `bytes_read` |
| `filesystems` | list | per partition: `label`, `fstype`, `offset` (bytes), `partition`, `files`, `dirs`, `deleted` |
| `fetched_utc`, `fetch_seconds` | str, float | acquisition by this project |
| `member_count`, `skipped`, `notes` | | archive members seen, skipped reasons, free notes |
| `files` | list | one entry per extracted file (§3.3) |
| `acquisition_md5` *(export)* | str | `image.stored_hashes.MD5` |

### 3.3 Extracted file — `MANIFEST.json → files[]`, tables `files` / `images` / `documents`

| Field | Type | Description |
|---|---|---|
| `path` | str | manifest: relative to the evidence dir (`fs/...`); export: `data/cases`-relative |
| `source_member` | str | origin inside the archive or image (`fs:/Documents and Settings/...`, archive member name) |
| `category` | enum | §6.2 |
| `size` | int | bytes |
| `sha256` | str | hash of the extracted file |
| `member_mtime` / `mtime` | str | modification time recorded by the archive / file system (UTC) |
| `inode` | int | disk images: MFT record / inode number |
| `macb` | object | `{m, a, c, b}` UTC instants: modified, accessed, metadata-changed, born (created). Missing keys = not recorded by that file system |
| `deleted` | bool | recovered from unallocated directory entries |
| `duplicate_of` *(export)* | str \| null | first `data/cases`-relative path with the same SHA-256; deduplicate before training |
| `case_id`, `evidence_id`, `split`, `platform` *(export)* | | denormalised keys |

### 3.4 Catalogue — `data/cases/catalog.sqlite`

| Table | Columns |
|---|---|
| `cases` | `case_id` PK, `title`, `year`, `kind`, `summary`, `origin`, `tags` (JSON), `evidence_items`, `evidence_fetched`, `brief_docs`, `answer_key_docs` |
| `evidence` | `evidence_id` PK, `case_id` FK, `title`, `platform`, `kind`, `fetched`, `files`, `bytes`, `sqlite_ok`, `images`, `images_gps`, `deleted_recovered`, `fs_entries`, `acquisition_md5`, `sandbox` |
| `files` | `case_id`, `evidence_id`, `path`, `category`, `size`, `sha256`, `modified_utc`, `created_utc`, `deleted`, `source_member` (indexes on `(case_id, evidence_id, category)` and `sha256`) |
| `docs` | `case_id`, `role` (`brief` / `answer_key` / `acquisition`), `name`, `chars`, `text_path` |

### 3.5 File-system entry — `FILELIST.jsonl.gz`

| Field | Type | Description |
|---|---|---|
| `fs` | str | partition label |
| `path` | str | absolute path inside the file system (`/$MFT`, `/Users/...`) — **not** sandbox-relative: most entries were not extracted |
| `inode` | int | MFT record / inode |
| `size` | int | bytes |
| `is_dir` | bool | |
| `allocated` | bool | `false` = deleted entry |
| `m`, `a`, `c`, `b` | str | MACB UTC instants |

---

## 4. Analysis records

### 4.1 Citation — `src/verification/models.py:Citation`

| Field | Type | Description |
|---|---|---|
| `file_path` | str | sandbox-relative file the evidence is in |
| `locator_type` | enum | `byte_offset`, `line_number`, `sql_query`, `regex`, `exif_tag`, `registry`, `evtx_record` |
| `byte_offset` | int | 0-based offset (with `byte_offset`) |
| `line_number` | int | 1-based line (with `line_number`) |
| `sql_query`, `sql_params` | str, list | one read-only `SELECT`/`WITH`, positional `?` params; select rows by `rowid`/primary key |
| `regex` | str | Python regex; named group `v` is the value, else the whole match |
| `exif_tag` | str | `"<IFD>.<TagName>"`, IFD ∈ `0th`, `Exif`, `GPS`, `1st`, `Interop` (e.g. `GPS.GPSLatitude`) |
| `registry` | str | `"<key path>::<value name>"` inside a hive (value rendered by `render_registry_value`), or `"<key path>"` for the key's last-write time |
| `evtx_record` | int | EventRecordID inside an `.evtx` file; snippet is `evtx_summary(xml)` |
| `expected_hash` | str | SHA-256 of `file_path` when read; a mismatch fails the citation |
| `side_hashes` | dict | SQL citations only: SHA-256 of the `-wal` / `-journal` next to the database when read (`{"-wal": sha}`); the verifier requires the same set and hashes |
| `raw_text_snippet` | str | the value the verifier must reproduce |
| `snippet_encoding` | enum | `utf-8` (default) or `hex` |
| `match_mode` | enum | `exact` (snippet equals a candidate) or `contains` |

Exactly the field named by `locator_type` must be set.

### 4.2 Finding — `src/verification/models.py:Finding`

| Field | Type | Description |
|---|---|---|
| `finding_id` | str | §1 |
| `agent` | str | class name of the emitting agent (`MultimodalVisionAgent`) |
| `artifact_path` | str | sandbox-relative artifact analysed |
| `category` | str | §6.4 |
| `claim` | str | human statement. Model observations are prefixed `[MODEL OBSERVATION - not evidence]` |
| `value` | any | structured payload (coordinates, OCR text + bbox, EXIF value, …) |
| `citations` | list[Citation] | ground truth; empty ⇒ `uncorroborated` |
| `observed_text` | str | OCR/VLM text that must be corroborated by the cited snippet |
| `model`, `confidence` | str, float | model that produced `observed_text`; its own score (not a verification score) |
| `status` | enum | `pending` → `verified` / `rejected` / `uncorroborated` (§6.5) |
| `verification` | list[object] | one VerificationResult per citation + corroboration check |
| `context_links` | list[object] | `{relation, value, why, …}` — relation in §6.6. Analysis aids, never evidence |
| `evidence_id` | str | evidence item the artifact belongs to |
| `created_utc` | str | creation instant |

### 4.3 VerificationResult

| Field | Type | Description |
|---|---|---|
| `verified` | bool | citation reproduced |
| `locator_type`, `file_path` | | copied from the citation |
| `reason` | str | short explanation (`EXIF tag value matches`, `bytes differ at offset`, …) |
| `actual_hash`, `hash_match` | str, bool | recomputed SHA-256 and comparison |
| `observed_snippet` | str | what the file actually contains at the locator |
| `checked_utc` | str | check instant |

### 4.4 Run trace — `workspace/runs/<run_id>/trace.jsonl`

Every line: `seq` (int, 1..n), `ts` (UTC), `t` (seconds since run start), `kind`, plus kind-specific fields.
Kinds: `run_start`, `index_built`, `evidence_start`, `artifact_start`, `exif`, `ocr`, `index_lookup`, `vlm`,
`finding`, `verification`, `context_link`, `memory`, `artifact_end`, `error`, `evidence_end`, `run_end`.
`run.json` holds run metadata and `status` (`running` / `done` / `failed`).

### 4.5 Agent findings file — `<agent>__<evidence_id>_findings.json`

`agent`, `evidence_id`, `run_id`, `device`, `artifacts_analyzed`, `runtime_s`, `status_counts`
(`{verified, rejected, uncorroborated, pending}`), `resources` (GPU name, VRAM budget/peaks), `errors`,
`verified_findings`, `rejected_findings`, `model_observations_uncorroborated` (lists of Finding).

---

## 5. ML dataset export — `data/datasets/<YYYYMMDD>/` (`python main.py dataset`)

| Table | One row per | Key fields |
|---|---|---|
| `cases.jsonl` | case | §3.1 + `evidence_registered`, `evidence_fetched` |
| `evidence.jsonl` | fetched evidence item | §3.2: `platform`, `kind`, `container`, `files`, `filesystems`, `acquisition_md5` |
| `files.jsonl` | extracted file | §3.3 |
| `images.jsonl` | image file | subset of `files` (categories `user_image`, `screenshot`, `app_image` or image extension) |
| `documents.jsonl` | text-bearing file | subset of `files` (`user_document`, `email`, `removable_file`, `cloud_record`, `system_log`, `shell_history`, `app_config`) |
| `fs_timelines.jsonl` | FILELIST | `path` to the `.jsonl.gz`, `entries`, `deleted_entries` |
| `findings.jsonl` | finding | §4.2 + `case_id`, `split`, `run_id` |
| `context_brief.jsonl` | brief document | `doc`, `text` — allowed model input |
| `answer_keys.jsonl` | answer-key document | `doc`, `text` — **evaluation labels only** |
| `DATASHEET.md` | — | counts, splits, provenance and rules |

---

## 6. Controlled vocabularies

### 6.1 Platform — `src/schemas/common.py:Platform`
`android`, `ios`, `windows`, `linux`, `macos`, `removable` (USB / SD media), `takeout` (cloud export), `""` (unknown).

### 6.2 File category — `src/schemas/common.py:FileCategory`

| Category | Meaning |
|---|---|
| `app_database` | SQLite database inside an app's private data directory |
| `database` | other SQLite / database file (system or unknown owner) |
| `app_config` | app settings (Android `shared_prefs` XML, plists, INI) |
| `app_image` | image cached or received by an app (chat media, caches) |
| `user_image` | camera roll / user picture |
| `screenshot` | screen capture |
| `thumbnail_cache` | `thumbs.db`, `thumbcache_*.db`, `iconcache*.db` |
| `user_document` | office/PDF/text document in a user folder |
| `email` | mail store (PST/DBX/MBOX/EML) |
| `system_log` | text logs (Android `/data/system`, Linux `/var/log`, Windows `setupapi*.log`, scheduled tasks) |
| `event_log` | Windows EVT/EVTX |
| `registry` | Windows hives SYSTEM, SOFTWARE, DEFAULT, COMPONENTS, NTUSER.DAT, UsrClass.dat, Amcache.hve (never SAM / SECURITY) |
| `prefetch` | Windows Prefetch `.pf` |
| `shortcut` | LNK files and jump lists |
| `browser_history` | Internet Explorer stores in a user profile (`index.dat`, Cookies, Temporary Internet Files); Firefox/Chrome SQLite stores are `app_database` |
| `recycle_bin` | `$Recycle.Bin` / `RECYCLER` entries |
| `removable_file` | any file on removable media |
| `cloud_record` | Takeout JSON/HTML/CSV/MBOX/ICS/VCF/KML |
| `fs_metadata` | NTFS system files `$MFT`, `$LogFile`, `$Boot`, `$Secure`, `$AttrDef` |
| `backup_meta` | iTunes backup metadata (`Manifest.db`, `Manifest.plist`, `Info.plist`, `Status.plist`) |
| `shell_history` | `.bash_history` and similar |
| `network_capture` | packet capture (`.pcap`, `.pcapng`, `.cap`) |
| `memory_image` | physical memory dump (`.mem`, `.vmem`, `.lime`, `.dmp`, RAM `.raw`) |
| `crash_dump` | small Windows minidump |
| `video` / `audio` | media files (CCTV footage, voice notes) |
| `artifact` | small file of no recognised type kept from a logical tree (≤ 32 MB) |
| `disk_image_unparsed` | image segment kept byte-for-byte because no file system was recognised (wiped disks, NAND / chip-off dumps) |
| `compressed_evidence` | the distribution archive itself, kept when its members are too large to store unpacked (sparse memory dumps, unallocated-space blobs) |

### 6.3 Epoch kind — `src/schemas/common.py:EpochKind`

| Value | Definition | Typical source |
|---|---|---|
| `unix_s` / `unix_ms` / `unix_us` / `unix_ns` | since 1970-01-01 UTC | Android providers (`ms`), Linux, Firefox (`us`) |
| `webkit_us` | µs since 1601-01-01 UTC | Chrome / WebKit history |
| `windows_filetime` | 100 ns since 1601-01-01 UTC | registry, LNK, NTFS |
| `cocoa_s` | s since 2001-01-01 UTC | iOS Core Data, Safari, KnowledgeC |
| `cocoa_ns` | ns since 2001-01-01 UTC | iOS `sms.db` (iOS 11+) |
| `hfs_s` | s since 1904-01-01 UTC | HFS+ |
| `iso8601` | text with offset / `Z` | JSON APIs, Takeout |
| `exif` | `YYYY:MM:DD HH:MM:SS` local time | EXIF `DateTime*` (UTC only with `OffsetTime*`) |
| `local_text` | other wall-clock text without offset | logs |

A zero numeric value means "unset" (`utc` stays `null`).

### 6.4 Finding category

| Category | Agent | Evidentiary when |
|---|---|---|
| `exif.camera_make`, `exif.camera_model`, `exif.software`, `exif.description` | vision | verified |
| `exif.capture_time`, `exif.digitized_time`, `exif.modified_time`, `exif.capture_utc_offset` | vision | verified |
| `exif.gps` | vision | verified (all GPS tags cited) |
| `ocr.corroborated` | vision | verified (OCR text matched a DB cell / text line) |
| `ocr.uncorroborated`, `vlm_text.uncorroborated`, `vlm.description` | vision | never (model observation) |

| `sms.message` | apps | verified (Android `mmssms.db` sms, Google Messages `bugle_db`) |
| `imessage.message`, `whatsapp.message`, `skype.message`, `fbmessenger.message`, `line.message`, `kik.message`, `tiktok.message` | apps | verified (each quoted cell re-queried by rowid) |
| `call.log`, `contact.entry`, `contact.person`, `contact.value`, `calendar.event`, `account.entry` | apps | verified |
| `chrome.url`, `chrome.search`, `chrome.download`, `firefox.url`, `safari.visit` | apps | verified |
| `knowledgec.event`, `notes.note`, `photos.asset` | apps | verified |
| `fs.magic_mismatch` | fs | verified (signature bytes re-read at the offset) |
| `fs.sqlite_unlisted` | fs | verified (SQLite header bytes) |
| `fs.orphan_wal` | fs | verified (WAL magic) |
| `fs.hidden_path`, `fs.deleted_recovered` | fs | verified (first 16 bytes + hash: the file exists as described) |
| `fs.mail_store` | fs | verified (first 16 bytes): a PST / OST / DBX / mbox container and where it is |
| `sqlite.carved_text` | fs | verified (exact bytes of the carved run in a free page / freeblock / unallocated cell area). Remnants of deleted rows: their *meaning* is an analyst's call. Deduplicated per database; `value.score` (0..1) ranks them for triage only |
| `win.system_info`, `win.computer_name`, `win.timezone`, `win.shutdown`, `win.network_interface`, `win.network_card`, `win.last_logon` | win | verified (each registry value re-read from the hive) |
| `win.usb_device`, `win.portable_device`, `win.mountpoint` | win | verified (USBSTOR / USB / MountedDevices / Portable Devices / EMDMgmt / MountPoints2 values and key last-write times) |
| `win.installed_program`, `win.user_profile` | win | verified (Uninstall, ProfileList, SAM user names; never hashes) |
| `win.userassist`, `win.recent_doc`, `win.typed_url`, `win.search_term`, `win.run_mru` | win | verified (NTUSER values; ROT13 / UTF-16 decoding derived from the cited bytes) |
| `win.prefetch`, `win.lnk`, `win.recycle_bin`, `ie.url` | win | verified (exact bytes at fixed offsets: names, FILETIMEs, run counts, paths) |
| `win.event` | win | verified (`evtx_record`: the record's summary re-rendered from the log) |

Apps findings carry `value = {table, rowid, fields, timestamps}`; every timestamp is a `ForensicTimestamp`.
FS findings carry the manifest's `sha256`, `size`, `macb` and `original_path` in `value`.

Windows findings carry `value.logical_path` (path inside the original volume) and decoded fields (UTC ISO times).

New agents add categories as `<domain>.<item>` (e.g. `sms.message`, `chrome.visit`, `fs.magic_mismatch`) and
register them here.

### 6.5 Finding status
`pending` (not checked yet), `verified` (every citation re-executed and matched; the only evidentiary status),
`rejected` (a citation failed: hallucination or tampering signal), `uncorroborated` (model observation without
ground truth; kept, labelled, never evidence).

### 6.6 Context relation
`corroborates`, `keyword_match`, `outside_timeframe`, `not_in_context`, `new_lead`, `unverified_mention`
(definitions in `src/context/case_context.py`).

### 6.7 Split
`train`, `val`, `test` — per case. Test: `nist_data_leakage`, `magnet_ctf_2022`, `hickman_ios17`,
`hickman_android_14`. Val: `lone_wolf`, `magnet_ctf_2019`, `hickman_ios16`, `hickman_android_13`. Others: train.

### 6.8 Fine-tuning task — `src/schemas/training.py:TaskType`
`image_text_grounding`, `finding_generation`, `evidence_status_classification` (§7).

---

## 7. FDX interchange records — `src/schemas/`

### 7.1 ArtifactRef

| Field | Type | Description |
|---|---|---|
| `case_id`, `evidence_id` | str | locate the sandbox |
| `path` | str | sandbox-relative POSIX path (validated: no absolute paths, `\`, or `..`) |
| `sha256` | str \| null | required whenever the bytes are used (image parts of training samples) |

`cases_path` gives the `data/cases`-relative form; `ArtifactRef.from_cases_path()` parses it.

### 7.2 ForensicTimestamp

| Field | Type | Description |
|---|---|---|
| `raw` | int / float / str | value exactly as stored |
| `epoch` | enum | §6.3 |
| `utc` | str \| null | UTC instant when the source defines one |
| `local` | str \| null | wall-clock time when it has no offset |
| `utc_offset` | str \| null | `±HH:MM` when known |

### 7.3 BridgeRecord

All Citation locator fields (§4.1) plus:

| Field | Type | Description |
|---|---|---|
| `schema_version` | str | `fdx/1.0` |
| `source` | str | accessor: `sqlite.cell`, `text.line`, `exif.tag`, `bytes.byte_offset`, … |
| `sha256` | str | file hash at read time (becomes `Citation.expected_hash`) |
| `value` | any | typed value (int, float, str, parsed object) |
| `field` | str | logical field name (`sms.body`, `GPS.GPSLatitude`) |
| `timestamps` | dict[str, ForensicTimestamp] | named times extracted from the record (`sent`, `read`, …) |
| `attributes` | dict | locator context (`table`, `column`, `rowid`, `mime`, …) |

### 7.4 TrainingSample (JSONL line in `data/datasets/<date>/finetune/<task>.jsonl`)

| Field | Type | Description |
|---|---|---|
| `schema_version` | str | `fdx/1.0` |
| `sample_id` | str | deterministic content hash (§1) |
| `task` | enum | §6.8 |
| `split` | enum | inherited from the case |
| `messages` | list[Message] | `role` ∈ `system`/`user`/`assistant`; `content` = parts `{type: "text", text}` or `{type: "image", image: ArtifactRef}`; last message is the assistant target |
| `target` | any | structured form of the assistant answer |
| `label` | str \| null | classification label (`evidence_status_classification`) |
| `target_status` | enum | verification status of the content the target comes from (must be `verified` for generation tasks) |
| `weight` | float | sample weight (default 1.0) |
| `provenance` | object | `case_id`, `evidence_id`, `run_id`, `finding_ids`, `artifacts` (every evidence file used, with hash), `generator`, `dataset`, `created_utc` |
