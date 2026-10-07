# ForensiAgent Data Interchange Standard (FDX) — version `fdx/1.0`

How data moves between **evidence → forensic bridge → agents → models**, and how it is packaged for
**fine-tuning**. Field definitions are in [`DATA_DICTIONARY.md`](DATA_DICTIONARY.md); the code is in
[`src/schemas/`](../src/schemas/); JSON Schemas are in [`schemas/`](schemas/).

```
 evidence file ──► bridge accessor ──► BridgeRecord ──► agent ──► Finding(citations) ──► Verifier ──► VERIFIED
 (sandbox, RO)     (src/bridge)        (value + locator      (parser / model)    (rec.to_citation())   │
                                        + hash + snippet)                                              ▼
                                                                       dataset export ──► TrainingSample JSONL
```

The invariant behind every rule below: **anything a model learns from or claims must be re-executable against
the physical evidence.** A value without a locator and hash is not data in this project; it is an opinion.

---

## 1. Scope

| Contract | Producer | Consumer | Model |
|---|---|---|---|
| `BridgeRecord` | bridge accessors, artifact parsers | agents, text index, training export | `src/schemas/bridge.py` |
| `Finding` / `Citation` | agents | verifier, reports, GUI, dataset export | `src/verification/models.py` |
| `TrainingSample` | `src/datasets/finetune.py` (or any exporter) | training / evaluation scripts | `src/schemas/training.py` |
| `ArtifactRef`, `ForensicTimestamp` | all of the above | all of the above | `src/schemas/common.py` |

---

## 2. Conventions (normative)

1. **Paths** inside records are sandbox-relative POSIX paths. Records that leave an evidence item's context
   (training samples, cross-case tables) carry an `ArtifactRef` (`case_id`, `evidence_id`, `path`, `sha256`).
2. **Hashes** are SHA-256 of the whole file, lowercase hex, computed through `Sandbox.hash()` (cached on path,
   size and mtime).
3. **Times** are `ForensicTimestamp`s built with `to_timestamp(raw, epoch, utc_offset=None)`. Keep `raw` and
   `epoch`; set `utc` only when the source defines an instant. Never assume a time zone for local times.
4. **Encoding**: UTF-8; binary values are lowercase hex with `snippet_encoding: "hex"`.
5. **IDs and vocabularies** come from the data dictionary (§1, §6). New values (a new file category, finding
   category or task) are added to the enum in `src/schemas/` *and* to the dictionary in the same change.
6. **JSONL**: one record per line, serialised with `model_dump_json(exclude_none=True)`.

---

## 3. Coding the forensic bridge

### 3.1 Rules for every accessor

| # | Rule | Why |
|---|---|---|
| B1 | Resolve every path through `Sandbox.resolve()`. Never build absolute evidence paths by hand. | blocks `..`, absolute paths and symlinks out of the sandbox |
| B2 | Never write to, or open in place, an evidence file. SQLite goes through `SQLiteBridge` (hash-keyed working copy, WAL merged on the copy, `mode=ro`, SELECT-only authorizer). | even read-only SQLite connections rewrite `-shm` and change the evidence hash |
| B3 | Any accessor that returns **evidence values to an agent** returns `BridgeRecord` (or a list), or `None` when the value cannot be cited. | the agent can then cite it with `record.to_citation()`; no uncitable values reach a finding |
| B4 | `raw_text_snippet` is rendered the way the verifier recomputes it (table below). Typed data goes in `value`. | a record that the verifier cannot reproduce is a REJECTED finding later |
| B5 | Times go in `timestamps` (or inside `value`) as `ForensicTimestamp`. | keeps raw values and epoch for court-defensible reporting |
| B6 | Deterministic: no network, no models, no randomness, stable ordering (sort directory listings, `ORDER BY rowid`). | reruns must produce identical records and hashes |
| B7 | Exploration helpers that return metadata only (`tables()`, `walk()`, `image_info()`) may return plain types, but their output is never quoted in a finding without going through a record accessor. | |
| B8 | Credential stores (SAM/SECURITY hives, keychains, `/etc/shadow`, password DBs) are never read. Values that look like secrets are not copied into `value`/`attributes` of records meant for reports or training. | |

### 3.2 Snippet rendering per locator

| `locator_type` | Locator fields | `raw_text_snippet` | Reference accessor |
|---|---|---|---|
| `sql_query` | `sql_query` = `SELECT "<col>" FROM "<table>" WHERE rowid = ?`, `sql_params` = `[rowid]` | `str(cell)` exactly as `sqlite3` returns it (ints as digits) | `bridge.sqlite.cell_record(db, table, column, rowid)` |
| `line_number` | `line_number` (1-based) | the full line without the line ending | `bridge.text.line_record(path, n)` |
| `exif_tag` | `exif_tag` = `"<IFD>.<TagName>"` | `render_exif_value(raw)` (rationals as `n/d`, tuples comma-joined) | `bridge.images.exif_record(path, tag)` |
| `byte_offset` | `byte_offset` | the bytes at the offset: UTF-8 text, or hex with `snippet_encoding: "hex"` | (Sprint 2: carving / BLOB accessors) |
| `regex` | `regex` (named group `v` = value) | the matched value; `match_mode: "contains"` only when the value is a substring | (Sprint 2: XML / plist / JSON accessors) |

BLOB cells: cite them with `byte_offset` + hex in the database file or with a `regex` over a decoded text
form produced by a bridge accessor; do not put `str(bytes)` in a snippet.

### 3.3 Writing an artifact parser (Sprint 2 pattern)

```python
from src.schemas import to_timestamp

def parse_sms(bridge, db: str) -> list[BridgeRecord]:
    out = []
    for (rowid,) in bridge.sqlite.query(db, "SELECT rowid FROM sms ORDER BY rowid"):
        body = bridge.sqlite.cell_record(db, "sms", "body", rowid, field="sms.body")
        date = bridge.sqlite.cell_record(db, "sms", "date", rowid, field="sms.date")
        if body is None:
            continue
        if date is not None:
            body.timestamps["sent"] = to_timestamp(date.value, "unix_ms")
        out += [r for r in (body, date) if r]
    return out

# in the agent
findings = [self.finding(db, "sms.message", f"SMS: {r.value!r}", value=r.value,
                         citations=[r.to_citation()]) for r in records if r.field == "sms.body"]
```

### 3.4 Required tests for a new accessor or parser

1. **Round trip**: every record it emits on a synthetic fixture becomes a finding that the
   `HeuristicVerificationEngine` marks `VERIFIED` (see `tests/test_schemas.py::test_every_bridge_record_verifies`).
2. **Forgery**: the same record with a changed snippet is `REJECTED`.
3. **Absence**: a missing row, line or tag returns `None`, not an empty record.
4. **Integrity**: the evidence file's SHA-256 is unchanged after the accessor runs.

---

## 4. Sending data to models

### 4.1 At inference (agents ↔ models)

* **Model input** is built from `BridgeRecord`s and evidence images referenced by `ArtifactRef` (the agent loads
  the bytes and checks the SHA-256). The brief (`context/brief`) may be included; `context/answer_key` never.
* **Model output** is never evidence. It enters the system as `Finding.observed_text` / `value` with
  `model` set. It becomes evidence only when the verifier corroborates it against a cited record
  (OCR/VLM text) or when the model returns citations copied from the records it was given, which the
  verifier then re-executes (`finding_generation` task, §4.3).
* Prompts that ask for structured output request JSON matching the `Finding` schema (`docs/schemas/finding.schema.json`).

### 4.2 Fine-tuning: the canonical file

One JSONL file per task in `data/datasets/<date>/finetune/`, one `TrainingSample` per line
(`python main.py finetune`). Each sample is a chat transcript:

```json
{"schema_version": "fdx/1.0", "sample_id": "84a98ef04c313deb586b", "task": "image_text_grounding", "split": "train",
 "messages": [
   {"role": "system", "content": [{"type": "text", "text": "You are a forensic OCR model. ..."}]},
   {"role": "user", "content": [
      {"type": "image", "image": {"case_id": "m57_patents", "evidence_id": "m57_charlie",
                                  "path": "Documents and Settings/Charlie/My Documents/.../20090196419-001.tif",
                                  "sha256": "73d93fdf..."}},
      {"type": "text", "text": "Region [x_min, y_min, x_max, y_max] = [262, 305, 734, 334]. Transcribe it."}]},
   {"role": "assistant", "content": [{"type": "text", "text": "{\"text\": \"QUANTUM CRYPTOGRAPHY APPARATUS\"}"}]}],
 "target": {"text": "QUANTUM CRYPTOGRAPHY APPARATUS"}, "target_status": "verified", "weight": 1.0,
 "provenance": {"case_id": "m57_patents", "evidence_id": "m57_charlie", "run_id": "20261006-161503-vision",
                "finding_ids": ["2307f828637a"], "artifacts": [...], "generator": "finetune.image_text_grounding",
                "dataset": "data/datasets/20261006"}}
```

### 4.3 Tasks

| Task | Input | Target | Built from |
|---|---|---|---|
| `image_text_grounding` | evidence image + region box | `{"text": ...}` | `ocr.corroborated` findings (text matched a DB cell) |
| `finding_generation` | the `BridgeRecord`s of one artifact (JSON lines) | `{"findings": [{category, claim, value, citations}]}` | verified findings, grouped per artifact |
| `evidence_status_classification` | claim + model text + cited ground truth | `verified` / `rejected` / `uncorroborated` | all findings |

New tasks are added to `TaskType`, get a builder in `src/datasets/finetune.py`, a row here and in the dictionary.

### 4.4 Rules (enforced by `TrainingSample` validation or the exporter)

| # | Rule | Enforced by |
|---|---|---|
| T1 | Generation targets come only from VERIFIED content (`target_status = "verified"`). Model observations appear only as classifier inputs. | schema |
| T2 | The last message is the assistant target; assistant messages contain no images. | schema |
| T3 | Images are referenced (`ArtifactRef` + SHA-256), never embedded. The loader verifies the hash. | schema + loader |
| T4 | All artifacts of a sample belong to `provenance.case_id`. | schema |
| T5 | Splits are per case. An image hash used in `test` is dropped from `val`/`train`; one used in `val` is dropped from `train`. Identical samples (same `sample_id`) are written once. | exporter (`guard_leakage`) |
| T6 | `answer_keys.jsonl` and `context/answer_key` are never inputs or targets. | exporter (does not read them) |
| T7 | No credentials or secrets in inputs or targets (rule B8). Credential stores are never extracted, so they cannot enter; there is **no automatic secret filter yet** for secrets inside ordinary artifacts (e.g. a password visible in a screenshot), so review samples before sharing a model trained on them. | extraction rules + manual review |
| T8 | Every sample keeps full provenance (case, evidence, run, finding ids, artifact hashes, dataset). | schema |

### 4.5 Converting to a training framework

The canonical file stays FDX; conversion happens in the training script. For Qwen2.5-VL / Hugging Face chat
templates:

```python
import hashlib, json
from pathlib import Path
from PIL import Image
from src.schemas import TrainingSample

CASES = Path("data/cases")

def to_hf(line: str):
    s = TrainingSample.model_validate_json(line)
    images, messages = [], []
    for m in s.messages:
        parts = []
        for p in m.content:
            if p.type == "image":
                path = CASES / p.image.cases_path
                assert hashlib.sha256(path.read_bytes()).hexdigest() == p.image.sha256, "evidence changed"
                images.append(Image.open(path).convert("RGB"))
                parts.append({"type": "image"})
            else:
                parts.append({"type": "text", "text": p.text})
        messages.append({"role": m.role, "content": parts})
    return {"messages": messages, "images": images, "split": s.split.value, "weight": s.weight}
```

### 4.6 Validation

```bash
python main.py finetune --validate data/datasets/<date>/finetune/<task>.jsonl   # any producer's file
python main.py schemas                                                          # refresh docs/schemas/*.json
python -m pytest tests/test_schemas.py
```

---

## 5. Versioning

`schema_version` is `fdx/MAJOR.MINOR`. Adding an optional field or an enum value is a **minor** change; renaming,
removing or changing the meaning of a field is a **major** change and needs a migration note in `HANDOVER.md`.
Regenerate `docs/schemas/` and update the dictionary in the same commit as the code change.

---

## 6. Checklist for a change that touches data

- [ ] New values added to the enums in `src/schemas/` and to `docs/DATA_DICTIONARY.md`
- [ ] Bridge accessors return `BridgeRecord` (B3) with verifier-compatible snippets (B4)
- [ ] Round-trip, forgery, absence and integrity tests (§3.4)
- [ ] Training samples validate (`main.py finetune --validate`) and no answer-key text is used (T6)
- [ ] `python main.py schemas` run and `docs/schemas/` committed
