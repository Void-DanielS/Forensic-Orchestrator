# Forensic Orchestrator — ForensiAgent-Multimodal

A multi-agent framework for automated analysis of forensic evidence (Android/iOS extractions, Windows /
Linux / removable disk images) where **no finding is reported unless it is re-verified against the
physical evidence**.

## Status

Sprint 1 (foundation):

- Case-based dataset fetcher for public DFIR corpora (NIST CFReDS, Digital Corpora, Magnet CTF, Hickman phones)
- Forensic tooling bridge: sandbox, read-only SQLite, EXIF/GPS, E01/raw images through libewf + The Sleuth Kit
- `HeuristicVerificationEngine`: deterministic citation checks, no ML
- `MultimodalVisionAgent`: EXIF + EasyOCR + Qwen2.5-VL (4-bit) on a single 12 GB GPU
- Local web console, Markdown case reports, ML dataset export

Planned: `FileSystemAgent`, `AppArtifactAgent`, `TimelineAgent`, `OrchestratorAuditorAgent`. See
[`HANDOVER.md`](HANDOVER.md) for the Sprint 1 state and the Sprint 2/3 plan.

## Zero-hallucination contract

```
 agent (EXIF / OCR / VLM)                     HeuristicVerificationEngine (no ML)
 ───────────────────────                      ─────────────────────────────────────
 Finding                                      for each Citation:
   claim, value, observed_text   ─────────▶     1. path inside sandbox?
   citations: [                                 2. SHA-256 == expected_hash?
     {file_path, locator,                       3. re-execute locator on the file
      expected_hash,                               (bytes / line / regex / read-only SQL / EXIF)
      raw_text_snippet}, ...]                   4. snippet matches?
                                              + model text ≈ verified snippet (OCR/VLM)
                                              ─▶ VERIFIED | REJECTED | UNCORROBORATED
```

* **VERIFIED**: every citation re-executed and matched. This is the only status that counts as evidence.
* **REJECTED**: a citation failed (wrong value, altered file, bad query). Treated as a hallucination.
* **UNCORROBORATED**: a model observation with no ground truth (e.g. a VLM scene description).
  It is kept as clearly labelled analyst context and is never presented as evidence.

OCR and VLM text become evidence only when the case text index finds the same text in an
extraction database. The verifier then re-runs that SQL and checks that the model's text
is explained by the stored value (≥ 0.85 coverage).

## Case database

Evidence is organised **by case** in `data/cases/<case_id>/` (see `HANDOVER.md` §1b): each investigation holds all
of its evidence items (Windows / removable / Linux disk images and phone extractions), its **brief** (given to
agents), its **answer key** (scoring only, never given to agents) and the agents' workspace (memory, run traces,
reports). 18 public cases: M57-Patents, Lone Wolf, Narcos, NIST Data Leakage, Owl, DFRWS 2021, Magnet CTF
2019-2022 and 8 Hickman reference phones.

Disk images (E01 / split raw) are read with libewf + The Sleuth Kit; the extractor keeps registry hives, event logs,
Prefetch, LNK/jump lists, browser stores, e-mail, user documents and images, `$MFT`, and recovers deleted files, and
writes `FILELIST.jsonl.gz` with every directory entry and its MACB times. Credential stores are never extracted.

**Evidence data is not included in this repository.** `data/` (about 20 GB of case evidence, datasets and the
case catalog) is excluded by `.gitignore`; rebuild it locally with `main.py cases build`. All cases are public
training and reference images.

## Quick start

Requires Python 3.12. CUDA 12.8 wheels are needed for RTX 50xx GPUs.

```bash
python -m venv .venv
.venv/Scripts/pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
.venv/Scripts/python -m pip install -r requirements.txt

.venv/Scripts/python main.py cases list                                  # cases, evidence, fetch status
.venv/Scripts/python main.py cases build --cases all --groups light      # documents + evidence (resumable)
.venv/Scripts/python main.py cases build --cases narcos --groups heavy   # spooled images (large)
.venv/Scripts/python main.py investigate --case m57_patents --limit 5    # vision agent over a whole case
.venv/Scripts/python main.py apps --case hickman_android_10             # AppArtifactAgent: SMS, chats, calls, browsers
.venv/Scripts/python main.py fs --case m57_patents                      # FileSystemAgent: signatures, hidden/deleted, SQLite carving
.venv/Scripts/python main.py corpus                                     # apps + fs over every READY case
.venv/Scripts/python main.py gui                                         # http://127.0.0.1:8765
.venv/Scripts/python main.py dataset                                     # ML tables -> data/datasets/<date>/
.venv/Scripts/python main.py finetune                                    # FDX training samples -> <date>/finetune/
.venv/Scripts/python -m pytest                                           # unit tests (no GPU / dataset needed)
```

In the GUI pick a case, press **Start agent run** and watch each image: OCR boxes (green = text backed by a
database cell or text-file line), EXIF/GPS, the VLM answer, every proposed finding and its verification, and the
links the agent draws to the brief. Other tabs: investigation memory, findings, report, context documents
(answer key locked), evidence statistics.

## Data dictionary and interchange standard

* [`docs/DATA_DICTIONARY.md`](docs/DATA_DICTIONARY.md): every table, field and controlled vocabulary.
* [`docs/DATA_INTERCHANGE_STANDARD.md`](docs/DATA_INTERCHANGE_STANDARD.md) (FDX): how the forensic bridge returns
  citable `BridgeRecord`s, and the `TrainingSample` format for sending data to models for fine-tuning.
* [`docs/schemas/`](docs/schemas/): JSON Schemas (`python main.py schemas`).

## Layout

```
src/config/settings.py              device allocation (SINGLE_GPU_MODE), VRAM budgets, model ids
src/utils/dataset_scraper.py        mobile corpus fetcher (range/stream, nested archives) + integrity inspection
src/utils/disk_image.py             computer / removable images: EWF/raw -> TSK walk, selection, deleted recovery
src/cases/                          case registry, builder + catalog.sqlite, document text extraction
src/context/case_context.py         brief entities, finding<->context contrast, investigation memory
src/pipeline/investigate.py         run an agent over a case (trace, memory, report)
src/reports/case_report.py          Markdown report
src/gui/                            local web console
src/datasets/export.py              ML dataset export
src/datasets/finetune.py            FDX TrainingSample export, leakage guard, validator
src/schemas/                        FDX contracts: ArtifactRef, ForensicTimestamp, BridgeRecord, TrainingSample
src/bridge/forensic_bridge.py       Sandbox, read-only SQLite, EXIF/GPS, text, FS traversal
src/bridge/text_index.py            inverted index of DB text, used to propose OCR/VLM citations
src/verification/models.py          Citation / Finding / VerificationResult (pydantic)
src/verification/heuristic_verifier.py  HeuristicVerificationEngine
src/agents/base_agent.py            BaseAgent: analyze -> verify -> report
src/agents/vision_agent.py          MultimodalVisionAgent (EXIF + EasyOCR + Qwen2.5-VL 4-bit)
main.py                             CLI
HANDOVER.md                         sprint state and roadmap
tests/                              unit tests (verifier, bridge, scraper, schemas)
docs/                               data dictionary, interchange standard, JSON Schemas
```

## Project scope and sprints

ForensiAgent-Multimodal is a multi-agent orchestration framework for automated analysis of logical forensic images
(Android/iOS extractions, disk logical backups, system file trees) with a zero-hallucination guarantee.

### Hardware

* **Sprint 1 (current):** a single GPU, `cuda:0` RTX 5070 with ~12 GB VRAM. All Phase 1 components (VLM, OCR,
  reasoning/embeddings) fit on it through 4-bit/8-bit quantization (`bitsandbytes`) and careful memory management.
* **Target:** dual GPU, `cuda:0` RTX 5070 for text and orchestration, `cuda:1` RTX 4060 for VLM and vision.
* **Device selection** (`src/config/settings.py`): `SINGLE_GPU_MODE = True` runs everything on `cuda:0`. Switching
  to dual GPU only requires setting it to `False` (or `FORENSI_SINGLE_GPU=0`); see `HANDOVER.md` §6.

### Agent architecture (5 specialist agents)

| Agent | Responsibility | Sprint |
|---|---|---|
| `MultimodalVisionAgent` | image and screenshot artifacts: EXIF metadata, OCR, VLM description | **1 (delivered)** |
| `FileSystemAgent` | directory trees, MACB timestamps, hashes, hidden files | 2 |
| `AppArtifactAgent` | SQLite databases, JSON, XML, chat logs, browser history | 2 |
| `TimelineAgent` | temporal correlation across multi-source findings | 3 |
| `OrchestratorAuditorAgent` | task orchestration, strict heuristic audit, final Markdown report | 3 |

### Sprint 1 scope

**A. Case-based dataset fetcher** (`src/utils/dataset_scraper.py`, plus `src/utils/disk_image.py` and `src/cases/`)
downloads public practice forensic images tied to real DFIR case scenarios (NIST CFReDS, Digital Corpora, Magnet
Forensics CTFs). Each fetch is inspected to confirm that databases (`.db`, `.sqlite`), system logs and geotagged or
text-bearing images are present and uncorrupted. Evidence is stored per case in `data/cases/<case_id>/evidence/`
(the original plan named this folder `data/raw_images/`).

**B. Deterministic heuristic verification engine** (`src/verification/heuristic_verifier.py`). Every finding must be
backed by ground truth from the extraction. The engine:

* receives a citation `{file_path, byte_offset / line_number / sql_query, expected_hash, raw_text_snippet}`;
* opens and inspects the file inside the logical sandbox;
* re-executes the exact regex match, SQL query or byte-level comparison;
* returns `verified: True/False` with confirmation metadata before any finding is accepted.

**C. Forensic tooling bridge** (`src/bridge/forensic_bridge.py`), Python wrappers around lightweight forensic tools:

* `sqlite3` in read-only mode (URI `filename?mode=ro`, on working copies so evidence is never touched);
* `exiftool` / Pillow / piexif for image metadata and geolocation;
* native OS / `pytsk3` / `dfvfs` traversal for logical directory browsing.

**D. `MultimodalVisionAgent`** (`src/agents/vision_agent.py`), inheriting from `BaseAgent` and running on `cuda:0`
within the ~12 GB budget:

* **inputs:** visual artifacts (photos, messaging screenshots, suspicious images);
* **operations:** extract EXIF and camera metadata; OCR embedded text; run VLM inference (Qwen2.5-VL in
  4-bit/8-bit) to describe the visual context; emit structured JSON findings with the verification metadata the
  `HeuristicVerificationEngine` requires.

### Sprint 1 plan

1. **Environment:** `requirements.txt` (torch, transformers, accelerate, bitsandbytes, pillow, pydantic, easyocr;
   sqlite3 ships with Python) and `src/config/settings.py` for single-GPU execution with dual-GPU flags.
2. **Dataset acquisition and inspection:** run the fetcher and confirm databases and case images are present.
3. **Core engine:** the heuristic verifier and the forensic bridge.
4. **Vision agent and pipeline test:** run `main.py` end to end (dataset → vision agent → heuristic verifier) on
   `cuda:0` with quantization.
5. **Handover documentation** (`HANDOVER.md`): what was built and tested, the verified dataset structure, how to
   enable dual-GPU mode once the RTX 4060 is installed, and the plan for Sprint 2 (`AppArtifactAgent`,
   `FileSystemAgent`) and Sprint 3 (`TimelineAgent`, `OrchestratorAuditorAgent`).
