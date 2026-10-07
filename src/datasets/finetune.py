"""Build FDX TrainingSamples (``src/schemas/training.py``) from a dataset export.

    python main.py finetune                       # latest data/datasets/<date>/ -> <date>/finetune/
    python main.py finetune --validate FILE.jsonl # check any file against the TrainingSample contract

Tasks (one JSONL file each):

  image_text_grounding.jsonl            evidence image + region -> text that a database cell corroborates
  finding_generation.jsonl              BridgeRecords of one artifact -> the verified Finding JSON (citations)
  evidence_status_classification.jsonl  finding + cited ground truth -> verified | rejected | uncorroborated

Only VERIFIED content becomes a generation target. Model observations (OCR/VLM text without
corroboration) appear only as inputs of the status classifier, labelled ``uncorroborated``.
Leakage guard: an image hash used in test is dropped from val/train, one used in val is dropped
from train; identical samples (same ``sample_id``) are written once.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable, Iterator

from pydantic import ValidationError

from src.config.settings import DATA_DIR
from src.schemas.bridge import BridgeRecord
from src.schemas.common import ArtifactRef
from src.schemas.training import ContentPart, Message, Provenance, TaskType, TrainingSample

SYSTEM = {
    TaskType.IMAGE_TEXT_GROUNDING: "You are a forensic OCR model. Transcribe exactly the text inside the given "
                                   "region of the evidence image. Output JSON {\"text\": ...}. Do not guess.",
    TaskType.FINDING_GENERATION: "You are a forensic analysis agent. From the bridge records of one artifact, emit "
                                 "findings as JSON. Every finding must cite the records it relies on, copying "
                                 "file_path, locator, expected_hash and raw_text_snippet exactly.",
    TaskType.EVIDENCE_STATUS: "You audit forensic findings. Given a claim, the model text it rests on and the cited "
                              "ground truth, answer with one word: verified, rejected or uncorroborated.",
}
MAX_CLAIM_CHARS = 2000
SPLIT_RANK = {"test": 0, "val": 1, "train": 2}


def latest_dataset() -> Path:
    dirs = sorted(p for p in (DATA_DIR / "datasets").glob("*") if (p / "findings.jsonl").exists())
    if not dirs:
        raise FileNotFoundError("no dataset export with findings.jsonl; run `python main.py dataset` first")
    return dirs[-1]


def _rows(path: Path) -> Iterator[dict]:
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.strip():
                yield json.loads(line)


def _bbox(poly) -> list[int] | None:
    """EasyOCR quadrilateral -> axis-aligned [x_min, y_min, x_max, y_max]."""
    try:
        xs, ys = [int(p[0]) for p in poly], [int(p[1]) for p in poly]
    except (TypeError, IndexError, ValueError):
        return None
    return [min(xs), min(ys), max(xs), max(ys)]


def _citation_json(c: dict) -> dict:
    loc = {k: c[k] for k in ("byte_offset", "line_number", "sql_query", "regex", "exif_tag") if c.get(k) is not None}
    out = {"file_path": c["file_path"], "locator_type": c["locator_type"], **loc,
           "expected_hash": c["expected_hash"], "raw_text_snippet": c["raw_text_snippet"]}
    if c.get("sql_params"):
        out["sql_params"] = c["sql_params"]
    if c.get("snippet_encoding", "utf-8") != "utf-8":
        out["snippet_encoding"] = c["snippet_encoding"]
    return out


class Builder:
    def __init__(self, dataset: Path):
        self.dataset = dataset
        self.findings = list(_rows(dataset / "findings.jsonl"))
        wanted = {f"{f['case_id']}/evidence/{f.get('evidence_id')}/fs/{f['artifact_path']}" for f in self.findings}
        self.sha = {r["path"]: r["sha256"] for r in _rows(dataset / "files.jsonl") if r["path"] in wanted}

    def _artifact(self, f: dict) -> ArtifactRef | None:
        sha = next((c["expected_hash"] for c in f.get("citations", []) if c["file_path"] == f["artifact_path"]), None)
        ref = ArtifactRef(case_id=f["case_id"], evidence_id=f["evidence_id"], path=f["artifact_path"])
        sha = sha or self.sha.get(ref.cases_path)
        return ref.model_copy(update={"sha256": sha}) if sha else None

    def _prov(self, task: TaskType, fs: list[dict], artifacts: list[ArtifactRef]) -> Provenance:
        f0 = fs[0]
        cited = {(c["file_path"], c["expected_hash"]) for f in fs for c in f.get("citations", [])}
        refs = {a.path: a for a in artifacts}
        for path, sha in sorted(cited):
            refs.setdefault(path, ArtifactRef(case_id=f0["case_id"], evidence_id=f0["evidence_id"], path=path, sha256=sha))
        return Provenance(case_id=f0["case_id"], evidence_id=f0.get("evidence_id"), run_id=f0.get("run_id"),
                          finding_ids=[f["finding_id"] for f in fs], artifacts=list(refs.values()),
                          generator=f"finetune.{task.value}", dataset=self.dataset.as_posix())

    # ------------------------------------------------------------------ tasks
    def image_text_grounding(self) -> Iterator[TrainingSample]:
        task = TaskType.IMAGE_TEXT_GROUNDING
        for f in self.findings:
            if f["category"] != "ocr.corroborated" or f["status"] != "verified" or not f.get("observed_text"):
                continue
            img, box = self._artifact(f), _bbox((f.get("value") or {}).get("bbox"))
            if img is None or box is None:
                continue
            target = {"text": f["observed_text"]}
            yield TrainingSample(
                task=task, split=f["split"], target=target, target_status="verified",
                messages=[Message.text("system", SYSTEM[task]),
                          Message(role="user", content=[ContentPart(type="image", image=img), ContentPart(
                              type="text", text=f"Region [x_min, y_min, x_max, y_max] = {box}. Transcribe it.")]),
                          Message.text("assistant", json.dumps(target, ensure_ascii=False))],
                provenance=self._prov(task, [f], [img]))

    def finding_generation(self) -> Iterator[TrainingSample]:
        task = TaskType.FINDING_GENERATION
        groups: dict[tuple, list[dict]] = defaultdict(list)
        for f in self.findings:
            if f["status"] == "verified" and f["category"].startswith("exif.") and f.get("citations"):
                groups[(f["case_id"], f["evidence_id"], f.get("run_id"), f["artifact_path"])].append(f)
        for (_, _, _, path), fs in groups.items():
            fs.sort(key=lambda f: f["category"])
            records: dict[str, dict] = {}
            for f in fs:
                for c in f["citations"]:
                    cj = _citation_json(c)
                    rec = BridgeRecord(source="exif.tag" if c["locator_type"] == "exif_tag" else f"bytes.{c['locator_type']}",
                                       sha256=cj.pop("expected_hash"), **cj)
                    records.setdefault(json.dumps(cj, sort_keys=True, default=str),
                                       rec.model_dump(mode="json", exclude_defaults=True))
            records = sorted(records.values(), key=lambda r: (r["locator_type"], str(r.get("exif_tag") or r.get("byte_offset"))))
            target = {"findings": [{"category": f["category"], "claim": f["claim"], "value": f.get("value"),
                                    "citations": [_citation_json(c) for c in f["citations"]]} for f in fs]}
            user = (f"Artifact: {path}\nBridge records (FDX BridgeRecord, JSON lines):\n"
                    + "\n".join(json.dumps(r, ensure_ascii=False) for r in records))
            yield TrainingSample(
                task=task, split=fs[0]["split"], target=target, target_status="verified",
                messages=[Message.text("system", SYSTEM[task]), Message.text("user", user),
                          Message.text("assistant", json.dumps(target, ensure_ascii=False, default=str))],
                provenance=self._prov(task, fs, []))

    def evidence_status(self) -> Iterator[TrainingSample]:
        task = TaskType.EVIDENCE_STATUS
        for f in self.findings:
            if f["status"] not in ("verified", "rejected", "uncorroborated"):
                continue
            truth = [f"- {c['locator_type']} on {c['file_path']}: {c['raw_text_snippet'][:500]!r}" for c in f.get("citations", [])]
            user = (f"Category: {f['category']}\nClaim: {f['claim'][:MAX_CLAIM_CHARS]}\n"
                    f"Model text: {(f.get('observed_text') or '(none)')[:MAX_CLAIM_CHARS]}\n"
                    f"Cited ground truth:\n" + ("\n".join(truth) if truth else "(no citations)"))
            img = self._artifact(f)
            yield TrainingSample(
                task=task, split=f["split"], label=f["status"], target={"status": f["status"]},
                target_status=f["status"],
                messages=[Message.text("system", SYSTEM[task]), Message.text("user", user),
                          Message.text("assistant", f["status"])],
                provenance=self._prov(task, [f], [img] if img else []))


def guard_leakage(samples: Iterable[TrainingSample]) -> tuple[list[TrainingSample], Counter]:
    """Dedupe by sample_id and keep each image hash in its most held-out split only."""
    samples = list(samples)
    home: dict[str, str] = {}
    for s in samples:
        for h in s.image_hashes():
            if h not in home or SPLIT_RANK[s.split.value] < SPLIT_RANK[home[h]]:
                home[h] = s.split.value
    kept, seen, dropped = [], set(), Counter()
    for s in samples:
        if s.sample_id in seen:
            dropped["duplicate"] += 1
        elif any(home[h] != s.split.value for h in s.image_hashes()):
            dropped["cross_split_image"] += 1
        else:
            seen.add(s.sample_id)
            kept.append(s)
    return kept, dropped


def export_finetune(dataset: Path | None = None) -> Path:
    dataset = dataset or latest_dataset()
    out = dataset / "finetune"
    out.mkdir(parents=True, exist_ok=True)
    b = Builder(dataset)
    rows, labels = [], Counter()
    for task, gen in ((TaskType.IMAGE_TEXT_GROUNDING, b.image_text_grounding),
                      (TaskType.FINDING_GENERATION, b.finding_generation),
                      (TaskType.EVIDENCE_STATUS, b.evidence_status)):
        kept, dropped = guard_leakage(gen())
        with open(out / f"{task.value}.jsonl", "w", encoding="utf-8") as fh:
            for s in kept:
                fh.write(s.model_dump_json(exclude_none=True) + "\n")
        sp = Counter(s.split.value for s in kept)
        rows.append(f"| {task.value} | {len(kept):,} | {sp['train']:,} | {sp['val']:,} | {sp['test']:,} | "
                    f"{dict(dropped) or '-'} |")
        if task is TaskType.EVIDENCE_STATUS:
            labels = Counter(s.label for s in kept)
    card = [f"# Fine-tuning set ({dataset.name})", "",
            "FDX `TrainingSample` JSONL (`src/schemas/training.py`, spec in `docs/DATA_INTERCHANGE_STANDARD.md`).",
            "Image parts are `ArtifactRef`s: load bytes from `data/cases/<case_id>/evidence/<evidence_id>/fs/<path>` "
            "and check the SHA-256 before use.", "",
            "| Task | Samples | train | val | test | Dropped |", "|---|---|---|---|---|---|", *rows, "",
            f"Status labels: {dict(labels)}. Classes are imbalanced (no REJECTED findings on real data yet): weight "
            "them or add synthetic rejections before training a classifier.", "",
            "Rules: only VERIFIED content is a generation target; answer keys are never used; splits are per case."]
    (out / "FINETUNE_CARD.md").write_text("\n".join(card) + "\n", encoding="utf-8")
    return out


def validate_jsonl(path: Path) -> tuple[int, list[str]]:
    """(valid line count, error messages) for a TrainingSample JSONL file."""
    ok, errors = 0, []
    with open(path, encoding="utf-8") as fh:
        for i, line in enumerate(fh, 1):
            if not line.strip():
                continue
            try:
                TrainingSample.model_validate_json(line)
                ok += 1
            except ValidationError as exc:
                errors.append(f"line {i}: {exc.errors()[0]['msg']}")
    return ok, errors
