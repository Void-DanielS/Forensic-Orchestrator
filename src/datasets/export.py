"""Export the case database as a machine-learning dataset (JSONL tables + datasheet).

    python main.py dataset export            # -> data/datasets/<YYYYMMDD>/

Tables (one JSON object per line; paths are relative to ``data/cases``):

  cases.jsonl          case metadata + split
  evidence.jsonl       evidence items (platform, kind, file system stats, acquisition hash)
  files.jsonl          every extracted file: category, size, sha256, MACB, deleted flag, duplicate group
  images.jsonl         subset of files that are images (for vision models)
  documents.jsonl      user documents / e-mails / text artefacts (for language models)
  fs_timelines.jsonl   pointer + counts for each FILELIST.jsonl.gz (all directory entries with MACB)
  findings.jsonl       every agent finding with its verification status (labels for verification /
                       corroboration models) and its context links
  context_brief.jsonl  brief documents' text per case (model input)
  answer_keys.jsonl    answer-key text per case (evaluation labels only; never model input)

Splits are assigned **per case** (all evidence of an investigation lands in one split) so that
no scenario leaks between train / validation / test.
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.cases.registry import CASES, CASES_DIR
from src.config.settings import DATA_DIR

IMAGE_CATS = {"user_image", "screenshot", "app_image"}
IMAGE_EXT = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".heic", ".webp", ".tif", ".tiff")
DOC_CATS = {"user_document", "email", "removable_file", "cloud_record", "system_log", "shell_history", "app_config"}


# Explicit, stratified by case kind and platform: each split holds a computer scenario with a documented answer
# key, a multi-device CTF and Android + iOS reference devices. Unlisted cases are training data.
SPLITS = {
    "nist_data_leakage": "test", "magnet_ctf_2022": "test", "hickman_ios17": "test", "hickman_android_14": "test",
    "lone_wolf": "val", "magnet_ctf_2019": "val", "hickman_ios16": "val", "hickman_android_13": "val",
}


def split_of(case_id: str) -> str:
    """Listed cases go to their split; every other case is training data."""
    return SPLITS.get(case_id, "train")


def _jl(path: Path, rows) -> int:
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False, default=str) + "\n")
            n += 1
    return n


def export(out: Path | None = None) -> Path:
    out = out or DATA_DIR / "datasets" / datetime.now().strftime("%Y%m%d")
    out.mkdir(parents=True, exist_ok=True)
    stats: dict = defaultdict(Counter)
    cases, evidence, files, fstl, findings, briefs, keys = [], [], [], [], [], [], []
    sha_first: dict[str, str] = {}
    for c in CASES.values():
        sp = split_of(c.case_id)
        fetched = []
        for eid in c.evidence_ids():
            d = c.dir / "evidence" / eid
            mp = d / "MANIFEST.json"
            if not mp.exists():
                continue
            fetched.append(eid)
            man = json.loads(mp.read_text(encoding="utf-8"))
            platform = man.get("platform") or man.get("spec", {}).get("platform") or ""
            img = man.get("image") or {}
            # Mobile manifests store the archive format (zip / tar / tar.gz) in "kind"; normalise to the evidence kind.
            kind = man.get("kind") or "mobile_extraction"
            container = None if kind == "disk_image" else kind
            if container:
                kind = "cloud_export" if platform == "takeout" else "mobile_extraction"
            evidence.append({"case_id": c.case_id, "evidence_id": eid, "split": sp, "platform": platform,
                             "kind": kind, "container": container or img.get("format"), "title": man.get("title"),
                             "files": len(man["files"]), "filesystems": man.get("filesystems"),
                             "acquisition_md5": (img.get("stored_hashes") or {}).get("MD5")})
            for f in man["files"]:
                sha = f["sha256"]
                dup = sha_first.setdefault(sha, f"{c.case_id}/{eid}/{f['path']}")
                rec = {"case_id": c.case_id, "evidence_id": eid, "split": sp, "platform": platform,
                       "path": f"{c.case_id}/evidence/{eid}/{f['path']}", "category": f["category"], "size": f["size"],
                       "sha256": sha, "mtime": f.get("member_mtime"), "macb": f.get("macb"),
                       "deleted": bool(f.get("deleted")), "source_member": f.get("source_member"),
                       "duplicate_of": None if dup.endswith(f["path"]) and dup.startswith(f"{c.case_id}/{eid}/") else dup}
                files.append(rec)
                stats["files_by_platform"][platform] += 1
                stats["files_by_category"][f["category"]] += 1
            if (d / "FILELIST.jsonl.gz").exists():
                fstl.append({"case_id": c.case_id, "evidence_id": eid, "split": sp,
                             "path": f"{c.case_id}/evidence/{eid}/FILELIST.jsonl.gz",
                             "entries": sum(fs.get("files", 0) + fs.get("dirs", 0) for fs in man.get("filesystems", [])),
                             "deleted_entries": sum(fs.get("deleted", 0) for fs in man.get("filesystems", []))})
        for run in sorted((c.dir / "workspace" / "runs").glob("*/*_findings.json")) if (c.dir / "workspace" / "runs").is_dir() else []:
            d = json.loads(run.read_text(encoding="utf-8"))
            for key in ("verified_findings", "rejected_findings", "model_observations_uncorroborated"):
                for f in d.get(key, []):
                    findings.append({"case_id": c.case_id, "split": sp, "run_id": run.parent.name, **f})
        for role, sink in (("brief", briefs), ("answer_key", keys)):
            rd = c.dir / "context" / role
            if rd.is_dir():
                for t in sorted(rd.glob("*.txt")):
                    sink.append({"case_id": c.case_id, "split": sp, "doc": t.name.removesuffix(".txt"),
                                 "text": t.read_text(encoding="utf-8", errors="replace")})
        cases.append({"case_id": c.case_id, "title": c.title, "kind": c.kind, "year": c.year, "split": sp,
                      "summary": c.summary, "tags": list(c.tags), "evidence_registered": len(c.evidence),
                      "evidence_fetched": fetched})
    images = [f for f in files if f["category"] in IMAGE_CATS or f["path"].lower().endswith(IMAGE_EXT)]
    docs = [f for f in files if f["category"] in DOC_CATS and not f["path"].lower().endswith(IMAGE_EXT)]
    counts = {name: _jl(out / f"{name}.jsonl", rows) for name, rows in (
        ("cases", cases), ("evidence", evidence), ("files", files), ("images", images), ("documents", docs),
        ("fs_timelines", fstl), ("findings", findings), ("context_brief", briefs), ("answer_keys", keys))}
    split_counts = {t: dict(Counter(r["split"] for r in rows)) for t, rows in
                    (("cases", cases), ("evidence", evidence), ("images", images), ("documents", docs))}
    fs_entries = sum(r["entries"] for r in fstl)
    sheet = [f"# ForensiAgent case dataset ({out.name})", "",
             f"Generated {datetime.now(timezone.utc).isoformat()} from `data/cases` "
             f"({len(cases)} cases, {len(evidence)} evidence items fetched).", "",
             "| Table | Rows |", "|---|---|", *[f"| {k} | {v:,} |" for k, v in counts.items()], "",
             f"File-system timeline entries (all directory entries with MACB, incl. deleted): **{fs_entries:,}**", "",
             "## Splits (by case)", "", "```", json.dumps(split_counts, indent=1), "```", "",
             "## Files by platform", "", *[f"- {k or 'unknown'}: {v:,}" for k, v in stats["files_by_platform"].most_common()], "",
             "## Files by category", "", *[f"- {k}: {v:,}" for k, v in stats["files_by_category"].most_common()], "",
             "## Provenance and rules", "",
             "- Sources: Digital Corpora (M57-Patents, Lone Wolf, Narcos, Owl, Hickman reference images, Magnet "
             "Forensics CTFs, DFRWS 2021), NIST CFReDS (Data Leakage). Public research/teaching images; cite them.",
             "- `answer_keys.jsonl` is evaluation-only. Never feed it to a model being evaluated on the same case.",
             "- `duplicate_of` marks byte-identical files seen earlier (OS sample pictures, shared media): "
             "deduplicate before training to avoid leakage across splits.",
             "- Credential stores (SAM/SECURITY hives, keychains, /etc/shadow, password files) were never extracted.",
             "- Paths are relative to `data/cases`; bytes stay there (this export holds metadata, labels and text)."]
    (out / "DATASHEET.md").write_text("\n".join(sheet) + "\n", encoding="utf-8")
    return out
