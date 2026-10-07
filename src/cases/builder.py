"""Build the case database: migrate, fetch documents and evidence, inspect, catalogue.

    python main.py cases list
    python main.py cases build --cases all --groups light          # docs + range-read evidence
    python main.py cases build --cases m57_patents,narcos --groups light,heavy
    python main.py cases catalog                                    # data/cases/catalog.sqlite
"""
from __future__ import annotations

import gzip
import io
import json
import os
import re
import shutil
import sqlite3
import time
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import quote, unquote

import requests

from src.cases.docs import ensure_text
from src.cases.registry import CASES, CASES_DIR, Case, Doc
from src.config.settings import RAW_IMAGES_DIR
from src.utils import dataset_scraper as ds
from src.utils.disk_image import DiskEvidence, fetch_disk

ROLES = ("brief", "answer_key", "acquisition")


# --------------------------------------------------------------------------------------
# Migration of the Sprint 1 corpus (data/raw_images/<source>) into case folders
# --------------------------------------------------------------------------------------
def migrate_legacy() -> list[str]:
    """Move already-fetched mobile sources into their case. A same-volume rename: no data is rewritten."""
    moved = []
    for case in CASES.values():
        for e in case.evidence:
            if not isinstance(e, str):
                continue
            old, new = RAW_IMAGES_DIR / e, case.dir / "evidence" / e
            if old.is_dir() and not new.exists():
                new.parent.mkdir(parents=True, exist_ok=True)
                os.rename(old, new)
                moved.append(f"{e} -> {new.relative_to(CASES_DIR.parent)}")
            _adopt_source_context(case, new)
    return moved


def _adopt_source_context(case: Case, ev_dir: Path) -> None:
    """Mobile sources carry their own _case_context; expose it at case level by role."""
    ctx = ev_dir / "_case_context"
    if not ctx.is_dir():
        return
    for f in ctx.iterdir():
        if f.suffix.lower() == ".txt" and any(s.stem == f.stem and s.suffix.lower() != ".txt" for s in ctx.iterdir()):
            continue  # legacy pypdf text beside a pdf: regenerated at case level as <name>.pdf.txt
        # Hash-named PDFs are extraction reports (acquisition); everything else describes the device/case.
        role = "acquisition" if re.fullmatch(r"[0-9a-f]{40}\.pdf", f.name) else "brief"
        dest = case.dir / "context" / role / f.name
        if not dest.exists():
            dest.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(f, dest)


# --------------------------------------------------------------------------------------
# Documents
# --------------------------------------------------------------------------------------
def _full(url: str) -> str:
    return url if url.startswith("http") else f"{ds.DC}/{quote(url)}"


def fetch_docs(case: Case) -> list[str]:
    notes = []
    for doc in case.docs:
        dest_dir = case.dir / "context" / doc.role
        dest_dir.mkdir(parents=True, exist_ok=True)
        try:
            if doc.members:
                _fetch_zip_docs(doc, dest_dir)
            else:
                name = doc.rename or unquote(_full(doc.url).rsplit("/", 1)[-1])
                dest = dest_dir / ds._safe_part(name)
                if not dest.exists():
                    ds._http_download(_full(doc.url), dest)
                    print(f"  [doc:{doc.role}] {dest.name}")
        except RuntimeError as exc:
            if "encrypted" not in str(exc):
                raise
            # Instructor material distributed with a password on request: record it, never attempt to open it.
            note = dest_dir / ("RESTRICTED__" + ds._safe_part(unquote(doc.url.rsplit("/", 1)[-1])) + ".md")
            note.write_text(f"# Restricted document\n\n`{_full(doc.url)}` is password-protected (instructor "
                            "material). Request the password from Digital Corpora, extract it, and place the files "
                            f"in this folder (`context/{doc.role}/`). Then run `python main.py cases build --docs-only "
                            f"--cases {case.case_id}`.\n", encoding="utf-8")
            print(f"  [doc] restricted (password-protected): {doc.url}")
        except Exception as exc:  # noqa: BLE001 - a missing document must not stop the build
            notes.append(f"{doc.url}: {type(exc).__name__}: {exc}")
            print(f"  [doc] unavailable: {notes[-1]}")
    for role in ROLES:
        d = case.dir / "context" / role
        if d.is_dir():
            for f in sorted(d.iterdir()):
                if not f.name.endswith(".txt") or not f.with_name(f.name[:-4]).exists():
                    ensure_text(f)
    return notes


def _fetch_zip_docs(doc: Doc, dest_dir: Path) -> None:
    src = ds.HttpSource(_full(doc.url))
    if src.size < 300 << 20:
        # Small archives are downloaded whole; some use ZIP features that member range-reads mis-handle.
        zf = zipfile.ZipFile(io.BytesIO(src.get(0, src.size)))
        read = zf.read
    else:
        zf = ds._open_zip(src)
        read = lambda info: ds._read_member(src, info)  # noqa: E731
    for info in zf.infolist():
        if info.is_dir() or info.file_size > 50 << 20 or "__MACOSX" in info.filename:
            continue
        if not re.search(doc.members, info.filename, re.I) or ds.SENSITIVE.search(info.filename):
            continue
        rel = info.filename.split("/")
        name = ds._safe_part("__".join(rel[-2:]) if len(rel) > 2 else rel[-1])
        dest = dest_dir / name
        if not dest.exists():
            dest.write_bytes(read(info))
            print(f"  [doc:{doc.role}] {name}")


# --------------------------------------------------------------------------------------
# Evidence
# --------------------------------------------------------------------------------------
def evidence_status(case: Case) -> list[dict]:
    rows = []
    for e in case.evidence:
        eid = e if isinstance(e, str) else e.evidence_id
        d = case.dir / "evidence" / eid
        row = {"evidence_id": eid, "kind": "mobile" if isinstance(e, str) else "disk_image",
               "fetched": (d / "MANIFEST.json").exists(), "inspected": (d / "INSPECTION.json").exists()}
        if isinstance(e, str):
            s = ds.SOURCES[e]
            row.update(title=s.title, platform=s.platform, group=s.group, transfer_gb=s.transfer_gb)
        else:
            row.update(title=e.title, platform=e.platform, group=e.group, transfer_gb=e.transfer_gb,
                       device=e.device, note=e.note)
        rows.append(row)
    return rows


def fetch_evidence(case: Case, groups: set[str], only: set[str] | None = None) -> list[str]:
    failures = []
    for e in case.evidence:
        eid = e if isinstance(e, str) else e.evidence_id
        group = ds.SOURCES[e].group if isinstance(e, str) else e.group
        if (only and eid not in only) or (group not in groups and not only):
            continue
        out = case.dir / "evidence" / eid
        try:
            if not (out / "MANIFEST.json").exists():
                if isinstance(e, DiskEvidence):
                    fetch_disk(e, out)
                else:
                    ds.fetch_source(ds.SOURCES[e])
            if isinstance(e, DiskEvidence) and (out / "_case_context").is_dir():
                for f in (out / "_case_context").iterdir():
                    dest = case.dir / "context" / "acquisition" / f"{eid}__{f.name}"
                    if not dest.exists():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, dest)
            if not (out / "INSPECTION.json").exists():
                rep = ds.inspect_path(out, eid)
                print("[inspect]\n" + rep.summary(), flush=True)
                ds.assert_source_usable(rep)
        except Exception as exc:  # noqa: BLE001 - keep going with the other items
            failures.append(f"{case.case_id}/{eid}: {type(exc).__name__}: {exc}")
            print(f"[error] {failures[-1]}", flush=True)
    return failures


def write_case_json(case: Case) -> dict:
    docs = {}
    for role in ROLES:
        d = case.dir / "context" / role
        docs[role] = sorted(f.name for f in d.iterdir() if not f.name.endswith(".txt") or
                            not f.with_name(f.name[:-4]).exists()) if d.is_dir() else []
    data = {"case_id": case.case_id, "title": case.title, "year": case.year, "kind": case.kind,
            "summary": case.summary, "origin": case.origin, "tags": list(case.tags), "questions": list(case.questions),
            "evidence": evidence_status(case), "context_docs": docs,
            "updated_utc": datetime.now(timezone.utc).isoformat()}
    case.dir.mkdir(parents=True, exist_ok=True)
    (case.dir / "case.json").write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")
    return data


def build(case_ids: list[str], groups: set[str], only: set[str] | None = None, docs_only: bool = False) -> int:
    moved = migrate_legacy()
    for m in moved:
        print(f"[migrate] {m}")
    failures = []
    for cid in case_ids:
        case = CASES[cid]
        print(f"\n===== case {cid}: {case.title}", flush=True)
        failures += [f"{cid} doc: {n}" for n in fetch_docs(case)]
        if not docs_only:
            failures += fetch_evidence(case, groups, only)
        write_case_json(case)
    build_catalog()
    if failures:
        print("[failures]", *failures, sep="\n  ")
    return 1 if failures else 0


# --------------------------------------------------------------------------------------
# Catalogue (one SQLite file describing the whole case database)
# --------------------------------------------------------------------------------------
SCHEMA = """
CREATE TABLE cases(case_id TEXT PRIMARY KEY, title TEXT, year INT, kind TEXT, summary TEXT, origin TEXT,
                   tags TEXT, evidence_items INT, evidence_fetched INT, brief_docs INT, answer_key_docs INT);
CREATE TABLE evidence(evidence_id TEXT PRIMARY KEY, case_id TEXT REFERENCES cases, title TEXT, platform TEXT,
                      kind TEXT, fetched INT, files INT, bytes INT, sqlite_ok INT, images INT, images_gps INT,
                      deleted_recovered INT, fs_entries INT, acquisition_md5 TEXT, sandbox TEXT);
CREATE TABLE files(case_id TEXT, evidence_id TEXT, path TEXT, category TEXT, size INT, sha256 TEXT,
                   modified_utc TEXT, created_utc TEXT, deleted INT, source_member TEXT);
CREATE TABLE docs(case_id TEXT, role TEXT, name TEXT, chars INT, text_path TEXT);
CREATE INDEX files_case ON files(case_id, evidence_id, category);
CREATE INDEX files_sha ON files(sha256);
"""


def build_catalog(path: Path | None = None) -> Path:
    path = path or CASES_DIR / "catalog.sqlite"
    CASES_DIR.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.unlink(missing_ok=True)
    con = sqlite3.connect(tmp)
    con.executescript(SCHEMA)
    for case in CASES.values():
        fetched = 0
        for e in case.evidence:
            eid = e if isinstance(e, str) else e.evidence_id
            d = case.dir / "evidence" / eid
            man_p, ins_p = d / "MANIFEST.json", d / "INSPECTION.json"
            title = ds.SOURCES[e].title if isinstance(e, str) else e.title
            platform = ds.SOURCES[e].platform if isinstance(e, str) else e.platform
            row = dict(evidence_id=eid, case_id=case.case_id, title=title, platform=platform,
                       kind="mobile" if isinstance(e, str) else "disk_image", fetched=man_p.exists(), files=0, bytes=0,
                       sqlite_ok=0, images=0, images_gps=0, deleted_recovered=0, fs_entries=0, acquisition_md5=None,
                       sandbox=(d / "fs").relative_to(CASES_DIR.parent.parent).as_posix())
            if man_p.exists():
                fetched += 1
                man = json.loads(man_p.read_text(encoding="utf-8"))
                files = man["files"]
                row.update(files=len(files), bytes=sum(f["size"] for f in files),
                           deleted_recovered=sum(1 for f in files if f.get("deleted")),
                           acquisition_md5=(man.get("image") or {}).get("stored_hashes", {}).get("MD5"),
                           fs_entries=sum(fs.get("files", 0) + fs.get("dirs", 0) for fs in man.get("filesystems", [])))
                con.executemany("INSERT INTO files VALUES (?,?,?,?,?,?,?,?,?,?)", [
                    (case.case_id, eid, f["path"], f["category"], f["size"], f["sha256"], f.get("member_mtime"),
                     (f.get("macb") or {}).get("b"), int(bool(f.get("deleted"))), f.get("source_member")) for f in files])
            if ins_p.exists():
                ins = json.loads(ins_p.read_text(encoding="utf-8"))
                row.update(sqlite_ok=ins["sqlite_ok"], images=ins["images_ok"], images_gps=len(ins["images_with_gps"]))
            con.execute(f"INSERT INTO evidence({','.join(row)}) VALUES ({','.join('?' * len(row))})", list(row.values()))
        n_docs = {}
        for role in ROLES:
            d = case.dir / "context" / role
            names = [f for f in d.iterdir() if f.name.endswith(".txt")] if d.is_dir() else []
            n_docs[role] = len(names)
            for t in names:
                con.execute("INSERT INTO docs VALUES (?,?,?,?,?)", (case.case_id, role, t.name[:-4] if
                            t.with_name(t.name[:-4]).exists() else t.name, len(t.read_text(encoding="utf-8",
                            errors="replace")), t.relative_to(CASES_DIR).as_posix()))
        con.execute("INSERT INTO cases VALUES (?,?,?,?,?,?,?,?,?,?,?)",
                    (case.case_id, case.title, case.year, case.kind, case.summary, case.origin, ",".join(case.tags),
                     len(case.evidence), fetched, n_docs["brief"], n_docs["answer_key"]))
    con.commit()
    con.close()
    tmp.replace(path)
    return path


def print_cases() -> None:
    for case in CASES.values():
        st = evidence_status(case)
        have = sum(r["fetched"] for r in st)
        print(f"{case.case_id:24} {case.kind:16} {have}/{len(st)} evidence  {case.title}")
        for r in st:
            print(f"    {'[x]' if r['fetched'] else '[ ]'} {r['evidence_id']:22} {r['platform']:9} {r['group']:8} "
                  f"~{r['transfer_gb']:5.1f} GB  {r['title']}")
