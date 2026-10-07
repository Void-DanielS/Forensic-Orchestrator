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
from src.cases.registry import CASES, CASES_DIR, Case, Doc, evidence_meta
from src.config.settings import RAW_IMAGES_DIR
from src.utils import dataset_scraper as ds
from src.utils import remote
from src.utils.bundle import RemoteEvidence, fetch_bundle
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
    return url if url.startswith(("http", "ia:")) else f"{ds.DC}/{quote(url)}"


def write_publisher_notes(case: Case) -> None:
    """The publisher's own description of the case is part of what an examiner is told."""
    if not case.publisher_notes and not case.answer_notes:
        return
    source = f"Source: {case.origin}" + (f" ({case.source_url})" if case.source_url else "")
    if case.publisher_notes:
        d = case.dir / "context" / "brief"
        d.mkdir(parents=True, exist_ok=True)
        (d / "publisher_description.md").write_text(
            f"# {case.title}\n\n{source}\n\nPublisher's description of the case (verbatim or lightly abridged):\n\n"
            f"{case.publisher_notes.strip()}\n", encoding="utf-8")
    if case.answer_notes:
        a = case.dir / "context" / "answer_key"
        a.mkdir(parents=True, exist_ok=True)
        (a / "publisher_solution_notes.md").write_text(
            f"# {case.title}: publisher's full story / solution notes\n\n{source}\n\n{case.answer_notes.strip()}\n",
            encoding="utf-8")


def fetch_docs(case: Case) -> list[str]:
    notes = []
    write_publisher_notes(case)
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
                    if doc.url.startswith("http") and not remote.gdrive_id(doc.url) and "web.archive.org" not in doc.url:
                        ds._http_download(_full(doc.url), dest)
                    else:
                        remote.stream_download(_full(doc.url), dest)
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
    src = remote.RemoteSource(_full(doc.url))
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
        meta = evidence_meta(e)
        d = case.dir / "evidence" / meta["evidence_id"]
        rows.append({**meta, "fetched": (d / "MANIFEST.json").exists(), "inspected": (d / "INSPECTION.json").exists()})
    return rows


def fetch_evidence(case: Case, groups: set[str], only: set[str] | None = None) -> list[str]:
    failures = []
    for e in case.evidence:
        meta = evidence_meta(e)
        eid, group = meta["evidence_id"], meta["group"]
        if (only and eid not in only) or (group not in groups and not only):
            continue
        out = case.dir / "evidence" / eid
        try:
            if not (out / "MANIFEST.json").exists():
                if isinstance(e, DiskEvidence):
                    fetch_disk(e, out)
                elif isinstance(e, RemoteEvidence):
                    fetch_bundle(e, out)
                else:
                    ds.fetch_source(ds.SOURCES[e])
            ctx = out / "_case_context"
            if isinstance(e, DiskEvidence) and ctx.is_dir():
                for f in ctx.iterdir():
                    dest = case.dir / "context" / "acquisition" / f"{eid}__{f.name}"
                    if not dest.exists():
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        shutil.copy2(f, dest)
            if isinstance(e, RemoteEvidence) and ctx.is_dir():
                # Documents shipped with the evidence keep the role the bundle assigned (brief / answer_key).
                for role_dir in ctx.iterdir():
                    if role_dir.is_dir() and role_dir.name in ROLES:
                        for f in role_dir.iterdir():
                            dest = case.dir / "context" / role_dir.name / f.name
                            if not dest.exists():
                                dest.parent.mkdir(parents=True, exist_ok=True)
                                shutil.copy2(f, dest)
                for role in ROLES:
                    d = case.dir / "context" / role
                    if d.is_dir():
                        for f in sorted(d.iterdir()):
                            if not f.name.endswith(".txt") or not f.with_name(f.name[:-4]).exists():
                                ensure_text(f)
            if not (out / "INSPECTION.json").exists():
                rep = ds.inspect_path(out, eid)
                print("[inspect]\n" + rep.summary(), flush=True)
                ds.assert_source_usable(rep)
        except Exception as exc:  # noqa: BLE001 - keep going with the other items
            failures.append(f"{case.case_id}/{eid}: {type(exc).__name__}: {exc}")
            print(f"[error] {failures[-1]}", flush=True)
    return failures


def _brief_chars(case: Case) -> int:
    d = case.dir / "context" / "brief"
    if not d.is_dir():
        return 0
    n = 0
    for f in d.iterdir():
        # Text is either the file itself (.txt/.md) or the extracted <doc>.txt beside a PDF/DOCX/HTML.
        if f.suffix.lower() in (".txt", ".md"):
            n += len(f.read_text(encoding="utf-8", errors="replace"))
    return n


def readiness(case: Case, evidence: list[dict] | None = None) -> dict:
    """Is the case usable for an investigation run: evidence on disk plus context to read?

    context_level: rich (>= 1500 characters of brief text, or >= 3 questions with a brief),
    basic (>= 300 characters), minimal (less). A case is *ready* with at least one fetched
    evidence item that extracted files and at least a basic brief."""
    evidence = evidence if evidence is not None else evidence_status(case)
    files = 0
    for e in evidence:
        man = case.dir / "evidence" / e["evidence_id"] / "MANIFEST.json"
        if man.exists():
            try:
                files += len(json.loads(man.read_text(encoding="utf-8")).get("files", []))
            except (OSError, ValueError):
                pass
    chars = _brief_chars(case)
    level = "rich" if chars >= 1500 or (len(case.questions) >= 3 and chars >= 300) else \
        "basic" if chars >= 300 else "minimal"
    ak = case.dir / "context" / "answer_key"
    fetched = sum(1 for e in evidence if e["fetched"])
    return {"evidence_fetched": fetched, "evidence_total": len(evidence), "extracted_files": files,
            "brief_chars": chars, "questions": len(case.questions), "context_level": level,
            "has_answer_key": ak.is_dir() and any(ak.iterdir()),
            "ready": fetched > 0 and files > 0 and level != "minimal"}


def write_case_json(case: Case) -> dict:
    docs = {}
    for role in ROLES:
        d = case.dir / "context" / role
        docs[role] = sorted(f.name for f in d.iterdir() if not f.name.endswith(".txt") or
                            not f.with_name(f.name[:-4]).exists()) if d.is_dir() else []
    evidence = evidence_status(case)
    data = {"case_id": case.case_id, "title": case.title, "year": case.year, "kind": case.kind,
            "summary": case.summary, "origin": case.origin, "tags": list(case.tags), "questions": list(case.questions),
            "source_url": case.source_url, "evidence": evidence, "context_docs": docs,
            "readiness": readiness(case, evidence), "updated_utc": datetime.now(timezone.utc).isoformat()}
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
                   tags TEXT, evidence_items INT, evidence_fetched INT, brief_docs INT, answer_key_docs INT,
                   questions INT, brief_chars INT, context_level TEXT, ready INT, source_url TEXT);
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
            meta = evidence_meta(e)
            eid = meta["evidence_id"]
            d = case.dir / "evidence" / eid
            man_p, ins_p = d / "MANIFEST.json", d / "INSPECTION.json"
            row = dict(evidence_id=eid, case_id=case.case_id, title=meta["title"], platform=meta["platform"],
                       kind=meta["kind"], fetched=man_p.exists(), files=0, bytes=0,
                       sqlite_ok=0, images=0, images_gps=0, deleted_recovered=0, fs_entries=0, acquisition_md5=None,
                       sandbox=(d / "fs").relative_to(CASES_DIR.parent.parent).as_posix())
            if man_p.exists():
                fetched += 1
                man = json.loads(man_p.read_text(encoding="utf-8"))
                files = man["files"]
                row.update(files=len(files), bytes=sum(f["size"] for f in files),
                           deleted_recovered=sum(1 for f in files if f.get("deleted")),
                           acquisition_md5=((man.get("image") or (man.get("images") or [{}])[0])
                                            .get("stored_hashes") or {}).get("MD5"),
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
        rd = readiness(case)
        con.execute("INSERT INTO cases VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (case.case_id, case.title, case.year, case.kind, case.summary, case.origin, ",".join(case.tags),
                     len(case.evidence), fetched, n_docs["brief"], n_docs["answer_key"], rd["questions"],
                     rd["brief_chars"], rd["context_level"], int(rd["ready"]), case.source_url))
    con.commit()
    con.close()
    tmp.replace(path)
    return path


def print_cases() -> None:
    for case in CASES.values():
        st = evidence_status(case)
        have = sum(r["fetched"] for r in st)
        rd = readiness(case, st)
        print(f"{case.case_id:28} {case.kind:16} {have}/{len(st)} evidence  ctx={rd['context_level']:7} "
              f"{'READY' if rd['ready'] else '-----'}  {case.title}")
        for r in st:
            print(f"    {'[x]' if r['fetched'] else '[ ]'} {r['evidence_id']:22} {r['platform']:9} {r['group']:8} "
                  f"~{r['transfer_gb']:5.1f} GB  {r['title']}")
