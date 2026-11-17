"""Evidence integrity: audit every extracted file against its MANIFEST, re-acquire damaged files from the original
source, and write-protect the files the host OS is known to rewrite.

    python main.py cases audit  [--cases a,b] [--no-hash]     # -> data/outputs/integrity/audit_<ts>.{json,md}
    python main.py cases repair [--cases a,b] [--dry-run]     # audit, re-acquire, re-audit, protect

Why: the corpus lives in a synced Windows folder. Explorer rewrites ``Recent/desktop.ini`` in place (432 -> 324
bytes), SQLite rewrites ``-shm`` files when a database is opened in place, and antivirus / sync clients remove
some files (executables, names with a leading space). Any such change breaks the citations that point at the file,
so it must be detected and undone before results are trusted.

Repair never trusts the network blindly: the re-acquired bytes are written only when their SHA-256 equals the
hash recorded at acquisition time. Sources handled (range reads, nothing downloaded in full):

* mobile ZIP extractions (``archive_url`` + ``source_member``)
* disk images given by URLs (EWF / raw) or as STORED members of a ZIP: TSK reads the recorded inode
* bundle images whose segments are plain remote files (archive.org, Zenodo, Digital Corpora)

Anything else (images inside 7z / deflated ZIPs) is listed as ``needs_manual`` with the reason.
"""
from __future__ import annotations

import ctypes
import hashlib
import json
import os
import stat
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from src.bridge.forensic_bridge import long_path
from src.config import settings

# file classes the host rewrites in place: protected read-only after acquisition / repair
AT_RISK_SUFFIXES = ("desktop.ini", "-shm", "-wal", "-journal", ".ini")


def _lp(p: Path) -> str:
    return str(long_path(Path(os.path.abspath(p))))


def _sha256(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _cloud_only(path: str) -> bool:
    if os.name != "nt":
        return False
    fn = ctypes.windll.kernel32.GetFileAttributesW
    fn.restype, fn.argtypes = ctypes.c_uint32, [ctypes.c_wchar_p]
    a = fn(path)
    return a != 0xFFFFFFFF and bool(a & (0x400000 | 0x40000 | 0x1000))


@dataclass
class Problem:
    case_id: str
    evidence_id: str
    path: str
    kind: str                 # missing | size_mismatch | hash_mismatch | cloud_only | read_error
    detail: str = ""


@dataclass
class AuditResult:
    checked: int = 0
    ok: int = 0
    problems: list[Problem] = field(default_factory=list)
    per_case: dict = field(default_factory=lambda: defaultdict(Counter))

    def to_dict(self) -> dict:
        return {"checked": self.checked, "ok": self.ok, "problems": [p.__dict__ for p in self.problems],
                "by_kind": dict(Counter(p.kind for p in self.problems)),
                "per_case": {c: dict(v) for c, v in self.per_case.items()}}


def _check(job, do_hash: bool):
    case_id, eid, ev, f = job
    p = _lp(ev / f["path"])
    if not os.path.exists(p):
        return Problem(case_id, eid, f["path"], "missing")
    if _cloud_only(p):
        return Problem(case_id, eid, f["path"], "cloud_only", "OneDrive placeholder: content not on disk")
    try:
        size = os.path.getsize(p)
        if f.get("size") is not None and size != f["size"]:
            return Problem(case_id, eid, f["path"], "size_mismatch", f"{size} != {f['size']}")
        if do_hash and f.get("sha256") and _sha256(p) != f["sha256"]:
            return Problem(case_id, eid, f["path"], "hash_mismatch")
    except OSError as exc:
        return Problem(case_id, eid, f["path"], "read_error", str(exc)[:160])
    return None


def _jobs(cases):
    for case in cases:
        for eid in case.evidence_ids():
            ev = case.dir / "evidence" / eid
            mp = ev / "MANIFEST.json"
            if mp.exists():
                for f in json.loads(mp.read_text(encoding="utf-8")).get("files", []):
                    yield case.case_id, eid, ev, f


def audit(cases=None, do_hash: bool = True, workers: int = 8, progress: bool = True) -> AuditResult:
    from src.cases.registry import CASES

    cases = list(CASES.values()) if cases is None else cases
    jobs = list(_jobs(cases))
    res = AuditResult()
    with ThreadPoolExecutor(workers) as ex:
        for i, (job, prob) in enumerate(zip(jobs, ex.map(lambda j: _check(j, do_hash), jobs, chunksize=64)), 1):
            res.checked += 1
            if prob:
                res.problems.append(prob)
                res.per_case[job[0]][prob.kind] += 1
            else:
                res.ok += 1
                res.per_case[job[0]]["ok"] += 1
            if progress and i % 20000 == 0:
                print(f"[audit] {i:,}/{len(jobs):,}", flush=True)
    return res


def write_audit(res: AuditResult, title: str = "audit") -> Path:
    out = settings.OUTPUTS_DIR / "integrity"
    out.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    (out / f"{title}_{ts}.json").write_text(json.dumps(res.to_dict(), indent=1), encoding="utf-8")
    lines = [f"# Evidence integrity {title} ({ts})", "",
             f"Files checked: {res.checked:,} · intact: {res.ok:,} · problems: {len(res.problems)} "
             f"{dict(Counter(p.kind for p in res.problems))}", ""]
    if res.problems:
        lines += ["| case | evidence | file | problem | detail |", "|---|---|---|---|---|"]
        lines += [f"| {p.case_id} | {p.evidence_id} | `{p.path}` | {p.kind} | {p.detail} |" for p in res.problems]
    md = out / f"{title}_{ts}.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return md


# ---------------------------------------------------------------------------------------------- re-acquisition
class Unrepairable(Exception):
    pass


def _manifest(case_dir: Path, eid: str) -> dict:
    return json.loads((case_dir / "evidence" / eid / "MANIFEST.json").read_text(encoding="utf-8"))


def _zip_member_bytes(man: dict, f: dict) -> bytes:
    from src.utils import dataset_scraper as ds

    url = man.get("archive_url")
    if not url or not f.get("source_member"):
        raise Unrepairable("no archive URL / member recorded")
    top = ds.HttpSource(url)
    zf = ds._open_zip(top)
    try:
        info = zf.getinfo(f["source_member"])
    except KeyError as exc:
        raise Unrepairable("member not in the archive") from exc
    return ds._read_member(top, info)


def _remote(url: str):
    from src.utils import remote

    src = remote.RemoteSource(url)
    if not src.ranges:
        raise Unrepairable(f"host does not support range reads: {url}")
    return src


def _disk_sources(man: dict, f: dict) -> tuple[list, str, str]:
    """(byte sources, fmt, fs label) of the image holding ``f``."""
    from src.utils import dataset_scraper as ds
    from src.utils.disk_image import DiskEvidence

    member = f.get("source_member") or ""
    if man.get("kind") == "disk_image":
        spec = man["spec"]
        dev = DiskEvidence(**{k: v for k, v in spec.items() if k in DiskEvidence.__dataclass_fields__})
        label = member.split(":", 1)[0]
        if dev.urls:
            return [_remote(dev.full(u)) for u in dev.urls], dev.fmt, label
        top = ds.HttpSource(dev.full(dev.zip_url))
        zf = ds._open_zip(top)
        import re
        import zipfile

        infos = sorted((i for i in zf.infolist() if re.search(dev.member_regex, i.filename)), key=lambda i: i.filename)
        if not infos or any(i.compress_type != zipfile.ZIP_STORED for i in infos):
            raise Unrepairable("image segments are compressed inside the ZIP (full download needed)")
        return [ds.SubSource(top, ds._data_offset(top, i), i.file_size) for i in infos], dev.fmt, label
    if man.get("kind") == "bundle":
        from src.utils import remote
        from src.utils.bundle import group_files

        prefix, _, rest = member.partition("!")
        label = rest.split(":", 1)[0]
        img = next((i for i in man.get("images", []) if i.get("prefix") == prefix), None)
        if not img:
            raise Unrepairable(f"bundle image {prefix!r} not recorded")
        files = [(rf.path, rf) for spec in man["spec"].get("sources", []) for rf in remote.list_remote(spec)]
        for kind, key, members in group_files(files):
            if kind in ("ewf", "raw") and (key == img["name"] or key.endswith("/" + img["name"])
                                           or Path(key).name == Path(img["name"]).name):
                return [_remote(m[1].url) for m in members], "ewf" if kind == "ewf" else "raw", label
        raise Unrepairable("image is packed in an archive at the source (7z / zip): full download needed")
    raise Unrepairable(f"unsupported evidence kind {man.get('kind')!r}")


def _disk_file_bytes(man: dict, f: dict) -> bytes:
    from src.utils.disk_image import find_filesystems, open_image, read_file

    if f.get("inode") is None:
        raise Unrepairable("no inode recorded")
    sources, fmt, label = _disk_sources(man, f)
    image = open_image(sources, fmt)
    for fs in find_filesystems(image.tsk):
        if (fs.label or "fs") == (label or "fs") or (not fs.label and label in ("", "fs")):
            return read_file(fs, int(f["inode"]), int(f["size"]))
    raise Unrepairable(f"file system {label!r} not found in the image")


def reacquire(case_dir: Path, eid: str, rel: str) -> bytes:
    man = _manifest(case_dir, eid)
    f = next((x for x in man["files"] if x["path"] == rel), None)
    if f is None:
        raise Unrepairable("not in MANIFEST")
    data = _zip_member_bytes(man, f) if man.get("archive_url") else _disk_file_bytes(man, f)
    if hashlib.sha256(data).hexdigest() != f["sha256"]:
        raise Unrepairable("re-acquired bytes do not match the acquisition SHA-256")
    return data


def _write_evidence(path: Path, data: bytes, mtime: float | None) -> None:
    p = _lp(path)
    if os.path.exists(p):
        os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
    os.makedirs(os.path.dirname(p), exist_ok=True)
    tmp = p + ".repair"
    with open(tmp, "wb") as fh:
        fh.write(data)
    os.replace(tmp, p)
    if mtime:
        os.utime(p, (mtime, mtime))


def _mtime(f: dict) -> float | None:
    v = f.get("member_mtime")
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


def protect(cases=None) -> int:
    """Read-only attribute on at-risk evidence files (desktop.ini, SQLite side files): the shell / SQLite then
    cannot rewrite them. Content is not touched."""
    n = 0
    for _, _, ev, f in _jobs(cases if cases is not None else _all_cases()):
        if f["path"].lower().endswith(AT_RISK_SUFFIXES):
            p = _lp(ev / f["path"])
            try:
                mode = os.stat(p).st_mode
                if mode & stat.S_IWRITE:
                    os.chmod(p, stat.S_IREAD)
                    n += 1
            except OSError:
                pass
    return n


def _all_cases():
    from src.cases.registry import CASES

    return list(CASES.values())


def repair(cases=None, dry_run: bool = False, progress: bool = True) -> dict:
    from src.cases.registry import CASES

    cases = _all_cases() if cases is None else cases
    before = audit(cases, progress=progress)
    report_before = write_audit(before, "audit_before_repair")
    fixed, failed = [], []
    for p in before.problems:
        case = CASES[p.case_id]
        try:
            if p.kind == "cloud_only":
                raise Unrepairable("OneDrive placeholder: set the folder to 'Always keep on this device'")
            data = reacquire(case.dir, p.evidence_id, p.path)
            if not dry_run:
                f = next(x for x in _manifest(case.dir, p.evidence_id)["files"] if x["path"] == p.path)
                _write_evidence(case.dir / "evidence" / p.evidence_id / p.path, data, _mtime(f))
            fixed.append({**p.__dict__, "bytes": len(data)})
            if progress:
                print(f"[repair] restored {p.case_id}/{p.evidence_id}/{p.path} ({len(data)} B, SHA-256 match)",
                      flush=True)
        except Exception as exc:  # noqa: BLE001 - every failure is reported, none is fatal
            failed.append({**p.__dict__, "reason": f"{type(exc).__name__}: {exc}"[:240]})
            if progress:
                print(f"[repair] NOT restored {p.case_id}/{p.path}: {failed[-1]['reason']}", flush=True)
    protected = 0 if dry_run else protect(cases)
    after = audit(cases, progress=progress) if not dry_run else before
    report_after = write_audit(after, "audit_after_repair")
    summary = {"before": before.to_dict()["by_kind"], "after": after.to_dict()["by_kind"], "restored": fixed,
               "not_restored": failed, "protected_read_only": protected, "dry_run": dry_run,
               "reports": [str(report_before), str(report_after)]}
    out = settings.OUTPUTS_DIR / "integrity" / f"repair_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    summary["summary_path"] = str(out)
    return summary
