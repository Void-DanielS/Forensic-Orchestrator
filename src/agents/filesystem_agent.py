"""FileSystemAgent (Sprint 2): file-level anomalies and deleted-data recovery, all byte-cited.

Per extracted file (from ``MANIFEST.json``, which carries the hashes and MACB times) it reports:

* ``fs.magic_mismatch``     the content signature contradicts the extension (renamed / disguised file)
* ``fs.sqlite_unlisted``    a SQLite database without a database-like extension
* ``fs.orphan_wal``         a SQLite ``-wal`` journal whose main database is absent (uncommitted data)
* ``fs.hidden_path``        a user file inside a dot-directory / dot-file (outside app-private trees)
* ``fs.deleted_recovered``  a file recovered from an unallocated directory entry (disk images)
* ``sqlite.carved_text``    readable text in SQLite free pages, freeblocks and unallocated cell space:
                            remnants of deleted rows

Every finding cites bytes in the file (``byte_offset`` + hex snippet): the signature for type
findings, the first bytes for presence findings, the exact carved run for carving. The verifier
re-reads those bytes and re-hashes the file, so nothing here rests on the agent's word.
MACB times, size and the file's SHA-256 come from the manifest and go into ``value``.
"""
from __future__ import annotations

import json
import re
import struct
from pathlib import Path, PurePosixPath
from typing import Sequence

from src.agents.base_agent import BaseAgent
from src.verification.models import Finding

# (type, signature, offset, extensions that legitimately carry it)
SIGNATURES = [
    ("sqlite", b"SQLite format 3\x00", 0, (".db", ".sqlite", ".sqlite3", ".sqlitedb", ".storedata", ".db3", ".sql")),
    ("jpeg", b"\xff\xd8\xff", 0, (".jpg", ".jpeg", ".jpe", ".jfif", ".thm")),
    ("png", b"\x89PNG\r\n\x1a\n", 0, (".png",)),
    ("gif", b"GIF8", 0, (".gif",)),
    ("pdf", b"%PDF-", 0, (".pdf", ".ai")),
    ("zip", b"PK\x03\x04", 0, (".zip", ".docx", ".xlsx", ".pptx", ".odt", ".ods", ".odp", ".jar", ".apk", ".epub",
                               ".kmz", ".xpi", ".ipa", ".vsdx", ".aar", ".crx", ".whl", ".nupkg")),
    ("ole2", b"\xd0\xcf\x11\xe0\xa1\xb1\x1a\xe1", 0, (".doc", ".xls", ".ppt", ".msg", ".msi", ".db", ".pub", ".vsd",
                                                     ".mdi", ".dot", ".xlt", ".pps")),
    ("pe_executable", b"MZ", 0, (".exe", ".dll", ".sys", ".scr", ".com", ".ocx", ".cpl", ".drv", ".efi", ".mui",
                                ".ax", ".acm", ".tlb", ".fon", ".mun", ".tsp", ".ime", ".pyd", ".winmd", ".rll")),
    ("elf_executable", b"\x7fELF", 0, (".so", ".bin", ".elf", ".o", ".ko", "")),
    ("7z", b"7z\xbc\xaf\x27\x1c", 0, (".7z",)),
    ("rar", b"Rar!\x1a\x07", 0, (".rar",)),
    ("gzip", b"\x1f\x8b\x08", 0, (".gz", ".tgz", ".svgz")),
    ("bplist", b"bplist00", 0, (".plist", ".bplist", ".webarchive", ".savedstate", ".nib")),
    ("mp4_family", b"ftyp", 4, (".mp4", ".m4a", ".m4v", ".mov", ".3gp", ".3g2", ".heic", ".heif", ".avif", ".m4r",
                                ".qt", ".f4v", ".m4b", ".mj2")),
    ("riff", b"RIFF", 0, (".wav", ".avi", ".webp", ".ani", ".rmi", ".cdr")),
    ("evtx", b"ElfFile\x00", 0, (".evtx",)),
    ("registry_hive", b"regf", 0, (".dat", ".hve", ".log1", ".log2", ".sav", "")),
    ("lnk", b"L\x00\x00\x00\x01\x14\x02\x00", 0, (".lnk",)),
]
TEXTUAL_EXTS = (".txt", ".log", ".csv", ".xml", ".json", ".html", ".htm", ".ini", ".cfg", ".conf", ".md", ".js")
DB_LIKE = re.compile(r"\.(db|sqlite\d?|sqlitedb|storedata|db3|sql|dat)$|(^|/)(history|cookies|web data|login data|"
                     r"top sites|favicons|shortcuts|network action predictor|visited links|threads_db2|naver_line|"
                     r"db_im_xx|[^/]*_db\d?)$", re.I)
APP_PRIVATE = re.compile(r"(^|/)(data/data|data/user(_de)?/\d+|private/var/mobile/Containers|appdata|library/"
                         r"(caches|application support|containers)|\.config|\.cache|\.local|\.mozilla|"
                         r"\.thumbnails|\.android_secure|\.git)(/|$)", re.I)

# Credential stores are never carved (FDX rule B8): saved passwords, card numbers, keychains, account tokens.
NO_CARVE = re.compile(r"(^|/)(login data( for account)?|web data|keychain[^/]*|key4\.db|logins\.json|signons\.sqlite|"
                      r"cookies|accounts(_ce|_de)?\.db|[^/]*password[^/]*|[^/]*passwd[^/]*|[^/]*credential[^/]*)$", re.I)
SCHEMA_TEXT = re.compile(r"CREATE\s+(TABLE|INDEX|UNIQUE|VIEW|TRIGGER|VIRTUAL)|sqlite_(autoindex|sequence|stat)|"
                         r"PRIMARY KEY|INTEGER NOT NULL|COLLATE", re.I)
MIN_RUN = 12            # minimum printable characters for a carved text run
MAX_CARVED_PER_DB = 300


def _sig(head: bytes) -> str | None:
    for kind, magic, off, _ in SIGNATURES:
        if head[off:off + len(magic)] == magic:
            return kind
    return None


class FileSystemAgent(BaseAgent):
    name = "FileSystemAgent"
    role = "text"

    def __init__(self, bridge, verifier=None, manifest: dict | None = None, carve: bool = True, **kw):
        super().__init__(bridge, verifier, **kw)
        self.carve = carve
        self.meta: dict[str, dict] = {}
        self.all_paths: set[str] = set()
        if manifest:
            self.load_manifest(manifest)

    def load_manifest(self, manifest: dict) -> list[str]:
        for f in manifest.get("files", []):
            rel = re.sub(r"^fs/", "", f["path"])
            self.meta[rel] = f
        self.all_paths = {p.lower() for p in self.meta}
        return sorted(self.meta)

    # ------------------------------------------------------------------ helpers
    def _head_cite(self, rel: str, n: int, offset: int = 0):
        with open(self.sandbox.resolve(rel), "rb") as fh:
            fh.seek(offset)
            data = fh.read(n)
        if not data:
            return None
        return self.cite(rel, "byte_offset", data.hex(), byte_offset=offset, snippet_encoding="hex")

    def _base_value(self, rel: str, **extra) -> dict:
        m = self.meta.get(rel, {})
        return {"sha256": m.get("sha256"), "size": m.get("size"), "category": m.get("category"),
                "macb": m.get("macb"), "member_mtime": m.get("member_mtime"),
                "original_path": m.get("original_path") or m.get("source_member"), **extra}

    # ------------------------------------------------------------------ analysis
    def analyze_artifact(self, rel: str) -> list[Finding]:
        path = self.sandbox.resolve(rel)
        size = path.stat().st_size
        if size == 0:
            return []
        with open(path, "rb") as fh:
            head = fh.read(64)
        name = PurePosixPath(rel).name
        ext = PurePosixPath(name).suffix.lower()
        kind = _sig(head)
        out: list[Finding] = []
        m = self.meta.get(rel, {})

        if kind:
            allowed = next(s[3] for s in SIGNATURES if s[0] == kind)
            magic = next((s[1], s[2]) for s in SIGNATURES if s[0] == kind)
            disguised = ext and ext not in allowed and not (kind == "zip" and ext in (".zip",)) and \
                not name.endswith(("-wal", "-shm", "-journal")) and len(ext) <= 6
            if kind == "sqlite" and not DB_LIKE.search(rel) and ext not in (".db", ".sqlite", ".sqlite3"):
                out.append(self.finding(rel, "fs.sqlite_unlisted",
                                        f"SQLite database without a database extension: {rel}",
                                        value=self._base_value(rel, detected_type=kind, extension=ext or None),
                                        citations=[self._head_cite(rel, len(magic[0]), magic[1])]))
            elif disguised and ext not in TEXTUAL_EXTS + (".tmp", ".bak", ".old", ".cache", ".bin", ".dat", ".data",
                                                             ".blob", ".chunk", ".part", ".crdownload", ".0", ".1"):
                out.append(self.finding(rel, "fs.magic_mismatch",
                                        f"Content is {kind} but the extension is '{ext}': {rel}",
                                        value=self._base_value(rel, detected_type=kind, extension=ext),
                                        citations=[self._head_cite(rel, len(magic[0]), magic[1])]))

        if name.lower().endswith("-wal") and head[:4] in (b"\x37\x7f\x06\x82", b"\x37\x7f\x06\x83"):
            main = rel[:-4].lower()
            if main not in self.all_paths and not self.sandbox.resolve(rel[:-4]).exists():
                out.append(self.finding(rel, "fs.orphan_wal",
                                        f"SQLite write-ahead log without its database (uncheckpointed data): {rel}",
                                        value=self._base_value(rel, wal_bytes=size), citations=[self._head_cite(rel, 4)]))

        parts = PurePosixPath(rel).parts
        if any(p.startswith(".") and p not in (".", "..") for p in parts) and not APP_PRIVATE.search(rel) \
                and m.get("category") in (None, "user_document", "user_image", "removable_file", "screenshot",
                                          "app_database", "database", "email", "artifact"):
            out.append(self.finding(rel, "fs.hidden_path", f"File in a hidden (dot) path: {rel}",
                                    value=self._base_value(rel), citations=[self._head_cite(rel, 16)]))

        if m.get("deleted"):
            out.append(self.finding(rel, "fs.deleted_recovered",
                                    f"Recovered from a deleted directory entry: {m.get('original_path') or rel}",
                                    value=self._base_value(rel, reliability=m.get("reliability"),
                                                           recovered_bytes=m.get("recovered_bytes")),
                                    citations=[self._head_cite(rel, 16)]))

        if self.carve and kind == "sqlite" and size <= 512 << 20 and not NO_CARVE.search(rel):
            out += self._carve_sqlite(rel, path)
        return [f for f in out if all(f.citations)]

    # ------------------------------------------------------------------ SQLite carving
    def _carve_sqlite(self, rel: str, path: Path) -> list[Finding]:
        data = path.read_bytes()
        if len(data) < 100:
            return []
        page_size = struct.unpack(">H", data[16:18])[0]
        page_size = 65536 if page_size == 1 else page_size
        if page_size < 512 or page_size & (page_size - 1):
            return []
        n_pages = len(data) // page_size
        regions: list[tuple[int, int, str, int]] = []      # (start, end, kind, page_no)
        # 1) freelist: trunk pages list leaf pages; both hold deleted content
        trunk, seen = struct.unpack(">I", data[32:36])[0], set()
        while 0 < trunk <= n_pages and trunk not in seen and len(seen) < 100000:
            seen.add(trunk)
            off = (trunk - 1) * page_size
            nxt, count = struct.unpack(">II", data[off:off + 8])
            regions.append((off + 8 + 4 * count, off + page_size, "freelist_trunk", trunk))
            for i in range(min(count, (page_size - 8) // 4)):
                leaf = struct.unpack(">I", data[off + 8 + 4 * i: off + 12 + 4 * i])[0]
                if 0 < leaf <= n_pages:
                    regions.append(((leaf - 1) * page_size, leaf * page_size, "freelist_leaf", leaf))
            trunk = nxt
        # 2) b-tree pages: unallocated gap after the cell pointer array, and the freeblock chain
        for pno in range(1, n_pages + 1):
            base = (pno - 1) * page_size
            hdr = base + (100 if pno == 1 else 0)
            ptype = data[hdr] if hdr < len(data) else 0
            if ptype not in (0x02, 0x05, 0x0A, 0x0D):
                continue
            leaf = ptype in (0x0A, 0x0D)
            n_cells, content = struct.unpack(">HH", data[hdr + 3:hdr + 7])
            content = 65536 if content == 0 else content
            ptr_end = hdr + (8 if leaf else 12) + 2 * n_cells
            if base + content > ptr_end:
                regions.append((ptr_end, base + content, "unallocated", pno))
            fb, hops = struct.unpack(">H", data[hdr + 1:hdr + 3])[0], 0
            while fb and hops < 1000:
                fo = base + fb
                if fo + 4 > base + page_size:
                    break
                nxt, ln = struct.unpack(">HH", data[fo:fo + 4])
                regions.append((fo + 4, min(fo + ln, base + page_size), "freeblock", pno))
                fb, hops = nxt, hops + 1
        out: list[Finding] = []
        pattern = re.compile(rb"(?:[\x20-\x7e]|[\xc2-\xdf][\x80-\xbf]|[\xe0-\xef][\x80-\xbf]{2}|\x09|\x0a){%d,}" % MIN_RUN)
        for start, end, kind, pno in regions:
            if end <= start:
                continue
            for mt in pattern.finditer(data, start, end):
                raw = mt.group(0)
                text = raw.decode("utf-8", "replace").strip()
                if len(set(text)) < 5 or len(re.findall(r"[A-Za-z]", text)) < 6 or SCHEMA_TEXT.search(text):
                    continue          # padding / binary noise / leftover schema text
                out.append(self.finding(rel, "sqlite.carved_text",
                                        f"Text in SQLite {kind} space (page {pno}) of {rel}: {text[:140]}",
                                        value={"text": text, "region": kind, "page": pno, "offset": mt.start(),
                                               "length": len(raw), "page_size": page_size,
                                               "sha256": self.meta.get(rel, {}).get("sha256")},
                                        citations=[self.cite(rel, "byte_offset", raw.hex(), byte_offset=mt.start(),
                                                             snippet_encoding="hex")]))
                if len(out) >= MAX_CARVED_PER_DB:
                    return out
        return out


def run_on_evidence(case, evidence_id: str, tracer=None, context=None, memory=None, run_id=None,
                    output_dir: Path | None = None, limit: int = 0, carve: bool = True):
    from src.bridge.forensic_bridge import ForensicBridge

    ev_dir = case.dir / "evidence" / evidence_id
    manifest = json.loads((ev_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    bridge = ForensicBridge(ev_dir / "fs")
    try:
        agent = FileSystemAgent(bridge, manifest=manifest, carve=carve, tracer=tracer, context=context, memory=memory,
                                evidence_id=evidence_id, run_id=run_id)
        files = sorted(agent.meta)
        if limit:
            files = files[:limit]
        return agent.run(files, output_dir=output_dir) if files else None
    finally:
        bridge.close()


def interesting(findings: Sequence[Finding]) -> dict[str, int]:
    from collections import Counter
    return dict(Counter(f.category for f in findings))
