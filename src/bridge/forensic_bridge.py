"""Thin, deterministic wrappers around forensic primitives.

Every accessor works inside a :class:`Sandbox`: paths are resolved relative to the
extraction root and anything escaping it (``..``, absolute paths, symlinks) is refused.
Evidence is never written to. SQLite databases are queried from a hash-keyed working
copy (the original plus its ``-wal``/``-journal``), the standard practice that lets
WAL content be read without touching the evidence files.
"""
from __future__ import annotations

import hashlib
import os
import re
import shutil
import sqlite3
import stat
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from fractions import Fraction
from pathlib import Path
from typing import Any, Iterator

SQLITE_MAGIC = b"SQLite format 3\x00"


class SandboxViolation(PermissionError):
    pass


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def long_path(p: Path) -> Path:
    """Windows extended-length form (``\\\\?\\C:\\...``) so evidence paths over 260 characters stay
    reachable without the LongPathsEnabled registry switch. No-op elsewhere."""
    if os.name != "nt":
        return p
    s = str(p)
    if s.startswith("\\\\?\\"):
        return p
    if s.startswith("\\\\"):
        return Path("\\\\?\\UNC\\" + s[2:])
    return Path("\\\\?\\" + s)


class Sandbox:
    """The logical extraction root. All bridge access goes through :meth:`resolve`."""

    def __init__(self, root: Path | str):
        self.root = long_path(Path(root).resolve())
        if not self.root.is_dir():
            raise FileNotFoundError(self.root)
        self._hash_cache: dict[tuple[str, int, int], str] = {}

    def resolve(self, rel: str | Path) -> Path:
        p = Path(rel)
        if p.is_absolute():
            raise SandboxViolation(f"absolute path not allowed: {rel}")
        full = (self.root / p).resolve()
        if full != self.root and self.root not in full.parents:
            raise SandboxViolation(f"path escapes sandbox: {rel}")
        return full

    def relative(self, path: Path | str) -> str:
        full = Path(path).resolve()
        if self.root not in full.parents:
            raise SandboxViolation(f"path outside sandbox: {path}")
        return full.relative_to(self.root).as_posix()

    def hash(self, rel: str | Path) -> str:
        """SHA-256 of a sandbox file, cached on (path, size, mtime)."""
        full = self.resolve(rel)
        st = full.stat()
        key = (str(full), st.st_size, st.st_mtime_ns)
        if key not in self._hash_cache:
            self._hash_cache[key] = sha256_file(full)
        return self._hash_cache[key]


# --------------------------------------------------------------------------------------
# SQLite
# --------------------------------------------------------------------------------------
_ALLOWED_SQL_ACTIONS = {
    sqlite3.SQLITE_SELECT,
    sqlite3.SQLITE_READ,
    sqlite3.SQLITE_FUNCTION,
    getattr(sqlite3, "SQLITE_RECURSIVE", 33),
}


# Android ships custom ICU collations; absent them, queries/indexes over those columns fail.
_ANDROID_COLLATIONS = ("PHONEBOOK", "LOCALIZED", "UNICODE")


def _casefold_collation(a: str, b: str) -> int:
    a, b = a.casefold(), b.casefold()
    return (a > b) - (a < b)


def register_android_collations(con: sqlite3.Connection) -> None:
    for name in _ANDROID_COLLATIONS:
        con.create_collation(name, _casefold_collation)


def working_copy(src: Path, dst_dir: Path) -> Path:
    """Copy a database (+ ``-wal``/``-journal``) and merge the WAL into the copy.

    Evidence is never opened directly: even ``mode=ro`` connections rewrite the ``-shm``
    WAL index, which changes the evidence hash.
    """
    dst_dir.mkdir(parents=True, exist_ok=True)
    dst = dst_dir / src.name
    shutil.copy2(src, dst)
    for suffix in ("-wal", "-journal"):
        side = src.with_name(src.name + suffix)
        if side.exists():
            shutil.copy2(side, dst.with_name(dst.name + suffix))
    if dst.with_name(dst.name + "-wal").exists():
        con = sqlite3.connect(dst)
        try:
            con.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            con.execute("PRAGMA journal_mode=DELETE")
        except sqlite3.DatabaseError:
            pass
        finally:
            con.close()
    return dst


def _readonly_authorizer(action: int, *_: Any) -> int:
    return sqlite3.SQLITE_OK if action in _ALLOWED_SQL_ACTIONS else sqlite3.SQLITE_DENY


class SQLiteBridge:
    """Read-only SQL over evidence databases.

    Queries run on a working copy in a temp directory, opened with ``mode=ro`` after
    WAL checkpointing on the copy, and guarded by an authorizer that denies anything
    other than SELECT/READ/FUNCTION. Only single statements are accepted.
    """

    def __init__(self, sandbox: Sandbox, workdir: Path | None = None):
        self.sandbox = sandbox
        self.workdir = Path(workdir or tempfile.mkdtemp(prefix="forensi_sqlite_"))
        self._copies: dict[str, Path] = {}

    @staticmethod
    def is_sqlite(path: Path) -> bool:
        try:
            with open(path, "rb") as fh:
                return fh.read(16) == SQLITE_MAGIC
        except OSError:
            return False

    def _working_copy(self, rel: str) -> Path:
        src = self.sandbox.resolve(rel)
        digest = self.sandbox.hash(rel)
        if digest in self._copies:
            return self._copies[digest]
        dst = working_copy(src, self.workdir / digest[:16])
        self._copies[digest] = dst
        return dst

    def connect(self, rel: str, guarded: bool = True) -> sqlite3.Connection:
        """Read-only connection to the working copy; ``guarded`` adds the SELECT-only authorizer."""
        copy = self._working_copy(rel)
        con = sqlite3.connect(f"{copy.resolve().as_uri()}?mode=ro", uri=True)
        register_android_collations(con)
        if guarded:
            con.set_authorizer(_readonly_authorizer)
        return con

    def query(self, rel: str, sql: str, params: list[Any] | tuple = (), row_limit: int = 5000) -> list[tuple]:
        if not re.match(r"^\s*(SELECT|WITH)\b", sql, re.IGNORECASE):
            raise ValueError("only SELECT/WITH queries are allowed")
        con = self.connect(rel)
        try:
            cur = con.execute(sql, list(params))
            return cur.fetchmany(row_limit)
        finally:
            con.close()

    def tables(self, rel: str) -> dict[str, list[tuple[str, str]]]:
        """{table: [(column, declared_type), ...]} for ordinary rowid tables."""
        con = self.connect(rel, guarded=False)  # PRAGMA needs the unguarded (still ro) handle
        try:
            out: dict[str, list[tuple[str, str]]] = {}
            for (name, sql) in con.execute("SELECT name, sql FROM sqlite_master WHERE type='table'"):
                if name.startswith("sqlite_") or (sql and "WITHOUT ROWID" in sql.upper()):
                    continue
                if sql and sql.upper().startswith("CREATE VIRTUAL"):
                    continue
                cols = [(r[1], (r[2] or "").upper()) for r in con.execute(
                    'PRAGMA table_info("' + name.replace('"', '""') + '")')]
                out[name] = cols
            return out
        finally:
            con.close()

    def cleanup(self) -> None:
        shutil.rmtree(self.workdir, ignore_errors=True)


# --------------------------------------------------------------------------------------
# Image metadata
# --------------------------------------------------------------------------------------
def render_exif_value(value: Any) -> str:
    """Canonical string form of a piexif value. Agent and verifier both use this."""
    if isinstance(value, bytes):
        return value.rstrip(b"\x00").decode("utf-8", errors="replace").strip()
    if isinstance(value, tuple) and value and all(isinstance(v, tuple) and len(v) == 2 for v in value):
        return ",".join(f"{n}/{d}" for n, d in value)
    if isinstance(value, tuple) and len(value) == 2 and all(isinstance(v, int) for v in value):
        return f"{value[0]}/{value[1]}"
    if isinstance(value, tuple):
        return ",".join(str(v) for v in value)
    return str(value)


def _dms_to_decimal(dms: tuple, ref: str) -> float:
    deg = sum(Fraction(n, d or 1) / (60 ** i) for i, (n, d) in enumerate(dms))
    val = float(deg)
    return -val if ref.upper() in ("S", "W") else val


class ImageMetadataBridge:
    """EXIF/GPS via piexif + Pillow; ``exiftool`` is used when present on PATH."""

    _IFDS = ("0th", "Exif", "GPS", "1st", "Interop")

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
        self.exiftool = shutil.which("exiftool")

    def raw_exif(self, rel: str) -> dict[str, Any]:
        """{"<IFD>.<TagName>": raw piexif value} (empty when the image has no EXIF)."""
        import piexif

        path = self.sandbox.resolve(rel)
        try:
            data = piexif.load(str(path))
        except Exception:  # noqa: BLE001 - PNG/WebP/no EXIF
            return {}
        out: dict[str, Any] = {}
        for ifd in self._IFDS:
            for tag_id, value in (data.get(ifd) or {}).items():
                name = piexif.TAGS.get(ifd, {}).get(tag_id, {}).get("name", f"Tag{tag_id}")
                out[f"{ifd}.{name}"] = value
        return out

    def exif(self, rel: str) -> dict[str, str]:
        return {k: render_exif_value(v) for k, v in self.raw_exif(rel).items() if k != "1st.JPEGInterchangeFormat"}

    def gps(self, rel: str) -> dict[str, float] | None:
        raw = self.raw_exif(rel)
        try:
            lat = _dms_to_decimal(raw["GPS.GPSLatitude"], render_exif_value(raw["GPS.GPSLatitudeRef"]))
            lon = _dms_to_decimal(raw["GPS.GPSLongitude"], render_exif_value(raw["GPS.GPSLongitudeRef"]))
        except (KeyError, ZeroDivisionError, TypeError):
            return None
        out = {"latitude": round(lat, 7), "longitude": round(lon, 7)}
        if "GPS.GPSAltitude" in raw:
            n, d = raw["GPS.GPSAltitude"]
            alt = n / d if d else 0.0
            if raw.get("GPS.GPSAltitudeRef") == 1:
                alt = -alt
            out["altitude_m"] = round(alt, 2)
        return out

    def image_info(self, rel: str) -> dict[str, Any]:
        from PIL import Image

        with Image.open(self.sandbox.resolve(rel)) as im:
            return {"format": im.format, "mode": im.mode, "width": im.width, "height": im.height}

    def exiftool_json(self, rel: str) -> dict[str, Any] | None:
        if not self.exiftool:
            return None
        import json

        res = subprocess.run([self.exiftool, "-json", "-n", str(self.sandbox.resolve(rel))],
                             capture_output=True, text=True, timeout=60)
        return json.loads(res.stdout)[0] if res.returncode == 0 else None


def find_byte_offset(path: Path, needle: bytes, limit: int = 1 << 20) -> int | None:
    """First offset of ``needle`` within the first ``limit`` bytes (EXIF lives in the header)."""
    with open(path, "rb") as fh:
        idx = fh.read(limit).find(needle)
    return None if idx < 0 else idx


# --------------------------------------------------------------------------------------
# Text and file-system traversal
# --------------------------------------------------------------------------------------
class TextBridge:
    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox

    def read_text(self, rel: str) -> str:
        raw = self.sandbox.resolve(rel).read_bytes()
        # UTF-16 only with a byte-order mark: without one, almost any even-length byte string "decodes".
        if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
            try:
                return raw.decode("utf-16")
            except UnicodeDecodeError:
                pass
        try:
            return raw.decode("utf-8")
        except UnicodeDecodeError:
            return raw.decode("latin-1")

    def line(self, rel: str, line_number: int) -> str | None:
        lines = self.read_text(rel).splitlines()
        return lines[line_number - 1] if 1 <= line_number <= len(lines) else None


@dataclass
class FileEntry:
    path: str
    size: int
    modified_utc: str
    accessed_utc: str
    changed_or_born_utc: str   # st_ctime: metadata change (POSIX) / creation (Windows)
    hidden: bool
    is_sqlite: bool
    extension: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _iso(ts: float) -> str:
    return datetime.fromtimestamp(ts, timezone.utc).isoformat()


class FileSystemBridge:
    """Logical directory traversal.

    Sprint 1 walks an extracted logical tree with the native OS. ``pytsk3``/``dfvfs`` are
    detected for future raw/E01 image support (FileSystemAgent, Sprint 2).
    """

    def __init__(self, sandbox: Sandbox):
        self.sandbox = sandbox
        self.tsk_available = self._has("pytsk3")
        self.dfvfs_available = self._has("dfvfs")

    @staticmethod
    def _has(mod: str) -> bool:
        import importlib.util

        return importlib.util.find_spec(mod) is not None

    def walk(self, rel: str = ".", extensions: tuple[str, ...] | None = None) -> Iterator[FileEntry]:
        base = self.sandbox.resolve(rel)
        for dirpath, dirnames, filenames in os.walk(base, followlinks=False):
            dirnames.sort()
            for name in sorted(filenames):
                full = Path(dirpath) / name
                if full.is_symlink():
                    continue
                if extensions and full.suffix.lower() not in extensions:
                    continue
                st = full.stat()
                win_hidden = bool(getattr(st, "st_file_attributes", 0) & getattr(stat, "FILE_ATTRIBUTE_HIDDEN", 0))
                yield FileEntry(
                    path=self.sandbox.relative(full),
                    size=st.st_size,
                    modified_utc=_iso(st.st_mtime),
                    accessed_utc=_iso(st.st_atime),
                    changed_or_born_utc=_iso(st.st_ctime),
                    hidden=name.startswith(".") or any(p.startswith(".") for p in full.relative_to(self.sandbox.root).parts[:-1]) or win_hidden,
                    is_sqlite=SQLiteBridge.is_sqlite(full),
                    extension=full.suffix.lower(),
                )


class ForensicBridge:
    """Facade bundling the bridges for one sandbox."""

    def __init__(self, root: Path | str):
        self.sandbox = Sandbox(root)
        self.sqlite = SQLiteBridge(self.sandbox)
        self.images = ImageMetadataBridge(self.sandbox)
        self.text = TextBridge(self.sandbox)
        self.fs = FileSystemBridge(self.sandbox)

    def close(self) -> None:
        self.sqlite.cleanup()
