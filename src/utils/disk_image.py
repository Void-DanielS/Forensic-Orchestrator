"""Computer / removable-media disk images (E01 or raw/split dd) as case evidence.

Disk images are 0.01-70 GB, so they are read **in place** whenever the bytes are
randomly addressable:

* a bare ``.E01``/``.E02``... or raw file on the server: HTTP range requests;
* an image STORED inside a ZIP: a byte sub-range of the ZIP (still range-read);
* an image DEFLATED inside a ZIP: inflated once to ``data/_spool`` (CRC-32 checked) and
  deleted after extraction.

Reads go through a block cache, then ``libewf`` (EWF) or a concatenation (split raw),
then The Sleuth Kit (``pytsk3``): partition table -> every file system -> directory walk.
Only the parts of the image the walk and the selected files touch are downloaded.

Per evidence item the extractor writes the same layout as the mobile fetcher, so the
bridge, inspection and agents work unchanged::

    <evidence dir>/
    ├── fs/                 selected files (allocated) and fs/_deleted/ (recovered deleted files)
    ├── FILELIST.jsonl.gz   every directory entry of every file system: path, inode, size,
    │                       allocated/deleted, MACB timestamps (UTC). Nothing is skipped here.
    ├── MANIFEST.json       extracted files with SHA-256, MACB, inode, image + FS metadata,
    │                       the EWF acquisition headers and the stored acquisition hash
    └── _case_context/      acquisition logs shipped with the image (FTK Imager .txt etc.)

Credential stores (``SAM``, ``SECURITY``, ``/etc/shadow``, keychains), memory and swap
(``pagefile.sys``, ``hiberfil.sys``) are never extracted.
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import io
import json
import os
import re
import shutil
import threading
import time
import zipfile
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from urllib.parse import quote

# --------------------------------------------------------------------------------------
# Evidence specification
# --------------------------------------------------------------------------------------
DC = "https://digitalcorpora.s3.amazonaws.com/corpora"


@dataclass(frozen=True)
class DiskEvidence:
    """A computer / removable-media image. Either ``urls`` (image segments in order) or
    ``zip_url`` + ``member_regex`` (segments are members of one ZIP) must be set."""

    evidence_id: str
    title: str
    platform: str                        # windows | linux | removable | android_raw
    urls: tuple[str, ...] = ()           # absolute or relative to the Digital Corpora bucket
    zip_url: str = ""
    member_regex: str = ""
    fmt: str = "ewf"                     # ewf | raw
    acquisition_docs: tuple[str, ...] = ()   # URLs, or regex of ZIP members (when zip_url) saved to _case_context
    device: str = ""                     # short human description (owner, role in the case)
    group: str = "light"                 # light (range-read) | heavy (spooled) | extended | future
    transfer_gb: float = 0.0
    max_total_bytes: int = 2_000_000_000
    max_images: int = 1500
    max_deleted_bytes: int = 300_000_000
    note: str = ""

    def full(self, url: str) -> str:
        return url if url.startswith("http") else f"{DC}/{quote(url)}"


# --------------------------------------------------------------------------------------
# Byte access: block cache over range sources, file objects for libewf, split raw
# --------------------------------------------------------------------------------------
class BlockCache:
    """LRU cache of aligned blocks over a ``.get(start, length)`` / ``.size`` byte source."""

    def __init__(self, src, block: int = 4 << 20, max_blocks: int = 256):
        self.src, self.block, self.max_blocks = src, block, max_blocks
        self.size = src.size
        self._blocks: collections.OrderedDict[int, bytes] = collections.OrderedDict()
        self._lock = threading.Lock()
        self.fetched = 0

    def _get_block(self, idx: int) -> bytes:
        with self._lock:
            if idx in self._blocks:
                self._blocks.move_to_end(idx)
                return self._blocks[idx]
        start = idx * self.block
        data = self.src.get(start, min(self.block, self.size - start))
        with self._lock:
            self.fetched += len(data)
            self._blocks[idx] = data
            while len(self._blocks) > self.max_blocks:
                self._blocks.popitem(last=False)
        return data

    def read(self, offset: int, n: int) -> bytes:
        if offset >= self.size or n <= 0:
            return b""
        n = min(n, self.size - offset)
        out = bytearray()
        while n > 0:
            idx, inner = divmod(offset, self.block)
            chunk = self._get_block(idx)[inner:inner + n]
            if not chunk:
                break
            out += chunk
            offset += len(chunk)
            n -= len(chunk)
        return bytes(out)


class CachedFileObject(io.RawIOBase):
    """Seekable read-only file object over a :class:`BlockCache` (what libewf/libbfio needs)."""

    def __init__(self, cache: BlockCache):
        super().__init__()
        self.cache, self.pos = cache, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.cache.size + off
        return self.pos

    def tell(self) -> int:
        return self.pos

    def get_offset(self) -> int:
        return self.pos

    def get_size(self) -> int:
        return self.cache.size

    def read(self, size: int = -1) -> bytes:
        if size is None or size < 0:
            size = self.cache.size - self.pos
        data = self.cache.read(self.pos, size)
        self.pos += len(data)
        return data

    def readinto(self, b) -> int:  # type: ignore[override]
        data = self.read(len(b))
        b[:len(data)] = data
        return len(data)


class ConcatCache:
    """Split raw image (``.001``, ``.002``...) presented as one address space."""

    def __init__(self, caches: list[BlockCache]):
        self.caches = caches
        self.starts, pos = [], 0
        for c in caches:
            self.starts.append(pos)
            pos += c.size
        self.size = pos

    @property
    def fetched(self) -> int:
        return sum(c.fetched for c in self.caches)

    def read(self, offset: int, n: int) -> bytes:
        out = bytearray()
        for start, c in zip(self.starts, self.caches):
            if n <= 0:
                break
            if offset < start + c.size and offset + n > start:
                inner = max(0, offset - start)
                piece = c.read(inner, min(n, c.size - inner))
                out += piece
                offset += len(piece)
                n -= len(piece)
        return bytes(out)


def _tsk():
    import pytsk3

    return pytsk3


class _ImgBase:
    """Mixin turning a ``read(offset, n)`` provider into a pytsk3 image."""


def make_tsk_image(reader, size: int):
    pytsk3 = _tsk()

    class _Img(pytsk3.Img_Info):
        def __init__(self):
            super().__init__(url="", type=pytsk3.TSK_IMG_TYPE_EXTERNAL)

        def read(self, offset, length):  # noqa: D401 - pytsk3 callback
            return reader(offset, length)

        def get_size(self):
            return size

        def close(self):
            pass

    return _Img()


@dataclass
class OpenedImage:
    tsk: object
    media_size: int
    info: dict
    fetched: callable  # () -> bytes fetched so far


def open_image(sources: list, fmt: str) -> OpenedImage:
    """Open EWF or raw segments (each a ``.get/.size`` byte source) as a TSK image."""
    caches = [BlockCache(s) for s in sources]
    if fmt == "ewf":
        import pyewf

        handle = pyewf.handle()
        handle.open_file_objects([CachedFileObject(c) for c in caches])
        size = handle.get_media_size()
        lock = threading.Lock()

        def reader(off: int, n: int) -> bytes:
            with lock:
                return handle.read_buffer_at_offset(n, off)

        info = {"format": "ewf", "media_size": size, "bytes_per_sector": handle.get_bytes_per_sector(),
                "headers": _safe_call(handle.get_header_values) or {},
                "stored_hashes": _safe_call(handle.get_hash_values) or {}}
        return OpenedImage(make_tsk_image(reader, size), size, info, lambda: sum(c.fetched for c in caches))
    concat = ConcatCache(caches)
    return OpenedImage(make_tsk_image(concat.read, concat.size), concat.size,
                       {"format": "raw", "media_size": concat.size, "segments": len(caches)}, lambda: concat.fetched)


def _safe_call(fn):
    try:
        return {str(k): str(v) for k, v in fn().items()}
    except Exception:  # noqa: BLE001 - optional metadata
        return None


# --------------------------------------------------------------------------------------
# File systems and directory walk
# --------------------------------------------------------------------------------------
@dataclass
class FsEntry:
    fs: str
    path: str               # absolute inside the file system, "/" separated
    inode: int
    size: int
    is_dir: bool
    allocated: bool         # False: deleted directory entry (content may be overwritten)
    m: str | None = None    # modified
    a: str | None = None    # accessed
    c: str | None = None    # metadata changed
    b: str | None = None    # born / created


@dataclass
class FileSystem:
    label: str
    offset: int
    fstype: str
    partition: str
    handle: object = field(repr=False, default=None)


def _iso(sec, nano=0) -> str | None:
    if not sec:
        return None
    try:
        return datetime.fromtimestamp(sec + (nano or 0) / 1e9, timezone.utc).isoformat().replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError):
        return None


class _RegionFile(io.RawIOBase):
    """File object over a byte region of a pytsk3 image (what libvslvm needs to open an LVM PV)."""

    def __init__(self, img, offset: int, size: int):
        super().__init__()
        self.img, self.base, self.size, self.pos = img, offset, size, 0
        self.lock = threading.Lock()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def tell(self) -> int:
        return self.pos

    def get_size(self) -> int:
        return self.size

    def read(self, n: int = -1) -> bytes:
        n = self.size - self.pos if n is None or n < 0 else min(n, self.size - self.pos)
        if n <= 0:
            return b""
        with self.lock:
            data = self.img.read(self.base + self.pos, n)
        self.pos += len(data)
        return data


def _lvm_filesystems(img, offset: int, size: int, keep: list) -> list[FileSystem]:
    """Logical volumes of an LVM2 physical volume at ``offset`` (Ubuntu / RHEL servers default to LVM)."""
    pytsk3 = _tsk()
    try:
        head = img.read(offset, 4096)
    except OSError:
        return []
    if b"LABELONE" not in head[:2048]:
        return []
    try:
        import pyvslvm
    except ImportError:
        return []
    region = _RegionFile(img, offset, size)
    handle = pyvslvm.handle()
    handle.open_file_object(region)
    pv = _RegionFile(img, offset, size)       # the PV data is read through its own file object
    handle.open_physical_volume_files_as_file_objects([pv])
    keep += [region, pv, handle]
    found = []
    vg = handle.get_volume_group()
    for i in range(vg.get_number_of_logical_volumes()):
        lv = vg.get_logical_volume(i)
        lock = threading.Lock()

        def reader(off: int, n: int, lv=lv, lock=lock) -> bytes:
            with lock:
                return lv.read_buffer_at_offset(n, off)

        lv_img = make_tsk_image(reader, lv.get_size())
        keep += [lv, lv_img]
        try:
            fs = pytsk3.FS_Info(lv_img, offset=0)
        except OSError:
            continue
        found.append(FileSystem("", 0, _fstype(fs), f"LVM {vg.get_name()}/{lv.get_name()}", fs))
    return found


_KEEP_ALIVE: list = []   # LVM handles / images must outlive the FS_Info objects built on them


def _dissect(img, offset: int, size: int, desc: str) -> list[FileSystem]:
    """XFS / APFS (not in this TSK build) through dissect; [] when neither is present or the parse fails."""
    from src.utils.dissect_fs import dissect_filesystems
    try:
        return [FileSystem("", offset, t, f"{desc}: {d}", h)
                for t, d, h in dissect_filesystems(img.read, offset, size, _KEEP_ALIVE)]
    except Exception as exc:  # noqa: BLE001 - an unsupported variant must not stop the other partitions
        print(f"    [dissect] {desc}: {type(exc).__name__}: {str(exc)[:160]}", flush=True)
        return []


def find_filesystems(img) -> list[FileSystem]:
    pytsk3 = _tsk()
    found: list[FileSystem] = []
    try:
        vol = pytsk3.Volume_Info(img)
        block = vol.info.block_size
        for part in vol:
            if not (part.flags & pytsk3.TSK_VS_PART_FLAG_ALLOC) or part.len < 2048:
                continue
            desc = part.desc.decode("utf-8", "replace") if isinstance(part.desc, bytes) else str(part.desc)
            try:
                fs = pytsk3.FS_Info(img, offset=part.start * block)
            except OSError:
                found += _lvm_filesystems(img, part.start * block, part.len * block, _KEEP_ALIVE) or                     _dissect(img, part.start * block, part.len * block, desc)
                continue
            found.append(FileSystem("", part.start * block, _fstype(fs), desc, fs))
    except OSError:
        pass
    if not found:
        try:
            fs = pytsk3.FS_Info(img, offset=0)
            found.append(FileSystem("", 0, _fstype(fs), "whole image", fs))
        except OSError:
            found += _lvm_filesystems(img, 0, img.get_size(), _KEEP_ALIVE) or _dissect(img, 0, img.get_size(), "whole image")
    for i, fs in enumerate(found):
        fs.label = "" if len(found) == 1 else _volume_label(fs, i)
    return found


def _volume_label(fs: FileSystem, i: int) -> str:
    name = re.sub(r"[^A-Za-z0-9]+", "_", fs.partition).strip("_").lower()
    return f"vol{i}_{name}"[:40] if name else f"vol{i}"


def _fstype(fs) -> str:
    pytsk3 = _tsk()
    try:
        code = int(fs.info.ftype)
    except Exception:  # noqa: BLE001
        return "unknown"
    names = {int(getattr(pytsk3, n)): n[len("TSK_FS_TYPE_"):].lower() for n in dir(pytsk3)
             if n.startswith("TSK_FS_TYPE_") and not n.endswith(("_DETECT", "_MASK", "_UNSUPP"))
             and isinstance(getattr(pytsk3, n), int)}
    return names.get(code, f"type_{code:#x}")


# Directories never descended into: huge, OS-owned, no investigative content.
PRUNE_DIRS = re.compile(r"^/(windows/(winsxs|installer|assembly|servicing|softwaredistribution|microsoft\.net)"
                        r"|\$extend|\$orphanfiles|system volume information/.+\{|proc|sys|dev|usr/(share|lib|lib64|include|src)"
                        r"|lib|lib64|bin|sbin|boot)(/|$)", re.I)


def walk(fs: FileSystem, max_depth: int = 64) -> Iterator[FsEntry]:
    if hasattr(fs.handle, "walk"):           # dissect adapter (XFS / APFS)
        yield from fs.handle.walk(fs.label, PRUNE_DIRS, max_depth)
        return
    pytsk3 = _tsk()
    root = fs.handle.open_dir(path="/")
    stack = [(root, "", 0)]
    seen: set[int] = set()
    while stack:
        directory, base, depth = stack.pop()
        for f in directory:
            name_b = f.info.name.name
            if name_b in (b".", b".."):
                continue
            name = name_b.decode("utf-8", "replace")
            path = f"{base}/{name}"
            meta = f.info.meta
            is_dir = bool(meta) and meta.type == pytsk3.TSK_FS_META_TYPE_DIR
            allocated = bool(f.info.name.flags & pytsk3.TSK_FS_NAME_FLAG_ALLOC)
            if meta:
                e = FsEntry(fs.label, path, int(meta.addr), int(meta.size), is_dir, allocated,
                            _iso(meta.mtime, getattr(meta, "mtime_nano", 0)), _iso(meta.atime, getattr(meta, "atime_nano", 0)),
                            _iso(meta.ctime, getattr(meta, "ctime_nano", 0)), _iso(meta.crtime, getattr(meta, "crtime_nano", 0)))
            else:
                e = FsEntry(fs.label, path, int(f.info.name.meta_addr), 0, False, allocated)
            yield e
            if is_dir and allocated and depth < max_depth and e.inode not in seen and not PRUNE_DIRS.match(path):
                seen.add(e.inode)
                try:
                    stack.append((f.as_directory(), path, depth + 1))
                except OSError:
                    continue


def read_file(fs: FileSystem, inode: int, size: int, chunk: int = 4 << 20) -> bytes:
    f = fs.handle.open_meta(inode=inode)
    out = bytearray()
    off = 0
    while off < size:
        data = f.read_random(off, min(chunk, size - off))
        if not data:
            break
        out += data
        off += len(data)
    return bytes(out)


# --------------------------------------------------------------------------------------
# Selection for computer file systems
# --------------------------------------------------------------------------------------
NEVER = re.compile(
    r"(^|/)(sam|security)(\.log\d?)?$|/(pagefile|hiberfil|swapfile)\.sys$|/etc/(shadow|gshadow)-?$|"
    r"\.(vmem|vmsn|dmp|mem|keychain|keychain-db)$|/memory\.dmp$|password|passwd\.bak|/\.ssh/id_[a-z0-9]+$|"
    r"/\.gnupg/", re.I)
DOC_EXTS_C = (".doc", ".docx", ".xls", ".xlsx", ".ppt", ".pptx", ".pdf", ".rtf", ".txt", ".odt", ".ods", ".csv",
              ".md", ".html", ".htm", ".eml", ".msg", ".vcf", ".ics", ".kml", ".kmz", ".gpx", ".xml", ".json")
IMG_EXTS_C = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".heic", ".webp", ".tif", ".tiff")
SQLITE_NAMES = {"history", "cookies", "web data", "login data", "top sites", "favicons", "shortcuts",
                "network action predictor", "visited links", "places.sqlite", "formhistory.sqlite",
                "cookies.sqlite", "downloads.sqlite", "webappsstore.sqlite", "permissions.sqlite"}
USER_ROOT = re.compile(r"^/(users|documents and settings|home|root)/[^/]+/", re.I)

COMPUTER_PRIORITY = {"fs_metadata": 0, "registry": 0, "event_log": 0, "prefetch": 0, "app_database": 0, "database": 0,
                     "browser_history": 0, "email": 1, "shortcut": 1, "recycle_bin": 1, "system_log": 1,
                     "shell_history": 0, "thumbnail_cache": 2, "user_document": 2, "screenshot": 2,
                     "user_image": 3, "app_image": 4, "removable_file": 2, "app_config": 3}


def auto_platform(fstype: str) -> str:
    """Selector set for a file system whose owner platform is not declared (mixed bundles)."""
    t = fstype.lower()
    if t.startswith("ntfs"):
        return "windows"
    if t.startswith(("ext", "yaffs", "f2fs", "btrfs", "xfs")):
        return "linux"
    if t.startswith(("hfs", "apfs")):
        return "macos"
    if t.startswith(("fat", "exfat", "iso9660", "udf")):
        return "removable"
    return "generic"


def classify_computer(path: str, platform: str, size: int) -> str | None:
    """Category for a file inside a computer / removable file system, or None to skip."""
    l = path.lower()
    name = l.rsplit("/", 1)[-1]
    ext = "." + name.rsplit(".", 1)[-1] if "." in name else ""
    if NEVER.search(l) or name.endswith((".part", ".tmp")) and size > 10 << 20:
        return None
    in_user = bool(USER_ROOT.match(l))
    if re.match(r"^/\$(mft|logfile|boot|secure|attrdef)$", l):
        return "fs_metadata" if size <= 1 << 30 else None   # NTFS metadata: MFT records, journal

    # ---- shared: SQLite and browser stores anywhere in a user profile
    if ext in (".db", ".sqlite", ".sqlite3", ".sqlitedb") or name in SQLITE_NAMES or (
            name.endswith(("-wal", "-journal")) and (in_user or "/var/lib/" in l)):
        if "thumbcache" in name or name == "thumbs.db" or name.startswith("iconcache"):
            return "thumbnail_cache"
        if in_user or "/programdata/" in l or "/var/lib/" in l or platform == "removable":
            return "app_database"
        return None
    if in_user and ("/cookies/" in l or "/temporary internet files/" in l and ext not in IMG_EXTS_C):
        return "browser_history" if size <= 4 << 20 else None
    if name == "index.dat" and in_user:
        return "browser_history"                     # Internet Explorer <= 9 (Windows XP / 7)

    if platform == "removable":
        if ext in IMG_EXTS_C:
            return "user_image"
        return "removable_file" if size <= 64 << 20 else None

    if platform == "windows":
        if re.search(r"/windows/system32/config/(system|software|default|components)(\.log\d?)?$", l) \
                or name in ("ntuser.dat", "usrclass.dat", "amcache.hve", "ntuser.dat.log1", "ntuser.dat.log2") \
                or re.search(r"/windows/appcompat/programs/amcache\.hve", l):
            return "registry"
        if ext in (".evtx", ".evt") or "/winevt/logs/" in l:
            return "event_log"
        if ext == ".pf" and "/prefetch/" in l:
            return "prefetch"
        if "/system32/sru/" in l and name == "srudb.dat":
            return "app_database"
        if re.search(r"/(\$recycle\.bin|recycler)/", l):
            return "recycle_bin" if (name.startswith("$i") or name == "info2" or ext in DOC_EXTS_C + IMG_EXTS_C) else None
        if ext in (".lnk",) or name.endswith(("automaticdestinations-ms", "customdestinations-ms")):
            return "shortcut" if in_user else None
        if ext in (".pst", ".ost", ".dbx", ".mbx", ".eml", ".msg", ".wab") or "/thunderbird/profiles/" in l and ext == "":
            return "email" if in_user else None
        if re.search(r"/windows/(setupapi[^/]*\.log|inf/setupapi[^/]*\.log|system32/logfiles/[^/]+\.log)$", l) \
                or "/windows/system32/tasks/" in l:
            return "system_log"
        if not in_user:
            return None
        if "/appdata/" in l or "/application data/" in l or "/local settings/" in l:
            if ext in IMG_EXTS_C:
                return "app_image"
            if ext in (".json", ".xml", ".ini", ".plist") and size <= 4 << 20:
                return "app_config"
            return None
        if ext in IMG_EXTS_C:
            return "screenshot" if "screenshot" in l or "screen shot" in l else "user_image"
        if ext in DOC_EXTS_C:
            return "user_document"
        return None

    if platform in ("linux", "chromeos"):
        if name in (".bash_history", ".zsh_history", ".python_history", ".mysql_history", ".lesshst", ".viminfo"):
            return "shell_history"
        if l.startswith("/var/log/") or "/recovery_logs" in l or "/log/" in l and ext in ("", ".log", ".txt"):
            return "system_log" if size <= 64 << 20 else None
        if re.match(r"^/etc/(hostname|hosts|passwd|group|fstab|crontab|timezone|os-release|issue|network/interfaces|"
                    r"dhcpcd\.conf|wpa_supplicant/wpa_supplicant\.conf|rc\.local)$", l):
            return "app_config"
        if in_user or l.startswith("/home/") or l.startswith("/opt/") or l.startswith("/srv/"):
            if ext in IMG_EXTS_C:
                return "user_image"
            if ext in DOC_EXTS_C + (".py", ".sh", ".js", ".conf", ".cfg", ".ini", ".yaml", ".yml", ".gcode"):
                return "user_document" if size <= 32 << 20 else None
        return None
    if platform == "macos":
        if l.startswith("/private/var/log/") or l.startswith("/library/logs/") or "/library/logs/" in l and in_user:
            return "system_log" if size <= 64 << 20 else None
        if name in (".bash_history", ".zsh_history", ".sh_history") or "/.bash_sessions/" in l:
            return "shell_history"
        if in_user:
            if ext == ".plist" and ("/library/preferences/" in l or "/library/application support/" in l) \
                    and size <= 4 << 20:
                return "app_config"
            if ext in (".emlx", ".mbox", ".eml") or "/library/mail/" in l and ext == "":
                return "email" if size <= 64 << 20 else None
            if ext in IMG_EXTS_C:
                return "screenshot" if "screen shot" in l or "screenshot" in l else (
                    "app_image" if "/library/" in l else "user_image")
            if "/library/" in l:
                return None
            if ext in DOC_EXTS_C:
                return "user_document"
        if re.match(r"^/(library|private/var)/db/", l) and (ext in (".db", ".plist") or name.endswith(".store.db")):
            return "app_database" if size <= 64 << 20 else None
        return None

    if platform in ("android", "ios"):
        # A phone partition (chip-off / JTAG / userdata image) opened by The Sleuth Kit: reuse the mobile
        # selectors. The userdata partition is mounted at /data, so its root has no "data/" prefix.
        from src.utils.dataset_scraper import classify as classify_mobile
        return classify_mobile(path, platform) or classify_mobile("/data" + path, platform)

    if platform in ("iot", "generic"):
        if ext in (".db", ".sqlite", ".sqlite3") or name.endswith(("-wal", "-journal")):
            return "app_database"
        if l.startswith("/var/log/") or ext == ".log" or "/log/" in l:
            return "system_log" if size <= 64 << 20 else None
        if ext in IMG_EXTS_C:
            return "user_image"
        if ext in (".json", ".xml", ".conf", ".cfg", ".ini", ".plist", ".yaml", ".yml") and size <= 4 << 20:
            return "app_config"
        if ext in DOC_EXTS_C + (".mp4", ".avi", ".h264", ".264", ".dav", ".mov") and size <= 256 << 20:
            return "user_document"
        return None
    raise ValueError(platform)


# --------------------------------------------------------------------------------------
# Extraction
# --------------------------------------------------------------------------------------
_SIZE_CAP = {"fs_metadata": 1 << 30, "app_database": 256 << 20, "registry": 256 << 20, "email": 1 << 30, "event_log": 128 << 20,
             "browser_history": 128 << 20, "removable_file": 64 << 20}
DELETED_CATEGORIES = {"user_document", "user_image", "screenshot", "email", "app_database", "shortcut",
                      "removable_file", "recycle_bin"}


class DiskFetcher:
    """Extract one disk image. ``sources`` (already-open byte sources) and ``prefix`` let a bundle
    (:mod:`src.utils.bundle`) run several images into one evidence directory: files go to
    ``fs/<prefix>/`` and every FILELIST row carries ``image=<prefix>``."""

    def __init__(self, ev: DiskEvidence, out: Path, prefix: str = "", sources: list | None = None):
        from src.utils import dataset_scraper as ds

        self.ds, self.ev, self.out = ds, ev, out
        self.prefix, self.sources = prefix, sources
        self.ctx = out / "_case_context"
        self.dest = ds._DestMap(out / "fs" / prefix if prefix else out / "fs")
        self.records: list[dict] = []
        self.notes: list[str] = []
        self.skipped: collections.Counter = collections.Counter()
        self.spooled: list[Path] = []

    # ---------------------------------------------------------------- segments
    def resolve_segments(self) -> list:
        ds = self.ds
        if self.ev.urls:
            for doc in self.ev.acquisition_docs:
                self._fetch_doc_url(doc)
            from src.utils.remote import open_source

            remote = [open_source(self.ev.full(u), ds.SPOOL_DIR, self.ev.evidence_id) for u in self.ev.urls]
            total = sum(s.size for s in remote)
            srcs = [self._local(s, u.rsplit("/", 1)[-1].replace("%23", "#"), total) for s, u in zip(remote, self.ev.urls)]
            print(f"  [image] {len(srcs)} segment(s), {total / 1e9:.2f} GB, "
                  f"{'spooled' if self.spooled else 'range-read'}", flush=True)
            return srcs
        top = ds.HttpSource(self.ev.full(self.ev.zip_url))
        zf = ds._open_zip(top)
        infos = sorted((i for i in zf.infolist() if re.search(self.ev.member_regex, i.filename)), key=lambda i: i.filename)
        if not infos:
            raise RuntimeError(f"no ZIP member matches {self.ev.member_regex}")
        for doc_re in self.ev.acquisition_docs:
            for i in zf.infolist():
                if re.search(doc_re, i.filename) and i.file_size < 20 << 20:
                    self._save_ctx(i.filename.rsplit("/", 1)[-1], ds._read_member(top, i))
        srcs = []
        for info in infos:
            if info.compress_type == zipfile.ZIP_STORED:
                srcs.append(self._local(ds.SubSource(top, ds._data_offset(top, info), info.file_size),
                                        info.filename.rsplit("/", 1)[-1], sum(i.file_size for i in infos)))
            else:
                srcs.append(ds.FileSource(self._spool(top, info)))
        print(f"  [image] {len(srcs)} segment(s), {sum(s.size for s in srcs) / 1e9:.2f} GB "
              f"({'range-read' if not self.spooled else f'{len(self.spooled)} spooled'})")
        return srcs

    def _local(self, src, name: str, total: int):
        """Spool a remote segment with parallel range requests (resumable) unless the image is huge.

        A directory walk touches scattered clusters; below ~45 GB a parallel full download
        (~25-30 MB/s) is far faster than random range reads (~1 request per cluster run)."""
        if total > SPOOL_MAX_BYTES:
            return src
        ds = self.ds
        ds.SPOOL_DIR.mkdir(parents=True, exist_ok=True)
        path = ds.SPOOL_DIR / f"{self.ev.evidence_id}_{name}"
        self.spooled.append(path)
        if path.exists() and path.stat().st_size == src.size and not path.with_name(path.name + ".todo").exists():
            return ds.FileSource(path)
        if shutil.disk_usage(ds.SPOOL_DIR).free < src.size + 15e9:
            raise RuntimeError(f"not enough disk to spool {name}")
        parallel_download(src, path, label=name)
        return ds.FileSource(path)

    def _spool(self, top, info) -> Path:
        ds = self.ds
        ds.SPOOL_DIR.mkdir(parents=True, exist_ok=True)
        path = ds.SPOOL_DIR / f"{self.ev.evidence_id}_{info.filename.rsplit('/', 1)[-1]}"
        if path.exists() and path.stat().st_size == info.file_size:
            print(f"  [spool] reuse {path.name}")
        else:
            free = shutil.disk_usage(ds.SPOOL_DIR).free
            if free < info.file_size * 1.1 + 15e9:
                raise RuntimeError(f"not enough disk to spool {info.filename} ({free / 1e9:.0f} GB free)")
            print(f"  [spool] {info.filename} deflated -> {info.file_size / 1e9:.2f} GB", flush=True)
            tmp = path.with_name(path.name + ".part")
            with open(tmp, "wb") as fh:   # _InflateReader raises on CRC mismatch at EOF
                shutil.copyfileobj(ds._member_stream(top, info, self.ev.evidence_id), fh, 8 << 20)
            tmp.replace(path)
        self.spooled.append(path)
        return path

    def _fetch_doc_url(self, url: str) -> None:
        from urllib.parse import unquote

        full = self.ev.full(url)
        dest = self.ctx / self.ds._safe_part(unquote(full.rsplit("/", 1)[-1]))
        if not dest.exists():
            try:
                self.ds._http_download(full, dest)
                print(f"  [context] {dest.name}")
            except Exception as exc:  # noqa: BLE001
                self.notes.append(f"acquisition doc unavailable: {full} ({exc})")

    def _save_ctx(self, name: str, data: bytes) -> None:
        dest = self.ctx / self.ds._safe_part(name)
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(data)
            print(f"  [context] {dest.name}")

    # ---------------------------------------------------------------- main
    def extract(self, listing) -> tuple[dict, list[dict]]:
        """Open the image, list every FS entry into ``listing`` and extract the selection."""
        image = open_image(self.sources or self.resolve_segments(), self.ev.fmt)
        print(f"  [image] media {image.media_size / 1e9:.2f} GB; acquisition hash: "
              f"{image.info.get('stored_hashes') or '-'}", flush=True)
        filesystems = find_filesystems(image.tsk)
        if not filesystems:
            raise RuntimeError("no file system recognised by The Sleuth Kit")
        fs_meta, candidates = [], []
        for fs in filesystems:
            plat = self.ev.platform if self.ev.platform != "auto" else auto_platform(fs.fstype)
            stats: collections.Counter = collections.Counter()
            print(f"  [fs] {fs.label or '/'} {fs.fstype} @ {fs.offset} ({fs.partition})", flush=True)
            for e in walk(fs):
                row = asdict(e)
                if self.prefix:
                    row["image"] = self.prefix
                listing.write(json.dumps(row, ensure_ascii=False) + "\n")
                stats["dirs" if e.is_dir else "files"] += 1
                stats["deleted"] += not e.allocated
                if not e.is_dir and e.size > 0:
                    cat = classify_computer(e.path, plat, e.size)
                    if cat:
                        candidates.append((fs, e, cat))
                n = stats["files"] + stats["dirs"]
                if n % 25000 == 0:
                    print(f"    walked {n} entries, {image.fetched() / 1e6:.0f} MB read", flush=True)
            fs_meta.append({"label": fs.label, "fstype": fs.fstype, "offset": fs.offset, "partition": fs.partition,
                            "platform": plat,
                            **({"image": self.prefix} if self.prefix else {}), **stats})
            print(f"    {dict(stats)}", flush=True)
        self._extract(candidates, image)
        return {**image.info, "bytes_read": image.fetched()}, fs_meta

    def run(self) -> dict:
        started = time.time()
        self.out.mkdir(parents=True, exist_ok=True)
        print(f"[fetch] {self.ev.evidence_id}: {self.ev.title}", flush=True)
        try:
            with gzip.open(self.out / "FILELIST.jsonl.gz", "wt", encoding="utf-8") as listing:
                image_info, fs_meta = self.extract(listing)
            ok = True
        finally:
            # Keep the spool after a failure so a retry resumes without re-downloading.
            if locals().get("ok"):
                for p in self.spooled:
                    p.unlink(missing_ok=True)
        self.records.sort(key=lambda r: r["path"])
        manifest = {"evidence_id": self.ev.evidence_id, "title": self.ev.title, "platform": self.ev.platform,
                    "kind": "disk_image", "spec": asdict(self.ev), "image": image_info, "filesystems": fs_meta,
                    "fetched_utc": datetime.now(timezone.utc).isoformat(), "fetch_seconds": round(time.time() - started, 1),
                    "member_count": len(self.records), "skipped": dict(self.skipped), "notes": self.notes,
                    "files": self.records}
        (self.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"  [done] {len(self.records)} files, {image_info['bytes_read'] / 1e9:.2f} GB read in "
              f"{manifest['fetch_seconds']:.0f}s (skipped {dict(self.skipped)})", flush=True)
        return manifest

    def _extract(self, candidates: list, image: OpenedImage) -> None:
        live = [c for c in candidates if c[1].allocated]
        deleted = [c for c in candidates if not c[1].allocated and c[2] in DELETED_CATEGORIES and c[1].size <= 32 << 20]
        live.sort(key=lambda c: (COMPUTER_PRIORITY.get(c[2], 5), c[1].size))
        deleted.sort(key=lambda c: (COMPUTER_PRIORITY.get(c[2], 5), c[1].size))
        budget, images, dbytes = 0, 0, 0
        chosen = []
        for fs, e, cat in live:
            if e.size > _SIZE_CAP.get(cat, 32 << 20):
                self.skipped["too_large"] += 1
                continue
            if cat in ("user_image", "app_image", "screenshot"):
                if e.size < 8 * 1024:
                    self.skipped["tiny_image"] += 1
                    continue
                if images >= self.ev.max_images:
                    self.skipped["image_budget"] += 1
                    continue
                images += 1
            if budget + e.size > self.ev.max_total_bytes:
                self.skipped["byte_budget"] += 1
                continue
            budget += e.size
            chosen.append((fs, e, cat, False))
        for fs, e, cat in deleted:
            if dbytes + e.size > self.ev.max_deleted_bytes:
                self.skipped["deleted_budget"] += 1
                continue
            dbytes += e.size
            chosen.append((fs, e, cat, True))
        print(f"  [select] {len(chosen)} files ({budget / 1e6:.0f} MB live, {dbytes / 1e6:.0f} MB deleted) "
              f"of {len(candidates)} candidates", flush=True)
        t0 = time.time()
        for i, (fs, e, cat, is_deleted) in enumerate(chosen, 1):
            try:
                self._write(fs, e, cat, is_deleted)
            except Exception as exc:  # noqa: BLE001 - unreadable / overwritten clusters
                self.skipped["read_error_deleted" if is_deleted else "read_error"] += 1
                if not is_deleted:
                    self.notes.append(f"read error {fs.label}:{e.path}: {type(exc).__name__}: {str(exc)[:120]}")
            if i % 500 == 0 or i == len(chosen):
                print(f"  [extract] {i}/{len(chosen)}  {image.fetched() / 1e9:.2f} GB read, "
                      f"{time.time() - t0:.0f}s", flush=True)

    def _write(self, fs: FileSystem, e: FsEntry, cat: str, is_deleted: bool) -> None:
        member = f"{self.prefix + '!' if self.prefix else ''}{fs.label or 'fs'}:{e.path}"
        if is_deleted:
            # Flat <inode>_<name>: deleted names collide with each other and with deleted directories.
            rel = "/".join(p for p in ("_deleted", fs.label, f"{e.inode}_{e.path.rsplit('/', 1)[-1]}") if p)
            member += f"#inode{e.inode}"
        else:
            rel = "/".join(p for p in (fs.label, e.path.lstrip("/")) if p)
        dest, original = self.dest.map(rel, member)
        if is_deleted:
            original = e.path
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        h, written = hashlib.sha256(), 0
        f = fs.handle.open_meta(inode=e.inode)
        with open(tmp, "wb") as out:
            while written < e.size:
                data = f.read_random(written, min(4 << 20, e.size - written))
                if not data:
                    break
                out.write(data)
                h.update(data)
                written += len(data)
        if written == 0 or (written != e.size and not is_deleted):
            tmp.unlink(missing_ok=True)
            raise IOError(f"short read {written}/{e.size}")
        tmp.replace(dest)
        mt = _ts(e.m)
        if mt:
            os.utime(dest, (_ts(e.a) or mt, mt))
        rec = {"path": dest.relative_to(self.out).as_posix(), "source_member": member, "category": cat,
               "size": written, "sha256": h.hexdigest(), "member_mtime": e.m, "inode": e.inode,
               "macb": {"m": e.m, "a": e.a, "c": e.c, "b": e.b}, "deleted": is_deleted}
        if is_deleted:
            rec["reliability"] = "recovered from a deleted entry; clusters may have been reallocated"
            if written != e.size:
                rec["recovered_bytes"] = f"{written}/{e.size}"
        if original:
            rec["original_path"] = original
        self.records.append(rec)


def _ts(iso: str | None) -> float | None:
    if not iso:
        return None
    try:
        return datetime.fromisoformat(iso.replace("Z", "+00:00")).timestamp()
    except ValueError:
        return None


SPOOL_MAX_BYTES = 45_000_000_000


def parallel_download(src, path: Path, label: str = "", chunk: int = 32 << 20, threads: int = 12) -> None:
    """Download a range source to ``path`` with parallel range requests; resumable via ``<path>.todo``."""
    from concurrent.futures import ThreadPoolExecutor, as_completed

    url = getattr(src, "url", "") or ""
    if "archive.org" in url:
        threads = 3      # archive.org throttles many parallel ranges per IP to a crawl (chunk size kept: .todo resume)
    elif "drive.usercontent.google.com" in url:
        threads = min(threads, 8)
    todo_p = path.with_name(path.name + ".todo")
    n = (src.size + chunk - 1) // chunk
    todo = set(json.loads(todo_p.read_text())) if todo_p.exists() and path.exists() else set(range(n))
    if not path.exists() or path.stat().st_size != src.size:
        with open(path, "wb") as fh:
            fh.truncate(src.size)
        todo = set(range(n))
    todo_p.write_text(json.dumps(sorted(todo)))
    lock = threading.Lock()
    t0, done_bytes = time.time(), 0
    print(f"  [download] {label} {src.size / 1e9:.2f} GB ({len(todo)}/{n} chunks to fetch)", flush=True)

    def work(i: int) -> int:
        start = i * chunk
        data = src.get(start, min(chunk, src.size - start))
        with lock, open(path, "r+b") as fh:
            fh.seek(start)
            fh.write(data)
        return i

    with ThreadPoolExecutor(threads) as pool:
        for k, fut in enumerate(as_completed([pool.submit(work, i) for i in sorted(todo)]), 1):
            i = fut.result()
            with lock:
                todo.discard(i)
                done_bytes += min(chunk, src.size - i * chunk)
                if k % 16 == 0 or not todo:
                    todo_p.write_text(json.dumps(sorted(todo)))
                    print(f"    {done_bytes / 1e9:.2f} GB, {done_bytes / 1e6 / max(1e-6, time.time() - t0):.1f} MB/s",
                          flush=True)
    todo_p.unlink(missing_ok=True)


def fetch_disk(ev: DiskEvidence, out: Path) -> dict:
    return DiskFetcher(ev, out).run()
