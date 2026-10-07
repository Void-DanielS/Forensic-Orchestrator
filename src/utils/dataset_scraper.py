"""Public forensic reference-image corpus fetcher.

Builds ``data/raw_images/<source_id>/`` for a catalogue of **public, documented** mobile
extractions: the Josh Hickman Android/iOS reference images, the Magnet Forensics CTF
images and the DFRWS 2021 challenge, all hosted by Digital Corpora. Each source keeps its
*case context* (image-creation documentation, scenario notes) in ``_case_context/``,
outside the agents' sandbox (``fs/``).

The archives are 0.2-40 GB and often nested (a ZIP inside a ZIP, a TAR inside a ZIP), so
the fetcher never stores whole images:

* **ZIP**: the central directory is read with HTTP range requests and only selected
  members are fetched, each CRC-32 checked against the archive.
* **Nested STORED member**: addressed as a byte sub-range of its parent, so it is also
  range-read without downloading the rest.
* **Nested DEFLATED member**: inflated on the fly. A TAR is streamed member by member, and
  a ZIP is spooled to a temp file because it needs random access.
* **TAR / TAR.GZ**: streamed with parallel read-ahead, keeping only selected members.
* **iTunes backup**: ``Manifest.db`` is fetched first, then hashed file IDs are mapped back
  to ``Domain/relativePath`` and only the selected files are fetched.

Selection keeps SQLite databases (+ WAL/SHM/journal), user photos and screenshots, app
preference files and system logs, within a per-source byte budget. Every file gets a
SHA-256 in ``MANIFEST.json``. :func:`inspect_source` then opens every file read-only
(on working copies for SQLite), re-hashes it, and writes ``INSPECTION.json``.
:func:`write_corpus_index` summarises all sources in ``data/raw_images/CORPUS_INDEX.json``.

Usage::

    python -m src.utils.dataset_scraper --list-sources
    python -m src.utils.dataset_scraper --sources light          # range-read sources (~1-2 GB transfer)
    python -m src.utils.dataset_scraper --sources heavy          # streamed full-file-system images (~37 GB transfer)
    python -m src.utils.dataset_scraper --sources android_10,magnet2021_ios
    python -m src.utils.dataset_scraper --inspect-only --sources all
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import io
import json
import os
import plistlib
import re
import shutil
import sqlite3
import struct
import sys
import tarfile
import tempfile
import threading
import time
import zipfile
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator
from urllib.parse import quote

import requests

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.bridge.forensic_bridge import register_android_collations, working_copy
from src.config.settings import DATA_DIR, RAW_IMAGES_DIR

DC = "https://digitalcorpora.s3.amazonaws.com/corpora"
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".heic", ".webp")
DB_EXTS = (".db", ".sqlite", ".sqlite3", ".sqlitedb", ".storedata")
TEXT_EXTS = (".xml", ".txt", ".log", ".json", ".plist")
SQLITE_MAGIC = b"SQLite format 3\x00"
DOC_EXTS = (".pdf", ".docx", ".xlsx", ".txt", ".md")
DOC_NAME = re.compile(r"image.?creation|data.?population|documentation|solution|readme|scenario", re.I)
# Credential material shipped with some CTF images is never copied into the corpus.
SENSITIVE = re.compile(r"password|keychain|passcode|_mem\.zip$", re.I)
MAX_WIN_PATH = 240
SPOOL_DIR = DATA_DIR / "_spool"
# Chromium / WebView databases have no extension and live outside databases/.
KNOWN_DB_NAMES = {"history", "cookies", "web data", "login data", "top sites", "favicons", "shortcuts",
                  "network action predictor", "media history", "visited links", "affiliation database",
                  "quotamanager", "origin bound certs", "reporting and nel", "trust tokens"}
IMAGE_CATEGORIES = ("user_image", "screenshot", "app_image")
MIN_IMAGE_BYTES = 8 * 1024  # smaller files are icons / UI assets, not evidence
CATEGORY_PRIORITY = {"app_database": 0, "database": 0, "screenshot": 1, "user_image": 2, "app_image": 3,
                     "cloud_record": 2, "app_config": 3, "system_log": 4}


# --------------------------------------------------------------------------------------
# Selection
# --------------------------------------------------------------------------------------
def classify(path: str, platform: str) -> str | None:
    """Category for an archive member, or None to skip it."""
    l = "/" + path.replace("\\", "/").lower().lstrip("/")
    name = l.rsplit("/", 1)[-1]
    ext = "." + name.rsplit(".", 1)[-1] if "." in name else ""
    if any(s in l for s in ("/sbin/.magisk/", "/proc/", "/sys/")):
        return None
    side = name.endswith(("-wal", "-shm", "-journal"))

    if platform == "takeout":
        if ext in IMAGE_EXTS:
            return "user_image"
        if ext in (".json", ".html", ".csv", ".txt", ".mbox", ".ics", ".vcf", ".kml"):
            return "cloud_record"
        return None

    if platform == "android":
        in_app = re.search(r"/data/(data|user(_de)?/\d+)/[^/]+/", l) or re.search(r"/apps/[^/]+/", l)
        if in_app and (re.search(r"/(databases|db)/", l) or name.removesuffix("-journal") in KNOWN_DB_NAMES):
            return "app_database"
        if ext in DB_EXTS or (side and (".db" in name or "/databases/" in l)):
            return "database"
        if in_app and re.search(r"/(shared_prefs|sp)/", l) and ext == ".xml":
            return "app_config"
        if ext in IMAGE_EXTS:
            if "screenshot" in l:
                return "screenshot"
            if in_app or "/android/data/" in l or "/android/media/" in l:
                return "app_image"   # chat media, received files, app caches
            if re.search(r"/(media/0|sdcard|storage/emulated/0|data/media)/", l):
                return "user_image"
            return None
        if re.search(r"/data/(system|misc|system_ce/0|system_de/0|log)/", l) and (ext in TEXT_EXTS or "/log/" in l):
            return "system_log"
        return None

    if platform == "ios":
        if ext in DB_EXTS or (side and any(e in name for e in DB_EXTS)):
            return "database"
        if ext in IMAGE_EXTS:
            if "/dcim/" in l:
                return "screenshot" if ext == ".png" else "user_image"
            if "/sms/attachments/" in l or "/message/media/" in l:
                return "user_image"
            if "screenshot" in l:
                return "screenshot"
            if re.search(r"/containers/(data/application|shared/appgroup)/", l) or "appdomain" in l:
                return "app_image"
            return None
        if "/library/preferences/" in l and ext == ".plist":
            return "app_config"
        if "/logs/" in l and ext in TEXT_EXTS:
            return "system_log"
        return None
    raise ValueError(platform)


def is_context_doc(path: str) -> bool:
    name = path.rsplit("/", 1)[-1]
    depth = path.strip("/").count("/")
    return (name.lower().endswith(DOC_EXTS) and DOC_NAME.search(name) is not None
            and not SENSITIVE.search(path) and depth <= 3)


# --------------------------------------------------------------------------------------
# Source catalogue
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Source:
    source_id: str
    title: str
    platform: str                       # android | ios | takeout
    url: str                            # absolute, or relative to the Digital Corpora bucket
    kind: str = "zip"                   # zip | tar | tar.gz | itunes
    chain: tuple[str, ...] = ()         # nested archive members, outermost first
    expand: tuple[str, ...] = ()        # STORED .tar members of the final zip to stream as well
    strip_prefix: str = ""              # regex removed from the start of member paths
    docs: tuple[str, ...] = ()          # URLs (abs or bucket-relative) or "member:<path in outer zip>"
    author: str = ""
    scenario: str = ""
    group: str = "light"                # light | heavy | extended (very large) | future (not fetchable yet)
    transfer_gb: float = 0.0            # approximate bytes downloaded
    max_total_bytes: int = 1_500_000_000
    max_images: int = 1500
    note: str = ""

    @property
    def full_url(self) -> str:
        return self.url if self.url.startswith("http") else f"{DC}/{quote(self.url)}"


_HICKMAN = "Joshua Hickman (Digital Corpora reference images)"
_MAGNET = "Magnet Forensics CTF (Digital Corpora scenarios/magnet)"
SOURCES: dict[str, Source] = {s.source_id: s for s in (
    # ------------------------------------------------------------------ light (range-read)
    Source("android_10", "Android 10 - Google Pixel 3, full file system", "android",
           "mobile/android_10/Non-Cellebrite Extraction/Pixel 3.zip", strip_prefix=r"Pixel 3/",
           docs=("mobile/android_10/Android10-ImageCreation.pdf", "mobile/android_10/Android_10.txt",
                 "mobile/android_10/SMS-Messages.xlsx"),
           author=_HICKMAN, scenario="Documented population of 31 apps plus stock apps, Jan-Feb 2020 (EST).",
           transfer_gb=0.25),
    Source("dfrws2021_galaxy_s10", "DFRWS 2021 challenge - Samsung Galaxy S10 (SM-G973F) full dump", "android",
           "dfrws/challenge-2021/3_Samsung GSM_SM-G973F_DS Galaxy S10.zip", strip_prefix=r"Dump/",
           docs=("https://raw.githubusercontent.com/dfrws/dfrws2021-challenge/main/README.md",),
           author="DFRWS 2021 Forensic Challenge", scenario="IoT/mobile investigation challenge device.",
           transfer_gb=1.5),
    Source("magnet2019_android", "Magnet CTF 2019 - Android (sdcard + adb backup)", "android",
           "scenarios/magnet/2019 CTF - Android.zip", expand=("adb-data.tar",),
           docs=("scenarios/magnet/README.md",), author=_MAGNET, transfer_gb=0.2),
    Source("magnet2021_ios", "Magnet CTF 2021 - iPhone 8 'Eli', full file system", "ios",
           "scenarios/magnet/2021 CTF - iOS.zip",
           chain=("2021 MVS CTF_Eli iPhone 8/518e8d766f9b3e76db216f35fdb6b0604e50f61b_files_full.zip",),
           docs=("member:2021 MVS CTF_Eli iPhone 8/518e8d766f9b3e76db216f35fdb6b0604e50f61b.pdf",
                 "scenarios/magnet/README.md"),
           author=_MAGNET, transfer_gb=1.5),
    Source("ios15_itunes", "iOS 15.3.1 - iTunes backup", "ios",
           "mobile/android_13/ios_15_3_1/c50d35ac85f428883e3b6fa3893599da85f708ea.zip", kind="itunes",
           docs=("mobile/android_13/ios_15_3_1/iOS15-ImageCreation.pdf",), author=_HICKMAN, transfer_gb=0.5,
           group="future", note="encrypted backup: needs the documented backup password to decrypt"),
    Source("ios13_4_1_itunes", "iOS 13.4.1 - iTunes backup", "ios",
           "mobile/ios_13_4_1/ios_13_4_1.zip", kind="itunes",
           chain=("iOS 13.4.1 Extraction/iTunes Backup/c623fbd7e91b041e07a68f8523f53a35973e475d.zip",),
           docs=("member:iOS 13.4.1 Extraction/iOS13-4-1-ImageCreation.pdf",), author=_HICKMAN, transfer_gb=0.42,
           group="future", note="encrypted backup: needs the documented backup password to decrypt"),
    Source("ios13_3_1_itunes", "iOS 13.3.1 - iTunes backup", "ios",
           "mobile/ios_13_3_1/ios_13_3_1.zip", kind="itunes",
           chain=("iOS 13.3.1 Extraction/iTunes Backup/c623fbd7e91b041e07a68f8523f53a35973e475d.zip",),
           docs=("member:iOS 13.3.1 Extraction/iOS13-3-1-ImageCreation.pdf",), author=_HICKMAN, transfer_gb=0.4,
           group="future", note="encrypted backup: needs the documented backup password to decrypt"),
    Source("owl_nexus5", "Owl scenario - LG Nexus 5 quick image (Agent Data + sdcard + adb backup)", "android",
           "scenarios/2019-owl/Nexus5-Quick/LGE Nexus 5 Quick Image.zip", expand=("adb-data.tar",),
           docs=("scenarios/2019-owl/Nexus5-Quick/image_info.txt", "scenarios/2019-owl/Nexus5-Quick/activity_log.txt"),
           author="Digital Corpora (2019 Owl scenario)", transfer_gb=0.13),
    Source("magnet2020_takeout", "Magnet CTF 2020 - Google Takeout", "takeout",
           "scenarios/magnet/2020 CTF - Takeout.zip", docs=("scenarios/magnet/README.md",), author=_MAGNET,
           transfer_gb=0.36),
    # ------------------------------------------------------------------ heavy (streamed)
    Source("ios13_4_1_ffs", "iOS 13.4.1 - iPhone full file system", "ios",
           "mobile/ios_13_4_1/ios_13_4_1.zip", kind="tar",
           chain=("iOS 13.4.1 Extraction/Extraction/13-4-1.tar",),
           docs=("member:iOS 13.4.1 Extraction/iOS13-4-1-ImageCreation.pdf",),
           author=_HICKMAN, group="heavy", transfer_gb=9.0),
    Source("android_9", "Android 9 - Google Pixel 3, full file system (nested tar)", "android",
           "mobile/android_9.tar.gz", kind="tar.gz", author=_HICKMAN, group="extended", transfer_gb=4.5),
    Source("magnet2020_android", "Magnet CTF 2020 - Android", "android",
           "scenarios/magnet/2020 CTF - Android.tar", kind="tar", docs=("scenarios/magnet/README.md",),
           author=_MAGNET, group="heavy", transfer_gb=8.9),
    Source("magnet2022_android", "Magnet CTF 2022 - Android", "android",
           "scenarios/magnet/2022 CTF - Android-001.tar", kind="tar", docs=("scenarios/magnet/README.md",),
           author=_MAGNET, group="heavy", transfer_gb=9.5),
    Source("magnet2022_ios", "Magnet CTF 2022 - iPhone 8 'Jess', iOS 15 full file system", "ios",
           "scenarios/magnet/2022 CTF - iOS Full File System.zip",
           chain=("Jess_CTF_iPhone8/fb028ddefa8af7df5b12d3e729f075d150637a31_files_full.zip",),
           docs=("member:Jess_CTF_iPhone8/fb028ddefa8af7df5b12d3e729f075d150637a31.pdf", "scenarios/magnet/README.md"),
           author=_MAGNET, group="heavy", transfer_gb=4.7, note="spools ~8.7 GB to data/_spool temporarily"),
    # ------------------------------------------------------------------ future (registered, not fetched by default)
    Source("android_7", "Android 7 - LG Nexus 5X physical image", "android", "mobile/android_7.tar.gz", kind="tar.gz",
           author=_HICKMAN, group="future", transfer_gb=6.5,
           note="raw physical dump blk0_mmcblk0.bin (31 GB): needs partition/ext4 parsing (pytsk3, Sprint 2)"),
    Source("android_8", "Android 8 - LG Nexus 5X physical image", "android", "mobile/android_8.tar.gz", kind="tar.gz",
           author=_HICKMAN, group="future", transfer_gb=5.7,
           note="raw physical dump blk0_mmcblk0.bin (31 GB): needs partition/ext4 parsing (pytsk3, Sprint 2)"),
    Source("android_13", "Android 13 - full file system", "android", "mobile/android_13/android_13_data.tar.gz",
           kind="tar.gz", docs=("mobile/android_13/android_13_imagecreation_solution.pdf",),
           author=_HICKMAN, group="extended", transfer_gb=19.8),
    Source("android_14", "Android 14 - full file system", "android", "mobile/android_14/Android_14_Public_Image.tar.gz",
           kind="tar.gz", docs=("mobile/android_14/Android14-ImageCreation.pdf",),
           author=_HICKMAN, group="extended", transfer_gb=18.4),
    Source("ios16", "iOS 16 - full file system", "ios", "mobile/iOS16/iOS_16_Public_Image.tar.gz", kind="tar.gz",
           docs=("mobile/iOS16/iOS16-ImageCreation.pdf",), author=_HICKMAN, group="extended", transfer_gb=20.1),
    Source("ios17", "iOS 17 - full file system", "ios", "mobile/iOS17/iOS_17_Public_Image.tar.gz", kind="tar.gz",
           docs=("mobile/iOS17/iOS17-ImageCreation.pdf",), author=_HICKMAN, group="extended", transfer_gb=22.1),
    Source("ios13_3_1_ffs", "iOS 13.3.1 - iPhone full file system", "ios", "mobile/ios_13_3_1/ios_13_3_1.zip",
           kind="tar", chain=("iOS 13.3.1 Extraction/Extraction/13-3-1.tar",),
           docs=("member:iOS 13.3.1 Extraction/iOS13-3-1-ImageCreation.pdf",),
           author=_HICKMAN, group="extended", transfer_gb=8.3),
    Source("magnet2020_ios", "Magnet CTF 2020 - iOS full file system", "ios", "scenarios/magnet/2020 CTF - iOS.zip",
           chain=("2020 CTF - iOS/iOS_Filesystem-001.zip",), author=_MAGNET, group="extended", transfer_gb=12.8,
           note="spools ~13 GB"),
)}
# Not representable yet: android_6 (raw ext4 data.img -> needs pytsk3, Sprint 2), android_11/12
# (deflated nested zips of 11-16 GB holding Cellebrite/TAR exports), 2011 NexusOne/NexusS (YAFFS2 dumps).


def resolve_sources(spec: str) -> list[Source]:
    if spec in ("light", "heavy", "extended", "future"):
        return [s for s in SOURCES.values() if s.group == spec]
    if spec == "all":
        return [s for s in SOURCES.values() if s.group != "future"]
    out = []
    for sid in spec.split(","):
        if sid.strip() not in SOURCES:
            raise SystemExit(f"unknown source '{sid}'. Use --list-sources")
        out.append(SOURCES[sid.strip()])
    return out


def source_dir(source: Source) -> Path:
    """Evidence directory: ``data/cases/<case>/evidence/<source_id>`` (legacy ``data/raw_images`` otherwise)."""
    from src.cases.registry import case_of_evidence

    case = case_of_evidence(source.source_id)
    if case is not None:
        return case.dir / "evidence" / source.source_id
    return RAW_IMAGES_DIR / source.source_id


# --------------------------------------------------------------------------------------
# Byte sources (HTTP / sub-range / local file) and streams
# --------------------------------------------------------------------------------------
class HttpSource:
    """Remote object addressed with retried, thread-safe HTTP range requests."""

    def __init__(self, url: str):
        self.url = url
        self._local = threading.local()
        head = self._session().head(url, allow_redirects=True, timeout=60)
        head.raise_for_status()
        if head.headers.get("Accept-Ranges", "").lower() != "bytes":
            raise RuntimeError(f"{url} does not support range requests")
        self.size = int(head.headers["Content-Length"])

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "s"):
            self._local.s = requests.Session()
        return self._local.s

    def get(self, start: int, length: int) -> bytes:
        if length <= 0:
            return b""
        for attempt in range(6):
            try:
                r = self._session().get(self.url, headers={"Range": f"bytes={start}-{start + length - 1}"}, timeout=180)
                if r.status_code == 206 and len(r.content) == length:
                    return r.content
                raise IOError(f"HTTP {r.status_code}, got {len(r.content)}/{length} bytes")
            except (requests.RequestException, IOError):
                if attempt == 5:
                    raise
                time.sleep(2 ** attempt)
        raise AssertionError("unreachable")


class SubSource:
    def __init__(self, parent, offset: int, size: int):
        self.parent, self.offset, self.size = parent, offset, size

    def get(self, start: int, length: int) -> bytes:
        return self.parent.get(self.offset + start, length)


class FileSource:
    def __init__(self, path: Path):
        self.path, self.size = path, path.stat().st_size

    def get(self, start: int, length: int) -> bytes:
        with open(self.path, "rb") as fh:
            fh.seek(start)
            return fh.read(length)


class _RangeFile(io.RawIOBase):
    """Seekable file view over a byte source (used for ZIP central-directory parsing)."""

    def __init__(self, src):
        self.src, self.pos = src, 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self.pos

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.src.size + off
        return self.pos

    def readinto(self, b) -> int:  # type: ignore[override]
        n = min(len(b), self.src.size - self.pos)
        if n <= 0:
            return 0
        data = self.src.get(self.pos, n)
        b[: len(data)] = data
        self.pos += len(data)
        return len(data)


class _PrefetchReader(io.RawIOBase):
    """Sequential reader with parallel read-ahead (several range requests in flight)."""

    def __init__(self, src, label: str = "", chunk: int = 16 << 20, ahead: int = 6):
        self.src, self.chunk, self.label = src, chunk, label
        self.pool = ThreadPoolExecutor(ahead)
        self.queue: collections.deque = collections.deque()
        self.next_off = self.consumed = self._last_report = 0
        self.buf, self.bufpos = b"", 0
        self.started = time.time()
        for _ in range(ahead):
            self._schedule()

    def _schedule(self) -> None:
        if self.next_off < self.src.size:
            n = min(self.chunk, self.src.size - self.next_off)
            self.queue.append(self.pool.submit(self.src.get, self.next_off, n))
            self.next_off += n

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:  # type: ignore[override]
        if self.bufpos >= len(self.buf):
            if not self.queue:
                return 0
            self.buf, self.bufpos = self.queue.popleft().result(), 0
            self._schedule()
        n = min(len(b), len(self.buf) - self.bufpos)
        b[:n] = self.buf[self.bufpos: self.bufpos + n]
        self.bufpos += n
        self.consumed += n
        if self.consumed - self._last_report >= 1 << 30:
            self._last_report = self.consumed
            rate = self.consumed / max(time.time() - self.started, 1e-3) / 1e6
            print(f"  [stream {self.label}] {self.consumed / 1e9:.1f}/{self.src.size / 1e9:.1f} GB ({rate:.0f} MB/s)", flush=True)
        return n

    def close(self) -> None:
        self.pool.shutdown(wait=False, cancel_futures=True)
        super().close()


class _InflateReader(io.RawIOBase):
    """Raw-deflate decoder over a compressed stream, verifying the ZIP CRC-32 at EOF."""

    def __init__(self, raw, expected_crc: int | None = None):
        self.raw, self.expected_crc = raw, expected_crc
        self.d = zlib.decompressobj(-15)
        self.buf, self.bufpos, self.crc, self.eof = b"", 0, 0, False

    def readable(self) -> bool:
        return True

    def readinto(self, b) -> int:  # type: ignore[override]
        while self.bufpos >= len(self.buf):
            if self.eof:
                return 0
            chunk = self.raw.read(1 << 20)
            if chunk:
                self.buf = self.d.decompress(chunk)
            else:
                self.buf, self.eof = self.d.flush(), True
                if self.expected_crc is not None and (zlib.crc32(self.buf, self.crc) & 0xFFFFFFFF) != self.expected_crc:
                    raise IOError("CRC mismatch on nested archive member")
            self.bufpos = 0
            self.crc = zlib.crc32(self.buf, self.crc)
        n = min(len(b), len(self.buf) - self.bufpos)
        b[:n] = self.buf[self.bufpos: self.bufpos + n]
        self.bufpos += n
        return n


def _open_zip(src) -> zipfile.ZipFile:
    return zipfile.ZipFile(io.BufferedReader(_RangeFile(src), buffer_size=4 << 20))


def _data_offset(src, info: zipfile.ZipInfo) -> int:
    hdr = src.get(info.header_offset, 30)
    if hdr[:4] != b"PK\x03\x04":
        raise IOError(f"bad local header for {info.filename}")
    name_len, extra_len = struct.unpack("<HH", hdr[26:30])
    return info.header_offset + 30 + name_len + extra_len


def _read_member(src, info: zipfile.ZipInfo) -> bytes:
    """Fetch one ZIP member directly (thread-safe), inflate and CRC-check it."""
    raw = src.get(_data_offset(src, info), info.compress_size)
    if info.compress_type == zipfile.ZIP_STORED:
        data = raw
    elif info.compress_type == zipfile.ZIP_DEFLATED:
        data = zlib.decompress(raw, -15)
    else:
        raise IOError(f"unsupported compression {info.compress_type} for {info.filename}")
    if zlib.crc32(data) & 0xFFFFFFFF != info.CRC:
        raise IOError(f"CRC mismatch for {info.filename}")
    return data


def _member_stream(src, info: zipfile.ZipInfo, label: str) -> io.RawIOBase:
    sub = SubSource(src, _data_offset(src, info), info.compress_size)
    reader = _PrefetchReader(sub, label)
    if info.compress_type == zipfile.ZIP_STORED:
        return reader
    return _InflateReader(reader, expected_crc=info.CRC)


def _http_download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    with requests.get(url, stream=True, timeout=120) as r:
        r.raise_for_status()
        tmp = dest.with_name(dest.name + ".part")
        with open(tmp, "wb") as fh:
            for chunk in r.iter_content(1 << 20):
                fh.write(chunk)
    tmp.replace(dest)


def _crc32_file(path: Path) -> int:
    crc = 0
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            crc = zlib.crc32(chunk, crc)
    return crc & 0xFFFFFFFF


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------------------
# Destination mapping (Windows-safe) and budget
# --------------------------------------------------------------------------------------
_INVALID = re.compile(r'[<>:"|?*\x00-\x1f]')
_RESERVED = {"con", "prn", "aux", "nul", *(f"com{i}" for i in range(1, 10)), *(f"lpt{i}" for i in range(1, 10))}


def _safe_part(part: str) -> str:
    p = _INVALID.sub("_", part).rstrip(" .") or "_"
    return "_" + p if p.split(".")[0].lower() in _RESERVED else p


class _DestMap:
    """Maps archive paths to safe local paths; records the original path when it differs."""

    def __init__(self, fs_root: Path):
        self.fs = fs_root
        self.used: dict[str, str] = {}

    def map(self, rel: str, member: str) -> tuple[Path, str | None]:
        parts = [p for p in rel.replace("\\", "/").split("/") if p not in ("", ".", "..")]
        safe = [_safe_part(p) for p in parts]
        dest = self.fs.joinpath(*safe)
        original = None if safe == parts else rel
        if len(str(dest)) > MAX_WIN_PATH:
            digest = hashlib.sha1(rel.encode("utf-8", "surrogateescape")).hexdigest()[:16]
            dest, original = self.fs / "_longpaths" / f"{digest}_{safe[-1][-80:]}", rel
        key, n = str(dest).lower(), 2
        while key in self.used and self.used[key] != member:  # case-insensitive FS collision
            dest = dest.with_name(f"{dest.stem}~{n}{dest.suffix}")
            key, n, original = str(dest).lower(), n + 1, rel
        self.used[key] = member
        return dest, original


@dataclass
class _Budget:
    max_bytes: int
    max_images: int
    bytes: int = 0
    images: int = 0
    skipped: collections.Counter = field(default_factory=collections.Counter)

    def accept(self, cat: str, size: int) -> bool:
        cap = 256 << 20 if cat in ("database", "app_database") else 32 << 20
        if size > cap:
            self.skipped["too_large"] += 1
            return False
        if self.bytes + size > self.max_bytes:
            self.skipped["byte_budget"] += 1
            return False
        if cat in IMAGE_CATEGORIES:
            if size < MIN_IMAGE_BYTES:
                self.skipped["tiny_image"] += 1
                return False
            if self.images >= self.max_images:
                self.skipped["image_budget"] += 1
                return False
            self.images += 1
        self.bytes += size
        return True


# --------------------------------------------------------------------------------------
# Fetch
# --------------------------------------------------------------------------------------
class _Fetcher:
    def __init__(self, source: Source, workers: int = 8):
        self.s = source
        self.out = source_dir(source)
        self.ctx = self.out / "_case_context"
        self.dest = _DestMap(self.out / "fs")
        self.budget = _Budget(source.max_total_bytes, source.max_images)
        self.workers = workers
        self.records: list[dict] = []
        self.notes: list[str] = []
        self.strip = re.compile("^" + source.strip_prefix) if source.strip_prefix else None
        self.spooled: list[Path] = []

    # ---------------------------------------------------------------- helpers
    def _rel(self, member: str) -> str:
        return self.strip.sub("", member, count=1) if self.strip else member

    def _write(self, member: str, cat: str, data: bytes, mtime: float | None, crc: int | None = None) -> None:
        dest, original = self.dest.map(self._rel(member), member)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        tmp.write_bytes(data)
        tmp.replace(dest)
        if mtime:
            os.utime(dest, (mtime, mtime))
        self.records.append(self._record(dest, member, cat, len(data), hashlib.sha256(data).hexdigest(), mtime, crc, original))

    def _record(self, dest, member, cat, size, sha, mtime, crc, original) -> dict:
        rec = {"path": dest.relative_to(self.out).as_posix(), "source_member": member, "category": cat, "size": size,
               "member_mtime": datetime.fromtimestamp(mtime).isoformat() if mtime else None, "sha256": sha}
        if crc is not None:
            rec["zip_crc32"] = f"{crc:08x}"
        if original:
            rec["original_path"] = original
        return rec

    def _save_doc(self, name: str, data: bytes) -> None:
        if SENSITIVE.search(name):
            return
        dest = self.ctx / _safe_part(name.rsplit("/", 1)[-1])
        dest.parent.mkdir(parents=True, exist_ok=True)
        if not dest.exists():
            dest.write_bytes(data)
            print(f"  [context] {dest.name}")

    # ---------------------------------------------------------------- context docs
    def fetch_url_docs(self) -> None:
        for doc in self.s.docs:
            if doc.startswith("member:"):
                continue
            url = doc if doc.startswith("http") else f"{DC}/{quote(doc)}"
            dest = self.ctx / requests.utils.unquote(url.rsplit("/", 1)[-1])
            if dest.exists():
                continue
            try:
                print(f"  [context] {dest.name}")
                _http_download(url, dest)
            except requests.RequestException as exc:
                self.notes.append(f"context doc unavailable: {url} ({exc})")

    def fetch_member_docs(self, top_src) -> None:
        members = [d[len("member:"):] for d in self.s.docs if d.startswith("member:")]
        if not members:
            return
        zf = _open_zip(top_src)
        for m in members:
            try:
                self._save_doc(m, _read_member(top_src, zf.getinfo(m)))
            except KeyError:
                self.notes.append(f"context member missing: {m}")

    # ---------------------------------------------------------------- archive resolution
    def resolve(self):
        """Walk the nested chain. Returns ('zip', byte_source) or ('stream', fileobj)."""
        src = top = HttpSource(self.s.full_url)
        print(f"  [archive] {self.s.full_url} ({src.size / 1e9:.1f} GB)")
        self.fetch_member_docs(top)
        for i, member in enumerate(self.s.chain):
            info = _open_zip(src).getinfo(member)
            last = i == len(self.s.chain) - 1
            if info.compress_type == zipfile.ZIP_STORED:
                print(f"  [nested] {member} stored -> range-addressable ({info.file_size / 1e9:.2f} GB)")
                src = SubSource(src, _data_offset(src, info), info.file_size)
                continue
            stream = _member_stream(src, info, self.s.source_id)
            if last and self.s.kind in ("tar", "tar.gz"):
                print(f"  [nested] {member} deflated -> streaming")
                return "stream", stream
            src = FileSource(self._spool(stream, member, info.file_size))
        if self.s.kind in ("zip", "itunes"):
            return "zip", src
        return "stream", _PrefetchReader(src, self.s.source_id)

    def _spool(self, stream, member: str, size: int) -> Path:
        SPOOL_DIR.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(SPOOL_DIR).free
        if free < size * 1.2 + 5e9:
            raise RuntimeError(f"not enough disk to spool {member} ({size / 1e9:.1f} GB, {free / 1e9:.0f} GB free)")
        path = SPOOL_DIR / f"{self.s.source_id}_{hashlib.sha1(member.encode()).hexdigest()[:8]}.zip"
        print(f"  [nested] {member} deflated zip -> spooling {size / 1e9:.2f} GB to {path}")
        with open(path, "wb") as fh:
            shutil.copyfileobj(stream, fh, 8 << 20)
        self.spooled.append(path)
        return path

    # ---------------------------------------------------------------- zip
    def fetch_zip(self, src, prefix: str = "") -> None:
        zf = _open_zip(src)
        selected: list[tuple[zipfile.ZipInfo, str]] = []
        for info in zf.infolist():
            if info.is_dir():
                continue
            if is_context_doc(prefix + info.filename) and info.file_size < 50 << 20:
                self._save_doc(info.filename, _read_member(src, info))
                continue
            cat = classify(self._rel(prefix + info.filename), self.s.platform)
            if cat:
                selected.append((info, cat))
        selected.sort(key=lambda x: (CATEGORY_PRIORITY[x[1]], x[0].file_size))
        selected = [(i, c) for i, c in selected if self.budget.accept(c, i.file_size)]
        self._download_members(src, [(i, c, prefix + i.filename) for i, c in selected])
        for name in self.s.expand:
            info = zf.getinfo(name)
            # STORED members are range-read; DEFLATED ones are inflated on the fly (CRC-checked) and streamed.
            print(f"  [expand] {name} ({info.file_size / 1e6:.0f} MB)")
            self.fetch_tar(_member_stream(src, info, name), "r|", prefix=name + "/")

    def _download_members(self, src, items: list[tuple[zipfile.ZipInfo, str, str]]) -> None:
        total = sum(i.file_size for i, _, _ in items)
        print(f"  [select] {len(items)} members, {total / 1e6:.1f} MB (skipped: {dict(self.budget.skipped)})")
        mapped = []
        for info, cat, logical in items:
            dest, original = self.dest.map(self._rel(logical), info.filename)
            mapped.append((info, cat, logical, dest, original))
        lock = threading.Lock()

        def work(item) -> dict:
            info, cat, logical, dest, original = item
            mtime = datetime(*info.date_time).timestamp()
            if not (dest.exists() and dest.stat().st_size == info.file_size and _crc32_file(dest) == info.CRC):
                data = _read_member(src, info)
                dest.parent.mkdir(parents=True, exist_ok=True)
                tmp = dest.with_name(dest.name + ".part")
                tmp.write_bytes(data)
                tmp.replace(dest)
            os.utime(dest, (mtime, mtime))
            return self._record(dest, logical, cat, info.file_size, sha256_file(dest), mtime, info.CRC, original)

        done = 0
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            for fut in as_completed([pool.submit(work, it) for it in mapped]):
                with lock:
                    self.records.append(fut.result())
                    done += 1
                    if done % 250 == 0 or done == len(mapped):
                        print(f"  [download] {done}/{len(mapped)}", flush=True)

    # ---------------------------------------------------------------- iTunes backup
    def fetch_itunes(self, src) -> None:
        zf = _open_zip(src)
        names = {i.filename: i for i in zf.infolist() if not i.is_dir()}
        mdb = next((n for n in names if n.endswith("Manifest.db")), None)
        if not mdb:
            raise RuntimeError("no Manifest.db in iTunes backup")
        prefix = mdb[: -len("Manifest.db")]
        for meta in ("Manifest.db", "Manifest.plist", "Info.plist", "Status.plist"):
            if prefix + meta in names:
                info = names[prefix + meta]
                self._write("_backup_meta/" + meta, "backup_meta", _read_member(src, info),
                            datetime(*info.date_time).timestamp(), info.CRC)
        mplist = plistlib.loads(_read_member(src, names[prefix + "Manifest.plist"])) if prefix + "Manifest.plist" in names else {}
        if mplist.get("IsEncrypted"):
            self.notes.append("iTunes backup is encrypted; only backup metadata kept")
            return
        with tempfile.TemporaryDirectory() as td:
            db = Path(td) / "Manifest.db"
            db.write_bytes(_read_member(src, names[mdb]))
            con = sqlite3.connect(db)
            rows = con.execute("SELECT fileID, domain, relativePath FROM Files WHERE flags = 1").fetchall()
            con.close()
        items = []
        for file_id, domain, rel in rows:
            member = f"{prefix}{file_id[:2]}/{file_id}"
            logical = f"{domain}/{rel}"
            cat = classify(logical, "ios")
            if cat and member in names:
                items.append((names[member], cat, logical))
        items.sort(key=lambda x: (CATEGORY_PRIORITY[x[1]], x[0].file_size))
        items = [it for it in items if self.budget.accept(it[1], it[0].file_size)]
        print(f"  [itunes] {len(rows)} files in Manifest.db, {len(items)} selected")
        saved, self.strip = self.strip, None
        try:
            self._download_members(src, items)
        finally:
            self.strip = saved

    def _fetch_spooled_zip(self, stream, name: str, size: int) -> None:
        path = self._spool(stream, name, size)
        try:
            src = FileSource(path)
            names = [i.filename for i in _open_zip(src).infolist()]
            if any(n.endswith("Manifest.db") for n in names):
                self.fetch_itunes(src)
            else:
                self.fetch_zip(src, prefix=name + "/")
        except Exception as exc:  # noqa: BLE001 - one bad nested archive must not end the stream
            self.notes.append(f"nested zip {name}: {type(exc).__name__}: {exc}")
            print(f"  [nested] {name} failed: {exc}")
        finally:
            path.unlink(missing_ok=True)
            if path in self.spooled:
                self.spooled.remove(path)

    # ---------------------------------------------------------------- tar stream
    def fetch_tar(self, stream, mode: str, prefix: str = "") -> None:
        n = 0
        with tarfile.open(fileobj=stream, mode=mode) as tf:  # type: ignore[call-overload]
            for m in tf:
                if not m.isfile():
                    continue
                name = prefix + m.name.lstrip("./")
                low = name.lower()
                if low.endswith((".tar", ".tar.gz", ".tgz")) and m.size > 1 << 20:
                    # Nested archive (e.g. android_9.tar.gz -> "Google Pixel 3.tar"): stream it in place.
                    print(f"  [nested] {name} ({m.size / 1e9:.2f} GB) -> streaming", flush=True)
                    self.fetch_tar(tf.extractfile(m), "r|" if low.endswith(".tar") else "r|gz", prefix=name + "/")
                    continue
                if low.endswith(".zip") and m.size > 1 << 20:
                    # Nested ZIP (e.g. Cellebrite EXTRACTION_FFS.zip, backup zips) needs random access: spool it.
                    self._fetch_spooled_zip(tf.extractfile(m), name, m.size)
                    continue
                if is_context_doc(name) and m.size < 50 << 20:
                    self._save_doc(name, tf.extractfile(m).read())  # type: ignore[union-attr]
                    continue
                cat = classify(self._rel(name), self.s.platform)
                fh = None
                if cat is None and 4096 <= m.size <= 256 << 20 and "." not in name.rsplit("/", 1)[-1] \
                        and re.search(r"/(data|private/var|mobile)/", "/" + name.lower()):
                    fh = tf.extractfile(m)
                    head = fh.read(16)  # type: ignore[union-attr]
                    if head == SQLITE_MAGIC:
                        cat = "database"
                    else:
                        continue
                if cat is None or not self.budget.accept(cat, m.size):
                    continue
                fh = fh or tf.extractfile(m)
                data = (SQLITE_MAGIC if fh.tell() == 16 else b"") + fh.read()  # type: ignore[union-attr]
                self._write(name, cat, data, float(m.mtime) if m.mtime else None)
                n += 1
                if n % 500 == 0:
                    print(f"  [tar] kept {n} files, {self.budget.bytes / 1e6:.0f} MB", flush=True)
        print(f"  [tar] kept {n} files (skipped: {dict(self.budget.skipped)})")

    # ---------------------------------------------------------------- run
    def run(self) -> dict:
        print(f"[fetch] {self.s.source_id}: {self.s.title} (~{self.s.transfer_gb} GB transfer)")
        self.out.mkdir(parents=True, exist_ok=True)
        self.fetch_url_docs()
        started = time.time()
        try:
            kind, obj = self.resolve()
            if self.s.kind == "itunes":
                self.fetch_itunes(obj)
            elif kind == "zip":
                self.fetch_zip(obj)
            else:
                self.fetch_tar(obj, "r|gz" if self.s.kind == "tar.gz" else "r|")
        finally:
            for p in self.spooled:
                p.unlink(missing_ok=True)
        extract_pdf_text(self.ctx)
        self.records.sort(key=lambda r: r["path"])
        manifest = {**{k: v for k, v in asdict(self.s).items() if k not in ("max_total_bytes", "max_images")},
                    "archive_url": self.s.full_url, "fetched_utc": datetime.now(timezone.utc).isoformat(),
                    "fetch_seconds": round(time.time() - started, 1), "member_count": len(self.records),
                    "skipped": dict(self.budget.skipped), "notes": self.notes, "files": self.records}
        (self.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        return manifest


def extract_pdf_text(ctx: Path) -> None:
    import logging

    from pypdf import PdfReader

    logging.getLogger("pypdf").setLevel(logging.ERROR)  # malformed xref warnings are benign
    for pdf in ctx.glob("*.pdf"):
        txt = pdf.with_suffix(".txt")
        if txt.exists():
            continue
        try:
            reader = PdfReader(str(pdf))
            txt.write_text("\n\n".join(f"--- page {i + 1} ---\n{p.extract_text() or ''}"
                                       for i, p in enumerate(reader.pages)), encoding="utf-8")
        except Exception as exc:  # noqa: BLE001 - unreadable PDF is reported, not fatal
            txt.write_text(f"[text extraction failed: {exc}]", encoding="utf-8")


def fetch_source(source: Source, workers: int = 8) -> dict:
    return _Fetcher(source, workers).run()


# --------------------------------------------------------------------------------------
# Inspection
# --------------------------------------------------------------------------------------
@dataclass
class InspectionReport:
    source_id: str
    counts: dict[str, int] = field(default_factory=dict)
    total_bytes: int = 0
    sqlite_ok: int = 0
    sqlite_failed: list[str] = field(default_factory=list)
    sqlite_partial: list[str] = field(default_factory=list)   # opens & reads, but quick_check flags an FTS/index issue
    sqlite_tables_total: int = 0
    images_ok: int = 0
    images_failed: list[str] = field(default_factory=list)
    images_with_exif: int = 0
    images_with_gps: list[str] = field(default_factory=list)
    text_ok: int = 0
    hash_mismatches: list[str] = field(default_factory=list)
    context_docs: list[str] = field(default_factory=list)

    def summary(self) -> str:
        return (
            f"source={self.source_id} files={sum(self.counts.values())} ({self.total_bytes / 1e6:.0f} MB) by_category={self.counts}\n"
            f"  sqlite: {self.sqlite_ok} ok / {len(self.sqlite_partial)} readable with FTS-index warnings / "
            f"{len(self.sqlite_failed)} corrupt in source image, {self.sqlite_tables_total} tables\n"
            f"  images: {self.images_ok} ok / {len(self.images_failed)} failed, "
            f"{self.images_with_exif} with EXIF, {len(self.images_with_gps)} with GPS\n"
            f"  text logs readable: {self.text_ok}\n"
            f"  hash mismatches after inspection: {len(self.hash_mismatches)}\n"
            f"  case context: {', '.join(self.context_docs) or '-'}"
        )


def _check_sqlite(path: Path, scratch: Path) -> tuple[str, int, str]:
    """Check a working copy. Returns (status, table_count, detail); status in ok|partial|corrupt."""
    copy = working_copy(path, scratch / hashlib.sha1(str(path).encode()).hexdigest()[:16])
    con = sqlite3.connect(f"{copy.resolve().as_uri()}?mode=ro", uri=True)
    register_android_collations(con)
    try:
        tables = con.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]
        result = con.execute("PRAGMA quick_check").fetchone()[0]
        if result == "ok":
            return "ok", tables, ""
        status = "partial" if ("fts" in result.lower() or "inverted index" in result.lower()) else "corrupt"
        return status, tables, result.splitlines()[0][:200]
    except sqlite3.DatabaseError as exc:
        return "corrupt", 0, str(exc).splitlines()[0][:200]
    finally:
        con.close()
        shutil.rmtree(copy.parent, ignore_errors=True)


def inspect_source(source: Source) -> InspectionReport:
    return inspect_path(source_dir(source), source.source_id)


def inspect_path(out: Path, evidence_id: str) -> InspectionReport:
    """Integrity inspection of one evidence directory (mobile source or disk image)."""
    from PIL import Image
    from pillow_heif import register_heif_opener

    register_heif_opener()
    import piexif

    from src.bridge.forensic_bridge import long_path

    out = long_path(Path(out).resolve())   # evidence paths can exceed 260 characters inside case folders
    manifest_path = out / "MANIFEST.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"No manifest at {manifest_path}; fetch {evidence_id} first")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    rep = InspectionReport(evidence_id)
    rep.context_docs = sorted(p.name for p in (out / "_case_context").glob("*")) if (out / "_case_context").exists() else []

    scratch = Path(tempfile.mkdtemp(prefix="forensi_inspect_"))
    for rec in manifest["files"]:
        path = out / rec["path"]
        cat = rec["category"]
        rep.counts[cat] = rep.counts.get(cat, 0) + 1
        rep.total_bytes += rec["size"]
        if not path.exists():
            rep.hash_mismatches.append(rec["path"] + " (missing)")
            continue
        with open(path, "rb") as fh:
            head = fh.read(16)
        if head == SQLITE_MAGIC:
            status, tables, detail = _check_sqlite(path, scratch)
            rep.sqlite_tables_total += tables
            if status == "ok":
                rep.sqlite_ok += 1
            elif status == "partial":
                rep.sqlite_partial.append(f"{rec['path']}: {detail}")
            else:
                # The archive CRC / stream already matched, so corruption is present in the source image.
                rep.sqlite_failed.append(f"{rec['path']}: {detail}")
        elif path.suffix.lower() in IMAGE_EXTS:
            try:
                with Image.open(path) as im:
                    im.verify()
                rep.images_ok += 1
                if path.suffix.lower() in (".jpg", ".jpeg"):
                    exif = piexif.load(str(path))
                    if any(exif.get(k) for k in ("0th", "Exif")):
                        rep.images_with_exif += 1
                    if exif.get("GPS", {}).get(piexif.GPSIFD.GPSLatitude):
                        rep.images_with_gps.append(rec["path"])
            except Exception as exc:  # noqa: BLE001 - any decode failure is a finding
                rep.images_failed.append(f"{rec['path']}: {str(exc)[:160]}")
        elif cat in ("system_log", "app_config", "cloud_record"):
            try:
                path.read_bytes().decode("utf-8")
                rep.text_ok += 1
            except UnicodeDecodeError:
                pass  # binary XML (ABX) / binary plist are valid; not an error
        if sha256_file(path) != rec["sha256"]:
            rep.hash_mismatches.append(rec["path"])

    shutil.rmtree(scratch, ignore_errors=True)
    (out / "INSPECTION.json").write_text(json.dumps(rep.__dict__, indent=2, ensure_ascii=False), encoding="utf-8")
    return rep


# Backwards-compatible name used by earlier code.
inspect_dataset = inspect_source


def assert_source_usable(rep: InspectionReport) -> None:
    problems = []
    if not rep.counts:
        problems.append("no files")
    if rep.hash_mismatches:
        problems.append(f"{len(rep.hash_mismatches)} files changed or missing since acquisition")
    if problems:
        raise RuntimeError(f"{rep.source_id} not usable: " + "; ".join(problems))


def write_corpus_index(reports: list[InspectionReport]) -> dict:
    rows = []
    for rep in reports:
        s = SOURCES[rep.source_id]
        rows.append({
            "source_id": rep.source_id, "title": s.title, "platform": s.platform, "author": s.author,
            "files": sum(rep.counts.values()), "mb": round(rep.total_bytes / 1e6, 1), "sqlite_ok": rep.sqlite_ok,
            "sqlite_tables": rep.sqlite_tables_total, "images": rep.images_ok, "images_gps": len(rep.images_with_gps),
            "screenshots": rep.counts.get("screenshot", 0), "app_images": rep.counts.get("app_image", 0), "context_docs": rep.context_docs,
            "hash_mismatches": len(rep.hash_mismatches),
            "sandbox": (source_dir(s) / "fs").relative_to(DATA_DIR.parent).as_posix(),
        })
    totals = {k: sum(r[k] for r in rows) for k in ("files", "mb", "sqlite_ok", "sqlite_tables", "images", "images_gps", "screenshots")}
    index = {"generated_utc": datetime.now(timezone.utc).isoformat(), "sources": rows, "totals": totals}
    RAW_IMAGES_DIR.mkdir(parents=True, exist_ok=True)
    (RAW_IMAGES_DIR / "CORPUS_INDEX.json").write_text(json.dumps(index, indent=2, ensure_ascii=False), encoding="utf-8")
    return index


def assert_corpus_usable(index: dict) -> None:
    t = index["totals"]
    problems = []
    if t["sqlite_ok"] < 20:
        problems.append(f"only {t['sqlite_ok']} valid SQLite databases (need >= 20)")
    if t["images"] < 5:
        problems.append(f"only {t['images']} valid images")
    if t["images_gps"] < 1:
        problems.append("no geotagged images")
    if t["screenshots"] < 1:
        problems.append("no screenshots")
    if not any(any(d.lower().endswith(".pdf") for d in r["context_docs"]) for r in index["sources"]):
        problems.append("no case-context documentation")
    if problems:
        raise RuntimeError("Corpus not usable: " + "; ".join(problems))


def fetched_sources() -> list[Source]:
    return [s for s in SOURCES.values() if (source_dir(s) / "MANIFEST.json").exists()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sources", "--source", default="light",
                    help="light | heavy | extended | future | all | comma-separated source ids (default: light)")
    ap.add_argument("--inspect-only", action="store_true")
    ap.add_argument("--list-sources", action="store_true")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args(argv)

    if args.list_sources:
        for s in SOURCES.values():
            have = "fetched" if (source_dir(s) / "MANIFEST.json").exists() else ""
            print(f"{s.source_id:22} {s.group:6} {s.platform:8} ~{s.transfer_gb:5.1f} GB  {have:8} {s.title}")
        return 0

    failures = []
    for source in resolve_sources(args.sources):
        if not args.inspect_only:
            try:
                fetch_source(source, workers=args.workers)
            except Exception as exc:  # noqa: BLE001 - keep going with the other sources
                failures.append(f"{source.source_id}: {type(exc).__name__}: {exc}")
                print(f"[error] {failures[-1]}")
                continue
        if (source_dir(source) / "MANIFEST.json").exists():
            rep = inspect_source(source)
            print("[inspect]\n" + rep.summary())
            try:
                assert_source_usable(rep)
            except RuntimeError as exc:
                failures.append(str(exc))

    reports = [inspect_source(s) if not (source_dir(s) / "INSPECTION.json").exists()
               else InspectionReport(**json.loads((source_dir(s) / "INSPECTION.json").read_text(encoding="utf-8")))
               for s in fetched_sources()]
    index = write_corpus_index(reports)
    t = index["totals"]
    print(f"[corpus] {len(reports)} sources, {t['files']} files, {t['mb']:.0f} MB, {t['sqlite_ok']} SQLite DBs "
          f"({t['sqlite_tables']} tables), {t['images']} images ({t['images_gps']} geotagged, {t['screenshots']} screenshots)")
    assert_corpus_usable(index)
    if failures:
        print("[failures]", *failures, sep="\n  ")
        return 1
    print("[ok] corpus verified present, uncorrupted and unmodified")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
