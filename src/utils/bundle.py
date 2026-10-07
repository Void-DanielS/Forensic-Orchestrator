"""Generic evidence bundles: whatever a public case ships, from any host, in any container.

A :class:`RemoteEvidence` is one seized item (or one acquisition set) described by a list of
remote files or folders (see :mod:`src.utils.remote`). The fetcher walks what it finds and
dispatches every piece by what it *is*, not by where it came from:

* **Disk / partition images**: EWF (``.E01``...) or raw (``.dd``, ``.raw``, ``.img``, ``.bin``,
  ``.001``...) segments are grouped and handed to :class:`~src.utils.disk_image.DiskFetcher`
  (The Sleuth Kit). Several images in one bundle go to ``fs/<image name>/`` and share one
  ``FILELIST.jsonl.gz`` (each row tagged ``image``).
* **ZIP**: range-read over HTTP; members are dispatched recursively (an image STORED inside the
  ZIP is still range-read). Split ZIPs (``.z01``) go through 7-Zip.
* **7z / RAR / TAR / gzip / bzip2 / xz / VMDK / VHD(X) / DMG**: spooled (parallel ranges when the
  host allows them) and unpacked with the bundled 7-Zip into ``data/_spool``, then dispatched.
* **Logical trees** (phone extractions, Takeout, collected folders): files are selected with the
  same classifiers as the mobile and disk extractors, plus artefact types the others never see
  as loose files: network captures, memory images, mailboxes, logs, CCTV/video.
* **Documents** shipped with the evidence (scenario, questions, solution, image-creation notes)
  go to ``_case_context/brief`` or ``_case_context/answer_key`` (never into the sandbox).

Executables and credential files are never copied. Every copied file gets a SHA-256 in
``MANIFEST.json``; anything over the size caps is listed under ``not_fetched`` with its source
URL and size, so the record of what the case contains stays complete.
"""
from __future__ import annotations

import collections
import gzip
import hashlib
import json
import os
import re
import shutil
import subprocess
import tarfile
import time
import zipfile
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from src.config.settings import DATA_DIR
from src.utils import dataset_scraper as ds
from src.utils import remote
from src.utils.disk_image import DiskEvidence, DiskFetcher, classify_computer, parallel_download

SEVEN_ZIP = Path(__file__).resolve().parents[2] / "tools" / "7zip" / "7z.exe"
SPOOL = DATA_DIR / "_spool"

EWF_SEG = re.compile(r"^(?P<base>.+)\.(?P<seg>[EeLs][0-9A-Za-z]{2})$")       # .E01 .. .EZZ, .L01, .s01
RAW_SPLIT = re.compile(r"^(?P<base>.+?)\.(?P<seg>\d{3})$")                   # .001 .002 ...
RAW_SPLIT_AA = re.compile(r"^(?P<base>.+?)(?P<seg>[._-]?(?:split-)?a[a-z])$")  # split -a: xaa / .aa
RAW_IMG_EXT = (".dd", ".raw", ".img", ".bin", ".ima", ".image", ".iso", ".mmc", ".emmc", ".nand", ".dump")
ARCHIVE_EXT = (".7z", ".rar", ".tar", ".tgz", ".tbz", ".tbz2", ".txz", ".gz", ".bz2", ".xz", ".zst", ".vmdk", ".vhd",
               ".vhdx", ".dmg", ".zipx", ".cab", ".lzma")
MULTI_ARCHIVE = re.compile(r"^(?P<base>.+\.(?:7z|zip|rar|tar\.gz|tgz|tar))\.(?P<seg>\d{3})$|"
                           r"^(?P<zbase>.+)\.(?P<zseg>z\d\d)$|^(?P<rbase>.+)\.part(?P<rseg>\d+)\.rar$", re.I)
DOC_EXT = (".pdf", ".docx", ".doc", ".txt", ".md", ".rtf", ".html", ".htm", ".xlsx", ".pptx", ".odt")
DOC_HINT = re.compile(r"readme|scenario|question|challenge|description|brief|case|instruction|overview|"
                      r"population|image.?creation|documentation|story|background|task|hint|notes?\b|"
                      r"solution|answer|write.?up|walkthrough|teacher|instructor|report|key", re.I)
ANSWER_HINT = re.compile(r"solution|answer|write.?up|walkthrough|teacher|instructor|key\b|_key|results?/", re.I)
ACQUISITION_HINT = re.compile(r"\.(e01|e02|001|dd|raw|img|l01|ad1)\.txt$|ftk.?imager|acquisition|activity_log|"
                              r"image_info|ewfinfo|verify|hash(es)?\.txt$|\.md5$|\.sha1$", re.I)
EXEC_EXT = (".obb", ".dex", ".odex", ".vdex", ".oat", ".exe", ".dll", ".scr", ".sys", ".msi", ".com", ".bat", ".ps1", ".vbs", ".jar", ".apk", ".elf", ".so",
            ".dylib", ".pyc", ".lnk_")
ARTIFACT_CATEGORIES = {
    "network_capture": (".pcap", ".pcapng", ".cap", ".pcap.gz", ".netflow", ".nfcapd", ".erf"),
    "memory_image": (".mem", ".vmem", ".lime", ".dmp", ".mddramimage", ".winddramimage", ".winddrawramimage",
                     ".crash", ".core", ".aff4"),
    "email": (".pst", ".ost", ".mbox", ".eml", ".msg", ".emlx", ".dbx"),
    "event_log": (".evtx", ".evt"),
    "system_log": (".log", ".syslog", ".journal", ".audit"),
    "database": (".db", ".sqlite", ".sqlite3", ".sqlitedb", ".realm", ".ldb"),
    "user_image": (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".heic", ".webp", ".tif", ".tiff"),
    "video": (".mp4", ".avi", ".mov", ".mkv", ".h264", ".264", ".dav", ".3gp", ".m4v", ".wmv", ".mpg", ".ts"),
    "audio": (".wav", ".mp3", ".m4a", ".amr", ".ogg", ".opus", ".aac"),
    "cloud_record": (".json", ".csv", ".xml", ".plist", ".html", ".htm", ".txt", ".ics", ".vcf", ".kml", ".gpx"),
}
FLASH_NAME = re.compile(r"^(mtd\d+(ro)?|mmcblk\d+(p\d+)?|mtdblock\d+|(user)?data|system|cache|sdcard|emmc|nand)"
                        r"(\.(img|bin|dd|raw|yaffs2?))?$", re.I)
MEMORY_NAME = re.compile(r"mem(ory)?|ram|\.vmem$|lime|dump", re.I)


@dataclass(frozen=True)
class RemoteEvidence:
    """One evidence item assembled from remote files / folders (see module docstring)."""

    evidence_id: str
    title: str
    platform: str                      # windows|linux|macos|android|ios|removable|iot|network|memory|takeout|mixed|generic
    sources: tuple[str, ...]           # file URLs or folder specs understood by remote.list_remote
    include: str = ""                  # regex (case-insensitive) on the relative path; empty = everything
    exclude: str = ""
    device: str = ""
    group: str = "light"               # light | heavy | extended | future
    transfer_gb: float = 0.0
    password: str = ""                 # published archive password, or "file:<relative path>" to read it
    extra_passwords: tuple[str, ...] = ()   # further published passwords (archives of one item may differ)
    answer_key: str = ""               # regex: shipped documents that reveal results (go to answer_key, not brief)
    skip_docs: str = ""                # regex: shipped documents to ignore
    max_total_bytes: int = 1_500_000_000
    max_artifact_bytes: int = 1_000_000_000
    max_images: int = 1500
    max_image_bytes: int = 0           # disk/flash images larger than this are listed in not_fetched (0 = no cap)
    note: str = ""

    @property
    def kind(self) -> str:
        return "bundle"


# --------------------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------------------
def _lower_name(p: str) -> str:
    return p.replace("\\", "/").rsplit("/", 1)[-1].lower()


def segment_key(path: str) -> tuple[str, str, int] | None:
    """(group kind, group key, order) for multi-file images / archives, else None."""
    name = path.replace("\\", "/")
    low = name.lower()
    if (m := MULTI_ARCHIVE.match(name)):
        if m.group("base"):
            return "archive", m.group("base"), int(m.group("seg"))
        if m.group("zbase"):
            return "archive", m.group("zbase") + ".zip", int(m.group("zseg")[1:])
        return "archive", m.group("rbase") + ".rar", int(m.group("rseg"))
    if low.endswith(".zip") or low.endswith(".rar"):
        return "archive", name, 10_000   # last volume of a split set (.z01.. + .zip) or a single archive
    if (m := EWF_SEG.match(name)) and re.fullmatch(r"[Ee](0[1-9]|[1-9]\d)|E[A-Z]{2}|[Ls](0[1-9]|[1-9]\d)",
                                                                 m.group("seg")):
        seg = m.group("seg")
        order = int(seg[1:]) if seg[1:].isdigit() else 100 + (ord(seg[1].upper()) - 65) * 26 + ord(seg[2].upper()) - 65
        return "ewf", m.group("base"), order
    if (m := RAW_SPLIT.match(name)):
        return "raw", m.group("base"), int(m.group("seg"))
    return None


def group_files(files: list) -> list[tuple[str, str, list]]:
    """Group a flat file list ((path, ...) tuples) into items: (kind, name, [members in order])."""
    groups: dict[tuple[str, str], list] = collections.defaultdict(list)
    singles = []
    for f in files:
        k = segment_key(f[0])
        if k:
            groups[(k[0], k[1])].append((k[2], f))
        else:
            singles.append(f)
    items = []
    for (kind, key), members in groups.items():
        members.sort(key=lambda t: t[0])
        items.append((kind, key, [m for _, m in members]))
    for f in singles:
        # Single .dd/.img/.bin files stay "file": they are sniffed first (a .bin can be firmware or RAM).
        low = f[0].lower()
        if low.endswith(ARCHIVE_EXT) or low.endswith((".tar.gz", ".tar.bz2", ".tar.xz")):
            items.append(("archive", f[0], [f]))
        else:
            items.append(("file", f[0], [f]))
    order = {"file": 0, "ewf": 1, "raw": 1, "archive": 2}
    items.sort(key=lambda t: (order[t[0]], t[1]))
    return items


def artifact_category(rel: str, size: int) -> str | None:
    low = rel.lower()
    name = _lower_name(rel)
    if name.endswith(EXEC_EXT) or ds.SENSITIVE.search(name) or name in ("sam", "security", "system.sav"):
        return None
    for cat, exts in ARTIFACT_CATEGORIES.items():
        if name.endswith(exts):
            if cat == "memory_image" and name.endswith(".dmp") and size < 16 << 20:
                return "crash_dump"
            return cat
    if name.endswith((".raw", ".bin")) and MEMORY_NAME.search(name) and size >= 64 << 20:
        return "memory_image"
    if "." not in name and size <= 4 << 20 and "/" in low:
        return "artifact"
    return "artifact" if size <= 32 << 20 and not name.endswith((".iso",)) else None


def sniff(head: bytes) -> str:
    if head.startswith(b"EVF\x09\x0d\x0a\xff\x00") or head.startswith(b"LVF\x09\x0d\x0a\xff\x00"):
        return "ewf"
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06"):
        return "zip"
    if head.startswith(b"7z\xbc\xaf\x27\x1c") or head.startswith(b"Rar!") or head[:3] == b"BZh" \
            or head[:2] == b"\x1f\x8b" or head.startswith(b"\xfd7zXZ\x00") or head[:4] == b"KDMV" \
            or head[:8] == b"vhdxfile" or (len(head) > 262 and head[257:262] == b"ustar"):
        return "archive"
    if head.startswith(b"SQLite format 3\x00"):
        return "sqlite"
    if head.startswith((b"\xd4\xc3\xb2\xa1", b"\xa1\xb2\xc3\xd4", b"\x0a\x0d\x0d\x0a", b"\x4d\x3c\xb2\xa1")):
        return "pcap"
    if len(head) >= 512 and head[510:512] == b"\x55\xaa":
        return "disk"          # MBR / VBR boot signature
    if len(head) >= 1082 and head[1080:1082] == b"\x53\xef":
        return "disk"          # ext2/3/4 superblock
    if head[3:11] in (b"NTFS    ", b"MSDOS5.0", b"EXFAT   ") or head[512:520] == b"EFI PART":
        return "disk"
    return "unknown"


# --------------------------------------------------------------------------------------
# Fetcher
# --------------------------------------------------------------------------------------
class BundleFetcher:
    def __init__(self, ev: RemoteEvidence, out: Path):
        self.ev, self.out = ev, out
        self.fs_root = out / "fs"
        self.ctx = out / "_case_context"
        self.dest = ds._DestMap(self.fs_root)
        self.records: list[dict] = []
        self.not_fetched: list[dict] = []
        self.images: list[dict] = []
        self.filesystems: list[dict] = []
        self.notes: list[str] = []
        self.skipped: collections.Counter = collections.Counter()
        self.bytes = 0
        self.n_images = 0
        self.spool = SPOOL / f"bundle_{ev.evidence_id}"
        self.include = re.compile(ev.include, re.I) if ev.include else None
        self.exclude = re.compile(ev.exclude, re.I) if ev.exclude else None
        self._password: str | None = None
        self._zip_spooled: list[Path] = []
        self.listing = None

    # ---------------------------------------------------------------- selection helpers
    def _wanted(self, rel: str) -> bool:
        if self.include and not self.include.search(rel):
            return False
        return not (self.exclude and self.exclude.search(rel))

    def password(self, files: list[remote.RemoteFile]) -> str | None:
        if self._password is not None or not self.ev.password:
            return self._password
        if self.ev.password.startswith("file:"):
            want = self.ev.password[5:].lower()
            hit = [f for f in files if f.path.lower().endswith(want)]
            if hit:
                import requests
                txt = requests.get(remote.direct_url(hit[0].url), headers=remote.UA, timeout=60).text
                m = re.search(r"password\s*[=:]\s*(\S+)", txt, re.I)
                lines = [ln.strip() for ln in txt.splitlines() if ln.strip()]
                self._password = m.group(1) if m else (lines[0] if lines else "")
        else:
            self._password = self.ev.password
        return self._password

    def _is_doc(self, rel: str, size: int) -> bool:
        name = _lower_name(rel)
        if not name.endswith(DOC_EXT) or size > 60 << 20 or ds.SENSITIVE.search(rel):
            return False
        depth = rel.strip("/").count("/")
        if depth > 3 or re.search(r"/(data|apps?|private|users|library|appdata|sdcard|media)/", "/" + rel.lower()):
            return False   # deep inside an extraction: evidence, not a document shipped with the case
        return bool(DOC_HINT.search(name)) or (depth <= 1 and name.endswith((".pdf", ".docx", ".md", ".rtf")))

    def _save_doc(self, rel: str, data: bytes) -> None:
        if self.ev.skip_docs and re.search(self.ev.skip_docs, rel, re.I):
            return
        if ANSWER_HINT.search(rel) or (self.ev.answer_key and re.search(self.ev.answer_key, rel, re.I)):
            role = "answer_key"
        elif ACQUISITION_HINT.search(rel):
            role = "acquisition"
        else:
            role = "brief"
        name = ds._safe_part(rel.replace("\\", "/").strip("/").replace("/", "__"))[-120:]
        dest = self.ctx / role / name
        try:
            if not dest.exists():
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_bytes(data)
                print(f"  [context:{role}] {name}", flush=True)
        except OSError as exc:   # a context document must never abort an evidence stream
            self.notes.append(f"context document not saved: {rel}: {exc}")

    def _record(self, dest: Path, member: str, cat: str, size: int, sha: str, mtime: float | None,
                original: str | None) -> None:
        rec = {"path": dest.relative_to(self.out).as_posix(), "source_member": member, "category": cat,
               "size": size, "member_mtime": datetime.fromtimestamp(mtime).isoformat() if mtime else None,
               "sha256": sha}
        if original:
            rec["original_path"] = original
        self.records.append(rec)

    def _budget_ok(self, cat: str, size: int, rel: str, url: str = "") -> bool:
        cap = self.ev.max_artifact_bytes if cat in ("memory_image", "network_capture", "email", "video") else 256 << 20
        if size > cap:
            self.not_fetched.append({"path": rel, "category": cat, "size": size, "source": url,
                                     "reason": f"over the {cap / 1e9:.1f} GB cap for {cat}"})
            return False
        if cat in ds.IMAGE_CATEGORIES or cat == "user_image":
            if size < ds.MIN_IMAGE_BYTES:
                self.skipped["tiny_image"] += 1
                return False
            if self.n_images >= self.ev.max_images:
                self.skipped["image_budget"] += 1
                return False
        big = cat in ("memory_image", "network_capture")
        if not big and self.bytes + size > self.ev.max_total_bytes:
            self.skipped["byte_budget"] += 1
            return False
        if cat in ds.IMAGE_CATEGORIES or cat == "user_image":
            self.n_images += 1
        if not big:
            self.bytes += size
        return True

    def _classify(self, rel: str, size: int) -> str | None:
        p = self.ev.platform
        cat = None
        try:
            if p in ("android", "ios", "takeout"):
                cat = ds.classify(rel, p)
            elif p in ("windows", "linux", "macos", "removable", "iot", "chromeos", "generic"):
                cat = classify_computer("/" + rel.replace("\\", "/").lstrip("/"), p, size)
        except ValueError:
            cat = None
        if cat:
            return cat
        fallback = artifact_category(rel, size)
        if p in ("android", "ios", "takeout") and fallback in ("artifact", "cloud_record", "user_image"):
            return None   # phone trees: the mobile selectors already decided; don't keep every small file
        return fallback

    # ---------------------------------------------------------------- writing
    def _write_stream(self, rel: str, member: str, cat: str, reader, size: int, mtime: float | None) -> None:
        dest, original = self.dest.map(rel, member)
        dest.parent.mkdir(parents=True, exist_ok=True)
        tmp = dest.with_name(dest.name + ".part")
        h, n = hashlib.sha256(), 0
        with open(tmp, "wb") as fh:
            while True:
                chunk = reader(8 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                h.update(chunk)
                n += len(chunk)
        if size >= 0 and n != size:
            tmp.unlink(missing_ok=True)
            raise IOError(f"short copy of {rel}: {n}/{size}")
        tmp.replace(dest)
        if mtime:
            os.utime(dest, (mtime, mtime))
        self._record(dest, member, cat, n, h.hexdigest(), mtime, original)

    # ---------------------------------------------------------------- dispatch: remote
    def run_remote(self, files: list[remote.RemoteFile]) -> None:
        items = group_files([(f.path, f) for f in files if self._wanted(f.path)])
        print(f"  [bundle] {len(files)} remote files -> {len(items)} items", flush=True)
        for kind, name, members in items:
            fobjs = [m[1] for m in members]
            try:
                if kind == "file":
                    self._remote_file(fobjs[0])
                elif kind in ("ewf", "raw"):
                    if self._image_too_big(name, fobjs):
                        continue
                    srcs = self._materialize([remote.open_source(f.url, self.spool, self.ev.evidence_id) for f in fobjs],
                                             [f.path for f in fobjs])
                    if not self._try_disk(name, srcs, kind):
                        self._keep_unparsed(name, list(zip([f.path for f in fobjs], srcs)))
                else:
                    self._remote_archive(name, fobjs, files)
            except Exception as exc:  # noqa: BLE001 - one bad item must not lose the rest
                self.notes.append(f"{kind} {name}: {type(exc).__name__}: {str(exc)[:300]}")
                print(f"  [warn] {kind} {name}: {type(exc).__name__}: {str(exc)[:300]}", flush=True)

    def _remote_file(self, f: remote.RemoteFile) -> None:
        size = f.size
        rel = f.path
        if self._is_doc(rel, size if size >= 0 else 0):
            tmp = self.spool / "doc.tmp"
            remote.stream_download(f.url, tmp)
            self._save_doc(rel, tmp.read_bytes())
            tmp.unlink(missing_ok=True)
            return
        src = remote.RemoteSource(f.url)
        size = src.size
        head = src.get(0, min(4096, size)) if src.ranges and size > 0 else b""
        kind = sniff(head) if head else "unknown"
        if self._looks_like_disk(rel, kind, size):
            if self._image_too_big(rel, [remote.RemoteFile(rel, f.url, size)]):
                return
            srcs = self._materialize([src if src.ranges else remote.open_source(f.url, self.spool, self.ev.evidence_id)],
                                     [rel])
            if self._try_disk(rel, srcs, "ewf" if kind == "ewf" else "raw"):
                return
            self._keep_unparsed(rel, [(rel, srcs[0])])
            return
        if kind == "zip":
            self._remote_archive(rel, [f], [])
            return
        if kind == "archive":
            self._remote_archive(rel, [f], [])
            return
        cat = self._classify(rel, size)
        if kind == "sqlite":
            cat = cat if cat and "database" in cat else "database"
        if kind == "pcap":
            cat = "network_capture"
        if cat is None:
            self.skipped["not_selected"] += 1
            return
        if not self._budget_ok(cat, size, rel, f.url):
            return
        if src.ranges and size > 64 << 20:
            tmp = self.spool / f"art_{hashlib.sha1(rel.encode()).hexdigest()[:10]}"
            parallel_download(src, tmp, label=rel)
            with open(tmp, "rb") as fh:
                self._write_stream(rel, f.url, cat, fh.read, size, None)
            tmp.unlink(missing_ok=True)
        else:
            import requests
            with requests.get(remote.direct_url(f.url), stream=True, timeout=180, headers=remote.UA) as r:
                r.raise_for_status()
                self._write_stream(rel, f.url, cat, r.raw.read, size, None)
        print(f"  [artifact] {cat:16} {size / 1e6:9.1f} MB  {rel}", flush=True)

    def _remote_archive(self, name: str, fobjs: list[remote.RemoteFile], all_files: list) -> None:
        low = name.lower()
        if len(fobjs) == 1:
            src = remote.RemoteSource(fobjs[0].url)
            head = src.get(0, min(512 * 2, src.size)) if src.ranges else b""
            if (low.endswith(".zip") or head.startswith(b"PK\x03\x04")) and src.ranges:
                self._zip(name, src)
                return
            if low.endswith((".tar", ".tar.gz", ".tgz")) or (len(head) > 262 and head[257:262] == b"ustar"):
                gz = low.endswith((".gz", ".tgz")) or head[:2] == b"\x1f\x8b"
                stream = ds._PrefetchReader(src, name) if src.ranges else self._http_stream(fobjs[0].url)
                try:
                    self._tar_stream(self._base(name), stream, "r|gz" if gz else "r|")
                    return
                except tarfile.ReadError as exc:
                    self.notes.append(f"{name}: not a streamable tar ({exc}); unpacking with 7-Zip")
        # Spool every volume, then let 7-Zip unpack (only) the members worth keeping.
        self.spool.mkdir(parents=True, exist_ok=True)
        srcs = [remote.RemoteSource(f.url) for f in fobjs]
        self._ensure_space(sum(s.size for s in srcs) * 1.3, name)
        local = []
        for f, src in zip(fobjs, srcs):
            dest = self.spool / ds._safe_part(_lower_name(f.path) or src.name)
            if not (dest.exists() and dest.stat().st_size == src.size and not dest.with_name(dest.name + ".todo").exists()):
                if src.ranges:
                    parallel_download(src, dest, label=f.path)
                else:
                    remote.stream_download(f.url, dest, src.size)
            local.append(dest)
        try:
            self._unpack_local(name, local, all_files)
        finally:
            for p in local:
                p.unlink(missing_ok=True)

    @staticmethod
    def _http_stream(url: str):
        import requests
        r = requests.get(remote.direct_url(url), stream=True, timeout=180, headers=remote.UA)
        r.raise_for_status()
        return r.raw

    @staticmethod
    def _base(name: str) -> str:
        return re.sub(r"\.(7z|zip|rar|tar|tgz|gz|bz2|xz|zst|tar\.gz|tar\.bz2|tar\.xz|vmdk|vhdx?|dmg)(\.\d{3})?$", "",
                      name.replace("\\", "/"), flags=re.I).strip("/") + "/"

    def _ensure_space(self, need: float, what: str) -> None:
        SPOOL.mkdir(parents=True, exist_ok=True)
        free = shutil.disk_usage(SPOOL).free
        if free < need + 25e9:
            raise RuntimeError(f"not enough disk for {what}: need {need / 1e9:.1f} GB + 25 GB reserve, "
                               f"{free / 1e9:.0f} GB free")

    def _want_member(self, rel: str, size: int) -> bool:
        """Is an archive member worth extracting (before knowing its content)?"""
        low = rel.lower()
        if size == 0 or ds.SENSITIVE.search(low) and not low.endswith(DOC_EXT):
            return False
        if size < 0:
            return not low.endswith(EXEC_EXT)   # single-stream .bz2/.gz/.xz: 7-Zip cannot tell the size
        if self._is_doc(rel, size) or segment_key(rel) or low.endswith(ARCHIVE_EXT + RAW_IMG_EXT + (".zip", ".e01")):
            return True
        if size >= 16 << 20 and "." not in _lower_name(rel)[-6:]:
            return True     # big extensionless member: possibly an image
        return self._classify(rel, size) is not None

    # ---------------------------------------------------------------- TAR (streamed, never spooled whole)
    def _tar_stream(self, prefix: str, stream, mode: str) -> None:
        deferred: list[tuple[str, Path]] = []
        n_kept = 0
        with tarfile.open(fileobj=stream, mode=mode) as tf:  # type: ignore[call-overload]
            for m in tf:
                if not m.isfile() or m.size == 0:
                    continue
                rel = prefix + m.name.lstrip("./")
                low = rel.lower()
                if low.endswith(EXEC_EXT):
                    self.skipped["executable"] += 1
                    continue
                if low.endswith((".tar", ".tar.gz", ".tgz")) and m.size > 1 << 20:
                    print(f"  [tar] nested {rel} ({m.size / 1e9:.2f} GB) -> streaming", flush=True)
                    self._tar_stream(self._base(rel), tf.extractfile(m), "r|" if low.endswith(".tar") else "r|gz")
                    continue
                if self._is_doc(rel, m.size):
                    self._save_doc(rel, tf.extractfile(m).read())  # type: ignore[union-attr]
                    continue
                fh = tf.extractfile(m)
                head = fh.read(4096)  # type: ignore[union-attr]
                k = sniff(head)
                container = (k in ("zip", "archive") and m.size >= 1 << 20) or k == "ewf" \
                    or segment_key(rel) is not None and k not in ("sqlite", "unknown") \
                    or (k == "disk" or low.endswith(RAW_IMG_EXT)) and m.size >= 16 << 20
                if container:
                    self._ensure_space(m.size * 1.2, rel)
                    dest = self.spool / "tar" / hashlib.sha1(rel.encode()).hexdigest()[:12] / ds._safe_part(_lower_name(rel))
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    print(f"  [tar] spooling {rel} ({m.size / 1e9:.2f} GB, {k})", flush=True)
                    with open(dest, "wb") as out:
                        out.write(head)
                        shutil.copyfileobj(fh, out, 8 << 20)
                    deferred.append((rel, dest))
                    continue
                cat = self._classify(rel, m.size)
                if k == "sqlite":
                    cat = cat if cat and "database" in cat else "database"
                elif k == "pcap":
                    cat = "network_capture"
                if cat is None:
                    self.skipped["not_selected"] += 1
                    continue
                if not self._budget_ok(cat, m.size, rel):
                    continue
                first = [head]

                def reader(n, fh=fh, first=first):
                    if first:
                        return first.pop()
                    return fh.read(n)  # type: ignore[union-attr]

                self._write_stream(rel, rel, cat, reader, m.size, float(m.mtime) if m.mtime else None)
                n_kept += 1
                if n_kept % 500 == 0:
                    print(f"  [tar] kept {n_kept} files, {self.bytes / 1e6:.0f} MB", flush=True)
        print(f"  [tar] {prefix}: kept {n_kept} files; {len(deferred)} containers to open", flush=True)
        if deferred:
            self._dispatch_local(deferred)
            for _, p in deferred:
                shutil.rmtree(p.parent, ignore_errors=True)

    # ---------------------------------------------------------------- ZIP over ranges
    def _zip(self, name: str, src, depth: int = 0) -> None:
        zf = ds._open_zip(src)
        infos = [i for i in zf.infolist() if not i.is_dir() and "__MACOSX" not in i.filename]
        if not isinstance(src, ds.FileSource) and len(infos) > 50 and src.size <= 6_000_000_000 \
                and shutil.disk_usage(SPOOL).free > src.size + 40e9:
            # Many members over HTTP: one parallel download beats hundreds of small range requests.
            dest = self.spool / f"dl_{hashlib.sha1(name.encode()).hexdigest()[:10]}.zip"
            parallel_download(src, dest, label=name)
            try:
                self._zip(name, ds.FileSource(dest), depth)
            finally:
                dest.unlink(missing_ok=True)
            return
        prefix = self._base(name)
        if any(i.flag_bits & 0x1 or i.compress_type not in (zipfile.ZIP_STORED, zipfile.ZIP_DEFLATED) for i in infos):
            # Encrypted or Deflate64/BZIP2/LZMA members cannot be range-read by zipfile: unpack with 7-Zip.
            print(f"  [zip] {name}: encrypted / non-deflate members -> 7-Zip", flush=True)
            if isinstance(src, ds.FileSource):
                self._unpack_local(name, [src.path], [])
            else:
                self._ensure_space(src.size * 1.3, name)
                dest = self.spool / ds._safe_part(_lower_name(name) or "archive.zip")
                parallel_download(src, dest, label=name)
                try:
                    self._unpack_local(name, [dest], [])
                finally:
                    dest.unlink(missing_ok=True)
            return
        print(f"  [zip] {name}: {len(infos)} members ({src.size / 1e9:.2f} GB)", flush=True)
        groups = group_files([(i.filename, i) for i in infos])
        for kind, gname, members in groups:
            infos_g = [m[1] for m in members]
            rel = prefix + gname
            mark = len(self._zip_spooled)   # files spooled by *this* member only (outer levels keep theirs)
            if _lower_name(gname).endswith(EXEC_EXT):
                self.skipped["executable"] += 1
                continue
            try:
                if kind in ("ewf", "raw"):
                    srcs = [self._zip_member_source(src, i) for i in infos_g]
                    if not self._try_disk(rel, srcs, kind):
                        self._keep_unparsed(rel, [(prefix + i.filename, s) for i, s in zip(infos_g, srcs)])
                elif kind == "archive":
                    if len(infos_g) == 1 and infos_g[0].filename.lower().endswith(".zip"):
                        self._zip(rel, self._zip_member_source(src, infos_g[0]), depth + 1)
                    elif len(infos_g) == 1 and re.search(r"\.(tar|tar\.gz|tgz)$", infos_g[0].filename, re.I):
                        i = infos_g[0]
                        self._tar_stream(self._base(rel), ds._member_stream(src, i, self.ev.evidence_id),
                                         "r|" if i.filename.lower().endswith(".tar") else "r|gz")
                    else:
                        local = [self._zip_spool(src, i) for i in infos_g]
                        try:
                            self._unpack_local(rel, local, [])
                        finally:
                            for p in local:
                                p.unlink(missing_ok=True)
                else:
                    self._zip_member(src, infos_g[0], prefix)
            except Exception as exc:  # noqa: BLE001
                self.notes.append(f"zip member {rel}: {type(exc).__name__}: {str(exc)[:300]}")
                print(f"  [warn] zip member {rel}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)
            finally:
                for p in self._zip_spooled[mark:]:
                    p.unlink(missing_ok=True)
                del self._zip_spooled[mark:]

    def _zip_member_source(self, src, info: zipfile.ZipInfo):
        if info.compress_type == zipfile.ZIP_STORED:
            return ds.SubSource(src, ds._data_offset(src, info), info.file_size)
        path = self._zip_spool(src, info)
        self._zip_spooled.append(path)
        return ds.FileSource(path)

    def _zip_spool(self, src, info: zipfile.ZipInfo) -> Path:
        self._ensure_space(info.file_size * 1.2, info.filename)
        self.spool.mkdir(parents=True, exist_ok=True)
        path = self.spool / f"z{hashlib.sha1(info.filename.encode()).hexdigest()[:8]}_{ds._safe_part(_lower_name(info.filename))}"
        if not (path.exists() and path.stat().st_size == info.file_size):
            print(f"  [spool] {info.filename} ({info.file_size / 1e9:.2f} GB)", flush=True)
            tmp = path.with_name(path.name + ".part")
            with open(tmp, "wb") as fh:
                shutil.copyfileobj(ds._member_stream(src, info, self.ev.evidence_id), fh, 8 << 20)
            tmp.replace(path)
        return path

    def _zip_member(self, src, info: zipfile.ZipInfo, prefix: str) -> None:
        rel = prefix + info.filename
        if self._is_doc(info.filename, info.file_size):
            self._save_doc(rel, ds._read_member(src, info))
            return
        k = "unknown"
        if info.file_size >= 4096:
            head = ds._member_stream(src, info, "sniff").read(4096) if info.compress_type else \
                src.get(ds._data_offset(src, info), 4096)
            k = sniff(head)
            if self._looks_like_disk(rel, k, info.file_size):
                s = self._zip_member_source(src, info)
                if not self._try_disk(rel, [s], "ewf" if k == "ewf" else "raw"):
                    self._keep_unparsed(rel, [(rel, s)])
                return
            if k == "zip" and info.file_size > 1 << 20:
                self._zip(rel, self._zip_member_source(src, info))
                return
            if k == "archive" and info.file_size > 1 << 20:
                if head[:2] == b"\x1f\x8b" or len(head) > 262 and head[257:262] == b"ustar":
                    try:
                        self._tar_stream(self._base(rel), ds._member_stream(src, info, self.ev.evidence_id),
                                         "r|gz" if head[:2] == b"\x1f\x8b" else "r|")
                        return
                    except tarfile.ReadError:
                        pass
                local = self._zip_spool(src, info)
                try:
                    self._unpack_local(rel, [local], [])
                finally:
                    local.unlink(missing_ok=True)
                return
        cat = self._classify(rel, info.file_size)
        if k == "sqlite":
            cat = cat if cat and "database" in cat else "database"
        elif k == "pcap":
            cat = "network_capture"
        if cat is None:
            self.skipped["not_selected"] += 1
            return
        if not self._budget_ok(cat, info.file_size, rel, ""):
            return
        mtime = datetime(*info.date_time).timestamp() if info.date_time[0] >= 1980 else None
        stream = ds._member_stream(src, info, self.ev.evidence_id)
        self._write_stream(rel, rel, cat, stream.read, info.file_size, mtime)

    # ---------------------------------------------------------------- 7-Zip (7z, rar, split zip, gz/bz2/xz, vmdk...)
    def _7z_list(self, archive: Path, pw: str) -> list[tuple[str, int]]:
        r = subprocess.run([str(SEVEN_ZIP), "l", "-slt", "-sccUTF-8", f"-p{pw}", str(archive)],
                           capture_output=True, text=True, encoding="utf-8", errors="replace")
        if r.returncode not in (0, 1):
            raise RuntimeError(f"7-Zip cannot list {archive.name} ({r.returncode}): "
                               f"{(r.stderr or r.stdout).strip()[-300:]}")
        entries, cur = [], {}
        body = r.stdout.split("----------", 1)[-1]
        for line in body.splitlines() + [""]:
            if not line.strip():
                if cur and "Path" not in cur and ("Size" in cur or "Packed Size" in cur):
                    # single-stream .bz2/.gz/.xz: the member is the archive name minus its extension
                    cur["Path"] = re.sub(r"\.(bz2|gz|xz|zst|lzma|z)$", "", archive.name, flags=re.I) \
                        if re.search(r"\.(bz2|gz|xz|zst|lzma|z)$", archive.name, re.I) else archive.stem
                if cur.get("Path") and "D" not in cur.get("Attributes", "") and cur.get("Folder") != "+":
                    entries.append((cur["Path"], int(cur["Size"]) if cur.get("Size", "").isdigit() else -1))
                cur = {}
                continue
            k, _, v = line.partition(" = ")
            cur[k.strip()] = v
        return entries

    def _unpack_local(self, name: str, volumes: list[Path], all_files: list) -> None:
        target = self.spool / ("x_" + hashlib.sha1(name.encode()).hexdigest()[:10])
        shutil.rmtree(target, ignore_errors=True)
        target.mkdir(parents=True)
        first = sorted(volumes, key=lambda p: (not re.search(r"\.(001|z01|part0*1\.rar)$", p.name, re.I), p.name))[0]
        if len(volumes) > 1 and any(v.name.lower().endswith(".zip") for v in volumes):
            first = [v for v in volumes if v.name.lower().endswith(".zip")][0]
        first_pw = self.password([f for f in all_files if isinstance(f, remote.RemoteFile)]) or ""
        candidates = list(dict.fromkeys([first_pw, *self.ev.extra_passwords, ""]))
        for i, pw in enumerate(candidates):
            try:
                entries = self._7z_list(first, pw)
            except RuntimeError:
                if i == len(candidates) - 1:
                    raise
                continue
            wanted = [p for p, size in entries if self._want_member(p, size)]
            need = sum(max(size, 0) for p, size in entries if p in set(wanted))
            self._ensure_space(need, name)
            print(f"  [7z] {name}: {len(entries)} members, extracting {len(wanted)} ({need / 1e9:.2f} GB)", flush=True)
            if not wanted:
                shutil.rmtree(target, ignore_errors=True)
                return
            cmd = [str(SEVEN_ZIP), "x", "-y", "-bso0", "-bsp0", "-sccUTF-8", f"-o{target}", f"-p{pw}", str(first)]
            if len(wanted) < len(entries):
                listfile = self.spool / f"list_{hashlib.sha1(name.encode()).hexdigest()[:8]}.txt"
                listfile.write_text("\n".join(wanted), encoding="utf-8")
                cmd += ["-scsUTF-8", f"-i@{listfile}"]
            r = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace")
            if r.returncode == 2 and "password" in (r.stderr + r.stdout).lower() and i < len(candidates) - 1:
                shutil.rmtree(target, ignore_errors=True)
                target.mkdir(parents=True)
                continue   # wrong password for this archive: try the next published one
            break
        if r.returncode not in (0, 1):   # 1 = warnings (e.g. a few unreadable entries)
            shutil.rmtree(target, ignore_errors=True)
            raise RuntimeError(f"7-Zip failed ({r.returncode}): {(r.stderr or r.stdout).strip()[-300:]}")
        if r.returncode == 1:
            self.notes.append(f"7-Zip warnings for {name}: {(r.stderr or '').strip()[:200]}")
        files = [(self._base(name) + p.relative_to(target).as_posix(), p) for p in target.rglob("*") if p.is_file()]
        n_records, n_missing = len(self.records), len(self.not_fetched)
        try:
            self._dispatch_local(files)
        finally:
            shutil.rmtree(target, ignore_errors=True)
        # Members too big to keep unpacked (sparse memory dumps, unallocated-space blobs) are often tiny
        # compressed: keep the distribution archive itself, byte for byte, as the evidence.
        skipped = self.not_fetched[n_missing:]
        packed = sum(v.stat().st_size for v in volumes)
        if skipped and len(self.records) == n_records and packed <= self.ev.max_artifact_bytes:
            for v in volumes:
                rel = self._base(name).rstrip("/") + "__archive/" + v.name
                with open(v, "rb") as fh:
                    self._write_stream(rel, name, "compressed_evidence", fh.read, v.stat().st_size, None)
            for nf in skipped:
                nf["kept_compressed_as"] = self._base(name).rstrip("/") + "__archive/"
            print(f"  [archive] kept {name} compressed ({packed / 1e6:.0f} MB): members too large to store unpacked",
                  flush=True)

    def _dispatch_local(self, files: list[tuple[str, Path]]) -> None:
        items = group_files(files)
        if len(files) > 1:
            print(f"  [tree] {len(files)} files -> {len(items)} items", flush=True)
        for kind, name, members in items:
            paths = [m[1] for m in members]
            try:
                if kind in ("ewf", "raw"):
                    srcs = [ds.FileSource(p) for p in paths]
                    if not self._try_disk(name, srcs, kind):
                        self._keep_unparsed(name, [(m[0], s) for m, s in zip(members, srcs)])
                elif kind == "archive":
                    self._local_archive(name, paths)
                else:
                    self._local_file(paths[0], name)
            except Exception as exc:  # noqa: BLE001
                self.notes.append(f"{kind} {name}: {type(exc).__name__}: {str(exc)[:300]}")
                print(f"  [warn] {kind} {name}: {type(exc).__name__}: {str(exc)[:200]}", flush=True)

    def _local_archive(self, name: str, paths: list[Path]) -> None:
        low = name.lower()
        with open(paths[0], "rb") as fh:
            head = fh.read(1024)
        if len(paths) == 1 and head.startswith(b"PK\x03\x04"):
            self._zip(name, ds.FileSource(paths[0]))
            return
        if len(paths) == 1 and (low.endswith((".tar", ".tar.gz", ".tgz")) or len(head) > 262 and head[257:262] == b"ustar"):
            try:
                with open(paths[0], "rb") as fh:
                    self._tar_stream(self._base(name), fh, "r|gz" if head[:2] == b"\x1f\x8b" else "r|")
                return
            except tarfile.ReadError:
                pass
        self._unpack_local(name, paths, [])

    def _local_file(self, path: Path, rel: str) -> None:
        size = path.stat().st_size
        if rel.lower().endswith(EXEC_EXT):
            self.skipped["executable"] += 1
            return
        if self._is_doc(rel, size):
            self._save_doc(rel, path.read_bytes())
            return
        if size == 0:
            return
        with open(path, "rb") as fh:
            head = fh.read(4096)
        k = sniff(head)
        if self._looks_like_disk(rel, k, size):
            src = ds.FileSource(path)
            if not self._try_disk(rel, [src], "ewf" if k == "ewf" else "raw"):
                self._keep_unparsed(rel, [(rel, src)])
            return
        if k in ("zip", "archive") and size > 1 << 20:
            self._local_archive(rel, [path])
            return
        cat = self._classify(rel, size)
        if k == "sqlite":
            cat = cat if cat and "database" in cat else "database"
        elif k == "pcap":
            cat = "network_capture"
        if cat is None:
            self.skipped["not_selected"] += 1
            if size >= 256 << 20:   # big unknown blobs (unallocated space, raw dumps) are listed, not silently lost
                self.not_fetched.append({"path": rel, "category": "unknown_blob", "size": size, "source": rel,
                                         "reason": "large file of unrecognised type"})
            return
        if not self._budget_ok(cat, size, rel):
            return
        with open(path, "rb") as fh:
            self._write_stream(rel, rel, cat, fh.read, size, path.stat().st_mtime)

    # ---------------------------------------------------------------- disk images
    def _image_too_big(self, name: str, fobjs: list) -> bool:
        cap = self.ev.max_image_bytes
        if not cap:
            return False
        sizes = [f.size if f.size >= 0 else remote.RemoteSource(f.url).size for f in fobjs]
        if sum(sizes) <= cap:
            return False
        self.not_fetched.append({"path": name, "category": "disk_image", "size": sum(sizes),
                                 "source": [f.url for f in fobjs],
                                 "reason": f"image over the {cap / 1e9:.0f} GB per-image cap of this evidence item"})
        print(f"  [skip] {name}: {sum(sizes) / 1e9:.1f} GB image over cap (listed in not_fetched)", flush=True)
        return True

    def _materialize(self, srcs: list, names: list[str]) -> list:
        """Download remote image segments to the spool (parallel ranges) unless the image is huge.

        A file-system walk touches scattered clusters; below ~45 GB a parallel download is far faster
        than random range reads. Larger images are read in place."""
        from src.utils.disk_image import SPOOL_MAX_BYTES

        remote_srcs = [s for s in srcs if isinstance(s, remote.RemoteSource)]
        total = sum(s.size for s in srcs)
        if not remote_srcs or total > SPOOL_MAX_BYTES or shutil.disk_usage(SPOOL).free < total + 40e9:
            return srcs
        out = []
        for s, n in zip(srcs, names):
            if not isinstance(s, remote.RemoteSource):
                out.append(s)
                continue
            dest = self.spool / "img" / ds._safe_part(_lower_name(n) or s.name)
            dest.parent.mkdir(parents=True, exist_ok=True)
            if not (dest.exists() and dest.stat().st_size == s.size and not dest.with_name(dest.name + ".todo").exists()):
                parallel_download(s, dest, label=n)
            fs = ds.FileSource(dest)
            fs.landing = s.landing
            out.append(fs)
        return out

    @staticmethod
    def _looks_like_disk(rel: str, kind: str, size: int) -> bool:
        name = _lower_name(rel)
        if kind == "ewf":
            return True
        if FLASH_NAME.search(name) and size >= 64 << 10:
            return True     # Android partition / NAND dumps: yaffs2 / ext4 / raw flash, often extensionless
        if size < 16 << 20 or MEMORY_NAME.search(name) and not name.endswith((".dd", ".img", ".e01")):
            return False
        return kind == "disk" or name.endswith(RAW_IMG_EXT)

    def _try_disk(self, rel: str, sources: list, kind: str) -> bool:
        """Disk extraction; False (fall back to artefact handling) when TSK finds no file system."""
        try:
            self._disk(rel, sources, kind)
            return True
        except RuntimeError as exc:
            if "no file system" not in str(exc):
                raise
            self.notes.append(f"{rel}: no file system recognised by The Sleuth Kit (kept as a raw artefact if "
                              "within the size caps)")
            return False

    def _keep_unparsed(self, name: str, parts: list) -> None:
        """An image without a recognisable file system (wiped, chip-off NAND, carving set) is evidence in
        itself: keep its segments byte-for-byte when they fit the cap, so carving tools can work on them."""
        total = sum(s.size for _, s in parts)
        if total > self.ev.max_artifact_bytes:
            self.not_fetched.append({"path": name, "category": "disk_image_unparsed", "size": total,
                                     "source": [getattr(s, "landing", "") or getattr(s, "url", "") for _, s in parts],
                                     "reason": "no file system recognised and larger than the raw-image cap"})
            return
        for rel, src in parts:
            pos = 0

            def reader(n, src=src):
                nonlocal pos
                data = src.get(pos, min(n, src.size - pos)) if pos < src.size else b""
                pos += len(data)
                return data

            self._write_stream(rel, rel, "disk_image_unparsed", reader, src.size, None)
            print(f"  [raw image] kept {rel} ({src.size / 1e6:.1f} MB)", flush=True)

    def _disk(self, name: str, sources: list, kind: str) -> None:
        prefix = ds._safe_part(re.sub(r"\.(e01|001|dd|raw|img|bin|iso)$", "", name.replace("\\", "/").split("/")[-1],
                                      flags=re.I))[:60] or "image"
        n = 2
        while any(i["prefix"] == prefix for i in self.images):
            prefix = f"{prefix}_{n}"
            n += 1
        size = sum(s.size for s in sources)
        print(f"  [disk] {name}: {kind}, {len(sources)} segment(s), {size / 1e9:.2f} GB -> fs/{prefix}/", flush=True)
        platform = self.ev.platform if self.ev.platform not in ("mixed", "network", "memory", "takeout") else "auto"
        dev = DiskEvidence(self.ev.evidence_id, f"{self.ev.title}: {name}", platform, fmt="ewf" if kind == "ewf" else "raw",
                           max_total_bytes=max(50_000_000, self.ev.max_total_bytes - self.bytes),
                           max_images=max(0, self.ev.max_images - self.n_images))
        fetcher = DiskFetcher(dev, self.out, prefix=prefix, sources=sources)
        try:
            info, fs_meta = fetcher.extract(self.listing)
        finally:
            self.records += fetcher.records
            self.notes += fetcher.notes
            self.skipped.update({f"{prefix}:{k}": v for k, v in fetcher.skipped.items()})
        self.bytes += sum(r["size"] for r in fetcher.records if not r.get("deleted"))
        self.n_images += sum(1 for r in fetcher.records if r["category"] in ("user_image", "app_image", "screenshot"))
        self.images.append({"prefix": prefix, "name": name, "kind": kind, "segments": len(sources), "bytes": size,
                            **{k: v for k, v in info.items() if k != "headers"}, "headers": info.get("headers")})
        self.filesystems += fs_meta

    # ---------------------------------------------------------------- main
    def run(self) -> dict:
        started = time.time()
        self.out.mkdir(parents=True, exist_ok=True)
        self.spool.mkdir(parents=True, exist_ok=True)
        print(f"[fetch] {self.ev.evidence_id}: {self.ev.title}", flush=True)
        files: list[remote.RemoteFile] = []
        for spec in self.ev.sources:
            got = remote.list_remote(spec)
            if not got:
                self.notes.append(f"empty source listing: {spec}")
            files += got
        files = [f for f in files if not ds.SENSITIVE.search(f.path) or self.ev.password.startswith("file:")
                 and f.path.lower().endswith(self.ev.password[5:].lower())]
        self.password(files)
        listing_path = self.out / "FILELIST.jsonl.gz"
        ok = False
        try:
            with gzip.open(listing_path, "wt", encoding="utf-8") as self.listing:
                self.run_remote([f for f in files if not (self.ev.password.startswith("file:")
                                                          and f.path.lower().endswith(self.ev.password[5:].lower()))])
            ok = True
        finally:
            if ok:
                shutil.rmtree(self.spool, ignore_errors=True)
        if not self.images:
            listing_path.unlink(missing_ok=True)
        ds.extract_pdf_text(self.ctx / "brief") if (self.ctx / "brief").is_dir() else None
        if not self.records:
            raise RuntimeError(f"nothing extracted for {self.ev.evidence_id}: {self.notes[-3:] or 'no files selected'}")
        self.records.sort(key=lambda r: r["path"])
        spec = {k: v for k, v in asdict(self.ev).items() if k not in ("password", "extra_passwords")}
        spec["password_protected"] = bool(self.ev.password)
        manifest = {"evidence_id": self.ev.evidence_id, "title": self.ev.title, "platform": self.ev.platform,
                    "kind": "bundle", "spec": spec, "images": self.images, "filesystems": self.filesystems,
                    "fetched_utc": datetime.now(timezone.utc).isoformat(),
                    "fetch_seconds": round(time.time() - started, 1), "member_count": len(self.records),
                    "skipped": dict(self.skipped), "not_fetched": self.not_fetched, "notes": self.notes,
                    "files": self.records}
        (self.out / "MANIFEST.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
        cats = collections.Counter(r["category"] for r in self.records)
        print(f"  [done] {len(self.records)} files, {sum(r['size'] for r in self.records) / 1e6:.0f} MB, "
              f"{len(self.images)} disk image(s), {dict(cats.most_common(8))}; not fetched: {len(self.not_fetched)}",
              flush=True)
        return manifest


def fetch_bundle(ev: RemoteEvidence, out: Path) -> dict:
    return BundleFetcher(ev, out).run()
