"""Shared vocabulary of the FDX contracts: identifiers, controlled vocabularies, timestamps.

Conventions (see ``docs/DATA_INTERCHANGE_STANDARD.md`` §2):

* **Paths** inside a record are *sandbox-relative* POSIX paths (``data/media/0/DCIM/x.jpg``); the sandbox
  is ``data/cases/<case_id>/evidence/<evidence_id>/fs``. :class:`ArtifactRef` carries the three parts and
  converts to/from the ``data/cases``-relative form used by the dataset tables.
* **Hashes** are lowercase hex SHA-256 of the whole file.
* **Times** are :class:`ForensicTimestamp`: the raw stored value, its epoch kind, and the UTC ISO-8601
  instant when the source defines one. Local-only times (EXIF without offset) never get a guessed UTC.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, ConfigDict, Field, field_validator

SCHEMA_VERSION = "fdx/1.0"

SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class Platform(str, Enum):
    ANDROID = "android"
    IOS = "ios"
    WINDOWS = "windows"
    LINUX = "linux"
    MACOS = "macos"
    REMOVABLE = "removable"
    TAKEOUT = "takeout"     # cloud export (Google Takeout and similar)
    UNKNOWN = ""


class EvidenceKind(str, Enum):
    MOBILE_EXTRACTION = "mobile_extraction"   # logical / full-file-system phone extraction
    DISK_IMAGE = "disk_image"                 # E01 / raw image walked with The Sleuth Kit
    CLOUD_EXPORT = "cloud_export"


class FileCategory(str, Enum):
    """Category assigned at extraction (``dataset_scraper.classify`` / ``disk_image.classify_computer``)."""
    APP_DATABASE = "app_database"
    DATABASE = "database"
    APP_CONFIG = "app_config"
    APP_IMAGE = "app_image"
    USER_IMAGE = "user_image"
    SCREENSHOT = "screenshot"
    THUMBNAIL_CACHE = "thumbnail_cache"
    USER_DOCUMENT = "user_document"
    EMAIL = "email"
    SYSTEM_LOG = "system_log"
    EVENT_LOG = "event_log"
    REGISTRY = "registry"
    PREFETCH = "prefetch"
    SHORTCUT = "shortcut"
    BROWSER_HISTORY = "browser_history"
    RECYCLE_BIN = "recycle_bin"
    REMOVABLE_FILE = "removable_file"
    CLOUD_RECORD = "cloud_record"
    FS_METADATA = "fs_metadata"
    BACKUP_META = "backup_meta"
    SHELL_HISTORY = "shell_history"


class Split(str, Enum):
    TRAIN = "train"
    VAL = "val"
    TEST = "test"


class ArtifactRef(BaseModel):
    """A file inside one evidence item's sandbox, pinned by its hash."""
    model_config = ConfigDict(frozen=True)

    case_id: str
    evidence_id: str
    path: str = Field(description="sandbox-relative POSIX path")
    sha256: Optional[str] = None

    @field_validator("path")
    @classmethod
    def _relative_posix(cls, v: str) -> str:
        if v.startswith(("/", "\\")) or re.match(r"^[A-Za-z]:", v) or "\\" in v or ".." in v.split("/"):
            raise ValueError(f"path must be sandbox-relative POSIX without '..': {v!r}")
        return v

    @field_validator("sha256")
    @classmethod
    def _hex(cls, v: Optional[str]) -> Optional[str]:
        if v is not None and not SHA256_RE.match(v):
            raise ValueError("sha256 must be 64 lowercase hex characters")
        return v

    @property
    def cases_path(self) -> str:
        """Path relative to ``data/cases`` (the form used by the dataset tables)."""
        return f"{self.case_id}/evidence/{self.evidence_id}/fs/{self.path}"

    @classmethod
    def from_cases_path(cls, cases_path: str, sha256: Optional[str] = None) -> "ArtifactRef":
        m = re.match(r"^([^/]+)/evidence/([^/]+)/fs/(.+)$", cases_path)
        if not m:
            raise ValueError(f"not a data/cases-relative sandbox path: {cases_path!r}")
        return cls(case_id=m[1], evidence_id=m[2], path=m[3], sha256=sha256)


# --------------------------------------------------------------------------------------
# Timestamps
# --------------------------------------------------------------------------------------
class EpochKind(str, Enum):
    UNIX_S = "unix_s"                    # seconds since 1970-01-01 UTC
    UNIX_MS = "unix_ms"                  # Java / JavaScript / most Android providers
    UNIX_US = "unix_us"
    UNIX_NS = "unix_ns"
    WEBKIT_US = "webkit_us"              # Chrome / WebKit: microseconds since 1601-01-01 UTC
    WINDOWS_FILETIME = "windows_filetime"  # 100 ns intervals since 1601-01-01 UTC
    COCOA_S = "cocoa_s"                  # Apple Core Data / Mac absolute time: seconds since 2001-01-01 UTC
    COCOA_NS = "cocoa_ns"                # iOS sms.db (iOS 11+): nanoseconds since 2001-01-01 UTC
    HFS_S = "hfs_s"                      # HFS+: seconds since 1904-01-01 UTC
    ISO8601 = "iso8601"                  # text with explicit offset or Z
    EXIF = "exif"                        # 'YYYY:MM:DD HH:MM:SS' local time; offset only from OffsetTime* tags
    LOCAL_TEXT = "local_text"            # any other local wall-clock text without offset


_EPOCHS = {
    EpochKind.UNIX_S: (datetime(1970, 1, 1, tzinfo=timezone.utc), 1),
    EpochKind.UNIX_MS: (datetime(1970, 1, 1, tzinfo=timezone.utc), 1_000),
    EpochKind.UNIX_US: (datetime(1970, 1, 1, tzinfo=timezone.utc), 1_000_000),
    EpochKind.UNIX_NS: (datetime(1970, 1, 1, tzinfo=timezone.utc), 1_000_000_000),
    EpochKind.WEBKIT_US: (datetime(1601, 1, 1, tzinfo=timezone.utc), 1_000_000),
    EpochKind.WINDOWS_FILETIME: (datetime(1601, 1, 1, tzinfo=timezone.utc), 10_000_000),
    EpochKind.COCOA_S: (datetime(2001, 1, 1, tzinfo=timezone.utc), 1),
    EpochKind.COCOA_NS: (datetime(2001, 1, 1, tzinfo=timezone.utc), 1_000_000_000),
    EpochKind.HFS_S: (datetime(1904, 1, 1, tzinfo=timezone.utc), 1),
}


class ForensicTimestamp(BaseModel):
    """A stored time value, never silently converted.

    ``utc`` is set only when the source defines an absolute instant (numeric epochs, ISO-8601 with
    offset, EXIF with an ``OffsetTime*`` tag). ``local`` holds wall-clock text otherwise.
    """
    raw: Any = Field(description="value exactly as stored (int, float or str)")
    epoch: EpochKind
    utc: Optional[str] = Field(default=None, description="ISO-8601 UTC instant, e.g. 2020-02-09T18:23:53+00:00")
    local: Optional[str] = Field(default=None, description="ISO-8601 wall-clock time without offset")
    utc_offset: Optional[str] = Field(default=None, description="'+HH:MM' when known")


def to_timestamp(raw: Any, epoch: EpochKind | str, utc_offset: Optional[str] = None) -> ForensicTimestamp:
    """Normalise a stored time value. Zero / empty numeric values are 'unset' (``utc`` stays None)."""
    epoch = EpochKind(epoch)
    ts = ForensicTimestamp(raw=raw, epoch=epoch, utc_offset=utc_offset)
    if epoch in _EPOCHS:
        if raw in (None, "", 0, "0"):
            return ts
        base, per_s = _EPOCHS[epoch]
        if isinstance(raw, int) or (isinstance(raw, str) and raw.strip().lstrip("-").isdigit()):
            delta = timedelta(microseconds=int(raw) * 1_000_000 // per_s)   # exact for integer epochs
        else:
            delta = timedelta(seconds=float(raw) / per_s)
        ts.utc = (base + delta).isoformat()
        return ts
    text = str(raw).strip()
    if epoch is EpochKind.ISO8601:
        dt = datetime.fromisoformat(text.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            raise ValueError(f"ISO-8601 value without offset is local time; use epoch=local_text: {raw!r}")
        ts.utc = dt.astimezone(timezone.utc).isoformat()
        return ts
    if epoch is EpochKind.EXIF:
        dt = datetime.strptime(text, "%Y:%m:%d %H:%M:%S")
    else:
        dt = datetime.fromisoformat(text)
    ts.local = dt.isoformat()
    if utc_offset:
        sign = -1 if utc_offset.startswith("-") else 1
        hh, mm = utc_offset.lstrip("+-").split(":")
        offset = timezone(sign * timedelta(hours=int(hh), minutes=int(mm)))
        ts.utc = dt.replace(tzinfo=offset).astimezone(timezone.utc).isoformat()
    return ts
