"""BridgeRecord: the one return shape for every evidence read the forensic bridge exposes to agents.

A record is a value *plus the way to find it again*: the sandbox-relative file, its SHA-256 at read
time, the locator that reproduces the value, and the exact snippet the verifier will compare. Because
the locator fields are the :class:`~src.verification.models.Citation` fields, ``record.to_citation()``
is lossless and the HeuristicVerificationEngine can re-execute any record a parser emits.

Rules (``docs/DATA_INTERCHANGE_STANDARD.md`` §3):

1. ``raw_text_snippet`` is what the verifier recomputes, rendered the verifier's way: ``str(cell)`` for
   SQL, ``render_exif_value`` for EXIF, the full line for text, hex for binary byte offsets.
2. ``value`` is the parsed, typed payload for downstream use; it never replaces the snippet.
3. Times inside ``value`` / ``timestamps`` are :class:`ForensicTimestamp` (raw + epoch + UTC).
4. A bridge method returns ``None`` (or an empty list) rather than a record it cannot cite.
"""
from __future__ import annotations

from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

from src.schemas.common import SCHEMA_VERSION, ForensicTimestamp
from src.verification.models import Citation, LocatorType


class BridgeRecord(BaseModel):
    schema_version: str = SCHEMA_VERSION
    source: str = Field(description="bridge accessor that produced the record, e.g. 'sqlite.cell', 'exif.tag'")
    file_path: str = Field(description="sandbox-relative path of the file the value was read from")
    sha256: str
    locator_type: LocatorType
    byte_offset: Optional[int] = None
    line_number: Optional[int] = None
    sql_query: Optional[str] = None
    sql_params: list[Any] = Field(default_factory=list)
    regex: Optional[str] = None
    exif_tag: Optional[str] = None
    raw_text_snippet: str
    snippet_encoding: Literal["utf-8", "hex"] = "utf-8"
    match_mode: Literal["exact", "contains"] = "exact"
    value: Any = None
    field: Optional[str] = Field(default=None, description="logical field name, e.g. 'sms.body', 'GPS.GPSLatitude'")
    timestamps: dict[str, ForensicTimestamp] = Field(default_factory=dict)
    attributes: dict[str, Any] = Field(default_factory=dict, description="table, column, rowid, mime, ...")

    def to_citation(self) -> Citation:
        return Citation(file_path=self.file_path, locator_type=self.locator_type, byte_offset=self.byte_offset,
                        line_number=self.line_number, sql_query=self.sql_query, sql_params=list(self.sql_params),
                        regex=self.regex, exif_tag=self.exif_tag, expected_hash=self.sha256,
                        raw_text_snippet=self.raw_text_snippet, snippet_encoding=self.snippet_encoding,
                        match_mode=self.match_mode)
