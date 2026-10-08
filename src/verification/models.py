"""Data contracts shared by every agent and the HeuristicVerificationEngine.

An agent never asserts a fact on its own authority. Each :class:`Finding` carries one
or more :class:`Citation` objects pointing at the exact bytes / line / SQL row / EXIF tag
in the extraction that support it. The verifier re-executes every citation against the
physical file before the finding can be accepted.
"""
from __future__ import annotations

import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, model_validator

LocatorType = Literal["byte_offset", "line_number", "sql_query", "regex", "exif_tag", "registry", "evtx_record"]

# SQLite side files whose content a query can read (the WAL is merged into the working copy). ``-shm`` is a
# rebuildable index, not data, so it is not part of the cited evidence.
SIDE_SUFFIXES = ("-wal", "-journal")


class Citation(BaseModel):
    """Pointer to ground truth inside the logical sandbox.

    ``file_path`` is relative to the sandbox root. ``expected_hash`` is the SHA-256 of the
    whole file as seen by the agent; a mismatch means the evidence changed and the
    citation fails. Exactly one locator field must be set, matching ``locator_type``.
    """

    file_path: str
    locator_type: LocatorType
    byte_offset: Optional[int] = None
    line_number: Optional[int] = None          # 1-based
    sql_query: Optional[str] = None            # single read-only SELECT
    sql_params: list[Any] = Field(default_factory=list)
    regex: Optional[str] = None
    exif_tag: Optional[str] = None             # "<IFD>.<TagName>", e.g. "GPS.GPSLatitude"
    registry: Optional[str] = None             # "<key path>" or "<key path>::<value name>" inside a hive
    evtx_record: Optional[int] = None          # EventRecordID inside an .evtx file
    expected_hash: str
    # SHA-256 of the -wal / -journal next to a cited database at read time ({"-wal": sha}); SQL citations only.
    side_hashes: dict[str, str] = Field(default_factory=dict)
    raw_text_snippet: str
    snippet_encoding: Literal["utf-8", "hex"] = "utf-8"
    match_mode: Literal["exact", "contains"] = "exact"

    @model_validator(mode="after")
    def _locator_present(self) -> "Citation":
        field_for = {
            "byte_offset": self.byte_offset,
            "line_number": self.line_number,
            "sql_query": self.sql_query,
            "regex": self.regex,
            "exif_tag": self.exif_tag,
            "registry": self.registry,
            "evtx_record": self.evtx_record,
        }
        if field_for[self.locator_type] is None:
            raise ValueError(f"locator_type={self.locator_type} requires field '{self.locator_type}'")
        return self


class FindingStatus(str, Enum):
    PENDING = "pending"
    VERIFIED = "verified"              # every citation re-executed and matched
    REJECTED = "rejected"              # at least one citation failed -> treated as hallucination
    UNCORROBORATED = "uncorroborated"  # model observation with no ground-truth backing; never evidentiary


class Finding(BaseModel):
    finding_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:12])
    agent: str
    artifact_path: str                  # sandbox-relative path of the analysed artifact
    category: str                       # e.g. "exif.gps", "ocr.corroborated", "vlm.description"
    claim: str                          # human-readable statement
    value: Any = None                   # structured payload (coordinates, text, ...)
    citations: list[Citation] = Field(default_factory=list)
    # Model-derived text (OCR/VLM) that must resemble the cited ground truth.
    observed_text: Optional[str] = None
    model: Optional[str] = None
    confidence: Optional[float] = None
    status: FindingStatus = FindingStatus.PENDING
    verification: list[dict[str, Any]] = Field(default_factory=list)
    # Links to the investigation context (see src/context/case_context.py). Analysis aids, never evidence.
    context_links: list[dict[str, Any]] = Field(default_factory=list)
    evidence_id: Optional[str] = None
    created_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


class VerificationResult(BaseModel):
    verified: bool
    locator_type: str
    file_path: str
    reason: str
    actual_hash: Optional[str] = None
    hash_match: Optional[bool] = None
    observed_snippet: Optional[str] = None
    checked_utc: str = Field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
