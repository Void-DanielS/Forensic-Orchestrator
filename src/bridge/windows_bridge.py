"""Windows artefact accessors: registry hives, event logs (EVTX), Prefetch, shortcuts (LNK), IE ``index.dat``.

Same contract as the other bridges (FDX §3): every accessor returns a value *plus the locator that
reproduces it*, and the HeuristicVerificationEngine re-executes that locator on the physical file.

* Registry values use the ``registry`` locator ``"<key path>::<value name>"`` (or ``"<key path>"`` for the
  key's last-write time). The snippet is :func:`render_registry_value` of the value regipy reads; the
  verifier re-opens the hive and renders it the same way.
* Event records use the ``evtx_record`` locator (EventRecordID); the snippet is :func:`evtx_summary` of the
  record XML, recomputed by the verifier.
* Prefetch, LNK and ``index.dat`` are fixed binary layouts: they are cited by ``byte_offset`` + hex of the
  exact bytes, and the decoded value (UTF-16 name, FILETIME, run count) is derived from those bytes only.

Hives are parsed read-only from the evidence file (regipy reads the bytes; it never writes). Transaction
logs (``.LOG1/.LOG2``) are not replayed: what is cited is what the primary hive file holds.
"""
from __future__ import annotations

import html
import logging
import re
import struct
from datetime import datetime, timedelta, timezone
from typing import Any, Iterator

from src.schemas.bridge import BridgeRecord

FILETIME_EPOCH = datetime(1601, 1, 1, tzinfo=timezone.utc)
logging.getLogger("regipy").setLevel(logging.ERROR)     # it logs every undecodable REG_SZ as a warning


def filetime_to_iso(ft: int) -> str | None:
    """Windows FILETIME (100 ns since 1601-01-01 UTC) as ISO-8601 UTC; ``None`` when zero / out of range."""
    if not ft or ft < 0:
        return None
    try:
        return (FILETIME_EPOCH + timedelta(microseconds=ft // 10)).isoformat()
    except OverflowError:
        return None


def render_registry_value(value: Any) -> str:
    """Canonical text of a regipy value: lists (REG_MULTI_SZ) joined by ``|``, bytes as hex, else ``str``."""
    if isinstance(value, (bytes, bytearray)):
        return bytes(value).hex()
    if isinstance(value, (list, tuple)):
        return "|".join(render_registry_value(v) for v in value)
    if value is None:
        return ""
    return str(value)


def split_registry_locator(locator: str) -> tuple[str, str | None]:
    key, sep, name = locator.partition("::")
    return key, (name if sep else None)


# ------------------------------------------------------------------------------------------- registry
class RegistryBridge:
    def __init__(self, sandbox):
        self.sandbox = sandbox
        self._hives: dict[str, Any] = {}

    def hive(self, rel: str):
        from regipy.registry import RegistryHive

        digest = self.sandbox.hash(rel)
        if digest not in self._hives:
            self._hives.clear()          # one hive at a time: SOFTWARE hives reach 100+ MB
            self._hives[digest] = RegistryHive(str(self.sandbox.resolve(rel)))
        return self._hives[digest]

    def key(self, rel: str, path: str):
        from regipy.exceptions import RegistryKeyNotFoundException

        try:
            return self.hive(rel).get_key(path)
        except (RegistryKeyNotFoundException, Exception):  # noqa: BLE001 - corrupt cells read as absent
            return None

    def control_sets(self, rel: str) -> list[str]:
        """``\\ControlSet00N`` paths of a SYSTEM hive, current set first."""
        sel = self.key(rel, "\\Select")
        current = None
        if sel is not None:
            current = next((v.value for v in sel.iter_values() if v.name == "Current"), None)
        sets = []
        root = self.key(rel, "\\")
        if root is not None:
            sets = [f"\\{s.name}" for s in root.iter_subkeys() if s.name.lower().startswith("controlset")]
        if current:
            first = f"\\ControlSet{int(current):03d}"
            sets = [first] + [s for s in sets if s.lower() != first.lower()]
        return sets

    def read(self, rel: str, locator: str) -> str | None:
        """Rendered value (or key last-write time) at ``locator``; ``None`` when absent."""
        path, name = split_registry_locator(locator)
        k = self.key(rel, path)
        if k is None:
            return None
        if name is None:
            return filetime_to_iso(k.header.last_modified)
        for v in k.iter_values():
            if v.name == name:
                return render_registry_value(v.value)
        return None

    def values(self, rel: str, path: str) -> dict[str, Any]:
        k = self.key(rel, path)
        return {} if k is None else {v.name: v.value for v in k.iter_values()}

    def subkeys(self, rel: str, path: str) -> list[str]:
        k = self.key(rel, path)
        if k is None:
            return []
        try:
            return [s.name for s in k.iter_subkeys()]
        except Exception:  # noqa: BLE001
            return []

    def record(self, rel: str, path: str, name: str | None = None, field: str | None = None) -> BridgeRecord | None:
        locator = path if name is None else f"{path}::{name}"
        snippet = self.read(rel, locator)
        if snippet in (None, ""):
            return None
        return BridgeRecord(source="registry.value", file_path=rel, sha256=self.sandbox.hash(rel),
                            locator_type="registry", registry=locator, raw_text_snippet=snippet, value=snippet,
                            field=field or (name or "key_last_write"), attributes={"key": path, "value": name})


# ------------------------------------------------------------------------------------------- EVTX
_EVTX_FIELDS = re.compile(r"<(EventID|TimeCreated|EventRecordID|Provider|Computer)\b([^>]*)>([^<]*)<")
_EVTX_DATA = re.compile(r'<Data(?: Name="([^"]*)")?>([^<]*)</Data>')


def evtx_summary(xml: str, max_data: int = 12) -> str:
    """Deterministic one-line rendering of an event record (provider, ID, time, named data)."""
    head: dict[str, str] = {}
    for tag, attrs, text in _EVTX_FIELDS.findall(xml):
        if tag == "Provider":
            m = re.search(r'Name="([^"]*)"', attrs)
            head[tag] = m.group(1) if m else ""
        elif tag == "TimeCreated":
            m = re.search(r'SystemTime="([^"]*)"', attrs)
            head[tag] = m.group(1) if m else ""
        else:
            head[tag] = html.unescape(text).strip()
    data = []
    for name, val in _EVTX_DATA.findall(xml)[:max_data]:
        val = re.sub(r"\s+", " ", html.unescape(val)).strip()
        if val:
            data.append(f"{name}={val}" if name else val)
    return (f"{head.get('Provider', '')} EventID={head.get('EventID', '')} Record={head.get('EventRecordID', '')} "
            f"Time={head.get('TimeCreated', '')} " + "; ".join(data)).strip()


class EvtxBridge:
    """Event records through pyevtx-rs (``pip install evtx``; Rust, ~1000x faster than python-evtx).

    :meth:`summary` renders the whole file once and keeps ``{EventRecordID: summary}`` for the most recent
    file, so re-verifying thousands of records costs one pass, not one pass per record.
    """

    def __init__(self, sandbox):
        self.sandbox = sandbox
        self._cache: tuple[str, dict[int, str]] | None = None

    def records(self, rel: str) -> Iterator[tuple[int, str]]:
        from evtx import PyEvtxParser

        try:
            parser = PyEvtxParser(str(self.sandbox.resolve(rel)))
            for r in parser.records():
                if isinstance(r, dict) and r.get("data"):
                    yield int(r["event_record_id"]), r["data"]
        except (OSError, RuntimeError):     # damaged / truncated log: keep what was read
            return

    def summary(self, rel: str, record_id: int) -> str | None:
        digest = self.sandbox.hash(rel)
        if self._cache is None or self._cache[0] != digest:
            self._cache = (digest, {rid: evtx_summary(xml) for rid, xml in self.records(rel)})
        return self._cache[1].get(record_id)

    def record(self, rel: str, record_id: int, xml: str, field: str | None = None) -> BridgeRecord:
        snippet = evtx_summary(xml)
        return BridgeRecord(source="evtx.record", file_path=rel, sha256=self.sandbox.hash(rel),
                            locator_type="evtx_record", evtx_record=record_id, raw_text_snippet=snippet, value=snippet,
                            field=field)


# ------------------------------------------------------------------------------------------- binary layouts
def bytes_record(sandbox, rel: str, offset: int, data: bytes, field: str, value: Any = None) -> BridgeRecord:
    return BridgeRecord(source="bytes", file_path=rel, sha256=sandbox.hash(rel), locator_type="byte_offset",
                        byte_offset=offset, raw_text_snippet=data.hex(), snippet_encoding="hex",
                        value=value, field=field)


# Prefetch: (format version) -> (offset of last-run FILETIME, offset of run count)
PREFETCH_LAYOUT = {17: (0x78, 0x90), 23: (0x80, 0x98), 26: (0x80, 0xD0), 30: (0x80, 0xD0)}


def parse_prefetch(data: bytes) -> dict | None:
    """Uncompressed SCCA Prefetch (XP 17, Vista/7 23, 8.1 26, 10 30 when not MAM-compressed)."""
    if len(data) < 0x100 or data[4:8] != b"SCCA":
        return None
    version = struct.unpack("<I", data[:4])[0]
    if version not in PREFETCH_LAYOUT:
        return None
    name_raw = data[0x10:0x10 + 60]
    end = next((i for i in range(0, 60, 2) if name_raw[i:i + 2] == b"\x00\x00"), 60)
    lr_off, rc_off = PREFETCH_LAYOUT[version]
    return {"version": version, "executable": name_raw[:end].decode("utf-16-le", "replace"),
            "name_bytes": (0x10, name_raw[:end]),
            "last_run": filetime_to_iso(struct.unpack("<Q", data[lr_off:lr_off + 8])[0]),
            "last_run_bytes": (lr_off, data[lr_off:lr_off + 8]),
            "run_count": struct.unpack("<I", data[rc_off:rc_off + 4])[0],
            "run_count_bytes": (rc_off, data[rc_off:rc_off + 4])}


def find_string(data: bytes, text: str) -> tuple[int, bytes] | None:
    """Offset and bytes of ``text`` in a binary structure, trying UTF-16LE, then ASCII/cp1252."""
    if not text:
        return None
    for enc in ("utf-16-le", "cp1252"):
        try:
            needle = text.encode(enc)
        except UnicodeEncodeError:
            continue
        i = data.find(needle)
        if i >= 0:
            return i, needle
    return None


# IE 5-9 cache / history index (``Client UrlCache MMF Ver 5.2``): "URL " records; FILETIMEs at +0x08 / +0x10.
IE_URL_RECORD = re.compile(rb"URL \x02|URL \x03|URL \x04|URL \x05|URL \x06|URL \x07|URL \x08|URL \x01")
IE_LOCATION = re.compile(rb"(?:Visited: |:\d{10}: )?[^\x00]{0,80}?(?:https?|ftp|file)://[\x21-\x7e]{3,1024}")
