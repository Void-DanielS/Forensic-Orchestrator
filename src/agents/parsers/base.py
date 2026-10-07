"""Declarative SQLite artifact parsers.

An artifact (SMS, call, browser visit, chat message, ...) is described by a :class:`TableSpec`:
which databases hold it (path regex), which table, which columns (several candidate names per
field, because schemas drift between OS / app versions), how times are encoded, and which
related rows to look up (WhatsApp chat -> JID, iMessage handle, Safari visit -> URL).

:func:`parse_table` turns every row into one :class:`ParsedRecord`: a claim, a typed value and a
list of :class:`~src.schemas.bridge.BridgeRecord`, one per cited cell. Each record cites its cell
with ``SELECT "<col>" FROM "<table>" WHERE rowid = ?`` and ``str(cell)`` as the snippet, exactly the
form the HeuristicVerificationEngine re-executes (FDX §3.2). BLOB cells are never quoted (a ``str``
of bytes is not reproducible text); their presence is recorded in ``value`` only.

Everything runs through ``bridge.sqlite`` (hash-keyed working copies, read-only, SELECT-only).
"""
from __future__ import annotations

import re
import sqlite3
from dataclasses import dataclass, field
from typing import Any, Callable

from src.bridge.forensic_bridge import ForensicBridge
from src.schemas.bridge import BridgeRecord
from src.schemas.common import EpochKind, to_timestamp


def q(identifier: str) -> str:
    return '"' + identifier.replace('"', '""') + '"'


@dataclass(frozen=True)
class Col:
    """One logical field: the first candidate column present in the table is used."""
    name: str
    columns: tuple[str, ...]
    epoch: str | None = None          # EpochKind value, or "cocoa_auto" / "unix_auto"
    cite: bool = True                 # quote the cell as a citation (False: value only, e.g. flags)


@dataclass(frozen=True)
class Lookup:
    """Related row: ``SELECT rowid, <value> FROM <table> WHERE <key> = <value of field 'via'>``."""
    name: str
    table: tuple[str, ...]
    key: tuple[str, ...]
    value: tuple[str, ...]
    via: str


@dataclass(frozen=True)
class TableSpec:
    category: str                     # finding category, e.g. "sms.message"
    path: str                         # regex on the sandbox-relative path (case-insensitive)
    table: tuple[str, ...]            # candidate table names
    cols: tuple[Col, ...]
    require: tuple[str, ...]          # fields that must resolve to a column for the spec to apply
    claim: str                        # str.format template over field values / labels
    text: tuple[str, ...] = ()        # fields of which at least one must be non-empty for a row to count
    lookups: tuple[Lookup, ...] = ()
    where: str = ""                   # extra SQL condition; {col:<field>} placeholders become quoted columns
    labels: dict[str, dict[Any, str]] = field(default_factory=dict)
    max_rows: int = 1000
    description: str = ""

    def matches(self, rel: str) -> bool:
        return re.search(self.path, rel, re.I) is not None


@dataclass
class ParsedRecord:
    spec: TableSpec
    db: str
    table: str
    rowid: int
    claim: str
    value: dict[str, Any]
    records: list[BridgeRecord]


# ---------------------------------------------------------------------------------- times
def timestamp(raw: Any, epoch: str):
    if raw is None or isinstance(raw, (bytes, bytearray)) or raw in ("", 0):
        return None
    try:
        if epoch == "cocoa_auto":     # iOS 11+ stores nanoseconds, older versions seconds
            epoch = EpochKind.COCOA_NS if abs(float(raw)) > 1e11 else EpochKind.COCOA_S
        elif epoch == "unix_auto":
            v = abs(float(raw))
            epoch = (EpochKind.UNIX_NS if v > 1e17 else EpochKind.UNIX_US if v > 1e14 else
                     EpochKind.UNIX_MS if v > 1e11 else EpochKind.UNIX_S)
        return to_timestamp(raw, epoch)
    except (ValueError, OverflowError, TypeError):
        return None


# ---------------------------------------------------------------------------------- engine
class _Missing(dict):
    def __missing__(self, key: str) -> str:
        return ""


def _resolve(cands: tuple[str, ...], available: dict[str, str]) -> str | None:
    for c in cands:
        hit = available.get(c.lower())
        if hit:
            return hit
    return None


def _cell(db: str, sha: str, table: str, column: str, rowid: int, value: Any, fieldname: str) -> BridgeRecord:
    return BridgeRecord(source="sqlite.cell", file_path=db, sha256=sha, locator_type="sql_query",
                        sql_query=f"SELECT {q(column)} FROM {q(table)} WHERE rowid = ?", sql_params=[rowid],
                        raw_text_snippet=str(value), value=value, field=fieldname,
                        attributes={"table": table, "column": column, "rowid": rowid})


def _short(v: Any, n: int = 160) -> str:
    s = re.sub(r"\s+", " ", str(v)).strip()
    return s if len(s) <= n else s[: n - 1] + "…"


def parse_table(bridge: ForensicBridge, db: str, spec: TableSpec,
                tables: dict[str, list[tuple[str, str]]] | None = None) -> list[ParsedRecord]:
    """All rows of ``spec`` in database ``db`` as citable records ([] when the spec does not apply)."""
    tables = tables if tables is not None else bridge.sqlite.tables(db)
    by_lower = {t.lower(): t for t in tables}
    table = next((by_lower[t.lower()] for t in spec.table if t.lower() in by_lower), None)
    if table is None:
        return []
    available = {c.lower(): c for c, _ in tables[table]}
    resolved = {col.name: _resolve(col.columns, available) for col in spec.cols}
    if any(resolved.get(r) is None for r in spec.require):
        return []
    used = [(col, resolved[col.name]) for col in spec.cols if resolved[col.name]]
    where = spec.where
    if where:
        needed = re.findall(r"\{col:(\w+)\}", where)
        if any(resolved.get(n) is None for n in needed):
            return []          # the condition needs a column this schema version lacks
        where = re.sub(r"\{col:(\w+)\}", lambda m: q(resolved[m.group(1)]), where)
    sql = (f"SELECT rowid, {', '.join(q(c) for _, c in used)} FROM {q(table)}"
           + (f" WHERE {where}" if where else "") + " ORDER BY rowid")
    try:
        rows = bridge.sqlite.query(db, sql, row_limit=spec.max_rows)
    except sqlite3.DatabaseError:
        return []
    sha = bridge.sandbox.hash(db)
    lookup_tables = {}
    for lk in spec.lookups:
        lt = next((by_lower[t.lower()] for t in lk.table if t.lower() in by_lower), None)
        if lt:
            lav = {c.lower(): c for c, _ in tables[lt]}
            key, val = _resolve(lk.key, lav), _resolve(lk.value, lav)
            if key and val:
                lookup_tables[lk.name] = (lt, key, val)
    out: list[ParsedRecord] = []
    for row in rows:
        rowid, cells = row[0], row[1:]
        values: dict[str, Any] = {}
        records: list[BridgeRecord] = []
        times: dict[str, Any] = {}
        blobs: list[str] = []
        for (col, column), v in zip(used, cells):
            if v is None or v == "":
                continue
            if isinstance(v, (bytes, bytearray)):
                blobs.append(col.name)
                continue
            values[col.name] = v
            if col.epoch:
                ts = timestamp(v, col.epoch)
                if ts is not None:
                    times[col.name] = ts
            if col.cite:
                rec = _cell(db, sha, table, column, rowid, v, f"{spec.category}.{col.name}")
                if col.name in times:
                    rec.timestamps[col.name] = times[col.name]
                records.append(rec)
        for lk in spec.lookups:
            if lk.name not in lookup_tables or values.get(lk.via) is None:
                continue
            lt, key, val = lookup_tables[lk.name]
            try:
                hit = bridge.sqlite.query(db, f"SELECT rowid, {q(val)} FROM {q(lt)} WHERE {q(key)} = ? ORDER BY rowid",
                                          [values[lk.via]], row_limit=1)
            except sqlite3.DatabaseError:
                continue
            if hit and hit[0][1] not in (None, "") and not isinstance(hit[0][1], (bytes, bytearray)):
                values[lk.name] = hit[0][1]
                records.append(_cell(db, sha, lt, val, hit[0][0], hit[0][1], f"{spec.category}.{lk.name}"))
        if spec.text and not any(values.get(t) not in (None, "") for t in spec.text):
            continue
        if not records:
            continue
        labels = {f"{k}_label": spec.labels[k].get(values.get(k), str(values.get(k, "?"))) for k in spec.labels}
        fmt = _Missing({**{k: _short(v) for k, v in values.items()}, **labels})
        for k, ts in times.items():
            fmt[f"{k}_utc"] = ts.utc or ts.local or str(ts.raw)
        claim = spec.claim.format_map(fmt)
        value = {"table": table, "rowid": rowid, "fields": values,
                 "timestamps": {k: t.model_dump(exclude_none=True) for k, t in times.items()}}
        if blobs:
            value["blob_fields"] = blobs
        out.append(ParsedRecord(spec, db, table, rowid, claim, value, records))
    return out


def detect(paths: list[str], specs: list[TableSpec]) -> dict[str, list[TableSpec]]:
    """{database path: [specs whose path pattern matches]} for the given sandbox paths."""
    hits: dict[str, list[TableSpec]] = {}
    for p in paths:
        for s in specs:
            if s.matches(p):
                hits.setdefault(p, []).append(s)
    return hits


Postprocess = Callable[[ParsedRecord], ParsedRecord]
