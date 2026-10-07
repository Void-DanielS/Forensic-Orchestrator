"""Inverted index over every text cell of every SQLite database in the sandbox.

Agents use it to *propose* ground-truth citations for model-derived text (OCR, VLM):
given an observed string, find the database row that contains it and emit a
``SELECT <col> FROM <table> WHERE rowid = ?`` citation. The verifier then re-executes
that query independently; the index itself is never trusted as evidence.
"""
from __future__ import annotations

import re
import sqlite3
from collections import defaultdict
from dataclasses import dataclass

from src.bridge.forensic_bridge import ForensicBridge
from src.verification.heuristic_verifier import corroboration, normalize_text

_TOKEN = re.compile(r"\w{4,}", re.UNICODE)


TEXT_FILE_EXTS = (".txt", ".csv", ".log", ".eml", ".html", ".htm", ".xml", ".json", ".ini", ".md", ".vcf", ".ics")
MAX_TEXT_FILE = 2 << 20


def _q(identifier: str) -> str:
    """SQL identifier quoting (embedded double quotes doubled)."""
    return '"' + identifier.replace('"', '""') + '"'


@dataclass(frozen=True)
class TextCell:
    db: str          # sandbox-relative path (database, or text file when kind == "line")
    table: str
    column: str
    rowid: int       # SQLite rowid, or 1-based line number when kind == "line"
    value: str
    kind: str = "sql"

    def citation_sql(self) -> str:
        return f"SELECT {_q(self.column)} FROM {_q(self.table)} WHERE rowid = ?"

    def citation_regex(self) -> str:
        """Literal match of the value inside the file's decoded text (binary plists keep ASCII strings raw)."""
        return re.escape(self.value)

    @property
    def where(self) -> str:
        if self.kind == "sql":
            return f"{self.table}.{self.column} row {self.rowid}"
        return f"line {self.rowid}" if self.kind == "line" else f"plist string at {self.column}"


class CaseTextIndex:
    def __init__(self, bridge: ForensicBridge, min_len: int = 8, max_len: int = 4000, per_column_limit: int = 20000):
        self.bridge = bridge
        self.cells: list[TextCell] = []
        self._postings: dict[str, set[int]] = defaultdict(set)
        self.skipped_dbs: list[str] = []
        self.skipped_columns: list[str] = []   # undecodable / corrupt columns, kept for the audit trail
        self.min_len, self.max_len, self.per_column_limit = min_len, max_len, per_column_limit

    def build(self, text_files: bool = True) -> "CaseTextIndex":
        for entry in self.bridge.fs.walk():
            if entry.is_sqlite:
                self._index_db(entry.path)
            elif text_files and entry.extension in TEXT_FILE_EXTS and 0 < entry.size <= MAX_TEXT_FILE:
                self._index_text(entry.path)
            elif text_files and entry.extension == ".plist" and 0 < entry.size <= MAX_TEXT_FILE:
                self._index_plist(entry.path)
        return self

    def _index_text(self, rel: str) -> None:
        """Index lines of a plain-text artifact (e-mails, notes, CSV, logs). Cited with a line_number locator;
        the verifier re-reads the line with the same decoder (``TextBridge.line``)."""
        try:
            lines = self.bridge.text.read_text(rel).splitlines()
        except OSError:
            return
        for n, line in enumerate(lines, 1):
            if not (self.min_len <= len(line) <= self.max_len):
                continue
            idx = len(self.cells)
            self.cells.append(TextCell(rel, "", "", n, line, kind="line"))
            for tok in set(_TOKEN.findall(normalize_text(line))):
                self._postings[tok].add(idx)

    def _index_plist(self, rel: str) -> None:
        """String values of a property list. XML plists are cited by line like any text file; binary plists by
        an escaped-literal regex over the decoded file text, which the verifier re-runs (``TextBridge.read_text``).
        Only strings that appear verbatim in that decoded text are indexed (ASCII strings in bplist00)."""
        import plistlib

        raw = self.bridge.sandbox.resolve(rel).read_bytes()
        if not raw.startswith(b"bplist00"):
            self._index_text(rel)
            return
        try:
            obj = plistlib.loads(raw)
        except Exception:  # noqa: BLE001 - NSKeyedArchiver oddities / corrupt plists are skipped
            self.skipped_dbs.append(f"{rel}: unreadable plist")
            return
        decoded = self.bridge.text.read_text(rel)
        strings: list[tuple[str, str]] = []

        def walk(o, key: str) -> None:
            if isinstance(o, str):
                strings.append((key, o))
            elif isinstance(o, dict):
                for k, v in o.items():
                    walk(v, f"{key}/{k}" if key else str(k))
            elif isinstance(o, (list, tuple)):
                for i, v in enumerate(o[:5000]):
                    walk(v, f"{key}[{i}]")

        walk(obj, "")
        for key, value in strings:
            if not (self.min_len <= len(value) <= self.max_len) or value not in decoded:
                continue
            idx = len(self.cells)
            self.cells.append(TextCell(rel, "", key[:200], 0, value, kind="regex"))
            for tok in set(_TOKEN.findall(normalize_text(value))):
                self._postings[tok].add(idx)

    def _index_db(self, rel: str) -> None:
        try:
            tables = self.bridge.sqlite.tables(rel)
            con = self.bridge.sqlite.connect(rel)
        except sqlite3.DatabaseError as exc:
            self.skipped_dbs.append(f"{rel}: {exc}")
            return
        try:
            for table, cols in tables.items():
                for col, decl in cols:
                    if decl and not any(k in decl for k in ("TEXT", "CHAR", "CLOB", "VARCHAR")):
                        continue
                    sql = (f"SELECT rowid, {_q(col)} FROM {_q(table)} WHERE typeof({_q(col)})='text' "
                           f"AND length({_q(col)}) BETWEEN ? AND ? LIMIT ?")
                    try:
                        rows = con.execute(sql, (self.min_len, self.max_len, self.per_column_limit)).fetchall()
                    except sqlite3.DatabaseError as exc:
                        self.skipped_columns.append(f"{rel}:{table}.{col}: {str(exc)[:120]}")
                        continue
                    for rowid, value in rows:
                        idx = len(self.cells)
                        self.cells.append(TextCell(rel, table, col, rowid, value))
                        for tok in set(_TOKEN.findall(normalize_text(value))):
                            self._postings[tok].add(idx)
        finally:
            con.close()

    def search(self, observed: str, top_k: int = 3) -> list[tuple[float, TextCell]]:
        """Cells that pass the verifier's corroboration rule, best observed-coverage first."""
        tokens = set(_TOKEN.findall(normalize_text(observed)))
        if not tokens:
            return []
        hits: dict[int, int] = defaultdict(int)
        for tok in tokens:
            postings = self._postings.get(tok)
            if postings and len(postings) < 5000:  # ignore near-stopwords
                for i in postings:
                    hits[i] += 1
        need = 1 if len(tokens) <= 2 else 2
        scored = []
        for i, n in hits.items():
            if n < need:
                continue
            cell = self.cells[i]
            ok, obs_cov, truth_cov = corroboration(observed, cell.value)
            if ok:
                scored.append((obs_cov, truth_cov, cell))
        scored.sort(key=lambda x: (-x[0], -x[1]))
        return [(o, c) for o, _, c in scored[:top_k]]

    def stats(self) -> dict:
        return {"cells": len(self.cells), "tokens": len(self._postings),
                "databases": len({c.db for c in self.cells if c.kind == "sql"}),
                "text_files": len({c.db for c in self.cells if c.kind == "line"}),
                "plists": len({c.db for c in self.cells if c.kind == "regex"}), "skipped_dbs": len(self.skipped_dbs),
                "skipped_columns": len(self.skipped_columns)}
