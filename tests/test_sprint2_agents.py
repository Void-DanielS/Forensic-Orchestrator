"""Sprint 2 agents: AppArtifactAgent parsers and FileSystemAgent (FDX §3.4 round trip, forgery, absence,
integrity), on synthetic fixtures built per test, plus a real-corpus integration test when present."""
from __future__ import annotations

import hashlib
import json
import re
import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.agents.app_artifact_agent import AppArtifactAgent
from src.agents.filesystem_agent import FileSystemAgent
from src.agents.parsers import ALL_SPECS, TableSpec, parse_table
from src.bridge.forensic_bridge import ForensicBridge
from src.verification.heuristic_verifier import HeuristicVerificationEngine
from src.verification.models import FindingStatus

SAMPLE = {"unix_ms": 1580322045251, "unix_s": 1580322045, "unix_us": 1580322045251000, "webkit_us": 13224795645251000,
          "cocoa_s": 602014845, "cocoa_auto": 602014845123456789, "unix_auto": 1580322045251}


REAL_NAMES = ("mmssms.db", "bugle_db", "contacts2.db", "calllog.db", "calendar.db", "accounts.db", "sms.db",
              "AddressBook.sqlitedb", "CallHistory.storedata", "knowledgeC.db", "NoteStore.sqlite", "Calendar.sqlitedb",
              "Photos.sqlite", "History", "History.db", "places.sqlite", "msgstore.db", "ChatStorage.sqlite", "main.db",
              "threads_db2", "naver_line", "kikDatabase.db", "db_im_xx")


def _fixture_path(spec: TableSpec) -> str:
    """A sandbox path with a real database file name that satisfies the spec's path regex."""
    name = next(n for n in REAL_NAMES if spec.matches(f"data/data/app/databases/{n}"))
    return f"data/data/app/databases/{name}"


def _build_db(path: Path, spec: TableSpec) -> dict:
    """Create the spec's table with its first candidate columns, its lookup tables, and one row."""
    path.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(path)
    cols, row = [], {}
    for c in spec.cols:
        col = c.columns[0]
        cols.append(col)
        row[col] = SAMPLE.get(c.epoch, 1) if c.epoch else (7 if c.name.endswith("_id") or c.name in (
            "type", "read", "from_me", "is_from_me", "was_me", "duration", "originated", "answered", "visit_count",
            "history_item", "record_id", "property", "received_bytes", "total_bytes", "id") else f"value of {c.name}")
    for c in spec.cols:      # numeric fields filtered by WHERE clauses
        if c.name in ("latitude",):
            row[c.columns[0]] = 35.66
        if c.name in ("longitude",):
            row[c.columns[0]] = -78.88
        if c.name == "stream":
            row[c.columns[0]] = "/app/inFocus"
    con.execute(f"CREATE TABLE {spec.table[0]} ({', '.join(cols)})")
    con.execute(f"INSERT INTO {spec.table[0]} VALUES ({','.join('?' * len(cols))})", [row[c] for c in cols])
    lookup_tables: dict[str, set] = {}
    for lk in spec.lookups:
        lookup_tables.setdefault(lk.table[0], set()).update({lk.key[0], lk.value[0]})
    if "mimetypes" in spec.where:
        lookup_tables.setdefault("mimetypes", set()).update({"_id", "mimetype"})
    for t, tcols in lookup_tables.items():
        tcols = sorted(tcols)
        con.execute(f"CREATE TABLE {t} ({', '.join(tcols)})")
        vals = []
        for c in tcols:
            if c == "mimetype":
                vals.append("vnd.android.cursor.item/phone_v2")
            elif c in {lk.key[0] for lk in spec.lookups} | {"_id"}:
                vals.append(7)
            elif c in {lk.value[0] for lk in spec.lookups if lk.name in {l.via for l in spec.lookups}}:
                vals.append(7)        # chained lookup (value feeds the next lookup's key)
            else:
                vals.append(f"looked-up {c}")
        con.execute(f"INSERT INTO {t} VALUES ({','.join('?' * len(tcols))})", vals)
    con.commit()
    con.close()
    return row


@pytest.mark.parametrize("spec", ALL_SPECS, ids=lambda s: f"{s.category}:{s.table[0]}")
def test_every_parser_spec_round_trips_and_detects_forgery(tmp_path, spec):
    root = tmp_path / "fs"
    rel = _fixture_path(spec)
    assert spec.matches(rel), rel
    _build_db(root / rel, spec)
    before = hashlib.sha256((root / rel).read_bytes()).hexdigest()
    bridge = ForensicBridge(root)
    agent = AppArtifactAgent(bridge, specs=[spec])
    findings = agent.analyze_artifact(rel)
    assert findings, f"{spec.category}: no record parsed from the fixture"
    verifier = HeuristicVerificationEngine(root, bridge=bridge)
    for f in findings:
        assert verifier.verify_finding(f).status == FindingStatus.VERIFIED, f.verification
        assert all(c.sql_query.startswith("SELECT") and c.sql_params for c in f.citations)
    # forgery: change one quoted value -> rejected
    forged = findings[0].model_copy(deep=True)
    forged.citations[0].raw_text_snippet += " (edited)"
    forged.status = FindingStatus.PENDING
    assert verifier.verify_finding(forged).status == FindingStatus.REJECTED
    bridge.close()
    assert hashlib.sha256((root / rel).read_bytes()).hexdigest() == before   # evidence untouched


def test_absent_table_and_missing_columns_yield_nothing(tmp_path):
    root = tmp_path / "fs"
    rel = "data/data/x/databases/mmssms.db"
    (root / rel).parent.mkdir(parents=True)
    con = sqlite3.connect(root / rel)
    con.execute("CREATE TABLE sms (address)")        # no body column
    con.execute("INSERT INTO sms VALUES ('123')")
    con.commit()
    con.close()
    bridge = ForensicBridge(root)
    assert all(not parse_table(bridge, rel, s) for s in ALL_SPECS if s.matches(rel))
    bridge.close()


def test_timestamps_are_normalised_with_raw_and_epoch(tmp_path):
    spec = next(s for s in ALL_SPECS if s.category == "sms.message" and s.table == ("sms",))
    root = tmp_path / "fs"
    rel = "data/data/com.android.providers.telephony/databases/mmssms.db"
    _build_db(root / rel, spec)
    bridge = ForensicBridge(root)
    rec = parse_table(bridge, rel, spec)[0]
    ts = rec.value["timestamps"]["date"]
    assert ts["raw"] == SAMPLE["unix_ms"] and ts["epoch"] == "unix_ms" and ts["utc"].startswith("2020-01-29T18:20:45")
    bridge.close()


# ---------------------------------------------------------------- FileSystemAgent
def _sqlite_with_deleted_row(path: Path, secret: str) -> None:
    con = sqlite3.connect(path)
    con.execute("PRAGMA secure_delete=OFF")
    con.execute("CREATE TABLE notes (body TEXT)")
    con.executemany("INSERT INTO notes VALUES (?)", [(f"keep this ordinary note number {i}",) for i in range(30)])
    con.execute("INSERT INTO notes VALUES (?)", (secret,))
    con.commit()
    con.execute("DELETE FROM notes WHERE body = ?", (secret,))
    con.commit()
    con.close()


def test_filesystem_agent_findings_verify(tmp_path):
    root = tmp_path / "fs"
    (root / "Users/bob/Pictures").mkdir(parents=True)
    (root / "Users/bob/Pictures/holiday.gif").write_bytes(b"\xff\xd8\xff\xe0" + b"\0" * 200)     # JPEG named .gif
    (root / "Users/bob/.secret").mkdir(parents=True)
    (root / "Users/bob/.secret/plans.txt").write_text("meet at the dock")
    (root / "Users/bob/AppData").mkdir(parents=True)
    (root / "Users/bob/AppData/cache_store").write_bytes(b"SQLite format 3\x00" + b"\0" * 500)
    (root / "data").mkdir()
    (root / "data/orphan.db-wal").write_bytes(b"\x37\x7f\x06\x82" + b"\0" * 100)
    _sqlite_with_deleted_row(root / "data/notes.db", "deleted message about the stolen recipe")
    _sqlite_with_deleted_row(root / "data/Login Data", "hunter2 password for the bank")
    manifest = {"files": [{"path": "fs/" + p.relative_to(root).as_posix(), "size": p.stat().st_size,
                           "category": "user_document", "sha256": ""} for p in root.rglob("*") if p.is_file()]}
    manifest["files"][0]["deleted"] = True
    bridge = ForensicBridge(root)
    agent = FileSystemAgent(bridge, manifest=manifest)
    rep = agent.run(sorted(agent.meta))
    cats = {f.category for f in rep.findings}
    assert {"fs.magic_mismatch", "fs.hidden_path", "fs.sqlite_unlisted", "fs.orphan_wal", "fs.deleted_recovered",
            "sqlite.carved_text"} <= cats
    assert all(f.status == FindingStatus.VERIFIED for f in rep.findings)
    carved = [f.value["text"] for f in rep.findings if f.category == "sqlite.carved_text"]
    assert any("stolen recipe" in t for t in carved)
    assert not any("hunter2" in t for t in carved), "credential stores must never be carved"
    bridge.close()


# ---------------------------------------------------------------- real corpus (skipped when absent)
ANDROID10 = Path(__file__).resolve().parents[1] / "data/cases/hickman_android_10"


@pytest.mark.skipif(not (ANDROID10 / "evidence/android_10/MANIFEST.json").exists(), reason="android_10 not fetched")
def test_android10_documented_sms_are_recovered_and_verified():
    """Every distinct SMS in Hickman's SMS-Messages.xlsx (mmssms.db / bugle_db) is a VERIFIED finding."""
    txt = (ANDROID10 / "context/brief/SMS-Messages.xlsx.txt").read_text(encoding="utf-8")
    documented = set()
    for line in txt.splitlines():
        cols = [c.strip() for c in line.split(" | ")]
        if len(cols) < 18 or not cols[0].isdigit() or not ("Table: sms" in cols[17] or "bugle_db" in cols[17]):
            continue
        m = re.match(r"(\d+/\d+/\d+ \d+:\d+:\d+ [AP]M)\(UTC\+0\)", cols[3])
        if m:
            t = datetime.strptime(m.group(1), "%m/%d/%Y %I:%M:%S %p").replace(tzinfo=timezone.utc)
            documented.add((cols[14].strip(), t.strftime("%Y-%m-%dT%H:%M:%S")))
    ev = ANDROID10 / "evidence/android_10"
    bridge = ForensicBridge(ev / "fs")
    agent = AppArtifactAgent(bridge, evidence_id="android_10")
    rep = agent.run(agent.discover(json.loads((ev / "MANIFEST.json").read_text(encoding="utf-8"))))
    bridge.close()
    got = set()
    for f in rep.findings:
        if f.category == "sms.message" and f.status == FindingStatus.VERIFIED:
            ts = f.value["timestamps"].get("date") or f.value["timestamps"].get("timestamp") or {}
            got.add((str(f.value["fields"]["body"]).strip(), ts.get("utc", "")[:19]))
    assert len(documented) >= 10
    assert documented <= got, sorted(documented - got)[:5]
    assert not rep.by_status(FindingStatus.REJECTED)


def test_binary_plist_strings_are_indexed_and_regex_citations_verify(tmp_path):
    import plistlib

    from src.bridge.text_index import CaseTextIndex
    from src.verification.models import Citation, Finding

    root = tmp_path / "fs"
    p = root / "private/var/mobile/Library/Preferences/com.example.plist"
    p.parent.mkdir(parents=True)
    p.write_bytes(plistlib.dumps({"LastSearch": "warehouse on Pier 39 at midnight", "n": 3}, fmt=plistlib.FMT_BINARY))
    bridge = ForensicBridge(root)
    idx = CaseTextIndex(bridge).build()
    assert idx.stats()["plists"] == 1
    hits = idx.search("warehouse on Pier 39 at midnight")
    assert hits and hits[0][1].kind == "regex"
    cell = hits[0][1]
    rel = "private/var/mobile/Library/Preferences/com.example.plist"
    c = Citation(file_path=rel, locator_type="regex", regex=cell.citation_regex(), expected_hash=bridge.sandbox.hash(rel),
                 raw_text_snippet=cell.value)
    f = Finding(agent="t", artifact_path=rel, category="ocr.corroborated", claim="x", citations=[c])
    assert HeuristicVerificationEngine(root, bridge=bridge).verify_finding(f).status == FindingStatus.VERIFIED
    forged = f.model_copy(deep=True)
    forged.citations[0].raw_text_snippet = "warehouse on Pier 40 at midnight"
    forged.citations[0].regex = re.escape("warehouse on Pier 40 at midnight")
    assert HeuristicVerificationEngine(root, bridge=bridge).verify_finding(forged).status == FindingStatus.REJECTED
    bridge.close()
