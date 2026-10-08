"""WindowsArtifactAgent (binary layouts, EVTX rendering), verifier independence and WAL side-file hashing,
carved-text ranking, and answer-key scoring. Synthetic fixtures; real-corpus tests skip when data is absent."""
from __future__ import annotations

import json
import shutil
import sqlite3
import struct
from datetime import datetime, timezone
from pathlib import Path

import pytest

from src.agents.app_artifact_agent import AppArtifactAgent
from src.agents.filesystem_agent import carve_score, rank_carved
from src.agents.windows_agent import WindowsArtifactAgent
from src.bridge.forensic_bridge import ForensicBridge
from src.bridge.windows_bridge import (evtx_summary, filetime_to_iso, parse_prefetch, render_registry_value,
                                       split_registry_locator)
from src.cases.registry import CASES_DIR
from src.eval.scoring import GOLD_DIR, Doc, norm, score_gold, score_question
from src.verification.heuristic_verifier import HeuristicVerificationEngine
from src.verification.models import Citation, FindingStatus

FT = 130717531931299204          # 2015-03-24T13:59:53.129920+00:00


def _ft(dt: datetime) -> int:
    return int((dt - datetime(1601, 1, 1, tzinfo=timezone.utc)).total_seconds() * 10_000_000)


def _agent(root: Path, files: list[str]) -> WindowsArtifactAgent:
    manifest = {"files": [{"path": f"fs/{f}"} for f in files]}
    return WindowsArtifactAgent(ForensicBridge(root), manifest=manifest)


# ------------------------------------------------------------------------------------------- binary layouts
def _prefetch_bytes(name: str, count: int, last: int, version: int = 23) -> bytes:
    data = bytearray(0x200)
    struct.pack_into("<I", data, 0, version)
    data[4:8] = b"SCCA"
    enc = name.encode("utf-16-le")
    data[0x10:0x10 + len(enc)] = enc
    lr, rc = {17: (0x78, 0x90), 23: (0x80, 0x98), 26: (0x80, 0xD0)}[version]
    struct.pack_into("<Q", data, lr, last)
    struct.pack_into("<I", data, rc, count)
    return bytes(data)


@pytest.mark.parametrize("version", [17, 23, 26])
def test_parse_prefetch_versions(version):
    pf = parse_prefetch(_prefetch_bytes("CCLEANER64.EXE", 2, FT, version))
    assert pf["executable"] == "CCLEANER64.EXE" and pf["run_count"] == 2
    assert pf["last_run"] == filetime_to_iso(FT)


def test_prefetch_finding_verifies_and_tamper_rejects(tmp_path):
    rel = "Windows/Prefetch/CCLEANER64.EXE-12345678.pf"
    p = tmp_path / rel
    p.parent.mkdir(parents=True)
    p.write_bytes(_prefetch_bytes("CCLEANER64.EXE", 2, FT))
    agent = _agent(tmp_path, [rel])
    rep = agent.run(agent.discover())
    (f,) = rep.findings
    assert f.category == "win.prefetch" and f.status == FindingStatus.VERIFIED
    assert f.value["run_count"] == 2 and "CCLEANER64.EXE executed 2 time(s)" in f.claim
    data = bytearray(p.read_bytes())
    data[0x98] = 9                                       # the run count changes after the agent read it
    p.write_bytes(bytes(data))
    assert HeuristicVerificationEngine(tmp_path).verify_finding(f.model_copy(deep=True)).status == FindingStatus.REJECTED
    agent.close()


def test_recycle_bin_i_file_vista_and_win10(tmp_path):
    path = "C:\\Users\\informant\\Desktop\\secret.docx"
    enc = path.encode("utf-16-le") + b"\x00\x00"
    deleted = _ft(datetime(2015, 3, 24, 19, 51, 47, tzinfo=timezone.utc))
    v1 = struct.pack("<QQQ", 1, 4096, deleted) + enc.ljust(520, b"\x00")
    v2 = struct.pack("<QQQI", 2, 4096, deleted, len(enc) // 2) + enc
    rels = []
    for i, blob in enumerate((v1, v2)):
        rel = f"$Recycle.Bin/S-1-5-21-1/$IABC{i}.docx"
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_bytes(blob)
        rels.append(rel)
    agent = _agent(tmp_path, rels)
    rep = agent.run(agent.discover())
    assert len(rep.findings) == 2
    for f in rep.findings:
        assert f.status == FindingStatus.VERIFIED and f.value["original_path"] == path
        assert f.value["deleted_utc"].startswith("2015-03-24T19:51:47")
    agent.close()


def test_xp_info2_records(tmp_path):
    rel = "RECYCLER/S-1-5-21-1-1003/INFO2"
    recs = b""
    for idx, name in enumerate(("lalsetup250.exe", "ethereal-setup-0.10.6.exe"), 1):
        full = f"C:\\Documents and Settings\\Mr. Evil\\Desktop\\{name}"
        r = full.encode("cp1252").ljust(260, b"\x00") + struct.pack("<IIQI", idx, 2, FT, 1000 * idx)
        recs += r + full.encode("utf-16-le").ljust(520, b"\x00")
    (tmp_path / rel).parent.mkdir(parents=True)
    (tmp_path / rel).write_bytes(struct.pack("<IIII", 5, 0, 0, 800) + b"\x00" * 4 + recs)
    agent = _agent(tmp_path, [rel])
    rep = agent.run(agent.discover())
    assert [f.value["original_path"].rsplit("\\", 1)[1] for f in rep.findings] == ["lalsetup250.exe",
                                                                                  "ethereal-setup-0.10.6.exe"]
    assert all(f.status == FindingStatus.VERIFIED for f in rep.findings)
    agent.close()


def test_kind_resolves_longpaths_through_manifest(tmp_path):
    (tmp_path / "_longpaths").mkdir()
    (tmp_path / "_longpaths/abc_SYSTEM").write_bytes(b"regf")
    manifest = {"files": [{"path": "fs/_longpaths/abc_SYSTEM",
                           "original_path": "vol1/Windows/System32/config/SYSTEM"},
                          {"path": "fs/vol1/Windows/System32/config/RegBack/SYSTEM"}]}
    agent = WindowsArtifactAgent(ForensicBridge(tmp_path), manifest=manifest)
    assert agent.kind("_longpaths/abc_SYSTEM") == "hive" and agent.hive_type("_longpaths/abc_SYSTEM") == "SYSTEM"
    assert agent.kind("vol1/Windows/System32/config/RegBack/SYSTEM") is None
    agent.close()


def test_registry_rendering_and_locator():
    assert render_registry_value(["a", "b"]) == "a|b"
    assert render_registry_value(b"\x01\xff") == "01ff"
    assert render_registry_value(300) == "300"
    assert split_registry_locator("\\Select::Current") == ("\\Select", "Current")
    assert split_registry_locator("\\Select") == ("\\Select", None)


def test_evtx_summary_is_deterministic_and_unescaped():
    xml = ('<Event><System><Provider Name="MsiInstaller"></Provider><EventID Qualifiers="0">11707</EventID>'
           '<TimeCreated SystemTime="2015-03-25 14:57:31+00:00"></TimeCreated><EventRecordID>42</EventRecordID>'
           '</System><EventData><Data>Product: Eraser -- Installation completed &amp;lt;ok&amp;gt;</Data></EventData></Event>')
    s = evtx_summary(xml)
    assert s == evtx_summary(xml)
    assert s.startswith("MsiInstaller EventID=11707 Record=42 Time=2015-03-25 14:57:31+00:00")
    assert "Product: Eraser" in s and "&amp;" not in s


# ------------------------------------------------------------------------------------------- verifier fixes
def _wal_db(root: Path) -> str:
    rel = "data/data/com.android.providers.telephony/databases/mmssms.db"
    db = root / rel
    db.parent.mkdir(parents=True)
    con = sqlite3.connect(db)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA wal_autocheckpoint=0")
    con.execute("CREATE TABLE sms (_id INTEGER PRIMARY KEY, address TEXT, body TEXT, date INTEGER, type INTEGER)")
    con.execute("INSERT INTO sms (address, body, date, type) VALUES ('+15551234567', 'meet at noon', 1580322045251, 1)")
    con.commit()
    # copy while the connection is open so the -wal survives (closing would checkpoint it away)
    snap = root.parent / "snap"
    shutil.copytree(root, snap)
    con.close()
    shutil.rmtree(root)
    shutil.copytree(snap, root)
    assert (root / f"{rel}-wal").exists()
    return rel


def test_sql_citation_covers_wal_side_file(tmp_path):
    root = tmp_path / "fs"
    rel = _wal_db(root)
    bridge = ForensicBridge(root)
    agent = AppArtifactAgent(bridge)
    rep = agent.run(agent.discover())
    f = next(f for f in rep.findings if f.category == "sms.message")
    assert f.status == FindingStatus.VERIFIED
    assert set(f.citations[0].side_hashes) == {"-wal"}
    wal = root / f"{rel}-wal"
    wal.write_bytes(wal.read_bytes() + b"\x00" * 8)        # the WAL changed after the agent read it
    out = HeuristicVerificationEngine(root).verify_finding(f.model_copy(deep=True))
    assert out.status == FindingStatus.REJECTED
    assert any("side-file mismatch" in (v.get("reason") or "") for v in out.verification)
    agent.close()
    bridge.close()


def test_sql_citation_without_side_hashes_rejected_when_wal_present(tmp_path):
    root = tmp_path / "fs"
    rel = _wal_db(root)
    bridge = ForensicBridge(root)
    c = Citation(file_path=rel, locator_type="sql_query", sql_query='SELECT "body" FROM "sms" WHERE rowid = ?',
                 sql_params=[1], expected_hash=bridge.sandbox.hash(rel), raw_text_snippet="meet at noon")
    assert not HeuristicVerificationEngine(root).verify_citation(c).verified
    c.side_hashes = bridge.sandbox.side_hashes(rel)
    assert HeuristicVerificationEngine(root).verify_citation(c).verified
    bridge.close()


def test_default_verifier_has_its_own_bridge(tmp_path):
    bridge = ForensicBridge(tmp_path)
    agent = AppArtifactAgent(bridge)
    assert agent.verifier.bridge is not bridge
    assert agent.verifier.sandbox._hash_cache is not bridge.sandbox._hash_cache
    assert agent.verifier.owns_bridge
    agent.close()
    shared = HeuristicVerificationEngine(tmp_path, bridge=bridge)
    assert not shared.owns_bridge
    bridge.close()


# ------------------------------------------------------------------------------------------- carving rank
def test_carve_score_prefers_prose_and_identifiers():
    prose = carve_score("Meet me at the north gate tomorrow at nine", "freelist_leaf")
    ident = carve_score("contact jo@m57.biz about the patent", "freeblock")
    noise = carve_score("aaaaaaaaaaaaaaaaaaaa", "unallocated")
    assert prose > noise and ident > noise


def test_rank_carved_dedupes_and_orders():
    fs = [{"value": {"text": "hello world again", "score": 0.4}}, {"value": {"text": "Hello, world again", "score": 0.5}},
          {"value": {"text": "other text here", "score": 0.9}}]
    out = rank_carved(fs)
    assert [f["value"]["text"] for f in out] == ["other text here", "Hello, world again"]


# ------------------------------------------------------------------------------------------- scoring
def _doc(fid: str, cat: str, claim: str) -> Doc:
    return Doc(fid, cat, "ev", claim, norm(claim))


def test_norm_decodes_urls_and_paths():
    assert norm("q=data+leakage%20methods") == "q=data leakage methods"
    assert norm(r"C:\Users\Informant\Desktop") == "c:/users/informant/desktop"


def test_score_question_groups_categories_and_same_finding():
    docs = [_doc("a", "win.timezone", "Time zone: Eastern Standard Time"),
            _doc("b", "win.prefetch", "CCLEANER64.EXE executed 2 time(s)"),
            _doc("c", "chrome.url", "Visited https://download.ccleaner.com/ccsetup504.exe")]
    q = {"id": "1", "question": "", "answer": "", "all": [["cat:win.timezone"], ["eastern"]], "same_finding": True}
    assert score_question(q, docs)["answered"]
    ran = {"id": "2", "question": "", "answer": "",
           "all": [["cat:win.prefetch|ccsetup504.exe", "cat:win.userassist|ccsetup504.exe"]]}
    assert not score_question(ran, docs)["answered"]          # a download is not an execution
    part = {"id": "3", "question": "", "answer": "", "all": [["ccleaner64.exe"], ["eraser.exe"]]}
    r = score_question(part, docs)
    assert r["credit"] == 0.5 and r["met"][0]["finding_id"] == "b" and r["missing"] == [["eraser.exe"]]


def test_gold_files_are_well_formed():
    files = sorted(GOLD_DIR.glob("*.json"))
    assert files
    for p in files:
        gold = json.loads(p.read_text(encoding="utf-8"))
        assert gold["case_id"] == p.stem and gold["source"]
        ids = [q["id"] for q in gold["questions"]]
        assert len(ids) == len(set(ids)) and not set(ids) & set(gold.get("excluded", {}))
        for q in gold["questions"]:
            assert q["question"] and q["answer"] and q["all"] and all(g for g in q["all"])
        assert score_gold(gold, [])["all"]["answered"] == 0


NIST_PC = CASES_DIR / "nist_data_leakage" / "evidence" / "nist_pc"


@pytest.mark.skipif(not (NIST_PC / "MANIFEST.json").exists(), reason="nist_data_leakage not fetched")
def test_nist_pc_registry_answers_are_verified():
    manifest = json.loads((NIST_PC / "MANIFEST.json").read_text(encoding="utf-8"))
    bridge = ForensicBridge(NIST_PC / "fs")
    agent = WindowsArtifactAgent(bridge, manifest=manifest)
    hives = [p for p in agent.discover() if agent.hive_type(p) in ("SYSTEM", "SOFTWARE")]
    rep = agent.run(hives)
    claims = {f.category: f.claim for f in rep.findings}
    assert all(f.status == FindingStatus.VERIFIED for f in rep.findings)
    assert claims["win.computer_name"] == "Computer name: INFORMANT-PC"
    assert "Windows 7 Ultimate" in claims["win.system_info"]
    serials = {f.value.get("serial") for f in rep.findings if f.category == "win.usb_device"}
    assert {"4C530012450531101593", "4C530012550531106501"} <= serials
    agent.close()
    bridge.close()
