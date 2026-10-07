"""Self-contained tests for the verification engine and bridge (no dataset or GPU needed)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import piexif
import pytest
from PIL import Image

from src.bridge.forensic_bridge import ForensicBridge, SandboxViolation
from src.bridge.text_index import CaseTextIndex
from src.verification.heuristic_verifier import HeuristicVerificationEngine, corroboration, text_similarity
from src.verification.models import Citation, Finding, FindingStatus

MSG = "Running late, see you at the north entrance in 10 minutes"


@pytest.fixture()
def sandbox(tmp_path: Path) -> Path:
    root = tmp_path / "fs"
    (root / "db").mkdir(parents=True)
    con = sqlite3.connect(root / "db" / "msgs.db")
    con.execute("CREATE TABLE m (_id INTEGER PRIMARY KEY, body TEXT)")
    con.executemany("INSERT INTO m (body) VALUES (?)", [(MSG,), ("ok",), ("Lunch tomorrow?",)])
    con.commit()
    con.close()
    (root / "log.txt").write_text("boot\nwifi connected ssid=HomeNet\nshutdown\n", encoding="utf-8")
    exif = {"0th": {piexif.ImageIFD.Make: b"TestCam", piexif.ImageIFD.Model: b"X1"},
            "GPS": {piexif.GPSIFD.GPSLatitudeRef: b"N", piexif.GPSIFD.GPSLatitude: ((35, 1), (39, 1), (3960, 100)),
                    piexif.GPSIFD.GPSLongitudeRef: b"W", piexif.GPSIFD.GPSLongitude: ((78, 1), (52, 1), (4770, 100))}}
    Image.new("RGB", (32, 32), "white").save(root / "img.jpg", exif=piexif.dump(exif))
    return root


@pytest.fixture()
def env(sandbox: Path):
    bridge = ForensicBridge(sandbox)
    yield bridge, HeuristicVerificationEngine(sandbox, bridge=bridge)
    bridge.close()


def _cite(bridge, path, locator_type, snippet, **kw) -> Citation:
    return Citation(file_path=path, locator_type=locator_type, expected_hash=bridge.sandbox.hash(path),
                    raw_text_snippet=snippet, **kw)


def _finding(*cites, observed=None) -> Finding:
    return Finding(agent="t", artifact_path="x", category="t", claim="t", citations=list(cites), observed_text=observed)


def test_sql_true_and_hallucinated(env):
    bridge, eng = env
    ok = eng.verify_finding(_finding(_cite(bridge, "db/msgs.db", "sql_query", MSG, sql_query="SELECT body FROM m")))
    bad = eng.verify_finding(_finding(_cite(bridge, "db/msgs.db", "sql_query", "Meet at the docks",
                                            sql_query="SELECT body FROM m")))
    assert ok.status == FindingStatus.VERIFIED
    assert bad.status == FindingStatus.REJECTED


@pytest.mark.parametrize("sql", ["DELETE FROM m", "UPDATE m SET body='x'", "DROP TABLE m",
                                 "WITH x AS (SELECT 1) DELETE FROM m", "ATTACH DATABASE 'y.db' AS y"])
def test_sql_writes_refused_and_evidence_unchanged(env, sql):
    bridge, eng = env
    before = bridge.sandbox.hash("db/msgs.db")
    f = eng.verify_finding(_finding(_cite(bridge, "db/msgs.db", "sql_query", MSG, sql_query=sql)))
    assert f.status == FindingStatus.REJECTED
    assert bridge.sqlite.query("db/msgs.db", "SELECT COUNT(*) FROM m")[0][0] == 3
    bridge.sandbox._hash_cache.clear()
    assert bridge.sandbox.hash("db/msgs.db") == before


def test_hash_mismatch_rejected(env):
    bridge, eng = env
    c = _cite(bridge, "log.txt", "line_number", "wifi", line_number=2, match_mode="contains")
    c.expected_hash = "f" * 64
    assert eng.verify_finding(_finding(c)).status == FindingStatus.REJECTED


def test_line_regex_byte_offset(env):
    bridge, eng = env
    data = (bridge.sandbox.root / "log.txt").read_bytes()
    cites = [
        _cite(bridge, "log.txt", "line_number", "ssid=HomeNet", line_number=2, match_mode="contains"),
        _cite(bridge, "log.txt", "regex", "HomeNet", regex=r"ssid=(?P<v>\w+)"),
        _cite(bridge, "log.txt", "byte_offset", "wifi connected", byte_offset=data.index(b"wifi")),
    ]
    assert eng.verify_finding(_finding(*cites)).status == FindingStatus.VERIFIED
    wrong = _cite(bridge, "log.txt", "line_number", "ssid=HomeNet", line_number=1, match_mode="contains")
    assert eng.verify_finding(_finding(wrong)).status == FindingStatus.REJECTED


def test_exif_and_gps(env):
    bridge, eng = env
    assert bridge.images.gps("img.jpg") == pytest.approx({"latitude": 35.661, "longitude": -78.87991667}, abs=1e-6)
    good = _cite(bridge, "img.jpg", "exif_tag", "TestCam", exif_tag="0th.Make")
    lat = _cite(bridge, "img.jpg", "exif_tag", "35/1,39/1,3960/100", exif_tag="GPS.GPSLatitude")
    bad = _cite(bridge, "img.jpg", "exif_tag", "Apple", exif_tag="0th.Make")
    assert eng.verify_finding(_finding(good, lat)).status == FindingStatus.VERIFIED
    assert eng.verify_finding(_finding(bad)).status == FindingStatus.REJECTED


def test_sandbox_escape(env, sandbox):
    bridge, eng = env
    with pytest.raises(SandboxViolation):
        bridge.sandbox.resolve("../outside.txt")
    c = Citation(file_path="../../etc/passwd", locator_type="line_number", line_number=1,
                 expected_hash="0" * 64, raw_text_snippet="root")
    assert eng.verify_finding(_finding(c)).status == FindingStatus.REJECTED


def test_ocr_corroboration(env):
    bridge, eng = env
    cite = _cite(bridge, "db/msgs.db", "sql_query", MSG, sql_query="SELECT body FROM m WHERE rowid = ?", sql_params=[1])
    # One OCR line of the message, with a small OCR error.
    assert eng.verify_finding(_finding(cite, observed="see you at the n0rth entrance")).status == FindingStatus.VERIFIED
    assert eng.verify_finding(_finding(cite, observed="totally different words entirely")).status == FindingStatus.REJECTED


def test_short_truth_cannot_verify_long_observation():
    # "Lunch tomorrow?" is contained in the observed line, but explains little of it.
    assert text_similarity("Lunch tomorrow? Bring the documents and the cash", "Lunch tomorrow?") < 0.5
    assert text_similarity("see you at the north", MSG) == 1.0


def test_uncited_is_uncorroborated(env):
    _, eng = env
    assert eng.verify_finding(_finding()).status == FindingStatus.UNCORROBORATED


def test_text_index_proposes_verifiable_citation(env):
    bridge, eng = env
    idx = CaseTextIndex(bridge).build()
    score, cell = idx.search("see you at the north entrance")[0]
    assert cell.value == MSG and score == 1.0
    cite = _cite(bridge, cell.db, "sql_query", cell.value, sql_query=cell.citation_sql(), sql_params=[cell.rowid])
    assert eng.verify_finding(_finding(cite, observed="see you at the north entrance")).status == FindingStatus.VERIFIED


def test_wal_db_read_without_touching_evidence(tmp_path):
    root = tmp_path / "fs"
    root.mkdir()
    db = root / "w.db"
    con = sqlite3.connect(db)
    con.execute("PRAGMA journal_mode=WAL")
    con.execute("PRAGMA wal_autocheckpoint=0")
    con.execute("CREATE TABLE t (v TEXT)")
    con.execute("INSERT INTO t VALUES ('only in wal')")
    con.commit()
    # Snapshot db + wal while the writer is open, as an acquisition would.
    snap = tmp_path / "snap"
    snap.mkdir()
    for suffix in ("", "-wal"):
        (snap / f"w.db{suffix}").write_bytes((root / f"w.db{suffix}").read_bytes())
    con.close()
    bridge = ForensicBridge(snap)
    before = {p.name: p.read_bytes() for p in snap.iterdir()}
    assert bridge.sqlite.query("w.db", "SELECT v FROM t") == [("only in wal",)]
    assert {p.name: p.read_bytes() for p in snap.iterdir()} == before
    bridge.close()


def test_short_text_inside_blob_is_not_specific():
    blob = "air.com.sgn.cookiejamblast.gp,app.WTInfoTech.WorldAroundMeLite,com.radioshack.RADIO SHACK,com.x.y" * 3
    assert corroboration("RADIO SHACK", blob)[0] is False
    assert corroboration("Super Secret Message", "Super Secret Message Apps")[0] is True
    assert corroboration("@user2812914319221", "user2812914319221")[0] is True
    # Long, distinctive text may be one line of a longer message.
    assert corroboration("see you at the north entrance in 10", MSG + " " + MSG)[0] is True


def test_vlm_placeholders_filtered():
    from src.agents.vision_agent import _PLACEHOLDER

    for s in ("not visible text", "No text visible", "N/A", "none", "..."):
        assert _PLACEHOLDER.match(s)
    assert not _PLACEHOLDER.match("Notes from meeting")
    assert not _PLACEHOLDER.match("Super Secret Message")
