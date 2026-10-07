"""Case registry, computer-image selection, context extraction/contrast, long paths, GUI guards."""
from __future__ import annotations

import json
import os
import sqlite3
from pathlib import Path

import pytest

from src.bridge.forensic_bridge import ForensicBridge, Sandbox, SandboxViolation
from src.cases.docs import extract_text
from src.cases.registry import CASES, case_of_evidence
from src.context.case_context import CaseContext, InvestigationMemory, contrast
from src.utils import dataset_scraper as ds
from src.utils.disk_image import BlockCache, ConcatCache, classify_computer
from src.verification.models import Citation, Finding, FindingStatus


# ---------------------------------------------------------------- registry
def test_every_evidence_id_is_unique_and_resolvable():
    ids = [e for c in CASES.values() for e in c.evidence_ids()]
    assert len(ids) == len(set(ids))
    for e in ids:
        assert case_of_evidence(e) is not None
        if e in ds.SOURCES:
            assert ds.source_dir(ds.SOURCES[e]).parent.name == "evidence"


def test_answer_keys_are_never_briefs():
    for c in CASES.values():
        roles = {d.url: d.role for d in c.docs}
        for url, role in roles.items():
            if any(k in url.lower() for k in ("answer", "teacher", "instructor", "solution")):
                assert role == "answer_key", url


def test_computer_cases_exist():
    platforms = {getattr(e, "platform", None) for c in CASES.values() for e in c.evidence}
    assert {"windows", "removable", "linux"} <= platforms
    assert sum(1 for c in CASES.values() if len(c.evidence) >= 3) >= 4   # multi-device cases


# ---------------------------------------------------------------- computer selection
@pytest.mark.parametrize("path,platform,expected", [
    ("/Windows/System32/config/SOFTWARE", "windows", "registry"),
    ("/Windows/System32/config/SAM", "windows", None),
    ("/Windows/System32/config/SECURITY", "windows", None),
    ("/pagefile.sys", "windows", None),
    ("/$MFT", "windows", "fs_metadata"),
    ("/Users/bob/NTUSER.DAT", "windows", "registry"),
    ("/Users/bob/AppData/Local/Google/Chrome/User Data/Default/History", "windows", "app_database"),
    ("/Documents and Settings/jo/Cookies/jo@google[1].txt", "windows", "browser_history"),
    ("/Documents and Settings/jo/My Documents/plan.doc", "windows", "user_document"),
    ("/Users/bob/Pictures/beach.jpg", "windows", "user_image"),
    ("/Windows/Prefetch/CMD.EXE-1234.pf", "windows", "prefetch"),
    ("/Windows/System32/winevt/Logs/Security.evtx", "windows", "event_log"),
    ("/Program Files/App/logo.png", "windows", None),
    ("/home/pi/.bash_history", "linux", "shell_history"),
    ("/etc/shadow", "linux", None),
    ("/etc/hostname", "linux", "app_config"),
    ("/DCIM/100/IMG_1.JPG", "removable", "user_image"),
])
def test_classify_computer(path, platform, expected):
    assert classify_computer(path, platform, 1000) == expected


class _Mem:
    def __init__(self, data: bytes):
        self.data, self.size, self.calls = data, len(data), 0

    def get(self, start, n):
        self.calls += 1
        return self.data[start:start + n]


def test_block_cache_and_concat():
    a, b = _Mem(bytes(range(256)) * 40), _Mem(b"XYZ" * 1000)
    ca, cb = BlockCache(a, block=1024, max_blocks=4), BlockCache(b, block=1024, max_blocks=4)
    assert ca.read(1000, 100) == a.data[1000:1100]
    calls = a.calls
    assert ca.read(1010, 10) == a.data[1010:1020] and a.calls == calls      # served from cache
    cc = ConcatCache([ca, cb])
    edge = a.size - 5
    assert cc.read(edge, 10) == a.data[edge:] + b.data[:5]
    assert cc.size == a.size + b.size


# ---------------------------------------------------------------- context
BRIEF = """AFFIDAVIT
On 11/20/2009 Detective Aaron Greene interviewed Jo Smith of M57.biz about leaked files.
Jo Smith said the file patent-plan.docx was emailed to rival@competitor.com from a Windows XP laptop.
Contact phone 831-555-0199. Pat said the leak happened before December 4, 2009.
"""


@pytest.fixture()
def ctx(tmp_path):
    d = tmp_path / "case1" / "context" / "brief"
    d.mkdir(parents=True)
    (d / "affidavit.txt").write_text(BRIEF, encoding="utf-8")
    (tmp_path / "case1" / "context" / "answer_key").mkdir()
    (tmp_path / "case1" / "context" / "answer_key" / "solution.txt").write_text("The thief was Zelda Secretname", encoding="utf-8")
    return CaseContext.load(tmp_path / "case1", "Start-up data leak.")


def test_context_entities(ctx):
    by = {(e.type, e.value) for e in ctx.entities.values()}
    assert ("person", "Jo Smith") in by and ("person", "Aaron Greene") in by
    assert ("email", "rival@competitor.com") in by
    assert ("filename", "patent-plan.docx") in by
    assert ("phone", "831-555-0199") in by
    assert any(t == "device" and "Windows XP" in v for t, v in by)
    assert ctx.timeframe and ctx.timeframe["start"] <= "2009-11-20" <= ctx.timeframe["end"]


def test_answer_key_never_loaded(ctx):
    assert not any("zelda" in e.norm for e in ctx.entities.values())


def _f(status, category, claim, value=None, observed=None):
    f = Finding(agent="t", artifact_path="a.jpg", category=category, claim=claim, value=value, observed_text=observed)
    f.status = status
    return f


def test_contrast_relations(ctx):
    v = FindingStatus.VERIFIED
    rel = lambda f: {l["relation"] for l in contrast(ctx, f)}  # noqa: E731
    assert "corroborates" in rel(_f(v, "ocr.corroborated", "text", {"text": "send to rival@competitor.com"}))
    assert "outside_timeframe" in rel(_f(v, "exif.capture_time", "t", "2015:01:01 10:00:00"))
    assert "outside_timeframe" not in rel(_f(v, "exif.capture_time", "t", "2009:11:25 10:00:00"))
    assert "not_in_context" in rel(_f(v, "exif.camera_model", "Model = 'Canon EOS'", "Canon EOS"))
    assert "new_lead" in rel(_f(v, "exif.gps", "geotag", {"latitude": 1.0}))
    assert rel(_f(FindingStatus.UNCORROBORATED, "vlm.description", "a photo of Jo Smith")) == {"unverified_mention"}
    assert rel(_f(FindingStatus.REJECTED, "ocr.corroborated", "Jo Smith")) == set()


def test_memory_roundtrip(ctx, tmp_path):
    mem = InvestigationMemory(tmp_path / "case1", ctx)
    f = _f(FindingStatus.VERIFIED, "ocr.corroborated", "Jo Smith")
    mem.record("run1", "ev1", f, contrast(ctx, f))
    mem.save()
    data = json.loads((tmp_path / "case1" / "workspace" / "memory.json").read_text(encoding="utf-8"))
    assert data["links"][0]["finding_id"] == f.finding_id and data["summary"]["links"]["corroborates"] >= 1
    assert InvestigationMemory(tmp_path / "case1", ctx).links == mem.links


# ---------------------------------------------------------------- documents
def test_docx_and_xlsx_text(tmp_path):
    import zipfile

    docx = tmp_path / "a.docx"
    with zipfile.ZipFile(docx, "w") as z:
        z.writestr("word/document.xml", '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
                   "<w:body><w:p><w:r><w:t>Hello brief</w:t></w:r></w:p></w:body></w:document>")
    assert "Hello brief" in extract_text(docx)
    html = tmp_path / "b.html"
    html.write_text("<html><script>x</script><p>Case &amp; scope</p></html>", encoding="utf-8")
    assert "Case & scope" in extract_text(html) and "x" not in extract_text(html).replace("Case", "")


# ---------------------------------------------------------------- long paths
@pytest.mark.skipif(os.name != "nt", reason="Windows extended-length paths")
def test_long_paths_resolve_and_stay_confined(tmp_path):
    (tmp_path / "fs").mkdir()
    sb = Sandbox(tmp_path / "fs")
    long_dir = sb.root / ("d" * 120) / ("e" * 120)
    os.makedirs(long_dir)
    (long_dir / "x.db").write_bytes(b"")
    con = sqlite3.connect(":memory:")
    con.close()
    rel = f"{'d' * 120}/{'e' * 120}/x.db"
    assert len(str(tmp_path / "fs" / rel)) > 260
    assert sb.resolve(rel).is_file()
    assert sb.hash(rel)
    with pytest.raises(SandboxViolation):
        sb.resolve("../outside")
    assert [e.path for e in ForensicBridge(tmp_path / "fs").fs.walk()] == [rel]


# ---------------------------------------------------------------- text-file corroboration, identifier quoting
def test_text_line_citation_and_quoted_columns(tmp_path):
    from src.bridge.text_index import CaseTextIndex
    from src.verification.heuristic_verifier import HeuristicVerificationEngine

    fs = tmp_path / "fs"
    (fs / "mail").mkdir(parents=True)
    (fs / "mail" / "msg.txt").write_text("From: jo\nPlease send the quantum cryptography patent drafts\n", encoding="utf-8")
    con = sqlite3.connect(fs / "q.db")
    con.execute('create table m("we""ird" text)')
    con.execute("insert into m values('a distinctive message body here')")
    con.commit()
    con.close()
    b = ForensicBridge(fs)
    idx = CaseTextIndex(b).build()
    assert idx.stats()["text_files"] == 1 and not idx.skipped_columns
    eng = HeuristicVerificationEngine(fs, bridge=b)
    (s1, line), = idx.search("send the quantum cryptography patent drafts")
    assert line.kind == "line" and line.rowid == 2
    c = Citation(file_path=line.db, locator_type="line_number", line_number=line.rowid,
                 expected_hash=b.sandbox.hash(line.db), raw_text_snippet=line.value)
    assert eng.verify_citation(c).verified
    (s2, cell), = idx.search("a distinctive message body here")
    c = Citation(file_path=cell.db, locator_type="sql_query", sql_query=cell.citation_sql(), sql_params=[cell.rowid],
                 expected_hash=b.sandbox.hash(cell.db), raw_text_snippet=cell.value)
    assert eng.verify_citation(c).verified
    b.close()


def test_latin1_not_misread_as_utf16(tmp_path):
    (tmp_path / "l.log").write_bytes("café olé!\n".encode("latin-1"))
    assert ForensicBridge(tmp_path).text.read_text("l.log") == "café olé!\n"
