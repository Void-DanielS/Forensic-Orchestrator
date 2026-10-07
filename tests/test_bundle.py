"""Generic evidence bundles (src/utils/bundle.py) and the remote transport (src/utils/remote.py).

Offline: no network access. Remote hosts are covered by URL-resolution tests only; the
container logic runs on local temporary files (ZIP, 7z with a password, nested archives).
"""
from __future__ import annotations

import gzip
import io
import json
import sqlite3
import subprocess
import tarfile
import zipfile
from pathlib import Path

import pytest

from src.cases.registry import CASES
from src.utils import bundle, remote
from src.utils.bundle import BundleFetcher, RemoteEvidence, artifact_category, group_files, segment_key, sniff


# ---------------------------------------------------------------- segment grouping
@pytest.mark.parametrize("name,expected", [
    ("case/disk.E01", ("ewf", "case/disk", 1)),
    ("case/disk.E10", ("ewf", "case/disk", 10)),
    ("disk.Ex01", None),
    ("SCHARDT.003", ("raw", "SCHARDT", 3)),
    ("BSidesAmman21.E01.7z.002", ("archive", "BSidesAmman21.E01.7z", 2)),
    ("CCTV/Crop_fit.z02", ("archive", "CCTV/Crop_fit.zip", 2)),
    ("CCTV/Crop_fit.zip", ("archive", "CCTV/Crop_fit.zip", 10_000)),
    ("set.part2.rar", ("archive", "set.rar", 2)),
    ("notes.txt", None),
    ("mail/logfile-2012-07-10-1000.eml", None),      # regression: .eml is not an EWF segment
    ("Message.framework/schema.sql", None),          # regression: .sql is not a SMART segment
    ("disk.EAB", ("ewf", "disk", 101)),
])
def test_segment_key(name, expected):
    assert segment_key(name) == expected


def test_group_files_orders_segments_and_keeps_singles():
    files = [(n, n) for n in ("a.E02", "a.E01", "b.zip", "b.z01", "readme.txt", "x.7z.002", "x.7z.001", "pic.jpg")]
    items = {(k, name): [m[0] for m in members] for k, name, members in group_files(files)}
    assert items[("ewf", "a")] == ["a.E01", "a.E02"]
    assert items[("archive", "b.zip")] == ["b.z01", "b.zip"]          # split zip: .z01 first, .zip last
    assert items[("archive", "x.7z")] == ["x.7z.001", "x.7z.002"]
    assert ("file", "readme.txt") in items and ("file", "pic.jpg") in items


def test_sniff_magic():
    assert sniff(b"EVF\x09\x0d\x0a\xff\x00" + b"\0" * 100) == "ewf"
    assert sniff(b"PK\x03\x04rest") == "zip"
    assert sniff(b"7z\xbc\xaf\x27\x1c....") == "archive"
    assert sniff(b"SQLite format 3\x00") == "sqlite"
    assert sniff(b"\xd4\xc3\xb2\xa1" + b"\0" * 20) == "pcap"
    mbr = bytearray(4096)
    mbr[510:512] = b"\x55\xaa"
    assert sniff(bytes(mbr)) == "disk"
    assert sniff(b"hello world") == "unknown"


@pytest.mark.parametrize("rel,size,expected", [
    ("net/capture.pcap", 50 << 20, "network_capture"),
    ("mem/physmem.raw", 512 << 20, "memory_image"),
    ("crash/minidump.dmp", 1 << 20, "crash_dump"),
    ("mail/box.pst", 10 << 20, "email"),
    ("tools/evil.exe", 1 << 20, None),                 # executables are never copied
    ("secrets/passwords.txt", 1000, None),             # credential files are never copied
    ("cctv/footage.mp4", 50 << 20, "video"),
    ("huge.iso", 4 << 30, None),
])
def test_artifact_category(rel, size, expected):
    assert artifact_category(rel, size) == expected


# ---------------------------------------------------------------- remote URL resolution (offline)
def test_direct_url_resolution():
    assert remote.direct_url("https://drive.google.com/file/d/1AbCdEfGhIjKlMn/view?usp=sharing") == \
        "https://drive.usercontent.google.com/download?id=1AbCdEfGhIjKlMn&export=download&confirm=t"
    assert remote.direct_url("ia:dfir-case1/Case1-Webserver.E01") == \
        "https://archive.org/download/dfir-case1/Case1-Webserver.E01"
    assert remote.direct_url("https://web.archive.org/web/20111012154048/http://x.org/a.tgz") == \
        "https://web.archive.org/web/20111012154048id_/http://x.org/a.tgz"
    assert remote.direct_url("https://www.dropbox.com/s/abc/file.zip?dl=0").endswith("dl=1")
    assert remote.direct_url("http://downloads.digitalcorpora.org/corpora/x/y.E01") == \
        "https://digitalcorpora.s3.amazonaws.com/corpora/x/y.E01"
    assert remote.gdrive_folder_id("https://drive.google.com/drive/folders/1-UrxFGpCo54bVujwFmmqNbsZEV28dSNz?usp=s") \
        == "1-UrxFGpCo54bVujwFmmqNbsZEV28dSNz"


# ---------------------------------------------------------------- local container dispatch
def _fetcher(tmp_path: Path, platform="generic", **kw) -> BundleFetcher:
    ev = RemoteEvidence("t_ev", "test evidence", platform, ("unused",), **kw)
    f = BundleFetcher(ev, tmp_path / "out")
    f.out.mkdir(parents=True)
    f.spool = tmp_path / "spool"
    f.spool.mkdir()
    return f


def _sqlite_bytes(tmp: Path) -> bytes:
    p = tmp / "x.db"
    con = sqlite3.connect(p)
    con.execute("create table t(a)")
    con.execute("insert into t values ('hello')")
    con.commit()
    con.close()
    return p.read_bytes()


def test_local_zip_tree_selects_artifacts_docs_and_skips_executables(tmp_path):
    inner = io.BytesIO()
    with zipfile.ZipFile(inner, "w") as z:
        z.writestr("photos/IMG_0001.jpg", b"\xff\xd8\xff\xe0" + b"\0" * 20000)
    outer = tmp_path / "evidence.zip"
    with zipfile.ZipFile(outer, "w") as z:
        z.writestr("README.md", "# Scenario\nThe suspect ...")
        z.writestr("solution.pdf", b"%PDF-1.4 answers")
        z.writestr("app/data.db", _sqlite_bytes(tmp_path))
        z.writestr("net/capture.pcap", b"\xd4\xc3\xb2\xa1" + b"\0" * 2000)
        z.writestr("tools/dropper.exe", b"MZ" + b"\0" * 5000)
        z.writestr("nested/inner.zip", inner.getvalue() + b"\0" * (2 << 20))  # > 1 MB: opened as a container
    f = _fetcher(tmp_path)
    with gzip.open(f.out / "FILELIST.jsonl.gz", "wt") as f.listing:
        f._dispatch_local([("evidence.zip", outer)])
    cats = {Path(r["path"]).name: r["category"] for r in f.records}
    assert cats["data.db"] in ("database", "app_database")
    assert cats["capture.pcap"] == "network_capture"
    assert "dropper.exe" not in cats
    assert (f.ctx / "brief" / "evidence__README.md").exists()
    assert (f.ctx / "answer_key" / "evidence__solution.pdf").exists()
    for r in f.records:   # every record hashes the bytes on disk
        import hashlib
        assert hashlib.sha256((f.out / r["path"]).read_bytes()).hexdigest() == r["sha256"]


def test_local_tar_stream(tmp_path):
    tarp = tmp_path / "phone.tar"
    with tarfile.open(tarp, "w") as t:
        data = _sqlite_bytes(tmp_path)
        info = tarfile.TarInfo("data/data/com.example/databases/msgs.db")
        info.size = len(data)
        t.addfile(info, io.BytesIO(data))
    f = _fetcher(tmp_path, platform="android")
    with gzip.open(f.out / "FILELIST.jsonl.gz", "wt") as f.listing:
        f._dispatch_local([("phone.tar", tarp)])
    assert [r["category"] for r in f.records] == ["app_database"]


@pytest.mark.skipif(not bundle.SEVEN_ZIP.exists(), reason="bundled 7-Zip not installed")
def test_7z_with_password_extracts_only_selected_members(tmp_path):
    src = tmp_path / "src"
    (src / "logs").mkdir(parents=True)
    (src / "logs" / "auth.log").write_text("Jan 1 sshd[1]: Accepted password for root\n")
    (src / "setup.exe").write_bytes(b"MZ" + b"\0" * 1000)
    arc = tmp_path / "case.7z"
    subprocess.run([str(bundle.SEVEN_ZIP), "a", "-pS3cret!", "-mhe=on", str(arc), str(src / "*")],
                   check=True, capture_output=True)
    f = _fetcher(tmp_path, platform="linux", password="S3cret!")
    with gzip.open(f.out / "FILELIST.jsonl.gz", "wt") as f.listing:
        f._dispatch_local([("case.7z", arc)])
    names = {Path(r["path"]).name for r in f.records}
    assert "auth.log" in names
    assert "setup.exe" not in names


@pytest.mark.skipif(not bundle.SEVEN_ZIP.exists(), reason="bundled 7-Zip not installed")
def test_single_stream_bz2_is_unpacked(tmp_path):
    import bz2
    arc = tmp_path / "capture.pcap.bz2"
    arc.write_bytes(bz2.compress(b"\xd4\xc3\xb2\xa1" + b"\0" * 4000))
    f = _fetcher(tmp_path, platform="network")
    with gzip.open(f.out / "FILELIST.jsonl.gz", "wt") as f.listing:
        f._dispatch_local([("capture.pcap.bz2", arc)])
    assert [r["category"] for r in f.records] == ["network_capture"]


def test_password_file_parsing(tmp_path, monkeypatch):
    class _R:
        def __init__(self, text):
            self.text = text

    import requests
    f = _fetcher(tmp_path, password="file:Password-Challenge1.txt")
    monkeypatch.setattr(requests, "get", lambda *a, **k: _R("Password = DFChallenge@s4a\n\nNote: use 7zip\n"))
    files = [remote.RemoteFile("Password-Challenge1.txt", "https://example.org/p.txt")]
    assert f.password(files) == "DFChallenge@s4a"
    f2 = _fetcher(tmp_path / "b", password="file:ZIP_Password.txt")
    monkeypatch.setattr(requests, "get", lambda *a, **k: _R("02DB2ECE91DB67E8FA939FC3DC15D16B\n"))
    assert f2.password([remote.RemoteFile("ZIP_Password.txt", "https://example.org/z.txt")]) == \
        "02DB2ECE91DB67E8FA939FC3DC15D16B"


# ---------------------------------------------------------------- registry invariants for the extended cases
def test_extended_registry_has_100_plus_cases_with_context():
    assert len(CASES) >= 100
    for c in CASES.values():
        assert c.evidence, c.case_id
        # every case carries context an examiner is given: publisher notes, a brief doc, or (core cases) docs
        assert c.publisher_notes or any(d.role == "brief" for d in c.docs) or c.summary, c.case_id


def test_publisher_notes_never_contain_solutions():
    leaks = ("the solution is", "the answer is", "the winner", "winning submission", "walk-through")
    for c in CASES.values():
        low = c.publisher_notes.lower()
        assert not any(k in low for k in leaks), c.case_id


def test_bundle_specs_do_not_ship_credentials_in_manifest():
    for c in CASES.values():
        for e in c.evidence:
            if isinstance(e, RemoteEvidence) and e.password:
                spec = {k: v for k, v in e.__dict__.items() if k not in ("password", "extra_passwords")}
                assert e.password not in json.dumps(spec)
