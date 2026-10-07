"""Offline tests for corpus selection, path mapping and streaming helpers."""
from __future__ import annotations

import io
import tarfile
import zipfile
import zlib
from pathlib import Path

import pytest

from src.utils import dataset_scraper as ds


@pytest.mark.parametrize("path,platform,expected", [
    ("data/data/com.whatsapp/databases/msgstore.db", "android", "app_database"),
    ("data/data/com.whatsapp/databases/msgstore.db-wal", "android", "app_database"),
    ("data/data/com.android.chrome/app_chrome/Default/History", "android", "app_database"),
    ("data/data/com.android.chrome/app_chrome/Default/Login Data-journal", "android", "app_database"),
    ("data/data/com.android.chrome/app_chrome/Default/Preferences", "android", None),
    ("data/user_de/0/com.android.providers.telephony/databases/mmssms.db", "android", "app_database"),
    ("apps/com.kik.android/db/kikDatabase.db", "android", "app_database"),
    ("data/data/com.x/shared_prefs/settings.xml", "android", "app_config"),
    ("data/media/0/DCIM/Camera/IMG_1.jpg", "android", "user_image"),
    ("data/media/0/Pictures/Screenshots/Screenshot_1.png", "android", "screenshot"),
    ("data/media/0/Android/data/com.app/cache/x.jpg", "android", "app_image"),
    ("data/data/org.telegram.messenger/files/photo.jpg", "android", "app_image"),
    ("private/var/mobile/Containers/Shared/AppGroup/ABC/Message/Media/a.jpg", "ios", "user_image"),
    ("private/var/mobile/Containers/Data/Application/ABC/Documents/x.png", "ios", "app_image"),
    ("private/var/containers/Bundle/Application/ABC/App.app/icon.png", "ios", None),
    ("sbin/.magisk/mirror/data/data/a/databases/b.db", "android", None),
    ("data/system/packages.xml", "android", "system_log"),
    ("private/var/mobile/Library/SMS/sms.db", "ios", "database"),
    ("HomeDomain/Library/SMS/sms.db", "ios", "database"),
    ("CameraRollDomain/Media/DCIM/100APPLE/IMG_0001.HEIC", "ios", "user_image"),
    ("private/var/mobile/Media/DCIM/100APPLE/IMG_0002.PNG", "ios", "screenshot"),
    ("HomeDomain/Library/Preferences/com.apple.x.plist", "ios", "app_config"),
    ("Takeout/Google Photos/2020/a.jpg", "takeout", "user_image"),
    ("Takeout/My Activity/Search/MyActivity.json", "takeout", "cloud_record"),
])
def test_classify(path, platform, expected):
    assert ds.classify(path, platform) == expected


def test_context_doc_detection_excludes_credentials():
    assert ds.is_context_doc("iOS 13.4.1 Extraction/iOS13-4-1-ImageCreation.pdf")
    assert not ds.is_context_doc("Jess_CTF_iPhone8/x_passwords.txt")
    assert not ds.is_context_doc("data/data/app/files/readme.txt/deep/a/b.pdf")


def test_dest_map_windows_safe(tmp_path: Path):
    m = ds._DestMap(tmp_path / "fs")
    p, orig = m.map("data/a:b/con/file?.db", "x")
    assert p.relative_to(tmp_path / "fs").as_posix() == "data/a_b/_con/file_.db" and orig == "data/a:b/con/file?.db"
    p1, _ = m.map("data/Foo.db", "m1")
    p2, orig2 = m.map("data/foo.db", "m2")  # differs only by case
    assert p1 != p2 and orig2 == "data/foo.db"
    long = "/".join(["d" * 50] * 8) + "/x.db"
    p3, orig3 = m.map(long, "m3")
    assert "_longpaths" in p3.parts and orig3 == long and len(str(p3)) <= ds.MAX_WIN_PATH


def test_budget():
    b = ds._Budget(max_bytes=100, max_images=1)
    assert b.accept("database", 60) and not b.accept("database", 60)
    b = ds._Budget(max_bytes=10**9, max_images=1)
    assert not b.accept("user_image", 100)            # icon-sized
    assert b.accept("user_image", 20000) and not b.accept("app_image", 20000)


class _Mem:
    def __init__(self, data: bytes):
        self.data, self.size = data, len(data)

    def get(self, start: int, length: int) -> bytes:
        return self.data[start:start + length]


def _zip_bytes(files: dict[str, bytes], compression=zipfile.ZIP_DEFLATED) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression) as z:
        for n, d in files.items():
            z.writestr(n, d)
    return buf.getvalue()


def test_member_stream_inflates_and_checks_crc():
    payload = b"hello forensic world " * 5000
    src = _Mem(_zip_bytes({"a.bin": payload}))
    info = ds._open_zip(src).getinfo("a.bin")
    assert ds._read_member(src, info) == payload
    assert ds._member_stream(src, info, "t").read() == payload


def test_nested_stored_zip_is_range_addressable():
    inner = _zip_bytes({"data/data/x/databases/a.db": b"SQLite format 3\x00" + b"\0" * 100})
    outer = _Mem(_zip_bytes({"inner.zip": inner}, zipfile.ZIP_STORED))
    info = ds._open_zip(outer).getinfo("inner.zip")
    sub = ds.SubSource(outer, ds._data_offset(outer, info), info.file_size)
    z = ds._open_zip(sub)
    assert ds._read_member(sub, z.getinfo("data/data/x/databases/a.db")).startswith(b"SQLite format 3")


def test_tar_stream_selects_and_sniffs_sqlite(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "RAW_IMAGES_DIR", tmp_path)
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w") as tf:
        for name, data in {
            "data/data/app/databases/a.db": b"SQLite format 3\x00" + b"1" * 5000,
            "data/data/app/files/noext_store": b"SQLite format 3\x00" + b"2" * 5000,   # sniffed by magic
            "data/data/app/files/blob": b"x" * 5000,                                  # skipped
            "data/media/0/DCIM/Camera/p.jpg": b"\xff\xd8" + b"3" * 10000,
            "Image Creation.pdf": b"%PDF-1.4",                                        # not matched (no keyword)
            "Android9-ImageCreation.pdf": b"%PDF-1.4 doc",
        }.items():
            ti = tarfile.TarInfo(name)
            ti.size, ti.mtime = len(data), 1580000000
            tf.addfile(ti, io.BytesIO(data))
    buf.seek(0)
    src = ds.Source("t", "t", "android", "x", kind="tar")
    f = ds._Fetcher(src)
    f.fetch_tar(buf, "r|")
    got = {r["source_member"]: r["category"] for r in f.records}
    assert got == {"data/data/app/databases/a.db": "app_database", "data/data/app/files/noext_store": "database",
                   "data/media/0/DCIM/Camera/p.jpg": "user_image"}
    assert (tmp_path / "t" / "fs" / "data/data/app/files/noext_store").read_bytes().startswith(b"SQLite format 3")
    assert (tmp_path / "t" / "_case_context" / "Android9-ImageCreation.pdf").exists()
    rec = next(r for r in f.records if r["category"] == "database")
    assert rec["size"] == 5016


def test_tar_stream_recurses_into_nested_tar(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "RAW_IMAGES_DIR", tmp_path)

    def tar_of(files: dict[str, bytes]) -> bytes:
        b = io.BytesIO()
        with tarfile.open(fileobj=b, mode="w") as tf:
            for n, d in files.items():
                ti = tarfile.TarInfo(n)
                ti.size = len(d)
                tf.addfile(ti, io.BytesIO(d))
        return b.getvalue()

    inner = tar_of({"data/data/app/databases/x.db": b"SQLite format 3\x00" + b"0" * (2 << 20)})
    outer = io.BytesIO(tar_of({"Android 9/Google Pixel 3.tar": inner}))
    f = ds._Fetcher(ds.Source("n", "n", "android", "x", kind="tar"))
    f.fetch_tar(outer, "r|")
    assert [r["source_member"] for r in f.records] == ["Android 9/Google Pixel 3.tar/data/data/app/databases/x.db"]


def test_tar_stream_spools_nested_zip(tmp_path, monkeypatch):
    monkeypatch.setattr(ds, "RAW_IMAGES_DIR", tmp_path)
    monkeypatch.setattr(ds, "SPOOL_DIR", tmp_path / "spool")
    payload = bytes([0x53, 0x51, 0x4C]) + b"ite format 3" + bytes([0]) + b"z" * (2 << 20)
    inner = _zip_bytes({"Dump/data/data/app/databases/c.db": payload}, zipfile.ZIP_STORED)
    b = io.BytesIO()
    with tarfile.open(fileobj=b, mode="w") as tf:
        ti = tarfile.TarInfo("Img/EXTRACTION_FFS 01/EXTRACTION_FFS.zip")
        ti.size = len(inner)
        tf.addfile(ti, io.BytesIO(inner))
    b.seek(0)
    f = ds._Fetcher(ds.Source("z", "z", "android", "x", kind="tar"))
    f.fetch_tar(b, "r|")
    assert [r["source_member"] for r in f.records] == ["Img/EXTRACTION_FFS 01/EXTRACTION_FFS.zip/Dump/data/data/app/databases/c.db"]
    assert not list((tmp_path / "spool").glob("*"))  # spool cleaned up
