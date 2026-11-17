"""Evidence integrity audit / repair (``src/utils/integrity.py``) on a synthetic case: no network."""
from __future__ import annotations

import hashlib
import json
import os
import stat
from pathlib import Path

import pytest

from src.utils import integrity


class FakeCase:
    def __init__(self, root: Path, case_id: str = "c1"):
        self.case_id, self.dir = case_id, root / case_id

    def evidence_ids(self):
        return ["ev"]


def _case(tmp_path: Path) -> tuple[FakeCase, dict[str, bytes]]:
    case = FakeCase(tmp_path)
    ev = case.dir / "evidence" / "ev"
    content = {"fs/a.txt": b"alpha", "fs/Recent/desktop.ini": b"[.ShellClassInfo]\r\nX=1\r\n",
               "fs/db.sqlite-shm": b"\x00" * 64, "fs/gone.exe": b"MZ" + b"\x00" * 30}
    files = []
    for rel, data in content.items():
        p = ev / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        files.append({"path": rel, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    (ev / "MANIFEST.json").write_text(json.dumps({"kind": "zip", "files": files}), encoding="utf-8")
    return case, content


def test_audit_detects_missing_resized_and_rewritten_files(tmp_path):
    case, _ = _case(tmp_path)
    ev = case.dir / "evidence" / "ev"
    (ev / "fs/gone.exe").unlink()
    (ev / "fs/Recent/desktop.ini").write_bytes(b"[.ShellClassInfo]\r\nCLSID={0}\r\nX=1\r\n")   # shell rewrite
    (ev / "fs/db.sqlite-shm").write_bytes(b"\x01" + b"\x00" * 63)                           # same size, new bytes
    res = integrity.audit([case], progress=False)
    kinds = {p.path: p.kind for p in res.problems}
    assert kinds == {"fs/gone.exe": "missing", "fs/Recent/desktop.ini": "size_mismatch",
                     "fs/db.sqlite-shm": "hash_mismatch"}
    assert res.ok == 1 and res.checked == 4
    fast = integrity.audit([case], do_hash=False, progress=False)
    assert "fs/db.sqlite-shm" not in {p.path for p in fast.problems}     # only hashing sees same-size rewrites


def test_reacquire_refuses_bytes_that_do_not_match_the_acquisition_hash(tmp_path, monkeypatch):
    case, content = _case(tmp_path)
    monkeypatch.setattr(integrity, "_zip_member_bytes", lambda man, f: b"tampered")
    man = json.loads((case.dir / "evidence/ev/MANIFEST.json").read_text(encoding="utf-8"))
    man["archive_url"] = "https://example.invalid/x.zip"
    (case.dir / "evidence/ev/MANIFEST.json").write_text(json.dumps(man), encoding="utf-8")
    with pytest.raises(integrity.Unrepairable):
        integrity.reacquire(case.dir, "ev", "fs/a.txt")
    monkeypatch.setattr(integrity, "_zip_member_bytes", lambda man, f: content[f["path"]])
    assert integrity.reacquire(case.dir, "ev", "fs/a.txt") == b"alpha"


def test_restore_and_protect(tmp_path):
    case, content = _case(tmp_path)
    ev = case.dir / "evidence" / "ev"
    target = ev / "fs/Recent/desktop.ini"
    target.write_bytes(b"rewritten")
    os.chmod(target, stat.S_IREAD)                                    # restore must work on read-only files
    integrity._write_evidence(target, content["fs/Recent/desktop.ini"], 1_600_000_000.0)
    assert target.read_bytes() == content["fs/Recent/desktop.ini"]
    assert int(target.stat().st_mtime) == 1_600_000_000
    n = integrity.protect([case])
    assert n == 2                                                     # desktop.ini and -shm only
    assert not os.stat(target).st_mode & stat.S_IWRITE
    assert os.stat(ev / "fs/a.txt").st_mode & stat.S_IWRITE
    with pytest.raises(PermissionError):
        target.write_bytes(b"shell rewrite")
    for p in (target, ev / "fs/db.sqlite-shm"):                       # let pytest clean up
        os.chmod(p, stat.S_IWRITE | stat.S_IREAD)
