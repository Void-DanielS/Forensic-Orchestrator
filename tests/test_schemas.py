"""FDX contracts: timestamps, artifact references, bridge records, training samples (no dataset or GPU needed)."""
from __future__ import annotations

import sqlite3
from pathlib import Path

import piexif
import pytest
from PIL import Image
from pydantic import ValidationError

from src.bridge.forensic_bridge import ForensicBridge
from src.datasets.finetune import guard_leakage, validate_jsonl
from src.schemas import ArtifactRef, ContentPart, Message, Provenance, TaskType, TrainingSample, to_timestamp
from src.verification.heuristic_verifier import HeuristicVerificationEngine
from src.verification.models import Finding, FindingStatus

UTC = "2020-02-09T18:23:53+00:00"
SHA = "a" * 64


# ------------------------------------------------------------------ timestamps
@pytest.mark.parametrize("raw,epoch", [
    (1581272633, "unix_s"), (1581272633000, "unix_ms"), ("1581272633000000", "unix_us"),
    (13225746233000000, "webkit_us"), (132257462330000000, "windows_filetime"),
    (602965433, "cocoa_s"), (602965433000000000, "cocoa_ns"), ("2020-02-09T13:23:53-05:00", "iso8601"),
])
def test_epochs_normalise_to_the_same_instant(raw, epoch):
    ts = to_timestamp(raw, epoch)
    assert ts.utc == UTC and ts.raw == raw and ts.epoch.value == epoch


def test_local_times_never_get_a_guessed_utc():
    ts = to_timestamp("2020:02:09 13:23:53", "exif")
    assert ts.utc is None and ts.local == "2020-02-09T13:23:53"
    assert to_timestamp("2020:02:09 13:23:53", "exif", utc_offset="-05:00").utc == UTC
    with pytest.raises(ValueError):
        to_timestamp("2020-02-09T13:23:53", "iso8601")


def test_zero_epoch_means_unset():
    assert to_timestamp(0, "unix_ms").utc is None


# ------------------------------------------------------------------ artifact references
def test_artifact_ref_round_trip_and_validation():
    ref = ArtifactRef(case_id="c1", evidence_id="e1", path="data/media/0/DCIM/x.jpg", sha256=SHA)
    assert ref.cases_path == "c1/evidence/e1/fs/data/media/0/DCIM/x.jpg"
    assert ArtifactRef.from_cases_path(ref.cases_path, SHA) == ref
    for bad in ("/abs/x", "C:/x", "a\\b", "a/../b"):
        with pytest.raises(ValidationError):
            ArtifactRef(case_id="c1", evidence_id="e1", path=bad)
    with pytest.raises(ValidationError):
        ArtifactRef(case_id="c1", evidence_id="e1", path="x", sha256="ABC")


# ------------------------------------------------------------------ bridge records
@pytest.fixture()
def bridge(tmp_path: Path) -> ForensicBridge:
    root = tmp_path / "fs"
    root.mkdir()
    con = sqlite3.connect(root / "sms.db")
    con.execute("CREATE TABLE sms (_id INTEGER PRIMARY KEY, body TEXT, date INTEGER)")
    con.execute("INSERT INTO sms (body, date) VALUES ('meet at the north gate', 1581272633000)")
    con.commit()
    con.close()
    (root / "wifi.log").write_text("boot\nconnected ssid=HomeNet\n", encoding="utf-8")
    exif = piexif.dump({"0th": {piexif.ImageIFD.Make: b"TestCam"}, "Exif": {}, "GPS": {}, "1st": {}})
    Image.new("RGB", (8, 8)).save(root / "p.jpg", exif=exif)
    b = ForensicBridge(root)
    yield b
    b.close()


def test_every_bridge_record_verifies(bridge: ForensicBridge):
    records = [bridge.sqlite.cell_record("sms.db", "sms", "body", 1, field="sms.body"),
               bridge.sqlite.cell_record("sms.db", "sms", "date", 1),
               bridge.text.line_record("wifi.log", 2),
               bridge.images.exif_record("p.jpg", "0th.Make")]
    assert [r.raw_text_snippet for r in records] == ["meet at the north gate", "1581272633000",
                                                      "connected ssid=HomeNet", "TestCam"]
    engine = HeuristicVerificationEngine(bridge.sandbox.root, bridge=bridge)
    for r in records:
        f = Finding(agent="test", artifact_path=r.file_path, category="t", claim="c", citations=[r.to_citation()])
        assert engine.verify_finding(f).status == FindingStatus.VERIFIED, r


def test_absent_values_yield_no_record(bridge: ForensicBridge):
    assert bridge.sqlite.cell_record("sms.db", "sms", "body", 99) is None
    assert bridge.text.line_record("wifi.log", 50) is None
    assert bridge.images.exif_record("p.jpg", "GPS.GPSLatitude") is None


def test_edited_record_is_rejected(bridge: ForensicBridge):
    r = bridge.sqlite.cell_record("sms.db", "sms", "body", 1)
    forged = r.model_copy(update={"raw_text_snippet": "meet at the south gate"})
    f = Finding(agent="test", artifact_path=r.file_path, category="t", claim="c", citations=[forged.to_citation()])
    assert HeuristicVerificationEngine(bridge.sandbox.root, bridge=bridge).verify_finding(f).status == FindingStatus.REJECTED


# ------------------------------------------------------------------ training samples
def _img(case="c1", sha=SHA):
    return ArtifactRef(case_id=case, evidence_id="e1", path="x.jpg", sha256=sha)


def _sample(split="train", case="c1", img=None, **kw) -> TrainingSample:
    args = dict(task=TaskType.IMAGE_TEXT_GROUNDING, split=split, target={"text": "hi"}, target_status="verified",
                messages=[Message(role="user", content=[ContentPart(type="image", image=img or _img(case)),
                                                        ContentPart(type="text", text="read")]),
                          Message.text("assistant", '{"text": "hi"}')],
                provenance=Provenance(case_id=case, generator="test"))
    args.update(kw)
    return TrainingSample(**args)


def test_sample_rules():
    s = _sample()
    assert s.sample_id == _sample().sample_id and len(s.sample_id) == 20
    with pytest.raises(ValidationError, match="VERIFIED"):
        _sample(target_status="uncorroborated")
    with pytest.raises(ValidationError, match="different case"):
        _sample(img=_img(case="other"))
    with pytest.raises(ValidationError, match="last message"):
        _sample(messages=[Message.text("user", "x")])
    with pytest.raises(ValidationError, match="sha256"):
        ContentPart(type="image", image=ArtifactRef(case_id="c1", evidence_id="e1", path="x.jpg"))
    with pytest.raises(ValidationError, match="label"):
        _sample(task=TaskType.EVIDENCE_STATUS, target_status="verified", label="maybe")


def test_leakage_guard_keeps_image_in_most_held_out_split():
    kept, dropped = guard_leakage([_sample("train"), _sample("test"), _sample("test")])
    assert [s.split.value for s in kept] == ["test"]
    assert dropped == {"cross_split_image": 1, "duplicate": 1}


def test_validate_jsonl(tmp_path: Path):
    p = tmp_path / "s.jsonl"
    p.write_text(_sample().model_dump_json() + "\n" + '{"task": "image_text_grounding"}\n', encoding="utf-8")
    ok, errors = validate_jsonl(p)
    assert ok == 1 and len(errors) == 1 and errors[0].startswith("line 2")
