"""Jev typed-answer model: fragment labelling, dataset build, a short training run, the query API and the
JevAgent contract (answers are UNCORROBORATED, never evidence), all on synthetic fixtures on CPU."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch

from src.agents.jev_agent import JevAgent
from src.bridge.forensic_bridge import ForensicBridge
from src.ml import fragments as fr
from src.ml.jev import FRAG, Jev, JevConfig, JevNet, encode_ext, score_targets
from src.ml.train import auroc, classification_metrics, ece, fit_temperature, train
from src.verification.models import FindingStatus


def _body(kind: str, n: int, rng: random.Random) -> bytes:
    """Distinct, learnable byte statistics per synthetic type."""
    if kind == "sqlite":
        page = b"SQLite format 3\x00" + bytes(rng.randrange(0, 16) for _ in range(FRAG - 16))
        return (page * (n // FRAG + 1))[:n]
    if kind == "jpeg":
        return bytes(rng.randrange(256) for _ in range(n))
    words = ["evidence", "report", "user", "login", "the", "file", "case"]
    return (" ".join(rng.choice(words) for _ in range(n)).encode())[:n]


HEADERS = {"sqlite": b"SQLite format 3\x00", "jpeg": b"\xff\xd8\xff\xe0", "text": b"plain text "}
EXT = {"sqlite": ".db", "jpeg": ".jpg", "text": ".txt"}


class FakeCase:
    def __init__(self, root: Path, case_id: str):
        self.case_id, self.dir = case_id, root / case_id

    def evidence_ids(self):
        return ["ev"]


def _make_case(root: Path, case_id: str, files_per_class: int, seed: int) -> FakeCase:
    rng = random.Random(seed)
    case = FakeCase(root, case_id)
    fs = case.dir / "evidence" / "ev" / "fs"
    fs.mkdir(parents=True)
    files = []
    for kind in HEADERS:
        for i in range(files_per_class):
            data = HEADERS[kind] + _body(kind, 8 * FRAG, rng)
            name = f"{kind}_{i}{EXT[kind]}"
            (fs / name).write_bytes(data)
            files.append({"path": f"fs/{name}", "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    (case.dir / "evidence" / "ev" / "MANIFEST.json").write_text(json.dumps({"files": files}), encoding="utf-8")
    return case


# ------------------------------------------------------------------------------------------- labelling
def test_type_label_and_text():
    assert fr.type_label(b"SQLite format 3\x00rest") == "sqlite"
    assert fr.type_label(b"hello world\nline two") == "text"
    assert fr.type_label(b"\x00\x01\x02binary") == "other"
    assert fr.is_text("café".encode()[:-1])          # a multi-byte character cut at the end is tolerated


def test_disguise_rule():
    assert fr.is_disguised("jpeg", "a/photo.jpg") == 0
    assert fr.is_disguised("jpeg", "a/photo.docx") == 1
    assert fr.is_disguised("jpeg", "a/photo.tmp") == 0          # tolerated generic extension
    assert fr.is_disguised("text", "a/notes.jpg") == -1          # undefined for non-signature types
    assert fr.is_disguised("sqlite", "db/main.db-wal") == 0


def test_fragment_offsets_skip_header():
    rng = random.Random(0)
    offs = fr.fragment_offsets(10 * FRAG, 4, rng)
    assert len(offs) == 4 and all(o >= FRAG and o % FRAG == 0 and o + FRAG <= 10 * FRAG for o in offs)
    assert fr.fragment_offsets(FRAG + 100, 4, rng) == []
    assert fr.is_blank(bytes(FRAG)) and not fr.is_blank(b"ab" * 256)


# ------------------------------------------------------------------------------------------- dataset + training
@pytest.fixture(scope="module")
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp("jev_cases")
    cases = [_make_case(root, "synthetic_train", 70, 1), _make_case(root, "lone_wolf", 15, 2),     # val split
             _make_case(root, "nist_data_leakage", 15, 3)]                                         # test split
    return fr.build(root / "fragments", cases=cases, progress=False)


def test_build_splits_and_labels(dataset):
    d = fr.load(dataset)
    assert d["x"].shape[1] == FRAG and d["x"].dtype == np.uint8
    assert set(d["split"]) == {"train", "val", "test"}
    assert set(d["label"]) == {"sqlite", "jpeg", "text"}
    assert (d["offset"] >= FRAG).all()                  # the header sector is never a training input
    assert (dataset / "DATASHEET.md").exists()


def test_train_query_and_calibration(dataset, tmp_path):
    res = train(dataset, epochs=3, batch=128, out_dir=tmp_path / "run", device="cpu", progress=False,
                checkpoint=tmp_path / "jev.pt")
    m = res["metrics"]
    assert m["test"]["choice"]["accuracy"] > 0.8         # the synthetic types are easy to separate
    assert m["config"]["t_choice"] > 0 and (tmp_path / "run" / "REPORT.md").exists()
    jev = Jev.load(tmp_path / "jev.pt", "cpu")
    frag = _body("sqlite", FRAG, random.Random(9))
    ans = jev.choice(frag)
    assert ans["query"] == "choice" and abs(sum(ans["probabilities"].values()) - 1) < 1e-3
    sub = jev.choice(frag, ["jpeg", "text"])
    assert set(sub["probabilities"]) == {"jpeg", "text"}
    with pytest.raises(ValueError):
        jev.choice(frag, ["pdf"])
    yn = jev.yes_no(frag, ".jpg")
    assert 0 <= yn["probability"] <= 1 and yn["confidence"] >= 0.5
    sc = jev.score(frag)
    assert 0 <= sc["answer"] <= 1 and len(sc["distribution"]) == 5


# ------------------------------------------------------------------------------------------- metrics
def test_metrics():
    y, p = np.array([0, 0, 1, 1]), np.array([0, 1, 1, 1])
    m = classification_metrics(y, p, ["a", "b"])
    assert m["accuracy"] == 0.75 and m["confusion"] == [[1, 1], [0, 2]]
    assert auroc(np.array([0, 0, 1, 1]), np.array([0.1, 0.2, 0.8, 0.9])) == 1.0
    assert auroc(np.array([1, 1]), np.array([0.1, 0.2])) is None
    assert ece(np.array([1.0, 1.0]), np.array([True, True])) == 0.0
    assert score_targets(torch.tensor([0.0, 0.5, 1.0])).tolist() == [0, 2, 4]
    assert encode_ext(".JPG")[:3] == [ord("j") + 1, ord("p") + 1, ord("g") + 1] and len(encode_ext("")) == 8
    # over-confident logits get a temperature above 1
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 500)
    z = np.stack([np.where(y == 0, 5.0, -5.0), np.where(y == 0, -5.0, 5.0)], 1)
    flip = rng.random(500) < 0.3
    z[flip] = -z[flip]
    assert fit_temperature(z, y) > 1.5


# ------------------------------------------------------------------------------------------- agent contract
def test_agent_answers_are_never_evidence(tmp_path):
    rng = random.Random(4)
    (tmp_path / "wiped.db").write_bytes(bytes(FRAG) + _body("jpeg", 6 * FRAG, rng))   # header wiped, body survives
    (tmp_path / "photo.jpg").write_bytes(HEADERS["jpeg"] + _body("jpeg", 6 * FRAG, rng))  # has a signature: skipped
    torch.manual_seed(0)
    net = JevNet(JevConfig(classes=["jpeg", "sqlite"], abstain_below=0.0))
    with torch.no_grad():                               # force a confident "jpeg / disguised" answer
        net.choice.bias.copy_(torch.tensor([20.0, -20.0]))
        net.yes_no[-1].bias.fill_(20.0)
    bridge = ForensicBridge(tmp_path)
    agent = JevAgent(bridge, jev=Jev(net, "cpu"),
                     manifest={"files": [{"path": "fs/wiped.db", "sha256": "x", "size": 7 * FRAG},
                                         {"path": "fs/photo.jpg", "sha256": "y", "size": 6 * FRAG + 4}]})
    try:
        rep = agent.run(["photo.jpg", "wiped.db"])
    finally:
        agent.close()
        bridge.close()
    cats = sorted(f.category for f in rep.findings)
    assert cats == ["jev.content_type", "jev.disguised"]
    assert all(f.artifact_path == "wiped.db" for f in rep.findings)
    assert all(f.status == FindingStatus.UNCORROBORATED and not f.citations for f in rep.findings)
    assert rep.findings[0].value["choice"]["answer"] == "jpeg"
