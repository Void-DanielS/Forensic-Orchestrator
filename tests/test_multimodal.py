"""Jev-MM multimodal model: labels, metadata features, dataset build, fusion network, a short training run, the
checkpoint round trip and the agent contract (answers are UNCORROBORATED). Synthetic images, CPU, no downloads:
the ViT is built without pretrained weights and OCR / text encoding are replaced by fixed fixtures."""
from __future__ import annotations

import hashlib
import io
import json
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from src.ml import mm_dataset as mmd
from src.ml.multimodal import MODALITIES, JevMM, JevMMNet, MMConfig

pytest.importorskip("timm")


# ------------------------------------------------------------------------------------------------ labels
@pytest.mark.parametrize("path,cat,expected", [
    ("fs/data/media/0/DCIM/Camera/IMG_20200209_132353.jpg", "user_image", "camera"),
    ("fs/private/var/mobile/Media/DCIM/100APPLE/IMG_0013.HEIC", "user_image", "camera"),
    ("fs/data/media/0/Pictures/Screenshots/Screenshot_20200213-213310.png", "screenshot", "screenshot"),
    ("fs/data/media/0/DCIM/.thumbnails/123.jpg", "user_image", "thumbnail"),
    ("fs/Users/x/AppData/Local/Microsoft/Windows/Explorer/thumbcache_256.db", "thumbnail_cache", "thumbnail"),
    ("fs/Documents and Settings/Jean/Local Settings/Temporary Internet Files/Content.IE5/AB/x[1].jpg",
     "app_image", "web_cache"),
    ("fs/data/media/0/WhatsApp/Media/WhatsApp Images/IMG-20200101-WA0001.jpg", "user_image", "messaging"),
    ("fs/data/data/com.whatsapp/files/emoji/e1.png", "app_image", "app_asset"),
    ("fs/data/data/com.example/files/banner.png", "app_image", "app_asset"),
    ("fs/home/user/Downloads/photo.jpg", "user_image", "user_file"),
])
def test_origin_label_rules(path, cat, expected):
    assert mmd.origin_label(path, cat) == expected
    assert (expected in mmd.USER_GENERATED) == (expected in {"camera", "screenshot", "messaging", "user_file"})


# ------------------------------------------------------------------------------------------------ features
def _jpeg(w=640, h=480, exif=True, quality=90, seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    img = Image.fromarray(rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    buf = io.BytesIO()
    kw = {"quality": quality}
    if exif:
        ex = Image.Exif()
        ex[0x010F] = "Google"
        ex[0x0110] = "Pixel 3"
        kw["exif"] = ex.tobytes()
    img.save(buf, "JPEG", **kw)
    return buf.getvalue()


def _png(w=1080, h=2160, seed=0) -> bytes:
    rng = np.random.default_rng(seed)
    a = np.full((h // 8, w // 8, 3), 240, np.uint8)
    k = max(1, h // 80)
    a[:k] = rng.integers(0, 60, (k, max(1, w // 8), 3))
    buf = io.BytesIO()
    Image.fromarray(a).resize((w, h)).save(buf, "PNG")
    return buf.getvalue()


def test_metadata_vector_reads_container_and_exif_not_the_name():
    data = _jpeg(exif=True, quality=75)
    with Image.open(io.BytesIO(data)) as img:
        v = mmd.metadata_vector(img, len(data))
    f = dict(zip(mmd.META_FEATURES, v))
    assert len(v) == mmd.N_META
    assert f["fmt_jpeg"] == 1 and f["exif_present"] == 1 and f["exif_make"] == 1 and f["exif_model"] == 1
    assert f["exif_gps"] == 0
    assert 0.65 <= f["jpeg_quality"] <= 0.85                 # libjpeg quality 75 recovered from the table
    data = _png()
    with Image.open(io.BytesIO(data)) as img:
        f = dict(zip(mmd.META_FEATURES, mmd.metadata_vector(img, len(data))))
    assert f["fmt_png"] == 1 and f["exif_present"] == 0 and f["screen_width"] == 1 and f["portrait"] == 1


def test_letterbox_keeps_whole_image():
    img = Image.new("RGB", (100, 400), (255, 0, 0))
    t = mmd.letterbox(img)
    assert t.shape == (224, 224, 3) and t.dtype == np.uint8
    assert tuple(t[112, 112]) == (255, 0, 0) and tuple(t[112, 2]) == (128, 128, 128)   # padded, not cropped


def test_text_stats_detects_status_bar_clock():
    boxes = [([[0, 0], [50, 0], [50, 10], [0, 10]], "12:45", 0.9), ([[0, 20], [90, 20], [90, 40], [0, 40]], "Inbox", 0.8)]
    s = dict(zip(mmd.TEXT_STATS, mmd.text_stats(boxes, 100, 100)))
    assert s["has_text"] == 1 and s["clock_pattern"] == 1 and 0 < s["text_area"] <= 1
    assert mmd.text_stats([], 10, 10)[0] == 0


# ------------------------------------------------------------------------------------------------ dataset
class FakeCase:
    def __init__(self, root: Path, case_id: str):
        self.case_id, self.dir = case_id, root / case_id

    def evidence_ids(self):
        return ["ev"]


LAYOUT = {  # origin -> (relative dir, category, maker)
    "camera": ("data/media/0/DCIM/Camera", "user_image", lambda s: _jpeg(800, 600, True, 92, s)),
    "screenshot": ("data/media/0/Pictures/Screenshots", "screenshot", lambda s: _png(540, 1080, s)),
    "app_asset": ("data/data/com.app/files", "app_image", lambda s: _png(64 + s % 7, 64, s)),
}


def _make_case(root: Path, case_id: str, n: int, seed: int) -> FakeCase:
    case = FakeCase(root, case_id)
    ev = case.dir / "evidence" / "ev"
    files = []
    for origin, (d, cat, make) in LAYOUT.items():
        for i in range(n):
            data = make(seed * 1000 + i)
            ext = ".jpg" if data[:2] == b"\xff\xd8" else ".png"
            rel = f"fs/{d}/{origin}_{case_id}_{i}{ext}"
            p = ev / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_bytes(data)
            files.append({"path": rel, "category": cat, "size": len(data), "sha256": hashlib.sha256(data).hexdigest()})
    (ev / "MANIFEST.json").write_text(json.dumps({"platform": "android", "files": files}), encoding="utf-8")
    return case


@pytest.fixture(scope="module")
def tiny_ds(tmp_path_factory):
    from src.datasets import export

    root = tmp_path_factory.mktemp("cases")
    cases = [_make_case(root, "mm_tr_a", 6, 1), _make_case(root, "mm_tr_b", 6, 2),
             _make_case(root, "mm_va", 3, 3), _make_case(root, "mm_te", 3, 4)]
    old = dict(export.SPLITS)
    export.SPLITS.update({"mm_va": "val", "mm_te": "test"})
    try:
        out = mmd.build(tmp_path_factory.mktemp("ds") / "20990101", cases=cases, workers=1, progress=False,
                        cache_dir=tmp_path_factory.mktemp("cache"))
    finally:
        export.SPLITS.clear()
        export.SPLITS.update(old)
    rows = mmd.read_rows(out)
    rng = np.random.default_rng(0)
    emb = rng.normal(size=(len(rows), 768)).astype(np.float32) * 0.01
    stats = np.zeros((len(rows), len(mmd.TEXT_STATS)), np.float32)
    for i, r in enumerate(rows):                     # screenshots "carry text" in this fixture
        if r["origin"] == "screenshot":
            stats[i, 0], emb[i, 0] = 1.0, 1.0
    np.savez_compressed(out / "text.npz", emb=emb.astype(np.float16), stats=stats)
    return out


def test_build_is_case_split_deduplicated_and_path_free(tiny_ds):
    d = mmd.load(tiny_ds)
    rows = d["rows"]
    assert len(rows) == 3 * (6 + 6 + 3 + 3)
    assert {r["split"] for r in rows} == {"train", "val", "test"}
    assert d["thumbs"].shape == (len(rows), 224, 224, 3) and d["meta"].shape == (len(rows), mmd.N_META)
    assert len({r["sha256"] for r in rows}) == len(rows)
    assert not any("path" in f or "name" in f for f in mmd.META_FEATURES)
    assert (tiny_ds / "DATASHEET.md").exists()


def test_thumbnail_cache_tamper_is_detected(tiny_ds, tmp_path):
    info = json.loads((tiny_ds / "build.json").read_text(encoding="utf-8"))
    t = Path(info["thumbs_path"])
    orig = t.read_bytes()
    try:
        t.write_bytes(orig[:-1] + bytes([orig[-1] ^ 1]))
        with pytest.raises(ValueError):
            mmd.load(tiny_ds)
    finally:
        t.write_bytes(orig)


# ------------------------------------------------------------------------------------------------ model
def _net(mods=MODALITIES) -> JevMMNet:
    return JevMMNet(MMConfig(classes=list(mmd.ORIGINS), modalities=tuple(mods), dim=64, heads=2, layers=1),
                    pretrained=False)


def test_fusion_handles_missing_modalities_and_returns_attention():
    torch.manual_seed(0)
    net = _net().eval()
    b = 4
    th = torch.randint(0, 255, (b, 224, 224, 3), dtype=torch.uint8)
    me, tx = torch.randn(b, mmd.N_META), torch.randn(b, 768 + len(mmd.TEXT_STATS))
    present = torch.tensor([[1, 1, 1], [1, 1, 0], [1, 0, 0], [0, 0, 0]], dtype=torch.bool)
    out = net(th, me, tx, present, attn=True)
    assert out["choice"].shape == (b, len(mmd.ORIGINS)) and out["yes_no"].shape == (b,)
    a = out["attention"]
    assert torch.allclose(a.sum(1), torch.ones(b), atol=1e-4)
    assert a[1, 2] < 1e-6 and a[2, 1] < 1e-6                    # masked modalities get no attention
    # an absent modality's input must not change the answer
    tx2 = tx.clone()
    tx2[1] += 100
    assert torch.allclose(net(th, me, tx2, present)["choice"][1], out["choice"][1], atol=1e-4)


def test_only_unfrozen_vit_blocks_train():
    net = _net()
    trainable = {n.split(".")[2] for n, p in net.named_parameters() if n.startswith("vit.blocks") and p.requires_grad}
    assert trainable == {str(len(net.vit.blocks) - 1)}
    assert not net.vit.patch_embed.proj.weight.requires_grad


def test_short_training_learns_and_checkpoint_round_trips(tiny_ds, tmp_path, monkeypatch):
    from src.ml import multimodal, mm_train

    torch.manual_seed(0)
    data = mm_train.MMData(tiny_ds)
    net, cfg, val, hist = mm_train.train_one(data, ("metadata", "text"), epochs=25, batch=16, lr=3e-3,
                                             device="cpu", progress=False, label="t", pretrained=False)
    assert len(hist) == 25 and cfg.t_choice > 0
    res = mm_train.test_metrics(net, cfg, data, "cpu", robustness=True)
    # metadata separates camera JPEGs / tall screenshots / tiny app PNGs perfectly in this fixture
    assert res["test"]["choice"]["accuracy"] >= 0.9
    assert res["test"]["yes_no"]["auroc"] >= 0.9
    assert set(res["robustness"]) == set(mm_train.WITHHELD)
    ck = JevMM.save(net, tmp_path / "jev_mm.pt")
    monkeypatch.setattr(multimodal, "build_vit", lambda pretrained=True: (_ for _ in ()).throw(AssertionError))
    jev = JevMM.load(ck, "cpu")
    d = mmd.load(tiny_ds)
    te = data.idx("test")
    p = jev.probabilities(np.asarray(d["thumbs"])[te], data.meta[te], data.text[te], data.present[te], attn=True)
    assert p["choice"].shape == (len(te), len(mmd.ORIGINS)) and np.allclose(p["choice"].sum(1), 1, atol=1e-4)
    ans = jev.choice_answer(p["choice"][0])
    assert ans["query"] == "choice" and ans["best"] in mmd.ORIGINS
    yn = jev.yes_no_answer(float(p["yes_no"][0]))
    assert yn["query"] == "yes_no" and 0 <= yn["probability"] <= 1


def test_seed_ensemble_round_trips_and_averages_logits(tmp_path):
    from src.ml.multimodal import JevMMEnsemble

    torch.manual_seed(0)
    mods = ("metadata", "text")
    a, b = _net(mods).eval(), _net(mods).eval()
    cfg = a.cfg
    me, tx = torch.randn(3, mmd.N_META), torch.randn(3, 768 + len(mmd.TEXT_STATS))
    pr = torch.tensor([[1, 1, 1]] * 3, dtype=torch.bool)
    want = (a(None, me, tx, pr)["choice"] + b(None, me, tx, pr)["choice"]) / 2
    assert torch.allclose(JevMMEnsemble([a, b], cfg)(None, me, tx, pr)["choice"], want, atol=1e-5)
    jev = JevMM.load(JevMM.save_ensemble([a, b], cfg, tmp_path / "ens.pt"), "cpu")
    assert isinstance(jev.net, JevMMEnsemble) and len(jev.net.members) == 2
    with torch.no_grad():
        assert torch.allclose(jev.net(None, me, tx, pr)["choice"], want, atol=1e-5)


def test_checkpoint_excludes_frozen_backbone(tmp_path):
    net = _net()
    ck = torch.load(JevMM.save(net, tmp_path / "x.pt"), weights_only=False)
    assert not any(k in ck["state_dict"] for k in net.frozen_prefix())
    assert any(k.startswith("vit.blocks.11.") for k in ck["state_dict"])


def test_agent_findings_are_uncorroborated_leads(tiny_ds, tmp_path):
    from src.agents.mm_agent import ImageTriageAgent
    from src.bridge.forensic_bridge import ForensicBridge
    from src.verification.models import FindingStatus

    class StubJev:
        cfg = MMConfig(abstain_below=0.5)

        def ask(self, paths, ocr=True):
            return [{"file": str(p), "choice": JevMM.choice_answer(self, np.eye(7)[0] * 0.9 + 0.1 / 7),
                     "yes_no": JevMM.yes_no_answer(self, 0.95), "modality_attention": {"vision": 0.5},
                     "modalities_present": {"vision": True}, "ocr_text": ""} for p in paths]

    fs = tmp_path / "fs"
    (fs / "DCIM").mkdir(parents=True)
    (fs / "DCIM" / "a.jpg").write_bytes(_jpeg())
    (fs / "notes.txt").write_text("x")
    bridge = ForensicBridge(fs)
    agent = ImageTriageAgent(bridge, jev=StubJev(), manifest={"files": [
        {"path": "fs/DCIM/a.jpg", "sha256": "0" * 64, "size": 1, "category": "user_image"},
        {"path": "fs/notes.txt", "sha256": "1" * 64, "size": 1, "category": "user_document"}]})
    try:
        res = agent.run(agent.image_files())
    finally:
        agent.close()
        bridge.close()
    fs_ = res.findings if hasattr(res, "findings") else res["findings"]
    assert fs_ and all(f.status == FindingStatus.UNCORROBORATED and not f.citations for f in fs_)
    assert {f.category for f in fs_} >= {"mm.origin"}
