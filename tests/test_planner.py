"""Investigation planner + reader: labels from an answer key, features, the fusion network, ranking metrics and the
reader's verified citations. Synthetic case on CPU; no answer key or model download needed."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.ml import planner_dataset as pd_
from src.ml.planner import MODALITIES, PlannerConfig, PlannerNet, encode_path, features
from src.ml.planner_train import average_precision, case_metrics, coverage_at


def test_answer_key_labels():
    key = pd_._norm_key("The suspect copied C:\\Users\\informant\\Desktop\\secret_plan.docx to the USB.")
    assert pd_.name_in_key("fs/Users/informant/Desktop/secret_plan.docx", key)
    assert not pd_.name_in_key("fs/Users/informant/Desktop/plan.docx", key)          # whole-name match only
    assert not pd_.name_in_key("fs/Users/x/desktop.ini", pd_._norm_key("desktop.ini everywhere"))   # generic
    pats = pd_.entity_patterns([{"type": "url", "value": "https://www.Evil.org/drop/"},
                                {"type": "email", "value": "spy@corp.com"}, {"type": "gps", "value": "1,2"}])
    assert [p[1] for p in pats] == [b"evil.org/drop", b"spy@corp.com"]
    m = pd_.Matcher(pats)
    data = b"\x00junk" + "EVIL.ORG/DROP/x".encode("utf-16-le") + b" mail spy@corp.com"
    assert m.hits(data.lower()) == {"https://www.Evil.org/drop/", "spy@corp.com"}
    assert not pd_.Matcher([])


class FakeCase:
    def __init__(self, root: Path):
        self.case_id, self.dir, self.summary = "fake", root / "fake", "A data leakage case."

    def evidence_ids(self):
        return ["ev"]


def _case(tmp_path) -> FakeCase:
    case = FakeCase(tmp_path)
    ev = case.dir / "evidence" / "ev"
    files = {"fs/Users/a/mail.pst": b"!BDN" + "contact spy@corp.com".encode("utf-16-le"),
             "fs/Windows/system32/x.dll": b"MZ" + b"\x00" * 600,
             "fs/Users/a/secret_plan.docx": b"PK\x03\x04 plan"}
    recs = []
    for rel, data in files.items():
        p = ev / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(data)
        recs.append({"path": rel, "size": len(data), "sha256": hashlib.sha256(data).hexdigest(),
                     "category": "email" if rel.endswith(".pst") else "user_document",
                     "macb": {"m": "2015-03-22T14:34:59Z"}})
    (ev / "MANIFEST.json").write_text(json.dumps({"platform": "windows", "files": recs}), encoding="utf-8")
    return case


def test_case_rows_labels_and_inputs(tmp_path):
    case = _case(tmp_path)
    key = pd_._norm_key("Answer: secret_plan.docx was sent to spy@corp.com")
    rows, head, meta, unreadable = pd_.case_rows(case, pd_.Matcher(pd_.entity_patterns(
        [{"type": "email", "value": "spy@corp.com"}])), key)
    by = {r["path"]: r for r in rows}
    assert unreadable == 0 and head.shape == (3, pd_.HEAD) and meta.shape == (3, pd_.N_META)
    assert by["fs/Users/a/mail.pst"]["entities"] == ["spy@corp.com"] and by["fs/Users/a/mail.pst"]["relevant"]
    assert by["fs/Users/a/secret_plan.docx"]["by_name"] == 1
    assert by["fs/Windows/system32/x.dll"]["relevant"] == 0
    assert "spy@corp.com" in by["fs/Users/a/mail.pst"]["snippet"]            # UTF-16 strings are read
    # inference mode: no answer key -> no labels, identical inputs
    rows2, head2, meta2, _ = pd_.case_rows(case)
    assert all(r["relevant"] == 0 for r in rows2) and np.array_equal(head, head2) and np.array_equal(meta, meta2)


def test_planner_net_masks_and_learns():
    torch.manual_seed(0)
    n = 64
    rng = np.random.default_rng(0)
    rows = [{"path_text": f"Users/a/{'mail.pst' if i % 2 else 'x.dll'}", "snippet": "x" if i % 3 else ""}
            for i in range(n)]
    x = features(rows, rng.integers(0, 255, (n, 256)).astype(np.uint8), rng.random((n, pd_.N_META), dtype=np.float32),
                 rng.normal(size=(n, 768)).astype(np.float32), np.zeros((n, 768), np.float32))
    assert x["path"].shape == (n, pd_.PATH_CHARS) and x["present"][:, 3].tolist() == [bool(i % 3) for i in range(n)]
    y = torch.tensor([float(i % 2) for i in range(n)])
    net = PlannerNet(PlannerConfig(modalities=("path",), dim=32, heads=2, layers=1))
    t = {k: torch.from_numpy(v) for k, v in x.items()}
    opt = torch.optim.Adam(net.parameters(), lr=3e-3)
    for _ in range(60):
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            net(t["path"], t["header"], t["meta"], t["content"], t["context"], t["present"])["logit"], y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    net.eval()
    out = net(t["path"], t["header"], t["meta"], t["content"], t["context"], t["present"], attn=True)
    assert ((out["logit"] > 0).float() == y).float().mean() == 1.0      # the file name decides in this toy task
    assert torch.allclose(out["attention"].sum(1), torch.ones(n), atol=1e-4)
    assert out["attention"][:, 1:].abs().max() < 1e-6                  # unused modalities get no attention


def test_encode_path_keeps_the_file_name():
    a = encode_path("x" * 500 + "/secret.docx")
    assert len(a) == pd_.PATH_CHARS and bytes((a[-11:] - 1).astype(np.uint8)) == b"secret.docx"


def test_ranking_metrics():
    y = np.array([1, 0, 1, 0, 0])
    s = np.array([0.9, 0.8, 0.7, 0.1, 0.0])
    assert average_precision(y, s) == pytest.approx((1 + 2 / 3) / 2)
    items = [{"a"}, set(), {"b", "a"}, set(), {"c"}]
    order = np.argsort(-s)
    assert coverage_at(order, items, 1) == pytest.approx(1 / 3) and coverage_at(order, items, 3) == pytest.approx(2 / 3)
    m = case_metrics(y.astype(np.float32), s, items)
    assert m["answer_items"] == 3 and m["coverage@100"] == 1.0


def test_reader_identifiers_are_verified(tmp_path):
    from src.agents.reader_agent import ReaderAgent, identifiers
    from src.bridge.forensic_bridge import ForensicBridge
    from src.verification.models import FindingStatus

    data = b"\x00\x01 visit https://www.evil.org/drop?id=7, then mail spy@corp.com.\x00" + \
        "secondary: boss@corp.com".encode("utf-16-le")
    got = {(k, v, enc) for k, v, _, _, enc in identifiers(data)}
    assert got == {("url", "https://www.evil.org/drop?id=7", "ascii"), ("email", "spy@corp.com", "ascii"),
                   ("email", "boss@corp.com", "utf-16le")}
    fs = tmp_path / "fs"
    fs.mkdir()
    (fs / "blob.bin").write_bytes(data)
    bridge = ForensicBridge(fs)
    agent = ReaderAgent(bridge)
    try:
        rep = agent.run(["blob.bin"])
    finally:
        agent.close()
        bridge.close()
    assert len(rep.findings) == 4 and all(f.status == FindingStatus.VERIFIED for f in rep.findings)
    assert {f.category for f in rep.findings} == {"reader.file", "reader.email", "reader.url"}
    # a tampered file makes the same statements fail verification
    fs2 = tmp_path / "fs2"
    fs2.mkdir()
    (fs2 / "blob.bin").write_bytes(data)
    bridge = ForensicBridge(fs2)
    agent = ReaderAgent(bridge)
    try:
        findings = agent.analyze(["blob.bin"])
        (fs2 / "blob.bin").write_bytes(data.replace(b"spy", b"spx"))
        checked = agent.verifier.verify_all(findings)
    finally:
        agent.close()
        bridge.close()
    assert all(f.status == FindingStatus.REJECTED for f in checked)


assert MODALITIES == ("path", "header", "metadata", "content", "context")


def test_augment_preserves_name_signature_and_labels():
    from src.ml.planner_train import MAGIC, META_CONT, NAME_CHARS, augment
    torch.manual_seed(0)
    b, n, h = 64, 160, 256
    t = {"path": torch.randint(1, 128, (b, n)), "header": torch.randint(0, 256, (b, h)),
         "meta": torch.rand(b, len(pd_.META_FEATURES)), "content": torch.randn(b, 768),
         "context": torch.zeros(b, 768), "present": torch.ones(b, 5, dtype=torch.bool)}
    y = (torch.arange(b) % 4 == 0).float()
    a = augment(t, y, synth=1.0)
    assert {k: v.shape for k, v in a.items()} == {k: v.shape for k, v in t.items()}
    assert torch.equal(a["path"][:, -NAME_CHARS:], t["path"][:, -NAME_CHARS:])        # file name untouched
    assert torch.equal(a["header"][:, :MAGIC], t["header"][:, :MAGIC])                 # signature untouched
    other = [i for i in range(t["meta"].shape[1]) if i not in META_CONT]
    assert torch.equal(a["meta"][:, other], t["meta"][:, other])                       # flags / categories untouched
    assert not torch.equal(a["meta"], t["meta"]) and torch.equal(t["path"], t["path"].clone())
    assert a["path"].min() >= 0 and a["header"].max() < 256


def test_wide_and_deep_starts_as_its_linear_part():
    torch.manual_seed(0)
    net = PlannerNet(PlannerConfig(modalities=("path", "header", "metadata", "content"), wide=True))
    b = 8
    meta = torch.rand(b, net.cfg.n_meta)
    o = net(torch.randint(0, 129, (b, 160)), torch.randint(0, 256, (b, 256)), meta, torch.randn(b, 768),
            torch.randn(b, 768), torch.ones(b, 5, dtype=torch.bool))
    assert torch.allclose(o["logit"], net.wide(meta).squeeze(-1))          # gate = 0: exactly the linear model
    with torch.no_grad():
        net.gate.fill_(1.0)
    o2 = net(torch.randint(0, 129, (b, 160)), torch.randint(0, 256, (b, 256)), meta, torch.randn(b, 768),
             torch.randn(b, 768), torch.ones(b, 5, dtype=torch.bool))
    assert not torch.allclose(o2["logit"], o2["anchor"])                    # the deep residual now contributes
