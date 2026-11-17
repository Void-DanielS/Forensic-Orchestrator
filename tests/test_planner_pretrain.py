"""Self-supervised planner encoders: contrastive loss, learning signal and the hand-off to PlannerNet (CPU)."""
from __future__ import annotations

import numpy as np
import torch

from src.ml import planner_pretrain as pp
from src.ml.planner import PlannerConfig, PlannerNet
from src.ml.planner_dataset import N_META


def _batch(n=32, seed=0):
    g = torch.Generator().manual_seed(seed)
    return {"path": torch.randint(1, 128, (n, 160), generator=g), "header": torch.randint(0, 256, (n, 256), generator=g),
            "meta": torch.rand(n, N_META, generator=g), "content": torch.randn(n, 768, generator=g),
            "has_content": torch.tensor([i % 4 != 0 for i in range(n)])}


def test_info_nce_perfect_alignment():
    z = torch.nn.functional.normalize(torch.randn(16, 8), dim=-1)
    lo, acc = pp.info_nce(z, z, torch.tensor(50.0))
    assert acc == 1.0 and lo < 0.1


def test_contrastive_loss_learns_and_handoff(tmp_path):
    torch.manual_seed(0)
    net = pp.ContrastiveEncoders(dropout=0.0)
    t = _batch()
    l0, accs = pp.contrastive_loss(net, t)
    assert set(accs) == {f"{a}-{b}" for a, b in pp.PAIRS}
    opt = torch.optim.Adam(net.parameters(), 1e-3)
    for _ in range(30):
        lo, _ = pp.contrastive_loss(net, t)
        opt.zero_grad()
        lo.backward()
        opt.step()
    assert lo < l0                                   # the views of one file become mutually identifiable
    ck = tmp_path / "enc.pt"
    torch.save({"dim": PlannerConfig.dim, "modalities": pp.PRE_MODS,
                "enc": {m: net.enc[m].state_dict() for m in pp.PRE_MODS}}, ck)
    planner = PlannerNet(PlannerConfig())
    assert pp.load_into(planner, ck, freeze=True) == list(pp.PRE_MODS)
    for m in pp.PRE_MODS:
        for k, v in planner.enc[m].state_dict().items():
            assert torch.equal(v, net.enc[m].state_dict()[k])
        assert not any(p.requires_grad for p in planner.enc[m].parameters())
    assert all(p.requires_grad for p in planner.enc["context"].parameters())


def test_weights_balance_cases():
    case = np.array(["a"] * 100 + ["b"] * 4)
    w = pp._weights(case, np.arange(len(case))).numpy()
    assert abs(w[:100].sum() / w[100:].sum() - 5.0) < 1e-6      # sqrt(100) / sqrt(4)


def test_anchored_fusion_starts_as_metadata_model():
    torch.manual_seed(0)
    net = PlannerNet(PlannerConfig(anchor=True, dropout=0.0)).eval()
    t = _batch(8)
    present = torch.ones(8, 5, dtype=torch.bool)
    o = net(t["path"], t["header"], t["meta"], t["content"], torch.randn(8, 768), present)
    assert torch.allclose(o["logit"], o["anchor"])            # gate = 0 at init: pure metadata model
    o2 = net(torch.zeros_like(t["path"]), t["header"], t["meta"], t["content"], torch.randn(8, 768), present)
    assert torch.allclose(o2["anchor"], o["anchor"])           # anchor ignores the other modalities
    net.gate.data.fill_(1.0)
    o3 = net(t["path"], t["header"], t["meta"], t["content"], torch.randn(8, 768), present)
    assert not torch.allclose(o3["logit"], o3["anchor"])


def test_curated_tokens_are_labels_only():
    from src.ml import planner_dataset as pd_

    tok = pd_.curated_tokens()
    assert "mvs_2024" in tok and all(len(t) >= 5 for v in tok.values() for t in v)
