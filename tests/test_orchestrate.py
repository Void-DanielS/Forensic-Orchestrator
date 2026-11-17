"""Orchestrated agents without a GPU: the LLM is replaced by a scripted fake, so the tests check the safeguards
(verbatim quotes verified, invented quotes rejected, only verified text becomes a lead) and the re-ranking."""
from __future__ import annotations

import json

import numpy as np

from src.agents.analyst_agent import AnalystAgent, locate, text_view
from src.bridge.forensic_bridge import ForensicBridge
from src.pipeline.orchestrate import _GENERIC, rerank
from src.verification.models import FindingStatus


class FakeLLM:
    model_id = "fake-llm"

    def __init__(self, answer: dict):
        self.answer = json.dumps(answer)
        self.prompts: list = []

    def chat(self, prompts, **_):
        self.prompts += prompts
        return [self.answer] * len(prompts)


DATA = (b"\x00\x00From: Home Technologies <HomeTechFLA@yahoo.com>\x00Subject: Ideas for drops?\x00"
        + "Mr. Evil wrote the plan".encode("utf-16-le") + b"\x00" * 8)


def _run(tmp_path, answer, leads=()):
    fs = tmp_path / "fs"
    fs.mkdir(exist_ok=True)
    (fs / "mail.dbx").write_bytes(DATA)
    bridge = ForensicBridge(fs)
    llm = FakeLLM(answer)
    ag = AnalystAgent(bridge, brief="Who planned the drops?", leads=leads, llm=llm)
    try:
        rep = ag.run(["mail.dbx"])
        return rep, ag, llm
    finally:
        ag.close()
        bridge.close()


def test_locate_ascii_utf16_and_case():
    assert locate(DATA, "Ideas for drops?")[2] == "utf-8"
    off, raw, enc = locate(DATA, "mr. evil wrote")
    assert enc == "utf-16-le" and raw.decode(enc) == "Mr. Evil wrote"         # the cited bytes are the file's own
    assert locate(DATA, "Ideas for robbery") is None


def test_text_view_puts_lead_context_first():
    v = text_view(b"A" * 5000 + b" secret lead here " + b"B" * 100, leads=["secret lead"], chars=600)
    assert v.startswith("...") and "secret lead" in v[:600]


def test_quotes_verified_and_invented_quotes_rejected(tmp_path):
    answer = {"quotes": [{"quote": "Ideas for drops?", "type": "message", "why": "planning"},
                         {"quote": "mr. evil wrote the plan", "type": "person", "why": "author"},
                         {"quote": "Transfer $5000 to account 991", "type": "other", "why": "invented"}],
              "leads": ["Mr. Evil", "Account 991", "Home Technologies"]}
    rep, ag, llm = _run(tmp_path, answer)
    st = {f.category: f.status for f in rep.findings}
    quotes = [f for f in rep.findings if not f.category.startswith("analyst.tool")]
    assert [f.status for f in quotes] == [FindingStatus.VERIFIED, FindingStatus.VERIFIED, FindingStatus.REJECTED]
    assert st["analyst.tool.email"] == FindingStatus.VERIFIED                   # the extraction tool ran too
    assert "Transfer" not in quotes[2].claim                                     # the invented text is not a claim
    assert "Mr. Evil wrote the plan" in quotes[1].claim                          # the file's bytes, not the model's
    leads = ag.verified_leads(rep.findings)
    assert "Mr. Evil" in leads and "HomeTechFLA@yahoo.com" in leads
    assert "Account 991" not in leads and "Home Technologies" not in leads      # not inside a verified quote


def test_leads_reach_the_prompt(tmp_path):
    _, _, llm = _run(tmp_path, {"quotes": [], "leads": []}, leads=["HomeTechFLA@yahoo.com"])
    assert "HomeTechFLA@yahoo.com" in llm.prompts[0][1]


def test_rerank_moves_files_with_verified_leads_up():
    rows = [{"snippet": "nothing here"}, {"snippet": "mail from hometechfla@yahoo.com"}, {"snippet": "x"}]
    p = np.array([0.30, 0.20, 0.25])
    assert int(np.argmax(rerank(p, rows, [], set()))) == 0
    s = rerank(p, rows, ["HomeTechFLA@yahoo.com"], {0})
    assert int(np.argmax(s)) == 1 and s[0] == -np.inf                            # done files never return
    assert _GENERIC.search("msoe@microsoft.com") and not _GENERIC.search("HomeTechFLA@yahoo.com")


def test_sources_split_llm_quotes_from_the_tool(tmp_path):
    from types import SimpleNamespace

    from src.pipeline import orchestrate as orch

    ans = {"quotes": [{"quote": "Mr. Evil wrote the plan", "type": "person", "why": "author"}], "leads": []}
    rep, _, _ = _run(tmp_path, ans)
    src = {orch.source(f) for f in rep.findings if f.status == FindingStatus.VERIFIED}
    assert src == {"llm", "tool"}                       # the e-mail comes from the tool, the quote from the LLM
    assert orch.source(SimpleNamespace(agent="ImageTriageAgent", category="mm.triage")) == "image"


def test_no_tool_mode_keeps_only_llm_quotes(tmp_path):
    fs = tmp_path / "fs"
    fs.mkdir()
    (fs / "mail.dbx").write_bytes(DATA)
    bridge = ForensicBridge(fs)
    ans = {"quotes": [{"quote": "Mr. Evil wrote the plan", "type": "person", "why": "author"}], "leads": []}
    ag = AnalystAgent(bridge, brief="b", llm=FakeLLM(ans), tools=False)
    try:
        rep = ag.run(["mail.dbx"])
    finally:
        ag.close()
        bridge.close()
    assert [f.category for f in rep.findings] == ["analyst.person"]


def test_split_evaluation_counts_per_source(monkeypatch):
    from types import SimpleNamespace

    from src.eval import scoring
    from src.pipeline import orchestrate as orch

    ents = [{"norm": "hometechfla yahoo com", "type": "email", "value": "HomeTechFLA@yahoo.com"},
            {"norm": "mr evil", "type": "person", "value": "Mr. Evil"}]
    monkeypatch.setattr(scoring, "key_entities", lambda *a, **k: ents)
    ver = [{"claim": "email HomeTechFLA@yahoo.com in mail", "value": {}, "_source": "tool"},
           {"claim": "“Mr. Evil wrote the plan” in mail", "value": {}, "_source": "llm"}]
    r = orch.split_evaluation(SimpleNamespace(dir=None, summary=""), ver)
    assert (r["llm"], r["tool"], r["image"], r["union"], r["llm_only"]) == (1, 1, 0, 2, 1)
