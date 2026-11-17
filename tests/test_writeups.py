"""Write-up corpus for the tuned prior: context/solution split, cleaning and the leak filter (no dataset case)."""
import json

from src.ml import writeups as w

SCENARIO = ("**Scenario:**\nA workstation of a small company was compromised after an employee opened a phishing "
            "attachment. The SOC collected a KAPE triage image of the host. Investigate what happened. " * 2)
SOLUTION = ("## Questions\n> Q1: What file was opened?\n\nParsing the MFT with MFTECmd and the Prefetch folder with "
            "PECmd shows the document in the Downloads folder.\n\n   Answer\n  invoice.docm\n" * 4)


def test_split_and_clean():
    text = w.clean(f"# Lab\n![x](a.png)\n{SCENARIO}\n{SOLUTION}")
    ctx, sol = w.split(text)
    assert "phishing" in ctx and "png" not in ctx
    assert sol.startswith("## Questions") and "MFTECmd" in sol


def test_split_on_task_lines():
    ctx, sol = w.split(f"{SCENARIO}\n>Task 1: Which IP brute-forced SSH?\n" + "auth.log shows it. " * 40)
    assert sol.startswith(">Task 1")


def test_leak_filter(tmp_path, monkeypatch):
    monkeypatch.setattr(w, "key_identifiers", lambda: ["jimcloudy@outlook.com"])
    repo = tmp_path / "repo"
    repo.mkdir()
    (repo / "ok.md").write_text(f"# Fine\n{SCENARIO}\n{SOLUTION}", encoding="utf-8")
    (repo / "case.md").write_text(f"# CFReDS Data Leakage Case\n{SCENARIO}\n{SOLUTION}", encoding="utf-8")
    (repo / "ident.md").write_text(f"# Other\n{SCENARIO}\nmail to JimCloudy@outlook.com\n{SOLUTION}", encoding="utf-8")
    (repo / "README.md").write_text(f"# Index\n{SCENARIO}\n{SOLUTION}", encoding="utf-8")
    p = w.build(tmp_path, tmp_path / "out")
    pairs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    assert [x["source"] for x in pairs] == ["repo/ok.md"]
    m = json.loads((p.parent / "manifest.json").read_text(encoding="utf-8"))
    assert m["dropped"]["names_case"] == 1 and m["dropped"]["has_identifier"] == 1


def test_every_dataset_case_has_aliases():
    assert len(w.CASE_ALIASES) == 16
    for case, pats in w.CASE_ALIASES.items():
        assert pats, case
