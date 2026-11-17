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
    assert sol.startswith("Question: What file was opened?\nParsing the MFT with MFTECmd")
    assert "Answer: invoice.docm" in sol and "## Questions" not in sol and "   Answer" not in sol


def test_split_on_task_lines():
    ctx, sol = w.split(f"{SCENARIO}\n>Task 1: Which IP brute-forced SSH?\n" + "auth.log shows it. " * 40)
    assert sol.startswith("Question: Which IP brute-forced SSH?\nauth.log")


def test_structure_keeps_actions_drops_noise():
    sol = ("## Walkthrough\nI start by mounting the image with Arsenal Image Mounter and parsing it with KAPE.\n\n"
           "> Q1: Which user?\nOpen NTUSER.DAT in Registry Explorer and go to the RecentDocs key.\n\n   Answer\n  bob\n\n"
           "> Q2: First flag?\n\n   Answer\n  FLAG{x}\n\n"
           "> Q3: Hash?\n```\n" + "\n".join(f"line {i}" for i in range(30)) + "\n```\nI ran md5sum on it.\n"
           "**Answer:** abc\n\n## Summary\nThanks for reading, follow me!\n")
    out = w.structure(w.clean(sol))
    assert out.startswith("Approach:\nI start by mounting the image")
    assert "Question: Which user?\nOpen NTUSER.DAT in Registry Explorer" in out and "Answer: bob" in out
    assert "First flag" not in out                       # answered without any step
    assert "line 7" in out and "line 8" not in out       # long tool output cut after 8 lines
    assert "Answer: abc" in out and "Thanks for reading" not in out


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
