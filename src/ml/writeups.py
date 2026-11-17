"""Corpus of public DFIR write-ups for fine-tuning the LLM prior (pre-registration, deviation of 2026-11-17).

    python -m src.ml.writeups                # data/external/writeups/<repo>/**.md -> data/datasets/writeups/

Each write-up becomes one plain-text pair: *context* (the scenario, the text before the first question / solution
section) -> *solution* (the rest). Nothing else is extracted. A write-up is dropped when it names one of the 16
dataset cases or contains any of their answer-key identifiers, so nothing about the evaluation cases is learned.
The cloned repositories are untrusted data: only their Markdown text is read, nothing in them is executed.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path

from src.config import settings

ROOT = settings.DATA_DIR / "external" / "writeups"
OUT = settings.DATA_DIR / "datasets" / "writeups"
MIN_CONTEXT, MIN_SOLUTION = 150, 400
MAX_CONTEXT, MAX_SOLUTION = 2_000, 12_000

# the 16 cases of the planner dataset, by every name a write-up would use
CASE_ALIASES = {
    "m57_patents": [r"m57", r"patents? (case|scenario)", r"pat mcgoo"],
    "dfrws2009": [r"dfrws.{0,3}2009", r"nssal", r"mardi gras"],
    "mvs_2024": [r"(magnet )?virtual summit.{0,10}2024", r"mvs.{0,3}2024", r"ethan allen park"],
    "dfrws2018": [r"dfrws.{0,3}2018", r"jessie pinkman"],
    "owl": [r"project owl", r"owl scenario"],
    "vto_parrot_bluegrass": [r"vto labs?", r"parrot bluegrass"],
    "nist_data_leakage": [r"data leakage case", r"iaman informant", r"cfreds.{0,40}leakage", r"mr\.? informant"],
    "lone_wolf": [r"lone.?wolf"],
    "ngdc_2012": [r"national gallery", r"ngdc"],
    "vto_qysea_fifish_p3": [r"qysea", r"fifish"],
    "nist_hacking_case": [r"hacking case", r"greg schardt", r"mr\.? evil"],
    "dfrws2017": [r"dfrws.{0,3}2017", r"iot forensic challenge"],
    "dstl_ground_truth": [r"dstl", r"ground truth datasets? for data recovery"],
    "nist_rhino_hunt": [r"rhino ?hunt", r"rhinoceros"],
    "dfrws2005": [r"dfrws.{0,3}2005", r"goatboy"],
    "dfrws2016": [r"dfrws.{0,3}2016"],
}
ALIASES = re.compile("|".join(f"(?:{p})" for ps in CASE_ALIASES.values() for p in ps), re.I)
SKIP_NAMES = {"readme.md", "claude.md", "license.md", "contributing.md", "summary.md", "index.md", "agents.md"}
# where the solution starts: a heading or line that opens the questions / walkthrough
SPLIT = re.compile(r"^(?:#{1,4}\s*(?:questions?|walk-?through|solutions?|answers?|analysis|investigation|tasks?"
                   r"|flags?|start investigation|q\s*\d|question\s*\d|task\s*\d|challenge\s*\d)\b.*"
                   r"|>?\s*\**\s*(?:q|question|task)\s*\d+\s*[:.)\-*].*"     # "> Task 1:", "**Q1.", "Question 2 -"
                   r"|>\s*\w.*\?\s*)$", re.I | re.M)                        # "> What was ...?" (first quoted question)


def clean(md: str) -> str:
    md = re.sub(r"<details>.*?</details>", lambda m: re.sub(r"<[^>]+>", " ", m.group(0)), md, flags=re.S)
    md = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", md)                    # images
    md = re.sub(r"<img[^>]*>", "", md, flags=re.I)
    md = re.sub(r"\[([^\]]+)\]\((?:https?://)[^)]*\)", r"\1", md)   # links -> their text
    md = re.sub(r"<[^>]+>", "", md)                                  # remaining html tags
    md = re.sub(r"^(created|last updated):.*$", "", md, flags=re.I | re.M)
    md = re.sub(r"[ \t]+\n", "\n", md)
    return re.sub(r"\n{3,}", "\n\n", md).strip()


def split(text: str) -> tuple[str, str] | None:
    m = SPLIT.search(text)
    if not m:
        return None
    ctx, sol = text[:m.start()].strip(), text[m.start():].strip()
    if len(ctx) < MIN_CONTEXT or len(sol) < MIN_SOLUTION:
        return None
    return ctx[:MAX_CONTEXT], sol[:MAX_SOLUTION]


def key_identifiers() -> list[str]:
    """Every answer-key identifier of the dataset cases (all of them, however generic: excluding too much is safe)."""
    from src.ml.planner_dataset import load

    return sorted({str(e).lower() for r in load()["rows"] for e in r["entities"] if len(str(e)) >= 4})


def build(root: Path = ROOT, out: Path = OUT) -> Path:
    ids = key_identifiers()
    pairs, dropped, seen = [], {"short_or_no_split": 0, "names_case": 0, "has_identifier": 0, "duplicate": 0}, set()
    for f in sorted(root.rglob("*.md")):
        if f.name.lower() in SKIP_NAMES:
            continue
        text = clean(f.read_text(encoding="utf-8", errors="replace"))
        rel = f.relative_to(root).as_posix()
        if ALIASES.search(rel) or ALIASES.search(text):
            dropped["names_case"] += 1
            continue
        low = text.lower()
        hit = next((i for i in ids if i in low), None)
        if hit:
            dropped["has_identifier"] += 1
            continue
        p = split(text)
        if p is None:
            dropped["short_or_no_split"] += 1
            continue
        h = hashlib.sha256(p[1].encode()).hexdigest()
        if h in seen:
            dropped["duplicate"] += 1
            continue
        seen.add(h)
        pairs.append({"source": rel, "context": p[0], "solution": p[1]})
    out.mkdir(parents=True, exist_ok=True)
    path = out / "pairs.jsonl"
    with path.open("w", encoding="utf-8") as fh:
        for p in pairs:
            fh.write(json.dumps(p, ensure_ascii=False) + "\n")
    (out / "manifest.json").write_text(json.dumps({
        "pairs": len(pairs), "dropped": dropped, "identifiers_checked": len(ids),
        "repositories": sorted(d.name for d in root.iterdir() if d.is_dir()),
        "context_chars": sum(len(p["context"]) for p in pairs), "solution_chars": sum(len(p["solution"]) for p in pairs),
    }, indent=1), encoding="utf-8")
    return path


if __name__ == "__main__":
    p = build()
    print(p, (p.parent / "manifest.json").read_text(encoding="utf-8"))
