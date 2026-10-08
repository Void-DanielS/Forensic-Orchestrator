"""Score a case's verified findings against its answer key.

Agents never see ``context/answer_key`` or this module's gold files. Scoring happens afterwards, on the
findings the agents wrote to ``workspace/runs``:

1. **Audit.** The latest findings of each (agent, evidence item) are re-verified with a *fresh*
   :class:`HeuristicVerificationEngine` (its own bridge and hash cache). Only findings that are still
   VERIFIED count; anything else is reported as an audit failure.
2. **Gold questions** (``src/eval/gold/<case>.json``, transcribed from the answer-key document). Each
   question lists requirement groups; a group is met when any of its alternatives occurs in a verified
   finding (claim + value text, normalised). ``cat:<category>`` matches a finding category and
   ``cat:<category>|<text>`` both in one finding. With ``same_finding`` every group must be met by one finding.
   Per question: credit = groups met / groups; *answered* = all groups met. Each met group records the
   finding that met it, so every point of the score can be checked against its citation.
3. **Answer-key entity recall** (every case with an answer key): the specific identifiers in the answer key
   (e-mail, URL, domain, phone, file name, GPS) that are *not* already in the brief, and the share of them that
   appear in verified findings. A coarse, automatic signal; the gold questions are the measured score.
"""
from __future__ import annotations

import json
import re
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable
from urllib.parse import unquote_plus

from src.verification.models import Finding, FindingStatus

GOLD_DIR = Path(__file__).with_name("gold")
KEY_ENTITY_TYPES = ("email", "url", "domain", "phone", "filename", "gps")


def norm(s: str) -> str:
    """Matching form: URL-decoded, case-folded, ``\\`` as ``/``, whitespace collapsed."""
    try:
        s = unquote_plus(s)
    except Exception:  # noqa: BLE001
        pass
    s = s.casefold().replace("\\", "/")
    return re.sub(r"\s+", " ", s).strip()


def _flatten(v: Any, out: list[str]) -> None:
    if isinstance(v, dict):
        for x in v.values():
            _flatten(x, out)
    elif isinstance(v, (list, tuple)):
        for x in v:
            _flatten(x, out)
    elif v is not None and not isinstance(v, (bytes, bytearray)):
        out.append(str(v))


@dataclass
class Doc:
    finding_id: str
    category: str
    evidence_id: str
    claim: str
    text: str            # normalised claim + value


def finding_doc(f: dict) -> Doc:
    parts = [f.get("claim", "")]
    _flatten(f.get("value"), parts)
    return Doc(f.get("finding_id", ""), f.get("category", ""), f.get("evidence_id") or "", f.get("claim", ""),
               norm(" \n ".join(parts)))


# ------------------------------------------------------------------------------------ loading + audit
def latest_findings_files(case_dir: Path) -> list[Path]:
    """Newest ``*_findings.json`` per (agent, evidence item) across the case's runs."""
    best: dict[str, Path] = {}
    for p in (case_dir / "workspace" / "runs").glob("*/*_findings.json"):
        prev = best.get(p.name)
        if prev is None or p.parent.name > prev.parent.name:      # run ids start with a sortable timestamp
            best[p.name] = p
    return sorted(best.values())


def load_verified(case_dir: Path) -> tuple[list[dict], list[dict]]:
    """(verified finding dicts, file summaries) from the latest run of every agent / evidence item."""
    findings, files = [], []
    for p in latest_findings_files(case_dir):
        d = json.loads(p.read_text(encoding="utf-8"))
        v = d.get("verified_findings", [])
        findings += v
        files.append({"file": str(p.relative_to(case_dir)), "agent": d.get("agent"), "evidence_id": d.get("evidence_id"),
                      "verified": len(v), "rejected": len(d.get("rejected_findings", []))})
    return findings, files


def audit(case_dir: Path, findings: list[dict]) -> tuple[list[dict], dict]:
    """Re-verify every finding with a fresh engine per evidence item; keep those still VERIFIED."""
    from src.verification.heuristic_verifier import HeuristicVerificationEngine

    by_ev: dict[str, list[dict]] = defaultdict(list)
    for f in findings:
        by_ev[f.get("evidence_id") or ""].append(f)
    kept, failed = [], []
    for eid, fs in by_ev.items():
        root = case_dir / "evidence" / eid / "fs"
        if not root.is_dir():
            failed += [{"finding_id": f["finding_id"], "reason": "evidence item missing"} for f in fs]
            continue
        eng = HeuristicVerificationEngine(root)
        try:
            for f in fs:
                obj = Finding.model_validate({**f, "status": "pending", "verification": []})
                eng.verify_finding(obj)
                if obj.status == FindingStatus.VERIFIED:
                    kept.append(f)
                else:
                    reasons = [v.get("reason") for v in obj.verification if not v.get("verified")]
                    failed.append({"finding_id": f["finding_id"], "category": f.get("category"), "reasons": reasons[:3]})
        finally:
            eng.close()
    return kept, {"checked": len(findings), "still_verified": len(kept), "failed": len(failed), "failures": failed[:50]}


# ------------------------------------------------------------------------------------ gold questions
def _alt_hits(alt: str, doc: Doc) -> bool:
    if alt.startswith("cat:"):
        cat, _, text = alt[4:].partition("|")
        return doc.category == cat and (not text or norm(text) in doc.text)
    return norm(alt) in doc.text


def _group_hit(group: list[str], docs: Iterable[Doc]) -> Doc | None:
    for d in docs:
        if any(_alt_hits(a, d) for a in group):
            return d
    return None


def score_question(q: dict, docs: list[Doc]) -> dict:
    groups = q["all"]
    if q.get("same_finding"):
        best, best_doc, best_hits = 0, None, []
        for d in docs:
            hits = [g for g in groups if any(_alt_hits(a, d) for a in g)]
            if len(hits) > best:
                best, best_doc, best_hits = len(hits), d, hits
                if best == len(groups):
                    break
        met = [{"group": g, "finding_id": best_doc.finding_id, "claim": best_doc.claim[:200]} for g in best_hits]
        missing = [g for g in groups if g not in best_hits]
    else:
        met, missing = [], []
        for g in groups:
            d = _group_hit(g, docs)
            if d:
                met.append({"group": g, "finding_id": d.finding_id, "claim": d.claim[:200]})
            else:
                missing.append(g)
    credit = len(met) / len(groups) if groups else 0.0
    return {"id": q["id"], "question": q["question"], "answer": q["answer"], "needs": q.get("needs"),
            "credit": round(credit, 3), "answered": credit == 1.0, "met": met, "missing": missing}


def score_gold(gold: dict, docs: list[Doc]) -> dict:
    rows = [score_question(q, docs) for q in gold["questions"]]

    def summary(rs: list[dict]) -> dict:
        n = len(rs)
        return {"questions": n, "answered": sum(r["answered"] for r in rs),
                "answered_pct": round(100 * sum(r["answered"] for r in rs) / n, 1) if n else None,
                "mean_credit_pct": round(100 * sum(r["credit"] for r in rs) / n, 1) if n else None,
                "any_credit": sum(r["credit"] > 0 for r in rs)}

    return {"source": gold.get("source"), "excluded": gold.get("excluded", {}), "all": summary(rows),
            "covered_capabilities": summary([r for r in rows if not r["needs"]]), "rows": rows}


# ------------------------------------------------------------------------------------ automatic entity recall
def key_entities(case_dir: Path, summary: str = "") -> list[dict]:
    """Specific identifiers in the answer key that the brief does not already give away."""
    from src.context.case_context import CaseContext

    key_dir = case_dir / "context" / "answer_key"
    docs = []
    for t in sorted(key_dir.glob("*.txt")) if key_dir.is_dir() else []:
        text = t.read_text(encoding="utf-8", errors="replace")
        if text.strip() and not text.startswith("[text extraction failed"):
            docs.append((t.name, text))
    if not docs:
        return []
    key = CaseContext(case_dir.name, "", docs)
    brief = CaseContext.load(case_dir, summary)
    given = {e.norm for e in brief.entities.values()}
    out = []
    for e in key.entities.values():
        if e.type in KEY_ENTITY_TYPES and e.norm not in given and len(e.norm) >= 5:
            out.append({"type": e.type, "value": e.value, "norm": e.norm})
    return out


def entity_recall(entities: list[dict], findings: list[dict]) -> dict:
    from src.verification.heuristic_verifier import normalize_text

    if not entities:
        return {"entities": 0}
    texts = []
    for f in findings:
        parts = [f.get("claim", "")]
        _flatten(f.get("value"), parts)
        texts.append(" " + normalize_text(" ".join(parts)) + " ")
    blob = "\n".join(texts)
    found = [e for e in entities if f" {e['norm']} " in blob]
    pattern = re.compile("|".join(re.escape(f" {e['norm']} ") for e in sorted(entities, key=lambda e: -len(e["norm"]))))
    on_target = sum(1 for t in texts if pattern.search(t))
    by_type: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    for e in entities:
        by_type[e["type"]][1] += 1
    for e in found:
        by_type[e["type"]][0] += 1
    return {"entities": len(entities), "found": len(found),
            "recall_pct": round(100 * len(found) / len(entities), 1),
            "by_type": {k: {"found": v[0], "total": v[1]} for k, v in sorted(by_type.items())},
            "findings_on_target": on_target, "findings": len(findings),
            "on_target_pct": round(100 * on_target / len(findings), 2) if findings else None,
            "found_examples": [e["value"] for e in found[:25]],
            "missed_examples": [e["value"] for e in entities if e not in found][:25]}


# ------------------------------------------------------------------------------------ case score
@dataclass
class CaseScore:
    case_id: str
    files: list[dict] = field(default_factory=list)
    audit: dict = field(default_factory=dict)
    gold: dict | None = None
    entity: dict = field(default_factory=dict)
    categories: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {"case_id": self.case_id, "files": self.files, "audit": self.audit, "gold": self.gold,
                "entity_recall": self.entity, "categories": self.categories}


def score_case(case, reverify: bool = True) -> CaseScore:
    from collections import Counter

    findings, files = load_verified(case.dir)
    cs = CaseScore(case.case_id, files=files)
    if reverify:
        findings, cs.audit = audit(case.dir, findings)
    cs.categories = dict(Counter(f.get("category") for f in findings).most_common())
    gold_path = GOLD_DIR / f"{case.case_id}.json"
    if gold_path.exists():
        cs.gold = score_gold(json.loads(gold_path.read_text(encoding="utf-8")), [finding_doc(f) for f in findings])
    cs.entity = entity_recall(key_entities(case.dir, case.summary), findings)
    return cs


def write_case_score(case, cs: CaseScore) -> Path:
    out = case.dir / "workspace" / "reports"
    out.mkdir(parents=True, exist_ok=True)
    (out / "score.json").write_text(json.dumps(cs.to_dict(), indent=2, ensure_ascii=False), encoding="utf-8")
    lines = [f"# Score against the answer key: {case.case_id}", ""]
    a = cs.audit
    if a:
        lines += [f"Audit (fresh re-verification): {a['still_verified']} of {a['checked']} findings still VERIFIED, "
                  f"{a['failed']} failed.", ""]
    if cs.gold:
        g = cs.gold
        lines += [f"Source: {g['source']}", "",
                  f"**Gold questions: {g['all']['answered']} / {g['all']['questions']} answered "
                  f"({g['all']['answered_pct']} %), mean credit {g['all']['mean_credit_pct']} %.** "
                  f"Questions needing capabilities not built yet are marked in *needs*; without them: "
                  f"{g['covered_capabilities']['answered']} / {g['covered_capabilities']['questions']} answered, "
                  f"mean credit {g['covered_capabilities']['mean_credit_pct']} %.", "",
                  "| Q | Question | Credit | Needs | Evidence (finding id: claim) | Missing |", "|---|---|---|---|---|---|"]
        for r in g["rows"]:
            ev = "<br>".join(f"`{m['finding_id']}` {m['claim'][:90]}".replace("|", "\\|") for m in r["met"][:3])
            miss = ", ".join(" / ".join(x) for x in r["missing"][:4]).replace("|", "\\|")
            lines.append(f"| {r['id']} | {r['question']} | {int(r['credit'] * 100)} % | {r['needs'] or ''} | {ev} | {miss} |")
        lines += ["", "Excluded questions: " + "; ".join(f"Q{k} ({v})" for k, v in g["excluded"].items()), ""]
    e = cs.entity
    if e.get("entities"):
        lines += [f"Answer-key entity recall: {e['found']} / {e['entities']} ({e['recall_pct']} %); "
                  f"{e['findings_on_target']} of {e['findings']} verified findings mention one ({e['on_target_pct']} %).", ""]
    path = out / "score.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path
