"""Markdown report of one vision run over a case: evidence, context contrast, observations.

Evidence sections contain VERIFIED findings only, each with its citation (file, locator,
SHA-256). Context links and model observations are in clearly labelled separate sections.
The consolidation agent reads the JSON findings files listed at the end, not this text.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

from src.context.case_context import CaseContext, InvestigationMemory


def _md(s: str, n: int = 160) -> str:
    s = " ".join(str(s).split())
    s = s if len(s) <= n else s[: n - 1] + "…"
    return s.replace("|", "\\|")


def _cite(c: dict) -> str:
    loc = {"byte_offset": c.get("byte_offset"), "line_number": c.get("line_number"), "sql_query": c.get("sql_query"),
           "regex": c.get("regex"), "exif_tag": c.get("exif_tag")}[c["locator_type"]]
    params = f" {c['sql_params']}" if c.get("sql_params") else ""
    return f"`{_md(c['file_path'], 70)}` {c['locator_type']}=`{_md(loc, 60)}`{params} sha256 `{c['expected_hash'][:12]}…`"


def write_vision_report(case, ctx: CaseContext, memory: InvestigationMemory, run_id: str, reports: list[dict],
                        run_dir: Path) -> Path:
    out = case.dir / "workspace" / "reports" / f"{run_id}_vision.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    cd = ctx.to_dict()
    verified = [f for r in reports for f in r["verified_findings"]]
    rejected = [f for r in reports for f in r["rejected_findings"]]
    observ = [f for r in reports for f in r["model_observations_uncorroborated"]]
    run_links = [l for l in memory.links if l["run_id"] == run_id]
    L: list[str] = []
    w = L.append

    w(f"# {case.title}")
    w(f"_Vision agent report, run `{run_id}`, generated {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}._\n")
    w("> Evidence statements below are **VERIFIED**: each was re-executed against the evidence file "
      "(hash + locator) by the deterministic verifier. Context links and model observations are analysis aids, "
      "not evidence.\n")

    w("## 1. Investigation brief (context given to the agent)")
    w(f"{case.summary}\n")
    w(f"Brief documents: {', '.join(d['name'] for d in cd['brief_docs']) or '—'}  ")
    tf = cd["timeframe"]
    w(f"Timeframe described by the brief: {tf['start']} → {tf['end']} ({tf['dated_mentions']} dated mentions)  "
      if tf else "Timeframe: not stated in the brief  ")
    by_type = defaultdict(list)
    for e in cd["entities"]:
        by_type[e["type"]].append(e)
    for t in ("person", "email", "phone", "device", "domain", "filename", "gps"):
        if by_type[t]:
            w(f"- **{t}**: " + ", ".join(f"{_md(e['value'], 40)}" for e in by_type[t][:15]))
    w("")

    w("## 2. What the agent examined")
    w("| Evidence | Images | Verified | Rejected | Observations | Runtime | Device / peak VRAM |")
    w("|---|---|---|---|---|---|---|")
    for r in reports:
        sc, res = r["status_counts"], r["resources"]
        w(f"| `{r['evidence_id']}` | {r['artifacts_analyzed']} | {sc['verified']} | {sc['rejected']} | "
          f"{sc['uncorroborated']} | {r['runtime_s']} s | {res.get('device')} / {res.get('peak_allocated_gb', '—')} GB |")
    errs = [e for r in reports for e in r["errors"]]
    if errs:
        w(f"\nArtifacts that could not be processed ({len(errs)}): " + "; ".join(_md(e, 100) for e in errs[:8]))
    w("")

    w("## 3. Verified evidence")
    if not verified:
        w("_No verified findings._\n")
    by_art = defaultdict(list)
    for f in verified:
        by_art[(f["evidence_id"], f["artifact_path"])].append(f)
    for (eid, art), fs in sorted(by_art.items()):
        w(f"**`{eid}` · `{_md(art, 110)}`**\n")
        w("| ID | Category | Statement | Citation(s) |")
        w("|---|---|---|---|")
        for f in fs:
            w(f"| `{f['finding_id']}` | {f['category']} | {_md(f['claim'])} | "
              + "<br>".join(_cite(c) for c in f["citations"]) + " |")
        w("")

    w("## 4. Contrast with the investigation context")
    rel = Counter(l["relation"] for l in run_links)
    w("| Relation | Count | Meaning |")
    w("|---|---|---|")
    meaning = {"corroborates": "verified evidence contains a person/e-mail/phone/file/device named in the brief",
               "keyword_match": "verified evidence contains a recurring keyword of the brief (weak)",
               "outside_timeframe": "verified timestamp outside the period the brief describes",
               "not_in_context": "verified device identity the brief never mentions",
               "new_lead": "verified finding with no counterpart in the brief",
               "unverified_mention": "model observation mentions a brief entity (follow up, not evidence)"}
    for k, v in meaning.items():
        w(f"| {k} | {rel.get(k, 0)} | {v} |")
    w("")
    for relation, title in (("corroborates", "Brief entities confirmed by evidence"),
                            ("outside_timeframe", "Timestamps outside the brief's timeframe"),
                            ("not_in_context", "Devices not mentioned in the brief"),
                            ("new_lead", "New leads (verified, not in the brief)"),
                            ("unverified_mention", "Follow-up: model observations mentioning brief entities")):
        links = [l for l in run_links if l["relation"] == relation]
        if not links:
            continue
        w(f"### {title}")
        seen = set()
        for l in links[:40]:
            key = (l.get("value"), l["artifact"])
            if key in seen:
                continue
            seen.add(key)
            w(f"- `{l['finding_id']}` `{l['evidence_id']}` · {_md(l['artifact'], 80)}: {_md(l['why'], 220)}")
        w("")

    w("## 5. Rejected claims (hallucination guard)")
    if not rejected:
        w("_None: every citation the agent produced was reproduced from the evidence._\n")
    for f in rejected[:30]:
        reasons = "; ".join(_md(v.get("reason", ""), 90) for v in f["verification"] if not v.get("verified"))
        w(f"- `{f['finding_id']}` {f['category']} {_md(f['claim'], 120)} — {reasons}")
    w("")

    w("## 6. Model observations (NOT evidence)")
    w(f"{len(observ)} OCR / VLM observations had no ground truth in the evidence. Scene descriptions:")
    for f in [f for f in observ if f["category"] == "vlm.description"][:20]:
        w(f"- `{f['evidence_id']}` {_md(f['artifact_path'].rsplit('/', 1)[-1], 50)}: {_md(f['claim'], 200)}")
    w("")

    w("## 7. Hand-off")
    w("Machine-readable inputs for the consolidation / audit agent:")
    for r in reports:
        stem = f"multimodalvisionagent__{r['evidence_id']}"
        w(f"- `{(run_dir / f'{stem}_findings.json').relative_to(case.dir).as_posix()}` (+ `_verification_audit.json`)")
    w(f"- `workspace/memory.json` (context entities and {len(memory.links)} cumulative links)")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out


def write_agents_report(case, ctx: CaseContext, memory: InvestigationMemory, run_id: str, reports: list[dict],
                        run_dir: Path, per_category: int = 25) -> Path:
    """Report of a run of the deterministic agents (AppArtifactAgent, FileSystemAgent).

    These agents emit thousands of rows; the report gives counts per category and evidence item and the
    first ``per_category`` verified findings of each category with their citations. The complete set is
    in the findings JSON files listed in the hand-off section.
    """
    out = case.dir / "workspace" / "reports" / f"{run_id}_agents.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    verified = [f for r in reports for f in r["verified_findings"]]
    rejected = [f for r in reports for f in r["rejected_findings"]]
    run_links = [l for l in memory.links if l["run_id"] == run_id]
    agents = sorted({r["agent"] for r in reports})
    L: list[str] = []
    w = L.append
    w(f"# {case.title}")
    w(f"_{' + '.join(agents)} report, run `{run_id}`, generated "
      f"{datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}._\n")
    w("> Every statement in sections 2-3 is **VERIFIED**: the deterministic verifier re-executed its citations "
      "(read-only SQL by rowid, or bytes at an offset) against the evidence file and re-checked its SHA-256.\n")
    w("## 1. Brief")
    w(f"{case.summary}\n")
    w("## 2. Coverage")
    w("| Agent | Evidence | Artifacts | Verified | Rejected | Runtime | Errors |")
    w("|---|---|---|---|---|---|---|")
    for r in reports:
        sc = r["status_counts"]
        w(f"| {r['agent']} | `{r['evidence_id']}` | {r['artifacts_analyzed']} | {sc['verified']} | {sc['rejected']} | "
          f"{r['runtime_s']} s | {len(r['errors'])} |")
    cats = Counter(f["category"] for f in verified)
    w("\n| Category | Verified findings |")
    w("|---|---|")
    for c, n in cats.most_common():
        w(f"| {c} | {n} |")
    w("")
    w("## 3. Verified evidence (first findings per category)")
    by_cat = defaultdict(list)
    for f in verified:
        by_cat[f["category"]].append(f)
    for c, fs in sorted(by_cat.items()):
        w(f"### {c} ({len(fs)})")
        w("| ID | Evidence | Statement | Citation(s) |")
        w("|---|---|---|---|")
        for f in fs[:per_category]:
            w(f"| `{f['finding_id']}` | `{f['evidence_id']}` | {_md(f['claim'])} | "
              + "<br>".join(_cite(c) for c in f["citations"][:3]) + (" …" if len(f["citations"]) > 3 else "") + " |")
        w("")
    w("## 4. Contrast with the investigation context")
    rel = Counter(l["relation"] for l in run_links)
    w(", ".join(f"{k}: {v}" for k, v in rel.most_common()) or "_No context links._")
    for l in [l for l in run_links if l["relation"] == "corroborates"][:40]:
        w(f"- `{l['finding_id']}` `{l['evidence_id']}` · {_md(l['artifact'], 80)}: {_md(l['why'], 200)}")
    w("")
    w("## 5. Rejected claims")
    if not rejected:
        w("_None: every citation was reproduced from the evidence._\n")
    for f in rejected[:30]:
        reasons = "; ".join(_md(v.get("reason", ""), 90) for v in f["verification"] if not v.get("verified"))
        w(f"- `{f['finding_id']}` {f['category']} {_md(f['claim'], 120)} — {reasons}")
    w("")
    w("## 6. Hand-off")
    for r in reports:
        stem = f"{r['agent'].lower()}__{r['evidence_id']}"
        w(f"- `{(run_dir / f'{stem}_findings.json').relative_to(case.dir).as_posix()}` (+ `_verification_audit.json`)")
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    return out
