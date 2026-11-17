"""ForensiAgent-Multimodal CLI.

    python main.py config                       # show device allocation
    python main.py fetch [--source android_10]  # download + inspect public reference image
    python main.py inspect                      # re-inspect without downloading
    python main.py verify-selftest              # verifier accepts true / rejects false claims
    python main.py vision [--limit N] [--no-vlm] [--no-ocr]
    python main.py pipeline [--limit N]         # inspect -> selftest -> vision -> verifier

Outputs go to data/outputs/.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from src.config import settings
from src.utils import dataset_scraper


def _sandbox_root(source_id: str) -> Path:
    from src.cases.registry import evidence_dir

    root = evidence_dir(source_id) / "fs"
    if not root.is_dir():
        sys.exit(f"Evidence missing at {root}. Run: python main.py cases build --only {source_id}")
    return root


def _all_evidence_ids() -> list[str]:
    from src.cases.registry import CASES

    return sorted({e for c in CASES.values() for e in c.evidence_ids()} | set(dataset_scraper.SOURCES))


def cmd_cases(a: argparse.Namespace) -> int:
    from src.cases import builder
    from src.cases.registry import CASES

    if a.action == "list":
        builder.print_cases()
        return 0
    if a.action == "catalog":
        print(f"[catalog] {builder.build_catalog()}")
        return 0
    ids = list(CASES) if a.cases == "all" else [c.strip() for c in a.cases.split(",") if c.strip()]
    unknown = [c for c in ids if c not in CASES]
    if unknown:
        sys.exit(f"unknown case(s) {unknown}; see: python main.py cases list")
    if a.action in ("audit", "repair"):
        from src.utils import integrity

        sel = [CASES[c] for c in ids]
        if a.action == "audit":
            res = integrity.audit(sel, do_hash=not a.no_hash)
            md = integrity.write_audit(res)
            print(md.read_text(encoding="utf-8")[:4000])
            return 1 if res.problems else 0
        out = integrity.repair(sel, dry_run=a.dry_run)
        print(json.dumps({k: v for k, v in out.items() if k not in ("restored",)}, indent=1)[:6000])
        print(f"[repair] restored {len(out['restored'])}, not restored {len(out['not_restored'])}, "
              f"protected {out['protected_read_only']} at-risk files -> {out['summary_path']}")
        return 0 if not out["not_restored"] else 1
    only = {x.strip() for x in a.only.split(",") if x.strip()} or None
    if only:
        ids = [c for c in ids if only & set(CASES[c].evidence_ids())] or ids
    return builder.build(ids, set(a.groups.split(",")), only=only, docs_only=a.docs_only)


def cmd_config(_: argparse.Namespace) -> int:
    print(settings.describe())
    import torch

    print(f"torch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  cuda:{i} {p.name} {p.total_memory / 1024**3:.1f} GB")
    print(f"vision device: {settings.device_for('vision')}, VLM: {settings.VISION.vlm_model_id} ({settings.VISION.quantization})")
    return 0


def cmd_fetch(a: argparse.Namespace) -> int:
    return dataset_scraper.main(["--source", a.source])


def cmd_inspect(a: argparse.Namespace) -> int:
    return dataset_scraper.main(["--source", a.source, "--inspect-only"])


def cmd_verify_selftest(a: argparse.Namespace) -> int:
    """Feed the verifier known-true and deliberately false claims built from the real dataset."""
    from src.bridge.forensic_bridge import ForensicBridge
    from src.verification.heuristic_verifier import HeuristicVerificationEngine
    from src.verification.models import Citation, Finding, FindingStatus

    bridge = ForensicBridge(_sandbox_root(a.source))
    eng = HeuristicVerificationEngine(bridge.sandbox.root, bridge=bridge)
    sb = bridge.sandbox
    sms = "data/data/com.android.providers.telephony/databases/mmssms.db"
    img = "data/media/0/DCIM/Camera/IMG_20200209_132353.jpg"
    body = bridge.sqlite.query(sms, "SELECT body FROM sms WHERE body IS NOT NULL ORDER BY _id LIMIT 1")[0][0]
    make = bridge.images.exif(img)["0th.Make"]
    lat = bridge.images.exif(img)["GPS.GPSLatitude"]
    log_rel = next(e.path for e in bridge.fs.walk("data/system") if e.extension == ".xml" and e.size > 200)
    line2 = bridge.text.line(log_rel, 2) or ""

    def f(name: str, *cites: Citation, observed: str | None = None) -> Finding:
        return Finding(agent="selftest", artifact_path=name, category="selftest", claim=name,
                       citations=list(cites), observed_text=observed)

    q = "SELECT body FROM sms WHERE body = ?"
    cases = [
        (True, f("sql: real SMS body", Citation(file_path=sms, locator_type="sql_query", sql_query=q, sql_params=[body],
                                                expected_hash=sb.hash(sms), raw_text_snippet=body))),
        (False, f("sql: hallucinated SMS body", Citation(file_path=sms, locator_type="sql_query",
                  sql_query="SELECT body FROM sms", expected_hash=sb.hash(sms),
                  raw_text_snippet="Meet me at the warehouse at midnight"))),
        (False, f("sql: write attempt blocked", Citation(file_path=sms, locator_type="sql_query",
                  sql_query="DELETE FROM sms", expected_hash=sb.hash(sms), raw_text_snippet=body))),
        (True, f("exif: real camera make", Citation(file_path=img, locator_type="exif_tag", exif_tag="0th.Make",
                                                    expected_hash=sb.hash(img), raw_text_snippet=make))),
        (True, f("exif: real GPS latitude", Citation(file_path=img, locator_type="exif_tag", exif_tag="GPS.GPSLatitude",
                                                     expected_hash=sb.hash(img), raw_text_snippet=lat))),
        (False, f("exif: wrong camera make", Citation(file_path=img, locator_type="exif_tag", exif_tag="0th.Make",
                                                      expected_hash=sb.hash(img), raw_text_snippet="Apple"))),
        (False, f("hash: tampered evidence", Citation(file_path=img, locator_type="exif_tag", exif_tag="0th.Make",
                                                      expected_hash="0" * 64, raw_text_snippet=make))),
        (False, f("sandbox: path traversal", Citation(file_path="../../../../main.py", locator_type="line_number",
                                                      line_number=1, expected_hash="0" * 64, raw_text_snippet="x"))),
        (True, f("line: real log line", Citation(file_path=log_rel, locator_type="line_number", line_number=2,
                                                 expected_hash=sb.hash(log_rel), raw_text_snippet=line2[:40],
                                                 match_mode="contains"))),
        (True, f("ocr: corroborated text", Citation(file_path=sms, locator_type="sql_query", sql_query=q,
                 sql_params=[body], expected_hash=sb.hash(sms), raw_text_snippet=body), observed=body)),
        (False, f("ocr: text does not match cited row", Citation(file_path=sms, locator_type="sql_query",
                  sql_query=q, sql_params=[body], expected_hash=sb.hash(sms), raw_text_snippet=body),
                  observed="completely unrelated words here")),
        (False, f("vlm: uncited claim")),
    ]
    failures = 0
    for expect, finding in cases:
        eng.verify_finding(finding)
        ok = finding.status == FindingStatus.VERIFIED
        passed = ok == expect
        failures += not passed
        reason = "; ".join(str(v.get("reason")) for v in finding.verification)
        print(f"  [{'PASS' if passed else 'FAIL'}] {finding.claim:<38} -> {finding.status.value:<14} {reason[:90]}")
    bridge.close()
    print(f"selftest: {len(cases) - failures}/{len(cases)} passed")
    return 1 if failures else 0


def cmd_vision(a: argparse.Namespace) -> int:
    from src.agents.vision_agent import MultimodalVisionAgent, discover_images
    from src.bridge.forensic_bridge import ForensicBridge
    from src.bridge.text_index import CaseTextIndex
    from src.verification.models import FindingStatus

    cfg = settings.VISION
    cfg.enable_vlm = cfg.enable_vlm and not a.no_vlm
    cfg.enable_ocr = cfg.enable_ocr and not a.no_ocr
    bridge = ForensicBridge(_sandbox_root(a.source))
    images = discover_images(bridge, cfg)
    # Prioritise user media (camera, screenshots) over app-bundled images.
    images.sort(key=lambda p: (not any(k in p for k in ("/DCIM/", "/Screenshots/")), p))
    if a.images:
        wanted = [x.strip() for x in a.images.split(",") if x.strip()]
        missing = [w for w in wanted if w not in images]
        if missing:
            sys.exit(f"not in sandbox: {missing}")
        images = wanted
    if a.limit:
        images = images[: a.limit]
    print(f"[vision] {len(images)} images, device={settings.device_for('vision')}, VLM={'on' if cfg.enable_vlm else 'off'}, OCR={'on' if cfg.enable_ocr else 'off'}")

    index = CaseTextIndex(bridge).build()
    print(f"[index] {index.stats()}")
    agent = MultimodalVisionAgent(bridge, text_index=index, cfg=cfg)
    out = settings.OUTPUTS_DIR / a.source
    report = agent.run(images, output_dir=out)
    agent.release()
    bridge.close()

    d = report.to_dict()
    print(f"[vision] done in {d['runtime_s']}s  status={d['status_counts']}  resources={d['resources']}")
    for fnd in report.by_status(FindingStatus.VERIFIED)[:25]:
        print(f"  VERIFIED  {fnd.category:<20} {fnd.artifact_path.rsplit('/', 1)[-1]:<32} {fnd.claim[:90]}")
    if report.errors:
        print("[errors]", *report.errors, sep="\n  ")
    print(f"[out] {out / 'multimodalvisionagent_findings.json'}")
    return 0 if not report.by_status(FindingStatus.REJECTED) else 2


def cmd_investigate(a: argparse.Namespace) -> int:
    from src.pipeline.investigate import investigate

    res = investigate(a.case, evidence=[x for x in a.items.split(",") if x] or None, limit=a.limit,
                      images=[x.strip() for x in a.images.split(",") if x.strip()] or None,
                      ocr=not a.no_ocr, vlm=not a.no_vlm, run_id=a.run_id or None)
    print(f"[investigate] run {res['run_id']} totals={res['totals']}")
    print(f"[report] {res['report']}")
    return 0 if not res["totals"]["rejected"] else 2


def cmd_agents(a: argparse.Namespace) -> int:
    """apps / fs / win / investigate --agents apps,fs,win: deterministic agents over one case."""
    from src.pipeline.investigate import investigate_agents

    agents = a.agents.split(",") if getattr(a, "agents", "") else [a.cmd]
    res = investigate_agents(a.case, agents, evidence=[x for x in a.items.split(",") if x] or None, limit=a.limit,
                             carve=not getattr(a, "no_carve", False), run_id=a.run_id or None)
    print(f"[{'+'.join(agents)}] run {res['run_id']} totals={res['totals']}")
    print(f"[report] {res['report']}")
    return 0 if not res["totals"]["rejected"] else 2


def cmd_corpus(a: argparse.Namespace) -> int:
    """Run the deterministic agents (apps, fs, win) over every READY case; summary in data/outputs/corpus_runs/."""
    from collections import Counter

    from src.cases.builder import readiness
    from src.cases.registry import CASES
    from src.pipeline.investigate import investigate_agents

    agents = a.agents.split(",")
    ready = [c for c in CASES.values() if readiness(c)["ready"]]
    if a.cases != "all":
        ready = [c for c in ready if c.case_id in a.cases.split(",")]
    out_dir = settings.OUTPUTS_DIR / "corpus_runs"
    out_dir.mkdir(parents=True, exist_ok=True)
    rows, cats, totals = [], Counter(), Counter()
    for i, case in enumerate(ready, 1):
        print(f"[corpus] {i}/{len(ready)} {case.case_id}", flush=True)
        try:
            res = investigate_agents(case.case_id, agents, limit=a.limit, carve=not a.no_carve)
        except (Exception, SystemExit) as exc:  # noqa: BLE001 - one case must not stop the corpus run
            rows.append({"case_id": case.case_id, "error": f"{type(exc).__name__}: {exc}"})
            print(f"  [error] {rows[-1]['error']}", flush=True)
            continue
        for r in res["reports"]:
            cats.update(f["category"] for f in r["verified_findings"])
        totals.update(res["totals"])
        rows.append({"case_id": case.case_id, "run_id": res["run_id"], "totals": res["totals"], "report": res["report"],
                     "per_evidence": [{"agent": r["agent"], "evidence_id": r["evidence_id"],
                                       "artifacts": r["artifacts_analyzed"], **r["status_counts"],
                                       "errors": len(r["errors"])} for r in res["reports"]]})
    summary = {"agents": agents, "cases": len(ready), "totals": dict(totals), "categories": dict(cats.most_common()),
               "runs": rows}
    path = out_dir / f"corpus_{'-'.join(agents)}.json"
    path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[corpus] {len(ready)} cases totals={dict(totals)} -> {path}")
    return 0 if not totals.get("rejected") else 2


def cmd_score(a: argparse.Namespace) -> int:
    """Score verified findings against answer keys (gold questions + answer-key entity recall)."""
    from src.cases.builder import readiness
    from src.cases.registry import CASES
    from src.eval.scoring import GOLD_DIR, score_case, write_case_score

    if a.cases == "gold":
        ids = sorted(p.stem for p in GOLD_DIR.glob("*.json"))
    elif a.cases == "all":
        ids = [c.case_id for c in CASES.values() if readiness(c)["ready"]]
    else:
        ids = a.cases.split(",")
    rows = []
    for cid in ids:
        case = CASES[cid]
        cs = score_case(case, reverify=not a.no_reverify)
        path = write_case_score(case, cs)
        d = cs.to_dict()
        g = d["gold"]["all"] if d["gold"] else None
        e = d["entity_recall"]
        rows.append({"case_id": cid, "audit": {k: v for k, v in d["audit"].items() if k != "failures"},
                     "gold": g, "gold_covered": d["gold"]["covered_capabilities"] if d["gold"] else None,
                     "entity_recall": {k: e.get(k) for k in ("entities", "found", "recall_pct", "on_target_pct")},
                     "report": str(path)})
        print(f"[score] {cid}: audit={rows[-1]['audit'] or '-'} gold={g or '-'} entity_recall={e.get('recall_pct', '-')} %"
              f" -> {path}", flush=True)
    out = settings.OUTPUTS_DIR / "scores"
    out.mkdir(parents=True, exist_ok=True)
    (out / f"scores_{a.cases if a.cases in ('all', 'gold') else 'selected'}.json").write_text(
        json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    failed = sum((r["audit"] or {}).get("failed", 0) for r in rows)
    return 0 if not failed else 2


def cmd_dataset(a: argparse.Namespace) -> int:
    from src.datasets.export import export

    out = export()
    print((out / "DATASHEET.md").read_text(encoding="utf-8"))
    return 0


def cmd_finetune(a: argparse.Namespace) -> int:
    from src.datasets.finetune import export_finetune, validate_jsonl

    if a.validate:
        ok, errors = validate_jsonl(Path(a.validate))
        print(f"[finetune] {ok} valid sample(s), {len(errors)} error(s)")
        for e in errors[:20]:
            print("  " + e)
        return 1 if errors else 0
    out = export_finetune(Path(a.dataset) if a.dataset else None)
    print((out / "FINETUNE_CARD.md").read_text(encoding="utf-8"))
    return 0


def cmd_jev(a: argparse.Namespace) -> int:
    """Jev typed-answer model: build the fragment dataset, train / evaluate it, or ask it about a file."""
    if a.action == "build":
        from src.ml.fragments import build

        out = build(max_files=a.max_files)
        print((out / "DATASHEET.md").read_text(encoding="utf-8"))
        return 0
    if a.action == "train":
        from src.ml.train import train

        res = train(Path(a.dataset) if a.dataset else None, epochs=a.epochs, batch=a.batch, lr=a.lr, seed=a.seed)
        print((Path(res["run"]) / "REPORT.md").read_text(encoding="utf-8"))
        print(f"[jev] checkpoint {res['checkpoint']}")
        return 0
    if a.action == "eval":
        from src.ml.train import evaluate

        print(json.dumps(evaluate(Path(a.checkpoint) if a.checkpoint else None,
                                  Path(a.dataset) if a.dataset else None), indent=2))
        return 0
    from src.ml.jev import FRAG, Jev

    if not a.file:
        raise SystemExit("jev ask needs --file (and optionally --offset, --ext)")
    with open(a.file, "rb") as fh:
        fh.seek(a.offset)
        frag = fh.read(FRAG)
    jev = Jev.load(Path(a.checkpoint) if a.checkpoint else None)
    ext = a.ext or Path(a.file).suffix
    print(json.dumps({"context": {"file": a.file, "offset": a.offset, "bytes": len(frag), "extension": ext},
                      "choice": jev.choice(frag, [o for o in a.options.split(",") if o] or None),
                      "yes_no": jev.yes_no(frag, ext), "score": jev.score(frag)}, indent=2))
    return 0


def cmd_mm(a: argparse.Namespace) -> int:
    """Jev-MM multimodal image-triage model: dataset (build / ocr / encode), train, eval, ask."""
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from src.ml import mm_dataset as mmd

    ds = Path(a.dataset) if a.dataset else None
    ck = Path(a.checkpoint) if a.checkpoint else None
    if a.action == "build":
        out = mmd.build(max_per_class=a.max_files)
        print((out / "DATASHEET.md").read_text(encoding="utf-8"))
        return 0
    if a.action == "ocr":
        print(f"[mm] OCR -> {mmd.run_ocr(ds)}")
        return 0
    if a.action == "encode":
        print(f"[mm] text features -> {mmd.encode_text(ds)}")
        return 0
    if a.action == "prepare":
        out = mmd.build(max_per_class=a.max_files)
        mmd.run_ocr(out)
        mmd.encode_text(out)
        print((out / "DATASHEET.md").read_text(encoding="utf-8"))
        return 0
    if a.action == "train":
        from src.ml.mm_train import train

        res = train(ds, epochs=a.epochs, batch=a.batch, lr=a.lr, seed=a.seed, ablations=not a.no_ablations,
                    seeds=tuple(int(x) for x in a.seeds.split(",") if x.strip()) or None)
        print((Path(res["run"]) / "REPORT.md").read_text(encoding="utf-8"))
        print(f"[mm] checkpoint {res['checkpoint']}")
        return 0
    if a.action == "eval":
        from src.ml.mm_train import evaluate

        print(json.dumps(evaluate(ck, ds), indent=2))
        return 0
    from src.ml.multimodal import JevMM

    if not a.file:
        raise SystemExit("mm ask needs --file IMAGE[,IMAGE...]")
    print(json.dumps(JevMM.load(ck).ask([f for f in a.file.split(",") if f], ocr=not a.no_ocr), indent=2,
                     ensure_ascii=False))
    return 0


def cmd_planner(a: argparse.Namespace) -> int:
    """Investigation planner (deep learning): learns from solved cases where to look in a new one."""
    import os

    os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from src.ml import planner_dataset as pd_

    ds = Path(a.dataset) if a.dataset else None
    if a.action == "build":
        out = pd_.build()
        pd_.encode(out)
        print((out / "build.json").read_text(encoding="utf-8")[:3000])
        return 0
    if a.action == "encode":
        print(f"[planner] text features -> {pd_.encode(ds)}")
        return 0
    if a.action == "pretrain-build":
        from src.ml.planner_pretrain import build as pre_build

        print((pre_build() / "build.json").read_text(encoding="utf-8")[:2000])
        return 0
    if a.action == "pretrain":
        from src.ml.planner_pretrain import pretrain

        res = pretrain(ds, epochs=a.epochs if a.epochs != 10 else 20, seed=a.seed)
        print((Path(res["run"]) / "REPORT.md").read_text(encoding="utf-8"))
        return 0
    if a.action == "train":
        from src.ml.planner_train import train

        res = train(ds, epochs=a.epochs, seed=a.seed, ablations=not a.no_ablations)
        print((Path(res["run"]) / "REPORT.md").read_text(encoding="utf-8"))
        return 0
    from src.cases.registry import CASES
    from src.pipeline.solve import plan, planner_for

    if a.case not in CASES:
        raise SystemExit("planner ask needs --case <id>")
    pl, ck = planner_for(a.case, Path(a.checkpoint) if a.checkpoint else None)
    rows, p, _ = plan(CASES[a.case], pl)
    for rank, i in enumerate(sorted(range(len(rows)), key=lambda j: -p[j])[:a.k], 1):
        print(f"{rank:4d}  p={p[i]:.3f}  {rows[i]['category']:<16} {rows[i]['name'][-100:]}")
    print(f"[planner] {ck}")
    return 0


def cmd_solve(a: argparse.Namespace) -> int:
    """Deep-learning investigation: planner -> image / reader agents -> verifier -> report."""
    from src.pipeline.solve import solve

    cases = [c.strip() for c in a.case.split(",") if c.strip()]
    out = [solve(c, k=a.k, checkpoint=Path(a.checkpoint) if a.checkpoint else None, vlm=a.vlm) for c in cases]
    print(json.dumps(out, indent=1, default=str)[:6000])
    return 0


def cmd_orchestrate(a: argparse.Namespace) -> int:
    """Orchestrated multi-agent investigation: planner orchestrator + LLM analyst + image agent, in rounds."""
    from src.pipeline.orchestrate import investigate

    cases = [c.strip() for c in a.case.split(",") if c.strip()]
    out = [investigate(c, k=a.k, batch=a.batch, feedback=not a.no_feedback, vlm=not a.no_vlm,
                       checkpoint=Path(a.checkpoint) if a.checkpoint else None, tools=not a.no_tool,
                       ranker=a.ranker) for c in cases]
    print(json.dumps(out, indent=1, default=str)[:6000])
    return 0


def cmd_schemas(a: argparse.Namespace) -> int:
    from src.schemas import BridgeRecord, TrainingSample
    from src.verification.models import Finding

    out = Path(__file__).parent / "docs" / "schemas"
    out.mkdir(parents=True, exist_ok=True)
    for name, model in (("bridge_record", BridgeRecord), ("training_sample", TrainingSample), ("finding", Finding)):
        (out / f"{name}.schema.json").write_text(json.dumps(model.model_json_schema(), indent=2) + "\n", encoding="utf-8")
        print(f"[schemas] {out / f'{name}.schema.json'}")
    return 0


def cmd_gui(a: argparse.Namespace) -> int:
    from src.gui.server import serve

    return serve(port=a.port, open_browser=not a.no_browser)


def cmd_pipeline(a: argparse.Namespace) -> int:
    rc = cmd_inspect(a)
    rc = rc or cmd_verify_selftest(a)
    vr = cmd_vision(a)
    summary = settings.OUTPUTS_DIR / a.source / "multimodalvisionagent_findings.json"
    if summary.exists():
        d = json.loads(summary.read_text(encoding="utf-8"))
        print(f"[pipeline] verified={d['status_counts']['verified']} rejected={d['status_counts']['rejected']} "
              f"uncorroborated={d['status_counts']['uncorroborated']}")
    return rc or vr


def main(argv: list[str] | None = None) -> int:
    for stream in (sys.stdout, sys.stderr):
        stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)  # type: ignore[attr-defined]
    ap = argparse.ArgumentParser(description="ForensiAgent-Multimodal")
    ap.add_argument("--source", "--evidence", dest="source", default="android_10", choices=_all_evidence_ids())
    sub = ap.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("cases", help="case database: list | build | catalog | audit | repair")
    pc.add_argument("action", choices=("list", "build", "catalog", "audit", "repair"))
    pc.add_argument("--no-hash", action="store_true", help="audit: existence and size only (fast)")
    pc.add_argument("--dry-run", action="store_true", help="repair: re-acquire and check, write nothing")
    pc.add_argument("--cases", default="all", help="all | comma-separated case ids")
    pc.add_argument("--groups", default="light", help="evidence groups to fetch: light,heavy,extended")
    pc.add_argument("--only", default="", help="comma-separated evidence ids (overrides --groups)")
    pc.add_argument("--docs-only", action="store_true")
    pc.set_defaults(fn=cmd_cases)
    pi = sub.add_parser("investigate", help="run the vision agent over a case (trace + memory + report)")
    pi.add_argument("--case", required=True)
    pi.add_argument("--items", default="", help="comma-separated evidence ids (default: all fetched)")
    pi.add_argument("--limit", type=int, default=0, help="max images per evidence item")
    pi.add_argument("--images", default="", help="comma-separated sandbox-relative image paths")
    pi.add_argument("--no-vlm", action="store_true")
    pi.add_argument("--no-ocr", action="store_true")
    pi.add_argument("--run-id", default="")
    pi.add_argument("--agents", default="", help="apps,fs,win,jev,mm: run these agents instead of vision")
    pi.add_argument("--no-carve", action="store_true")
    pi.set_defaults(fn=lambda a: cmd_agents(a) if a.agents else cmd_investigate(a))
    for name, helptext in (("apps", "AppArtifactAgent: SMS, calls, contacts, chats, browser history (verified rows)"),
                           ("fs", "FileSystemAgent: signature mismatches, hidden/deleted files, SQLite carving"),
                           ("win", "WindowsArtifactAgent: registry, Prefetch, LNK, EVTX, Recycle Bin, IE history")):
        pa = sub.add_parser(name, help=helptext)
        pa.add_argument("--case", required=True)
        pa.add_argument("--items", default="", help="comma-separated evidence ids (default: all fetched)")
        pa.add_argument("--limit", type=int, default=0, help="max artifacts per evidence item")
        pa.add_argument("--run-id", default="")
        pa.add_argument("--no-carve", action="store_true")
        pa.set_defaults(fn=cmd_agents, agents="")
    pco = sub.add_parser("corpus", help="run apps+fs+win over every READY case (summary in data/outputs/corpus_runs)")
    pco.add_argument("--agents", default="apps,fs,win")
    pco.add_argument("--cases", default="all")
    pco.add_argument("--limit", type=int, default=0)
    pco.add_argument("--no-carve", action="store_true")
    pco.set_defaults(fn=cmd_corpus)
    psc = sub.add_parser("score", help="score verified findings against answer keys (workspace/reports/score.md)")
    psc.add_argument("--cases", default="gold", help="gold (cases with gold questions) | all (READY) | ids")
    psc.add_argument("--no-reverify", action="store_true", help="skip the fresh re-verification audit")
    psc.set_defaults(fn=cmd_score)
    sub.add_parser("dataset", help="export the case database as ML dataset tables (data/datasets/)").set_defaults(
        fn=cmd_dataset)
    pf = sub.add_parser("finetune", help="build FDX TrainingSample JSONL from a dataset export (or --validate a file)")
    pf.add_argument("--dataset", default="", help="data/datasets/<date> (default: latest)")
    pf.add_argument("--validate", default="", help="validate a TrainingSample JSONL file and exit")
    pf.set_defaults(fn=cmd_finetune)
    pj = sub.add_parser("jev", help="Jev typed-answer model (deep learning): build | train | eval | ask")
    pj.add_argument("action", choices=("build", "train", "eval", "ask"))
    pj.add_argument("--dataset", default="", help="data/datasets/fragments/<date> (default: latest)")
    pj.add_argument("--checkpoint", default="", help="default: data/models/jev/jev.pt")
    pj.add_argument("--epochs", type=int, default=20)
    pj.add_argument("--batch", type=int, default=512)
    pj.add_argument("--lr", type=float, default=2e-3)
    pj.add_argument("--seed", type=int, default=13)
    pj.add_argument("--max-files", type=int, default=400, help="build: files per case and type")
    pj.add_argument("--file", default="", help="ask: any file to read one fragment from")
    pj.add_argument("--offset", type=int, default=512, help="ask: byte offset of the 512-byte fragment")
    pj.add_argument("--ext", default="", help="ask: extension to judge (default: the file's own)")
    pj.add_argument("--options", default="", help="ask: comma-separated subset of types for the choice query")
    pj.set_defaults(fn=cmd_jev)
    pm = sub.add_parser("mm", help="Jev-MM multimodal model (vision + metadata + OCR text): "
                                   "build | ocr | encode | prepare | train | eval | ask")
    pm.add_argument("action", choices=("build", "ocr", "encode", "prepare", "train", "eval", "ask"))
    pm.add_argument("--dataset", default="", help="data/datasets/multimodal/<date> (default: latest)")
    pm.add_argument("--checkpoint", default="", help="default: data/models/multimodal/jev_mm.pt")
    pm.add_argument("--epochs", type=int, default=12)
    pm.add_argument("--batch", type=int, default=128)
    pm.add_argument("--lr", type=float, default=1e-3)
    pm.add_argument("--seed", type=int, default=13)
    pm.add_argument("--max-files", type=int, default=400, help="build: images per case and origin")
    pm.add_argument("--no-ablations", action="store_true", help="train: full model only")
    pm.add_argument("--seeds", default="13,14,15", help="train: comma-separated seeds (mean +- std in the report)")
    pm.add_argument("--file", default="", help="ask: image file(s), comma-separated")
    pm.add_argument("--no-ocr", action="store_true", help="ask: skip OCR (text modality absent)")
    pm.set_defaults(fn=cmd_mm)
    pp = sub.add_parser("planner", help="investigation planner (deep learning): build | encode | pretrain-build | "
                                        "pretrain | train | ask")
    pp.add_argument("action", choices=("build", "encode", "pretrain-build", "pretrain", "train", "ask"))
    pp.add_argument("--dataset", default="", help="data/datasets/planner/<date> (default: latest)")
    pp.add_argument("--checkpoint", default="", help="default: the case's leave-one-out model, else planner.pt")
    pp.add_argument("--epochs", type=int, default=10)
    pp.add_argument("--seed", type=int, default=13)
    pp.add_argument("--no-ablations", action="store_true")
    pp.add_argument("--case", default="", help="ask: case to rank")
    pp.add_argument("--k", type=int, default=30, help="ask: how many files to list")
    pp.set_defaults(fn=cmd_planner)
    ps = sub.add_parser("solve", help="deep-learning investigation of a case: planner -> agents -> verifier")
    ps.add_argument("--case", required=True, help="case id (comma-separated for several)")
    ps.add_argument("--k", type=int, default=100, help="files dispatched from the planner's ranking")
    ps.add_argument("--checkpoint", default="")
    ps.add_argument("--vlm", action="store_true", help="also run the Qwen2.5-VL vision agent on dispatched images")
    ps.set_defaults(fn=cmd_solve)
    po = sub.add_parser("orchestrate", help="orchestrated multi-agent investigation (planner + LLM analyst + image "
                                           "agent, rounds with a blackboard of verified leads)")
    po.add_argument("--case", required=True, help="case id (comma-separated for several)")
    po.add_argument("--k", type=int, default=100, help="file budget")
    po.add_argument("--batch", type=int, default=25, help="files per round")
    po.add_argument("--checkpoint", default="")
    po.add_argument("--no-feedback", action="store_true", help="ablation: one round, no blackboard / re-ranking")
    po.add_argument("--no-vlm", action="store_true", help="image agent without the Qwen2.5-VL reader")
    po.add_argument("--no-tool", action="store_true", help="ablation: analyst without the identifier-extraction tool")
    po.add_argument("--ranker", choices=("planner", "lr"), default="planner",
                    help="lr = control: top K of the logistic-regression baseline, read once")
    po.set_defaults(fn=cmd_orchestrate)
    sub.add_parser("schemas", help="write FDX JSON Schemas to docs/schemas/").set_defaults(fn=cmd_schemas)
    pg = sub.add_parser("gui", help="local web interface (http://127.0.0.1:8765)")
    pg.add_argument("--port", type=int, default=8765)
    pg.add_argument("--no-browser", action="store_true")
    pg.set_defaults(fn=cmd_gui)
    sub.add_parser("config").set_defaults(fn=cmd_config)
    sub.add_parser("fetch").set_defaults(fn=cmd_fetch)
    sub.add_parser("inspect").set_defaults(fn=cmd_inspect)
    sub.add_parser("verify-selftest").set_defaults(fn=cmd_verify_selftest)
    for name, fn in (("vision", cmd_vision), ("pipeline", cmd_pipeline)):
        p = sub.add_parser(name)
        p.add_argument("--limit", type=int, default=0)
        p.add_argument("--images", default="", help="comma-separated sandbox-relative image paths")
        p.add_argument("--no-vlm", action="store_true")
        p.add_argument("--no-ocr", action="store_true")
        p.set_defaults(fn=fn)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    raise SystemExit(main())
