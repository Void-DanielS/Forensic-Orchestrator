"""ForensiAgent-Multimodal CLI (Sprint 1).

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
    """apps / fs / investigate --agents apps,fs: deterministic agents over one case."""
    from src.pipeline.investigate import investigate_agents

    agents = a.agents.split(",") if getattr(a, "agents", "") else [a.cmd]
    res = investigate_agents(a.case, agents, evidence=[x for x in a.items.split(",") if x] or None, limit=a.limit,
                             carve=not getattr(a, "no_carve", False), run_id=a.run_id or None)
    print(f"[{'+'.join(agents)}] run {res['run_id']} totals={res['totals']}")
    print(f"[report] {res['report']}")
    return 0 if not res["totals"]["rejected"] else 2


def cmd_corpus(a: argparse.Namespace) -> int:
    """Run the AppArtifactAgent + FileSystemAgent over every READY case; summary in data/outputs/corpus_runs/."""
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
    ap = argparse.ArgumentParser(description="ForensiAgent-Multimodal (Sprint 1)")
    ap.add_argument("--source", "--evidence", dest="source", default="android_10", choices=_all_evidence_ids())
    sub = ap.add_subparsers(dest="cmd", required=True)
    pc = sub.add_parser("cases", help="case database: list | build | catalog")
    pc.add_argument("action", choices=("list", "build", "catalog"))
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
    pi.add_argument("--agents", default="", help="apps,fs: run the deterministic agents instead of vision")
    pi.add_argument("--no-carve", action="store_true")
    pi.set_defaults(fn=lambda a: cmd_agents(a) if a.agents else cmd_investigate(a))
    for name, helptext in (("apps", "AppArtifactAgent: SMS, calls, contacts, chats, browser history (verified rows)"),
                           ("fs", "FileSystemAgent: signature mismatches, hidden/deleted files, SQLite carving")):
        pa = sub.add_parser(name, help=helptext)
        pa.add_argument("--case", required=True)
        pa.add_argument("--items", default="", help="comma-separated evidence ids (default: all fetched)")
        pa.add_argument("--limit", type=int, default=0, help="max artifacts per evidence item")
        pa.add_argument("--run-id", default="")
        pa.add_argument("--no-carve", action="store_true")
        pa.set_defaults(fn=cmd_agents, agents="")
    pco = sub.add_parser("corpus", help="run apps+fs over every READY case (summary in data/outputs/corpus_runs)")
    pco.add_argument("--agents", default="apps,fs")
    pco.add_argument("--cases", default="all")
    pco.add_argument("--limit", type=int, default=0)
    pco.add_argument("--no-carve", action="store_true")
    pco.set_defaults(fn=cmd_corpus)
    sub.add_parser("dataset", help="export the case database as ML dataset tables (data/datasets/)").set_defaults(
        fn=cmd_dataset)
    pf = sub.add_parser("finetune", help="build FDX TrainingSample JSONL from a dataset export (or --validate a file)")
    pf.add_argument("--dataset", default="", help="data/datasets/<date> (default: latest)")
    pf.add_argument("--validate", default="", help="validate a TrainingSample JSONL file and exit")
    pf.set_defaults(fn=cmd_finetune)
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
