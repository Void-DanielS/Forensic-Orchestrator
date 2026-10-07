"""Local web console: watch agents work on a case, inspect memory, findings, reports, documents.

    python main.py gui            # http://127.0.0.1:8765

Standard library only. Binds to 127.0.0.1. Read-only on evidence: images are served through
the case sandbox (path-confined) and re-rendered exactly as the agent saw them (EXIF-rotated,
longest side 2048 px) so OCR boxes overlay correctly. Answer-key documents are listed but their
text is only returned on an explicit ``reveal=1`` request from the examiner.
"""
from __future__ import annotations

import io
import json
import re
import subprocess
import sys
import threading
import webbrowser
from collections import Counter, OrderedDict
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

from src.bridge.forensic_bridge import Sandbox, SandboxViolation
from src.cases.registry import CASES
from src.config.settings import PROJECT_ROOT

STATIC = Path(__file__).parent / "static"
_ID = re.compile(r"^[\w.\-]{1,120}$")
_JOBS: dict[str, dict] = {}
_JOBS_LOCK = threading.Lock()
_IMG_CACHE: OrderedDict[tuple, bytes] = OrderedDict()


class ApiError(Exception):
    def __init__(self, code: int, msg: str):
        super().__init__(msg)
        self.code = code


def _case(cid: str):
    if cid not in CASES:
        raise ApiError(404, f"unknown case {cid}")
    return CASES[cid]


def _safe_id(v: str, what: str) -> str:
    if not v or not _ID.match(v):
        raise ApiError(400, f"bad {what}")
    return v


def _read_json(p: Path, default=None):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


# --------------------------------------------------------------------------------------
# API
# --------------------------------------------------------------------------------------
def api_cases(_: dict) -> dict:
    out = []
    for c in CASES.values():
        cj = _read_json(c.dir / "case.json", {}) or {}
        ev = cj.get("evidence") or []
        fetched = sum(1 for e in c.evidence_ids() if (c.dir / "evidence" / e / "MANIFEST.json").exists())
        runs = sorted((c.dir / "workspace" / "runs").glob("*/run.json")) if (c.dir / "workspace" / "runs").is_dir() else []
        out.append({"case_id": c.case_id, "title": c.title, "kind": c.kind, "year": c.year, "tags": list(c.tags),
                    "evidence_total": len(c.evidence), "evidence_fetched": fetched, "runs": len(runs),
                    "platforms": sorted({e.get("platform") for e in ev if e.get("platform")})})
    return {"cases": out}


def api_case(q: dict) -> dict:
    c = _case(q.get("id", ""))
    from src.cases.builder import evidence_status

    ev = evidence_status(c)
    for e in ev:
        man = _read_json(c.dir / "evidence" / e["evidence_id"] / "MANIFEST.json")
        if man:
            e["files"] = len(man["files"])
            e["categories"] = dict(Counter(f["category"] for f in man["files"]).most_common())
            e["bytes"] = sum(f["size"] for f in man["files"])
            e["deleted_recovered"] = sum(1 for f in man["files"] if f.get("deleted"))
            img = man.get("image") or {}
            e["acquisition"] = {"stored_hashes": img.get("stored_hashes"), "headers": img.get("headers"),
                                "media_size": img.get("media_size")} if img else None
            e["filesystems"] = man.get("filesystems")
    docs = {}
    for role in ("brief", "answer_key", "acquisition"):
        d = c.dir / "context" / role
        docs[role] = sorted(f.name for f in d.iterdir() if not f.name.endswith(".txt")
                            or not f.with_name(f.name[:-4]).exists()) if d.is_dir() else []
    runs = []
    rdir = c.dir / "workspace" / "runs"
    if rdir.is_dir():
        for r in sorted(rdir.iterdir(), reverse=True):
            meta = _read_json(r / "run.json")
            if meta:
                runs.append(meta)
    reports = sorted((p.name for p in (c.dir / "workspace" / "reports").glob("*.md")), reverse=True) \
        if (c.dir / "workspace" / "reports").is_dir() else []
    return {"case_id": c.case_id, "title": c.title, "summary": c.summary, "kind": c.kind, "year": c.year,
            "origin": c.origin, "tags": list(c.tags), "evidence": ev, "docs": docs, "runs": runs, "reports": reports}


def api_trace(q: dict) -> dict:
    c = _case(q.get("case", ""))
    run = _safe_id(q.get("run", ""), "run")
    since = int(q.get("since", "0") or 0)
    rdir = c.dir / "workspace" / "runs" / run
    events = []
    p = rdir / "trace.jsonl"
    if p.exists():
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                try:
                    ev = json.loads(line)
                except ValueError:
                    continue   # partially written last line
                if ev.get("seq", 0) > since:
                    events.append(ev)
    return {"run": _read_json(rdir / "run.json", {}), "events": events}


def api_findings(q: dict) -> dict:
    c = _case(q.get("case", ""))
    run = _safe_id(q.get("run", ""), "run")
    out = []
    for p in sorted((c.dir / "workspace" / "runs" / run).glob("*_findings.json")):
        d = _read_json(p, {})
        for key in ("verified_findings", "rejected_findings", "model_observations_uncorroborated"):
            out += d.get(key, [])
    return {"findings": out}


def api_memory(q: dict) -> dict:
    c = _case(q.get("id", ""))
    mem = _read_json(c.dir / "workspace" / "memory.json")
    if mem is None:
        from src.context.case_context import CaseContext

        mem = {**CaseContext.load(c.dir, c.summary).to_dict(), "links": [], "runs": [], "summary": {}}
    return mem


def api_report(q: dict) -> dict:
    c = _case(q.get("case", ""))
    name = _safe_id(q.get("name", ""), "report")
    p = c.dir / "workspace" / "reports" / name
    if not p.is_file():
        raise ApiError(404, "no such report")
    return {"name": name, "markdown": p.read_text(encoding="utf-8")}


def api_doc(q: dict) -> dict:
    c = _case(q.get("case", ""))
    role = q.get("role", "")
    if role not in ("brief", "answer_key", "acquisition"):
        raise ApiError(400, "bad role")
    if role == "answer_key" and q.get("reveal") != "1":
        raise ApiError(403, "answer key is hidden from agents; reveal explicitly")
    d = c.dir / "context" / role
    name = q.get("name", "")
    if "/" in name or "\\" in name or name.startswith("."):
        raise ApiError(400, "bad name")
    for cand in (d / (name + ".txt"), d / name):
        if cand.is_file() and cand.suffix.lower() in (".txt", ".md", ".log", ".csv"):
            return {"name": name, "role": role, "text": cand.read_text(encoding="utf-8", errors="replace")[:400_000]}
    raise ApiError(404, "no text for document")


def api_images(q: dict) -> dict:
    """Ranked images of one evidence item (what the agent would pick first)."""
    c = _case(q.get("case", ""))
    eid = _safe_id(q.get("evidence", ""), "evidence")
    if eid not in c.evidence_ids():
        raise ApiError(404, "evidence not in case")
    from src.agents.vision_agent import rank_images
    from src.bridge.forensic_bridge import ForensicBridge

    ev = c.dir / "evidence" / eid
    man = _read_json(ev / "MANIFEST.json")
    if not man:
        raise ApiError(404, "evidence not fetched")
    b = ForensicBridge(ev / "fs")
    try:
        imgs = rank_images(b, man)
    finally:
        b.close()
    return {"images": imgs[: int(q.get("limit", "200"))], "total": len(imgs)}


def image_bytes(q: dict) -> bytes:
    c = _case(q.get("case", ""))
    eid = _safe_id(q.get("evidence", ""), "evidence")
    if eid not in c.evidence_ids():
        raise ApiError(404, "evidence not in case")
    rel = q.get("path", "")
    key = (c.case_id, eid, rel, q.get("thumb", ""))
    if key in _IMG_CACHE:
        _IMG_CACHE.move_to_end(key)
        return _IMG_CACHE[key]
    try:
        path = Sandbox(c.dir / "evidence" / eid / "fs").resolve(rel)
    except SandboxViolation as exc:
        raise ApiError(403, str(exc)) from exc
    if not path.is_file():
        raise ApiError(404, "no such image")
    from PIL import Image, ImageOps
    from pillow_heif import register_heif_opener

    register_heif_opener()
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im).convert("RGB")
    im.thumbnail((360, 360) if q.get("thumb") else (2048, 2048))   # same transform as the agent
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=88)
    data = buf.getvalue()
    _IMG_CACHE[key] = data
    while len(_IMG_CACHE) > 300:
        _IMG_CACHE.popitem(last=False)
    return data


def api_jobs(_: dict) -> dict:
    with _JOBS_LOCK:
        for j in _JOBS.values():
            if j["status"] == "running" and j["proc"].poll() is not None:
                j["status"] = "done" if j["proc"].returncode == 0 or j["proc"].returncode == 2 else "failed"
                j["returncode"] = j["proc"].returncode
        return {"jobs": [{k: v for k, v in j.items() if k != "proc"} for j in _JOBS.values()]}


def api_investigate(body: dict) -> dict:
    c = _case(body.get("case", ""))
    items = [e for e in body.get("evidence", []) if e in c.evidence_ids()]
    limit = max(0, min(int(body.get("limit", 6) or 0), 2000))
    if any(j["status"] == "running" for j in api_jobs({})["jobs"]):
        raise ApiError(409, "an agent run is already in progress (one GPU job at a time)")
    from src.agents.trace import new_run_id

    run_id = new_run_id("vision")
    cmd = [sys.executable, str(PROJECT_ROOT / "main.py"), "investigate", "--case", c.case_id, "--limit", str(limit),
           "--run-id", run_id]
    if items:
        cmd += ["--items", ",".join(items)]
    if not body.get("ocr", True):
        cmd.append("--no-ocr")
    if not body.get("vlm", True):
        cmd.append("--no-vlm")
    log = c.dir / "workspace" / "runs" / f"{run_id}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    env = {**__import__("os").environ, "HF_HUB_OFFLINE": "1", "PYTHONIOENCODING": "utf-8"}
    proc = subprocess.Popen(cmd, cwd=PROJECT_ROOT, stdout=open(log, "w", encoding="utf-8"), stderr=subprocess.STDOUT,
                            env=env)
    with _JOBS_LOCK:
        _JOBS[run_id] = {"run_id": run_id, "case_id": c.case_id, "status": "running", "proc": proc, "cmd": cmd[2:]}
    return {"run_id": run_id}


ROUTES = {"/api/cases": api_cases, "/api/case": api_case, "/api/trace": api_trace, "/api/findings": api_findings,
          "/api/memory": api_memory, "/api/report": api_report, "/api/doc": api_doc, "/api/images": api_images,
          "/api/jobs": api_jobs}


class Handler(BaseHTTPRequestHandler):
    server_version = "ForensiAgentConsole/1.0"

    def log_message(self, fmt, *args):  # quiet
        pass

    def _send(self, code: int, body: bytes, ctype: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code: int, obj) -> None:
        self._send(code, json.dumps(obj, ensure_ascii=False, default=str).encode("utf-8"), "application/json; charset=utf-8")

    def do_GET(self):  # noqa: N802
        u = urlparse(self.path)
        q = {k: v[0] for k, v in parse_qs(u.query).items()}
        try:
            if u.path in ("/", "/index.html"):
                return self._send(200, (STATIC / "index.html").read_bytes(), "text/html; charset=utf-8")
            if u.path == "/api/image":
                return self._send(200, image_bytes(q), "image/jpeg")
            if u.path in ROUTES:
                return self._json(200, ROUTES[u.path](q))
            raise ApiError(404, "not found")
        except ApiError as exc:
            self._json(exc.code, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def do_POST(self):  # noqa: N802
        u = urlparse(self.path)
        try:
            if self.headers.get("Origin") not in (None, f"http://{self.headers.get('Host')}"):
                raise ApiError(403, "cross-origin request refused")
            n = int(self.headers.get("Content-Length", "0") or 0)
            body = json.loads(self.rfile.read(min(n, 1 << 16)) or b"{}")
            if u.path == "/api/investigate":
                return self._json(200, api_investigate(body))
            raise ApiError(404, "not found")
        except ApiError as exc:
            self._json(exc.code, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self._json(500, {"error": f"{type(exc).__name__}: {exc}"})


def serve(port: int = 8765, open_browser: bool = True) -> int:
    httpd = ThreadingHTTPServer(("127.0.0.1", port), Handler)
    url = f"http://127.0.0.1:{port}/"
    print(f"[gui] ForensiAgent console at {url}  (Ctrl+C to stop)", flush=True)
    if open_browser:
        threading.Timer(0.8, lambda: webbrowser.open(url)).start()
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    return 0
