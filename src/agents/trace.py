"""Run tracing: an append-only event log of what an agent does, read live by the GUI.

``data/cases/<case>/workspace/runs/<run_id>/``::

    run.json        run metadata and status (running | done | failed), updated at start/end
    trace.jsonl     one JSON event per line: {"seq", "ts", "kind", ...}

Event kinds (vision agent): ``run_start``, ``index_built``, ``evidence_start``, ``artifact_start``,
``exif``, ``ocr``, ``index_lookup``, ``vlm``, ``finding``, ``verification``, ``context_link``,
``memory``, ``artifact_end``, ``error``, ``evidence_end``, ``run_end``.
"""
from __future__ import annotations

import json
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def new_run_id(agent: str) -> str:
    return f"{datetime.now().strftime('%Y%m%d-%H%M%S')}-{agent.lower()}"


class Tracer:
    def __init__(self, run_dir: Path, meta: dict[str, Any]):
        self.dir = run_dir
        self.dir.mkdir(parents=True, exist_ok=True)
        self.path = self.dir / "trace.jsonl"
        self._lock = threading.Lock()
        self._seq = 0
        self._t0 = time.perf_counter()
        self.meta = {**meta, "status": "running", "started_utc": datetime.now(timezone.utc).isoformat()}
        self._write_meta()

    def _write_meta(self) -> None:
        tmp = self.dir / "run.json.tmp"
        tmp.write_text(json.dumps(self.meta, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
        tmp.replace(self.dir / "run.json")

    def emit(self, kind: str, **data: Any) -> None:
        with self._lock:
            self._seq += 1
            event = {"seq": self._seq, "ts": datetime.now(timezone.utc).isoformat(),
                     "t": round(time.perf_counter() - self._t0, 3), "kind": kind, **data}
            with open(self.path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(event, ensure_ascii=False, default=str) + "\n")

    def update(self, **meta: Any) -> None:
        with self._lock:
            self.meta.update(meta)
            self._write_meta()

    def finish(self, status: str, **summary: Any) -> None:
        self.update(status=status, finished_utc=datetime.now(timezone.utc).isoformat(), **summary)


class NullTracer:
    meta: dict = {}

    def emit(self, kind: str, **data: Any) -> None:
        pass

    def update(self, **meta: Any) -> None:
        pass

    def finish(self, status: str, **summary: Any) -> None:
        pass
