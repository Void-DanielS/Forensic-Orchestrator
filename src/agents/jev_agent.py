"""JevAgent: typed answers from the trained Jev model (``src/ml/jev.py``) as investigative leads.

For every file without a recognisable header signature (wiped / overwritten headers, unknown containers,
recovered deleted files) it samples body fragments and asks the model:

* ``jev.content_type``  choice: which file type the body bytes look like (only signature types, i.e. not
                        ``text`` / ``other``, and only above the confidence threshold)
* ``jev.disguised``     yes/no: whether that content contradicts the file's extension

Answers are model observations with **no citations**, so the HeuristicVerificationEngine marks them
UNCORROBORATED: they rank files for an examiner and never count as evidence. Inference for all files is batched
in ``prepare()``; ``analyze_artifact()`` only turns the cached answers into findings.
"""
from __future__ import annotations

import json
import random
from pathlib import Path, PurePosixPath
from typing import Sequence

import numpy as np

from src.agents.base_agent import BaseAgent
from src.agents.filesystem_agent import _sig
from src.ml import fragments as fr
from src.verification.models import Finding

MIN_CONFIDENCE = 0.8


class JevAgent(BaseAgent):
    name = "JevAgent"
    role = "text"

    def __init__(self, bridge, verifier=None, manifest: dict | None = None, jev=None, checkpoint: Path | None = None,
                 min_confidence: float = MIN_CONFIDENCE, **kw):
        super().__init__(bridge, verifier, **kw)
        self.meta = {PurePosixPath(f["path"]).relative_to("fs").as_posix(): f
                     for f in (manifest or {}).get("files", []) if f["path"].startswith("fs/")}
        self._jev, self.checkpoint = jev, checkpoint
        self.min_confidence = min_confidence
        self.answers: dict[str, dict] = {}

    @property
    def jev(self):
        if self._jev is None:
            from src.ml.jev import Jev

            self._jev = Jev.load(self.checkpoint, self.device)
        return self._jev

    def _name(self, rel: str) -> str:
        m = self.meta.get(rel, {})
        return m.get("original_path") or m.get("source_member") or rel

    def prepare(self, artifacts: Sequence[str]) -> None:
        rng = random.Random(0)
        frags, owner, exts = [], [], []
        for rel in artifacts:
            path = self.sandbox.resolve(rel)
            size = path.stat().st_size
            if size < 2 * fr.FRAG:
                continue
            with open(path, "rb") as fh:
                if _sig(fh.read(64)):
                    continue        # a header signature is a deterministic answer: the FileSystemAgent has it
                for off in fr.fragment_offsets(size, fr.FRAGS_PER_FILE, rng):
                    fh.seek(off)
                    b = fh.read(fr.FRAG)
                    if len(b) == fr.FRAG and not fr.is_blank(b):
                        frags.append(np.frombuffer(b, dtype=np.uint8))
                        owner.append((rel, off))
                        exts.append(fr.extension_of(self._name(rel)))
        if not frags:
            return
        p = self.jev.probabilities(np.stack(frags), exts)
        by_file: dict[str, list[int]] = {}
        for i, (rel, _) in enumerate(owner):
            by_file.setdefault(rel, []).append(i)
        for rel, idx in by_file.items():
            self.answers[rel] = {"offsets": [owner[i][1] for i in idx],
                                 "choice": p["choice"][idx].mean(0), "yes_no": float(p["yes_no"][idx].mean())}
        self.tracer.emit("model", agent=self.name, model=self.jev.cfg.version, fragments=len(frags),
                         files=len(by_file))

    def analyze_artifact(self, rel: str) -> list[Finding]:
        a = self.answers.get(rel)
        if a is None:
            return []
        ch = self.jev.choice_answer(a["choice"])
        if ch["abstained"] or ch["answer"] in ("text", "other") or ch["confidence"] < self.min_confidence:
            return []
        m = self.meta.get(rel, {})
        name = self._name(rel)
        value = {"sha256": m.get("sha256"), "size": m.get("size"), "original_path": name,
                 "fragment_offsets": a["offsets"], "fragment_bytes": fr.FRAG, "choice": ch}
        out = [self.finding(rel, "jev.content_type",
                            f"Headerless content looks like {ch['answer']} (p={ch['confidence']:.2f}): {name}",
                            value=value, model=ch["model"], confidence=ch["confidence"])]
        yn = self.jev.yes_no_answer(a["yes_no"])
        if yn["answer"] and yn["confidence"] >= self.min_confidence:
            ext = fr.extension_of(name)
            out.append(self.finding(rel, "jev.disguised",
                                    f"Content ({ch['answer']}) likely contradicts the extension '{ext or '(none)'}' "
                                    f"(p={yn['probability']:.2f}): {name}",
                                    value={**value, "extension": ext, "yes_no": yn}, model=yn["model"],
                                    confidence=yn["confidence"]))
        return out


def run_on_evidence(case, evidence_id: str, tracer=None, context=None, memory=None, run_id=None,
                    output_dir: Path | None = None, limit: int = 0, jev=None):
    from src.bridge.forensic_bridge import ForensicBridge

    ev_dir = case.dir / "evidence" / evidence_id
    manifest = json.loads((ev_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    bridge = ForensicBridge(ev_dir / "fs")
    agent = None
    try:
        agent = JevAgent(bridge, manifest=manifest, jev=jev, tracer=tracer, context=context, memory=memory,
                         evidence_id=evidence_id, run_id=run_id)
        files = sorted(agent.meta)
        if limit:
            files = files[:limit]
        return agent.run(files, output_dir=output_dir) if files else None
    finally:
        if agent is not None:
            agent.close()
        bridge.close()
