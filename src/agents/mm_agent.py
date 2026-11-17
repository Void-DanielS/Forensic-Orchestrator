"""ImageTriageAgent: multimodal (vision + metadata + OCR text) leads from the trained Jev-MM model.

For every image in an evidence item it asks Jev-MM (``src/ml/multimodal.py``):

* ``mm.origin``           choice: camera / screenshot / messaging / user file / web cache / thumbnail / app asset,
                          plus yes/no "user-generated?" and the attention paid to each modality
* ``mm.origin_mismatch``  the content and metadata confidently contradict where the image is stored (a camera
                          photo inside an app cache, a screenshot renamed into a system folder), or the image was
                          recovered from deleted space, where the path says nothing about its origin

Answers carry **no citations**, so the HeuristicVerificationEngine marks them UNCORROBORATED: they rank images for
an examiner (``rank()`` orders them by P(user-generated) for the vision agent) and never count as evidence.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Sequence

from src.agents.base_agent import BaseAgent
from src.ml.mm_dataset import IMAGE_CATEGORIES, IMAGE_EXTS, USER_GENERATED, origin_label
from src.verification.models import Finding

MIN_CONFIDENCE = 0.6
MISMATCH_CONFIDENCE = 0.85


class ImageTriageAgent(BaseAgent):
    name = "ImageTriageAgent"
    role = "vision"

    def __init__(self, bridge, verifier=None, manifest: dict | None = None, jev=None, checkpoint: Path | None = None,
                 min_confidence: float = MIN_CONFIDENCE, ocr: bool = True, **kw):
        super().__init__(bridge, verifier, **kw)
        self.meta = {PurePosixPath(f["path"]).relative_to("fs").as_posix(): f
                     for f in (manifest or {}).get("files", []) if f["path"].startswith("fs/")}
        self._jev, self.checkpoint = jev, checkpoint
        self.min_confidence, self.ocr = min_confidence, ocr
        self.answers: dict[str, dict] = {}

    @property
    def jev(self):
        if self._jev is None:
            from src.ml.multimodal import JevMM

            self._jev = JevMM.load(self.checkpoint, self.device)
        return self._jev

    def image_files(self) -> list[str]:
        return sorted(rel for rel, m in self.meta.items()
                      if m.get("category") in IMAGE_CATEGORIES or rel.lower().endswith(IMAGE_EXTS))

    def _name(self, rel: str) -> str:
        m = self.meta.get(rel, {})
        return m.get("original_path") or rel

    def prepare(self, artifacts: Sequence[str]) -> None:
        paths = [str(self.sandbox.resolve(rel)) for rel in artifacts]
        for rel, ans in zip(artifacts, self.jev.ask(paths, ocr=self.ocr) if paths else []):
            self.answers[rel] = ans
        self.tracer.emit("model", agent=self.name, model=self.jev.cfg.version, images=len(paths),
                         answered=sum("error" not in a for a in self.answers.values()))

    def rank(self) -> list[tuple[str, float]]:
        """Images ordered by P(user-generated), most likely first (for the vision agent's queue)."""
        return sorted(((rel, a["yes_no"]["probability"]) for rel, a in self.answers.items() if "error" not in a),
                      key=lambda t: -t[1])

    def analyze_artifact(self, rel: str) -> list[Finding]:
        a = self.answers.get(rel)
        if not a or "error" in a:
            return []
        ch, yn = a["choice"], a["yes_no"]
        if ch["abstained"] or ch["confidence"] < self.min_confidence:
            return []
        m = self.meta.get(rel, {})
        name = self._name(rel)
        value = {"sha256": m.get("sha256"), "size": m.get("size"), "original_path": name, "choice": ch, "yes_no": yn,
                 "modality_attention": a["modality_attention"], "modalities_present": a["modalities_present"],
                 "ocr_text": a.get("ocr_text", "")}
        ug = "user-generated" if yn["probability"] >= 0.5 else "not user-generated"
        out = [self.finding(rel, "mm.origin",
                            f"Image looks like {ch['answer'].replace('_', ' ')} (p={ch['confidence']:.2f}; "
                            f"{ug}, p={yn['probability']:.2f}): {name}",
                            value=value, model=ch["model"], confidence=ch["confidence"])]
        stored = origin_label(name, m.get("category") or "user_image")
        deleted = rel.startswith("_deleted/") or bool(m.get("deleted"))
        if ch["confidence"] >= MISMATCH_CONFIDENCE and (deleted or (
                ch["answer"] != stored and (ch["answer"] in USER_GENERATED) != (stored in USER_GENERATED))):
            why = ("recovered from deleted space" if deleted
                   else f"stored where {stored.replace('_', ' ')} images are kept")
            out.append(self.finding(rel, "mm.origin_mismatch",
                                    f"Content and metadata say {ch['answer'].replace('_', ' ')} "
                                    f"(p={ch['confidence']:.2f}) but the image is {why}: {name}",
                                    value={**value, "path_origin": stored, "deleted": deleted}, model=ch["model"],
                                    confidence=ch["confidence"]))
        return out

    def release(self) -> None:
        self._jev = None
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception:  # noqa: BLE001
            pass


def run_on_evidence(case, evidence_id: str, tracer=None, context=None, memory=None, run_id=None,
                    output_dir: Path | None = None, limit: int = 0, jev=None):
    from src.bridge.forensic_bridge import ForensicBridge

    ev_dir = case.dir / "evidence" / evidence_id
    manifest = json.loads((ev_dir / "MANIFEST.json").read_text(encoding="utf-8"))
    bridge = ForensicBridge(ev_dir / "fs")
    agent = None
    try:
        agent = ImageTriageAgent(bridge, manifest=manifest, jev=jev, tracer=tracer, context=context, memory=memory,
                                 evidence_id=evidence_id, run_id=run_id)
        files = agent.image_files()
        if limit:
            files = files[:limit]
        return agent.run(files, output_dir=output_dir) if files else None
    finally:
        if agent is not None:
            agent.release()
            agent.close()
        bridge.close()
