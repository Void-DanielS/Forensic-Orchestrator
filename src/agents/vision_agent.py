"""MultimodalVisionAgent (Sprint 1).

Pipeline per image artifact:

1. **EXIF / camera metadata** (deterministic): camera make/model, capture time, software,
   GPS. Each finding cites the EXIF tag (re-extracted by the verifier) and, for ASCII
   tags, the exact byte offset of the value inside the file.
2. **OCR** (EasyOCR on the agent's GPU): each text line is looked up in the case text
   index; when a database row contains it, the finding cites that row via a read-only
   SQL query and carries the OCR text as ``observed_text`` so the verifier checks the
   similarity. Lines with no ground-truth match are kept as UNCORROBORATED observations.
3. **VLM** (Qwen2.5-VL, 4-bit/8-bit via bitsandbytes): structured scene description. The
   description is an UNCORROBORATED observation; any text the VLM reports reading goes
   through the same corroboration path as OCR.

Every model output therefore lands in exactly one bucket: verified against the
extraction, rejected, or explicitly labelled as non-evidentiary context.
"""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Sequence

from src.agents.base_agent import BaseAgent
from src.bridge.forensic_bridge import ForensicBridge, find_byte_offset
from src.bridge.text_index import CaseTextIndex
from src.config import settings
from src.verification.heuristic_verifier import HeuristicVerificationEngine, normalize_text
from src.verification.models import Finding

ASCII_TAGS = {
    "0th.Make": "exif.camera_make",
    "0th.Model": "exif.camera_model",
    "0th.Software": "exif.software",
    "0th.DateTime": "exif.modified_time",
    "Exif.DateTimeOriginal": "exif.capture_time",
    "Exif.DateTimeDigitized": "exif.digitized_time",
    "Exif.OffsetTimeOriginal": "exif.capture_utc_offset",
    "0th.ImageDescription": "exif.description",
    "0th.Artist": "exif.artist",
}
GPS_TAGS = ("GPS.GPSLatitudeRef", "GPS.GPSLatitude", "GPS.GPSLongitudeRef", "GPS.GPSLongitude")

VLM_PROMPT = (
    "You are assisting a digital forensic examiner. Describe this image factually and conservatively. "
    "Do not guess identities. Reply with JSON only, using exactly these keys:\n"
    '{"image_type": "photo|screenshot|document|other", '
    '"scene": "one or two sentence description", '
    '"app_or_ui": "application or UI shown, or null", '
    '"visible_text": ["verbatim text lines you can read, at most 10"], '
    '"objects": ["notable objects"], '
    '"people_count": 0, '
    '"forensic_relevance": "why an examiner might care, or null"}'
)


class MultimodalVisionAgent(BaseAgent):
    name = "MultimodalVisionAgent"
    role = "vision"

    def __init__(self, bridge: ForensicBridge, verifier: HeuristicVerificationEngine | None = None,
                 text_index: CaseTextIndex | None = None, cfg: settings.VisionSettings | None = None, **kw):
        super().__init__(bridge, verifier, **kw)
        self.cfg = cfg or settings.VISION
        self.text_index = text_index
        self.ocr_device = settings.device_for("ocr")
        self._ocr = None
        self._vlm = None
        self._processor = None

    # ------------------------------------------------------------------ models
    def _load_ocr(self):
        if self._ocr is None:
            key = ("ocr", tuple(self.cfg.ocr_languages), self.ocr_device)
            if key not in _MODEL_CACHE:
                import easyocr

                gpu: Any = self.ocr_device if self.ocr_device.startswith("cuda") else False
                _MODEL_CACHE[key] = easyocr.Reader(self.cfg.ocr_languages, gpu=gpu, verbose=False)
            self._ocr = _MODEL_CACHE[key]
        return self._ocr

    def _load_vlm(self):
        key = ("vlm", self.cfg.vlm_model_id, self.cfg.quantization, self.device)
        if self._vlm is None and key in _MODEL_CACHE:
            self._vlm, self._processor = _MODEL_CACHE[key]
        if self._vlm is None:
            import torch
            from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig

            quant = None
            if self.cfg.quantization == "4bit":
                quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                                           bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)
            elif self.cfg.quantization == "8bit":
                quant = BitsAndBytesConfig(load_in_8bit=True)
            device_map = {"": self.device} if self.device.startswith("cuda") else "cpu"
            self._vlm = AutoModelForImageTextToText.from_pretrained(
                self.cfg.vlm_model_id,
                quantization_config=quant if self.device.startswith("cuda") else None,
                dtype=torch.bfloat16 if self.device.startswith("cuda") else torch.float32,
                device_map=device_map,
            ).eval()
            # Qwen2-VL image processor: size bounds are total pixel counts (caps visual tokens / VRAM).
            self._processor = AutoProcessor.from_pretrained(
                self.cfg.vlm_model_id,
                size={"shortest_edge": self.cfg.vlm_min_pixels, "longest_edge": self.cfg.vlm_max_pixels})
            _MODEL_CACHE[key] = (self._vlm, self._processor)
        return self._vlm, self._processor

    def release(self, keep_models: bool = False) -> None:
        """Drop model references. Models stay in the process cache (shared by the agents of one case run)
        unless ``keep_models`` is False, which frees GPU memory."""
        self._vlm = self._processor = self._ocr = None
        if keep_models:
            return
        _MODEL_CACHE.clear()
        try:
            import gc

            import torch

            gc.collect()
            torch.cuda.empty_cache()
        except ImportError:
            pass

    # ------------------------------------------------------------------ analysis
    def prepare(self, artifacts: Sequence[str]) -> None:
        self.enforce_vram_budget()
        if self.text_index is None and (self.cfg.enable_ocr or self.cfg.enable_vlm):
            self.text_index = CaseTextIndex(self.bridge).build()

    def analyze_artifact(self, rel: str) -> list[Finding]:
        import time

        print(f"  [vision] {rel}", flush=True)
        path = self.sandbox.resolve(rel)
        findings = self.exif_findings(rel)
        img, info = self._load_image(rel)
        self.tracer.emit("image", artifact=rel, bytes=path.stat().st_size, sha256=self.sandbox.hash(rel), **info)
        if self.cfg.enable_ocr:
            t0 = time.perf_counter()
            findings += self.ocr_findings(rel, img)
            self.tracer.emit("stage_done", artifact=rel, stage="ocr", seconds=round(time.perf_counter() - t0, 2))
        if self.cfg.enable_vlm:
            findings += self.vlm_findings(rel, img)
        return findings

    def _load_image(self, rel: str):
        from PIL import Image, ImageOps
        from pillow_heif import register_heif_opener

        register_heif_opener()  # iPhone photos are HEIC
        with Image.open(self.sandbox.resolve(rel)) as im:
            fmt, orig = im.format, im.size
            im = ImageOps.exif_transpose(im).convert("RGB")
        im.thumbnail((2048, 2048))
        # OCR boxes are in this display space (EXIF-rotated, longest side <= 2048); the GUI renders the same transform.
        return im, {"format": fmt, "original_size": list(orig), "display_size": list(im.size)}

    # -- 1. EXIF
    def exif_findings(self, rel: str) -> list[Finding]:
        out: list[Finding] = []
        tags = self.bridge.images.exif(rel)
        path = self.sandbox.resolve(rel)
        self.tracer.emit("exif", artifact=rel, tag_count=len(tags),
                         tags={k: v[:120] for k, v in tags.items() if k in ASCII_TAGS or k.startswith("GPS.")})
        for tag, category in ASCII_TAGS.items():
            value = tags.get(tag)
            if not value:
                continue
            cites = [self.cite(rel, "exif_tag", value, exif_tag=tag)]
            offset = find_byte_offset(path, value.encode("utf-8"))
            if offset is not None:
                cites.append(self.cite(rel, "byte_offset", value, byte_offset=offset))
            out.append(self.finding(rel, category, f"{tag.split('.', 1)[1]} = {value!r}", value=value, citations=cites))

        gps = self.bridge.images.gps(rel)
        if gps:
            cites = [self.cite(rel, "exif_tag", tags[t], exif_tag=t) for t in GPS_TAGS]
            if "GPS.GPSAltitude" in tags:
                cites.append(self.cite(rel, "exif_tag", tags["GPS.GPSAltitude"], exif_tag="GPS.GPSAltitude"))
            for t in ("GPS.GPSDateStamp", "GPS.GPSTimeStamp"):
                if t in tags:
                    cites.append(self.cite(rel, "exif_tag", tags[t], exif_tag=t))
            self.tracer.emit("gps", artifact=rel, **gps)
            out.append(self.finding(
                rel, "exif.gps",
                f"Image geotagged at {gps['latitude']:.6f}, {gps['longitude']:.6f}",
                value={**gps, "map": f"https://www.openstreetmap.org/?mlat={gps['latitude']}&mlon={gps['longitude']}"},
                citations=cites))
        return out

    # -- 2. OCR
    def ocr_findings(self, rel: str, img) -> list[Finding]:
        import numpy as np

        reader = self._load_ocr()
        results = reader.readtext(np.asarray(img), detail=1, paragraph=False)
        self.tracer.emit("ocr", artifact=rel, display_size=list(img.size), boxes=[
            {"bbox": [[int(x), int(y)] for x, y in bbox], "text": text, "conf": round(float(conf), 3),
             "kept": conf >= self.cfg.ocr_min_confidence and len(normalize_text(text)) >= 4}
            for bbox, text, conf in results])
        out: list[Finding] = []
        for bbox, text, conf in results:
            text = text.strip()
            if conf < self.cfg.ocr_min_confidence or len(normalize_text(text)) < 4:
                continue
            value = {"text": text, "bbox": [[int(x), int(y)] for x, y in bbox], "confidence": round(float(conf), 3)}
            out.append(self._text_finding(rel, text, "ocr", "easyocr", float(conf), value))
        return out

    def _text_finding(self, rel: str, text: str, source: str, model: str, conf: float | None, value: dict) -> Finding:
        """Corroborated finding if a DB row contains the text, else an uncorroborated observation."""
        hit = None
        if self.text_index is not None and len(normalize_text(text)) >= self.verifier.MIN_CORROBORATION_CHARS:
            matches = self.text_index.search(text, top_k=1)
            hit = matches[0] if matches else None
            self.tracer.emit("index_lookup", artifact=rel, source=source, text=text, hit=None if not hit else {
                "db": hit[1].db, "kind": hit[1].kind, "where": hit[1].where, "rowid": hit[1].rowid,
                "value": hit[1].value[:300], "coverage": round(hit[0], 3)})
        if hit:
            score, cell = hit
            if cell.kind == "line":
                cite = self.cite(cell.db, "line_number", cell.value, line_number=cell.rowid)
            else:
                cite = self.cite(cell.db, "sql_query", cell.value, sql_query=cell.citation_sql(), sql_params=[cell.rowid])
            return self.finding(
                rel, f"{source}.corroborated",
                f"Text read from image ({source}) matches {cell.where} in {cell.db}",
                value={**value, "ground_truth": cell.value, "similarity": round(score, 3)},
                citations=[cite], observed_text=text, model=model, confidence=conf)
        return self.finding(rel, f"{source}.uncorroborated", f"Text read from image ({source}): {text!r}",
                            value=value, observed_text=text, model=model, confidence=conf)

    # -- 3. VLM
    def vlm_findings(self, rel: str, img) -> list[Finding]:
        import torch

        model, processor = self._load_vlm()
        messages = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": VLM_PROMPT}]}]
        inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                               return_dict=True, return_tensors="pt").to(model.device)
        import time

        t0 = time.perf_counter()
        self.tracer.emit("vlm_start", artifact=rel, model=self.cfg.vlm_model_id, prompt=VLM_PROMPT,
                         visual_tokens=int(inputs["input_ids"].shape[1]))
        with torch.inference_mode():
            gen = model.generate(**inputs, max_new_tokens=self.cfg.vlm_max_new_tokens, do_sample=False)
        raw = processor.batch_decode(gen[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        parsed = _parse_json(raw)
        self.tracer.emit("vlm", artifact=rel, model=self.cfg.vlm_model_id, raw=raw, parsed=parsed,
                         seconds=round(time.perf_counter() - t0, 2))
        model_id = self.cfg.vlm_model_id
        out = [self.finding(
            rel, "vlm.description",
            f"[MODEL OBSERVATION - not evidence] {parsed.get('scene') or raw[:300]}",
            value={"parsed": parsed, "raw": raw if not parsed else None}, model=model_id)]
        for line in (parsed.get("visible_text") or [])[:10]:
            if isinstance(line, str) and line.strip() and not _PLACEHOLDER.match(line.strip()):
                out.append(self._text_finding(rel, line.strip(), "vlm_text", model_id, None, {"text": line.strip()}))
        return out


# Loaded models shared by every MultimodalVisionAgent in the process (one per evidence item in a case run).
_MODEL_CACHE: dict[tuple, Any] = {}

# VLMs fill "visible_text" with placeholders when nothing is legible; these are not readings.
_PLACEHOLDER = re.compile(r"^\W*(no|not|none|n/?a|null|unreadable|illegible)\b.*$|^\W*$", re.IGNORECASE)


def _parse_json(raw: str) -> dict[str, Any]:
    m = re.search(r"\{.*\}", raw, re.DOTALL)
    if not m:
        return {}
    try:
        data = json.loads(m.group(0))
        return data if isinstance(data, dict) else {}
    except json.JSONDecodeError:
        return {}


def discover_images(bridge: ForensicBridge, cfg: settings.VisionSettings | None = None) -> list[str]:
    cfg = cfg or settings.VISION
    return [e.path for e in bridge.fs.walk(extensions=cfg.image_extensions) if e.size > 0]


_IMAGE_CATEGORY_RANK = {"screenshot": 0, "user_image": 1, "removable_file": 1, "recycle_bin": 1, "app_image": 3}
_USER_HINT = re.compile(r"/(dcim|screenshots?|pictures|my pictures|desktop|documents|my documents|downloads|"
                        r"whatsapp|telegram|signal|sms/attachments|_deleted)/", re.I)
_BORING = re.compile(r"/(sample pictures|windows/web|program files|thumbnails?|cache|\.thumbnails)/", re.I)


def rank_images(bridge: ForensicBridge, manifest: dict | None, cfg: settings.VisionSettings | None = None) -> list[str]:
    """Images of an evidence item, most investigative first: screenshots and user photos, user folders,
    recovered deleted images, then app caches; OS sample pictures last."""
    cfg = cfg or settings.VISION
    cats = {}
    if manifest:
        for f in manifest.get("files", []):
            p = f["path"][3:] if f["path"].startswith("fs/") else f["path"]
            cats[p] = f["category"]
    imgs = discover_images(bridge, cfg)

    def key(p: str):
        return (1 if _BORING.search("/" + p) else 0, _IMAGE_CATEGORY_RANK.get(cats.get(p, ""), 2),
                0 if _USER_HINT.search("/" + p) else 1, p)
    return sorted(imgs, key=key)
