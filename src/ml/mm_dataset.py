"""Multimodal image dataset for the triage model (``src/ml/multimodal.py``).

    python main.py mm build      # images -> labels + metadata vectors + 224 px thumbnails     (CPU, parallel)
    python main.py mm ocr        # EasyOCR text + text statistics for every image              (GPU, resumable)
    python main.py mm encode     # multilingual sentence embedding of the OCR text            (GPU)

Three modalities per image, all taken from the file itself:

  vision     the pixels: a 224 x 224 letterboxed RGB thumbnail (aspect ratio kept, nothing cropped away)
  metadata   ``META_FEATURES``: container format, geometry, file size / bits per pixel, colour mode, EXIF tags
             present (make, model, GPS, capture time, software, UTC offset, exposure), JPEG quality and chroma
             subsampling, ICC profile, PNG text chunks, animation
  text       what OCR reads in the image (EasyOCR) -> a frozen multilingual sentence embedding + text statistics

Labels are deterministic and come from **where the device stored the file** (``origin_label``): camera, screenshot,
messaging media, user file, web cache, thumbnail, app asset; plus ``user_generated`` (camera, screenshot, messaging,
user file). The path is never an input, so the model must recognise the origin from content and metadata: the
situation of a carved, deleted or renamed image, where the path is gone.

Splits are per case (``src/datasets/export.split_of``); identical files (SHA-256) are kept once, test before val
before train. Small tables live in ``data/datasets/multimodal/<date>/``; the thumbnail tensor (~150 KB per image)
lives in ``settings.CACHE_DIR`` outside the synced project folder and is checked by its SHA-256 on load.
"""
from __future__ import annotations

import hashlib
import io
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np

from src.config import settings
from src.config.settings import DATA_DIR

THUMB = 224
OCR_MAX_SIDE = 1280
MAX_PER_CASE_CLASS = 400          # one photo library or one app cache must not dominate a class
SPLIT_ORDER = ("test", "val", "train")
IMAGE_CATEGORIES = ("user_image", "app_image", "screenshot", "thumbnail_cache")
IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".gif", ".bmp", ".webp", ".heic", ".heif", ".tif", ".tiff", ".jfif")

ORIGINS = ["camera", "screenshot", "messaging", "user_file", "web_cache", "thumbnail", "app_asset"]
USER_GENERATED = {"camera", "screenshot", "messaging", "user_file"}

# ------------------------------------------------------------------------------------------------ labels
_THUMB_RE = re.compile(r"thumbnail|/\.?thumbs?/|thumbcache|thumbs\.db|/\.thumbnails|/cache/thumb")
_SHOT_RE = re.compile(r"screen ?shot|screencap|screen_capture")
_WEB_RE = re.compile(r"temporary internet files|inetcache|content\.ie5|/cache2/|cache_data|webview|web_asset|"
                     r"http-?cache|/webkit/|/image_manager|/okhttp|/glide|/fresco|/picasso|imagepipeline|"
                     r"(chrome|firefox|mozilla|safari|opera|edge|browser)[^/]*/.*cache")
_MSG_RE = re.compile(r"whatsapp|telegram|org\.thoughtcrime|securesms|signal|viber|skype|kik|naver\.line|groupme|"
                     r"facebook\.orca|snapchat|discord|wechat|tencent\.mm|/sms/attachments|threema|wickr|messenger")
_MSG_ASSET_RE = re.compile(r"emoji|emo_cache|emoticon|sticker|effect|asset|resource|theme|icon|font|splash|filter|"
                           r"/res/|mask|/package/|projector|/yuki/|/anim")
_CAMERA_RE = re.compile(r"/dcim/|camera roll|/camera/|/100apple|/1\d\dapple|/media/dcim")


def origin_label(path: str, category: str) -> str:
    """Deterministic origin of an image from its storage path (first matching rule wins)."""
    p = "/" + path.replace("\\", "/").lower()
    if category == "thumbnail_cache" or _THUMB_RE.search(p):
        return "thumbnail"
    if category == "screenshot" or _SHOT_RE.search(p):
        return "screenshot"
    if _WEB_RE.search(p):
        return "web_cache"
    if _MSG_RE.search(p) and not _MSG_ASSET_RE.search(p):
        return "messaging"
    if _CAMERA_RE.search(p):
        return "camera"
    return "app_asset" if category == "app_image" else "user_file"


# ------------------------------------------------------------------------------------------------ metadata
FORMATS = ["JPEG", "PNG", "GIF", "BMP", "WEBP", "HEIF", "TIFF"]
MODES = ["RGB", "RGBA", "L", "P"]
SCREEN_WIDTHS = {320, 360, 480, 540, 640, 720, 750, 768, 800, 828, 1024, 1080, 1125, 1170, 1179, 1242, 1280, 1284,
                 1290, 1366, 1440, 1536, 1600, 1920, 2560}
META_FEATURES = (
    [f"fmt_{f.lower()}" for f in FORMATS] + ["fmt_other"]
    + ["log_width", "log_height", "log_aspect", "log_megapixels", "log_file_kb", "log_bits_per_pixel",
       "screen_width", "portrait"]
    + [f"mode_{m.lower()}" for m in MODES] + ["mode_other", "has_alpha", "animated"]
    + ["exif_present", "log_exif_tags", "exif_make", "exif_model", "exif_gps", "exif_datetime_original",
       "exif_software", "exif_offset_time", "exif_exposure", "exif_thumbnail", "exif_rotated"]
    + ["jpeg_quality", "jpeg_progressive", "jpeg_420", "jpeg_444", "jpeg_jfif", "jpeg_adobe"]
    + ["icc_profile", "log_png_text_chunks", "dpi_present"])
N_META = len(META_FEATURES)

_STD_LUM = np.array([16, 11, 10, 16, 24, 40, 51, 61, 12, 12, 14, 19, 26, 58, 60, 55, 14, 13, 16, 24, 40, 57, 69, 56,
                     14, 17, 22, 29, 51, 87, 80, 62, 18, 22, 37, 56, 68, 109, 103, 77, 24, 35, 55, 64, 81, 104, 113,
                     92, 49, 64, 78, 87, 103, 121, 120, 101, 72, 92, 95, 98, 112, 100, 103, 99], dtype=float)


def jpeg_quality(quant: dict | None) -> float:
    """libjpeg quality (1..100) implied by the luminance quantisation table; 0 when unknown."""
    if not quant or 0 not in quant:
        return 0.0
    s = float(np.mean(np.asarray(quant[0], dtype=float)) / _STD_LUM.mean() * 100)
    q = (200 - s) / 2 if s <= 100 else 5000 / s
    return float(min(max(q, 1), 100))


def _exif_tags(img) -> dict[int, object]:
    try:
        ex = img.getexif()
    except Exception:  # noqa: BLE001
        return {}
    tags = dict(ex)
    for ifd in (0x8769, 0x8825):            # Exif IFD, GPS IFD
        try:
            sub = ex.get_ifd(ifd)
            if sub:
                tags.update({(ifd << 16) | k: v for k, v in sub.items()})
                tags[ifd] = True
        except Exception:  # noqa: BLE001
            pass
    return tags


def metadata_vector(img, file_size: int) -> np.ndarray:
    """``META_FEATURES`` for one opened PIL image (no path, no file name)."""
    f = dict.fromkeys(META_FEATURES, 0.0)
    fmt = (img.format or "").upper()
    fmt = "HEIF" if fmt in ("HEIC", "HEIF", "AVIF") else fmt
    f[f"fmt_{fmt.lower()}" if fmt in FORMATS else "fmt_other"] = 1.0
    w, h = img.size
    px = max(w * h, 1)
    f.update(log_width=math.log1p(w) / 8, log_height=math.log1p(h) / 8, log_aspect=math.log(max(w, 1) / max(h, 1)),
             log_megapixels=math.log1p(px / 1e6), log_file_kb=math.log1p(file_size / 1024) / 8,
             log_bits_per_pixel=math.log1p(file_size * 8 / px), screen_width=float(min(w, h) in SCREEN_WIDTHS),
             portrait=float(h > w))
    f[f"mode_{img.mode.lower()}" if img.mode in MODES else "mode_other"] = 1.0
    f["has_alpha"] = float("A" in img.mode or "transparency" in img.info)
    f["animated"] = float(getattr(img, "n_frames", 1) > 1)
    tags = _exif_tags(img)
    if tags:
        exif = 0x8769 << 16
        gps = 0x8825
        f.update(exif_present=1.0, log_exif_tags=math.log1p(len(tags)) / 5, exif_make=float(0x010F in tags),
                 exif_model=float(0x0110 in tags), exif_gps=float(bool(tags.get(gps))),
                 exif_datetime_original=float((exif | 0x9003) in tags), exif_software=float(0x0131 in tags),
                 exif_offset_time=float((exif | 0x9011) in tags or (exif | 0x9010) in tags),
                 exif_exposure=float((exif | 0x829A) in tags or (exif | 0x829D) in tags),
                 exif_rotated=float(tags.get(0x0112, 1) not in (0, 1)))
    f["exif_thumbnail"] = float(b"\xff\xd8" in (img.info.get("exif") or b"")[8:])
    if fmt == "JPEG":
        f["jpeg_quality"] = jpeg_quality(getattr(img, "quantization", None)) / 100
        f["jpeg_progressive"] = float(bool(img.info.get("progressive") or img.info.get("progression")))
        try:
            from PIL import JpegImagePlugin

            sub = JpegImagePlugin.get_sampling(img)
        except Exception:  # noqa: BLE001
            sub = -1
        f["jpeg_420"], f["jpeg_444"] = float(sub == 2), float(sub == 0)
        f["jpeg_jfif"] = float("jfif" in img.info)
        f["jpeg_adobe"] = float("adobe" in img.info)
    f["icc_profile"] = float(bool(img.info.get("icc_profile")))
    if fmt == "PNG":
        f["log_png_text_chunks"] = math.log1p(len(getattr(img, "text", {}) or {}))
    f["dpi_present"] = float("dpi" in img.info)
    return np.array([f[k] for k in META_FEATURES], dtype=np.float32)


def letterbox(img, size: int = THUMB) -> np.ndarray:
    """RGB, longest side -> ``size``, centred on mid-grey (nothing cropped). uint8 (size, size, 3)."""
    from PIL import Image

    if img.mode in ("RGBA", "LA", "P"):
        img = img.convert("RGBA")
        bg = Image.new("RGBA", img.size, (128, 128, 128, 255))
        img = Image.alpha_composite(bg, img)
    img = img.convert("RGB")
    img.thumbnail((size, size), Image.BICUBIC)
    canvas = Image.new("RGB", (size, size), (128, 128, 128))
    canvas.paste(img, ((size - img.width) // 2, (size - img.height) // 2))
    return np.asarray(canvas, dtype=np.uint8)


def _register_heif() -> None:
    try:
        from pillow_heif import register_heif_opener

        register_heif_opener()
    except Exception:  # noqa: BLE001
        pass


def _abs(path: Path) -> str:
    from src.bridge.forensic_bridge import long_path

    return str(long_path(Path(os.path.abspath(path))))


def load_image_features(path: str, size: int) -> tuple[np.ndarray, np.ndarray] | str:
    """(metadata vector, thumbnail) or an error string. Runs in worker processes."""
    from PIL import Image

    _register_heif()
    Image.MAX_IMAGE_PIXELS = 200_000_000
    try:
        with open(path, "rb") as fh:
            data = fh.read()
        with Image.open(io.BytesIO(data)) as img:
            meta = metadata_vector(img, len(data))
            img.seek(0)
            img.draft("RGB", (THUMB * 2, THUMB * 2))     # fast JPEG decode at reduced scale
            thumb = letterbox(img)
        return meta, thumb
    except Exception as exc:  # noqa: BLE001
        return f"{type(exc).__name__}: {str(exc)[:80]}"


def _work(args):
    return load_image_features(*args)


# ------------------------------------------------------------------------------------------------ build
def mm_dir() -> Path:
    return DATA_DIR / "datasets" / "multimodal"


def _ready_cases():
    from src.cases.builder import readiness
    from src.cases.registry import CASES

    return [c for c in CASES.values() if readiness(c)["ready"]]


def collect(cases=None, seed: int = 13, max_per_class: int = MAX_PER_CASE_CLASS) -> tuple[list[dict], Counter]:
    """Choose the images (dedupe, per-case class cap) without opening them."""
    from src.datasets.export import split_of

    cases = sorted(cases if cases is not None else _ready_cases(),
                   key=lambda c: (SPLIT_ORDER.index(split_of(c.case_id)), c.case_id))
    rng = random.Random(seed)
    seen: set[str] = set()
    rows, skipped = [], Counter()
    for case in cases:
        split = split_of(case.case_id)
        cand = []
        for eid in case.evidence_ids():
            ev = case.dir / "evidence" / eid
            mp = ev / "MANIFEST.json"
            if not mp.exists():
                continue
            man = json.loads(mp.read_text(encoding="utf-8"))
            platform = man.get("platform") or (man.get("spec") or {}).get("platform") or ""
            for f in man.get("files", []):
                cat = f.get("category") or ""
                name = f.get("original_path") or f["path"]
                if cat not in IMAGE_CATEGORIES and not name.lower().endswith(IMAGE_EXTS):
                    continue
                if not f.get("sha256"):
                    continue
                cand.append((eid, ev, f, cat, name, platform))
        rng.shuffle(cand)
        per_class: Counter = Counter()
        for eid, ev, f, cat, name, platform in cand:
            if f["sha256"] in seen:
                skipped["duplicate"] += 1
                continue
            origin = origin_label(name, cat or "user_image")
            if per_class[origin] >= max_per_class:
                skipped["class_cap"] += 1
                continue
            seen.add(f["sha256"])
            per_class[origin] += 1
            rows.append({"case_id": case.case_id, "evidence_id": eid, "split": split, "platform": platform,
                         "path": f["path"], "name": name, "category": cat, "origin": origin,
                         "user_generated": int(origin in USER_GENERATED), "sha256": f["sha256"],
                         "size": int(f.get("size") or 0), "file": _abs(ev / f["path"])})
    return rows, skipped


def build(out: Path | None = None, cases=None, seed: int = 13, workers: int | None = None,
          max_per_class: int = MAX_PER_CASE_CLASS, progress: bool = True, cache_dir: Path | None = None) -> Path:
    out = out or mm_dir() / datetime.now().strftime("%Y%m%d")
    out.mkdir(parents=True, exist_ok=True)
    rows, skipped = collect(cases, seed, max_per_class)
    if progress:
        print(f"[mm build] {len(rows):,} images selected ({dict(skipped)}); decoding ...", flush=True)
    workers = workers or max(1, min(12, (os.cpu_count() or 4) - 2))
    metas, thumbs, keep = [], [], []
    errors = Counter()
    jobs = [(r["file"], THUMB) for r in rows]
    if workers > 1 and len(jobs) > 50:
        with ProcessPoolExecutor(workers) as ex:
            results = ex.map(_work, jobs, chunksize=16)
            results = list(_progress(results, len(jobs), progress))
    else:
        results = [_work(j) for j in jobs]
    for r, res in zip(rows, results):
        if isinstance(res, str):
            errors[res.split(":")[0]] += 1
            continue
        metas.append(res[0])
        thumbs.append(res[1])
        keep.append(r)
    if not keep:
        raise SystemExit("no decodable images; are any cases READY? (python main.py cases list)")
    thumbs_arr = np.stack(thumbs)
    cache = (cache_dir or settings.CACHE_DIR) / "multimodal"
    cache.mkdir(parents=True, exist_ok=True)
    tpath = cache / f"thumbs_{out.name}.npy"
    np.save(tpath, thumbs_arr)
    for r in keep:
        r.pop("file")
    np.savez_compressed(out / "meta.npz", meta=np.stack(metas))
    (out / "images.jsonl").write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in keep) + "\n",
                                      encoding="utf-8")
    info = {"created": datetime.now().isoformat(timespec="seconds"), "seed": seed, "max_per_case_class": max_per_class,
            "n": len(keep), "skipped": dict(skipped), "decode_errors": dict(errors), "thumb_size": THUMB,
            "thumbs_path": str(tpath), "thumbs_sha256": _sha_file(tpath), "meta_features": META_FEATURES,
            "origins": ORIGINS}
    (out / "build.json").write_text(json.dumps(info, indent=2), encoding="utf-8")
    write_datasheet(out)
    if progress:
        print(f"[mm build] {len(keep):,} images -> {out} (thumbnails {tpath}, decode errors {dict(errors)})",
              flush=True)
    return out


def _progress(it, n, on):
    for i, x in enumerate(it, 1):
        if on and (i % 1000 == 0 or i == n):
            print(f"[mm build] decoded {i:,}/{n:,}", flush=True)
        yield x


def _sha_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(8 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ------------------------------------------------------------------------------------------------ OCR + text
TEXT_STATS = ["has_text", "log_boxes", "mean_conf", "text_area", "log_chars", "digit_frac", "alpha_frac",
              "clock_pattern", "upper_frac"]
_CLOCK = re.compile(r"\b([01]?\d|2[0-3])[:.][0-5]\d\b")


def text_stats(boxes: list[tuple[list, str, float]], width: int, height: int) -> np.ndarray:
    """Statistics of OCR output (boxes are EasyOCR ``(quad, text, conf)``)."""
    txt = " ".join(t for _, t, _ in boxes)
    n = len(txt) or 1
    area = 0.0
    for quad, _, _ in boxes:
        xs, ys = [p[0] for p in quad], [p[1] for p in quad]
        area += (max(xs) - min(xs)) * (max(ys) - min(ys))
    return np.array([float(bool(boxes)), math.log1p(len(boxes)) / 4,
                     float(np.mean([c for *_, c in boxes])) if boxes else 0.0,
                     min(area / max(width * height, 1), 1.0), math.log1p(len(txt)) / 7,
                     sum(ch.isdigit() for ch in txt) / n, sum(ch.isalpha() for ch in txt) / n,
                     float(bool(_CLOCK.search(txt))), sum(ch.isupper() for ch in txt) / n], dtype=np.float32)


def run_ocr(ds: Path | None = None, min_conf: float = 0.3, langs=("en", "es"), progress: bool = True,
            save_every: int = 500) -> Path:
    """EasyOCR over every image of a built dataset; resumable (``ocr.jsonl`` is appended)."""
    import easyocr
    from PIL import Image

    _register_heif()
    Image.MAX_IMAGE_PIXELS = 200_000_000
    ds = ds or latest()
    rows = read_rows(ds)
    out = ds / "ocr.jsonl"
    done = set()
    if out.exists():
        done = {json.loads(line)["i"] for line in out.read_text(encoding="utf-8").splitlines() if line.strip()}
    reader = easyocr.Reader(list(langs), gpu=settings.device_for("ocr").startswith("cuda"), verbose=False)
    from src.cases.registry import CASES_DIR

    buf = []
    t0 = datetime.now()
    todo = [i for i in range(len(rows)) if i not in done]
    for k, i in enumerate(todo, 1):
        r = rows[i]
        path = _abs(CASES_DIR / r["case_id"] / "evidence" / r["evidence_id"] / r["path"])
        try:
            with Image.open(path) as img:
                img.draft("RGB", (OCR_MAX_SIDE, OCR_MAX_SIDE))
                img = img.convert("RGB")
                img.thumbnail((OCR_MAX_SIDE, OCR_MAX_SIDE))
                arr = np.asarray(img)
            res = [(q, t, float(c)) for q, t, c in reader.readtext(arr, detail=1, paragraph=False) if c >= min_conf]
            res = [([[float(x), float(y)] for x, y in q], t, c) for q, t, c in res]
            rec = {"i": i, "text": " | ".join(t for _, t, _ in res)[:1000],
                   "stats": text_stats(res, arr.shape[1], arr.shape[0]).round(4).tolist()}
        except Exception as exc:  # noqa: BLE001
            rec = {"i": i, "text": "", "stats": [0.0] * len(TEXT_STATS), "error": f"{type(exc).__name__}"}
        buf.append(rec)
        if len(buf) >= save_every or k == len(todo):
            with open(out, "a", encoding="utf-8") as fh:
                fh.write("".join(json.dumps(b, ensure_ascii=False) + "\n" for b in buf))
            buf = []
            if progress:
                rate = k / max((datetime.now() - t0).total_seconds(), 1e-6)
                print(f"[mm ocr] {len(done) + k:,}/{len(rows):,} ({rate:.1f} img/s)", flush=True)
    return out


TEXT_MODEL = "sentence-transformers/paraphrase-multilingual-mpnet-base-v2"


class TextEncoder:
    """Frozen multilingual sentence encoder (mean pooling), loaded from the local HF cache."""

    def __init__(self, device: str | None = None, model_id: str = TEXT_MODEL):
        from transformers import AutoModel, AutoTokenizer

        os.environ.setdefault("HF_HUB_OFFLINE", "1")
        self.tok = AutoTokenizer.from_pretrained(model_id)
        self.device = device or settings.device_for("embeddings")
        self.model = AutoModel.from_pretrained(model_id).to(self.device).eval()
        self.dim = self.model.config.hidden_size

    def encode(self, texts: list[str], batch: int = 64) -> np.ndarray:
        import torch

        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        idx = [i for i, t in enumerate(texts) if t.strip()]
        with torch.no_grad():
            for s in range(0, len(idx), batch):
                ii = idx[s:s + batch]
                enc = self.tok([texts[i].replace(" | ", ". ") for i in ii], padding=True, truncation=True,
                               max_length=128, return_tensors="pt").to(self.device)
                h = self.model(**enc).last_hidden_state
                m = enc["attention_mask"].unsqueeze(-1).float()
                e = (h * m).sum(1) / m.sum(1).clamp(min=1)
                out[ii] = torch.nn.functional.normalize(e, dim=-1).float().cpu().numpy()
        return out


def encode_text(ds: Path | None = None, progress: bool = True) -> Path:
    ds = ds or latest()
    n = len(read_rows(ds))
    ocr = {}
    for line in (ds / "ocr.jsonl").read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            ocr[r["i"]] = r
    missing = n - len(ocr)
    if missing:
        raise SystemExit(f"OCR incomplete ({missing} images left); run `python main.py mm ocr` first")
    texts = [ocr[i]["text"] for i in range(n)]
    stats = np.array([ocr[i]["stats"] for i in range(n)], dtype=np.float32)
    emb = TextEncoder().encode(texts)
    np.savez_compressed(ds / "text.npz", emb=emb.astype(np.float16), stats=stats)
    if progress:
        print(f"[mm encode] {int(stats[:, 0].sum()):,} of {n:,} images carry OCR text -> {ds / 'text.npz'}",
              flush=True)
    write_datasheet(ds)
    return ds / "text.npz"


# ------------------------------------------------------------------------------------------------ load
def latest() -> Path:
    dirs = sorted(p for p in mm_dir().glob("*") if (p / "images.jsonl").exists())
    if not dirs:
        raise FileNotFoundError("no multimodal dataset; run `python main.py mm build` first")
    return dirs[-1]


def read_rows(ds: Path) -> list[dict]:
    return [json.loads(line) for line in (ds / "images.jsonl").read_text(encoding="utf-8").splitlines() if line]


def load(ds: Path | None = None, check_thumbs: bool = True) -> dict:
    """Rows, metadata, thumbnails (memory-mapped) and text features of a built dataset."""
    ds = ds or latest()
    info = json.loads((ds / "build.json").read_text(encoding="utf-8"))
    tpath = Path(info["thumbs_path"])
    if not tpath.exists():
        raise FileNotFoundError(f"thumbnail cache missing ({tpath}); rebuild with `python main.py mm build`")
    if check_thumbs and _sha_file(tpath) != info["thumbs_sha256"]:
        raise ValueError(f"thumbnail cache {tpath} does not match its recorded SHA-256; rebuild the dataset")
    d = {"dir": ds, "info": info, "rows": read_rows(ds), "meta": np.load(ds / "meta.npz")["meta"],
         "thumbs": np.load(tpath, mmap_mode="r")}
    if (ds / "text.npz").exists():
        with np.load(ds / "text.npz") as z:
            d["text_emb"], d["text_stats"] = z["emb"].astype(np.float32), z["stats"]
    return d


def write_datasheet(ds: Path) -> Path:
    rows = read_rows(ds)
    info = json.loads((ds / "build.json").read_text(encoding="utf-8"))
    cnt = defaultdict(Counter)
    for r in rows:
        cnt[r["split"]][r["origin"]] += 1
    table = "\n".join(f"| {o} | " + " | ".join(f"{cnt[s][o]:,}" for s in ("train", "val", "test")) + " |"
                      for o in ORIGINS)
    ug = Counter((r["split"], r["user_generated"]) for r in rows)
    text_line = "not yet extracted (`python main.py mm ocr`, then `mm encode`)"
    if (ds / "text.npz").exists():
        with np.load(ds / "text.npz") as z:
            has = z["stats"][:, 0]
        text_line = f"{int(has.sum()):,} of {len(has):,} images ({has.mean():.1%}) carry OCR text"
    plat = Counter(r["platform"] for r in rows)
    md = f"""# Multimodal image dataset ({ds.name})

Built {info['created']} by `python main.py mm build` (seed {info['seed']}, at most {info['max_per_case_class']} images per
case and origin). Images: **{len(rows):,}** from {len({r['case_id'] for r in rows})} cases. Skipped: {info['skipped']};
undecodable: {info['decode_errors']}.

Modalities: **vision** (224 px letterboxed thumbnail, cache `{Path(info['thumbs_path']).name}`), **metadata**
({len(info['meta_features'])} features: format, geometry, EXIF, JPEG quantisation, ICC, PNG chunks), **text** (OCR):
{text_line}.

## Origin label per split (case-level splits)

| origin | train | val | test |
|---|---|---|---|
{table}

User-generated (camera / screenshot / messaging / user file): train {ug[('train', 1)]:,} of
{ug[('train', 0)] + ug[('train', 1)]:,}; val {ug[('val', 1)]:,} of {ug[('val', 0)] + ug[('val', 1)]:,}; test
{ug[('test', 1)]:,} of {ug[('test', 0)] + ug[('test', 1)]:,}.

Platforms: {dict(plat.most_common())}.

Labels come from the storage path with fixed rules (`src/ml/mm_dataset.origin_label`); the path and file name are
never model inputs.
"""
    p = ds / "DATASHEET.md"
    p.write_text(md, encoding="utf-8")
    return p
