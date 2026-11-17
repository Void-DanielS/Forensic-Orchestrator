"""Dataset for the investigation planner (``src/ml/planner.py``): "where should the investigator look?"

    python main.py planner build      # files of every solved case -> labels from the answer key + model inputs
    python main.py planner encode     # sentence embeddings of content snippets and case briefs

One row per extracted file of every READY case that has a published answer key. **Label** (from the answer key,
used only as supervision, never as an input):

  relevant = the answer key names the file (its base name appears in the key text)
             OR the file's bytes contain an identifier of the solution (e-mail, URL, domain, phone, file name that
             the key gives and the brief does not; ``src/eval/scoring.key_entities``), as UTF-8 or UTF-16LE

**Inputs** (what the planner may see for a new, unsolved case):

  path       the file's path inside the evidence (text)
  metadata   size, category, platform, deleted / hidden flags, depth, MACB times as within-case percentiles
  bytes      the first 256 bytes of the file (format signature and header structure)
  content    printable strings from the first 64 KB (UTF-8 and UTF-16LE) -> sentence embedding
  context    the case brief (what examiners are told) -> sentence embedding

The ``entities`` hit list of each file is kept for the coverage@K evaluation ("how much of the answer key would an
examiner reach by reading only the top-K files?"), never as a training input.
"""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from pathlib import Path, PurePosixPath

import numpy as np

from src.bridge.forensic_bridge import long_path
from src.config.settings import DATA_DIR

HEAD = 256
SNIPPET_BYTES = 64 * 1024
SNIPPET_CHARS = 600
SCAN_CAP = 64 << 20
PATH_CHARS = 160
GENERIC_NAMES = {"desktop.ini", "thumbs.db", ".ds_store", "index.html", "readme.txt", "license.txt"}
CATEGORIES = ["app_config", "database", "app_database", "app_image", "user_document", "user_image", "system_log",
              "artifact", "prefetch", "shortcut", "browser_history", "event_log", "removable_file", "audio",
              "thumbnail_cache", "cloud_record", "registry", "email", "screenshot", "recycle_bin", "fs_metadata",
              "video", "network_capture", "memory_image", "other"]
PLATFORMS = ["windows", "android", "ios", "linux", "removable", "mixed", "iot", "takeout", "network", "other"]
META_FEATURES = (["log_size", "deleted", "hidden", "depth"] + [f"t_{k}" for k in "macb"] + [f"has_{k}" for k in "macb"]
                 + [f"cat_{c}" for c in CATEGORIES] + [f"plat_{p}" for p in PLATFORMS])
N_META = len(META_FEATURES)
_STRINGS = re.compile(rb"[\x20-\x7e]{5,}")
_STRINGS16 = re.compile(rb"(?:[\x20-\x7e]\x00){5,}")
NUL = bytes(1)


def planner_dir() -> Path:
    return DATA_DIR / "datasets" / "planner"


def _key_text(case_dir: Path) -> str:
    out = []
    kd = case_dir / "context" / "answer_key"
    for t in sorted(kd.glob("*.txt")) if kd.is_dir() else []:
        s = t.read_text(encoding="utf-8", errors="replace")
        if not s.startswith("[text extraction failed"):
            out.append(s)
    return "\n".join(out)


def brief_text(case) -> str:
    parts = [case.summary or ""]
    bd = case.dir / "context" / "brief"
    for t in sorted(bd.glob("*.txt")) if bd.is_dir() else []:
        parts.append(t.read_text(encoding="utf-8", errors="replace"))
    return "\n".join(parts)[:20000]


def _norm_key(s: str) -> str:
    return re.sub(r"\s+", " ", s.lower().replace("\\", "/"))


def entity_patterns(entities: list[dict]) -> list[tuple[str, bytes, bytes]]:
    """(entity value, utf-8 pattern, utf-16le pattern), lower-cased; URLs without scheme / www / trailing slash."""
    out, seen = [], set()
    for e in entities:
        v = e["value"].strip().lower()
        if e["type"] == "gps":
            continue
        if e["type"] == "url":
            v = re.sub(r"^[a-z]+://", "", v)
            v = re.sub(r"^www\.", "", v).rstrip("/")
        if len(v) < 5 or v in seen:
            continue
        seen.add(v)
        out.append((e["value"], v.encode("utf-8", "ignore"), v.encode("utf-16-le", "ignore")))
    return out


def name_in_key(name: str, key_norm: str) -> bool:
    base = PurePosixPath(name.replace("\\", "/")).name.lower()
    if len(base) < 5 or base in GENERIC_NAMES or not re.search(r"[a-z]", base):
        return False
    return re.search(r"(?<![a-z0-9_])" + re.escape(base) + r"(?![a-z0-9_])", key_norm) is not None


def snippet(data: bytes) -> str:
    s8 = [m.group().decode("ascii") for m in _STRINGS.finditer(data)]
    s16 = [m.group().decode("utf-16-le", "ignore") for m in _STRINGS16.finditer(data)]
    return " ".join(s8 + s16)[:SNIPPET_CHARS]


def _ts(v) -> float | None:
    if not v:
        return None
    try:
        return datetime.fromisoformat(str(v).replace("Z", "+00:00")).timestamp()
    except (ValueError, OSError, OverflowError):     # out-of-range FS times (1601 epoch on Windows)
        return None


TOKENS_FILE = Path(__file__).resolve().parents[1] / "eval" / "planner_tokens.json"


def curated_tokens() -> dict[str, list[str]]:
    """Hand-transcribed solution tokens per case (``src/eval/planner_tokens.json``); supervision only."""
    if not TOKENS_FILE.exists():
        return {}
    d = json.loads(TOKENS_FILE.read_text(encoding="utf-8"))
    return {k: v["tokens"] for k, v in d.items() if not k.startswith("_")}


def solved_cases():
    from src.cases.builder import readiness
    from src.cases.registry import CASES
    from src.eval.scoring import key_entities

    tokens = curated_tokens()
    out = []
    for c in CASES.values():
        if not readiness(c)["ready"]:
            continue
        key = _key_text(c.dir)
        if key.strip() or c.case_id in tokens:
            ents = key_entities(c.dir, c.summary) if key.strip() else []
            ents += [{"type": "token", "value": t} for t in tokens.get(c.case_id, [])]
            out.append((c, ents, key))
    return out


class Matcher:
    """All identifier patterns (UTF-8 and UTF-16LE) in one Aho-Corasick automaton: one linear pass per file."""

    def __init__(self, pats: list[tuple[str, bytes, bytes]]):
        import ahocorasick

        self.n = len(pats)
        self.auto = ahocorasick.Automaton(ahocorasick.STORE_ANY, ahocorasick.KEY_STRING)
        for value, p8, p16 in pats:
            for p in (p8, p16):
                if p:
                    self.auto.add_word(p.decode("latin-1"), value)   # bytes <-> chars 1:1
        if self.n:
            self.auto.make_automaton()

    def __len__(self) -> int:
        return self.n

    def __bool__(self) -> bool:
        return self.n > 0

    def hits(self, data: bytes) -> set[str]:
        return {v for _, v in self.auto.iter(data.decode("latin-1"))} if self.n else set()


def scan_file(path: str, size: int, pats: Matcher | None) -> tuple[bytes, str, list[str]]:
    cap = SCAN_CAP if pats else SNIPPET_BYTES          # no labels to compute: the inputs need only the first 64 KB
    with open(path, "rb") as fh:
        data = fh.read(min(size, cap) if size else cap)
    hits = sorted(pats.hits(data.lower())) if pats else []
    return data[:HEAD], snippet(data[:SNIPPET_BYTES]), hits


def case_rows(case, pats: Matcher | None = None, key_norm: str = "", workers: int = 8):
    """Model inputs (and, when ``pats`` / ``key_norm`` are given, labels) for every file of ``case``.

    Shared by the training dataset and by inference on a new case, so both see exactly the same features."""
    jobs = []
    for eid in case.evidence_ids():
        ev = case.dir / "evidence" / eid
        mp = ev / "MANIFEST.json"
        if not mp.exists():
            continue
        man = json.loads(mp.read_text(encoding="utf-8"))
        platform = man.get("platform") or (man.get("spec") or {}).get("platform") or "other"
        for f in man.get("files", []):
            jobs.append((eid, ev, f, platform))

    def work(j):
        eid, ev, f, platform = j
        p = str(long_path(Path(os.path.abspath(ev / f["path"]))))
        try:
            return scan_file(p, int(f.get("size") or 0), pats)
        except OSError:
            return None

    with ThreadPoolExecutor(workers) as ex:
        scanned = list(ex.map(work, jobs, chunksize=32))

    def macb(f, k):
        return _ts((f.get("macb") or {}).get(k) or (f.get("member_mtime") if k == "m" else None))

    times = {k: np.sort(np.array([t for (_, _, f, _) in jobs if (t := macb(f, k))])) for k in "macb"}
    rows, heads, metas, unreadable = [], [], [], 0
    for (eid, ev, f, platform), sc in zip(jobs, scanned):
        if sc is None:
            unreadable += 1
            continue
        head, snip, hits = sc
        name = f.get("original_path") or f["path"]
        rel = f["path"][3:] if f["path"].startswith("fs/") else f["path"]
        by_name = name_in_key(name, key_norm) if key_norm else False
        meta = dict.fromkeys(META_FEATURES, 0.0)
        meta["log_size"] = math.log1p(int(f.get("size") or 0)) / 20
        meta["deleted"] = float(bool(f.get("deleted")) or rel.startswith("_deleted/"))
        parts = rel.replace("\\", "/").split("/")
        meta["hidden"] = float(any(p.startswith((".", "$")) for p in parts))
        meta["depth"] = min(len(parts), 20) / 20
        for k in "macb":
            t = macb(f, k)
            if t and len(times[k]):
                meta[f"t_{k}"] = float(np.searchsorted(times[k], t) / len(times[k]))
                meta[f"has_{k}"] = 1.0
        cat = f.get("category") if f.get("category") in CATEGORIES else "other"
        meta[f"cat_{cat}"] = 1.0
        meta[f"plat_{platform if platform in PLATFORMS else 'other'}"] = 1.0
        rows.append({"case_id": case.case_id, "evidence_id": eid, "path": f["path"], "name": name[-300:],
                     "path_text": rel[-PATH_CHARS:], "category": cat, "platform": platform,
                     "size": int(f.get("size") or 0), "sha256": f.get("sha256"), "snippet": snip,
                     "by_name": int(by_name), "entities": hits, "relevant": int(by_name or bool(hits))})
        heads.append(np.frombuffer(head.ljust(HEAD, NUL), dtype=np.uint8))
        metas.append(np.array([meta[k] for k in META_FEATURES], dtype=np.float32))
    head_arr = np.stack(heads) if heads else np.zeros((0, HEAD), np.uint8)
    meta_arr = np.stack(metas) if metas else np.zeros((0, N_META), np.float32)
    return rows, head_arr, meta_arr, unreadable


def build(out: Path | None = None, progress: bool = True, min_positive: int = 1) -> Path:
    out = out or planner_dir() / datetime.now().strftime("%Y%m%d")
    out.mkdir(parents=True, exist_ok=True)
    rows, heads, metas, briefs = [], [], [], {}
    stats = Counter()
    for case, ents, key in solved_cases():
        pats = Matcher(entity_patterns(ents))
        crow, chead, cmeta, unreadable = case_rows(case, pats, _norm_key(key))
        stats["unreadable"] += unreadable
        if not crow:
            continue
        npos = sum(r["relevant"] for r in crow)
        stats[f"case:{case.case_id}"] = npos
        if progress:
            print(f"[planner build] {case.case_id}: {len(crow):,} files, {npos:,} relevant "
                  f"({sum(r['by_name'] for r in crow)} named in the key, "
                  f"{sum(bool(r['entities']) for r in crow)} contain key identifiers; {len(pats)} identifiers)",
                  flush=True)
        if npos < min_positive:          # nothing to learn from this case
            stats["dropped_cases"] += 1
            continue
        rows += crow
        heads.append(chead)
        metas.append(cmeta)
        briefs[case.case_id] = brief_text(case)
    np.savez_compressed(out / "arrays.npz", head=np.concatenate(heads), meta=np.concatenate(metas))
    with open(out / "files.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False))
            fh.write("\n")
    (out / "briefs.json").write_text(json.dumps(briefs, indent=1, ensure_ascii=False), encoding="utf-8")
    info = {"created": datetime.now().isoformat(timespec="seconds"), "n": len(rows), "cases": sorted(briefs),
            "positives": sum(r["relevant"] for r in rows), "stats": dict(stats), "meta_features": META_FEATURES}
    (out / "build.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    if progress:
        print(f"[planner build] {len(rows):,} files from {len(briefs)} solved cases, {info['positives']:,} relevant "
              f"-> {out}", flush=True)
    return out


def latest() -> Path:
    dirs = sorted(p for p in planner_dir().glob("*") if (p / "files.jsonl").exists())
    if not dirs:
        raise FileNotFoundError("no planner dataset; run `python main.py planner build` first")
    return dirs[-1]


def load(ds: Path | None = None) -> dict:
    ds = ds or latest()
    rows = [json.loads(line) for line in (ds / "files.jsonl").read_text(encoding="utf-8").splitlines() if line]
    with np.load(ds / "arrays.npz") as z:
        d = {"dir": ds, "rows": rows, "head": z["head"], "meta": z["meta"]}
    d["briefs"] = json.loads((ds / "briefs.json").read_text(encoding="utf-8"))
    if (ds / "text.npz").exists():
        with np.load(ds / "text.npz") as z:
            d["snip_emb"], d["brief_emb"] = z["snip"].astype(np.float32), z["brief"].astype(np.float32)
            d["brief_cases"] = list(z["brief_cases"])
    return d


def encode(ds: Path | None = None) -> Path:
    """Sentence embeddings of the content snippets and of each case brief (frozen multilingual MPNet)."""
    from src.ml.mm_dataset import TextEncoder

    ds = ds or latest()
    d = load(ds)
    enc = TextEncoder()
    snip = enc.encode([r["snippet"] for r in d["rows"]], batch=128)
    cases = sorted(d["briefs"])
    brief = enc.encode([d["briefs"][c][:4000] for c in cases])
    np.savez_compressed(ds / "text.npz", snip=snip.astype(np.float16), brief=brief.astype(np.float16),
                        brief_cases=np.array(cases))
    return ds / "text.npz"
