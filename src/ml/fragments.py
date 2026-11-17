"""Byte-fragment dataset for the Jev typed-answer model (``src/ml/jev.py``).

    python main.py jev build            # -> data/datasets/fragments/<YYYYMMDD>/fragments.npz + DATASHEET.md

Every READY case contributes 512-byte, sector-aligned fragments taken **after** the first sector of each file, so
the model never sees the magic number it is labelled with: it has to recognise the type from the body bytes, as a
carver must with headerless or deleted blocks. Labels are deterministic and come from the file itself:

  type        the header signature (``filesystem_agent.SIGNATURES``), else ``text`` (printable head), else ``other``
  disguised   the FileSystemAgent rule "signature contradicts the extension" (signature types only, else -1)
  position    fragment offset / (size - 512): 0 = start of the file, 1 = end

Splits are per case (``src/datasets/export.split_of``). Files are deduplicated by SHA-256 with test before val
before train, so an identical file never sits on both sides of a split.
"""
from __future__ import annotations

import json
import os
import random
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path, PurePosixPath

import numpy as np

from src.agents.filesystem_agent import SIGNATURES, TEXTUAL_EXTS, _sig
from src.bridge.forensic_bridge import long_path
from src.config.settings import DATA_DIR
from src.datasets.export import split_of

FRAG = 512
HEAD = 4096
FRAGS_PER_FILE = 4
MAX_FILES_PER_CASE_CLASS = 400   # keeps one photo library or one OS tree from dominating a class
SPLIT_ORDER = ("test", "val", "train")
# extensions that are generic containers or temp names: never evidence of disguise (same list as FileSystemAgent)
TOLERATED_EXTS = TEXTUAL_EXTS + (".tmp", ".bak", ".old", ".cache", ".bin", ".dat", ".data", ".blob", ".chunk",
                                 ".part", ".crdownload", ".0", ".1")
ALLOWED_EXTS = {kind: exts for kind, _, _, exts in SIGNATURES}


def fragments_dir() -> Path:
    return DATA_DIR / "datasets" / "fragments"


def is_text(data: bytes) -> bool:
    """Printable ASCII / UTF-8 with no NUL bytes (deterministic)."""
    if not data or b"\x00" in data:
        return False
    try:
        s = data.decode("utf-8")
    except UnicodeDecodeError as exc:
        if exc.start < len(data) - 4:   # only a multi-byte character cut at the end is tolerated
            return False
        s = data[:exc.start].decode("utf-8")
    ok = sum(ch.isprintable() or ch in "\t\r\n" for ch in s)
    return ok >= 0.97 * max(len(s), 1)


def type_label(head: bytes) -> str:
    return _sig(head) or ("text" if is_text(head) else "other")


def extension_of(name: str) -> str:
    return PurePosixPath(name.replace("\\", "/")).suffix.lower()


def is_disguised(kind: str, name: str) -> int:
    """1 if the extension contradicts a signature type, 0 if not, -1 when undefined (text / other)."""
    if kind not in ALLOWED_EXTS:
        return -1
    ext = extension_of(name)
    base = PurePosixPath(name.replace("\\", "/")).name
    if not ext or len(ext) > 6 or base.endswith(("-wal", "-shm", "-journal")) or ext in TOLERATED_EXTS:
        return 0
    return int(ext not in ALLOWED_EXTS[kind])


def is_blank(block: bytes) -> bool:
    """A block of one repeated byte (zero fill, sparse region) says nothing about the file type."""
    return len(set(block)) <= 1


def fragment_offsets(size: int, n: int, rng: random.Random) -> list[int]:
    """Up to ``n`` distinct sector-aligned offsets, all past the first sector."""
    sectors = (size - FRAG) // FRAG          # sectors 1..sectors start a full fragment
    if sectors < 1:
        return []
    picks = rng.sample(range(1, sectors + 1), min(n, sectors))
    return sorted(p * FRAG for p in picks)


def _manifest_files(case):
    for eid in case.evidence_ids():
        ev = case.dir / "evidence" / eid
        mp = ev / "MANIFEST.json"
        if not mp.exists():
            continue
        for f in json.loads(mp.read_text(encoding="utf-8")).get("files", []):
            if (f.get("size") or 0) >= 2 * FRAG and f.get("sha256"):
                yield eid, ev, f


def build(out: Path | None = None, cases=None, seed: int = 13, frags_per_file: int = FRAGS_PER_FILE,
          max_files: int = MAX_FILES_PER_CASE_CLASS, progress: bool = True) -> Path:
    """Extract fragments from every READY case (or ``cases``) into one ``fragments.npz``."""
    if cases is None:
        from src.cases.builder import readiness
        from src.cases.registry import CASES

        cases = [c for c in CASES.values() if readiness(c)["ready"]]
    out = out or fragments_dir() / datetime.now().strftime("%Y%m%d")
    out.mkdir(parents=True, exist_ok=True)
    rng = random.Random(seed)
    cases = sorted(cases, key=lambda c: (SPLIT_ORDER.index(split_of(c.case_id)), c.case_id))
    seen: set[str] = set()
    cols: dict[str, list] = defaultdict(list)
    stats: dict = {"files_by_split": Counter(), "skipped": Counter(), "files_by_class": defaultdict(Counter)}
    for ci, case in enumerate(cases, 1):
        split = split_of(case.case_id)
        per_class: Counter = Counter()
        n0 = len(cols["label"])
        files = list(_manifest_files(case))
        rng.shuffle(files)          # the per-class cap must not keep only the first directories
        for eid, ev, f in files:
            if f["sha256"] in seen:
                stats["skipped"]["duplicate"] += 1
                continue
            # extended-length form: evidence paths over 260 characters were silently skipped as "unreadable"
            path = long_path(Path(os.path.abspath(ev / f["path"])))
            name = f.get("original_path") or f.get("source_member") or f["path"]
            try:
                with open(path, "rb") as fh:
                    head = fh.read(HEAD)
                    kind = type_label(head)
                    if per_class[kind] >= max_files:
                        stats["skipped"]["class_cap"] += 1
                        continue
                    size = f["size"]
                    blocks = []
                    for off in fragment_offsets(size, frags_per_file, rng):
                        fh.seek(off)
                        b = fh.read(FRAG)
                        if len(b) == FRAG and not is_blank(b):
                            blocks.append((off, b))
            except OSError:
                stats["skipped"]["unreadable"] += 1
                continue
            if not blocks:
                stats["skipped"]["no_informative_block"] += 1
                continue
            seen.add(f["sha256"])
            per_class[kind] += 1
            stats["files_by_split"][split] += 1
            stats["files_by_class"][split][kind] += 1
            disg = is_disguised(kind, name)
            for off, b in blocks:
                cols["x"].append(np.frombuffer(b, dtype=np.uint8))
                cols["label"].append(kind)
                cols["disguised"].append(disg)
                cols["position"].append(off / max(size - FRAG, 1))
                cols["ext"].append(extension_of(name)[:9])
                cols["split"].append(split)
                cols["case_id"].append(case.case_id)
                cols["evidence_id"].append(eid)
                cols["path"].append(f["path"])
                cols["sha256"].append(f["sha256"])
                cols["offset"].append(off)
        if progress:
            print(f"[jev build] {ci}/{len(cases)} {case.case_id} ({split}): {len(cols['label']) - n0} fragments",
                  flush=True)
    if not cols["label"]:
        raise SystemExit("no fragments extracted: are any cases READY? (python main.py cases list)")
    np.savez_compressed(
        out / "fragments.npz", x=np.stack(cols["x"]), label=np.array(cols["label"]),
        disguised=np.array(cols["disguised"], dtype=np.int8), position=np.array(cols["position"], dtype=np.float32),
        ext=np.array(cols["ext"]), split=np.array(cols["split"]), case_id=np.array(cols["case_id"]),
        evidence_id=np.array(cols["evidence_id"]), path=np.array(cols["path"]), sha256=np.array(cols["sha256"]),
        offset=np.array(cols["offset"], dtype=np.int64))
    _datasheet(out, cols, stats, seed, frags_per_file, max_files)
    return out


def _datasheet(out: Path, cols, stats, seed, frags_per_file, max_files) -> None:
    frag = defaultdict(Counter)
    for lab, sp in zip(cols["label"], cols["split"]):
        frag[sp][lab] += 1
    labels = sorted({*cols["label"]}, key=lambda k: -sum(frag[s][k] for s in SPLIT_ORDER))
    rows = "\n".join(f"| {k} | " + " | ".join(f"{stats['files_by_class'][s][k]:,} / {frag[s][k]:,}"
                                               for s in ("train", "val", "test")) + " |" for k in labels)
    nat = Counter((sp, d) for sp, d in zip(cols["split"], cols["disguised"]) if d >= 0)
    md = f"""# Fragment dataset ({out.name})

Generated {datetime.now().isoformat(timespec='seconds')} by `python main.py jev build` (seed {seed}, up to
{frags_per_file} fragments of {FRAG} B per file, at most {max_files} files per case and class, first sector never used).

Cases: {len({*cols['case_id']})} · fragments: {len(cols['label']):,} · files: {sum(stats['files_by_split'].values()):,}
· skipped: {dict(stats['skipped'])}

## Files / fragments per type and split

| type | train | val | test |
|---|---|---|---|
{rows}

## Natural "disguised" labels (signature types only)

{ {f'{sp}:{"disguised" if d else "consistent"}': n for (sp, d), n in sorted(nat.items())} }

Labels are deterministic (header signature, printable test, FileSystemAgent extension rule). The model is trained on
body fragments only, so it never sees the bytes its type label was derived from.
"""
    (out / "DATASHEET.md").write_text(md, encoding="utf-8")


def latest() -> Path:
    dirs = sorted(p for p in fragments_dir().glob("*") if (p / "fragments.npz").exists())
    if not dirs:
        raise FileNotFoundError("no fragment dataset; run `python main.py jev build` first")
    return dirs[-1]


def load(path: Path | None = None) -> dict[str, np.ndarray]:
    path = path or latest()
    if path.is_dir():
        path = path / "fragments.npz"
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
