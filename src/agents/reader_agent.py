"""ReaderAgent: reads the non-image files the planner sends it and reports the identifiers they contain.

It is the planner's executor for everything that is not an image: for each file (in the planner's order) it finds
e-mail addresses and URLs in the raw bytes, in UTF-8 / ASCII and in UTF-16LE (Windows strings), and reports each
distinct one as a finding cited by ``byte_offset`` + the exact bytes in hex. The HeuristicVerificationEngine re-reads
those bytes from the evidence file, so every reported identifier is VERIFIED or REJECTED: this is the safeguard that
keeps the deep-learning pipeline free of hallucinations.
"""
from __future__ import annotations

import re
from typing import Sequence

from src.agents.base_agent import BaseAgent
from src.verification.models import Finding

SCAN_CAP = 64 << 20
MAX_PER_FILE = 60
EMAIL = re.compile(rb"[A-Za-z0-9._%+-]{1,64}@[A-Za-z0-9-]{1,63}(?:\.[A-Za-z0-9-]{1,63})*\.[A-Za-z]{2,12}")
URL = re.compile(rb"(?:https?|ftp)://[A-Za-z0-9.-]{3,253}(?:[/?#][\x21-\x7e]{0,300})?")
_U16 = re.compile(rb"(?:[\x20-\x7e]\x00){6,}")
_TRAIL = b"\"'<>)]},;.\\"


def _domain(kind: str, value: str) -> str:
    if kind == "email":
        return value.rsplit("@", 1)[-1].lower()
    m = re.match(r"[a-z]+://([^/?#:]+)", value.lower())
    return m.group(1) if m else ""


def identifiers(data: bytes) -> list[tuple[str, str, int, bytes, str]]:
    """(kind, value, byte offset, exact bytes, encoding) for every e-mail / URL, first occurrence of each value."""
    out, seen = [], set()

    def add(kind, value, off, raw, enc):
        key = (kind, value.lower())
        if key not in seen and len(out) < MAX_PER_FILE:
            seen.add(key)
            out.append((kind, value, off, raw, enc))

    for kind, rx in (("email", EMAIL), ("url", URL)):
        for m in rx.finditer(data):
            raw = m.group().rstrip(_TRAIL)
            if len(raw) >= 6:
                add(kind, raw.decode("ascii", "replace"), m.start(), raw, "ascii")
    for m in _U16.finditer(data):
        text = m.group().decode("utf-16-le", "ignore").encode("ascii", "ignore")
        for kind, rx in (("email", EMAIL), ("url", URL)):
            for mm in rx.finditer(text):
                raw = mm.group().rstrip(_TRAIL)
                if len(raw) >= 6:
                    add(kind, raw.decode("ascii"), m.start() + 2 * mm.start(), raw.decode("ascii").encode("utf-16-le"),
                        "utf-16le")
    return out


class ReaderAgent(BaseAgent):
    name = "ReaderAgent"
    role = "text"

    def __init__(self, bridge, verifier=None, names: dict[str, str] | None = None, **kw):
        super().__init__(bridge, verifier, **kw)
        self.names = names or {}

    def analyze_artifact(self, rel: str) -> list[Finding]:
        path = self.sandbox.resolve(rel)
        with open(path, "rb") as fh:
            data = fh.read(SCAN_CAP)
        name = self.names.get(rel, rel)
        out = []
        if data:
            # the file itself, as selected by the planner: name + size + its first bytes (cited at offset 0)
            head = data[:16]
            out.append(self.finding(rel, "reader.file", f"Selected for review: {name} ({len(data):,} bytes read, "
                                                        f"starts {head[:8].hex()})",
                                    value={"original_path": name, "bytes_read": len(data), "head_hex": head.hex()},
                                    citations=[self.cite(rel, "byte_offset", head.hex(), byte_offset=0,
                                                         snippet_encoding="hex")]))
        for kind, value, off, raw, enc in identifiers(data):
            c = self.cite(rel, "byte_offset", raw.hex(), byte_offset=off, snippet_encoding="hex")
            out.append(self.finding(rel, f"reader.{kind}", f"{'E-mail address' if kind == 'email' else 'URL'} "
                                                           f"{value} in {name} (offset {off}, {enc})",
                                    value={"identifier": value, "type": kind, "domain": _domain(kind, value),
                                           "offset": off, "encoding": enc, "original_path": name},
                                    citations=[c]))
        return out

    def analyze(self, artifacts: Sequence[str]) -> list[Finding]:
        out = []
        for rel in artifacts:
            try:
                out += self.analyze_artifact(rel)
            except OSError as exc:
                self.errors.append(f"{rel}: {exc}")
        return out
