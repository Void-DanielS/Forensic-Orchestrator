"""Deterministic Heuristic Verification Engine.

The engine is the only component allowed to promote a finding to ``VERIFIED``. For each
citation it physically re-opens the cited file inside the sandbox and:

1. checks the file exists and lives inside the sandbox;
2. recomputes SHA-256 and compares with ``expected_hash`` (evidence integrity);
3. re-executes the locator (byte comparison, line read, regex, read-only SQL, EXIF
   re-extraction) and compares the result with ``raw_text_snippet``.

A finding is VERIFIED only if *every* citation passes and, when the finding carries
model-derived ``observed_text`` (OCR/VLM), that text is sufficiently similar to the
verified ground-truth snippet. Findings without citations are UNCORROBORATED: they are
kept for analyst context but are never reported as evidence. No model is involved here.
"""
from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from pathlib import Path
from typing import Iterable

from src.bridge.forensic_bridge import ForensicBridge, SandboxViolation, render_exif_value
from src.config.settings import VERIFICATION
from src.verification.models import Citation, Finding, FindingStatus, VerificationResult


def normalize_text(s: str) -> str:
    s = unicodedata.normalize("NFKC", s).casefold()
    s = re.sub(r"[^\w]+", " ", s)
    return re.sub(r"\s+", " ", s).strip()


def text_similarity(observed: str, truth: str) -> float:
    """Fraction of the *observed* (model-read) text explained by the ground truth.

    1.0 when the observed text is contained in the truth (e.g. one OCR line of a longer
    message). A short truth value inside a long observed line scores low, so a common
    word in a database cannot "verify" an unrelated sentence.
    """
    a, b = normalize_text(observed), normalize_text(truth)
    if not a or not b:
        return 0.0
    matched = sum(m.size for m in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    return matched / len(a)


def corroboration(observed: str, truth: str) -> tuple[bool, float, float]:
    """Deterministic rule deciding whether model-read text is backed by a ground-truth value.

    Returns ``(ok, observed_coverage, truth_coverage)``. Requires that the ground truth explains
    the observed text (coverage >= threshold) and, for short text, that the match is specific:
    the observed text must make up a substantial share of the cited value.
    """
    a, b = normalize_text(observed), normalize_text(truth)
    if not a or not b:
        return False, 0.0, 0.0
    matched = sum(m.size for m in SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks())
    obs_cov, truth_cov = matched / len(a), matched / len(b)
    specific = len(a) >= VERIFICATION.corroboration_min_specific_chars or truth_cov >= VERIFICATION.corroboration_min_truth_coverage
    return obs_cov >= VERIFICATION.ocr_corroboration_threshold and specific, obs_cov, truth_cov


class HeuristicVerificationEngine:
    MIN_CORROBORATION_CHARS = 10

    def __init__(self, sandbox_root: Path | str, bridge: ForensicBridge | None = None):
        # Without an explicit bridge the engine opens its own: its own hash cache and its own SQLite working
        # copies, so nothing the agent computed (or cached) is reused to check the agent.
        self.owns_bridge = bridge is None
        self.bridge = bridge or ForensicBridge(sandbox_root)
        self.sandbox = self.bridge.sandbox
        self.audit_log: list[dict] = []

    def close(self) -> None:
        if self.owns_bridge:
            self.bridge.close()

    # ------------------------------------------------------------------ public API
    def verify_citation(self, c: Citation) -> VerificationResult:
        base = {"locator_type": c.locator_type, "file_path": c.file_path}
        try:
            path = self.sandbox.resolve(c.file_path)
        except SandboxViolation as exc:
            return self._log(VerificationResult(verified=False, reason=f"sandbox violation: {exc}", **base))
        if not path.is_file():
            return self._log(VerificationResult(verified=False, reason="file not found", **base))

        actual = self.sandbox.hash(c.file_path)
        if actual.lower() != c.expected_hash.lower():
            return self._log(VerificationResult(verified=False, reason="hash mismatch: evidence differs from cited version",
                                                actual_hash=actual, hash_match=False, **base))
        if c.locator_type == "sql_query":
            sides = self.sandbox.side_hashes(c.file_path)
            if {k: v.lower() for k, v in sides.items()} != {k: v.lower() for k, v in c.side_hashes.items()}:
                return self._log(VerificationResult(
                    verified=False, reason=f"side-file mismatch: cited {sorted(c.side_hashes)} vs present {sorted(sides)} "
                                           "(WAL / journal differs from the version the agent read)",
                    actual_hash=actual, hash_match=False, **base))
        try:
            ok, observed, reason = getattr(self, f"_check_{c.locator_type}")(c, path)
        except Exception as exc:  # noqa: BLE001 - any re-execution error is a failed verification
            ok, observed, reason = False, None, f"re-execution error: {type(exc).__name__}: {exc}"
        return self._log(VerificationResult(verified=ok, reason=reason, actual_hash=actual, hash_match=True,
                                            observed_snippet=observed, **base))

    def verify_finding(self, f: Finding) -> Finding:
        if not f.citations:
            f.status = FindingStatus.UNCORROBORATED
            f.verification = [{"verified": False, "reason": "no citation: model observation only"}]
            return f
        results = [self.verify_citation(c) for c in f.citations]
        f.verification = [r.model_dump() for r in results]
        ok = all(r.verified for r in results)
        if ok and f.observed_text is not None:
            ok = self._check_corroboration(f, results)
        f.status = FindingStatus.VERIFIED if ok else FindingStatus.REJECTED
        return f

    def verify_all(self, findings: Iterable[Finding]) -> list[Finding]:
        return [self.verify_finding(f) for f in findings]

    def dump_audit_log(self, path: Path) -> None:
        path.write_text(json.dumps(self.audit_log, indent=2, default=str), encoding="utf-8")

    # ------------------------------------------------------------------ locators
    def _check_byte_offset(self, c: Citation, path: Path) -> tuple[bool, str | None, str]:
        expected = bytes.fromhex(c.raw_text_snippet) if c.snippet_encoding == "hex" else c.raw_text_snippet.encode("utf-8")
        with open(path, "rb") as fh:
            fh.seek(c.byte_offset)  # type: ignore[arg-type]
            got = fh.read(len(expected))
        observed = got.hex() if c.snippet_encoding == "hex" else got.decode("utf-8", errors="replace")
        return got == expected, observed, "bytes match at offset" if got == expected else "bytes differ at offset"

    def _check_line_number(self, c: Citation, path: Path) -> tuple[bool, str | None, str]:
        line = self.bridge.text.line(c.file_path, c.line_number)  # type: ignore[arg-type]
        if line is None:
            return False, None, "line out of range"
        ok = self._match(c, [line])
        return ok, line, "snippet found on line" if ok else "snippet not on line"

    def _check_regex(self, c: Citation, path: Path) -> tuple[bool, str | None, str]:
        text = self.bridge.text.read_text(c.file_path)
        matches = [m.group("v") if "v" in m.re.groupindex else m.group(0) for m in re.finditer(c.regex, text)]  # type: ignore[arg-type]
        ok = self._match(c, matches)
        return ok, (matches[0] if matches else None), f"{len(matches)} regex match(es)" + ("" if ok else ", snippet absent")

    def _check_sql_query(self, c: Citation, path: Path) -> tuple[bool, str | None, str]:
        rows = self.bridge.sqlite.query(c.file_path, c.sql_query, c.sql_params, VERIFICATION.sql_row_limit)  # type: ignore[arg-type]
        cells = [str(v) for row in rows for v in row if v is not None]
        ok = self._match(c, cells)
        observed = next((v for v in cells if self._match(c, [v])), cells[0] if cells else None)
        return ok, observed, f"query returned {len(rows)} row(s)" + ("" if ok else ", snippet absent")

    def _check_exif_tag(self, c: Citation, path: Path) -> tuple[bool, str | None, str]:
        raw = self.bridge.images.raw_exif(c.file_path)
        if c.exif_tag not in raw:
            return False, None, f"EXIF tag {c.exif_tag} not present"
        observed = render_exif_value(raw[c.exif_tag])
        ok = self._match(c, [observed])
        return ok, observed, "EXIF tag value matches" if ok else "EXIF tag value differs"

    def _check_registry(self, c: Citation, path: Path) -> tuple[bool, str | None, str]:
        observed = self.bridge.registry.read(c.file_path, c.registry)  # type: ignore[arg-type]
        if observed is None:
            return False, None, "registry key / value not present"
        ok = self._match(c, [observed])
        return ok, observed, "registry value matches" if ok else "registry value differs"

    def _check_evtx_record(self, c: Citation, path: Path) -> tuple[bool, str | None, str]:
        observed = self.bridge.evtx.summary(c.file_path, c.evtx_record)  # type: ignore[arg-type]
        if observed is None:
            return False, None, "event record not present"
        ok = self._match(c, [observed])
        return ok, observed, "event record matches" if ok else "event record differs"

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _match(c: Citation, candidates: list[str]) -> bool:
        if c.match_mode == "contains":
            return any(c.raw_text_snippet in v for v in candidates)
        return c.raw_text_snippet in candidates

    def _check_corroboration(self, f: Finding, results: list[VerificationResult]) -> bool:
        observed = f.observed_text or ""
        if len(normalize_text(observed)) < self.MIN_CORROBORATION_CHARS:
            f.verification.append({"verified": False, "reason": "observed text too short to corroborate"})
            return False
        checks = [corroboration(observed, r.observed_snippet or "") for r in results]
        ok, obs_cov, truth_cov = max(checks, key=lambda c: (c[0], c[1]))
        f.verification.append({
            "verified": ok,
            "reason": f"model text vs ground truth: observed coverage {obs_cov:.3f} "
                      f"(threshold {VERIFICATION.ocr_corroboration_threshold}), truth coverage {truth_cov:.3f}"
                      + ("" if ok else " - not corroborated / not specific"),
        })
        return ok

    def _log(self, r: VerificationResult) -> VerificationResult:
        self.audit_log.append(r.model_dump())
        return r
