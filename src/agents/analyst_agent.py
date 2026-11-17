"""AnalystAgent: an LLM reads the files the orchestrator sends it and quotes what matters for the investigation.

For each file the agent builds a text view (printable ASCII and UTF-16LE strings, plus windows around the leads the
other agents have already verified), and asks the local LLM, given the case brief and those leads, to return JSON:
up to ``MAX_QUOTES`` *verbatim* quotes worth an examiner's attention, each with a type and a reason, plus names or
identifiers to look for in other files.

The safeguard is the same as for every agent: a quote becomes a finding cited by ``byte_offset`` + the exact bytes,
and the verifier re-reads them from the evidence.

* quote found in the file -> the citation carries the bytes actually at that offset -> VERIFIED;
* tool: the deterministic e-mail / URL extractor of the ReaderAgent runs on every file the agent reads (the LLM
  decides what matters, the tool makes sure no identifier in the file is missed);
* quote not in the file (the model paraphrased or invented it) -> the citation carries the model's text -> REJECTED.
  These are hallucinations caught by the verifier; they are counted and never reported as evidence.

The finding's claim and value contain only the verified quote and the file name: the model's reasoning is kept in
the trace, so nothing the model wrote outside a verified quote can reach the evidence or the evaluation. Leads
returned to the orchestrator are restricted to strings inside verified quotes.
"""
from __future__ import annotations

import re
import time
from typing import Sequence

from src.agents.base_agent import BaseAgent
from src.agents.llm import LocalLLM, parse_json
from src.agents.reader_agent import EMAIL, URL, identifiers
from src.verification.models import Finding, FindingStatus

READ_CAP = 64 << 20
VIEW_CHARS = 3500
WINDOW = 240
MAX_QUOTES = 6
MIN_QUOTE = 4
MAX_QUOTE = 300
_S8 = re.compile(rb"[\x20-\x7e]{4,}")
_S16 = re.compile(rb"(?:[\x20-\x7e]\x00){4,}")
TYPES = ("person", "email", "url", "account", "phone", "file", "location", "date", "message", "device", "other")

SYSTEM = ("You are a digital forensic analyst. You read text extracted from one evidence file and copy, word for "
          "word, the passages that could answer the investigation. Never invent or paraphrase text: every quote "
          "must appear exactly in the file text you are given. Answer with JSON only.")

USER = """Investigation brief:
{brief}

Leads already verified in other files: {leads}

File: {name}
Text extracted from the file (strings, may be fragmentary):
<<<
{text}
>>>

Return JSON: {{"quotes": [{{"quote": "<exact text copied from the file, 4-200 characters>", "type": "<one of {types}>", "why": "<short reason>"}}], "leads": ["<names, accounts, e-mails, file names to search for in other files>"]}}
Give at most {n} quotes, the most relevant first. If nothing in the file is relevant, return {{"quotes": [], "leads": []}}."""


def text_view(data: bytes, leads: Sequence[str] = (), chars: int = VIEW_CHARS) -> str:
    """Printable strings of the file; passages around verified leads come first."""
    s8 = [m.group().decode("ascii") for m in _S8.finditer(data)]
    s16 = [m.group().decode("utf-16-le", "ignore") for m in _S16.finditer(data)]
    full = "\n".join(s8 + s16)
    low = full.lower()
    parts, used = [], 0
    for lead in leads:
        i = low.find(lead.lower())
        if i >= 0 and used < chars // 2:
            w = full[max(0, i - WINDOW):i + len(lead) + WINDOW]
            parts.append("..." + w + "...")
            used += len(w)
    rest = chars - used
    return ("\n".join(parts) + "\n" if parts else "") + full[:max(rest, 0)]


def locate(data: bytes, quote: str) -> tuple[int, bytes, str] | None:
    """(offset, exact bytes at that offset, encoding) of ``quote`` in ``data``; case-insensitive, first hit."""
    low = data.lower()
    for enc in ("utf-8", "utf-16-le"):
        try:
            q = quote.encode(enc)
        except UnicodeEncodeError:
            continue
        i = data.find(q)
        if i < 0:
            i = low.find(q.lower())
        if i >= 0:
            return i, data[i:i + len(q)], enc
    return None


class AnalystAgent(BaseAgent):
    name = "AnalystAgent"
    role = "vision"                    # shares the GPU and the model with the vision agent

    def __init__(self, bridge, verifier=None, names: dict[str, str] | None = None, brief: str = "",
                 leads: Sequence[str] = (), llm: LocalLLM | None = None, tools: bool = True, **kw):
        super().__init__(bridge, verifier, **kw)
        self.names = names or {}
        self.brief = brief
        self.leads = list(leads)
        self._llm = llm
        self.tools = tools
        self.answers: dict[str, dict] = {}
        self.data: dict[str, bytes] = {}

    @property
    def llm(self) -> LocalLLM:
        if self._llm is None:
            self._llm = LocalLLM()
        return self._llm

    def prepare(self, artifacts: Sequence[str]) -> None:
        prompts, rels = [], []
        for rel in artifacts:
            try:
                with open(self.sandbox.resolve(rel), "rb") as fh:
                    data = fh.read(READ_CAP)
            except OSError as exc:
                self.errors.append(f"{rel}: {exc}")
                continue
            view = text_view(data, self.leads)
            if len(view.strip()) < 20:
                continue
            self.data[rel] = data
            rels.append(rel)
            prompts.append((SYSTEM, USER.format(brief=self.brief[:1500], leads=", ".join(self.leads[:20]) or "none",
                                                name=self.names.get(rel, rel), text=view, types=", ".join(TYPES),
                                                n=MAX_QUOTES)))
        t0 = time.perf_counter()
        raws = self.llm.chat(prompts) if prompts else []
        for rel, raw in zip(rels, raws):
            self.answers[rel] = {"raw": raw, "parsed": parse_json(raw)}
            self.tracer.emit("llm", agent=self.name, artifact=rel, model=self.llm.model_id, raw=raw[:2000])
        self.tracer.emit("model", agent=self.name, model=self.llm.model_id, files=len(rels),
                         seconds=round(time.perf_counter() - t0, 1))

    def analyze_artifact(self, rel: str) -> list[Finding]:
        ans = self.answers.get(rel)
        if not ans:
            return []
        data, name = self.data[rel], self.names.get(rel, rel)
        out, seen = [], set()
        if self.tools:                    # tool call: exhaustive e-mail / URL extraction (deterministic, cited)
            for kind, value, off, raw, enc in identifiers(data):
                c = self.cite(rel, "byte_offset", raw.hex(), byte_offset=off, snippet_encoding="hex")
                out.append(self.finding(rel, f"analyst.tool.{kind}", f"{kind} {value} in {name} (offset {off}, {enc})",
                                        value={"identifier": value, "type": kind, "offset": off, "encoding": enc,
                                               "original_path": name}, citations=[c]))
        for q in (ans["parsed"].get("quotes") or [])[:MAX_QUOTES]:
            if not isinstance(q, dict) or not isinstance(q.get("quote"), str):
                continue
            quote = q["quote"].strip()[:MAX_QUOTE]
            if len(quote) < MIN_QUOTE or quote.lower() in seen:
                continue
            seen.add(quote.lower())
            kind = q.get("type") if q.get("type") in TYPES else "other"
            hit = locate(data, quote)
            if hit:
                off, raw, enc = hit
                text = raw.decode(enc, "replace")
                c = self.cite(rel, "byte_offset", raw.hex(), byte_offset=off, snippet_encoding="hex")
                out.append(self.finding(rel, f"analyst.{kind}", f"“{text}” in {name} (offset {off}, {enc})",
                                        value={"quote": text, "type": kind, "offset": off, "encoding": enc,
                                               "original_path": name},
                                        citations=[c], model=self.llm.model_id))
            else:                         # the verifier will reject it: the model's text is not at that location
                c = self.cite(rel, "byte_offset", quote.encode("utf-8").hex(), byte_offset=0, snippet_encoding="hex")
                out.append(self.finding(rel, f"analyst.{kind}", f"[model quote, not found in file] {name}",
                                        value={"type": kind, "original_path": name, "model_quote": quote},
                                        citations=[c], model=self.llm.model_id))
            self.tracer.emit("llm_reason", agent=self.name, artifact=rel, quote=quote[:200], why=str(q.get("why"))[:300],
                             found=bool(hit))
        return out

    def verified_leads(self, findings: Sequence[Finding]) -> list[str]:
        """Leads for the orchestrator: identifiers inside VERIFIED quotes, plus the model's suggested leads that
        occur inside a verified quote of the same file (so an invented lead can never steer the search)."""
        out: list[str] = []
        tool_ids: list[str] = []                 # exhaustive tool output: lowest priority
        by_file: dict[str, list[str]] = {}
        for f in findings:
            if f.agent != self.name or f.status != FindingStatus.VERIFIED or not isinstance(f.value, dict):
                continue
            if "identifier" in f.value:
                tool_ids.append(f.value["identifier"])
            elif "quote" in f.value:
                by_file.setdefault(f.artifact_path, []).append(f.value["quote"])
        for rel, quotes in by_file.items():
            blob = "\n".join(quotes)
            for rx in (EMAIL, URL):
                out += [m.group().decode("ascii", "ignore") for m in rx.finditer(blob.encode("utf-8", "ignore"))]
            for lead in (self.answers.get(rel, {}).get("parsed", {}).get("leads") or []):
                if isinstance(lead, str) and len(lead.strip()) >= 4 and lead.strip().lower() in blob.lower():
                    out.append(lead.strip())
        return list(dict.fromkeys(out + tool_ids))
