"""Investigation context: what the examiner was told, held as testable statements.

The :class:`CaseContext` is built **only from the brief** (case summary + ``context/brief``
documents). Answer keys are never loaded here, so an agent cannot "find" what it was told
the answer is. Extraction is deterministic (regular expressions and frequency rules), and
every entity keeps the document line it came from.

The :class:`InvestigationMemory` is the agent's working memory for the case
(``workspace/memory.json``): the context entities plus every link between a finding and
the context. :func:`contrast` decides each link deterministically:

==================  =========================================================================
relation            meaning
==================  =========================================================================
corroborates        a VERIFIED finding contains a context entity (name, e-mail, phone, file, device...)
keyword_match       weaker: a VERIFIED finding contains a recurring capitalised keyword of the brief
outside_timeframe   a VERIFIED timestamp falls outside the period the brief describes
not_in_context      a VERIFIED device identity (camera make/model) the brief never mentions
new_lead            a VERIFIED finding of an informative kind with no context entity
unverified_mention  a model observation (not evidence) mentions a context entity: follow up
==================  =========================================================================

Links are analysis aids. Only the finding's own verification status makes it evidence.
"""
from __future__ import annotations

import json
import re
import threading
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable

from src.verification.heuristic_verifier import normalize_text
from src.verification.models import Finding, FindingStatus

# --------------------------------------------------------------------------------------
# Entity extraction
# --------------------------------------------------------------------------------------
_EMAIL = re.compile(r"\b[\w.+-]+@[\w-]+(?:\.[\w-]+)+\b")
_URL = re.compile(r"\bhttps?://[^\s<>\"')]+", re.I)
_DOMAIN = re.compile(r"\b(?:[a-z0-9-]+\.)+(?:com|net|org|biz|info|edu|gov|io|co|uk|de|ru|cn|me|us)\b", re.I)
_PHONE = re.compile(r"(?<![\w.])(?:\+?1[\s.-]?)?\(?\d{3}\)?[\s.-]\d{3}[\s.-]\d{4}(?!\w)")
_GPS = re.compile(r"(-?\d{1,2}\.\d{4,})\s*,\s*(-?\d{1,3}\.\d{4,})")
_FILENAME = re.compile(r"(?<![\w.\-])[\w\-()]{1,60}(?:\.[\w\-]+)*\.(?:docx?|xlsx?|pptx?|pdf|txt|jpe?g|png|gif|zip|rar|7z|exe|pst|eml|csv|db|sqlite)\b",
                       re.I)
_DEVICE = re.compile(r"\b(?:iPhone(?:\s?\d{1,2}(?:\s?(?:Pro|Max|Plus|mini))*)?|iPad\w*|MacBook\w*|Pixel\s?\d\w*|"
                     r"Galaxy\s?[A-Z]?\d+\w*|Nexus\s?\d+\w*|Chromebook|Raspberry\s?Pi|Windows\s?(?:XP|Vista|7|8\.1|8|10|11)|"
                     r"Android\s?\d+(?:\.\d+)*|iOS\s?\d+(?:\.\d+)*|Ubuntu|Linux|USB|CD-R|SD card|microSD|NAS|QNAP)\b", re.I)
_MONTHS = "jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec"
_DATE_PATTERNS = (
    (re.compile(r"\b(20\d\d|19\d\d)[-/.](\d{1,2})[-/.](\d{1,2})\b"), "ymd"),
    (re.compile(r"\b(\d{1,2})/(\d{1,2})/(20\d\d|19\d\d)\b"), "mdy"),
    (re.compile(rf"\b({_MONTHS})[a-z]*\.?\s+(\d{{1,2}})(?:st|nd|rd|th)?,?\s+(20\d\d|19\d\d)\b", re.I), "Mdy"),
    (re.compile(rf"\b(\d{{1,2}})(?:st|nd|rd|th)?\s+({_MONTHS})[a-z]*\.?,?\s+(20\d\d|19\d\d)\b", re.I), "dMy"),
)
_STOP = set("""a about above after again against all also am an and any are as at be because been before being below
between both but by can could did do does doing down during each few for from further had has have having he her here
hers herself him himself his how i if in into is it its itself just me more most my myself no nor not now of off on once
only or other our ours ourselves out over own same she should so some such than that the their theirs them themselves
then there these they this those through to too under until up very was we were what when where which while who whom why
will with would you your yours yourself yourselves case evidence image images device devices data file files report
reports police officer detective examiner investigation investigator suspect suspects scenario section page figure table
note notes information created creation date time user users phone computer drive drives disk system windows android
ios iphone google apple microsoft monday tuesday wednesday thursday friday saturday sunday january february march april
june july august september october november december version please also however therefore question questions answer
answers yes using used use new one two three first second third last next part following total number name names may
company office internet email emails message messages app apps application applications account accounts download
downloaded installed install browser web website search searched photo photos picture pictures video videos
folder folders screen screens button home settings menu icon chat call calls contact contacts camera gallery map maps
music status action size type list format find identify hint test sent received inbox outbox missed incoming outgoing
unread read signed sign shared added started private thanks hey address document documents documentation digital auto
extraction open close back next done cancel ok today yesterday tomorrow""".split())

# Common given names (English / Spanish). A capitalised pair counts as a person only when its first word is a
# given name or it follows an honorific; other recurring capitalised words become keywords.
_GIVEN = set("""aaron adam adrian alan albert alex alexander alice alicia alix amanda amy ana andrea andrew angela anna
anne anthony antonio ashley barbara ben benjamin beth betty bill bob brandon brenda brian bruce carl carla carlos carol
caroline catherine charles charlie chris christian christina christine christopher cindy claire craig cynthia daniel
david deborah dennis diana diego donald donna dorothy douglas dylan ed edward elena eli elizabeth emily emma eric erik
eva evelyn frank fred gary george gerald grace greg gregory hannah harold harry heather helen henry ian isabel jack
jacob james jamie jane janet jason javier jeff jeffrey jennifer jeremy jerry jess jessica jesus jim jo joan joe joel
john johnny jon jonathan jordan jorge jose joseph josh joshua juan judy julia julie justin karen kate katherine kathy
keith kelly ken kenneth kevin kim kyle larry laura lauren linda lisa louis lucas lucia luis madison madi manuel marc
marcus margaret maria marie mario mark martha martin mary matt matthew max maxwell megan melissa michael michelle
miguel mike nancy natalie nathan nicholas nicole oliver olivia pablo pat patricia patrick paul paula pedro peter
philip rachel ralph randy raul raymond rebecca richard rick rob robert roberto roger ronald rose roy ruby russell ryan
sam samantha samuel sandra sara sarah scott sean sergio sharon shawn stephanie stephen steve steven susan terry thomas
tim timothy tina todd tom tony tyler valerie victor victoria vincent walter wayne william zachary""".split())


@dataclass
class Entity:
    id: str
    type: str                 # person | keyword | email | url | domain | phone | gps | filename | device | date
    value: str
    norm: str
    mentions: int = 0
    sources: list[dict] = field(default_factory=list)   # {"doc", "line", "excerpt"}


def _date(m: re.Match, kind: str) -> datetime | None:
    try:
        g = m.groups()
        if kind == "ymd":
            y, mo, d = int(g[0]), int(g[1]), int(g[2])
        elif kind == "mdy":
            mo, d, y = int(g[0]), int(g[1]), int(g[2])
        elif kind == "Mdy":
            mo, d, y = _MONTHS.split("|").index(g[0][:3].lower()) + 1, int(g[1]), int(g[2])
        else:
            d, mo, y = int(g[0]), _MONTHS.split("|").index(g[1][:3].lower()) + 1, int(g[2])
        return datetime(y, mo, d, tzinfo=timezone.utc)
    except (ValueError, IndexError):
        return None


class CaseContext:
    """Entities and timeframe extracted from the case brief."""

    def __init__(self, case_id: str, summary: str, docs: list[tuple[str, str]]):
        self.case_id, self.summary = case_id, summary
        self.docs = [{"name": n, "chars": len(t)} for n, t in docs]
        self.entities: dict[str, Entity] = {}
        self.dates: list[tuple[datetime, str]] = []
        self._extract([("case summary", summary), *docs])

    # ------------------------------------------------------------------ build
    @classmethod
    def load(cls, case_dir: Path, summary: str = "") -> "CaseContext":
        brief = case_dir / "context" / "brief"
        docs = []
        if brief.is_dir():
            for t in sorted(brief.glob("*.txt")):
                text = t.read_text(encoding="utf-8", errors="replace")
                if text.strip() and not text.startswith("[text extraction failed"):
                    docs.append((t.name.removesuffix(".txt") if t.with_name(t.name[:-4]).exists() else t.name, text))
        return cls(case_dir.name, summary, docs)

    def _add(self, etype: str, value: str, doc: str, line_no: int, line: str) -> None:
        value = value.strip(" .,;:()[]'\"")
        norm = normalize_text(value)
        if len(norm) < 3:
            return
        key = f"{etype}:{norm}"
        e = self.entities.get(key)
        if e is None:
            e = self.entities[key] = Entity(f"E{len(self.entities) + 1:04d}", etype, value, norm)
        e.mentions += 1
        if len(e.sources) < 5:
            e.sources.append({"doc": doc, "line": line_no, "excerpt": line.strip()[:200]})

    def _extract(self, docs: list[tuple[str, str]]) -> None:
        caps: Counter = Counter()
        cap_src: dict[str, tuple[str, int, str]] = {}
        names: Counter = Counter()
        name_src: dict[str, tuple[str, int, str]] = {}
        for doc, text in docs:
            for i, line in enumerate(text.splitlines(), 1):
                if not line.strip():
                    continue
                for rx, t in ((_EMAIL, "email"), (_URL, "url"), (_PHONE, "phone"), (_FILENAME, "filename"),
                              (_DEVICE, "device")):
                    for m in rx.finditer(line):
                        self._add(t, m.group(0), doc, i, line)
                for m in _DOMAIN.finditer(line):
                    if "@" not in line[max(0, m.start() - 1):m.start()]:
                        self._add("domain", m.group(0).lower(), doc, i, line)
                for m in _GPS.finditer(line):
                    self._add("gps", f"{m.group(1)},{m.group(2)}", doc, i, line)
                for rx, kind in _DATE_PATTERNS:
                    for m in rx.finditer(line):
                        d = _date(m, kind)
                        if d and 1990 <= d.year <= 2035:
                            self.dates.append((d, f"{doc}:{i}"))
                            self._add("date", d.date().isoformat(), doc, i, line)
                # Person names: honorific + name, or <given name> <Surname>. Other capitalised words -> keywords.
                for m in re.finditer(r"\b(?:Mr|Mrs|Ms|Dr|Det|Officer|Agent)\.?\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)", line):
                    names[m.group(1)] += 2
                    name_src.setdefault(m.group(1), (doc, i, line))
                for m in re.finditer(r"\b(?=([A-Z][a-z]{1,15})\s+([A-Z][a-z]{1,20})\b)", line):  # overlapping pairs
                    first, last = m.groups()
                    if first.lower() in _GIVEN and last.lower() not in _STOP:
                        names[f"{first} {last}"] += 2
                        name_src.setdefault(f"{first} {last}", (doc, i, line))
                for m in re.finditer(r"(?<![.!?]\s)(?<!^)\b([A-Z][a-z]{2,15})\b", line):
                    w = m.group(1)
                    if w.lower() not in _STOP:
                        caps[w] += 1
                        cap_src.setdefault(w, (doc, i, line))
        for n, c in names.items():
            if c >= 2:
                doc, i, line = name_src[n]
                self._add("person", n, doc, i, line)
                self.entities[f"person:{normalize_text(n)}"].mentions = c
        person_words = {w for k, e in self.entities.items() if e.type == "person" for w in e.value.split()}
        for w, c in caps.most_common(80):
            is_person = w in person_words or w.lower() in _GIVEN
            if c >= 3 or (is_person and c >= 2):
                doc, i, line = cap_src[w]
                kind = "person" if is_person else "keyword"
                self._add(kind, w, doc, i, line)
                self.entities[f"{kind}:{normalize_text(w)}"].mentions = c

    # ------------------------------------------------------------------ queries
    @property
    def timeframe(self) -> dict | None:
        """Period the brief talks about: the central 90% of dated mentions, padded by 30 days."""
        if not self.dates:
            return None
        ds = sorted(d for d, _ in self.dates)
        lo, hi = ds[int(len(ds) * 0.05)], ds[min(len(ds) - 1, int(len(ds) * 0.95))]
        return {"start": (lo - timedelta(days=30)).date().isoformat(), "end": (hi + timedelta(days=30)).date().isoformat(),
                "dated_mentions": len(ds)}

    def matches(self, text: str) -> list[Entity]:
        t = f" {normalize_text(text)} "
        out = []
        for e in self.entities.values():
            if e.type == "date":
                continue
            if e.type == "keyword" and e.mentions < 3:
                continue
            if f" {e.norm} " in t:
                out.append(e)
        return out

    def to_dict(self) -> dict:
        ents = sorted(self.entities.values(), key=lambda e: (e.type, -e.mentions))
        return {"case_id": self.case_id, "summary": self.summary, "brief_docs": self.docs,
                "timeframe": self.timeframe, "entities": [asdict(e) for e in ents],
                "entity_counts": dict(Counter(e.type for e in ents))}


# --------------------------------------------------------------------------------------
# Contrast and memory
# --------------------------------------------------------------------------------------
INFORMATIVE = ("exif.gps", "ocr.corroborated", "vlm_text.corroborated")
_TIME_CATEGORIES = ("exif.capture_time", "exif.digitized_time", "exif.modified_time")
_DEVICE_CATEGORIES = ("exif.camera_make", "exif.camera_model")


def _flatten(v: Any) -> Iterable[str]:
    if isinstance(v, dict):
        for x in v.values():
            yield from _flatten(x)
    elif isinstance(v, (list, tuple)):
        for x in v:
            yield from _flatten(x)
    elif isinstance(v, str):
        yield v


def contrast(ctx: CaseContext, f: Finding) -> list[dict]:
    """Deterministic links between one finding and the case context (see module docstring)."""
    text = " ".join([f.claim, f.observed_text or "", *(_flatten(f.value))])
    ents = ctx.matches(text)
    links: list[dict] = []
    if f.status == FindingStatus.VERIFIED:
        for e in ents:
            links.append({"relation": "keyword_match" if e.type == "keyword" else "corroborates", "entity": e.id, "entity_type": e.type, "value": e.value,
                          "why": f"verified {f.category} contains '{e.value}' from the brief ({e.sources[0]['doc']})"})
        tf = ctx.timeframe
        if f.category in _TIME_CATEGORIES and tf and isinstance(f.value, str):
            m = re.match(r"(\d{4}):(\d\d):(\d\d)", f.value)
            if m:
                day = f"{m.group(1)}-{m.group(2)}-{m.group(3)}"
                if not tf["start"] <= day <= tf["end"]:
                    links.append({"relation": "outside_timeframe", "value": day,
                                  "why": f"{f.category} {day} is outside the brief's period {tf['start']}..{tf['end']}"})
        if f.category == "exif.camera_model" and not ents:
            devices = [e.value for e in ctx.entities.values() if e.type == "device"]
            links.append({"relation": "not_in_context", "value": f.value,
                          "why": f"device identity '{f.value}' not mentioned in the brief"
                                 + (f" (brief mentions: {', '.join(devices[:6])})" if devices else "")})
        if not ents and f.category in INFORMATIVE:
            links.append({"relation": "new_lead", "value": str(f.value.get("text") if isinstance(f.value, dict) else f.value)[:120],
                          "why": f"verified {f.category} with no counterpart in the brief"})
    elif f.status == FindingStatus.UNCORROBORATED:
        for e in ents:
            if e.type in ("keyword",) and len(e.norm) < 5:
                continue
            links.append({"relation": "unverified_mention", "entity": e.id, "entity_type": e.type, "value": e.value,
                          "why": f"model observation mentions '{e.value}'; not evidence until verified"})
    return links


class InvestigationMemory:
    """Persistent per-case working memory (``workspace/memory.json``). Thread-safe, append-mostly."""

    def __init__(self, case_dir: Path, ctx: CaseContext):
        self.path = case_dir / "workspace" / "memory.json"
        self.ctx = ctx
        self._lock = threading.Lock()
        old = json.loads(self.path.read_text(encoding="utf-8")) if self.path.exists() else {}
        self.links: list[dict] = old.get("links", [])
        self.runs: list[dict] = old.get("runs", [])

    def record(self, run_id: str, evidence_id: str, f: Finding, links: list[dict]) -> None:
        with self._lock:
            for link in links:
                self.links.append({"run_id": run_id, "evidence_id": evidence_id, "finding_id": f.finding_id,
                                   "artifact": f.artifact_path, "category": f.category, "status": f.status.value,
                                   "claim": f.claim[:200], **link})

    def add_run(self, run: dict) -> None:
        with self._lock:
            self.runs.append(run)

    def summary(self) -> dict:
        by_rel = Counter(l["relation"] for l in self.links)
        ent_hits = defaultdict(int)
        for l in self.links:
            if l["relation"] == "corroborates":
                ent_hits[l["entity"]] += 1
        return {"links": dict(by_rel), "entities_corroborated": len(ent_hits), "runs": len(self.runs)}

    def save(self) -> None:
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            data = {**self.ctx.to_dict(), "updated_utc": datetime.now(timezone.utc).isoformat(),
                    "summary": self.summary(), "runs": self.runs, "links": self.links}
            tmp = self.path.with_name(self.path.name + ".tmp")
            tmp.write_text(json.dumps(data, indent=1, ensure_ascii=False, default=str), encoding="utf-8")
            tmp.replace(self.path)
