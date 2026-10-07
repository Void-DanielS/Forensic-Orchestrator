"""Plain-text extraction for case documents (PDF, DOCX, XLSX, HTML, Markdown, text).

Each document ``X.ext`` gets a sibling ``X.ext.txt`` so agents, the GUI and the catalog work
on text without format-specific code. Pure standard library + pypdf.
"""
from __future__ import annotations

import html
import re
import zipfile
from pathlib import Path
from xml.etree import ElementTree as ET

TEXT_SUFFIX = ".txt"


def extract_text(path: Path) -> str:
    ext = path.suffix.lower()
    if ext == ".pdf":
        import logging

        from pypdf import PdfReader

        logging.getLogger("pypdf").setLevel(logging.ERROR)
        return "\n".join((p.extract_text() or "") for p in PdfReader(str(path)).pages)
    if ext in (".docx", ".odt"):
        return _docx(path)
    if ext == ".xlsx":
        return _xlsx(path)
    raw = path.read_bytes()
    text = _decode(raw)
    if ext in (".html", ".htm"):
        text = re.sub(r"(?is)<(script|style).*?</\1>", " ", text)
        text = re.sub(r"(?i)<br\s*/?>|</(p|div|li|tr|h\d)>", "\n", text)
        text = html.unescape(re.sub(r"<[^>]+>", " ", text))
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n\s*\n+", "\n\n", text)
    return text


def _decode(raw: bytes) -> str:
    if raw.startswith((b"\xff\xfe", b"\xfe\xff")):
        return raw.decode("utf-16")
    try:
        return raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        return raw.decode("cp1252", errors="replace")


def _docx(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        name = "word/document.xml" if "word/document.xml" in z.namelist() else "content.xml"
        root = ET.fromstring(z.read(name))
    paras = []
    for p in root.iter():
        if p.tag.endswith("}p"):
            t = "".join(n.text or "" for n in p.iter() if n.tag.endswith("}t") or n.tag.endswith("}span") and n.text)
            if t.strip():
                paras.append(t)
    return "\n".join(paras)


def _xlsx(path: Path) -> str:
    with zipfile.ZipFile(path) as z:
        shared: list[str] = []
        if "xl/sharedStrings.xml" in z.namelist():
            for si in ET.fromstring(z.read("xl/sharedStrings.xml")):
                shared.append("".join(t.text or "" for t in si.iter() if t.tag.endswith("}t")))
        out = []
        for name in sorted(n for n in z.namelist() if n.startswith("xl/worksheets/sheet") and n.endswith(".xml")):
            out.append(f"# {name.rsplit('/', 1)[-1]}")
            for row in ET.fromstring(z.read(name)).iter():
                if not row.tag.endswith("}row"):
                    continue
                cells = []
                for c in row:
                    v = next((x.text for x in c if x.tag.endswith("}v")), None)
                    if v is None:
                        v = "".join(t.text or "" for t in c.iter() if t.tag.endswith("}t")) or None
                    elif c.get("t") == "s":
                        v = shared[int(v)] if int(v) < len(shared) else v
                    if v is not None:
                        cells.append(v)
                if cells:
                    out.append(" | ".join(cells))
        return "\n".join(out)


def ensure_text(path: Path) -> Path | None:
    """Write ``<doc>.txt`` beside ``path`` (once). Returns the text path, or None if unsupported."""
    if path.suffix.lower() == TEXT_SUFFIX:
        return path
    out = path.with_name(path.name + TEXT_SUFFIX)
    if out.exists():
        return out
    try:
        text = extract_text(path)
    except Exception as exc:  # noqa: BLE001 - corrupt / unsupported document
        text = f"[text extraction failed: {type(exc).__name__}: {exc}]"
    out.write_text(text, encoding="utf-8")
    return out
