"""Remote evidence transport: one byte-source interface for every public host the corpus uses.

Public forensic cases are scattered over many hosts. Each one gets a resolver that turns a
"landing" link into a direct, range-readable URL, plus a folder lister:

=====================  ===============================================================
Host / scheme          How it is read
=====================  ===============================================================
S3 / plain HTTP        HTTP range requests (Digital Corpora, NIST CFReDS archive, GCS, B2)
Google Drive file      ``drive.usercontent.google.com/download?id=..&confirm=t`` (ranges work)
Google Drive folder    ``embeddedfolderview`` pages, walked recursively
MediaFire              the direct ``downloadN.mediafire.com`` link scraped from the file page;
                       folders through the public ``folder/get_content`` API
archive.org            ``ia:<identifier>``: metadata API, ``/download/<id>/<file>``
Wayback Machine        ``web/<ts>id_/<url>`` (raw bytes, no toolbar rewriting)
GitHub repository      ``gh:<owner>/<repo>[@branch][/subdir]``: git tree API, raw / LFS media URLs
Digital Corpora prefix ``dc:<prefix>/``: S3 ListObjectsV2
figshare / Zenodo      ``figshare:<article id>`` / ``zenodo:<record id>``: their public APIs
Dropbox shared file    ``dl=1``
=====================  ===============================================================

Hosts without range support are streamed once to the spool and then read locally
(:func:`open_source`). Nothing here interprets the bytes; that is left to the extractors.
"""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
from dataclasses import dataclass
from html import unescape
from pathlib import Path
from urllib.parse import parse_qs, quote, unquote, urlparse

import requests

UA = {"User-Agent": "Mozilla/5.0 (ForensiAgent corpus builder; research use)", "Accept-Encoding": "identity"}
DC_BUCKET = "https://digitalcorpora.s3.amazonaws.com"
_DIRECT_CACHE: dict[str, str] = {}
_lock = threading.Lock()


class RemoteError(RuntimeError):
    pass


# --------------------------------------------------------------------------------------
# Direct-URL resolution
# --------------------------------------------------------------------------------------
def gdrive_id(url: str) -> str | None:
    m = re.search(r"drive\.google\.com/(?:file/d/|open\?id=|uc\?(?:[^#]*&)?id=)([A-Za-z0-9_-]{10,})", url) \
        or re.search(r"drive\.usercontent\.google\.com/download\?(?:[^#]*&)?id=([A-Za-z0-9_-]{10,})", url)
    return m.group(1) if m else None


def gdrive_folder_id(url: str) -> str | None:
    m = re.search(r"drive\.google\.com/drive/(?:u/\d+/)?folders/([A-Za-z0-9_-]{10,})", url)
    return m.group(1) if m else None


def direct_url(url: str) -> str:
    """The URL whose bytes are the file itself (follows host-specific landing pages)."""
    url = url.strip()
    if url in _DIRECT_CACHE:
        return _DIRECT_CACHE[url]
    out = url
    if url.startswith("ia:"):
        ident, _, name = url[3:].partition("/")
        out = f"https://archive.org/download/{ident}/{quote(name)}"
    elif (gid := gdrive_id(url)):
        out = f"https://drive.usercontent.google.com/download?id={gid}&export=download&confirm=t"
    elif "mediafire.com/file/" in url or "mediafire.com/?" in url:
        out = _mediafire_direct(url)
    elif "web.archive.org/web/" in url:
        out = re.sub(r"(web\.archive\.org/web/\d+)(?:[a-z]{2}_)?/", r"\1id_/", url, count=1)
    elif "dropbox.com/" in url:
        out = re.sub(r"([?&])dl=0", r"\1dl=1", url) if "dl=0" in url else url + ("&" if "?" in url else "?") + "dl=1"
    elif url.startswith("http://downloads.digitalcorpora.org/") or url.startswith("https://downloads.digitalcorpora.org/") \
            or url.startswith("https://search.digitalcorpora.org/"):
        out = DC_BUCKET + "/" + url.split(".org/", 1)[1]
    with _lock:
        _DIRECT_CACHE[url] = out
    return out


def _mediafire_direct(url: str) -> str:
    r = requests.get(url, headers=UA, timeout=60)
    r.raise_for_status()
    m = re.search(r'href="(https?://download\d*\.mediafire\.com/[^"]+)"', r.text)
    if not m:
        m = re.search(r'data-scrambled-url="([^"]+)"', r.text)
        if m:
            import base64
            return base64.b64decode(m.group(1)).decode()
        raise RemoteError(f"MediaFire: no direct link on {url}")
    return unescape(m.group(1))


# --------------------------------------------------------------------------------------
# Range-readable source
# --------------------------------------------------------------------------------------
class RemoteSource:
    """Retried, thread-safe HTTP range reads of one remote object (``.size`` / ``.get``)."""

    def __init__(self, url: str, name: str = ""):
        self.landing = url
        self.url = direct_url(url)
        self.name = name or unquote(urlparse(self.landing).path.rsplit("/", 1)[-1])
        self._local = threading.local()
        self.size, self.ranges, self.filename = self._probe()
        if self.filename and not name:
            self.name = self.filename

    def _session(self) -> requests.Session:
        if not hasattr(self._local, "s"):
            s = requests.Session()
            s.headers.update(UA)
            self._local.s = s
        return self._local.s

    def _probe(self) -> tuple[int, bool, str]:
        # A 1-byte GET works on every host (HEAD is refused or lies on Drive, MediaFire, GCS signed URLs).
        last = None
        for attempt in range(5):
            try:
                r = self._session().get(self.url, headers={"Range": "bytes=0-0"}, stream=True, timeout=90,
                                        allow_redirects=True)
                ctype = r.headers.get("Content-Type", "")
                fname = ""
                cd = r.headers.get("Content-Disposition", "")
                m = re.search(r"filename\*=UTF-8''([^;]+)|filename=\"?([^\";]+)", cd)
                if m:
                    fname = unquote(m.group(1) or m.group(2)).strip()
                if r.status_code == 206 and "Content-Range" in r.headers:
                    r.close()
                    size = int(r.headers["Content-Range"].rsplit("/", 1)[1])
                    if "raw.githubusercontent.com" in self.url and size < 1024 and not self.url.endswith("#plain"):
                        self._resolve_lfs()
                        return self._probe()
                    # The Wayback Machine answers the first range and then ignores later ones.
                    return size, "web.archive.org" not in self.url, fname
                if r.status_code == 200:
                    if "text/html" in ctype and ("google" in self.url or "mediafire" in self.url):
                        body = r.raw.read(4096, decode_content=True).decode("utf-8", "replace")
                        r.close()
                        raise RemoteError(f"host returned an HTML page instead of the file ({self.url}): "
                                          f"{re.sub(r'<[^>]+>|\\s+', ' ', body)[:200]}")
                    size = int(r.headers.get("Content-Length") or -1)
                    r.close()
                    return size, False, fname
                r.close()
                raise RemoteError(f"HTTP {r.status_code} for {self.url}")
            except (requests.RequestException, RemoteError) as exc:
                last = exc
                if isinstance(exc, RemoteError) and ("HTML page" in str(exc) or "HTTP 404" in str(exc)):
                    raise
                time.sleep(2 ** attempt)
        raise RemoteError(f"cannot open {self.url}: {last}")

    def _resolve_lfs(self) -> None:
        body = self._session().get(self.url, timeout=60).content
        if body.startswith(b"version https://git-lfs"):
            self.url = self.url.replace("raw.githubusercontent.com", "media.githubusercontent.com/media")
        else:
            self.url += "#plain"   # small regular blob: probe again without the LFS check

    def get(self, start: int, length: int) -> bytes:
        if length <= 0:
            return b""
        if not self.ranges:
            raise RemoteError(f"{self.url} does not support range requests")
        throttled = 0
        attempt = 0
        while True:
            try:
                r = self._session().get(self.url, headers={"Range": f"bytes={start}-{start + length - 1}"},
                                        timeout=180)
                if r.status_code == 206 and len(r.content) == length:
                    return r.content
                if "google" in self.url and r.status_code in (200, 403, 429) and \
                        "text/html" in r.headers.get("Content-Type", "") and throttled < 6:
                    # Drive answers an HTML quota page when it rate-limits an IP: wait it out, don't burn retries.
                    throttled += 1
                    with _lock:
                        print(f"  [drive] rate-limited; pausing {120 * throttled}s ({throttled}/6)", flush=True)
                    time.sleep(120 * throttled)
                    continue
                raise IOError(f"HTTP {r.status_code}, got {len(r.content)}/{length} bytes")
            except (requests.RequestException, IOError):
                if attempt == 6:
                    raise
                time.sleep(min(60, 2 ** attempt))
                attempt += 1
        raise AssertionError("unreachable")


def stream_download(url: str, dest: Path, expected: int = -1) -> None:
    """Single-stream download (hosts without range support), resumable when the host allows it."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    for attempt in range(5):
        try:
            with requests.get(direct_url(url), stream=True, timeout=180, headers=UA) as r:
                r.raise_for_status()
                with open(tmp, "wb") as fh:
                    shutil.copyfileobj(r.raw, fh, 8 << 20) if not r.headers.get("Content-Encoding") else \
                        [fh.write(c) for c in r.iter_content(8 << 20)]
            if expected > 0 and tmp.stat().st_size != expected:
                raise IOError(f"short download {tmp.stat().st_size}/{expected}")
            tmp.replace(dest)
            return
        except (requests.RequestException, IOError):
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))


def download(url: str, dest: Path) -> Path:
    """Whole-file download (documents, small artefacts). Parallel ranges when the host allows it."""
    from src.utils.disk_image import parallel_download

    src = RemoteSource(url)
    if src.ranges and src.size > 64 << 20:
        parallel_download(src, dest, label=dest.name)
    else:
        stream_download(url, dest, src.size)
    return dest


# --------------------------------------------------------------------------------------
# Folder listing
# --------------------------------------------------------------------------------------
@dataclass(frozen=True)
class RemoteFile:
    path: str            # relative path inside the listed folder
    url: str             # landing / direct URL (resolved again by direct_url)
    size: int = -1


_LIST_CACHE: dict[str, list] = {}


def list_remote(spec: str) -> list[RemoteFile]:
    """Expand a folder-like spec into files (cached per process). A plain file URL yields itself."""
    spec = spec.strip()
    if spec not in _LIST_CACHE:
        _LIST_CACHE[spec] = _list_remote(spec)
    return list(_LIST_CACHE[spec])


def _list_remote(spec: str) -> list[RemoteFile]:
    if (fid := gdrive_folder_id(spec)):
        return _gdrive_folder(fid)
    if spec.startswith("ia:") and "/" not in spec[3:]:
        return _archive_org(spec[3:])
    if spec.startswith("dc:"):
        return _dc_prefix(spec[3:])
    if spec.startswith("gh:"):
        return _github(spec[3:])
    if spec.startswith("figshare:"):
        return _figshare(spec.split(":", 1)[1])
    if spec.startswith("zenodo:"):
        return _zenodo(spec.split(":", 1)[1])
    if "mediafire.com/folder/" in spec:
        return _mediafire_folder(re.search(r"/folder/([a-z0-9]+)", spec).group(1))
    name = unquote(urlparse(spec if not spec.startswith("ia:") else "x://" + spec[3:]).path.rsplit("/", 1)[-1])
    if gdrive_id(spec) or "dropbox.com/" in spec or not name or "." not in name:
        try:   # landing URLs (".../view", "?dl=0") carry no file name: ask the host
            src = RemoteSource(spec)
            return [RemoteFile(src.name or name, spec, src.size)]
        except RemoteError:
            pass
    return [RemoteFile(name, spec)]


def _gdrive_folder(fid: str, prefix: str = "", depth: int = 0) -> list[RemoteFile]:
    if depth > 8:
        return []
    for attempt in range(5):
        try:
            r = requests.get(f"https://drive.google.com/embeddedfolderview?id={fid}", headers=UA, timeout=90)
            if r.status_code >= 500 and attempt < 4:
                time.sleep(10 * (attempt + 1))   # Drive rate-limits rapid folder walks with 500s
                continue
            break
        except requests.RequestException:
            if attempt == 4:
                raise
            time.sleep(5 * (attempt + 1))
    if r.status_code != 200:
        raise RemoteError(f"Google Drive folder {fid}: HTTP {r.status_code}")
    out: list[RemoteFile] = []
    for m in re.finditer(r'<a href="([^"]+)"[^>]*>.*?flip-entry-title">([^<]*)<', r.text, re.S):
        href, title = unescape(m.group(1)), unescape(m.group(2)).strip()
        if (sub := gdrive_folder_id(href)):
            out += _gdrive_folder(sub, f"{prefix}{title}/", depth + 1)
        elif (gid := gdrive_id(href)):
            out.append(RemoteFile(prefix + title, f"https://drive.google.com/file/d/{gid}/view"))
    return out


def _archive_org(ident: str) -> list[RemoteFile]:
    d = requests.get(f"https://archive.org/metadata/{ident}", headers=UA, timeout=60).json()
    return [RemoteFile(f["name"], f"ia:{ident}/{f['name']}", int(f.get("size", -1)))
            for f in d.get("files", []) if f.get("source") == "original"
            and not re.search(r"_(files|meta)\.(xml|sqlite)$|__ia_thumb|\.torrent$", f["name"])]


def _dc_prefix(prefix: str) -> list[RemoteFile]:
    prefix = prefix if prefix.startswith("corpora/") else "corpora/" + prefix
    out, token = [], None
    while True:
        params = {"list-type": "2", "prefix": prefix}
        if token:
            params["continuation-token"] = token
        x = requests.get(DC_BUCKET + "/", params=params, timeout=60).text
        for key, size in re.findall(r"<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>", x, re.S):
            key = unescape(key)
            if not key.endswith("/") and not key.endswith(".htaccess"):
                out.append(RemoteFile(key[len(prefix):].lstrip("/"), f"{DC_BUCKET}/{quote(key)}", int(size)))
        m = re.search(r"<NextContinuationToken>([^<]+)</NextContinuationToken>", x)
        if not m:
            return out
        token = unescape(m.group(1))


def _github(spec: str) -> list[RemoteFile]:
    m = re.match(r"([^/@]+)/([^/@]+)(?:@([^/]+))?(?:/(.*))?$", spec)
    owner, repo, branch, sub = m.group(1), m.group(2), m.group(3), (m.group(4) or "").strip("/")
    api = f"https://api.github.com/repos/{owner}/{repo}"
    if not branch:
        branch = requests.get(api, headers=UA, timeout=60).json().get("default_branch", "main")
    tree = requests.get(f"{api}/git/trees/{branch}?recursive=1", headers=UA, timeout=60).json()
    if "tree" not in tree:
        raise RemoteError(f"GitHub {spec}: {tree.get('message', tree)}")
    out = []
    for t in tree["tree"]:
        if t["type"] != "blob" or (sub and not t["path"].startswith(sub + "/")):
            continue
        rel = t["path"][len(sub) + 1:] if sub else t["path"]
        # raw.githubusercontent serves blobs; an LFS pointer is swapped for the media URL when probed.
        out.append(RemoteFile(rel, f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{quote(t['path'])}",
                              int(t.get("size", -1))))
    return out


def _figshare(article: str) -> list[RemoteFile]:
    d = requests.get(f"https://api.figshare.com/v2/articles/{article}", headers=UA, timeout=60).json()
    return [RemoteFile(f["name"], f["download_url"], int(f.get("size", -1))) for f in d.get("files", [])]


def _zenodo(record: str) -> list[RemoteFile]:
    d = requests.get(f"https://zenodo.org/api/records/{record}", headers=UA, timeout=60).json()
    return [RemoteFile(f["key"], f["links"]["self"], int(f.get("size", -1))) for f in d.get("files", [])]


def _mediafire_folder(key: str, prefix: str = "", depth: int = 0) -> list[RemoteFile]:
    api = "https://www.mediafire.com/api/1.5/folder/get_content.php"
    out: list[RemoteFile] = []
    for ctype in ("files", "folders"):
        chunk = 1
        while True:
            d = requests.get(api, params={"folder_key": key, "content_type": ctype, "response_format": "json",
                                          "chunk": chunk}, headers=UA, timeout=60).json()["response"]
            fc = d.get("folder_content", {})
            for f in fc.get("files") or []:
                out.append(RemoteFile(prefix + f["filename"], f["links"]["normal_download"], int(f.get("size", -1))))
            for f in fc.get("folders") or []:
                if depth < 6:
                    out += _mediafire_folder(f["folderkey"], f"{prefix}{f['name']}/", depth + 1)
            if fc.get("more_chunks") != "yes":
                break
            chunk += 1
    return out


def open_source(url: str, spool: Path, label: str = ""):
    """A range-readable source for ``url``: remote when the host supports ranges, else a spooled copy."""
    from src.utils import dataset_scraper as ds

    src = RemoteSource(url)
    if src.ranges:
        return src
    dest = spool / f"{label or 'stream'}_{re.sub(r'[^A-Za-z0-9._-]', '_', src.name)[-80:]}"
    if not (dest.exists() and dest.stat().st_size == src.size):
        print(f"  [stream] {src.name}: host has no range support, downloading {src.size / 1e9:.2f} GB", flush=True)
        stream_download(url, dest, src.size)
    fs = ds.FileSource(dest)
    fs.name = src.name
    return fs


def describe(spec: str) -> str:
    return json.dumps([f.__dict__ for f in list_remote(spec)], indent=1)
