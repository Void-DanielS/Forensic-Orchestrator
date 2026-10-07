"""File systems The Sleuth Kit build cannot read (XFS, APFS), through Fox-IT's ``dissect`` parsers.

The adapter makes a dissect file system look like what :mod:`src.utils.disk_image` expects from
pytsk3: :meth:`DissectHandle.walk` yields the same :class:`FsEntry` rows (path, inode, size,
MACB times) and :meth:`DissectHandle.open_meta` returns an object with ``read_random``. Live
(allocated) entries only: these parsers do not recover deleted directory entries.
"""
from __future__ import annotations

import io
import stat
import threading
from datetime import datetime, timezone
from typing import Iterator


def _iso(v) -> str | None:
    if v is None:
        return None
    try:
        if isinstance(v, datetime):
            dt = v if v.tzinfo else v.replace(tzinfo=timezone.utc)
        else:
            dt = datetime.fromtimestamp(float(v), timezone.utc)
        return dt.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")
    except (OverflowError, OSError, ValueError, TypeError):
        return None


class _Region(io.RawIOBase):
    """Seekable file object over ``reader(offset, n)`` starting at ``base`` (an image region)."""

    def __init__(self, reader, base: int, size: int):
        super().__init__()
        self.reader, self.base, self.size, self.pos = reader, base, size, 0
        self.lock = threading.Lock()

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def seek(self, off: int, whence: int = 0) -> int:
        self.pos = off if whence == 0 else self.pos + off if whence == 1 else self.size + off
        return self.pos

    def tell(self) -> int:
        return self.pos

    def readinto(self, b) -> int:  # type: ignore[override]
        n = min(len(b), self.size - self.pos)
        if n <= 0:
            return 0
        with self.lock:
            data = self.reader(self.base + self.pos, n)
        b[:len(data)] = data
        self.pos += len(data)
        return len(data)


class _Reader:
    def __init__(self, stream):
        self.stream = stream
        self.lock = threading.Lock()

    def read_random(self, off: int, n: int) -> bytes:
        with self.lock:
            self.stream.seek(off)
            return self.stream.read(n)


class DissectHandle:
    """pytsk3-like handle over a dissect XFS file system or one APFS volume."""

    def __init__(self, kind: str, fs, label: str):
        self.kind, self.fs, self.label = kind, fs, label

    # -- XFS / APFS differences
    def _root(self):
        return self.fs.root if self.kind == "xfs" else self.fs.get("/")

    def _children(self, node):
        if self.kind == "xfs":
            for child in node.iterdir():
                if child.filename in (".", ".."):
                    continue
                yield child.filename, child
        else:
            for entry in node.iterdir():
                child = entry.inode if not isinstance(entry.inode, int) else self.fs.inode(entry.inode)
                yield entry.name, child

    def _inum(self, node) -> int:
        return int(node.inum if self.kind == "xfs" else node.inode)

    def _is_dir(self, node) -> bool:
        return node.filetype == stat.S_IFDIR if self.kind == "xfs" else node.is_dir()

    def _is_file(self, node) -> bool:
        return node.filetype == stat.S_IFREG if self.kind == "xfs" else node.is_file()

    def walk(self, fs_label: str, prune, max_depth: int = 64) -> Iterator:
        from src.utils.disk_image import FsEntry

        stack = [(self._root(), "", 0)]
        seen: set[int] = set()
        while stack:
            node, base, depth = stack.pop()
            try:
                children = list(self._children(node))
            except Exception:  # noqa: BLE001 - one unreadable directory must not stop the walk
                continue
            for name, child in children:
                path = f"{base}/{name}"
                try:
                    is_dir = self._is_dir(child)
                    if not is_dir and not self._is_file(child):
                        continue          # symlinks, devices, sockets
                    inum = self._inum(child)
                    size = 0 if is_dir else int(child.size)
                    e = FsEntry(fs_label, path, inum, size, is_dir, True, _iso(child.mtime), _iso(child.atime),
                                _iso(child.ctime), _iso(getattr(child, "crtime", None) or getattr(child, "btime", None)))
                except Exception:  # noqa: BLE001
                    continue
                yield e
                if is_dir and depth < max_depth and inum not in seen and not prune.match(path):
                    seen.add(inum)
                    stack.append((child, path, depth + 1))

    def open_meta(self, inode: int):
        node = self.fs.get_inode(inode) if self.kind == "xfs" else self.fs.inode(inode)
        return _Reader(node.open())


def dissect_filesystems(reader, offset: int, size: int, keep: list) -> list[tuple[str, str, DissectHandle]]:
    """(fstype, description, handle) for XFS / APFS found at ``offset`` of an image."""
    try:
        head = reader(offset, 4096)
    except OSError:
        return []
    region = _Region(reader, offset, size)
    keep.append(region)
    found = []
    if head[:4] == b"XFSB":
        from dissect.xfs import XFS
        found.append(("xfs", "XFS (dissect)", DissectHandle("xfs", XFS(region), "")))
    elif head[32:36] == b"NXSB":
        from dissect.apfs import APFS
        container = APFS(region)
        keep.append(container)
        for vol in container.volumes:
            try:
                if vol.is_encrypted:
                    continue
                name = vol.name
            except Exception:  # noqa: BLE001
                continue
            found.append(("apfs", f"APFS volume {name} (dissect)", DissectHandle("apfs", vol, name)))
    return found
