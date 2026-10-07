"""Small file helpers for workspace (non-evidence) files."""
from __future__ import annotations

import time
from pathlib import Path


def atomic_replace(tmp: Path, dest: Path, attempts: int = 20) -> None:
    """``tmp.replace(dest)`` that tolerates a brief lock on ``dest``.

    On Windows a sync client (OneDrive) or an indexer can hold the destination open for a moment,
    which makes ``os.replace`` fail with WinError 5 / 32. Retry with a short back-off instead of
    losing the run.
    """
    for i in range(attempts):
        try:
            tmp.replace(dest)
            return
        except PermissionError:
            if i == attempts - 1:
                raise
            time.sleep(min(2.0, 0.05 * 2 ** i))
