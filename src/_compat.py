"""Shared runtime guards for all pipeline scripts.

1. UTF-8 stdout/stderr — Windows consoles default to cp1252, so the first
   emoji ``print`` (e.g. in ``data_download.py``) crashed with
   ``UnicodeEncodeError`` unless the user set ``PYTHONUTF8=1``. Importing this
   module reconfigures the streams (best effort), so every script only needs
   ``import src._compat`` at the top before any other local import.
2. Repo-root-anchored input lookup — scripts are documented to run from the
   repository root, but resolving inputs via :func:`data_path` (current
   directory first, repo root as fallback) keeps them working when invoked
   from elsewhere, without breaking tests that ``chdir`` into a tmp dir.
3. Logging setup — :func:`configure_logging` must be called from a ``main``
   function rather than at import time, so importing a module never mutates
   the root logger of the host application.
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

from src.config import ROOT


def ensure_utf8_stdout() -> None:
    """Force UTF-8 on stdout/stderr so emoji prints never raise on Windows."""
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:  # pragma: no cover - depends on the host stream
            pass


def configure_logging(level: int = logging.INFO, fmt: str | None = None) -> None:
    """Install the pipeline log format. Call from ``main``, never at import."""
    logging.basicConfig(
        level=level,
        format=fmt or "%(asctime)s | %(name)s | %(levelname)s | %(message)s",
    )


def data_path(name: str) -> str:
    """Resolve an input artifact, checking cwd, then ``CERNDATA_DIR``, then the repo root.

    ``os.path.join`` with an absolute ``name`` returns ``name`` unchanged, so
    absolute paths pass through untouched. The ``CERNDATA_DIR`` hop matters in
    the container, where ``src/data_download.py`` writes the dataset to a mounted
    volume but every reader would otherwise look in the image's read-only source
    tree and fail with FileNotFoundError.
    """
    if os.path.isabs(name):
        return name
    candidates = [os.path.join(os.getcwd(), name)]
    data_dir = os.environ.get("CERNDATA_DIR")
    if data_dir:
        candidates.append(os.path.join(data_dir, name))
    candidates.append(os.path.join(str(ROOT), name))
    for candidate in candidates:
        if os.path.exists(candidate):
            return candidate
    # Nothing exists yet: return the conventional location so the error message
    # points at where the file is expected.
    return candidates[0]


def repo_root() -> Path:
    """Absolute path of the repository root (tests anchor their fixtures here)."""
    return ROOT


ensure_utf8_stdout()  # run on import, before any emoji print below/importer
