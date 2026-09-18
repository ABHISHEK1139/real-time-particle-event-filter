"""Shared runtime guards for all pipeline scripts.

1. UTF-8 stdout/stderr — Windows consoles default to cp1252, so the first
   emoji ``print`` (e.g. in ``data_download.py``) crashed with
   ``UnicodeEncodeError`` unless the user set ``PYTHONUTF8=1``. Importing this
   module reconfigures the streams (best effort), so every script only needs
   ``import src._compat`` at the top before any other local import.
2. Repo-root-anchored input lookup — scripts are documented to run from the
   repository root, but resolving inputs via :func:`data_path` (current
   directory first, repo root as fallback) keeps them working when invoked
   from elsewhere, without breaking tests that ``chdir`` into a tmp dir
   (outputs intentionally stay cwd-relative so tests never pollute the repo).
"""
import os
import sys
from pathlib import Path


def ensure_utf8_stdout() -> None:
    for stream in (sys.stdout, sys.stderr):
        try:
            if stream is not None and hasattr(stream, "reconfigure"):
                stream.reconfigure(encoding="utf-8", errors="replace")
        except Exception:
            pass


ensure_utf8_stdout()  # run on import, before any emoji print below/importer

ROOT = Path(__file__).resolve().parent.parent


def data_path(name: str) -> str:
    """Return ``name`` from the cwd if present, else from the repo root.

    ``os.path.join`` with an absolute ``name`` returns ``name`` unchanged,
    so absolute paths pass through untouched.
    """
    here = os.path.join(os.getcwd(), name)
    if os.path.exists(here):
        return here
    return str(ROOT / name)
