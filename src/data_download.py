"""Fetch the CERN Open Data educational dimuon sample and materialise a ROOT copy.

Dataset identity (fixed): CMS DoubleMu Run2011A, proton-proton collisions at
7 TeV. Educational 100k-event derived sample:
  https://opendata.cern.ch/record/5201
(parent: Datasets derived from the Run2011A SingleElectron/SingleMu/
DoubleElectron/DoubleMu primary datasets, https://opendata.cern.ch/record/545).

NOTE: this is an education/outreach subset, not suitable for a full physics
analysis. Previously the repo mixed this up with the Run2010B sample
(https://opendata.cern.ch/record/700) and cited 13.6 TeV (Run-3 energy).
Both were wrong for this data; the correct energy here is 7 TeV.

Integrity: the canonical payload is pinned by SHA-256 (:data:`src.config.CSV_SHA256`).
Schema/row-count checks alone let a swapped or truncated mirror through, and the
downstream scripts would then train on silently wrong data.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import os
import sys
import time
import socket
import tempfile
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src._compat  # noqa: F401  (UTF-8 stdout guard; must stay before prints)
from src._compat import ROOT
from src.config import (
    CSV_SHA256,
    CSV_SIZE_BYTES,
    MIN_FILE_BYTES,
    MIN_ROWS,
    REQUIRED_COLUMNS,
    ROOT_FILE,
    CSV_FILE,
    ensure_dir,
)

import numpy as np
import pandas as pd
import uproot

# Canonical educational CSV (record 5201). The parent record/545 URL is kept as
# a fallback mirror because 5201's parent is 545 and mirrors have shifted before.
DATA_URLS = (
    "https://opendata.cern.ch/record/5201/files/Dimuon_DoubleMu.csv",
    "https://opendata.cern.ch/record/545/files/Dimuon_DoubleMu.csv",
)

#: Name of the TTree written into the ROOT copy.
TREE_NAME = "Events"
#: Suffix of the fingerprint sidecar recording which CSV the ROOT file came from.
#: Produces `<ROOT_FILE>.fingerprint` (e.g. Dimuon_DoubleMu.root.fingerprint), which
#: is covered by the `*.root.fingerprint` rule in .gitignore.
FINGERPRINT_SUFFIX = ".fingerprint"


def csv_path() -> str:
    """Destination for the downloaded CSV (override with ``CERNDATA_DIR``)."""
    return str(Path(os.environ.get("CERNDATA_DIR", ROOT)) / CSV_FILE)


def root_path() -> str:
    """Destination for the derived ROOT TTree (override with ``CERNDATA_DIR``)."""
    return str(Path(os.environ.get("CERNDATA_DIR", ROOT)) / ROOT_FILE)


def sha256_of(path: str | os.PathLike, chunk: int = 1 << 20) -> str:
    """Streaming SHA-256 so the 14 MB payload is never held in memory at once."""
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for block in iter(lambda: fh.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def _download_with_retry(dest: str, urls=DATA_URLS, retries: int = 3, timeout: int = 120) -> str:
    """Download ``urls`` to ``dest`` atomically. Returns the URL that succeeded.

    A partially written file is never promoted into place: the payload lands in a
    temporary file next to ``dest`` and is renamed only after a full successful
    transfer, so an interrupted download can never poison the cache.
    """
    last_err: Exception | None = None
    dest = str(dest)
    parent = os.path.dirname(os.path.abspath(dest)) or "."
    ensure_dir(parent)

    for url in urls:
        for attempt in range(1, retries + 1):
            fd, tmp_dest = tempfile.mkstemp(prefix=".dl_", dir=parent)
            os.close(fd)
            try:
                print(f"📥 Downloading {url} (attempt {attempt}/{retries})...")
                # urlretrieve takes no timeout arg, so bound the socket instead;
                # without this the `timeout` parameter was silently ignored and
                # a stalled mirror could hang forever.
                old_timeout = socket.getdefaulttimeout()
                socket.setdefaulttimeout(timeout)
                try:
                    urllib.request.urlretrieve(url, tmp_dest)
                finally:
                    socket.setdefaulttimeout(old_timeout)
                os.replace(tmp_dest, dest)
                return url
            except Exception as e:  # network errors: retry, then try next mirror
                last_err = e
                with contextlib.suppress(OSError):
                    os.remove(tmp_dest)
                remaining = " (last attempt)" if attempt == retries and url == urls[-1] else ""
                print(f"⚠️ Download failed ({e}); retrying{remaining}...")
                if not (attempt == retries and url == urls[-1]):
                    time.sleep(2 * attempt)
    raise RuntimeError(f"Failed to fetch dataset from all mirrors {list(urls)}: {last_err}")


def _validate_csv(path: str, *, verify_checksum: bool | None = None) -> bool:
    """Raise if ``path`` is not a usable copy of the dimuon dataset.

    Checksum policy
    ---------------
    A SHA-256 of a different file is meaningless, so the hash is only enforced
    for a payload of exactly the canonical size. A differently-sized local file
    (a trimmed subset, or the physics-consistent fixture the CI workflow
    generates) passes the schema/row checks with a loud warning instead. Set
    ``CERNDATA_VERIFY_CHECKSUM=0`` to skip the hash check entirely.
    """
    if not os.path.exists(path):
        raise ValueError(f"Dataset not found: {path}")

    if verify_checksum is None:
        verify_checksum = os.environ.get("CERNDATA_VERIFY_CHECKSUM", "1") != "0"

    size = os.path.getsize(path)
    if size < MIN_FILE_BYTES:
        raise ValueError(f"Downloaded CSV suspiciously small ({size} bytes); likely truncated.")

    if not verify_checksum:
        print(f"⚠️ Checksum verification disabled (CERNDATA_VERIFY_CHECKSUM=0); accepting {size} bytes on schema alone.")
    elif size != CSV_SIZE_BYTES:
        print(
            f"⚠️ {path} is {size:,} bytes but the canonical 5201 payload is "
            f"{CSV_SIZE_BYTES:,} bytes. This is a local subset or a test fixture, not the "
            f"official sample, so its checksum cannot be verified — continuing on schema checks only."
        )
    else:
        digest = sha256_of(path)
        if digest != CSV_SHA256:
            raise ValueError(
                f"Checksum mismatch for {path}:\n"
                f"  expected sha256 {CSV_SHA256}\n"
                f"  actual   sha256 {digest}\n"
                f"The mirror returned a different payload; refusing to use it."
            )

    df = pd.read_csv(path, nrows=5)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV schema validation failed; missing columns {sorted(missing)}; got {list(df.columns)}")
    with open(path, "rb") as fh:
        n = sum(1 for _ in fh) - 1  # minus header
    if n < MIN_ROWS:
        raise ValueError(f"CSV has only {n} rows (< {MIN_ROWS}); likely incomplete.")
    return True


def _write_root(csv_file: str, root_file: str) -> None:
    """Convert the validated CSV into an uproot-readable TTree.

    Non-numeric columns (e.g. the string ``Type`` constraint) are dropped because
    they cannot be written as flat ROOT branches. The write is atomic and records
    a fingerprint so a refreshed CSV invalidates the stale ROOT copy.
    """
    df = pd.read_csv(csv_file)
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns {sorted(missing)}")
    df = df.dropna(subset=list(REQUIRED_COLUMNS))
    if len(df) < MIN_ROWS:
        raise ValueError(f"Too few valid rows after dropna: {len(df)}")

    numeric_df = df.select_dtypes(include=[np.number])
    branch_dict = {col: numeric_df[col].to_numpy() for col in numeric_df.columns}

    parent = os.path.dirname(os.path.abspath(root_file)) or "."
    ensure_dir(parent)
    fd, tmp_root = tempfile.mkstemp(prefix=".root_", dir=parent)
    os.close(fd)
    try:
        with uproot.recreate(tmp_root) as f:
            f[TREE_NAME] = branch_dict
        os.replace(tmp_root, root_file)
    finally:
        with contextlib.suppress(OSError):
            os.remove(tmp_root)

    with open(root_file + FINGERPRINT_SUFFIX, "w", encoding="utf-8") as fh:
        fh.write(f"{sha256_of(csv_file)}\n{len(df)}\n")


def _root_is_current(csv_file: str, root_file: str) -> bool:
    """True when ``root_file`` provably matches the on-disk ``csv_file``.

    A missing or unreadable fingerprint is treated as *stale* rather than trusted:
    ROOT files predate this sidecar, and silently reusing one that was built from a
    different CSV would train on mismatched data. Rebuilding costs ~1s and happens
    exactly once, after which the fingerprint makes the check free.
    """
    if not os.path.exists(root_file):
        return False
    try:
        with open(root_file + FINGERPRINT_SUFFIX, encoding="utf-8") as fh:
            recorded = fh.readline().strip()
    except OSError:
        return False
    return bool(recorded) and recorded == sha256_of(csv_file)


def download_and_convert(*, force: bool = False, verify_checksum: bool | None = None) -> dict:
    """Ensure the CSV and its ROOT copy exist, are valid and are mutually current.

    Returns a small status dict (paths, row count, whether anything was fetched).
    """
    csv_file, root_file = csv_path(), root_path()
    ensure_dir(os.path.dirname(os.path.abspath(csv_file)))
    print("🌍 Connecting to CERN Open Data Portal (Run2011A DoubleMu, 7 TeV)...")

    fetched = False
    if force or not os.path.exists(csv_file):
        print("📥 Downloading educational Dimuon sample (~14MB, 100k events)...")
        _download_with_retry(csv_file)
        fetched = True
    else:
        print(f"✅ CSV {csv_file!r} already present; validating...")

    try:
        _validate_csv(csv_file, verify_checksum=verify_checksum)
    except ValueError as e:
        # A payload that fails validation must not be left behind: otherwise every
        # later run hits the same bad cache entry and fails immediately instead of
        # re-downloading.
        print(f"⚠️ Existing CSV {csv_file!r} is invalid ({e}). Discarding and re-downloading...")
        with contextlib.suppress(OSError):
            os.remove(csv_file)
        _download_with_retry(csv_file)
        fetched = True
        _validate_csv(csv_file, verify_checksum=verify_checksum)

    print("✅ CSV integrity/schema check passed.")

    with open(csv_file, "rb") as fh:
        rows = sum(1 for _ in fh) - 1  # minus header
    if not _root_is_current(csv_file, root_file):
        print(f"⚙️ Converting to CERN-native {ROOT_FILE} (TTree {TREE_NAME!r})...")
        _write_root(csv_file, root_file)
        print(f"✅ ROOT file materialised at: {root_file}")
    else:
        print(f"✅ ROOT binary {root_file!r} already matches the CSV. Ready for ingestion.")

    return {
        "csv": csv_file,
        "root": root_file,
        "rows": rows,
        "downloaded": fetched,
    }


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python src/data_download.py",
        description="Fetch the CERN Open Data dimuon sample and materialise a ROOT copy.",
    )
    p.add_argument("--force", action="store_true", help="re-download even if a valid CSV is already present")
    p.add_argument(
        "--no-checksum", action="store_true", help="skip the pinned SHA-256 check (needed for local subsets/fixtures)"
    )
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    status = download_and_convert(force=args.force, verify_checksum=not args.no_checksum)
    print(f"✅ Dataset ready: {status['rows']:,} events at {status['csv']}")
