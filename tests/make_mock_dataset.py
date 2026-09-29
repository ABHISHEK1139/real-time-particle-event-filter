"""Generate a physics-consistent stand-in for the CERN DoubleMu sample.

Used by CI and by the container smoke test, and by anyone who wants to run the
pipeline without a 14 MB download:

    python tests/make_mock_dataset.py --rows 1000 --out Dimuon_DoubleMu.csv

The generator lives in :mod:`src.synthetic`, not in ``conftest.py``, so this
script has no pytest dependency. That matters: the Docker image installs only
``requirements.txt``, and an earlier version imported conftest and died with
``ModuleNotFoundError: No module named 'pytest'`` inside the container.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

# Allow `python tests/make_mock_dataset.py` from a bare checkout. The generator
# is imported as `src.synthetic` rather than `tests.conftest` because an
# unrelated `tests` package in site-packages shadows this directory when it is
# imported as a package, and because conftest pulls in pytest.
_HERE = Path(__file__).resolve().parent
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import src._compat  # noqa: E402  (UTF-8 stdout guard, before the emoji print)
from src.synthetic import synthetic_dimuon_frame  # noqa: E402

DEFAULT_MOCK_ROWS = 1000
DEFAULT_OUT = "Dimuon_DoubleMu.csv"
SIGNAL_MIN, SIGNAL_MAX = 80.0, 100.0


def make_mock_dataset(rows: int = DEFAULT_MOCK_ROWS, seed: int = 42):
    """Return a physics-consistent frame with the real sample's 21-column schema."""
    return synthetic_dimuon_frame(rows, seed=seed, full_schema=True)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--rows", type=int, default=DEFAULT_MOCK_ROWS)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--out", default=DEFAULT_OUT)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    df = make_mock_dataset(rows=args.rows, seed=args.seed)
    Path(args.out).write_text(df.to_csv(index=False), encoding="utf-8")
    signal = int(((df["M"] > SIGNAL_MIN) & (df["M"] < SIGNAL_MAX)).sum())
    print(f"✅ Wrote {len(df):,} physics-consistent events to {args.out} ({signal:,} inside the Z mass window)")
