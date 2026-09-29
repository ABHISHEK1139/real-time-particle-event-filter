"""Generate a physics-consistent stand-in for the CERN DoubleMu sample.

Used by CI (and by anyone who wants to run the pipeline without a 14 MB
download). The invariant mass is *computed* from the kinematics, so the
signal/background structure matches the real sample — the previous CI fixture
drew a random ``M`` unrelated to ``pt/eta/phi``, which only proved that the code
executes, not that the label boundary is learnable.

    python tests/make_mock_dataset.py --rows 1000 --out Dimuon_DoubleMu.csv
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

# Allow `python tests/make_mock_dataset.py` from a bare checkout. `conftest` is
# imported as a top-level module because an unrelated `tests` package in
# site-packages shadows this directory when it is imported as a package.
_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))
if str(_HERE.parent) not in sys.path:
    sys.path.insert(0, str(_HERE.parent))

import src._compat  # noqa: E402  (UTF-8 stdout guard, before the emoji print)
from conftest import synthetic_dimuon_frame  # noqa: E402

DEFAULT_MOCK_ROWS = 1000
DEFAULT_OUT = "Dimuon_DoubleMu.csv"


def make_mock_dataset(rows: int = DEFAULT_MOCK_ROWS, seed: int = 42) -> pd.DataFrame:
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
    signal = int(((df["M"] > 80) & (df["M"] < 100)).sum())
    print(f"✅ Wrote {len(df):,} physics-consistent events to {args.out} ({signal:,} inside the Z mass window)")
