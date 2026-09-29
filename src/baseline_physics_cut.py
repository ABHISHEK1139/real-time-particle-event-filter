"""Deterministic mass-cut reference: the accuracy ceiling any filter can reach.

This script is intentionally a tautology — it applies the 80<M<100 cut and scores
it against the same cut used to build the labels — and it exists only to state
that ceiling out loud. Use it to sanity-check the supervised pipeline in
``src/train_model.py``; compare models through AUROC/AUPRC in ``results/metrics.md``.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src._compat  # noqa: F401  (UTF-8 stdout guard; must stay before prints)

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report

from src._compat import data_path
from src.config import (
    CSV_FILE,
    MASS_WINDOW,
    REQUIRED_COLUMNS,
)

# Config: centralise the few hard-coded experiment knobs (replaces magic numbers
# scattered across scripts). Full YAML config is overkill for this prototype;
# keep it importable with zero new dependencies.
CONFIG = {
    "mass_window": MASS_WINDOW,
}


def run_physics_baseline(csv: str = CSV_FILE) -> dict:
    print("🔬 [EXPERIMENT: PURE PHYSICS BASELINE]")
    print("Deterministic 80<M<100 cut vs itself = 100% by construction (ceiling).\n")

    path = data_path(csv)
    if not Path(path).exists():
        print("❌ Dataset not found. Run python src/data_download.py first.")
        raise SystemExit(1)

    data = pd.read_csv(path).dropna()
    missing = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")
    print(f"✅ Loaded {len(data)} events.\n")

    lo, hi = CONFIG["mass_window"]
    mass = data["M"].to_numpy()
    y_true = ((mass > lo) & (mass < hi)).astype(int)

    # Vectorised: the previous version used DataFrame.apply, which is ~50x slower
    # for a two-comparison predicate and dominated the reported runtime.
    start_time = time.perf_counter()
    y_pred_naive = ((mass > lo) & (mass < hi)).astype(int)
    execution_time = time.perf_counter() - start_time

    acc = accuracy_score(y_true, y_pred_naive)
    print("==========================================")
    print(f"🎯 Pure Mass (M) Cut Accuracy: {acc * 100:.4f}% (tautology: cut == label)")
    print(f"⏱️ Cut Execution Time: {execution_time:.6f} seconds")
    print("==========================================\n")

    print("[Classification Report]")
    print(classification_report(y_true, y_pred_naive, target_names=["Background", "Signal"]))

    print("\n📝 CONCLUSION:")
    print("This 100% is the ceiling BY CONSTRUCTION (same rule defines labels).")
    print("An ML model on (pt, eta, phi) scoring ~99% is reconstructing M geometrically,")
    print("not discovering new physics. Compare models via AUROC/AUPRC in results/metrics.md.")
    return {"accuracy": float(acc), "execution_time_s": execution_time, "events": len(data)}


if __name__ == "__main__":
    run_physics_baseline()
