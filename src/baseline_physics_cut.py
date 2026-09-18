import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src._compat  # noqa: F401  (UTF-8 stdout guard; must stay before prints)
from src._compat import data_path

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import time

# Config: centralise the few hard-coded experiment knobs (replaces magic numbers
# scattered across scripts). Full YAML config is overkill for this prototype;
# keep it importable with zero new dependencies.
CONFIG = {
    'mass_window': (80.0, 100.0),
}


def run_physics_baseline(csv="Dimuon_DoubleMu.csv"):
    print("🔬 [EXPERIMENT: PURE PHYSICS BASELINE]")
    print("Deterministic 80<M<100 cut vs itself = 100% by construction (ceiling).\n")

    try:
        data = pd.read_csv(data_path(csv))  # cwd first, repo root fallback
    except FileNotFoundError:
        print("❌ Dataset not found. Run python src/data_download.py first.")
        raise SystemExit(1)

    data = data.dropna()
    if 'M' not in data.columns:
        raise ValueError("Dataset missing required invariant mass column 'M'.")
    print(f"✅ Loaded {len(data)} events.\n")
    lo, hi = CONFIG['mass_window']
    y_true = data['M'].apply(lambda x: 1 if lo < x < hi else 0)

    start_time = time.time()
    y_pred_naive = data['M'].apply(lambda x: 1 if lo < x < hi else 0)
    execution_time = time.time() - start_time

    acc = accuracy_score(y_true, y_pred_naive)
    print("==========================================")
    print(f"🎯 Pure Mass (M) Cut Accuracy: {acc*100:.4f}% (tautology: cut == label)")
    print(f"⏱️ Cut Execution Time: {execution_time:.4f} seconds")
    print("==========================================\n")

    print("[Classification Report]")
    print(classification_report(y_true, y_pred_naive, target_names=["Background", "Signal"]))

    print("\n📝 CONCLUSION:")
    print("This 100% is the ceiling BY CONSTRUCTION (same rule defines labels).")
    print("An ML model on (pt, eta, phi) scoring ~99% is reconstructing M geometrically,")
    print("not discovering new physics. Compare models via AUROC/AUPRC in results/metrics.md.")


if __name__ == '__main__':
    run_physics_baseline()
