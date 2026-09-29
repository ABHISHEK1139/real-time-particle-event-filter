"""Unsupervised Trigger Anomaly Detection Baseline.

Trains an Isolation Forest anomaly detector exclusively on collision kinematics
(pt1, pt2, eta1, eta2, phi1, phi2) WITHOUT using invariant mass labels.
Demonstrates trigger-level new physics discovery by flagging resonant events
as kinematic anomalies from the background continuum.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src._compat  # noqa: F401  (UTF-8 stdout guard)

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import auc, average_precision_score, roc_curve
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src._compat import data_path
from src.config import (
    ANOMALY_JOBLIB_FILE,
    BUDGET,
    CSV_FILE,
    FEATURES,
    RANDOM_STATE,
    REQUIRED_COLUMNS,
    labels_from_mass,
    output_path,
    rates_at_threshold,
    threshold_for_budget,
)


def train_anomaly_detector(X_train, budget: float = BUDGET):
    """Fits an Isolation Forest anomaly detector on background/unlabeled kinematics.

    ``contamination`` only sets the score offset used by ``IsolationForest.predict``;
    the benchmark below ranks on ``decision_function``, so the operative budget is
    applied at calibration time by :func:`threshold_for_budget`. It is still set so
    the persisted artifact has a sensible ``offset_`` for ``predict()`` callers.
    """
    print("🌲 Fitting Isolation Forest Anomaly Detector on kinematic features...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train)
    iso = IsolationForest(n_estimators=100, contamination=budget, random_state=RANDOM_STATE, n_jobs=-1)
    s = time.perf_counter()
    iso.fit(X_scaled)
    print(f"✅ Anomaly Detector trained in {time.perf_counter() - s:.2f} seconds.")
    return iso, scaler


def anomaly_scores(iso: IsolationForest, scaler: StandardScaler, X) -> np.ndarray:
    """Higher = more anomalous. Isolation Forest's decision_function is inverted."""
    return -iso.decision_function(scaler.transform(X))


def evaluate_anomaly_detector(iso, scaler, X_val, y_val, X_test, y_test, budget: float = BUDGET):
    """
    Calibrates threshold on VALIDATION background, evaluates frozen on TEST.
    In Isolation Forest, decision_function returns negative for anomalies.
    We negate so higher score = higher anomaly probability (signal candidate).
    """
    print(f"\n📏 Evaluating Anomaly Detection (Fixed Budget = {budget * 100}%)")
    scores_val = anomaly_scores(iso, scaler, X_val)
    scores_test = anomaly_scores(iso, scaler, X_test)

    y_val = np.asarray(y_val)
    y_test = np.asarray(y_test)

    threshold = threshold_for_budget(scores_val[y_val == 0], budget)
    r = rates_at_threshold(scores_test, y_test, threshold)

    fpr, tpr, _ = roc_curve(y_test, scores_test)
    auroc = float(auc(fpr, tpr))
    auprc = float(average_precision_score(y_test, scores_test))

    print(f"   => Anomaly Score Threshold: >= {threshold:.6g} (fit on VALIDATION)")
    print(f"   => TEST Background Retained: {r['bg_retained'] * 100:.2f}% (Target was {budget * 100}%)")
    print(f"   => TEST Unsupervised Signal Efficiency: {r['sig_eff'] * 100:.2f}%")
    print(f"   => TEST AUROC: {auroc:.4f} | AUPRC: {auprc:.4f}")

    _plot_roc(fpr, tpr, auroc)

    return {
        "threshold": threshold,
        "bg_retained": r["bg_retained"],
        "sig_eff": r["sig_eff"],
        "auroc": auroc,
        "auprc": auprc,
    }


def _plot_roc(fpr, tpr, auroc) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(fpr, tpr, color="forestgreen", lw=2, label=f"Isolation Forest (AUROC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Random Chance")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title(
        "Unsupervised Anomaly Detection ROC\n(Trained without mass labels, evaluated on TEST)",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="lower right")
    ax.grid(True, linestyle="--", alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path("plots", "7_anomaly_detection_roc.png"), dpi=150)
    plt.close(fig)
    print("✅ ROC plot saved to plots/7_anomaly_detection_roc.png")


def write_report(metrics: dict, budget: float = BUDGET) -> Path:
    path = output_path("results", "anomaly_metrics.md")
    with open(path, "w", encoding="utf-8") as f:
        f.write("# Unsupervised Anomaly Detection Benchmark\n\n")
        f.write("> Evaluates Isolation Forest trained exclusively on continuum background kinematics\n")
        f.write("> without invariant mass labels. Models new physics resonant discovery at trigger level.\n\n")
        f.write(f"**Target Background Acceptance Budget**: {budget * 100}%\n\n")
        f.write("| Model | Training Paradigm | Signal Efficiency @5% bg | AUROC | AUPRC |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(
            f"| Isolation Forest | Unsupervised (Background only) | **{metrics['sig_eff'] * 100:.2f}%** | "
            f"{metrics['auroc']:.4f} | {metrics['auprc']:.4f} |\n\n"
        )
        f.write(f"TEST Background Retained: {metrics['bg_retained'] * 100:.2f}% (Target: {budget * 100}%).\n\n")
        f.write(
            "> Conclusion: Without ever seeing an invariant mass label, the unsupervised detector isolates "
            f"{metrics['sig_eff'] * 100:.2f}% of Z-boson candidate events at 5% false positive rate.\n"
        )

    json_path = output_path("results", "anomaly_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump({"budget": budget, "model": "IsolationForest", **metrics}, f, indent=2)
        f.write("\n")
    return path


def run_anomaly_benchmark(csv_file: str = CSV_FILE, budget: float = BUDGET) -> dict:
    csv_path = data_path(csv_file)
    if not Path(csv_path).exists():
        print(f"❌ Dataset {csv_file} not found. Run python src/data_download.py first.")
        raise SystemExit(1)

    df = pd.read_csv(csv_path).dropna()
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    y = labels_from_mass(df["M"].to_numpy())
    X = df[FEATURES]

    # Stratified 60/20/20 train/val/test
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.25, random_state=RANDOM_STATE, stratify=y_temp
    )

    # Train on background collisions only (modeling normal collision physics)
    X_train_bg = X_train[y_train == 0]
    iso, scaler = train_anomaly_detector(X_train_bg, budget=budget)
    metrics = evaluate_anomaly_detector(iso, scaler, X_val, y_val, X_test, y_test, budget=budget)

    print(f"✅ Anomaly metrics written to {write_report(metrics, budget=budget)}")
    joblib.dump({"model": iso, "scaler": scaler}, str(output_path(ANOMALY_JOBLIB_FILE)))
    print(f"💾 Anomaly model saved to {ANOMALY_JOBLIB_FILE}")
    return metrics


if __name__ == "__main__":
    run_anomaly_benchmark()
