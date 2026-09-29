"""Speed benchmark: XGBoost CPU vs XGBoost GPU (apples-to-apples) + RF reference.

Old script compared RandomForest-CPU vs XGBoost-GPU and called the gap 'GPU
speedup' — different algorithms, so that was operational, not a GPU claim.
Now: same XGBoost hyperparams on CPU and (if available) CUDA, with warmup,
synchronisation, and median/p95 over repeats. RF numbers are kept only as a
separate architecture reference. Output goes to plots/speed_comparison.png
(not repo root) so committed figures don't go stale.

Correctness note: the benchmark must NOT rewrite the device parameter of the
model it is timing. Doing so silently downgraded the "GPU" run to CPU, which is
why the GPU row used to be indistinguishable from the CPU row. Each variant is
now built and timed on exactly the device its name claims.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src._compat  # noqa: F401  (UTF-8 stdout guard; must stay before prints)

import matplotlib
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt

from src._compat import data_path
from src.config import (
    CSV_FILE,
    FEATURES,
    RANDOM_STATE,
    REQUIRED_COLUMNS,
    labels_from_mass,
    output_path,
)

N_REPEATS = 20
WARMUP = 5
#: Fixed benchmark slice so every architecture sees byte-identical input.
BENCH_ROWS = 5000
XGB_PARAMS = {"n_estimators": 100, "learning_rate": 0.1, "tree_method": "hist", "random_state": RANDOM_STATE}


def cuda_available() -> bool:
    """True when torch can see a CUDA device. Never raises."""
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def _synchronize() -> None:
    """Block until queued accelerator work completes, so timings are not lies."""
    if not cuda_available():
        return
    try:
        import torch

        torch.cuda.synchronize()
    except Exception:
        pass


def timed_predict(model, X, repeats: int = N_REPEATS, warmup: int = WARMUP) -> dict:
    """Median/p95/mean wall-clock latency in ms, after warmup."""
    for _ in range(warmup):
        model.predict(X)
    _synchronize()
    lat = []
    for _ in range(repeats):
        s = time.perf_counter()
        model.predict(X)
        _synchronize()
        lat.append((time.perf_counter() - s) * 1000)
    return {"median": float(np.median(lat)), "p95": float(np.percentile(lat, 95)), "mean": float(np.mean(lat))}


def benchmark(name: str, model, X_train, y_train, X_bench, y_bench) -> dict | None:
    """Fit + time one architecture, preserving its configured device."""
    try:
        t0 = time.perf_counter()
        model.fit(X_train, y_train)
        _synchronize()
        train_t = time.perf_counter() - t0

        preds = model.predict(X_bench)
        acc = accuracy_score(y_bench, preds)
        try:
            auroc = roc_auc_score(y_bench, model.predict_proba(X_bench)[:, 1])
        except Exception:
            auroc = float("nan")
        lat = timed_predict(model, X_bench)
    except Exception as e:
        # A CUDA failure here is an expected, non-fatal environment limitation.
        print(f"  ⚠️ {name} unavailable, skipping (not a failure): {e}")
        return None

    print(
        f"  ➜ {name}: train {train_t:.3f}s | infer median {lat['median']:.3f}ms "
        f"p95 {lat['p95']:.3f}ms | acc {acc * 100:.2f}% AUROC {auroc:.4f}"
    )
    return {"name": name, "accuracy": acc, "auroc": auroc, "train_s": train_t, **lat}


def main(bench_rows: int = BENCH_ROWS, repeats: int = N_REPEATS) -> list[dict]:
    print("🚀 Loading CERN Dimuon Collision Data (Run2011A DoubleMu, 7 TeV educational sample)...")
    csv_path = data_path(CSV_FILE)
    if not Path(csv_path).exists():
        print(f"❌ Cannot find {CSV_FILE!r}. Run python src/data_download.py first.")
        raise SystemExit(1)

    data = pd.read_csv(csv_path).dropna()
    missing = set(REQUIRED_COLUMNS) - set(data.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    print("⚡ Preprocessing and creating labels (1 = 80<M<100, else 0)...")
    data["label"] = labels_from_mass(data["M"].to_numpy())
    X = data[FEATURES]
    y = data["label"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y)
    print(f"✅ Data prepared (stratified): {len(X_train)} train, {len(X_test)} test.\n")

    n_bench = min(bench_rows, len(X_test))
    X_bench = X_test.iloc[:n_bench]  # identical slice for every architecture
    y_bench = y_test.iloc[:n_bench]
    results: list[dict] = []

    print("🟢 RF baseline (CPU, architecture reference only)")
    rf = benchmark(
        "RF-100 (CPU ref)",
        RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE, n_jobs=-1),
        X_train,
        y_train,
        X_bench,
        y_bench,
    )
    if rf:
        results.append(rf)

    print("🟡 XGBoost CPU (same hyperparams as GPU run)")
    cpu = benchmark("XGB CPU", xgb.XGBClassifier(**XGB_PARAMS, n_jobs=-1), X_train, y_train, X_bench, y_bench)
    if cpu:
        results.append(cpu)

    print("🔴 XGBoost GPU (same hyperparams; skipped cleanly if no CUDA)")
    if cuda_available():
        # Timed on the GPU it claims: the device parameter is left untouched.
        gpu = benchmark("XGB GPU", xgb.XGBClassifier(**XGB_PARAMS, device="cuda"), X_train, y_train, X_bench, y_bench)
        if gpu:
            results.append(gpu)
    else:
        print("  ⚠️ No CUDA device visible, skipping the GPU variant (not a failure).")

    if not results:
        raise RuntimeError("No architecture completed; nothing to plot.")

    _plot(results, n_bench, repeats)
    return results


def _plot(results: list[dict], n_bench: int, repeats: int) -> None:
    print("🔥 Plotting inference latency (median) vs accuracy...")
    names = [r["name"] for r in results]
    meds = [r["median"] for r in results]
    accs = [r["accuracy"] * 100 for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = "tab:red"
    ax1.set_xlabel("Model", fontsize=12, fontweight="bold")
    ax1.set_ylabel(f"Median inference latency, {n_bench} events (ms)", color=color, fontsize=12, fontweight="bold")
    ax1.bar(names, meds, color=color, alpha=0.6, width=0.4)
    ax1.tick_params(axis="y", labelcolor=color)
    plt.setp(ax1.get_xticklabels(), rotation=15, ha="right")

    ax2 = ax1.twinx()
    color = "tab:blue"
    ax2.set_ylabel("Accuracy (%)", color=color, fontsize=12, fontweight="bold")
    ax2.plot(names, accs, color=color, marker="o", markersize=10, linewidth=3)
    ax2.tick_params(axis="y", labelcolor=color)

    ax1.set_title(f"Inference latency (median of {repeats}, warmed up) vs Accuracy", fontsize=14, fontweight="bold")
    fig.tight_layout()
    target = output_path("plots", "speed_comparison.png")
    fig.savefig(target)
    plt.close(fig)
    print(f"✅ Saved to {target} (warmup + median/p95; identical data slice; each variant timed on its own device)")


if __name__ == "__main__":
    main()
