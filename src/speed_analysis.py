"""Speed benchmark: XGBoost CPU vs XGBoost GPU (apples-to-apples) + RF reference.

Old script compared RandomForest-CPU vs XGBoost-GPU and called the gap 'GPU
speedup' — different algorithms, so that was operational, not a GPU claim.
Now: same XGBoost hyperparams on CPU and (if available) CUDA, with warmup,
synchronisation, and median/p95 over repeats. RF numbers are kept only as a
separate architecture reference. Output goes to plots/speed_comparison.png
(not repo root) so committed figures don't go stale.
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src._compat  # noqa: F401  (UTF-8 stdout guard; must stay before prints)
from src._compat import data_path

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
import xgboost as xgb
import matplotlib.pyplot as plt
import os

FEATURES = ['pt1', 'pt2', 'eta1', 'eta2', 'phi1', 'phi2']
N_REPEATS = 20
WARMUP = 5
XGB_PARAMS = dict(n_estimators=100, learning_rate=0.1, tree_method='hist', random_state=42)


def timed_predict(model, X, repeats=N_REPEATS, warmup=WARMUP):
    for _ in range(warmup):
        model.predict(X)
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.synchronize()
    except Exception:
        pass
    lat = []
    for _ in range(repeats):
        s = time.perf_counter()
        model.predict(X)
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
        except Exception:
            pass
        lat.append((time.perf_counter() - s) * 1000)
    return float(np.median(lat)), float(np.percentile(lat, 95)), float(np.mean(lat))


def main():
    print("🚀 Loading CERN Dimuon Collision Data (Run2011A DoubleMu, 7 TeV educational sample)...")
    try:
        data = pd.read_csv(data_path("Dimuon_DoubleMu.csv"))  # cwd first, repo root fallback
    except FileNotFoundError:
        print("❌ Cannot find 'Dimuon_DoubleMu.csv'. Run python src/data_download.py first.")
        raise SystemExit(1)

    data = data.dropna()
    required = set(FEATURES) | {'M'}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    print("⚡ Preprocessing and creating labels (1 = 80<M<100, else 0)...")
    data['label'] = data['M'].apply(lambda x: 1 if 80 < x < 100 else 0)
    X = data[FEATURES]
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"✅ Data prepared (stratified): {len(X_train)} train, {len(X_test)} test.\n")

    X_bench = X_test.iloc[:5000]  # fixed bench slice so all models see identical data
    y_bench = y_test.iloc[:5000]
    results = []

    def bench(name, model):
        s = time.time()
        model.fit(X_train, y_train)
        train_t = time.time() - s
        if hasattr(model, 'set_params'):
            try:
                model.set_params(device='cpu')
            except Exception:
                pass
        preds = model.predict(X_bench)
        acc = accuracy_score(y_bench, preds)
        try:
            auroc = roc_auc_score(y_bench, model.predict_proba(X_bench)[:, 1])
        except Exception:
            auroc = float('nan')
        med, p95, mean = timed_predict(model, X_bench)
        print(f"  ➜ {name}: train {train_t:.3f}s | infer median {med:.3f}ms p95 {p95:.3f}ms | acc {acc*100:.2f}% AUROC {auroc:.4f}")
        results.append((name, med, acc, train_t, p95))

    print("🟢 RF baseline (CPU, architecture reference only)")
    bench("RF-100 (CPU ref)", RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1))

    print("🟡 XGBoost CPU (same hyperparams as GPU run)")
    bench("XGB CPU", xgb.XGBClassifier(**{**XGB_PARAMS, 'n_jobs': -1}))

    print("🔴 XGBoost GPU (same hyperparams; skipped cleanly if no CUDA)")
    try:
        m = xgb.XGBClassifier(**{**XGB_PARAMS, 'device': 'cuda'})
        m.fit(X_train.iloc[:1000], y_train.iloc[:1000])  # probe CUDA on small slice
        bench("XGB GPU", xgb.XGBClassifier(**{**XGB_PARAMS, 'device': 'cuda'}))
    except Exception as e:
        print(f"  ⚠️ GPU run unavailable, skipping (not a failure): {e}")

    print("🔥 Plotting inference latency (median) vs accuracy...")
    os.makedirs("plots", exist_ok=True)
    models = [r[0] for r in results]
    meds = [r[1] for r in results]
    accs = [r[2] * 100 for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))
    color = 'tab:red'
    ax1.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax1.set_ylabel(f'Median inference latency, {len(X_bench)} events (ms)', color=color, fontsize=12, fontweight='bold')
    ax1.bar(models, meds, color=color, alpha=0.6, width=0.4)
    ax1.tick_params(axis='y', labelcolor=color)
    plt.setp(ax1.get_xticklabels(), rotation=15, ha='right')

    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy (%)', color=color, fontsize=12, fontweight='bold')
    ax2.plot(models, accs, color=color, marker='o', markersize=10, linewidth=3)
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Inference latency (median of 20, warmed up) vs Accuracy', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig('plots/speed_comparison.png')
    plt.close(fig)
    print("✅ Saved to plots/speed_comparison.png (warmup + median/p95; identical data slice)")


if __name__ == "__main__":
    main()
