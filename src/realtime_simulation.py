"""Offline batch replay / real-time inference simulation (NOT a live stream).

Reads the static CSV in chunks, runs XGBoost batch inference, and reports:
separate data-prep / inference / post-processing timings (with warmup),
plus TP/FP/FN/TN against the known 80<M<100 label. The old script timed the
whole loop and called total_time/total_events 'average latency' — that is
replay throughput, not inference latency. Both are now reported honestly.
Also validates the model's actual device instead of assuming CUDA.
"""
import time
import pandas as pd
import numpy as np
import joblib
import logging
from termcolor import colored

logging.basicConfig(level=logging.INFO, format='%(asctime)s | streaming_engine | %(message)s')
FEATURES = ['pt1', 'pt2', 'eta1', 'eta2', 'phi1', 'phi2']


def _model_device_info(model):
    info = {}
    try:
        import xgboost as xgb
        info['xgboost_version'] = xgb.__version__
    except Exception:
        info['xgboost_version'] = 'unknown'
    try:
        params = model.get_params()
        info['model_device_param'] = params.get('device', params.get('tree_method', 'unknown'))
    except Exception:
        info['model_device_param'] = 'unknown'
    try:
        import torch
        info['cuda_available_torch'] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info['gpu_name'] = torch.cuda.get_device_name(0)
    except Exception:
        info['cuda_available_torch'] = 'unknown'
    return info


def simulate_realtime_stream(batch_size=50000, max_batches=None):
    print(colored("\n📡 OFFLINE BATCH REPLAY: CERN collision-data inference simulation...", "cyan", attrs=['bold']))
    print(colored("   (static CSV replay in chunks — no live queue/socket/Kafka/trigger interface)", "cyan"))

    try:
        model = joblib.load('z_boson_xgb_model.joblib')
        print(colored("✅ XGBoost Model Loaded successfully.", "green"))
    except FileNotFoundError:
        print(colored("❌ CRITICAL: z_boson_xgb_model.joblib not found. Run: python src/data_download.py && python src/train_model.py", "red"))
        raise SystemExit(1)

    for k, v in _model_device_info(model).items():
        print(f"   device info | {k}: {v}")

    try:
        full_data = pd.read_csv("Dimuon_DoubleMu.csv").dropna()
        features_only = full_data[FEATURES]
        y_true = ((full_data['M'] > 80) & (full_data['M'] < 100)).astype(int).values
    except FileNotFoundError:
        print(colored("❌ CRITICAL: Dimuon_DoubleMu.csv not found. Run python src/data_download.py first.", "red"))
        raise SystemExit(1)

    print(colored("🚨 REPLAY START: batched inference...\n", "yellow"))
    total_events = len(features_only)
    signals_detected = 0
    tp = fp = fn = tn = 0

    # Warmup (excluded from timings) to avoid counting CUDA/model init overhead
    warm = features_only.iloc[:min(1000, len(features_only))]
    try:
        model.predict(warm)
    except Exception:
        pass

    t_prep = t_inf = t_post = 0.0
    start_global = time.perf_counter()
    n_batches = 0
    for i in range(0, total_events, batch_size):
        if max_batches is not None and n_batches >= max_batches:
            break
        t0 = time.perf_counter()
        batch_features = features_only.iloc[i:i + batch_size]
        batch_true = y_true[i:i + batch_size]
        t1 = time.perf_counter()

        preds = model.predict(batch_features)
        t2 = time.perf_counter()

        signals_in_batch = int(np.sum(preds == 1))
        signals_detected += signals_in_batch
        tp += int(np.sum((preds == 1) & (batch_true == 1)))
        fp += int(np.sum((preds == 1) & (batch_true == 0)))
        fn += int(np.sum((preds == 0) & (batch_true == 1)))
        tn += int(np.sum((preds == 0) & (batch_true == 0)))
        t3 = time.perf_counter()

        t_prep += (t1 - t0)
        t_inf += (t2 - t1)
        t_post += (t3 - t2)
        n_batches += 1

        inf_ms = (t2 - t1) * 1000
        status = colored(f"[BATCH PROCESSED] {len(batch_features)} Events | Signals: {signals_in_batch}", "green", attrs=['bold'])
        logging.info(f"Batch {n_batches} | Inference: {inf_ms:.2f}ms | {status}")

    total_time = time.perf_counter() - start_global
    processed = min(total_events, (max_batches * batch_size) if max_batches else total_events)
    eff = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    print(colored("\n=============================================", "cyan"))
    print(colored("🏁 BATCH REPLAY SUMMARY (offline, not a live trigger)", "cyan", attrs=['bold']))
    print(colored(f"     Total Events Processed : {processed:,}", "white"))
    print(colored(f"     Z-Boson Predicted      : {signals_detected:,}", "white"))
    print(colored(f"     TP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,}", "white"))
    print(colored(f"     Signal efficiency (recall): {eff*100:.2f}% | FPR: {fpr*100:.2f}%", "white"))
    print(colored(f"     Wall-clock total       : {total_time:.4f}s", "white"))
    print(colored(f"     Replay throughput      : {processed/total_time:,.0f} events/s", "white"))
    print(colored(f"     Breakdown — prep: {t_prep:.3f}s | inference: {t_inf:.3f}s | post: {t_post:.3f}s", "white"))
    print(colored(f"     Pure inference latency : {(t_inf/processed)*1000:.6f} ms/event", "white"))
    print(colored("=============================================\n", "cyan"))


if __name__ == "__main__":
    try:
        simulate_realtime_stream()
    except KeyboardInterrupt:
        print(colored("\n🛑 REPLAY STOPPED BY USER.", "red", attrs=['bold']))
