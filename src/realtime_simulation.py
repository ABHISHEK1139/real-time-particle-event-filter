"""Offline batch replay / real-time inference simulation (NOT a live stream).

Reads the static CSV in chunks, runs XGBoost batch inference, and reports:
separate data-prep / inference / post-processing timings (with warmup),
plus TP/FP/FN/TN against the known 80<M<100 label. The old script timed the
whole loop and called total_time/total_events 'average latency' — that is
replay throughput, not inference latency. Both are now reported honestly.
Also validates the model's actual device instead of assuming CUDA.
"""

from __future__ import annotations

import argparse
import contextlib
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src._compat  # noqa: F401  (UTF-8 stdout guard; must stay before prints)
from src._compat import configure_logging, data_path

import joblib
import numpy as np
import pandas as pd
from termcolor import colored

from src.config import (
    CSV_FILE,
    FEATURES,
    REQUIRED_COLUMNS,
    XGB_JOBLIB_FILE,
    labels_from_mass,
)

LOGGER = logging.getLogger("streaming_engine")
DEFAULT_BATCH_SIZE = 50_000


def _model_device_info(model) -> dict:
    info = {}
    try:
        import xgboost as xgb

        info["xgboost_version"] = xgb.__version__
    except Exception:
        info["xgboost_version"] = "unknown"
    try:
        params = model.get_params()
        info["model_device_param"] = params.get("device", params.get("tree_method", "unknown"))
    except Exception:
        info["model_device_param"] = "unknown"
    try:
        import torch

        info["cuda_available_torch"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["gpu_name"] = torch.cuda.get_device_name(0)
    except Exception:
        info["cuda_available_torch"] = "unknown"
    return info


def _align_inference_device(model):
    """Silence XGBoost 3.x's CPU/GPU DMatrix fallback warning on CUDA-less hosts.

    A booster trained with ``device='cuda'`` warns on every ``predict`` when
    the input lives on CPU. If this host has no CUDA, switch the (already
    trained) booster back to CPU inference — weights are unchanged.
    """
    try:
        import torch

        if torch.cuda.is_available():
            return model
    except Exception:
        pass
    with contextlib.suppress(Exception):
        model.set_params(device="cpu")
    return model


def _load_payload(model_file: str = XGB_JOBLIB_FILE, csv_file: str = CSV_FILE):
    """Load the classifier and the replay payload, with actionable error messages."""
    try:
        model = joblib.load(data_path(model_file))  # cwd first, repo root fallback
    except FileNotFoundError as e:
        raise SystemExit(
            f"❌ CRITICAL: {model_file} not found. Run: python src/data_download.py && python src/train_model.py"
        ) from e
    model = _align_inference_device(model)

    csv_path = data_path(csv_file)
    if not Path(csv_path).exists():
        raise SystemExit(f"❌ CRITICAL: {csv_file} not found. Run python src/data_download.py first.")
    full_data = pd.read_csv(csv_path).dropna()
    missing = set(REQUIRED_COLUMNS) - set(full_data.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {sorted(missing)}")
    return model, full_data[FEATURES], labels_from_mass(full_data["M"].to_numpy())


def simulate_realtime_stream(
    batch_size: int = DEFAULT_BATCH_SIZE,
    max_batches: int | None = None,
    model_file: str = XGB_JOBLIB_FILE,
    csv_file: str = CSV_FILE,
) -> dict:
    """Replay the static CSV in chunks and return a latency/accuracy summary.

    ``batch_size`` must be positive; ``max_batches`` may be 0, in which case the
    loop is skipped entirely and an all-zero summary is returned without raising.
    """
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, got {batch_size}")

    print(colored("\n📡 OFFLINE BATCH REPLAY: CERN collision-data inference simulation...", "cyan", attrs=["bold"]))
    print(colored("   (static CSV replay in chunks — no live queue/socket/Kafka/trigger interface)", "cyan"))

    model, features_only, y_true = _load_payload(model_file=model_file, csv_file=csv_file)
    print(colored("✅ XGBoost Model Loaded successfully.", "green"))
    for k, v in _model_device_info(model).items():
        print(f"   device info | {k}: {v}")

    total_events = len(features_only)
    signals_detected = 0
    tp = fp = fn = tn = 0

    # Warmup (excluded from timings) to avoid counting CUDA/model init overhead
    if total_events > 0:
        warm = features_only.iloc[: min(1000, total_events)]
        with contextlib.suppress(Exception):
            model.predict(warm)

    t_prep = t_inf = t_post = 0.0
    start_global = time.perf_counter()
    n_batches = 0
    processed = 0

    for start in range(0, total_events, batch_size):
        if max_batches is not None and n_batches >= max_batches:
            break
        t0 = time.perf_counter()
        batch_features = features_only.iloc[start : start + batch_size]
        batch_true = y_true[start : start + batch_size]
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

        t_prep += t1 - t0
        t_inf += t2 - t1
        t_post += t3 - t2
        n_batches += 1
        processed += len(batch_features)

        inf_ms = (t2 - t1) * 1000
        status = colored(
            f"[BATCH PROCESSED] {len(batch_features)} Events | Signals: {signals_in_batch}", "green", attrs=["bold"]
        )
        LOGGER.info("Batch %d | Inference: %.2fms | %s", n_batches, inf_ms, status)

    total_time = time.perf_counter() - start_global
    summary = {
        "processed": processed,
        "signals_detected": signals_detected,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "signal_efficiency": tp / max(tp + fn, 1),
        "false_positive_rate": fp / max(fp + tn, 1),
        "wall_clock_s": total_time,
        "throughput_eps": (processed / total_time) if total_time > 0 else 0.0,
        "prep_s": t_prep,
        "inference_s": t_inf,
        "post_s": t_post,
        "latency_ms_per_event": ((t_inf / processed) * 1000) if processed > 0 else 0.0,
        "batches": n_batches,
    }

    print(colored("\n=============================================", "cyan"))
    print(colored("🏁 BATCH REPLAY SUMMARY (offline, not a live trigger)", "cyan", attrs=["bold"]))
    print(colored(f"     Total Events Processed : {processed:,}", "white"))
    print(colored(f"     Z-Boson Predicted      : {signals_detected:,}", "white"))
    print(colored(f"     TP={tp:,} FP={fp:,} FN={fn:,} TN={tn:,}", "white"))
    print(
        colored(
            f"     Signal efficiency (recall): {summary['signal_efficiency'] * 100:.2f}% | "
            f"FPR: {summary['false_positive_rate'] * 100:.2f}%",
            "white",
        )
    )
    print(colored(f"     Wall-clock total       : {total_time:.4f}s", "white"))
    print(colored(f"     Replay throughput      : {summary['throughput_eps']:,.0f} events/s", "white"))
    print(colored(f"     Breakdown — prep: {t_prep:.3f}s | inference: {t_inf:.3f}s | post: {t_post:.3f}s", "white"))
    print(colored(f"     Pure inference latency : {summary['latency_ms_per_event']:.6f} ms/event", "white"))
    print(colored("=============================================\n", "cyan"))
    return summary


def main(batch_size: int = DEFAULT_BATCH_SIZE, max_batches: int | None = None) -> dict:
    configure_logging()
    return simulate_realtime_stream(batch_size=batch_size, max_batches=max_batches)


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="python src/realtime_simulation.py", description=__doc__.splitlines()[0])
    p.add_argument(
        "--batch-size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help=f"events per replayed batch (default: {DEFAULT_BATCH_SIZE})",
    )
    p.add_argument(
        "--max-batches", type=int, default=None, help="stop after N batches (0 replays nothing; default: all)"
    )
    p.add_argument("--model", default=XGB_JOBLIB_FILE)
    p.add_argument("--csv", default=CSV_FILE)
    return p.parse_args(argv)


if __name__ == "__main__":
    try:
        args = _parse_args()
        main(batch_size=args.batch_size, max_batches=args.max_batches)
    except KeyboardInterrupt:
        print(colored("\n🛑 REPLAY STOPPED BY USER.", "red", attrs=["bold"]))
