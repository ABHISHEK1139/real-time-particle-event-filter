"""Sustained GPU stress test (SYNTHETIC workload — NOT 10M unique collisions).

Builds a large batch by repeating the real feature rows 100x (~10M rows).
Useful as a software/GPU saturation test only; never present throughput on
it as physics-event throughput. Validates CUDA presence before the loop and
supports --iterations for CI (default remains infinite until Ctrl+C).
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src._compat  # noqa: F401  (UTF-8 stdout guard; must stay before prints)

import joblib
import pandas as pd
from termcolor import colored

from src._compat import data_path
from src.config import CSV_FILE, FEATURES, XGB_JOBLIB_FILE

# 10M rows x 6 float64 columns ≈ 480 MB. Refuse clearly rather than OOM-kill.
MAX_SYNTHETIC_ROWS = 20_000_000


def _require_cuda() -> None:
    """Exit(1) with a clear message unless a CUDA device is actually visible."""
    try:
        import torch
    except Exception as e:
        raise SystemExit(colored(f"❌ Could not import torch ({e}). Exiting.", "red")) from e

    cuda = torch.cuda.is_available()
    print(f"   torch CUDA available: {cuda}" + (f" | {torch.cuda.get_device_name(0)}" if cuda else ""))
    if not cuda:
        print(colored("⚠️ No CUDA device visible — this test is meaningless on CPU at 10M rows. Exiting.", "yellow"))
        raise SystemExit(1)


def _load_model(model_file: str = XGB_JOBLIB_FILE):
    path = data_path(model_file)
    if not Path(path).exists():
        raise SystemExit(colored(f"❌ {model_file} not found. Run train_model.py first.", "red"))
    model = joblib.load(path)
    try:
        model.set_params(device="cuda")
        print(colored("✅ Model loaded (requested CUDA).", "green"))
    except Exception as e:
        print(colored(f"⚠️ Model loaded but device=cuda not accepted: {e}", "yellow"))
    return model


def _load_synthetic_batch(csv_file: str = CSV_FILE, repeat: int = 100):
    path = data_path(csv_file)
    if not Path(path).exists():
        raise SystemExit(colored(f"❌ {csv_file} not found. Run data_download.py first.", "red"))
    features = pd.read_csv(path).dropna()[FEATURES]
    n_real = len(features)
    n_synth = n_real * repeat
    if n_synth > MAX_SYNTHETIC_ROWS:
        print(
            colored(
                f"❌ Requested {n_synth:,} synthetic rows exceeds the "
                f"{MAX_SYNTHETIC_ROWS:,} safety cap (~1 GB). Lower --repeat.",
                "red",
            )
        )
        raise SystemExit(1)
    print(
        colored(
            f"📦 SYNTHETIC step: repeating {n_real:,} real rows x{repeat} "
            f"= {n_synth:,} duplicated rows (NOT unique events)...",
            "yellow",
        )
    )
    return features, pd.concat([features] * repeat, ignore_index=True), n_real, n_synth


def main(
    iterations: int = 0, repeat: int = 100, model_file: str = XGB_JOBLIB_FILE, csv_file: str = CSV_FILE
) -> float | None:
    print(colored("🚀 GPU STRESS TEST (synthetic duplicated payload)...", "cyan", attrs=["bold"]))

    _require_cuda()
    model = _load_model(model_file)
    _features, massive_batch, _n_real, _n_synth = _load_synthetic_batch(csv_file, repeat)

    print(colored("🚨 STARTING INFERENCE LOOP (Ctrl+C to stop).", "red", attrs=["bold"]))
    throughput = None
    iteration = 0
    try:
        while True:
            iteration += 1
            start_t = time.perf_counter()
            model.predict(massive_batch)
            inf_time = time.perf_counter() - start_t
            throughput = len(massive_batch) / max(inf_time, 1e-12)
            print(
                f"🔄 Loop #{iteration} | {len(massive_batch):,} duplicated rows in {inf_time:.2f}s | "
                f"Throughput: {throughput:,.0f} rows/s (synthetic)"
            )
            if iterations and iteration >= iterations:
                break
    except KeyboardInterrupt:
        print(colored("\n🛑 STRESS TEST STOPPED BY USER.", "red", attrs=["bold"]))
    return throughput


def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--iterations", type=int, default=0, help="0 = infinite until Ctrl+C")
    p.add_argument("--repeat", type=int, default=100, help="duplication factor for synthetic batch")
    p.add_argument("--model", default=XGB_JOBLIB_FILE)
    p.add_argument("--csv", default=CSV_FILE)
    return p.parse_args(argv)


if __name__ == "__main__":
    args = _parse_args()
    main(iterations=args.iterations, repeat=args.repeat, model_file=args.model, csv_file=args.csv)
