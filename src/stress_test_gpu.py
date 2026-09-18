"""Sustained GPU stress test (SYNTHETIC workload — NOT 10M unique collisions).

Builds a large batch by repeating the real feature rows 100x (~10M rows).
Useful as a software/GPU saturation test only; never present throughput on
it as physics-event throughput. Validates CUDA presence before the loop and
supports --iterations for CI (default remains infinite until Ctrl+C).
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src._compat  # noqa: F401  (UTF-8 stdout guard; must stay before prints)
from src._compat import data_path

import pandas as pd
import joblib
import time
from termcolor import colored

FEATURES = ['pt1', 'pt2', 'eta1', 'eta2', 'phi1', 'phi2']
# 10M rows x 6 float64 columns ≈ 480 MB. Refuse clearly rather than OOM-kill.
MAX_SYNTHETIC_ROWS = 20_000_000


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--iterations', type=int, default=0, help='0 = infinite until Ctrl+C')
    ap.add_argument('--repeat', type=int, default=100, help='duplication factor for synthetic batch')
    args = ap.parse_args()

    print(colored("🚀 GPU STRESS TEST (synthetic duplicated payload)...", "cyan", attrs=['bold']))

    try:
        import torch
        cuda = torch.cuda.is_available()
        print(f"   torch CUDA available: {cuda}" + (f" | {torch.cuda.get_device_name(0)}" if cuda else ""))
        if not cuda:
            print(colored("⚠️ No CUDA device visible — this test is meaningless on CPU at 10M rows. Exiting.", "yellow"))
            raise SystemExit(1)
    except SystemExit:
        raise
    except Exception as e:
        print(colored(f"❌ Could not query CUDA status ({e}). Exiting.", "red"))
        raise SystemExit(1)

    try:
        model = joblib.load(data_path('z_boson_xgb_model.joblib'))
        try:
            model.set_params(device='cuda')
            print(colored("✅ Model loaded (requested CUDA).", "green"))
        except Exception as e:
            print(colored(f"⚠️ Model loaded but device='cuda' not accepted: {e}", "yellow"))
    except FileNotFoundError:
        print(colored("❌ z_boson_xgb_model.joblib not found. Run train_model.py first.", "red"))
        raise SystemExit(1)

    try:
        full_data = pd.read_csv(data_path("Dimuon_DoubleMu.csv")).dropna()
        features = full_data[FEATURES]
        n_real = len(features)
        n_synth = n_real * args.repeat
        if n_synth > MAX_SYNTHETIC_ROWS:
            print(colored(f"❌ Requested {n_synth:,} synthetic rows exceeds the "
                          f"{MAX_SYNTHETIC_ROWS:,} safety cap (~1 GB). Lower --repeat.", "red"))
            raise SystemExit(1)
        print(colored(f"📦 SYNTHETIC step: repeating {n_real:,} real rows x{args.repeat} "
                       f"= {n_synth:,} duplicated rows (NOT unique events)...", "yellow"))
        massive_batch = pd.concat([features] * args.repeat, ignore_index=True)
    except FileNotFoundError:
        print(colored("❌ Dimuon_DoubleMu.csv not found. Run data_download.py first.", "red"))
        raise SystemExit(1)

    print(colored("🚨 STARTING INFERENCE LOOP (Ctrl+C to stop).", "red", attrs=['bold']))
    time.sleep(1)
    try:
        iteration = 1
        while True:
            start_t = time.perf_counter()
            _ = model.predict(massive_batch)
            inf_time = time.perf_counter() - start_t
            throughput = len(massive_batch) / inf_time
            print(f"🔄 Loop #{iteration} | {len(massive_batch):,} duplicated rows in {inf_time:.2f}s | "
                  f"Throughput: {throughput:,.0f} rows/s (synthetic)")
            iteration += 1
            if args.iterations and iteration > args.iterations:
                break
    except KeyboardInterrupt:
        print(colored("\n🛑 STRESS TEST STOPPED BY USER.", "red", attrs=['bold']))


if __name__ == "__main__":
    main()
