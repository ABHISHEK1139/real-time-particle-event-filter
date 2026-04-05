import time
import pandas as pd
import joblib
import random
import logging
from termcolor import colored

# Setup basic logging format for simulation realism
logging.basicConfig(level=logging.INFO, format='%(asctime)s | streaming_engine | %(message)s')

def simulate_realtime_stream():
    print(colored("\n📡 INITIALIZING REAL-TIME CERN COLLISION DATA PIPELINE...", "cyan", attrs=['bold']))
    time.sleep(1)

    try:
        # Load the fully-trained, GPU-accelerated XGBoost model
        # Note: Model expects features ['pt1', 'pt2', 'eta1', 'eta2', 'phi1', 'phi2'] (no invariant mass cheat!)
        model = joblib.load('z_boson_xgb_model.joblib')
        print(colored("✅ XGBoost Model Loaded successfully.", "green"))
    except FileNotFoundError:
        print(colored("❌ CRITICAL: z_boson_xgb_model.joblib not found. Run train_model.py first.", "red"))
        return

    try:
        # We load the dataset and simulate it arriving piece-by-piece
        # We drop the 'M' column just to prove the prediction does not rely on it
        full_data = pd.read_csv("Dimuon_DoubleMu.csv").dropna()
        features_only = full_data[['pt1', 'pt2', 'eta1', 'eta2', 'phi1', 'phi2']]
        actual_mass = full_data['M'] # To verify ground truth
    except FileNotFoundError:
        print(colored("❌ CRITICAL: Dimuon data CSV not found.", "red"))
        return

    print(colored("🚨 INCOMING STREAM DETECTED: STARTING HIGH-THROUGHPUT BATCH INFERENCE...\n", "yellow", attrs=['blink']))
    time.sleep(1.5)

    try:
        total_events = len(features_only)
        signals_detected = 0
        batch_size = 50000  # Large batches to actually utilize GPU parallel cores
        
        start_global = time.perf_counter()

        # Process the entire dataset in batches
        for i in range(0, total_events, batch_size):
            batch_features = features_only.iloc[i:i+batch_size]
            batch_mass = actual_mass.iloc[i:i+batch_size].values
            
            start_t = time.perf_counter()
            # GPU Batch Inference
            preds = model.predict(batch_features)
            inf_time = (time.perf_counter() - start_t) * 1000  # ms
            
            signals_in_batch = sum(preds == 1)
            signals_detected += signals_in_batch
            
            status = colored(f"[BATCH PROCESSED] {len(batch_features)} Events | Signals: {signals_in_batch}", "green", attrs=['bold'])
            logging.info(f"Batch {i//batch_size + 1} | Inference: {inf_time:.2f}ms | {status}")

        total_time = time.perf_counter() - start_global

        print(colored(f"\n=============================================", "cyan"))
        print(colored(f"🏁 FULL DATASET PROCESSING SUMMARY", "cyan", attrs=['bold']))
        print(colored(f"     Total Events Processed : {total_events:,}", "white"))
        print(colored(f"     Z-Boson Signals Found  : {signals_detected:,}", "white"))
        print(colored(f"     Total Inference Time   : {total_time:.4f} seconds", "white"))
        print(colored(f"     Throughput             : {total_events/total_time:,.0f} events / sec", "white"))
        print(colored(f"     Average Latency        : {(total_time/total_events)*1000:.6f} ms / event", "white"))
        print(colored(f"=============================================\n", "cyan"))

    except KeyboardInterrupt:
        print(colored("\n🛑 STREAM DISCONNECTED BY USER.", "red", attrs=['bold']))

if __name__ == "__main__":
    simulate_realtime_stream()
