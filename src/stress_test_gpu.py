import pandas as pd
import joblib
import time
import xgboost as xgb
from termcolor import colored

def main():
    print(colored("🚀 INITIATING SUSTAINED GPU STRESS TEST...", "cyan", attrs=['bold']))
    
    try:
        model = joblib.load('z_boson_xgb_model.joblib')
        # Enforce GPU parameters for Inference specifically (XGBoost defaults to CPU for inference if not explicit)
        model.set_params(device='cuda')
        print(colored("✅ XGBoost Model Loaded successfully (Forced CUDA Device).", "green"))
    except Exception as e:
        print(colored(f"❌ CRITICAL Error loading model: {e}", "red"))
        return

    try:
        full_data = pd.read_csv("Dimuon_DoubleMu.csv").dropna()
        features = full_data[['pt1', 'pt2', 'eta1', 'eta2', 'phi1', 'phi2']]
        
        # MASSIVE Dataset amplification (duplicate 100x to make 10,000,000 rows payload)
        print(colored("📦 Duplicating payload to 10 MILLION events to force GPU saturation...", "yellow"))
        massive_batch = pd.concat([features] * 100, ignore_index=True)
        
    except Exception as e:
        print(colored(f"❌ CRITICAL Data error: {e}", "red"))
        return

    print(colored("🚨 STARTING CONTINUOUS GPU INFERENCE LOOP!", "red", attrs=['blink', 'bold']))
    print(colored("👉 OPEN TASK MANAGER NOW! Look at the GPU tab.", "yellow"))
    print(colored("👉 If you are on Windows, click the arrow above the GPU graph and change '3D' to 'Cuda' or 'Compute_0' to see ML loads!", "magenta", attrs=['bold']))
    print("Press Ctrl+C to stop...\n")
    
    time.sleep(2)

    try:
        iteration = 1
        while True:
            start_t = time.perf_counter()
            # Feed 10 Million events to the GPU at once
            _ = model.predict(massive_batch)
            inf_time = time.perf_counter() - start_t
            
            throughput = len(massive_batch) / inf_time
            print(f"🔄 Loop #{iteration} | 10,000,000 events processed in {inf_time:.2f}s | Throughput: {throughput:,.0f} events/sec")
            iteration += 1
            
    except KeyboardInterrupt:
        print(colored("\n🛑 STRESS TEST STOPPED BY USER.", "red", attrs=['bold']))

if __name__ == "__main__":
    main()
