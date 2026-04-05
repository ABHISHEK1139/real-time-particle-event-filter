import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import time

def run_physics_baseline():
    print("🔬 [EXPERIMENT: PURE PHYSICS BASELINE]")
    print("Executing a deterministic hard-cut on Invariant Mass (M) to explicitly evaluate mathematical data leakage.\n")
    
    try:
        data = pd.read_csv("Dimuon_DoubleMu.csv")
    except FileNotFoundError:
        print("❌ Dataset not found.")
        return

    data = data.dropna()
    print(f"✅ Loaded {len(data)} events.\n")

    # The ground truth evaluation (Z Boson resonance window)
    y_true = data['M'].apply(lambda x: 1 if 80 < x < 100 else 0)

    # ==========================================
    # 🧮 NAIVE MATHEMATICAL CUT (NO ML)
    # ==========================================
    start_time = time.time()
    
    # We simulate a "dumb" sensor that simply filters perfectly reconstructed mass bounds
    y_pred_naive = data['M'].apply(lambda x: 1 if 80 < x < 100 else 0)
    
    execution_time = time.time() - start_time
    
    # Evaluation
    acc = accuracy_score(y_true, y_pred_naive)
    print("==========================================")
    print(f"🎯 Pure Mass (M) Cut Accuracy: {acc*100:.4f}%")
    print(f"⏱️ Cut Execution Time: {execution_time:.4f} seconds")
    print("==========================================\n")
    
    print("[Classification Report]")
    print(classification_report(y_true, y_pred_naive, target_names=["Background", "Signal"]))

    print("\n📝 CONCLUSION:")
    print("This perfectly deterministic 100% accuracy (or ~99.30% if predicting off subsets) represents the theoretical ceiling of the dataset.")
    print("Any ML model acting on (pt, eta, phi) that achieves ~99.33% is effectively reconstructing 'M' geometrically.")

if __name__ == '__main__':
    run_physics_baseline()
