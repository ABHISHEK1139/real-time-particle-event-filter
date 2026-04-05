import time
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    print("🚀 Loading CERN Dimuon Collision Data...")
    try:
        data = pd.read_csv("Dimuon_DoubleMu.csv")
    except FileNotFoundError:
        print("❌ Cannot find 'Dimuon_DoubleMu.csv'. Please ensure it's in the same directory.")
        return

    # Clean data (drop na)
    data = data.dropna()

    # Create target label (1 = Z boson signal, 0 = Background noise)
    print("⚡ Preprocessing and creating labels...")
    data['label'] = data['M'].apply(lambda x: 1 if 80 < x < 100 else 0)

    # Features: Momentum, Eta, Phi for both muons
    features = ['pt1', 'pt2', 'eta1', 'eta2', 'phi1', 'phi2']
    X = data[features]
    y = data['label']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"✅ Data prepared: {len(X_train)} training samples, {len(X_test)} testing samples.\n")

    results = []

    # ==========================================
    # LEVEL 1: Standard Random Forest (Baseline)
    # ==========================================
    print("🟢 LEVEL 1: Standard Scikit-Learn Random Forest (CPU)")
    rf_base = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    
    # Train
    start = time.time()
    rf_base.fit(X_train, y_train)
    train_time_base = time.time() - start
    
    # Predict (Measure Inference Speed)
    start = time.time()
    preds_base = rf_base.predict(X_test)
    inf_time_base = time.time() - start
    acc_base = accuracy_score(y_test, preds_base)
    
    print(f"  ➜ Training Time:  {train_time_base:.4f}s")
    print(f"  ➜ Inference Time: {inf_time_base:.4f}s")
    print(f"  ➜ Accuracy:       {acc_base*100:.2f}%\n")
    results.append(("Standard RF", inf_time_base, acc_base))

    # ==========================================
    # LEVEL 2a: Optimized Random Forest (Fewer Estimators)
    # ==========================================
    print("🟡 LEVEL 2a: Optimized Random Forest (Reduced Estimators)")
    rf_fast = RandomForestClassifier(n_estimators=20, random_state=42, n_jobs=-1)
    
    # Train
    start = time.time()
    rf_fast.fit(X_train, y_train)
    train_time_fast = time.time() - start
    
    # Predict
    start = time.time()
    preds_fast = rf_fast.predict(X_test)
    inf_time_fast = time.time() - start
    acc_fast = accuracy_score(y_test, preds_fast)
    
    print(f"  ➜ Training Time:  {train_time_fast:.4f}s")
    print(f"  ➜ Inference Time: {inf_time_fast:.4f}s")
    print(f"  ➜ Accuracy:       {acc_fast*100:.2f}%\n")
    results.append(("Optimized RF", inf_time_fast, acc_fast))

    # ==========================================
    # LEVEL 3: GPU-Accelerated XGBoost Classifier
    # ==========================================
    print("🔴 LEVEL 3: GPU-Accelerated XGBoost Classifier")
    try:
        # Use tree_method='hist' and device='cuda' for GPU acceleration in newer XGBoost versions
        xgb_model = xgb.XGBClassifier(
            n_estimators=100, 
            learning_rate=0.1, 
            tree_method='hist',
            device='cuda',
            random_state=42
        )
        
        # Train
        start = time.time()
        xgb_model.fit(X_train, y_train)
        train_time_xgb = time.time() - start
        
        # Predict
        start = time.time()
        preds_xgb = xgb_model.predict(X_test)
        inf_time_xgb = time.time() - start
        acc_xgb = accuracy_score(y_test, preds_xgb)
        
        print(f"  ➜ Training Time:  {train_time_xgb:.4f}s")
        print(f"  ➜ Inference Time: {inf_time_xgb:.4f}s")
        print(f"  ➜ Accuracy:       {acc_xgb*100:.2f}%\n")
        results.append(("GPU XGBoost", inf_time_xgb, acc_xgb))
    except Exception as e:
        print("  ⚠️ Could not run GPU XGBoost (ensure CUDA is set up correctly natively).")
        print(f"      Error: {e}")
        # Fallback to CPU XGBoost if CUDA is not available or driver fails
        print("  ➜ Falling back to CPU XGBoost...")
        xgb_model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, tree_method='hist', random_state=42, n_jobs=-1)
        start = time.time()
        xgb_model.fit(X_train, y_train)
        start_inf = time.time()
        preds_xgb = xgb_model.predict(X_test)
        inf_time_xgb = time.time() - start_inf
        acc_xgb = accuracy_score(y_test, preds_xgb)
        print(f"  ➜ Inference Time: {inf_time_xgb:.4f}s")
        print(f"  ➜ Accuracy:       {acc_xgb*100:.2f}%\n")
        results.append(("CPU XGBoost", inf_time_xgb, acc_xgb))


    # ==========================================
    # Plotting Speed Comparison
    # ==========================================
    print("🔥 Plotting Performance vs Accuracy Comparison...")
    models = [r[0] for r in results]
    times = [r[1] for r in results]
    accs = [r[2]*100 for r in results]

    fig, ax1 = plt.subplots(figsize=(10, 6))

    # Bar plot for Inference Time
    color = 'tab:red'
    ax1.set_xlabel('Model Iteration', fontsize=12, fontweight='bold')
    ax1.set_ylabel('Inference Time (s)', color=color, fontsize=12, fontweight='bold')
    bars = ax1.bar(models, times, color=color, alpha=0.6, width=0.4, label='Inference Speed')
    ax1.tick_params(axis='y', labelcolor=color)

    # Line plot for Accuracy
    ax2 = ax1.twinx()  
    color = 'tab:blue'
    ax2.set_ylabel('Accuracy (%)', color=color, fontsize=12, fontweight='bold')
    line = ax2.plot(models, accs, color=color, marker='o', markersize=10, linewidth=3, label='Accuracy')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Performance Analysis: Inference Speed vs Accuracy', fontsize=14, fontweight='bold')
    fig.tight_layout()
    plt.savefig('speed_comparison.png')
    print("✅ Saved comparison graph as 'speed_comparison.png'")
    
    # Try to show the plot if running locally
    try:
        # plt.show() # Uncomment to show directly during run
        pass
    except:
        pass

if __name__ == "__main__":
    main()
