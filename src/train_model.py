import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import joblib
import time
import os

FEATURES = ['pt1', 'pt2', 'eta1', 'eta2', 'phi1', 'phi2']

def load_data(filepath="Dimuon_DoubleMu.root", max_rows=None):
    """Loads ROOT TTree binary format using uproot."""
    print("🚀 Loading CERN Dimuon Collision Data (ROOT TTree)...")
    try:
        import uproot
        import awkward as ak
        file = uproot.open(filepath)
        data = file["Events"].arrays()
        if hasattr(data, 'to_dataframe'):
            data = data.to_dataframe()
        else:
            data = ak.to_dataframe(data)
        if max_rows is not None:
            data = data[:max_rows]
        return data.dropna()
    except FileNotFoundError:
        print("❌ ROOT dataset not found. Run src/data_download.py first.")
        raise
    except ImportError:
        print("❌ uproot library missing. Please install dependencies.")
        raise

def build_features_and_labels(df):
    """Constructs the kinematic features and implicit labels (Z Boson peak 80 < M < 100)."""
    # Labeling: Signal = 1 (Z Boson, roughly 80 < M < 100 GeV), Background = 0
    df = df.copy()
    df['label'] = df['M'].apply(lambda x: 1 if 80 < x < 100 else 0)
    return df[FEATURES], df['label'], df

def train_xgboost(X_train, y_train):
    """Trains an XGBoost classifier with GPU fallback mechanisms."""
    print("\n⚡ Starting Model Training with XGBoost (GPU Optimized)...")
    try:
        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, tree_method='hist', device='cuda', random_state=42)
        start_time = time.time()
        model.fit(X_train, y_train)
        print(f"✅ GPU Training Complete in {time.time() - start_time:.4f} seconds.")
    except Exception:
        print("⚠️ GPU acceleration not available, falling back to CPU optimization...")
        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1, tree_method='hist', n_jobs=-1, random_state=42)
        start_time = time.time()
        model.fit(X_train, y_train)
        print(f"✅ CPU Training Complete in {time.time() - start_time:.4f} seconds.")
    return model

def evaluate_rule_based_baseline(df, background_budget_pct=0.05):
    """
    Establishes a naive physics baseline using simple kinematic cuts (pT).
    We find the pT cut that restricts background leakage to `background_budget_pct`
    and measure how much Signal is retained.
    """
    print(f"\n📏 Evaluating Rule-Based Baseline (Fixed Background Budget = {background_budget_pct*100}%)")
    signal_mask = df['label'] == 1
    background_mask = df['label'] == 0
    
    # We will compute a simple cut on the leading muon's transverse momentum (pt1)
    pt1_bg = df[background_mask]['pt1'].values
    
    # To keep only `background_budget_pct` of the background, we find the (1 - budget) percentile of BG pt1
    pt_cut = np.percentile(pt1_bg, (1 - background_budget_pct) * 100)
    
    # Calculate efficiencies
    bg_retained = np.sum(pt1_bg >= pt_cut) / len(pt1_bg)
    
    pt1_sig = df[signal_mask]['pt1'].values
    sig_retained = np.sum(pt1_sig >= pt_cut) / len(pt1_sig)
    
    print(f"   => Naive Cut Rule: pT1 >= {pt_cut:.2f} GeV")
    print(f"   => Background Retained: {bg_retained*100:.2f}% (Target was {background_budget_pct*100}%)")
    print(f"   => Baseline Signal Efficiency: {sig_retained*100:.2f}%")
    return sig_retained

def evaluate_ml_model(model, X_test, y_test, background_budget_pct=0.05):
    """
    Evaluates the trained ML model against the exact same background budget.
    We threshold the output probability to match the exact background leakage,
    and compare the signal efficiency to the rule-based baseline.
    """
    print(f"\n🧠 Evaluating ML Classification (Fixed Background Budget = {background_budget_pct*100}%)")
    y_prob = model.predict_proba(X_test)[:, 1]
    
    # Isolate targets
    bg_probs = y_prob[y_test == 0]
    sig_probs = y_prob[y_test == 1]
    
    # Find probability threshold that permits exactly `background_budget_pct` leakage
    prob_cut = np.percentile(bg_probs, (1 - background_budget_pct) * 100)
    
    bg_retained = np.sum(bg_probs >= prob_cut) / len(bg_probs)
    sig_retained = np.sum(sig_probs >= prob_cut) / len(sig_probs)
    
    print(f"   => ML Probability Threshold: >= {prob_cut:.4f}")
    print(f"   => Background Retained: {bg_retained*100:.2f}% (Target was {background_budget_pct*100}%)")
    print(f"   => ML Signal Efficiency: {sig_retained*100:.2f}%")
    
    # Plot Significance Sweep
    eff_thresholds = np.linspace(0.01, 0.99, 50)
    significances = []
    for eff in eff_thresholds:
        thresh = np.percentile(sig_probs, (1 - eff) * 100)
        s = np.sum(sig_probs >= thresh)
        b = np.sum(bg_probs >= thresh)
        sig = s / np.sqrt(b + 1e-9)
        significances.append(sig)

    os.makedirs("plots", exist_ok=True)
    plt.figure(figsize=(8,6))
    plt.plot(eff_thresholds, significances, color='darkviolet', lw=2)
    plt.xlabel('Signal Efficiency', fontsize=12)
    plt.ylabel('Significance ($S/\sqrt{B}$)', fontsize=12)
    plt.title('Physics Validation: Signal Significance Profile', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/6b_physics_significance.png")
    plt.close()
    
    return sig_retained

def main():
    df_raw = load_data()
    X, y, df = build_features_and_labels(df_raw)
    
    # Evaluation Budget
    BUDGET = 0.05  # We tolerate 5% background leakage
    
    # 1. Physics Baseline
    baseline_sig_eff = evaluate_rule_based_baseline(df, background_budget_pct=BUDGET)
    
    # 2. ML Training
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = train_xgboost(X_train, y_train)
    
    # 3. ML Evaluation
    ml_sig_eff = evaluate_ml_model(model, X_test, y_test, background_budget_pct=BUDGET)
    
    # 4. Generate Results Report
    os.makedirs("results", exist_ok=True)
    with open("results/metrics.md", "w") as f:
        f.write("# Physics Filtering Benchmark\n\n")
        f.write(f"**Target Background Noise Budget**: {BUDGET*100}%\n\n")
        f.write("### Architectures\n")
        f.write("| Architecture | Signal Efficiency (Target Z Bosons Caught) |\n")
        f.write("|---|---|\n")
        f.write(f"| Rule-Based Cut (pT threshold) | **{baseline_sig_eff*100:.2f}%** |\n")
        f.write(f"| Kinematic ML Classifier (XGBoost) | **{ml_sig_eff*100:.2f}%** |\n\n")
        f.write(f"> Result: The ML pipeline improves signal detection by **+{(ml_sig_eff - baseline_sig_eff)*100:.2f}%** compared to naive theoretical bounds at the exact same noise acceptance rate.\n")
        
    print("\n✅ Physics Benchmarks compiled natively to results/metrics.md")
    joblib.dump(model, 'z_boson_xgb_model.joblib')
    print("💾 Model saved locally as 'z_boson_xgb_model.joblib'")

if __name__ == "__main__":
    main()
