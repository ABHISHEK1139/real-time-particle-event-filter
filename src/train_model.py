"""Supervised demonstration of learning a Z-mass-window selection from muon kinematics.

Scope (honest): labels are `1 if 80 < M < 100 else 0` while features
(pt/eta/phi) mathematically determine M. The model therefore learns an
approximation to a known deterministic selection — it does NOT discover an
unknown Z signature. Educational CERN Open Data only (Run2011A DoubleMu,
7 TeV, record 5201; parent record 545).

Experimental protocol (fixed):
  TRAIN (60%) -> fit model / fit baseline cut
  VALIDATION (20%) -> calibrate thresholds for 5% background acceptance
  TEST (20%, frozen) -> report signal efficiency, AUROC/AUPRC, confusion matrix
Stratified splits throughout. Thresholds are NEVER fit on TEST.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src._compat  # noqa: F401  (UTF-8 stdout guard; must stay before prints)
from src._compat import data_path

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report,
    roc_curve, auc, average_precision_score, precision_recall_curve,
)
import joblib
import time
import os

FEATURES = ['pt1', 'pt2', 'eta1', 'eta2', 'phi1', 'phi2']
REQUIRED_COLUMNS = set(FEATURES) | {'M'}
RANDOM_STATE = 42


def load_data(filepath="Dimuon_DoubleMu.root", max_rows=None):
    """Loads ROOT TTree binary format using uproot."""
    filepath = data_path(filepath)  # cwd first, repo root as fallback
    print("🚀 Loading CERN Dimuon Collision Data (ROOT TTree)...")
    try:
        import uproot
        import awkward as ak
        with uproot.open(filepath) as file:
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
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns {sorted(missing)}; got {list(df.columns)}")
    df = df.copy()
    df['label'] = df['M'].apply(lambda x: 1 if 80 < x < 100 else 0)
    return df[FEATURES], df['label'], df


def _is_cuda_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    keywords = ("cuda", "gpu", "device", "libcuda", "ptx", "nvml",
                "no cuda", "cuda error", "xgboosterror", "device='cuda'",
                'device "cuda"', "thrust", "nccl")
    return any(k in msg for k in keywords)


def train_xgboost(X_train, y_train):
    """Trains an XGBoost classifier, trying CUDA but only falling back on genuine CUDA errors."""
    print("\n⚡ Starting Model Training with XGBoost (GPU attempt, CPU fallback)...")
    try:
        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1,
                                  tree_method='hist', device='cuda',
                                  random_state=RANDOM_STATE)
        start_time = time.time()
        model.fit(X_train, y_train)
        print(f"✅ GPU Training Complete in {time.time() - start_time:.4f} seconds.")
        try:
            model.set_params(device='cpu')
        except Exception:
            pass
    except Exception as e:
        # A real model/data/programming bug must NOT be misreported as "no GPU".
        if not _is_cuda_error(e):
            print(f"❌ Training failed with a non-CUDA error; NOT falling back silently: {e}")
            raise
        print(f"⚠️ CUDA unavailable ({e}); falling back to CPU...")
        model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1,
                                  tree_method='hist', n_jobs=-1,
                                  random_state=RANDOM_STATE)
        start_time = time.time()
        model.fit(X_train, y_train)
        print(f"✅ CPU Training Complete in {time.time() - start_time:.4f} seconds.")
    return model


def stratified_train_val_test_split(X, y, df, test_size=0.2, val_size=0.2, random_state=RANDOM_STATE):
    """60/20/20 stratified split (val_size/test_size are fractions of the whole)."""
    X_temp, X_test, y_temp, y_test, df_temp, df_test = train_test_split(
        X, y, df, test_size=test_size, random_state=random_state, stratify=y)
    # val fraction relative to temp
    val_rel = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val, df_train, df_val = train_test_split(
        X_temp, y_temp, df_temp, test_size=val_rel, random_state=random_state, stratify=y_temp)
    return (X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test)


def evaluate_rule_based_baseline(df_train, df_test, background_budget_pct=0.05):
    """
    Standard physics baseline: fit a leading-muon pT (max(pT1, pT2)) cut on TRAIN
    background only (no test leakage), then measure frozen performance on TEST.
    (Muon indices 1 and 2 are unordered in open data; cutting on max(pT1, pT2)
    avoids unphysical muon assignment asymmetry).
    """
    print(f"\n📏 Evaluating Rule-Based Baseline (Fixed Background Budget = {background_budget_pct*100}%)")
    print("   (cut fit on TRAIN, evaluated on frozen TEST)")
    pt_lead_train = np.maximum(df_train['pt1'], df_train['pt2'])
    pt_lead_test = np.maximum(df_test['pt1'], df_test['pt2'])

    pt_lead_bg_train = pt_lead_train[df_train['label'] == 0].values
    if len(pt_lead_bg_train) == 0:
        raise ValueError("No background events in training split.")
    pt_cut = np.percentile(pt_lead_bg_train, (1 - background_budget_pct) * 100)

    # Frozen evaluation on TEST
    pt_lead_bg_test = pt_lead_test[df_test['label'] == 0].values
    pt_lead_sig_test = pt_lead_test[df_test['label'] == 1].values
    bg_retained = float(np.sum(pt_lead_bg_test >= pt_cut) / max(len(pt_lead_bg_test), 1))
    sig_retained = float(np.sum(pt_lead_sig_test >= pt_cut) / max(len(pt_lead_sig_test), 1))

    print(f"   => Baseline Leading Muon Cut Rule: pT_lead >= {pt_cut:.2f} GeV (fit on TRAIN)")
    print(f"   => TEST Background Retained: {bg_retained*100:.2f}% (Target was {background_budget_pct*100}%)")
    print(f"   => TEST Baseline Signal Efficiency: {sig_retained*100:.2f}%")
    return {"pt_cut": float(pt_cut), "bg_retained": bg_retained, "sig_eff": sig_retained}


def evaluate_ml_model(model, X_val, y_val, X_test, y_test, background_budget_pct=0.05):
    """
    Calibrate the probability threshold on VALIDATION for the background budget,
    then evaluate FROZEN on TEST. Also reports AUROC/AUPRC/accuracy/confusion.
    Replaces the old circular significance sweep with ROC-based evaluation.
    """
    print(f"\n🧠 Evaluating ML Classification (Fixed Background Budget = {background_budget_pct*100}%)")
    print("   (threshold fit on VALIDATION, evaluated on frozen TEST)")
    y_val = np.asarray(y_val)
    y_test = np.asarray(y_test)
    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    bg_val = p_val[y_val == 0]
    if len(bg_val) == 0:
        raise ValueError("No background events in validation split.")
    prob_cut = float(np.percentile(bg_val, (1 - background_budget_pct) * 100))

    bg_test = p_test[y_test == 0]
    sig_test = p_test[y_test == 1]
    bg_retained = float(np.sum(bg_test >= prob_cut) / max(len(bg_test), 1))
    sig_retained = float(np.sum(sig_test >= prob_cut) / max(len(sig_test), 1))

    # Threshold-independent metrics on TEST
    fpr, tpr, _ = roc_curve(y_test, p_test)
    auroc = float(auc(fpr, tpr))
    auprc = float(average_precision_score(y_test, p_test))
    y_pred_default = (p_test >= 0.5).astype(int)
    acc = float(accuracy_score(y_test, y_pred_default))
    cm = confusion_matrix(y_test, y_pred_default, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()

    print(f"   => ML Probability Threshold: >= {prob_cut:.4f} (fit on VALIDATION)")
    print(f"   => TEST Background Retained: {bg_retained*100:.2f}% (Target was {background_budget_pct*100}%)")
    print(f"   => TEST ML Signal Efficiency @5% FPR budget: {sig_retained*100:.2f}%")
    print(f"   => TEST AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | Accuracy@0.5: {acc*100:.2f}%")
    print(f"   => TEST Confusion@0.5: TN={tn} FP={fp} FN={fn} TP={tp}")

    os.makedirs("plots", exist_ok=True)

    # ROC curve (threshold-independent, no leakage: single TEST evaluation)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'XGBoost (AUROC = {auroc:.3f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=1, linestyle='--', label='Chance')
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve (TEST set, frozen)', fontsize=14, fontweight='bold')
    plt.legend(loc='lower right')
    plt.tight_layout()
    plt.savefig("plots/6_roc_curve.png")
    plt.close()

    # Significance profile: thresholds from VALIDATION score grid, evaluated on TEST.
    # (Old code chose thresholds from the TEST signal itself — circular.)
    grid = np.quantile(p_val, np.linspace(0.01, 0.99, 50))
    effs, sigs = [], []
    for thr in grid:
        s = np.sum(p_test[y_test == 1] >= thr)
        b = np.sum(p_test[y_test == 0] >= thr)
        effs.append(s / max(np.sum(y_test == 1), 1))
        sigs.append(s / np.sqrt(b + 1e-9))
    # Sort points ascending by efficiency for monotonic curve rendering
    sorted_pairs = sorted(zip(effs, sigs), key=lambda x: x[0])
    effs_sorted = [p[0] for p in sorted_pairs]
    sigs_sorted = [p[1] for p in sorted_pairs]

    plt.figure(figsize=(8, 6))
    plt.plot(effs_sorted, sigs_sorted, color='darkviolet', lw=2)
    plt.xlabel('Signal Efficiency (TEST)', fontsize=12)
    plt.ylabel('Significance ($S/\\sqrt{B}$, TEST)', fontsize=12)
    plt.title('Physics Validation: Significance Profile\n(thresholds from VALIDATION, measured on TEST)',
              fontsize=12, fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/6b_physics_significance.png")
    plt.close()

    # Confusion matrix @0.5 on TEST
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix @0.5 (TEST)', fontweight='bold')
    plt.tight_layout()
    plt.savefig("plots/5_confusion_matrix.png")
    plt.close()

    return {"prob_cut": prob_cut, "bg_retained": bg_retained, "sig_eff": sig_retained,
            "auroc": auroc, "auprc": auprc, "accuracy": acc,
            "tn": tn, "fp": fp, "fn": fn, "tp": tp}


def main():
    df_raw = load_data()
    X, y, df = build_features_and_labels(df_raw)

    # Evaluation Budget
    BUDGET = 0.05  # We tolerate 5% background leakage

    # Leakage-free stratified split: 60% train / 20% val / 20% test
    X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test = \
        stratified_train_val_test_split(X, y, df)

    print(f"✅ Split (stratified): train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    print(f"   Train signal frac: {np.mean(np.asarray(y_train)==1):.4f} | "
          f"Val: {np.mean(np.asarray(y_val)==1):.4f} | Test: {np.mean(np.asarray(y_test)==1):.4f}")

    # 1. Physics Baseline (fit TRAIN, eval TEST)
    baseline = evaluate_rule_based_baseline(df_train, df_test, background_budget_pct=BUDGET)

    # 2. ML Training (TRAIN only)
    model = train_xgboost(X_train, y_train)

    # 3. ML Evaluation (calibrate VAL, eval frozen TEST)
    ml = evaluate_ml_model(model, X_val, y_val, X_test, y_test, background_budget_pct=BUDGET)

    # 4. Generate Results Report
    os.makedirs("results", exist_ok=True)
    dummy_acc = max(np.mean(np.asarray(y_test) == 0), np.mean(np.asarray(y_test) == 1))
    with open("results/metrics.md", "w", encoding="utf-8") as f:
        f.write("# Physics Filtering Benchmark\n\n")
        f.write("> Supervised demonstration of learning a Z-mass-window (80<M<100 GeV)\n")
        f.write("> selection from muon kinematics. Labels are a deterministic function of M;\n")
        f.write("> features already encode M. Not a trigger/discovery benchmark.\n\n")
        f.write(f"**Target Background Acceptance Budget**: {BUDGET*100}%\n\n")
        f.write("**Protocol**: stratified 60/20/20 train/val/test; thresholds fit on\n")
        f.write("TRAIN (baseline) or VALIDATION (ML) and measured once on frozen TEST.\n\n")
        f.write("### Architectures (TEST set, frozen)\n")
        f.write("| Architecture | Signal Efficiency @5% bg | AUROC | AUPRC | Acc@0.5 |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(f"| Rule-Based Cut (pT_lead >= {baseline['pt_cut']:.2f} GeV) | **{baseline['sig_eff']*100:.2f}%** | — | — | — |\n")
        f.write(f"| Kinematic ML Classifier (XGBoost) | **{ml['sig_eff']*100:.2f}%** | {ml['auroc']:.4f} | {ml['auprc']:.4f} | {ml['accuracy']*100:.2f}% |\n\n")
        f.write(f"TEST background retained: baseline {baseline['bg_retained']*100:.2f}%, ML {ml['bg_retained']*100:.2f}% "
                f"(target {BUDGET*100}%).\n\n")
        f.write(f"TEST confusion @0.5: TN={ml['tn']} FP={ml['fp']} FN={ml['fn']} TP={ml['tp']}.\n\n")
        f.write(f"Dummy (all-background) accuracy on TEST would be {dummy_acc*100:.2f}%; "
                "accuracy alone is misleading under this class imbalance — prefer AUROC/AUPRC above.\n\n")
        f.write(f"> Result: at the same ≈5% background acceptance, ML signal efficiency "
                f"({ml['sig_eff']*100:.2f}%) vs naive pT cut ({baseline['sig_eff']*100:.2f}%): "
                f"**{((ml['sig_eff'] - baseline['sig_eff'])*100):+.2f}pp** difference on frozen TEST.\n")

    print("\n✅ Physics Benchmarks compiled natively to results/metrics.md")
    # Reset device to CPU on saved estimator for portable cross-platform inference without warnings
    if hasattr(model, 'set_params'):
        try:
            model.set_params(device='cpu')
        except Exception:
            pass
    joblib.dump(model, 'z_boson_xgb_model.joblib')
    print("💾 Model saved locally as 'z_boson_xgb_model.joblib'")

    # Save native XGBoost JSON format (production format for C++ / Triton / ROOT TMVA)
    os.makedirs('models', exist_ok=True)
    json_path = 'models/z_boson_xgb_model.json'
    try:
        model.save_model(json_path)
        print(f"💾 Production native model saved as '{json_path}'")
    except Exception as e:
        print(f"⚠️ Could not export native model: {e}")

    # Generate full invariant mass spectrum plot
    try:
        from src.plot_mass_spectrum import plot_invariant_mass_spectrum
        plot_invariant_mass_spectrum()
    except Exception as e:
        print(f"⚠️ Could not generate mass spectrum plot: {e}")


if __name__ == "__main__":
    main()
