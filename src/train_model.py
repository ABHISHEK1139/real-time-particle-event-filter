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

from __future__ import annotations

import contextlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src._compat  # noqa: F401  (UTF-8 stdout guard; must stay before prints)

import joblib
import matplotlib
import numpy as np
import seaborn as sns
import xgboost as xgb
from sklearn.metrics import (
    accuracy_score,
    auc,
    average_precision_score,
    confusion_matrix,
    roc_curve,
)
from sklearn.model_selection import train_test_split

matplotlib.use("Agg")  # headless-safe: never try to open a GUI window
import matplotlib.pyplot as plt

from src.config import (
    BUDGET,
    FEATURES,
    MASS_SIGNAL_MAX,
    MASS_SIGNAL_MIN,
    RANDOM_STATE,
    REQUIRED_COLUMNS,
    ROOT_FILE,
    XGB_JOBLIB_FILE,
    XGB_JSON_FILE,
    output_path,
    rates_at_threshold,
    threshold_for_budget,
)

TREE_NAME = "Events"


def load_data(filepath: str = ROOT_FILE, max_rows: int | None = None):
    """Loads ROOT TTree binary format using uproot."""
    from src._compat import data_path

    filepath = data_path(filepath)  # cwd first, repo root as fallback
    print("🚀 Loading CERN Dimuon Collision Data (ROOT TTree)...")
    try:
        import awkward as ak
        import uproot
    except ImportError as e:
        print("❌ uproot/awkward missing. Please install dependencies.")
        raise e from None

    if not Path(filepath).exists():
        print(f"❌ ROOT dataset not found at {filepath}. Run src/data_download.py first.")
        raise FileNotFoundError(filepath)

    with uproot.open(filepath) as f:
        arrays = f[TREE_NAME].arrays()
        df = arrays.to_dataframe() if hasattr(arrays, "to_dataframe") else ak.to_dataframe(arrays)
        if max_rows is not None:
            df = df[:max_rows]
        return df.dropna()


def build_features_and_labels(df):
    """Constructs the kinematic features and implicit labels (Z Boson peak 80 < M < 100)."""
    # Labeling: Signal = 1 (Z Boson, roughly 80 < M < 100 GeV), Background = 0
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Input is missing required columns {sorted(missing)}; got {list(df.columns)}")
    df = df.copy()
    df["label"] = ((df["M"] > MASS_SIGNAL_MIN) & (df["M"] < MASS_SIGNAL_MAX)).astype(int)
    return df[FEATURES], df["label"], df


def _is_cuda_error(exc: Exception) -> bool:
    msg = str(exc).lower()
    keywords = (
        "cuda",
        "gpu",
        "device",
        "libcuda",
        "ptx",
        "nvml",
        "no cuda",
        "cuda error",
        "xgboosterror",
        "device='cuda'",
        'device "cuda"',
        "thrust",
        "nccl",
    )
    return any(k in msg for k in keywords)


def train_xgboost(X_train, y_train):
    """Trains an XGBoost classifier, trying CUDA but only falling back on genuine CUDA errors."""
    print("\n⚡ Starting Model Training with XGBoost (GPU attempt, CPU fallback)...")
    try:
        model = xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.1, tree_method="hist", device="cuda", random_state=RANDOM_STATE
        )
        start_time = time.time()
        model.fit(X_train, y_train)
        print(f"✅ GPU Training Complete in {time.time() - start_time:.4f} seconds.")
        # Reset the device so the pickled booster infers on any host. Weights are
        # unchanged; this only avoids XGBoost 3.x's CPU-fallback warning.
        with contextlib.suppress(Exception):
            model.set_params(device="cpu")
    except Exception as e:
        # A real model/data/programming bug must NOT be misreported as "no GPU".
        if not _is_cuda_error(e):
            print(f"❌ Training failed with a non-CUDA error; NOT falling back silently: {e}")
            raise
        print(f"⚠️ CUDA unavailable ({e}); falling back to CPU...")
        model = xgb.XGBClassifier(
            n_estimators=100, learning_rate=0.1, tree_method="hist", n_jobs=-1, random_state=RANDOM_STATE
        )
        start_time = time.time()
        model.fit(X_train, y_train)
        print(f"✅ CPU Training Complete in {time.time() - start_time:.4f} seconds.")
    return model


def stratified_train_val_test_split(X, y, df, test_size=0.2, val_size=0.2, random_state=RANDOM_STATE):
    """60/20/20 stratified split (val_size/test_size are fractions of the whole)."""
    X_temp, X_test, y_temp, y_test, df_temp, df_test = train_test_split(
        X, y, df, test_size=test_size, random_state=random_state, stratify=y
    )
    # val fraction relative to temp
    val_rel = val_size / (1.0 - test_size)
    X_train, X_val, y_train, y_val, df_train, df_val = train_test_split(
        X_temp, y_temp, df_temp, test_size=val_rel, random_state=random_state, stratify=y_temp
    )
    return (X_train, X_val, X_test, y_train, y_val, y_test, df_train, df_val, df_test)


def evaluate_rule_based_baseline(df_train, df_test, background_budget_pct: float = BUDGET):
    """
    Standard physics baseline: fit a leading-muon pT (max(pT1, pT2)) cut on TRAIN
    background only (no test leakage), then measure frozen performance on TEST.
    (Muon indices 1 and 2 are unordered in open data; cutting on max(pT1, pT2)
    avoids unphysical muon assignment asymmetry).
    """
    print(f"\n📏 Evaluating Rule-Based Baseline (Fixed Background Budget = {background_budget_pct * 100}%)")
    print("   (cut fit on TRAIN, evaluated on frozen TEST)")
    pt_lead_train = np.maximum(df_train["pt1"], df_train["pt2"])
    pt_lead_test = np.maximum(df_test["pt1"], df_test["pt2"])

    pt_cut = threshold_for_budget(pt_lead_train[df_train["label"] == 0], background_budget_pct)
    r = rates_at_threshold(pt_lead_test, df_test["label"], pt_cut)

    print(f"   => Baseline Leading Muon Cut Rule: pT_lead >= {pt_cut:.2f} GeV (fit on TRAIN)")
    print(f"   => TEST Background Retained: {r['bg_retained'] * 100:.2f}% (Target was {background_budget_pct * 100}%)")
    print(f"   => TEST Baseline Signal Efficiency: {r['sig_eff'] * 100:.2f}%")
    return {"pt_cut": pt_cut, "bg_retained": r["bg_retained"], "sig_eff": r["sig_eff"]}


def evaluate_ml_model(model, X_val, y_val, X_test, y_test, background_budget_pct: float = BUDGET):
    """
    Calibrate the probability threshold on VALIDATION for the background budget,
    then evaluate FROZEN on TEST. Also reports AUROC/AUPRC/accuracy/confusion.
    Thresholds come from a budget-safe cut so ties can never overshoot the budget.
    """
    print(f"\n🧠 Evaluating ML Classification (Fixed Background Budget = {background_budget_pct * 100}%)")
    print("   (threshold fit on VALIDATION, evaluated on frozen TEST)")
    y_val = np.asarray(y_val)
    y_test = np.asarray(y_test)
    p_val = model.predict_proba(X_val)[:, 1]
    p_test = model.predict_proba(X_test)[:, 1]

    prob_cut = threshold_for_budget(p_val[y_val == 0], background_budget_pct)
    r = rates_at_threshold(p_test, y_test, prob_cut)

    # Threshold-independent metrics on TEST
    fpr, tpr, _ = roc_curve(y_test, p_test)
    auroc = float(auc(fpr, tpr))
    auprc = float(average_precision_score(y_test, p_test))
    y_pred_default = (p_test >= 0.5).astype(int)
    acc = float(accuracy_score(y_test, y_pred_default))
    cm = confusion_matrix(y_test, y_pred_default, labels=[0, 1])
    tn, fp, fn, tp = (int(v) for v in cm.ravel().tolist())

    print(f"   => ML Probability Threshold: >= {prob_cut:.6g} (fit on VALIDATION)")
    print(f"   => TEST Background Retained: {r['bg_retained'] * 100:.2f}% (Target was {background_budget_pct * 100}%)")
    print(f"   => TEST ML Signal Efficiency @5% FPR budget: {r['sig_eff'] * 100:.2f}%")
    print(f"   => TEST AUROC: {auroc:.4f} | AUPRC: {auprc:.4f} | Accuracy@0.5: {acc * 100:.2f}%")
    print(f"   => TEST Confusion@0.5: TN={tn} FP={fp} FN={fn} TP={tp}")

    _plot_roc(fpr, tpr, auroc)
    _plot_significance(p_val, p_test, y_test)
    _plot_confusion(cm)

    return {
        "prob_cut": prob_cut,
        "bg_retained": r["bg_retained"],
        "sig_eff": r["sig_eff"],
        "auroc": auroc,
        "auprc": auprc,
        "accuracy": acc,
        "tn": tn,
        "fp": fp,
        "fn": fn,
        "tp": tp,
    }


def _plot_roc(fpr, tpr, auroc) -> None:
    """ROC curve (threshold-independent, no leakage: single TEST evaluation)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, color="darkorange", lw=2, label=f"XGBoost (AUROC = {auroc:.3f})")
    ax.plot([0, 1], [0, 1], color="navy", lw=1, linestyle="--", label="Chance")
    ax.set_xlabel("False Positive Rate", fontsize=12)
    ax.set_ylabel("True Positive Rate", fontsize=12)
    ax.set_title("ROC Curve (TEST set, frozen)", fontsize=14, fontweight="bold")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(output_path("plots", "6_roc_curve.png"))
    plt.close(fig)


def _plot_significance(p_val, p_test, y_test) -> None:
    """Significance profile: thresholds from VALIDATION score grid, measured on TEST.

    (The old code chose thresholds from the TEST signal itself — circular.)
    """
    sig_mask = y_test == 1
    bg_mask = y_test == 0
    n_sig = max(int(sig_mask.sum()), 1)
    sig_scores = p_test[sig_mask]
    bg_scores = p_test[bg_mask]

    points = []
    for thr in np.quantile(p_val, np.linspace(0.01, 0.99, 50)):
        s = int(np.sum(sig_scores >= thr))
        b = int(np.sum(bg_scores >= thr))
        points.append((s / n_sig, s / np.sqrt(b) if b > 0 else 0.0))
    # Sort ascending by efficiency for monotonic curve rendering.
    points.sort(key=lambda pt: pt[0])

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot([p[0] for p in points], [p[1] for p in points], color="darkviolet", lw=2)
    ax.set_xlabel("Signal Efficiency (TEST)", fontsize=12)
    ax.set_ylabel("Significance ($S/\\sqrt{B}$, TEST)", fontsize=12)
    ax.set_title(
        "Physics Validation: Significance Profile\n(thresholds from VALIDATION, measured on TEST)",
        fontsize=12,
        fontweight="bold",
    )
    fig.tight_layout()
    fig.savefig(output_path("plots", "6b_physics_significance.png"))
    plt.close(fig)


def _plot_confusion(cm) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix @0.5 (TEST)", fontweight="bold")
    fig.tight_layout()
    fig.savefig(output_path("plots", "5_confusion_matrix.png"))
    plt.close(fig)


def write_metrics_report(baseline: dict, ml: dict, y_test, budget: float = BUDGET) -> Path:
    """Emit both the human-readable report and a machine-readable JSON twin.

    The dashboard reads the JSON rather than hard-coding numbers, which is how
    its reference table drifted away from these results before.
    """
    path = output_path("results", "metrics.md")
    y_test = np.asarray(y_test)
    dummy_acc = max(np.mean(y_test == 0), np.mean(y_test == 1))

    with open(path, "w", encoding="utf-8") as f:
        f.write("# Physics Filtering Benchmark\n\n")
        f.write("> Supervised demonstration of learning a Z-mass-window (80<M<100 GeV)\n")
        f.write("> selection from muon kinematics. Labels are a deterministic function of M;\n")
        f.write("> features already encode M. Not a trigger/discovery benchmark.\n\n")
        f.write(f"**Target Background Acceptance Budget**: {budget * 100}%\n\n")
        f.write("**Protocol**: stratified 60/20/20 train/val/test; thresholds fit on\n")
        f.write("TRAIN (baseline) or VALIDATION (ML) and measured once on frozen TEST.\n")
        f.write("Cuts are placed so background retention cannot exceed the budget\n")
        f.write("(`src.config.threshold_for_budget`).\n\n")
        f.write("### Architectures (TEST set, frozen)\n")
        f.write("| Architecture | Signal Efficiency @5% bg | AUROC | AUPRC | Acc@0.5 |\n")
        f.write("|---|---|---|---|---|\n")
        f.write(
            f"| Rule-Based Cut (pT_lead >= {baseline['pt_cut']:.2f} GeV) | "
            f"**{baseline['sig_eff'] * 100:.2f}%** | — | — | — |\n"
        )
        f.write(
            f"| Kinematic ML Classifier (XGBoost) | **{ml['sig_eff'] * 100:.2f}%** | "
            f"{ml['auroc']:.4f} | {ml['auprc']:.4f} | {ml['accuracy'] * 100:.2f}% |\n\n"
        )
        f.write(
            f"TEST background retained: baseline {baseline['bg_retained'] * 100:.2f}%, "
            f"ML {ml['bg_retained'] * 100:.2f}% (target {budget * 100}%).\n\n"
        )
        f.write(f"TEST confusion @0.5: TN={ml['tn']} FP={ml['fp']} FN={ml['fn']} TP={ml['tp']}.\n\n")
        f.write(
            f"Dummy (all-background) accuracy on TEST would be {dummy_acc * 100:.2f}%; "
            "accuracy alone is misleading under this class imbalance — prefer AUROC/AUPRC above.\n\n"
        )
        f.write(
            f"> Result: at the same ≈5% background acceptance, ML signal efficiency "
            f"({ml['sig_eff'] * 100:.2f}%) vs naive pT cut ({baseline['sig_eff'] * 100:.2f}%): "
            f"**{((ml['sig_eff'] - baseline['sig_eff']) * 100):+.2f}pp** difference on frozen TEST.\n"
        )

    json_path = output_path("results", "metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "budget": budget,
                "protocol": "stratified 60/20/20 train/val/test; thresholds fit on TRAIN (baseline) "
                "or VALIDATION (ML), measured once on frozen TEST",
                "test_events": int(y_test.size),
                "test_signal_fraction": float(np.mean(y_test == 1)),
                "dummy_accuracy": float(dummy_acc),
                "baseline": {k: float(v) for k, v in baseline.items()},
                "ml": {k: (int(v) if isinstance(v, (int, np.integer)) else float(v)) for k, v in ml.items()},
                "xgboost_version": xgb.__version__,
            },
            f,
            indent=2,
        )
        f.write("\n")
    return path


def main(budget: float = BUDGET) -> dict:
    df_raw = load_data()
    X, y, df = build_features_and_labels(df_raw)

    # Leakage-free stratified split: 60% train / 20% val / 20% test
    X_train, X_val, X_test, y_train, y_val, y_test, df_train, _df_val, df_test = stratified_train_val_test_split(
        X, y, df
    )

    print(f"✅ Split (stratified): train={len(X_train)} val={len(X_val)} test={len(X_test)}")
    print(
        f"   Train signal frac: {np.mean(np.asarray(y_train) == 1):.4f} | "
        f"Val: {np.mean(np.asarray(y_val) == 1):.4f} | Test: {np.mean(np.asarray(y_test) == 1):.4f}"
    )

    # 1. Physics Baseline (fit TRAIN, eval TEST)
    baseline = evaluate_rule_based_baseline(df_train, df_test, background_budget_pct=budget)

    # 2. ML Training (TRAIN only)
    model = train_xgboost(X_train, y_train)

    # 3. ML Evaluation (calibrate VAL, eval frozen TEST)
    ml = evaluate_ml_model(model, X_val, y_val, X_test, y_test, background_budget_pct=budget)

    # 4. Generate Results Report
    report = write_metrics_report(baseline, ml, y_test, budget=budget)
    print(f"\n✅ Physics Benchmarks compiled natively to {report}")

    # Persist the classifier. Device was already reset to CPU inside train_xgboost,
    # so the artifact is portable across hosts without warnings.
    joblib_path = output_path(XGB_JOBLIB_FILE)
    joblib.dump(model, str(joblib_path))
    print(f"💾 Model saved locally as {joblib_path}")

    # Save native XGBoost JSON format (production format for C++ / Triton / ROOT TMVA)
    json_path = output_path(XGB_JSON_FILE)
    try:
        model.save_model(str(json_path))
        print(f"💾 Production native model saved as {json_path}")
    except Exception as e:
        print(f"⚠️ Could not export native model: {e}")

    # Generate full invariant mass spectrum plot
    try:
        from src.plot_mass_spectrum import plot_invariant_mass_spectrum

        plot_invariant_mass_spectrum(model_file=str(joblib_path))
    except Exception as e:
        print(f"⚠️ Could not generate mass spectrum plot: {e}")

    return {"baseline": baseline, "ml": ml, "report": str(report)}


if __name__ == "__main__":
    main()
