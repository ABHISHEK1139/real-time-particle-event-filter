"""Reconstruct and visualize the Dimuon Invariant Mass spectrum.

Demonstrates how the Z-boson Breit-Wigner peak (~91.2 GeV) is isolated from
the low-mass Drell-Yan and QCD continuum by the leading-pT baseline cut
and the machine learning kinematic filter.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import src._compat  # noqa: F401  (UTF-8 stdout guard)

import joblib
import matplotlib
import numpy as np

matplotlib.use("Agg")  # headless-safe
import matplotlib.pyplot as plt
import pandas as pd

from src._compat import data_path
from src.config import (
    BUDGET,
    CSV_FILE,
    FEATURES,
    MASS_SIGNAL_MAX,
    MASS_SIGNAL_MIN,
    PDG_Z_MASS,
    REQUIRED_COLUMNS,
    XGB_JOBLIB_FILE,
    output_path as resolve_output_path,
    threshold_for_budget,
)

SPECTRUM_FILE = "plots/1_invariant_mass_spectrum.png"


def _calibrate_cut(scores, bg_mask: np.ndarray) -> float:
    """Cut at the background budget using background events only."""
    return threshold_for_budget(np.asarray(scores)[bg_mask], BUDGET)


def plot_invariant_mass_spectrum(
    csv_file: str = CSV_FILE, model_file: str = XGB_JOBLIB_FILE, output_path: str | None = None
) -> str:
    print("🔬 [PHYSICS VISUALIZATION] Reconstructing Dimuon Invariant Mass Spectrum...")

    csv_path = data_path(csv_file)
    if not Path(csv_path).exists():
        raise FileNotFoundError(f"Dataset {csv_file} not found. Run python src/data_download.py first.")

    df = pd.read_csv(csv_path).dropna()
    missing = set(REQUIRED_COLUMNS) - set(df.columns)
    if missing:
        raise ValueError(f"Dataset missing required columns: {sorted(missing)}")

    target = Path(output_path) if output_path else resolve_output_path(SPECTRUM_FILE)

    # 1. Baseline cut on leading muon pT
    pt_lead = np.maximum(df["pt1"], df["pt2"])
    bg_mask = (df["M"] <= MASS_SIGNAL_MIN) | (df["M"] >= MASS_SIGNAL_MAX)
    pt_cut = _calibrate_cut(pt_lead, bg_mask)  # 5% background acceptance budget
    baseline_pass = pt_lead >= pt_cut

    # 2. ML Filter
    model_path = data_path(model_file)
    ml_pass = np.zeros(len(df), dtype=bool)
    has_model = Path(model_path).exists()
    if has_model:
        try:
            model = joblib.load(model_path)
            if hasattr(model, "set_params"):
                model.set_params(device="cpu")
            scores = model.predict_proba(df[FEATURES])[:, 1]
            ml_pass = scores >= _calibrate_cut(scores, bg_mask)
        except Exception as e:
            print(f"⚠️ Could not evaluate ML model for spectrum ({e}); proceeding with baseline only.")
            has_model = False

    # Plotting: Two panels (Global spectrum log-scale + Z-peak zoom linear scale)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # --- Panel 1: Full Range 2 - 110 GeV (Log Scale) ---
    bins_full = np.linspace(2, 110, 108)
    ax1.hist(
        df["M"],
        bins=bins_full,
        histtype="stepfilled",
        alpha=0.3,
        color="steelblue",
        label=f"Raw Data ({len(df):,} evts)",
    )
    ax1.hist(
        df.loc[baseline_pass, "M"],
        bins=bins_full,
        histtype="step",
        lw=2,
        color="darkorange",
        label=f"Leading $p_T \\geq {pt_cut:.1f}$ GeV ({np.sum(baseline_pass):,} evts)",
    )
    if has_model:
        ax1.hist(
            df.loc[ml_pass, "M"],
            bins=bins_full,
            histtype="step",
            lw=2.5,
            color="crimson",
            label=f"XGBoost Filter ({np.sum(ml_pass):,} evts)",
        )

    ax1.set_yscale("log")
    ax1.set_xlabel("Dimuon Invariant Mass $M_{\\mu\\mu}$ [GeV]", fontsize=12, fontweight="bold")
    ax1.set_ylabel("Events / 1.0 GeV", fontsize=12, fontweight="bold")
    ax1.set_title(
        "Full Invariant Mass Spectrum (Log Scale)\nCMS Open Data DoubleMu Run2011A $\\sqrt{s}=7$ TeV", fontsize=12
    )
    ax1.grid(True, which="both", linestyle="--", alpha=0.5)
    ax1.legend(loc="upper right", frameon=True)

    # --- Panel 2: Z Boson Resonance Window 60 - 120 GeV (Linear Scale) ---
    bins_z = np.linspace(60, 120, 60)
    z_mask = (df["M"] >= 60) & (df["M"] <= 120)

    ax2.hist(
        df.loc[z_mask, "M"], bins=bins_z, histtype="stepfilled", alpha=0.25, color="steelblue", label="Raw (Z Window)"
    )
    ax2.hist(
        df.loc[z_mask & baseline_pass, "M"],
        bins=bins_z,
        histtype="step",
        lw=2,
        color="darkorange",
        label="Leading $p_T$ Cut",
    )
    if has_model:
        ax2.hist(
            df.loc[z_mask & ml_pass, "M"], bins=bins_z, histtype="step", lw=2.5, color="crimson", label="XGBoost Filter"
        )

    ax2.axvline(PDG_Z_MASS, color="black", linestyle=":", lw=2, label="PDG $M_Z = 91.19$ GeV")
    ax2.axvspan(MASS_SIGNAL_MIN, MASS_SIGNAL_MAX, color="gold", alpha=0.15, label="Signal Mass Window [80, 100] GeV")

    ax2.set_xlabel("Dimuon Invariant Mass $M_{\\mu\\mu}$ [GeV]", fontsize=12, fontweight="bold")
    ax2.set_ylabel("Events / 1.0 GeV", fontsize=12, fontweight="bold")
    ax2.set_title("Z Boson Resonance Peak (Linear Scale)\nBreit-Wigner Line Shape Isolation", fontsize=12)
    ax2.grid(True, linestyle="--", alpha=0.5)
    ax2.legend(loc="upper right", frameon=True)

    fig.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=150)
    plt.close(fig)
    print(f"✅ Invariant mass spectrum figure saved to {target}")
    return str(target)


if __name__ == "__main__":
    plot_invariant_mass_spectrum()
