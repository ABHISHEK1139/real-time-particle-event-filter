"""CERN CMS Real-Time Particle Event Filter - Interactive Streamlit Dashboard.

Allows interactive exploration of CMS Open Data Dimuon events, tuning kinematic
trigger thresholds (leading pT cut vs machine learning filter), and visualizing
the isolated Breit-Wigner Z-boson resonance peak in real time.

Design notes
------------
* The primary ML control is a **background-retention budget**, not a raw
  probability. The trained classifier is so confident that its 5%-background cut
  sits around 2e-4; a probability slider quantised to 0.01 rounds that to 0.0
  and silently admits 100% of events. Driving the cut from the budget via
  :func:`src.config.threshold_for_budget` removes the quantisation trap and is
  also the quantity a trigger is actually specified in.
* Every displayed benchmark number is read from ``results/*.json`` produced by
  the training scripts. Nothing is hard-coded, so the reference tab cannot
  drift away from the measured results.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # headless-safe
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Setup path and environment
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import src._compat  # noqa: F401,E402
from src._compat import data_path  # noqa: E402
from src.config import (  # noqa: E402
    ANOMALY_JOBLIB_FILE,
    BUDGET,
    CSV_FILE,
    FEATURES,
    MASS_SIGNAL_MAX,
    MASS_SIGNAL_MIN,
    PDG_Z_MASS,
    REQUIRED_COLUMNS,
    XGB_JOBLIB_FILE,
    threshold_for_budget,
)

# Page configuration must be the first Streamlit command in the script.
st.set_page_config(
    page_title="CMS Real-Time Particle Event Filter",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

#: Seed for the burst-emulation sampler, so "real-time" runs are reproducible.
EMULATION_SEED = 42
#: Events drawn per burst / number of bursts in the latency emulation.
BURST_SIZES = (100, 500, 1000, 5000)


def _synthetic_events(n: int = 20_000, seed: int = 42) -> pd.DataFrame:
    """Physics-consistent fallback payload (M computed from kinematics)."""
    rng = np.random.default_rng(seed)
    pt1 = rng.uniform(5, 100, n)
    pt2 = rng.uniform(5, 100, n)
    eta1 = rng.uniform(-2.4, 2.4, n)
    eta2 = rng.uniform(-2.4, 2.4, n)
    phi1 = rng.uniform(-np.pi, np.pi, n)
    phi2 = rng.uniform(-np.pi, np.pi, n)
    m = np.sqrt(np.maximum(2 * pt1 * pt2 * (np.cosh(eta1 - eta2) - np.cos(phi1 - phi2)), 1e-9))
    return pd.DataFrame({"pt1": pt1, "pt2": pt2, "eta1": eta1, "eta2": eta2, "phi1": phi1, "phi2": phi2, "M": m})


@st.cache_data(show_spinner=False)
def load_collision_data(max_rows: int = 100_000) -> tuple[pd.DataFrame, str]:
    """Loads CMS Dimuon collision data with caching."""
    csv_file = data_path(CSV_FILE)
    if Path(csv_file).exists():
        df = pd.read_csv(csv_file, nrows=max_rows)
        missing = set(REQUIRED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(f"{CSV_FILE} is missing required columns: {sorted(missing)}")
        return df.dropna(), "CMS Open Data Run2011A DoubleMu"
    return _synthetic_events(), "Physics-Consistent Synthetic Fallback"


@st.cache_resource(show_spinner=False)
def load_ml_model():
    """Loads the trained XGBoost classifier, pinned to CPU inference."""
    model_path = data_path(XGB_JOBLIB_FILE)
    if not Path(model_path).exists():
        return None
    try:
        model = joblib.load(model_path)
        if hasattr(model, "set_params"):
            model.set_params(device="cpu")
        return model
    except Exception as e:
        st.warning(f"Could not load {XGB_JOBLIB_FILE}: {e}")
        return None


@st.cache_resource(show_spinner=False)
def load_anomaly_model():
    """Loads the trained Isolation Forest anomaly detector + its scaler."""
    model_path = data_path(ANOMALY_JOBLIB_FILE)
    if not Path(model_path).exists():
        return None, None
    try:
        payload = joblib.load(model_path)
        return payload["model"], payload["scaler"]
    except Exception as e:
        st.warning(f"Could not load {ANOMALY_JOBLIB_FILE}: {e}")
        return None, None


@st.cache_data(show_spinner=False)
def load_metrics(name: str) -> dict | None:
    """Read a machine-readable metrics file written by the training scripts."""
    path = data_path(f"results/{name}")
    if not Path(path).exists():
        return None
    try:
        with open(path, encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, json.JSONDecodeError):
        return None


@st.cache_data(show_spinner=False)
def score_dataframe(_model, features: pd.DataFrame) -> np.ndarray:
    """P(signal) from the XGBoost classifier.

    ``_model`` is excluded from the cache key on purpose: a fitted estimator is
    unhashable by Streamlit and is already a process-wide singleton from
    :func:`load_ml_model`, so the DataFrame alone is the correct cache key.
    """
    return _model.predict_proba(features)[:, 1]


@st.cache_data(show_spinner=False)
def score_anomaly(_iso, _scaler, features: pd.DataFrame) -> np.ndarray:
    """Higher = more anomalous (Isolation Forest's decision_function is inverted)."""
    return -_iso.decision_function(_scaler.transform(features))


def _clamp(value: float, lo: float, hi: float) -> float:
    """Keep a derived default inside its slider range.

    Streamlit raises if a slider's value falls outside [min_value, max_value];
    thresholds derived from data can do exactly that when the payload changes.
    """
    return float(min(max(value, lo), hi))


def _rejection_factor(bg_retained_pct: float) -> float:
    """1 / background retention, i.e. how much background is thrown away."""
    return float("inf") if bg_retained_pct <= 0 else 100.0 / bg_retained_pct


def main() -> None:
    st.title("⚛️ CERN CMS Real-Time Particle Event Filter")
    st.caption("CMS Open Data DoubleMu Run2011A ($\\sqrt{s} = 7\\text{ TeV}$) — Trigger & Resonance Filtering System")

    # --- Sidebar Controls ---
    st.sidebar.header("🎛️ Filter Controls")

    sample_size = st.sidebar.slider("Events to Analyze", min_value=5000, max_value=100000, value=50000, step=5000)
    with st.spinner("Loading collision data..."):
        try:
            df, data_source = load_collision_data(max_rows=sample_size)
        except (ValueError, OSError) as e:
            st.error(f"Could not load collision data: {e}")
            st.stop()

    st.sidebar.info(f"📁 **Source**: {data_source}\n\n📊 **Loaded**: {len(df):,} events")

    pt_lead = np.maximum(df["pt1"], df["pt2"])
    bg_mask = ((df["M"] <= MASS_SIGNAL_MIN) | (df["M"] >= MASS_SIGNAL_MAX)).to_numpy()
    sig_mask = ~bg_mask

    # 1. Physics baseline cut
    st.sidebar.markdown("---")
    st.sidebar.subheader("1. Physics Baseline Trigger")
    rec_pt_cut = _clamp(np.percentile(pt_lead[bg_mask], (1 - BUDGET) * 100), 5.0, 60.0)
    pt_cut = st.sidebar.slider(
        "Leading Muon $p_T$ Cut [GeV]",
        min_value=5.0,
        max_value=60.0,
        value=round(rec_pt_cut, 1),
        step=0.5,
        help="Filters events where max(pt1, pt2) >= threshold. Standard LHC trigger baseline.",
    )

    # 2. ML Classifier Filter
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Machine Learning Filter")
    xgb_model = load_ml_model()
    has_xgb = xgb_model is not None
    ml_probs = np.zeros(len(df))
    prob_cut = 0.0

    if has_xgb:
        with st.spinner("Scoring events..."):
            ml_probs = score_dataframe(xgb_model, df[FEATURES])
        # Primary control: the background budget, expressed the way a real
        # trigger is specified. Derived cuts cannot be rounded to a useless 0.0.
        budget_pct = st.sidebar.slider(
            "Background Retention Budget [%]",
            min_value=0.1,
            max_value=25.0,
            value=round(BUDGET * 100, 2),
            step=0.1,
            help="Fraction of continuum background the filter is allowed to keep. "
            "Equivalent to a probability cut calibrated on the loaded events.",
        )
        prob_cut = threshold_for_budget(ml_probs[bg_mask], budget_pct / 100.0)

        manual = st.sidebar.checkbox(
            "Override with manual probability cut",
            value=False,
            help="Bypass the budget control and set the classifier probability threshold directly.",
        )
        if manual:
            prob_cut = _clamp(
                st.sidebar.number_input(
                    "Manual P(signal) threshold",
                    min_value=0.0,
                    max_value=1.0,
                    value=float(np.clip(prob_cut, 0.0, 1.0)),
                    step=0.0001,
                    format="%.6f",
                ),
                0.0,
                1.0,
            )
        st.sidebar.caption(f"Effective P(signal) cut: **≥ {prob_cut:.6g}**")
    else:
        st.sidebar.warning(f"XGBoost model not found. Run `python src/train_model.py` to train ({XGB_JOBLIB_FILE}).")

    # 3. Unsupervised Anomaly Detector Filter
    iso_model, iso_scaler = load_anomaly_model()
    has_iso = iso_model is not None and iso_scaler is not None
    iso_scores = np.zeros(len(df))
    iso_cut = 0.0

    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Unsupervised Anomaly Filter")
    use_iso = st.sidebar.checkbox(
        "Enable Anomaly Detector",
        value=False,
        disabled=not has_iso,
        help="Isolation Forest trained without mass labels on collision kinematics.",
    )
    if use_iso and has_iso:
        with st.spinner("Scoring anomaly scores..."):
            iso_scores = score_anomaly(iso_model, iso_scaler, df[FEATURES])
        lo, hi = float(np.min(iso_scores)), float(np.max(iso_scores))
        if hi <= lo:  # degenerate scores: a slider needs a non-zero span
            hi = lo + 1e-6
        rec_iso_cut = _clamp(np.percentile(iso_scores[bg_mask], (1 - BUDGET) * 100), lo, hi)
        iso_cut = st.sidebar.slider(
            "Anomaly Score Threshold", min_value=lo, max_value=hi, value=rec_iso_cut, step=max((hi - lo) / 500.0, 1e-9)
        )
    elif not has_iso:
        st.sidebar.caption(f"No anomaly model found ({ANOMALY_JOBLIB_FILE}); run `python src/anomaly_detection.py`.")

    # Compute masks
    pass_baseline = pt_lead >= pt_cut
    pass_ml = (ml_probs >= prob_cut) if has_xgb else np.zeros(len(df), dtype=bool)
    pass_iso = (iso_scores >= iso_cut) if (use_iso and has_iso) else np.zeros(len(df), dtype=bool)

    # --- KPI Metrics Bar ---
    total_events = len(df)
    total_sig = int(np.sum(sig_mask))
    total_bg = int(np.sum(bg_mask))

    def _rates(mask):
        return (
            (np.sum(mask & sig_mask) / max(total_sig, 1)) * 100,
            (np.sum(mask & bg_mask) / max(total_bg, 1)) * 100,
        )

    base_sig_eff, base_bg_rate = _rates(pass_baseline)
    ml_sig_eff, ml_bg_rate = _rates(pass_ml) if has_xgb else (0.0, 0.0)

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events", f"{total_events:,}", f"Signals: {total_sig:,}")
    with col2:
        st.metric(
            "Baseline Trigger Pass",
            f"{int(np.sum(pass_baseline)):,}",
            f"Eff: {base_sig_eff:.1f}% | BG: {base_bg_rate:.1f}%",
        )
    with col3:
        st.metric(
            "ML Filter Pass",
            f"{int(np.sum(pass_ml)):,}" if has_xgb else "N/A",
            f"Eff: {ml_sig_eff:.1f}% | BG: {ml_bg_rate:.1f}%" if has_xgb else "Model offline",
        )
    with col4:
        if has_xgb:
            st.metric(
                "ML Background Rejection",
                f"{_rejection_factor(ml_bg_rate):,.1f}x",
                f"{ml_sig_eff - base_sig_eff:+.1f}pp signal vs cut",
            )
        else:
            st.metric("ML Background Rejection", "N/A", "Model offline")

    st.markdown("---")

    # --- Main Tabbed Views ---
    tab1, tab2, tab3, tab4 = st.tabs(
        [
            "📊 Invariant Mass Spectrum",
            "🎯 Kinematic Distributions",
            "⚡ Real-Time Emulation",
            "📑 Physics & Metrics Reference",
        ]
    )

    # === TAB 1: Invariant Mass Spectrum ===
    with tab1:
        st.subheader("Dimuon Invariant Mass ($M_{\\mu\\mu}$) Reconstruction")
        st.write(
            "Visualizing how kinematic trigger cuts isolate the Breit-Wigner resonance of the **Z boson** "
            f"($M_Z = {PDG_Z_MASS:.2f}\\text{{ GeV}}$) while discarding non-resonant continuum background."
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Log scale full spectrum
        bins_full = np.linspace(2, 110, 108)
        ax1.hist(
            df["M"],
            bins=bins_full,
            histtype="stepfilled",
            alpha=0.3,
            color="steelblue",
            label=f"Raw Collisions ({len(df):,})",
        )
        ax1.hist(
            df.loc[pass_baseline, "M"],
            bins=bins_full,
            histtype="step",
            lw=2,
            color="darkorange",
            label=f"Baseline $p_T \\geq {pt_cut:.1f}$ GeV ({int(np.sum(pass_baseline)):,})",
        )
        if has_xgb:
            ax1.hist(
                df.loc[pass_ml, "M"],
                bins=bins_full,
                histtype="step",
                lw=2.5,
                color="crimson",
                label=f"XGBoost Filter $P \\geq {prob_cut:.4g}$ ({int(np.sum(pass_ml)):,})",
            )
        if use_iso and has_iso:
            ax1.hist(
                df.loc[pass_iso, "M"],
                bins=bins_full,
                histtype="step",
                lw=1.5,
                color="forestgreen",
                linestyle="--",
                label=f"Anomaly Filter ({int(np.sum(pass_iso)):,})",
            )

        ax1.set_yscale("log")
        ax1.set_xlabel("Dimuon Invariant Mass $M_{\\mu\\mu}$ [GeV]", fontsize=11, fontweight="bold")
        ax1.set_ylabel("Events / 1.0 GeV", fontsize=11, fontweight="bold")
        ax1.set_title("Global Spectrum (2 - 110 GeV, Log Scale)", fontsize=12, fontweight="bold")
        ax1.grid(True, which="both", linestyle="--", alpha=0.4)
        ax1.legend(loc="upper right", frameon=True, fontsize=9)

        # Panel 2: Linear scale Z-peak zoom
        bins_z = np.linspace(60, 120, 60)
        z_mask = (df["M"] >= 60) & (df["M"] <= 120)

        ax2.hist(
            df.loc[z_mask, "M"],
            bins=bins_z,
            histtype="stepfilled",
            alpha=0.25,
            color="steelblue",
            label="Raw (Z-Window)",
        )
        ax2.hist(
            df.loc[z_mask & pass_baseline, "M"],
            bins=bins_z,
            histtype="step",
            lw=2,
            color="darkorange",
            label="Baseline Cut",
        )
        if has_xgb:
            ax2.hist(
                df.loc[z_mask & pass_ml, "M"], bins=bins_z, histtype="step", lw=2.5, color="crimson", label="ML Filter"
            )
        if use_iso and has_iso:
            ax2.hist(
                df.loc[z_mask & pass_iso, "M"],
                bins=bins_z,
                histtype="step",
                lw=1.5,
                color="forestgreen",
                linestyle="--",
                label="Anomaly Filter",
            )

        ax2.axvline(PDG_Z_MASS, color="black", linestyle=":", lw=2, label=f"PDG $M_Z = {PDG_Z_MASS:.2f}$ GeV")
        ax2.axvspan(MASS_SIGNAL_MIN, MASS_SIGNAL_MAX, color="gold", alpha=0.15, label="Signal Mass Window")

        ax2.set_xlabel("Dimuon Invariant Mass $M_{\\mu\\mu}$ [GeV]", fontsize=11, fontweight="bold")
        ax2.set_ylabel("Events / 1.0 GeV", fontsize=11, fontweight="bold")
        ax2.set_title("Z Boson Peak Zoom (60 - 120 GeV, Linear Scale)", fontsize=12, fontweight="bold")
        ax2.grid(True, linestyle="--", alpha=0.4)
        ax2.legend(loc="upper right", frameon=True, fontsize=9)

        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)

    # === TAB 2: Kinematic Distributions ===
    with tab2:
        st.subheader("Kinematic Feature Space")
        st.write("Compare transverse momentum ($p_T$), pseudorapidity ($\\eta$), and angular distributions.")

        feat_col1, feat_col2 = st.columns(2)

        with feat_col1:
            # pT1 vs pT2 scatter (sample subset for render performance)
            plot_df = df.sample(min(3000, len(df)), random_state=42)
            # Reuse the already-computed scores: re-running the model here both
            # wasted ~100x the work and could disagree with the displayed cut.
            plot_scores = ml_probs[plot_df.index] if has_xgb else np.zeros(len(plot_df))
            scatter_mask = plot_scores >= prob_cut

            fig_sc, ax_sc = plt.subplots(figsize=(6, 5))
            if has_xgb:
                ax_sc.scatter(
                    plot_df.loc[~scatter_mask, "pt1"],
                    plot_df.loc[~scatter_mask, "pt2"],
                    s=12,
                    c="gray",
                    alpha=0.3,
                    label="Rejected Background",
                )
                ax_sc.scatter(
                    plot_df.loc[scatter_mask, "pt1"],
                    plot_df.loc[scatter_mask, "pt2"],
                    s=18,
                    c="crimson",
                    alpha=0.6,
                    label="Accepted Candidate",
                )
            else:
                ax_sc.scatter(plot_df["pt1"], plot_df["pt2"], s=12, c="steelblue", alpha=0.4, label="Events")
            ax_sc.set_xlabel("$p_{T1}$ [GeV]", fontsize=11, fontweight="bold")
            ax_sc.set_ylabel("$p_{T2}$ [GeV]", fontsize=11, fontweight="bold")
            ax_sc.set_title("Transverse Momentum Correlation: $p_{T1}$ vs $p_{T2}$", fontsize=11, fontweight="bold")
            ax_sc.set_xlim(0, 80)
            ax_sc.set_ylim(0, 80)
            ax_sc.grid(True, linestyle="--", alpha=0.4)
            ax_sc.legend(loc="upper right")
            fig_sc.tight_layout()
            st.pyplot(fig_sc)
            plt.close(fig_sc)

        with feat_col2:
            # Delta Phi (back-to-back topology); wrapped to [0, pi].
            dphi = np.abs(df["phi1"] - df["phi2"])
            dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)

            fig_ang, ax_ang = plt.subplots(figsize=(6, 5))
            ax_ang.hist(
                dphi[sig_mask],
                bins=40,
                density=True,
                histtype="step",
                lw=2,
                color="crimson",
                label="Signal Window ($80 < M < 100$)",
            )
            ax_ang.hist(
                dphi[bg_mask],
                bins=40,
                density=True,
                histtype="step",
                lw=2,
                color="steelblue",
                label="Background Continuum",
            )
            ax_ang.set_xlabel("Opening Angle $\\Delta\\phi$ [rad]", fontsize=11, fontweight="bold")
            ax_ang.set_ylabel("Probability Density", fontsize=11, fontweight="bold")
            ax_ang.set_title(
                "Muon Back-to-Back Topology ($\\Delta\\phi \\approx \\pi$)", fontsize=11, fontweight="bold"
            )
            ax_ang.grid(True, linestyle="--", alpha=0.4)
            ax_ang.legend(loc="upper left")
            fig_ang.tight_layout()
            st.pyplot(fig_ang)
            plt.close(fig_ang)

    # === TAB 3: Real-Time Trigger Emulation ===
    with tab3:
        st.subheader("⚡ Real-Time Trigger Latency & Throughput Emulation")
        st.write(
            "Replays the loaded events in bursts and times the classifier on each burst. "
            "This is a local benchmark of the model, not a live HLT feed."
        )

        stream_col1, stream_col2 = st.columns([1, 2])

        with stream_col1:
            batch_size = st.selectbox("Batch Size (Collisions / Burst)", BURST_SIZES, index=1)
            num_batches = st.slider("Number of Bursts", min_value=5, max_value=50, value=15)
            run_btn = st.button("🚀 Run Trigger Emulation", type="primary", disabled=not has_xgb)

        with stream_col2:
            if not run_btn:
                st.info('Click "Run Trigger Emulation" to benchmark burst latency and throughput.')
            elif not has_xgb:
                st.error("Cannot run ML emulation: XGBoost model not loaded.")
            else:
                progress_bar = st.progress(0.0)
                status_text = st.empty()
                metrics_holder = st.empty()

                rng = np.random.default_rng(EMULATION_SEED)  # reproducible bursts
                latencies = []
                passed_counts = 0
                total_simulated = 0
                sample_features = df[FEATURES].to_numpy()
                n_available = len(sample_features)

                for i in range(num_batches):
                    idx = rng.choice(n_available, batch_size, replace=batch_size > n_available)
                    batch_X = sample_features[idx]

                    t0 = time.perf_counter()
                    probs = xgb_model.predict_proba(batch_X)[:, 1]
                    passed_counts += int(np.sum(probs >= prob_cut))
                    total_simulated += batch_size
                    latencies.append((time.perf_counter() - t0) * 1000)  # ms

                    progress_bar.progress((i + 1) / num_batches)
                    status_text.text(f"Processed Burst {i + 1}/{num_batches} — Latency: {latencies[-1]:.2f} ms")
                    time.sleep(0.03)  # brief pacing for visual feedback

                latencies = np.asarray(latencies)
                total_inference_s = float(latencies.sum()) / 1000.0
                avg_lat = float(latencies.mean())
                p99_lat = float(np.percentile(latencies, 99))
                # Guard the divide: a sub-microsecond total used to raise
                # ZeroDivisionError instead of reporting the throughput.
                ev_per_sec = (total_simulated / total_inference_s) if total_inference_s > 0 else 0.0

                with metrics_holder.container():
                    m1, m2, m3 = st.columns(3)
                    m1.metric("Avg Burst Latency", f"{avg_lat:.2f} ms", f"P99: {p99_lat:.2f} ms")
                    m2.metric("Throughput", f"{ev_per_sec:,.0f} ev/s", f"{total_simulated:,} ev total")
                    m3.metric(
                        "Filtered Storage Rate",
                        f"{(passed_counts / total_simulated) * 100:.1f}%",
                        f"{passed_counts:,} saved to tape",
                    )

                    fig_lat, ax_lat = plt.subplots(figsize=(6, 3))
                    ax_lat.plot(range(1, num_batches + 1), latencies, marker="o", color="teal")
                    ax_lat.axhline(avg_lat, color="red", linestyle="--", label=f"Mean: {avg_lat:.2f} ms")
                    ax_lat.set_xlabel("Burst Number")
                    ax_lat.set_ylabel("Inference Latency [ms]")
                    ax_lat.set_title("Trigger Burst Latency Stability")
                    ax_lat.grid(True, linestyle="--", alpha=0.4)
                    ax_lat.legend()
                    fig_lat.tight_layout()
                    st.pyplot(fig_lat)
                    plt.close(fig_lat)

    # === TAB 4: Architecture & Physics Reference ===
    with tab4:
        st.subheader("📑 Physics Benchmark & Architecture Comparison")
        st.write(
            "In HEP triggers the event rate must be cut by orders of magnitude while preserving clean "
            "resonances. The table below is generated from `results/metrics.json` and "
            "`results/anomaly_metrics.json`; run the training scripts to populate it."
        )

        sup = load_metrics("metrics.json")
        ano = load_metrics("anomaly_metrics.json")
        if sup is None:
            st.info("No supervised metrics found — run `python src/train_model.py`.")
        if ano is None:
            st.info("No anomaly metrics found — run `python src/anomaly_detection.py`.")

        if sup is not None:
            budget_pct = sup.get("budget", BUDGET) * 100
            b, m = sup["baseline"], sup["ml"]
            rows = [
                (
                    "Rule-Based Cut (physics baseline)",
                    f"$p_T^{{\\text{{lead}}}} \\geq {b['pt_cut']:.2f}$ GeV",
                    "Physics Baseline",
                    f"{b['sig_eff'] * 100:.1f}%",
                    "—",
                    "Hardcoded / L1",
                ),
                (
                    "Gradient Boosted Trees (XGBoost)",
                    f"$P(\\mu\\mu) \\geq {m['prob_cut']:.4g}$",
                    "Supervised ML",
                    f"{m['sig_eff'] * 100:.1f}%",
                    f"{m['auroc']:.3f}",
                    "JSON / Treelite / Triton",
                ),
            ]
            if ano is not None:
                rows.append(
                    (
                        "Isolation Forest",
                        f"score $\\geq {ano['threshold']:.4g}$",
                        "Unsupervised Anomaly",
                        f"{ano['sig_eff'] * 100:.1f}%",
                        f"{ano['auroc']:.3f}",
                        "Joblib / ONNX",
                    )
                )

            st.markdown(
                f"**Protocol**: {sup.get('protocol', 'n/a')}\n\n"
                f"**Frozen TEST split**: {sup.get('test_events', 0):,} events, "
                f"signal fraction {sup.get('test_signal_fraction', 0) * 100:.2f}%, "
                f"all-background dummy accuracy {sup.get('dummy_accuracy', 0) * 100:.2f}%."
            )
            table = [
                "| Architecture | Selection | Paradigm | Signal Eff @budget | AUROC | Deployment |",
                "|---|---|---|---|---|---|",
            ]
            table += [f"| **{name}** | {sel} | {par} | {eff} | {au} | {dep} |" for name, sel, par, eff, au, dep in rows]
            st.markdown("\n".join(table))
            st.caption(f"All efficiencies measured at a {budget_pct:.1f}% background budget on the frozen test split.")

        st.markdown("\n#### Key Invariant Mass Formula")
        st.latex(r"M_{\mu\mu} = \sqrt{2\,p_{T1} p_{T2}\,(\cosh(\eta_1 - \eta_2) - \cos(\phi_1 - \phi_2))}")
        st.write(
            "Supervised models are trained **strictly without** $M_{\\mu\\mu}$ to prevent circular "
            "label leakage in the feature matrix — but since kinematics determine $M$, this is "
            "mass-window learning, not trigger-level discovery. See `PHYSICS.md`."
        )


if __name__ == "__main__":
    main()
