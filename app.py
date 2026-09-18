"""CERN CMS Real-Time Particle Event Filter - Interactive Streamlit Dashboard.

Allows interactive exploration of CMS Open Data Dimuon events, tuning kinematic
trigger thresholds (leading pT cut vs machine learning filter), and visualizing
the isolated Breit-Wigner Z-boson resonance peak in real time.
"""
import os
import sys
import time
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib

# Setup path and environment
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
import src._compat  # noqa: F401
from src._compat import data_path

# Page configuration
st.set_page_config(
    page_title="CMS Real-Time Particle Event Filter",
    page_icon="⚛️",
    layout="wide",
    initial_sidebar_state="expanded",
)

FEATURES = ['pt1', 'pt2', 'eta1', 'eta2', 'phi1', 'phi2']
MASS_SIGNAL_MIN = 80.0
MASS_SIGNAL_MAX = 100.0
PDG_Z_MASS = 91.1876


@st.cache_data(show_spinner=False)
def load_collision_data(max_rows=100000):
    """Loads CMS Dimuon collision data with caching."""
    csv_file = data_path("Dimuon_DoubleMu.csv")
    if os.path.exists(csv_file):
        df = pd.read_csv(csv_file, nrows=max_rows).dropna()
        return df, "CMS Open Data Run2011A DoubleMu"

    # Fallback to physics-consistent synthetic data
    rng = np.random.default_rng(42)
    n = 20000
    pt1 = rng.uniform(5, 100, n)
    pt2 = rng.uniform(5, 100, n)
    eta1 = rng.uniform(-2.4, 2.4, n)
    eta2 = rng.uniform(-2.4, 2.4, n)
    phi1 = rng.uniform(-np.pi, np.pi, n)
    phi2 = rng.uniform(-np.pi, np.pi, n)
    m = np.sqrt(np.maximum(2 * pt1 * pt2 * (np.cosh(eta1 - eta2) - np.cos(phi1 - phi2)), 1e-9))
    df = pd.DataFrame({'pt1': pt1, 'pt2': pt2, 'eta1': eta1, 'eta2': eta2,
                       'phi1': phi1, 'phi2': phi2, 'M': m})
    return df, "Physics-Consistent Synthetic Fallback"


@st.cache_resource(show_spinner=False)
def load_ml_model():
    """Loads trained XGBoost classifier."""
    model_path = data_path("z_boson_xgb_model.joblib")
    if os.path.exists(model_path):
        try:
            model = joblib.load(model_path)
            if hasattr(model, 'set_params'):
                model.set_params(device='cpu')
            return model, True
        except Exception as e:
            return None, False
    return None, False


@st.cache_resource(show_spinner=False)
def load_anomaly_model():
    """Loads trained Isolation Forest anomaly detector."""
    model_path = data_path("models/anomaly_detector.joblib")
    if os.path.exists(model_path):
        try:
            data = joblib.load(model_path)
            return data['model'], data['scaler'], True
        except Exception:
            return None, None, False
    return None, None, False


def main():
    st.title("⚛️ CERN CMS Real-Time Particle Event Filter")
    st.caption("CMS Open Data DoubleMu Run2011A ($\\sqrt{s} = 7\\text{ TeV}$) — Trigger & Resonance Filtering System")

    # --- Sidebar Controls ---
    st.sidebar.header("🎛️ Filter Controls")

    # Data limits
    sample_size = st.sidebar.slider("Events to Analyze", min_value=5000, max_value=100000, value=50000, step=5000)
    with st.spinner("Loading collision data..."):
        df, data_source = load_collision_data(max_rows=sample_size)

    st.sidebar.info(f"📁 **Source**: {data_source}\n\n📊 **Loaded**: {len(df):,} events")

    # 1. Physics baseline cut
    pt_lead = np.maximum(df['pt1'], df['pt2'])
    bg_mask = (df['M'] <= MASS_SIGNAL_MIN) | (df['M'] >= MASS_SIGNAL_MAX)
    sig_mask = ~bg_mask

    # Recommended 5% budget threshold
    rec_pt_cut = float(np.percentile(pt_lead[bg_mask], 95)) if np.any(bg_mask) else 30.0

    st.sidebar.markdown("---")
    st.sidebar.subheader("1. Physics Baseline Trigger")
    pt_cut = st.sidebar.slider(
        "Leading Muon $p_T$ Cut [GeV]",
        min_value=5.0,
        max_value=60.0,
        value=round(rec_pt_cut, 1),
        step=0.5,
        help="Filters events where max(pt1, pt2) >= threshold. Standard LHC trigger baseline."
    )

    # 2. ML Classifier Filter
    st.sidebar.markdown("---")
    st.sidebar.subheader("2. Machine Learning Filter")
    xgb_model, has_xgb = load_ml_model()

    if has_xgb:
        ml_probs = xgb_model.predict_proba(df[FEATURES])[:, 1]
        rec_prob_cut = float(np.percentile(ml_probs[bg_mask], 95)) if np.any(bg_mask) else 0.84
        prob_cut = st.sidebar.slider(
            "XGBoost Probability Threshold",
            min_value=0.0,
            max_value=1.0,
            value=round(rec_prob_cut, 2),
            step=0.01,
            help="Threshold for kinematic ML decision: P(signal) >= threshold."
        )
    else:
        st.sidebar.warning("XGBoost model not found. Run `python src/train_model.py` to train.")
        ml_probs = np.zeros(len(df))
        prob_cut = 0.5

    # 3. Unsupervised Anomaly Detector Filter
    iso_model, iso_scaler, has_iso = load_anomaly_model()
    st.sidebar.markdown("---")
    st.sidebar.subheader("3. Unsupervised Anomaly Filter")
    use_iso = st.sidebar.checkbox("Enable Anomaly Detector", value=False,
                                  help="Isolation Forest trained without mass labels on collision kinematics.")
    if use_iso and has_iso:
        X_scaled = iso_scaler.transform(df[FEATURES])
        iso_scores = -iso_model.decision_function(X_scaled)
        rec_iso_cut = float(np.percentile(iso_scores[bg_mask], 95)) if np.any(bg_mask) else 0.0
        iso_cut = st.sidebar.slider(
            "Anomaly Score Threshold",
            min_value=float(np.min(iso_scores)),
            max_value=float(np.max(iso_scores)),
            value=round(rec_iso_cut, 3),
            step=0.01
        )
    else:
        iso_scores = np.zeros(len(df))
        iso_cut = 0.0

    # Compute masks
    pass_baseline = pt_lead >= pt_cut
    pass_ml = ml_probs >= prob_cut if has_xgb else np.zeros(len(df), dtype=bool)
    pass_iso = (iso_scores >= iso_cut) if (use_iso and has_iso) else np.zeros(len(df), dtype=bool)

    # --- KPI Metrics Bar ---
    total_events = len(df)
    total_sig = int(np.sum(sig_mask))
    total_bg = int(np.sum(bg_mask))

    base_sig_eff = (np.sum(pass_baseline & sig_mask) / max(total_sig, 1)) * 100
    base_bg_rate = (np.sum(pass_baseline & bg_mask) / max(total_bg, 1)) * 100

    ml_sig_eff = (np.sum(pass_ml & sig_mask) / max(total_sig, 1)) * 100 if has_xgb else 0
    ml_bg_rate = (np.sum(pass_ml & bg_mask) / max(total_bg, 1)) * 100 if has_xgb else 0

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total Events", f"{total_events:,}", f"Signals: {total_sig:,}")
    with col2:
        st.metric("Baseline Trigger Pass", f"{np.sum(pass_baseline):,}",
                  f"Eff: {base_sig_eff:.1f}% | BG: {base_bg_rate:.1f}%")
    with col3:
        st.metric("ML Filter Pass", f"{np.sum(pass_ml):,}" if has_xgb else "N/A",
                  f"Eff: {ml_sig_eff:.1f}% | BG: {ml_bg_rate:.1f}%" if has_xgb else "Model offline")
    with col4:
        bg_rejection = (100.0 / max(ml_bg_rate, 0.001)) if has_xgb else 1.0
        st.metric("ML Background Rejection", f"{bg_rejection:.1f}x",
                  f"+{(ml_sig_eff - base_sig_eff):.1f}pp Signal vs Cut" if has_xgb else "")

    st.markdown("---")

    # --- Main Tabbed Views ---
    tab1, tab2, tab3, tab4 = st.tabs([
        "📊 Invariant Mass Spectrum",
        "🎯 Kinematic Distributions",
        "⚡ Real-Time Emulation",
        "📑 Physics & Metrics Reference",
    ])

    # === TAB 1: Invariant Mass Spectrum ===
    with tab1:
        st.subheader("Dimuon Invariant Mass ($M_{\\mu\\mu}$) Reconstruction")
        st.write(
            "Visualizing how kinematic trigger cuts isolate the Breit-Wigner resonance of the **Z boson** "
            "($M_Z = 91.19\\text{ GeV}$) while discarding non-resonant continuum background."
        )

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

        # Panel 1: Log scale full spectrum
        bins_full = np.linspace(2, 110, 108)
        ax1.hist(df['M'], bins=bins_full, histtype='stepfilled', alpha=0.3, color='steelblue',
                 label=f'Raw Collisions ({len(df):,})')
        ax1.hist(df.loc[pass_baseline, 'M'], bins=bins_full, histtype='step', lw=2, color='darkorange',
                 label=f'Baseline $p_T \\geq {pt_cut:.1f}$ GeV ({np.sum(pass_baseline):,})')
        if has_xgb:
            ax1.hist(df.loc[pass_ml, 'M'], bins=bins_full, histtype='step', lw=2.5, color='crimson',
                     label=f'XGBoost Filter $P \\geq {prob_cut:.2f}$ ({np.sum(pass_ml):,})')
        if use_iso and has_iso:
            ax1.hist(df.loc[pass_iso, 'M'], bins=bins_full, histtype='step', lw=1.5, color='forestgreen',
                     linestyle='--', label=f'Anomaly Filter ({np.sum(pass_iso):,})')

        ax1.set_yscale('log')
        ax1.set_xlabel('Dimuon Invariant Mass $M_{\\mu\\mu}$ [GeV]', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Events / 1.0 GeV', fontsize=11, fontweight='bold')
        ax1.set_title('Global Spectrum (2 - 110 GeV, Log Scale)', fontsize=12, fontweight='bold')
        ax1.grid(True, which='both', linestyle='--', alpha=0.4)
        ax1.legend(loc='upper right', frameon=True, fontsize=9)

        # Panel 2: Linear scale Z-peak zoom
        bins_z = np.linspace(60, 120, 60)
        z_mask = (df['M'] >= 60) & (df['M'] <= 120)

        ax2.hist(df.loc[z_mask, 'M'], bins=bins_z, histtype='stepfilled', alpha=0.25, color='steelblue', label='Raw (Z-Window)')
        ax2.hist(df.loc[z_mask & pass_baseline, 'M'], bins=bins_z, histtype='step', lw=2, color='darkorange', label='Baseline Cut')
        if has_xgb:
            ax2.hist(df.loc[z_mask & pass_ml, 'M'], bins=bins_z, histtype='step', lw=2.5, color='crimson', label='ML Filter')
        if use_iso and has_iso:
            ax2.hist(df.loc[z_mask & pass_iso, 'M'], bins=bins_z, histtype='step', lw=1.5, color='forestgreen',
                     linestyle='--', label='Anomaly Filter')

        ax2.axvline(PDG_Z_MASS, color='black', linestyle=':', lw=2, label=f'PDG $M_Z = {PDG_Z_MASS:.2f}$ GeV')
        ax2.axvspan(MASS_SIGNAL_MIN, MASS_SIGNAL_MAX, color='gold', alpha=0.15, label='Signal Mass Window')

        ax2.set_xlabel('Dimuon Invariant Mass $M_{\\mu\\mu}$ [GeV]', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Events / 1.0 GeV', fontsize=11, fontweight='bold')
        ax2.set_title('Z Boson Peak Zoom (60 - 120 GeV, Linear Scale)', fontsize=12, fontweight='bold')
        ax2.grid(True, linestyle='--', alpha=0.4)
        ax2.legend(loc='upper right', frameon=True, fontsize=9)

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
            fig_sc, ax_sc = plt.subplots(figsize=(6, 5))
            if has_xgb:
                scatter_mask = xgb_model.predict_proba(plot_df[FEATURES])[:, 1] >= prob_cut
                ax_sc.scatter(plot_df.loc[~scatter_mask, 'pt1'], plot_df.loc[~scatter_mask, 'pt2'],
                              s=12, c='gray', alpha=0.3, label='Rejected Background')
                ax_sc.scatter(plot_df.loc[scatter_mask, 'pt1'], plot_df.loc[scatter_mask, 'pt2'],
                              s=18, c='crimson', alpha=0.6, label='Accepted Candidate')
            else:
                ax_sc.scatter(plot_df['pt1'], plot_df['pt2'], s=12, c='steelblue', alpha=0.4, label='Events')
            ax_sc.set_xlabel('$p_{T1}$ [GeV]', fontsize=11, fontweight='bold')
            ax_sc.set_ylabel('$p_{T2}$ [GeV]', fontsize=11, fontweight='bold')
            ax_sc.set_title('Transverse Momentum Correlation: $p_{T1}$ vs $p_{T2}$', fontsize=11, fontweight='bold')
            ax_sc.set_xlim(0, 80)
            ax_sc.set_ylim(0, 80)
            ax_sc.grid(True, linestyle='--', alpha=0.4)
            ax_sc.legend(loc='upper right')
            fig_sc.tight_layout()
            st.pyplot(fig_sc)
            plt.close(fig_sc)

        with feat_col2:
            # Delta Phi and Delta Eta
            deta = np.abs(df['eta1'] - df['eta2'])
            dphi = np.abs(df['phi1'] - df['phi2'])
            dphi = np.where(dphi > np.pi, 2 * np.pi - dphi, dphi)

            fig_ang, ax_ang = plt.subplots(figsize=(6, 5))
            ax_ang.hist(dphi[sig_mask], bins=40, density=True, histtype='step', lw=2, color='crimson',
                        label='Signal Window ($80 < M < 100$)')
            ax_ang.hist(dphi[bg_mask], bins=40, density=True, histtype='step', lw=2, color='steelblue',
                        label='Background Continuum')
            ax_ang.set_xlabel('Opening Angle $\\Delta\\phi$ [rad]', fontsize=11, fontweight='bold')
            ax_ang.set_ylabel('Probability Density', fontsize=11, fontweight='bold')
            ax_ang.set_title('Muon Back-to-Back Topology ($\\Delta\\phi \\approx \\pi$)', fontsize=11, fontweight='bold')
            ax_ang.grid(True, linestyle='--', alpha=0.4)
            ax_ang.legend(loc='upper left')
            fig_ang.tight_layout()
            st.pyplot(fig_ang)
            plt.close(fig_ang)

    # === TAB 3: Real-Time Trigger Emulation ===
    with tab3:
        st.subheader("⚡ Real-Time Trigger Latency & Throughput Emulation")
        st.write("Simulates microsecond-level collision batch inference matching LHC High-Level Trigger (HLT) pipeline.")

        stream_col1, stream_col2 = st.columns([1, 2])

        with stream_col1:
            batch_size = st.selectbox("Batch Size (Collisions / Burst)", [100, 500, 1000, 5000], index=1)
            num_batches = st.slider("Number of Bursts", min_value=5, max_value=50, value=15)
            run_btn = st.button("🚀 Run Trigger Emulation", type="primary")

        with stream_col2:
            if run_btn:
                if not has_xgb:
                    st.error("Cannot run ML emulation: XGBoost model not loaded.")
                else:
                    progress_bar = st.progress(0.0)
                    status_text = st.empty()
                    metrics_holder = st.empty()

                    latencies = []
                    passed_counts = 0
                    total_simulated = 0

                    sample_features = df[FEATURES].values

                    for i in range(num_batches):
                        # Sample random batch
                        idx = np.random.choice(len(sample_features), batch_size, replace=True)
                        batch_X = sample_features[idx]

                        t0 = time.perf_counter()
                        probs = xgb_model.predict_proba(batch_X)[:, 1]
                        accepted = np.sum(probs >= prob_cut)
                        dt = (time.perf_counter() - t0) * 1000  # ms

                        latencies.append(dt)
                        passed_counts += accepted
                        total_simulated += batch_size

                        progress_bar.progress((i + 1) / num_batches)
                        status_text.text(f"Processed Burst {i+1}/{num_batches} — Latency: {dt:.2f} ms")
                        time.sleep(0.03)  # brief pacing for visual feedback

                    avg_lat = np.mean(latencies)
                    p99_lat = np.percentile(latencies, 99)
                    ev_per_sec = (total_simulated / (np.sum(latencies) / 1000.0))

                    with metrics_holder.container():
                        m1, m2, m3 = st.columns(3)
                        m1.metric("Avg Burst Latency", f"{avg_lat:.2f} ms", f"P99: {p99_lat:.2f} ms")
                        m2.metric("Throughput", f"{ev_per_sec:,.0f} ev/s", f"{total_simulated:,} ev total")
                        m3.metric("Filtered Storage Rate", f"{(passed_counts / total_simulated)*100:.1f}%",
                                  f"{passed_counts:,} saved to tape")

                        # Plot latency distribution
                        fig_lat, ax_lat = plt.subplots(figsize=(6, 3))
                        ax_lat.plot(range(1, num_batches + 1), latencies, marker='o', color='teal')
                        ax_lat.axhline(avg_lat, color='red', linestyle='--', label=f'Mean: {avg_lat:.2f} ms')
                        ax_lat.set_xlabel('Burst Number')
                        ax_lat.set_ylabel('Inference Latency [ms]')
                        ax_lat.set_title('Trigger Burst Latency Stability')
                        ax_lat.grid(True, linestyle='--', alpha=0.4)
                        ax_lat.legend()
                        fig_lat.tight_layout()
                        st.pyplot(fig_lat)
                        plt.close(fig_lat)
            else:
                st.info("Click 'Run Trigger Emulation' to simulate live LHC event filtering and benchmark latency.")

    # === TAB 4: Architecture & Physics Reference ===
    with tab4:
        st.subheader("📑 Physics Benchmark & Architecture Comparison")
        st.markdown("""
        ### Target: Filter Dimuon Events at 5% Background Budget
        In High-Energy Physics triggers, event rate must be reduced by orders of magnitude while preserving 
        clean physics resonances (like the $Z$ boson, Higgs, or Beyond-Standard-Model signatures).

        | Architecture | Paradigm | Signal Eff (@5% BG) | AUROC | Latency | Deployment Format |
        |---|---|---|---|---|---|
        | **Rule-Based Cut** ($p_T^{\\text{lead}} \\geq 29.7$ GeV) | Physics Baseline | ~96.7% | — | < 0.1 $\\mu$s | Hardcoded FPGA / L1 |
        | **Gradient Boosted Trees (XGBoost)** | Supervised ML | **~98.9%** | **0.992** | ~2 $\\mu$s / event | JSON / Treelite / Triton |
        | **Isolation Forest** | Unsupervised Anomaly | ~98.7% | 0.988 | ~10 $\\mu$s / event | Joblib / ONNX |
        | **Graph Neural Network (GCN)** | Relational Graph | Prototype | — | ~50 $\\mu$s / event | PyTorch C++ LibTorch |

        #### Key Invariant Mass Formula
        $$M_{\\mu\\mu} = \\sqrt{2 p_{T1} p_{T2} (\\cosh(\\eta_1 - \\eta_2) - \\cos(\\phi_1 - \\phi_2))}$$
        - Supervised models are trained **strictly without** $M_{\\mu\\mu}$ to prevent circular label leakage.
        """)


if __name__ == '__main__':
    main()
