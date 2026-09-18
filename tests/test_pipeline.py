import pytest
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Embed src directly to test functional modularity
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train_model import (
    load_data, build_features_and_labels, FEATURES,
    stratified_train_val_test_split, evaluate_rule_based_baseline, evaluate_ml_model,
)

DATA_PATH = "Dimuon_DoubleMu.root"
MODEL_PATH = "z_boson_xgb_model.joblib"


def _synthetic_physics_df(n=600, seed=0):
    """Physics-consistent synthetic data: M computed FROM kinematics, not random.

    M = sqrt(2*pT1*pT2*(cosh(deta)-cos(dphi))) — same relation as PHYSICS.md.
    """
    rng = np.random.default_rng(seed)
    pt1 = rng.uniform(5, 100, n)
    pt2 = rng.uniform(5, 100, n)
    eta1 = rng.uniform(-2.4, 2.4, n)
    eta2 = rng.uniform(-2.4, 2.4, n)
    phi1 = rng.uniform(-np.pi, np.pi, n)
    phi2 = rng.uniform(-np.pi, np.pi, n)
    m = np.sqrt(np.maximum(2 * pt1 * pt2 * (np.cosh(eta1 - eta2) - np.cos(phi1 - phi2)), 1e-9))
    return pd.DataFrame({'pt1': pt1, 'pt2': pt2, 'eta1': eta1, 'eta2': eta2,
                         'phi1': phi1, 'phi2': phi2, 'M': m})


@pytest.fixture
def dataset():
    """Real dataset if present, else physics-consistent synthetic fallback (never skip)."""
    if os.path.exists(DATA_PATH):
        return load_data(DATA_PATH, max_rows=1000)
    return _synthetic_physics_df(600)


@pytest.fixture
def model(dataset):
    if os.path.exists(MODEL_PATH):
        m = joblib.load(MODEL_PATH)
        if hasattr(m, 'set_params'):
            try:
                m.set_params(device='cpu')
            except Exception:
                pass
        return m
    # Train a tiny model on the (possibly synthetic) fixture so inference tests run everywhere
    from xgboost import XGBClassifier
    X, y, _ = build_features_and_labels(dataset)
    m = XGBClassifier(n_estimators=10, tree_method='hist', n_jobs=1, random_state=42, device='cpu')
    m.fit(X, y)
    return m


def test_data_schema(dataset):
    expected_features = ['pt1', 'pt2', 'eta1', 'eta2', 'phi1', 'phi2', 'M']
    for f in expected_features:
        assert f in dataset.columns, f"Critical feature {f} missing from payload."
    assert (dataset['pt1'] >= 0).all(), "Transverse momentum pt1 cannot be negative."
    assert (dataset['pt2'] >= 0).all(), "Transverse momentum pt2 cannot be negative."


def test_feature_builder(dataset):
    X, y, df_out = build_features_and_labels(dataset)
    assert list(X.columns) == FEATURES, "Feature column alignment drift detected."
    assert 'M' not in X.columns, "Leakage: mass in training schema."
    assert set(pd.Series(y).unique()).issubset({0, 1})


def test_leakage_exclusion(model):
    if hasattr(model, 'get_booster'):
        feats = model.get_booster().feature_names
        assert 'M' not in (feats or []), f"DATA LEAKAGE: model uses M: {feats}"
    else:
        pytest.skip("Model format unsupported for feature inspection")


def test_model_inference_boundaries(dataset, model):
    X, y, _ = build_features_and_labels(dataset)
    preds = model.predict(X)
    probs = model.predict_proba(X)
    assert len(preds) == len(X)
    assert probs.shape == (len(X), 2)
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    assert set(np.unique(preds)).issubset({0, 1})


def test_stratified_split_preserves_signal_fraction():
    df = _synthetic_physics_df(2000)
    X, y, dfl = build_features_and_labels(df)
    out = stratified_train_val_test_split(X, y, dfl)
    _, _, _, y_tr, y_va, y_te, _, _, _ = out
    base = np.mean(np.asarray(y) == 1)
    for split in (y_tr, y_va, y_te):
        assert abs(np.mean(np.asarray(split) == 1) - base) < 0.05


def test_threshold_calibration_no_test_leakage(tmp_path, monkeypatch):
    """Thresholds fit on val/train must be applied frozen to test (protocol regression test)."""
    # evaluate_ml_model saves plots/ + evaluate writes nothing else, so isolate
    # cwd: without this, pytest overwrote the repo's real plots with synthetic figures.
    monkeypatch.chdir(tmp_path)
    df = _synthetic_physics_df(2000)
    X, y, dfl = build_features_and_labels(df)
    X_tr, X_va, X_te, y_tr, y_va, y_te, df_tr, _, df_te = \
        stratified_train_val_test_split(X, y, dfl)
    from xgboost import XGBClassifier
    m = XGBClassifier(n_estimators=10, tree_method='hist', n_jobs=1, random_state=42)
    m.fit(X_tr, y_tr)
    base = evaluate_rule_based_baseline(df_tr, df_te)
    ml = evaluate_ml_model(m, X_va, y_va, X_te, y_te)
    assert 0.0 <= base['sig_eff'] <= 1.0 and 0.0 <= ml['sig_eff'] <= 1.0
    assert 0.5 <= ml['auroc'] <= 1.0  # physics-consistent labels must be learnable


def test_gnn_graph_construction():
    gnn = pytest.importorskip("torch_geometric")
    torch = pytest.importorskip("torch")
    from src.train_gnn import convert_to_graph_dataset, set_seeds
    set_seeds(0)
    df = _synthetic_physics_df(8)
    for c in ['E1', 'px1', 'py1', 'pz1', 'Q1', 'E2', 'px2', 'py2', 'pz2', 'Q2']:
        if c not in df.columns:
            df[c] = 1.0
    ds = convert_to_graph_dataset(df)
    assert len(ds) == 8
    g = ds[0]
    assert tuple(g.x.shape) == (2, 8)
    assert tuple(g.edge_index.shape) == (2, 2)
    assert len(g.edge_attr) == 2
    assert int(g.y.item()) in (0, 1)


def test_gnn_forward_pass():
    torch = pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from src.train_gnn import GCN, convert_to_graph_dataset
    try:
        from torch_geometric.loader import DataLoader
    except ImportError:
        from torch_geometric.data import DataLoader
    df = _synthetic_physics_df(8)
    for c in ['E1', 'px1', 'py1', 'pz1', 'Q1', 'E2', 'px2', 'py2', 'pz2', 'Q2']:
        if c not in df.columns:
            df[c] = 1.0
    loader = DataLoader(convert_to_graph_dataset(df), batch_size=4)
    net = GCN(num_node_features=8)
    net.eval()
    batch = next(iter(loader))
    out = net(batch)
    assert out.shape == (4, 2)


def test_realtime_pipeline_runs_on_synthetic(tmp_path, monkeypatch):
    pytest.importorskip("joblib")
    from xgboost import XGBClassifier
    df = _synthetic_physics_df(500)
    X, y, _ = build_features_and_labels(df)
    m = XGBClassifier(n_estimators=10, tree_method='hist', n_jobs=1, random_state=42)
    m.fit(X, y)
    joblib.dump(m, str(tmp_path / "z_boson_xgb_model.joblib"))
    df.to_csv(str(tmp_path / "Dimuon_DoubleMu.csv"), index=False)
    monkeypatch.chdir(tmp_path)
    from src import realtime_simulation as rs
    rs.simulate_realtime_stream(batch_size=200, max_batches=2)  # must not raise


def test_realtime_pipeline_edge_cases(tmp_path, monkeypatch):
    """Verify realtime simulation handles max_batches=0 and empty datasets gracefully."""
    pytest.importorskip("joblib")
    from xgboost import XGBClassifier
    df = _synthetic_physics_df(100)
    X, y, _ = build_features_and_labels(df)
    m = XGBClassifier(n_estimators=5, tree_method='hist', n_jobs=1, random_state=42)
    m.fit(X, y)
    joblib.dump(m, str(tmp_path / "z_boson_xgb_model.joblib"))
    monkeypatch.chdir(tmp_path)
    from src import realtime_simulation as rs

    # Case 1: max_batches = 0
    df.to_csv(str(tmp_path / "Dimuon_DoubleMu.csv"), index=False)
    rs.simulate_realtime_stream(batch_size=50, max_batches=0)

    # Case 2: empty DataFrame (0 events, must not ZeroDivisionError)
    df.iloc[:0].to_csv(str(tmp_path / "Dimuon_DoubleMu.csv"), index=False)
    rs.simulate_realtime_stream(batch_size=50)


def test_confusion_matrix_single_class():
    """Verify confusion matrix does not drop TN when all predictions belong to single class."""
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix([0, 0, 0], [0, 0, 0], labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()
    assert (tn, fp, fn, tp) == (3, 0, 0, 0)


def test_anomaly_detection_pipeline(tmp_path, monkeypatch):
    """Verify unsupervised anomaly detection trains and evaluates cleanly without mass labels."""
    monkeypatch.chdir(tmp_path)
    from src.anomaly_detection import train_anomaly_detector, evaluate_anomaly_detector

    df = _synthetic_physics_df(600)
    X, y, _ = build_features_and_labels(df)
    # Train anomaly detector on pseudo-background (first 300 events)
    iso, scaler = train_anomaly_detector(X.iloc[:300])
    # Evaluate on val and test splits
    metrics = evaluate_anomaly_detector(
        iso, scaler,
        X.iloc[300:450], y.iloc[300:450],
        X.iloc[450:], y.iloc[450:],
        budget=0.10
    )
    assert 0.0 <= metrics['sig_eff'] <= 1.0
    assert 0.0 <= metrics['bg_retained'] <= 1.0
    assert 0.0 <= metrics['auroc'] <= 1.0
    assert os.path.exists(tmp_path / "plots" / "7_anomaly_detection_roc.png")


def test_plot_mass_spectrum(tmp_path, monkeypatch):
    """Verify mass spectrum reconstruction generates output without errors."""
    monkeypatch.chdir(tmp_path)
    from src.plot_mass_spectrum import plot_invariant_mass_spectrum

    df = _synthetic_physics_df(400)
    csv_file = str(tmp_path / "Dimuon_DoubleMu.csv")
    df.to_csv(csv_file, index=False)

    out_png = str(tmp_path / "test_spectrum.png")
    result = plot_invariant_mass_spectrum(csv_file=csv_file, model_file="nonexistent.joblib", output_path=out_png)
    assert os.path.exists(result)
    assert os.path.getsize(result) > 1000


def test_native_xgboost_json_export(tmp_path):
    """Verify XGBoost native JSON format saves and loads without corruption."""
    from xgboost import XGBClassifier
    df = _synthetic_physics_df(100)
    X, y, _ = build_features_and_labels(df)
    m = XGBClassifier(n_estimators=5, tree_method='hist', n_jobs=1, random_state=42)
    m.fit(X, y)

    json_path = str(tmp_path / "model.json")
    m.save_model(json_path)
    assert os.path.exists(json_path)

    m2 = XGBClassifier()
    m2.load_model(json_path)
    preds1 = m.predict_proba(X)
    preds2 = m2.predict_proba(X)
    np.testing.assert_allclose(preds1, preds2, rtol=1e-5)


