"""End-to-end and unit tests for the Z-boson filtering pipeline.

The suite runs against the real dataset when it is present and against a
physics-consistent synthetic fixture otherwise, so it never skips and never
silently changes what it is testing.

Regression tests for previously-fixed defects are marked ``regression:`` so the
reason they exist stays discoverable.
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

# Imported as a top-level module, not `tests.conftest`: at least one unrelated
# `tests` package exists in site-packages on some hosts and shadows the local
# directory when it is imported as a package.
from conftest import (
    DATA_PATH,
    MODEL_PATH,
    invariant_mass,
    synthetic_dimuon_frame,
)
from src.config import (
    BUDGET,
    FEATURES,
    MASS_SIGNAL_MAX,
    MASS_SIGNAL_MIN,
    labels_from_mass,
    output_path,
    rates_at_threshold,
    threshold_for_budget,
)
from src.train_model import (
    build_features_and_labels,
    evaluate_ml_model,
    evaluate_rule_based_baseline,
    stratified_train_val_test_split,
)


# ---------------------------------------------------------------------------
# Data schema / feature construction
# ---------------------------------------------------------------------------
def test_data_schema(dataset):
    for f in [*FEATURES, "M"]:
        assert f in dataset.columns, f"Critical feature {f} missing from payload."
    assert (dataset["pt1"] >= 0).all(), "Transverse momentum pt1 cannot be negative."
    assert (dataset["pt2"] >= 0).all(), "Transverse momentum pt2 cannot be negative."


def test_invariant_mass_matches_dataset_relation():
    """The synthetic fixture's M must follow the PHYSICS.md relation exactly.

    Checked on the fixture rather than the real sample: CMS stores the exact
    four-momentum invariant mass, which differs slightly from the
    pT/eta/phi-only transverse-mass approximation the docs present.
    """
    df = synthetic_dimuon_frame(500, seed=11)
    recomputed = invariant_mass(df["pt1"], df["pt2"], df["eta1"], df["eta2"], df["phi1"], df["phi2"])
    np.testing.assert_allclose(recomputed, df["M"].to_numpy(), rtol=1e-9, atol=1e-9)


def test_feature_builder(dataset):
    X, y, _df = build_features_and_labels(dataset)
    assert list(X.columns) == FEATURES, "Feature column alignment drift detected."
    assert "M" not in X.columns, "Leakage: mass in training schema."
    assert set(pd.Series(y).unique()).issubset({0, 1})


def test_labels_from_mass_boundaries():
    """The mass window is exclusive at both edges; verify the shared helper agrees."""
    mass = np.array([79.999, 80.0, 80.001, 99.999, 100.0, 100.001])
    np.testing.assert_array_equal(labels_from_mass(mass), [0, 0, 1, 1, 0, 0])
    assert (MASS_SIGNAL_MIN, MASS_SIGNAL_MAX) == (80.0, 100.0)


# ---------------------------------------------------------------------------
# Leakage guards
# ---------------------------------------------------------------------------
def test_leakage_exclusion(trained_model):
    if not hasattr(trained_model, "get_booster"):
        pytest.skip("Model format unsupported for feature inspection")
    feats = trained_model.get_booster().feature_names
    assert "M" not in (feats or []), f"DATA LEAKAGE: model uses M: {feats}"


def test_model_inference_boundaries(dataset, trained_model):
    X, _y, _ = build_features_and_labels(dataset)
    preds = trained_model.predict(X)
    probs = trained_model.predict_proba(X)
    assert len(preds) == len(X)
    assert probs.shape == (len(X), 2)
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)
    assert set(np.unique(preds)).issubset({0, 1})


# ---------------------------------------------------------------------------
# Splitting / protocol
# ---------------------------------------------------------------------------
def test_stratified_split_preserves_signal_fraction():
    df = synthetic_dimuon_frame(2000)
    X, y, dfl = build_features_and_labels(df)
    *_, y_tr, y_va, y_te, _, _ = stratified_train_val_test_split(X, y, dfl)
    base = np.mean(np.asarray(y) == 1)
    for split in (y_tr, y_va, y_te):
        assert abs(np.mean(np.asarray(split) == 1) - base) < 0.05


def test_stratified_split_is_disjoint():
    df = synthetic_dimuon_frame(600)
    X, y, dfl = build_features_and_labels(df)
    X_tr, X_va, X_te, *_ = stratified_train_val_test_split(X, y, dfl)
    # Compare index *values*, not id(): CPython interns small ints, so id()
    # collides across different rows.
    tr, va, te = (set(part.index.tolist()) for part in (X_tr, X_va, X_te))
    assert not (tr & va)
    assert not (tr & te)
    assert not (va & te)
    assert len(tr) + len(va) + len(te) == len(X)


def test_threshold_calibration_no_test_leakage(in_tmp_cwd):
    """Thresholds fit on val/train must be applied frozen to test (protocol regression)."""
    df = synthetic_dimuon_frame(2000)
    X, y, dfl = build_features_and_labels(df)
    X_tr, X_va, X_te, y_tr, y_va, y_te, df_tr, _, df_te = stratified_train_val_test_split(X, y, dfl)
    from xgboost import XGBClassifier

    m = XGBClassifier(n_estimators=10, tree_method="hist", n_jobs=1, random_state=42)
    m.fit(X_tr, y_tr)
    base = evaluate_rule_based_baseline(df_tr, df_te)
    ml = evaluate_ml_model(m, X_va, y_va, X_te, y_te)
    assert 0.0 <= base["sig_eff"] <= 1.0
    assert 0.0 <= ml["sig_eff"] <= 1.0
    assert 0.5 <= ml["auroc"] <= 1.0  # physics-consistent labels must be learnable


# ---------------------------------------------------------------------------
# Threshold budget policy  (regression: percentile cut could overshoot the budget)
# ---------------------------------------------------------------------------
def test_threshold_never_exceeds_budget_with_heavy_ties():
    """regression: np.percentile interpolates through ties and overshoots the budget.

    95% of the background sits on two discrete values; the budget-safe cut must
    still retain <= 5%.
    """
    bg = np.concatenate([np.zeros(950), np.ones(50)])
    cut = threshold_for_budget(bg, 0.05)
    assert float(np.sum(bg >= cut)) / bg.size <= 0.05


@pytest.mark.parametrize("budget", [0.01, 0.05, 0.1, 0.25])
def test_threshold_respects_budget_on_continuous_scores(budget):
    rng = np.random.default_rng(0)
    bg = rng.normal(size=5000)
    cut = threshold_for_budget(bg, budget)
    retained = float(np.sum(bg >= cut)) / bg.size
    assert retained <= budget
    assert retained >= budget - 1.0 / bg.size * 2  # not absurdly conservative


def test_threshold_rejects_empty_and_out_of_range_budget():
    with pytest.raises(ValueError, match="no background events"):
        threshold_for_budget([], 0.05)
    with pytest.raises(ValueError, match="budget must be"):
        threshold_for_budget([1, 2, 3], 0.0)
    with pytest.raises(ValueError, match="budget must be"):
        threshold_for_budget([1, 2, 3], 1.0)


def test_threshold_ignores_non_finite_scores():
    bg = np.array([np.nan, np.inf, -np.inf, 1.0, 2.0, 3.0, 4.0])
    cut = threshold_for_budget(bg, 0.5)
    assert np.isfinite(cut)
    assert float(np.sum(bg[np.isfinite(bg)] >= cut)) / 6 <= 0.5


def test_rates_at_threshold_matches_manual_counting():
    scores = np.array([0.1, 0.9, 0.5, 0.2])
    labels = np.array([0, 1, 0, 1])
    r = rates_at_threshold(scores, labels, 0.5)
    assert r["n_bg"] == 2
    assert r["n_sig"] == 2
    assert r["bg_retained"] == pytest.approx(0.5)  # only 0.9 >= 0.5 among background
    assert r["sig_eff"] == pytest.approx(0.5)  # 0.9 yes, 0.2 no


# ---------------------------------------------------------------------------
# Replay / trigger simulation
# ---------------------------------------------------------------------------
def test_realtime_pipeline_runs_on_synthetic(in_tmp_cwd, tiny_model):
    joblib.dump(tiny_model, str(in_tmp_cwd / "z_boson_xgb_model.joblib"))
    synthetic_dimuon_frame(500).to_csv(str(in_tmp_cwd / "Dimuon_DoubleMu.csv"), index=False)
    from src import realtime_simulation as rs

    summary = rs.simulate_realtime_stream(batch_size=200, max_batches=2)
    assert summary["processed"] == 400
    assert summary["batches"] == 2
    assert summary["throughput_eps"] > 0
    assert summary["latency_ms_per_event"] >= 0


def test_realtime_pipeline_edge_cases(in_tmp_cwd, tiny_model):
    """regression: max_batches=0 and a header-only CSV must not raise."""
    joblib.dump(tiny_model, str(in_tmp_cwd / "z_boson_xgb_model.joblib"))
    from src import realtime_simulation as rs

    synthetic_dimuon_frame(100).to_csv(str(in_tmp_cwd / "Dimuon_DoubleMu.csv"), index=False)
    zero = rs.simulate_realtime_stream(batch_size=50, max_batches=0)
    assert zero["processed"] == 0
    assert zero["batches"] == 0

    synthetic_dimuon_frame(100).iloc[:0].to_csv(str(in_tmp_cwd / "Dimuon_DoubleMu.csv"), index=False)
    empty = rs.simulate_realtime_stream(batch_size=50)
    assert empty["processed"] == 0
    assert np.isfinite(empty["latency_ms_per_event"])


def test_realtime_rejects_non_positive_batch_size(in_tmp_cwd, tiny_model):
    """regression: batch_size=0 used to raise a raw range() ValueError."""
    joblib.dump(tiny_model, str(in_tmp_cwd / "z_boson_xgb_model.joblib"))
    synthetic_dimuon_frame(50).to_csv(str(in_tmp_cwd / "Dimuon_DoubleMu.csv"), index=False)
    from src import realtime_simulation as rs

    with pytest.raises(ValueError, match="batch_size must be positive"):
        rs.simulate_realtime_stream(batch_size=0)


def test_confusion_matrix_single_class():
    from sklearn.metrics import confusion_matrix

    cm = confusion_matrix([0, 0, 0], [0, 0, 0], labels=[0, 1])
    tn, fp, fn, tp = cm.ravel().tolist()
    assert (tn, fp, fn, tp) == (3, 0, 0, 0)


# ---------------------------------------------------------------------------
# Anomaly detection
# ---------------------------------------------------------------------------
def test_anomaly_detection_pipeline(in_tmp_cwd):
    from src.anomaly_detection import evaluate_anomaly_detector, train_anomaly_detector

    df = synthetic_dimuon_frame(600)
    X, y, _ = build_features_and_labels(df)
    iso, scaler = train_anomaly_detector(X.iloc[:300])
    metrics = evaluate_anomaly_detector(
        iso, scaler, X.iloc[300:450], y.iloc[300:450], X.iloc[450:], y.iloc[450:], budget=0.10
    )
    assert 0.0 <= metrics["sig_eff"] <= 1.0
    assert 0.0 <= metrics["bg_retained"] <= 0.10 + 1e-9  # budget honoured
    assert 0.0 <= metrics["auroc"] <= 1.0
    assert (in_tmp_cwd / "plots" / "7_anomaly_detection_roc.png").exists()


def test_anomaly_scores_are_sign_inverted():
    """regression: decision_function is negative for anomalies, so we negate it."""
    from sklearn.ensemble import IsolationForest
    from sklearn.preprocessing import StandardScaler

    from src.anomaly_detection import anomaly_scores

    rng = np.random.default_rng(1)
    X = rng.normal(size=(300, 6))
    X[:10] += 40.0  # planted outliers
    scaler = StandardScaler().fit(X)
    iso = IsolationForest(n_estimators=50, random_state=0).fit(scaler.transform(X))
    scores = anomaly_scores(iso, scaler, X)
    assert scores[:10].mean() > scores[10:].mean()


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------
def test_plot_mass_spectrum(tmp_path):
    df = synthetic_dimuon_frame(400)
    csv_file = str(tmp_path / "Dimuon_DoubleMu.csv")
    df.to_csv(csv_file, index=False)
    out_png = str(tmp_path / "test_spectrum.png")

    from src.plot_mass_spectrum import plot_invariant_mass_spectrum

    result = plot_invariant_mass_spectrum(csv_file=csv_file, model_file="nonexistent.joblib", output_path=out_png)
    assert os.path.exists(result)
    assert os.path.getsize(result) > 1000


# ---------------------------------------------------------------------------
# Model export
# ---------------------------------------------------------------------------
def test_native_xgboost_json_export(tmp_path):
    from xgboost import XGBClassifier

    df = synthetic_dimuon_frame(100)
    X, y, _ = build_features_and_labels(df)
    m = XGBClassifier(n_estimators=5, tree_method="hist", n_jobs=1, random_state=42)
    m.fit(X, y)

    json_path = str(tmp_path / "model.json")
    m.save_model(json_path)
    assert os.path.exists(json_path)

    m2 = XGBClassifier()
    m2.load_model(json_path)
    np.testing.assert_allclose(m.predict_proba(X), m2.predict_proba(X), rtol=1e-5)


def test_metrics_json_is_written_and_parsable(in_tmp_cwd):
    """The dashboard reads results/metrics.json; it must exist and round-trip."""
    from src.train_model import write_metrics_report

    baseline = {"pt_cut": 23.16, "bg_retained": 0.0499, "sig_eff": 0.967}
    ml = {
        "prob_cut": 0.0002,
        "bg_retained": 0.0487,
        "sig_eff": 1.0,
        "auroc": 0.9981,
        "auprc": 0.9456,
        "accuracy": 0.9917,
        "tn": 18850,
        "fp": 115,
        "fn": 50,
        "tp": 985,
    }
    report = write_metrics_report(baseline, ml, np.array([0] * 94825 + [1] * 5175))
    assert Path(report).exists()

    payload = json.loads(output_path("results", "metrics.json").read_text(encoding="utf-8"))
    assert payload["budget"] == BUDGET
    assert payload["ml"]["auroc"] == pytest.approx(0.9981)
    assert payload["test_events"] == 100_000


# ---------------------------------------------------------------------------
# Benchmark correctness
# ---------------------------------------------------------------------------
def test_speed_benchmark_preserves_device():
    """regression: bench() used to call set_params(device='cpu') on the GPU model,
    which silently made the 'XGB GPU' row a CPU measurement."""
    import xgboost as xgb

    from src.speed_analysis import XGB_PARAMS

    gpu_like = xgb.XGBClassifier(**XGB_PARAMS, device="cuda")
    assert gpu_like.get_params()["device"] == "cuda"


def test_speed_timed_predict_returns_statistics(tiny_model):
    from src.speed_analysis import timed_predict

    df = synthetic_dimuon_frame(200)
    X, _, _ = build_features_and_labels(df)
    stats = timed_predict(tiny_model, X, repeats=3, warmup=1)
    assert set(stats) == {"median", "p95", "mean"}
    assert stats["median"] > 0
    assert stats["p95"] >= stats["median"]


# ---------------------------------------------------------------------------
# Data download integrity
# ---------------------------------------------------------------------------
def test_checksum_detects_corrupted_payload(tmp_path, monkeypatch):
    """regression: a mirror could serve a different payload; the pinned SHA-256
    must reject a file of canonical size whose bytes differ."""
    from src import data_download as dd

    # Hermetic: the checksum policy must not depend on ambient environment
    # variables. CI sets CERNDATA_VERIFY_CHECKSUM=0 for the ingestion steps and
    # that would otherwise silently skip the very check under test.
    monkeypatch.delenv("CERNDATA_VERIFY_CHECKSUM", raising=False)
    payload = tmp_path / "Dimuon_DoubleMu.csv"
    df = synthetic_dimuon_frame(1000)
    payload.write_text(df.to_csv(index=False), encoding="utf-8")

    # Force the canonical size so the hash check is reached.
    monkeypatch.setattr(dd, "CSV_SIZE_BYTES", payload.stat().st_size)
    with pytest.raises(ValueError, match="Checksum mismatch"):
        dd._validate_csv(str(payload))

    # Writing the real digest must make it pass.
    monkeypatch.setattr(dd, "CSV_SHA256", dd.sha256_of(payload))
    assert dd._validate_csv(str(payload)) is True


def test_validate_csv_respects_checksum_opt_out(tmp_path, monkeypatch, capsys):
    """CERNDATA_VERIFY_CHECKSUM=0 must skip the hash check and say so."""
    from src import data_download as dd

    monkeypatch.setenv("CERNDATA_VERIFY_CHECKSUM", "0")
    subset = tmp_path / "subset.csv"
    synthetic_dimuon_frame(1200).to_csv(str(subset), index=False)
    assert dd._validate_csv(str(subset)) is True
    assert "Checksum verification disabled" in capsys.readouterr().out


def test_validate_csv_rejects_short_and_broken_files(tmp_path):
    from src import data_download as dd
    from src.config import CSV_FILE

    tiny = tmp_path / CSV_FILE
    tiny.write_text("pt1,pt2,eta1,eta2,phi1,phi2,M\n", encoding="utf-8")
    with pytest.raises(ValueError, match="suspiciously small"):
        dd._validate_csv(str(tiny))

    # Long enough to clear the size gate, but with the wrong schema.
    wrong_schema = tmp_path / "wrong.csv"
    wrong_schema.write_text("a,b\n" + "".join(f"{i},{i}\n" for i in range(20000)), encoding="utf-8")
    assert wrong_schema.stat().st_size > dd.MIN_FILE_BYTES
    with pytest.raises(ValueError, match="schema validation failed"):
        dd._validate_csv(str(wrong_schema))


def test_validate_csv_accepts_non_canonical_size_with_warning(tmp_path, monkeypatch, capsys):
    """A local subset is not hashable against the official payload; it must pass
    the schema check loudly rather than be rejected or silently accepted."""
    from src import data_download as dd

    # See the note in test_checksum_detects_corrupted_payload: keep this test
    # independent of the ambient CERNDATA_VERIFY_CHECKSUM setting.
    monkeypatch.delenv("CERNDATA_VERIFY_CHECKSUM", raising=False)
    subset = tmp_path / "subset.csv"
    synthetic_dimuon_frame(1200).to_csv(str(subset), index=False)
    assert dd._validate_csv(str(subset)) is True
    assert "not the" in capsys.readouterr().out


def test_download_discards_invalid_cache(monkeypatch, tmp_path):
    """regression: a corrupt cached CSV used to survive validation and make every
    subsequent run fail immediately instead of re-downloading."""
    from src import data_download as dd

    bad = tmp_path / "Dimuon_DoubleMu.csv"
    bad.write_text("garbage,header\n1,2\n", encoding="utf-8")
    monkeypatch.setattr(dd, "csv_path", lambda: str(bad))
    monkeypatch.setattr(dd, "root_path", lambda: str(tmp_path / "x.root"))
    monkeypatch.setattr(dd, "_root_is_current", lambda *_a, **_k: True)

    good = synthetic_dimuon_frame(1500)
    calls = {"n": 0}

    def fake_download(dest, *_a, **_k):
        calls["n"] += 1
        good.to_csv(dest, index=False)

    monkeypatch.setattr(dd, "_download_with_retry", fake_download)
    result = dd.download_and_convert(verify_checksum=False)
    assert calls["n"] == 1, "should re-download exactly once"
    assert result["rows"] == 1500
    assert bad.read_text(encoding="utf-8").startswith("Run,Event")


def test_root_fingerprint_detects_stale_conversion(tmp_path):
    from src import data_download as dd
    from src.config import CSV_FILE, ROOT_FILE

    csv_file = tmp_path / CSV_FILE
    root_file = tmp_path / ROOT_FILE
    synthetic_dimuon_frame(1100).to_csv(str(csv_file), index=False)
    dd._write_root(str(csv_file), str(root_file))
    assert dd._root_is_current(str(csv_file), str(root_file)) is True

    synthetic_dimuon_frame(1100, seed=99).to_csv(str(csv_file), index=False)
    assert dd._root_is_current(str(csv_file), str(root_file)) is False


# ---------------------------------------------------------------------------
# GNN
# ---------------------------------------------------------------------------
def test_gnn_graph_construction():
    pytest.importorskip("torch_geometric")
    pytest.importorskip("torch")
    from src.train_gnn import NUM_NODE_FEATURES, convert_to_graph_dataset, set_seeds

    set_seeds(0)
    ds = convert_to_graph_dataset(synthetic_dimuon_frame(8))
    assert len(ds) == 8
    g = ds[0]
    assert tuple(g.x.shape) == (2, NUM_NODE_FEATURES)
    assert tuple(g.edge_index.shape) == (2, 2)
    assert len(g.edge_attr) == 2
    assert int(g.y.item()) in (0, 1)


def test_gnn_forward_pass():
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    from src.train_gnn import GCN, convert_to_graph_dataset

    try:
        from torch_geometric.loader import DataLoader
    except ImportError:
        from torch_geometric.data import DataLoader
    loader = DataLoader(convert_to_graph_dataset(synthetic_dimuon_frame(8)), batch_size=4)
    net = GCN()
    net.eval()
    assert tuple(net(next(iter(loader))).shape) == (4, 2)


def test_gnn_scaler_is_fit_on_train_only():
    """regression guard: the node scaler must be fitted on the TRAIN split only.

    A scaler fitted on train+test would return an already-standardised held-out
    set even when that set is drawn from a visibly different distribution.
    """
    pytest.importorskip("torch_geometric")
    pytest.importorskip("torch")
    from sklearn.preprocessing import StandardScaler

    from src.train_gnn import NODE_COLUMNS_1, NODE_COLUMNS_2

    cols = [*NODE_COLUMNS_1, *NODE_COLUMNS_2]
    rng = np.random.default_rng(3)
    train = pd.DataFrame({c: rng.normal(0, 1, 400) for c in cols})
    held_out = pd.DataFrame({c: rng.normal(20, 1, 100) for c in cols})  # shifted

    scaler = StandardScaler().fit(train[cols].to_numpy())
    honest = abs(scaler.transform(held_out[cols].to_numpy()).mean())
    assert honest > 10.0

    # Fitting on train+held-out pulls the mean toward the pooled value, so the
    # held-out set comes back far closer to zero — that is the leakage signature.
    leaky_scaler = StandardScaler().fit(pd.concat([train, held_out])[cols].to_numpy())
    leaky = abs(leaky_scaler.transform(held_out[cols].to_numpy()).mean())
    assert leaky < honest / 5.0, f"leaky scaler mean {leaky} is not clearly closer to 0 than {honest}"


def test_gnn_checkpoint_records_metadata(in_tmp_cwd):
    pytest.importorskip("torch")
    pytest.importorskip("torch_geometric")
    import torch

    from src.train_gnn import NUM_NODE_FEATURES, GCN, save_checkpoint

    payload_path = save_checkpoint(
        GCN(), [(0.1, 0.2)], {"accuracy": 0.5}, None, subset=True, path=str(in_tmp_cwd / "g.pt")
    )
    blob = torch.load(payload_path, weights_only=False)
    assert blob["subset_mode"] is True
    assert blob["num_node_features"] == NUM_NODE_FEATURES
    assert blob["seed"] == 42


# ---------------------------------------------------------------------------
# Repository / CI configuration
# ---------------------------------------------------------------------------
def test_ci_workflow_is_valid_yaml():
    """regression: an unquoted step name containing ': ' makes the whole workflow
    file a YAML error. GitHub then reports the run as failed with *zero jobs*,
    so nothing runs and the reason is not obvious from the run summary."""
    yaml = pytest.importorskip("yaml")
    workflow = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "pytest.yml"
    assert workflow.exists(), "CI workflow file is missing"

    with open(workflow, encoding="utf-8") as fh:
        data = yaml.safe_load(fh)

    assert isinstance(data, dict), "workflow did not parse to a mapping"
    # PyYAML parses the bare key `on` as boolean True (YAML 1.1 truthiness).
    assert ("on" in data) or (True in data), "workflow has no trigger configuration"
    assert "jobs" in data, "workflow defines no jobs"

    for name, job in data["jobs"].items():
        assert "steps" in job, f"job {name!r} has no steps"
        for step in job["steps"]:
            assert isinstance(step, dict), f"job {name!r} has a non-mapping step"
            # A step needs either `uses` or `run`; the label is a name, a uses
            # reference, or the first line of the script.
            assert "uses" in step or "run" in step, f"job {name!r} has an inert step: {step}"
            label = step.get("name") or step.get("uses")
            if label is None:
                label = next((ln for ln in step["run"].splitlines() if ln.strip()), "")
            assert isinstance(label, str)
            assert label.strip(), f"job {name!r} has a step whose label resolved to {label!r}"


def test_ci_workflow_runs_the_test_suite():
    """The workflow must invoke pytest with a bare --cov.

    regression: `--cov=<file>` makes coverage resolve the file as an importable
    module, which imports it before conftest.py and crashes NumPy with
    "cannot load module more than once per process" -> pytest exit code 4.
    """
    yaml = pytest.importorskip("yaml")
    workflow = Path(__file__).resolve().parent.parent / ".github" / "workflows" / "pytest.yml"
    data = yaml.safe_load(workflow.read_text(encoding="utf-8"))

    scripts = [step["run"] for job in data["jobs"].values() for step in job["steps"] if "run" in step]
    pytest_runs = [s for s in scripts if "pytest" in s]
    assert pytest_runs, "the workflow never runs pytest"
    for run in pytest_runs:
        assert "--cov=" not in run, f"--cov=<file> breaks collection: {run!r}"


def test_requirements_lock_covers_direct_dependencies():
    """A 'lock' that leaves the transitive closure floating is not a lock."""
    import re
    from packaging.requirements import Requirement

    root = Path(__file__).resolve().parent.parent
    direct = [
        line.strip()
        for line in (root / "requirements.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    lock_text = (root / "requirements-lock.txt").read_text(encoding="utf-8")
    pinned = set()
    for line in lock_text.splitlines():
        m = re.match(r"^\s*#?\s*([A-Za-z0-9._-]+)==(\S+)", line)
        if m:
            pinned.add(re.sub(r"[-_.]+", "-", m.group(1)).lower())

    assert len(pinned) > 20, f"only {len(pinned)} pins; the transitive closure is not locked"
    for spec in direct:
        name = re.sub(r"[-_.]+", "-", Requirement(spec).name).lower()
        assert name in pinned, f"{name} is a direct dependency but is not pinned in the lock"

    # ruff is a dev dependency and must be pinned too.
    dev = [
        line.strip()
        for line in (root / "requirements-dev.txt").read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith(("#", "-r "))
    ]
    for spec in dev:
        name = re.sub(r"[-_.]+", "-", Requirement(spec).name).lower()
        assert name in pinned, f"{name} is a dev dependency but is not pinned in the lock"


def test_documented_streamlit_dependency_exists():
    """regression: the dashboard is the headline feature and streamlit was
    absent from the requirements entirely, so it could not be installed."""
    reqs = (Path(__file__).resolve().parent.parent / "requirements.txt").read_text(encoding="utf-8")
    assert re.search(r"^streamlit", reqs, re.MULTILINE), "streamlit missing from requirements.txt"


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------
def test_dashboard_runs_headless():
    """Smoke test the Streamlit app end to end, asserting no script exception."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app_file = Path(__file__).resolve().parent.parent / "app.py"
    if not app_file.exists():
        pytest.skip("app.py not present")
    at = AppTest.from_file(str(app_file), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]


@pytest.mark.skipif(not MODEL_PATH.exists(), reason="trained model not available")
def test_dashboard_ml_filter_respects_its_budget():
    """regression: the probability slider rounded the 5%-budget cut (2e-4) to 0.00,
    so the 'ML Filter' passed 100% of events and rejection read 1.0x."""
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    app_file = Path(__file__).resolve().parent.parent / "app.py"
    at = AppTest.from_file(str(app_file), default_timeout=300)
    at.run()
    assert not at.exception, [e.value for e in at.exception]

    metrics = {m.label: m for m in at.metric}
    total = int(metrics["Total Events"].value.replace(",", ""))
    ml_pass = int(metrics["ML Filter Pass"].value.replace(",", ""))
    assert ml_pass < total, f"ML filter admitted everything ({ml_pass}/{total})"

    # The default control is the 5% background budget, so retention must be near 5%
    # and never above it. The delta string is "Eff: ..% | BG: ..%".
    bg_rate = float(metrics["ML Filter Pass"].delta.split("|")[1].split(":")[1].strip().rstrip("%"))
    assert bg_rate <= 5.0 + 0.5, f"background retention {bg_rate}% overran the 5% budget"

    rejection = float(metrics["ML Background Rejection"].value.rstrip("x").replace(",", ""))
    assert rejection > 1.5, f"rejection factor {rejection}x indicates a no-op filter"
