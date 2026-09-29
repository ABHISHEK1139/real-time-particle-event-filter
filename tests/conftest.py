"""Shared pytest fixtures and physics-consistent test data generation.

Two things live here that used to be duplicated (and drift) across
``tests/test_pipeline.py`` and the CI workflow:

* :func:`synthetic_dimuon_frame` — a fixture whose invariant mass is *computed*
  from its kinematics via the same relation as ``PHYSICS.md``. Random ``M`` would
  prove only that the code runs, not that the physics is consistent.
* :func:`repo_root` / ``DATA_PATH`` — paths anchored to the repository root.
  The old tests used bare relative paths, so running ``pytest`` from anywhere
  other than the repo root silently switched the whole suite onto synthetic
  data instead of the real dataset.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import CSV_FILE, FEATURES, MASS_COLUMN, ROOT_FILE  # noqa: E402

DATA_PATH = REPO_ROOT / ROOT_FILE
CSV_PATH = REPO_ROOT / CSV_FILE
MODEL_PATH = REPO_ROOT / "z_boson_xgb_model.joblib"

#: Full 21-column schema of the CERN DoubleMu sample, so the fixture is a drop-in
#: replacement for the real CSV (the GNN needs E/px/py/pz/Q, not just pt/eta/phi).
FULL_COLUMNS = (
    "Run",
    "Event",
    "type1",
    "E1",
    "px1",
    "py1",
    "pz1",
    "pt1",
    "eta1",
    "phi1",
    "Q1",
    "type2",
    "E2",
    "px2",
    "py2",
    "pz2",
    "pt2",
    "eta2",
    "phi2",
    "Q2",
    "M",
)


def repo_root() -> Path:
    """Absolute repository root."""
    return REPO_ROOT


def invariant_mass(pt1, pt2, eta1, eta2, phi1, phi2) -> np.ndarray:
    """Dimuon invariant mass from kinematics (see PHYSICS.md)."""
    rad = np.maximum(2 * pt1 * pt2 * (np.cosh(eta1 - eta2) - np.cos(phi1 - phi2)), 0.0)
    return np.sqrt(rad)


def synthetic_dimuon_frame(n: int = 600, seed: int = 0, full_schema: bool = True) -> pd.DataFrame:
    """Physics-consistent dimuon events with M derived from (pt, eta, phi)."""
    rng = np.random.default_rng(seed)
    pt1 = rng.uniform(5, 100, n)
    pt2 = rng.uniform(5, 100, n)
    eta1 = rng.uniform(-2.4, 2.4, n)
    eta2 = rng.uniform(-2.4, 2.4, n)
    phi1 = rng.uniform(-np.pi, np.pi, n)
    phi2 = rng.uniform(-np.pi, np.pi, n)
    m = invariant_mass(pt1, pt2, eta1, eta2, phi1, phi2)

    data = {"pt1": pt1, "pt2": pt2, "eta1": eta1, "eta2": eta2, "phi1": phi1, "phi2": phi2, MASS_COLUMN: m}
    if not full_schema:
        return pd.DataFrame(data)

    # Derive the four-momenta consistently with pt/eta/phi so the fixture is a
    # faithful stand-in for the real sample (the GNN consumes E and p components).
    p1 = pt1 * np.cosh(eta1)
    p2 = pt2 * np.cosh(eta2)
    return pd.DataFrame(
        {
            "Run": 1,
            "Event": np.arange(n),
            "type1": 13,
            "E1": p1,
            "px1": pt1 * np.cos(phi1),
            "py1": pt1 * np.sin(phi1),
            "pz1": pt1 * np.sinh(eta1),
            "pt1": pt1,
            "eta1": eta1,
            "phi1": phi1,
            "Q1": 1.0,
            "type2": -13,
            "E2": p2,
            "px2": pt2 * np.cos(phi2),
            "py2": pt2 * np.sin(phi2),
            "pz2": pt2 * np.sinh(eta2),
            "pt2": pt2,
            "eta2": eta2,
            "phi2": phi2,
            "Q2": -1.0,
            "M": m,
        }
    )


@pytest.fixture(scope="session")
def synthetic_df() -> pd.DataFrame:
    return synthetic_dimuon_frame(600, seed=0)


@pytest.fixture(scope="session")
def dataset():
    """The real dataset if it is on disk, else a physics-consistent fixture.

    Anchored to the repo root, so the answer no longer depends on the pytest cwd.
    Never skips: the suite must exercise the same code paths either way.
    """
    if DATA_PATH.exists():
        from src.train_model import load_data

        return load_data(str(DATA_PATH), max_rows=1000)
    return synthetic_dimuon_frame(600)


@pytest.fixture(scope="session")
def trained_model(dataset):
    """The persisted classifier if present, else a small one fitted on the fixture."""
    import contextlib

    import joblib

    if MODEL_PATH.exists():
        model = joblib.load(MODEL_PATH)
        if hasattr(model, "set_params"):
            with contextlib.suppress(Exception):
                model.set_params(device="cpu")
        return model

    from xgboost import XGBClassifier

    from src.train_model import build_features_and_labels

    X, y, _ = build_features_and_labels(dataset)
    model = XGBClassifier(n_estimators=10, tree_method="hist", n_jobs=1, random_state=42, device="cpu")
    return model.fit(X, y)


@pytest.fixture
def tiny_model():
    """A cheap classifier fitted on synthetic data, for tests that need isolation."""
    from xgboost import XGBClassifier

    from src.train_model import build_features_and_labels

    X, y, _ = build_features_and_labels(synthetic_dimuon_frame(300, seed=7))
    return XGBClassifier(n_estimators=10, tree_method="hist", n_jobs=1, random_state=42).fit(X, y)


@pytest.fixture
def in_tmp_cwd(tmp_path, monkeypatch):
    """Run the body with cwd pointed at a tmp dir so generated files stay isolated."""
    monkeypatch.chdir(tmp_path)
    return tmp_path


__all__ = [
    "CSV_PATH",
    "DATA_PATH",
    "FEATURES",
    "FULL_COLUMNS",
    "MODEL_PATH",
    "REPO_ROOT",
    "invariant_mass",
    "repo_root",
    "synthetic_dimuon_frame",
]
