"""Shared pytest fixtures.

Two things live here that used to be duplicated (and drift) across
``tests/test_pipeline.py`` and the CI workflow:

* Repo-anchored paths. The old tests used bare relative paths, so running
  ``pytest`` from anywhere other than the repo root silently switched the whole
  suite onto synthetic data instead of the real dataset.
* The real-dataset-or-fixture ``dataset`` fixture.

The physics-consistent event generator lives in :mod:`src.synthetic` rather than
here, so ``tests/make_mock_dataset.py`` can use it without dragging pytest into
the runtime image.
"""

from __future__ import annotations

import contextlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.config import CSV_FILE, ROOT_FILE  # noqa: E402
from src.synthetic import (  # noqa: E402  (re-exported for test call sites)
    FULL_COLUMNS,
    invariant_mass,
    synthetic_dimuon_frame,
)

DATA_PATH = REPO_ROOT / ROOT_FILE
CSV_PATH = REPO_ROOT / CSV_FILE
MODEL_PATH = REPO_ROOT / "z_boson_xgb_model.joblib"


def repo_root() -> Path:
    """Absolute repository root."""
    return REPO_ROOT


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
    "FULL_COLUMNS",
    "MODEL_PATH",
    "REPO_ROOT",
    "invariant_mass",
    "repo_root",
    "synthetic_dimuon_frame",
]
