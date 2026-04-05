import pytest
import pandas as pd
import numpy as np
import joblib
import os
import sys

# Embed src directly to test functional modularity
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.train_model import load_data, build_features_and_labels, FEATURES

# Hardcoded reference since functions are entangled in CLI main wrappers currently
DATA_PATH = "Dimuon_DoubleMu.root"
MODEL_PATH = "z_boson_xgb_model.joblib"

@pytest.fixture
def dataset():
    """Load a subset of the dataset using the verified ingestion pipeline to validate schemas."""
    if not os.path.exists(DATA_PATH):
        pytest.skip("Dataset not present in CI environment.")
    return load_data(DATA_PATH, max_rows=1000)

@pytest.fixture
def model():
    """Load the compiled binary to ensure compatibility."""
    if not os.path.exists(MODEL_PATH):
        pytest.skip("Model binary not compiled yet. Run train_model.py")
    return joblib.load(MODEL_PATH)

def test_data_schema(dataset):
    """Ensure the physical properties exist correctly before inference."""
    expected_features = ['pt1', 'pt2', 'eta1', 'eta2', 'phi1', 'phi2', 'M']
    for f in expected_features:
        assert f in dataset.columns, f"Critical feature {f} missing from Open Data payload."
    
    # Assert kinematics are mathematically sane
    assert dataset['pt1'].min() >= 0, "Transverse momentum cannot be negative."

def test_feature_builder(dataset):
    """Deeply inspect the isolated feature builder decoupling constraints."""
    X, y, df_out = build_features_and_labels(dataset)
    
    # Assert Exact order constraint matching
    assert list(X.columns) == FEATURES, "Architecture failed! Feature column alignment drift detected."
    assert 'M' not in X.columns, "Leakage Alert! Mass explicitly pushed into training schema."
    
    # Assert labels valid
    assert set(y.unique()).issubset({0, 1}), "Target labels incorrectly formulated."

def test_leakage_exclusion(model):
    """Deeply interrogate the compiled XGBoost structural features to guarantee invariant mass wasn't dynamically learned."""
    if hasattr(model, 'get_booster'):
        extracted_features = model.get_booster().feature_names
        assert 'M' not in extracted_features, f"FATAL DATA LEAKAGE: Model compiled explicitly with Invariant Mass node mapping: {extracted_features}"
        assert extracted_features == FEATURES, f"Feature misalignment! Expected {FEATURES}, received {extracted_features}"
    else:
        pytest.skip("Model format unsupported for deep AST inspection")

def test_model_inference_boundaries(dataset, model):
    """Validate that the ML outputs strictly bound probability distributions."""
    X, y, _ = build_features_and_labels(dataset)
    
    preds = model.predict(X)
    probs = model.predict_proba(X)
    
    # Assert output dimensionality matches
    assert len(preds) == len(X)
    assert probs.shape == (len(X), 2)
    
    # Assert probabilities are valid bounds [0, 1]
    assert np.all(probs >= 0.0) and np.all(probs <= 1.0)
    
    # Assert predicted classes are binary {0, 1}
    assert set(np.unique(preds)).issubset({0, 1})
