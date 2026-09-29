"""Physics-consistent synthetic dimuon event generation.

Shared by the test suite (``tests/conftest.py``) and the standalone fixture
generator (``tests/make_mock_dataset.py``).

This lives in its own module rather than in ``conftest.py`` on purpose: a
conftest imports pytest, so anything that reused it inherited a hard dependency
on the test framework. The image installs only ``requirements.txt``, so
generating a fixture inside the container died with
``ModuleNotFoundError: No module named 'pytest'``.

The invariant mass is *computed* from the kinematics via the relation in
PHYSICS.md, so the label structure matches the real sample. A fixture with a
random ``M`` unrelated to ``pt/eta/phi`` would only prove the code executes, not
that the label boundary is learnable.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

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

    data = {"pt1": pt1, "pt2": pt2, "eta1": eta1, "eta2": eta2, "phi1": phi1, "phi2": phi2, "M": m}
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
