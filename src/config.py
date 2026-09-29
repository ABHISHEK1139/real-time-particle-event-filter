"""Single source of truth for experiment constants, artifact paths and thresholding.

Every script, the tests and the Streamlit dashboard previously hard-coded their own
copy of the feature list, mass window and background budget. Those copies drifted
(the dashboard ended up displaying numbers that contradicted ``results/metrics.md``),
so they now all import from here.

Thresholding policy
--------------------
A "background budget" is a *rate ceiling*, so the cut must never admit more
background than allowed. ``np.percentile`` interpolates between tied scores and
can therefore land a threshold that overshoots the budget whenever the score
distribution is discrete (tree models emit large numbers of exactly-0.0 / 1.0
probabilities). ``threshold_for_budget`` instead returns the smallest score such
that the retained fraction is guaranteed ``<= budget``, and the realised rate is
always reported alongside it.
"""

from __future__ import annotations

import os
from pathlib import Path

import numpy as np

# --- Experiment constants -------------------------------------------------
FEATURES: list[str] = ["pt1", "pt2", "eta1", "eta2", "phi1", "phi2"]
MASS_COLUMN = "M"
REQUIRED_COLUMNS: tuple[str, ...] = (*FEATURES, MASS_COLUMN)

MASS_SIGNAL_MIN = 80.0
MASS_SIGNAL_MAX = 100.0
MASS_WINDOW: tuple[float, float] = (MASS_SIGNAL_MIN, MASS_SIGNAL_MAX)
PDG_Z_MASS = 91.1876

#: Fraction of background events a filter is allowed to retain.
BUDGET = 0.05

RANDOM_STATE = 42
SEED = RANDOM_STATE

CSV_FILE = "Dimuon_DoubleMu.csv"
ROOT_FILE = "Dimuon_DoubleMu.root"
XGB_JOBLIB_FILE = "z_boson_xgb_model.joblib"
ANOMALY_JOBLIB_FILE = "models/anomaly_detector.joblib"
XGB_JSON_FILE = "models/z_boson_xgb_model.json"
GNN_CHECKPOINT_FILE = "models/gnn_prototype.pt"

# --- Repository / output locations ---------------------------------------
ROOT: Path = Path(__file__).resolve().parent.parent


def output_dir() -> Path:
    """Directory for generated plots/results/logs.

    Defaults to the current working directory so the scripts keep writing
    ``plots/`` and ``results/`` next to wherever the user ran them from (and so
    tests that ``chdir`` into a tmp dir never pollute the repo). Override with
    ``CERNDATA_OUTPUT_DIR`` for a fixed, out-of-tree output location.
    """
    override = os.environ.get("CERNDATA_OUTPUT_DIR")
    return Path(override) if override else Path.cwd()


def ensure_dir(path: Path | str) -> Path:
    """``mkdir -p`` that returns the path, for terse call sites."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def output_path(*parts: str) -> Path:
    """Resolve an output artifact under :func:`output_dir`, creating parent dirs."""
    p = output_dir().joinpath(*parts)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# --- Dataset identity ----------------------------------------------------
#: CMS DoubleMu Run2011A, pp collisions at 7 TeV, educational 100k-event sample.
DATASET_NAME = "CMS DoubleMu Run2011A"
DATASET_ENERGY = "7 TeV"
DATASET_RECORD_URL = "https://opendata.cern.ch/record/5201"

#: SHA-256 of the canonical 100k-event educational CSV. Pinned so a swapped or
#: truncated mirror payload is rejected instead of silently training on it.
CSV_SHA256 = "58caa45c580f8bcf4b457281c8525a56de0e0b8eb0e0b29e86a9b655a8e64d35"
CSV_SIZE_BYTES = 13_935_840

MIN_ROWS = 1000
MIN_FILE_BYTES = 100_000


def labels_from_mass(mass: np.ndarray | pd.Series):  # noqa: F821
    """Signal label: 1 inside the Z mass window, 0 outside."""
    m = np.asarray(mass, dtype=float)
    return ((m > MASS_SIGNAL_MIN) & (m < MASS_SIGNAL_MAX)).astype(int)


# --- Thresholding --------------------------------------------------------
def threshold_for_budget(background_scores, budget: float = BUDGET) -> float:
    """Return the smallest cut whose background retention cannot exceed ``budget``.

    With ``n`` background calibration scores and ``allowed = floor(budget * n)``,
    the cut is placed just above the ``allowed``-th largest score. Selection with
    ``score >= cut`` then admits at most ``allowed`` events, so the realised
    background rate is bounded by the budget even when scores are heavily tied.
    Ties at the boundary make the realised rate *lower* than the budget, never
    higher; the realised value is always reported next to the budget.

    Raises
    ------
    ValueError
        If no background scores are supplied or ``budget`` is outside ``(0, 1)``.
    """
    if not 0.0 < budget < 1.0:
        raise ValueError(f"budget must be in (0, 1), got {budget!r}")
    bg = np.asarray(background_scores, dtype=float).ravel()
    bg = bg[np.isfinite(bg)]
    n = bg.size
    if n == 0:
        raise ValueError("Cannot calibrate a threshold: no background events supplied.")

    allowed = int(np.floor(budget * n))
    if allowed >= n:  # budget so loose that everything may pass
        return float(np.nextafter(bg.min(), -np.inf))
    return float(np.nextafter(np.sort(bg)[n - allowed - 1], np.inf))


def rates_at_threshold(scores, labels, threshold: float) -> dict[str, float]:
    """Background retention and signal efficiency for a fixed cut."""
    s = np.asarray(scores, dtype=float).ravel()
    y = np.asarray(labels).ravel().astype(int)
    passed = s >= threshold
    bg = y == 0
    sig = y == 1
    n_bg, n_sig = int(bg.sum()), int(sig.sum())
    return {
        "threshold": float(threshold),
        "bg_retained": float(passed[bg].sum() / n_bg) if n_bg else float("nan"),
        "sig_eff": float(passed[sig].sum() / n_sig) if n_sig else float("nan"),
        "n_bg": n_bg,
        "n_sig": n_sig,
    }
