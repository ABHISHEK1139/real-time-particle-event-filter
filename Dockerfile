# syntax=docker/dockerfile:1
#
# Two stages: the builder resolves dependencies into a self-contained virtualenv,
# the runtime stage copies only that venv and the source. That keeps compilers,
# pip caches and the build toolchain out of the shipped image and makes the
# runtime reproducible without pinning transitive deps in this file.
#
# Only requirements.txt is installed, so the shipped image carries no pytest/ruff.
# The image also ships no dataset and no weights: both are produced at run time.
#
#   docker build -t cern-zboson-ml .
#   docker run --rm -v cern-data:/data cern-zboson-ml   # download + train + anomaly
#   docker run --rm cern-zboson-ml python src/realtime_simulation.py
#   docker run --rm -p 8501:8501 cern-zboson-ml streamlit run app.py \
#       --server.address=0.0.0.0 --server.port=8501

# ---------------------------------------------------------------------------
# Stage 1: builder
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS builder

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    VIRTUAL_ENV=/opt/venv

RUN python -m venv "$VIRTUAL_ENV"
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

WORKDIR /build
COPY requirements.txt ./

# Install into the venv only. The runtime stage reuses this venv, so nothing
# installed here leaks into the shipped image's system site-packages.
RUN pip install --upgrade pip setuptools wheel \
 && pip install -r requirements.txt

# ---------------------------------------------------------------------------
# Stage 2: runtime
# ---------------------------------------------------------------------------
FROM python:3.12-slim AS runtime

LABEL org.opencontainers.image.title="CERN Z-boson ML filter" \
      org.opencontainers.image.description="ML filtering of Z -> mu+mu- events from CMS Open Data (Run2011A DoubleMu, 7 TeV)" \
      org.opencontainers.image.source="https://github.com/ABHISHEK1139/real-time-particle-event-filter" \
      org.opencontainers.image.licenses="MIT"

# Windows hosts default to cp1252, which crashed emoji prints; force UTF-8
# everywhere so container logs match local runs. MPLBACKEND=Agg keeps matplotlib
# headless (no DISPLAY in a container, and no GUI toolkit installed).
ENV PYTHONUTF8=1 \
    PYTHONIOENCODING=utf-8 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONFAULTHANDLER=1 \
    MPLBACKEND=Agg \
    CERNDATA_DIR=/data \
    CERNDATA_OUTPUT_DIR=/app \
    VIRTUAL_ENV=/opt/venv

ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Runtime system deps only: uproot/awkward need none, but libgomp is required by
# scikit-learn/XGBoost and its absence is a classic late-startup ImportError.
RUN apt-get update \
 && apt-get install -y --no-install-recommends libgomp1 \
 && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv

WORKDIR /app

# Run as a non-root user. The dataset and every generated artifact are written
# under /data and /app, both of which this user owns.
RUN useradd --create-home --uid 10001 appuser \
 && mkdir -p /data /app/plots /app/results /app/models \
 && chown -R appuser:appuser /data /app

COPY --chown=appuser:appuser . /app/

USER appuser

# Declared so `docker run --rm -v ...:/data` persists the downloaded dataset and
# the ROOT conversion between runs instead of re-fetching 14 MB every time.
VOLUME ["/data"]

# Fail fast if the installed package cannot even be imported, instead of a
# container that looks healthy while every entry point raises on start.
# (The dashboard has its own probe: GET /_stcore/health on port 8501.)
HEALTHCHECK --interval=30s --timeout=10s --start-period=20s --retries=3 \
    CMD ["python", "-c", "import src.train_model, src.config; print('ok')"]

# Exit 143 propagates from a SIGTERM so orchestrators see a clean shutdown.
STOPSIGNAL SIGTERM

# Default: fetch the dataset (if absent) then train and benchmark end to end.
# Each stage is chained with && so a failure surfaces as a non-zero exit rather
# than a container that reports success having trained nothing.
#
# Long-running dashboard mode:
#   docker run --rm -p 8501:8501 cern-zboson-ml \
#     streamlit run app.py --server.address=0.0.0.0 --server.port=8501
CMD ["sh", "-c", "python src/data_download.py && python src/train_model.py && python src/anomaly_detection.py && python src/baseline_physics_cut.py"]

