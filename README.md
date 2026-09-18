# Machine Learning-Based Detection of Z Boson Signals using CERN Open Data

![Python](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch_Geometric-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-14354C?style=for-the-badge&logo=xgboost&logoColor=white)
![pytest](https://img.shields.io/badge/Pytest_CI%2FCD-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)

> **Scope:** supervised demonstration of learning a Z-mass-window selection
> (80 < M < 100 GeV) from muon kinematics on educational CERN Open Data —
> plus an offline batch-replay inference prototype. Not a live trigger,
> not a discovery benchmark.

## Why I Built This

I built this project to better understand how machine learning can be applied to high-energy physics data. While reading about CERN and event filtering, I realized I understood the idea at a high level but not the actual workflow, so I decided to learn by building a small prototype on open data.

This repository is not meant to present a production-level detector system. It is a learning project where I tried to connect machine learning concepts with particle physics ideas such as kinematic features, invariant mass, and signal selection.

---

## What This Project Does

Uses the CERN Open Data educational dimuon sample (**CMS DoubleMu Run2011A,
7 TeV pp, 100k events — record [5201](https://opendata.cern.ch/record/5201),
parent record [545](https://opendata.cern.ch/record/545)**) and trains models
to reproduce the Z-mass-window label from kinematics (`pt`, `eta`, `phi`).

Approaches: stratified XGBoost classifier (train/val/test, validation-calibrated
thresholds, AUROC/AUPRC reported) and a small two-node dimuon GNN prototype
(train/val/test, seeded, checkpointed). Throughput numbers come from offline
batch replay (`realtime_simulation.py`), not a live event stream.

---

## Results (frozen TEST, 20% = 20,000 events, 5.17% signal)

Protocol: stratified 60/20/20 train/val/test. Thresholds are fit on TRAIN
(baseline) or VALIDATION (ML) and measured once on frozen TEST — never fit on
TEST. Full report: `results/metrics.md`.

| Model | Signal eff. @ ≈5% bg | AUROC | AUPRC | Acc@0.5 |
|---|---|---|---|---|
| Naive pT1 cut (≥18.94 GeV) | 95.46% | — | — | — |
| XGBoost (100k events) | **100.00%** | **0.9981** | **0.9456** | 99.17% |
| GNN, full data (100k, 50 ep, best-val) | — | 0.9946 | 0.8610 | 98.30% |
| GNN, subset (10k, 10 ep) | — | 0.9942 | 0.7877 | 98.05% |

Note: a 10-epoch full-data GNN run scored 98.27% / 0.9945 / 0.8523 — the extra
40 epochs bought only +0.009 AUPRC (train loss kept falling while val loss
bounced), confirming the network plateaus early on this geometric task.

Speed (median of 20 warmed-up runs, identical 5k-event slice — `plots/speed_comparison.png`):

| Model | Train | Infer (median) | Acc |
|---|---|---|---|
| RF-100 CPU (reference) | ~2.4 s | ~44 ms | 98.94% |
| XGB CPU | ~0.55 s | ~3.1 ms | 99.08% |
| XGB GPU | ~0.49 s | ~2.9 ms | 99.08% |

---

## What I Learned

One of the biggest lessons from this project was that a strong accuracy number does not always mean the problem is truly difficult. In this case, the labels are closely tied to invariant mass, and the input features already contain much of the information needed to reconstruct that relationship.

Because of that, I do not treat the result as evidence of a realistic trigger-level solution. Instead, I treat it as a useful step in understanding the physics structure of the data, the limits of simplified ML pipelines, and the difference between a clean educational setup and real detector environments. A dummy all-background classifier already scores ~95% accuracy here, so the headline metrics are AUROC/AUPRC and signal efficiency at fixed background acceptance — see `results/metrics.md`.

---

## Project Structure

```text
src/
  _compat.py              # UTF-8 stdout guard + repo-root input lookup (shared)
  data_download.py        # fetch record 5201 CSV (retry/timeout), validate, -> .root
  train_model.py          # stratified 60/20/20 XGBoost, VAL-calibrated thresholds, JSON export
  train_gnn.py            # 2-node dimuon GCN (StandardScaler normalized, train/val/test)
  plot_mass_spectrum.py   # Invariant mass spectrum & Breit-Wigner Z-peak isolation
  anomaly_detection.py    # Unsupervised Isolation Forest trigger on kinematics without mass labels
  realtime_simulation.py  # offline batch replay with TP/FP/FN/TN + split timings
  speed_analysis.py       # XGB-CPU vs XGB-GPU (same params) + RF reference
  stress_test_gpu.py      # synthetic duplicated-batch GPU saturation test
  baseline_physics_cut.py # deterministic M-cut ceiling (100% by construction)
app.py                    # Interactive Streamlit dashboard for real-time trigger & spectrum filtering
tests/test_pipeline.py    # 14 tests: schema, leakage, stratification, GNN, replay, anomaly, spectrum
results/                  # metrics.md, anomaly_metrics.md
plots/                    # 1_invariant_mass_spectrum, 5_confusion_matrix, 6_roc_curve, 6b_significance, 7_anomaly_detection_roc, speed
models/                   # z_boson_xgb_model.json, anomaly_detector.joblib, gnn_prototype.pt
logs/                     # UTF-8 snapshots of baseline + benchmark runs
```

---

## Running the Project

### Interactive Dashboard (Streamlit)
Launch the interactive web dashboard to explore collisions, tune trigger thresholds in real time, and inspect the Breit-Wigner mass peak:
```bash
streamlit run app.py
```

### Docker
```bash
docker build -t cern-zboson-ml .
# Fresh clone: fetch data + train inside the container (data/model are NOT shipped):
docker run --rm -v ${PWD}/plots:/app/plots -v ${PWD}/results:/app/results \
  cern-zboson-ml sh -c "python src/data_download.py && python src/train_model.py"
# Offline batch replay (needs the trained model + CSV present):
docker run --rm cern-zboson-ml python src/realtime_simulation.py
```

### Local Python
```bash
pip install -r requirements.txt          # or requirements-lock.txt for pinned repro
python src/data_download.py        # fetches data from CERN Open Data Portal (record 5201)
python src/train_model.py          # stratified train/val/test, writes results/metrics.md + plots/
python src/plot_mass_spectrum.py   # reconstructs dimuon mass spectrum & Breit-Wigner peak
python src/anomaly_detection.py    # unsupervised trigger anomaly detection without mass labels
python src/train_gnn.py            # 10k-subset GNN prototype (add --full for all 100k)
python src/speed_analysis.py       # XGB-CPU vs XGB-GPU (same params) + RF reference -> plots/speed_comparison.png
python src/realtime_simulation.py  # offline batch replay with TP/FP/FN/TN
streamlit run app.py               # launches live interactive trigger & mass peak dashboard
python -m pytest tests/test_pipeline.py -v
```

Notes:
- Run from the repository root; scripts also fall back to the repo root when
  looking up `Dimuon_DoubleMu.*` / `z_boson_xgb_model.joblib`.
- Console output is forced to UTF-8 by `src/_compat.py`, so no `PYTHONUTF8=1`
  workaround is needed on Windows anymore.
- `train_gnn.py` defaults to a seeded random 10k subset for speed; its numbers
  are not comparable to full-data XGBoost — use `--full` for the 100k run.

---

## Limitations & How I'd Improve It

Honest gaps (see `PHYSICS.md` / `MODEL.md` for the full story):
- Labels are a deterministic function of M, which the features already encode —
  the models re-learn a known geometric cut; AUROC ≈ 0.998 reflects task ease,
  not trigger readiness.
- GNN is a 2-node dimuon graph, not a general event topology; no pile-up,
  no detector simulation, no live stream (replay is static-CSV batches).
- Stress test duplicates rows (synthetic saturation, not 10M unique collisions).

If I continue: central YAML config for all hyperparameters, Monte Carlo
backgrounds, richer graph topologies (multi-object events), anomaly-detection
baselines that don't rely on the clean mass-window label, and proper
per-event latency histograms (p50/p95/p99) instead of batch means.

---

## Data Source

[CERN Open Data Portal — Educational dimuon sample, DoubleMu Run2011A 7 TeV (record 5201)](https://opendata.cern.ch/record/5201) — parent derived-dataset record [545](http://opendata.cern.ch/record/545). (The Run2010B dimuon sample is a separate record [700](https://opendata.cern.ch/record/700); this project does not use it.) Dataset released CC0; code in this repo is MIT (see LICENSE).

---

*This project is part of my ongoing effort to bridge computer science and high energy physics. There is a lot I still do not know about how real detector systems work, but this repository reflects a genuine attempt to learn it from the data up.*
