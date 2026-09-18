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

## What I Learned

One of the biggest lessons from this project was that a strong accuracy number does not always mean the problem is truly difficult. In this case, the labels are closely tied to invariant mass, and the input features already contain much of the information needed to reconstruct that relationship.

Because of that, I do not treat the result as evidence of a realistic trigger-level solution. Instead, I treat it as a useful step in understanding the physics structure of the data, the limits of simplified ML pipelines, and the difference between a clean educational setup and real detector environments. A dummy all-background classifier already scores ~95% accuracy here, so the headline metrics are AUROC/AUPRC and signal efficiency at fixed background acceptance — see `results/metrics.md`.

---

## Next Steps

If I continue this project, I want to work closer to the actual structure of HEP workflows by integrating Monte Carlo simulated backgrounds, improving the graph topology representation, and exploring anomaly detection tools that do not rely on such clean label definitions.

For me, the main value of this project is not that it is finished, but that it helped me move from reading about these systems to actually experimenting with them and understanding where my knowledge is still incomplete.

---

## Running the Project

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
pip install -r requirements.txt
python src/data_download.py   # fetches data from CERN Open Data Portal (record 5201)
python src/train_model.py     # stratified train/val/test, writes results/metrics.md + plots/
python src/speed_analysis.py  # XGB-CPU vs XGB-GPU (same params) + RF reference -> plots/speed_comparison.png
python -m pytest tests/test_pipeline.py -v
```

---

## Data Source

[CERN Open Data Portal — Educational dimuon sample, DoubleMu Run2011A 7 TeV (record 5201)](https://opendata.cern.ch/record/5201) — parent derived-dataset record [545](http://opendata.cern.ch/record/545). (The Run2010B dimuon sample is a separate record [700](https://opendata.cern.ch/record/700); this project does not use it.) Dataset released CC0; code in this repo is MIT (see LICENSE).

---

*This project is part of my ongoing effort to bridge computer science and high energy physics. There is a lot I still do not know about how real detector systems work, but this repository reflects a genuine attempt to learn it from the data up.*
