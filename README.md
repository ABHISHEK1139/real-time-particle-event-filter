# Machine Learning-Based Detection of Z Boson Signals using CERN Open Data

![Python](https://img.shields.io/badge/Python-14354C?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch_Geometric-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-14354C?style=for-the-badge&logo=xgboost&logoColor=white)
![pytest](https://img.shields.io/badge/Pytest_CI%2FCD-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)

## Why I Built This

I built this project to better understand how machine learning can be applied to high-energy physics data. While reading about CERN and event filtering, I realized I understood the idea at a high level but not the actual workflow, so I decided to learn by building a small prototype on open data.

This repository is not meant to present a production-level detector system. It is a learning project where I tried to connect machine learning concepts with particle physics ideas such as kinematic features, invariant mass, and signal selection.

---

## What This Project Does

This project uses CERN Open Data dimuon events and trains machine learning models to distinguish Z boson candidate events from background events using kinematic features such as `pt`, `eta`, and `phi`.

I experimented with different approaches, including a tree-based baseline and an early graph-based prototype, mainly to understand how model choice, feature design, and throughput relate to event filtering problems in high-energy physics.

---

## What I Learned

One of the biggest lessons from this project was that a strong accuracy number does not always mean the problem is truly difficult. In this case, the labels are closely tied to invariant mass, and the input features already contain much of the information needed to reconstruct that relationship.

Because of that, I do not treat the result as evidence of a realistic trigger-level solution. Instead, I treat it as a useful step in understanding the physics structure of the data, the limits of simplified ML pipelines, and the difference between a clean educational setup and real detector environments.

---

## Next Steps

If I continue this project, I want to work closer to the actual structure of HEP workflows by integrating Monte Carlo simulated backgrounds, improving the graph topology representation, and exploring anomaly detection tools that do not rely on such clean label definitions.

For me, the main value of this project is not that it is finished, but that it helped me move from reading about these systems to actually experimenting with them and understanding where my knowledge is still incomplete.

---

## Running the Project

### Docker (Recommended)
```bash
docker build -t cern-zboson-ml .
docker run --rm --gpus all -v ${PWD}/plots:/app/plots cern-zboson-ml python src/train_model.py
```

### Local Python
```bash
pip install -r requirements.txt
python src/data_download.py   # fetches data from CERN Open Data Portal
python src/train_model.py
python src/speed_analysis.py
python -m pytest tests/test_pipeline.py -v
```

---

## Data Source

[CERN Open Data Portal — CMS Dimuon Run2010B](http://opendata.cern.ch/record/545)

---

*This project is part of my ongoing effort to bridge computer science and high energy physics. There is a lot I still do not know about how real detector systems work, but this repository reflects a genuine attempt to learn it from the data up.*
