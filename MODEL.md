# Model Reproduction & Output Binding

Raw compiled model binaries (`.joblib`, `.pt`, `.onnx`) are deliberately excluded from this Git repository to keep the repo slim. Training data (`.csv`/`.root`) is likewise excluded. You must generate them locally.

## Reproduction Instructions

From the repository root, fetch data then train:

```bash
pip install -r requirements.txt
python src/data_download.py   # fetches CERN Open Data CSV -> Dimuon_DoubleMu.csv (+ .root)
python src/train_model.py     # trains XGBoost -> z_boson_xgb_model.joblib
```

This procedure will:
1. Download the educational CSV, validate its schema/row count, and convert the
   numeric branches to a CERN-native ROOT `TTree` (`Events`).
   `src/train_model.py` and `src/train_gnn.py` read that **ROOT** file for
   training, while `src/realtime_simulation.py` and `src/speed_analysis.py`
   replay the **CSV** — both derive from the same download, and the kinematic
   feature matrix is always (`pt1, pt2, eta1, eta2, phi1, phi2`; invariant mass
   `M` is used only for labels/evaluation, never as a feature).
2. Try XGBoost with `device='cuda'` and automatically fall back to CPU (`tree_method='hist'`) only when CUDA is genuinely unavailable or fails (other errors propagate instead of being misreported as "no GPU").
3. Write the trained classifier to `z_boson_xgb_model.joblib` in the repository root and write benchmark metrics/plots to `results/` and `plots/`.

Once present, downstream scripts (e.g. `python src/realtime_simulation.py`) will discover and load the `.joblib` file. If it is missing they exit with a clear error telling you to run the two commands above first.
