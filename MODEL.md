# Model Reproduction & Output Binding

Compiled model binaries (`.joblib`, `.pt`, `.onnx`, `models/`) and derived data
(`.root`, the checksum sidecar) are deliberately **excluded from this Git
repository** to keep it small. Every one of them is reproducible from source.
`Dimuon_DoubleMu.csv` *is* tracked, because it is the official CERN educational
sample the README and CI validate against.

## Quick start

From the repository root:

```bash
pip install -r requirements.txt
python src/data_download.py   # fetch + SHA-256 verify the CSV, then write Dimuon_DoubleMu.root
python src/train_model.py     # trains XGBoost -> z_boson_xgb_model.joblib + models/z_boson_xgb_model.json
```

## What each stage produces

| Path | Producer | Contents |
|---|---|---|
| `Dimuon_DoubleMu.csv` | `src/data_download.py` | Official educational sample (tracked in git) |
| `Dimuon_DoubleMu.root` | `src/data_download.py` | uproot-readable `TTree` named `Events` |
| `Dimuon_DoubleMu.root.fingerprint` | `src/data_download.py` | SHA-256 of the CSV the ROOT file was built from |
| `z_boson_xgb_model.joblib` | `src/train_model.py` | Portable scikit-learn pickle, pinned to CPU inference |
| `models/z_boson_xgb_model.json` | `src/train_model.py` | Native XGBoost JSON (C++ / Treelite / Triton / ROOT TMVA) |
| `models/anomaly_detector.joblib` | `src/anomaly_detection.py` | Isolation Forest + its `StandardScaler` |
| `models/gnn_prototype.pt` | `src/train_gnn.py` | GCN weights, scaler, history, metrics, node-column schema |
| `results/metrics.{md,json}` | `src/train_model.py` | Benchmark report (human + machine readable) |
| `results/anomaly_metrics.{md,json}` | `src/anomaly_detection.py` | Unsupervised benchmark report |
| `plots/*.png` | multiple | Committed figures referenced by the README |

## Pipeline notes

1. **Ingestion is integrity-checked.** The download is atomic (a partial transfer
   is never promoted into place), the payload is verified against a pinned
   SHA-256 and exact byte size, and the derived ROOT file records a fingerprint
   so a refreshed CSV automatically invalidates a stale ROOT copy. A payload that
   fails validation is deleted and re-fetched, rather than left in place to fail
   every subsequent run.

2. **Where files are read from and written to.**
   - *Reads* resolve via `src._compat.data_path`: current directory → `CERNDATA_DIR` → repository root. Absolute paths pass through untouched.
   - *Writes* resolve via `src.config.output_path`: `CERNDATA_OUTPUT_DIR` if set, otherwise the current directory. This keeps generated `plots/`, `results/` and `models/` next to wherever you ran the command, and keeps the test suite from polluting the repository.

3. **Training device handling.** `src/train_model.py` attempts
   `device='cuda'` and falls back to CPU (`tree_method='hist'`) **only** when CUDA
   is genuinely unavailable. Any other exception propagates instead of being
   misreported as "no GPU". The device is reset to `cpu` before the booster is
   persisted, so the artifact infers on any host without a fallback warning.

4. **The speed benchmark does not rewrite the device it is timing.** An earlier
   version called `set_params(device='cpu')` on the "GPU" model to silence a
   warning, which silently downgraded the GPU row to a CPU measurement. Each
   variant is now fitted and timed on exactly the device its label claims, and
   `torch.cuda.synchronize()` is issued around each timing sample.

5. **Threshold policy.** Every cut is derived with
   `src.config.threshold_for_budget`, which guarantees background retention
   cannot exceed the budget even under heavy score ties. Thresholds are always
   fit on TRAIN (baseline) or VALIDATION (ML), never on TEST.

## Environment variables

| Variable | Effect |
|---|---|
| `CERNDATA_DIR` | Directory the downloader writes the CSV/ROOT to, and the second search path for readers. |
| `CERNDATA_OUTPUT_DIR` | Directory for generated `plots/`, `results/`, `models/`. Defaults to the cwd. |
| `CERNDATA_VERIFY_CHECKSUM` | `0` disables the pinned SHA-256 check (required for local subsets and the CI fixture). |
