# Model Reproduction & Output Binding

As per formal CERN MLOps best practices, raw compiled model binaries (`.joblib`, `.pt`, `.onnx`) are deliberately excluded from this Git repository to maintain slim payload limits and prevent unbounded version-control bloat.

Instead of cloning a stagnant binary, researchers are expected to execute the native training pipeline utilizing local GPU acceleration to compile the freshest model dynamically.

## Reproduction Instructions

To generate the active predictive model (`z_boson_xgb_model.joblib`), ensure you have fetched the associated CERN Dimuon Open Data CSV and execute the following from the master directory:

```bash
# Execute the native execution pipeline
python train_model.py
```

This procedure will:
1. Parse the local CSV and construct the kinematic feature grids.
2. Delegate memory execution sequentially to the NVIDIA CUDA core architecture.
3. Spool the newly trained XGBoost classifier and emit a localized `z_boson_xgb_model.joblib` natively onto your hardware.

Once populated, downstream simulators (like `realtime_simulation.py`) will automatically discover and load the `.joblib` binary payload.
