# CERN Z-boson ML filter — task runner
#
# Every target mirrors a documented entry point, so `make help` is the single
# place to discover what this project can do. Each target is safe to re-run.

PYTHON ?= python
IMAGE  ?= cern-zboson-ml
DATA_VOLUME ?= cern-data

.DEFAULT_GOAL := help
.PHONY: help install install-dev data train anomaly gnn spectrum speed replay \
        baseline all test lint format coverage check mock serve stress \
        docker-build docker-run docker-serve docker-clean clean

help: ## Show this help
	@grep -hE '^[a-zA-Z_-]+:.*?## ' $(MAKEFILE_LIST) \
	  | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-16s\033[0m %s\n", $$1, $$2}'

# --- Setup -----------------------------------------------------------------
install: ## Install runtime dependencies
	$(PYTHON) -m pip install -r requirements.txt

install-dev: ## Install runtime + test + lint dependencies
	$(PYTHON) -m pip install -r requirements-dev.txt

# --- Pipeline --------------------------------------------------------------
data: ## Fetch the CERN dataset (SHA-256 pinned) and build the ROOT copy
	$(PYTHON) src/data_download.py

train: ## Train the supervised XGBoost classifier
	$(PYTHON) src/train_model.py

anomaly: ## Train the unsupervised Isolation Forest anomaly detector
	$(PYTHON) src/anomaly_detection.py

gnn: ## Train the graph neural network (GNN_EPOCHS=10, add FULL=1 for all 100k)
	$(PYTHON) src/train_gnn.py --epochs $(or $(GNN_EPOCHS),10) $(if $(FULL),--full,)

spectrum: ## Render the invariant mass spectrum figure
	$(PYTHON) src/plot_mass_spectrum.py

speed: ## Run the CPU/GPU inference latency benchmark
	$(PYTHON) src/speed_analysis.py

replay: ## Replay the dataset in batches (BATCH=50000 BATCHES= to limit)
	$(PYTHON) src/realtime_simulation.py $(if $(BATCH),--batch-size $(BATCH),)

baseline: ## Print the deterministic mass-cut accuracy ceiling
	$(PYTHON) src/baseline_physics_cut.py

stress: ## GPU saturation stress test (requires CUDA; ITER=1 for a single loop)
	$(PYTHON) src/stress_test_gpu.py --iterations $(or $(ITER),2) --repeat $(or $(REPEAT),10)

all: data train anomaly gnn spectrum speed baseline ## Run the full pipeline

serve: ## Launch the Streamlit dashboard
	streamlit run app.py

mock: ## Generate a physics-consistent stand-in dataset (ROWS=1000)
	$(PYTHON) tests/make_mock_dataset.py --rows $(or $(ROWS),1000) --out Dimuon_DoubleMu.csv

# --- Quality ---------------------------------------------------------------
test: ## Run the pytest suite
	$(PYTHON) -m pytest tests/ -v

coverage: ## Run the suite with a coverage report
	$(PYTHON) -m pytest tests/ --cov --cov-report=term-missing --cov-report=html

lint: ## Check lint and formatting
	$(PYTHON) -m ruff check .
	$(PYTHON) -m ruff format --check .

format: ## Apply lint fixes and reformat in place
	$(PYTHON) -m ruff check --fix .
	$(PYTHON) -m ruff format .

check: lint test ## Lint + test

# --- Docker ----------------------------------------------------------------
docker-build: ## Build the production image
	docker build -t $(IMAGE) .

docker-run: ## Run the containerised training pipeline (persists the dataset)
	docker run --rm -v $(DATA_VOLUME):/data $(IMAGE)

docker-serve: ## Run the containerised dashboard on :8501
	docker run --rm -p 8501:8501 -v $(DATA_VOLUME):/data $(IMAGE) \
	  streamlit run app.py --server.address=0.0.0.0 --server.port=8501

docker-clean: ## Remove the built image
	docker image rm $(IMAGE) || true

# --- Housekeeping ----------------------------------------------------------
clean: ## Remove generated artifacts (keeps the tracked dataset and figures)
	rm -rf .pytest_cache .ruff_cache .coverage coverage.xml htmlcov pytest-report.xml
	rm -f z_boson_xgb_model.joblib
	rm -rf models
