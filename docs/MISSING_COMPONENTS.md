# ParkVoice AI — Missing Components & Functional Gaps

This document enumerates all missing modules, uncompiled binaries, dataset gaps, unexecuted validation notebooks, and missing tests grouped by category.

---

## 1. Missing Binary Model & Metric Artifacts (`backend/models/`)

The directory [backend/models/](file:///d:/Projects/parkinson/backend/models) is currently empty. The following binary artifacts are required for production inference and API benchmarking:

| Missing Artifact File | Expected Content | Required Generator Script | Criticality |
| :--- | :--- | :--- | :--- |
| `tabular_scaler.pkl` | Fitted `RobustScaler` instance for 22 clinical voice features | `python backend/ml/train_classical.py` | **CRITICAL** |
| `rf_model.pkl` | Calibrated Random Forest Classifier | `python backend/ml/train_classical.py` | **CRITICAL** |
| `xgb_model.pkl` | Calibrated XGBoost Classifier | `python backend/ml/train_classical.py` | **CRITICAL** |
| `lgbm_model.pkl` | Calibrated LightGBM Classifier | `python backend/ml/train_classical.py` | **CRITICAL** |
| `ensemble_model.pkl` | Soft-voting `WeightedEnsemble` & Logistic Stacking Meta-Learner | `python backend/ml/train_ensemble.py` | **CRITICAL** |
| `cnn_model.pt` | Fine-tuned EfficientNet-B0 PyTorch weights | `python backend/ml/train_cnn.py` | **HIGH** |
| `severity_model.pkl` | Fitted `SeverityModel` (XGBRegressor + conformal `qhat`) | `python backend/ml/severity_model.py` | **HIGH** |
| `ensemble_quantized.onnx` | Dynamically quantized INT8 ONNX model for edge inference | `python backend/ml/export_onnx.py` | **HIGH** |
| `classical_metrics.json` | Evaluation metrics dict for RF, XGBoost, LightGBM | `python backend/ml/train_classical.py` | **MEDIUM** |
| `ensemble_metrics.json` | Evaluation metrics for ensemble on train/val/test splits | `python backend/ml/train_ensemble.py` | **MEDIUM** |
| `cnn_metrics.json` | Accuracy/AUC epoch metrics for EfficientNet CNN | `python backend/ml/train_cnn.py` | **MEDIUM** |
| `severity_metrics.json` | MAE, RMSE, $R^2$, and tier accuracy metrics for severity model | `python backend/ml/severity_model.py` | **MEDIUM** |
| `onnx_benchmarks.json` | Hardware latency (ms), size (MB), and throughput results | `python backend/evaluation/edge_benchmark.py` | **MEDIUM** |

---

## 2. Dataset & Validation Deficiencies

1. **Missing Real Raw Audio Dataset**:
   - The repository contains no real raw audio dataset files (e.g., mDPBD, Saardbrücken Voice Database, or Italian Parkinson's Voice DB).
   - [dataset_loader.py](file:///d:/Projects/parkinson/backend/pipeline/dataset_loader.py) uses `ucimlrepo` to download the 2008 Little et al. UCI dataset (which consists of 195 pre-extracted tabular rows, not raw audio).
   - If offline or network-blocked, `dataset_loader.py` calls `_generate_synthetic_dataset()`, generating 240 synthetic rows.
2. **Synthetic UPDRS Targets**:
   - The UCI dataset does not contain UPDRS motor severity scores.
   - Function `add_synthetic_severity()` in `dataset_loader.py` artificially generates UPDRS scores ($0\text{--}108$) using an empirical formula based on Jitter, Shimmer, RPDE, and PPE.
3. **Synthetic Spectrogram Images for CNN**:
   - In [train_cnn.py](file:///d:/Projects/parkinson/backend/ml/train_cnn.py#L76), `SyntheticSpectrogramDataset` synthesizes fake 224x224 RGB Mel-spectrogram images from the 22 tabular features because no real audio files exist for the UCI subjects.
4. **Unexecuted Validation Notebook**:
   - [notebooks/00_dataset_validation.ipynb](file:///d:/Projects/parkinson/notebooks/00_dataset_validation.ipynb) is an unexecuted Jupyter notebook.
   - `dataset_report.json` in the project root contains hardcoded zero values: `{"subjects": 0, "recordings": 0, "audio_availability": false}`.

---

## 3. Missing / Defective API Components

1. **Defective `/api/models/status` Endpoint**:
   - `model_status()` in [models_endpoint.py](file:///d:/Projects/parkinson/backend/api/endpoints/models_endpoint.py#L31) lacks a `return` statement.
2. **Floating Unreachable Code**:
   - Lines 52--62 in `models_endpoint.py` are positioned after `return registry` in `get_model_registry()`, causing `scaler_ready` and pipeline status checks to be dead code.
3. **Synchronous Subprocess Trigger in `/api/models/train`**:
   - Endpoint `trigger_training()` launches training scripts via `subprocess.Popen` without storing process handles, tracking status, or logging progress to a database table or task registry.

---

## 4. Test Coverage Gaps

The test suite in [backend/tests/test_screening_modules.py](file:///d:/Projects/parkinson/backend/tests/test_screening_modules.py) contains only 3 basic contract unit tests:
- `test_audio_quality_contract()`
- `test_uncertainty_contract()`
- `test_biomarker_dashboard_required_rows()`

**Missing Unit and Integration Tests**:
- ❌ No API integration tests for `POST /api/analysis/upload`.
- ❌ No API integration tests for Patient CRUD endpoints (`/api/patients`).
- ❌ No tests for audio preprocessor noise reduction or silence trimming.
- ❌ No tests for acoustic feature extractor mathematical accuracy.
- ❌ No tests for ML training routines (`train_classical.py`, `train_ensemble.py`, `severity_model.py`).
- ❌ No tests for ONNX export or quantization validity.
- ❌ No tests for database ORM models or async database session lifecycle.
