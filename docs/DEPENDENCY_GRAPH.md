# ParkVoice AI — Dependency Graph & Validation Matrix

This document defines the dependency chain for the current implementation and adds validation metadata for every runtime dependency. The focus is execution readiness, not architectural redesign.

## 1. Layer Separation

### Research Layer
- Training scripts
- Dataset validation
- Evaluation and benchmarking scripts
- Experimental research code

### Inference Layer
- Preprocessing
- Feature extraction
- Model loading
- Severity estimation
- SHAP + risk breakdown
- API response generation

### Product Layer
- Frontend rendering
- Database persistence
- Authentication
- Deployment packaging
- Monitoring and release governance

---

## 2. End-to-End Runtime Dependency Tree

```text
User Upload
  ↓
Frontend
  ↓
POST /api/analysis/upload
  ↓
backend/api/endpoints/analysis.py
  ↓
backend/uploads/
  ↓
backend/core/database.py
  ↓
backend/pipeline/pipeline.py
  ├─ preprocessing → backend/pipeline/audio_preprocessor.py
  ├─ feature extraction → backend/pipeline/feature_extractor.py
  ├─ normalization → backend/pipeline/normalization.py
  ├─ classification → backend/ml/train_classical.py + backend/ml/train_ensemble.py
  ├─ severity → backend/ml/severity_model.py
  ├─ explainability → backend/explainability/shap_explainer.py
  └─ persistence → backend/core/database.py
  ↓
API Response
```

---

## 3. Dependency Validation Matrix

| Dependency | Input | Output | Artifact Produced | Artifact Consumed | Validation Method | Failure Condition |
|---|---|---|---|---|---|---|
| Dataset validation | Raw dataset source | Populated dataset summary | `dataset_report.json` | Training scripts, notebook validation | Load JSON and verify schema, duplication, and feature coverage | Missing or inconsistent subject counts / columns |
| Audio preprocessing | Uploaded audio file | Resampled waveform and quality report | None | `pipeline.py` | Run `preprocess_pipeline()` on a sample WAV file | Corrupt file, empty waveform, or failed quality gate |
| Feature extraction | Preprocessed waveform | Clinical feature vector and spectrogram payload | None | `normalization.py`, model inference | Verify output keys exist and vector shape is consistent | Missing feature keys or invalid shape |
| Normalization | Clinical feature vector | Normalized feature vector | `tabular_scaler.pkl` | Model inference | Load scaler and run `transform()` | Missing scaler file or shape mismatch |
| Classical training | Dataset and feature schema | RF / XGB / LGBM outputs | `rf_model.pkl`, `xgb_model.pkl`, `lgbm_model.pkl`, `classical_metrics.json` | `pipeline.py` | Load each artifact and run `predict_proba()` | Missing files, low-quality metrics, failed serialization |
| Ensemble training | Classical model probability outputs | Ensemble artifact and metrics | `ensemble_model.pkl`, `ensemble_metrics.json` | `pipeline.py` | Load ensemble and run `predict_proba()` | Missing calibration artifact or invalid probabilities |
| Severity training | Clinical vector + severity labels | Severity regressor and intervals | `severity_model.pkl`, `severity_metrics.json` | `pipeline.py` | Load model and call `predict_with_tier()` | Missing severity artifact or invalid interval output |
| ONNX export | Ensemble or classical model | Quantized ONNX artifact | `ensemble_quantized.onnx` | Edge fallback and benchmark route | Load with `onnxruntime.InferenceSession` and run a smoke pass | Export failure, unsupported opset, runtime session failure |
| Edge benchmarking | ONNX artifact | Latency, size, throughput report | `onnx_benchmarks.json` | `GET /api/models/benchmarks` | Parse metrics and confirm latency and size fields | Missing benchmark output or invalid numeric payload |
| Analysis upload | Audio file + patient metadata | Analysis DB record | Upload file in `backend/uploads/` | `analysis.py`, `pipeline.py` | Validate file type and size, create DB record | Unsupported file type, oversized audio, patient not found |
| Analysis retrieval | Analysis ID | Stored analysis result | DB-backed analysis record | Frontend pages and detail routes | Query DB and verify response schema | Missing record, wrong status field |
| Explainability | Completed inference result + RF model | SHAP and clinical explanation payload | None | `GET /api/analysis/{id}/explain` | Run explainability on a known sample | Missing feature importance or invalid SHAP shape |
| Frontend binding | API response payload | Rendered UI view | None | `Record.tsx`, `Overview.tsx`, `Analysis.tsx`, `Report.tsx` | Execute a live upload workflow and verify UI rendering | Broken response mapping or unsupported contract mismatch |

---

## 4. Critical Dependency Chains

### Chain A: Training to Inference
- `dataset_report.json` → `train_classical.py` → `rf_model.pkl`, `xgb_model.pkl`, `lgbm_model.pkl`
- `dataset_report.json` → `train_ensemble.py` → `ensemble_model.pkl`
- `dataset_report.json` → `severity_model.py` → `severity_model.pkl`

### Chain B: Feature Extraction to Runtime Prediction
- `audio_preprocessor.py` → `feature_extractor.py` → `normalization.py` → `pipeline.py`

### Chain C: API to User Observability
- `POST /api/analysis/upload` → `analysis.py` → `pipeline.py` → DB persistence → `GET /api/analysis/{id}` → frontend report rendering

### Chain D: Deployment Readiness
- `train_ensemble.py` → `export_onnx.py` → `edge_benchmark.py` → `GET /api/models/benchmarks`

---

## 5. Dependency Validation Rules

1. Every runtime artifact must have a named producer and a named consumer.
2. No inference stage may be marked complete if its artifact dependency is absent from disk.
3. Every model artifact must have a concrete verification step before it is treated as production-ready.
4. Frontend work is never considered integrated until the live API contract is verified with real responses.
5. Research artifacts and production artifacts must remain clearly separated in execution tracking and release review.

This document should be treated as the dependency authority for implementation sequencing and milestone verification.
