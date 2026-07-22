# Pipeline Trace

This document provides the execution trace for a single inference request from audio upload to API response. It is the canonical end-to-end flow for debugging, release verification, and operational support.

## Milestone Alignment

This trace is governed by the same milestone numbering used in [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) and the backlog IDs defined in [BACKLOG.md](BACKLOG.md). The environment-validation work belongs to Milestone 0 and backlog ID `PV-000`; all downstream runtime and product work remains scoped to later milestone IDs.

## End-to-End Inference Flow

Upload
↓
`backend/api/endpoints/analysis.py`
↓
`backend/pipeline/pipeline.py`
↓
`backend/pipeline/audio_preprocessor.py`
↓
`backend/pipeline/feature_extractor.py`
↓
`backend/pipeline/normalization.py`
↓
`backend/ml/train_classical.py` / `backend/ml/train_ensemble.py`
↓
`backend/ml/severity_model.py`
↓
`backend/explainability/shap_explainer.py`
↓
`backend/explainability/risk_breakdown.py`
↓
`backend/core/database.py`
↓
API Response

## Step-by-Step Trace

| Stage | Function | File | Input | Output | Dependencies | Failure Modes | Next Stage |
|---|---|---|---|---|---|---|---|
| 1 | `upload_and_analyze()` | `backend/api/endpoints/analysis.py` | Multipart audio upload | `analysis_id`, saved file path, DB record | Input format validation, storage directory, DB session | Unsupported file type, oversized audio, patient lookup failure | Background processing |
| 2 | `_run_analysis_background()` | `backend/api/endpoints/analysis.py` | `analysis_id`, file path | DB status transitions | Async DB session, `run_inference()` | DB write failure, exception in inference | Master pipeline |
| 3 | `run_inference()` | `backend/pipeline/pipeline.py` | Audio file path, optional patient ID | Full result dictionary | Audio preprocessing, feature extraction, model loading | Missing artifacts, invalid audio, model load exceptions | Quality gate / feature extraction |
| 4 | `preprocess_pipeline()` | `backend/pipeline/audio_preprocessor.py` | Raw waveform file | Resampled audio, quality report | `librosa`, `soundfile`, noise reduction helpers | Corrupt file, low signal quality, empty waveform | Feature extraction |
| 5 | `extract_all_features()` | `backend/pipeline/feature_extractor.py` | Resampled waveform | Clinical feature dictionary + spectral data | `librosa`, `scipy`, `pyin`, `numpy` | Missing pitch, invalid coefficients, unsupported sample length | Normalization |
| 6 | `get_clinical_features()` | `backend/pipeline/feature_extractor.py` | Full feature dictionary | 1D clinical vector | Internal feature schema | Missing expected feature keys | Normalization |
| 7 | `tabular_normalizer.transform()` | `backend/pipeline/normalization.py` | Clinical feature vector | Normalized tabular vector | `tabular_scaler.pkl` | Missing scaler file, shape mismatch | Model inference |
| 8 | `_load_models()` | `backend/pipeline/pipeline.py` | None | Model registry | Serialized `.pkl`, `.pt`, `.onnx` artifacts | Missing model files, bad pickle, torch load errors | Classical and ensemble prediction |
| 9 | Classical inference | `backend/pipeline/pipeline.py` | Normalized clinical vector | Per-model probabilities | `rf_model.pkl`, `xgb_model.pkl`, `lgbm_model.pkl` | Missing files, prediction exception | Ensemble merge |
| 10 | Ensemble inference | `backend/pipeline/pipeline.py` | Normalized clinical vector | `ensemble_prob` | `ensemble_model.pkl` | Missing ensemble artifact, bad calibration | Confidence + uncertainty |
| 11 | Severity inference | `backend/ml/severity_model.py` | Normalized clinical vector | `predicted_updrs`, bounds, tier | `severity_model.pkl` | Missing severity artifact, invalid bounds | Risk classification |
| 12 | `estimate_uncertainty()` | `backend/pipeline/uncertainty.py` | model probabilities, quality metrics | Confidence and uncertainty payload | Ensemble disagreement, quality gate data | Low-confidence or invalid quality input | Explainability |
| 13 | `explain_prediction()` | `backend/explainability/shap_explainer.py` | Normalized vector, feature columns | SHAP values, feature importance | RF or tree model artifacts | SHAP disabled or model not supported | Risk breakdown |
| 14 | `build_risk_breakdown()` | `backend/explainability/risk_breakdown.py` | SHAP output and feature dictionary | Clinical categories and factors | Feature names and clinical references | Missing biomarker mapping | Biomarker dashboard |
| 15 | `build_biomarker_dashboard()` | `backend/explainability/clinical_rules.py` | Full feature dictionary | Human-readable biomarker dashboard | Clinical interpretation rules | Missing rules or out-of-range biomarker keys | DB persistence |
| 16 | `Analysis` DB persistence | `backend/core/database.py` | Result payload | Saved analysis row | SQLite session, ORM model | DB write error, schema mismatch | API retrieval |
| 17 | `get_analysis()` | `backend/api/endpoints/analysis.py` | `analysis_id` | JSON analysis response | DB session | Analysis record missing | Frontend render |

## Trace Ownership Boundary

- Research Layer: dataset validation, training scripts, metric generation, experimentation.
- Inference Layer: preprocessing, feature extraction, normalization, model loading, SHAP, severity, API persistence.
- Product Layer: frontend rendering, history, monitoring, deployment, endpoint contract.

## Operational Notes

- The pipeline must never skip the `quality` gate when audio preprocessing fails.
- Missing model artifacts must be treated as a hard operational blocker, not a soft warning.
- Every downstream stage depends on stable input keys and output schemas remaining consistent across the training and inference layers.
