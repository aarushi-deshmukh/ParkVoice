# ParkVoice AI — Phase 0 Project Audit Report

**Date of Audit:** July 23, 2026  
**Auditor Role:** Lead Software Architect & Technical Auditor  
**Project:** ParkVoice AI — Research-Grade Parkinson's Disease Acoustic Screening Support Platform  

---

## 1. Repository Overview

ParkVoice AI is designed as a research-grade acoustic screening and risk assessment platform for Parkinson's Disease (PD). The system processes human voice recordings (sustained vowel phonation /a/) to extract 20+ clinical acoustic biomarkers, compute calibrated machine learning ensemble risk scores, estimate UPDRS motor severity with conformal prediction intervals, provide SHAP game-theoretic explainability, and package inference for ONNX INT8 edge deployment.

### Workspace Structure & File Inventory

```
d:\Projects\parkinson
├── .gitignore
├── README.md
├── dataset_report.json
├── docker-compose.yml
├── implementation_plan.md
│
├── backend/
│   ├── .env.example
│   ├── Dockerfile
│   ├── main.py
│   ├── model_registry.json
│   ├── parkvoice.db
│   ├── railway.json
│   ├── requirements.txt
│   ├── api/
│   │   ├── __init__.py
│   │   └── endpoints/
│   │       ├── __init__.py
│   │       ├── analysis.py
│   │       ├── models_endpoint.py
│   │       └── patients.py
│   ├── core/
│   │   ├── __init__.py
│   │   ├── config.py
│   │   ├── database.py
│   │   └── models.py
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── edge_benchmark.py
│   │   └── metrics.py
│   ├── explainability/
│   │   ├── __init__.py
│   │   ├── clinical_rules.py
│   │   ├── risk_breakdown.py
│   │   └── shap_explainer.py
│   ├── ml/
│   │   ├── __init__.py
│   │   ├── export_onnx.py
│   │   ├── severity_model.py
│   │   ├── train_classical.py
│   │   ├── train_cnn.py
│   │   └── train_ensemble.py
│   ├── models/                  (Empty Directory — Missing Binaries)
│   ├── pipeline/
│   │   ├── __init__.py
│   │   ├── audio_preprocessor.py
│   │   ├── audio_quality.py
│   │   ├── dataset_loader.py
│   │   ├── feature_extractor.py
│   │   ├── normalization.py
│   │   ├── pipeline.py
│   │   └── uncertainty.py
│   ├── research/
│   │   └── experimental/
│   │       ├── __init__.py
│   │       └── temporal_progression.py
│   ├── tests/
│   │   └── test_screening_modules.py
│   └── uploads/                 (Directory for uploaded audio files)
│
├── frontend/
│   ├── .gitignore
│   ├── Dockerfile
│   ├── README.md
│   ├── eslint.config.js
│   ├── index.html
│   ├── nginx.conf
│   ├── package-lock.json
│   ├── package.json
│   ├── tsconfig.app.json
│   ├── tsconfig.json
│   ├── tsconfig.node.json
│   ├── vercel.json
│   ├── vite.config.ts
│   └── src/
│       ├── App.css
│       ├── App.tsx
│       ├── index.css
│       ├── main.tsx
│       ├── api/
│       │   └── client.ts
│       ├── components/
│       │   ├── AudioWaveform.tsx
│       │   ├── FeatureChart.tsx
│       │   ├── Navbar.tsx
│       │   └── RiskGauge.tsx
│       ├── pages/
│       │   ├── Analysis.tsx
│       │   ├── Benchmarks.tsx
│       │   ├── Biomarkers.tsx
│       │   ├── Explainability.tsx
│       │   ├── History.tsx
│       │   ├── Home.tsx
│       │   ├── Overview.tsx
│       │   ├── Record.tsx
│       │   └── Report.tsx
│       └── store/
│           └── useStore.ts
│
├── docs/
│   ├── architecture.md
│   └── deployment.md
│
└── notebooks/
    ├── 00_dataset_validation.ipynb
    └── methodology.md
```

---

## 2. Target Architecture vs. Implementation Reality

| Target Architecture Stage | Target Specification | Actual Implementation Status | Empirical Evidence / Finding |
| :--- | :--- | :--- | :--- |
| **1. Upload Voice** | Ingestion of WAV, MP3, FLAC, M4A | ✅ Fully Implemented | Handled by `POST /api/analysis/upload` in [analysis.py](file:///d:/Projects/parkinson/backend/api/endpoints/analysis.py#L154) and frontend [Record.tsx](file:///d:/Projects/parkinson/frontend/src/pages/Record.tsx) & [Home.tsx](file:///d:/Projects/parkinson/frontend/src/pages/Home.tsx). |
| **2. Dataset Validation** | Ingestion & validation of UCI / clinical audio datasets | 🔴 Stub / Mocked | `dataset_report.json` shows 0 subjects and 0 recordings. `00_dataset_validation.ipynb` is unexecuted. `dataset_loader.py` falls back to generating synthetic data via `_generate_synthetic_dataset()`. |
| **3. Audio Quality Assessment** | SNR, clipping ratio, silence ratio, signal stability check | ✅ Fully Implemented | Module [audio_quality.py](file:///d:/Projects/parkinson/backend/pipeline/audio_quality.py) and `validate_recording()` in [audio_preprocessor.py](file:///d:/Projects/parkinson/backend/pipeline/audio_preprocessor.py#L114). |
| **4. Acoustic Feature Extraction** | 22 UCI tabular biomarkers, 40 MFCCs, Mel-spectrograms | ✅ Fully Implemented | Extracted in 533-line module [feature_extractor.py](file:///d:/Projects/parkinson/backend/pipeline/feature_extractor.py) using `librosa`, `scipy`, and `pyin`. |
| **5. Classical ML Models** | Calibrated Random Forest, XGBoost, LightGBM | 🟡 Partially Implemented | Code in [train_classical.py](file:///d:/Projects/parkinson/backend/ml/train_classical.py), but binary artifacts (`rf_model.pkl`, `xgb_model.pkl`, `lgbm_model.pkl`) are missing from `backend/models/`. |
| **6. EfficientNet Spectrogram Model** | EfficientNet-B0 fine-tuned on 2D Mel-spectrogram images | 🟡 Partially Implemented / Mocked Data | Model structure defined in [train_cnn.py](file:///d:/Projects/parkinson/backend/ml/train_cnn.py#L37), but trained on synthetic spectrogram images (`SyntheticSpectrogramDataset`) because real audio files are absent. Binary `cnn_model.pt` missing on disk. |
| **7. Calibrated Ensemble** | Soft-voting ensemble weighted by validation ROC-AUC | 🟡 Partially Implemented | Class `WeightedEnsemble` in [train_ensemble.py](file:///d:/Projects/parkinson/backend/ml/train_ensemble.py#L29), but relies on trained base model `.pkl` files which are missing. |
| **8. Risk Prediction** | Calibrated probability & 4-tier risk classification | ✅ Implemented in Code | Functions `_classify_risk()` and `run_inference()` in [pipeline.py](file:///d:/Projects/parkinson/backend/pipeline/pipeline.py#L141). |
| **9. Severity Estimation** | UPDRS-III regressor with 95% conformal prediction intervals | 🟡 Partially Implemented / Synthetic Labels | Implemented in [severity_model.py](file:///d:/Projects/parkinson/backend/ml/severity_model.py) using `conformal_qhat`, but trained on synthetic UPDRS targets generated by `add_synthetic_severity()`. Binary missing. |
| **10. Explainability** | SHAP values mapped to clinical biomarker clusters | ✅ Fully Implemented | Modules [shap_explainer.py](file:///d:/Projects/parkinson/backend/explainability/shap_explainer.py), [risk_breakdown.py](file:///d:/Projects/parkinson/backend/explainability/risk_breakdown.py), and [clinical_rules.py](file:///d:/Projects/parkinson/backend/explainability/clinical_rules.py). |
| **11. Uncertainty Estimation** | Ensemble disagreement + calibration + audio quality | ✅ Fully Implemented | Function `estimate_uncertainty()` in [uncertainty.py](file:///d:/Projects/parkinson/backend/pipeline/uncertainty.py). |
| **12. Clinical Dashboard** | Responsive React UI with gauges, radar, charts, and history | 🟡 Partially Implemented | 9 pages built in [frontend/src/pages](file:///d:/Projects/parkinson/frontend/src/pages), but contains fallback mock data (`demoAnalysis` in `Overview.tsx`). |
| **13. ONNX Edge Inference** | INT8 dynamically quantized ONNX edge model | 🟡 Partially Implemented | Export logic in [export_onnx.py](file:///d:/Projects/parkinson/backend/ml/export_onnx.py), but `ensemble_quantized.onnx` does not exist in `backend/models/`. |
| **14. Benchmark Reporting** | Hardware latency, size, and throughput profiling | 🟡 Partially Implemented | Profiler [edge_benchmark.py](file:///d:/Projects/parkinson/backend/evaluation/edge_benchmark.py) works, but output `onnx_benchmarks.json` is not present on disk. |

---

## 3. Implementation Summary

1. **Code Completeness**: The pipeline logic, mathematical feature extraction equations, explainability mapping, and frontend React components are almost completely written.
2. **Missing Binary Artifacts**: The primary reason the backend cannot execute real inference out-of-the-box is that the `backend/models/` directory is **completely empty**. None of the `.pkl`, `.pt`, `.onnx`, or `.json` metric files have been trained or compiled onto disk.
3. **Data Dependency Deficit**: The system lacks real raw audio datasets. The UCI Parkinson's dataset (which is tabular-only) is downloaded or generated synthetically, and real audio for training the EfficientNet-B0 CNN is unavailable.
4. **Endpoint Defect**: A critical syntax error / unreachable code block exists in `backend/api/endpoints/models_endpoint.py` (lines 31–63), preventing `/api/models/status` from returning status data and leaving lines 52–62 unreachable.
