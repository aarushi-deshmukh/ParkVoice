# Parkinson's Disease Early Detection Platform — Implementation Plan

## Overview

A production-ready, research-grade AI platform for early Parkinson's Disease detection from voice recordings. The system provides explainable risk assessment, severity prediction, longitudinal patient tracking, and a novel temporal progression model — all within a premium healthcare UI.

---

## Architecture

```
parkinson/
├── backend/                    # FastAPI Python backend
│   ├── api/                    # REST endpoints
│   ├── core/                   # Config, security, DB
│   ├── data/                   # Raw datasets (gitignored)
│   ├── models/                 # Saved ML models (.pkl, .onnx)
│   ├── pipeline/               # Feature extraction + preprocessing
│   ├── ml/                     # Training scripts & model definitions
│   ├── explainability/         # SHAP + visualization
│   ├── research/               # Temporal progression model
│   └── evaluation/             # Metrics + reporting
├── frontend/                   # Vite + React + TypeScript
│   ├── src/
│   │   ├── pages/              # All 6 UI pages
│   │   ├── components/         # Reusable UI components
│   │   ├── store/              # Zustand state
│   │   └── api/                # Axios API layer
└── notebooks/                  # Jupyter research notebooks
    ├── 01_eda.ipynb
    ├── 02_feature_engineering.ipynb
    ├── 03_model_comparison.ipynb
    ├── 04_temporal_model.ipynb
    └── methodology.md          # Publication-style write-up
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| Backend API | FastAPI + Uvicorn |
| ML / DL | scikit-learn, XGBoost, LightGBM, PyTorch |
| Audio Processing | librosa, soundfile, noisereduce, pyworld |
| Explainability | SHAP, matplotlib, seaborn |
| Database | SQLite (dev) → PostgreSQL (prod) via SQLAlchemy |
| Frontend | Vite + React + TypeScript |
| Charts | Recharts + D3.js |
| State | Zustand |
| ONNX Export | onnx, skl2onnx, torch.onnx |
| Packaging | Docker + docker-compose |

---

## Proposed Changes

### 1. Data Pipeline (`backend/pipeline/`)

#### [NEW] `audio_preprocessor.py`
- Load audio files (WAV, MP3, M4A)
- Noise reduction using `noisereduce` spectral gating
- Silence trimming, resampling to 22050 Hz
- Normalization

#### [NEW] `feature_extractor.py`
Extracts all acoustic biomarkers:
- **MFCC** (13–40 coefficients + delta + delta-delta)
- **Mel-spectrogram** (for CNN input)
- **Pitch (F0)** via `pyworld` HARVEST algorithm
- **Jitter** (local, local absolute, RAP, PPQ5, DDP)
- **Shimmer** (local, local dB, APQ3, APQ5, APQ11, DDA)
- **HNR** (Harmonics-to-Noise Ratio)
- **NHR** (Noise-to-Harmonics Ratio)
- **RPDE, DFA, PPE** — nonlinear dynamics
- **Spread1, Spread2, D2** — signal complexity
- Voice stability metrics (pitch SD, jitter SD over time)

#### [NEW] `normalization.py`
- RobustScaler for tabular features (outlier-resistant)
- Min-Max for spectrogram pixel data
- Saves fitted scalers as `.pkl` artifacts

#### [NEW] `dataset_loader.py`
- Auto-downloads UCI Parkinson's Dataset (ucimlrepo)
- Loads Parkinson Speech Dataset CSV + audio files
- Merges and labels datasets
- Train/Val/Test stratified splits (70/15/15)

---

### 2. ML Models (`backend/ml/`)

#### [NEW] `train_classical.py`
Trains tabular models on extracted features:
- **Random Forest** (GridSearchCV optimized)
- **XGBoost** (with early stopping)
- **LightGBM** (with DART boosting)
- Saves each model as `.pkl` + exports to ONNX

#### [NEW] `train_cnn.py`
- Input: Mel-spectrograms (224×224 images)
- Architecture: EfficientNet-B0 backbone (transfer learning)
- Fine-tuned on Parkinson spectrograms
- Outputs: PD probability + severity score

#### [NEW] `train_cnn_lstm.py`
- Hybrid architecture for sequential voice segments
- CNN encoder → temporal LSTM → classification head
- Captures temporal dynamics within a single recording

#### [NEW] `train_ensemble.py`
- Soft-voting ensemble combining all 5 models
- Calibrated probabilities via Platt scaling
- Weighted by validation ROC-AUC

#### [NEW] `severity_model.py`
- UPDRS score regression (not binary classification)
- Trained on Parkinson Speech Dataset severity labels
- Outputs: UPDRS score (0–108), severity tier (Mild/Moderate/Severe)

#### [NEW] `temporal_progression.py` *(Novel Research Component)*
- Transformer-based temporal model (research contribution)
- Input: Time-series of voice feature vectors from patient history
- Predicts **6-month progression risk** score
- Architecture: Multi-head self-attention over longitudinal recordings

---

### 3. Explainability (`backend/explainability/`)

#### [NEW] `shap_explainer.py`
- TreeExplainer for RF/XGB/LGBM
- KernelExplainer for CNN (sampling-based)
- Returns: SHAP values, feature importance ranking, waterfall plot data

#### [NEW] `risk_breakdown.py`
- Maps SHAP values to clinical feature categories
- Groups into: Pitch Instability, Tremor Markers, Noise Ratio, Complexity
- Returns structured risk-factor breakdown with confidence intervals

#### [NEW] `report_generator.py`
- Generates PDF-ready assessment reports
- Includes: Risk score, severity tier, feature breakdown, SHAP plots, recommendations

---

### 4. Evaluation (`backend/evaluation/`)

#### [NEW] `metrics.py`
Computes: Accuracy, Precision, Recall, F1, ROC-AUC, Sensitivity, Specificity, Cohen's Kappa, MCC

#### [NEW] `model_comparison.py`
- Cross-validation comparison table
- Bootstrap confidence intervals
- Generates publication-ready comparison DataFrame

---

### 5. API (`backend/api/`)

#### [NEW] `endpoints/analysis.py`
- `POST /api/analyze` — Upload audio → run full pipeline → return results
- `GET /api/analysis/{id}` — Fetch saved analysis
- `GET /api/explain/{id}` — Get SHAP explanation for analysis

#### [NEW] `endpoints/patients.py`
- `POST /api/patients` — Create patient profile
- `GET /api/patients/{id}/history` — Longitudinal history
- `GET /api/patients/{id}/trends` — Progression trends

#### [NEW] `endpoints/models.py`
- `GET /api/models/metrics` — All model performance metrics
- `GET /api/models/comparison` — Comparison table

---

### 6. Frontend (`frontend/`)

Six premium healthcare pages:

| Page | Route | Description |
|---|---|---|
| Home / Upload | `/` | Drag-drop audio upload with waveform preview |
| Record | `/record` | In-browser microphone recording with live visualizer |
| Analysis | `/analysis/:id` | Real-time processing progress + feature extraction results |
| Risk Report | `/report/:id` | Full clinical risk report, severity gauge, radar chart |
| Explainability | `/explain/:id` | SHAP waterfall, feature importance bar charts |
| History | `/history` | Longitudinal trends, progression chart, patient tracker |

Design System:
- Dark healthcare theme (deep navy + clinical teal + alert amber)
- Inter + JetBrains Mono fonts (Google Fonts)
- Glassmorphism panels, subtle grid texture
- Animated waveform visualizer for recording
- Smooth page transitions
- Responsive (desktop-first, mobile-aware)

---

### 7. Research Component (`notebooks/`)

#### [NEW] `methodology.md`
Publication-style write-up covering:
- Abstract, Introduction, Related Work
- Dataset description + preprocessing
- Feature engineering rationale
- Model architectures
- Temporal progression model (novel contribution)
- Results + ablation study
- Conclusion

---

### 8. Deployment

#### [NEW] `Dockerfile` + `docker-compose.yml`
- Multi-stage build (builder + runtime)
- CPU-optimized inference container
- ONNX Runtime for edge inference

#### [NEW] `export_onnx.py`
- Exports all models to ONNX format
- INT8 quantization for edge deployment
- Benchmark inference speed

---

## Open Questions

> [!IMPORTANT]
> **Q1: Do you have local audio files or datasets already downloaded?**
> The system will auto-download UCI Parkinson's Dataset via `ucimlrepo`. For the Parkinson Speech Dataset (with audio recordings), it requires manual download from Kaggle/UCI due to authentication. Should I include automated Kaggle CLI download instructions or assume manual download?

> [!IMPORTANT]
> **Q2: Frontend Framework Preference**
> The UI will use **Vite + React + TypeScript** for the 6-page SPA. This gives the best component architecture for charts and real-time audio. Is this acceptable, or would you prefer a simpler HTML/CSS/JS approach?

> [!IMPORTANT]
> **Q3: Deployment Target**
> Should the system run entirely **locally** (localhost dev server), or should I also configure for cloud deployment (Railway, Render, or Docker)? The ONNX edge export will be included either way.

> [!IMPORTANT]
> **Q4: GPU Availability**
> The CNN and CNN-LSTM models train significantly faster with a GPU. Should I include GPU training scripts (CUDA) or optimize entirely for CPU training (with smaller models)?

> [!NOTE]
> **Execution Order**: I'll build this in 6 phases — (1) Project scaffold, (2) Data pipeline, (3) ML training scripts, (4) Backend API, (5) Frontend UI, (6) Research docs + export. Total estimated scope: ~80 files.

## Verification Plan

### Automated Tests
```bash
pytest backend/tests/ -v --cov=backend
```

### Manual Verification
- Run `python backend/ml/train_classical.py` — verify models train without errors
- Start `uvicorn backend.main:app` — test `/api/analyze` with sample WAV
- `npm run dev` in frontend — verify all 6 pages render
- Upload a sample WAV file end-to-end through the UI
