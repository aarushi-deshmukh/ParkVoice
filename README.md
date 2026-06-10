# ParkVoice AI - Parkinson's Risk Assessment Platform

ParkVoice AI is a research-grade acoustic screening support platform for Parkinson's risk assessment from voice recordings. It is not a diagnostic tool.

**Global disclaimer:** This system is intended for research and screening support purposes only and is not a diagnostic medical device.

## Architecture

Dataset Validation -> Audio Quality Assessment -> Feature Extraction -> Calibrated Ensemble Risk Assessment -> Severity Estimation -> Explainability -> Uncertainty Estimation -> ONNX Edge Inference -> Benchmark Reporting -> Executive Dashboard

## Models Kept

- Random Forest with `CalibratedClassifierCV(method="sigmoid", cv=5)`
- XGBoost with `CalibratedClassifierCV(method="sigmoid", cv=5)`
- LightGBM with `CalibratedClassifierCV(method="sigmoid", cv=5)`
- EfficientNet-B0 spectrogram classifier

Redundant deep learning model paths have been removed from the active platform.

## Screening Outputs

- Calibrated risk score and risk tier
- Confidence and uncertainty score
- Audio quality score, category, and warnings
- UPDRS severity estimate with conformal prediction bounds
- SHAP feature contributions
- Clinician-friendly biomarker explanations
- Biomarker dashboard rows for jitter, shimmer, HNR, NHR, RPDE, DFA, PPE, pitch variability, and signal complexity

## Backend

```bash
cd backend
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

API docs are available at `http://localhost:8000/api/docs`.

Useful endpoints:

- `POST /api/analysis/upload`
- `GET /api/analysis/{analysis_id}`
- `GET /api/models/registry`
- `GET /api/models/benchmarks`
- `GET /api/models/status`

## Frontend

```bash
cd frontend
npm install
npm run dev
```

The Overview page is the primary demo page and shows risk score, confidence, uncertainty, audio quality, severity, biomarkers, SHAP insights, and real benchmark results when available.

## Dataset Validation

Use `notebooks/00_dataset_validation.ipynb` to create or refresh `dataset_report.json`.

Tracked validation fields:

- subjects
- recordings
- class balance
- audio availability
- UPDRS availability
- longitudinal availability

Unsupported features are disabled when their required dataset fields are unavailable.

## ONNX And Benchmarks

Export and quantize:

```bash
python backend/ml/train_classical.py
python backend/ml/severity_model.py
python backend/ml/train_ensemble.py
python backend/ml/export_onnx.py
python backend/evaluation/edge_benchmark.py
```

Benchmark reporting only displays real measurements from the hardware where `edge_benchmark.py` was executed. Raspberry Pi and Jetson results require runs on those devices.

## Development Deployment

```bash
docker compose up --build
```

## Production Deployment

- Frontend: Vercel
- Backend: Railway
- Database: Neon PostgreSQL
- Storage: Supabase Storage

Configure production environment variables from `backend/.env.example` and use managed storage for uploaded voice recordings.
