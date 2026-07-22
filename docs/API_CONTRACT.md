# API Contract

This document defines the production-facing API contract used by the frontend and the backend governance processes. The contract is intentionally stable and must not be changed during documentation-only work.

## Milestone Governance

The endpoint contract in this document is only evaluated within the milestone that owns it. Future milestone backlog items such as `PV-010` or `PV-011` must not be used to reject a current milestone review. The milestone numbering and backlog IDs referenced elsewhere in the execution docs remain authoritative and consistent with [IMPLEMENTATION_ROADMAP.md](IMPLEMENTATION_ROADMAP.md) and [BACKLOG.md](BACKLOG.md).

## Endpoint Catalog

### 1. `POST /api/analysis/upload`

Purpose
- Upload a voice recording and queue background analysis.

Request Schema
- Multipart form field `file` required.
- Optional form field `patient_id` string.

Response Schema
- `id`: string
- `status`: `pending | processing | complete | failed`
- `filename`: string
- `created_at`: timestamp
- `pd_probability`: optional float
- `risk_tier`: optional string
- `confidence`: optional float
- `confidence_category`: optional string
- `uncertainty_score`: optional float
- `severity_score`: optional float
- `predicted_updrs`: optional float
- `severity_tier`: optional string
- `model_predictions`: optional object
- `features`: optional object
- `shap_values`: optional object
- `feature_importance`: optional array
- `risk_breakdown`: optional object
- `clinical_explanations`: optional array
- `biomarkers`: optional array
- `quality`: optional object
- `error_message`: optional string
- `disclaimer`: string

Validation
- Accepted file extensions must match `settings.ALLOWED_AUDIO_FORMATS`.
- Uploaded file size must not exceed `settings.MAX_UPLOAD_SIZE_MB`.
- `patient_id` must resolve to an existing patient record when supplied.

Downstream Dependencies
- `backend/core/database.py`
- `backend/pipeline/pipeline.py`
- `backend/uploads/`

Frontend Consumer
- `frontend/src/pages/Record.tsx`
- `frontend/src/pages/Home.tsx`
- `frontend/src/pages/Analysis.tsx`

Failure Cases
- `415 Unsupported Media Type`
- `413 Payload Too Large`
- `404 Patient not found`
- Async inference failure stored as `status="failed"`

### 2. `GET /api/analysis/{analysis_id}`

Purpose
- Retrieve analysis result status and payload for a given analysis ID.

Request Schema
- Path parameter `analysis_id` string.

Response Schema
- Same response model as upload response.

Validation
- `analysis_id` must exist in the database.

Downstream Dependencies
- `backend/core/database.py`
- `backend/api/endpoints/analysis.py`

Frontend Consumer
- `frontend/src/pages/Analysis.tsx`
- `frontend/src/pages/Report.tsx`

Failure Cases
- `404 Analysis not found`

### 3. `GET /api/analysis/{analysis_id}/explain`

Purpose
- Return explainability payload for a completed analysis.

Request Schema
- Path parameter `analysis_id` string.

Response Schema
- `analysis_id`: string
- `feature_importance`: array
- `shap_values`: object
- `risk_breakdown`: object
- `radar_data`: array

Validation
- Analysis must exist and have `status == "complete"`.

Downstream Dependencies
- `backend/explainability/risk_breakdown.py`
- `backend/explainability/shap_explainer.py`

Frontend Consumer
- `frontend/src/pages/Explainability.tsx`

Failure Cases
- `404 Analysis not found`
- `422 Analysis is not complete`

### 4. `DELETE /api/analysis/{analysis_id}`

Purpose
- Remove a stored analysis record and its uploaded file.

Validation
- Analysis must exist before delete.

Failure Cases
- `404 Analysis not found`

### 5. `GET /api/analysis/`

Purpose
- List recently created analyses with optional patient filter.

Validation
- Page size and offset are optional and bounded by the router implementation.

### 6. `GET /api/models/status`

Purpose
- Return model-file readiness status for each inference component.

Response Schema
- `models`: object keyed by model component
- `scaler_ready`: boolean
- `any_model_ready`: boolean
- `full_pipeline_ready`: boolean

Downstream Dependencies
- `backend/models/`
- `backend/model_registry.json`

Frontend Consumer
- `frontend/src/pages/Overview.tsx`
- `frontend/src/pages/Benchmarks.tsx`

### 7. `GET /api/models/registry`

Purpose
- Return model registry metadata.

### 8. `GET /api/models/metrics`

Purpose
- Return classical, ensemble, CNN, severity, and ONNX benchmark metrics.

### 9. `GET /api/models/comparison`

Purpose
- Return structured comparison table for frontend dashboards.

### 10. `GET /api/models/benchmarks`

Purpose
- Return hardware benchmark profiles for ONNX edge deployment.

### 11. `POST /api/models/train`

Purpose
- Trigger the authoritative training pipeline.

Validation
- The script paths referenced by the endpoint must exist before launch.

Failure Cases
- `script_not_found` for missing training entrypoint.

### 12. Patient Endpoints

- `POST /api/patients/`
- `GET /api/patients/`
- `GET /api/patients/{patient_id}`
- `PUT /api/patients/{patient_id}`
- `DELETE /api/patients/{patient_id}`
- `GET /api/patients/{patient_id}/history`
- `GET /api/patients/{patient_id}/trends`

These endpoints must remain stable with the current response shapes for the frontend history and longitudinal trend pages.

## Contract Governance

1. Do not change fields in existing response models without updating every frontend consumer.
2. No endpoint may be treated as complete until its schema and downstream dependencies are explicitly verified.
3. Any contract change must be documented here first and then reflected in the implementation checklist.
