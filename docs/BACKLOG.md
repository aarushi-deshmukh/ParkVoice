# ParkVoice AI — Authoritative Implementation Backlog

**Version:** 1.0.0-Phase0  
**Status:** Authoritative Post-Audit Backlog  
**Sorting:** Strict Dependency Order (Tasks are ordered sequentially so each task unlocks downstream work without blocking).

---

## Governing Execution Order

0. Validate the development environment.
1. Stabilize the runtime contract.
2. Validate dataset and feature path.
3. Train the classical ML path.
4. Train the ensemble and severity path.
5. Integrate the frontend against live backend responses.
6. Package and benchmark the edge deployment path.
7. Execute release verification and regression checks.

---

## Atomic Task Groups

### PV-000: Development Environment Validation

#### PV-000.1
Create the Python virtual environment.

#### PV-000.2
Install dependencies.

#### PV-000.3
Run the backend and confirm startup readiness.

#### PV-000.4
Run the frontend and confirm build readiness.

#### PV-000.5
Execute `pytest` in the configured environment.

#### PV-000.6
Resolve import path issues so the repository can be executed cleanly.

#### PV-000.7
Verify Docker build readiness.

#### PV-000.8
Verify required environment variables and repository configuration.

#### PV-000.9
Generate the environment validation report.

**Definition of Done**
- Development environment is fully reproducible.

---

### PV-001: Core API Contract Repair

#### PV-001.1
Verify `GET /api/models/status` response shape.

#### PV-001.2
Fix unreachable logic in `backend/api/endpoints/models_endpoint.py`.

#### PV-001.3
Return a stable `models`, `scaler_ready`, `any_model_ready`, and `full_pipeline_ready` payload.

#### PV-001.4
Verify `GET /api/models/registry` remains callable with the current registry file.

#### PV-001.5
Smoke-test backend startup and route accessibility.

---

### PV-002: Data Path Validation

#### PV-002.1
Execute notebook validation workflow for the dataset.

#### PV-002.2
Verify `dataset_report.json` schema and population completeness.

#### PV-002.3
Confirm the audio preprocessing stage can open a sample input.

#### PV-002.4
Confirm feature extraction outputs the expected clinical vector and spectral payload.

#### PV-002.5
Validate the normalized feature contract expected by downstream inference.

---

### PV-003: Classical ML Training Path

#### PV-003.1
Verify dataset features are available for the training script.

#### PV-003.2
Fit the scaler artifact.

#### PV-003.3
Train the Random Forest classifier.

#### PV-003.4
Train the XGBoost classifier.

#### PV-003.5
Train the LightGBM classifier.

#### PV-003.6
Generate `classical_metrics.json`.

#### PV-003.7
Serialize all classical model artifacts to `backend/models/`.

#### PV-003.8
Smoke-load each model and confirm `predict_proba()` works.

---

### PV-004: Ensemble and Severity Path

#### PV-004.1
Train the weighted ensemble model.

#### PV-004.2
Generate `ensemble_metrics.json`.

#### PV-004.3
Serialize `ensemble_model.pkl`.

#### PV-004.4
Train the severity regressor.

#### PV-004.5
Generate severity interval outputs and metrics.

#### PV-004.6
Serialize `severity_model.pkl` and `severity_metrics.json`.

#### PV-004.7
Verify the pipeline can return calibrated probability and severity values in one response.

---

### PV-005: Deployment-Ready Export Path

#### PV-005.1
Export the ensemble artifact to ONNX.

#### PV-005.2
Apply dynamic INT8 quantization and produce `ensemble_quantized.onnx`.

#### PV-005.3
Benchmark the ONNX artifact on the host CPU.

#### PV-005.4
Write `onnx_benchmarks.json` with valid profile values.

#### PV-005.5
Verify the benchmark endpoint returns the expected shape for the frontend.

---

### PV-006: Frontend Integration Path

#### PV-006.1
Verify the upload page posts to the live API without mock-only data paths.

#### PV-006.2
Verify `Analysis.tsx` polls the backend result correctly.

#### PV-006.3
Verify the report page renders persisted analysis fields from the API response.

#### PV-006.4
Verify the model status and benchmark pages read live outputs rather than placeholder data.

#### PV-006.5
Confirm the frontend integration path is stable across the main user workflow.

---

### PV-007: Quality Gates and Release

#### PV-007.1
Create or expand regression tests for the analysis upload path.

#### PV-007.2
Create or expand regression tests for the model status path.

#### PV-007.3
Create or expand regression tests for feature extraction and normalization.

#### PV-007.4
Create or expand regression tests for inference smoke checks.

#### PV-007.5
Run the full backend regression suite.

#### PV-007.6
Confirm milestone gates and required artifacts are all present.

#### PV-007.7
Compile the final release evidence for the current implementation state.

---

## Atomic Task Rule

Every task above must be executable independently enough to produce a testable outcome. Broad labels such as “train model” must be decomposed into atomic validation, serialization, and smoke-test steps before they are considered complete.
