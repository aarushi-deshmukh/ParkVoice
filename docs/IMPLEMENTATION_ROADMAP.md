# ParkVoice AI — Milestone-Based Execution Roadmap

This roadmap is the authoritative execution plan for the remainder of the project. It replaces the previous phase-only structure with dependency-ordered milestones that produce working subsystems after each step.

## Execution Model

The project is organized into three governance layers:

- Research Layer
  - Training
  - Evaluation
  - Experiments
  - Datasets

- Inference Layer
  - Preprocessing
  - Feature Extraction
  - Models
  - SHAP
  - Severity
  - API

- Product Layer
  - Frontend
  - Database
  - Authentication
  - Deployment
  - Monitoring

The implementation order below minimizes blockers and maximizes the number of working subsystems after every milestone.

---

## Milestone Overview

0. Milestone 0 — Development Environment Validation
1. Milestone 1 — Core Infrastructure
2. Milestone 2 — Data Pipeline
3. Milestone 3 — Classical ML
4. Milestone 4 — Ensemble + Severity
5. Milestone 5 — Frontend Integration
6. Milestone 6 — Deployment & Edge
7. Milestone 7 — Quality Assurance & Release

---

## Review Discipline

A milestone review must evaluate only the work that belongs to that milestone:
- backlog items assigned to the milestone
- definition of done
- acceptance criteria
- verification gate

Future milestone tasks must never block a current milestone review. A milestone review shall not fail because of unrelated later-stage work such as frontend mock data, CNN completion, or ONNX export if those items are assigned to later milestones.

---

## Milestone 0 — Development Environment Validation

Goal
- Verify that the project can be developed reproducibly before any backlog item is implemented.

Scope
- Create a Python virtual environment.
- Install dependencies.
- Verify backend startup.
- Verify frontend build.
- Verify `pytest` execution.
- Resolve import path problems.
- Verify Docker build.
- Verify environment variables.
- Verify database initialization.
- Verify repository structure.

Definition of Done
- ✓ Backend starts.
- ✓ Frontend builds.
- ✓ `pytest` runs.
- ✓ Imports resolve.
- ✓ Database initializes.
- ✓ Docker builds successfully.

Acceptance Criteria
- The development environment is reproducible from a clean clone.
- The environment validation evidence can be used to determine whether the next milestone may proceed.

Verification Steps
1. Create a clean Python virtual environment.
2. Install backend and frontend dependencies.
3. Start the backend and confirm it reaches the expected runtime state.
4. Build the frontend and confirm the production bundle can be generated.
5. Run `pytest` and confirm the environment can execute the targeted test entrypoint.
6. Resolve import path or package discovery blockers.
7. Verify Docker build and required environment variables.
8. Confirm the database initializes cleanly.

Expected Deliverables
- Reproducible environment setup instructions.
- Environment validation report.
- Verified readiness to begin Milestone 1.

Gate 0
Development Environment Ready
↓
PASS
↓
Proceed

---

## Milestone 1 — Core Infrastructure

Goal
- Restore the backend contract and make the API operational enough to support downstream execution.

Scope
- Fix the `GET /api/models/status` control flow issue.
- Stabilize the model / registry status contract.
- Confirm backend startup, DB initialization, and health of the core API endpoints.
- Establish the minimum execution envelope needed for all later milestones.

Definition of Done
- ✓ `GET /api/models/status` returns a valid JSON response.
- ✓ `GET /api/models/registry` serves registry metadata when available.
- ✓ Backend starts successfully.
- ✓ Core API routes are reachable.
- ✓ `backend/models/` and `backend/uploads/` contract is understood by the runtime.

Acceptance Criteria
- The API no longer returns null or unreachable code paths for the model status contract.
- The backend can start in local development mode without schema or import blockers.
- Model status handling is deterministic and read from disk.

Verification Steps
1. Start the backend application.
2. Call `GET /api/models/status`.
3. Call `GET /api/models/registry`.
4. Confirm the response payloads match the documented contract.

Expected Deliverables
- Stable backend startup path.
- Corrected model status route behavior.
- Verified API contract for model readiness.

Gate 1
Backend starts
↓
All core endpoints respond
↓
Continue

---

## Milestone 2 — Data Pipeline

Goal
- Establish a validated data path that can feed inference and training without synthetic assumptions.

Scope
- Validate the dataset report.
- Confirm data schema and train/test feature availability.
- Verify the audio preprocessing and feature extraction path.
- Ensure the inference layer can execute on real audio inputs once model artifacts exist.

Definition of Done
- ✓ Dataset validation notebook completed.
- ✓ `dataset_report.json` populated with real schema and counts.
- ✓ Audio preprocessing succeeds on an input sample.
- ✓ Feature extraction succeeds and produces a stable clinical vector.
- ✓ Inference pipeline can reach model-loading without runtime contract failures.

Acceptance Criteria
- The dataset path is deterministic and schema-validated.
- Audio preprocessing and feature extraction produce the expected vector contract.
- The raw feature extraction output is consistent with the downstream normalization and model-loading flow.

Verification Steps
1. Run dataset validation workflow.
2. Inspect `dataset_report.json` for schema, counts, and completeness.
3. Execute preprocessing over a sample recording.
4. Execute feature extraction over the same sample.
5. Confirm output keys are compatible with `pipeline.py` and `normalization.py`.

Expected Deliverables
- Validated dataset report.
- Working preprocessing and feature extraction path.
- Traceable artifact inventory for downstream model training.

Gate 2
Dataset validation succeeds
↓
Audio preprocessing succeeds
↓
Feature extraction succeeds
↓
Continue

---

## Milestone 3 — Classical ML

Goal
- Train and serialize the classical inference stack so the tabular classification path becomes operational.

Scope
- Fit the scaler.
- Train RF, XGB, and LGBM.
- Generate classical metrics.
- Ensure the classical models can load and predict successfully.

Definition of Done
- ✓ RF trained
- ✓ XGB trained
- ✓ LGBM trained
- ✓ Scaler serialized
- ✓ Metrics generated
- ✓ Models load successfully
- ✓ API can perform prediction

Acceptance Criteria
- `tabular_scaler.pkl`, `rf_model.pkl`, `xgb_model.pkl`, and `lgbm_model.pkl` are stored in `backend/models/`.
- `classical_metrics.json` contains verifiable metrics.
- The classical model branch is available through the inference pipeline.

Verification Steps
1. Execute the classical training script.
2. Verify the artifact files exist in `backend/models/`.
3. Load each model with the appropriate runtime path.
4. Run a smoke inference with a normalized vector.
5. Confirm the `model_predictions` dictionary contains the expected model keys.

Expected Deliverables
- Serialized classical models.
- Validation metrics artifact.
- Working tabular classification inference branch.

Gate 3
Models load
↓
Inference succeeds
↓
Continue

---

## Milestone 4 — Ensemble + Severity

Goal
- Produce the calibrated ensemble and severity branch needed for integrated risk scoring.

Scope
- Train the weighted ensemble.
- Train and serialize the severity regressor.
- Confirm probability and severity outputs are compatible with the API response contract.

Definition of Done
- ✓ Ensemble model trained
- ✓ Ensemble metrics generated
- ✓ Severity model trained
- ✓ Severity metrics generated
- ✓ Inference response contains calibrated risk and severity outputs
- ✓ Ensemble and severity artifacts are loadable

Acceptance Criteria
- `ensemble_model.pkl` and `severity_model.pkl` exist in `backend/models/`.
- The model response payload contains `pd_probability`, `severity_score`, interval bounds, and tier labels.
- The full inference path remains stable when the ensemble and severity branches are enabled.

Verification Steps
1. Execute the ensemble training script.
2. Execute the severity training script.
3. Verify the artifact files exist.
4. Run a smoke inference using a sample recording.
5. Confirm the output fields are populated and consistent with the API contract.

Expected Deliverables
- Ensemble artifact set.
- Severity artifact set.
- Integrated risk and severity inference output.

Gate 4
Ensemble loads
↓
Severity loads
↓
Full inference payload is valid
↓
Continue

---

## Milestone 5 — Frontend Integration

Goal
- Bind the frontend to the live backend contract, replacing placeholder and mock-driven behavior where necessary.

Scope
- Confirm the frontend consumes the live analysis, model status, patient history, and benchmark endpoints.
- Ensure the user-facing pages show real backend state rather than synthetic fallback content.

Definition of Done
- ✓ Frontend pages call live backend endpoints.
- ✓ Analysis upload flow works end to end.
- ✓ Model status is displayed from the backend response.
- ✓ History and report pages render from persisted analysis data.

Acceptance Criteria
- No frontend page depends on unsupported demo-only data for its primary working path.
- The UI can display the response shape defined in the API contract without type or wire-format mismatches.

Verification Steps
1. Start the frontend.
2. Trigger analysis upload through the UI.
3. Poll the analysis response and verify the result page renders.
4. Confirm the model status and history components read from the live API output.

Expected Deliverables
- Live frontend-to-backend binding.
- Verified user workflow for upload, polling, and report view.

Gate 5
Frontend build starts
↓
Live API flow succeeds
↓
UI renders real results
↓
Continue

---

## Milestone 6 — Deployment & Edge

Goal
- Make the trained inference path deployment-ready and verify edge portability.

Scope
- Export the ensemble to ONNX.
- Apply quantization and benchmark the exported model.
- Verify deployment packaging for the backend and frontend.

Definition of Done
- ✓ ONNX artifact generated
- ✓ Benchmark artifact generated
- ✓ Backend deployment path is documented and runnable
- ✓ Edge model can be loaded and benchmarked

Acceptance Criteria
- `ensemble_quantized.onnx` exists and loads through ONNX Runtime.
- `onnx_benchmarks.json` reflects the host benchmark output.
- The deployment configuration remains aligned with the current product structure.

Verification Steps
1. Export the ONNX artifact.
2. Load the session with ONNX Runtime.
3. Run a benchmark smoke test.
4. Confirm benchmark output contains the expected latency and size entries.

Expected Deliverables
- Edge inference artifact.
- Host benchmark profile.
- Deployment configuration consistent with the current system design.

Gate 6
ONNX export succeeds
↓
Benchmark output is valid
↓
Deployment path is runnable
↓
Continue

---

## Milestone 7 — Quality Assurance & Release

Goal
- Bring the full stack to a releasable state with explicit verification evidence.

Scope
- Expand automated testing.
- Verify regression coverage for API, pipeline, and model loading.
- Confirm the release path is complete and auditable.

Definition of Done
- ✓ Automated test suite covers the critical runtime paths.
- ✓ API contract smoke checks pass.
- ✓ Pipeline verification is repeatable.
- ✓ Release summary can be produced from the merged documentation and evidence set.

Acceptance Criteria
- The core upload-to-report workflow is covered by verification.
- Regression risk is bounded by automated and manual release checks.
- The implementation remains aligned with the documented architecture and milestone order.

Verification Steps
1. Run backend tests.
2. Run contract and smoke checks across the main API endpoints.
3. Verify the pipeline trace and artifact lifecycle remain in sync with the working system.
4. Confirm all milestone gates have completed successfully.

Expected Deliverables
- Verified release checklist.
- Test evidence for critical runtime workflows.
- Final execution guide for the remainder of the project.

Gate 7
Regression checks pass
↓
Release evidence is complete
↓
Release candidate is ready

---

## Dependency Order Summary

Milestone 0 unlocks a reproducible development environment.
Milestone 1 unlocks the backend contract.
Milestone 2 unlocks the data and feature path.
Milestone 3 unlocks inferential model artifacts.
Milestone 4 unlocks calibrated ensemble and severity outputs.
Milestone 5 unlocks end-to-end product workflow.
Milestone 6 unlocks deployment and edge readiness.
Milestone 7 unlocks release confidence and operational closure.

This order is the governing implementation sequence for the remainder of the project.
