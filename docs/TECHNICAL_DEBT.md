# ParkVoice AI — Technical Debt & Code Defect Audit

This document cataloging broken imports, syntax/control-flow defects, dead code, duplicate logic, fallback mocks, and architectural debt across the codebase.

---

## 1. Syntax & Control Flow Defects

### Defect 1: Broken Control Flow in `backend/api/endpoints/models_endpoint.py`

**Location**: [backend/api/endpoints/models_endpoint.py:L31-L64](file:///d:/Projects/parkinson/backend/api/endpoints/models_endpoint.py#L31-L64)

```python
# ── Existing Defective Code ──
@router.get("/status")
async def model_status():
    """Check which models are available on disk."""
    models = {
        "random_forest": os.path.exists(os.path.join(MODEL_DIR, "rf_model.pkl")),
        "xgboost": os.path.exists(os.path.join(MODEL_DIR, "xgb_model.pkl")),
        "lightgbm": os.path.exists(os.path.join(MODEL_DIR, "lgbm_model.pkl")),
        "ensemble": os.path.exists(os.path.join(MODEL_DIR, "ensemble_model.pkl")),
        "efficientnet_b0": os.path.exists(os.path.join(MODEL_DIR, "cnn_model.pt")),
        "severity": os.path.exists(os.path.join(MODEL_DIR, "severity_model.pkl")),
        "onnx": os.path.exists(os.path.join(MODEL_DIR, "ensemble_quantized.onnx")),
    }
    # MISSING RETURN STATEMENT HERE!

@router.get("/registry")
async def get_model_registry():
    """Return the production model registry metadata."""
    registry = _load_json(REGISTRY_PATH)
    if registry is None:
        raise HTTPException(status_code=404, detail="model_registry.json not found")
    return registry
    # UNREACHABLE DEAD CODE BELOW:
    scaler_ready = os.path.exists(os.path.join(MODEL_DIR, "tabular_scaler.pkl"))
    return {
        "models": models,
        "scaler_ready": scaler_ready,
        "any_model_ready": any(models.values()),
        "full_pipeline_ready": all([
            models["ensemble"] or models["random_forest"],
            scaler_ready,
        ]),
    }
```

**Impact**: 
1. `GET /api/models/status` returns `null` or `200 OK` with an empty body because `model_status()` completes without a return statement.
2. `get_model_registry()` returns `registry` on line 51, rendering lines 52–62 completely unreachable.
3. The frontend `fetchModelStatus()` call in Zustand store receives invalid/empty data, reporting `full_pipeline_ready: false`.

---

## 2. Dead & Unused Code

1. **Unused Frontend Stylesheet**:
   - File: [frontend/src/App.css](file:///d:/Projects/parkinson/frontend/src/App.css)
   - Reason: Contains legacy template CSS from Vite initialization. All active styling is driven by [frontend/src/index.css](file:///d:/Projects/parkinson/frontend/src/index.css) (20,715 bytes).
2. **Experimental Research Sandbox**:
   - File: [backend/research/experimental/temporal_progression.py](file:///d:/Projects/parkinson/backend/research/experimental/temporal_progression.py)
   - Reason: Located under `research/experimental/`. While imported by `patients.py` for `/history`, it is explicitly disabled unless $\ge 3$ historical recordings exist and trained weights exist.

---

## 3. Mock Data & Synthetic Fallbacks

| File | Component | Fallback Behavior | Operational Risk |
| :--- | :--- | :--- | :--- |
| `dataset_loader.py` | `load_uci_dataset()` | If `ucimlrepo` download fails, generates 240 synthetic rows via `_generate_synthetic_dataset()`. | Model trained on synthetic distribution rather than real patient data. |
| `dataset_loader.py` | `add_synthetic_severity()` | Generates fake UPDRS scores ($0\text{--}108$) using an empirical formula. | Severity regressor is trained on synthetic regression targets. |
| `train_cnn.py` | `SyntheticSpectrogramDataset` | Synthesizes fake 224x224 RGB Mel-spectrogram images from 22 tabular numbers. | CNN cannot learn real speech acoustic spectrogram textures. |
| `Overview.tsx` | `demoAnalysis` | If the API call fails or returns empty, renders hardcoded fallback object `demoAnalysis`. | Frontend masks backend failures by displaying demo mock data. |
| `dataset_report.json` | Root Config | Hardcoded to 0 subjects, 0 recordings, all features disabled. | Out of sync with actual codebase capabilities. |
| `model_registry.json` | Root Metadata | Hardcoded with `metrics: {}` and `onnx_status: "pending"`. | Out of sync until training pipeline runs. |

---

## 4. Subprocess Execution & Process Safety Risks

In [backend/api/endpoints/models_endpoint.py](file:///d:/Projects/parkinson/backend/api/endpoints/models_endpoint.py#L185), the `POST /train` endpoint triggers training scripts asynchronously using `subprocess.Popen`:

```python
for script in scripts:
    if os.path.exists(script):
        proc = subprocess.Popen([sys.executable, script], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        results[script] = {"pid": proc.pid, "status": "started"}
```

**Architectural Issues**:
1. Processes are spawned without tracking completion, exit codes, or failure logs.
2. Multiple sequential calls to `POST /train` will spawn concurrent competing training jobs on the CPU/GPU, causing memory corruption or resource exhaustion.
3. No background task queue (such as Celery, ARQ, or FastAPI BackgroundTasks with status tracking) is utilized.
