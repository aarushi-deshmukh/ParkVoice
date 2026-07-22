# Environment Validation Guide

This document is the authoritative environment validation record for the development workflow. Its purpose is to confirm that the repository can be developed from a clean clone without ambiguity.

## Required Versions

- Python version:
- Node version:
- npm version:
- Docker version:

## Package Installation

Expected commands:

```bash
python -m venv .venv
. .venv/bin/activate
pip install -r backend/requirements.txt
cd frontend
npm install
```

Startup note:
- The backend uses `sqlite+aiosqlite` for the default SQLite configuration, so `aiosqlite` must be present in the backend environment.
- If `ModuleNotFoundError: No module named 'aiosqlite'` occurs during startup, reinstall the backend requirements in the active environment.

Expected output:
- The virtual environment is created successfully.
- Backend dependencies install without unresolved package errors.
- Frontend dependencies install without unresolved package errors.

## Backend Startup

Expected command:

```bash
cd backend
python -m uvicorn main:app --reload
```

Expected output:
- Application startup completes.
- API routes are available.
- Import resolution succeeds.

## Frontend Startup

Expected command:

```bash
cd frontend
npm run build
```

Expected output:
- Vite build completes successfully.
- Static assets are generated.

## Database Initialization

Expected command:

```bash
python -c "from backend.core.database import init_db; init_db()"
```

Expected output:
- Database file or schema is initialized successfully.
- No startup-time initialization errors occur.

## Docker Validation

Expected command:

```bash
docker compose build
```

Expected output:
- Both backend and frontend containers build successfully.
- No missing-file or dependency-level build failure occurs.

## Pytest Execution

Expected command:

```bash
python -m pytest backend/tests -q
```

Expected output:
- Test collection succeeds.
- Test execution begins with a clean import path.
- The suite reports pass/fail without collection-time import failures.

Configuration note:
- The repository uses a minimal `pytest.ini` with `pythonpath = backend` so `explainability` and `pipeline` imports resolve correctly when running the test entrypoint from the repository root.

## Import Validation

Expected checks:
- `from explainability.clinical_rules import build_biomarker_dashboard`
- `from pipeline.audio_quality import assess_audio_quality`
- `from pipeline.uncertainty import estimate_uncertainty`

Expected output:
- All imports resolve from the active environment.
- No `ModuleNotFoundError` occurs during collection.

## Environment Variables

Expected variables:
- `DATABASE_URL`
- `SECRET_KEY`
- `ALLOWED_AUDIO_FORMATS`
- `MAX_UPLOAD_SIZE_MB`

Expected output:
- Required configuration values are defined.
- Backend startup does not fail due to missing environment configuration.

## Common Failures

- `ModuleNotFoundError` for package imports
- Missing Python interpreter or virtual environment
- Node dependency install failure
- Docker build errors caused by missing context or invalid paths
- Database initialization or migration errors
- Missing or misnamed environment variables

## Resolution Steps

1. Recreate the virtual environment.
2. Reinstall requirements from the documented lock or requirements file.
3. Re-run backend startup with the active environment.
4. Resolve import path issues before continuing to milestone execution.
5. Re-run Docker build after dependency installation and configuration updates.
6. Confirm database initialization with the current project structure.
7. Only then proceed to the next milestone.
