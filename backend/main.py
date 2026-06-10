"""
ParkVoice AI — FastAPI Application Entry Point
===============================================
Research-grade Parkinson's Disease acoustic screening support platform.
"""

import os
import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from sqlalchemy import text
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import JSONResponse

from core.config import settings
from core.database import engine, Base
from api.endpoints import analysis, patients, models_endpoint

# ── Logging ──────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("parkvoice")


# ── Lifespan ─────────────────────────────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup and shutdown lifecycle management."""
    logger.info("🧠 ParkVoice AI starting up...")

    # Create all database tables
    async with engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)
        if settings.DATABASE_URL.startswith("sqlite"):
            existing = await conn.execute(text("PRAGMA table_info(analyses)"))
            columns = {row[1] for row in existing.fetchall()}
            for name, ddl in {
                "uncertainty_score": "FLOAT",
                "uncertainty": "JSON",
                "quality": "JSON",
                "predicted_updrs": "FLOAT",
                "lower_bound": "FLOAT",
                "upper_bound": "FLOAT",
                "clinical_explanations": "JSON",
                "biomarkers": "JSON",
            }.items():
                if name not in columns:
                    await conn.execute(text(f"ALTER TABLE analyses ADD COLUMN {name} {ddl}"))

    # Ensure required directories exist
    for directory in [settings.UPLOAD_DIR, settings.MODEL_DIR]:
        os.makedirs(directory, exist_ok=True)
        logger.info(f"  ✓ Directory ready: {directory}")

    logger.info("✅ ParkVoice AI ready for inference.")
    yield
    logger.info("🛑 ParkVoice AI shutting down.")


# ── Application ───────────────────────────────────────────────────────────────
app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description=(
        "Research-grade AI platform for Parkinson's Disease risk assessment "
        "from voice recordings with SHAP explainability, severity scoring, "
        "and longitudinal patient tracking."
    ),
    docs_url="/api/docs",
    redoc_url="/api/redoc",
    lifespan=lifespan,
)

# ── CORS ──────────────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Routers ───────────────────────────────────────────────────────────────────
app.include_router(analysis.router,  prefix="/api/analysis",  tags=["Analysis"])
app.include_router(patients.router,  prefix="/api/patients",  tags=["Patients"])
app.include_router(models_endpoint.router, prefix="/api/models", tags=["Models"])

# ── Static Uploads ────────────────────────────────────────────────────────────
os.makedirs(settings.UPLOAD_DIR, exist_ok=True)
app.mount("/uploads", StaticFiles(directory=settings.UPLOAD_DIR), name="uploads")


# ── Health Check ──────────────────────────────────────────────────────────────
@app.get("/api/health", tags=["System"])
async def health():
    return {
        "status": "healthy",
        "app": settings.APP_NAME,
        "version": settings.APP_VERSION,
    }


@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    logger.error(f"Unhandled exception: {exc}", exc_info=True)
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error. Please check server logs."},
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=settings.DEBUG,
        log_level="info",
    )
