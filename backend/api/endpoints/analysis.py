"""
Analysis API Endpoints
=======================
Handles audio upload, async analysis processing, result retrieval,
and SHAP explanation endpoints.

DISCLAIMER:
  "This system is intended for research and screening support purposes only
  and is not a diagnostic medical device."
"""

import os
import uuid
import logging
import asyncio
from datetime import datetime
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, UploadFile, File, Form, Depends, HTTPException, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select

from core.config import settings
from core.database import get_db
from core.models import Analysis, Patient

logger = logging.getLogger(__name__)
router = APIRouter()


# ── Pydantic Schemas ───────────────────────────────────────────────────────────

SYSTEM_DISCLAIMER = (
    "This system is intended for research and screening support purposes only "
    "and is not a diagnostic medical device."
)


class AnalysisResponse(BaseModel):
    id: str
    status: str
    filename: str
    created_at: datetime
    # ── Calibrated risk assessment ────────────────────────────────────────────
    pd_probability: Optional[float] = None
    risk_tier: Optional[str] = None              # low | moderate | high | very_high
    confidence: Optional[float] = None           # numeric 0–1 for backward compat
    confidence_category: Optional[str] = None   # High | Moderate | Low
    uncertainty_score: Optional[float] = None
    uncertainty: Optional[dict] = None
    # ── UPDRS severity regression with 95% CI ─────────────────────────────────
    severity_score: Optional[float] = None
    predicted_updrs: Optional[float] = None
    severity_tier: Optional[str] = None
    severity_lower_ci: Optional[float] = None
    severity_upper_ci: Optional[float] = None
    lower_bound: Optional[float] = None
    upper_bound: Optional[float] = None
    # ── Model outputs ──────────────────────────────────────────────────────────
    model_predictions: Optional[dict] = None
    features: Optional[dict] = None
    shap_values: Optional[dict] = None
    feature_importance: Optional[list] = None
    risk_breakdown: Optional[dict] = None
    clinical_explanations: Optional[list] = None
    biomarkers: Optional[list] = None
    quality: Optional[dict] = None
    error_message: Optional[str] = None
    # ── Regulatory & research context ─────────────────────────────────────────
    disclaimer: str = SYSTEM_DISCLAIMER

    class Config:
        from_attributes = True


# ── Background Analysis Task ───────────────────────────────────────────────────

async def _run_analysis_background(analysis_id: str, file_path: str):
    """Run the full inference pipeline in a background task."""
    from core.database import AsyncSessionLocal
    from pipeline.pipeline import run_inference

    logger.info(f"Background analysis started: {analysis_id}")
    async with AsyncSessionLocal() as db:
        try:
            # Update status to processing
            result = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
            analysis = result.scalar_one_or_none()
            if not analysis:
                return
            analysis.status = "processing"
            await db.commit()

            # Run inference in thread pool (CPU-bound)
            loop = asyncio.get_event_loop()
            inference_result = await loop.run_in_executor(
                None, lambda: run_inference(file_path, include_shap=True)
            )

            # Save results
            await db.execute(select(Analysis).where(Analysis.id == analysis_id))
            result = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
            analysis = result.scalar_one_or_none()

            if inference_result.get("status") == "failed":
                analysis.status = "failed"
                analysis.error_message = inference_result.get("error", "Unknown error")
            else:
                analysis.status = "complete"
                analysis.pd_probability = inference_result.get("pd_probability")
                analysis.risk_tier = inference_result.get("risk_tier")
                analysis.confidence = inference_result.get("confidence")
                analysis.confidence_category = inference_result.get("confidence_category")
                analysis.uncertainty_score = inference_result.get("uncertainty_score")
                analysis.uncertainty = inference_result.get("uncertainty")
                analysis.severity_score = inference_result.get("severity_score")
                analysis.predicted_updrs = inference_result.get("predicted_updrs")
                analysis.severity_tier = inference_result.get("severity_tier")
                analysis.severity_lower_ci = inference_result.get("severity_lower_ci")
                analysis.severity_upper_ci = inference_result.get("severity_upper_ci")
                analysis.lower_bound = inference_result.get("lower_bound")
                analysis.upper_bound = inference_result.get("upper_bound")
                analysis.model_predictions = inference_result.get("model_predictions")
                analysis.features = inference_result.get("features")
                analysis.shap_values = inference_result.get("shap_values")
                analysis.feature_importance = inference_result.get("feature_importance")
                analysis.risk_breakdown = inference_result.get("risk_breakdown")
                analysis.clinical_explanations = inference_result.get("clinical_explanations")
                analysis.biomarkers = inference_result.get("biomarkers")
                analysis.quality = inference_result.get("quality")

            await db.commit()
            logger.info(f"Analysis complete: {analysis_id} → {analysis.status}")

        except Exception as e:
            logger.error(f"Analysis failed for {analysis_id}: {e}", exc_info=True)
            try:
                result = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
                analysis = result.scalar_one_or_none()
                if analysis:
                    analysis.status = "failed"
                    analysis.error_message = str(e)
                    await db.commit()
            except Exception:
                pass


# ── Endpoints ─────────────────────────────────────────────────────────────────

@router.post("/upload", response_model=AnalysisResponse, status_code=202)
async def upload_and_analyze(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
    patient_id: Optional[str] = Form(None),
    db: AsyncSession = Depends(get_db),
):
    """
    Upload an audio file and trigger async Parkinson's risk assessment.
    Returns immediately with analysis ID; poll GET /analysis/{id} for results.
    """
    # Validate format
    ext = Path(file.filename).suffix.lower().lstrip(".")
    if ext not in settings.ALLOWED_AUDIO_FORMATS:
        raise HTTPException(
            status_code=415,
            detail=f"Unsupported format '{ext}'. Allowed: {settings.ALLOWED_AUDIO_FORMATS}",
        )

    # Validate size
    content = await file.read()
    size_mb = len(content) / (1024 * 1024)
    if size_mb > settings.MAX_UPLOAD_SIZE_MB:
        raise HTTPException(
            status_code=413,
            detail=f"File too large ({size_mb:.1f} MB). Max: {settings.MAX_UPLOAD_SIZE_MB} MB",
        )

    # Save to disk
    analysis_id = str(uuid.uuid4())
    filename = f"{analysis_id}_{file.filename}"
    file_path = os.path.join(settings.UPLOAD_DIR, filename)
    os.makedirs(settings.UPLOAD_DIR, exist_ok=True)

    with open(file_path, "wb") as f:
        f.write(content)

    # Validate patient if provided
    if patient_id:
        result = await db.execute(select(Patient).where(Patient.id == patient_id))
        patient = result.scalar_one_or_none()
        if not patient:
            raise HTTPException(status_code=404, detail="Patient not found")

    # Create analysis record
    analysis = Analysis(
        id=analysis_id,
        patient_id=patient_id,
        filename=file.filename,
        file_path=file_path,
        status="pending",
    )
    db.add(analysis)
    await db.commit()
    await db.refresh(analysis)

    # Queue background processing
    background_tasks.add_task(_run_analysis_background, analysis_id, file_path)

    logger.info(f"Analysis queued: {analysis_id} ({file.filename})")
    return analysis


@router.get("/{analysis_id}", response_model=AnalysisResponse)
async def get_analysis(analysis_id: str, db: AsyncSession = Depends(get_db)):
    """Retrieve analysis results by ID. Poll until status == 'complete'."""
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    return analysis


@router.get("/{analysis_id}/explain")
async def get_explanation(analysis_id: str, db: AsyncSession = Depends(get_db)):
    """Get detailed SHAP explanation for a completed analysis."""
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")
    if analysis.status != "complete":
        raise HTTPException(status_code=422, detail=f"Analysis is {analysis.status}, not complete")

    from explainability.risk_breakdown import compute_group_radar_data
    radar = compute_group_radar_data(analysis.risk_breakdown) if analysis.risk_breakdown else []

    return {
        "analysis_id": analysis_id,
        "feature_importance": analysis.feature_importance or [],
        "shap_values": analysis.shap_values or {},
        "risk_breakdown": analysis.risk_breakdown or {},
        "radar_data": radar,
    }


@router.delete("/{analysis_id}", status_code=204)
async def delete_analysis(analysis_id: str, db: AsyncSession = Depends(get_db)):
    """Delete an analysis record and its uploaded file."""
    result = await db.execute(select(Analysis).where(Analysis.id == analysis_id))
    analysis = result.scalar_one_or_none()
    if not analysis:
        raise HTTPException(status_code=404, detail="Analysis not found")

    # Remove file
    if os.path.exists(analysis.file_path):
        os.remove(analysis.file_path)

    await db.delete(analysis)
    await db.commit()


@router.get("/")
async def list_analyses(
    skip: int = 0, limit: int = 20,
    patient_id: Optional[str] = None,
    db: AsyncSession = Depends(get_db),
):
    """List recent analyses with optional patient filter."""
    query = select(Analysis).order_by(Analysis.created_at.desc()).offset(skip).limit(limit)
    if patient_id:
        query = query.where(Analysis.patient_id == patient_id)
    result = await db.execute(query)
    analyses = result.scalars().all()
    return {
        "items": [AnalysisResponse.model_validate(a) for a in analyses],
        "total": len(analyses),
    }
