"""
Patient Management Endpoints
=============================
CRUD for patient profiles and longitudinal tracking.
"""

from datetime import datetime
from typing import List, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy import select, func

from core.database import get_db
from core.models import Patient, Analysis

router = APIRouter()


class PatientCreate(BaseModel):
    name: str
    age: Optional[int] = None
    gender: Optional[str] = None
    notes: Optional[str] = None


class PatientUpdate(BaseModel):
    name: Optional[str] = None
    age: Optional[int] = None
    gender: Optional[str] = None
    notes: Optional[str] = None


class PatientResponse(BaseModel):
    id: str
    name: str
    age: Optional[int]
    gender: Optional[str]
    notes: Optional[str]
    created_at: datetime
    analysis_count: Optional[int] = 0
    latest_risk_tier: Optional[str] = None
    latest_pd_probability: Optional[float] = None

    class Config:
        from_attributes = True


@router.post("/", response_model=PatientResponse, status_code=201)
async def create_patient(data: PatientCreate, db: AsyncSession = Depends(get_db)):
    """Create a new patient profile."""
    patient = Patient(**data.model_dump())
    db.add(patient)
    await db.commit()
    await db.refresh(patient)
    return PatientResponse(
        id=patient.id, name=patient.name, age=patient.age,
        gender=patient.gender, notes=patient.notes,
        created_at=patient.created_at,
    )


@router.get("/", response_model=List[PatientResponse])
async def list_patients(skip: int = 0, limit: int = 50, db: AsyncSession = Depends(get_db)):
    """List all patients with summary stats."""
    result = await db.execute(
        select(Patient).order_by(Patient.created_at.desc()).offset(skip).limit(limit)
    )
    patients = result.scalars().all()
    responses = []
    for p in patients:
        # Get latest analysis
        anal_result = await db.execute(
            select(Analysis)
            .where(Analysis.patient_id == p.id)
            .order_by(Analysis.created_at.desc())
            .limit(1)
        )
        latest = anal_result.scalar_one_or_none()

        # Count analyses
        count_result = await db.execute(
            select(func.count()).where(Analysis.patient_id == p.id)
        )
        count = count_result.scalar() or 0

        responses.append(PatientResponse(
            id=p.id, name=p.name, age=p.age,
            gender=p.gender, notes=p.notes,
            created_at=p.created_at,
            analysis_count=count,
            latest_risk_tier=latest.risk_tier if latest else None,
            latest_pd_probability=latest.pd_probability if latest else None,
        ))
    return responses


@router.get("/{patient_id}", response_model=PatientResponse)
async def get_patient(patient_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    return patient


@router.put("/{patient_id}", response_model=PatientResponse)
async def update_patient(
    patient_id: str, data: PatientUpdate, db: AsyncSession = Depends(get_db)
):
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    for key, val in data.model_dump(exclude_none=True).items():
        setattr(patient, key, val)
    await db.commit()
    await db.refresh(patient)
    return patient


@router.delete("/{patient_id}", status_code=204)
async def delete_patient(patient_id: str, db: AsyncSession = Depends(get_db)):
    result = await db.execute(select(Patient).where(Patient.id == patient_id))
    patient = result.scalar_one_or_none()
    if not patient:
        raise HTTPException(status_code=404, detail="Patient not found")
    await db.delete(patient)
    await db.commit()


@router.get("/{patient_id}/history")
async def get_patient_history(patient_id: str, db: AsyncSession = Depends(get_db)):
    """
    Get complete analysis history for a patient.
    Returns time-series of PD probability, severity, and key biomarkers.
    """
    result = await db.execute(
        select(Analysis)
        .where(Analysis.patient_id == patient_id)
        .where(Analysis.status == "complete")
        .order_by(Analysis.created_at.asc())
    )
    analyses = result.scalars().all()

    if not analyses:
        return {"patient_id": patient_id, "recordings": [], "trends": None}

    # Build time-series
    recordings = []
    pd_probabilities = []
    severity_scores = []

    for a in analyses:
        recordings.append({
            "id": a.id,
            "date": a.created_at.isoformat(),
            "filename": a.filename,
            "pd_probability": a.pd_probability,
            "risk_tier": a.risk_tier,
            "severity_score": a.severity_score,
            "severity_tier": a.severity_tier,
            "severity_lower_ci": a.severity_lower_ci,
            "severity_upper_ci": a.severity_upper_ci,
        })
        if a.pd_probability is not None:
            pd_probabilities.append(a.pd_probability)
        if a.severity_score is not None:
            severity_scores.append(a.severity_score)

    # Longitudinal trends
    import numpy as np

    def _trend(values):
        if len(values) < 2:
            return "insufficient_data"
        delta = values[-1] - values[0]
        if abs(delta) < 0.05:
            return "stable"
        return "increasing" if delta > 0 else "decreasing"

    # Progression prediction from temporal model (if enough recordings)
    progression = None
    if len(analyses) >= 3:
        try:
            from research.experimental.temporal_progression import predict_progression
            from pipeline.feature_extractor import get_clinical_features
            feat_sequence = [
                get_clinical_features(a.features) for a in analyses if a.features
            ]
            if len(feat_sequence) >= 3:
                progression = predict_progression(feat_sequence)
        except Exception as e:
            pass

    return {
        "patient_id": patient_id,
        "recordings": recordings,
        "trends": {
            "pd_probability": _trend(pd_probabilities),
            "severity": _trend(severity_scores),
            "n_recordings": len(analyses),
        },
        "progression_prediction": progression,
    }


@router.get("/{patient_id}/trends")
async def get_patient_trends(patient_id: str, db: AsyncSession = Depends(get_db)):
    """Get visualization-ready trend data for charts."""
    result = await db.execute(
        select(Analysis)
        .where(Analysis.patient_id == patient_id)
        .where(Analysis.status == "complete")
        .order_by(Analysis.created_at.asc())
    )
    analyses = result.scalars().all()

    trend_data = {
        "dates": [a.created_at.isoformat() for a in analyses],
        "pd_probability": [a.pd_probability for a in analyses],
        "severity_score": [a.severity_score for a in analyses],
        "early_detection_score": [a.early_detection_score for a in analyses],
        "progression_risk": [a.progression_risk for a in analyses],
    }

    # Extract key biomarkers over time
    biomarkers = ["fo_mean", "jitter_local", "shimmer_local", "hnr", "ppe"]
    for bm in biomarkers:
        trend_data[bm] = [
            a.features.get(bm) if a.features else None for a in analyses
        ]

    return trend_data
