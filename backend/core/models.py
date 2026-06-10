"""
SQLAlchemy ORM Models — All database tables
"""
import uuid
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    String, Float, Integer, Boolean, DateTime, Text, ForeignKey, JSON
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from core.database import Base


def gen_uuid() -> str:
    return str(uuid.uuid4())


class Patient(Base):
    """Longitudinal patient profile."""
    __tablename__ = "patients"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    name: Mapped[str] = mapped_column(String(128))
    age: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    gender: Mapped[Optional[str]] = mapped_column(String(16), nullable=True)
    notes: Mapped[Optional[str]] = mapped_column(Text, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    analyses: Mapped[list["Analysis"]] = relationship(
        back_populates="patient", cascade="all, delete-orphan"
    )


class Analysis(Base):
    """Single voice analysis record."""
    __tablename__ = "analyses"

    id: Mapped[str] = mapped_column(String(36), primary_key=True, default=gen_uuid)
    patient_id: Mapped[Optional[str]] = mapped_column(
        String(36), ForeignKey("patients.id"), nullable=True
    )

    # File info
    filename: Mapped[str] = mapped_column(String(256))
    file_path: Mapped[str] = mapped_column(String(512))
    duration_seconds: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sample_rate: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    # Status
    status: Mapped[str] = mapped_column(String(32), default="pending")
    # pending | processing | complete | failed
    error_message: Mapped[Optional[str]] = mapped_column(Text, nullable=True)

    # ── Core Results ─────────────────────────────────────────────────
    # PD probability (0–1)
    pd_probability: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Risk tier: low | moderate | high | very_high
    risk_tier: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    # Confidence of the ensemble (0–1)
    confidence: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Confidence category: High | Moderate | Low
    confidence_category: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    uncertainty_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    uncertainty: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    quality: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # ── Severity ─────────────────────────────────────────────────────
    # UPDRS-equivalent severity score (0–108)
    severity_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Mild | Moderate | Severe | Healthy
    severity_tier: Mapped[Optional[str]] = mapped_column(String(32), nullable=True)
    # 95% Confidence Intervals for UPDRS score prediction
    severity_lower_ci: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    severity_upper_ci: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    predicted_updrs: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    lower_bound: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    upper_bound: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # Early-stage detection score (0–1)
    early_detection_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    # Progression risk (0–1)
    progression_risk: Mapped[Optional[float]] = mapped_column(Float, nullable=True)

    # ── Per-model Probabilities ───────────────────────────────────────
    model_predictions: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    # {"rf": 0.72, "xgb": 0.68, "lgbm": 0.74, "efficientnet_b0": 0.71}

    # ── Extracted Features ────────────────────────────────────────────
    features: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    # ── SHAP Explanation ──────────────────────────────────────────────
    shap_values: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    feature_importance: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    risk_breakdown: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    clinical_explanations: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)
    biomarkers: Mapped[Optional[list]] = mapped_column(JSON, nullable=True)

    # ── Temporal Progression (if patient has history) ─────────────────
    progression_prediction: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
    updated_at: Mapped[datetime] = mapped_column(
        DateTime, default=datetime.utcnow, onupdate=datetime.utcnow
    )

    patient: Mapped[Optional["Patient"]] = relationship(back_populates="analyses")


class ModelMetric(Base):
    """Stores trained model evaluation metrics."""
    __tablename__ = "model_metrics"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    model_name: Mapped[str] = mapped_column(String(64))
    dataset: Mapped[str] = mapped_column(String(128), default="uci_parkinson")
    accuracy: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    precision: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    recall: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    f1_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    roc_auc: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    sensitivity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    specificity: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    training_time_sec: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    params: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)
