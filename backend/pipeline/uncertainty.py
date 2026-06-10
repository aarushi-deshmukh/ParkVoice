"""
Uncertainty estimation for calibrated screening outputs.
"""

from __future__ import annotations

from typing import Any, Dict

import numpy as np


def estimate_uncertainty(model_predictions: Dict[str, float], calibrated_probability: float, quality: Dict[str, Any]) -> Dict[str, Any]:
    individual = [
        float(v)
        for key, v in model_predictions.items()
        if key not in {"ensemble", "onnx"} and isinstance(v, (int, float))
    ]
    disagreement = float(np.std(individual)) if len(individual) > 1 else 0.0
    boundary_uncertainty = float(1.0 - abs(calibrated_probability - 0.5) * 2.0)
    quality_penalty = float(1.0 - quality.get("quality_score", 0.0))

    uncertainty_score = float(np.clip(
        0.40 * min(disagreement / 0.25, 1.0)
        + 0.35 * boundary_uncertainty
        + 0.25 * quality_penalty,
        0.0,
        1.0,
    ))
    confidence = float(np.clip(1.0 - uncertainty_score, 0.0, 1.0))

    warnings: list[str] = []
    if disagreement > 0.15:
        warnings.append("Model disagreement is elevated; interpret the screening score cautiously.")
    if boundary_uncertainty > 0.65:
        warnings.append("The calibrated risk score is close to the decision boundary.")
    if quality.get("quality_category") in {"Fair", "Poor"}:
        warnings.append("Audio quality reduces confidence in the acoustic biomarkers.")

    return {
        "confidence": round(confidence, 3),
        "uncertainty_score": round(uncertainty_score, 3),
        "warnings": warnings,
        "ensemble_disagreement": round(disagreement, 3),
        "calibration_confidence": round(1.0 - boundary_uncertainty, 3),
        "audio_quality_factor": round(float(quality.get("quality_score", 0.0)), 3),
    }
