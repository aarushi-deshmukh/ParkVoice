"""
Clinician-friendly biomarker explanations.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List


BIOMARKER_RANGES: Dict[str, Dict[str, Any]] = {
    "jitter_local": {"label": "Jitter", "healthy_range": "0.00-1.00%", "normal": (0.0, 1.0), "borderline": (1.0, 1.8), "message": "Elevated jitter may indicate reduced vocal fold stability."},
    "shimmer_local": {"label": "Shimmer", "healthy_range": "0.00-3.80%", "normal": (0.0, 3.8), "borderline": (3.8, 6.0), "message": "Elevated shimmer may indicate irregular vocal loudness control."},
    "hnr": {"label": "HNR", "healthy_range": "18-35 dB", "normal": (18.0, 35.0), "borderline": (12.0, 18.0), "inverse": True, "message": "Reduced HNR may indicate increased breathiness."},
    "nhr": {"label": "NHR", "healthy_range": "0.00-0.02", "normal": (0.0, 0.02), "borderline": (0.02, 0.05), "message": "Elevated NHR may indicate a noisier phonation signal."},
    "rpde": {"label": "RPDE", "healthy_range": "0.20-0.45", "normal": (0.20, 0.45), "borderline": (0.45, 0.60), "message": "Elevated RPDE may indicate less regular vocal dynamics."},
    "dfa": {"label": "DFA", "healthy_range": "0.50-0.75", "normal": (0.50, 0.75), "borderline": (0.75, 0.90), "message": "DFA outside the reference band may indicate altered signal complexity."},
    "ppe": {"label": "PPE", "healthy_range": "0.05-0.15", "normal": (0.05, 0.15), "borderline": (0.15, 0.25), "message": "Elevated PPE may indicate unstable pitch period control."},
    "pitch_cv": {"label": "Pitch Variability", "healthy_range": "0.00-0.12", "normal": (0.0, 0.12), "borderline": (0.12, 0.20), "message": "Elevated pitch variability may reflect reduced phonatory steadiness."},
    "d2": {"label": "Signal Complexity", "healthy_range": "1.40-3.70", "normal": (1.40, 3.70), "borderline": (0.90, 1.40), "inverse": True, "message": "Reduced signal complexity may reflect a less flexible vocal pattern."},
}


def biomarker_status(key: str, value: float) -> str:
    meta = BIOMARKER_RANGES[key]
    lo, hi = meta["normal"]
    if lo <= value <= hi:
        return "Normal"
    border_lo, border_hi = meta["borderline"]
    if meta.get("inverse"):
        return "Borderline" if border_lo <= value < border_hi else "Abnormal"
    return "Borderline" if border_lo < value <= border_hi else "Abnormal"


def build_biomarker_dashboard(features: Dict[str, float]) -> List[Dict[str, Any]]:
    rows = []
    for key, meta in BIOMARKER_RANGES.items():
        value = float(features.get(key, 0.0))
        status = biomarker_status(key, value)
        rows.append({
            "key": key,
            "name": meta["label"],
            "value": round(value, 4),
            "healthy_range": meta["healthy_range"],
            "status": status,
            "explanation": meta["message"],
        })
    return rows


def explain_top_biomarkers(feature_importance: Iterable[Dict[str, Any]], features: Dict[str, float], limit: int = 5) -> List[Dict[str, Any]]:
    insights = []
    for item in feature_importance:
        feature = item.get("feature")
        if feature not in BIOMARKER_RANGES:
            continue
        value = float(features.get(feature, 0.0))
        meta = BIOMARKER_RANGES[feature]
        insights.append({
            "feature": feature,
            "label": meta["label"],
            "value": round(value, 4),
            "status": biomarker_status(feature, value),
            "direction": item.get("direction"),
            "shap_value": item.get("shap_value"),
            "clinical_explanation": meta["message"],
        })
        if len(insights) >= limit:
            break
    return insights
