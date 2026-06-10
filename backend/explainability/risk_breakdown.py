"""
Risk Factor Breakdown
======================
Maps SHAP values to clinically meaningful risk categories.
Groups features into biomarker clusters and computes aggregate risk per group.
"""

from typing import Any, Dict, List

import numpy as np

# Clinical feature groupings
RISK_GROUPS = {
    "pitch_instability": {
        "display": "Pitch Instability",
        "features": ["fo_mean", "fo_max", "fo_min", "ppe", "spread1", "spread2", "d2"],
        "description": "Measures of fundamental frequency variability and pitch control",
        "icon": "wave",
        "clinical_note": "Reduced pitch range and increased pitch entropy are hallmark early PD markers.",
    },
    "jitter": {
        "display": "Vocal Tremor (Jitter)",
        "features": ["jitter_local", "jitter_abs", "jitter_rap", "jitter_ppq5", "jitter_ddp"],
        "description": "Cycle-to-cycle frequency perturbations indicating laryngeal tremor",
        "icon": "activity",
        "clinical_note": "Elevated jitter reflects neuromotor control deficits in the larynx.",
    },
    "shimmer": {
        "display": "Amplitude Instability (Shimmer)",
        "features": ["shimmer_local", "shimmer_db", "shimmer_apq3", "shimmer_apq5",
                     "shimmer_apq11", "shimmer_dda"],
        "description": "Cycle-to-cycle amplitude perturbations indicating vocal fold irregularity",
        "icon": "bar-chart-2",
        "clinical_note": "Increased shimmer correlates with reduced glottal closure efficiency.",
    },
    "noise_ratio": {
        "display": "Voice Noise Ratio",
        "features": ["nhr", "hnr"],
        "description": "Signal-to-noise ratio of the voice — turbulent vs. harmonic components",
        "icon": "radio",
        "clinical_note": "Higher NHR / lower HNR indicates increased vocal fold irregularity.",
    },
    "nonlinear_dynamics": {
        "display": "Signal Complexity",
        "features": ["rpde", "dfa"],
        "description": "Nonlinear dynamical measures of vocal signal complexity",
        "icon": "cpu",
        "clinical_note": "Abnormal RPDE and DFA values suggest loss of physiological complexity.",
    },
}

NORMAL_RANGES = {
    "fo_mean": (140.0, 220.0, "Hz"),
    "jitter_local": (0.0, 0.004, "%"),
    "shimmer_local": (0.0, 0.02, "%"),
    "hnr": (18.0, 35.0, "dB"),
    "nhr": (0.0, 0.02, ""),
    "rpde": (0.2, 0.46, ""),
    "dfa": (0.5, 0.74, ""),
    "ppe": (0.05, 0.15, ""),
}


def build_risk_breakdown(
    shap_result: Dict,
    features_raw: Dict[str, float],
) -> Dict[str, Any]:
    """
    Build structured clinical risk breakdown from SHAP values.

    Args:
        shap_result: Output from explain_prediction()
        features_raw: Raw (unnormalized) feature values

    Returns:
        Dict mapping risk groups → aggregate risk, feature details, alerts
    """
    shap_values = shap_result.get("shap_values", {})

    breakdown = {}
    total_positive_shap = sum(v for v in shap_values.values() if v > 0) + 1e-10

    for group_key, group_meta in RISK_GROUPS.items():
        group_features = group_meta["features"]
        group_shap_vals = {f: shap_values.get(f, 0.0) for f in group_features}

        # Aggregate risk: sum of positive SHAP contributions from this group
        group_risk_shap = sum(v for v in group_shap_vals.values() if v > 0)
        group_risk_score = float(np.clip(group_risk_shap / total_positive_shap, 0, 1))

        # Per-feature details
        feature_details = []
        for feat in group_features:
            raw_val = features_raw.get(feat)
            shap_val = shap_values.get(feat, 0.0)

            detail = {
                "feature": feat,
                "display_name": _display_name(feat),
                "value": round(float(raw_val), 5) if raw_val is not None else None,
                "shap_contribution": round(float(shap_val), 5),
                "direction": "risk" if shap_val > 0 else "protective",
                "status": _range_status(feat, raw_val),
            }
            feature_details.append(detail)

        # Alerts for abnormal features
        alerts = [
            d["display_name"]
            for d in feature_details
            if d["status"] == "abnormal" and d["value"] is not None
        ]

        breakdown[group_key] = {
            "display": group_meta["display"],
            "description": group_meta["description"],
            "clinical_note": group_meta["clinical_note"],
            "icon": group_meta["icon"],
            "risk_score": round(group_risk_score, 4),
            "risk_level": _risk_level(group_risk_score),
            "features": feature_details,
            "alerts": alerts,
        }

    return breakdown


def _display_name(feature: str) -> str:
    from explainability.shap_explainer import FEATURE_DISPLAY_NAMES
    return FEATURE_DISPLAY_NAMES.get(feature, feature)


def _range_status(feature: str, value: float | None) -> str:
    """Returns 'normal', 'borderline', 'abnormal', or 'unknown'."""
    if value is None or feature not in NORMAL_RANGES:
        return "unknown"
    lo, hi, _ = NORMAL_RANGES[feature]
    if lo <= value <= hi:
        return "normal"
    margin = (hi - lo) * 0.2
    if (lo - margin) <= value <= (hi + margin):
        return "borderline"
    return "abnormal"


def _risk_level(score: float) -> str:
    if score < 0.2:
        return "low"
    elif score < 0.45:
        return "moderate"
    elif score < 0.70:
        return "high"
    return "very_high"


def compute_group_radar_data(breakdown: Dict[str, Any]) -> List[Dict]:
    """Format breakdown for radar chart visualization on frontend."""
    return [
        {
            "group": meta["display"],
            "risk": round(meta["risk_score"] * 100, 1),  # 0–100 scale
            "level": meta["risk_level"],
        }
        for meta in breakdown.values()
    ]
