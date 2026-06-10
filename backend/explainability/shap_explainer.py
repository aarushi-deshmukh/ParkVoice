"""
SHAP Explainability Engine
===========================
Generates SHAP explanations for all model types:
- TreeExplainer for RF / XGBoost / LightGBM
- Returns SHAP values, feature importance rankings, and plot data
"""

import logging
from typing import Any, Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

FEATURE_DISPLAY_NAMES = {
    "fo_mean": "Mean Pitch (Hz)",
    "fo_max": "Max Pitch (Hz)",
    "fo_min": "Min Pitch (Hz)",
    "jitter_local": "Jitter Local (%)",
    "jitter_abs": "Jitter Absolute",
    "jitter_rap": "Jitter RAP",
    "jitter_ppq5": "Jitter PPQ5",
    "jitter_ddp": "Jitter DDP",
    "shimmer_local": "Shimmer Local (%)",
    "shimmer_db": "Shimmer (dB)",
    "shimmer_apq3": "Shimmer APQ3",
    "shimmer_apq5": "Shimmer APQ5",
    "shimmer_apq11": "Shimmer APQ11",
    "shimmer_dda": "Shimmer DDA",
    "nhr": "Noise-Harmonics Ratio",
    "hnr": "Harmonics-Noise Ratio",
    "rpde": "RPDE (Recurrence Entropy)",
    "dfa": "DFA (Fractal Scaling)",
    "spread1": "Fundamental Spread 1",
    "spread2": "Fundamental Spread 2",
    "d2": "Correlation Dimension",
    "ppe": "Pitch Period Entropy",
}

FEATURE_NORMAL_RANGES = {
    "fo_mean": (160.0, 220.0),
    "jitter_local": (0.0, 0.004),
    "shimmer_local": (0.0, 0.02),
    "hnr": (18.0, 35.0),
    "nhr": (0.0, 0.02),
    "rpde": (0.2, 0.45),
    "dfa": (0.5, 0.75),
    "ppe": (0.05, 0.15),
}


def explain_prediction(
    model: Any,
    X: np.ndarray,
    feature_names: List[str],
    background_data: Optional[np.ndarray] = None,
    max_features: int = 22,
) -> Dict:
    """
    Generate SHAP explanation for a single prediction.

    Args:
        model: sklearn-compatible model (RF, XGB, LGBM)
        X: (1, n_features) sample to explain
        feature_names: ordered list of feature names
        background_data: reference dataset for KernelExplainer fallback

    Returns:
        Dict with shap_values, feature_importance, and waterfall_data
    """
    try:
        import shap
    except ImportError:
        logger.error("SHAP not installed. Run: pip install shap")
        return {}

    try:
        # TreeExplainer works for RF, XGB, LGBM
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        # For binary classifiers, shap_values may be a list [neg_class, pos_class]
        if isinstance(shap_values, list):
            sv = shap_values[1][0]  # SHAP values for positive (PD) class, first sample
        else:
            sv = shap_values[0]

        expected_value = float(
            explainer.expected_value[1]
            if isinstance(explainer.expected_value, (list, np.ndarray))
            else explainer.expected_value
        )

    except Exception as e:
        logger.warning(f"TreeExplainer failed ({e}). Using sampling-based explanation.")
        sv = _fallback_importance(model, X, feature_names)
        expected_value = 0.5

    sv = np.array(sv).flatten()[:len(feature_names)]

    # Build feature importance ranking
    abs_shap = np.abs(sv)
    ranked_idx = np.argsort(abs_shap)[::-1][:max_features]

    feature_importance = [
        {
            "feature": feature_names[i],
            "display_name": FEATURE_DISPLAY_NAMES.get(feature_names[i], feature_names[i]),
            "shap_value": round(float(sv[i]), 6),
            "abs_shap": round(float(abs_shap[i]), 6),
            "direction": "increases_risk" if sv[i] > 0 else "decreases_risk",
        }
        for i in ranked_idx
    ]

    # Waterfall data for visualization
    waterfall_data = {
        "expected_value": round(expected_value, 4),
        "prediction": round(float(expected_value + sv.sum()), 4),
        "features": [
            {
                "name": feature_names[i],
                "display_name": FEATURE_DISPLAY_NAMES.get(feature_names[i], feature_names[i]),
                "shap_value": round(float(sv[i]), 6),
                "feature_value": round(float(X[0, i]), 6) if X is not None else None,
            }
            for i in ranked_idx[:15]  # Top 15 for waterfall
        ],
    }

    return {
        "shap_values": {fn: round(float(s), 6) for fn, s in zip(feature_names, sv)},
        "expected_value": round(expected_value, 4),
        "feature_importance": feature_importance,
        "waterfall_data": waterfall_data,
    }


def _fallback_importance(
    model: Any, X: np.ndarray, feature_names: List[str]
) -> np.ndarray:
    """
    Fallback: estimate feature importance via perturbation
    when SHAP TreeExplainer is not available.
    """
    baseline_prob = model.predict_proba(X)[0, 1]
    n_features = X.shape[1]
    importances = np.zeros(n_features)

    for i in range(n_features):
        X_perturbed = X.copy()
        X_perturbed[0, i] = 0.0
        perturbed_prob = model.predict_proba(X_perturbed)[0, 1]
        importances[i] = baseline_prob - perturbed_prob

    return importances
