"""
Master Inference Pipeline
=========================
Orchestrates: audio load → preprocess → feature extract → normalize → predict.

DISCLAIMER:
  "This system is intended for research and screening support purposes only
  and is not a diagnostic medical device."

Phased model architecture:
  1. Classical ensemble (RF, XGBoost, LightGBM) — calibrated probability
  2. EfficientNet-B0 CNN spectrogram classifier
  3. ONNX quantized edge fallback
  4. Severity estimation (UPDRS regression with 95% CI)
  5. SHAP explainability
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict

import numpy as np

from pipeline.audio_preprocessor import preprocess_pipeline
from pipeline.uncertainty import estimate_uncertainty
from pipeline.feature_extractor import (
    extract_all_features,
    extract_mel_spectrogram,
    get_clinical_features,
)
from pipeline.normalization import tabular_normalizer, spectrogram_normalizer

logger = logging.getLogger(__name__)

SYSTEM_DISCLAIMER = (
    "This system is intended for research and screening support purposes only "
    "and is not a diagnostic medical device."
)

# Lazy-loaded model registry (populated on first call)
_model_registry: Dict[str, Any] = {}


def _load_models() -> Dict[str, Any]:
    """Lazily load all serialized models."""
    global _model_registry
    if _model_registry:
        return _model_registry

    import pickle
    from core.config import settings

    registry: Dict[str, Any] = {}

    # Classical calibrated models (pkl)
    for name, path in [
        ("rf",       "./models/rf_model.pkl"),
        ("xgb",      "./models/xgb_model.pkl"),
        ("lgbm",     "./models/lgbm_model.pkl"),
        ("ensemble", "./models/ensemble_model.pkl"),
    ]:
        if os.path.exists(path):
            with open(path, "rb") as f:
                registry[name] = pickle.load(f)
            logger.info(f"  ✓ Loaded {name} (calibrated)")
        else:
            logger.warning(f"  ✗ {name} not found at {path} — run training first")

    # ONNX quantized edge model
    onnx_path = "./models/ensemble_quantized.onnx"
    if os.path.exists(onnx_path):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.log_severity_level = 3
        registry["onnx"] = ort.InferenceSession(onnx_path, sess_options=opts)
        logger.info("  ✓ Loaded ONNX edge model")

    # EfficientNet-B0 CNN spectrogram classifier
    cnn_path = "./models/cnn_model.pt"
    if os.path.exists(cnn_path):
        try:
            import torch
            device = settings.effective_device
            from ml.train_cnn import ParkVoiceCNN
            model = ParkVoiceCNN(num_classes=2)
            model.load_state_dict(torch.load(cnn_path, map_location=device))
            model.eval()
            registry["cnn"] = {"model": model, "device": device}
            logger.info("  ✓ Loaded EfficientNet-B0 CNN")
        except Exception as e:
            logger.warning(f"  ✗ CNN load failed: {e}")

    # Severity model (UPDRS regression with CI)
    severity_path = "./models/severity_model.pkl"
    if os.path.exists(severity_path):
        with open(severity_path, "rb") as f:
            registry["severity"] = pickle.load(f)
        logger.info("  ✓ Loaded severity model")

    _model_registry = registry
    return registry


def _classify_risk(prob: float) -> str:
    if prob < 0.30:
        return "low"
    elif prob < 0.60:
        return "moderate"
    elif prob < 0.85:
        return "high"
    return "very_high"


def _confidence_category(probs: list[float]) -> str:
    """
    Derive calibrated confidence category from model consensus spread.
    High   = all models agree (low variance)
    Low    = models disagree (high variance)
    """
    if not probs:
        return "Low"
    std = float(np.std(probs))
    if std < 0.08:
        return "High"
    elif std < 0.18:
        return "Moderate"
    return "Low"


def _severity_tier(score: float) -> str:
    if score < 15:
        return "Healthy"
    elif score < 30:
        return "Mild"
    elif score < 55:
        return "Moderate"
    return "Severe"


def run_inference(
    file_path: str | Path,
    patient_id: str | None = None,
    include_shap: bool = True,
) -> Dict[str, Any]:
    """
    Full inference pipeline on a single audio file.

    Returns a comprehensive result dict with calibrated risk assessment,
    severity estimation with 95% CI, explainability, and signal quality metrics.
    """
    logger.info(f"Starting screening inference on: {file_path}")

    # ── 1. Preprocess ─────────────────────────────────────────────────────────
    y, sr, quality = preprocess_pipeline(file_path)

    if not quality["valid"]:
        return {
            "status": "failed",
            "error": "; ".join(quality["issues"]),
            "quality": quality,
            "disclaimer": SYSTEM_DISCLAIMER,
        }

    # ── 2. Feature Extraction ─────────────────────────────────────────────────
    all_features = extract_all_features(y, sr)
    clinical_vec = get_clinical_features(all_features).reshape(1, -1)

    # ── 3. Normalize tabular features ─────────────────────────────────────────
    try:
        clinical_norm = tabular_normalizer.transform(clinical_vec)
    except FileNotFoundError:
        logger.warning("Scaler not fitted — using raw features. Run training first.")
        clinical_norm = clinical_vec

    # ── 4. Load models ────────────────────────────────────────────────────────
    models = _load_models()

    # ── 5. Classical model predictions (calibrated) ───────────────────────────
    model_preds: Dict[str, float] = {}

    for name in ["rf", "xgb", "lgbm"]:
        if name in models:
            try:
                prob = float(models[name].predict_proba(clinical_norm)[0, 1])
                model_preds[name] = round(prob, 4)
            except Exception as e:
                logger.warning(f"Model {name} inference failed: {e}")

    # Ensemble prediction (weighted soft-voting)
    if "ensemble" in models:
        try:
            prob = float(models["ensemble"].predict_proba(clinical_norm)[0, 1])
            model_preds["ensemble"] = round(prob, 4)
        except Exception:
            pass

    # ONNX edge fallback
    if not model_preds and "onnx" in models:
        try:
            ort_session = models["onnx"]
            input_name = ort_session.get_inputs()[0].name
            out = ort_session.run(None, {input_name: clinical_norm.astype(np.float32)})
            prob = float(out[1][0, 1]) if isinstance(out[1], np.ndarray) else float(out[0][0])
            model_preds["onnx"] = round(prob, 4)
        except Exception as e:
            logger.warning(f"ONNX inference failed: {e}")

    # ── 6. EfficientNet-B0 spectrogram prediction ─────────────────────────────
    mel_spec = extract_mel_spectrogram(y, sr)
    mel_norm = spectrogram_normalizer.normalize(mel_spec)
    mel_norm = spectrogram_normalizer.standardize(mel_norm)

    if "cnn" in models:
        try:
            import torch
            device = models["cnn"]["device"]
            cnn = models["cnn"]["model"]
            tensor = torch.tensor(mel_norm.transpose(2, 0, 1)).unsqueeze(0).to(device)
            with torch.no_grad():
                logits = cnn(tensor)
                prob = float(torch.softmax(logits, dim=1)[0, 1].cpu())
            model_preds["efficientnet_b0"] = round(prob, 4)
        except Exception as e:
            logger.warning(f"EfficientNet inference failed: {e}")

    # ── 7. Calibrated ensemble probability ───────────────────────────────────
    if model_preds:
        # Prefer ensemble model; fall back to mean of individual classifiers
        if "ensemble" in model_preds:
            ensemble_prob = model_preds["ensemble"]
        else:
            ensemble_prob = float(np.mean(list(model_preds.values())))
    else:
        ensemble_prob = 0.0
        logger.warning("No models available — returning zero probability. Train models first.")

    # ── 8. Confidence and uncertainty ─────────────────────────────────────────
    individual_probs = [v for k, v in model_preds.items() if k != "ensemble"]
    confidence_cat = _confidence_category(individual_probs)
    uncertainty = estimate_uncertainty(model_preds, ensemble_prob, quality)
    confidence_numeric = uncertainty["confidence"]

    # ── 9. Severity estimation with conformal intervals ───────────────────────
    severity_score = 0.0
    severity_tier_label = "Healthy"
    severity_lower_bound = 0.0
    severity_upper_bound = 0.0

    if "severity" in models:
        try:
            sev_model = models["severity"]
            sev_preds_with_tier = sev_model.predict_with_tier(clinical_norm)
            sev_result = sev_preds_with_tier[0]
            severity_score = float(sev_result.get("predicted_updrs", sev_result["score"]))
            severity_tier_label = sev_result["tier"]
            severity_lower_bound = float(sev_result.get("lower_bound", max(0.0, severity_score - 5.0)))
            severity_upper_bound = float(sev_result.get("upper_bound", min(108.0, severity_score + 5.0)))
        except Exception as e:
            logger.warning(f"Severity model failed: {e}")

    # ── 10. Risk classification ───────────────────────────────────────────────
    risk_tier = _classify_risk(ensemble_prob)

    # ── 11. SHAP Explainability ───────────────────────────────────────────────
    shap_values = None
    feature_importance = None
    risk_breakdown = None
    clinical_explanations = []

    if include_shap and "rf" in models:
        try:
            from explainability.shap_explainer import explain_prediction
            from explainability.risk_breakdown import build_risk_breakdown
            from explainability.clinical_rules import explain_top_biomarkers
            from pipeline.dataset_loader import INTERNAL_FEATURE_COLS
            shap_result = explain_prediction(
                models["rf"], clinical_norm, INTERNAL_FEATURE_COLS
            )
            shap_values = shap_result["shap_values"]
            feature_importance = shap_result["feature_importance"]
            risk_breakdown = build_risk_breakdown(shap_result, all_features)
            clinical_explanations = explain_top_biomarkers(feature_importance, all_features)
        except Exception as e:
            logger.warning(f"SHAP explanation failed: {e}")

    from explainability.clinical_rules import build_biomarker_dashboard
    biomarker_dashboard = build_biomarker_dashboard(all_features)

    # ── 12. Compile result ────────────────────────────────────────────────────
    return {
        "status": "complete",
        "pd_probability": round(ensemble_prob, 4),
        "risk_tier": risk_tier,
        "confidence": round(confidence_numeric, 4),
        "confidence_category": confidence_cat,
        "uncertainty": uncertainty,
        "uncertainty_score": uncertainty["uncertainty_score"],
        "severity_score": round(severity_score, 2),
        "predicted_updrs": round(severity_score, 2),
        "severity_tier": severity_tier_label,
        "severity_lower_ci": round(severity_lower_bound, 1),
        "severity_upper_ci": round(severity_upper_bound, 1),
        "lower_bound": round(severity_lower_bound, 1),
        "upper_bound": round(severity_upper_bound, 1),
        "model_predictions": model_preds,
        "features": all_features,
        "shap_values": shap_values,
        "feature_importance": feature_importance,
        "risk_breakdown": risk_breakdown,
        "clinical_explanations": clinical_explanations,
        "biomarkers": biomarker_dashboard,
        "quality": quality,
        "disclaimer": SYSTEM_DISCLAIMER,
    }
