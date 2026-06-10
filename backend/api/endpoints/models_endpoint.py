"""
Model Metrics Endpoints
========================
Returns training metrics, comparison tables, and model status for the UI.
"""

import json
import logging
import os
from typing import Dict, List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from datetime import datetime

logger = logging.getLogger(__name__)
router = APIRouter()

MODEL_DIR = "./models"
REGISTRY_PATH = "./model_registry.json"


def _load_json(path: str) -> Optional[dict]:
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None


@router.get("/status")
async def model_status():
    """Check which models are available on disk."""
    models = {
        "random_forest": os.path.exists(os.path.join(MODEL_DIR, "rf_model.pkl")),
        "xgboost": os.path.exists(os.path.join(MODEL_DIR, "xgb_model.pkl")),
        "lightgbm": os.path.exists(os.path.join(MODEL_DIR, "lgbm_model.pkl")),
        "ensemble": os.path.exists(os.path.join(MODEL_DIR, "ensemble_model.pkl")),
        "efficientnet_b0": os.path.exists(os.path.join(MODEL_DIR, "cnn_model.pt")),
        "severity": os.path.exists(os.path.join(MODEL_DIR, "severity_model.pkl")),
        "onnx": os.path.exists(os.path.join(MODEL_DIR, "ensemble_quantized.onnx")),
    }


@router.get("/registry")
async def get_model_registry():
    """Return the production model registry metadata."""
    registry = _load_json(REGISTRY_PATH)
    if registry is None:
        raise HTTPException(status_code=404, detail="model_registry.json not found")
    return registry
    scaler_ready = os.path.exists(os.path.join(MODEL_DIR, "tabular_scaler.pkl"))

    return {
        "models": models,
        "scaler_ready": scaler_ready,
        "any_model_ready": any(models.values()),
        "full_pipeline_ready": all([
            models["ensemble"] or models["random_forest"],
            scaler_ready,
        ]),
    }


@router.get("/metrics")
async def get_all_metrics():
    """Return all model training metrics."""
    metrics = {}

    classical = _load_json(os.path.join(MODEL_DIR, "classical_metrics.json"))
    if classical:
        metrics.update(classical)

    ensemble = _load_json(os.path.join(MODEL_DIR, "ensemble_metrics.json"))
    if ensemble:
        metrics["ensemble"] = ensemble.get("val", {})

    cnn = _load_json(os.path.join(MODEL_DIR, "cnn_metrics.json"))
    if cnn:
        metrics["cnn"] = cnn

    severity = _load_json(os.path.join(MODEL_DIR, "severity_metrics.json"))
    if severity:
        metrics["severity_model"] = severity

    onnx = _load_json(os.path.join(MODEL_DIR, "onnx_benchmarks.json"))

    return {
        "classification_metrics": metrics,
        "onnx_benchmarks": onnx,
    }


@router.get("/comparison")
async def get_model_comparison():
    """
    Return a structured model comparison table for the UI.
    Shows all classification metrics side-by-side.
    """
    classical = _load_json(os.path.join(MODEL_DIR, "classical_metrics.json")) or {}
    ensemble_data = _load_json(os.path.join(MODEL_DIR, "ensemble_metrics.json")) or {}
    cnn_data = _load_json(os.path.join(MODEL_DIR, "cnn_metrics.json")) or {}

    def _row(name: str, display: str, data: dict, is_dl: bool = False) -> dict:
        return {
            "model": name,
            "display_name": display,
            "type": "Deep Learning" if is_dl else "Classical ML",
            "accuracy": data.get("accuracy"),
            "roc_auc": data.get("roc_auc") or data.get("best_val_auc"),
            "f1": data.get("f1"),
            "sensitivity": data.get("sensitivity"),
            "specificity": data.get("specificity"),
            "precision": data.get("precision"),
            "training_time_sec": data.get("training_time_sec"),
        }

    rows = []
    for name, display in [
        ("random_forest", "Random Forest"),
        ("xgboost", "XGBoost"),
        ("lightgbm", "LightGBM"),
    ]:
        if name in classical:
            rows.append(_row(name, display, classical[name]))

    if "val" in ensemble_data:
        rows.append(_row("ensemble", "Weighted Ensemble", ensemble_data["val"]))

    if cnn_data:
        rows.append(_row("efficientnet_b0", "EfficientNet-B0 CNN", cnn_data, is_dl=True))

    # Sort by ROC-AUC descending
    rows.sort(key=lambda x: x.get("roc_auc") or 0, reverse=True)

    return {
        "comparison_table": rows,
        "best_model": rows[0]["model"] if rows else None,
        "metrics_available": len(rows) > 0,
    }


@router.get("/benchmarks")
async def get_benchmarks():
    """Return real benchmark measurements for ONNX models."""
    onnx_path = os.path.join(MODEL_DIR, "ensemble_quantized.onnx")
    onnx_available = os.path.exists(onnx_path)

    bench_data = _load_json(os.path.join(MODEL_DIR, "onnx_benchmarks.json"))

    profiles = []
    if bench_data:
        for model_key, model_data in bench_data.items():
            quant_m = model_data.get("quantized_metrics") or model_data.get("quantized") or {}
            if not quant_m:
                continue
            latency = quant_m.get("latency_ms") or quant_m.get("inference_ms")
            size_mb = quant_m.get("size_mb")
            if size_mb is None and quant_m.get("model_size_kb") is not None:
                size_mb = quant_m["model_size_kb"] / 1024
            if latency is None or size_mb is None:
                continue
            profiles.append({
                "device": "host_cpu",
                "model": model_key,
                "latency_ms": round(float(latency), 3),
                "throughput_samples_per_sec": round(1000.0 / max(float(latency), 0.1), 1),
                "model_size_mb": round(float(size_mb), 3),
                "memory_usage_mb": quant_m.get("memory_usage_mb") or quant_m.get("ram_mb"),
                "compression_ratio": model_data.get("compression_ratio") or model_data.get("size_reduction_pct"),
                "quantization": "INT8",
                "deployable": float(latency) <= 500.0,
                "notes": "Measured on the current host CPU. Raspberry Pi or Jetson values require measurements from those devices."
            })

    return {
        "profiles": profiles,
        "onnx_available": onnx_available,
        "real_measurements_only": True,
        "message": None if profiles else "No real benchmark measurements found. Run backend/evaluation/edge_benchmark.py on the target hardware.",
        "benchmark_timestamp": datetime.utcnow().isoformat() + "Z"
    }


@router.post("/train")
async def trigger_training():
    """
    Trigger the full training pipeline.
    Returns immediately — training runs as a subprocess.
    Check /status to see when models are ready.
    """
    import subprocess, sys
    scripts = [
        "ml/train_classical.py",
        "ml/severity_model.py",
        "ml/train_ensemble.py",
    ]
    results = {}
    for script in scripts:
        if os.path.exists(script):
            proc = subprocess.Popen(
                [sys.executable, script],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            results[script] = {"pid": proc.pid, "status": "started"}
        else:
            results[script] = {"status": "script_not_found"}

    return {
        "message": "Training jobs started. Monitor logs for progress.",
        "jobs": results,
    }
