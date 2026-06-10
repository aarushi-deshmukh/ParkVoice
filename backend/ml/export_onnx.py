"""
ONNX Export and INT8 Quantization
==================================
Exports trained classical models to ONNX format for edge inference.
Applies INT8 quantization for reduced model size and faster CPU inference.
"""

import json
import logging
import os
import pickle
from typing import Dict

import numpy as np
import onnx
import onnxruntime as ort
from skl2onnx import convert_sklearn
from skl2onnx.common.data_types import FloatTensorType

from pipeline.dataset_loader import load_prepared_dataset
from pipeline.normalization import tabular_normalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = "./models"
FEATURE_DIM = 22


def export_sklearn_to_onnx(model, model_name: str, n_features: int = FEATURE_DIM) -> str:
    """Export an sklearn-compatible model to ONNX format."""
    initial_type = [("float_input", FloatTensorType([None, n_features]))]
    onnx_model = convert_sklearn(
        model, initial_types=initial_type,
        target_opset=18,
        options={type(model): {"zipmap": False}},
    )
    output_path = os.path.join(MODEL_DIR, f"{model_name}.onnx")
    onnx.save(onnx_model, output_path)
    logger.info(f"  ✓ Exported {model_name} → {output_path}")
    return output_path


def quantize_onnx(input_path: str, output_path: str) -> str:
    """Apply INT8 dynamic quantization to an ONNX model."""
    from onnxruntime.quantization import quantize_dynamic, QuantType
    quantize_dynamic(
        model_input=input_path,
        model_output=output_path,
        weight_type=QuantType.QInt8,
    )
    logger.info(f"  ✓ Quantized → {output_path}")
    return output_path


def benchmark_onnx(model_path: str, X_test: np.ndarray, n_runs: int = 100) -> Dict:
    """Benchmark ONNX inference speed."""
    import time
    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    session = ort.InferenceSession(model_path, sess_options=opts)
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(5):
        session.run(None, {input_name: X_test[:1].astype(np.float32)})

    # Benchmark
    t0 = time.perf_counter()
    for _ in range(n_runs):
        session.run(None, {input_name: X_test[:1].astype(np.float32)})
    elapsed = (time.perf_counter() - t0) / n_runs * 1000

    size_kb = os.path.getsize(model_path) / 1024
    logger.info(f"  {os.path.basename(model_path)}: {elapsed:.2f} ms/inference | {size_kb:.1f} KB")

    return {
        "inference_ms": round(elapsed, 3),
        "model_size_kb": round(size_kb, 1),
        "path": model_path,
    }


def main():
    logger.info("=" * 60)
    logger.info("ONNX Export Pipeline")

    _, (X_val, y_val), (X_test, y_test), _ = load_prepared_dataset("./data")
    X_val_n = tabular_normalizer.transform(X_val)
    X_test_n = tabular_normalizer.transform(X_test)

    export_results = {}

    for name in ["rf", "xgb", "lgbm", "ensemble"]:
        pkl_path = os.path.join(MODEL_DIR, f"{name}_model.pkl")
        if not os.path.exists(pkl_path):
            logger.warning(f"  Skipping {name} — model not found.")
            continue

        with open(pkl_path, "rb") as f:
            model = pickle.load(f)

        try:
            # Export base model (for ensemble, export underlying components)
            if name == "ensemble":
                # Export as RF (first model in ensemble)
                base_model = model.models[0][1] if hasattr(model, "models") else model
                onnx_path = export_sklearn_to_onnx(base_model, f"{name}_base")
            else:
                onnx_path = export_sklearn_to_onnx(model, name)

            # Quantize
            quant_path = onnx_path.replace(".onnx", "_quantized.onnx")
            quantize_onnx(onnx_path, quant_path)

            # Benchmark both
            orig_bench = benchmark_onnx(onnx_path, X_test_n)
            quant_bench = benchmark_onnx(quant_path, X_test_n)

            export_results[name] = {
                "original": orig_bench,
                "quantized": quant_bench,
                "size_reduction_pct": round(
                    (1 - quant_bench["model_size_kb"] / orig_bench["model_size_kb"]) * 100, 1
                ),
                "speedup": round(orig_bench["inference_ms"] / quant_bench["inference_ms"], 2),
            }

        except Exception as e:
            logger.error(f"  ✗ Failed to export {name}: {e}")

    # Copy best quantized model as primary ONNX
    primary_src = os.path.join(MODEL_DIR, "ensemble_base_quantized.onnx")
    if not os.path.exists(primary_src):
        primary_src = os.path.join(MODEL_DIR, "rf_quantized.onnx")
    if os.path.exists(primary_src):
        import shutil
        shutil.copy(primary_src, os.path.join(MODEL_DIR, "ensemble_quantized.onnx"))
        logger.info("  ✓ Primary ONNX edge model: ensemble_quantized.onnx")

    with open(os.path.join(MODEL_DIR, "onnx_benchmarks.json"), "w") as f:
        json.dump(export_results, f, indent=2)

    logger.info("\n✅ ONNX export complete.")
    return export_results


if __name__ == "__main__":
    main()
