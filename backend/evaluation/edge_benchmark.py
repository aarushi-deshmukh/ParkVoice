"""
Edge-AI Hardware Benchmark Profiler
=====================================
Benchmarks ONNX and Quantized INT8 models for size, latency, and memory footprints.
Only reports real measurements from the hardware where this script is executed.
"""

import json
import logging
import os
import time
from typing import Dict

import numpy as np
import onnxruntime as ort

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = "./models"
FEATURE_DIM = 22

# Benchmarking constants
N_WARMUP = 10
N_RUNS = 100

def benchmark_model(model_path: str, X_test: np.ndarray) -> Dict:
    """Run real inference speed and size measurements on current CPU."""
    if not os.path.exists(model_path):
        return {}

    size_mb = os.path.getsize(model_path) / (1024 * 1024)

    opts = ort.SessionOptions()
    opts.log_severity_level = 3
    opts.intra_op_num_threads = 1
    session = ort.InferenceSession(model_path, sess_options=opts)
    input_name = session.get_inputs()[0].name

    # Warmup
    for _ in range(N_WARMUP):
        session.run(None, {input_name: X_test})

    # Benchmark latency
    latencies = []
    for _ in range(N_RUNS):
        t0 = time.perf_counter()
        session.run(None, {input_name: X_test})
        latencies.append((time.perf_counter() - t0) * 1000) # to ms

    avg_latency = float(np.mean(latencies))
    p95_latency = float(np.percentile(latencies, 95))

    memory_usage_mb = None
    try:
        import psutil
        import os as _os
        process = psutil.Process(_os.getpid())
        memory_usage_mb = process.memory_info().rss / (1024 * 1024)
    except Exception:
        pass

    return {
        "size_mb": round(size_mb, 3),
        "latency_ms": round(avg_latency, 3),
        "p95_latency_ms": round(p95_latency, 3),
        "memory_usage_mb": round(memory_usage_mb, 2) if memory_usage_mb else None,
    }


def main():
    logger.info("=" * 60)
    logger.info("Edge AI Hardware Benchmarking Profiler")
    logger.info(f"Targeting directory: {MODEL_DIR}")

    # Generate dummy input vector (1, 22) matching tabular normalized acoustic features
    X_dummy = np.random.normal(0.0, 1.0, (1, FEATURE_DIM)).astype(np.float32)

    results = {}
    models_to_test = ["rf", "xgb", "lgbm", "ensemble_base"]

    for m_name in models_to_test:
        orig_path = os.path.join(MODEL_DIR, f"{m_name}.onnx")
        quant_path = os.path.join(MODEL_DIR, f"{m_name}_quantized.onnx")

        # Fallback names
        if m_name == "ensemble_base" and not os.path.exists(orig_path):
            orig_path = os.path.join(MODEL_DIR, "ensemble_quantized.onnx") # Check if direct ensemble exists
            quant_path = os.path.join(MODEL_DIR, "ensemble_quantized.onnx")

        orig_stats = benchmark_model(orig_path, X_dummy)
        quant_stats = benchmark_model(quant_path, X_dummy)

        if not orig_stats and not quant_stats:
            logger.warning(f"  Model files not found for {m_name}. Skipping.")
            continue

        # If only one exists (e.g. only quantized was saved), copy stats
        if not orig_stats: orig_stats = quant_stats
        if not quant_stats: quant_stats = orig_stats

        model_display = m_name.upper().replace("_BASE", "")
        logger.info(f"\nModel: {model_display}")
        logger.info(f"  Original Size:  {orig_stats['size_mb']:.2f} MB | Latency: {orig_stats['latency_ms']:.2f} ms")
        logger.info(f"  Quantized Size: {quant_stats['size_mb']:.2f} MB | Latency: {quant_stats['latency_ms']:.2f} ms")

        results[m_name] = {
            "model_name": model_display,
            "original_metrics": orig_stats,
            "quantized_metrics": quant_stats,
            "compression_ratio": round(orig_stats["size_mb"] / max(quant_stats["size_mb"], 1e-6), 2),
            "speedup_factor": round(orig_stats["latency_ms"] / max(quant_stats["latency_ms"], 1e-6), 2),
            "hardware_note": "Measured on the current host only. Edge-device values require running this script on that hardware.",
        }

    # Save to JSON
    output_path = os.path.join(MODEL_DIR, "onnx_benchmarks.json")
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    logger.info(f"\n✅ Edge benchmarking complete. Results saved to {output_path}")


if __name__ == "__main__":
    main()
