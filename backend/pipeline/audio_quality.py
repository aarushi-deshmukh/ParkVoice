"""
Audio quality assessment for screening support.

The module produces a compact report used by inference, uncertainty scoring,
and UI display. It is intentionally model-independent so unsupported datasets
or edge deployments can still quality-gate recordings.
"""

from __future__ import annotations

from typing import Any, Dict

import librosa
import numpy as np


def _estimate_snr_db(y: np.ndarray) -> float:
    rms = float(np.sqrt(np.mean(np.square(y))) + 1e-12)
    noise_floor = float(np.percentile(np.abs(y), 10) + 1e-12)
    return float(20.0 * np.log10(rms / noise_floor))


def _silence_ratio(y: np.ndarray, sr: int) -> float:
    if len(y) == 0:
        return 1.0
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    if len(rms) == 0:
        return 1.0
    threshold = max(np.percentile(rms, 20), 1e-5)
    return float(np.mean(rms <= threshold))


def _signal_stability(y: np.ndarray) -> float:
    if len(y) < 2048:
        return 0.0
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    if len(rms) < 2:
        return 0.0
    cv = float(np.std(rms) / (np.mean(rms) + 1e-8))
    return float(np.clip(1.0 - cv, 0.0, 1.0))


def assess_audio_quality(y: np.ndarray, sr: int) -> Dict[str, Any]:
    duration = float(len(y) / sr) if sr else 0.0
    snr_db = _estimate_snr_db(y) if len(y) else 0.0
    clipping_ratio = float(np.mean(np.abs(y) >= 0.98)) if len(y) else 0.0
    silence_ratio = _silence_ratio(y, sr)
    stability = _signal_stability(y)

    duration_score = np.clip(duration / 8.0, 0.0, 1.0)
    snr_score = np.clip((snr_db - 5.0) / 25.0, 0.0, 1.0)
    clipping_score = np.clip(1.0 - (clipping_ratio / 0.03), 0.0, 1.0)
    silence_score = np.clip(1.0 - (silence_ratio / 0.65), 0.0, 1.0)

    quality_score = float(
        0.25 * duration_score
        + 0.30 * snr_score
        + 0.20 * clipping_score
        + 0.15 * silence_score
        + 0.10 * stability
    )

    if quality_score >= 0.85:
        category = "Excellent"
    elif quality_score >= 0.70:
        category = "Good"
    elif quality_score >= 0.50:
        category = "Fair"
    else:
        category = "Poor"

    warnings: list[str] = []
    if duration < 3.0:
        warnings.append("Recording duration is short; at least 3 seconds is required and 8 seconds is preferred.")
    if snr_db < 10.0:
        warnings.append("Estimated SNR is low; record in a quieter environment if possible.")
    if clipping_ratio > 0.01:
        warnings.append("Potential clipping detected; reduce microphone gain or increase distance.")
    if silence_ratio > 0.45:
        warnings.append("High silence ratio may reduce biomarker reliability.")
    if stability < 0.45:
        warnings.append("Signal energy is unstable; repeat a steady sustained phonation if available.")

    return {
        "quality_score": round(quality_score, 3),
        "quality_category": category,
        "warnings": warnings,
        "duration": round(duration, 2),
        "snr": round(snr_db, 1),
        "clipping": round(clipping_ratio, 4),
        "silence_ratio": round(silence_ratio, 3),
        "signal_stability": round(stability, 3),
    }
