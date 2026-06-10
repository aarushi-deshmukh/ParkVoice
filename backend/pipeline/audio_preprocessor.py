"""
Audio Preprocessor
==================
Handles loading, noise reduction, silence trimming, resampling, and
normalization of audio files for voice biomarker analysis.
"""

import io
import logging
from pathlib import Path
from typing import Tuple

import numpy as np
import librosa
import soundfile as sf
import noisereduce as nr

from core.config import settings
from pipeline.audio_quality import assess_audio_quality

logger = logging.getLogger(__name__)

SUPPORTED_FORMATS = {".wav", ".mp3", ".m4a", ".ogg", ".flac"}


def load_audio(file_path: str | Path) -> Tuple[np.ndarray, int]:
    """
    Load audio file and return (signal, sample_rate).
    Converts to mono automatically.
    """
    path = Path(file_path)
    if path.suffix.lower() not in SUPPORTED_FORMATS:
        raise ValueError(f"Unsupported audio format: {path.suffix}")

    try:
        y, sr = librosa.load(str(path), sr=None, mono=True)
        logger.debug(f"Loaded audio: {path.name} | SR={sr} | Duration={len(y)/sr:.2f}s")
        return y, sr
    except Exception as e:
        raise RuntimeError(f"Failed to load audio file {path.name}: {e}") from e


def load_audio_bytes(audio_bytes: bytes, ext: str = ".wav") -> Tuple[np.ndarray, int]:
    """Load audio from raw bytes (e.g. from an HTTP upload)."""
    buf = io.BytesIO(audio_bytes)
    try:
        y, sr = librosa.load(buf, sr=None, mono=True)
        return y, sr
    except Exception as e:
        raise RuntimeError(f"Failed to decode audio bytes: {e}") from e


def resample(y: np.ndarray, orig_sr: int, target_sr: int = None) -> Tuple[np.ndarray, int]:
    """Resample audio to the configured target sample rate."""
    target = target_sr or settings.SAMPLE_RATE
    if orig_sr == target:
        return y, orig_sr
    y_resampled = librosa.resample(y, orig_sr=orig_sr, target_sr=target)
    return y_resampled, target


def reduce_noise(y: np.ndarray, sr: int, prop_decrease: float = 0.85) -> np.ndarray:
    """
    Spectral gating noise reduction.
    Uses the first 0.3 s as noise profile (assumes brief silence at start).
    Falls back gracefully if signal is too short.
    """
    try:
        noise_duration = min(0.3, len(y) / sr * 0.2)
        noise_samples = int(noise_duration * sr)
        if noise_samples < 512:
            # Not enough signal for profiling — apply without noise sample
            y_clean = nr.reduce_noise(y=y, sr=sr, prop_decrease=prop_decrease)
        else:
            noise_clip = y[:noise_samples]
            y_clean = nr.reduce_noise(
                y=y, sr=sr,
                y_noise=noise_clip,
                prop_decrease=prop_decrease,
                stationary=True,
            )
        return y_clean
    except Exception as e:
        logger.warning(f"Noise reduction failed ({e}), returning original signal.")
        return y


def trim_silence(
    y: np.ndarray,
    top_db: float = 30.0,
    frame_length: int = 2048,
    hop_length: int = 512,
) -> np.ndarray:
    """Remove leading and trailing silence."""
    y_trimmed, _ = librosa.effects.trim(
        y, top_db=top_db, frame_length=frame_length, hop_length=hop_length
    )
    return y_trimmed


def normalize_amplitude(y: np.ndarray, target_rms: float = 0.05) -> np.ndarray:
    """
    Normalize to target RMS to ensure consistent loudness across recordings.
    Prevents clipping by clamping to [-1, 1].
    """
    rms = np.sqrt(np.mean(y ** 2))
    if rms < 1e-8:
        logger.warning("Signal is near-silent; skipping RMS normalization.")
        return y
    y_norm = y * (target_rms / rms)
    return np.clip(y_norm, -1.0, 1.0)


def validate_recording(y: np.ndarray, sr: int) -> dict:
    """
    Quality gate: check minimum duration, SNR estimate, and clipping.
    Returns a dict with 'valid', 'warnings', and quality metrics.
    """
    duration = len(y) / sr
    warnings = []
    issues = []
    quality_report = assess_audio_quality(y, sr)

    if duration < 1.5:
        issues.append("Recording too short (< 1.5 s). At least 3 s recommended.")
    elif duration < 3.0:
        warnings.append("Recording is short (< 3 s). Longer recordings improve accuracy.")

    # Clipping detection
    clip_ratio = np.mean(np.abs(y) > 0.98)
    if clip_ratio > 0.01:
        warnings.append(f"Detected clipping in {clip_ratio*100:.1f}% of samples.")

    # Simple SNR estimate using energy ratio
    rms = np.sqrt(np.mean(y ** 2))
    noise_floor = np.percentile(np.abs(y), 5)
    snr_db = 20 * np.log10(rms / (noise_floor + 1e-8))
    if snr_db < 10:
        warnings.append(f"Low estimated SNR ({snr_db:.1f} dB). Consider recording in a quieter environment.")

    return {
        "valid": len(issues) == 0,
        "issues": issues,
        "warnings": warnings,
        "duration_seconds": round(duration, 2),
        "estimated_snr_db": round(snr_db, 1),
        "clip_ratio": round(float(clip_ratio), 4),
        "rms": round(float(rms), 5),
        **quality_report,
    }


def preprocess_pipeline(
    file_path: str | Path,
    *,
    denoise: bool = True,
    trim: bool = True,
    normalize: bool = True,
    target_sr: int = None,
) -> Tuple[np.ndarray, int, dict]:
    """
    Full preprocessing pipeline.

    Returns:
        (processed_signal, sample_rate, quality_report)
    """
    # 1. Load
    y, sr = load_audio(file_path)

    # 2. Resample
    y, sr = resample(y, sr, target_sr)

    # 3. Noise reduction
    if denoise:
        y = reduce_noise(y, sr)

    # 4. Trim silence
    if trim:
        y = trim_silence(y)

    # 5. Normalize amplitude
    if normalize:
        y = normalize_amplitude(y)

    # 6. Quality check
    quality = validate_recording(y, sr)

    logger.info(
        f"Preprocessing complete | "
        f"Duration={quality['duration_seconds']:.2f}s | "
        f"SNR≈{quality['estimated_snr_db']:.1f}dB | "
        f"Valid={quality['valid']}"
    )

    return y, sr, quality
