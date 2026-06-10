"""
Voice Feature Extractor
========================
Extracts all acoustic biomarkers used in clinical Parkinson's assessment:
- MFCC (40 coefficients + Δ + ΔΔ)
- Mel-spectrogram (for CNN input)
- Pitch (F0): mean, min, max
- Jitter variants: local, absolute, RAP, PPQ5, DDP
- Shimmer variants: local, dB, APQ3, APQ5, APQ11, DDA
- HNR, NHR
- Nonlinear dynamics: RPDE, DFA, PPE, spread1, spread2, D2
- Voice stability: pitch/energy SD over time windows
"""

import logging
from typing import Dict, Any, Optional

import numpy as np
import librosa
import scipy.signal as signal
import scipy.stats as stats

logger = logging.getLogger(__name__)


# ─── Pitch / F0 Extraction ────────────────────────────────────────────────────

def extract_f0(
    y: np.ndarray,
    sr: int,
    fmin: float = 50.0,
    fmax: float = 600.0,
    hop_length: int = 256,
) -> np.ndarray:
    """
    Extract fundamental frequency (F0) sequence using PYIN (probabilistic YIN).
    Returns array of voiced F0 values (unvoiced frames excluded).
    """
    f0, voiced_flag, _ = librosa.pyin(
        y, fmin=fmin, fmax=fmax,
        sr=sr, hop_length=hop_length,
        fill_na=None,
    )
    voiced_f0 = f0[voiced_flag & ~np.isnan(f0)] if f0 is not None else np.array([])
    return voiced_f0.astype(np.float64)


def _periods_from_f0(f0: np.ndarray) -> np.ndarray:
    """Convert F0 array to glottal period array T = 1/F0."""
    safe_f0 = np.where(f0 > 0, f0, np.nan)
    return 1.0 / safe_f0


# ─── Jitter ────────────────────────────────────────────────────────────────────

def compute_jitter(f0: np.ndarray) -> Dict[str, float]:
    """
    Compute jitter metrics from F0 sequence.
    All values expressed as ratios (dimensionless).
    """
    if len(f0) < 3:
        return {k: 0.0 for k in
                ["jitter_local", "jitter_abs", "jitter_rap", "jitter_ppq5", "jitter_ddp"]}

    T = _periods_from_f0(f0)
    T = T[~np.isnan(T)]
    if len(T) < 3:
        return {k: 0.0 for k in
                ["jitter_local", "jitter_abs", "jitter_rap", "jitter_ppq5", "jitter_ddp"]}

    T_mean = np.mean(T)

    # Local jitter: mean abs diff of consecutive periods / mean period
    local_abs_diffs = np.abs(np.diff(T))
    jitter_abs = float(np.mean(local_abs_diffs))
    jitter_local = jitter_abs / T_mean if T_mean > 0 else 0.0

    # RAP: 3-point average
    if len(T) >= 3:
        rap_diffs = np.abs(T[1:-1] - (T[:-2] + T[1:-1] + T[2:]) / 3.0)
        jitter_rap = float(np.mean(rap_diffs) / T_mean) if T_mean > 0 else 0.0
    else:
        jitter_rap = 0.0

    # PPQ5: 5-point average
    if len(T) >= 5:
        ppq_diffs = np.array([
            abs(T[i] - np.mean(T[max(0, i-2):i+3])) for i in range(2, len(T) - 2)
        ])
        jitter_ppq5 = float(np.mean(ppq_diffs) / T_mean) if T_mean > 0 else 0.0
    else:
        jitter_ppq5 = jitter_rap

    # DDP = 3 * RAP
    jitter_ddp = 3.0 * jitter_rap

    return {
        "jitter_local": round(jitter_local * 100, 6),  # in %
        "jitter_abs": round(jitter_abs * 1000, 6),      # in ms
        "jitter_rap": round(jitter_rap * 100, 6),
        "jitter_ppq5": round(jitter_ppq5 * 100, 6),
        "jitter_ddp": round(jitter_ddp * 100, 6),
    }


# ─── Shimmer ───────────────────────────────────────────────────────────────────

def _amplitude_envelope(y: np.ndarray, sr: int, hop_length: int = 256) -> np.ndarray:
    """RMS energy envelope per frame as amplitude proxy."""
    rms = librosa.feature.rms(y=y, frame_length=1024, hop_length=hop_length)[0]
    return rms.astype(np.float64)


def compute_shimmer(y: np.ndarray, sr: int, f0: np.ndarray) -> Dict[str, float]:
    """
    Compute shimmer metrics from amplitude envelope aligned to F0 frames.
    """
    if len(f0) < 3:
        return {k: 0.0 for k in
                ["shimmer_local", "shimmer_db", "shimmer_apq3",
                 "shimmer_apq5", "shimmer_apq11", "shimmer_dda"]}

    A = _amplitude_envelope(y, sr)
    if len(A) < 3:
        return {k: 0.0 for k in
                ["shimmer_local", "shimmer_db", "shimmer_apq3",
                 "shimmer_apq5", "shimmer_apq11", "shimmer_dda"]}

    A = A[A > 1e-8]  # Remove silence
    if len(A) < 3:
        return {k: 0.0 for k in
                ["shimmer_local", "shimmer_db", "shimmer_apq3",
                 "shimmer_apq5", "shimmer_apq11", "shimmer_dda"]}

    A_mean = np.mean(A)

    # Local shimmer
    abs_diffs = np.abs(np.diff(A))
    shimmer_local = float(np.mean(abs_diffs) / A_mean) * 100 if A_mean > 0 else 0.0

    # Shimmer in dB
    db_ratios = np.abs(
        20 * np.log10(A[1:] / (A[:-1] + 1e-10) + 1e-10)
    )
    shimmer_db = float(np.mean(db_ratios))

    # APQ3 — 3-point amplitude perturbation quotient
    if len(A) >= 3:
        apq3_vals = np.abs(A[1:-1] - (A[:-2] + A[1:-1] + A[2:]) / 3.0)
        shimmer_apq3 = float(np.mean(apq3_vals) / A_mean) * 100 if A_mean > 0 else 0.0
    else:
        shimmer_apq3 = 0.0

    # APQ5
    if len(A) >= 5:
        apq5_vals = np.array([
            abs(A[i] - np.mean(A[max(0, i-2):i+3])) for i in range(2, len(A)-2)
        ])
        shimmer_apq5 = float(np.mean(apq5_vals) / A_mean) * 100 if A_mean > 0 else 0.0
    else:
        shimmer_apq5 = shimmer_apq3

    # APQ11
    if len(A) >= 11:
        apq11_vals = np.array([
            abs(A[i] - np.mean(A[max(0, i-5):i+6])) for i in range(5, len(A)-5)
        ])
        shimmer_apq11 = float(np.mean(apq11_vals) / A_mean) * 100 if A_mean > 0 else 0.0
    else:
        shimmer_apq11 = shimmer_apq5

    # DDA = 3 * APQ3
    shimmer_dda = 3.0 * shimmer_apq3

    return {
        "shimmer_local": round(shimmer_local, 6),
        "shimmer_db": round(shimmer_db, 6),
        "shimmer_apq3": round(shimmer_apq3, 6),
        "shimmer_apq5": round(shimmer_apq5, 6),
        "shimmer_apq11": round(shimmer_apq11, 6),
        "shimmer_dda": round(shimmer_dda, 6),
    }


# ─── HNR / NHR ────────────────────────────────────────────────────────────────

def compute_hnr(y: np.ndarray, sr: int) -> Dict[str, float]:
    """
    Estimate Harmonics-to-Noise Ratio and Noise-to-Harmonics Ratio
    using the autocorrelation method (Boersma, 1993 approximation).
    """
    try:
        # Compute autocorrelation
        frame_len = 1024
        if len(y) < frame_len:
            return {"hnr": 0.0, "nhr": 1.0}

        # Windowed frames
        frames = librosa.util.frame(y, frame_length=frame_len, hop_length=256).T
        hnr_vals = []

        for frame in frames:
            if np.max(np.abs(frame)) < 1e-6:
                continue
            # Normalized autocorrelation
            acf = np.correlate(frame, frame, mode='full')
            acf = acf[len(acf)//2:]
            r0 = acf[0]
            if r0 < 1e-10:
                continue
            acf_norm = acf / r0

            # Find first peak after zero
            r_peak = np.max(acf_norm[10:frame_len//2])
            r_peak = np.clip(r_peak, 0, 1 - 1e-8)

            if r_peak > 0:
                hnr_db = 10 * np.log10(r_peak / (1 - r_peak + 1e-10))
                hnr_vals.append(hnr_db)

        hnr = float(np.mean(hnr_vals)) if hnr_vals else 0.0
        nhr = float(10 ** (-hnr / 10)) if hnr != 0 else 1.0

        return {
            "hnr": round(hnr, 4),
            "nhr": round(np.clip(nhr, 0, 1), 6),
        }
    except Exception as e:
        logger.warning(f"HNR computation failed: {e}")
        return {"hnr": 0.0, "nhr": 1.0}


# ─── Nonlinear Dynamics ────────────────────────────────────────────────────────

def compute_ppe(f0: np.ndarray) -> float:
    """
    Pitch Period Entropy (PPE) — Shannon entropy of the normalized pitch period.
    High PPE indicates unstable pitch control (Parkinson marker).
    """
    if len(f0) < 2:
        return 0.0
    T = _periods_from_f0(f0)
    T = T[~np.isnan(T)]
    if len(T) < 2:
        return 0.0
    # Normalize periods
    T_norm = (T - np.min(T)) / (np.ptp(T) + 1e-10)
    # Histogram entropy
    hist, _ = np.histogram(T_norm, bins=min(30, len(T)//2 + 1), density=True)
    hist = hist[hist > 0]
    return float(-np.sum(hist * np.log2(hist + 1e-10)) / np.log2(len(hist) + 1))


def compute_dfa(signal_arr: np.ndarray, scales: Optional[np.ndarray] = None) -> float:
    """
    Detrended Fluctuation Analysis (DFA) — measures long-range correlations.
    Returns the scaling exponent α. α ≈ 0.5 → uncorrelated; α > 0.5 → persistent.
    """
    if len(signal_arr) < 16:
        return 0.5
    try:
        y = np.cumsum(signal_arr - np.mean(signal_arr))
        if scales is None:
            scales = np.logspace(1, np.log10(len(y) // 4), 20).astype(int)
            scales = np.unique(scales)
            scales = scales[scales > 1]

        flucts = []
        for s in scales:
            n_segments = len(y) // s
            if n_segments < 2:
                continue
            resid_rms = []
            for i in range(n_segments):
                seg = y[i*s:(i+1)*s]
                x = np.arange(len(seg))
                poly = np.polyfit(x, seg, 1)
                trend = np.polyval(poly, x)
                resid_rms.append(np.sqrt(np.mean((seg - trend) ** 2)))
            flucts.append((s, np.mean(resid_rms)))

        if len(flucts) < 2:
            return 0.5

        log_s = np.log10([f[0] for f in flucts])
        log_f = np.log10([f[1] + 1e-10 for f in flucts])
        alpha, _ = np.polyfit(log_s, log_f, 1)
        return round(float(np.clip(alpha, 0, 2)), 4)
    except Exception:
        return 0.5


def compute_rpde(y: np.ndarray, m: int = 4, tau: int = 1, epsilon: float = 0.2) -> float:
    """
    Recurrence Period Density Entropy (RPDE) approximation.
    Measures signal unpredictability via recurrence quantification.
    Higher RPDE → more irregular (Parkinson marker).
    """
    if len(y) < m * tau + 100:
        return 0.5
    try:
        # Phase space embedding
        N = len(y)
        M = N - (m - 1) * tau
        if M < 10:
            return 0.5
        X = np.array([y[i:i + (m-1)*tau + 1:tau] for i in range(M)])

        # Pairwise distances (subsample for performance)
        subsample = min(500, M)
        idx = np.random.choice(M, subsample, replace=False)
        X_sub = X[idx]

        # Recurrence periods
        eps = epsilon * np.std(y)
        periods = []
        for i, xi in enumerate(X_sub[:50]):
            dists = np.linalg.norm(X_sub - xi, axis=1)
            recurrent = np.where(dists < eps)[0]
            if len(recurrent) > 1:
                p_diffs = np.diff(recurrent)
                periods.extend(p_diffs.tolist())

        if not periods:
            return 0.5

        hist, _ = np.histogram(periods, bins=min(20, len(periods)//2 + 1))
        hist = hist[hist > 0].astype(float)
        hist /= hist.sum()
        entropy = -np.sum(hist * np.log2(hist + 1e-10))
        max_ent = np.log2(len(hist))
        return round(float(entropy / max_ent) if max_ent > 0 else 0.5, 4)
    except Exception:
        return 0.5


def compute_spread_d2(f0: np.ndarray) -> Dict[str, float]:
    """
    Compute spread1, spread2, D2 — nonlinear dynamical measures from UCI dataset.
    - spread1, spread2: first two modes of F0 variance structure
    - D2: correlation dimension estimate
    """
    if len(f0) < 10:
        return {"spread1": -5.0, "spread2": 0.2, "d2": 1.5}

    log_f0 = np.log(f0 + 1e-10)
    spread1 = float(np.min(log_f0) - np.mean(log_f0))  # typically negative
    spread2 = float(np.std(log_f0))

    # D2 approximation via Grassberger-Procaccia (simplified)
    # Use embedding dimension m=2
    if len(f0) >= 4:
        diffs = np.diff(f0)
        # Correlation integral approximation
        eps = np.std(diffs) * 0.5
        n = len(diffs)
        count = np.sum(np.abs(diffs[:-1] - diffs[1:]) < eps)
        d2 = np.log(count / n + 1e-10) / np.log(eps + 1e-10)
        d2 = float(np.clip(d2, 0.5, 5.0))
    else:
        d2 = 1.5

    return {
        "spread1": round(spread1, 6),
        "spread2": round(spread2, 6),
        "d2": round(d2, 6),
    }


# ─── MFCC ─────────────────────────────────────────────────────────────────────

def extract_mfcc(
    y: np.ndarray, sr: int,
    n_mfcc: int = 40,
    hop_length: int = 512,
) -> Dict[str, float]:
    """
    Extract MFCC features with delta and delta-delta coefficients.
    Returns statistics (mean, std) for each coefficient.
    """
    mfcc = librosa.feature.mfcc(
        y=y, sr=sr, n_mfcc=n_mfcc,
        hop_length=hop_length,
        n_fft=2048,
        n_mels=128,
    )
    delta = librosa.feature.delta(mfcc)
    delta2 = librosa.feature.delta(mfcc, order=2)

    features = {}
    for i, (m, d, d2) in enumerate(zip(mfcc, delta, delta2)):
        features[f"mfcc_{i+1}_mean"] = round(float(np.mean(m)), 6)
        features[f"mfcc_{i+1}_std"] = round(float(np.std(m)), 6)
        features[f"delta_mfcc_{i+1}_mean"] = round(float(np.mean(d)), 6)
        features[f"delta2_mfcc_{i+1}_mean"] = round(float(np.mean(d2)), 6)

    return features


# ─── Mel Spectrogram ──────────────────────────────────────────────────────────

def extract_mel_spectrogram(
    y: np.ndarray, sr: int,
    n_mels: int = 128,
    hop_length: int = 512,
    n_fft: int = 2048,
    img_size: int = 224,
) -> np.ndarray:
    """
    Generate a log-Mel spectrogram as a (img_size, img_size) float32 array,
    normalized to [0, 1] for CNN input. Output is 3-channel (RGB) via replication.
    """
    mel = librosa.feature.melspectrogram(
        y=y, sr=sr, n_mels=n_mels,
        hop_length=hop_length, n_fft=n_fft,
    )
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Normalize to [0, 1]
    mel_norm = (mel_db - mel_db.min()) / (mel_db.max() - mel_db.min() + 1e-8)

    # Resize to img_size x img_size using scipy zoom
    from scipy.ndimage import zoom
    h, w = mel_norm.shape
    mel_resized = zoom(mel_norm, (img_size / h, img_size / w), order=1)

    # 3-channel (H, W, 3)
    mel_rgb = np.stack([mel_resized] * 3, axis=-1).astype(np.float32)
    return mel_rgb


# ─── Voice Stability Metrics ──────────────────────────────────────────────────

def compute_voice_stability(y: np.ndarray, sr: int, f0: np.ndarray) -> Dict[str, float]:
    """
    Longitudinal stability metrics over time windows.
    - Pitch coefficient of variation (CV)
    - Energy CV
    - Voiced fraction
    - Short-time energy variance
    """
    # Pitch CV
    pitch_cv = float(np.std(f0) / (np.mean(f0) + 1e-10)) if len(f0) > 2 else 0.0

    # Energy CV
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    energy_cv = float(np.std(rms) / (np.mean(rms) + 1e-10))

    # Voiced fraction
    total_frames = len(librosa.pyin(y, fmin=50, fmax=600, sr=sr)[0] or [0])
    voiced_fraction = len(f0) / max(total_frames, 1)

    # Zero-crossing rate (tremor indicator)
    zcr = librosa.feature.zero_crossing_rate(y, frame_length=2048, hop_length=512)[0]
    zcr_mean = float(np.mean(zcr))

    # Spectral centroid stability
    centroid = librosa.feature.spectral_centroid(y=y, sr=sr, hop_length=512)[0]
    centroid_cv = float(np.std(centroid) / (np.mean(centroid) + 1e-10))

    return {
        "pitch_cv": round(pitch_cv, 6),
        "energy_cv": round(energy_cv, 6),
        "voiced_fraction": round(float(voiced_fraction), 4),
        "zcr_mean": round(zcr_mean, 6),
        "spectral_centroid_cv": round(centroid_cv, 6),
    }


# ─── Master Extractor ─────────────────────────────────────────────────────────

def extract_all_features(y: np.ndarray, sr: int) -> Dict[str, Any]:
    """
    Master feature extraction pipeline.
    Returns a flat dict of all acoustic biomarkers.
    """
    logger.info("Extracting voice biomarkers...")

    # F0
    f0 = extract_f0(y, sr)
    f0_mean = float(np.mean(f0)) if len(f0) > 0 else 0.0
    f0_max = float(np.max(f0)) if len(f0) > 0 else 0.0
    f0_min = float(np.min(f0)) if len(f0) > 0 else 0.0

    features: Dict[str, Any] = {
        "fo_mean": round(f0_mean, 4),
        "fo_max": round(f0_max, 4),
        "fo_min": round(f0_min, 4),
    }

    # Jitter
    features.update(compute_jitter(f0))

    # Shimmer
    features.update(compute_shimmer(y, sr, f0))

    # HNR / NHR
    features.update(compute_hnr(y, sr))

    # Nonlinear dynamics
    features["ppe"] = compute_ppe(f0)
    features["dfa"] = compute_dfa(f0 if len(f0) > 10 else y[:2048])
    features["rpde"] = compute_rpde(y)
    features.update(compute_spread_d2(f0))

    # Voice stability
    features.update(compute_voice_stability(y, sr, f0))

    # MFCC
    mfcc_feats = extract_mfcc(y, sr)
    features.update(mfcc_feats)

    logger.info(f"Extracted {len(features)} acoustic features.")
    return features


def get_clinical_features(features: Dict[str, Any]) -> np.ndarray:
    """
    Extract the fixed-order 22-feature clinical feature vector
    matching the UCI Parkinson's dataset column layout.
    Used for classical ML models.
    """
    CLINICAL_COLS = [
        "fo_mean", "fo_max", "fo_min",
        "jitter_local", "jitter_abs", "jitter_rap", "jitter_ppq5", "jitter_ddp",
        "shimmer_local", "shimmer_db", "shimmer_apq3", "shimmer_apq5",
        "shimmer_apq11", "shimmer_dda",
        "nhr", "hnr",
        "rpde", "dfa",
        "spread1", "spread2", "d2", "ppe",
    ]
    return np.array([features.get(col, 0.0) for col in CLINICAL_COLS], dtype=np.float32)
