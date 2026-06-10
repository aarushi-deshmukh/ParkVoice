"""
Feature Normalization
=====================
Fit and persist RobustScaler (tabular) and MinMaxScaler (spectrogram).
Provides transform utilities used by all model inference paths.
"""

import logging
import os
import pickle
from pathlib import Path
from typing import Tuple

import numpy as np
from sklearn.preprocessing import RobustScaler, MinMaxScaler

from core.config import settings

logger = logging.getLogger(__name__)

TABULAR_SCALER_PATH = os.path.join(settings.MODEL_DIR, "tabular_scaler.pkl")
SPECTROGRAM_SCALER_PATH = os.path.join(settings.MODEL_DIR, "spectrogram_scaler.pkl")


class FeatureNormalizer:
    """
    Wraps RobustScaler for tabular clinical features.
    Persists to disk so the same scaler is used in training and inference.
    """

    def __init__(self, scaler_path: str = TABULAR_SCALER_PATH):
        self.scaler_path = scaler_path
        self._scaler: RobustScaler | None = None

    def fit(self, X: np.ndarray) -> "FeatureNormalizer":
        self._scaler = RobustScaler()
        self._scaler.fit(X)
        self._save()
        logger.info(f"Tabular scaler fitted on shape {X.shape}, saved to {self.scaler_path}")
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        scaler = self._load()
        return scaler.transform(X)

    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        self.fit(X)
        return self._scaler.transform(X)

    def _save(self):
        os.makedirs(os.path.dirname(self.scaler_path) or ".", exist_ok=True)
        with open(self.scaler_path, "wb") as f:
            pickle.dump(self._scaler, f)

    def _load(self) -> RobustScaler:
        if self._scaler is not None:
            return self._scaler
        if os.path.exists(self.scaler_path):
            with open(self.scaler_path, "rb") as f:
                self._scaler = pickle.load(f)
            return self._scaler
        raise FileNotFoundError(
            f"Scaler not found at {self.scaler_path}. "
            "Run training pipeline first to fit and save the scaler."
        )

    @property
    def is_fitted(self) -> bool:
        return os.path.exists(self.scaler_path)


class SpectrogramNormalizer:
    """
    Per-channel min-max normalization for mel spectrograms.
    Stateless — applied per image in-place during preprocessing.
    """

    @staticmethod
    def normalize(spec: np.ndarray) -> np.ndarray:
        """Normalize spectrogram image to [0, 1] per channel."""
        out = spec.copy().astype(np.float32)
        for c in range(out.shape[-1]):
            cmin, cmax = out[..., c].min(), out[..., c].max()
            out[..., c] = (out[..., c] - cmin) / (cmax - cmin + 1e-8)
        return out

    @staticmethod
    def standardize(spec: np.ndarray) -> np.ndarray:
        """ImageNet-style standardize (mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])."""
        MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return (spec - MEAN) / STD


# Module-level singletons
tabular_normalizer = FeatureNormalizer()
spectrogram_normalizer = SpectrogramNormalizer()
