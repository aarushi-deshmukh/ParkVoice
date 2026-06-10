"""
Severity Prediction Model
=========================
Predicts UPDRS-equivalent severity score (continuous, 0–108)
from voice biomarkers. Trained as a regression task.
Also predicts severity tier and conformal prediction intervals.
"""

import json
import logging
import os
import pickle
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb

from pipeline.dataset_loader import (
    load_uci_dataset, add_synthetic_severity,
    split_dataset, prepare_features, INTERNAL_FEATURE_COLS,
)
from pipeline.normalization import tabular_normalizer

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = "./models"
os.makedirs(MODEL_DIR, exist_ok=True)


def severity_tier(score: float) -> str:
    if score < 15:
        return "Healthy"
    elif score < 30:
        return "Mild"
    elif score < 55:
        return "Moderate"
    return "Severe"


class SeverityModel:
    """
    XGBoost regressor for UPDRS severity score prediction.
    Exposes predict() returning (score, tier).
    """

    def __init__(self):
        self.model: xgb.XGBRegressor | None = None
        self.conformal_alpha: float = 0.10
        self.conformal_qhat: float = 5.0

    def fit(self, X_train: np.ndarray, y_train: np.ndarray,
            X_val: np.ndarray, y_val: np.ndarray) -> "SeverityModel":

        self.model = xgb.XGBRegressor(
            n_estimators=500,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=1.0,
            eval_metric="mae",
            early_stopping_rounds=50,
            random_state=42,
            verbosity=0,
        )
        self.model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=False,
        )

        # Split conformal calibration from validation residuals.
        val_preds = self.predict(X_val)
        residuals = np.abs(y_val - val_preds)
        n = len(residuals)
        quantile = np.ceil((n + 1) * (1 - self.conformal_alpha)) / n
        self.conformal_qhat = float(np.quantile(residuals, min(quantile, 1.0), method="higher"))
        logger.info(f"  ✓ Calculated conformal qhat: {self.conformal_qhat:.3f}")
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not fitted.")
        preds = self.model.predict(X)
        return np.clip(preds, 0, 108)

    def predict_with_tier(self, X: np.ndarray) -> list:
        scores = self.predict(X)
        results = []
        for s in scores:
            lower = float(np.clip(s - self.conformal_qhat, 0, 108))
            upper = float(np.clip(s + self.conformal_qhat, 0, 108))
            results.append({
                "predicted_updrs": round(float(s), 1),
                "score": round(float(s), 1),
                "tier": severity_tier(s),
                "lower_bound": round(lower, 1),
                "upper_bound": round(upper, 1),
            })
        return results

    def evaluate(self, X: np.ndarray, y: np.ndarray) -> Dict:
        preds = self.predict(X)
        mae = float(mean_absolute_error(y, preds))
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        r2 = float(r2_score(y, preds))
        # Tier accuracy
        pred_tiers = [severity_tier(p) for p in preds]
        true_tiers = [severity_tier(t) for t in y]
        tier_acc = float(np.mean([p == t for p, t in zip(pred_tiers, true_tiers)]))
        return {
            "mae": round(mae, 3),
            "rmse": round(rmse, 3),
            "r2": round(r2, 3),
            "tier_accuracy": round(tier_acc, 3),
        }

    def save(self, path: str):
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info(f"  ✓ Severity model saved → {path}")

    @classmethod
    def load(cls, path: str) -> "SeverityModel":
        with open(path, "rb") as f:
            return pickle.load(f)


def train_severity_model() -> Tuple[SeverityModel, Dict]:
    logger.info("=" * 60)
    logger.info("Training Severity Prediction Model...")

    # Load data with severity labels
    df = load_uci_dataset("./data")
    df = add_synthetic_severity(df)

    train_df, val_df, test_df = split_dataset(df, label_col="label")

    # Prepare features
    X_train, _ = prepare_features(train_df)
    X_val, _ = prepare_features(val_df)
    X_test, _ = prepare_features(test_df)

    y_train = train_df["updrs_score"].values.astype(np.float32)
    y_val = val_df["updrs_score"].values.astype(np.float32)
    y_test = test_df["updrs_score"].values.astype(np.float32)

    # Normalize
    X_train_n = tabular_normalizer.transform(X_train)
    X_val_n = tabular_normalizer.transform(X_val)
    X_test_n = tabular_normalizer.transform(X_test)

    # Train
    sev_model = SeverityModel()
    sev_model.fit(X_train_n, y_train, X_val_n, y_val)

    # Evaluate
    val_metrics = sev_model.evaluate(X_val_n, y_val)
    test_metrics = sev_model.evaluate(X_test_n, y_test)

    logger.info(f"  Val  → MAE: {val_metrics['mae']:.2f} | RMSE: {val_metrics['rmse']:.2f} | R²: {val_metrics['r2']:.3f} | Tier Acc: {val_metrics['tier_accuracy']:.3f}")
    logger.info(f"  Test → MAE: {test_metrics['mae']:.2f} | RMSE: {test_metrics['rmse']:.2f} | R²: {test_metrics['r2']:.3f} | Tier Acc: {test_metrics['tier_accuracy']:.3f}")

    # Save
    sev_model.save(os.path.join(MODEL_DIR, "severity_model.pkl"))

    metrics = {"val": val_metrics, "test": test_metrics}
    with open(os.path.join(MODEL_DIR, "severity_metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    return sev_model, metrics


if __name__ == "__main__":
    train_severity_model()
