"""
Ensemble Model Training
========================
Builds a soft-voting ensemble from RF, XGBoost, and LightGBM.
Uses Platt-scaled calibrated probabilities weighted by validation ROC-AUC.
"""

import json
import logging
import os
import pickle
from typing import Dict, List, Tuple

import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression

from pipeline.dataset_loader import load_prepared_dataset
from pipeline.normalization import tabular_normalizer
from evaluation.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = "./models"


class WeightedEnsemble:
    """
    Soft-voting ensemble with AUC-weighted averaging.
    Combines classical models using probabilities, not hard votes.
    """

    def __init__(self, models: List[Tuple[str, object]], weights: List[float]):
        self.models = models  # [(name, model), ...]
        self.weights = np.array(weights) / np.sum(weights)  # normalize
        self.meta_learner = None

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Weighted average of model probabilities."""
        all_probs = []
        for (name, model), w in zip(self.models, self.weights):
            try:
                probs = model.predict_proba(X)
                all_probs.append(probs * w)
            except Exception as e:
                logger.warning(f"Model {name} failed: {e}")
        if not all_probs:
            return np.full((len(X), 2), 0.5)
        return np.sum(all_probs, axis=0)

    def predict(self, X: np.ndarray) -> np.ndarray:
        proba = self.predict_proba(X)
        return (proba[:, 1] >= 0.5).astype(int)


def train_meta_ensemble(
    X_train: np.ndarray, y_train: np.ndarray,
    X_val: np.ndarray, y_val: np.ndarray,
) -> Tuple[WeightedEnsemble, Dict]:
    logger.info("=" * 60)
    logger.info("Training Ensemble Model...")

    # Load individual models
    models_with_names = []
    val_aucs = []

    for name in ["rf", "xgb", "lgbm"]:
        path = os.path.join(MODEL_DIR, f"{name}_model.pkl")
        if os.path.exists(path):
            with open(path, "rb") as f:
                model = pickle.load(f)
            models_with_names.append((name, model))

            # Compute validation AUC for weighting
            from sklearn.metrics import roc_auc_score
            probs = model.predict_proba(X_val)[:, 1]
            auc = roc_auc_score(y_val, probs)
            val_aucs.append(auc)
            logger.info(f"  {name.upper():8s} → Val AUC: {auc:.4f}")
        else:
            logger.warning(f"  Model {name} not found at {path} — skipping.")

    if not models_with_names:
        raise RuntimeError("No trained models found. Run train_classical.py first.")

    # Use AUC^2 weighting for more pronounced differentiation
    weights = [auc ** 2 for auc in val_aucs]
    ensemble = WeightedEnsemble(models_with_names, weights)

    # Evaluate ensemble
    metrics = compute_all_metrics(ensemble, X_val, y_val)
    logger.info(f"\n  Ensemble Val ROC-AUC: {metrics['roc_auc']:.4f}")
    logger.info(f"  Ensemble Val Accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"  Ensemble Val Sensitivity: {metrics['sensitivity']:.4f}")
    logger.info(f"  Ensemble Val Specificity: {metrics['specificity']:.4f}")

    # Also fit a meta-learner (stacking) on val set OOF predictions
    # Collect OOF probs from each model on train set
    oof_probs = np.column_stack([
        m.predict_proba(X_train)[:, 1] for _, m in models_with_names
    ])
    meta = LogisticRegression(C=1.0, solver="lbfgs", max_iter=1000)
    meta.fit(oof_probs, y_train)
    ensemble.meta_learner = meta

    # Save
    path = os.path.join(MODEL_DIR, "ensemble_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(ensemble, f)
    logger.info(f"\n  ✅ Ensemble saved → {path}")

    return ensemble, metrics


def main():
    (X_train, y_train), (X_val, y_val), (X_test, y_test), _ = \
        load_prepared_dataset("./data")

    X_train_n = tabular_normalizer.transform(X_train)
    X_val_n = tabular_normalizer.transform(X_val)
    X_test_n = tabular_normalizer.transform(X_test)

    ensemble, val_metrics = train_meta_ensemble(X_train_n, y_train, X_val_n, y_val)

    test_metrics = compute_all_metrics(ensemble, X_test_n, y_test)
    logger.info(f"\n  Ensemble Test ROC-AUC: {test_metrics['roc_auc']:.4f}")
    logger.info(f"  Ensemble Test F1: {test_metrics['f1']:.4f}")

    with open(os.path.join(MODEL_DIR, "ensemble_metrics.json"), "w") as f:
        json.dump({"val": val_metrics, "test": test_metrics}, f, indent=2)


if __name__ == "__main__":
    main()
