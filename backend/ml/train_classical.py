"""
Classical ML Training Script
=============================
Trains Random Forest, XGBoost, and LightGBM with hyperparameter tuning.
Saves models as .pkl and exports ONNX versions.
Persists evaluation metrics to disk.
"""

import json
import logging
import os
import pickle
import time
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, roc_auc_score, confusion_matrix,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
import xgboost as xgb
import lightgbm as lgb

from pipeline.dataset_loader import load_prepared_dataset, INTERNAL_FEATURE_COLS
from pipeline.normalization import tabular_normalizer
from evaluation.metrics import compute_all_metrics

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger(__name__)

MODEL_DIR = "./models"
DATA_DIR = "./data"
os.makedirs(MODEL_DIR, exist_ok=True)


def save_model(model, name: str):
    path = os.path.join(MODEL_DIR, f"{name}_model.pkl")
    with open(path, "wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(f"  ✓ Saved {name} → {path}")
    return path


def train_random_forest(X_train, y_train, X_val, y_val) -> Tuple[object, Dict]:
    logger.info("=" * 50)
    logger.info("Training Random Forest...")
    t0 = time.time()

    param_grid = {
        "n_estimators": [200, 400],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "class_weight": ["balanced"],
    }

    base_rf = RandomForestClassifier(random_state=42, n_jobs=-1)
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid = GridSearchCV(
        base_rf, param_grid, cv=cv,
        scoring="roc_auc", n_jobs=-1, verbose=0,
    )
    grid.fit(X_train, y_train)
    best_rf = grid.best_estimator_

    # Calibrate probabilities
    calibrated_rf = CalibratedClassifierCV(best_rf, method="sigmoid", cv=5)
    calibrated_rf.fit(X_train, y_train)

    metrics = compute_all_metrics(calibrated_rf, X_val, y_val)
    metrics["training_time_sec"] = round(time.time() - t0, 2)
    metrics["best_params"] = grid.best_params_

    logger.info(f"  RF Val ROC-AUC: {metrics['roc_auc']:.4f} | Acc: {metrics['accuracy']:.4f}")
    save_model(calibrated_rf, "rf")

    return calibrated_rf, metrics


def train_xgboost(X_train, y_train, X_val, y_val) -> Tuple[object, Dict]:
    logger.info("=" * 50)
    logger.info("Training XGBoost...")
    t0 = time.time()

    scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_val, label=y_val)

    params = {
        "objective": "binary:logistic",
        "eval_metric": ["logloss", "auc"],
        "scale_pos_weight": scale_pos,
        "max_depth": 6,
        "eta": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 3,
        "gamma": 0.1,
        "lambda": 1.0,
        "alpha": 0.1,
        "seed": 42,
        "verbosity": 0,
    }

    evals_result = {}
    bst = xgb.train(
        params, dtrain,
        num_boost_round=1000,
        evals=[(dtrain, "train"), (dval, "val")],
        early_stopping_rounds=50,
        evals_result=evals_result,
        verbose_eval=False,
    )

    # Wrap in sklearn interface for compatibility
    xgb_clf = xgb.XGBClassifier(
        n_estimators=bst.best_iteration if bst.best_iteration > 0 else 100,
        max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=3, gamma=0.1,
        reg_lambda=1.0, reg_alpha=0.1,
        scale_pos_weight=scale_pos,
        use_label_encoder=False,
        eval_metric="logloss",
        random_state=42,
    )
    xgb_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Calibrate probabilities
    calibrated_xgb = CalibratedClassifierCV(xgb_clf, method="sigmoid", cv=5)
    calibrated_xgb.fit(X_train, y_train)

    metrics = compute_all_metrics(calibrated_xgb, X_val, y_val)
    metrics["training_time_sec"] = round(time.time() - t0, 2)
    metrics["best_iteration"] = int(bst.best_iteration)

    logger.info(f"  XGB Val ROC-AUC: {metrics['roc_auc']:.4f} | Acc: {metrics['accuracy']:.4f}")
    save_model(calibrated_xgb, "xgb")

    return calibrated_xgb, metrics


def train_lightgbm(X_train, y_train, X_val, y_val) -> Tuple[object, Dict]:
    logger.info("=" * 50)
    logger.info("Training LightGBM...")
    t0 = time.time()

    train_data = lgb.Dataset(X_train, label=y_train)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)

    params = {
        "objective": "binary",
        "metric": ["binary_logloss", "auc"],
        "boosting_type": "dart",
        "num_leaves": 63,
        "max_depth": -1,
        "learning_rate": 0.05,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "min_child_samples": 10,
        "reg_alpha": 0.1,
        "reg_lambda": 1.0,
        "is_unbalance": True,
        "verbose": -1,
        "seed": 42,
    }

    callbacks = [lgb.early_stopping(stopping_rounds=50, verbose=False),
                 lgb.log_evaluation(period=-1)]

    bst = lgb.train(
        params, train_data,
        num_boost_round=1000,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    lgbm_clf = lgb.LGBMClassifier(
        n_estimators=bst.best_iteration if bst.best_iteration > 0 else 100,
        boosting_type="dart",
        num_leaves=63,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_child_samples=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        is_unbalance=True,
        random_state=42,
        verbose=-1,
    )
    lgbm_clf.fit(X_train, y_train, eval_set=[(X_val, y_val)])

    # Calibrate probabilities
    calibrated_lgbm = CalibratedClassifierCV(lgbm_clf, method="sigmoid", cv=5)
    calibrated_lgbm.fit(X_train, y_train)

    metrics = compute_all_metrics(calibrated_lgbm, X_val, y_val)
    metrics["training_time_sec"] = round(time.time() - t0, 2)

    logger.info(f"  LGBM Val ROC-AUC: {metrics['roc_auc']:.4f} | Acc: {metrics['accuracy']:.4f}")
    save_model(calibrated_lgbm, "lgbm")

    return calibrated_lgbm, metrics


def main():
    logger.info("🧠 ParkVoice Classical ML Training Pipeline")
    logger.info(f"  Data dir: {DATA_DIR}")
    logger.info(f"  Model dir: {MODEL_DIR}")

    # ── Load data ─────────────────────────────────────────────────────────────
    (X_train, y_train), (X_val, y_val), (X_test, y_test), feat_cols = \
        load_prepared_dataset(DATA_DIR)

    logger.info(f"  Training samples: {len(X_train)}")
    logger.info(f"  Feature columns: {len(feat_cols)}")
    logger.info(f"  Class balance (train): {(y_train==1).sum()} PD / {(y_train==0).sum()} healthy")

    # ── Normalize ─────────────────────────────────────────────────────────────
    X_train_n = tabular_normalizer.fit_transform(X_train)
    X_val_n = tabular_normalizer.transform(X_val)
    X_test_n = tabular_normalizer.transform(X_test)

    # ── Train models ──────────────────────────────────────────────────────────
    all_metrics = {}

    rf, rf_metrics = train_random_forest(X_train_n, y_train, X_val_n, y_val)
    all_metrics["random_forest"] = rf_metrics

    xgb_model, xgb_metrics = train_xgboost(X_train_n, y_train, X_val_n, y_val)
    all_metrics["xgboost"] = xgb_metrics

    lgbm_model, lgbm_metrics = train_lightgbm(X_train_n, y_train, X_val_n, y_val)
    all_metrics["lightgbm"] = lgbm_metrics

    # ── Test set evaluation ───────────────────────────────────────────────────
    logger.info("\n" + "=" * 50)
    logger.info("Test Set Evaluation")
    for name, model in [("RF", rf), ("XGBoost", xgb_model), ("LightGBM", lgbm_model)]:
        test_metrics = compute_all_metrics(model, X_test_n, y_test)
        logger.info(
            f"  {name:10s} | "
            f"Acc={test_metrics['accuracy']:.3f} | "
            f"AUC={test_metrics['roc_auc']:.3f} | "
            f"F1={test_metrics['f1']:.3f} | "
            f"Sens={test_metrics['sensitivity']:.3f} | "
            f"Spec={test_metrics['specificity']:.3f}"
        )

    # ── Save metrics ──────────────────────────────────────────────────────────
    metrics_path = os.path.join(MODEL_DIR, "classical_metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    logger.info(f"\n✅ Metrics saved to {metrics_path}")

    return rf, xgb_model, lgbm_model, all_metrics


if __name__ == "__main__":
    main()
