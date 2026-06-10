"""
Evaluation Metrics
==================
Comprehensive classification and regression metrics for Parkinson's detection.
All metrics aligned with clinical reporting standards.
"""

import logging
from typing import Any, Dict, Optional

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    average_precision_score,
    matthews_corrcoef,
    cohen_kappa_score,
)

logger = logging.getLogger(__name__)


def compute_all_metrics(
    model: Any,
    X: np.ndarray,
    y: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Compute all classification metrics from a sklearn-compatible model.
    Includes sensitivity, specificity, PPV, NPV, MCC, Cohen's Kappa.
    """
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    return compute_metrics_from_arrays(y, preds, probs, threshold)


def compute_metrics_from_arrays(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: Optional[np.ndarray] = None,
    threshold: float = 0.5,
) -> Dict[str, float]:
    """Compute metrics from prediction arrays directly."""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    sensitivity = tp / (tp + fn + 1e-10)   # recall for positive class
    specificity = tn / (tn + fp + 1e-10)   # recall for negative class
    ppv = tp / (tp + fp + 1e-10)           # precision
    npv = tn / (tn + fn + 1e-10)           # negative predictive value

    metrics = {
        "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
        "precision": round(float(ppv), 4),
        "recall": round(float(sensitivity), 4),
        "f1": round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "sensitivity": round(float(sensitivity), 4),
        "specificity": round(float(specificity), 4),
        "ppv": round(float(ppv), 4),
        "npv": round(float(npv), 4),
        "mcc": round(float(matthews_corrcoef(y_true, y_pred)), 4),
        "cohen_kappa": round(float(cohen_kappa_score(y_true, y_pred)), 4),
        "tp": int(tp), "tn": int(tn), "fp": int(fp), "fn": int(fn),
        "threshold": threshold,
    }

    if y_prob is not None:
        try:
            metrics["roc_auc"] = round(float(roc_auc_score(y_true, y_prob)), 4)
            metrics["average_precision"] = round(
                float(average_precision_score(y_true, y_prob)), 4
            )
            from sklearn.metrics import brier_score_loss
            metrics["brier_score"] = round(float(brier_score_loss(y_true, y_prob)), 4)
        except Exception:
            metrics["roc_auc"] = 0.5
            metrics["average_precision"] = 0.0
            metrics["brier_score"] = 0.25

    return metrics


def compute_torch_metrics(
    model: Any, loader: Any, device: str = "cpu"
) -> Dict[str, float]:
    """
    Compute metrics for a PyTorch model from a DataLoader.
    Expects outputs to be class logits.
    """
    import torch
    model.eval()
    all_probs, all_labels = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            logits = model(X)
            probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(y.numpy())

    probs = np.array(all_probs)
    labels = np.array(all_labels)
    preds = (probs >= 0.5).astype(int)
    return compute_metrics_from_arrays(labels, preds, probs)


def find_optimal_threshold(
    y_true: np.ndarray, y_prob: np.ndarray, metric: str = "f1"
) -> Dict[str, float]:
    """
    Find the probability threshold that maximizes a given metric.
    Useful for clinical settings where sensitivity/specificity tradeoffs matter.
    """
    best_score = 0.0
    best_threshold = 0.5

    for thresh in np.linspace(0.1, 0.9, 81):
        preds = (y_prob >= thresh).astype(int)
        if metric == "f1":
            score = float(f1_score(y_true, preds, zero_division=0))
        elif metric == "sensitivity":
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            score = tp / (tp + fn + 1e-10)
        elif metric == "specificity":
            tn, fp, fn, tp = confusion_matrix(y_true, preds, labels=[0, 1]).ravel()
            score = tn / (tn + fp + 1e-10)
        elif metric == "balanced_accuracy":
            from sklearn.metrics import balanced_accuracy_score
            score = float(balanced_accuracy_score(y_true, preds))
        else:
            score = float(accuracy_score(y_true, preds))

        if score > best_score:
            best_score = score
            best_threshold = thresh

    return {
        "optimal_threshold": round(float(best_threshold), 3),
        f"best_{metric}": round(best_score, 4),
    }


def compute_bootstrap_ci(
    y_true: np.ndarray, y_prob: np.ndarray,
    n_bootstrap: int = 1000, alpha: float = 0.05,
) -> Dict[str, Dict[str, float]]:
    """
    Bootstrap 95% confidence intervals for key metrics.
    """
    rng = np.random.default_rng(42)
    metrics_boot = {
        "roc_auc": [], "f1": [], "sensitivity": [], "specificity": []
    }

    n = len(y_true)
    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, n)
        y_t = y_true[idx]
        y_p = y_prob[idx]
        y_preds = (y_p >= 0.5).astype(int)

        if len(np.unique(y_t)) < 2:
            continue

        try:
            metrics_boot["roc_auc"].append(roc_auc_score(y_t, y_p))
        except Exception:
            pass

        metrics_boot["f1"].append(f1_score(y_t, y_preds, zero_division=0))

        tn, fp, fn, tp = confusion_matrix(y_t, y_preds, labels=[0, 1]).ravel()
        metrics_boot["sensitivity"].append(tp / (tp + fn + 1e-10))
        metrics_boot["specificity"].append(tn / (tn + fp + 1e-10))

    result = {}
    for metric, values in metrics_boot.items():
        if values:
            arr = np.array(values)
            lo = float(np.percentile(arr, alpha / 2 * 100))
            hi = float(np.percentile(arr, (1 - alpha / 2) * 100))
            result[metric] = {
                "mean": round(float(np.mean(arr)), 4),
                "lower_95ci": round(lo, 4),
                "upper_95ci": round(hi, 4),
            }
    return result


def generate_evaluation_plots(y_true: np.ndarray, y_prob: np.ndarray, model_name: str, output_dir: str = "./evaluation/plots"):
    """
    Generate publication-ready ROC, Precision-Recall, and Calibration curves.
    """
    import os
    import matplotlib.pyplot as plt
    from sklearn.metrics import roc_curve, auc, precision_recall_curve
    from sklearn.calibration import calibration_curve

    os.makedirs(output_dir, exist_ok=True)
    
    # Styled plots
    plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # 1. ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    axes[0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[0].plot([0, 1], [0, 1], color='navy', lw=1.5, linestyle='--')
    axes[0].set_xlim([0.0, 1.0])
    axes[0].set_ylim([0.0, 1.05])
    axes[0].set_xlabel('False Positive Rate')
    axes[0].set_ylabel('True Positive Rate')
    axes[0].set_title(f'Receiver Operating Characteristic ({model_name})')
    axes[0].legend(loc="lower right")
    
    # 2. Precision-Recall Curve
    precision, recall, _ = precision_recall_curve(y_true, y_prob)
    avg_prec = average_precision_score(y_true, y_prob)
    axes[1].plot(recall, precision, color='teal', lw=2, label=f'PR curve (AP = {avg_prec:.3f})')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].set_title(f'Precision-Recall Curve ({model_name})')
    axes[1].legend(loc="lower left")
    
    # 3. Calibration Curve (Reliability Diagram)
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=5)
    axes[2].plot(prob_pred, prob_true, marker='o', linewidth=2, color='purple', label='Calibrated Model')
    axes[2].plot([0, 1], [0, 1], linestyle='--', color='gray', label='Perfectly Calibrated')
    axes[2].set_xlabel('Mean Predicted Probability')
    axes[2].set_ylabel('Fraction of Positives')
    axes[2].set_title(f'Calibration Curve ({model_name})')
    axes[2].legend(loc="upper left")
    
    plt.tight_layout()
    output_path = os.path.join(output_dir, f"{model_name.lower().replace(' ', '_')}_evaluation.png")
    plt.savefig(output_path, dpi=300)
    plt.close()
    logger.info(f"  ✓ Evaluation curves generated → {output_path}")
