import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve


def auc_score(y_true, proba):
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)
    return float(roc_auc_score(y_true, proba))


def choose_threshold_min_recall(y_true, proba, min_recall: float = 0.55):
    y_true = np.asarray(y_true).astype(int)
    proba = np.asarray(proba).astype(float)

    precision, recall, thresholds = precision_recall_curve(y_true, proba)

    valid_indices = np.where(recall >= float(min_recall))[0]

    # Select threshold that gives maximum precision while satisfying minimum recall
    if len(valid_indices) > 0:
        best_idx = int(valid_indices[np.argmax(precision[valid_indices])])
        best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        chosen_by = "max_precision_subject_to_min_recall"
        
    # Fallback to max F1 score if no threshold meets the minimum recall
    else:
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_idx = int(np.argmax(f1_scores))
        best_threshold = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.5
        chosen_by = "max_f1_fallback"

    info = {
        "chosen_by": chosen_by,
        "min_recall_target": float(min_recall),
        "precision_at_threshold": float(precision[best_idx]),
        "recall_at_threshold": float(recall[best_idx]),
    }
    return best_threshold, info


def apply_threshold(proba, threshold):
    proba = np.asarray(proba).astype(float)
    return (proba >= float(threshold)).astype(int)
