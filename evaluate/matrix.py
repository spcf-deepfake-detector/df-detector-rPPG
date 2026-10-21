
import numpy as np
from typing import Dict, Any, Union
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


def compute_all_metrics(
    y_true: Union[np.ndarray, list],
    y_scores: Union[np.ndarray, list],
    threshold: float = 0.5
) -> Dict[str, Any]:
    """Compute binary classification metrics for FalseFacet rPPG.

    Labels:
        0 = REAL
        1 = FAKE

    y_scores must contain probabilities for the FAKE class.
    AUC is NaN if it cannot be calculated.
    """

    # Convert inputs to flat arrays.
    y_true_arr = np.asarray(y_true).flatten()
    y_scores_arr = np.asarray(y_scores, dtype=np.float64).flatten()

    # Validate threshold.
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a finite value between 0 and 1.")

    # Validate lengths.
    if len(y_true_arr) != len(y_scores_arr):
        raise ValueError(
            f"Input length mismatch: y_true has {len(y_true_arr)} samples, "
            f"but y_scores has {len(y_scores_arr)} samples."
        )

    # Empty input: return zero counts and undefined AUC.
    if len(y_true_arr) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "f1": 0.0,
            "auc": float("nan"),
            "tp": 0,
            "tn": 0,
            "fp": 0,
            "fn": 0
        }

    # Validate labels before converting to integers.
    if not np.all(np.isin(y_true_arr, [0, 1])):
        raise ValueError("Labels must be 0 (REAL) or 1 (FAKE).")

    y_true_arr = y_true_arr.astype(np.int32)

    # Validate prediction scores.
    if not np.all(np.isfinite(y_scores_arr)):
        raise ValueError("Prediction scores must all be finite.")

    if np.any((y_scores_arr < 0.0) | (y_scores_arr > 1.0)):
        raise ValueError(
            "Prediction scores must be probabilities between 0 and 1."
        )

    # Convert probabilities into predicted labels.
    y_pred_arr = (y_scores_arr >= threshold).astype(np.int32)

    # Calculate standard classification metrics.
    accuracy = accuracy_score(y_true_arr, y_pred_arr)
    precision = precision_score(
        y_true_arr, y_pred_arr, zero_division=0
    )
    recall = recall_score(
        y_true_arr, y_pred_arr, zero_division=0
    )
    f1 = f1_score(
        y_true_arr, y_pred_arr, zero_division=0
    )

    # Confusion matrix: rows are true labels, columns are predictions.
    cm = confusion_matrix(y_true_arr, y_pred_arr, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    # Specificity: proportion of REAL videos correctly identified.
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0

    # AUC requires both classes to be present.
    if len(np.unique(y_true_arr)) == 2:
        auc = float(roc_auc_score(y_true_arr, y_scores_arr))
    else:
        auc = float("nan")

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "specificity": float(specificity),
        "f1": float(f1),
        "auc": auc,
        "tp": int(tp),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn)
    }


if __name__ == "__main__":
    print("[i] Testing evaluate/metrics.py")

    # 0 = REAL, 1 = FAKE
    mock_true = [0, 0, 0, 1, 1, 1]
    mock_scores = [0.1, 0.2, 0.8, 0.9, 0.7, 0.3]

    results = compute_all_metrics(
        mock_true,
        mock_scores,
        threshold=0.5
    )

    print("-" * 40)
    print("MOCK EVALUATION OUTPUT")
    print("-" * 40)

    for name, value in results.items():
        if isinstance(value, (float, np.floating)):
            print(f"{name:<15}: {value:.4f}")
        else:
            print(f"{name:<15}: {value}")

    print("-" * 40)
