
import pandas as pd
import numpy as np
from typing import List, Dict, Any


def perform_error_analysis(
    samples_metadata: List[Dict[str, Any]],
    y_true: np.ndarray,
    y_scores: np.ndarray,
    threshold: float = 0.5
) -> pd.DataFrame:
    """Identify false positives and false negatives with metadata.

    Label convention:
        0 = REAL
        1 = FAKE

    y_scores contains the model's predicted probability of FAKE.
    """

    # Convert inputs to consistent NumPy arrays.
    y_true = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores, dtype=float).ravel()

    # Validate the threshold.
    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be a finite value between 0 and 1.")

    # Validate input lengths.
    if not (
        len(samples_metadata) == len(y_true) == len(y_scores)
    ):
        raise ValueError(
            "samples_metadata, y_true, and y_scores "
            "must contain the same number of samples."
        )

    if len(y_true) == 0:
        raise ValueError("Cannot analyze an empty dataset.")

    # Validate labels and scores.
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("Labels must be 0 (REAL) or 1 (FAKE).")

    if not np.all(np.isfinite(y_scores)):
        raise ValueError("All prediction scores must be finite.")

    if np.any((y_scores < 0.0) | (y_scores > 1.0)):
        raise ValueError(
            "y_scores must be probabilities between 0 and 1."
        )

    # Convert FAKE probabilities into predicted labels.
    y_pred = (y_scores >= threshold).astype(int)

    records = []

    for meta, true_lbl, pred_lbl, score in zip(
        samples_metadata, y_true, y_pred, y_scores
    ):
        true_lbl = int(true_lbl)
        pred_lbl = int(pred_lbl)

        is_error = true_lbl != pred_lbl

        if true_lbl == 0 and pred_lbl == 1:
            error_type = "False Positive (REAL misclassified as FAKE)"
        elif true_lbl == 1 and pred_lbl == 0:
            error_type = "False Negative (FAKE misclassified as REAL)"
        else:
            error_type = "Correct"

        records.append({
            "video_path": meta.get("path", "N/A"),
            "true_label": "FAKE" if true_lbl == 1 else "REAL",
            "pred_label": "FAKE" if pred_lbl == 1 else "REAL",
            "score": float(score),
            "threshold": float(threshold),
            "signal_quality": meta.get("signal_quality", np.nan),
            "is_error": is_error,
            "error_type": error_type
        })

    df = pd.DataFrame(records)

    total_errors = int(df["is_error"].sum())
    total_samples = len(df)

    print(
        f"[i] Error Analysis Complete. "
        f"Total Errors: {total_errors} / {total_samples}"
    )

    return df
