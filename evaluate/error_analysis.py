import pandas as pd
import numpy as np
from typing import List, Dict, Any

def perform_error_analysis(samples_metadata: List[Dict[str, Any]], y_true: np.ndarray, y_scores: np.ndarray, threshold: float = 0.5) -> pd.DataFrame:
    """Identifies false positives and false negatives along with diagnostic factors."""
    y_pred = (y_scores >= threshold).astype(int)
    records = []

    for meta, true_lbl, pred_lbl, score in zip(samples_metadata, y_true, y_pred, y_scores):
        is_error = (true_lbl != pred_lbl)
        error_type = "Correct"
        if true_lbl == 0 and pred_lbl == 1:
            error_type = "False Positive (Real misclassified as Fake)"
        elif true_lbl == 1 and pred_lbl == 0:
            error_type = "False Negative (Fake misclassified as Real)"

        rec = {
            "video_path": meta.get("path", "N/A"),
            "true_label": "FAKE" if true_lbl == 1 else "REAL",
            "pred_label": "FAKE" if pred_lbl == 1 else "REAL",
            "score": float(score),
            "signal_quality": meta.get("signal_quality", 0.0),
            "is_error": is_error,
            "error_type": error_type
        }
        records.append(rec)

    df = pd.DataFrame(records)
    print(f"[i] Error Analysis Complete. Total Errors: {df['is_error'].sum()} / {len(df)}")
    return dfgit 