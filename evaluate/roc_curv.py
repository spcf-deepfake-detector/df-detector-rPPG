import os
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc


def plot_roc_curve(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    save_path: str = "results/roc_curve.png"
) -> float:
    """Generate and save the ROC curve.

    Label convention:
        0 = REAL
        1 = FAKE

    y_scores must contain predicted probabilities for the FAKE class.
    Returns:
        Area under the ROC curve (AUC).
    """

    # Convert inputs to consistent NumPy arrays.
    y_true = np.asarray(y_true).ravel()
    y_scores = np.asarray(y_scores, dtype=float).ravel()

    # Validate input lengths.
    if len(y_true) != len(y_scores):
        raise ValueError(
            "y_true and y_scores must contain the same number of samples."
        )

    if len(y_true) == 0:
        raise ValueError("Cannot plot an ROC curve with no samples.")

    # Validate labels.
    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError("Labels must be 0 (REAL) or 1 (FAKE).")

    # Both classes are required for a meaningful ROC curve.
    if len(np.unique(y_true)) != 2:
        raise ValueError(
            "ROC curve requires both REAL and FAKE samples."
        )

    # Validate prediction scores.
    if not np.all(np.isfinite(y_scores)):
        raise ValueError("All prediction scores must be finite.")

    if np.any((y_scores < 0.0) | (y_scores > 1.0)):
        raise ValueError(
            "y_scores must be probabilities between 0 and 1. "
            "Apply sigmoid to model logits first."
        )

    # Calculate ROC curve and area under the curve.
    fpr, tpr, _ = roc_curve(y_true, y_scores, pos_label=1)
    roc_auc = auc(fpr, tpr)

    # Create the output directory if one is specified.
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Plot and save.
    plt.figure(figsize=(6, 5))
    plt.plot(
        fpr,
        tpr,
        color="darkorange",
        lw=2,
        label=f"ROC curve (AUC = {roc_auc:.3f})"
    )
    plt.plot(
        [0, 1],
        [0, 1],
        color="navy",
        lw=2,
        linestyle="--",
        label="Random classifier"
    )

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("FalseFacet rPPG ROC Curve")
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[+] ROC curve saved to {save_path}")
    print(f"[+] ROC AUC: {roc_auc:.4f}")

    return float(roc_auc)
