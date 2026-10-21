
import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    save_path: str = "results/heatmaps/confusion_matrix.png"
):
    """Generate and save a confusion matrix plot.

    Label convention:
        0 = REAL
        1 = FAKE
    """
    # Ensure the output directory exists.
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Convert inputs to NumPy arrays.
    y_true = np.asarray(y_true).ravel()
    y_pred = np.asarray(y_pred).ravel()

    # Validate input lengths.
    if len(y_true) != len(y_pred):
        raise ValueError(
            "y_true and y_pred must contain the same number of samples."
        )

    if len(y_true) == 0:
        raise ValueError("Cannot plot a confusion matrix with no samples.")

    # Calculate the confusion matrix: 0 = REAL, 1 = FAKE.
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])

    # Create and save the heatmap.
    plt.figure(figsize=(6, 5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["REAL", "FAKE"],
        yticklabels=["REAL", "FAKE"],
        cbar=True
    )

    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    plt.title("FalseFacet rPPG Confusion Matrix")
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[+] Confusion matrix plot saved to {save_path}")

    return cm
