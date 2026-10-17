
import os
import argparse
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from training.dataset import RPPGDataset
from model.rppg_model import RPPG1DCNN
from evaluate.metrics import compute_all_metrics
from evaluate.confusion_matrix import plot_confusion_matrix
from evaluate.roc_curve import plot_roc_curve


def load_checkpoint(model, checkpoint_path, device):
    """Load model weights from a state dict or common checkpoint dictionary."""
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        # Compatibility with older PyTorch versions.
        # Only load checkpoint files from a trusted source.
        checkpoint = torch.load(checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict):
        if "model_state_dict" in checkpoint:
            state_dict = checkpoint["model_state_dict"]
        elif "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint
    else:
        raise ValueError("Checkpoint does not contain a valid state dictionary.")

    model.load_state_dict(state_dict)
    return model


def run_evaluation(
    data_dir: str,
    checkpoint_path: str,
    batch_size: int = 4,
    threshold: float = 0.5,
    save_dir: str = "results"
):
    """Evaluate the trained rPPG classifier on an independent test dataset.

    Label convention:
        0 = REAL
        1 = FAKE
    Model scores must represent the probability of the FAKE class.
    """

    # 1. Validate arguments and paths.
    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if not np.isfinite(threshold) or not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between 0 and 1.")

    if not os.path.isdir(data_dir):
        raise FileNotFoundError(
            f"Test dataset directory does not exist: {data_dir}"
        )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Trained model checkpoint not found: {checkpoint_path}"
        )

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "heatmaps"), exist_ok=True)

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )

    print(f"[i] Execution device: {device}")
    print(f"[i] Test data directory: {data_dir}")
    print(f"[i] Model checkpoint: {checkpoint_path}")

    # 2. Load the test dataset.
    test_dataset = RPPGDataset(root_dir=data_dir)

    if len(test_dataset) == 0:
        raise ValueError(
            f"No video samples found in test directory: {data_dir}"
        )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    print(f"[i] Total test samples: {len(test_dataset)}")

    # 3. Load the trained model.
    model = RPPG1DCNN().to(device)
    model = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    print("[+] Trained model weights loaded successfully.")

    # 4. Run inference.
    y_true_list = []
    y_scores_list = []

    print("[i] Running inference on test samples...")

    with torch.no_grad():
        for x, y in test_loader:
            x = x.to(device=device, dtype=torch.float32)

            logits = model(x).reshape(-1)
            scores = torch.sigmoid(logits)

            scores = scores.cpu().numpy()
            labels = y.reshape(-1).cpu().numpy()

            if len(scores) != len(labels):
                raise ValueError(
                    "The number of predictions does not match "
                    "the number of ground-truth labels."
                )

            y_scores_list.extend(scores.tolist())
            y_true_list.extend(labels.tolist())

    y_true = np.asarray(y_true_list)
    y_scores = np.asarray(y_scores_list, dtype=np.float64)

    # 5. Validate predictions and labels.
    if len(y_true) != len(test_dataset):
        raise ValueError(
            "Collected label count does not match the dataset size."
        )

    if not np.all(np.isin(y_true, [0, 1])):
        raise ValueError(
            "Test labels must be 0 (REAL) or 1 (FAKE)."
        )

    y_true = y_true.astype(np.int32)

    if not np.all(np.isfinite(y_scores)):
        raise ValueError("Model produced non-finite prediction scores.")

    if np.any((y_scores < 0.0) | (y_scores > 1.0)):
        raise ValueError("Predicted probabilities must be between 0 and 1.")

    if len(np.unique(y_true)) != 2:
        raise ValueError(
            "The test dataset must contain both REAL and FAKE samples "
            "to calculate ROC-AUC meaningfully."
        )

    # 6. Calculate evaluation metrics.
    metrics = compute_all_metrics(
        y_true,
        y_scores,
        threshold=threshold
    )

    print("\n" + "=" * 55)
    print("             FALSEFACET rPPG EVALUATION")
    print("=" * 55)
    print(f"Accuracy         : {metrics['accuracy']:.4f}")
    print(f"Precision        : {metrics['precision']:.4f}")
    print(f"Recall           : {metrics['recall']:.4f}")
    print(f"Specificity      : {metrics['specificity']:.4f}")
    print(f"F1-Score         : {metrics['f1']:.4f}")
    print(f"ROC-AUC          : {metrics['auc']:.4f}")
    print("-" * 55)
    print(f"True Positives (TP)  : {metrics['tp']}")
    print(f"True Negatives (TN)  : {metrics['tn']}")
    print(f"False Positives (FP) : {metrics['fp']}")
    print(f"False Negatives (FN) : {metrics['fn']}")
    print("=" * 55)

    # 7. Save the metrics summary.
    metrics_df = pd.DataFrame([metrics])
    metrics_df["threshold"] = threshold
    metrics_df["test_samples"] = len(test_dataset)

    csv_out = os.path.join(save_dir, "evaluation_metrics.csv")
    metrics_df.to_csv(csv_out, index=False)
    print(f"[+] Metrics saved to: {csv_out}")

    # 8. Generate evaluation plots.
    y_pred = (y_scores >= threshold).astype(np.int32)

    cm_plot_path = os.path.join(
        save_dir, "heatmaps", "confusion_matrix.png"
    )
    roc_plot_path = os.path.join(save_dir, "roc_curve.png")

    plot_confusion_matrix(
        y_true,
        y_pred,
        save_path=cm_plot_path
    )

    plot_roc_curve(
        y_true,
        y_scores,
        save_path=roc_plot_path
    )

    print("[+] Evaluation completed successfully.")

    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FALSEFACET - rPPG Test Evaluation"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset/test",
        help="Test directory containing real/ and fake/ subfolders"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model/checkpoints/best_rppg_model.pt",
        help="Path to the trained model checkpoint"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Evaluation batch size"
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for classifying FAKE probabilities"
    )
    parser.add_argument(
        "--save_dir",
        type=str,
        default="results",
        help="Directory for evaluation reports and plots"
    )

    args = parser.parse_args()

    run_evaluation(
        data_dir=args.data_dir,
        checkpoint_path=args.checkpoint,
        batch_size=args.batch_size,
        threshold=args.threshold,
        save_dir=args.save_dir
    )
#python -m evaluate.evaluate
