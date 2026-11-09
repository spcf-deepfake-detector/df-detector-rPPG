import os
import time
import random
import tempfile
import argparse

import numpy as np
import torch
import torch.nn as nn

from pathlib import Path
from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from model.rppg_model import RPPG1DCNN
from training.dataset import RPPGDataset


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"[i] Random seed: {seed}")


# ==========================================================
# DEVICE INFORMATION
# ==========================================================

def print_device_information(device):

    print("\n========== DEVICE INFORMATION ==========")

    print(f"Selected device: {device}")

    if device.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(device)}"
        )

        print(
            f"CUDA version: {torch.version.cuda}"
        )

        memory = (
            torch.cuda.get_device_properties(device)
            .total_memory / (1024 ** 3)
        )

        print(f"GPU memory: {memory:.2f} GB")

    else:

        print("Running on CPU.")

    print("========================================\n")


# ==========================================================
# SAFE CHECKPOINT SAVING
# ==========================================================

def save_checkpoint(
    model,
    optimizer,
    epoch,
    best_val_loss,
    checkpoint_path,
):

    checkpoint_path = Path(checkpoint_path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = None

    checkpoint = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    try:

        with tempfile.NamedTemporaryFile(
            dir=checkpoint_path.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary_file:

            temporary_path = temporary_file.name

        torch.save(
            checkpoint,
            temporary_path,
        )

        os.replace(
            temporary_path,
            checkpoint_path,
        )

        print(
            f"[CHECKPOINT] Saved: {checkpoint_path}",
            flush=True,
        )

    finally:

        if (
            temporary_path is not None
            and os.path.exists(temporary_path)
        ):
            os.remove(temporary_path)


# ==========================================================
# DATASET CREATION
# ==========================================================

def create_datasets(data_dir):

    data_dir = Path(data_dir).resolve()

    print("\n========== LOADING DATASETS ==========")

    train_path = data_dir / "train"

    val_path = data_dir / "val"

    # Support older directory naming.
    if not val_path.is_dir():

        val_path = data_dir / "validation"

    test_path = data_dir / "test"

    for path in (
        train_path,
        val_path,
        test_path,
    ):

        if not path.is_dir():

            raise FileNotFoundError(
                f"Dataset directory not found: {path}"
            )

    print("\n[1/3] Loading training dataset...")

    train_dataset = RPPGDataset(
        root_dir=str(train_path),
        target_length=300,
    )

    print("\n[2/3] Loading validation dataset...")

    val_dataset = RPPGDataset(
        root_dir=str(val_path),
        target_length=300,
    )

    print("\n[3/3] Loading testing dataset...")

    test_dataset = RPPGDataset(
        root_dir=str(test_path),
        target_length=300,
    )

    print("\n========== DATASET SUMMARY ==========")

    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(val_dataset)}")
    print(f"Testing samples:    {len(test_dataset)}")

    print("=====================================\n")

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


# ==========================================================
# DATALOADERS
# ==========================================================

def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size,
    device,
):

    use_cuda = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
        drop_last=False,
    )

    print("========== DATALOADERS ==========")

    print(f"Training batches:   {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Testing batches:    {len(test_loader)}")

    print(f"Batch size: {batch_size}")
    print(f"Pinned memory: {use_cuda}")

    print("=================================\n")

    return (
        train_loader,
        val_loader,
        test_loader,
    )


# ==========================================================
# BATCH VALIDATION
# ==========================================================

def prepare_batch(x, y, device):

    if not torch.is_tensor(x):
        raise TypeError("Input must be a PyTorch tensor.")

    if not torch.is_tensor(y):
        raise TypeError("Labels must be PyTorch tensors.")

    if x.ndim != 2:
        raise ValueError(
            f"Expected input shape (B,T), got {tuple(x.shape)}"
        )

    if x.size(1) != 300:
        raise ValueError(
            f"Expected 300 temporal samples, got {x.size(1)}"
        )

    if not torch.isfinite(x).all():
        raise ValueError(
            "Input contains NaN or infinite values."
        )

    if not torch.isfinite(y).all():
        raise ValueError(
            "Labels contain NaN or infinite values."
        )

    if not torch.all((y == 0) | (y == 1)):
        raise ValueError(
            "Labels must contain only 0 or 1."
        )

    x = x.to(
        device,
        dtype=torch.float32,
        non_blocking=device.type == "cuda",
    )

    y = y.to(
        device,
        dtype=torch.float32,
        non_blocking=device.type == "cuda",
    ).view(-1)

    return x, y


# ==========================================================
# CALCULATE CLASSIFICATION METRICS
# ==========================================================

def calculate_metrics(all_labels, all_predictions):

    return {
        "accuracy": accuracy_score(
            all_labels,
            all_predictions,
        ),

        "precision": precision_score(
            all_labels,
            all_predictions,
            zero_division=0,
        ),

        "recall": recall_score(
            all_labels,
            all_predictions,
            zero_division=0,
        ),

        "f1": f1_score(
            all_labels,
            all_predictions,
            zero_division=0,
        ),
    }


# ==========================================================
# TRAINING AND EVALUATION LOOP
# ==========================================================

def run_epoch(
    model,
    loader,
    criterion,
    device,
    optimizer=None,
    epoch=None,
    total_epochs=None,
    phase="validation",
):

    is_training = optimizer is not None

    if is_training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    total_samples = 0

    all_labels = []

    all_predictions = []

    start_time = time.time()

    context = torch.enable_grad if is_training else torch.inference_mode

    with context():

        for batch_idx, (x, y) in enumerate(
            loader,
            start=1,
        ):

            x, y = prepare_batch(
                x,
                y,
                device,
            )

            if is_training:

                optimizer.zero_grad(
                    set_to_none=True,
                )

            logits = model(x).view(-1)

            if logits.shape != y.shape:
                raise ValueError(
                    f"Logits shape {logits.shape} "
                    f"does not match labels {y.shape}."
                )

            loss = criterion(
                logits,
                y,
            )

            if not torch.isfinite(loss):
                raise ValueError(
                    f"{phase} loss became NaN or infinite."
                )

            if is_training:

                loss.backward()

                optimizer.step()

            batch_count = y.size(0)

            total_loss += (
                loss.item() * batch_count
            )

            total_samples += batch_count

            probabilities = torch.sigmoid(logits)

            predictions = (
                probabilities >= 0.5
            ).long()

            all_predictions.extend(
                predictions.detach().cpu().tolist()
            )

            all_labels.extend(
                y.detach().long().cpu().tolist()
            )

            print(
                f"[{phase.upper()}] "
                f"Batch {batch_idx}/{len(loader)}",
                flush=True,
            )

    if total_samples == 0:
        raise ValueError(
            f"No samples processed during {phase}."
        )

    average_loss = total_loss / total_samples

    metrics = calculate_metrics(
        all_labels,
        all_predictions,
    )

    elapsed = time.time() - start_time

    print(f"\n========== {phase.upper()} SUMMARY ==========")

    print(f"Loss:      {average_loss:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")

    print(f"Samples:   {total_samples}")
    print(f"Time:      {elapsed:.2f} seconds")

    print("============================================\n")

    return average_loss, metrics


# ==========================================================
# LOAD BEST CHECKPOINT
# ==========================================================

def load_checkpoint(model, checkpoint_path, device):

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    # Support both the new checkpoint format
    # and older checkpoints containing only state_dict.
    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print(
            f"[i] Loaded best checkpoint from epoch "
            f"{checkpoint['epoch']}"
        )

    else:

        model.load_state_dict(checkpoint)

        print("[i] Loaded legacy model checkpoint.")

    return model


# ==========================================================
# FINAL TESTING
# ==========================================================

def test_model(
    model,
    loader,
    criterion,
    device,
    checkpoint_path,
):

    print("\n========== FINAL TESTING ==========")

    model = load_checkpoint(
        model,
        checkpoint_path,
        device,
    )

    test_loss, metrics = run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        phase="test",
    )

    # Collect predictions for confusion matrix.
    all_labels = []
    all_predictions = []

    model.eval()

    with torch.inference_mode():

        for x, y in loader:

            x, y = prepare_batch(
                x,
                y,
                device,
            )

            logits = model(x).view(-1)

            predictions = (
                torch.sigmoid(logits) >= 0.5
            ).long()

            all_predictions.extend(
                predictions.cpu().tolist()
            )

            all_labels.extend(
                y.long().cpu().tolist()
            )

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[0, 1],
    )

    print("\n========== CONFUSION MATRIX ==========")

    print("                 Predicted")

    print("                 Real   Fake")

    print(
        f"Actual Real      {matrix[0][0]:5d}  "
        f"{matrix[0][1]:5d}"
    )

    print(
        f"Actual Fake      {matrix[1][0]:5d}  "
        f"{matrix[1][1]:5d}"
    )

    print("=======================================\n")

    print("========== FINAL TEST RESULTS ==========")

    print(f"Test loss:      {test_loss:.4f}")
    print(f"Test accuracy:  {metrics['accuracy']:.4f}")
    print(f"Test precision: {metrics['precision']:.4f}")
    print(f"Test recall:    {metrics['recall']:.4f}")
    print(f"Test F1-score:  {metrics['f1']:.4f}")

    print("=========================================\n")

    return {
        "loss": test_loss,
        **metrics,
        "confusion_matrix": matrix.tolist(),
    }


# ==========================================================
# MAIN TRAINING FUNCTION
# ==========================================================

def train_model(
    data_dir,
    epochs=20,
    batch_size=2,
    lr=1e-3,
    save_dir="model/checkpoints",
    device_name="auto",
    seed=42,
):

    # ------------------------------------------
    # VALIDATE CONFIGURATION
    # ------------------------------------------

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if lr <= 0:
        raise ValueError("Learning rate must be positive.")

    set_seed(seed)

    # ------------------------------------------
    # SELECT DEVICE
    # ------------------------------------------

    if device_name == "auto":

        device = torch.device(
            "cuda:0"
            if torch.cuda.is_available()
            else "cpu"
        )

    elif device_name == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA requested but unavailable."
            )

        device = torch.device("cuda:0")

    else:

        device = torch.device("cpu")

    print_device_information(device)

    # ------------------------------------------
    # CONFIGURATION
    # ------------------------------------------

    print("========== TRAINING CONFIGURATION ==========")

    print(f"Epochs:       {epochs}")
    print(f"Batch size:   {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Data directory: {data_dir}")
    print(f"Device:       {device}")

    print("Model: RPPG1DCNN")
    print("Input channels: 1")
    print("Signal length: 300")
    print("Real label: 0")
    print("Fake label: 1")

    print("=============================================\n")

    # ------------------------------------------
    # LOAD DATASETS
    # ------------------------------------------

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = create_datasets(data_dir)

    # ------------------------------------------
    # CREATE DATALOADERS
    # ------------------------------------------

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders(
        train_dataset,
        val_dataset,
        test_dataset,
        batch_size,
        device,
    )

    # ------------------------------------------
    # INITIALIZE MODEL
    # ------------------------------------------

    print("[i] Initializing RPPG1DCNN...")

    model = RPPG1DCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    print("[i] Model initialized successfully.")

    # ------------------------------------------
    # CHECKPOINT CONFIGURATION
    # ------------------------------------------

    os.makedirs(
        save_dir,
        exist_ok=True,
    )

    checkpoint_path = os.path.join(
        save_dir,
        "best_rppg_model.pt",
    )

    best_val_loss = float("inf")

    # ------------------------------------------
    # TRAINING
    # ------------------------------------------

    training_start = time.time()

    for epoch in range(1, epochs + 1):

        print("\n" + "=" * 55)

        print(f"TRAINING EPOCH {epoch}/{epochs}")

        print("=" * 55)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # TRAIN
        train_loss, train_metrics = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            epoch=epoch,
            total_epochs=epochs,
            phase="training",
        )

        # VALIDATE
        val_loss, val_metrics = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=epochs,
            phase="validation",
        )

        # --------------------------------------
        # EPOCH SUMMARY
        # --------------------------------------

        print("\n========== EPOCH RESULTS ==========")

        print(f"Epoch: {epoch}/{epochs}")

        print(f"Train loss: {train_loss:.4f}")

        print(
            f"Train accuracy: "
            f"{train_metrics['accuracy']:.4f}"
        )

        print(f"Validation loss: {val_loss:.4f}")

        print(
            f"Validation accuracy: "
            f"{val_metrics['accuracy']:.4f}"
        )

        print("===================================\n")

        # --------------------------------------
        # SAVE BEST MODEL
        # --------------------------------------

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                checkpoint_path=checkpoint_path,
            )

        # --------------------------------------
        # GPU MEMORY
        # --------------------------------------

        if device.type == "cuda":

            allocated = (
                torch.cuda.memory_allocated(device)
                / (1024 ** 3)
            )

            peak = (
                torch.cuda.max_memory_allocated(device)
                / (1024 ** 3)
            )

            print(
                f"[GPU MEMORY] "
                f"Allocated: {allocated:.2f} GB | "
                f"Peak: {peak:.2f} GB"
            )

    # ------------------------------------------
    # TRAINING COMPLETE
    # ------------------------------------------

    elapsed = time.time() - training_start

    print("\n========== TRAINING COMPLETE ==========")

    print(
        f"Training time: {elapsed / 60:.2f} minutes"
    )

    print(f"Best validation loss: {best_val_loss:.4f}")

    print(f"Best checkpoint: {checkpoint_path}")

    print("========================================\n")

    # ------------------------------------------
    # FINAL TESTING
    # ------------------------------------------

    test_results = test_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        checkpoint_path=checkpoint_path,
    )

    print("[i] FALSEFACET rPPG training finished.")

    return test_results


# ==========================================================
# COMMAND-LINE INTERFACE
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="FALSEFACET RPPG1DCNN Training"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="processed",
        help="Processed rPPG dataset directory.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate.",
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="model/checkpoints",
        help="Model checkpoint directory.",
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Training device.",
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )

    args = parser.parse_args()

    if args.epochs < 1:
        parser.error("--epochs must be at least 1.")

    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1.")

    if args.lr <= 0:
        parser.error("--lr must be positive.")

    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        device_name=args.device,
        seed=args.seed,
    )