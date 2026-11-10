import os
import json
import time
import random
import tempfile
import argparse

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from tqdm import tqdm

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
# CONFIGURATION
# ==========================================================

TARGET_LENGTH = 300
REAL_LABEL = 0
FAKE_LABEL = 1

CHECKPOINT_NAME = "best_rppg_model.pt"
HISTORY_NAME = "training_history.json"


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    print(f"[INFO] Random seed: {seed}")


# ==========================================================
# DEVICE INFORMATION
# ==========================================================

def select_device(device_name="auto"):

    if device_name == "auto":
        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    elif device_name == "cuda":

        if not torch.cuda.is_available():
            raise RuntimeError(
                "CUDA was requested but is unavailable. "
                "Check your PyTorch CUDA installation."
            )

        device = torch.device("cuda:0")

    else:
        device = torch.device("cpu")

    return device


def print_device_information(device):

    print("\n========== DEVICE INFORMATION ==========")

    print(f"Selected device: {device}")

    if device.type == "cuda":

        print(f"GPU: {torch.cuda.get_device_name(device)}")
        print(f"CUDA version: {torch.version.cuda}")

        memory = (
            torch.cuda.get_device_properties(device).total_memory
            / (1024 ** 3)
        )

        print(f"GPU memory: {memory:.2f} GB")

    else:
        print("Running on CPU.")

    print("========================================\n")


# ==========================================================
# CHECKPOINT SAVING
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

    checkpoint = {
        "epoch": epoch,
        "best_val_loss": best_val_loss,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
    }

    temporary_path = None

    try:

        with tempfile.NamedTemporaryFile(
            dir=checkpoint_path.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary_file:

            temporary_path = temporary_file.name

        torch.save(checkpoint, temporary_path)

        os.replace(
            temporary_path,
            checkpoint_path,
        )

        print(
            f"[CHECKPOINT] Saved best model: {checkpoint_path}",
            flush=True,
        )

    finally:

        if (
            temporary_path is not None
            and os.path.exists(temporary_path)
        ):
            os.remove(temporary_path)


def load_checkpoint(model, checkpoint_path, device):

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True,
    )

    if (
        isinstance(checkpoint, dict)
        and "model_state_dict" in checkpoint
    ):

        model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        print(
            f"[INFO] Loaded best checkpoint from epoch "
            f"{checkpoint.get('epoch', 'unknown')}"
        )

    else:

        # Support older checkpoints containing only state_dict.
        model.load_state_dict(checkpoint)

        print("[INFO] Loaded legacy model checkpoint.")

    return model


# ==========================================================
# DATASET CREATION
# ==========================================================

def create_datasets(data_dir):

    data_dir = Path(data_dir).resolve()

    print("\n========== LOADING DATASETS ==========")

    train_path = data_dir / "train"
    val_path = data_dir / "val"
    test_path = data_dir / "test"

    # Support both val and validation folder names.
    if not val_path.is_dir():
        val_path = data_dir / "validation"

    required_paths = {
        "training": train_path,
        "validation": val_path,
        "testing": test_path,
    }

    for name, path in required_paths.items():

        if not path.is_dir():
            raise FileNotFoundError(
                f"{name.capitalize()} dataset directory not found: {path}"
            )

    print("\n[1/3] Loading training dataset...")

    train_dataset = RPPGDataset(
        root_dir=str(train_path),
        target_length=TARGET_LENGTH,
    )

    print("\n[2/3] Loading validation dataset...")

    val_dataset = RPPGDataset(
        root_dir=str(val_path),
        target_length=TARGET_LENGTH,
    )

    print("\n[3/3] Loading testing dataset...")

    test_dataset = RPPGDataset(
        root_dir=str(test_path),
        target_length=TARGET_LENGTH,
    )

    datasets = {
        "Training": train_dataset,
        "Validation": val_dataset,
        "Testing": test_dataset,
    }

    print("\n========== DATASET SUMMARY ==========")

    for name, dataset in datasets.items():

        if len(dataset) == 0:
            raise ValueError(
                f"{name} dataset contains no samples."
            )

        print(f"{name} samples: {len(dataset)}")

    print("=====================================\n")

    return (
        train_dataset,
        val_dataset,
        test_dataset,
    )


# ==========================================================
# CLASS DISTRIBUTION
# ==========================================================

def print_class_distribution(dataset, name):

    real_count = 0
    fake_count = 0

    # The current RPPGDataset stores samples as (path, label).
    if hasattr(dataset, "samples"):

        for _, label in dataset.samples:

            if int(label) == REAL_LABEL:
                real_count += 1

            elif int(label) == FAKE_LABEL:
                fake_count += 1

    else:

        # Fallback for datasets without a samples attribute.
        for index in range(len(dataset)):

            _, label = dataset[index]

            label = int(label)

            if label == REAL_LABEL:
                real_count += 1

            elif label == FAKE_LABEL:
                fake_count += 1

    print(f"\n{name.upper()} CLASS DISTRIBUTION")
    print(f"Real: {real_count}")
    print(f"Fake: {fake_count}")
    print(f"Total: {real_count + fake_count}")

    if real_count == 0 or fake_count == 0:
        print(
            f"[WARNING] {name} dataset does not contain "
            "both classes."
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

    print("\n========== DATALOADERS ==========")

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

    if x.size(1) != TARGET_LENGTH:
        raise ValueError(
            f"Expected {TARGET_LENGTH} temporal samples, "
            f"got {x.size(1)}"
        )

    if not torch.isfinite(x).all().item():
        raise ValueError(
            "Input contains NaN or infinite values."
        )

    if not torch.isfinite(y).all().item():
        raise ValueError(
            "Labels contain NaN or infinite values."
        )

    if not torch.all((y == REAL_LABEL) | (y == FAKE_LABEL)).item():
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
# CLASSIFICATION METRICS
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
    phase="Validation",
    epoch=None,
    total_epochs=None,
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

    progress_description = phase

    if epoch is not None and total_epochs is not None:
        progress_description = (
            f"Epoch {epoch}/{total_epochs} - {phase}"
        )

    context = (
        torch.enable_grad()
        if is_training
        else torch.inference_mode()
    )

    with context:

        progress_bar = tqdm(
            loader,
            desc=progress_description,
            unit="batch",
            leave=True,
        )

        for x, y in progress_bar:

            x, y = prepare_batch(
                x,
                y,
                device,
            )

            if is_training:

                optimizer.zero_grad(
                    set_to_none=True
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

            if not torch.isfinite(loss).item():
                raise ValueError(
                    f"{phase} loss became NaN or infinite."
                )

            if is_training:

                loss.backward()
                optimizer.step()

            batch_count = y.size(0)

            total_loss += loss.item() * batch_count
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

            current_loss = total_loss / total_samples

            progress_bar.set_postfix(
                loss=f"{current_loss:.4f}"
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

    return {
        "loss": average_loss,
        **metrics,
        "labels": all_labels,
        "predictions": all_predictions,
        "time": elapsed,
    }


# ==========================================================
# SAVE TRAINING HISTORY
# ==========================================================

def save_training_history(history, history_path):

    history_path = Path(history_path)

    history_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temporary_path = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=history_path.parent,
            suffix=".tmp",
            delete=False,
        ) as temporary_file:

            temporary_path = temporary_file.name

            json.dump(
                history,
                temporary_file,
                indent=4,
            )

        os.replace(
            temporary_path,
            history_path,
        )

        print(
            f"[HISTORY] Saved: {history_path}"
        )

    finally:

        if (
            temporary_path is not None
            and os.path.exists(temporary_path)
        ):
            os.remove(temporary_path)


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

    # Always evaluate the best validation checkpoint.
    model = load_checkpoint(
        model,
        checkpoint_path,
        device,
    )

    # One pass through the test dataset.
    test_results = run_epoch(
        model=model,
        loader=loader,
        criterion=criterion,
        device=device,
        phase="Testing",
    )

    all_labels = test_results["labels"]
    all_predictions = test_results["predictions"]

    matrix = confusion_matrix(
        all_labels,
        all_predictions,
        labels=[REAL_LABEL, FAKE_LABEL],
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

    print(f"Test loss:      {test_results['loss']:.4f}")
    print(f"Test accuracy:  {test_results['accuracy']:.4f}")
    print(f"Test precision: {test_results['precision']:.4f}")
    print(f"Test recall:    {test_results['recall']:.4f}")
    print(f"Test F1-score:  {test_results['f1']:.4f}")

    print("=========================================\n")

    return {
        "loss": test_results["loss"],
        "accuracy": test_results["accuracy"],
        "precision": test_results["precision"],
        "recall": test_results["recall"],
        "f1": test_results["f1"],
        "confusion_matrix": matrix.tolist(),
    }


# ==========================================================
# MAIN TRAINING FUNCTION
# ==========================================================

def train_model(
    data_dir,
    epochs=20,
    batch_size=4,
    lr=1e-3,
    save_dir="model/checkpoints",
    device_name="auto",
    seed=42,
):

    # ------------------------------------------------------
    # VALIDATE CONFIGURATION
    # ------------------------------------------------------

    if epochs < 1:
        raise ValueError("epochs must be at least 1.")

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1.")

    if not np.isfinite(lr) or lr <= 0:
        raise ValueError("Learning rate must be finite and positive.")

    set_seed(seed)

    # ------------------------------------------------------
    # SELECT DEVICE
    # ------------------------------------------------------

    device = select_device(device_name)

    print_device_information(device)

    # ------------------------------------------------------
    # TRAINING CONFIGURATION
    # ------------------------------------------------------

    print("========== TRAINING CONFIGURATION ==========")

    print(f"Epochs:        {epochs}")
    print(f"Batch size:    {batch_size}")
    print(f"Learning rate: {lr}")
    print(f"Data directory: {data_dir}")
    print(f"Device:        {device}")

    print("Model: RPPG1DCNN")
    print("Input channels: 1")
    print(f"Signal length: {TARGET_LENGTH}")
    print("Real label: 0")
    print("Fake label: 1")

    print("=============================================\n")

    # ------------------------------------------------------
    # LOAD DATASETS
    # ------------------------------------------------------

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = create_datasets(data_dir)

    # Print class distributions.
    print_class_distribution(
        train_dataset,
        "Training",
    )

    print_class_distribution(
        val_dataset,
        "Validation",
    )

    print_class_distribution(
        test_dataset,
        "Testing",
    )

    # ------------------------------------------------------
    # CREATE DATALOADERS
    # ------------------------------------------------------

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

    # ------------------------------------------------------
    # INITIALIZE MODEL
    # ------------------------------------------------------

    print("[INFO] Initializing RPPG1DCNN...")

    model = RPPG1DCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr,
    )

    print("[INFO] Model initialized successfully.")

    # ------------------------------------------------------
    # CHECKPOINT AND HISTORY PATHS
    # ------------------------------------------------------

    save_dir = Path(save_dir)

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = save_dir / CHECKPOINT_NAME
    history_path = save_dir / HISTORY_NAME

    best_val_loss = float("inf")
    best_epoch = 0

    history = {
        "configuration": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "target_length": TARGET_LENGTH,
            "device": str(device),
            "seed": seed,
            "model": "RPPG1DCNN",
        },
        "epochs": [],
    }

    # ------------------------------------------------------
    # TRAINING
    # ------------------------------------------------------

    training_start = time.time()

    print("\n========== STARTING TRAINING ==========\n")

    for epoch in range(1, epochs + 1):

        print("\n" + "=" * 60)
        print(f"EPOCH {epoch}/{epochs}")
        print("=" * 60)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # --------------------------------------------------
        # TRAIN
        # --------------------------------------------------

        train_results = run_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            device=device,
            optimizer=optimizer,
            phase="Training",
            epoch=epoch,
            total_epochs=epochs,
        )

        # --------------------------------------------------
        # VALIDATE
        # --------------------------------------------------

        val_results = run_epoch(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            phase="Validation",
            epoch=epoch,
            total_epochs=epochs,
        )

        # --------------------------------------------------
        # EPOCH SUMMARY
        # --------------------------------------------------

        print("\n========== EPOCH RESULTS ==========")

        print(f"Epoch: {epoch}/{epochs}")

        print(f"Train loss:      {train_results['loss']:.4f}")
        print(f"Train accuracy:  {train_results['accuracy']:.4f}")
        print(f"Train precision: {train_results['precision']:.4f}")
        print(f"Train recall:    {train_results['recall']:.4f}")
        print(f"Train F1-score:  {train_results['f1']:.4f}")

        print(f"Validation loss:      {val_results['loss']:.4f}")
        print(f"Validation accuracy:  {val_results['accuracy']:.4f}")
        print(f"Validation precision: {val_results['precision']:.4f}")
        print(f"Validation recall:    {val_results['recall']:.4f}")
        print(f"Validation F1-score:  {val_results['f1']:.4f}")

        print("===================================\n")

        # --------------------------------------------------
        # SAVE HISTORY
        # --------------------------------------------------

        epoch_record = {
            "epoch": epoch,

            "train_loss": train_results["loss"],
            "train_accuracy": train_results["accuracy"],
            "train_precision": train_results["precision"],
            "train_recall": train_results["recall"],
            "train_f1": train_results["f1"],

            "validation_loss": val_results["loss"],
            "validation_accuracy": val_results["accuracy"],
            "validation_precision": val_results["precision"],
            "validation_recall": val_results["recall"],
            "validation_f1": val_results["f1"],

            "train_time_seconds": train_results["time"],
            "validation_time_seconds": val_results["time"],
        }

        history["epochs"].append(epoch_record)

        # --------------------------------------------------
        # SAVE BEST MODEL
        # --------------------------------------------------

        if val_results["loss"] < best_val_loss:

            best_val_loss = val_results["loss"]
            best_epoch = epoch

            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val_loss=best_val_loss,
                checkpoint_path=checkpoint_path,
            )

            print(
                f"[INFO] New best validation loss: "
                f"{best_val_loss:.4f}"
            )

        # Save history after every epoch.
        save_training_history(
            history,
            history_path,
        )

        # --------------------------------------------------
        # GPU MEMORY
        # --------------------------------------------------

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

    # ------------------------------------------------------
    # TRAINING COMPLETE
    # ------------------------------------------------------

    elapsed = time.time() - training_start

    history["best_epoch"] = best_epoch
    history["best_validation_loss"] = best_val_loss
    history["total_training_time_seconds"] = elapsed

    save_training_history(
        history,
        history_path,
    )

    print("\n========== TRAINING COMPLETE ==========")

    print(f"Training time: {elapsed / 60:.2f} minutes")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation loss: {best_val_loss:.4f}")

    print(f"Best checkpoint: {checkpoint_path}")
    print(f"Training history: {history_path}")

    print("========================================\n")

    # ------------------------------------------------------
    # FINAL TESTING
    # ------------------------------------------------------

    test_results = test_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        checkpoint_path=checkpoint_path,
    )

    print("[INFO] FALSEFACET rPPG training finished.")

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

    if not np.isfinite(args.lr) or args.lr <= 0:
        parser.error("--lr must be positive and finite.")

    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        device_name=args.device,
        seed=args.seed,
    )