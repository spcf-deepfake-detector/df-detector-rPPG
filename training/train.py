import os
import json
import time
import random
import tempfile
import argparse

from pathlib import Path
from numbers import Integral, Real

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

DEFAULT_BATCH_SIZE = 2
DEFAULT_EPOCHS = 20
DEFAULT_LEARNING_RATE = 1e-3

NUM_WORKERS = 0

# Keep disabled until there is a reason to change it.
PIN_MEMORY = True


# ==========================================================
# CONFIGURATION VALIDATION
# ==========================================================

def validate_training_config(
    epochs,
    batch_size,
    learning_rate,
    seed,
):
    """Validate training parameters before allocating resources."""

    if (
        isinstance(epochs, bool)
        or not isinstance(epochs, Integral)
        or epochs < 1
    ):
        raise ValueError("epochs must be an integer >= 1.")

    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, Integral)
        or batch_size < 1
    ):
        raise ValueError("batch_size must be an integer >= 1.")

    if (
        isinstance(learning_rate, bool)
        or not isinstance(learning_rate, Real)
        or not np.isfinite(learning_rate)
        or learning_rate <= 0
    ):
        raise ValueError(
            "learning_rate must be a finite positive number."
        )

    if (
        isinstance(seed, bool)
        or not isinstance(seed, Integral)
        or seed < 0
    ):
        raise ValueError("seed must be a non-negative integer.")


# ==========================================================
# REPRODUCIBILITY
# ==========================================================

def set_seed(seed=42):
    """
    Set random seeds for reproducible training.

    Note:
        Exact reproducibility can still depend on CUDA,
        PyTorch, and hardware-specific operations.
    """

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
    """Select CPU or CUDA device."""

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

    elif device_name == "cpu":

        device = torch.device("cpu")

    else:
        raise ValueError(
            "device_name must be 'auto', 'cuda', or 'cpu'."
        )

    return device


def print_device_information(device):
    """Display selected hardware information."""

    print("\n========== DEVICE INFORMATION ==========")

    print(f"Selected device: {device}")

    if device.type == "cuda":

        print(
            f"GPU: {torch.cuda.get_device_name(device)}"
        )

        print(f"CUDA version: {torch.version.cuda}")

        properties = torch.cuda.get_device_properties(device)

        memory = properties.total_memory / (1024 ** 3)

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
    configuration=None,
):
    """
    Save the best model checkpoint atomically.

    The temporary file is created in the destination
    directory so os.replace() can perform the replacement
    without moving the file across filesystems.
    """

    checkpoint_path = Path(checkpoint_path)

    checkpoint_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint = {
        "epoch": int(epoch),
        "best_val_loss": float(best_val_loss),

        "model_state_dict": model.state_dict(),

        "optimizer_state_dict": optimizer.state_dict(),

        "model_name": "RPPG1DCNN",

        "target_length": TARGET_LENGTH,

        "real_label": REAL_LABEL,
        "fake_label": FAKE_LABEL,

        "configuration": configuration or {},
    }

    temporary_path = None

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

        # Flush the saved file before replacing the checkpoint.
        with open(temporary_path, "rb") as saved_file:
            os.fsync(saved_file.fileno())

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
# CHECKPOINT LOADING
# ==========================================================

def load_checkpoint(
    model,
    checkpoint_path,
    device,
):
    """
    Load a model checkpoint.

    Supports:
        1. Current checkpoint dictionaries.
        2. Older checkpoints containing only state_dict.
    """

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )

    except Exception as exc:

        raise RuntimeError(
            f"Could not load checkpoint: {checkpoint_path}\n"
            f"Reason: {exc}"
        ) from exc

    if not isinstance(checkpoint, dict):
        raise ValueError(
            "Checkpoint must contain a dictionary."
        )

    if "model_state_dict" in checkpoint:

        state_dict = checkpoint["model_state_dict"]

        if (
            checkpoint.get("target_length", TARGET_LENGTH)
            != TARGET_LENGTH
        ):
            raise ValueError(
                "Checkpoint target length does not match "
                f"the current configuration ({TARGET_LENGTH})."
            )

        if (
            checkpoint.get("real_label", REAL_LABEL)
            != REAL_LABEL
            or checkpoint.get("fake_label", FAKE_LABEL)
            != FAKE_LABEL
        ):
            raise ValueError(
                "Checkpoint label mapping does not match "
                "the current configuration."
            )

        model.load_state_dict(state_dict)

        print(
            "[INFO] Loaded checkpoint from epoch "
            f"{checkpoint.get('epoch', 'unknown')}."
        )

    else:

        # Legacy checkpoint containing only model parameters.
        model.load_state_dict(checkpoint)

        print("[INFO] Loaded legacy model state_dict.")

    model.to(device)

    return model


# ==========================================================
# DATASET CREATION
# ==========================================================

def create_datasets(data_dir):
    """
    Load train, validation, and test datasets.

    Supports either:
        processed/val/
    or:
        processed/validation/
    """

    data_dir = Path(data_dir).expanduser().resolve()

    print("\n========== LOADING DATASETS ==========")

    train_path = data_dir / "train"

    val_path = data_dir / "val"

    test_path = data_dir / "test"

    if not val_path.is_dir():
        val_path = data_dir / "validation"

    required_paths = {
        "Training": train_path,
        "Validation": val_path,
        "Testing": test_path,
    }

    for name, path in required_paths.items():

        if not path.is_dir():
            raise FileNotFoundError(
                f"{name} dataset directory not found: {path}"
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

        print(
            f"{name} samples: {len(dataset)}"
        )

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
    """Display the number of real and fake windows."""

    real_count = 0
    fake_count = 0

    if not hasattr(dataset, "samples"):
        raise TypeError(
            "Dataset must expose a samples attribute "
            "containing (path, label) entries."
        )

    for _, label in dataset.samples:

        if label == REAL_LABEL:
            real_count += 1

        elif label == FAKE_LABEL:
            fake_count += 1

        else:
            raise ValueError(
                f"Unexpected dataset label: {label}"
            )

    total = real_count + fake_count

    print(f"\n{name.upper()} CLASS DISTRIBUTION")

    print(f"Real: {real_count}")
    print(f"Fake: {fake_count}")
    print(f"Total: {total}")

    if real_count == 0 or fake_count == 0:

        print(
            f"[WARNING] {name} dataset does not contain "
            "both classes."
        )

    return {
        "real": real_count,
        "fake": fake_count,
        "total": total,
    }


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
    """
    Create DataLoaders.

    num_workers=0 is intentional for the current system.
    It avoids additional worker processes and memory usage.
    """

    use_pinned_memory = (
        device.type == "cuda" and PIN_MEMORY
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=use_pinned_memory,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_pinned_memory,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=use_pinned_memory,
        drop_last=False,
    )

    print("\n========== DATALOADERS ==========")

    print(f"Training batches:   {len(train_loader)}")
    print(f"Validation batches: {len(val_loader)}")
    print(f"Testing batches:    {len(test_loader)}")

    print(f"Batch size: {batch_size}")
    print(f"Workers: {NUM_WORKERS}")
    print(f"Pinned memory: {use_pinned_memory}")

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
    """
    Validate and move one batch to the selected device.

    Expected input:
        x: (batch_size, target_length)
        y: (batch_size,)
    """

    if not torch.is_tensor(x):
        raise TypeError(
            "Input must be a PyTorch tensor."
        )

    if not torch.is_tensor(y):
        raise TypeError(
            "Labels must be PyTorch tensors."
        )

    if x.ndim != 2:
        raise ValueError(
            f"Expected input shape (B,T), "
            f"received {tuple(x.shape)}."
        )

    if x.size(0) == 0:
        raise ValueError("Batch is empty.")

    if x.size(1) != TARGET_LENGTH:
        raise ValueError(
            f"Expected {TARGET_LENGTH} temporal samples, "
            f"received {x.size(1)}."
        )

    if y.ndim != 1:
        raise ValueError(
            f"Expected labels shape (B,), "
            f"received {tuple(y.shape)}."
        )

    if x.size(0) != y.size(0):
        raise ValueError(
            "Input batch size does not match label batch size."
        )

    if not torch.isfinite(x).all().item():
        raise ValueError(
            "Input contains NaN or infinite values."
        )

    if not torch.isfinite(y).all().item():
        raise ValueError(
            "Labels contain NaN or infinite values."
        )

    valid_labels = (
        (y == REAL_LABEL) | (y == FAKE_LABEL)
    )

    if not valid_labels.all().item():
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
    )

    return x, y


# ==========================================================
# MODEL OUTPUT VALIDATION
# ==========================================================

def prepare_model_output(logits, batch_size):
    """
    Validate model output and convert it to one logit
    per sample.

    Supported outputs:
        (B,)
        (B, 1)

    The model must produce logits, not sigmoid probabilities,
    because BCEWithLogitsLoss applies sigmoid internally.
    """

    if not torch.is_tensor(logits):
        raise TypeError(
            "Model output must be a PyTorch tensor."
        )

    if logits.ndim == 2 and logits.size(1) == 1:
        logits = logits.squeeze(1)

    if logits.ndim != 1:
        raise ValueError(
            "RPPG1DCNN must return one logit per sample. "
            f"Received shape: {tuple(logits.shape)}"
        )

    if logits.size(0) != batch_size:
        raise ValueError(
            f"Model returned {logits.size(0)} predictions "
            f"for a batch of {batch_size}."
        )

    if not torch.isfinite(logits).all().item():
        raise ValueError(
            "Model produced NaN or infinite logits."
        )

    return logits


# ==========================================================
# CLASSIFICATION METRICS
# ==========================================================

def calculate_metrics(
    all_labels,
    all_predictions,
):
    """Calculate binary classification metrics."""

    return {
        "accuracy": float(
            accuracy_score(
                all_labels,
                all_predictions,
            )
        ),

        "precision": float(
            precision_score(
                all_labels,
                all_predictions,
                zero_division=0,
            )
        ),

        "recall": float(
            recall_score(
                all_labels,
                all_predictions,
                zero_division=0,
            )
        ),

        "f1": float(
            f1_score(
                all_labels,
                all_predictions,
                zero_division=0,
            )
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
    """
    Run one training or evaluation epoch.

    Training:
        Forward pass
        Loss calculation
        Backpropagation
        Optimizer update

    Evaluation:
        Forward pass
        Loss calculation
        Metric calculation
    """

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

    if is_training:
        context = torch.enable_grad()
    else:
        context = torch.inference_mode()

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

            # ------------------------------------------
            # FORWARD PASS
            # ------------------------------------------

            logits = model(x)

            logits = prepare_model_output(
                logits,
                batch_size=y.size(0),
            )

            # ------------------------------------------
            # LOSS
            # ------------------------------------------

            loss = criterion(
                logits,
                y,
            )

            if not torch.isfinite(loss).item():
                raise ValueError(
                    f"{phase} loss became NaN or infinite."
                )

            # ------------------------------------------
            # BACKPROPAGATION
            # ------------------------------------------

            if is_training:

                loss.backward()

                # Check gradients before updating parameters.
                for parameter in model.parameters():

                    if parameter.grad is not None:

                        if not torch.isfinite(
                            parameter.grad
                        ).all().item():

                            raise ValueError(
                                "Model gradients contain NaN "
                                "or infinite values."
                            )

                optimizer.step()

            # ------------------------------------------
            # ACCUMULATE LOSS
            # ------------------------------------------

            batch_count = y.size(0)

            total_loss += (
                loss.item() * batch_count
            )

            total_samples += batch_count

            # ------------------------------------------
            # PREDICTIONS
            # ------------------------------------------

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

            current_loss = (
                total_loss / total_samples
            )

            progress_bar.set_postfix(
                loss=f"{current_loss:.4f}"
            )

    # ======================================================
    # EPOCH SUMMARY
    # ======================================================

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

    print(
        f"\n========== {phase.upper()} SUMMARY =========="
    )

    print(f"Loss:      {average_loss:.4f}")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1-score:  {metrics['f1']:.4f}")

    print(f"Samples:   {total_samples}")
    print(f"Time:      {elapsed:.2f} seconds")

    print("============================================\n")

    return {
        "loss": float(average_loss),
        **metrics,
        "labels": all_labels,
        "predictions": all_predictions,
        "time": float(elapsed),
    }


# ==========================================================
# SAVE TRAINING HISTORY
# ==========================================================

def save_training_history(
    history,
    history_path,
):
    """Save training history using atomic replacement."""

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
                allow_nan=False,
            )

            temporary_file.flush()
            os.fsync(temporary_file.fileno())

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
    """
    Evaluate the best validation checkpoint on the test set.

    The test set is not used for model selection.
    """

    print("\n========== FINAL TESTING ==========")

    model = load_checkpoint(
        model,
        checkpoint_path,
        device,
    )

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
        labels=[
            REAL_LABEL,
            FAKE_LABEL,
        ],
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

    print(
        f"Test loss:      {test_results['loss']:.4f}"
    )

    print(
        f"Test accuracy:  {test_results['accuracy']:.4f}"
    )

    print(
        f"Test precision: {test_results['precision']:.4f}"
    )

    print(
        f"Test recall:    {test_results['recall']:.4f}"
    )

    print(
        f"Test F1-score:  {test_results['f1']:.4f}"
    )

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
    epochs=DEFAULT_EPOCHS,
    batch_size=DEFAULT_BATCH_SIZE,
    lr=DEFAULT_LEARNING_RATE,
    save_dir="model/checkpoints",
    device_name="auto",
    seed=42,
):
    """
    Train FALSEFACET's RPPG1DCNN.

    Workflow:
        1. Validate configuration.
        2. Select device.
        3. Load train, validation, and test datasets.
        4. Create DataLoaders.
        5. Initialize model, loss, and optimizer.
        6. Train and validate each epoch.
        7. Save the best validation checkpoint.
        8. Evaluate the best checkpoint on the test set.
    """

    # ======================================================
    # 1. VALIDATE CONFIGURATION
    # ======================================================

    validate_training_config(
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=lr,
        seed=seed,
    )

    epochs = int(epochs)
    batch_size = int(batch_size)
    seed = int(seed)

    set_seed(seed)

    # ======================================================
    # 2. SELECT DEVICE
    # ======================================================

    device = select_device(device_name)

    print_device_information(device)

    # ======================================================
    # 3. DISPLAY CONFIGURATION
    # ======================================================

    print("========== TRAINING CONFIGURATION ==========")

    print(f"Epochs:         {epochs}")
    print(f"Batch size:     {batch_size}")
    print(f"Learning rate:  {lr}")

    print(f"Data directory: {data_dir}")
    print(f"Device:         {device}")

    print("Model: RPPG1DCNN")
    print(f"Signal length: {TARGET_LENGTH}")

    print("Real label: 0")
    print("Fake label: 1")

    print("=============================================\n")

    # ======================================================
    # 4. LOAD DATASETS
    # ======================================================

    (
        train_dataset,
        val_dataset,
        test_dataset,
    ) = create_datasets(data_dir)

    # ------------------------------------------------------
    # CLASS DISTRIBUTION
    # ------------------------------------------------------

    train_distribution = print_class_distribution(
        train_dataset,
        "Training",
    )

    val_distribution = print_class_distribution(
        val_dataset,
        "Validation",
    )

    test_distribution = print_class_distribution(
        test_dataset,
        "Testing",
    )

    # Training must contain both classes.
    if (
        train_distribution["real"] == 0
        or train_distribution["fake"] == 0
    ):
        raise ValueError(
            "Training dataset must contain both real and fake signals."
        )

    # Validation must contain both classes for meaningful
    # binary validation metrics.
    if (
        val_distribution["real"] == 0
        or val_distribution["fake"] == 0
    ):
        raise ValueError(
            "Validation dataset must contain both real and fake signals."
        )

    # The test split should also contain both classes.
    if (
        test_distribution["real"] == 0
        or test_distribution["fake"] == 0
    ):
        raise ValueError(
            "Test dataset must contain both real and fake signals."
        )

    # ======================================================
    # 5. CREATE DATALOADERS
    # ======================================================

    (
        train_loader,
        val_loader,
        test_loader,
    ) = create_dataloaders(
        train_dataset=train_dataset,
        val_dataset=val_dataset,
        test_dataset=test_dataset,
        batch_size=batch_size,
        device=device,
    )

    # ======================================================
    # 6. INITIALIZE MODEL
    # ======================================================

    print("[INFO] Initializing RPPG1DCNN...")

    model = RPPG1DCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(lr),
    )

    print("[INFO] Model initialized successfully.")

    # ======================================================
    # 7. CHECKPOINT AND HISTORY PATHS
    # ======================================================

    save_dir = Path(save_dir).expanduser().resolve()

    save_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    checkpoint_path = save_dir / CHECKPOINT_NAME

    history_path = save_dir / HISTORY_NAME

    best_val_loss = float("inf")
    best_epoch = 0

    configuration = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": float(lr),
        "target_length": TARGET_LENGTH,
        "device": str(device),
        "seed": seed,
        "model": "RPPG1DCNN",
        "real_label": REAL_LABEL,
        "fake_label": FAKE_LABEL,
    }

    history = {
        "configuration": configuration,
        "epochs": [],
        "best_epoch": None,
        "best_validation_loss": None,
        "total_training_time_seconds": None,
        "test_results": None,
    }

    # ======================================================
    # 8. TRAINING
    # ======================================================

    training_start = time.time()

    print("\n========== STARTING TRAINING ==========\n")

    for epoch in range(1, epochs + 1):

        print("\n" + "=" * 60)
        print(f"EPOCH {epoch}/{epochs}")
        print("=" * 60)

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)

        # --------------------------------------------------
        # 8.1 TRAINING PHASE
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
        # 8.2 VALIDATION PHASE
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
        # 8.3 EPOCH SUMMARY
        # --------------------------------------------------

        print("\n========== EPOCH RESULTS ==========")

        print(f"Epoch: {epoch}/{epochs}")

        print(
            f"Train loss:      {train_results['loss']:.4f}"
        )

        print(
            f"Train accuracy:  {train_results['accuracy']:.4f}"
        )

        print(
            f"Train precision: {train_results['precision']:.4f}"
        )

        print(
            f"Train recall:    {train_results['recall']:.4f}"
        )

        print(
            f"Train F1-score:  {train_results['f1']:.4f}"
        )

        print(
            f"Validation loss:      {val_results['loss']:.4f}"
        )

        print(
            f"Validation accuracy:  {val_results['accuracy']:.4f}"
        )

        print(
            f"Validation precision: {val_results['precision']:.4f}"
        )

        print(
            f"Validation recall:    {val_results['recall']:.4f}"
        )

        print(
            f"Validation F1-score:  {val_results['f1']:.4f}"
        )

        print("===================================\n")

        # --------------------------------------------------
        # 8.4 SAVE EPOCH HISTORY
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
        # 8.5 SAVE BEST VALIDATION CHECKPOINT
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
                configuration=configuration,
            )

            print(
                f"[INFO] New best validation loss: "
                f"{best_val_loss:.4f}"
            )

        # --------------------------------------------------
        # 8.6 SAVE HISTORY AFTER EVERY EPOCH
        # --------------------------------------------------

        history["best_epoch"] = best_epoch

        history["best_validation_loss"] = (
            best_val_loss
        )

        save_training_history(
            history=history,
            history_path=history_path,
        )

        # --------------------------------------------------
        # 8.7 GPU MEMORY INFORMATION
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

    # ======================================================
    # 9. TRAINING COMPLETE
    # ======================================================

    elapsed = time.time() - training_start

    history["best_epoch"] = best_epoch

    history["best_validation_loss"] = best_val_loss

    history["total_training_time_seconds"] = elapsed

    save_training_history(
        history=history,
        history_path=history_path,
    )

    print("\n========== TRAINING COMPLETE ==========")

    print(
        f"Training time: {elapsed / 60:.2f} minutes"
    )

    print(f"Best epoch: {best_epoch}")

    print(
        f"Best validation loss: {best_val_loss:.4f}"
    )

    print(f"Best checkpoint: {checkpoint_path}")

    print(f"Training history: {history_path}")

    print("========================================\n")

    # ======================================================
    # 10. FINAL TESTING
    # ======================================================

    test_results = test_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        checkpoint_path=checkpoint_path,
    )

    # Save final test results in training history.
    history["test_results"] = test_results

    save_training_history(
        history=history,
        history_path=history_path,
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
        help="Processed dataset directory.",
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=DEFAULT_EPOCHS,
        help="Number of training epochs.",
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=DEFAULT_BATCH_SIZE,
        help="Training batch size.",
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=DEFAULT_LEARNING_RATE,
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
        choices=[
            "auto",
            "cuda",
            "cpu",
        ],
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

    try:

        validate_training_config(
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            seed=args.seed,
        )

    except ValueError as exc:
        parser.error(str(exc))

    train_model(
        data_dir=args.data_dir,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        save_dir=args.save_dir,
        device_name=args.device,
        seed=args.seed,
    )