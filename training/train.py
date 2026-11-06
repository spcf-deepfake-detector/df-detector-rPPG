import os
import argparse
import time
import tempfile

import torch
import torch.nn as nn

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)

from model.rppg_model import RPPG1DCNN
from training.dataset import RPPGDataset 



# ==========================================================
# DEVICE INFORMATION
# ==========================================================

def print_device_information(device):

    print("\n========== DEVICE INFORMATION ==========")

    print(f"Selected device: {device}")

    if device.type == "cuda":

        print(f"GPU: {torch.cuda.get_device_name(device)}")

        print(
            f"CUDA version: {torch.version.cuda}"
        )

        print(
            f"GPU memory: "
            f"{torch.cuda.get_device_properties(device).total_memory / (1024 ** 3):.2f} GB"
        )

    else:

        print("Running on CPU.")

    print("========================================\n")


# ==========================================================
# CUDA MEMORY MONITORING
# ==========================================================

def print_cuda_memory(device):

    if device.type != "cuda":
        return

    allocated = torch.cuda.memory_allocated(device) / (1024 ** 3)

    reserved = torch.cuda.memory_reserved(device) / (1024 ** 3)

    peak = torch.cuda.max_memory_allocated(device) / (1024 ** 3)

    print(
        f"[GPU MEMORY] "
        f"Allocated: {allocated:.2f} GB | "
        f"Reserved: {reserved:.2f} GB | "
        f"Peak: {peak:.2f} GB",
        flush=True
    )


# ==========================================================
# SAFE CHECKPOINT SAVING
# ==========================================================

def save_checkpoint(model, checkpoint_path):

    directory = os.path.dirname(checkpoint_path)

    os.makedirs(directory, exist_ok=True)

    temporary_path = None

    try:

        with tempfile.NamedTemporaryFile(
            dir=directory,
            suffix=".tmp",
            delete=False
        ) as temporary_file:

            temporary_path = temporary_file.name

        torch.save(
            model.state_dict(),
            temporary_path
        )

        os.replace(
            temporary_path,
            checkpoint_path
        )

        print(
            f"[CHECKPOINT] Saved: {checkpoint_path}",
            flush=True
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

    print("\n========== LOADING DATASETS ==========")

    train_path = os.path.join(data_dir, "train")

    val_path = os.path.join(data_dir, "validation")

    test_path = os.path.join(data_dir, "test")

    for path in (train_path, val_path, test_path):

        if not os.path.isdir(path):

            raise FileNotFoundError(
                f"Dataset directory not found: {path}"
            )

    print("\n[1/3] Loading training dataset...")

    train_dataset = RPPGDataset(
        root_dir=train_path,
        target_length=300,
    )

    print("\n[2/3] Loading validation dataset...")

    val_dataset = RPPGDataset(
        root_dir=val_path,
        target_length=300,

    )

    print("\n[3/3] Loading testing dataset...")

    test_dataset = RPPGDataset(
        root_dir=test_path,
        target_length=300,

    )

    if len(train_dataset) == 0:

        raise ValueError(
            "Training dataset is empty."
        )

    if len(val_dataset) == 0:

        raise ValueError(
            "Validation dataset is empty."
        )

    if len(test_dataset) == 0:

        raise ValueError(
            "Testing dataset is empty."
        )

    print("\n========== DATASET SUMMARY ==========")

    print(f"Training samples:   {len(train_dataset)}")

    print(f"Validation samples: {len(val_dataset)}")

    print(f"Testing samples:    {len(test_dataset)}")

    print("=====================================\n")

    return train_dataset, val_dataset, test_dataset


# ==========================================================
# DATALOADER CREATION
# ==========================================================

def create_dataloaders(
    train_dataset,
    val_dataset,
    test_dataset,
    batch_size,
    device,
):

    print("[i] Creating DataLoaders...")


    use_cuda = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0,
        pin_memory=use_cuda,
        drop_last=False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
        drop_last=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=use_cuda,
        drop_last=False
    )

    print(f"Training batches:   {len(train_loader)}")

    print(f"Validation batches: {len(val_loader)}")

    print(f"Testing batches:    {len(test_loader)}")

    print(f"Pinned memory: {use_cuda}")

    print("[i] DataLoaders ready.\n")

    return train_loader, val_loader, test_loader


# ==========================================================
# BATCH VALIDATION
# ==========================================================

def prepare_batch(x, y, device):

    if not torch.is_tensor(x) or not torch.is_tensor(y):
        raise TypeError(
            "Dataset must return PyTorch tensors."
        )

    if x.ndim < 2:
        raise ValueError(
            f"Invalid input dimensions: {x.shape}"
        )

    # Validate on CPU before transferring.
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

    # Transfer validated tensors to the selected device.
    non_blocking = device.type == "cuda"

    x = x.to(
        device,
        dtype=torch.float32,
        non_blocking=non_blocking,
    )

    y = y.to(
        device,
        dtype=torch.float32,
        non_blocking=non_blocking,
    ).view(-1)

    return x, y

# ==========================================================
# TRAINING ONE EPOCH
# ==========================================================

def train_one_epoch(
    model,
    loader,
    criterion,
    optimizer,
    device,
    epoch,
    total_epochs
):

    model.train()

    total_loss = 0.0

    total_correct = 0

    total_samples = 0

    number_of_batches = len(loader)

    start_time = time.time()

    print(
        f"\n========== TRAINING EPOCH "
        f"{epoch}/{total_epochs} ==========",
        flush=True
    )

    for batch_idx, (x, y) in enumerate(loader, start=1):

        print(
            f"[TRAIN] Epoch {epoch}/{total_epochs} "
            f"Batch {batch_idx}/{number_of_batches}",
            flush=True
        )

        try:

            x, y = prepare_batch(
                x,
                y,
                device
            )

            optimizer.zero_grad(set_to_none=True)

            logits = model(x).view(-1)

            if logits.shape != y.shape:

                raise ValueError(
                    f"Output shape {logits.shape} "
                    f"does not match label shape {y.shape}."
                )

            loss = criterion(
                logits,
                y
            )

            if not torch.isfinite(loss):

                raise ValueError(
                    "Training loss became NaN or infinite."
                )

            loss.backward()

            optimizer.step()

            batch_count = y.size(0)

            total_loss += loss.item() * batch_count

            predictions = (
                torch.sigmoid(logits) >= 0.5
            )

            labels = y >= 0.5

            total_correct += (
                predictions == labels
            ).sum().item()

            total_samples += batch_count

        except torch.cuda.OutOfMemoryError as exc:

            optimizer.zero_grad(set_to_none=True)

            if device.type == "cuda":

                torch.cuda.empty_cache()

            raise RuntimeError(
                "CUDA ran out of memory during training. "
                "Try reducing batch_size."
            ) from exc

        if device.type == "cuda":

            if batch_idx % 5 == 0:

                print_cuda_memory(device)

    if total_samples == 0:

        raise ValueError(
            "No training samples were processed."
        )

    average_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    elapsed = time.time() - start_time

    print("\n========== TRAINING SUMMARY ==========")

    print(f"Training loss: {average_loss:.4f}")

    print(f"Training accuracy: {accuracy:.4f}")

    print(f"Processed samples: {total_samples}")

    print(f"Time: {elapsed:.2f} seconds")

    print("======================================")

    return average_loss, accuracy


# ==========================================================
# VALIDATION
# ==========================================================

def validate_model(
    model,
    loader,
    criterion,
    device,
    epoch,
    total_epochs
):

    model.eval()

    total_loss = 0.0

    total_correct = 0

    total_samples = 0

    print(
        f"\n========== VALIDATION "
        f"{epoch}/{total_epochs} ==========",
        flush=True
    )

    with torch.inference_mode():

        for batch_idx, (x, y) in enumerate(
            loader,
            start=1
        ):

            print(
                f"[VALIDATION] Batch "
                f"{batch_idx}/{len(loader)}",
                flush=True
            )

            try:

                x, y = prepare_batch(
                    x,
                    y,
                    device
                )

                logits = model(x).view(-1)

                if logits.shape != y.shape:

                    raise ValueError(
                        "Validation output and labels do not match."
                    )

                loss = criterion(
                    logits,
                    y
                )

                if not torch.isfinite(loss):

                    raise ValueError(
                        "Validation loss became NaN or infinite."
                    )

                batch_count = y.size(0)

                total_loss += loss.item() * batch_count

                predictions = (
                    torch.sigmoid(logits) >= 0.5
                )

                labels = y >= 0.5

                total_correct += (
                    predictions == labels
                ).sum().item()

                total_samples += batch_count

            except torch.cuda.OutOfMemoryError as exc:

                if device.type == "cuda":

                    torch.cuda.empty_cache()

                raise RuntimeError(
                    "CUDA ran out of memory during validation."
                ) from exc

    if total_samples == 0:

        raise ValueError(
            "No validation samples were processed."
        )

    average_loss = total_loss / total_samples

    accuracy = total_correct / total_samples

    print("\n========== VALIDATION SUMMARY ==========")

    print(f"Validation loss: {average_loss:.4f}")

    print(f"Validation accuracy: {accuracy:.4f}")

    print(f"Processed samples: {total_samples}")

    print("========================================")

    return average_loss, accuracy


# ==========================================================
# FINAL TESTING
# ==========================================================

def test_model(
    model,
    loader,
    criterion,
    device,
    checkpoint_path
):

    print("\n========== FINAL TESTING ==========")

    if not os.path.exists(checkpoint_path):

        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )

    model.load_state_dict(state_dict)

    model.eval()

    total_loss = 0.0

    total_samples = 0

    all_labels = []

    all_predictions = []

    with torch.inference_mode():

        for batch_idx, (x, y) in enumerate(
            loader,
            start=1
        ):

            print(
                f"[TEST] Batch {batch_idx}/{len(loader)}",
                flush=True
            )

            try:

                x, y = prepare_batch(
                    x,
                    y,
                    device
                )

                logits = model(x).view(-1)

                if logits.shape != y.shape:

                    raise ValueError(
                        "Test output and labels do not match."
                    )

                loss = criterion(
                    logits,
                    y
                )

                batch_count = y.size(0)

                total_loss += loss.item() * batch_count

                total_samples += batch_count

                probabilities = torch.sigmoid(logits)

                predictions = (
                    probabilities >= 0.5
                ).long()

                all_predictions.extend(
                    predictions.cpu().tolist()
                )

                all_labels.extend(
                    y.long().cpu().tolist()
                )

            except torch.cuda.OutOfMemoryError as exc:

                if device.type == "cuda":

                    torch.cuda.empty_cache()

                raise RuntimeError(
                    "CUDA ran out of memory during testing."
                ) from exc

    if total_samples == 0:

        raise ValueError(
            "No test samples were processed."
        )

    average_loss = total_loss / total_samples

    accuracy = accuracy_score(
        all_labels,
        all_predictions
    )

    precision = precision_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    recall = recall_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    f1 = f1_score(
        all_labels,
        all_predictions,
        zero_division=0
    )

    print("\n========== FINAL TEST RESULTS ==========")

    print(f"Test Loss:      {average_loss:.4f}")

    print(f"Test Accuracy:  {accuracy:.4f}")

    print(f"Test Precision: {precision:.4f}")

    print(f"Test Recall:    {recall:.4f}")

    print(f"Test F1-Score:  {f1:.4f}")

    print("========================================")

    return {
        "loss": average_loss,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }


# ==========================================================
# MAIN TRAINING FUNCTION
# ==========================================================

def train_model(
    data_dir: str,
    epochs: int = 20,
    batch_size: int = 2,
    lr: float = 1e-3,
    save_dir: str = "model/checkpoints",
    device_name: str = "auto"
):

    # ------------------------------------------------------
    # STEP 1: VALIDATE CONFIGURATION
    # ------------------------------------------------------

    if epochs < 1:

        raise ValueError(
            "epochs must be at least 1."
        )

    if batch_size < 1:

        raise ValueError(
            "batch_size must be at least 1."
        )

    if lr <= 0:

        raise ValueError(
            "Learning rate must be positive."
        )

    os.makedirs(
        save_dir,
        exist_ok=True
    )

    # ------------------------------------------------------
    # STEP 2: SELECT DEVICE
    # ------------------------------------------------------

    if device_name == "auto":

        device = torch.device(
            "cuda:0" if torch.cuda.is_available() else "cpu"
        )

    elif device_name == "cuda":

        if not torch.cuda.is_available():

            raise RuntimeError(
                "CUDA was requested but is unavailable."
            )

        device = torch.device("cuda:0")

    else:

        device = torch.device("cpu")

    print_device_information(device)

    print("========== TRAINING CONFIGURATION ==========")

    print(f"Epochs: {epochs}")

    print(f"Batch size: {batch_size}")

    print(f"Learning rate: {lr}")

    print(f"Data directory: {data_dir}")

    print(f"Checkpoint directory: {save_dir}")

    print("DataLoader workers: 0")

    print(f"Pinned memory: {device.type == 'cuda'}")

    print("============================================\n")

    # ------------------------------------------------------
    # STEP 3: LOAD DATASETS
    # ------------------------------------------------------

    train_dataset, val_dataset, test_dataset = (
        create_datasets(data_dir)
    )

    # ------------------------------------------------------
    # STEP 4: CREATE DATALOADERS
    # ------------------------------------------------------

    train_loader, val_loader, test_loader = (
        create_dataloaders(
            train_dataset,
            val_dataset,
            test_dataset,
            batch_size,
            device,
        )
    )

    # ------------------------------------------------------
    # STEP 5: CREATE MODEL
    # ------------------------------------------------------

    print("[i] Initializing RPPG1DCNN...")

    model = RPPG1DCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=lr
    )

    print("[i] Model initialized successfully.")

    # ------------------------------------------------------
    # STEP 6: CHECKPOINT CONFIGURATION
    # ------------------------------------------------------

    checkpoint_path = os.path.join(
        save_dir,
        "best_rppg_model.pt"
    )

    best_val_loss = float("inf")

    # ------------------------------------------------------
    # STEP 7: TRAINING
    # ------------------------------------------------------

    training_start = time.time()

    for epoch in range(1, epochs + 1):

        if device.type == "cuda":

            torch.cuda.reset_peak_memory_stats(device)

        # TRAINING

        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            total_epochs=epochs
        )

        # VALIDATION

        val_loss, val_accuracy = validate_model(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            epoch=epoch,
            total_epochs=epochs
        )

        # EPOCH RESULTS

        print("\n========== EPOCH RESULTS ==========")

        print(f"Epoch: {epoch}/{epochs}")

        print(f"Train Loss: {train_loss:.4f}")

        print(f"Train Accuracy: {train_accuracy:.4f}")

        print(f"Validation Loss: {val_loss:.4f}")

        print(f"Validation Accuracy: {val_accuracy:.4f}")

        print("===================================\n")

        # SAVE BEST CHECKPOINT

        if val_loss < best_val_loss:

            best_val_loss = val_loss

            save_checkpoint(
                model,
                checkpoint_path
            )

        # MEMORY INFORMATION

        print_cuda_memory(device)

    # ------------------------------------------------------
    # STEP 8: TRAINING COMPLETE
    # ------------------------------------------------------

    total_training_time = (
        time.time() - training_start
    )

    print("\n========== TRAINING COMPLETE ==========")

    print(
        f"Total training time: "
        f"{total_training_time / 60:.2f} minutes"
    )

    print(f"Best validation loss: {best_val_loss:.4f}")

    print("========================================")

    # ------------------------------------------------------
    # STEP 9: FINAL TESTING
    # ------------------------------------------------------

    test_results = test_model(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
        checkpoint_path=checkpoint_path
    )

    print("\n[i] FALSEFACET rPPG training finished.")

    return test_results


# ==========================================================
# COMMAND-LINE INTERFACE
# ==========================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="FALSEFACET rPPG Training"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="processed",
        help="Root directory containing preprocessed rPPG signals."
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=20,
        help="Number of training epochs."
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Training batch size."
    )

    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Learning rate."
    )

    parser.add_argument(
        "--save_dir",
        type=str,
        default="model/checkpoints",
        help="Checkpoint output directory."
    )

    parser.add_argument(
        "--device",
        type=str,
        choices=["auto", "cuda", "cpu"],
        default="auto",
        help="Training device."
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
        device_name=args.device
    )