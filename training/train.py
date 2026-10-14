import os
import argparse

import torch
import torch.nn as nn

from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

from model.rppg_model import RPPG1DCNN
from training.dataset import RPPGDataset

def train_model(data_dir: str, 
                epochs: int = 20, 
                batch_size: int = 8, 
                lr: float = 1e-3, 
                save_dir: str = "model/checkpoints"
    ):
    
    os.makedirs(save_dir, 
                exist_ok=True
    )

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(f"[i] Training on device: {device}")

    train_dataset = RPPGDataset(
        root_dir=os.path.join(data_dir, "train"))
    
    val_dataset = RPPGDataset(
        root_dir=os.path.join(data_dir, "val"))

    test_dataset = RPPGDataset(
        root_dir=os.path.join(data_dir, "test")
    )

    if len(train_dataset) == 0:
        raise ValueError("Training dataset is empty")
    if len(val_dataset) == 0:
        raise ValueError("Validated dataset is empty")
    if len(test_dataset) == 0:
        raise ValueError("Test dataset is empty")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False
    )

    model = RPPG1DCNN().to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=lr
    )

    best_val_loss = float('inf')

    ckpt_path = os.path.join(
        save_dir,
        "best_rppg_model.pt  "
    )

    for epoch in range(1, epochs + 1):

        model.train()

        train_loss = 0.0
        train_correct = 0
        total_train = 0

        for x, y in train_loader:

            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32).view(-1)
            optimizer.zero_grad()
            
            logits = model(x).view(-1)

            if logits.shape != y.shape:
                raise ValueError(
                    "Model output and label shapes do not match"
                )

            loss = criterion(logits, y)
     
            loss.backward()
            optimizer.step()

            batch_size_actual = y.size(0)

            train_loss += loss.item() * batch_size_actual

            preds = torch.sigmoid(logits) >= 0.5
            labels = y >= 0.5

            train_correct += (
                preds == labels
                ).sum().item()
            
            total_train += batch_size_actual

    if total_train == 0:
        raise ValueError("No training samples were processed.")
    
    train_acc = train_correct / total_train 
    avg_train_loss = train_loss / total_train 

        # Validation Loop
    model.eval()
    val_loss = 0.0
    val_correct = 0
    total_val = 0

    with torch.no_grad():

        for x, y in val_loader:
            
            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32).view(-1)

            logits = model(x).view(-1)

            loss = criterion(logits, y)

            batch_size_actual = y.size(0)

            val_loss += loss.item() * batch_size_actual

            preds = torch.sigmoid(logits) >= 0.5
            labels = y >= 0.5

            val_correct += (preds == labels).sum().item()
            total_val += batch_size_actual
    
    if total_val == 0:
        raise ValueError("No validation samples were processed.")
        
    val_acc = val_correct / total_val 
    avg_val_loss = val_loss / total_val

    print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

    if avg_val_loss < best_val_loss:

        best_val_loss = avg_val_loss

        torch.save(model.state_dict(), ckpt_path)
        print(f"  [+] Saved new best model checkpoint to {ckpt_path}")


    #testing
    print("\n[i] Loading best checkpoint for final testing...")

    model.load_state_dict(
        torch.load(
            ckpt_path,
            map_location=device,
            weights_only=True
        )
    )

    model.eval()
    
    test_loss = 0.0
    total_test = 0

    all_labels = []
    all_preds = []

    with torch.no_grad():

        for x,y in test_loader:

            x = x.to(device, dtype=torch.float32)
            y = y.to(device, dtype=torch.float32).view(-1)

            logits = model(x).view(-1)

            if logits.shape != y.shape:
                raise ValueError(
                    "Model output and label shapes to not match"
                )

            loss = criterion(logits, y)
            batch_size_actual = y.size(0)

            test_loss += loss.item() * batch_size_actual
            total_test += batch_size_actual

            probabilities = torch.sigmoid(logits)

            preds = (probabilities >= 0.5).long()

            all_preds.extend(
                preds.cpu().numpy().tolist()
            )

            all_labels.extend(
                y.long().cpu().tolist()
            )

    avg_test_loss = test_loss/total_test
    test_accuracy = accuracy_score(
        all_labels,
        all_preds
    )

    test_precision = precision_score(
        all_labels,
        all_preds,
        all_preds,
        zero_division=0
    )

    test_recall = recall_score(
        all_labels,
        all_preds,
        zero_division=0

    )

    test_f1 = f1_score(
        all_labels,
        all_preds,
        zero_division=0
    )

    print("\n ========== FINAL TEST RESULT ========== ")
    print(f"Test Loss: {avg_test_loss:.4f}")
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print(f"Test Precision: {test_precision:.4f}")
    print(f"Test Recall: {test_recall:.4f}")
    print(f"Test F1-Score: {test_f1:.4f}")
    print("==========================================")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FALSE FACET - Model Training Script")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Root dataset folder")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    if args.epoch < 1:
        parser.error("--epoch must be at least 1")
    if args.batch_size < 1:
        parser.error("--batch_size must be at least 1")
    if args.lr <= 0:
        parser.error("--lr must be positive")

    train_model(
        data_dir=args.data_dir, 
        epochs=args.epochs, 
        batch_size=args.batch_size, 
        lr=args.lr
        )