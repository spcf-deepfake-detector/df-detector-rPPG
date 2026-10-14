import os
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.rppg_model import RPPG1DCNN
from training.dataset import RPPGDataset

def train_model(data_dir: str, epochs: int = 20, batch_size: int = 8, lr: float = 1e-3, save_dir: str = "model/checkpoints"):
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"[i] Training on device: {device}")

    train_dataset = RPPGDataset(root_dir=os.path.join(data_dir, "train"))
    val_dataset = RPPGDataset(root_dir=os.path.join(data_dir, "val"))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = RPPG1DCNN().to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_val_loss = float('inf')

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss, train_correct = 0.0, 0
        total_train = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            
            logits = model(x).squeeze(-1)
            loss = criterion(logits, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            preds = torch.sigmoid(logits) >= 0.5
            train_correct += (preds == y).sum().item()
            total_train += len(y)

        train_acc = train_correct / total_train if total_train > 0 else 0.0
        avg_train_loss = train_loss / total_train if total_train > 0 else 0.0

        # Validation Loop
        model.eval()
        val_loss, val_correct = 0.0, 0
        total_val = 0

        with torch.no_grad():
            for x, y in val_loader:
                x, y = x.to(device), y.to(device)
                logits = model(x).squeeze(-1)
                loss = criterion(logits, y)

                val_loss += loss.item() * len(y)
                preds = torch.sigmoid(logits) >= 0.5
                val_correct += (preds == y).sum().item()
                total_val += len(y)

        val_acc = val_correct / total_val if total_val > 0 else 0.0
        avg_val_loss = val_loss / total_val if total_val > 0 else 0.0

        print(f"Epoch {epoch:02d}/{epochs:02d} | Train Loss: {avg_train_loss:.4f} | Train Acc: {train_acc:.4f} | Val Loss: {avg_val_loss:.4f} | Val Acc: {val_acc:.4f}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            ckpt_path = os.path.join(save_dir, "best_rppg_model.pt")
            torch.save(model.state_dict(), ckpt_path)
            print(f"  [+] Saved new best model checkpoint to {ckpt_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FALSE FACET - Model Training Script")
    parser.add_argument("--data_dir", type=str, default="dataset", help="Root dataset folder")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    train_model(args.data_dir, args.epochs, args.batch_size, args.lr)