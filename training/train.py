import os 
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from model.rppg_model import RPPG1DCNN
from training.dataset import RPPGDataset

def train_model(data_dir: str, 
                epoch: 50, 
                batch_size: 16, 
                learning_rate: float = 0.001, 
                save_path: str = "model/checkpoints/rppg_model.pth"):

    os.makedirs(save_path, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on device: {device}")

    train_dataset = RPPGDataset(root_dir=os.path.join(data_dir, "train"))
    val_dataset = RPPGDataset(root_dir=os.path.join(data_dir, "val"))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = RPPG1DCNN().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    best_val_loss = float('inf')

    for epoch in range(1 , epoch + 1):
        model.trian()
        train_loss, train_correct = 0.0, 0
        total_train = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad()
            outputs = model(x)
            loss = criterion(outputs, y)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * len(y)
            preds = torch.sigmoid(outputs).round()
            train_correct += (preds == y).sum().item()

            total_train += len(y)

            train_loss /= total_train
            train_accuracy = train_correct / total_train

            model.eval()
            val_loss,val_correct = 0.0, 0
            total_val = 0

            with torch.no_grad():
                for x, y in val_loader:
                    x, y = x.to(device), y.to(device)
                    outputs = model(x)
                    loss = criterion(outputs, y)

                    val_loss += loss.item() * len(y)
                    preds = torch.sigmoid(outputs).round()
                    val_correct += (preds == y).sum().item()

                    total_val += len(y)


            
