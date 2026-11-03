import torch
import torch.nn as nn
import torch.optim as optim

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from pathlib import Path

from model.efficientnet_model import create_efficientnet_b3


# ==========================================
# CONFIGURATION
# ==========================================

DATASET_DIR = Path("extracted_faces")
CHECKPOINT_DIR = Path("models/checkpoints/efficientnet_v2")

IMAGE_SIZE = 300
BATCH_SIZE = 8
EPOCHS = 10
LEARNING_RATE = 0.0001

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

print("Using device:", DEVICE)


# ==========================================
# IMAGE TRANSFORMATIONS
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),

    transforms.ColorJitter(
        brightness=0.1,
        contrast=0.1,
        saturation=0.1
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


eval_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# LOAD DATASET
# ==========================================

train_dataset = datasets.ImageFolder(
    DATASET_DIR / "train",
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    DATASET_DIR / "val",
    transform=eval_transform
)

test_dataset = datasets.ImageFolder(
    DATASET_DIR / "test",
    transform=eval_transform
)


# ==========================================
# STANDARDIZE DATASET LABELS
# ==========================================

# Standard label convention:
# Real = 0
# Fake = 1

class_mapping = {
    "real": 0,
    "fake": 1
}

for dataset in [train_dataset, val_dataset, test_dataset]:

    dataset.samples = [
        (
            path,
            class_mapping[
                Path(path).relative_to(dataset.root).parts[0].lower()
            ]
        )
        for path, _ in dataset.samples
    ]

    dataset.targets = [
        label for _, label in dataset.samples
    ]

    dataset.class_to_idx = class_mapping.copy()


print("\nUpdated Class Mapping:", train_dataset.class_to_idx)


# ==========================================
# CHECK DATASET
# ==========================================

print("\nUpdated Class Mapping:", train_dataset.class_to_idx)

assert train_dataset.class_to_idx == val_dataset.class_to_idx
assert train_dataset.class_to_idx == test_dataset.class_to_idx

print("\nDataset summary:")

for name, dataset in [
    ("Training", train_dataset),
    ("Validation", val_dataset),
    ("Testing", test_dataset)
]:

    real_count = dataset.targets.count(0)
    fake_count = dataset.targets.count(1)

    print(f"\n{name} images: {len(dataset)}")
    print(f"Real images (0): {real_count}")
    print(f"Fake images (1): {fake_count}")


# ==========================================
# DATA LOADERS
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=2
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=2
)


# ==========================================
# INITIALIZE MODEL
# ==========================================

model = create_efficientnet_b3(
    num_classes=2,
    pretrained=True
)

model = model.to(DEVICE)


# ==========================================
# LOSS AND OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss()

optimizer = optim.AdamW(
    model.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# VALIDATION FUNCTION
# ==========================================

def evaluate(model, loader):

    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = torch.argmax(outputs, dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# ==========================================
# TRAINING LOOP
# ==========================================

best_val_accuracy = 0.0

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0
    correct = 0
    total = 0

    for images, labels in train_loader:

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        # Clear previous gradients
        optimizer.zero_grad(set_to_none=True)

        # Forward propagation
        outputs = model(images)

        # Calculate loss
        loss = criterion(outputs, labels)

        # Backpropagation
        loss.backward()

        # Update model weights
        optimizer.step()

        running_loss += loss.item() * images.size(0)

        predictions = torch.argmax(outputs, dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    train_loss = running_loss / total
    train_accuracy = correct / total

    # Validation
    val_loss, val_accuracy = evaluate(
        model,
        val_loader
    )

    print(f"\nEpoch [{epoch + 1}/{EPOCHS}]")

    print(f"Train Loss: {train_loss:.4f}")
    print(f"Train Accuracy: {train_accuracy:.4f}")

    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_accuracy:.4f}")

    # Save best model
    if val_accuracy > best_val_accuracy:

        best_val_accuracy = val_accuracy

        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "class_to_idx": class_mapping,
                "image_size": IMAGE_SIZE,
                "best_val_accuracy": best_val_accuracy
            },
            CHECKPOINT_DIR / "efficientnet_b3_best.pth"
        )

        print("Best model checkpoint saved!")


# ==========================================
# TEST BEST MODEL
# ==========================================

print("\nLoading best checkpoint...")

checkpoint = torch.load(
    CHECKPOINT_DIR / "efficientnet_b3_best.pth",
    map_location=DEVICE
)

model.load_state_dict(checkpoint["model_state_dict"])

test_loss, test_accuracy = evaluate(
    model,
    test_loader
)

print("\n========== FINAL TEST RESULTS ==========")

print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_accuracy:.4f}")

print("\nTraining completed!")