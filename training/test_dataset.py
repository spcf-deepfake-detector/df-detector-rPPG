import torch
from torch.utils.data import DataLoader
from training.dataset import RPPGDataset


# 1. Load the training videos
dataset = RPPGDataset(
    root_dir="dataset/train",
    target_length=300,
    fps=30.0
)


# 2. Count the videos
print("Number of videos:", len(dataset))


# 3. Stop if no videos were found
if len(dataset) == 0:
    raise RuntimeError(
        "No videos found. Check dataset/train/real "
        "and dataset/train/fake."
    )


# 4. Process one video
signals, labels = dataset[0]


# 5. Display information about the signal
print("Signal shape:", signals.shape)
print("Signal dtype:", signals.dtype)
print("Label:", labels.item())


# 6. Check for invalid numerical values
finite = torch.isfinite(signals).all().item()

print("Signal contains only finite values:", finite)

if not finite:
    raise ValueError(
        "The signal contains NaN or infinite values."
    )


# 7. Prepare the DataLoader
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)


# 8. Test the first batch
print("\nTesting DataLoader...")

for batch_signals, batch_labels in loader:
    print("Batch signal shape:", batch_signals.shape)
    print("Batch labels:", batch_labels)
    break


# 9. Finish
print("\nDataset test completed!")