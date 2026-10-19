from torch.utils.data import DataLoader
from dataset import RPPGDataset

# Load the training videos
dataset = RPPGDataset(
    root_dir="dataset/train",
    target_length=300,
    fps=30.0
)

print("Number of videos:", len(dataset))

if len(dataset) == 0:
    raise RuntimeError(
        "No videos found. Check dataset/train/real "
        "and dataset/train/fake."
    )

# Process one video
signals, labels = dataset[0]

print("Signal shape:", signals.shape)
print("Signal dtype:", signals.dtype)
print("Label:", labels.item())
print("Signal contains only finite values:", signals.isfinite().all().item())

# Prepare a DataLoader
loader = DataLoader(
    dataset,
    batch_size=2,
    shuffle=False,
    num_workers=0
)

print("\nTesting DataLoader...")

for batch_signals, batch_labels in loader:
    print("Batch signal shape:", batch_signals.shape)
    print("Batch labels:", batch_labels)
    break

print("\nDataset test completed!")