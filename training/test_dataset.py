from training.dataset import RPPGDataset

dataset = RPPGDataset(
    root_dir="dataset/train",
    target_length=300,
    fps=30.0
)

print("Dataset size:", len(dataset))

if len(dataset) > 0:
    x,y = dataset[0]

    print("Signal Shape:", x.shape)
    print("Signa dtype:", x.dtype)
    print("Label:", y.item())
    print("Signal Finite:", x.isfinite().all().item())
    print("Signal Standard deviation:", x.std().itwm)