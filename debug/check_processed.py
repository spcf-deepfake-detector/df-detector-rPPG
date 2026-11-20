from pathlib import Path

PROCESSED_DIR = Path("processed")

splits = ["train", "val", "test"]
classes = ["real", "fake"]

total = 0

print("\n========== PROCESSED DATASET SUMMARY ==========\n")

for split in splits:
    print(f" {split.upper()} SET")

    split_total = 0

    for class_name in classes:
        folder = PROCESSED_DIR / split / class_name

        if not folder.exists():
            print(f"  {class_name.upper()}: Folder not found")
            continue

        files = list(folder.glob("*.npy"))
        count = len(files)

        split_total += count
        total += count

        print(f"  {class_name.upper()}: {count} files")

    print(f"  Total: {split_total}\n")

print("===============================================")
print(f"TOTAL .NPY FILES: {total}")
print("===============================================\n")