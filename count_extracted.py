from pathlib import Path

dataset_dir = Path("extracted_faces")

splits = ["train", "val", "test"]
classes = ["real", "fake"]

total_images = 0

for split in splits:

    print(f"\n========== {split.upper()} ==========")

    for label in classes:

        folder = dataset_dir / split / label

        images = list(folder.rglob("*.jpg"))

        count = len(images)

        total_images += count

        print(f"{label}: {count} images")

print("\n========== TOTAL ==========")
print(f"Total extracted images: {total_images}")