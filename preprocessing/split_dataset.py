import random
import shutil
import argparse
import hashlib

from pathlib import Path
from collections import Counter


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DIR = PROJECT_ROOT / "raw_dataset"
OUTPUT_DIR = PROJECT_ROOT / "dataset"

SPLIT_RATIOS = {
    "train": 0.70,
    "val": 0.15,
    "test": 0.15,
}

RANDOM_SEED = 42

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv",
    ".webm", ".mpeg", ".mpg", ".m4v",
}


# ============================================================
# FILE HASH
# ============================================================

def calculate_hash(file_path):
    """Calculate SHA256 hash to identify duplicate videos."""

    sha256 = hashlib.sha256()

    with open(file_path, "rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


# ============================================================
# FIND VIDEOS
# ============================================================

def get_videos(class_dir):

    if not class_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {class_dir}"
        )

    return sorted(
        path for path in class_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    )


# ============================================================
# SPLIT VIDEOS
# ============================================================

def split_videos(videos, seed):

    videos = list(videos)

    rng = random.Random(seed)
    rng.shuffle(videos)

    total = len(videos)

    train_count = round(total * SPLIT_RATIOS["train"])
    val_count = round(total * SPLIT_RATIOS["val"])

    train_videos = videos[:train_count]

    val_videos = videos[
        train_count:train_count + val_count
    ]

    test_videos = videos[
        train_count + val_count:
    ]

    return {
        "train": train_videos,
        "val": val_videos,
        "test": test_videos,
    }


# ============================================================
# EXISTING VIDEO HASHES
# ============================================================

def get_existing_hashes():

    hashes = set()

    if not OUTPUT_DIR.exists():
        return hashes

    for video in OUTPUT_DIR.rglob("*"):

        if (
            video.is_file()
            and video.suffix.lower() in VIDEO_EXTENSIONS
        ):
            hashes.add(calculate_hash(video))

    return hashes


# ============================================================
# SAFE COPY
# ============================================================

def copy_videos(split_data, class_name, existing_hashes):

    copied = Counter()
    skipped = 0

    for split_name, videos in split_data.items():

        destination_dir = OUTPUT_DIR / split_name / class_name

        destination_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        for video in videos:

            video_hash = calculate_hash(video)

            # Skip exact duplicates already in the dataset.
            if video_hash in existing_hashes:
                skipped += 1
                continue

            relative_path = video.relative_to(
                INPUT_DIR / class_name
            )

            destination = destination_dir / relative_path

            destination.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            # Avoid overwriting a different video with the same name.
            if destination.exists():

                counter = 1

                while True:

                    new_name = (
                        f"{destination.stem}_new_{counter}"
                        f"{destination.suffix}"
                    )

                    new_destination = (
                        destination.parent / new_name
                    )

                    if not new_destination.exists():
                        destination = new_destination
                        break

                    counter += 1

            shutil.copy2(video, destination)

            existing_hashes.add(video_hash)
            copied[split_name] += 1

    for split_name in ["train", "val", "test"]:

        print(
            f"{split_name.upper():5s} | "
            f"{class_name:5s} | "
            f"{copied[split_name]} copied"
        )

    print(f"Skipped duplicates: {skipped}")

    return copied


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Incrementally add new FALSEFACET videos."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED
    )

    args = parser.parse_args()

    print("\n======================================")
    print(" FALSEFACET INCREMENTAL DATASET SPLITTER")
    print("======================================")

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {INPUT_DIR}"
        )

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    existing_hashes = get_existing_hashes()

    total_counts = Counter()

    for class_name in ["real", "fake"]:

        print(f"\nProcessing: {class_name.upper()}")

        videos = get_videos(INPUT_DIR / class_name)

        if not videos:
            print("No new videos found.")
            continue

        print(f"New videos found: {len(videos)}")

        split_data = split_videos(
            videos,
            seed=args.seed
        )

        copied = copy_videos(
            split_data,
            class_name,
            existing_hashes
        )

        total_counts[class_name] = sum(copied.values())

    print("\n======================================")
    print(" INCREMENTAL SPLIT SUMMARY")
    print("======================================")

    print(f"New real videos copied: {total_counts['real']}")
    print(f"New fake videos copied: {total_counts['fake']}")

    print(
        f"Total new videos copied: "
        f"{sum(total_counts.values())}"
    )

    print(f"\nDataset location: {OUTPUT_DIR}")

    print("\nExisting dataset was preserved.")
    print("Dataset update completed!")


if __name__ == "__main__":
    main()