import random
import shutil
import argparse

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
    ".mp4",
    ".avi",
    ".mov",
    ".mkv",
    ".webm",
    ".mpeg",
    ".mpg",
    ".m4v",
}


# ============================================================
# FIND VIDEOS
# ============================================================

def get_videos(class_dir: Path):
    """Find all supported video files recursively."""

    if not class_dir.exists():
        raise FileNotFoundError(
            f"Dataset directory does not exist: {class_dir}"
        )

    videos = [
        path
        for path in class_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    return sorted(videos)


# ============================================================
# SPLIT VIDEOS
# ============================================================

def split_videos(videos, seed=RANDOM_SEED):
    """Split videos into train, validation, and test."""

    videos = list(videos)

    rng = random.Random(seed)
    rng.shuffle(videos)

    total = len(videos)

    train_count = int(total * SPLIT_RATIOS["train"])
    val_count = int(total * SPLIT_RATIOS["val"])

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
# COPY VIDEOS
# ============================================================

def copy_videos(split_data, class_name):
    """Copy videos into their corresponding split folders."""

    for split_name, videos in split_data.items():

        destination_dir = OUTPUT_DIR / split_name / class_name
        destination_dir.mkdir(parents=True, exist_ok=True)

        for video in videos:

            # Preserve relative paths to avoid filename collisions.
            relative_path = video.relative_to(
                INPUT_DIR / class_name
            )

            destination = destination_dir / relative_path
            destination.parent.mkdir(
                parents=True,
                exist_ok=True
            )

            shutil.copy2(video, destination)

        print(
            f"{split_name.upper():5s} | "
            f"{class_name:5s} | "
            f"{len(videos)} videos"
        )


# ============================================================
# CLEAN OUTPUT
# ============================================================

def prepare_output_directory(overwrite=False):
    """Prepare the output directory."""

    if OUTPUT_DIR.exists():

        if not overwrite:
            raise FileExistsError(
                f"\nOutput directory already exists:\n"
                f"{OUTPUT_DIR}\n\n"
                f"Use --overwrite to rebuild the splits."
            )

        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="Automatically split FALSEFACET video dataset."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete and rebuild the existing output directory."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=RANDOM_SEED,
        help="Random seed for reproducible splitting."
    )

    args = parser.parse_args()

    print("\n======================================")
    print(" FALSEFACET DATASET SPLITTER")
    print("======================================\n")

    prepare_output_directory(args.overwrite)

    total_counts = Counter()

    for class_name in ["real", "fake"]:

        print(f"\nProcessing: {class_name.upper()}")

        class_dir = INPUT_DIR / class_name

        videos = get_videos(class_dir)

        if not videos:
            raise ValueError(
                f"No videos found in {class_dir}"
            )

        split_data = split_videos(
            videos,
            seed=args.seed
        )

        copy_videos(split_data, class_name)

        total_counts[class_name] = len(videos)

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n======================================")
    print(" DATASET SPLIT SUMMARY")
    print("======================================")

    print(f"Real videos: {total_counts['real']}")
    print(f"Fake videos: {total_counts['fake']}")

    print(
        f"Total videos: "
        f"{sum(total_counts.values())}"
    )

    print("\nOutput directory:")

    print(OUTPUT_DIR)

    print("\nDataset splitting completed successfully!")


if __name__ == "__main__":
    main()
