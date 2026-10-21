import random
import shutil
import argparse
from pathlib import Path


def split_dataset(source_dir, output_dir, seed=42):
    source_dir = Path(source_dir).resolve()
    output_dir = Path(output_dir).resolve()

    # Supported video formats
    video_extensions = {
        ".mp4",
        ".mov",
        ".avi",
        ".mkv",
        ".webm"
    }

    # Check the source directory
    if not source_dir.is_dir():
        raise FileNotFoundError(
            f"Source directory not found: {source_dir}"
        )

    # Prevent accidentally mixing files from previous runs
    if output_dir.exists() and any(
        path.is_file() for path in output_dir.rglob("*")
    ):
        raise FileExistsError(
            f"The output directory already contains files: {output_dir}\n"
            "Choose a new output directory or clear the old dataset first."
        )

    # Create a reproducible random generator
    rng = random.Random(seed)

    # Dataset split ratios
    train_ratio = 0.70
    val_ratio = 0.15

    # Process real and fake videos separately
    for label in ["real", "fake"]:

        class_dir = source_dir / label

        if not class_dir.is_dir():
            raise FileNotFoundError(
                f"Missing class directory: {class_dir}"
            )

        # Find videos directly inside the class directory
        videos = sorted(
            path
            for path in class_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in video_extensions
        )

        if len(videos) < 3:
            raise ValueError(
                f"The '{label}' class contains only {len(videos)} videos. "
                "At least 3 videos are required for this split."
            )

        # Shuffle videos reproducibly
        rng.shuffle(videos)

        total = len(videos)

        # Calculate split sizes
        train_count = max(1, int(total * train_ratio))
        val_count = max(1, int(total * val_ratio))

        # Ensure the test split also has at least one video
        if train_count + val_count >= total:
            train_count = total - 2
            val_count = 1

        train_videos = videos[:train_count]

        val_videos = videos[
            train_count:train_count + val_count
        ]

        test_videos = videos[
            train_count + val_count:
        ]

        # Copy videos into their respective directories
        splits = {
            "train": train_videos,
            "val": val_videos,
            "test": test_videos
        }

        for split_name, split_videos in splits.items():

            destination_dir = output_dir / split_name / label
            destination_dir.mkdir(
                parents=True,
                exist_ok=True
            )

            for video in split_videos:
                destination = destination_dir / video.name

                shutil.copy2(video, destination)

        # Print the results
        print(f"\nClass: {label.upper()}")
        print(f"Total videos: {total}")
        print(f"Training videos: {len(train_videos)}")
        print(f"Validation videos: {len(val_videos)}")
        print(f"Testing videos: {len(test_videos)}")

    print("\nDataset splitting completed!")
    print(f"Dataset location: {output_dir}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description="Split real and fake videos into train, val, and test."
    )

    parser.add_argument(
        "--source",
        default="video",
        help="Directory containing real/ and fake/ videos."
    )

    parser.add_argument(
        "--output",
        default="dataset",
        help="Output directory for the split dataset."
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducible splitting."
    )

    args = parser.parse_args()

    split_dataset(
        source_dir=args.source,
        output_dir=args.output,
        seed=args.seed
    )