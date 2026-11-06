import os
import json
import argparse
import tempfile

from pathlib import Path

import numpy as np

from training.video_preprocessing_dataset import RPPGDataset


# ==========================================================
# SAVE NUMPY SIGNAL SAFELY
# ==========================================================

def save_signal(signal, output_path):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="wb",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as temp_file:

            temp_path = temp_file.name

            np.save(
                temp_file,
                signal,
                allow_pickle=False,
            )

            temp_file.flush()
            os.fsync(temp_file.fileno())

        os.replace(
            temp_path,
            output_path,
        )

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ==========================================================
# SAVE MANIFEST SAFELY
# ==========================================================

def save_manifest(records, output_dir):

    output_dir = Path(output_dir)

    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    manifest_path = output_dir / "manifest.json"

    temp_path = None

    try:

        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            suffix=".tmp",
            dir=output_dir,
            delete=False,
        ) as temp_file:

            temp_path = temp_file.name

            json.dump(
                records,
                temp_file,
                indent=4,
            )

            temp_file.flush()
            os.fsync(temp_file.fileno())

        os.replace(
            temp_path,
            manifest_path,
        )

    finally:

        if temp_path and os.path.exists(temp_path):
            os.remove(temp_path)


# ==========================================================
# VALIDATE EXISTING SIGNAL
# ==========================================================

def validate_existing_signal(signal_path, target_length=300):

    try:

        signal = np.load(
            signal_path,
            allow_pickle=False,
        )

        if signal.shape != (target_length,):
            return False

        if not np.issubdtype(signal.dtype, np.number):
            return False

        if not np.all(np.isfinite(signal)):
            return False

        return True

    except Exception:

        return False


# ==========================================================
# PREPROCESS ONE DATASET SPLIT
# ==========================================================

def preprocess_split(
    split_name,
    data_dir,
    output_dir,
    debug=False,
    overwrite=False,
):

    print("\n" + "=" * 60)
    print(f"PREPROCESSING: {split_name.upper()}")
    print("=" * 60)

    # ------------------------------------------------------
    # RESOLVE DIRECTORIES
    # ------------------------------------------------------

    data_dir = Path(data_dir).resolve()
    output_dir = Path(output_dir).resolve()

    split_input_dir = (
        data_dir / split_name
    ).resolve()

    split_output_dir = (
        output_dir / split_name
    )

    if not split_input_dir.is_dir():

        print(
            f"[WARNING] Dataset split not found: "
            f"{split_input_dir}"
        )

        return []

    split_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Input directory: {split_input_dir}")
    print(f"Output directory: {split_output_dir}")

    # ------------------------------------------------------
    # CREATE DATASET
    # ------------------------------------------------------

    dataset = RPPGDataset(
        root_dir=str(split_input_dir),
        target_length=300,
        fps=30.0,
        min_valid_rgb_fraction=0.80,
        min_rgb_frames=10,
        detection_confidence=0.30,
        model_selection=1,
        frame_scale=1.0,#turn into 1 temporarily for testing
        debug=debug,
    )

    records = []

    successful = 0
    skipped = 0
    failed = 0

    total = len(dataset)

    # ------------------------------------------------------
    # PROCESS VIDEOS
    # ------------------------------------------------------

    for index in range(total):

        video_path, label = dataset.samples[index]

        # Convert video path to an absolute path.
        video_path = Path(video_path).resolve()

        if label not in (0, 1):
            raise ValueError(
                f"Invalid label {label} for {video_path}"
            )

        class_name = "real" if label == 0 else "fake"

        # --------------------------------------------------
        # CALCULATE RELATIVE PATH
        # --------------------------------------------------

        try:

            relative_path = video_path.relative_to(
                split_input_dir
            )

        except ValueError as exc:

            raise ValueError(
                "Video path is outside the dataset split.\n"
                f"Video: {video_path}\n"
                f"Dataset root: {split_input_dir}"
            ) from exc

        # --------------------------------------------------
        # CREATE OUTPUT PATH
        # --------------------------------------------------

        output_path = (
            split_output_dir
            / relative_path.parent
            / f"{video_path.stem}.npy"
        )

        print("\n" + "-" * 60)

        print(
            f"[{index + 1}/{total}] "
            f"{video_path.name}"
        )

        print(f"Class: {class_name}")
        print(f"Label: {label}")
        print(f"Output: {output_path}")

        # --------------------------------------------------
        # SKIP VALID EXISTING SIGNALS
        # --------------------------------------------------

        if output_path.exists() and not overwrite:

            if validate_existing_signal(output_path):

                print(
                    "[SKIPPED] Existing signal is valid."
                )

                skipped += 1

                records.append({
                    "video": str(video_path),
                    "signal": str(output_path),
                    "label": label,
                    "class": class_name,
                    "status": "skipped_existing",
                })

                save_manifest(
                    records,
                    split_output_dir,
                )

                continue

            else:

                print(
                    "[WARNING] Existing signal is invalid. "
                    "Reprocessing video."
                )

        # --------------------------------------------------
        # PROCESS VIDEO
        # --------------------------------------------------

        try:

            x_tensor, y_tensor = dataset[index]

            signal = x_tensor.detach().cpu().numpy()

            signal = signal.astype(
                np.float32,
                copy=False,
            )

            saved_label = int(y_tensor.item())

            # ----------------------------------------------
            # VALIDATE SIGNAL
            # ----------------------------------------------

            if signal.ndim != 1:

                raise ValueError(
                    f"Expected 1D signal, got {signal.shape}"
                )

            if len(signal) != dataset.target_length:

                raise ValueError(
                    f"Expected {dataset.target_length} samples, "
                    f"got {len(signal)}"
                )

            if not np.all(np.isfinite(signal)):

                raise ValueError(
                    "Signal contains NaN or infinite values."
                )

            if saved_label != label:

                raise ValueError(
                    "Saved label does not match dataset label."
                )

            # ----------------------------------------------
            # SAVE SIGNAL
            # ----------------------------------------------

            save_signal(
                signal,
                output_path,
            )

            successful += 1

            print("[SUCCESS] Signal saved.")
            print(f"Signal shape: {signal.shape}")
            print(f"Signal dtype: {signal.dtype}")

            records.append({
                "video": str(video_path),
                "signal": str(output_path),
                "label": label,
                "class": class_name,
                "shape": list(signal.shape),
                "dtype": str(signal.dtype),
                "status": "success",
            })

        except Exception as exc:

            failed += 1

            print("[FAILED] Could not process video.")
            print(f"Reason: {exc}")

            records.append({
                "video": str(video_path),
                "label": label,
                "class": class_name,
                "status": "failed",
                "error": str(exc),
            })

        # Save progress after every video.
        save_manifest(
            records,
            split_output_dir,
        )

    # ------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------

    print("\n" + "=" * 60)
    print(f"{split_name.upper()} PREPROCESSING SUMMARY")
    print("=" * 60)

    print(f"Total videos: {total}")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")

    print(f"Output directory: {split_output_dir}")

    print("=" * 60)

    return records


# ==========================================================
# MAIN
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description="FALSEFACET rPPG Dataset Preprocessing"
    )

    parser.add_argument(
        "--data_dir",
        type=str,
        default="dataset",
        help="Original video dataset directory.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="processed",
        help="Directory for processed rPPG signals.",
    )

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed preprocessing diagnostics.",
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess signals that already exist.",
    )

    args = parser.parse_args()

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    print("\n==========================================")
    print("FALSEFACET rPPG PREPROCESSING")
    print("==========================================")

    print(f"Input: {data_dir}")
    print(f"Output: {output_dir}")

    print("Signal length: 300")
    print("Signal format: NumPy float32")

    print("==========================================")

    splits = [
        "train",
        "validation",
        "test",
    ]

    all_records = []

    for split_name in splits:

        records = preprocess_split(
            split_name=split_name,
            data_dir=data_dir,
            output_dir=output_dir,
            debug=args.debug,
            overwrite=args.overwrite,
        )

        all_records.extend(records)

    # ------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------

    print("\n==========================================")
    print("PREPROCESSING FINISHED")
    print("==========================================")

    print(f"Total recorded videos: {len(all_records)}")

    for status in [
        "success",
        "skipped_existing",
        "failed",
    ]:

        count = sum(
            1 for record in all_records
            if record["status"] == status
        )

        print(f"{status}: {count}")

    print(f"Output directory: {output_dir}")

    print("==========================================")


if __name__ == "__main__":
    main()