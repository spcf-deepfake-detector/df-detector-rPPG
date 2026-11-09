import os
import json
import argparse
import tempfile

from pathlib import Path

import numpy as np

from training.video_preprocessing_dataset import RPPGDataset
from training.dataset import RPPGDataset


# ==========================================================
# CONFIGURATION
# ==========================================================

TARGET_LENGTH = 300
TARGET_FPS = 30.0

MIN_VALID_RGB_FRACTION = 0.80
MIN_RGB_FRAMES = 10

# Input now contains extracted face-image sequences.
DEFAULT_DATA_DIR = "extracted_faces"

# Output contains processed NumPy rPPG signals.
DEFAULT_OUTPUT_DIR = "processed"


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
# LOAD EXISTING MANIFEST
# ==========================================================

def load_manifest(output_dir):

    output_dir = Path(output_dir)

    manifest_path = output_dir / "manifest.json"

    if not manifest_path.exists():
        return []

    try:

        with open(
            manifest_path,
            "r",
            encoding="utf-8",
        ) as file:

            records = json.load(file)

        if not isinstance(records, list):

            raise ValueError(
                "Manifest must contain a JSON list."
            )

        return records

    except Exception as exc:

        raise RuntimeError(
            f"Could not load manifest: {manifest_path}\n"
            f"Reason: {exc}"
        ) from exc


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
# INSERT OR UPDATE MANIFEST RECORD
# ==========================================================

def upsert_manifest_record(records, new_record):

    sequence = new_record["sequence"]

    for index, existing_record in enumerate(records):

        if existing_record.get("sequence") == sequence:

            records[index] = new_record
            return

    records.append(new_record)


# ==========================================================
# VALIDATE EXISTING SIGNAL
# ==========================================================

def validate_existing_signal(
    signal_path,
    expected_label,
    target_length=TARGET_LENGTH,
):

    try:

        signal = np.load(
            signal_path,
            allow_pickle=False,
        )

        if signal.shape != (target_length,):
            return False

        if not np.issubdtype(
            signal.dtype,
            np.number,
        ):
            return False

        if not np.all(np.isfinite(signal)):
            return False

        return True

    except Exception:

        return False


# ==========================================================
# VALIDATE NEW SIGNAL
# ==========================================================

def validate_signal(
    signal,
    expected_length,
    expected_label,
    saved_label,
):

    if signal.ndim != 1:

        raise ValueError(
            f"Expected a 1D signal, got shape {signal.shape}"
        )

    if len(signal) != expected_length:

        raise ValueError(
            f"Expected {expected_length} samples, "
            f"got {len(signal)}"
        )

    if not np.issubdtype(
        signal.dtype,
        np.number,
    ):

        raise ValueError(
            f"Signal must be numeric, got {signal.dtype}"
        )

    if not np.all(np.isfinite(signal)):

        raise ValueError(
            "Signal contains NaN or infinite values."
        )

    if saved_label != expected_label:

        raise ValueError(
            f"Label mismatch: expected {expected_label}, "
            f"got {saved_label}"
        )


# ==========================================================
# PREPROCESS ONE DATASET SPLIT
# ==========================================================

def preprocess_split(
    split_name,
    data_dir,
    output_dir,
    debug=False,
    overwrite=False,
    limit=None,
):

    print("\n" + "=" * 65)
    print(f"PREPROCESSING: {split_name.upper()}")
    print("=" * 65)

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
    ).resolve()

    if not split_input_dir.is_dir():

        print(
            f"[WARNING] Dataset split not found:\n"
            f"{split_input_dir}"
        )

        return []

    split_output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    print(f"Input directory:  {split_input_dir}")
    print(f"Output directory: {split_output_dir}")

    # ------------------------------------------------------
    # LOAD EXISTING MANIFEST
    # ------------------------------------------------------

    records = load_manifest(
        split_output_dir
    )

    print(
        f"Existing manifest records: {len(records)}"
    )

    # ------------------------------------------------------
    # CREATE IMAGE-BASED DATASET
    # ------------------------------------------------------

    print("\nInitializing RPPGDataset...")

    dataset = RPPGDataset(
        root_dir=str(split_input_dir),

        target_length=TARGET_LENGTH,
        fps=TARGET_FPS,

        min_valid_rgb_fraction=MIN_VALID_RGB_FRACTION,
        min_rgb_frames=MIN_RGB_FRAMES,

        debug=debug,
    )

    # ------------------------------------------------------
    # DETERMINE PROCESSING LIMIT
    # ------------------------------------------------------

    dataset_total = len(dataset)

    if limit is None:

        total = dataset_total

    else:

        total = min(
            limit,
            dataset_total,
        )

    print("\nDataset information:")
    print(f"Available sequences: {dataset_total}")
    print(f"Sequences to process: {total}")

    if total == 0:

        print("[WARNING] No sequences to process.")

        return []

    # ------------------------------------------------------
    # COUNTERS
    # ------------------------------------------------------

    successful = 0
    skipped = 0
    failed = 0

    run_records = []

    # ------------------------------------------------------
    # PROCESS SEQUENCES
    # ------------------------------------------------------

    for index in range(total):

        sequence_dir, label = dataset.samples[index]

        sequence_dir = Path(sequence_dir).resolve()

        print("\n" + "-" * 65)

        print(
            f"[{index + 1}/{total}] "
            f"{sequence_dir.name}"
        )

        try:

            # ----------------------------------------------
            # VALIDATE LABEL
            # ----------------------------------------------

            if label not in (0, 1):

                raise ValueError(
                    f"Invalid label: {label}"
                )

            label = int(label)

            class_name = (
                "real" if label == 0 else "fake"
            )

            print(f"Class: {class_name}")
            print(f"Label: {label}")

            # ----------------------------------------------
            # VALIDATE SEQUENCE DIRECTORY
            # ----------------------------------------------

            if not sequence_dir.is_dir():

                raise FileNotFoundError(
                    f"Sequence directory not found: "
                    f"{sequence_dir}"
                )

            metadata_path = (
                sequence_dir / "metadata.json"
            )

            if not metadata_path.is_file():

                raise FileNotFoundError(
                    f"metadata.json not found: "
                    f"{metadata_path}"
                )

            # ----------------------------------------------
            # CALCULATE RELATIVE SEQUENCE PATH
            # ----------------------------------------------

            try:

                relative_path = sequence_dir.relative_to(
                    split_input_dir
                )

            except ValueError as exc:

                raise ValueError(
                    "Sequence directory is outside the dataset split.\n"
                    f"Sequence: {sequence_dir}\n"
                    f"Dataset root: {split_input_dir}"
                ) from exc

            # ----------------------------------------------
            # CREATE OUTPUT PATH
            # ----------------------------------------------

            output_path = (
                split_output_dir
                / relative_path
            ).with_suffix(".npy")

            print(f"Sequence: {sequence_dir}")
            print(f"Output:   {output_path}")

            # ----------------------------------------------
            # SKIP VALID EXISTING SIGNAL
            # ----------------------------------------------

            if output_path.exists() and not overwrite:

                if validate_existing_signal(
                    output_path,
                    expected_label=label,
                    target_length=TARGET_LENGTH,
                ):

                    existing_signal = np.load(
                        output_path,
                        allow_pickle=False,
                    )

                    print(
                        "[SKIPPED] Existing signal is valid."
                    )

                    skipped += 1

                    record = {
                        "sequence": str(relative_path),
                        "signal": str(
                            output_path.relative_to(output_dir)
                        ),
                        "label": label,
                        "class": class_name,
                        "shape": list(existing_signal.shape),
                        "dtype": str(existing_signal.dtype),
                        "status": "skipped_existing",
                    }

                    upsert_manifest_record(
                        records,
                        record,
                    )

                    run_records.append(record)

                    save_manifest(
                        records,
                        split_output_dir,
                    )

                    continue

                print(
                    "[WARNING] Existing signal is invalid. "
                    "Reprocessing sequence."
                )

            # ----------------------------------------------
            # PROCESS SEQUENCE THROUGH RPPGDATASET
            # ----------------------------------------------

            print(
                "[PROCESSING] Extracting rPPG signal "
                "from cropped face images..."
            )

            x_tensor, y_tensor = dataset[index]

            # ----------------------------------------------
            # CONVERT TENSOR TO NUMPY
            # ----------------------------------------------

            signal = (
                x_tensor
                .detach()
                .cpu()
                .numpy()
            )

            signal = signal.astype(
                np.float32,
                copy=False,
            )

            saved_label = int(
                y_tensor.item()
            )

            # ----------------------------------------------
            # VALIDATE SIGNAL
            # ----------------------------------------------

            validate_signal(
                signal=signal,
                expected_length=TARGET_LENGTH,
                expected_label=label,
                saved_label=saved_label,
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

            # ----------------------------------------------
            # CREATE MANIFEST RECORD
            # ----------------------------------------------

            record = {
                "sequence": str(relative_path),
                "signal": str(
                    output_path.relative_to(output_dir)
                ),
                "label": label,
                "class": class_name,
                "shape": list(signal.shape),
                "dtype": str(signal.dtype),
                "status": "success",
            }

            upsert_manifest_record(
                records,
                record,
            )

            run_records.append(record)

        except Exception as exc:

            failed += 1

            print("[FAILED] Could not process sequence.")
            print(f"Reason: {exc}")

            record = {
                "sequence": str(sequence_dir),
                "label": (
                    int(label)
                    if label in (0, 1)
                    else str(label)
                ),
                "status": "failed",
                "error": str(exc),
            }

            upsert_manifest_record(
                records,
                record,
            )

            run_records.append(record)

        # --------------------------------------------------
        # SAVE PROGRESS AFTER EVERY SEQUENCE
        # --------------------------------------------------

        save_manifest(
            records,
            split_output_dir,
        )

    # ------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------

    print("\n" + "=" * 65)
    print(f"{split_name.upper()} PREPROCESSING SUMMARY")
    print("=" * 65)

    print(f"Available sequences: {dataset_total}")
    print(f"Sequences selected: {total}")
    print(f"Successful: {successful}")
    print(f"Skipped: {skipped}")
    print(f"Failed: {failed}")

    print(f"Manifest records: {len(records)}")
    print(f"Output directory: {split_output_dir}")

    print("=" * 65)

    return run_records


# ==========================================================
# MAIN
# ==========================================================

def main():

    parser = argparse.ArgumentParser(
        description="FALSEFACET rPPG Image-Based Preprocessing"
    )

    # ------------------------------------------------------
    # DATASET DIRECTORIES
    # ------------------------------------------------------

    parser.add_argument(
        "--data_dir",
        type=str,
        default=DEFAULT_DATA_DIR,
        help="Extracted face-image dataset directory.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory for processed rPPG signals.",
    )

    # ------------------------------------------------------
    # DATASET SPLIT
    # ------------------------------------------------------

    parser.add_argument(
        "--split",
        type=str,
        choices=[
            "all",
            "train",
            "val",
            "validation",
            "test",
        ],
        default="all",
        help="Dataset split to preprocess.",
    )

    # ------------------------------------------------------
    # PROCESSING LIMIT
    # ------------------------------------------------------

    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of sequences to process per split.",
    )

    # ------------------------------------------------------
    # DEBUGGING
    # ------------------------------------------------------

    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable detailed preprocessing diagnostics.",
    )

    # ------------------------------------------------------
    # OVERWRITE
    # ------------------------------------------------------

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Reprocess signals that already exist.",
    )

    args = parser.parse_args()

    # ------------------------------------------------------
    # VALIDATE ARGUMENTS
    # ------------------------------------------------------

    if args.limit is not None and args.limit <= 0:

        parser.error(
            "--limit must be greater than 0."
        )

    # ------------------------------------------------------
    # RESOLVE DIRECTORIES
    # ------------------------------------------------------

    data_dir = Path(args.data_dir).resolve()
    output_dir = Path(args.output_dir).resolve()

    # ------------------------------------------------------
    # SELECT SPLITS
    # ------------------------------------------------------

    if args.split == "all":

        splits = [
            "train",
            "val",
            "test",
        ]

    else:

        selected_split = (
            "val"
            if args.split == "validation"
            else args.split
        )

        splits = [selected_split]

    # ------------------------------------------------------
    # START PREPROCESSING
    # ------------------------------------------------------

    print("\n" + "=" * 65)
    print("FALSEFACET rPPG PREPROCESSING")
    print("=" * 65)

    print(f"Input directory: {data_dir}")
    print(f"Output directory: {output_dir}")

    print(f"Selected split: {args.split}")
    print(f"Processing limit: {args.limit}")

    print(f"Signal length: {TARGET_LENGTH}")
    print(f"Reference FPS: {TARGET_FPS}")
    print("Input format: Cropped face images")
    print("Output format: NumPy float32")

    print("=" * 65)

    all_records = []

    for split_name in splits:

        records = preprocess_split(
            split_name=split_name,
            data_dir=data_dir,
            output_dir=output_dir,
            debug=args.debug,
            overwrite=args.overwrite,
            limit=args.limit,
        )

        all_records.extend(records)

    # ------------------------------------------------------
    # FINAL SUMMARY
    # ------------------------------------------------------

    print("\n" + "=" * 65)
    print("PREPROCESSING FINISHED")
    print("=" * 65)

    print(
        f"Sequences processed in this run: "
        f"{len(all_records)}"
    )

    for status in [
        "success",
        "skipped_existing",
        "failed",
    ]:

        count = sum(
            1
            for record in all_records
            if record["status"] == status
        )

        print(f"{status}: {count}")

    print(f"Output directory: {output_dir}")

    print("=" * 65)


# ==========================================================
# ENTRY POINT
# ==========================================================

if __name__ == "__main__":
    main()