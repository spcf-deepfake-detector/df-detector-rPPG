"""
FalseFacet rPPG Dataset Diagnostic
===================================

Scans videos in:

    dataset/train/real
    dataset/train/fake

    dataset/validation/real
    dataset/validation/fake

    dataset/test/real
    dataset/test/fake

Diagnostic stages:
    1. Face detection
    2. Face tracking
    3. RGB extraction
    4. POS signal processing

Safety features:
    - Processes videos sequentially.
    - Processes one video at a time.
    - Does not train a model.
    - Does not modify or delete dataset videos.
    - Saves CSV results after every video.
    - Releases preprocessing resources after every video.
    - Runs garbage collection between videos.
    - Continues to the next video if one fails.

Default:
    Tests one real and one fake video per split.

Run from project root:

    python -u -m training.diagnose_dataset

Scan every video:

    python -u -m training.diagnose_dataset --all

Scan only training videos:

    python -u -m training.diagnose_dataset --split train

Scan two real and two fake videos per split:

    python -u -m training.diagnose_dataset --limit-per-class 2

Reports:
    reports/rppg_dataset_diagnostics.csv
    reports/rppg_dataset_summary.txt
"""

import os
import csv
import glob
import gc
import argparse

from collections import Counter, defaultdict

import numpy as np

from preprocessing.face_detection import FaceDetector
from preprocessing.face_tracking import FaceTracker
from preprocessing.rgb_extraction import RGBExtractor
from preprocessing.rppg import pos_method


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_ROOTS = {
    "train": os.path.join("dataset", "train"),
    "validation": os.path.join("dataset", "validation"),
    "test": os.path.join("dataset", "test"),
}

REAL_FOLDER = "real"
FAKE_FOLDER = "fake"

MIN_DETECTION_CONFIDENCE = 0.3
MODEL_SELECTION = 1

MIN_VALID_RGB_FRACTION = 0.80

REPORT_DIR = "reports"

CSV_REPORT = os.path.join(
    REPORT_DIR,
    "rppg_dataset_diagnostics.csv",
)

SUMMARY_REPORT = os.path.join(
    REPORT_DIR,
    "rppg_dataset_summary.txt",
)


# ============================================================
# CSV STRUCTURE
# ============================================================

FIELDNAMES = [
    "split",
    "class",
    "label",
    "video_path",
    "status",
    "failure_stage",
    "failure_reason",
    "fps",
    "total_frames",
    "detected_frames",
    "detection_rate_pct",
    "tracking_valid_frames",
    "tracking_rejected_frames",
    "rgb_total_frames",
    "rgb_valid_frames",
    "rgb_missing_frames",
    "rgb_valid_pct",
    "missing_run_count",
    "missing_runs",
    "pos_signal_length",
]


# ============================================================
# VIDEO DISCOVERY
# ============================================================

def find_videos(root_dir):
    """
    Find real and fake MP4 videos.

    Returns:
        List of (video_path, class_name, label).
    """

    videos = []

    for class_name, label in (
        (REAL_FOLDER, 0),
        (FAKE_FOLDER, 1),
    ):

        class_dir = os.path.join(
            root_dir,
            class_name,
        )

        if not os.path.isdir(class_dir):
            print(
                f"[WARNING] Folder not found: {class_dir}",
                flush=True,
            )
            continue

        paths = sorted(
            glob.glob(
                os.path.join(class_dir, "*.mp4")
            )
        )

        for path in paths:
            videos.append(
                (path, class_name, label)
            )

    return videos


def select_videos(videos, limit_per_class):
    """
    Select videos without accidentally excluding fake videos.

    limit_per_class=None means all videos.
    """

    if limit_per_class is None:
        return videos

    selected = []

    counts = Counter()

    for item in videos:

        video_path, class_name, label = item

        if counts[class_name] >= limit_per_class:
            continue

        selected.append(item)

        counts[class_name] += 1

    return selected


# ============================================================
# GENERAL HELPERS
# ============================================================

def safe_close(resource, resource_name):
    """
    Safely close a preprocessing resource.
    """

    if resource is None:
        return

    close_function = getattr(
        resource,
        "close",
        None,
    )

    if not callable(close_function):
        return

    try:
        close_function()

    except Exception as exc:
        print(
            f"[WARNING] Could not close {resource_name}: {exc}",
            flush=True,
        )


def is_detected(item):
    """
    Safely interpret a detection flag.
    """

    if not isinstance(item, dict):
        return False

    value = item.get("detected", False)

    try:
        return bool(value)
    except (TypeError, ValueError):
        return False


def count_missing_runs(rgb_signals):
    """
    Count consecutive runs of invalid RGB measurements.

    Returns:
        run_count, formatted_runs
    """

    if rgb_signals is None:
        return 0, ""

    rgb = np.asarray(
        rgb_signals,
        dtype=np.float64,
    )

    if (
        rgb.ndim != 2
        or rgb.shape[1] != 3
        or len(rgb) == 0
    ):
        return 0, ""

    valid = np.all(
        np.isfinite(rgb),
        axis=1,
    )

    missing = ~valid

    runs = []

    start = None

    for index, is_missing in enumerate(missing):

        if is_missing and start is None:
            start = index

        elif not is_missing and start is not None:

            runs.append(
                (start, index - 1)
            )

            start = None

    if start is not None:
        runs.append(
            (start, len(missing) - 1)
        )

    run_text = "; ".join(
        f"{start}-{end}"
        for start, end in runs
    )

    return len(runs), run_text


def create_empty_row(
    video_path,
    split,
    class_name,
    label,
):
    """
    Create a report row with default values.
    """

    return {
        "split": split,
        "class": class_name,
        "label": label,
        "video_path": video_path,

        "status": "FAILED",
        "failure_stage": "",
        "failure_reason": "",

        "fps": "",
        "total_frames": 0,

        "detected_frames": 0,
        "detection_rate_pct": "",

        "tracking_valid_frames": 0,
        "tracking_rejected_frames": 0,

        "rgb_total_frames": 0,
        "rgb_valid_frames": 0,
        "rgb_missing_frames": 0,
        "rgb_valid_pct": "",

        "missing_run_count": 0,
        "missing_runs": "",

        "pos_signal_length": 0,
    }


# ============================================================
# MAIN VIDEO DIAGNOSTIC
# ============================================================

def diagnose_video(
    video_path,
    split,
    class_name,
    label,
):
    """
    Diagnose one video.

    Each stage is executed sequentially.

    Returns:
        A dictionary containing diagnostic results.
    """

    row = create_empty_row(
        video_path,
        split,
        class_name,
        label,
    )

    detector = None
    extractor = None

    detections = None
    tracked_detections = None
    rgb_signals = None
    signal = None

    print(
        "\n--------------------------------------------",
        flush=True,
    )

    print(
        f"VIDEO: {video_path}",
        flush=True,
    )

    print(
        "--------------------------------------------",
        flush=True,
    )

    try:

        # ====================================================
        # STAGE 1: FACE DETECTION
        # ====================================================

        row["failure_stage"] = "face_detection"

        print(
            "\n[1/4] Starting face detection...",
            flush=True,
        )

        detector = FaceDetector(
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            model_selection=MODEL_SELECTION,
        )

        detection_summary = detector.process_video(
            video_path
        )

        if not isinstance(
            detection_summary,
            dict,
        ):
            raise ValueError(
                "FaceDetector did not return a dictionary."
            )

        detections = detection_summary.get(
            "detections"
        )

        if not isinstance(detections, list):
            raise ValueError(
                "FaceDetector returned an invalid detections list."
            )

        total_frames = int(
            detection_summary.get(
                "total_frames",
                len(detections),
            )
        )

        if total_frames <= 0:
            raise ValueError(
                "Video yielded zero decoded frames."
            )

        row["total_frames"] = total_frames

        fps = detection_summary.get(
            "fps",
            30.0,
        )

        try:
            fps = float(fps)
        except (TypeError, ValueError):
            fps = 30.0

        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0

        row["fps"] = fps

        detected_frames = sum(
            1
            for item in detections
            if is_detected(item)
        )

        row["detected_frames"] = detected_frames

        row["detection_rate_pct"] = round(
            100.0 * detected_frames / total_frames,
            2,
        )

        print(
            f"[DETECTION] Frames: {total_frames}",
            flush=True,
        )

        print(
            f"[DETECTION] Faces detected in frames: "
            f"{detected_frames}",
            flush=True,
        )

        print(
            f"[DETECTION] Detection rate: "
            f"{row['detection_rate_pct']}%",
            flush=True,
        )

        # Release detector before the next stage.
        safe_close(
            detector,
            "FaceDetector",
        )

        detector = None

        gc.collect()

        # ====================================================
        # STAGE 2: FACE TRACKING
        # ====================================================

        row["failure_stage"] = "face_tracking"

        print(
            "\n[2/4] Starting face tracking...",
            flush=True,
        )

        tracker = FaceTracker()

        tracked_detections = tracker.track_faces(
            detections
        )

        if not isinstance(
            tracked_detections,
            list,
        ):
            raise ValueError(
                "FaceTracker did not return a list."
            )

        if len(tracked_detections) != len(detections):
            raise ValueError(
                "FaceTracker output length does not match "
                "the detection input length."
            )

        tracking_valid = sum(
            1
            for item in tracked_detections
            if (
                isinstance(item, dict)
                and item.get("detected", False)
                and item.get("track_valid", True)
            )
        )

        tracking_rejected = sum(
            1
            for item in tracked_detections
            if (
                isinstance(item, dict)
                and item.get("detected", False)
                and not item.get("track_valid", True)
            )
        )

        row["tracking_valid_frames"] = tracking_valid

        row["tracking_rejected_frames"] = tracking_rejected

        print(
            f"[TRACKING] Accepted: {tracking_valid}",
            flush=True,
        )

        print(
            f"[TRACKING] Rejected: {tracking_rejected}",
            flush=True,
        )

        del tracker

        gc.collect()

        # ====================================================
        # STAGE 3: RGB EXTRACTION
        # ====================================================

        row["failure_stage"] = "rgb_extraction"

        print(
            "\n[3/4] Starting RGB extraction...",
            flush=True,
        )

        extractor = RGBExtractor()

        rgb_signals = extractor.process_video_frames(
            video_path,
            tracked_detections,
        )

        if rgb_signals is None:
            raise ValueError(
                "RGBExtractor returned None."
            )

        rgb = np.asarray(
            rgb_signals,
            dtype=np.float64,
        )

        if (
            rgb.ndim != 2
            or rgb.shape[1] != 3
        ):
            raise ValueError(
                f"Unexpected RGB shape: {rgb.shape}. "
                "Expected (frames, 3)."
            )

        rgb_total = len(rgb)

        rgb_valid_mask = np.all(
            np.isfinite(rgb),
            axis=1,
        )

        valid_count = int(
            np.sum(rgb_valid_mask)
        )

        missing_count = rgb_total - valid_count

        valid_percentage = (
            100.0 * valid_count / rgb_total
            if rgb_total > 0
            else 0.0
        )

        row["rgb_total_frames"] = rgb_total

        row["rgb_valid_frames"] = valid_count

        row["rgb_missing_frames"] = missing_count

        row["rgb_valid_pct"] = round(
            valid_percentage,
            2,
        )

        run_count, run_text = count_missing_runs(
            rgb
        )

        row["missing_run_count"] = run_count

        row["missing_runs"] = run_text

        print(
            f"[RGB] Total: {rgb_total}",
            flush=True,
        )

        print(
            f"[RGB] Valid: {valid_count}",
            flush=True,
        )

        print(
            f"[RGB] Missing: {missing_count}",
            flush=True,
        )

        print(
            f"[RGB] Valid percentage: "
            f"{row['rgb_valid_pct']}%",
            flush=True,
        )

        if rgb_total != total_frames:

            print(
                "[WARNING] RGB frame count differs from "
                "the decoded video frame count.",
                flush=True,
            )

        safe_close(
            extractor,
            "RGBExtractor",
        )

        extractor = None

        gc.collect()

        # ====================================================
        # STAGE 4: POS PROCESSING
        # ====================================================

        row["failure_stage"] = "pos_processing"

        print(
            "\n[4/4] Starting POS processing...",
            flush=True,
        )

        if rgb_total == 0:
            raise ValueError(
                "RGB signal contains zero frames."
            )

        rgb_fraction = valid_count / rgb_total

        if rgb_fraction < MIN_VALID_RGB_FRACTION:

            row["failure_stage"] = (
                "insufficient_rgb_validity"
            )

            raise ValueError(
                f"Valid RGB percentage is "
                f"{row['rgb_valid_pct']}%, but "
                f"{MIN_VALID_RGB_FRACTION * 100:.2f}% "
                "is required."
            )

        signal = pos_method(
            rgb,
            fps=fps,
        )

        signal = np.asarray(
            signal,
            dtype=np.float32,
        )

        if signal.ndim != 1:
            raise ValueError(
                f"Unexpected POS signal shape: {signal.shape}"
            )

        if signal.size == 0:
            raise ValueError(
                "POS signal is empty."
            )

        if not np.all(np.isfinite(signal)):
            raise ValueError(
                "POS signal contains non-finite values."
            )

        row["pos_signal_length"] = int(
            signal.size
        )

        row["status"] = "PASSED"

        row["failure_stage"] = ""

        row["failure_reason"] = ""

        print(
            f"[POS] Signal length: {signal.size}",
            flush=True,
        )

        print(
            "\n[RESULT] PASSED",
            flush=True,
        )

    except Exception as exc:

        row["status"] = "FAILED"

        row["failure_reason"] = (
            f"{type(exc).__name__}: {exc}"
        )

        print(
            "\n[RESULT] FAILED",
            flush=True,
        )

        print(
            f"[FAILURE STAGE] {row['failure_stage']}",
            flush=True,
        )

        print(
            f"[ERROR] {row['failure_reason']}",
            flush=True,
        )

    finally:

        # Release resources even if a stage fails.
        safe_close(
            detector,
            "FaceDetector",
        )

        safe_close(
            extractor,
            "RGBExtractor",
        )

        # Release large arrays before processing another video.
        detections = None
        tracked_detections = None
        rgb_signals = None
        signal = None

        gc.collect()

    return row


# ============================================================
# CSV REPORT
# ============================================================

def initialize_csv():
    """
    Create a fresh CSV report for the current diagnostic run.
    """

    os.makedirs(
        REPORT_DIR,
        exist_ok=True,
    )

    with open(
        CSV_REPORT,
        "w",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=FIELDNAMES,
        )

        writer.writeheader()


def append_csv_row(row):
    """
    Save one completed video immediately.
    """

    with open(
        CSV_REPORT,
        "a",
        newline="",
        encoding="utf-8",
    ) as csv_file:

        writer = csv.DictWriter(
            csv_file,
            fieldnames=FIELDNAMES,
        )

        writer.writerow(row)

        csv_file.flush()


# ============================================================
# SUMMARY REPORT
# ============================================================

def write_summary(all_rows):
    """
    Generate a summary from completed diagnostic results.
    """

    counts = defaultdict(Counter)

    failure_counts = defaultdict(Counter)

    for row in all_rows:

        key = (
            row["split"],
            row["class"],
        )

        counts[key]["total"] += 1

        counts[key][
            row["status"].lower()
        ] += 1

        if row["status"] != "PASSED":

            failure_counts[key][
                row["failure_stage"] or "unknown"
            ] += 1

    passed = sum(
        row["status"] == "PASSED"
        for row in all_rows
    )

    failed = sum(
        row["status"] != "PASSED"
        for row in all_rows
    )

    lines = [
        "FALSEFACET rPPG DATASET DIAGNOSTIC SUMMARY",
        "=" * 48,
        f"Videos processed: {len(all_rows)}",
        f"Passed: {passed}",
        f"Failed: {failed}",
        "",
        "RESULTS BY SPLIT AND CLASS:",
    ]

    for split in DATASET_ROOTS:

        for class_name in (
            REAL_FOLDER,
            FAKE_FOLDER,
        ):

            key = (
                split,
                class_name,
            )

            stats = counts[key]

            if stats["total"] == 0:
                continue

            lines.append(
                f"\n{split}/{class_name}: "
                f"total={stats['total']}, "
                f"passed={stats['passed']}, "
                f"failed={stats['failed']}"
            )

            if failure_counts[key]:

                details = ", ".join(
                    f"{stage}={count}"
                    for stage, count
                    in failure_counts[key].most_common()
                )

                lines.append(
                    f"Failure stages: {details}"
                )

    overall_failures = Counter(
        row["failure_stage"]
        for row in all_rows
        if row["status"] != "PASSED"
    )

    lines.extend([
        "",
        "OVERALL FAILURE STAGES:",
    ])

    if overall_failures:

        for stage, count in overall_failures.most_common():

            lines.append(
                f"- {stage}: {count}"
            )

    else:

        lines.append("- None")

    with open(
        SUMMARY_REPORT,
        "w",
        encoding="utf-8",
    ) as summary_file:

        summary_file.write(
            "\n".join(lines) + "\n"
        )

    print(
        "\n" + "\n".join(lines),
        flush=True,
    )


# ============================================================
# MAIN
# ============================================================

def main():

    parser = argparse.ArgumentParser(
        description="FalseFacet rPPG dataset diagnostic"
    )

    parser.add_argument(
        "--all",
        action="store_true",
        help="Process every video in every selected split.",
    )

    parser.add_argument(
        "--limit-per-class",
        type=int,
        default=1,
        help="Maximum videos per class per split. Default: 1.",
    )

    parser.add_argument(
        "--split",
        choices=[
            "train",
            "validation",
            "test",
            "all",
        ],
        default="all",
        help="Dataset split to diagnose. Default: all.",
    )

    args = parser.parse_args()

    if args.limit_per_class < 1:
        parser.error(
            "--limit-per-class must be at least 1."
        )

    # --all overrides the per-class limit.
    limit_per_class = (
        None if args.all
        else args.limit_per_class
    )

    os.makedirs(
        REPORT_DIR,
        exist_ok=True,
    )

    initialize_csv()

    all_rows = []

    if args.split == "all":

        selected_splits = list(
            DATASET_ROOTS.items()
        )

    else:

        selected_splits = [
            (
                args.split,
                DATASET_ROOTS[args.split],
            )
        ]

    print(
        "\n============================================",
        flush=True,
    )

    print(
        "FALSEFACET rPPG DATASET DIAGNOSTIC",
        flush=True,
    )

    print(
        "============================================",
        flush=True,
    )

    print(
        f"Scan mode: "
        f"{'ALL VIDEOS' if args.all else 'LIMITED'}",
        flush=True,
    )

    print(
        f"Videos per class: "
        f"{'Unlimited' if limit_per_class is None else limit_per_class}",
        flush=True,
    )

    print(
        "Videos are processed sequentially.",
        flush=True,
    )

    print(
        "============================================",
        flush=True,
    )

    try:

        for split, root_dir in selected_splits:

            print(
                f"\n========== {split.upper()} ==========",
                flush=True,
            )

            videos = find_videos(root_dir)

            videos = select_videos(
                videos,
                limit_per_class,
            )

            print(
                f"Selected videos: {len(videos)}",
                flush=True,
            )

            if not videos:

                print(
                    f"[INFO] No videos found in {root_dir}",
                    flush=True,
                )

                continue

            for index, item in enumerate(
                videos,
                start=1,
            ):

                video_path, class_name, label = item

                print(
                    f"\n[{index}/{len(videos)}] "
                    f"{class_name.upper()}",
                    flush=True,
                )

                row = diagnose_video(
                    video_path,
                    split,
                    class_name,
                    label,
                )

                all_rows.append(row)

                # Save each result immediately.
                append_csv_row(row)

                print(
                    f"[SAVED] {CSV_REPORT}",
                    flush=True,
                )

                # Release memory before the next video.
                gc.collect()

    except KeyboardInterrupt:

        print(
            "\n[INTERRUPTED] Diagnostic stopped by user.",
            flush=True,
        )

        print(
            "Previously completed results remain in the CSV.",
            flush=True,
        )

    finally:

        # Always generate a summary for completed videos.
        write_summary(all_rows)

        gc.collect()

    print(
        "\n============================================",
        flush=True,
    )

    print(
        "DIAGNOSTIC FINISHED",
        flush=True,
    )

    print(
        f"Videos processed: {len(all_rows)}",
        flush=True,
    )

    print(
        f"CSV: {CSV_REPORT}",
        flush=True,
    )

    print(
        f"Summary: {SUMMARY_REPORT}",
        flush=True,
    )

    print(
        "No model training or dataset deletion was performed.",
        flush=True,
    )

    print(
        "============================================",
        flush=True,
    )


if __name__ == "__main__":
    main()