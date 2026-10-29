"""
FalseFacet rPPG dataset diagnostic
===================================
Scans every .mp4 video in:
    dataset/train/real
    dataset/train/fake
    dataset/validation/real
    dataset/validation/fake
    dataset/test/real
    dataset/test/fake

Writes:
    reports/rppg_dataset_diagnostics.csv
    reports/rppg_dataset_summary.txt

Run from the project root:
    python -m training.diagnose_dataset

This script diagnoses videos; it does NOT train a model or delete files.
"""

import csv
import os
import glob
import traceback
from collections import Counter, defaultdict

import numpy as np

from preprocessing.face_detection import FaceDetector
from preprocessing.face_tracking import FaceTracker
from preprocessing.rgb_extraction import RGBExtractor
from preprocessing.rppg import pos_method


# Adjust these paths if your dataset uses different folder names.
DATASET_ROOTS = {
    "train": os.path.join("dataset", "train"),
    "validation": os.path.join("dataset", "validation"),
    "test": os.path.join("dataset", "test"),
}

REAL_FOLDER = "real"
FAKE_FOLDER = "fake"

# Match the detector configuration currently used in dataset.py.
MIN_DETECTION_CONFIDENCE = 0.3
MODEL_SELECTION = 1

# Keep this consistent with prepare_rgb_signal() in preprocessing/rppg.py.
MIN_VALID_RGB_FRACTION = 0.80

REPORT_DIR = "reports"
CSV_REPORT = os.path.join(REPORT_DIR, "rppg_dataset_diagnostics.csv")
SUMMARY_REPORT = os.path.join(REPORT_DIR, "rppg_dataset_summary.txt")


def find_videos(root_dir):
    """Return sorted .mp4 files from the real and fake folders."""
    videos = []

    for class_name, label in ((REAL_FOLDER, 0), (FAKE_FOLDER, 1)):
        class_dir = os.path.join(root_dir, class_name)

        if not os.path.isdir(class_dir):
            print(f"[WARNING] Folder not found: {class_dir}")
            continue

        for path in sorted(glob.glob(os.path.join(class_dir, "*.mp4"))):
            videos.append((path, class_name, label))

    return videos


def count_missing_runs(rgb_signals):
    """Count consecutive runs of frames with missing/non-finite RGB."""
    if rgb_signals is None:
        return 0, ""

    rgb = np.asarray(rgb_signals, dtype=np.float64)
    if rgb.ndim != 2 or rgb.shape[1] != 3 or len(rgb) == 0:
        return 0, ""

    valid = np.all(np.isfinite(rgb), axis=1)
    missing = ~valid
    runs = []
    start = None

    for i, is_missing in enumerate(missing):
        if is_missing and start is None:
            start = i
        elif not is_missing and start is not None:
            runs.append((start, i - 1))
            start = None

    if start is not None:
        runs.append((start, len(missing) - 1))

    # Store a compact representation in the CSV.
    return len(runs), "; ".join(f"{a}-{b}" for a, b in runs)


def diagnose_video(video_path, split, class_name, label):
    """Run the existing preprocessing stages and return one report row."""
    row = {
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
        "rgb_valid_frames": 0,
        "rgb_missing_frames": 0,
        "rgb_valid_pct": "",
        "missing_run_count": 0,
        "missing_runs": "",
        "pos_signal_length": 0,
    }

    detector = None
    extractor = None

    try:
        # Stage 1: face detection
        row["failure_stage"] = "face_detection"
        detector = FaceDetector(
            min_detection_confidence=MIN_DETECTION_CONFIDENCE,
            model_selection=MODEL_SELECTION,
        )

        try:
            detection_summary = detector.process_video(video_path)
        finally:
            detector.close()
            detector = None

        if not isinstance(detection_summary, dict):
            raise ValueError("FaceDetector did not return a dictionary.")

        detections = detection_summary.get("detections")
        if detections is None:
            raise ValueError("FaceDetector returned no 'detections' list.")

        total_frames = int(detection_summary.get("total_frames", len(detections)))
        row["total_frames"] = total_frames
        row["fps"] = detection_summary.get("fps", "")
        detected_frames = sum(
            1 for item in detections
            if isinstance(item, dict) and item.get("detected") is True
        )
        row["detected_frames"] = detected_frames
        row["detection_rate_pct"] = (
            round(100.0 * detected_frames / total_frames, 2)
            if total_frames else 0.0
        )

        if total_frames == 0:
            raise ValueError("Video yielded zero decoded frames.")

        # Stage 2: tracking
        row["failure_stage"] = "face_tracking"
        tracker = FaceTracker()
        tracked_detections = tracker.track_faces(detections)

        if tracked_detections is None:
            raise ValueError("FaceTracker returned None.")

        tracking_valid = sum(
            1 for item in tracked_detections
            if isinstance(item, dict)
            and item.get("detected", False)
            and item.get("track_valid", True)
        )
        tracking_rejected = sum(
            1 for item in tracked_detections
            if isinstance(item, dict)
            and item.get("detected", False)
            and not item.get("track_valid", True)
        )
        row["tracking_valid_frames"] = tracking_valid
        row["tracking_rejected_frames"] = tracking_rejected

        # Stage 3: RGB extraction
        row["failure_stage"] = "rgb_extraction"
        extractor = RGBExtractor()

        try:
            rgb_signals = extractor.process_video_frames(
                video_path,
                tracked_detections,
            )
        finally:
            extractor.close()
            extractor = None

        rgb = np.asarray(rgb_signals, dtype=np.float64)
        if rgb.ndim != 2 or rgb.shape[1] != 3:
            raise ValueError(
                f"RGBExtractor returned shape {rgb.shape}; expected (frames, 3)."
            )

        rgb_valid = np.all(np.isfinite(rgb), axis=1)
        valid_count = int(np.sum(rgb_valid))
        total_rgb = len(rgb)
        missing_count = total_rgb - valid_count
        valid_pct = (100.0 * valid_count / total_rgb) if total_rgb else 0.0

        row["rgb_valid_frames"] = valid_count
        row["rgb_missing_frames"] = missing_count
        row["rgb_valid_pct"] = round(valid_pct, 2)
        run_count, run_text = count_missing_runs(rgb)
        row["missing_run_count"] = run_count
        row["missing_runs"] = run_text

        # Stage 4: POS signal. This calls the project's existing validator,
        # including its current RGB validity threshold.
        row["failure_stage"] = "pos_processing"
        fps = detection_summary.get("fps", 30.0)
        try:
            fps = float(fps)
        except (TypeError, ValueError):
            fps = 30.0
        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0

        signal = pos_method(rgb, fps=fps)
        signal = np.asarray(signal, dtype=np.float32)

        if signal.ndim != 1:
            raise ValueError(f"POS signal has unexpected shape {signal.shape}.")
        if signal.size == 0:
            raise ValueError("POS signal is empty.")
        if not np.all(np.isfinite(signal)):
            raise ValueError("POS signal contains non-finite values.")

        row["pos_signal_length"] = int(signal.size)
        row["status"] = "PASSED"
        row["failure_stage"] = ""
        row["failure_reason"] = ""

    except Exception as exc:
        row["status"] = "FAILED"
        row["failure_reason"] = f"{type(exc).__name__}: {exc}"

        # If the POS validator rejected a video because RGB validity is low,
        # report that more specifically in the stage column.
        if row["failure_stage"] == "pos_processing":
            fraction = (
                float(row["rgb_valid_pct"]) / 100.0
                if row["rgb_valid_pct"] != "" else None
            )
            if fraction is not None and fraction < MIN_VALID_RGB_FRACTION:
                row["failure_stage"] = "insufficient_rgb_validity"

    finally:
        if detector is not None:
            try:
                detector.close()
            except Exception:
                pass
        if extractor is not None:
            try:
                extractor.close()
            except Exception:
                pass

    return row


def main():
    os.makedirs(REPORT_DIR, exist_ok=True)

    videos = find_videos(DATASET_ROOTS["train"])

    if not videos:
        print("No training videos found.")
        return

    video_path, class_name, label = videos[0]

    print(f"Testing one video: {video_path}", flush=True)

    row = diagnose_video(video_path, "train", class_name, label)

    print("\nDiagnostic result:", flush=True)
    print(row, flush=True)

    all_rows = []
    for split, root_dir in DATASET_ROOTS.items():
        videos = find_videos(root_dir)
        videos = videos[:1]

        if not videos:
            print(f"[INFO] No videos found under: {root_dir}")
            continue

        print(f"\n========== SCANNING {split.upper()} ==========")
        print(f"Root: {root_dir}")
        print(f"Videos found: {len(videos)}")

        for index, (video_path, class_name, label) in enumerate(videos, start=1):
            print(f"\n[{index}/{len(videos)}] {video_path}")
            row = diagnose_video(video_path, split, class_name, label)
            all_rows.append(row)

            if row["status"] == "PASSED":
                print(
                    f"[PASS] RGB valid={row['rgb_valid_pct']}%, "
                    f"POS length={row['pos_signal_length']}"
                )
            else:
                print(
                    f"[FAIL] Stage={row['failure_stage']} | "
                    f"{row['failure_reason']}"
                )

    if not all_rows:
        print(
            "\nNo videos were found. Check DATASET_ROOTS near the top "
            "of this script."
        )
        return

    # CSV report
    fieldnames = list(all_rows[0].keys())
    with open(CSV_REPORT, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)

    # Summaries by split and class
    counts = defaultdict(Counter)
    failure_counts = defaultdict(Counter)

    for row in all_rows:
        key = (row["split"], row["class"])
        counts[key]["total"] += 1
        counts[key][row["status"].lower()] += 1
        if row["status"] != "PASSED":
            failure_counts[key][row["failure_stage"] or "unknown"] += 1

    lines = [
        "FALSEFACET rPPG DATASET DIAGNOSTIC SUMMARY",
        "=" * 46,
        f"Videos scanned: {len(all_rows)}",
        f"Passed: {sum(r['status'] == 'PASSED' for r in all_rows)}",
        f"Failed: {sum(r['status'] != 'PASSED' for r in all_rows)}",
        "",
        "Results by split and class:",
    ]

    for split in DATASET_ROOTS:
        for class_name in (REAL_FOLDER, FAKE_FOLDER):
            key = (split, class_name)
            stats = counts[key]
            if stats["total"] == 0:
                continue

            lines.append(
                f"- {split}/{class_name}: total={stats['total']}, "
                f"passed={stats['passed']}, failed={stats['failed']}"
            )

            if failure_counts[key]:
                details = ", ".join(
                    f"{stage}={count}"
                    for stage, count in failure_counts[key].most_common()
                )
                lines.append(f"  Failure stages: {details}")

    # Overall failure-stage counts
    overall_failures = Counter(
        row["failure_stage"]
        for row in all_rows
        if row["status"] != "PASSED"
    )
    lines.extend(["", "Overall failure stages:"])
    if overall_failures:
        for stage, count in overall_failures.most_common():
            lines.append(f"- {stage}: {count}")
    else:
        lines.append("- None")

    with open(SUMMARY_REPORT, "w", encoding="utf-8") as summary_file:
        summary_file.write("\n".join(lines) + "\n")

    print("\n\n" + "\n".join(lines))
    print(f"\nCSV report saved to: {CSV_REPORT}")
    print(f"Summary saved to: {SUMMARY_REPORT}")
    print("\nThis diagnostic did not train a model or delete any videos.")


if __name__ == "__main__":
    main()