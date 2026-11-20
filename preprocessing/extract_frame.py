import cv2
import json
import argparse
import math
import os
import shutil
import tempfile

from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

VIDEO_EXTENSIONS = {
    ".mp4", ".avi", ".mov", ".mkv",
    ".webm", ".mpeg", ".mpg", ".m4v",
}

IMAGE_EXTENSION = ".jpg"

METADATA_FILENAME = "metadata.json"
SUCCESS_FILENAME = "_SUCCESS"


# ============================================================
# GET VIDEO METADATA
# ============================================================

def get_video_metadata(cap):
    """
    PURPOSE:
        Read the original video's basic properties.

    RETURNS:
        FPS, reported frame count, dimensions, and estimated duration.
    """

    fps = float(cap.get(cv2.CAP_PROP_FPS))

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if not math.isfinite(fps) or fps <= 0:
        fps = None

    duration = (
        total_frames / fps
        if fps is not None and total_frames > 0
        else None
    )

    return {
        "original_fps": fps,
        "total_frames_reported": total_frames,
        "width": width,
        "height": height,
        "duration_seconds": duration,
    }


# ============================================================
# ATOMIC JSON SAVING
# ============================================================

def save_json_atomic(data, output_path):
    """
    PURPOSE:
        Save JSON safely.

    A temporary file is written first. The final metadata file
        is replaced only after the temporary file is complete.
    """

    output_path = Path(output_path)

    temporary_path = output_path.with_name(
        output_path.name + ".tmp"
    )

    try:
        with open(
            temporary_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                data,
                file,
                indent=4,
                allow_nan=False
            )

            file.flush()
            os.fsync(file.fileno())

        os.replace(
            temporary_path,
            output_path
        )

    finally:
        temporary_path.unlink(missing_ok=True)


# ============================================================
# VALIDATE EXISTING EXTRACTION
# ============================================================

def is_extraction_complete(output_dir):
    """
    PURPOSE:
        Determine whether a video's frame extraction is complete.

    CHECKS:
        1. Success marker exists.
        2. Metadata exists and is valid.
        3. Extracted frame count is positive.
        4. Frame metadata count matches.
        5. Expected sequential frame filenames exist.
        6. Frame files are not empty.
    """

    output_dir = Path(output_dir)

    success_marker = output_dir / SUCCESS_FILENAME
    metadata_path = output_dir / METADATA_FILENAME

    if not success_marker.is_file():
        return False

    if not metadata_path.is_file():
        return False

    try:
        with open(
            metadata_path,
            "r",
            encoding="utf-8"
        ) as file:

            metadata = json.load(file)

        expected_count = metadata.get("extracted_frames")
        frames_metadata = metadata.get("frames")

        if (
            isinstance(expected_count, bool)
            or not isinstance(expected_count, int)
            or expected_count <= 0
        ):
            return False

        if not isinstance(frames_metadata, list):
            return False

        if len(frames_metadata) != expected_count:
            return False

        existing_frames = sorted(
            output_dir.glob(f"frame_*{IMAGE_EXTENSION}")
        )

        if len(existing_frames) != expected_count:
            return False

        for index, frame_path in enumerate(existing_frames):

            expected_name = (
                f"frame_{index:05d}{IMAGE_EXTENSION}"
            )

            if frame_path.name != expected_name:
                return False

            if not frame_path.is_file():
                return False

            if frame_path.stat().st_size <= 0:
                return False

            record = frames_metadata[index]

            if not isinstance(record, dict):
                return False

            if record.get("frame_index") != index:
                return False

            if record.get("filename") != expected_name:
                return False

            timestamp = record.get("timestamp")

            if (
                not isinstance(timestamp, (int, float))
                or isinstance(timestamp, bool)
                or not math.isfinite(timestamp)
            ):
                return False

        return True

    except (
        OSError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ):
        return False


# ============================================================
# EXTRACT FRAMES INTO TEMPORARY DIRECTORY
# ============================================================

def _extract_video_to_directory(video_path, temporary_dir):
    """
    PURPOSE:
        Extract one video's frames into a temporary directory.

    IMPORTANT:
        The original output directory is not modified here.
        This allows the previous successful extraction to remain
        available if the new extraction fails.
    """

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Cannot open video: {video_path}"
        )

    try:
        video_metadata = get_video_metadata(cap)

        original_fps = video_metadata["original_fps"]

        if original_fps is None:
            raise ValueError(
                f"Invalid video FPS: {video_path}"
            )

        frame_count = 0
        frame_metadata = []

        while True:
            success, frame = cap.read()

            if not success:
                break

            if frame is None or frame.size == 0:
                raise ValueError(
                    f"Invalid frame at index {frame_count}"
                )

            if (
                frame.shape[1] != video_metadata["width"]
                or frame.shape[0] != video_metadata["height"]
            ):
                raise ValueError(
                    f"Unexpected frame dimensions at index {frame_count}"
                )

            frame_index = frame_count

            # Assumes constant-frame-rate video.
            timestamp = frame_index / original_fps

            frame_name = (
                f"frame_{frame_index:05d}{IMAGE_EXTENSION}"
            )

            frame_path = temporary_dir / frame_name

            saved = cv2.imwrite(
                str(frame_path),
                frame
            )

            if not saved:
                raise OSError(
                    f"Failed to save frame: {frame_path}"
                )

            frame_metadata.append({
                "frame_index": frame_index,
                "timestamp": float(timestamp),
                "filename": frame_name,
            })

            frame_count += 1

        if frame_count == 0:
            raise ValueError(
                f"No frames extracted from: {video_path}"
            )

        metadata = {
            "source_video": video_path.name,
            "source_path": str(video_path.resolve()),

            "original_fps": float(original_fps),

            "total_frames_reported": video_metadata[
                "total_frames_reported"
            ],

            "extracted_frames": frame_count,

            "width": video_metadata["width"],
            "height": video_metadata["height"],

            "duration_seconds": (
                frame_count / original_fps
            ),

            "timestamp_method": (
                "frame_index_divided_by_original_fps"
            ),

            "frames": frame_metadata,
        }

        metadata_path = temporary_dir / METADATA_FILENAME

        save_json_atomic(
            metadata,
            metadata_path
        )

        # The success marker is created last.
        success_marker = temporary_dir / SUCCESS_FILENAME

        success_marker.write_text(
            f"Successfully extracted {frame_count} frames.\n",
            encoding="utf-8"
        )

        return frame_count, metadata

    finally:
        cap.release()


# ============================================================
# SAFELY REPLACE EXISTING OUTPUT
# ============================================================

def _replace_output_directory(temporary_dir, output_dir):
    """
    PURPOSE:
        Replace an existing extraction only after the new
        extraction has completed successfully.

    If replacement fails, attempt to restore the previous output.
    """

    temporary_dir = Path(temporary_dir)
    output_dir = Path(output_dir)

    backup_dir = None

    if output_dir.exists():

        backup_dir = Path(
            tempfile.mkdtemp(
                prefix=f".{output_dir.name}.backup-",
                dir=output_dir.parent
            )
        )

        backup_dir.rmdir()

        os.replace(
            output_dir,
            backup_dir
        )

    try:
        os.replace(
            temporary_dir,
            output_dir
        )

    except Exception:

        if backup_dir is not None and backup_dir.exists():
            os.replace(
                backup_dir,
                output_dir
            )

        raise

    else:
        if backup_dir is not None and backup_dir.exists():
            shutil.rmtree(backup_dir)


# ============================================================
# EXTRACT FRAMES FROM ONE VIDEO
# ============================================================

def extract_frames(video_path, output_dir, overwrite=False):
    """
    PURPOSE:
        Extract every frame from one video.

    OUTPUT:
        A dedicated video directory containing:
            - frame_00000.jpg
            - frame_00001.jpg
            - ...
            - metadata.json
            - _SUCCESS

    RETURNS:
        Number of newly extracted frames.
        Returns 0 when an existing complete extraction is skipped.
    """

    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.is_file():
        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    output_dir.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    if output_dir.exists() and not output_dir.is_dir():
        raise NotADirectoryError(
            f"Output path is not a directory: {output_dir}"
        )

    # Skip completed extractions.
    if not overwrite and is_extraction_complete(output_dir):

        print(
            f"[SKIPPED] Already extracted: {video_path.name}"
        )

        return 0

    # Create a temporary directory beside the final output.
    temporary_dir = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent
        )
    )

    try:
        frame_count, metadata = _extract_video_to_directory(
            video_path,
            temporary_dir
        )

        # Validate temporary output before replacing old output.
        if not is_extraction_complete(temporary_dir):
            raise RuntimeError(
                f"Temporary extraction validation failed: {video_path}"
            )

        _replace_output_directory(
            temporary_dir,
            output_dir
        )

    except Exception:
        if temporary_dir.exists():
            shutil.rmtree(
                temporary_dir,
                ignore_errors=True
            )

        raise

    print(f"\n[SUCCESS] {video_path.name}")
    print(f"Frames extracted: {frame_count}")
    print(f"Original FPS: {metadata['original_fps']}")
    print(
        f"Duration: {metadata['duration_seconds']:.2f} seconds"
    )
    print(f"Metadata: {output_dir / METADATA_FILENAME}")
    print(f"Output: {output_dir}")

    return frame_count


# ============================================================
# FIND ALL VIDEOS
# ============================================================

def find_videos(input_dir):
    """
    PURPOSE:
        Recursively discover supported video files.
    """

    input_dir = Path(input_dir)

    if not input_dir.is_dir():
        raise FileNotFoundError(
            f"Input directory not found: {input_dir}"
        )

    videos = [
        path
        for path in input_dir.rglob("*")
        if path.is_file()
        and path.suffix.lower() in VIDEO_EXTENSIONS
    ]

    return sorted(videos)


# ============================================================
# BATCH EXTRACTION
# ============================================================

def batch_extract(
    input_dir,
    output_dir,
    overwrite=False,
):
    """
    PURPOSE:
        Extract all videos while preserving the original
        directory structure.

    EXAMPLE:
        dataset/train/real/video001.mp4

        becomes:

        extracted_frames/train/real/video001/
    """

    input_dir = Path(input_dir)
    output_dir = Path(output_dir)

    videos = find_videos(input_dir)

    if not videos:
        print("No videos found.")

        return {
            "total_videos": 0,
            "successful": 0,
            "skipped": 0,
            "failed": 0,
            "total_frames": 0,
        }

    print("\n======================================")
    print(" FALSEFACET BATCH FRAME EXTRACTION")
    print("======================================")

    print(f"Videos found: {len(videos)}")

    total_frames = 0
    successful = 0
    skipped = 0
    failed = 0

    for index, video in enumerate(videos, start=1):

        print(
            f"\n[{index}/{len(videos)}] Processing: {video}"
        )

        relative_path = video.relative_to(input_dir)

        video_output = (
            output_dir
            / relative_path.parent
            / video.stem
        )

        try:
            frames = extract_frames(
                video_path=video,
                output_dir=video_output,
                overwrite=overwrite,
            )

            if frames == 0:
                skipped += 1

            else:
                successful += 1
                total_frames += frames

        except Exception as error:
            failed += 1

            print(f"[FAILED] {video}")
            print(f"Reason: {error}")

    print("\n======================================")
    print(" EXTRACTION SUMMARY")
    print("======================================")

    print(f"Total videos: {len(videos)}")
    print(f"Successfully extracted: {successful}")
    print(f"Already extracted: {skipped}")
    print(f"Failed: {failed}")

    print(
        f"Total newly extracted frames: {total_frames}"
    )

    print(
        f"\nOutput directory: {output_dir.resolve()}"
    )

    return {
        "total_videos": len(videos),
        "successful": successful,
        "skipped": skipped,
        "failed": failed,
        "total_frames": total_frames,
    }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        description=(
            "FALSEFACET: Extract video frames and generate metadata."
        )
    )

    parser.add_argument(
        "--video",
        help="Extract frames from one video."
    )

    parser.add_argument(
        "--input",
        default="dataset",
        help="Input dataset directory."
    )

    parser.add_argument(
        "--output",
        default="extracted_frames",
        help="Output directory."
    )

    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Re-extract even when a complete extraction exists."
    )

    args = parser.parse_args()

    if args.video:

        video_path = Path(args.video)

        if args.output == "extracted_frames":
            output_dir = (
                Path("extracted_frames")
                / video_path.stem
            )

        else:
            output_dir = Path(args.output)

        extract_frames(
            video_path=video_path,
            output_dir=output_dir,
            overwrite=args.overwrite,
        )

    else:

        batch_extract(
            input_dir=args.input,
            output_dir=args.output,
            overwrite=args.overwrite,
        )