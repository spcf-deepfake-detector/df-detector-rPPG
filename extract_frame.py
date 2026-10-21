import cv2
import argparse
from pathlib import Path


def extract_frames(video_path, output_dir):
    video_path = Path(video_path)
    output_dir = Path(output_dir)

    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")

    frame_count = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            frame_path = output_dir / f"frame_{frame_count:05d}.jpg"

            if not cv2.imwrite(str(frame_path), frame):
                raise OSError(f"Failed to save frame: {frame_path}")

            frame_count += 1
    finally:
        cap.release()

    if frame_count == 0:
        raise ValueError(f"No frames extracted from: {video_path}")

    print(f"\nVideo: {video_path}")
    print(f"Frames extracted: {frame_count}")
    print(f"Saved to: {output_dir.resolve()}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Extract video frames into JPG images"
    )

    parser.add_argument("--video", required=True)
    parser.add_argument("--output", default="processing_result")

    args = parser.parse_args()

    extract_frames(args.video, args.output)