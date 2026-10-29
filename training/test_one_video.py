
from pathlib import Path

import cv2
import numpy as np

from preprocessing.face_detection import FaceDetector
from preprocessing.rgb_extraction import RGBExtractor


# --------------------------------------------------
# 1. Select exactly one video
# --------------------------------------------------

VIDEO_PATH = Path(
    "dataset/train/fake/Screen Recording 2026-01-23 100053.mp4"
)

OUTPUT_DIR = Path("debug_frames")

# Frames near the beginning, middle, and end.
FRAME_INDICES = [0, 10, 50, 100, 150, 200, 221, 224, 225, 230, 231]


if not VIDEO_PATH.is_file():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH.resolve()}"
    )


# --------------------------------------------------
# 2. Save selected frames with detection bounding boxes
# --------------------------------------------------

def save_debug_frames(video_path, detections):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cap = cv2.VideoCapture(str(video_path))

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    saved_frames = 0
    frame_index = 0

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            if frame_index in FRAME_INDICES:
                display = frame.copy()

                # Match this video frame to its detection record.
                if frame_index < len(detections):
                    record = detections[frame_index]

                    if isinstance(record, dict):
                        bbox = record.get("bbox")

                        if bbox is not None and len(bbox) == 4:
                            x, y, w, h = map(int, bbox)

                            if w > 0 and h > 0:
                                cv2.rectangle(
                                    display,
                                    (x, y),
                                    (x + w, y + h),
                                    (0, 255, 0),
                                    2,
                                )

                                cv2.putText(
                                    display,
                                    f"Face: {w}x{h}",
                                    (10, 60),
                                    cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7,
                                    (0, 255, 0),
                                    2,
                                )

                cv2.putText(
                    display,
                    f"Frame {frame_index}",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 255),
                    2,
                )

                output_path = (
                    OUTPUT_DIR / f"frame_{frame_index}.jpg"
                )

                if not cv2.imwrite(str(output_path), display):
                    raise RuntimeError(
                        f"Failed to save {output_path}"
                    )

                print(f"Saved: {output_path}")
                saved_frames += 1

            frame_index += 1

    finally:
        cap.release()

    print(
        f"\nSaved {saved_frames} diagnostic frames "
        f"to: {OUTPUT_DIR.resolve()}"
    )


# --------------------------------------------------
# 3. Main test
# --------------------------------------------------

def main():
    detector = None
    rgb_extractor = None

    try:
        # ------------------------------------------
        # Initialize face detection
        # ------------------------------------------

        detector = FaceDetector(
            min_detection_confidence=0.3,
            model_selection=1,
            frame_scale=1.0,
        )

        # ------------------------------------------
        # Detect faces
        # ------------------------------------------

        detection_result = detector.process_video(
            str(VIDEO_PATH),
            multi_face=False,
            verbose=True,
        )

        detections = detection_result["detections"]

        print("\nDetection records:", len(detections))

        # ------------------------------------------
        # Save diagnostic frames
        # ------------------------------------------

        save_debug_frames(
            VIDEO_PATH,
            detections,
        )

        # ------------------------------------------
        # Initialize RGB extraction
        # ------------------------------------------

        rgb_extractor = RGBExtractor(
            min_roi_width=5,
            min_roi_height=5,
            min_skin_pixels=1,
            debug=True,
        )

        # ------------------------------------------
        # Extract RGB measurements
        # ------------------------------------------

        rgb_signal = rgb_extractor.process_video_frames(
            str(VIDEO_PATH),
            detections,
        )

        # ------------------------------------------
        # Summarize results
        # ------------------------------------------

        valid_frames = int(
            np.isfinite(rgb_signal).all(axis=1).sum()
        )

        total_frames = len(rgb_signal)

        print("\n========== TEST RESULT ==========")
        print(f"Video: {VIDEO_PATH.name}")
        print(f"Detection records: {len(detections)}")
        print(f"RGB records: {total_frames}")
        print(f"Valid RGB frames: {valid_frames}")

        if total_frames:
            print(
                f"RGB validity: "
                f"{valid_frames / total_frames * 100:.2f}%"
            )

        print("================================")

    finally:
        if rgb_extractor is not None:
            rgb_extractor.close()

        if detector is not None:
            detector.close()


if __name__ == "__main__":
    main()
