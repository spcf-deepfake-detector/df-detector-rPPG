
from pathlib import Path

from preprocessing.face_detection import FaceDetector
from preprocessing.rgb_extraction import RGBExtractor


# --------------------------------------------------
# 1. Select exactly one video
# --------------------------------------------------

VIDEO_PATH = Path(
    "dataset/train/fake/Screen Recording 2026-01-23 100053.mp4"
)

if not VIDEO_PATH.is_file():
    raise FileNotFoundError(
        f"Video not found: {VIDEO_PATH.resolve()}"
    )


def main():
    detector = None
    rgb_extractor = None

    try:
        # ------------------------------------------
        # 2. Initialize face detection
        # ------------------------------------------

        detector = FaceDetector(
            min_detection_confidence=0.3,
            model_selection=1,
            frame_scale=1.0,
        )

        # ------------------------------------------
        # 3. Detect faces in this video
        # ------------------------------------------

        detection_result = detector.process_video(
            str(VIDEO_PATH),
            multi_face=False,
            verbose=True,
        )

        detections = detection_result["detections"]

        print("\nDetection records:", len(detections))

        # ------------------------------------------
        # 4. Initialize RGB extraction
        # ------------------------------------------

        rgb_extractor = RGBExtractor(
            min_roi_width=5,
            min_roi_height=5,
            min_skin_pixels=1,
            debug=True,
        )

        # ------------------------------------------
        # 5. Extract RGB measurements
        # ------------------------------------------

        rgb_signal = rgb_extractor.process_video_frames(
            str(VIDEO_PATH),
            detections,
        )

        # ------------------------------------------
        # 6. Summarize results
        # ------------------------------------------

        valid_frames = (
            rgb_signal.shape[0] > 0
            and __import__("numpy").isfinite(rgb_signal).all(axis=1).sum()
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
