import cv2
import numpy as np

from preprocessing.face_detection import FaceDetector
from preprocessing.face_tracking import FaceTracker
from preprocessing.rgb_extraction import RGBExtractor


# ==========================================================
# CONFIGURATION
# ==========================================================

VIDEO_PATH = (
    "dataset/train/fake/"
    "Screen Recording 2026-01-23 091908.mp4"
)


# ==========================================================
# MAIN TEST
# ==========================================================

def main():

    print("\n========== SINGLE-PASS RGB EXTRACTION TEST ==========\n")

    cap = cv2.VideoCapture(VIDEO_PATH)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO_PATH}")

    detector = None
    rgb_extractor = None

    rgb_signals = []
    frame_idx = 0

    try:

        # --------------------------------------------------
        # INITIALIZE COMPONENTS
        # --------------------------------------------------

        detector = FaceDetector(
            min_detection_confidence=0.30,
            model_selection=1,
            frame_scale=1.0,
            max_faces=None,
            use_dual_model=False
        )

        tracker = FaceTracker(
            iou_threshold=0.20,
            smoothing_alpha=0.65,
            fast_motion_threshold=0.15,
            max_center_distance=1.5,
            max_size_ratio=2.5,
            max_missed_frames=5,
            max_prediction_frames=5,
            max_target_lost_frames=30,
            debug=False
        )

        rgb_extractor = RGBExtractor(
            min_roi_width=5,
            min_roi_height=5,
            min_skin_pixels=1,
            debug=False
        )

        print("[INFO] Starting single-pass processing...\n")

        # --------------------------------------------------
        # ONE VIDEO DECODING LOOP
        # --------------------------------------------------

        while True:

            ret, frame = cap.read()

            if not ret:
                break

            # 1. Detect faces
            face_detections = detector.detect_frame(frame)

            # 2. Convert detections into a dictionary
            detection = {
                "frame_idx": frame_idx,
                "faces": face_detections,
                "candidate_count": len(face_detections),
                "detected": len(face_detections) > 0
            }

            # Select the largest face
            if face_detections:

                selected_face = max(
                    face_detections,
                    key=lambda face: (
                        face["bbox"][2] * face["bbox"][3]
                    )
                )

                detection["bbox"] = selected_face["bbox"]
                detection["confidence"] = selected_face["confidence"]

            else:

                detection["bbox"] = [0, 0, 0, 0]
                detection["confidence"] = 0.0

            # 3. Track the face
            tracking = tracker.track_frame(detection)

            # 4. Extract RGB from the SAME frame
            rgb = rgb_extractor.process_frame(
                frame,
                tracking
            )

            # Append exactly ONE measurement per frame
            rgb_signals.append(rgb)

            # Debug first five frames
            if frame_idx < 5:

                print(f"\nFrame: {frame_idx}")
                print("Detection count:", len(face_detections))
                print("Tracking status:", tracking.get("status"))
                print("Tracking valid:", tracking.get("track_valid"))
                print("Tracking bbox:", tracking.get("bbox"))
                print("RGB:", rgb)

            frame_idx += 1

            if frame_idx % 50 == 0:
                print(f"[PROGRESS] Processed {frame_idx} frames")

        # --------------------------------------------------
        # RESULTS
        # --------------------------------------------------

        rgb_signals = np.asarray(
            rgb_signals,
            dtype=np.float32
        )

        valid_mask = np.isfinite(rgb_signals).all(axis=1)

        valid_count = int(valid_mask.sum())
        invalid_count = len(rgb_signals) - valid_count

        print("\n========== RGB EXTRACTION RESULTS ==========")

        print(f"Total frames: {frame_idx}")
        print(f"RGB shape: {rgb_signals.shape}")
        print(f"Valid RGB frames: {valid_count}")
        print(f"Invalid RGB frames: {invalid_count}")

        if frame_idx > 0:
            print(
                f"RGB valid rate: "
                f"{valid_count / frame_idx * 100:.2f}%"
            )

        print("\nFirst 5 RGB measurements:")
        print(rgb_signals[:5])

        print("\n========== TEST COMPLETED ==========\n")

    finally:

        cap.release()

        if detector is not None:
            detector.close()

        if rgb_extractor is not None:
            rgb_extractor.close()


if __name__ == "__main__":
    main()