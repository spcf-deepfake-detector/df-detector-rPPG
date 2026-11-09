import cv2

from preprocessing.face_detection import FaceDetector


video_path = "dataset/test/fake/Screen Recording 2026-01-23 100015.mp4"

detector = FaceDetector(
    min_detection_confidence=0.3,
    model_selection=0,
    frame_scale=1.0,
    multi_scale_factors=(1.0, 1.25, 1.5),
)

try:

    result = detector.process_video(video_path)

    print("\n========== DETECTION RESULTS ==========")

    print("Total frames:", result["total_frames"])

    print("Statistics:")

    for key, value in result["stats"].items():
        print(f"{key}: {value}")

finally:

    detector.close()