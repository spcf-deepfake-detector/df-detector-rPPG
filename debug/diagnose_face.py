import cv2

from preprocessing.face_detection import FaceDetector

VIDEO_PATH = "dataset/train/fake/Screen Recording 2026-01-23 100053.mp4"

TARGET_FRAMES = [0, 50, 93, 94, 150, 225]

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

detector = FaceDetector()

frame_number = 0

try:
    while True:
        success, frame = cap.read()

        if not success:
            break

        if frame_number in TARGET_FRAMES:

            height, width = frame.shape[:2]

            faces = detector.detect_frame(frame)

            print(f"\n========== FRAME {frame_number} ==========")
            print(f"Frame dimensions: {width} x {height}")

            if not faces:
                print("No face detected.")
            else:
                for index, face in enumerate(faces):

                    x, y, w, h = face["bbox"]

                    face_percentage = (
                        (w * h) / (width * height)
                    ) * 100

                    print(f"Face {index + 1}")
                    print(f"BBox: {face['bbox']}")
                    print(f"Confidence: {face['confidence']:.4f}")
                    print(f"Face width: {w} pixels")
                    print(f"Face height: {h} pixels")
                    print(f"Face area: {face_percentage:.2f}%")

        frame_number += 1

finally:
    cap.release()
    detector.close()

print("\nDiagnostic completed.")