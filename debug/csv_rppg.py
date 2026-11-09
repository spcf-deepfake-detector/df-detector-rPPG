import cv2
import csv

from preprocessing.face_detection import FaceDetector

VIDEO = "dataset/test/fake/Screen Recording 2026-01-23 100015.mp4"

CONFIGURATIONS = [
    {"name": "A_model0_scale1.0", "model": 0, "scale": 1.0},
    {"name": "B_model1_scale1.0", "model": 1, "scale": 1.0},
    {"name": "C_model0_scale1.5", "model": 0, "scale": 1.5},
    {"name": "D_model1_scale1.5", "model": 1, "scale": 1.5},
]

results = []

for config in CONFIGURATIONS:

    print("\n======================================")
    print("Testing:", config["name"])
    print("Model:", config["model"])
    print("Frame scale:", config["scale"])
    print("======================================")

    cap = cv2.VideoCapture(VIDEO)

    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {VIDEO}")

    detector = FaceDetector(
        min_detection_confidence=0.3,
        model_selection=config["model"],
        frame_scale=config["scale"],
        use_dual_model=False,
    )

    frame_index = 0
    detected_total = 0
    detected_interval = 0
    interval_total = 0

    try:
        while True:

            success, frame = cap.read()

            if not success:
                break

            detections = detector.detect_frame(frame)

            detected = bool(detections)
            confidence = 0.0
            bbox = None

            if detected:
                detected_total += 1

                best = max(
                    detections,
                    key=lambda item: item["confidence"],
                )

                confidence = float(best["confidence"])
                bbox = best["bbox"]

            if 58 <= frame_index <= 230:
                interval_total += 1

                if detected:
                    detected_interval += 1

            results.append({
                "configuration": config["name"],
                "frame_index": frame_index,
                "detected": detected,
                "confidence": confidence,
                "bbox": bbox,
            })

            frame_index += 1

    finally:
        cap.release()
        detector.close()

    overall_rate = (
        detected_total / frame_index * 100
        if frame_index else 0
    )

    interval_rate = (
        detected_interval / interval_total * 100
        if interval_total else 0
    )

    print(f"Frames processed: {frame_index}")
    print(f"Detected: {detected_total}")
    print(f"Overall detection rate: {overall_rate:.2f}%")
    print(f"Frames 58-230 detected: {detected_interval}/{interval_total}")
    print(f"Interval detection rate: {interval_rate:.2f}%")

# Save detailed results.
csv_path = "detection_comparison_100015.csv"

with open(csv_path, "w", newline="", encoding="utf-8") as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "configuration",
            "frame_index",
            "detected",
            "confidence",
            "bbox",
        ],
    )

    writer.writeheader()
    writer.writerows(results)

print("\n======================================")
print("ALL TESTS COMPLETED")
print("CSV saved:", csv_path)
print("======================================")
