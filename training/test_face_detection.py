from preprocessing.face_detection import FaceDetector

video_path = "dataset/train/real/16.mp4"

configs = [
    {"confidence": 0.5, "scale": 1.0},
    {"confidence": 0.3, "scale": 1.0},
    {"confidence": 0.5, "scale": 1.5},
]

for config in configs:
    print("\nTesting configuration:", config)

    detector = FaceDetector(
        min_detection_confidence=config["confidence"],
        model_selection=0,
        frame_scale=config["scale"],
    )

    try:
        result = detector.process_video(video_path)

        print(
            f"FINAL DETECTION RATE: "
            f"{result['detection_rate']:.2f}%"
        )
    finally:
        detector.close()