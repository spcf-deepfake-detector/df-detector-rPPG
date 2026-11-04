import os
import time

from preprocessing.face_detection import FaceDetector


VIDEO = (
    "dataset/train/fake/"
    "Screen Recording 2026-01-23 103830.mp4"
)

CONFIGS = [
    {
        "name": "Current",
        "confidence": 0.30,
        "model": 1,
        "scale": 1.0,
    },
    {
        "name": "Upscaling",
        "confidence": 0.30,
        "model": 1,
        "scale": 2.0,
    },
    {
        "name": "Short-range",
        "confidence": 0.30,
        "model": 0,
        "scale": 1.0,
    },
]


if not os.path.isfile(VIDEO):
    raise FileNotFoundError(VIDEO)


for config in CONFIGS:

    print("\n" + "=" * 50)
    print(f"TEST: {config['name']}")
    print("=" * 50)

    detector = FaceDetector(
        min_detection_confidence=config["confidence"],
        model_selection=config["model"],
        frame_scale=config["scale"],
    )

    start = time.perf_counter()

    try:
        result = detector.process_video(
            VIDEO,
            verbose=True,
        )
    finally:
        detector.close()

    elapsed = time.perf_counter() - start

    print(f"Processing time: {elapsed:.2f} seconds")
    print(f"Detection rate: {result['detection_rate']:.2f}%")