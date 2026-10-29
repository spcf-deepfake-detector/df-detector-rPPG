import cv2
import json
from pathlib import Path

VIDEO_PATH = (
    "dataset/train/fake/"
    "Screen Recording 2026-01-23 100053.mp4"
)

OUTPUT_DIR = Path("debug_frames")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Frames near the beginning, middle, and end.
FRAME_INDICES = [0, 10, 50, 100, 150, 200, 224, 225, 230, 231]

# If your test script saves tracking records to a JSON file,
# set its path here. Otherwise, leave this as None.
TRACKING_JSON = None

tracking_records = {}

if TRACKING_JSON:
    with open(TRACKING_JSON, "r", encoding="utf-8") as file:
        tracking_records = json.load(file)

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    raise RuntimeError(f"Could not open video: {VIDEO_PATH}")

frame_index = 0
saved = 0

while True:
    success, frame = cap.read()

    if not success:
        break

    if frame_index in FRAME_INDICES:
        display = frame.copy()

        # Draw the detector/tracker bounding box if records
        # are available in the JSON file.
        record = tracking_records.get(str(frame_index))

        if record:
            bbox = record.get("bbox")

            if bbox and len(bbox) == 4:
                x, y, w, h = map(int, bbox)

                cv2.rectangle(
                    display,
                    (x, y),
                    (x + w, y + h),
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

        output_path = OUTPUT_DIR / f"frame_{frame_index}.jpg"

        if not cv2.imwrite(str(output_path), display):
            raise RuntimeError(
                f"Could not save {output_path}"
            )

        print(f"Saved: {output_path}")
        saved += 1

    frame_index += 1

cap.release()

print(f"\nFinished. Saved {saved} frames to {OUTPUT_DIR}")