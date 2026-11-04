import cv2
import os

video_path = "dataset/train/fake/Screen Recording 2026-01-23 092550.mp4"
output_dir = "debug/092550_frames"

os.makedirs(output_dir, exist_ok=True)

frame_numbers = [
    329, 370, 412,
    418, 439, 460,
    495, 509, 523
]

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError("Could not open video.")

for frame_number in frame_numbers:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_number)

    success, frame = cap.read()

    if not success:
        print(f"Could not read frame {frame_number}")
        continue

    cv2.putText(
        frame,
        f"Frame: {frame_number}",
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 0),
        2
    )

    output_path = os.path.join(
        output_dir,
        f"frame_{frame_number}.jpg"
    )

    cv2.imwrite(output_path, frame)
    print(f"Saved: {output_path}")

cap.release()
print("Finished extracting frames.")