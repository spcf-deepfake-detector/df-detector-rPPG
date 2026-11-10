
import cv2
import os

video_path = (
    "dataset/train/fake/"
    "Screen Recording 2026-01-23 100053.mp4"
)

output_dir = "debug_roi_frames"
os.makedirs(output_dir, exist_ok=True)

cap = cv2.VideoCapture(video_path)

if not cap.isOpened():
    raise RuntimeError(f"Could not open: {video_path}")

# One frame where Face Mesh succeeded and one where it failed.
frame_indices = [150, 230]

for frame_idx in frame_indices:
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)

    success, frame = cap.read()

    if not success:
        print(f"Could not read frame {frame_idx}")
        continue

    output_path = os.path.join(
        output_dir,
        f"frame_{frame_idx}.jpg",
    )

    cv2.imwrite(output_path, frame)
    print(f"Saved: {output_path}")

cap.release()
