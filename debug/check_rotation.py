
import cv2

video_path = (
    "dataset/train/fake/"
    "Screen Recording 2026-01-23 100053.mp4"
)


def inspect_video(auto_rotate):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise RuntimeError(
            f"Could not open video: {video_path}"
        )

    cap.set(
        cv2.CAP_PROP_ORIENTATION_AUTO,
        1 if auto_rotate else 0,
    )

    rotation = cap.get(
        cv2.CAP_PROP_ORIENTATION_META
    )

    success, frame = cap.read()

    print(f"\nAutomatic rotation: {auto_rotate}")
    print(f"Rotation metadata: {rotation}")

    if success:
        print(f"Decoded frame shape: {frame.shape}")
    else:
        print("Could not decode the first frame.")

    cap.release()


inspect_video(True)
inspect_video(False)
