import cv2
import mediapipe as mp

from preprocessing.face_detection import FaceDetector


# ==========================================
# CONFIGURATION
# ==========================================

VIDEO_PATH = "dataset/train/fake/Screen Recording 2026-01-23 100053.mp4"

TARGET_SIZES = [128, 256, 384]


# ==========================================
# INITIALIZE MODELS
# ==========================================

face_detector = FaceDetector()

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)


# ==========================================
# OPEN VIDEO
# ==========================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():
    face_detector.close()
    face_mesh.close()
    raise RuntimeError("Could not open video.")


# ==========================================
# INITIALIZE COUNTERS
# ==========================================

total_frames = 0

original_success = 0

resize_success = {
    size: 0
    for size in TARGET_SIZES
}


# ==========================================
# PROCESS VIDEO
# ==========================================

try:

    while True:

        success, frame = cap.read()

        if not success:
            break

        total_frames += 1

        height, width = frame.shape[:2]

        # --------------------------------------
        # 1. DETECT FACE
        # --------------------------------------

        detections = face_detector.detect_frame(frame)

        if not detections:
            continue

        x, y, w, h = detections[0]["bbox"]

        x = int(x)
        y = int(y)
        w = int(w)
        h = int(h)

        # --------------------------------------
        # 2. TEST ORIGINAL FRAME
        # --------------------------------------

        original_rgb = cv2.cvtColor(
            frame,
            cv2.COLOR_BGR2RGB
        )

        original_result = face_mesh.process(original_rgb)

        if original_result.multi_face_landmarks:
            original_success += 1

        # --------------------------------------
        # 3. CROP DETECTED FACE
        # --------------------------------------

        x1 = max(0, x)
        y1 = max(0, y)

        x2 = min(width, x + w)
        y2 = min(height, y + h)

        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            continue

        # --------------------------------------
        # 4. TEST DIFFERENT RESOLUTIONS
        # --------------------------------------

        for size in TARGET_SIZES:

            resized_face = cv2.resize(
                face_crop,
                (size, size),
                interpolation=cv2.INTER_CUBIC
            )

            resized_rgb = cv2.cvtColor(
                resized_face,
                cv2.COLOR_BGR2RGB
            )

            resized_result = face_mesh.process(resized_rgb)

            if resized_result.multi_face_landmarks:
                resize_success[size] += 1


# ==========================================
# DISPLAY RESULTS
# ==========================================

finally:

    cap.release()
    face_detector.close()
    face_mesh.close()


print("\n========== FACE MESH COMPARISON ==========")

print(f"Total frames: {total_frames}")

print("\nOriginal frame:")

original_percentage = (
    original_success / total_frames * 100
    if total_frames else 0
)

print(f"Successful: {original_success}")
print(f"Failed: {total_frames - original_success}")
print(f"Detection rate: {original_percentage:.2f}%")


print("\nResized face crops:")

for size in TARGET_SIZES:

    successful = resize_success[size]

    percentage = (
        successful / total_frames * 100
        if total_frames else 0
    )

    print(f"\nResolution: {size} x {size}")
    print(f"Successful: {successful}")
    print(f"Failed: {total_frames - successful}")
    print(f"Detection rate: {percentage:.2f}%")


print("\n==========================================")
print("Diagnostic completed.")