import cv2
import mediapipe as mp

image_path = "debug_roi_frames/frame_230.jpg"

frame = cv2.imread(image_path)

if frame is None:
    raise RuntimeError(f"Could not read image: {image_path}")

# Bounding box reported by your debugging output.
x, y, w, h = 259, 187, 66, 66

# Add padding around the detected face.
padding = int(max(w, h) * 0.30)

height, width = frame.shape[:2]

x1 = max(0, x - padding)
y1 = max(0, y - padding)
x2 = min(width, x + w + padding)
y2 = min(height, y + h + padding)

face_crop = frame[y1:y2, x1:x2]

if face_crop.size == 0:
    raise RuntimeError("The face crop is empty.")

# Test the original crop and an enlarged version.
enlarged_crop = cv2.resize(
    face_crop,
    None,
    fx=3,
    fy=3,
    interpolation=cv2.INTER_CUBIC,
)

cv2.imwrite("debug_roi_frames/face_crop.jpg", face_crop)
cv2.imwrite("debug_roi_frames/face_crop_enlarged.jpg", enlarged_crop)

mp_face_mesh = mp.solutions.face_mesh

with mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.3,
) as face_mesh:

    for name, image in [
        ("Original frame", frame),
        ("Face crop", face_crop),
        ("Enlarged face crop", enlarged_crop),
    ]:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        count = (
            len(results.multi_face_landmarks)
            if results.multi_face_landmarks
            else 0
        )

        print(f"{name}: {count} face(s) detected")

print("\nSaved face crops in debug_roi_frames/")
