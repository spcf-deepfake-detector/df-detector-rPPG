import cv2
import mediapipe as mp

frame_paths = [
    "debug_roi_frames/frame_150.jpg",
    "debug_roi_frames/frame_230.jpg",
]

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=5,
    refine_landmarks=True,
    min_detection_confidence=0.3,
)

for path in frame_paths:
    frame = cv2.imread(path)

    if frame is None:
        print(f"\nCould not read: {path}")
        continue

    rgb_frame = cv2.cvtColor(
        frame,
        cv2.COLOR_BGR2RGB,
    )

    results = face_mesh.process(rgb_frame)

    face_count = (
        len(results.multi_face_landmarks)
        if results.multi_face_landmarks
        else 0
    )

    print(f"\nImage: {path}")
    print(f"Image dimensions: {frame.shape[1]}x{frame.shape[0]}")
    print(f"Faces detected by Face Mesh: {face_count}")

face_mesh.close()
