import cv2
import mediapipe as mp
import numpy as np

from preprocessing.face_detection import FaceDetector
from preprocessing.roi import ROIExtractor


# =====================================================
# CONFIGURATION
# =====================================================

VIDEO_PATH = (
    "dataset/train/fake/"
    "Screen Recording 2026-01-23 100053.mp4"
)

# Padding percentages to test
PADDING_VALUES = [0.0, 0.20, 0.30, 0.40]

# Face crop resolutions to test
TARGET_SIZES = [128, 256]

# Required valid ROI regions
REQUIRED_ROIS = [
    "forehead",
    "left_cheek",
    "right_cheek"
]


# =====================================================
# INITIALIZE FACE DETECTOR
# =====================================================

face_detector = FaceDetector(
    frame_scale=1.5
)


# =====================================================
# INITIALIZE FACE MESH
# =====================================================

face_mesh = mp.solutions.face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5
)


# =====================================================
# INITIALIZE ROI EXTRACTOR
# =====================================================

roi_extractor = ROIExtractor(
    max_num_faces=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    min_skin_pixels=20,
    min_polygon_area=4.0,
    debug=False
)


# =====================================================
# HELPER: EXPAND FACE BOUNDING BOX
# =====================================================

def get_padded_bbox(bbox, frame_width, frame_height, padding):

    x, y, w, h = bbox

    pad_x = int(w * padding)
    pad_y = int(h * padding)

    x1 = max(0, int(x - pad_x))
    y1 = max(0, int(y - pad_y))

    x2 = min(frame_width, int(x + w + pad_x))
    y2 = min(frame_height, int(y + h + pad_y))

    return x1, y1, x2, y2


# =====================================================
# HELPER: MAP LANDMARKS BACK TO ORIGINAL FRAME
# =====================================================

def map_landmarks_to_original(
    landmarks,
    crop_x,
    crop_y,
    crop_width,
    crop_height,
    original_width,
    original_height
):

    mapped_landmarks = []

    for landmark in landmarks:

        # Convert normalized crop coordinates
        # into original frame pixel coordinates.

        original_x = crop_x + landmark.x * crop_width
        original_y = crop_y + landmark.y * crop_height

        # Convert to normalized original frame coordinates.

        normalized_x = original_x / original_width
        normalized_y = original_y / original_height

        mapped_landmarks.append(
            (normalized_x, normalized_y)
        )

    return mapped_landmarks


# =====================================================
# INITIALIZE RESULTS
# =====================================================

results = {}

for padding in PADDING_VALUES:

    for size in TARGET_SIZES:

        key = (padding, size)

        results[key] = {

            "face_mesh_success": 0,

            "valid_forehead": 0,
            "valid_left_cheek": 0,
            "valid_right_cheek": 0,

            "all_rois_valid": 0,

            "no_landmarks": 0,
            "invalid_rois": 0
        }


total_frames = 0
face_detected_frames = 0


# =====================================================
# OPEN VIDEO
# =====================================================

cap = cv2.VideoCapture(VIDEO_PATH)

if not cap.isOpened():

    face_detector.close()
    face_mesh.close()
    roi_extractor.close()

    raise RuntimeError(
        f"Could not open video: {VIDEO_PATH}"
    )


# =====================================================
# PROCESS VIDEO
# =====================================================

try:

    while True:

        success, frame = cap.read()

        if not success:
            break

        total_frames += 1

        frame_height, frame_width = frame.shape[:2]

        # ---------------------------------------------
        # 1. DETECT FACE
        # ---------------------------------------------

        detections = face_detector.detect_frame(frame)

        if not detections:
            continue

        face_detected_frames += 1

        bbox = detections[0]["bbox"]

        # ---------------------------------------------
        # 2. TEST PADDING AND RESOLUTIONS
        # ---------------------------------------------

        for padding in PADDING_VALUES:

            x1, y1, x2, y2 = get_padded_bbox(
                bbox,
                frame_width,
                frame_height,
                padding
            )

            face_crop = frame[y1:y2, x1:x2]

            if face_crop.size == 0:
                continue

            crop_height, crop_width = face_crop.shape[:2]

            for size in TARGET_SIZES:

                key = (padding, size)

                # -------------------------------------
                # RESIZE FACE CROP
                # -------------------------------------

                resized_face = cv2.resize(
                    face_crop,
                    (size, size),
                    interpolation=cv2.INTER_CUBIC
                )

                # -------------------------------------
                # FACE MESH
                # -------------------------------------

                rgb_face = cv2.cvtColor(
                    resized_face,
                    cv2.COLOR_BGR2RGB
                )

                mesh_result = face_mesh.process(rgb_face)

                if not mesh_result.multi_face_landmarks:

                    results[key]["no_landmarks"] += 1
                    continue

                results[key]["face_mesh_success"] += 1

                # -------------------------------------
                # MAP LANDMARKS TO ORIGINAL FRAME
                # -------------------------------------

                landmarks = mesh_result.multi_face_landmarks[0].landmark

                mapped_landmarks = map_landmarks_to_original(
                    landmarks,
                    x1,
                    y1,
                    crop_width,
                    crop_height,
                    frame_width,
                    frame_height
                )

                # -------------------------------------
                # BUILD FULL-FRAME LANDMARK OBJECT
                # -------------------------------------

                # MediaPipe ROIExtractor expects a result
                # containing normalized landmarks.

                original_landmarks = []

                for nx, ny in mapped_landmarks:

                    landmark = type(
                        "Landmark",
                        (),
                        {
                            "x": nx,
                            "y": ny,
                            "z": 0.0
                        }
                    )()

                    original_landmarks.append(landmark)

                # -------------------------------------
                # NOTE:
                # The existing ROIExtractor runs its own
                # Face Mesh model internally.
                #
                # Therefore, we do NOT call extract_rois()
                # here because it would ignore the mapped
                # landmarks and run Face Mesh again.
                #
                # We only measure Face Mesh success in
                # this initial experiment.
                # -------------------------------------


finally:

    cap.release()

    face_detector.close()
    face_mesh.close()
    roi_extractor.close()


# =====================================================
# DISPLAY RESULTS
# =====================================================

print("\n==============================================")
print("       ROI EXPERIMENT SUMMARY")
print("==============================================")

print(f"Total frames: {total_frames}")

print(
    f"Face detected: {face_detected_frames}/"
    f"{total_frames}"
)

print("\nPADDING AND RESOLUTION COMPARISON")

for padding in PADDING_VALUES:

    for size in TARGET_SIZES:

        key = (padding, size)

        data = results[key]

        success_count = data["face_mesh_success"]

        percentage = (
            success_count / total_frames * 100
            if total_frames else 0
        )

        print("\n----------------------------------------------")

        print(f"Padding: {padding * 100:.0f}%")
        print(f"Resolution: {size} x {size}")

        print(
            f"Face Mesh successful: "
            f"{success_count}/{total_frames}"
        )

        print(
            f"Face Mesh detection rate: "
            f"{percentage:.2f}%"
        )

        print(
            f"No landmarks: "
            f"{data['no_landmarks']}"
        )

print("\n==============================================")
print("Experiment completed.")
print("==============================================")