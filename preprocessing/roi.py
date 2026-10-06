import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, Any, List, Tuple

class ROIExtractor:
    """Extracts skin ROIs (Forehead, Left Cheek, Right Cheek) using MediaPipe Face Mesh."""
    # MediaPipe Face Mesh landmark indices
    FOREHEAD_INDICES = [10, 67, 109, 151, 337, 297, 9]
    LEFT_CHEEK_INDICES = [117, 118, 100, 126, 209, 198]
    RIGHT_CHEEK_INDICES = [346, 347, 329, 355, 429, 419]

    def __init__(self):
        self.mp_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )

    def extract_rois(self, frame: np.ndarray, bbox: List[int]) -> Dict[str, np.ndarray]:
        h, w, _ = frame.shape
        x, y, bw, bh = bbox

        # Fallback bounding-box based ROI if bbox is missing or invalid
        if bw <= 0 or bh <= 0:
            return {
                "forehead": np.zeros((10, 10, 3), dtype=np.uint8),
                "left_cheek": np.zeros((10, 10, 3), dtype=np.uint8),
                "right_cheek": np.zeros((10, 10, 3), dtype=np.uint8)
            }

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            # Bounding box bounding fallbacks
            fh_crop = frame[max(0, y):max(0, y + int(0.25 * bh)), max(0, x + int(0.2 * bw)):max(0, x + int(0.8 * bw))]
            lc_crop = frame[max(0, y + int(0.5 * bh)):max(0, y + int(0.8 * bh)), max(0, x):max(0, x + int(0.35 * bw))]
            rc_crop = frame[max(0, y + int(0.5 * bh)):max(0, y + int(0.8 * bh)), max(0, x + int(0.65 * bw)):max(0, x + bw)]
            return {
                "forehead": fh_crop if fh_crop.size > 0 else np.zeros((10, 10, 3), dtype=np.uint8),
                "left_cheek": lc_crop if lc_crop.size > 0 else np.zeros((10, 10, 3), dtype=np.uint8),
                "right_cheek": rc_crop if rc_crop.size > 0 else np.zeros((10, 10, 3), dtype=np.uint8)
            }

        landmarks = results.multi_face_landmarks[0].landmark
        
        def get_mask_crop(indices: List[int]) -> np.ndarray:
            pts = np.array([[int(landmarks[idx].x * w), int(landmarks[idx].y * h)] for idx in indices], dtype=np.int32)
            rect = cv2.boundingRect(pts)
            rx, ry, rw, rh = rect
            crop = frame[max(0, ry):max(0, ry + rh), max(0, rx):max(0, rx + rw)]
            return crop if crop.size > 0 else np.zeros((10, 10, 3), dtype=np.uint8)

        return {
            "forehead": get_mask_crop(self.FOREHEAD_INDICES),
            "left_cheek": get_mask_crop(self.LEFT_CHEEK_INDICES),
            "right_cheek": get_mask_crop(self.RIGHT_CHEEK_INDICES)
        }

    def close(self):
        self.face_mesh.close()