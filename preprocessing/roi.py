import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple


class ROIExtractor:
    """Extracts forehead and cheek regions using MediaPipe Face Mesh."""

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

    def extract_rois(
        self,
        frame: np.ndarray,
        bbox: List[int]
    ) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:

        # Prepare empty results for failed extractions.
        empty_roi = (
            np.empty((0, 0, 3), dtype=np.uint8),
            np.empty((0, 0), dtype=np.uint8)
        )

        empty_results = {
            "forehead": empty_roi,
            "left_cheek": empty_roi,
            "right_cheek": empty_roi
        }

        # Validate the input frame.
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            return empty_results

        h, w = frame.shape[:2]

        # Validate the bounding box.
        if bbox is None or len(bbox) != 4:
            return empty_results

        x, y, bw, bh = bbox

        if bw <= 0 or bh <= 0:
            return empty_results

        # Clip the bounding box to the image boundaries.
        x1 = max(0, min(w, x))
        y1 = max(0, min(h, y))
        x2 = max(0, min(w, x + bw))
        y2 = max(0, min(h, y + bh))

        if x2 <= x1 or y2 <= y1:
            return empty_results

        # MediaPipe expects RGB input.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)

        # Return empty regions if no face is detected.
        if not results.multi_face_landmarks:
            return empty_results

        landmarks = results.multi_face_landmarks[0].landmark

        def get_mask_crop(
            indices: List[int]
        ) -> Tuple[np.ndarray, np.ndarray]:

            # Convert normalized landmarks to pixel coordinates.
            pts = np.array(
                [
                    [
                        int(landmarks[idx].x * w),
                        int(landmarks[idx].y * h)
                    ]
                    for idx in indices
                ],
                dtype=np.int32
            )

            # Reject landmarks that are outside the frame.
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

            # Find a rectangle containing the polygon.
            rx, ry, rw, rh = cv2.boundingRect(pts)

            # Clip the rectangle to the frame.
            cx1 = max(0, rx)
            cy1 = max(0, ry)
            cx2 = min(w, rx + rw)
            cy2 = min(h, ry + rh)

            if cx2 <= cx1 or cy2 <= cy1:
                return empty_roi

            # Crop the frame.
            crop = frame[cy1:cy2, cx1:cx2]

            # Create a mask for the polygon.
            mask = np.zeros(
                (cy2 - cy1, cx2 - cx1),
                dtype=np.uint8
            )

            # Convert polygon points to crop-relative coordinates.
            local_pts = pts - np.array([cx1, cy1])

            # Fill the polygon.
            cv2.fillPoly(mask, [local_pts], 255)

            # Reject empty crops or masks.
            if crop.size == 0 or not np.any(mask):
                return empty_roi

            return crop, mask

        # Extract the three facial regions.
        return {
            "forehead": get_mask_crop(self.FOREHEAD_INDICES),
            "left_cheek": get_mask_crop(self.LEFT_CHEEK_INDICES),
            "right_cheek": get_mask_crop(self.RIGHT_CHEEK_INDICES)
        }

    def close(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()