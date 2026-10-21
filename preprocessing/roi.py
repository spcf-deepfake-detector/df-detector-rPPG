import cv2
import numpy as np
import mediapipe as mp
from typing import Dict, List, Tuple


class ROIExtractor:
    """Extract forehead and cheek regions using MediaPipe Face Mesh."""

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

        empty_roi = (
            np.empty((0, 0, 3), dtype=np.uint8),
            np.empty((0, 0), dtype=np.uint8)
        )

        empty_results = {
            "forehead": empty_roi,
            "left_cheek": empty_roi,
            "right_cheek": empty_roi
        }

        # 1. Validate the frame.
        if (
            frame is None
            or not isinstance(frame, np.ndarray)
            or frame.ndim != 3
            or frame.shape[2] != 3
        ):
            return empty_results

        h, w = frame.shape[:2]

        # 2. Validate the bounding box.
        if (
            bbox is None
            or len(bbox) != 4
        ):
            return empty_results

        try:
            x, y, bw, bh = map(int, bbox)
        except (TypeError, ValueError, OverflowError):
            return empty_results

        if bw <= 0 or bh <= 0:
            return empty_results

        # 3. Add padding around the detected face.
        pad_x = int(bw * 0.20)
        pad_y = int(bh * 0.20)

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)
        x2 = min(w, x + bw + pad_x)
        y2 = min(h, y + bh + pad_y)

        if x2 <= x1 or y2 <= y1:
            return empty_results

        # 4. Crop the face region.
        face_crop = frame[y1:y2, x1:x2]

        if face_crop.size == 0:
            return empty_results

        # 5. Convert the crop from BGR to RGB.
        rgb_crop = cv2.cvtColor(
            face_crop,
            cv2.COLOR_BGR2RGB
        )

        # 6. Detect facial landmarks within the crop.
        results = self.face_mesh.process(rgb_crop)

        if not results.multi_face_landmarks:
            return empty_results

        landmarks = results.multi_face_landmarks[0].landmark

        def get_mask_crop(
            indices: List[int]
        ) -> Tuple[np.ndarray, np.ndarray]:

            # 7. Convert crop-relative landmarks to original
            # frame coordinates.
            pts = np.array(
                [
                    [
                        int(landmarks[idx].x * (x2 - x1)) + x1,
                        int(landmarks[idx].y * (y2 - y1)) + y1
                    ]
                    for idx in indices
                ],
                dtype=np.int32
            )

            # 8. Keep coordinates inside the original frame.
            pts[:, 0] = np.clip(pts[:, 0], 0, w - 1)
            pts[:, 1] = np.clip(pts[:, 1], 0, h - 1)

            # 9. Find the smallest rectangle around the polygon.
            rx, ry, rw, rh = cv2.boundingRect(pts)

            cx1 = max(0, rx)
            cy1 = max(0, ry)
            cx2 = min(w, rx + rw)
            cy2 = min(h, ry + rh)

            if cx2 <= cx1 or cy2 <= cy1:
                return empty_roi

            # 10. Extract the image region.
            crop = frame[cy1:cy2, cx1:cx2]

            if crop.size == 0:
                return empty_roi

            # 11. Build a mask for the facial polygon.
            mask = np.zeros(
                (cy2 - cy1, cx2 - cx1),
                dtype=np.uint8
            )

            local_pts = pts - np.array([cx1, cy1])

            cv2.fillPoly(
                mask,
                [local_pts],
                255
            )

            if not np.any(mask):
                return empty_roi

            return crop, mask

        # 12. Extract the forehead and cheeks.
        rois = {
            "forehead": get_mask_crop(self.FOREHEAD_INDICES),
            "left_cheek": get_mask_crop(self.LEFT_CHEEK_INDICES),
            "right_cheek": get_mask_crop(self.RIGHT_CHEEK_INDICES)
        }

        return rois

    def close(self):
        """Release MediaPipe resources."""
        self.face_mesh.close()