import cv2
import numpy as np
import mediapipe as mp

from typing import Dict, List, Optional, Tuple, Any


class ROIExtractor:
    """
    Extract forehead and cheek regions using MediaPipe Face Mesh.

    The input frame is a full BGR video frame.
    The bounding box format is [x, y, width, height].

    Each ROI is returned as a dictionary:
        {
            "crop": BGR image crop or None,
            "mask": single-channel mask or None
        }
    """

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
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def _empty_roi() -> Dict[str, Optional[np.ndarray]]:
        """Return one consistently shaped empty ROI record."""
        return {"crop": None, "mask": None}

    def _empty_results(self) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
        """Return all expected ROI keys, even when extraction fails."""
        return {
            "forehead": self._empty_roi(),
            "left_cheek": self._empty_roi(),
            "right_cheek": self._empty_roi(),
        }

    def extract_rois(
        self,
        frame: np.ndarray,
        bbox: List[int],
    ) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
        """
        Detect face landmarks and extract masked forehead and cheek crops.

        Args:
            frame: Full BGR video frame (height, width, 3).
            bbox: Face box [x, y, width, height].

        Returns:
            Dictionary with forehead, left_cheek, right_cheek.
            Each value contains "crop" and "mask". Invalid regions
            return {"crop": None, "mask": None}.
        """
        empty_results = self._empty_results()

        # 1. Validate the frame before reading its dimensions.
        if not isinstance(frame, np.ndarray):
            return empty_results
        if frame.size == 0 or frame.ndim != 3 or frame.shape[2] != 3:
            return empty_results

        frame_h, frame_w = frame.shape[:2]

        # 2. Validate and normalize the bounding box.
        if bbox is None or len(bbox) != 4:
            return empty_results

        try:
            values = np.asarray(bbox, dtype=np.float64)
        except (TypeError, ValueError):
            return empty_results

        if not np.isfinite(values).all():
            return empty_results

        x, y, box_w, box_h = np.round(values).astype(int).tolist()

        if box_w <= 0 or box_h <= 0:
            return empty_results

        # Clip the box to the frame and reject boxes outside the image.
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_w, x + box_w)
        y2 = min(frame_h, y + box_h)

        if x2 <= x1 or y2 <= y1:
            return empty_results

        # 3. MediaPipe Face Mesh expects RGB, while OpenCV frames are BGR.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 4. Process the frame exactly once.
        results = self.face_mesh.process(rgb_frame)
        if not results.multi_face_landmarks:
            return empty_results

        landmarks = results.multi_face_landmarks[0].landmark

        # 5. Confirm the landmarks are near the expected face box.
        pad_x = int(box_w * 0.30)
        pad_y = int(box_h * 0.30)

        box_x1 = max(0, x - pad_x)
        box_y1 = max(0, y - pad_y)
        box_x2 = min(frame_w, x + box_w + pad_x)
        box_y2 = min(frame_h, y + box_h + pad_y)

        landmark_x = np.asarray(
            [point.x for point in landmarks], dtype=np.float64
        )
        landmark_y = np.asarray(
            [point.y for point in landmarks], dtype=np.float64
        )

        if not np.isfinite(landmark_x).all() or not np.isfinite(landmark_y).all():
            return empty_results

        face_center_x = float(np.mean(landmark_x) * frame_w)
        face_center_y = float(np.mean(landmark_y) * frame_h)

        if not (
            box_x1 <= face_center_x < box_x2
            and box_y1 <= face_center_y < box_y2
        ):
            return empty_results

        def get_mask_crop(
            indices: List[int],
        ) -> Dict[str, Optional[np.ndarray]]:
            """Return a BGR crop and its polygon mask, or an empty ROI."""
            if (
                not indices
                or min(indices) < 0
                or max(indices) >= len(landmarks)
            ):
                return self._empty_roi()

            points = []
            for idx in indices:
                landmark = landmarks[idx]

                if not (
                    np.isfinite(landmark.x)
                    and np.isfinite(landmark.y)
                ):
                    return self._empty_roi()

                px = int(round(landmark.x * frame_w))
                py = int(round(landmark.y * frame_h))
                points.append([px, py])

            points = np.asarray(points, dtype=np.int32)

            # 6. Clip polygon points to image boundaries.
            points[:, 0] = np.clip(points[:, 0], 0, frame_w - 1)
            points[:, 1] = np.clip(points[:, 1], 0, frame_h - 1)

            # 7. Reject polygons that cannot define a useful region.
            if len(np.unique(points, axis=0)) < 3:
                return self._empty_roi()

            if abs(cv2.contourArea(points)) < 4:
                return self._empty_roi()

            # 8. Crop the smallest rectangle around the polygon.
            rx, ry, rw, rh = cv2.boundingRect(points)
            cx1 = max(0, rx)
            cy1 = max(0, ry)
            cx2 = min(frame_w, rx + rw)
            cy2 = min(frame_h, ry + rh)

            if cx2 <= cx1 or cy2 <= cy1:
                return self._empty_roi()

            crop = frame[cy1:cy2, cx1:cx2]
            if crop.size == 0 or crop.ndim != 3 or crop.shape[2] != 3:
                return self._empty_roi()

            # 9. Draw the polygon in crop-relative coordinates.
            mask = np.zeros(crop.shape[:2], dtype=np.uint8)
            local_points = points - np.array([cx1, cy1], dtype=np.int32)
            cv2.fillPoly(mask, [local_points], 255)

            if mask.shape != crop.shape[:2] or cv2.countNonZero(mask) == 0:
                return self._empty_roi()

            return {"crop": crop, "mask": mask}

        # 10. Extract the three regions.
        return {
            "forehead": get_mask_crop(self.FOREHEAD_INDICES),
            "left_cheek": get_mask_crop(self.LEFT_CHEEK_INDICES),
            "right_cheek": get_mask_crop(self.RIGHT_CHEEK_INDICES),
        }

    def close(self):
        """Release MediaPipe Face Mesh resources."""
        self.face_mesh.close()