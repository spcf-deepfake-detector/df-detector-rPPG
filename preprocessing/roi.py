import cv2
import numpy as np
import mediapipe as mp

from typing import Dict, List, Optional


class ROIExtractor:
    """
    Extract forehead and cheek regions using MediaPipe Face Mesh.

    Input:
        frame: Full BGR video frame.
        bbox: [x, y, width, height] from face tracking.

    Output:
        {
            "forehead": {"crop": ndarray or None, "mask": ndarray or None},
            "left_cheek": {"crop": ndarray or None, "mask": ndarray or None},
            "right_cheek": {"crop": ndarray or None, "mask": ndarray or None},
        }
    """

    FOREHEAD_INDICES = [10, 67, 109, 151, 337, 297, 9]
    LEFT_CHEEK_INDICES = [117, 118, 100, 126, 209, 198]
    RIGHT_CHEEK_INDICES = [346, 347, 329, 355, 429, 419]

    def __init__(self, max_num_faces: int = 5):#usually number of faces each video
        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    @staticmethod
    def _empty_roi() -> Dict[str, Optional[np.ndarray]]:
        return {"crop": None, "mask": None}

    def _empty_results(self) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
        return {
            "forehead": self._empty_roi(),
            "left_cheek": self._empty_roi(),
            "right_cheek": self._empty_roi(),
        }

    @staticmethod
    def _box_iou(box_a, box_b) -> float:
        """
        Calculate intersection over union (IoU) for boxes in
        [x1, y1, x2, y2] format.
        """
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)
        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        intersection_w = max(0, ix2 - ix1)
        intersection_h = max(0, iy2 - iy1)
        intersection = intersection_w * intersection_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return intersection / union

    def _select_face_landmarks(
        self,
        results,
        frame_w: int,
        frame_h: int,
        target_box,
    ):
        """
        Select the detected face whose landmark bounding box has
        the greatest IoU with the tracked face box.
        """
        faces = results.multi_face_landmarks or []

        best_landmarks = None
        best_iou = 0.0

        for face in faces:
            landmarks = face.landmark

            xs = np.asarray(
                [point.x for point in landmarks], dtype=np.float64
            )
            ys = np.asarray(
                [point.y for point in landmarks], dtype=np.float64
            )

            if (
                not np.isfinite(xs).all()
                or not np.isfinite(ys).all()
            ):
                continue

            face_box = [
                int(np.floor(xs.min() * frame_w)),
                int(np.floor(ys.min() * frame_h)),
                int(np.ceil(xs.max() * frame_w)),
                int(np.ceil(ys.max() * frame_h)),
            ]

            # Clip the landmark box to the image.
            face_box[0] = max(0, min(frame_w, face_box[0]))
            face_box[1] = max(0, min(frame_h, face_box[1]))
            face_box[2] = max(0, min(frame_w, face_box[2]))
            face_box[3] = max(0, min(frame_h, face_box[3]))

            score = self._box_iou(face_box, target_box)

            if score > best_iou:
                best_iou = score
                best_landmarks = landmarks

        if best_landmarks is None:
            return None

        return best_landmarks

    def extract_rois(
        self,
        frame: np.ndarray,
        bbox: List[int],
    ) -> Dict[str, Dict[str, Optional[np.ndarray]]]:

        empty_results = self._empty_results()

        # 1. Validate the frame.
        if not isinstance(frame, np.ndarray):
            return empty_results

        if (
            frame.size == 0
            or frame.ndim != 3
            or frame.shape[2] != 3
        ):
            return empty_results

        frame_h, frame_w = frame.shape[:2]

        # 2. Validate the bounding box.
        if bbox is None:
            return empty_results

        try:
            values = np.asarray(bbox, dtype=np.float64)

            if values.shape != (4,) or not np.isfinite(values).all():
                return empty_results

            x, y, box_w, box_h = np.round(values).astype(int).tolist()

        except (TypeError, ValueError, OverflowError):
            return empty_results

        if box_w <= 0 or box_h <= 0:
            return empty_results

        # 3. Clip the tracked box to the frame.
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(frame_w, x + box_w)
        y2 = min(frame_h, y + box_h)

        if x2 <= x1 or y2 <= y1:
            return empty_results

        # Add a margin to tolerate small differences in box position.
        pad_x = int(box_w * 0.30)
        pad_y = int(box_h * 0.30)

        target_box = [
            max(0, x - pad_x),
            max(0, y - pad_y),
            min(frame_w, x + box_w + pad_x),
            min(frame_h, y + box_h + pad_y),
        ]

        # 4. Convert BGR to RGB for MediaPipe.
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # 5. Detect facial landmarks.
        results = self.face_mesh.process(rgb_frame)

        if not results.multi_face_landmarks:
            return empty_results

        # 6. Select the face that best overlaps the tracked box.
        landmarks = self._select_face_landmarks(
            results,
            frame_w,
            frame_h,
            target_box,
        )

        if landmarks is None:
            return empty_results

        # 7. Extract one masked region from selected landmarks.
        def get_mask_crop(
            indices: List[int],
        ) -> Dict[str, Optional[np.ndarray]]:

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

            # 8. Keep all polygon points inside the image.
            points[:, 0] = np.clip(points[:, 0], 0, frame_w - 1)
            points[:, 1] = np.clip(points[:, 1], 0, frame_h - 1)

            if len(np.unique(points, axis=0)) < 3:
                return self._empty_roi()

            if abs(cv2.contourArea(points)) < 4:
                return self._empty_roi()

            # 9. Create the smallest rectangle around the polygon.
            rx, ry, rw, rh = cv2.boundingRect(points)

            if rw <= 0 or rh <= 0:
                return self._empty_roi()

            crop = frame[ry:ry + rh, rx:rx + rw]

            if (
                crop.size == 0
                or crop.ndim != 3
                or crop.shape[2] != 3
            ):
                return self._empty_roi()

            # 10. Create a mask relative to the cropped image.
            mask = np.zeros(crop.shape[:2], dtype=np.uint8)

            local_points = points - np.array(
                [rx, ry], dtype=np.int32
            )

            cv2.fillPoly(mask, [local_points], 255)

            # 11. Reject empty masks.
            if (
                mask.shape != crop.shape[:2]
                or cv2.countNonZero(mask) == 0
            ):
                return self._empty_roi()

            return {"crop": crop, "mask": mask}

        # 12. Return all three regions using the same structure.
        return {
            "forehead": get_mask_crop(self.FOREHEAD_INDICES),
            "left_cheek": get_mask_crop(self.LEFT_CHEEK_INDICES),
            "right_cheek": get_mask_crop(self.RIGHT_CHEEK_INDICES),
        }

    def close(self):
        """Release MediaPipe Face Mesh resources."""
        self.face_mesh.close()