import cv2
import numpy as np
import mediapipe as mp

from typing import Any, Dict, List, Optional, Tuple


class ROIExtractor:
    """
    Extract forehead and cheek regions from a full BGR video frame.

    Input:
        frame: BGR image with shape (height, width, 3).
        bbox: [x, y, width, height] for the tracked face.

    Output:
        {"forehead": {"crop": ndarray or None, "mask": ndarray or None},
         "left_cheek": {"crop": ndarray or None, "mask": ndarray or None},
         "right_cheek": {"crop": ndarray or None, "mask": ndarray or None}}

    Crops are rectangular; masks identify pixels inside the ROI polygon.
    A mask does not guarantee every included pixel is skin.
    """

    FOREHEAD_INDICES = [10, 67, 109, 151, 337, 297, 9]
    LEFT_CHEEK_INDICES = [117, 118, 100, 126, 209, 198]
    RIGHT_CHEEK_INDICES = [346, 347, 329, 355, 429, 419]
    ROI_NAMES = ("forehead", "left_cheek", "right_cheek")

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_skin_pixels: int = 20,
        min_polygon_area: float = 4.0,
        min_match_iou: float = 0.0,
        bbox_padding_fraction: float = 0.30,
        debug: bool = False,
    ):
        if not isinstance(max_num_faces, int) or max_num_faces < 1:
            raise ValueError("max_num_faces must be a positive integer.")

        for name, value in (
            ("min_detection_confidence", min_detection_confidence),
            ("min_tracking_confidence", min_tracking_confidence),
            ("min_match_iou", min_match_iou),
        ):
            if not np.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1.")

        if not isinstance(min_skin_pixels, int) or min_skin_pixels < 1:
            raise ValueError("min_skin_pixels must be a positive integer.")
        if not np.isfinite(min_polygon_area) or min_polygon_area < 0:
            raise ValueError("min_polygon_area must be finite and nonnegative.")
        if (
            not np.isfinite(bbox_padding_fraction)
            or bbox_padding_fraction < 0
            or bbox_padding_fraction > 2
        ):
            raise ValueError("bbox_padding_fraction must be between 0 and 2.")

        self.min_skin_pixels = min_skin_pixels
        self.min_polygon_area = float(min_polygon_area)
        self.min_match_iou = float(min_match_iou)
        self.bbox_padding_fraction = float(bbox_padding_fraction)
        self.debug = bool(debug)

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=float(min_detection_confidence),
            min_tracking_confidence=float(min_tracking_confidence),
        )
        self.stats = self._new_stats()
        self._closed = False

    @staticmethod
    def _new_stats() -> Dict[str, Any]:
        return {
            "frames_processed": 0,
            "invalid_frame": 0,
            "invalid_bbox": 0,
            "face_mesh_errors": 0,
            "no_landmarks": 0,
            "invalid_landmarks": 0,
            "no_face_match": 0,
            "face_selected": 0,
            "forehead_valid": 0,
            "left_cheek_valid": 0,
            "right_cheek_valid": 0,
            "frames_with_valid_roi": 0,
            "frames_without_valid_roi": 0,
            "roi_failures": {
                name: {
                    "invalid_indices": 0,
                    "invalid_landmarks": 0,
                    "too_few_points": 0,
                    "polygon_too_small": 0,
                    "invalid_crop": 0,
                    "too_few_mask_pixels": 0,
                }
                for name in ROIExtractor.ROI_NAMES
            },
        }

    def reset_stats(self) -> None:
        """Reset counters without recreating Face Mesh."""
        self.stats = self._new_stats()

    def get_stats(self) -> Dict[str, Any]:
        """Return a copy of the current counters."""
        result = {
            key: value
            for key, value in self.stats.items()
            if key != "roi_failures"
        }
        result["roi_failures"] = {
            name: counts.copy()
            for name, counts in self.stats["roi_failures"].items()
        }
        return result

    def print_stats(self) -> None:
        s = self.stats
        print("\n========== ROI EXTRACTION SUMMARY ==========")
        labels = (
            ("Frames processed", "frames_processed"),
            ("Invalid frames", "invalid_frame"),
            ("Invalid bounding boxes", "invalid_bbox"),
            ("Face Mesh errors", "face_mesh_errors"),
            ("No landmarks detected", "no_landmarks"),
            ("Invalid landmark sets", "invalid_landmarks"),
            ("No matching face", "no_face_match"),
            ("Face selected", "face_selected"),
            ("Valid forehead ROIs", "forehead_valid"),
            ("Valid left cheek ROIs", "left_cheek_valid"),
            ("Valid right cheek ROIs", "right_cheek_valid"),
            ("Frames with at least one valid ROI", "frames_with_valid_roi"),
            ("Frames without a valid ROI", "frames_without_valid_roi"),
        )
        for label, key in labels:
            print(f"{label}: {s[key]}")

        print("\nROI failure details:")
        for name, failures in s["roi_failures"].items():
            print(f"\n  {name}:")
            for reason, count in failures.items():
                print(f"    {reason}: {count}")
        print("============================================\n")

    @staticmethod
    def _empty_roi() -> Dict[str, Optional[np.ndarray]]:
        return {"crop": None, "mask": None}

    def _empty_results(self) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
        return {name: self._empty_roi() for name in self.ROI_NAMES}

    @staticmethod
    def _valid_bbox(bbox) -> Optional[List[int]]:
        """Validate [x, y, width, height]."""
        if bbox is None:
            return None
        try:
            values = np.asarray(bbox, dtype=np.float64)
            if values.shape != (4,) or not np.isfinite(values).all():
                return None
            x, y, width, height = np.round(values).astype(int).tolist()
        except (TypeError, ValueError, OverflowError):
            return None
        if width <= 0 or height <= 0:
            return None
        return [x, y, width, height]

    @staticmethod
    def _box_iou(box_a: List[int], box_b: List[int]) -> float:
        """IoU for boxes in [x1, y1, x2, y2] format."""
        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b
        ix1, iy1 = max(ax1, bx1), max(ay1, by1)
        ix2, iy2 = min(ax2, bx2), min(ay2, by2)
        intersection = max(0, ix2 - ix1) * max(0, iy2 - iy1)
        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - intersection
        return float(intersection / union) if union > 0 else 0.0

    def _select_face_landmarks(
        self,
        results,
        frame_w: int,
        frame_h: int,
        target_box: List[int],
    ) -> Tuple[Optional[Any], float]:
        """Choose the landmark face that best overlaps the tracked box."""
        faces = results.multi_face_landmarks or []
        best_landmarks = None
        best_iou = -1.0

        for face in faces:
            landmarks = face.landmark
            if not landmarks:
                continue
            xs = np.asarray([p.x for p in landmarks], dtype=np.float64)
            ys = np.asarray([p.y for p in landmarks], dtype=np.float64)
            if not np.isfinite(xs).all() or not np.isfinite(ys).all():
                continue

            face_box = [
                max(0, min(frame_w, int(np.floor(xs.min() * frame_w)))),
                max(0, min(frame_h, int(np.floor(ys.min() * frame_h)))),
                max(0, min(frame_w, int(np.ceil(xs.max() * frame_w)))),
                max(0, min(frame_h, int(np.ceil(ys.max() * frame_h)))),
            ]
            if face_box[2] <= face_box[0] or face_box[3] <= face_box[1]:
                continue

            score = self._box_iou(face_box, target_box)
            if self.debug:
                print(
                    "[FACE MATCH DEBUG] "
                    f"target={target_box}, mesh_box={face_box}, IoU={score:.4f}"
                )
            if score > best_iou:
                best_iou = score
                best_landmarks = landmarks

        if best_landmarks is None or best_iou <= self.min_match_iou:
            return None, max(best_iou, 0.0)
        return best_landmarks, best_iou

    def extract_rois(
        self,
        frame: np.ndarray,
        bbox: List[int],
    ) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
        """Extract masked forehead and cheek regions from one BGR frame."""
        if self._closed:
            raise RuntimeError("ROIExtractor is closed; create a new instance.")

        self.stats["frames_processed"] += 1
        empty = self._empty_results()

        if (
            not isinstance(frame, np.ndarray)
            or frame.size == 0
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.dtype != np.uint8
        ):
            self.stats["invalid_frame"] += 1
            if self.debug:
                print("[ROI DEBUG] Invalid frame: expected non-empty uint8 BGR image.")
            return empty

        frame_h, frame_w = frame.shape[:2]
        valid_bbox = self._valid_bbox(bbox)
        if valid_bbox is None:
            self.stats["invalid_bbox"] += 1
            if self.debug:
                print(f"[ROI DEBUG] Invalid bbox: {bbox!r}")
            return empty

        x, y, box_w, box_h = valid_bbox
        x1, y1 = max(0, x), max(0, y)
        x2, y2 = min(frame_w, x + box_w), min(frame_h, y + box_h)
        if x2 <= x1 or y2 <= y1:
            self.stats["invalid_bbox"] += 1
            if self.debug:
                print(f"[ROI DEBUG] Bbox lies outside frame: {valid_bbox}")
            return empty

        pad_x = int(round(box_w * self.bbox_padding_fraction))
        pad_y = int(round(box_h * self.bbox_padding_fraction))
        target_box = [
            max(0, x - pad_x),
            max(0, y - pad_y),
            min(frame_w, x + box_w + pad_x),
            min(frame_h, y + box_h + pad_y),
        ]

        try:
            # OpenCV frames are BGR; MediaPipe Face Mesh expects RGB.
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
        except Exception as exc:
            self.stats["face_mesh_errors"] += 1
            if self.debug:
                print(f"[ROI ERROR] Face Mesh processing failed: {exc}")
            return empty

        if not results.multi_face_landmarks:
            self.stats["no_landmarks"] += 1
            if self.debug:
                print(
                    "[NO LANDMARKS DEBUG] "
                    f"Frame={frame_w}x{frame_h}, bbox={valid_bbox}, "
                    f"clipped face-box={x2 - x1}x{y2 - y1}, "
                    f"target_box={target_box}"
                )
            return empty

        landmarks, match_iou = self._select_face_landmarks(
            results, frame_w, frame_h, target_box
        )
        if landmarks is None:
            self.stats["no_face_match"] += 1
            if self.debug:
                print(
                    "[ROI DEBUG] Landmarks returned, but no face matched "
                    f"the target box. Best IoU={match_iou:.4f}; "
                    f"target_box={target_box}"
                )
            return empty

        self.stats["face_selected"] += 1
        if self.debug:
            print(f"[ROI DEBUG] Face selected; IoU={match_iou:.4f}")

        def make_roi(
            indices: List[int],
            roi_name: str,
        ) -> Dict[str, Optional[np.ndarray]]:
            failures = self.stats["roi_failures"][roi_name]

            if not indices or min(indices) < 0 or max(indices) >= len(landmarks):
                failures["invalid_indices"] += 1
                return self._empty_roi()

            points = []
            for index in indices:
                point = landmarks[index]
                if not np.isfinite(point.x) or not np.isfinite(point.y):
                    failures["invalid_landmarks"] += 1
                    return self._empty_roi()
                points.append([
                    int(round(point.x * frame_w)),
                    int(round(point.y * frame_h)),
                ])

            points = np.asarray(points, dtype=np.int32)
            points[:, 0] = np.clip(points[:, 0], 0, frame_w - 1)
            points[:, 1] = np.clip(points[:, 1], 0, frame_h - 1)

            if len(np.unique(points, axis=0)) < 3:
                failures["too_few_points"] += 1
                if self.debug:
                    print(f"[ROI DEBUG] {roi_name}: fewer than 3 unique polygon points.")
                return self._empty_roi()

            polygon_area = abs(float(cv2.contourArea(points)))
            if polygon_area < self.min_polygon_area:
                failures["polygon_too_small"] += 1
                if self.debug:
                    print(
                        f"[ROI DEBUG] {roi_name}: polygon area={polygon_area:.2f} px^2; "
                        f"minimum={self.min_polygon_area:.2f}; "
                        f"points={points.tolist()}"
                    )
                return self._empty_roi()

            rx, ry, rw, rh = cv2.boundingRect(points)
            if rw <= 0 or rh <= 0:
                failures["invalid_crop"] += 1
                return self._empty_roi()

            crop = frame[ry:ry + rh, rx:rx + rw]
            if crop.size == 0 or crop.ndim != 3 or crop.shape[2] != 3:
                failures["invalid_crop"] += 1
                return self._empty_roi()

            mask = np.zeros(crop.shape[:2], dtype=np.uint8)
            local_points = points - np.array([rx, ry], dtype=np.int32)
            cv2.fillPoly(mask, [local_points], 255)

            if mask.shape != crop.shape[:2]:
                failures["invalid_crop"] += 1
                return self._empty_roi()

            mask_pixels = int(cv2.countNonZero(mask))
            if mask_pixels < self.min_skin_pixels:
                failures["too_few_mask_pixels"] += 1
                if self.debug:
                    print(
                        f"[ROI DEBUG] {roi_name}: mask has {mask_pixels} pixels; "
                        f"minimum={self.min_skin_pixels}."
                    )
                return self._empty_roi()

            return {"crop": crop, "mask": mask}

        roi_results = {
            "forehead": make_roi(self.FOREHEAD_INDICES, "forehead"),
            "left_cheek": make_roi(self.LEFT_CHEEK_INDICES, "left_cheek"),
            "right_cheek": make_roi(self.RIGHT_CHEEK_INDICES, "right_cheek"),
        }

        for name in self.ROI_NAMES:
            roi = roi_results[name]
            if roi["crop"] is not None and roi["mask"] is not None:
                self.stats[f"{name}_valid"] += 1

        if any(
            roi_results[name]["crop"] is not None
            and roi_results[name]["mask"] is not None
            for name in self.ROI_NAMES
        ):
            self.stats["frames_with_valid_roi"] += 1
        else:
            self.stats["frames_without_valid_roi"] += 1

        return roi_results

    def close(self) -> None:
        """Release MediaPipe Face Mesh resources."""
        if not self._closed:
            self.face_mesh.close()
            self._closed = True

    def __enter__(self):
        if self._closed:
            raise RuntimeError("Cannot reuse a closed ROIExtractor.")
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()
        return False
