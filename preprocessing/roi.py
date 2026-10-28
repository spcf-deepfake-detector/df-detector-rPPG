import cv2
import numpy as np
import mediapipe as mp

from typing import Dict, List, Optional, Any


class ROIExtractor:
    """
    Extract forehead and cheek regions using MediaPipe Face Mesh.

    Input:
        frame: Full BGR video frame.
        bbox: [x, y, width, height] from face tracking.

    Output:
        {
            "forehead": {
                "crop": ndarray or None,
                "mask": ndarray or None
            },
            "left_cheek": {
                "crop": ndarray or None,
                "mask": ndarray or None
            },
            "right_cheek": {
                "crop": ndarray or None,
                "mask": ndarray or None
            }
        }

    Diagnostic counters are available through self.stats.

    Notes:
        - The crop is rectangular.
        - The mask identifies the polygon within that crop.
        - The mask does not guarantee that every pixel represents skin.
        - RGB extraction is performed by RGBExtractor, not this class.
    """

    FOREHEAD_INDICES = [10, 67, 109, 151, 337, 297, 9]

    LEFT_CHEEK_INDICES = [117, 118, 100, 126, 209, 198]

    RIGHT_CHEEK_INDICES = [346, 347, 329, 355, 429, 419]

    ROI_NAMES = (
        "forehead",
        "left_cheek",
        "right_cheek",
    )

    def __init__(
        self,
        max_num_faces: int = 1,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        min_skin_pixels: int = 20,
        min_polygon_area: float = 4.0,
        min_match_iou: float = 0.0,
        debug: bool = False,
    ):
        """
        Initialize the MediaPipe Face Mesh model.

        Args:
            max_num_faces:
                Maximum number of faces MediaPipe should detect.

            min_detection_confidence:
                Minimum confidence for face detection.

            min_tracking_confidence:
                Minimum confidence for face tracking.

            min_skin_pixels:
                Minimum number of pixels inside an ROI mask.

                20 is an initial configurable value, not a
                scientifically validated threshold.

            min_polygon_area:
                Minimum polygon area in square pixels.

            min_match_iou:
                Minimum IoU required to accept a face match.

                The default of 0.0 permits a positive-area
                candidate with zero IoU to be considered, but
                it does not permit a zero-IoU match to be
                accepted because the comparison is strict.

            debug:
                Print detailed diagnostics when True.
        """

        if not isinstance(max_num_faces, int) or max_num_faces < 1:
            raise ValueError(
                "max_num_faces must be a positive integer."
            )

        if not 0.0 <= min_detection_confidence <= 1.0:
            raise ValueError(
                "min_detection_confidence must be between 0 and 1."
            )

        if not 0.0 <= min_tracking_confidence <= 1.0:
            raise ValueError(
                "min_tracking_confidence must be between 0 and 1."
            )

        if (
            not isinstance(min_skin_pixels, int)
            or min_skin_pixels < 1
        ):
            raise ValueError(
                "min_skin_pixels must be a positive integer."
            )

        if (
            not np.isfinite(min_polygon_area)
            or min_polygon_area < 0
        ):
            raise ValueError(
                "min_polygon_area must be finite and nonnegative."
            )

        if (
            not np.isfinite(min_match_iou)
            or not 0.0 <= min_match_iou <= 1.0
        ):
            raise ValueError(
                "min_match_iou must be between 0 and 1."
            )

        self.min_skin_pixels = min_skin_pixels
        self.min_polygon_area = float(min_polygon_area)
        self.min_match_iou = float(min_match_iou)
        self.debug = bool(debug)

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )

        self.stats = self._new_stats()

        self._closed = False

    # ---------------------------------------------------------
    # Diagnostics
    # ---------------------------------------------------------

    @staticmethod
    def _new_stats() -> Dict[str, Any]:
        """Create a fresh set of diagnostic counters."""

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
        """Reset diagnostic counters without resetting Face Mesh."""

        self.stats = self._new_stats()

    def get_stats(self) -> Dict[str, Any]:
        """Return a copy of the diagnostic counters."""

        return {
            **{
                key: value
                for key, value in self.stats.items()
                if key != "roi_failures"
            },
            "roi_failures": {
                name: counts.copy()
                for name, counts in self.stats["roi_failures"].items()
            },
        }

    def print_stats(self) -> None:
        """Print a readable diagnostic summary."""

        stats = self.stats

        print("\n========== ROI EXTRACTION SUMMARY ==========")

        print(
            f"Frames processed: "
            f"{stats['frames_processed']}"
        )

        print(
            f"Invalid frames: "
            f"{stats['invalid_frame']}"
        )

        print(
            f"Invalid bounding boxes: "
            f"{stats['invalid_bbox']}"
        )

        print(
            f"Face Mesh errors: "
            f"{stats['face_mesh_errors']}"
        )

        print(
            f"No landmarks detected: "
            f"{stats['no_landmarks']}"
        )

        print(
            f"Invalid landmark sets: "
            f"{stats['invalid_landmarks']}"
        )

        print(
            f"No matching face: "
            f"{stats['no_face_match']}"
        )

        print(
            f"Face selected: "
            f"{stats['face_selected']}"
        )

        print(
            f"Valid forehead ROIs: "
            f"{stats['forehead_valid']}"
        )

        print(
            f"Valid left cheek ROIs: "
            f"{stats['left_cheek_valid']}"
        )

        print(
            f"Valid right cheek ROIs: "
            f"{stats['right_cheek_valid']}"
        )

        print(
            f"Frames with at least one valid ROI: "
            f"{stats['frames_with_valid_roi']}"
        )

        print(
            f"Frames without a valid ROI: "
            f"{stats['frames_without_valid_roi']}"
        )

        print("\nROI failure details:")

        for roi_name, failures in stats["roi_failures"].items():
            print(f"\n  {roi_name}:")

            for reason, count in failures.items():
                print(f"    {reason}: {count}")

        print("============================================\n")

    # ---------------------------------------------------------
    # Empty results
    # ---------------------------------------------------------

    @staticmethod
    def _empty_roi() -> Dict[str, Optional[np.ndarray]]:
        """Return an empty ROI using the expected output structure."""

        return {
            "crop": None,
            "mask": None,
        }

    def _empty_results(
        self,
    ) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
        """Return empty forehead and cheek regions."""

        return {
            "forehead": self._empty_roi(),
            "left_cheek": self._empty_roi(),
            "right_cheek": self._empty_roi(),
        }

    # ---------------------------------------------------------
    # Bounding-box utilities
    # ---------------------------------------------------------

    @staticmethod
    def _box_iou(box_a, box_b) -> float:
        """
        Calculate intersection over union.

        Boxes use [x1, y1, x2, y2] format.
        """

        ax1, ay1, ax2, ay2 = box_a
        bx1, by1, bx2, by2 = box_b

        intersection_x1 = max(ax1, bx1)
        intersection_y1 = max(ay1, by1)

        intersection_x2 = min(ax2, bx2)
        intersection_y2 = min(ay2, by2)

        intersection_width = max(
            0,
            intersection_x2 - intersection_x1,
        )

        intersection_height = max(
            0,
            intersection_y2 - intersection_y1,
        )

        intersection_area = (
            intersection_width * intersection_height
        )

        area_a = (
            max(0, ax2 - ax1)
            * max(0, ay2 - ay1)
        )

        area_b = (
            max(0, bx2 - bx1)
            * max(0, by2 - by1)
        )

        union_area = (
            area_a + area_b - intersection_area
        )

        if union_area <= 0:
            return 0.0

        return float(intersection_area / union_area)

    @staticmethod
    def _valid_bbox(bbox) -> Optional[List[int]]:
        """
        Validate and normalize [x, y, width, height].

        Returns None for invalid input.
        """

        if bbox is None:
            return None

        try:
            values = np.asarray(
                bbox,
                dtype=np.float64,
            )

            if values.shape != (4,):
                return None

            if not np.isfinite(values).all():
                return None

            x, y, width, height = (
                np.round(values).astype(int).tolist()
            )

        except (TypeError, ValueError, OverflowError):
            return None

        if width <= 0 or height <= 0:
            return None

        return [x, y, width, height]

    # ---------------------------------------------------------
    # Face landmark selection
    # ---------------------------------------------------------

    def _select_face_landmarks(
        self,
        results,
        frame_w: int,
        frame_h: int,
        target_box,
    ):
        """
        Select the face whose landmark bounding box has
        the greatest IoU with the tracked face box.

        Returns:
            (landmarks, best_iou)

        If no acceptable face is found:
            (None, best_iou)
        """

        faces = results.multi_face_landmarks or []

        best_landmarks = None
        best_iou = -1.0

        for face in faces:
            landmarks = face.landmark

            xs = np.asarray(
                [point.x for point in landmarks],
                dtype=np.float64,
            )

            ys = np.asarray(
                [point.y for point in landmarks],
                dtype=np.float64,
            )

            # Reject malformed landmark coordinates.
            if (
                len(xs) == 0
                or len(ys) == 0
                or not np.isfinite(xs).all()
                or not np.isfinite(ys).all()
            ):
                continue

            # MediaPipe coordinates are normalized relative
            # to the dimensions of the original frame.
            face_box = [
                int(np.floor(xs.min() * frame_w)),
                int(np.floor(ys.min() * frame_h)),
                int(np.ceil(xs.max() * frame_w)),
                int(np.ceil(ys.max() * frame_h)),
            ]

            # Clip the box to the image boundaries.
            face_box[0] = max(
                0,
                min(frame_w, face_box[0]),
            )

            face_box[1] = max(
                0,
                min(frame_h, face_box[1]),
            )

            face_box[2] = max(
                0,
                min(frame_w, face_box[2]),
            )

            face_box[3] = max(
                0,
                min(frame_h, face_box[3]),
            )

            # Reject degenerate bounding boxes.
            if (
                face_box[2] <= face_box[0]
                or face_box[3] <= face_box[1]
            ):
                continue

            score = self._box_iou(
                face_box,
                target_box,
            )

            if self.debug:
                print(
                    "[FACE MATCH DEBUG] "
                    f"Target box: {target_box}, "
                    f"Face Mesh box: {face_box}, "
                    f"IoU: {score:.4f}"
                )

            # Use >= so the first valid candidate can be
            # selected even if its IoU is exactly zero.
            if score > best_iou:
                best_iou = score
                best_landmarks = landmarks

        # No candidate had usable landmark coordinates.
        if best_landmarks is None:
            return None, 0.0

        # Reject candidates that do not overlap the target
        # sufficiently. The default threshold is 0.0.
        #
        # A score of exactly zero is rejected even at the
        # default threshold because it provides no evidence
        # of spatial overlap.
        if best_iou <= self.min_match_iou:
            return None, best_iou

        return best_landmarks, best_iou

    # ---------------------------------------------------------
    # ROI extraction
    # ---------------------------------------------------------

    def extract_rois(
        self,
        frame: np.ndarray,
        bbox: List[int],
    ) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
        """
        Extract forehead and cheek crops and masks from one frame.

        Returns the same dictionary structure as the original
        implementation so RGBExtractor can consume the results.
        """

        if self._closed:
            raise RuntimeError(
                "ROIExtractor is closed. Create a new instance "
                "before extracting more ROIs."
            )

        self.stats["frames_processed"] += 1

        empty_results = self._empty_results()

        # 1. Validate the frame.
        if not isinstance(frame, np.ndarray):
            self.stats["invalid_frame"] += 1
            return empty_results

        if (
            frame.size == 0
            or frame.ndim != 3
            or frame.shape[2] != 3
        ):
            self.stats["invalid_frame"] += 1
            return empty_results

        frame_h, frame_w = frame.shape[:2]

        # 2. Validate the tracked bounding box.
        valid_bbox = self._valid_bbox(bbox)

        if valid_bbox is None:
            self.stats["invalid_bbox"] += 1
            return empty_results

        x, y, box_w, box_h = valid_bbox

        # 3. Clip the tracked box to the image.
        x1 = max(0, x)
        y1 = max(0, y)

        x2 = min(frame_w, x + box_w)
        y2 = min(frame_h, y + box_h)

        if x2 <= x1 or y2 <= y1:
            self.stats["invalid_bbox"] += 1
            return empty_results

        # 4. Expand the target box to tolerate differences
        # between the detector and landmark bounding boxes.
        pad_x = int(box_w * 0.30)
        pad_y = int(box_h * 0.30)

        target_box = [
            max(0, x - pad_x),
            max(0, y - pad_y),
            min(frame_w, x + box_w + pad_x),
            min(frame_h, y + box_h + pad_y),
        ]

        # 5. Convert BGR to RGB for MediaPipe.
        try:
            rgb_frame = cv2.cvtColor(
                frame,
                cv2.COLOR_BGR2RGB,
            )

        except cv2.error:
            self.stats["invalid_frame"] += 1
            return empty_results

        # 6. Detect facial landmarks.
        try:
            results = self.face_mesh.process(rgb_frame)

        except Exception as exc:
            self.stats["face_mesh_errors"] += 1

            if self.debug:
                print(
                    "[ROI ERROR] Face Mesh failed: "
                    f"{exc}"
                )

            return empty_results

        if not results.multi_face_landmarks:
            self.stats["no_landmarks"] += 1
            return empty_results

        # 7. Select the best matching face.
        landmarks, match_iou = self._select_face_landmarks(
            results,
            frame_w,
            frame_h,
            target_box,
        )

        if landmarks is None:
            self.stats["no_face_match"] += 1

            if self.debug:
                print(
                    "[ROI DEBUG] No acceptable face match. "
                    f"Best IoU: {match_iou:.4f}"
                )

            return empty_results

        self.stats["face_selected"] += 1

        if self.debug:
            print(
                "[ROI DEBUG] Face selected. "
                f"IoU: {match_iou:.4f}"
            )

        # -----------------------------------------------------
        # 8. Extract one masked facial region.
        # -----------------------------------------------------

        def get_mask_crop(
            indices: List[int],
            roi_name: str,
        ) -> Dict[str, Optional[np.ndarray]]:
            """Create a crop and polygon mask for one ROI."""

            failures = self.stats["roi_failures"][roi_name]

            # Validate landmark indices.
            if (
                not indices
                or min(indices) < 0
                or max(indices) >= len(landmarks)
            ):
                failures["invalid_indices"] += 1
                return self._empty_roi()

            points = []

            # Convert normalized landmark coordinates to pixels.
            for idx in indices:
                landmark = landmarks[idx]

                if not (
                    np.isfinite(landmark.x)
                    and np.isfinite(landmark.y)
                ):
                    failures["invalid_landmarks"] += 1
                    return self._empty_roi()

                px = int(round(landmark.x * frame_w))
                py = int(round(landmark.y * frame_h))

                points.append([px, py])

            points = np.asarray(
                points,
                dtype=np.int32,
            )

            # Keep polygon points inside the image.
            points[:, 0] = np.clip(
                points[:, 0],
                0,
                frame_w - 1,
            )

            points[:, 1] = np.clip(
                points[:, 1],
                0,
                frame_h - 1,
            )

            # At least three distinct points are required
            # to form a polygon.
            if len(np.unique(points, axis=0)) < 3:
                failures["too_few_points"] += 1
                return self._empty_roi()

            # Reject tiny or degenerate polygons.
            polygon_area = abs(
                cv2.contourArea(points)
            )

            if polygon_area < self.min_polygon_area:
                failures["polygon_too_small"] += 1
                return self._empty_roi()

            # Find the smallest rectangle containing the polygon.
            rx, ry, rw, rh = cv2.boundingRect(points)

            if rw <= 0 or rh <= 0:
                failures["invalid_crop"] += 1
                return self._empty_roi()

            # Extract the rectangular crop from the original
            # BGR frame.
            crop = frame[
                ry:ry + rh,
                rx:rx + rw,
            ]

            if (
                crop.size == 0
                or crop.ndim != 3
                or crop.shape[2] != 3
            ):
                failures["invalid_crop"] += 1
                return self._empty_roi()

            # Create a mask whose dimensions match the crop.
            mask = np.zeros(
                crop.shape[:2],
                dtype=np.uint8,
            )

            # Convert full-frame polygon points into crop-relative
            # coordinates.
            local_points = points - np.array(
                [rx, ry],
                dtype=np.int32,
            )

            # White pixels belong to the ROI polygon.
            cv2.fillPoly(
                mask,
                [local_points],
                255,
            )

            # Validate mask dimensions.
            if mask.shape != crop.shape[:2]:
                failures["invalid_crop"] += 1
                return self._empty_roi()

            # Count the pixels included by the polygon.
            mask_pixels = cv2.countNonZero(mask)

            if mask_pixels < self.min_skin_pixels:
                failures["too_few_mask_pixels"] += 1
                return self._empty_roi()

            # The returned structure remains compatible with
            # the existing RGBExtractor.
            return {
                "crop": crop,
                "mask": mask,
            }

        # 9. Extract the three regions independently.
        roi_results = {
            "forehead": get_mask_crop(
                self.FOREHEAD_INDICES,
                "forehead",
            ),
            "left_cheek": get_mask_crop(
                self.LEFT_CHEEK_INDICES,
                "left_cheek",
            ),
            "right_cheek": get_mask_crop(
                self.RIGHT_CHEEK_INDICES,
                "right_cheek",
            ),
        }

        # 10. Update per-region success counters.
        for roi_name in self.ROI_NAMES:
            roi = roi_results[roi_name]

            if (
                roi["crop"] is not None
                and roi["mask"] is not None
            ):
                self.stats[f"{roi_name}_valid"] += 1

        # 11. Count whether at least one region is usable.
        has_valid_roi = any(
            roi_results[name]["crop"] is not None
            and roi_results[name]["mask"] is not None
            for name in self.ROI_NAMES
        )

        if has_valid_roi:
            self.stats["frames_with_valid_roi"] += 1
        else:
            self.stats["frames_without_valid_roi"] += 1

        return roi_results

    # ---------------------------------------------------------
    # Cleanup
    # ---------------------------------------------------------

    def close(self) -> None:
        """Release MediaPipe Face Mesh resources."""

        if not self._closed:
            self.face_mesh.close()
            self._closed = True

    def __enter__(self):
        """Allow use with a context manager."""

        if self._closed:
            raise RuntimeError(
                "Cannot reuse a closed ROIExtractor."
            )

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Automatically release resources when leaving a context."""

        self.close()
        return False
