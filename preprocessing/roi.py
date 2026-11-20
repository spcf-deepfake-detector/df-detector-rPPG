import cv2
import numpy as np
import mediapipe as mp

from numbers import Integral, Real
from typing import Any, Dict, List, Optional, Tuple
from types import SimpleNamespace


class ROIExtractor:
    """
    PURPOSE:
        Extract forehead and cheek regions from a full BGR video frame
        using MediaPipe Face Mesh landmarks.

    INPUT:
        frame:
            Original OpenCV BGR image with shape (height, width, 3).

        bbox:
            Tracked face bounding box in [x, y, width, height] format.

    OUTPUT:
        Dictionary containing:
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

    HOW IT WORKS:
        1. Validate the frame and tracked face bounding box.
        2. Add padding around the tracked face.
        3. Crop and resize the padded face while preserving aspect ratio.
        4. Convert BGR to RGB for MediaPipe.
        5. Detect facial landmarks.
        6. Map landmarks back to original-frame coordinates.
        7. Match the detected face against the tracked face.
        8. Build forehead and cheek polygon masks.
        9. Return each ROI crop and its mask.

    IMPORTANT:
        - This class does not calculate RGB averages.
        - RGBExtractor uses the returned crops and masks.
        - A predicted tracking box must never be passed here as
          a valid measurement. The caller must verify detected=True
          and track_valid=True before requesting ROI extraction.
        - A polygon mask identifies the selected facial region,
          but does not guarantee that every pixel is skin.
    """

    # MediaPipe Face Mesh landmark groups.
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
        bbox_padding_fraction: float = 0.20,
        mesh_input_size: int = 128,
        debug: bool = False,
    ):
        # --------------------------------------------------
        # 1. VALIDATE CONFIGURATION
        # --------------------------------------------------

        if (
            isinstance(max_num_faces, bool)
            or not isinstance(max_num_faces, Integral)
            or max_num_faces < 1
        ):
            raise ValueError(
                "max_num_faces must be a positive integer."
            )

        for name, value in (
            ("min_detection_confidence", min_detection_confidence),
            ("min_tracking_confidence", min_tracking_confidence),
            ("min_match_iou", min_match_iou),
        ):
            self._validate_numeric_range(
                name,
                value,
                0.0,
                1.0,
            )

        if (
            isinstance(min_skin_pixels, bool)
            or not isinstance(min_skin_pixels, Integral)
            or min_skin_pixels < 1
        ):
            raise ValueError(
                "min_skin_pixels must be a positive integer."
            )

        self._validate_numeric_range(
            "min_polygon_area",
            min_polygon_area,
            0.0,
            float("inf"),
        )

        self._validate_numeric_range(
            "bbox_padding_fraction",
            bbox_padding_fraction,
            0.0,
            2.0,
        )

        if (
            isinstance(mesh_input_size, bool)
            or not isinstance(mesh_input_size, Integral)
            or mesh_input_size < 32
        ):
            raise ValueError(
                "mesh_input_size must be an integer of at least 32."
            )

        if not isinstance(debug, bool):
            raise TypeError(
                "debug must be a boolean."
            )

        self.max_num_faces = int(max_num_faces)

        self.min_detection_confidence = float(
            min_detection_confidence
        )

        self.min_tracking_confidence = float(
            min_tracking_confidence
        )

        self.min_skin_pixels = int(min_skin_pixels)

        self.min_polygon_area = float(min_polygon_area)

        self.min_match_iou = float(min_match_iou)

        self.bbox_padding_fraction = float(
            bbox_padding_fraction
        )

        self.mesh_input_size = int(mesh_input_size)

        self.debug = debug

        # --------------------------------------------------
        # 2. INITIALIZE MEDIAPIPE FACE MESH
        # --------------------------------------------------

        self.face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=False,
            max_num_faces=self.max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=(
                self.min_detection_confidence
            ),
            min_tracking_confidence=(
                self.min_tracking_confidence
            ),
        )

        self.stats = self._new_stats()

        self._closed = False

    # ======================================================
    # CONFIGURATION VALIDATION
    # ======================================================

    @staticmethod
    def _validate_numeric_range(
        name: str,
        value: Any,
        minimum: float,
        maximum: float,
    ) -> None:
        """
        Validate a finite numeric configuration value.

        Booleans are rejected even though Python treats them
        as integer values.
        """

        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not np.isfinite(value)
            or value < minimum
            or value > maximum
        ):
            raise ValueError(
                f"{name} must be between "
                f"{minimum} and {maximum}."
            )

    # ======================================================
    # STATISTICS
    # ======================================================

    @staticmethod
    def _new_stats() -> Dict[str, Any]:
        """
        Create a fresh statistics dictionary.
        """

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
        """
        Reset statistics without recreating MediaPipe Face Mesh.
        """

        self.stats = self._new_stats()

    def get_stats(self) -> Dict[str, Any]:
        """
        Return a copy of the statistics.

        Nested ROI failure dictionaries are copied too,
        preventing callers from modifying internal counters.
        """

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
        """
        Print ROI extraction statistics.
        """

        stats = self.stats

        print("\n========== ROI EXTRACTION SUMMARY ==========")

        labels = (
            ("Frames processed", "frames_processed"),
            ("Invalid frames", "invalid_frame"),
            ("Invalid bounding boxes", "invalid_bbox"),
            ("Face Mesh errors", "face_mesh_errors"),
            ("No landmarks detected", "no_landmarks"),
            ("Invalid landmark sets", "invalid_landmarks"),
            ("No matching face", "no_face_match"),
            ("Faces selected", "face_selected"),
            ("Valid forehead ROIs", "forehead_valid"),
            ("Valid left cheek ROIs", "left_cheek_valid"),
            ("Valid right cheek ROIs", "right_cheek_valid"),
            (
                "Frames with at least one valid ROI",
                "frames_with_valid_roi",
            ),
            (
                "Frames without a valid ROI",
                "frames_without_valid_roi",
            ),
        )

        for label, key in labels:
            print(f"{label}: {stats[key]}")

        print("\nROI failure details:")

        for name, failures in stats["roi_failures"].items():
            print(f"\n  {name}:")

            for reason, count in failures.items():
                print(f"    {reason}: {count}")

        print("============================================\n")

    # ======================================================
    # EMPTY RESULTS
    # ======================================================

    @staticmethod
    def _empty_roi() -> Dict[str, Optional[np.ndarray]]:
        """
        Return an empty ROI result.
        """

        return {
            "crop": None,
            "mask": None,
        }

    def _empty_results(
        self,
    ) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
        """
        Return empty results for all three facial regions.
        """

        return {
            name: self._empty_roi()
            for name in self.ROI_NAMES
        }

    def _return_empty_frame(
        self,
        results: Dict[str, Dict[str, Optional[np.ndarray]]],
    ) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
        """
        Record a frame with no usable ROI.

        Use this for early-return failure paths so that
        frames_without_valid_roi is consistently updated.
        """

        self.stats["frames_without_valid_roi"] += 1

        return results

    # ======================================================
    # BOUNDING-BOX VALIDATION
    # ======================================================

    @staticmethod
    def _valid_bbox(
        bbox: Any,
    ) -> Optional[List[int]]:
        """
        Validate [x, y, width, height].

        Returns:
            Integer bounding box or None.
        """

        if bbox is None:
            return None

        try:
            values = np.asarray(
                bbox,
                dtype=np.float64,
            )

            if (
                values.shape != (4,)
                or not np.isfinite(values).all()
            ):
                return None

            x, y, width, height = (
                np.round(values).astype(int).tolist()
            )

        except (
            TypeError,
            ValueError,
            OverflowError,
        ):
            return None

        if width <= 0 or height <= 0:
            return None

        return [
            x,
            y,
            width,
            height,
        ]

    # ======================================================
    # FACE-BOX GEOMETRY
    # ======================================================

    @staticmethod
    def _box_iou(
        box_a: List[int],
        box_b: List[int],
    ) -> float:
        """
        Calculate Intersection over Union.

        Both boxes use:
            [x1, y1, x2, y2]
        """

        ax1, ay1, ax2, ay2 = box_a

        bx1, by1, bx2, by2 = box_b

        ix1 = max(ax1, bx1)
        iy1 = max(ay1, by1)

        ix2 = min(ax2, bx2)
        iy2 = min(ay2, by2)

        intersection = (
            max(0, ix2 - ix1)
            * max(0, iy2 - iy1)
        )

        area_a = (
            max(0, ax2 - ax1)
            * max(0, ay2 - ay1)
        )

        area_b = (
            max(0, bx2 - bx1)
            * max(0, by2 - by1)
        )

        union = area_a + area_b - intersection

        if union <= 0:
            return 0.0

        return float(intersection / union)

    # ======================================================
    # SELECT THE FACE MATCHING THE TRACKED TARGET
    # ======================================================

    def _select_face_landmarks(
        self,
        results: Any,
        frame_w: int,
        frame_h: int,
        target_box: List[int],
    ) -> Tuple[Optional[Any], float]:
        """
        Select the Face Mesh landmark set that best overlaps
        the padded bounding box of the tracked target.

        This helps avoid extracting ROIs from another face
        when multiple faces are present.
        """

        faces = results.multi_face_landmarks or []

        best_landmarks = None

        best_iou = 0.0

        for face in faces:
            landmarks = getattr(face, "landmark", None)

            if not landmarks:
                self.stats["invalid_landmarks"] += 1
                continue

            try:
                xs = np.asarray(
                    [point.x for point in landmarks],
                    dtype=np.float64,
                )

                ys = np.asarray(
                    [point.y for point in landmarks],
                    dtype=np.float64,
                )

            except (
                AttributeError,
                TypeError,
                ValueError,
            ):
                self.stats["invalid_landmarks"] += 1
                continue

            if (
                not np.isfinite(xs).all()
                or not np.isfinite(ys).all()
            ):
                self.stats["invalid_landmarks"] += 1
                continue

            # Convert normalized landmark coordinates
            # into a face bounding box in original pixels.
            face_box = [
                max(
                    0,
                    min(
                        frame_w,
                        int(np.floor(xs.min() * frame_w)),
                    ),
                ),
                max(
                    0,
                    min(
                        frame_h,
                        int(np.floor(ys.min() * frame_h)),
                    ),
                ),
                max(
                    0,
                    min(
                        frame_w,
                        int(np.ceil(xs.max() * frame_w)),
                    ),
                ),
                max(
                    0,
                    min(
                        frame_h,
                        int(np.ceil(ys.max() * frame_h)),
                    ),
                ),
            ]

            if (
                face_box[2] <= face_box[0]
                or face_box[3] <= face_box[1]
            ):
                self.stats["invalid_landmarks"] += 1
                continue

            score = self._box_iou(
                face_box,
                target_box,
            )

            if self.debug:
                print(
                    "[FACE MATCH DEBUG] "
                    f"target={target_box}, "
                    f"mesh_box={face_box}, "
                    f"IoU={score:.4f}"
                )

            if score > best_iou:
                best_iou = score
                best_landmarks = landmarks

        # Require positive overlap even when the configured
        # minimum IoU is zero.
        if (
            best_landmarks is None
            or best_iou <= 0.0
            or best_iou < self.min_match_iou
        ):
            return None, best_iou

        return best_landmarks, best_iou

    # ======================================================
    # EXTRACT ROIS FROM ONE FRAME
    # ======================================================

    def extract_rois(
        self,
        frame: np.ndarray,
        bbox: List[int],
    ) -> Dict[str, Dict[str, Optional[np.ndarray]]]:
        """
        Extract masked forehead and cheek regions from one BGR frame.

        Args:
            frame:
                Original OpenCV BGR image.

            bbox:
                Actual tracked face bounding box in
                [x, y, width, height] format.

        Returns:
            Dictionary containing the three ROI crops and masks.

        Frames without usable ROIs return None for their
        crop and mask values.
        """

        if self._closed:
            raise RuntimeError(
                "ROIExtractor is closed; create a new instance."
            )

        self.stats["frames_processed"] += 1

        empty = self._empty_results()

        # --------------------------------------------------
        # STEP 1: VALIDATE FRAME
        # --------------------------------------------------

        if (
            not isinstance(frame, np.ndarray)
            or frame.size == 0
            or frame.ndim != 3
            or frame.shape[2] != 3
            or frame.dtype != np.uint8
        ):
            self.stats["invalid_frame"] += 1

            if self.debug:
                print(
                    "[ROI DEBUG] Invalid frame: "
                    "expected non-empty uint8 BGR image."
                )

            return self._return_empty_frame(empty)

        frame_h, frame_w = frame.shape[:2]

        # --------------------------------------------------
        # STEP 2: VALIDATE TRACKED FACE BOUNDING BOX
        # --------------------------------------------------

        valid_bbox = self._valid_bbox(bbox)

        if valid_bbox is None:
            self.stats["invalid_bbox"] += 1

            if self.debug:
                print(
                    f"[ROI DEBUG] Invalid bbox: {bbox!r}"
                )

            return self._return_empty_frame(empty)

        x, y, box_w, box_h = valid_bbox

        # Clip the original face box to the image.
        x1 = max(0, x)
        y1 = max(0, y)

        x2 = min(frame_w, x + box_w)
        y2 = min(frame_h, y + box_h)

        if x2 <= x1 or y2 <= y1:
            self.stats["invalid_bbox"] += 1

            if self.debug:
                print(
                    f"[ROI DEBUG] Bbox lies outside frame: "
                    f"{valid_bbox}"
                )

            return self._return_empty_frame(empty)

        # --------------------------------------------------
        # STEP 3: ADD PADDING
        # --------------------------------------------------

        pad_x = int(
            round(box_w * self.bbox_padding_fraction)
        )

        pad_y = int(
            round(box_h * self.bbox_padding_fraction)
        )

        target_box = [
            max(0, x - pad_x),
            max(0, y - pad_y),
            min(frame_w, x + box_w + pad_x),
            min(frame_h, y + box_h + pad_y),
        ]

        crop_x1, crop_y1, crop_x2, crop_y2 = target_box

        # --------------------------------------------------
        # STEP 4: CROP PADDED FACE REGION
        # --------------------------------------------------

        face_crop = frame[
            crop_y1:crop_y2,
            crop_x1:crop_x2,
        ]

        if face_crop.size == 0:
            self.stats["invalid_frame"] += 1

            return self._return_empty_frame(empty)

        crop_h, crop_w = face_crop.shape[:2]

        # --------------------------------------------------
        # STEP 5: RESIZE WHILE PRESERVING ASPECT RATIO
        # --------------------------------------------------

        scale = (
            self.mesh_input_size
            / max(crop_w, crop_h)
        )

        new_w = max(
            1,
            int(round(crop_w * scale)),
        )

        new_h = max(
            1,
            int(round(crop_h * scale)),
        )

        interpolation = (
            cv2.INTER_AREA
            if scale < 1.0
            else cv2.INTER_CUBIC
        )

        try:
            resized_crop = cv2.resize(
                face_crop,
                (new_w, new_h),
                interpolation=interpolation,
            )

            # --------------------------------------------------
            # STEP 6: CONVERT BGR TO RGB
            # --------------------------------------------------

            rgb_crop = cv2.cvtColor(
                resized_crop,
                cv2.COLOR_BGR2RGB,
            )

            # --------------------------------------------------
            # STEP 7: RUN MEDIAPIPE FACE MESH
            # --------------------------------------------------

            results = self.face_mesh.process(rgb_crop)

        except Exception as exc:
            self.stats["face_mesh_errors"] += 1

            if self.debug:
                print(
                    "[ROI ERROR] Face Mesh processing failed: "
                    f"{type(exc).__name__}: {exc}"
                )

            return self._return_empty_frame(empty)

        # --------------------------------------------------
        # STEP 8: CHECK LANDMARK RESULTS
        # --------------------------------------------------

        if not results.multi_face_landmarks:
            self.stats["no_landmarks"] += 1

            if self.debug:
                print(
                    "[ROI DEBUG] No landmarks detected "
                    f"in padded crop {crop_w}x{crop_h}."
                )

            return self._return_empty_frame(empty)

        # --------------------------------------------------
        # STEP 9: MAP LANDMARKS TO ORIGINAL FRAME
        # --------------------------------------------------

        mapped_faces = []

        for face in results.multi_face_landmarks:
            mapped_landmarks = []

            for point in face.landmark:
                # Face Mesh coordinates are normalized relative
                # to the resized crop. Since resizing preserves
                # aspect ratio, these coordinates can be mapped
                # back to the original crop dimensions.

                original_x = (
                    crop_x1 + point.x * crop_w
                ) / frame_w

                original_y = (
                    crop_y1 + point.y * crop_h
                ) / frame_h

                mapped_landmarks.append(
                    SimpleNamespace(
                        x=original_x,
                        y=original_y,
                    )
                )

            mapped_faces.append(
                SimpleNamespace(
                    landmark=mapped_landmarks,
                )
            )

        mapped_results = SimpleNamespace(
            multi_face_landmarks=mapped_faces,
        )

        # --------------------------------------------------
        # STEP 10: SELECT THE MATCHING FACE
        # --------------------------------------------------

        landmarks, match_iou = self._select_face_landmarks(
            mapped_results,
            frame_w,
            frame_h,
            target_box,
        )

        if landmarks is None:
            self.stats["no_face_match"] += 1

            if self.debug:
                print(
                    "[ROI DEBUG] No matching face found. "
                    f"Best IoU={match_iou:.4f}"
                )

            return self._return_empty_frame(empty)

        self.stats["face_selected"] += 1

        # --------------------------------------------------
        # STEP 11: CREATE EACH FACIAL ROI
        # --------------------------------------------------

        def make_roi(
            indices: List[int],
            roi_name: str,
        ) -> Dict[str, Optional[np.ndarray]]:
            """
            Create one rectangular crop and polygon mask
            from the requested Face Mesh landmarks.
            """

            failures = self.stats["roi_failures"][roi_name]

            # ----------------------------------------------
            # Validate landmark indexes
            # ----------------------------------------------

            if (
                not indices
                or min(indices) < 0
                or max(indices) >= len(landmarks)
            ):
                failures["invalid_indices"] += 1

                return self._empty_roi()

            # ----------------------------------------------
            # Convert selected landmarks to pixel coordinates
            # ----------------------------------------------

            points = []

            for index in indices:
                point = landmarks[index]

                if (
                    not np.isfinite(point.x)
                    or not np.isfinite(point.y)
                ):
                    failures["invalid_landmarks"] += 1

                    return self._empty_roi()

                points.append([
                    int(round(point.x * frame_w)),
                    int(round(point.y * frame_h)),
                ])

            points = np.asarray(
                points,
                dtype=np.int32,
            )

            # Keep all polygon points inside the original frame.
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

            # ----------------------------------------------
            # Validate polygon points
            # ----------------------------------------------

            if len(np.unique(points, axis=0)) < 3:
                failures["too_few_points"] += 1

                if self.debug:
                    print(
                        f"[ROI DEBUG] {roi_name}: "
                        "fewer than 3 unique polygon points."
                    )

                return self._empty_roi()

            # ----------------------------------------------
            # Validate polygon area
            # ----------------------------------------------

            polygon_area = abs(
                float(cv2.contourArea(points))
            )

            if polygon_area < self.min_polygon_area:
                failures["polygon_too_small"] += 1

                if self.debug:
                    print(
                        f"[ROI DEBUG] {roi_name}: "
                        f"polygon area={polygon_area:.2f} px^2; "
                        f"minimum={self.min_polygon_area:.2f}; "
                        f"points={points.tolist()}"
                    )

                return self._empty_roi()

            # ----------------------------------------------
            # Calculate rectangular crop
            # ----------------------------------------------

            rx, ry, rw, rh = cv2.boundingRect(points)

            if rw <= 0 or rh <= 0:
                failures["invalid_crop"] += 1

                return self._empty_roi()

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

            # ----------------------------------------------
            # Create polygon mask
            # ----------------------------------------------

            mask = np.zeros(
                crop.shape[:2],
                dtype=np.uint8,
            )

            local_points = (
                points
                - np.array(
                    [rx, ry],
                    dtype=np.int32,
                )
            )

            cv2.fillPoly(
                mask,
                [local_points],
                255,
            )

            if mask.shape != crop.shape[:2]:
                failures["invalid_crop"] += 1

                return self._empty_roi()

            # ----------------------------------------------
            # Validate selected pixels
            # ----------------------------------------------

            mask_pixels = int(
                cv2.countNonZero(mask)
            )

            if mask_pixels < self.min_skin_pixels:
                failures["too_few_mask_pixels"] += 1

                if self.debug:
                    print(
                        f"[ROI DEBUG] {roi_name}: "
                        f"mask has {mask_pixels} pixels; "
                        f"minimum={self.min_skin_pixels}."
                    )

                return self._empty_roi()

            return {
                "crop": crop,
                "mask": mask,
            }

        # --------------------------------------------------
        # STEP 12: EXTRACT FOREHEAD AND CHEEKS
        # --------------------------------------------------

        roi_results = {
            "forehead": make_roi(
                self.FOREHEAD_INDICES,
                "forehead",
            ),

            "left_cheek": make_roi(
                self.LEFT_CHEEK_INDICES,
                "left_cheek",
            ),

            "right_cheek": make_roi(
                self.RIGHT_CHEEK_INDICES,
                "right_cheek",
            ),
        }

        # --------------------------------------------------
        # STEP 13: UPDATE ROI STATISTICS
        # --------------------------------------------------

        valid_roi_count = 0

        for name in self.ROI_NAMES:
            roi = roi_results[name]

            if (
                roi["crop"] is not None
                and roi["mask"] is not None
            ):
                self.stats[f"{name}_valid"] += 1

                valid_roi_count += 1

        if valid_roi_count > 0:
            self.stats["frames_with_valid_roi"] += 1
        else:
            self.stats["frames_without_valid_roi"] += 1

        if self.debug:
            print(
                f"[ROI DEBUG] Valid ROIs: "
                f"{valid_roi_count}/{len(self.ROI_NAMES)}"
            )

        return roi_results

    # ======================================================
    # RESOURCE CLEANUP
    # ======================================================

    def close(self) -> None:
        """
        Release MediaPipe Face Mesh resources.

        Calling close() more than once is safe.
        """

        if not self._closed:
            self.face_mesh.close()
            self._closed = True

    def __enter__(self):
        """
        Support use with a context manager.
        """

        if self._closed:
            raise RuntimeError(
                "Cannot reuse a closed ROIExtractor."
            )

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):
        """
        Automatically release Face Mesh resources.
        """

        self.close()

        return False
