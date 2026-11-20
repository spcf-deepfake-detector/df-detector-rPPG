
"""
FALSEFACET - Face Tracking
==========================

PURPOSE:
    Select one target face and follow it across consecutive
    video frames using bounding-box geometry and motion.

PIPELINE POSITION:
    extract_frame.py
        |
        v
    face_detection.py
        |
        v
    face_tracking.py
        |
        v
    roi.py
        |
        v
    rgb_extraction.py
        |
        v
    rppg.py

RESPONSIBILITIES:
    1. Initialize the target using the largest detected face.
    2. Match the target across consecutive frames.
    3. Support multiple face detections per frame.
    4. Estimate target movement using motion prediction.
    5. Smooth bounding-box movement.
    6. Handle temporary target disappearance.
    7. Attempt to reacquire the target.
    8. Avoid switching to another person unnecessarily.
    9. Distinguish actual detections from predicted positions.
    10. Report tracking statistics.

IMPORTANT:
    This is a geometric tracker, not facial recognition.
    It cannot guarantee identity when people cross paths.

    Predicted positions are used only for matching.
    They must NEVER be treated as valid RGB measurements.

OUTPUT:
    bbox:
        Smoothed bounding box for tracking and cropping.

    detected_bbox:
        Original bounding box from the face detector.

    detected:
        True only when an actual detection is matched.

    track_valid:
        True only when an actual target detection is accepted.
"""

import math

from numbers import Integral, Real
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


class FaceTracker:
    """
    Multi-face-aware, single-target geometric tracker.

    Supports:
        - Single-frame processing with track_frame().
        - Complete-sequence processing with track_faces().

    Both methods use the same tracking algorithm.
    """

    def __init__(
        self,
        iou_threshold: float = 0.20,
        smoothing_alpha: float = 0.65,
        fast_motion_threshold: float = 0.15,
        max_center_distance: float = 1.5,
        max_size_ratio: float = 2.5,
        max_missed_frames: int = 5,
        max_prediction_frames: int = 5,
        max_target_lost_frames: int = 30,
        max_match_score: float = 0.90,
        debug: bool = False,
    ):

        # =================================================
        # 1. VALIDATE CONFIGURATION
        # =================================================

        self._validate_range(
            "iou_threshold",
            iou_threshold,
            0.0,
            1.0,
        )

        self._validate_range(
            "smoothing_alpha",
            smoothing_alpha,
            0.0,
            1.0,
        )

        self._validate_range(
            "fast_motion_threshold",
            fast_motion_threshold,
            0.0,
            10.0,
        )

        self._validate_range(
            "max_center_distance",
            max_center_distance,
            0.1,
            10.0,
        )

        self._validate_range(
            "max_size_ratio",
            max_size_ratio,
            1.0,
            10.0,
        )

        self._validate_range(
            "max_match_score",
            max_match_score,
            0.0,
            1.0,
        )

        for name, value in (
            ("max_missed_frames", max_missed_frames),
            ("max_prediction_frames", max_prediction_frames),
            ("max_target_lost_frames", max_target_lost_frames),
        ):

            if (
                isinstance(value, bool)
                or not isinstance(value, Integral)
                or value < 0
            ):
                raise ValueError(
                    f"{name} must be an integer >= 0."
                )

        if max_missed_frames < 1:

            raise ValueError(
                "max_missed_frames must be at least 1."
            )

        if max_target_lost_frames < max_missed_frames:

            raise ValueError(
                "max_target_lost_frames must be >= "
                "max_missed_frames."
            )

        if not isinstance(debug, bool):

            raise TypeError(
                "debug must be a boolean."
            )

        # =================================================
        # 2. SAVE CONFIGURATION
        # =================================================

        self.iou_threshold = float(iou_threshold)

        self.smoothing_alpha = float(smoothing_alpha)

        self.fast_motion_threshold = float(
            fast_motion_threshold
        )

        self.max_center_distance = float(
            max_center_distance
        )

        self.max_size_ratio = float(max_size_ratio)

        self.max_missed_frames = int(
            max_missed_frames
        )

        self.max_prediction_frames = int(
            max_prediction_frames
        )

        self.max_target_lost_frames = int(
            max_target_lost_frames
        )

        self.max_match_score = float(
            max_match_score
        )

        self.debug = debug

        self.reset()

    # =====================================================
    # 3. VALIDATION
    # =====================================================

    @staticmethod
    def _validate_range(
        name: str,
        value: Any,
        minimum: float,
        maximum: float,
    ) -> None:
        """
        Validate a numeric configuration value.
        """

        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or not minimum <= value <= maximum
        ):

            raise ValueError(
                f"{name} must be between "
                f"{minimum} and {maximum}."
            )

    @staticmethod
    def _valid_bbox(bbox: Any) -> bool:
        """
        Check whether a bounding box is valid.

        Expected format:
            [x, y, width, height]
        """

        if not isinstance(bbox, (list, tuple)):
            return False

        if len(bbox) != 4:
            return False

        for value in bbox:

            if (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
            ):
                return False

        _, _, width, height = bbox

        return width > 0 and height > 0

    @staticmethod
    def _validate_confidence(value: Any) -> float:
        """
        Validate detector confidence.

        Invalid confidence values are converted to zero.
        """

        if (
            isinstance(value, bool)
            or not isinstance(value, Real)
            or not math.isfinite(value)
            or not 0.0 <= value <= 1.0
        ):

            return 0.0

        return float(value)

    @staticmethod
    def _is_detected(value: Any) -> bool:
        """
        Accept Python and NumPy boolean values.

        Other values are not treated as detections.
        """

        if isinstance(value, bool):
            return value

        if type(value).__name__ == "bool_":
            return bool(value)

        return False

    # =====================================================
    # 4. RESET TRACKING STATE
    # =====================================================

    def reset(self) -> None:
        """
        Reset the current tracking session.

        This is called automatically by track_faces().
        It should not be called between frames when using
        track_frame().
        """

        self.last_processed_frame_idx = -1

        # Most recently accepted target bounding box.
        self.last_valid_bbox: Optional[List[float]] = None

        # Previous accepted target bounding box.
        self.previous_valid_bbox: Optional[List[float]] = None

        # Frame index of the most recent accepted box.
        self.last_valid_frame: Optional[int] = None

        # Frame index of the previous accepted box.
        self.previous_valid_frame: Optional[int] = None

        # Tracking state.
        self.missed_frames = 0

        self.was_tracking = False

        self.target_lost = False

        # Statistics.
        self.accepted_frames = 0

        self.rejected_frames = 0

        self.reacquired_frames = 0

        self.iou_values: List[float] = []

        self.displacements: List[float] = []

        self.match_scores: List[float] = []

    # =====================================================
    # 5. GEOMETRY
    # =====================================================

    @staticmethod
    def _center(
        bbox: List[float],
    ) -> Tuple[float, float]:
        """
        Calculate the center of a bounding box.
        """

        x, y, width, height = bbox

        return (
            x + width / 2.0,
            y + height / 2.0,
        )

    @staticmethod
    def _area(
        bbox: List[float],
    ) -> float:
        """
        Calculate bounding-box area.
        """

        return bbox[2] * bbox[3]

    @staticmethod
    def compute_iou(
        boxA: List[float],
        boxB: List[float],
    ) -> float:
        """
        Calculate Intersection over Union.

        IoU measures bounding-box overlap.
        It does not establish facial identity.
        """

        if not FaceTracker._valid_bbox(boxA):
            return 0.0

        if not FaceTracker._valid_bbox(boxB):
            return 0.0

        ax, ay, aw, ah = map(float, boxA)

        bx, by, bw, bh = map(float, boxB)

        x_left = max(ax, bx)

        y_top = max(ay, by)

        x_right = min(
            ax + aw,
            bx + bw,
        )

        y_bottom = min(
            ay + ah,
            by + bh,
        )

        intersection_width = max(
            0.0,
            x_right - x_left,
        )

        intersection_height = max(
            0.0,
            y_bottom - y_top,
        )

        intersection_area = (
            intersection_width * intersection_height
        )

        union_area = (
            aw * ah
            + bw * bh
            - intersection_area
        )

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area

    def _normalized_displacement(
        self,
        old_bbox: List[float],
        new_bbox: List[float],
    ) -> float:
        """
        Calculate center movement relative to the old
        bounding-box diagonal.

        The result is independent of image resolution.
        """

        old_center = self._center(old_bbox)

        new_center = self._center(new_bbox)

        dx = new_center[0] - old_center[0]

        dy = new_center[1] - old_center[1]

        diagonal = math.hypot(
            old_bbox[2],
            old_bbox[3],
        )

        if diagonal <= 0:
            return 0.0

        return math.hypot(dx, dy) / diagonal

    # =====================================================
    # 6. EXTRACT DETECTION CANDIDATES
    # =====================================================

    def _get_candidates(
        self,
        record: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Extract valid actual face detections.

        Supports the multi-face format:

            {
                "faces": [
                    {
                        "bbox": [x, y, width, height],
                        "confidence": 0.95,
                        "detected": True
                    }
                ]
            }

        Also supports legacy single-face records.

        Original face-list indexes are preserved.
        """

        candidates = []

        # ---------------------------------------------
        # MULTI-FACE DETECTOR FORMAT
        # ---------------------------------------------

        if "faces" in record:

            raw_faces = record.get("faces")

            if not isinstance(raw_faces, list):
                return candidates

            for index, face in enumerate(raw_faces):

                if not isinstance(face, dict):
                    continue

                bbox = face.get("bbox")

                if not self._valid_bbox(bbox):
                    continue

                # Only actual detector results are eligible.
                if not self._is_detected(
                    face.get("detected", False)
                ):
                    continue

                candidates.append({
                    "bbox": [
                        float(value)
                        for value in bbox
                    ],
                    "confidence": self._validate_confidence(
                        face.get("confidence", 0.0)
                    ),
                    "face_index": index,
                })

        # ---------------------------------------------
        # LEGACY SINGLE-FACE FORMAT
        # ---------------------------------------------

        else:

            bbox = record.get("bbox")

            if (
                self._valid_bbox(bbox)
                and self._is_detected(
                    record.get("detected", False)
                )
            ):

                candidates.append({
                    "bbox": [
                        float(value)
                        for value in bbox
                    ],
                    "confidence": self._validate_confidence(
                        record.get("confidence", 0.0)
                    ),
                    "face_index": 0,
                })

        return candidates

    # =====================================================
    # 7. MOTION PREDICTION
    # =====================================================

    def _predict_bbox(
        self,
        current_frame: int,
    ) -> Optional[List[float]]:
        """
        Estimate the target bounding box at the current frame.

        Constant-velocity prediction is applied to:
            - Center X.
            - Center Y.
            - Width.
            - Height.

        Prediction is used only for candidate matching.

        It is NOT an actual face detection.
        """

        if self.last_valid_bbox is None:
            return None

        last = self.last_valid_bbox

        if (
            self.previous_valid_bbox is None
            or self.last_valid_frame is None
            or self.previous_valid_frame is None
            or self.max_prediction_frames == 0
        ):

            return last.copy()

        elapsed = (
            self.last_valid_frame
            - self.previous_valid_frame
        )

        if elapsed <= 0:
            return last.copy()

        missing_steps = max(
            0,
            current_frame - self.last_valid_frame,
        )

        prediction_steps = min(
            missing_steps,
            self.max_prediction_frames,
        )

        if prediction_steps == 0:
            return last.copy()

        previous = self.previous_valid_bbox

        # ---------------------------------------------
        # PREDICT CENTER MOVEMENT
        # ---------------------------------------------

        last_center = self._center(last)

        previous_center = self._center(previous)

        velocity_x = (
            last_center[0] - previous_center[0]
        ) / elapsed

        velocity_y = (
            last_center[1] - previous_center[1]
        ) / elapsed

        predicted_center_x = (
            last_center[0]
            + velocity_x * prediction_steps
        )

        predicted_center_y = (
            last_center[1]
            + velocity_y * prediction_steps
        )

        # ---------------------------------------------
        # PREDICT SIZE CHANGE
        # ---------------------------------------------

        width_velocity = (
            last[2] - previous[2]
        ) / elapsed

        height_velocity = (
            last[3] - previous[3]
        ) / elapsed

        predicted_width = max(
            1.0,
            last[2] + width_velocity * prediction_steps,
        )

        predicted_height = max(
            1.0,
            last[3] + height_velocity * prediction_steps,
        )

        predicted = [
            predicted_center_x - predicted_width / 2.0,
            predicted_center_y - predicted_height / 2.0,
            predicted_width,
            predicted_height,
        ]

        if not self._valid_bbox(predicted):
            return last.copy()

        return predicted

    # =====================================================
    # 8. MATCH DETECTION TO TARGET
    # =====================================================

    def _match_candidate(
        self,
        predicted_bbox: List[float],
        candidate_bbox: List[float],
        fast_motion: bool = False,
        reacquiring: bool = False,
        extended_reacquiring: bool = False,
    ) -> Optional[Dict[str, float]]:
        """
        Calculate whether a candidate plausibly belongs
        to the previously selected target.

        Lower scores indicate better matches.
        """

        predicted_center = self._center(
            predicted_bbox
        )

        candidate_center = self._center(
            candidate_bbox
        )

        dx = (
            candidate_center[0]
            - predicted_center[0]
        )

        dy = (
            candidate_center[1]
            - predicted_center[1]
        )

        diagonal = math.hypot(
            predicted_bbox[2],
            predicted_bbox[3],
        )

        if diagonal <= 0:
            return None

        center_distance = (
            math.hypot(dx, dy) / diagonal
        )

        # ---------------------------------------------
        # SIZE SIMILARITY
        # ---------------------------------------------

        width_ratio = max(
            predicted_bbox[2] / candidate_bbox[2],
            candidate_bbox[2] / predicted_bbox[2],
        )

        height_ratio = max(
            predicted_bbox[3] / candidate_bbox[3],
            candidate_bbox[3] / predicted_bbox[3],
        )

        size_ratio = max(
            width_ratio,
            height_ratio,
        )

        # ---------------------------------------------
        # IOU
        # ---------------------------------------------

        iou = self.compute_iou(
            predicted_bbox,
            candidate_bbox,
        )

        # ---------------------------------------------
        # DISTANCE GATES
        # ---------------------------------------------

        allowed_distance = self.max_center_distance

        if fast_motion:
            allowed_distance *= 1.5

        if reacquiring:

            allowed_distance = min(
                allowed_distance,
                1.0,
            )

        if extended_reacquiring:

            allowed_distance = min(
                allowed_distance,
                0.75,
            )

        if center_distance > allowed_distance:
            return None

        if size_ratio > self.max_size_ratio:
            return None

        # ---------------------------------------------
        # IOU GATE
        # ---------------------------------------------

        if (
            iou < self.iou_threshold
            and center_distance > allowed_distance * 0.5
        ):

            return None

        # ---------------------------------------------
        # MATCHING SCORE
        # ---------------------------------------------

        distance_score = min(
            center_distance / allowed_distance,
            1.0,
        )

        if self.max_size_ratio > 1.0:

            size_score = min(
                math.log(max(size_ratio, 1.0))
                / math.log(self.max_size_ratio),
                1.0,
            )

        else:
            size_score = 0.0

        iou_score = 1.0 - iou

        score = (
            0.50 * distance_score
            + 0.30 * size_score
            + 0.20 * iou_score
        )

        return {
            "score": score,
            "iou": iou,
            "center_distance": center_distance,
            "size_ratio": size_ratio,
        }

    def _select_matching_candidate(
        self,
        candidates: List[Dict[str, Any]],
        predicted_bbox: List[float],
        fast_motion: bool,
        reacquiring: bool,
        extended_reacquiring: bool,
    ) -> Optional[Dict[str, Any]]:
        """
        Select the candidate with the lowest acceptable
        matching score.

        Confidence is used only as a small tie-breaker.
        """

        best_candidate = None

        best_score = float("inf")

        for candidate in candidates:

            result = self._match_candidate(
                predicted_bbox=predicted_bbox,
                candidate_bbox=candidate["bbox"],
                fast_motion=fast_motion,
                reacquiring=reacquiring,
                extended_reacquiring=extended_reacquiring,
            )

            if result is None:
                continue

            adjusted_score = (
                result["score"]
                - candidate["confidence"] * 0.02
            )

            if adjusted_score > self.max_match_score:
                continue

            if adjusted_score < best_score:

                best_score = adjusted_score

                best_candidate = {
                    **candidate,
                    **result,
                    "adjusted_score": adjusted_score,
                }

        return best_candidate

    # =====================================================
    # 9. BOUNDING-BOX SMOOTHING
    # =====================================================

    def _smooth_bbox(
        self,
        previous_bbox: List[float],
        current_bbox: List[float],
    ) -> List[float]:
        """
        Smooth ordinary movement while responding faster
        to larger movements.

        Higher alpha means more weight is given to the
        current detector bounding box.
        """

        displacement = self._normalized_displacement(
            previous_bbox,
            current_bbox,
        )

        if displacement >= self.fast_motion_threshold:

            alpha = max(
                self.smoothing_alpha,
                0.85,
            )

        else:

            alpha = self.smoothing_alpha

        smoothed = [
            previous_bbox[index]
            + alpha * (
                current_bbox[index]
                - previous_bbox[index]
            )
            for index in range(4)
        ]

        if not self._valid_bbox(smoothed):
            return current_bbox.copy()

        return smoothed

    # =====================================================
    # 10. UPDATE TARGET POSITION
    # =====================================================

    def _update_position(
        self,
        current_bbox: List[float],
        current_frame: int,
    ) -> List[float]:
        """
        Update the accepted target position and motion history.

        Only accepted actual detections update this state.
        """

        if self.last_valid_bbox is not None:

            displacement = self._normalized_displacement(
                self.last_valid_bbox,
                current_bbox,
            )

            self.displacements.append(
                displacement
            )

            smoothed_bbox = self._smooth_bbox(
                self.last_valid_bbox,
                current_bbox,
            )

            self.previous_valid_bbox = (
                self.last_valid_bbox.copy()
            )

            self.previous_valid_frame = (
                self.last_valid_frame
            )

        else:

            smoothed_bbox = current_bbox.copy()

        self.last_valid_bbox = smoothed_bbox.copy()

        self.last_valid_frame = current_frame

        return smoothed_bbox

    # =====================================================
    # 11. TRACKING STATUS
    # =====================================================

    @staticmethod
    def _set_status(
        record: Dict[str, Any],
        status: str,
        valid: bool,
    ) -> None:
        """
        Add tracking status fields to a frame record.

        track_valid is the field that downstream processing
        should use to decide whether RGB extraction is allowed.
        """

        record["tracking_status"] = status

        record["status"] = status

        record["track_valid"] = valid

    # =====================================================
    # 12. HANDLE MISSING TARGET
    # =====================================================

    def _handle_missing(
        self,
        record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Mark the target as missing.

        A predicted position is deliberately NOT returned
        as bbox or detected_bbox.

        This prevents predicted boxes from being mistaken
        for real face measurements by RGB extraction.
        """

        self.missed_frames += 1

        self.rejected_frames += 1

        # No valid face measurement.
        record["bbox"] = [
            0.0,
            0.0,
            0.0,
            0.0,
        ]

        record["detected_bbox"] = None

        record["confidence"] = 0.0

        record["detected"] = False

        record["selected_face_index"] = None

        record["tracking_iou"] = None

        record["predicted_iou"] = None

        record["motion_displacement"] = None

        record["match_score"] = None

        # Permanently stop this tracking session after
        # the configured target-loss limit is exceeded.
        if self.missed_frames > self.max_target_lost_frames:

            self.target_lost = True

        if self.target_lost:

            status = "target_lost"

        elif self.missed_frames > self.max_missed_frames:

            status = "reacquiring"

        else:

            status = "missing"

        self._set_status(
            record,
            status,
            False,
        )

        if self.debug:

            print(
                f"[TRACKER] Frame {record['frame_idx']}: "
                f"{status} | "
                f"missed={self.missed_frames}"
            )

        return record

    # =====================================================
    # 13. INITIALIZE TARGET
    # =====================================================

    def _initialize_target(
        self,
        record: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        current_frame: int,
    ) -> Dict[str, Any]:
        """
        Initialize tracking using the largest detected face.

        This happens once per tracking session, unless the
        tracker has not yet found any valid target.
        """

        largest_face = max(
            candidates,
            key=lambda candidate: self._area(
                candidate["bbox"]
            ),
        )

        bbox = largest_face["bbox"]

        self.last_valid_bbox = bbox.copy()

        self.previous_valid_bbox = None

        self.last_valid_frame = current_frame

        self.previous_valid_frame = None

        self.missed_frames = 0

        self.was_tracking = True

        record["bbox"] = bbox.copy()

        record["detected_bbox"] = bbox.copy()

        record["confidence"] = (
            largest_face["confidence"]
        )

        record["detected"] = True

        record["selected_face_index"] = (
            largest_face["face_index"]
        )

        record["tracking_iou"] = None

        record["predicted_iou"] = None

        record["motion_displacement"] = 0.0

        record["match_score"] = 0.0

        self._set_status(
            record,
            "initialized",
            True,
        )

        self.accepted_frames += 1

        if self.debug:

            print(
                f"[TRACKER] Frame {current_frame}: "
                f"initialized with largest face | "
                f"area={self._area(bbox):.1f} | "
                f"face_index={largest_face['face_index']}"
            )

        return record

    # =====================================================
    # 14. PROCESS ONE FRAME
    # =====================================================

    def track_frame(
        self,
        detection: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process one frame while preserving tracking state.

        Use this method for single-pass video processing.

        IMPORTANT:
            Do not call reset() between consecutive frames.
        """

        if not isinstance(detection, dict):

            raise TypeError(
                "detection must be a dictionary."
            )

        record = dict(detection)

        # ---------------------------------------------
        # 14.1 FRAME INDEX
        # ---------------------------------------------

        frame_idx = record.get(
            "frame_idx",
            self.last_processed_frame_idx + 1,
        )

        if (
            isinstance(frame_idx, bool)
            or not isinstance(frame_idx, Integral)
            or frame_idx < 0
            or frame_idx <= self.last_processed_frame_idx
        ):

            raise ValueError(
                f"Invalid or non-increasing frame index: "
                f"{frame_idx}"
            )

        frame_idx = int(frame_idx)

        record["frame_idx"] = frame_idx

        self.last_processed_frame_idx = frame_idx

        # ---------------------------------------------
        # 14.2 EXTRACT ACTUAL DETECTION CANDIDATES
        # ---------------------------------------------

        candidates = self._get_candidates(
            record
        )

        record["candidate_count"] = len(
            candidates
        )

        record["tracking_iou"] = None

        record["predicted_iou"] = None

        record["motion_displacement"] = None

        record["match_score"] = None

        record["selected_face_index"] = None

        record["detected_bbox"] = None

        # ---------------------------------------------
        # 14.3 TARGET ALREADY LOST
        # ---------------------------------------------

        if self.target_lost:

            return self._handle_missing(
                record
            )

        # ---------------------------------------------
        # 14.4 INITIALIZE TARGET
        # ---------------------------------------------

        if self.last_valid_bbox is None:

            if not candidates:

                return self._handle_missing(
                    record
                )

            return self._initialize_target(
                record,
                candidates,
                frame_idx,
            )

        # ---------------------------------------------
        # 14.5 NO DETECTION CANDIDATES
        # ---------------------------------------------

        if not candidates:

            return self._handle_missing(
                record
            )

        # ---------------------------------------------
        # 14.6 PREDICT TARGET POSITION
        # ---------------------------------------------

        predicted_bbox = self._predict_bbox(
            frame_idx
        )

        if predicted_bbox is None:

            return self._handle_missing(
                record
            )

        # ---------------------------------------------
        # 14.7 DETERMINE FAST MOTION
        # ---------------------------------------------

        fast_motion = False

        if (
            self.previous_valid_bbox is not None
            and self.last_valid_bbox is not None
            and self.previous_valid_frame is not None
            and self.last_valid_frame is not None
        ):

            elapsed = (
                self.last_valid_frame
                - self.previous_valid_frame
            )

            if elapsed > 0:

                total_displacement = (
                    self._normalized_displacement(
                        self.previous_valid_bbox,
                        self.last_valid_bbox,
                    )
                )

                # Normalize movement by elapsed frames.
                # This avoids treating a multi-frame gap as
                # a single-frame movement.
                displacement_per_frame = (
                    total_displacement / elapsed
                )

                fast_motion = (
                    displacement_per_frame
                    >= self.fast_motion_threshold
                )

        # ---------------------------------------------
        # 14.8 REACQUISITION STATE
        # ---------------------------------------------

        reacquiring = (
            self.missed_frames > 0
        )

        extended_reacquiring = (
            self.missed_frames > self.max_missed_frames
        )

        # ---------------------------------------------
        # 14.9 FIND THE ORIGINAL TARGET
        # ---------------------------------------------

        selected = self._select_matching_candidate(
            candidates=candidates,
            predicted_bbox=predicted_bbox,
            fast_motion=fast_motion,
            reacquiring=reacquiring,
            extended_reacquiring=extended_reacquiring,
        )

        # ---------------------------------------------
        # 14.10 NO ACCEPTABLE MATCH
        # ---------------------------------------------

        if selected is None:

            return self._handle_missing(
                record
            )

        # ---------------------------------------------
        # 14.11 ACCEPT MATCH
        # ---------------------------------------------

        current_bbox = selected["bbox"]

        old_missed_frames = self.missed_frames

        previous_bbox = self.last_valid_bbox.copy()

        smoothed_bbox = self._update_position(
            current_bbox=current_bbox,
            current_frame=frame_idx,
        )

        displacement = self._normalized_displacement(
            previous_bbox,
            current_bbox,
        )

        # Smoothed coordinates are used for tracking
        # and cropping.
        record["bbox"] = smoothed_bbox.copy()

        # Raw detector coordinates remain available.
        record["detected_bbox"] = current_bbox.copy()

        record["confidence"] = selected["confidence"]

        record["detected"] = True

        record["selected_face_index"] = (
            selected["face_index"]
        )

        record["tracking_iou"] = selected["iou"]

        record["predicted_iou"] = self.compute_iou(
            predicted_bbox,
            current_bbox,
        )

        record["motion_displacement"] = displacement

        record["match_score"] = selected["score"]

        self.iou_values.append(
            selected["iou"]
        )

        self.match_scores.append(
            selected["score"]
        )

        self.missed_frames = 0

        if old_missed_frames > 0:

            status = "reacquired"

            self.reacquired_frames += 1

        else:

            status = "matched"

        self._set_status(
            record,
            status,
            True,
        )

        self.accepted_frames += 1

        if self.debug:

            print(
                f"[TRACKER] Frame {frame_idx}: "
                f"{status} | "
                f"face_index={selected['face_index']} | "
                f"IoU={selected['iou']:.3f} | "
                f"distance={selected['center_distance']:.3f} | "
                f"size_ratio={selected['size_ratio']:.3f} | "
                f"score={selected['score']:.3f}"
            )

        return record

    # =====================================================
    # 15. PROCESS COMPLETE SEQUENCE
    # =====================================================

    def track_faces(
        self,
        raw_detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Track a complete sequence of detection records.

        Uses track_frame() internally so batch processing
        and single-pass processing share the same algorithm.

        A fresh tracking session is started automatically.
        """

        if not isinstance(raw_detections, list):

            raise TypeError(
                "raw_detections must be a list."
            )

        self.reset()

        tracked = []

        for frame_position, detection in enumerate(
            raw_detections
        ):

            if not isinstance(detection, dict):

                raise TypeError(
                    f"Detection at position {frame_position} "
                    "must be a dictionary."
                )

            record = dict(detection)

            record.setdefault(
                "frame_idx",
                frame_position,
            )

            tracked_record = self.track_frame(
                record
            )

            tracked.append(
                tracked_record
            )

        self._print_summary(
            tracked
        )

        return tracked

    # =====================================================
    # 16. TRACKING SUMMARY
    # =====================================================

    def _print_summary(
        self,
        tracked: List[Dict[str, Any]],
    ) -> None:
        """
        Print tracking statistics for a complete sequence.
        """

        status_counts = Counter(
            record.get(
                "tracking_status",
                "unknown",
            )
            for record in tracked
        )

        total_frames = len(tracked)

        print(
            "\n========== FACE TRACKING SUMMARY =========="
        )

        print(
            f"Total frames: {total_frames}"
        )

        print(
            f"Accepted frames: {self.accepted_frames}"
        )

        print(
            f"Rejected frames: {self.rejected_frames}"
        )

        print(
            f"Reacquired frames: {self.reacquired_frames}"
        )

        print(
            "\nTracking statuses:"
        )

        for status in (
            "initialized",
            "matched",
            "reacquired",
            "missing",
            "reacquiring",
            "target_lost",
        ):

            print(
                f"  {status}: "
                f"{status_counts.get(status, 0)}"
            )

        if total_frames:

            tracking_rate = (
                self.accepted_frames / total_frames
            ) * 100.0

            print(
                f"\nTracking-valid rate: "
                f"{tracking_rate:.2f}%"
            )

        if self.iou_values:

            print(
                "\nIoU diagnostics:"
            )

            print(
                f"  Comparisons: {len(self.iou_values)}"
            )

            print(
                f"  Minimum IoU: "
                f"{min(self.iou_values):.3f}"
            )

            print(
                f"  Mean IoU: "
                f"{sum(self.iou_values) / len(self.iou_values):.3f}"
            )

            print(
                f"  Maximum IoU: "
                f"{max(self.iou_values):.3f}"
            )

        if self.displacements:

            print(
                "\nMotion diagnostics:"
            )

            print(
                f"  Mean normalized displacement: "
                f"{sum(self.displacements) / len(self.displacements):.3f}"
            )

            print(
                f"  Maximum normalized displacement: "
                f"{max(self.displacements):.3f}"
            )

        if self.match_scores:

            print(
                "\nMatching diagnostics:"
            )

            print(
                f"  Mean match score: "
                f"{sum(self.match_scores) / len(self.match_scores):.3f}"
            )

        print(
            "============================================\n"
        )