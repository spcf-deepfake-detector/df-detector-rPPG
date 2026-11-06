import math

from numbers import Real
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


class FaceTracker:
    """
    Multi-face-aware, single-target tracker for FalseFacet.

    Features:
        1. Selects the largest face during initialization.
        2. Tracks the initially selected person.
        3. Supports multiple detected faces.
        4. Handles changes in face position and size.
        5. Uses motion prediction for fast movement.
        6. Handles temporary face disappearance.
        7. Attempts to reacquire the original target.
        8. Avoids automatically switching to another person.
        9. Rejects unreliable candidate matches.
        10. Never treats predicted boxes as valid RGB measurements.

    Expected detector input:

        {
            "frame_idx": 0,
            "bbox": [x, y, width, height],
            "confidence": 0.95,
            "detected": True,
            "faces": [
                {
                    "bbox": [x, y, width, height],
                    "confidence": 0.95,
                    "detected": True
                }
            ]
        }

    Legacy single-face detection records are also supported.

    Important:
        This is a geometric tracker, not facial recognition.
        It cannot guarantee identity when people cross paths.
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

        # --------------------------------------------------
        # VALIDATE PARAMETERS
        # --------------------------------------------------

    

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
                or not isinstance(value, int)
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

        # --------------------------------------------------
        # SAVE CONFIGURATION
        # --------------------------------------------------

        self.iou_threshold = float(iou_threshold)

        self.smoothing_alpha = float(smoothing_alpha)

        self.fast_motion_threshold = float(
            fast_motion_threshold
        )

        self.max_center_distance = float(
            max_center_distance
        )

        self.max_size_ratio = float(max_size_ratio)

        self.max_missed_frames = max_missed_frames

        self.max_prediction_frames = max_prediction_frames

        self.max_target_lost_frames = max_target_lost_frames

        self.max_match_score = float(max_match_score)

        self.debug = debug

        self.reset()

    # ======================================================
    # VALIDATION
    # ======================================================

    @staticmethod
    def _validate_range(
        name: str,
        value: Any,
        minimum: float,
        maximum: float,
    ) -> None:

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

        if isinstance(value, bool):
            return value

        # Support NumPy boolean values.
        if type(value).__name__ == "bool_":
            return bool(value)

        return False

    # ======================================================
    # RESET
    # ======================================================

    def reset(self):
        """
        Reset tracking state and statistics.

        Called automatically whenever track_faces() starts.
        """

        self.last_processed_frame_idx = -1
        # Most recently accepted bounding box.
        self.last_valid_bbox: Optional[List[float]] = None

        # Previous accepted bounding box.
        self.previous_valid_bbox: Optional[List[float]] = None

        # Frame indexes of accepted boxes.
        self.last_valid_frame: Optional[int] = None

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

    # ======================================================
    # GEOMETRY
    # ======================================================

    @staticmethod
    def _center(
        bbox: List[float],
    ) -> Tuple[float, float]:

        x, y, width, height = bbox

        return (
            x + width / 2.0,
            y + height / 2.0,
        )

    @staticmethod
    def _area(
        bbox: List[float],
    ) -> float:

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

    # ======================================================
    # DETECTION CANDIDATES
    # ======================================================

    def _get_candidates(
        self,
        record: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """
        Extract valid face candidates.

        The original face-list indexes are preserved so that
        selected_face_index corresponds to detector output.
        """

        candidates = []

        # New multi-face detector.
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

                detected = face.get(
                    "detected",
                    True,
                )

                if not self._is_detected(detected):
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

        else:

            # Legacy single-face detector.
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

    # ======================================================
    # MOTION PREDICTION
    # ======================================================

    def _predict_bbox(
        self,
        current_frame: int,
    ) -> Optional[List[float]]:
        """
        Predict the target's bounding box.

        Constant-velocity prediction is used for:
            - Center X.
            - Center Y.
            - Width.
            - Height.

        Prediction is used only for candidate matching.

        It is NEVER used as a valid RGB measurement.
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

        # ----------------------------------------------
        # PREDICT CENTER
        # ----------------------------------------------

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

        # ----------------------------------------------
        # PREDICT SIZE
        # ----------------------------------------------

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

    # ======================================================
    # CANDIDATE MATCHING
    # ======================================================

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

        Lower match scores indicate better matches.
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

        # ----------------------------------------------
        # SIZE SIMILARITY
        # ----------------------------------------------

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

        # ----------------------------------------------
        # IOU
        # ----------------------------------------------

        iou = self.compute_iou(
            predicted_bbox,
            candidate_bbox,
        )

        # ----------------------------------------------
        # DISTANCE GATES
        # ----------------------------------------------

        allowed_distance = self.max_center_distance

        if fast_motion:
            allowed_distance *= 1.5

        if reacquiring:

            # Normal temporary-loss recovery.
            allowed_distance = min(
                allowed_distance,
                1.0,
            )

        if extended_reacquiring:

            # More conservative after several missing frames.
            allowed_distance = min(
                allowed_distance,
                0.75,
            )

        if center_distance > allowed_distance:
            return None

        if size_ratio > self.max_size_ratio:
            return None

        # ----------------------------------------------
        # IOU GATE
        # ----------------------------------------------

        # Low IoU can happen during fast movement.
        # However, a candidate with both low IoU and
        # a large center displacement is suspicious.

        if (
            iou < self.iou_threshold
            and center_distance > allowed_distance * 0.5
        ):

            return None

        # ----------------------------------------------
        # MATCHING SCORE
        # ----------------------------------------------

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
        Select the candidate with the lowest valid match score.
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

            # Confidence is only a small tie-breaker.
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

    # ======================================================
    # BOUNDING-BOX SMOOTHING
    # ======================================================

    def _smooth_bbox(
        self,
        previous_bbox: List[float],
        current_bbox: List[float],
    ) -> List[float]:
        """
        Smooth ordinary movement while responding faster
        to larger movements.
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

    # ======================================================
    # UPDATE TARGET POSITION
    # ======================================================

    def _update_position(
        self,
        current_bbox: List[float],
        current_frame: int,
    ) -> List[float]:
        """
        Update the accepted target position and motion history.
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

        self.last_valid_bbox = (
            smoothed_bbox.copy()
        )

        self.last_valid_frame = current_frame

        return smoothed_bbox

    # ======================================================
    # STATUS
    # ======================================================

    @staticmethod
    def _set_status(
        record: Dict[str, Any],
        status: str,
        valid: bool,
    ) -> None:

        record["tracking_status"] = status

        record["status"] = status

        record["track_valid"] = valid

    # ======================================================
    # HANDLE MISSING TARGET
    # ======================================================

    def _handle_missing(
        self,
        record: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Mark the target as missing.

        The predicted position is deliberately not returned
        as the actual target bounding box.
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

        record["confidence"] = 0.0

        record["detected"] = False

        record["selected_face_index"] = None

        record["tracking_iou"] = None

        record["predicted_iou"] = None

        record["motion_displacement"] = None

        record["match_score"] = None

        # Permanently lock this tracking session after
        # the original target has been lost too long.
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

    # ======================================================
    # INITIALIZE TARGET
    # ======================================================

    def _initialize_target(
        self,
        record: Dict[str, Any],
        candidates: List[Dict[str, Any]],
        current_frame: int,
    ) -> Dict[str, Any]:
        """
        Select the largest detected face.

        This happens only once per tracking session.
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

    # ======================================================
    # MAIN TRACKING FUNCTION
    # ======================================================
    
        # ======================================================
    # PROCESS ONE FRAME
    # ======================================================

    def track_frame(
        self,
        detection: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Process one frame while preserving tracking state.

        Unlike track_faces(), this method does not reset
        the tracker or print the final summary.

        This is used for single-pass video processing.
        """

        if not isinstance(detection, dict):

            raise TypeError(
                "detection must be a dictionary."
            )

        record = dict(detection)

        # ----------------------------------------------
        # FRAME INDEX
        # ----------------------------------------------

        frame_idx = record.get(
            "frame_idx",
            self.last_processed_frame_idx + 1,
        )

        if (
            isinstance(frame_idx, bool)
            or not isinstance(frame_idx, int)
            or frame_idx < 0
            or frame_idx <= self.last_processed_frame_idx
        ):

            frame_idx = self.last_processed_frame_idx + 1

        record["frame_idx"] = frame_idx

        self.last_processed_frame_idx = frame_idx

        # ----------------------------------------------
        # EXTRACT CANDIDATES
        # ----------------------------------------------

        candidates = self._get_candidates(record)

        record["candidate_count"] = len(candidates)

        record["tracking_iou"] = None

        record["predicted_iou"] = None

        record["motion_displacement"] = None

        record["match_score"] = None

        record["selected_face_index"] = None

        # ----------------------------------------------
        # CASE 1: TARGET ALREADY LOST
        # ----------------------------------------------

        if self.target_lost:

            return self._handle_missing(record)

        # ----------------------------------------------
        # CASE 2: INITIALIZE TARGET
        # ----------------------------------------------

        if self.last_valid_bbox is None:

            if not candidates:

                return self._handle_missing(record)

            return self._initialize_target(
                record,
                candidates,
                frame_idx,
            )

        # ----------------------------------------------
        # CASE 3: NO CANDIDATES
        # ----------------------------------------------

        if not candidates:

            return self._handle_missing(record)

        # ----------------------------------------------
        # CASE 4: PREDICT TARGET POSITION
        # ----------------------------------------------

        predicted_bbox = self._predict_bbox(frame_idx)

        if predicted_bbox is None:

            return self._handle_missing(record)

        # ----------------------------------------------
        # CASE 5: DETERMINE MOTION
        # ----------------------------------------------

        fast_motion = False

        if (
            self.previous_valid_bbox is not None
            and self.last_valid_bbox is not None
        ):

            recent_displacement = (
                self._normalized_displacement(
                    self.previous_valid_bbox,
                    self.last_valid_bbox,
                )
            )

            fast_motion = (
                recent_displacement
                >= self.fast_motion_threshold
            )

        reacquiring = self.missed_frames > 0

        extended_reacquiring = (
            self.missed_frames > self.max_missed_frames
        )

        # ----------------------------------------------
        # CASE 6: FIND ORIGINAL TARGET
        # ----------------------------------------------

        selected = self._select_matching_candidate(
            candidates=candidates,
            predicted_bbox=predicted_bbox,
            fast_motion=fast_motion,
            reacquiring=reacquiring,
            extended_reacquiring=extended_reacquiring,
        )

        # ----------------------------------------------
        # CASE 7: NO ACCEPTABLE MATCH
        # ----------------------------------------------

        if selected is None:

            return self._handle_missing(record)

        # ----------------------------------------------
        # CASE 8: ACCEPT MATCH
        # ----------------------------------------------

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

        record["bbox"] = smoothed_bbox

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

        self.iou_values.append(selected["iou"])

        self.match_scores.append(selected["score"])

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

    # ======================================================
    # MAIN TRACKING FUNCTION
    # ======================================================

    def track_faces(
        self,
        raw_detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Track a complete list of detection records.

        Uses track_frame() internally so batch processing
        and single-pass processing share the same algorithm.
        """

        if not isinstance(raw_detections, list):

            raise TypeError(
                "raw_detections must be a list."
            )

        # Start a fresh tracking session.
        self.reset()

        tracked = []

        for detection in raw_detections:

            tracked_record = self.track_frame(detection)

            tracked.append(tracked_record)

        self._print_summary(tracked)

        return tracked
        
    """
    Track the initially selected largest face for a single frame.

    Returns one tracking record for the input detection record.
    """


    def track_faces(
        self,
        raw_detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Track the initially selected largest face.

        Returns one tracking record per input detection record.
        """

        if not isinstance(raw_detections, list):

            raise TypeError(
                "raw_detections must be a list."
            )

        self.reset()

        tracked = []

        previous_frame_idx = -1

        for frame_position, detection in enumerate(
            raw_detections
        ):

            if not isinstance(detection, dict):

                raise TypeError(
                    f"Detection at position {frame_position} "
                    "must be a dictionary."
                )

            record = dict(detection)

            # ----------------------------------------------
            # FRAME INDEX
            # ----------------------------------------------

            frame_idx = record.get(
                "frame_idx",
                frame_position,
            )

            if (
                isinstance(frame_idx, bool)
                or not isinstance(frame_idx, int)
                or frame_idx < 0
                or frame_idx <= previous_frame_idx
            ):

                frame_idx = max(
                    frame_position,
                    previous_frame_idx + 1,
                )

            record["frame_idx"] = frame_idx

            previous_frame_idx = frame_idx

            # ----------------------------------------------
            # EXTRACT CANDIDATES
            # ----------------------------------------------

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

            # ----------------------------------------------
            # CASE 1: TARGET ALREADY LOST
            # ----------------------------------------------

            # Do not attempt to match another person after
            # the original target has exceeded its loss limit.

            if self.target_lost:

                tracked.append(
                    self._handle_missing(record)
                )

                continue

            # ----------------------------------------------
            # CASE 2: INITIALIZE TARGET
            # ----------------------------------------------

            if self.last_valid_bbox is None:

                if not candidates:

                    tracked.append(
                        self._handle_missing(record)
                    )

                    continue

                tracked.append(
                    self._initialize_target(
                        record,
                        candidates,
                        frame_idx,
                    )
                )

                continue

            # ----------------------------------------------
            # CASE 3: NO CANDIDATES
            # ----------------------------------------------

            if not candidates:

                tracked.append(
                    self._handle_missing(record)
                )

                continue

            # ----------------------------------------------
            # CASE 4: PREDICT TARGET POSITION
            # ----------------------------------------------

            predicted_bbox = self._predict_bbox(
                frame_idx
            )

            if predicted_bbox is None:

                tracked.append(
                    self._handle_missing(record)
                )

                continue

            # ----------------------------------------------
            # CASE 5: DETERMINE MOTION
            # ----------------------------------------------

            fast_motion = False

            if (
                self.previous_valid_bbox is not None
                and self.last_valid_bbox is not None
            ):

                recent_displacement = (
                    self._normalized_displacement(
                        self.previous_valid_bbox,
                        self.last_valid_bbox,
                    )
                )

                fast_motion = (
                    recent_displacement
                    >= self.fast_motion_threshold
                )

            reacquiring = self.missed_frames > 0

            extended_reacquiring = (
                self.missed_frames > self.max_missed_frames
            )

            # ----------------------------------------------
            # CASE 6: FIND ORIGINAL TARGET
            # ----------------------------------------------

            selected = self._select_matching_candidate(
                candidates=candidates,
                predicted_bbox=predicted_bbox,
                fast_motion=fast_motion,
                reacquiring=reacquiring,
                extended_reacquiring=extended_reacquiring,
            )

            # ----------------------------------------------
            # CASE 7: NO ACCEPTABLE MATCH
            # ----------------------------------------------

            if selected is None:

                tracked.append(
                    self._handle_missing(record)
                )

                continue

            # ----------------------------------------------
            # CASE 8: ACCEPT MATCH
            # ----------------------------------------------

            current_bbox = selected["bbox"]

            old_missed_frames = self.missed_frames

            previous_bbox = (
                self.last_valid_bbox.copy()
            )

            smoothed_bbox = self._update_position(
                current_bbox=current_bbox,
                current_frame=frame_idx,
            )

            displacement = self._normalized_displacement(
                previous_bbox,
                current_bbox,
            )

            record["bbox"] = smoothed_bbox

            record["confidence"] = (
                selected["confidence"]
            )

            record["detected"] = True

            record["selected_face_index"] = (
                selected["face_index"]
            )

            record["tracking_iou"] = (
                selected["iou"]
            )

            record["predicted_iou"] = self.compute_iou(
                predicted_bbox,
                current_bbox,
            )

            record["motion_displacement"] = (
                displacement
            )

            record["match_score"] = (
                selected["score"]
            )

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

            tracked.append(record)

        self._print_summary(tracked)

        return tracked

    # ======================================================
    # SUMMARY
    # ======================================================

    def _print_summary(
        self,
        tracked: List[Dict[str, Any]],
    ) -> None:

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

        print("\nTracking statuses:")

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
            ) * 100

            print(
                f"\nTracking-valid rate: "
                f"{tracking_rate:.2f}%"
            )

        if self.iou_values:

            print("\nIoU diagnostics:")

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

            print("\nMotion diagnostics:")

            print(
                f"  Mean normalized displacement: "
                f"{sum(self.displacements) / len(self.displacements):.3f}"
            )

            print(
                f"  Maximum normalized displacement: "
                f"{max(self.displacements):.3f}"
            )

        if self.match_scores:

            print("\nMatching diagnostics:")

            print(
                f"  Mean match score: "
                f"{sum(self.match_scores) / len(self.match_scores):.3f}"
            )

        print(
            "============================================\n"
        )