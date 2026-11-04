import math
from numbers import Real
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple


class FaceTracker:
    """
    Adaptive single-face tracker for the FalseFacet rPPG pipeline.

    Input:
        One detection record per video frame.

    Output:
        One tracking record per input frame.

    Bounding box:
        [x, y, width, height]

    Tracking statuses:
        initialized
        matched
        predicted_match
        missing
        low_iou
        reacquired

    Important:
        IoU measures bounding-box overlap, not facial identity.
        This tracker uses geometric prediction and is not an
        identity-recognition tracker.
    """

    def __init__(
        self,
        iou_threshold: float = 0.30,
        prediction_threshold: float = 0.20,
        max_missed_frames: int = 5,
        max_prediction_frames: int = 2,
        debug: bool = False,
    ):

        if (
            isinstance(iou_threshold, bool)
            or not isinstance(iou_threshold, Real)
            or not math.isfinite(iou_threshold)
            or not 0.0 <= iou_threshold <= 1.0
        ):
            raise ValueError(
                "iou_threshold must be between 0 and 1."
            )

        if (
            isinstance(prediction_threshold, bool)
            or not isinstance(prediction_threshold, Real)
            or not math.isfinite(prediction_threshold)
            or not 0.0 <= prediction_threshold <= 1.0
        ):
            raise ValueError(
                "prediction_threshold must be between 0 and 1."
            )

        if (
            isinstance(max_missed_frames, bool)
            or not isinstance(max_missed_frames, int)
            or max_missed_frames < 1
        ):
            raise ValueError(
                "max_missed_frames must be an integer >= 1."
            )

        if (
            isinstance(max_prediction_frames, bool)
            or not isinstance(max_prediction_frames, int)
            or max_prediction_frames < 0
        ):
            raise ValueError(
                "max_prediction_frames must be an integer >= 0."
            )

        if not isinstance(debug, bool):
            raise TypeError("debug must be a boolean.")

        self.iou_threshold = float(iou_threshold)
        self.prediction_threshold = float(prediction_threshold)

        self.max_missed_frames = max_missed_frames
        self.max_prediction_frames = max_prediction_frames

        self.debug = debug

        self.reset()

    # ==================================================
    # RESET TRACKING
    # ==================================================

    def reset(self):

        self.last_valid_bbox: Optional[List[float]] = None

        self.previous_valid_bbox: Optional[List[float]] = None

        self.missed_frames = 0

        self.accepted_frames = 0

        self.rejected_frames = 0

    # ==================================================
    # VALIDATE BOUNDING BOX
    # ==================================================

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

    # ==================================================
    # VALIDATE DETECTION FLAG
    # ==================================================

    @staticmethod
    def _is_detected(value: Any) -> bool:

        if isinstance(value, bool):
            return value

        # Supports numpy.bool_ without requiring NumPy import.
        if type(value).__name__ == "bool_":
            return bool(value)

        return False

    # ==================================================
    # VALIDATE CONFIDENCE
    # ==================================================

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

    # ==================================================
    # CALCULATE IOU
    # ==================================================

    @staticmethod
    def compute_iou(
        boxA: List[float],
        boxB: List[float],
    ) -> float:

        if not FaceTracker._valid_bbox(boxA):
            return 0.0

        if not FaceTracker._valid_bbox(boxB):
            return 0.0

        ax, ay, aw, ah = map(float, boxA)
        bx, by, bw, bh = map(float, boxB)

        x_left = max(ax, bx)
        y_top = max(ay, by)

        x_right = min(ax + aw, bx + bw)
        y_bottom = min(ay + ah, by + bh)

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
            aw * ah + bw * bh - intersection_area
        )

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area

    # ==================================================
    # PREDICT NEXT BOUNDING BOX
    # ==================================================

    def _predict_bbox(self) -> Optional[List[float]]:

        if (
            self.last_valid_bbox is None
            or self.previous_valid_bbox is None
        ):
            return self.last_valid_bbox

        last = self.last_valid_bbox
        previous = self.previous_valid_bbox

        # Estimate movement from the last two accepted boxes.
        predicted = []

        for index in range(4):

            movement = last[index] - previous[index]

            predicted.append(
                last[index] + movement
            )

        # Avoid invalid predicted dimensions.
        if not self._valid_bbox(predicted):
            return self.last_valid_bbox.copy()

        return predicted

    # ==================================================
    # UPDATE TRACKING POSITION
    # ==================================================

    def _update_position(self, current_bbox: List[float]):

        self.previous_valid_bbox = (
            self.last_valid_bbox.copy()
            if self.last_valid_bbox is not None
            else None
        )

        self.last_valid_bbox = current_bbox.copy()

    # ==================================================
    # SET TRACKING STATUS
    # ==================================================

    @staticmethod
    def _set_status(
        record: Dict[str, Any],
        status: str,
        valid: bool,
    ):

        record["tracking_status"] = status

        # Backward compatibility with dataset diagnostics.
        record["status"] = status

        record["track_valid"] = valid

    # ==================================================
    # MAIN TRACKING FUNCTION
    # ==================================================

    def track_faces(
        self,
        raw_detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        if not isinstance(raw_detections, list):
            raise TypeError(
                "raw_detections must be a list."
            )

        self.reset()

        tracked = []

        iou_values = []

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

            bbox = record.get("bbox")

            detected = self._is_detected(
                record.get("detected", False)
            )

            valid_detection = (
                detected
                and self._valid_bbox(bbox)
            )

            record["tracking_iou"] = None

            record["track_valid"] = False

            # ------------------------------------------
            # CASE 1: FACE MISSING
            # ------------------------------------------

            if not valid_detection:

                self.missed_frames += 1

                record["bbox"] = [0, 0, 0, 0]

                record["confidence"] = 0.0

                record["detected"] = False

                self._set_status(
                    record,
                    "missing",
                    False,
                )

                self.rejected_frames += 1

                tracked.append(record)

                if (
                    self.missed_frames
                    >= self.max_missed_frames
                ):

                    self.last_valid_bbox = None

                    self.previous_valid_bbox = None

                continue

            current_bbox = [
                float(value)
                for value in bbox
            ]

            record["confidence"] = (
                self._validate_confidence(
                    record.get("confidence", 0.0)
                )
            )

            record["detected"] = True

            # ------------------------------------------
            # CASE 2: INITIALIZE
            # ------------------------------------------

            if self.last_valid_bbox is None:

                self._update_position(current_bbox)

                self.missed_frames = 0

                record["bbox"] = current_bbox

                self._set_status(
                    record,
                    "initialized",
                    True,
                )

                self.accepted_frames += 1

                tracked.append(record)

                continue

            # ------------------------------------------
            # CASE 3: NORMAL IOU MATCH
            # ------------------------------------------

            normal_iou = self.compute_iou(
                self.last_valid_bbox,
                current_bbox,
            )

            iou_values.append(normal_iou)

            record["tracking_iou"] = normal_iou

            if normal_iou >= self.iou_threshold:

                self._update_position(current_bbox)

                self.missed_frames = 0

                record["bbox"] = current_bbox

                self._set_status(
                    record,
                    "matched",
                    True,
                )

                self.accepted_frames += 1

                tracked.append(record)

                continue

            # ------------------------------------------
            # CASE 4: PREDICTED POSITION MATCH
            # ------------------------------------------

            predicted_bbox = self._predict_bbox()

            predicted_iou = 0.0

            if predicted_bbox is not None:

                predicted_iou = self.compute_iou(
                    predicted_bbox,
                    current_bbox,
                )

            record["predicted_iou"] = predicted_iou

            if (
                predicted_iou >= self.prediction_threshold
                and self.missed_frames < self.max_prediction_frames
            ):

                self._update_position(current_bbox)

                self.missed_frames = 0

                record["bbox"] = current_bbox

                self._set_status(
                    record,
                    "predicted_match",
                    True,
                )

                self.accepted_frames += 1

                tracked.append(record)

                continue

            # ------------------------------------------
            # CASE 5: LOW IOU
            # ------------------------------------------

            self.missed_frames += 1

            record["bbox"] = current_bbox

            self._set_status(
                record,
                "low_iou",
                False,
            )

            self.rejected_frames += 1

            # ------------------------------------------
            # CASE 6: REACQUIRE
            # ------------------------------------------

            if (
                self.missed_frames
                >= self.max_missed_frames
            ):

                self._update_position(current_bbox)

                self.missed_frames = 0

                self._set_status(
                    record,
                    "reacquired",
                    True,
                )

                self.accepted_frames += 1

                self.rejected_frames -= 1

            tracked.append(record)

        # ==================================================
        # TRACKING SUMMARY
        # ==================================================

        status_counts = Counter(
            record["tracking_status"]
            for record in tracked
        )

        total_frames = len(tracked)

        print("\n========== FACE TRACKING SUMMARY ==========")

        print(f"Total frames: {total_frames}")

        print(f"Accepted frames: {self.accepted_frames}")

        print(f"Rejected frames: {self.rejected_frames}")

        print("\nTracking statuses:")

        for status in (
            "initialized",
            "matched",
            "predicted_match",
            "missing",
            "low_iou",
            "reacquired",
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

        if iou_values:

            print("\nIoU diagnostics:")

            print(
                f"  Comparisons: {len(iou_values)}"
            )

            print(
                f"  Minimum IoU: {min(iou_values):.3f}"
            )

            print(
                f"  Mean IoU: "
                f"{sum(iou_values) / len(iou_values):.3f}"
            )

            print(
                f"  Maximum IoU: {max(iou_values):.3f}"
            )

            below_threshold = sum(
                value < self.iou_threshold
                for value in iou_values
            )

            print(
                f"  Below threshold: {below_threshold}"
            )

        print("============================================\n")

        return tracked