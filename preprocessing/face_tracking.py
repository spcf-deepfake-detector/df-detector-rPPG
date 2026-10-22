import math
from numbers import Real
from typing import Any, Dict, List, Optional, Tuple


class FaceTracker:
    """
    Basic single-face tracker for the FalseFacet rPPG pipeline.

    Input:
        A list of frame-level face detection dictionaries.

    Output:
        One tracking record per input frame.

    Important fields:
        bbox:
            Current detected bounding box in [x, y, width, height] format.
            It is [0, 0, 0, 0] when no face was detected.

        detected:
            Whether the face detector found a valid face in this frame.

        track_valid:
            Whether this frame's detection is currently accepted as part
            of the tracked face.

        tracking_status:
            Explains what happened to the detection.

        tracking_iou:
            Overlap between the previous accepted box and current box,
            when both are available.

    Limitations:
        This is a basic single-face tracker, not a face-identity system.
        It cannot guarantee that the same person is followed throughout
        a video, especially when multiple faces appear.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed_frames: int = 5,
    ):
        # Validate the tracking settings.
        if (
            isinstance(iou_threshold, bool)
            or not isinstance(iou_threshold, Real)
            or not math.isfinite(iou_threshold)
            or not 0.0 <= iou_threshold <= 1.0
        ):
            raise ValueError(
                "iou_threshold must be a finite number between 0 and 1."
            )

        if (
            isinstance(max_missed_frames, bool)
            or not isinstance(max_missed_frames, int)
            or max_missed_frames < 1
        ):
            raise ValueError(
                "max_missed_frames must be an integer greater than or equal to 1."
            )

        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = max_missed_frames

        # Most recent bounding box accepted into the current track.
        self.last_valid_bbox: Optional[List[float]] = None

        # Number of consecutive frames without an accepted track match.
        self.missed_frames = 0

    @staticmethod
    def _valid_bbox(bbox: Any) -> bool:
        """
        Return True when bbox contains finite coordinates and
        positive width and height.
        """

        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False

        # Reject strings, booleans, NaN, infinity, and non-numeric values.
        for value in bbox:
            if isinstance(value, bool) or not isinstance(value, Real):
                return False

            if not math.isfinite(value):
                return False

        x, y, width, height = bbox

        # x and y may be negative in some intermediate bounding boxes,
        # but width and height must be positive.
        return width > 0 and height > 0

    @staticmethod
    def compute_iou(
        boxA: List[float],
        boxB: List[float],
    ) -> float:
        """
        Calculate Intersection over Union (IoU).

        Both boxes use [x, y, width, height] format.
        """

        if not FaceTracker._valid_bbox(boxA):
            return 0.0

        if not FaceTracker._valid_bbox(boxB):
            return 0.0

        # Top-left corner of the overlapping region.
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])

        # Bottom-right corner of the overlapping region.
        xB = min(
            boxA[0] + boxA[2],
            boxB[0] + boxB[2],
        )

        yB = min(
            boxA[1] + boxA[3],
            boxB[1] + boxB[3],
        )

        # If boxes do not overlap, intersection dimensions are zero.
        intersection_width = max(0.0, xB - xA)
        intersection_height = max(0.0, yB - yA)

        intersection_area = (
            intersection_width * intersection_height
        )

        areaA = boxA[2] * boxA[3]
        areaB = boxB[2] * boxB[3]

        union_area = areaA + areaB - intersection_area

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area

    def reset(self) -> None:
        """Clear tracking state before starting another video."""

        self.last_valid_bbox = None
        self.missed_frames = 0

    def track_faces(
        self,
        raw_detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Process frame-level face detections.

        Low-IoU detections are marked as not valid for the current track.
        After enough consecutive missed or rejected frames, the tracker
        may reacquire a face at a new position.

        This helps handle temporary detection failures and large movements,
        but reacquisition can switch to another person.
        """

        if not isinstance(raw_detections, list):
            raise TypeError(
                "raw_detections must be a list."
            )

        # Start a fresh tracking session for this video.
        self.reset()

        tracked: List[Dict[str, Any]] = []

        for frame_position, detection in enumerate(raw_detections):

            if not isinstance(detection, dict):
                raise TypeError(
                    f"Detection at position {frame_position} must be a dictionary."
                )

            # Preserve original metadata and fill in a missing frame index.
            record = dict(detection)
            record.setdefault("frame_idx", frame_position)

            bbox = record.get("bbox")
            detected_flag = record.get("detected", False)

            # Only a literal True means the detector reports a face.
            # This prevents values such as "False" (a non-empty string)
            # from accidentally being treated as True.
            valid_detection = (
                detected_flag is True
                and self._valid_bbox(bbox)
            )

            # Normalize the output fields for each frame.
            record["tracking_iou"] = None
            record["track_valid"] = False

            # -------------------------------------------------
            # CASE 1: No usable face detection in this frame.
            # -------------------------------------------------
            if not valid_detection:

                self.missed_frames += 1

                # Do not pretend the old bounding box belongs to this frame.
                record["bbox"] = [0, 0, 0, 0]
                record["confidence"] = 0.0
                record["detected"] = False
                record["track_valid"] = False
                record["tracking_status"] = "missing"

                tracked.append(record)

                # We keep the last accepted box internally for comparison.
                # After enough missed frames, forget it so the tracker can
                # reacquire a face at a new position.
                if self.missed_frames >= self.max_missed_frames:
                    self.last_valid_bbox = None

                continue

            # We have a valid detection.
            current_bbox = [float(value) for value in bbox]

            # Normalize confidence to a finite number in [0, 1].
            confidence = record.get("confidence", 0.0)

            if (
                isinstance(confidence, bool)
                or not isinstance(confidence, Real)
                or not math.isfinite(confidence)
                or not 0.0 <= confidence <= 1.0
            ):
                confidence = 0.0

            record["confidence"] = float(confidence)
            record["detected"] = True

            # -------------------------------------------------
            # CASE 2: No previous box exists.
            # Accept the first valid face as the starting track.
            # -------------------------------------------------
            if self.last_valid_bbox is None:

                self.last_valid_bbox = current_bbox
                self.missed_frames = 0

                record["bbox"] = current_bbox
                record["track_valid"] = True
                record["tracking_status"] = "initialized"

                tracked.append(record)
                continue

            # -------------------------------------------------
            # CASE 3: Compare the current box with the previous
            # accepted box.
            # -------------------------------------------------
            iou = self.compute_iou(
                self.last_valid_bbox,
                current_bbox,
            )

            record["tracking_iou"] = iou

            # A sufficiently overlapping box is considered a match.
            if iou >= self.iou_threshold:

                self.last_valid_bbox = current_bbox
                self.missed_frames = 0

                record["bbox"] = current_bbox
                record["track_valid"] = True
                record["tracking_status"] = "matched"

                tracked.append(record)
                continue

            # -------------------------------------------------
            # CASE 4: A face was detected, but its box overlaps
            # too little with the previous accepted box.
            #
            # This could be a different person, a large movement,
            # or a sudden detector change. We cannot know which
            # from IoU alone.
            # -------------------------------------------------
            self.missed_frames += 1

            # Keep the actual detected box for debugging.
            # Downstream rPPG code must check track_valid before
            # using this frame.
            record["bbox"] = current_bbox
            record["track_valid"] = False
            record["tracking_status"] = "low_iou"

            # If the mismatch continues long enough, allow the tracker
            # to start following the new location.
            if self.missed_frames >= self.max_missed_frames:

                self.last_valid_bbox = current_bbox
                self.missed_frames = 0

                record["track_valid"] = True
                record["tracking_status"] = "reacquired"

            tracked.append(record)

        return tracked