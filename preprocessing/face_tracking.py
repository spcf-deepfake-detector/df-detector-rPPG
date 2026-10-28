import math
from numbers import Real
from typing import Any, Dict, List, Optional


class FaceTracker:
    """
    Single-face tracker for the FalseFacet rPPG pipeline.

    Input:
        One detection record per video frame.

    Output:
        One tracking record per input frame.

    Important fields:
        bbox:
            Detected bounding box in [x, y, width, height] format.
            A rejected detection retains its actual box for debugging.

        detected:
            Whether the face detector found a valid face.

        track_valid:
            Whether this detection is accepted into the current track.

        tracking_status:
            Explains the tracking decision.

        tracking_iou:
            Overlap between the previous accepted box and current box.

    Important limitation:
        IoU is a geometric measure, not a face-identity measure.
        This tracker cannot guarantee that the same person is followed
        throughout a video.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed_frames: int = 5,
    ):
        # Validate IoU threshold.
        if (
            isinstance(iou_threshold, bool)
            or not isinstance(iou_threshold, Real)
            or not math.isfinite(iou_threshold)
            or not 0.0 <= iou_threshold <= 1.0
        ):
            raise ValueError(
                "iou_threshold must be a finite number between 0 and 1."
            )

        # Validate the number of consecutive missed frames.
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

        self.last_valid_bbox: Optional[List[float]] = None
        self.missed_frames = 0

    @staticmethod
    def _valid_bbox(bbox: Any) -> bool:
        """
        Check that a bounding box contains finite numeric coordinates
        and positive width and height.
        """

        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False

        for value in bbox:
            if isinstance(value, bool) or not isinstance(value, Real):
                return False

            if not math.isfinite(value):
                return False

        x, y, width, height = bbox

        return width > 0 and height > 0

    @staticmethod
    def compute_iou(
        boxA: List[float],
        boxB: List[float],
    ) -> float:
        """
        Calculate Intersection over Union (IoU).

        Bounding-box format:
            [x, y, width, height]
        """

        if not FaceTracker._valid_bbox(boxA):
            return 0.0

        if not FaceTracker._valid_bbox(boxB):
            return 0.0

        # Calculate the intersection rectangle.
        x_left = max(boxA[0], boxB[0])
        y_top = max(boxA[1], boxB[1])

        x_right = min(
            boxA[0] + boxA[2],
            boxB[0] + boxB[2],
        )

        y_bottom = min(
            boxA[1] + boxA[3],
            boxB[1] + boxB[3],
        )

        intersection_width = max(0.0, x_right - x_left)
        intersection_height = max(0.0, y_bottom - y_top)

        intersection_area = (
            intersection_width * intersection_height
        )

        # Calculate each box's area.
        areaA = boxA[2] * boxA[3]
        areaB = boxB[2] * boxB[3]

        union_area = areaA + areaB - intersection_area

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area

    def reset(self) -> None:
        """
        Reset tracking state before processing another video.
        """

        self.last_valid_bbox = None
        self.missed_frames = 0

    def track_faces(
        self,
        raw_detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Track detected faces across consecutive frames.

        Tracking statuses:
            initialized:
                First accepted detection.

            matched:
                Detection overlaps sufficiently with the previous box.

            missing:
                No usable detection was available.

            low_iou:
                A face was detected, but overlap was too low.

            reacquired:
                A low-IoU detection was accepted after enough
                consecutive tracking mismatches.
        """

        if not isinstance(raw_detections, list):
            raise TypeError(
                "raw_detections must be a list."
            )

        # A new call represents a new video/tracking sequence.
        self.reset()

        tracked: List[Dict[str, Any]] = []

        # Diagnostic counters.
        initialized_count = 0
        matched_count = 0
        missing_count = 0
        low_iou_count = 0
        reacquired_count = 0

        for frame_position, detection in enumerate(raw_detections):

            if not isinstance(detection, dict):
                raise TypeError(
                    f"Detection at position {frame_position} "
                    "must be a dictionary."
                )

            # Preserve the detector's original fields.
            record = dict(detection)
            record.setdefault("frame_idx", frame_position)

            bbox = record.get("bbox")
            detected_flag = record.get("detected", False)

            # Accept only explicit True plus a valid bounding box.
            valid_detection = (
                detected_flag is True
                and self._valid_bbox(bbox)
            )

            # Set tracking defaults.
            record["tracking_iou"] = None
            record["track_valid"] = False

            # -------------------------------------------------
            # CASE 1: No valid detection.
            # -------------------------------------------------
            if not valid_detection:

                self.missed_frames += 1
                missing_count += 1

                record["bbox"] = [0, 0, 0, 0]
                record["confidence"] = 0.0
                record["detected"] = False
                record["track_valid"] = False
                record["tracking_status"] = "missing"

                tracked.append(record)

                # Forget the old position after enough consecutive
                # missing frames. The next valid detection can then
                # initialize a new track.
                if self.missed_frames >= self.max_missed_frames:
                    self.last_valid_bbox = None

                continue

            # The bounding box is valid.
            current_bbox = [
                float(value) for value in bbox
            ]

            # Normalize the confidence score.
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
            # CASE 2: No previous accepted face exists.
            # -------------------------------------------------
            if self.last_valid_bbox is None:

                self.last_valid_bbox = current_bbox
                self.missed_frames = 0

                record["bbox"] = current_bbox
                record["track_valid"] = True
                record["tracking_status"] = "initialized"

                initialized_count += 1
                tracked.append(record)

                continue

            # -------------------------------------------------
            # CASE 3: Compare this detection with the previous
            # accepted bounding box.
            # -------------------------------------------------
            iou = self.compute_iou(
                self.last_valid_bbox,
                current_bbox,
            )

            record["tracking_iou"] = iou

            # Sufficient overlap: accept the detection.
            if iou >= self.iou_threshold:

                self.last_valid_bbox = current_bbox
                self.missed_frames = 0

                record["bbox"] = current_bbox
                record["track_valid"] = True
                record["tracking_status"] = "matched"

                matched_count += 1
                tracked.append(record)

                continue

            # -------------------------------------------------
            # CASE 4: The face was detected, but its box has
            # insufficient overlap with the previous accepted box.
            # -------------------------------------------------
            self.missed_frames += 1

            low_iou_count += 1

            # Keep the actual detected box for debugging.
            # Downstream code MUST check track_valid before
            # extracting RGB measurements.
            record["bbox"] = current_bbox
            record["track_valid"] = False
            record["tracking_status"] = "low_iou"

            # If mismatches continue long enough, accept the new
            # position as a possible new track.
            if self.missed_frames >= self.max_missed_frames:

                self.last_valid_bbox = current_bbox
                self.missed_frames = 0

                record["track_valid"] = True
                record["tracking_status"] = "reacquired"

                reacquired_count += 1

            tracked.append(record)

        # Print a summary so we can identify tracking losses.
        total_frames = len(tracked)

        accepted_count = sum(
            1 for record in tracked
            if record["track_valid"]
        )

        print("\n========== FACE TRACKING SUMMARY ==========")
        print(f"Total frames: {total_frames}")
        print(f"Initialized tracks: {initialized_count}")
        print(f"Matched frames: {matched_count}")
        print(f"Missing detections: {missing_count}")
        print(f"Low-IoU events: {low_iou_count}")
        print(f"Reacquired frames: {reacquired_count}")
        print(f"Accepted tracking frames: {accepted_count}")

        if total_frames > 0:
            tracking_rate = (
                accepted_count / total_frames
            ) * 100.0

            print(
                f"Tracking-valid rate: {tracking_rate:.2f}%"
            )

        print("============================================")

        return tracked