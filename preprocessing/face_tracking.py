
import math
from numbers import Real
from collections import Counter
from typing import Any, Dict, List, Optional


class FaceTracker:
    """
    Single-face tracker for the FalseFacet rPPG pipeline.

    Input:
        One detection record per video frame.

    Output:
        One tracking record per input frame.

    Bounding-box format:
        [x, y, width, height]

    Important fields:
        detected:
            Whether the detector found a valid face.

        track_valid:
            Whether the detection is accepted for downstream processing.

        tracking_status:
            Explanation of the tracking decision.

        tracking_iou:
            IoU between the current detection and the previous
            accepted bounding box.

    Important limitation:
        IoU measures geometric overlap, not facial identity.
        This tracker cannot guarantee identity consistency.
    """

    def __init__(
        self,
        iou_threshold: float = 0.3,
        max_missed_frames: int = 5,
        debug: bool = False,
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

        # Validate recovery interval.
        if (
            isinstance(max_missed_frames, bool)
            or not isinstance(max_missed_frames, int)
            or max_missed_frames < 1
        ):
            raise ValueError(
                "max_missed_frames must be an integer >= 1."
            )

        if not isinstance(debug, bool):
            raise TypeError("debug must be a boolean.")

        self.iou_threshold = float(iou_threshold)
        self.max_missed_frames = max_missed_frames
        self.debug = debug

        self.last_valid_bbox: Optional[List[float]] = None
        self.missed_frames = 0

    @staticmethod
    def _valid_bbox(bbox: Any) -> bool:
        """Return True if the bounding box is finite and has positive size."""

        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
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
    def compute_iou(
        boxA: List[float],
        boxB: List[float],
    ) -> float:
        """
        Calculate Intersection over Union.

        IoU = intersection area / union area.
        """

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

        intersection_width = max(0.0, x_right - x_left)
        intersection_height = max(0.0, y_bottom - y_top)

        intersection_area = (
            intersection_width * intersection_height
        )

        area_a = aw * ah
        area_b = bw * bh

        union_area = area_a + area_b - intersection_area

        if union_area <= 0.0:
            return 0.0

        return intersection_area / union_area

    def reset(self) -> None:
        """Reset tracking state before processing another video."""

        self.last_valid_bbox = None
        self.missed_frames = 0

    def track_faces(
        self,
        raw_detections: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Process detections in frame order.

        Tracking statuses:
            initialized:
                First accepted detection.

            matched:
                Detection meets the IoU threshold.

            missing:
                No valid detection is available.

            low_iou:
                Detection overlap is below the threshold.

            reacquired:
                Detection accepted after consecutive mismatches.
        """

        if not isinstance(raw_detections, list):
            raise TypeError("raw_detections must be a list.")

        # Each call represents a separate video.
        self.reset()

        tracked: List[Dict[str, Any]] = []
        iou_values: List[float] = []

        for frame_position, detection in enumerate(raw_detections):

            if not isinstance(detection, dict):
                raise TypeError(
                    f"Detection at position {frame_position} "
                    "must be a dictionary."
                )

            # Preserve the original detection record.
            record = dict(detection)
            record.setdefault("frame_idx", frame_position)

            bbox = record.get("bbox")
            detected_flag = record.get("detected", False)

            valid_detection = (
                detected_flag is True
                and self._valid_bbox(bbox)
            )

            record["tracking_iou"] = None
            record["track_valid"] = False

            # -------------------------------------------------
            # CASE 1: No valid detection.
            # -------------------------------------------------
            if not valid_detection:

                self.missed_frames += 1

                record["bbox"] = [0, 0, 0, 0]
                record["confidence"] = 0.0
                record["detected"] = False
                record["tracking_status"] = "missing"

                tracked.append(record)

                if self.debug:
                    print(
                        f"[TRACK DEBUG] Frame "
                        f"{record['frame_idx']}: missing detection"
                    )

                # Discard stale tracking position after enough misses.
                if self.missed_frames >= self.max_missed_frames:
                    self.last_valid_bbox = None

                continue

            current_bbox = [float(value) for value in bbox]

            # Validate confidence without rejecting an otherwise
            # valid bounding box.
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
            # CASE 2: Initialize a new track.
            # -------------------------------------------------
            if self.last_valid_bbox is None:

                self.last_valid_bbox = current_bbox
                self.missed_frames = 0

                record["bbox"] = current_bbox
                record["track_valid"] = True
                record["tracking_status"] = "initialized"

                tracked.append(record)

                if self.debug:
                    print(
                        f"[TRACK DEBUG] Frame "
                        f"{record['frame_idx']}: initialized"
                    )

                continue

            # -------------------------------------------------
            # CASE 3: Compare with the previous accepted box.
            # -------------------------------------------------
            iou = self.compute_iou(
                self.last_valid_bbox,
                current_bbox,
            )

            record["tracking_iou"] = iou
            iou_values.append(iou)

            if iou >= self.iou_threshold:

                self.last_valid_bbox = current_bbox
                self.missed_frames = 0

                record["bbox"] = current_bbox
                record["track_valid"] = True
                record["tracking_status"] = "matched"

                tracked.append(record)

                if self.debug:
                    print(
                        f"[TRACK DEBUG] Frame "
                        f"{record['frame_idx']}: matched, "
                        f"IoU={iou:.3f}"
                    )

                continue

            # -------------------------------------------------
            # CASE 4: Low-IoU detection.
            # -------------------------------------------------
            self.missed_frames += 1

            record["bbox"] = current_bbox
            record["track_valid"] = False
            record["tracking_status"] = "low_iou"

            # Recover after consecutive mismatches.
            if self.missed_frames >= self.max_missed_frames:

                self.last_valid_bbox = current_bbox
                self.missed_frames = 0

                record["track_valid"] = True
                record["tracking_status"] = "reacquired"

            tracked.append(record)

            if self.debug:
                print(
                    f"[TRACK DEBUG] Frame "
                    f"{record['frame_idx']}: "
                    f"{record['tracking_status']}, "
                    f"IoU={iou:.3f}"
                )

        # -----------------------------------------------------
        # Diagnostic summary.
        # -----------------------------------------------------
        status_counts = Counter(
            record.get("tracking_status", "unknown")
            for record in tracked
        )

        accepted_count = sum(
            1
            for record in tracked
            if record["track_valid"]
        )

        total_frames = len(tracked)

        print("\n========== FACE TRACKING SUMMARY ==========")
        print(f"Total frames: {total_frames}")
        print(f"Accepted frames: {accepted_count}")
        print(f"Rejected frames: {total_frames - accepted_count}")

        print("\nTracking statuses:")

        for status in (
            "initialized",
            "matched",
            "missing",
            "low_iou",
            "reacquired",
        ):
            print(f"  {status}: {status_counts.get(status, 0)}")

        if total_frames:
            tracking_rate = accepted_count / total_frames * 100.0
            print(f"\nTracking-valid rate: {tracking_rate:.2f}%")

        if iou_values:
            print("\nIoU diagnostics:")
            print(f"  Comparisons: {len(iou_values)}")
            print(f"  Minimum IoU: {min(iou_values):.3f}")
            print(
                f"  Mean IoU: "
                f"{sum(iou_values) / len(iou_values):.3f}"
            )
            print(f"  Maximum IoU: {max(iou_values):.3f}")
            print(
                f"  Below threshold: "
                f"{sum(i < self.iou_threshold for i in iou_values)}"
            )

        print("============================================")

        return tracked
