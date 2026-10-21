import os
import cv2
import numpy as np
import mediapipe as mp

from typing import List, Dict, Any, Optional


class FaceTracker:
    """
    Tracks face detections across video frames for FalseFacet rPPG.

    Each detection contains:
        - frame_idx: Video frame number
        - bbox: [x, y, width, height]
        - confidence: Face detection confidence
        - detected: Whether a face was detected

    This is a basic single-face tracker. It preserves missing
    detections but does not guarantee identity tracking when
    multiple faces appear.
    """

    def __init__(self, iou_threshold: float = 0.3):
        if not 0.0 <= iou_threshold <= 1.0:
            raise ValueError(
                "iou_threshold must be between 0 and 1."
            )

        self.iou_threshold = iou_threshold
        self.last_valid_bbox: Optional[List[int]] = None

    @staticmethod
    def _valid_bbox(bbox: Any) -> bool:
        """Check whether a bounding box contains valid coordinates."""

        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            return False

        try:
            x, y, width, height = bbox

            if not all(
                isinstance(value, (int, float))
                for value in bbox
            ):
                return False

            return (
                width > 0
                and height > 0
            )

        except (TypeError, ValueError):
            return False

    @staticmethod
    def compute_iou(
        boxA: List[int],
        boxB: List[int]
    ) -> float:
        """
        Calculate Intersection over Union (IoU).

        IoU = Intersection Area / Union Area
        """

        if not FaceTracker._valid_bbox(boxA):
            return 0.0

        if not FaceTracker._valid_bbox(boxB):
            return 0.0

        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])

        xB = min(
            boxA[0] + boxA[2],
            boxB[0] + boxB[2]
        )

        yB = min(
            boxA[1] + boxA[3],
            boxB[1] + boxB[3]
        )

        intersection_width = max(0, xB - xA)
        intersection_height = max(0, yB - yA)

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
        """Reset tracking state before processing a new video."""

        self.last_valid_bbox = None

    def track_faces(
        self,
        raw_detections: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """
        Process face detections while preserving frame alignment.

        Args:
            raw_detections: List of frame detection dictionaries.

        Returns:
            A list containing one tracking record per input record.

        Missing detections remain marked as detected=False.
        """

        if not raw_detections:
            self.reset()
            return []

        self.reset()

        tracked = []

        for detection in raw_detections:

            if not isinstance(detection, dict):
                raise TypeError(
                    "Each detection must be a dictionary."
                )

            bbox = detection.get("bbox")
            detected = detection.get("detected", False)

            valid_bbox = self._valid_bbox(bbox)

            if detected and valid_bbox:

                current_bbox = list(bbox)

                # Calculate IoU against the previous valid detection.
                # Low overlap is recorded conceptually as a possible
                # sudden movement or face change.
                if self.last_valid_bbox is not None:
                    iou = self.compute_iou(
                        self.last_valid_bbox,
                        current_bbox
                    )

                    # This basic tracker does not reject detections
                    # based on IoU alone because a real face can move
                    # substantially between consecutive frames.
                    _ = iou

                tracked.append({
                    **detection,
                    "bbox": current_bbox,
                    "detected": True
                })

                self.last_valid_bbox = current_bbox

            else:

                # Keep the previous box as a reference, but preserve
                # the fact that no face was detected in this frame.
                reference_bbox = (
                    self.last_valid_bbox.copy()
                    if self.last_valid_bbox is not None
                    else [0, 0, 0, 0]
                )

                tracked.append({
                    **detection,
                    "bbox": reference_bbox,
                    "confidence": 0.0,
                    "detected": False
                })

        return tracked