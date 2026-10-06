import numpy as np
from typing import List, Dict, Any

class FaceTracker:
    """Tracks a primary face across contiguous frames to prevent target switching."""
    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold

    @staticmethod
    def compute_iou(boxA: List[int], boxB: List[int]) -> float:
        xA = max(boxA[0], boxB[0])
        yA = max(boxA[1], boxB[1])
        xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
        yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

        interArea = max(0, xB - xA) * max(0, yB - yA)
        boxAArea = boxA[2] * boxA[3]
        boxBArea = boxB[2] * boxB[3]

        denom = float(boxAArea + boxBArea - interArea)
        return interArea / denom if denom > 0 else 0.0

    def track(self, raw_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not raw_detections:
            return []

        tracked = []
        last_bbox = None

        for det in raw_detections:
            current_bbox = det["bbox"]
            if last_bbox is None or current_bbox == [0, 0, 0, 0]:
                tracked.append(det)
                if current_bbox != [0, 0, 0, 0]:
                    last_bbox = current_bbox
            else:
                iou = self.compute_iou(last_bbox, current_bbox)
                if iou < self.iou_threshold and det["detected"]:
                    tracked.append({
                        "bbox": last_bbox,
                        "confidence": det["confidence"],
                        "detected": False,
                        "frame_idx": det["frame_idx"]
                    })
                else:
                    tracked.append(det)
                    if det["detected"]:
                        last_bbox = current_bbox
        return tracked