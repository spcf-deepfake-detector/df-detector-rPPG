import numpy as np
from typing import Dict, List, Any

class FaceTracker:
    def __init__(self, iou_threshold: float = 0.3):
        self.iou_threshold = iou_threshold

        @staticmethod
        def compute_iou(boxA: List[int], boxB: List[int]) -> float:
            """Computes Intersection over Union (IoU) between two bounding boxes."""
            xA = max(boxA[0], boxB[0])
            yA = max(boxA[1], boxB[1])
            xB = min(boxA[0] + boxA[2], boxB[0] + boxB[2])
            yB = min(boxA[1] + boxA[3], boxB[1] + boxB[3])

            interArea = max(0, xB - xA) * max(0, yB - yA)
            boxAArea = boxA[2] * boxA[3]
            boxBArea = boxB[2] * boxB[3]

            iou = interArea / float(boxAArea + boxBArea - interArea + 1e-8)
            return iou
        

