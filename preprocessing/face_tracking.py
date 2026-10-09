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

        denom = float(boxAArea + boxBArea - interArea)

        return interArea / denom if denom > 0 else 0.0

    def track_faces(self, raw_detections: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Track Faces across frames using intersection over union to maintain the consistency of the detected face bounding boxes."""
        
        if not raw_detections:
            return []
            
        tracked = []
        last_bbox = None

        for detection in raw_detections:
            current_bbox = detection["bbox"]

            if last_bbox is None or current_bbox == [0, 0, 0, 0]:
                tracked.append(detection)

                if current_bbox != [0,0,0,0] and detection["detected"]:
                    last_bbox = current_bbox
            else:
                iou = self.compute_iou(last_bbox, current_bbox)
                if iou < self.iou_threshold and detection["detected"]:
                    tracked.append({
                        "bbox": last_bbox,
                        "confidence": detection["confidence"],
                        "detected": False,
                        "frame_idx": detection["frame_idx"]
                            }) 
                else:
                    tracked.append(detection)

                    if detection["detected"]:
                        last_bbox = current_bbox

        return tracked        

tracker = FaceTracker(iou_threshold=0.3)

raw_detections = [
    {
        "bbox": [100, 50, 200, 250],
        "confidence": 0.95,
        "detected": True,
        "frame_idx": 0
    },
    {
        "bbox": [110, 55, 200, 250],
        "confidence": 0.92,
        "detected": True,
        "frame_idx": 1
    }
]

results = tracker.track_faces(raw_detections)

for result in results:
    print(result)