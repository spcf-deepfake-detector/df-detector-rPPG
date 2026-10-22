import os
import cv2
import numpy as np
import mediapipe as mp

from typing import List, Dict, Any, Tuple, Optional

class FaceDetector:
    def __init__(self, min_detection_confidence: float = 0.5, model_selection: int = 0):
        self.min_confidence = min_detection_confidence
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            min_detection_confidence=self.min_confidence,
            model_selection=model_selection
        )

    def detect_frame(self,frame:np.ndarray) -> List[Dict[str, Any]]:
        if frame is None or frame.ndim != 3 or frame.shape[2] != 3:
            return []

        h, w, _ = frame.shape

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = self.detector.process(rgb_frame)

        detections = []
        if result.detections:
            for detection in result.detections:
                score = float(detection.score[0])
                bbox_data = detection.location_data.relative_bounding_box
                x1 = int(bbox_data.xmin * w)
                y1 = int(bbox_data.ymin * h)
                x2 = int((bbox_data.xmin + bbox_data.width) * w)
                y2 = int((bbox_data.ymin + bbox_data.height) * h)

                x1 = max(0, min(w, x1))
                y1 = max(0, min(h, y1))
                x2 = max(0, min(w, x2))
                y2 = max(0, min(h, y2))

                bbox_w = x2 - x1
                bbox_h = y2 - y1

                if bbox_w <= 0 or bbox_h <= 0:
                    continue

                detections.append({
                    "bbox": [x1, y1, bbox_w, bbox_h],
                    "confidence": score,
                    "detected": True
                })
        return detections

    def process_video(self, video_path: str) -> Dict[str, Any]:
        if not os.path.exists(video_path):
            raise FileNotFoundError(f"Video file not found: {video_path}")
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            cap.release()
            raise ValueError(f"Could not open video: {video_path}")

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0 or np.isnan(fps):
            fps = 30.0
            
        frame_detections = []
        
        valid_count = 0
        frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            detections = self.detect_frame(frame)
            if detections:
                best_detection = max(detections, key=lambda x: x["confidence"])
                bbox = best_detection["bbox"]
                confidence = best_detection["confidence"]
                detected = True
                
                valid_count += 1
            else:
                bbox = [0, 0, 0, 0]
                confidence = 0.0
                detected = False

            frame_detections.append({
                "frame_idx": frame_idx,
                "bbox": bbox,
                "confidence": confidence,
                "detected": detected
            })
            frame_idx += 1

        cap.release()

        total_frames = len(detections)

        detected_frames = sum(1 for item in detections if item.get("detected", False))

        detection_rate = (
            detected_frames/total_frames + 100
            if total_frames > 0
            else 0.0
        )

        print(f"Total frames: {total_frames}")
        print(f"Detected faces: {detected_frames}")
        print(f"Detection rate: {detection_rate:.2f}%")

        print("\n Face Detection Summary")
        print(f"Video: {video_path}")
        print(f"Total frames: {frame_idx}")
        print(f"Detected frames: {valid_count}")
        print(f"Missing detections: {frame_idx - valid_count}")
        print(f"Detection rate: {detection_rate:.2%}")

        return {
            "fps": fps,
            "total_frames": frame_idx,
            "detections": frame_detections,
            "detection_rate": detection_rate

            }

    

    def close(self):
        self.detector.close()

    
    
       
    # print(
    #     f"Frame: {frame_idx}",
    #     f"Face detected: {detected}",
    #     f"Bounding Box: {bbox}"
    # )