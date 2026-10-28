import os
import cv2
import numpy as np
import mediapipe as mp

from typing import List, Dict, Any


class FaceDetector:
    """
    Detect faces in video frames for the FalseFacet rPPG pipeline.

    Features:
    - Configurable detection confidence.
    - Configurable MediaPipe model selection.
    - Optional frame upscaling for small faces.
    - Bounding boxes returned in original frame coordinates.
    - Detection statistics for debugging.

    Each valid decoded video frame produces one record in process_video().
    If no face is detected, the bounding box is [0, 0, 0, 0].
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.3,
        model_selection: int = 1,
        frame_scale: float = 1.0,
    ):
        # Validate the confidence threshold.
        if not np.isfinite(min_detection_confidence):
            raise ValueError(
                "min_detection_confidence must be a finite number."
            )

        if not 0.0 < min_detection_confidence <= 1.0:
            raise ValueError(
                "min_detection_confidence must be between 0 and 1."
            )

        # Validate the MediaPipe model selection.
        if model_selection not in (0, 1):
            raise ValueError(
                "model_selection must be 0 or 1."
            )

        # Validate the optional image scaling factor.
        if not np.isfinite(frame_scale) or frame_scale < 1.0:
            raise ValueError(
                "frame_scale must be a finite number greater than "
                "or equal to 1.0."
            )

        self.min_confidence = float(min_detection_confidence)
        self.model_selection = model_selection
        self.frame_scale = float(frame_scale)

        # Initialize MediaPipe's face detector.
        try:
            self.mp_face_detection = mp.solutions.face_detection

            self.detector = (
                self.mp_face_detection.FaceDetection(
                    min_detection_confidence=self.min_confidence,
                    model_selection=self.model_selection,
                )
            )

        except AttributeError as exc:
            raise RuntimeError(
                "MediaPipe's mp.solutions.face_detection API is "
                "unavailable in this environment. Check the installed "
                "MediaPipe version and API compatibility."
            ) from exc

        self.closed = False

    @staticmethod
    def _is_valid_frame(frame: Any) -> bool:
        """
        Check whether a frame is a valid three-channel image.
        """

        return (
            isinstance(frame, np.ndarray)
            and frame.ndim == 3
            and frame.shape[2] == 3
            and frame.size > 0
            and frame.shape[0] > 0
            and frame.shape[1] > 0
        )

    def detect_frame(
        self,
        frame: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Detect faces in one BGR frame.

        Returns:
            A list of detected faces.

            Each face contains:
                bbox: [x, y, width, height]
                confidence: detection confidence
                detected: True

        Coordinates refer to the original frame, even when upscaling
        is enabled.
        """

        if self.closed:
            raise RuntimeError(
                "FaceDetector has been closed. Create a new instance "
                "before detecting more frames."
            )

        # Invalid frames cannot be processed.
        if not self._is_valid_frame(frame):
            return []

        original_h, original_w = frame.shape[:2]

        # Optionally enlarge the frame to help with small faces.
        if self.frame_scale > 1.0:
            detection_frame = cv2.resize(
                frame,
                None,
                fx=self.frame_scale,
                fy=self.frame_scale,
                interpolation=cv2.INTER_CUBIC,
            )
        else:
            detection_frame = frame

        detect_h, detect_w = detection_frame.shape[:2]

        # Use actual dimensions instead of assuming the resize factor
        # produced perfectly exact dimensions.
        scale_x = detect_w / original_w
        scale_y = detect_h / original_h

        # OpenCV frames use BGR, while MediaPipe expects RGB.
        rgb_frame = cv2.cvtColor(
            detection_frame,
            cv2.COLOR_BGR2RGB,
        )

        # Run face detection.
        result = self.detector.process(rgb_frame)

        if not result.detections:
            return []

        detections = []

        for detection in result.detections:
            # Safely obtain the confidence score.
            if (
                not detection.score
                or not np.isfinite(detection.score[0])
            ):
                continue

            confidence = float(detection.score[0])

            # Obtain the normalized bounding box.
            relative_bbox = (
                detection.location_data.relative_bounding_box
            )

            # Convert normalized coordinates to pixels in the
            # detection frame.
            x1 = int(round(relative_bbox.xmin * detect_w))
            y1 = int(round(relative_bbox.ymin * detect_h))

            x2 = int(
                round(
                    (relative_bbox.xmin + relative_bbox.width)
                    * detect_w
                )
            )

            y2 = int(
                round(
                    (relative_bbox.ymin + relative_bbox.height)
                    * detect_h
                )
            )

            # Clamp coordinates to the resized frame.
            x1 = max(0, min(detect_w, x1))
            y1 = max(0, min(detect_h, y1))
            x2 = max(0, min(detect_w, x2))
            y2 = max(0, min(detect_h, y2))

            # Convert coordinates back to the original frame.
            x1 = int(round(x1 / scale_x))
            y1 = int(round(y1 / scale_y))
            x2 = int(round(x2 / scale_x))
            y2 = int(round(y2 / scale_y))

            # Clamp again to the original frame.
            x1 = max(0, min(original_w, x1))
            y1 = max(0, min(original_h, y1))
            x2 = max(0, min(original_w, x2))
            y2 = max(0, min(original_h, y2))

            bbox_w = x2 - x1
            bbox_h = y2 - y1

            # Ignore boxes that have no usable area.
            if bbox_w <= 0 or bbox_h <= 0:
                continue

            detections.append({
                "bbox": [x1, y1, bbox_w, bbox_h],
                "confidence": confidence,
                "detected": True,
            })

        return detections

    def process_video(
        self,
        video_path: str,
    ) -> Dict[str, Any]:
        """
        Process a video and produce one record per decoded frame.

        The existing output format is preserved:

            {
                "fps": ...,
                "total_frames": ...,
                "detections": [...],
                "detection_rate": ...
            }

        Additional summary information is printed for debugging.
        """

        if self.closed:
            raise RuntimeError(
                "FaceDetector has been closed. Create a new instance "
                "before processing another video."
            )

        if not os.path.isfile(video_path):
            raise FileNotFoundError(
                f"Video file not found: {video_path}"
            )

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            cap.release()
            raise ValueError(
                f"Could not open video: {video_path}"
            )

        fps = cap.get(cv2.CAP_PROP_FPS)

        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0

        frame_detections = []

        detected_count = 0
        missed_count = 0
        invalid_frame_count = 0
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()

                # End of video or decoding failure.
                if not ret:
                    break

                # Record invalid decoded frames separately.
                if not self._is_valid_frame(frame):
                    invalid_frame_count += 1

                    frame_detections.append({
                        "frame_idx": frame_idx,
                        "bbox": [0, 0, 0, 0],
                        "confidence": 0.0,
                        "detected": False,
                    })

                    frame_idx += 1
                    continue

                # Detect all faces in this frame.
                detections = self.detect_frame(frame)

                if detections:
                    # Preserve the existing pipeline behavior:
                    # select the highest-confidence face.
                    best = max(
                        detections,
                        key=lambda item: item["confidence"],
                    )

                    bbox = best["bbox"]
                    confidence = best["confidence"]
                    detected = True

                    detected_count += 1

                else:
                    bbox = [0, 0, 0, 0]
                    confidence = 0.0
                    detected = False

                    missed_count += 1

                # Maintain one record for every decoded frame.
                frame_detections.append({
                    "frame_idx": frame_idx,
                    "bbox": bbox,
                    "confidence": confidence,
                    "detected": detected,
                })

                frame_idx += 1

        finally:
            cap.release()

        total_frames = len(frame_detections)

        # Keep the same detection-rate definition as before:
        # detected frames divided by all decoded frames.
        detection_rate = (
            detected_count / total_frames * 100.0
            if total_frames > 0
            else 0.0
        )

        print("\n========== FACE DETECTION SUMMARY ==========")
        print(f"Video: {video_path}")
        print(
            f"Confidence threshold: {self.min_confidence:.2f}"
        )
        print(f"Model selection: {self.model_selection}")
        print(f"Frame scale: {self.frame_scale:.2f}")
        print(f"FPS: {fps:.2f}")
        print(f"Total decoded frames: {total_frames}")
        print(f"Detected frames: {detected_count}")
        print(f"Frames without a detection: {missed_count}")
        print(f"Invalid decoded frames: {invalid_frame_count}")
        print(f"Detection rate: {detection_rate:.2f}%")
        print("============================================")

        return {
            "fps": fps,
            "total_frames": total_frames,
            "detections": frame_detections,
            "detection_rate": detection_rate,
        }

    def close(self):
        """
        Release MediaPipe resources safely.
        """

        if not self.closed:
            self.detector.close()
            self.closed = True