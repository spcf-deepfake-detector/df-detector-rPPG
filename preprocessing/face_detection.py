import os
import cv2
import numpy as np
import mediapipe as mp

from typing import List, Dict, Any, Optional


class FaceDetector:
    """
    Face detector for the FalseFacet rPPG pipeline.

    Features:
    - Attempts detection on every valid decoded video frame.
    - Detects multiple faces in a frame when MediaPipe finds them.
    - Supports optional frame upscaling for small faces.
    - Converts bounding boxes back to original frame coordinates.
    - Supports the existing single-face output format.
    - Optionally preserves all detected faces per frame.
    - Records detection statistics for debugging.
    - Handles invalid frames and video decoding errors safely.

    IMPORTANT:
    Face detection does not guarantee that a face has usable skin
    regions for rPPG. ROI extraction and RGB validation are separate
    stages.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.3,
        model_selection: int = 1,
        frame_scale: float = 1.0,
        max_faces: Optional[int] = None,
    ):
        # ---------------------------------------------------------
        # 1. Validate configuration
        # ---------------------------------------------------------

        if (
            not np.isfinite(min_detection_confidence)
            or not 0.0 < min_detection_confidence <= 1.0
        ):
            raise ValueError(
                "min_detection_confidence must be between 0 and 1."
            )

        if model_selection not in (0, 1):
            raise ValueError(
                "model_selection must be 0 or 1."
            )

        if (
            not np.isfinite(frame_scale)
            or frame_scale < 1.0
        ):
            raise ValueError(
                "frame_scale must be finite and >= 1.0."
            )

        if max_faces is not None:
            if (
                isinstance(max_faces, bool)
                or not isinstance(max_faces, int)
                or max_faces < 1
            ):
                raise ValueError(
                    "max_faces must be a positive integer or None."
                )

        self.min_confidence = float(min_detection_confidence)
        self.model_selection = int(model_selection)
        self.frame_scale = float(frame_scale)
        self.max_faces = max_faces

        self.closed = False

        # ---------------------------------------------------------
        # 2. Initialize MediaPipe
        # ---------------------------------------------------------

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
                "unavailable. Check your installed MediaPipe version."
            ) from exc

        # ---------------------------------------------------------
        # 3. Statistics
        # ---------------------------------------------------------

        self.stats = self._new_stats()

    @staticmethod
    def _new_stats() -> Dict[str, int]:
        """Create a fresh statistics dictionary."""

        return {
            "frames_attempted": 0,
            "valid_frames": 0,
            "invalid_frames": 0,
            "frames_with_faces": 0,
            "frames_without_faces": 0,
            "total_faces_detected": 0,
        }

    @staticmethod
    def _is_valid_frame(frame: Any) -> bool:
        """
        Check whether a frame is a non-empty, three-channel image.

        OpenCV normally returns frames in BGR format.
        """

        return (
            isinstance(frame, np.ndarray)
            and frame.ndim == 3
            and frame.shape[2] == 3
            and frame.shape[0] > 0
            and frame.shape[1] > 0
            and frame.size > 0
        )

    def detect_frame(
        self,
        frame: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Detect all faces found in one BGR frame.

        Returns a list. Each detected face contains:

            {
                "bbox": [x, y, width, height],
                "confidence": float,
                "detected": True
            }

        Bounding boxes use coordinates from the ORIGINAL frame.

        An empty list means no usable detection was returned.
        """

        if self.closed:
            raise RuntimeError(
                "FaceDetector has been closed."
            )

        if not self._is_valid_frame(frame):
            return []

        original_h, original_w = frame.shape[:2]

        # ---------------------------------------------------------
        # 1. Optionally enlarge the frame
        # ---------------------------------------------------------

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

        # Use actual dimensions to handle rounding correctly.
        scale_x = detect_w / original_w
        scale_y = detect_h / original_h

        # ---------------------------------------------------------
        # 2. Convert BGR to RGB
        # ---------------------------------------------------------

        rgb_frame = cv2.cvtColor(
            detection_frame,
            cv2.COLOR_BGR2RGB,
        )

        # ---------------------------------------------------------
        # 3. Run MediaPipe face detection
        # ---------------------------------------------------------

        result = self.detector.process(rgb_frame)

        if not result.detections:
            return []

        faces = []

        # ---------------------------------------------------------
        # 4. Process every returned detection
        # ---------------------------------------------------------

        for detection in result.detections:

            # Obtain the confidence score safely.
            scores = detection.score

            if not scores:
                continue

            confidence = float(scores[0])

            if not np.isfinite(confidence):
                continue

            # Obtain the normalized bounding box.
            relative_bbox = (
                detection.location_data.relative_bounding_box
            )

            values = [
                relative_bbox.xmin,
                relative_bbox.ymin,
                relative_bbox.width,
                relative_bbox.height,
            ]

            if not np.all(np.isfinite(values)):
                continue

            if relative_bbox.width <= 0 or relative_bbox.height <= 0:
                continue

            # Convert normalized coordinates into pixels in the
            # resized detection frame.
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

            # Clamp to the resized frame.
            x1 = max(0, min(detect_w, x1))
            y1 = max(0, min(detect_h, y1))
            x2 = max(0, min(detect_w, x2))
            y2 = max(0, min(detect_h, y2))

            # Convert coordinates to the original frame.
            x1 = int(round(x1 / scale_x))
            y1 = int(round(y1 / scale_y))
            x2 = int(round(x2 / scale_x))
            y2 = int(round(y2 / scale_y))

            # Clamp to original frame boundaries.
            x1 = max(0, min(original_w, x1))
            y1 = max(0, min(original_h, y1))
            x2 = max(0, min(original_w, x2))
            y2 = max(0, min(original_h, y2))

            bbox_w = x2 - x1
            bbox_h = y2 - y1

            # Ignore unusable bounding boxes.
            if bbox_w <= 0 or bbox_h <= 0:
                continue

            faces.append({
                "bbox": [x1, y1, bbox_w, bbox_h],
                "confidence": confidence,
                "detected": True,
            })

        # Optionally limit the number of returned faces.
        # None means keep every usable detection returned by MediaPipe.
        if self.max_faces is not None and len(faces) > self.max_faces:
            faces = sorted(
                faces,
                key=lambda face: face["confidence"],
                reverse=True,
            )[:self.max_faces]

        return faces

    def process_video(
        self,
        video_path: str,
        multi_face: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Process every decoded frame in a video.

        Parameters
        ----------
        video_path:
            Path to the video file.

        multi_face:
            False:
                Preserve the existing single-face output format.

            True:
                Return every detected face in each frame using
                the additional "faces" field.

                This mode requires downstream code to understand
                the multi-face output format.

        verbose:
            Print a detection summary when True.

        Returns
        -------
        {
            "fps": float,
            "total_frames": int,
            "detections": list,
            "detection_rate": float,
            "total_faces_detected": int
        }

        The "detections" list contains one record for every decoded
        frame, including frames where no face was detected.

        A frame that fails to decode is not counted as a decoded frame.
        """

        if self.closed:
            raise RuntimeError(
                "FaceDetector has been closed."
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

        # Reset statistics for this video.
        self.stats = self._new_stats()

        frame_records = []
        frame_idx = 0

        try:
            while True:
                ret, frame = cap.read()

                # No more frames, or decoding failed.
                if not ret:
                    break

                self.stats["frames_attempted"] += 1

                # -------------------------------------------------
                # Invalid decoded frame
                # -------------------------------------------------

                if not self._is_valid_frame(frame):
                    self.stats["invalid_frames"] += 1

                    record = {
                        "frame_idx": frame_idx,
                        "bbox": [0, 0, 0, 0],
                        "confidence": 0.0,
                        "detected": False,
                    }

                    if multi_face:
                        record["faces"] = []

                    frame_records.append(record)
                    frame_idx += 1
                    continue

                self.stats["valid_frames"] += 1

                # -------------------------------------------------
                # Detect all faces in this frame
                # -------------------------------------------------

                faces = self.detect_frame(frame)

                if faces:
                    self.stats["frames_with_faces"] += 1
                    self.stats["total_faces_detected"] += len(faces)

                    # Preserve the old single-face behavior.
                    best = max(
                        faces,
                        key=lambda face: face["confidence"],
                    )

                    record = {
                        "frame_idx": frame_idx,
                        "bbox": best["bbox"],
                        "confidence": best["confidence"],
                        "detected": True,
                    }

                    # Preserve all faces when explicitly requested.
                    if multi_face:
                        record["faces"] = faces

                else:
                    self.stats["frames_without_faces"] += 1

                    record = {
                        "frame_idx": frame_idx,
                        "bbox": [0, 0, 0, 0],
                        "confidence": 0.0,
                        "detected": False,
                    }

                    if multi_face:
                        record["faces"] = []

                frame_records.append(record)
                frame_idx += 1

        finally:
            cap.release()

        total_frames = len(frame_records)

        detected_frames = self.stats["frames_with_faces"]

        # Percentage of decoded frames where at least one face
        # was detected. This is NOT the percentage of all faces found.
        detection_rate = (
            detected_frames / total_frames * 100.0
            if total_frames > 0
            else 0.0
        )

        summary = {
            "fps": float(fps),
            "total_frames": total_frames,
            "detections": frame_records,
            "detection_rate": detection_rate,
            "total_faces_detected": (
                self.stats["total_faces_detected"]
            ),
        }

        if verbose:
            print("\n========== FACE DETECTION SUMMARY ==========")
            print(f"Video: {video_path}")
            print(f"Confidence threshold: {self.min_confidence:.2f}")
            print(f"Model selection: {self.model_selection}")
            print(f"Frame scale: {self.frame_scale:.2f}")
            print(f"Multi-face output: {multi_face}")
            print(f"FPS: {fps:.2f}")
            print(f"Total decoded frames: {total_frames}")
            print(
                f"Valid frames: {self.stats['valid_frames']}"
            )
            print(
                f"Invalid frames: {self.stats['invalid_frames']}"
            )
            print(
                f"Frames with faces: {detected_frames}"
            )
            print(
                "Frames without faces: "
                f"{self.stats['frames_without_faces']}"
            )
            print(
                "Total face detections across frames: "
                f"{self.stats['total_faces_detected']}"
            )
            print(f"Frame detection rate: {detection_rate:.2f}%")
            print("============================================")

        return summary

    def close(self):
        """Release MediaPipe resources safely."""

        if not self.closed:
            self.detector.close()
            self.closed = True

    def __enter__(self):
        """Allow use with a 'with' statement."""

        if self.closed:
            raise RuntimeError(
                "This FaceDetector has already been closed."
            )

        return self

    def __exit__(self, exc_type, exc_value, traceback):
        """Close the detector automatically when leaving a 'with' block."""

        self.close()
