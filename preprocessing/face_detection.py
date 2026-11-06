import os
import cv2
import numpy as np
import mediapipe as mp

from typing import List, Dict, Any, Optional


class FaceDetector:
    """
    Multi-scale, multi-face detector for the FalseFacet rPPG pipeline.

    Features:
    - Uses MediaPipe short-range and full-range models.
    - Detects multiple faces in each frame.
    - Supports small and large faces.
    - Supports optional frame upscaling.
    - Merges detections from both models.
    - Removes duplicate bounding boxes.
    - Sorts faces by bounding-box area.
    - Selects the largest face for the legacy single-face output.
    - Preserves all detections when multi_face=True.
    - Returns bounding boxes in original frame coordinates.
    - Records detection statistics.

    IMPORTANT:
    This class detects faces, but does not track identity.

    FaceTracker is responsible for following the same person
    across consecutive frames.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.3,
        model_selection: int = 1,
        frame_scale: float = 1.5,
        max_faces: Optional[int] = None,
        use_dual_model: bool = False, #for testing, set to False
        duplicate_iou_threshold: float = 0.5,
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

        if (
            not np.isfinite(duplicate_iou_threshold)
            or not 0.0 <= duplicate_iou_threshold <= 1.0
        ):
            raise ValueError(
                "duplicate_iou_threshold must be between 0 and 1."
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
        self.use_dual_model = bool(use_dual_model)

        self.duplicate_iou_threshold = float(
            duplicate_iou_threshold
        )

        self.closed = False

        self.detectors = {}


        # ---------------------------------------------------------
        # 2. Initialize MediaPipe
        # ---------------------------------------------------------

        try:

            self.mp_face_detection = mp.solutions.face_detection

            
            # Always initialize the requested model.
            self.detectors[self.model_selection] = (
                self.mp_face_detection.FaceDetection(
                    min_detection_confidence=self.min_confidence,
                    model_selection=self.model_selection,
                )
            )

            # Initialize the other model for additional detections.
            if self.use_dual_model:

                other_model = 1 - self.model_selection

                self.detectors[other_model] = (
                    self.mp_face_detection.FaceDetection(
                        min_detection_confidence=self.min_confidence,
                        model_selection=other_model,
                    )
                )

        except AttributeError as exc:

            raise RuntimeError(
                "MediaPipe's mp.solutions.face_detection API "
                "is unavailable. Check your MediaPipe installation."
            ) from exc

        except Exception:

            self.close()
            raise

        # ---------------------------------------------------------
        # 3. Initialize statistics
        # ---------------------------------------------------------

        self.stats = self._new_stats()

    # =============================================================
    # STATISTICS
    # =============================================================

    @staticmethod
    def _new_stats() -> Dict[str, int]:

        return {
            "frames_attempted": 0,
            "valid_frames": 0,
            "invalid_frames": 0,
            "frames_with_faces": 0,
            "frames_without_faces": 0,
            "total_faces_detected": 0,
            "duplicate_faces_removed": 0,
        }

    # =============================================================
    # FRAME VALIDATION
    # =============================================================

    @staticmethod
    def _is_valid_frame(frame: Any) -> bool:

        return (
            isinstance(frame, np.ndarray)
            and frame.ndim == 3
            and frame.shape[2] == 3
            and frame.shape[0] > 0
            and frame.shape[1] > 0
            and frame.size > 0
        )

    # =============================================================
    # FACE AREA
    # =============================================================

    @staticmethod
    def _get_face_area(face: Dict[str, Any]) -> int:

        x, y, width, height = face["bbox"]

        return max(0, width) * max(0, height)

    # =============================================================
    # FACE SELECTION
    # =============================================================

    @classmethod
    def _select_largest_face(
        cls,
        faces: List[Dict[str, Any]],
    ) -> Optional[Dict[str, Any]]:

        if not faces:
            return None

        return max(
            faces,
            key=cls._get_face_area,
        )

    # =============================================================
    # SORT FACES
    # =============================================================

    @classmethod
    def _sort_faces_by_area(
        cls,
        faces: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:

        return sorted(
            faces,
            key=cls._get_face_area,
            reverse=True,
        )

    # =============================================================
    # BOUNDING BOX IOU
    # =============================================================

    @staticmethod
    def _calculate_iou(
        bbox_a: List[int],
        bbox_b: List[int],
    ) -> float:
        """
        Calculate Intersection over Union.

        bbox format:
            [x, y, width, height]
        """

        ax, ay, aw, ah = bbox_a
        bx, by, bw, bh = bbox_b

        a_x2 = ax + aw
        a_y2 = ay + ah

        b_x2 = bx + bw
        b_y2 = by + bh

        intersection_x1 = max(ax, bx)
        intersection_y1 = max(ay, by)

        intersection_x2 = min(a_x2, b_x2)
        intersection_y2 = min(a_y2, b_y2)

        intersection_w = max(
            0,
            intersection_x2 - intersection_x1,
        )

        intersection_h = max(
            0,
            intersection_y2 - intersection_y1,
        )

        intersection_area = (
            intersection_w * intersection_h
        )

        area_a = aw * ah
        area_b = bw * bh

        union_area = (
            area_a + area_b - intersection_area
        )

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area

    # =============================================================
    # REMOVE DUPLICATE DETECTIONS
    # =============================================================

    def _remove_duplicate_faces(
        self,
        faces: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Remove overlapping detections from the two MediaPipe models.

        The higher-confidence detection is kept.
        """

        if not faces:
            return []

        # Process highest-confidence detections first.
        ordered = sorted(
            faces,
            key=lambda face: face["confidence"],
            reverse=True,
        )

        unique_faces = []

        for candidate in ordered:

            duplicate = False

            for existing in unique_faces:

                iou = self._calculate_iou(
                    candidate["bbox"],
                    existing["bbox"],
                )

                if iou >= self.duplicate_iou_threshold:

                    duplicate = True

                    self.stats["duplicate_faces_removed"] += 1

                    break

            if not duplicate:
                unique_faces.append(candidate)

        return unique_faces

    # =============================================================
    # RUN ONE MEDIAPIPE MODEL
    # =============================================================

    def _run_model(
        self,
        detector: Any,
        rgb_frame: np.ndarray,
        original_width: int,
        original_height: int,
        detection_width: int,
        detection_height: int,
    ) -> List[Dict[str, Any]]:
        """
        Run one MediaPipe detector and convert its boxes
        into original-frame coordinates.
        """

        result = detector.process(rgb_frame)

        if not result.detections:
            return []

        scale_x = detection_width / original_width
        scale_y = detection_height / original_height

        faces = []

        for detection in result.detections:

            # -----------------------------------------------------
            # Confidence
            # -----------------------------------------------------

            if not detection.score:
                continue

            confidence = float(detection.score[0])

            if (
                not np.isfinite(confidence)
                or confidence < self.min_confidence
            ):
                continue

            # -----------------------------------------------------
            # Normalized bounding box
            # -----------------------------------------------------

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

            if (
                relative_bbox.width <= 0
                or relative_bbox.height <= 0
            ):
                continue

            # -----------------------------------------------------
            # Convert normalized coordinates to resized pixels
            # -----------------------------------------------------

            x1 = int(
                round(relative_bbox.xmin * detection_width)
            )

            y1 = int(
                round(relative_bbox.ymin * detection_height)
            )

            x2 = int(
                round(
                    (
                        relative_bbox.xmin
                        + relative_bbox.width
                    ) * detection_width
                )
            )

            y2 = int(
                round(
                    (
                        relative_bbox.ymin
                        + relative_bbox.height
                    ) * detection_height
                )
            )

            # -----------------------------------------------------
            # Clamp to resized frame
            # -----------------------------------------------------

            x1 = max(0, min(detection_width, x1))
            y1 = max(0, min(detection_height, y1))

            x2 = max(0, min(detection_width, x2))
            y2 = max(0, min(detection_height, y2))

            # -----------------------------------------------------
            # Convert to original frame coordinates
            # -----------------------------------------------------

            x1 = int(round(x1 / scale_x))
            y1 = int(round(y1 / scale_y))

            x2 = int(round(x2 / scale_x))
            y2 = int(round(y2 / scale_y))

            # -----------------------------------------------------
            # Clamp to original frame
            # -----------------------------------------------------

            x1 = max(0, min(original_width, x1))
            y1 = max(0, min(original_height, y1))

            x2 = max(0, min(original_width, x2))
            y2 = max(0, min(original_height, y2))

            width = x2 - x1
            height = y2 - y1

            if width <= 0 or height <= 0:
                continue

            faces.append({

                "bbox": [
                    x1,
                    y1,
                    width,
                    height,
                ],

                "confidence": confidence,

                "detected": True,
            })

        return faces

    # =============================================================
    # DETECT FACES IN ONE FRAME
    # =============================================================

    def detect_frame(
        self,
        frame: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Detect faces using both MediaPipe models.

        The frame is optionally enlarged before detection.

        Returns all unique faces sorted from largest to smallest.

        Bounding boxes use ORIGINAL frame coordinates.
        """

        if self.closed:

            raise RuntimeError(
                "FaceDetector has been closed."
            )

        if not self._is_valid_frame(frame):
            return []

        original_height, original_width = frame.shape[:2]

        # ---------------------------------------------------------
        # 1. Upscale frame if configured
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

        detection_height, detection_width = (
            detection_frame.shape[:2]
        )

        # ---------------------------------------------------------
        # 2. Convert BGR to RGB
        # ---------------------------------------------------------

        rgb_frame = cv2.cvtColor(
            detection_frame,
            cv2.COLOR_BGR2RGB,
        )

        # ---------------------------------------------------------
        # 3. Run the configured MediaPipe models
        # ---------------------------------------------------------

        all_faces = []

        for model_id, detector in self.detectors.items():

            model_faces = self._run_model(
                detector=detector,
                rgb_frame=rgb_frame,
                original_width=original_width,
                original_height=original_height,
                detection_width=detection_width,
                detection_height=detection_height,
            )

            all_faces.extend(model_faces)

        # ---------------------------------------------------------
        # 4. Remove duplicate detections
        # ---------------------------------------------------------

        unique_faces = self._remove_duplicate_faces(
            all_faces
        )

        # ---------------------------------------------------------
        # 5. Sort from largest to smallest
        # ---------------------------------------------------------

        unique_faces = self._sort_faces_by_area(
            unique_faces
        )

        # ---------------------------------------------------------
        # 6. Apply optional face limit
        # ---------------------------------------------------------

        if (
            self.max_faces is not None
            and len(unique_faces) > self.max_faces
        ):

            unique_faces = unique_faces[:self.max_faces]

        return unique_faces

    # =============================================================
    # PROCESS VIDEO
    # =============================================================

    def process_video(
        self,
        video_path: str,
        multi_face: bool = False,
        verbose: bool = True,
    ) -> Dict[str, Any]:
        """
        Process every decoded video frame.

        Each frame receives one record.

        Legacy fields:
            frame_idx
            bbox
            confidence
            detected

        When multi_face=True, also returns:
            faces

        The legacy bbox is always the largest detected face.
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

        # ---------------------------------------------------------
        # Video metadata
        # ---------------------------------------------------------

        cap.set(
            cv2.CAP_PROP_ORIENTATION_AUTO,
            1,
        )

        fps = cap.get(cv2.CAP_PROP_FPS)

        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0

        # Reset per-video statistics.
        self.stats = self._new_stats()

        frame_records = []

        frame_idx = 0

        try:

            while True:

                ret, frame = cap.read()

                if not ret:
                    break

                self.stats["frames_attempted"] += 1

                # -------------------------------------------------
                # 1. Invalid frame
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
                # 2. Detect all faces
                # -------------------------------------------------

                faces = self.detect_frame(frame)

                # -------------------------------------------------
                # 3. Select largest face for legacy output
                # -------------------------------------------------

                if faces:

                    self.stats["frames_with_faces"] += 1

                    self.stats["total_faces_detected"] += len(faces)

                    largest_face = self._select_largest_face(
                        faces
                    )

                    record = {

                        "frame_idx": frame_idx,

                        "bbox": largest_face["bbox"].copy(),

                        "confidence": largest_face["confidence"],

                        "detected": True,
                    }

                    if multi_face:

                        # Preserve all detected faces.
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

        # ---------------------------------------------------------
        # 4. Calculate statistics
        # ---------------------------------------------------------

        total_frames = len(frame_records)

        detected_frames = self.stats["frames_with_faces"]

        detection_rate = (

            detected_frames / total_frames * 100.0

            if total_frames > 0

            else 0.0
        )

        # ---------------------------------------------------------
        # 5. Prepare output
        # ---------------------------------------------------------

        summary = {

            "fps": float(fps),

            "total_frames": total_frames,

            "detections": frame_records,

            "detection_rate": detection_rate,

            "total_faces_detected": (
                self.stats["total_faces_detected"]
            ),

            "stats": self.stats.copy(),
        }

        # ---------------------------------------------------------
        # 6. Print summary
        # ---------------------------------------------------------

        if verbose:

            print(
                "\n========== FACE DETECTION SUMMARY =========="
            )

            print(f"Video: {video_path}")

            print(
                f"Confidence threshold: {self.min_confidence:.2f}"
            )

            print(
                f"Primary model: {self.model_selection}"
            )

            # print(
            #     f"Dual model enabled: {self.use_dual_model}"
            # )

            print(
                f"Frame scale: {self.frame_scale:.2f}"
            )

            print(
                f"Maximum faces: {self.max_faces}"
            )

            print(
                "Face selection: Largest bounding-box area"
            )

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
                "Total unique face detections: "
                f"{self.stats['total_faces_detected']}"
            )

            print(
                "Duplicate detections removed: "
                f"{self.stats['duplicate_faces_removed']}"
            )

            print(
                f"Frame detection rate: {detection_rate:.2f}%"
            )

            print(
                "============================================"
            )

        return summary

    # =============================================================
    # RESOURCE MANAGEMENT
    # =============================================================

    def close(self):
        """
        Close all initialized MediaPipe detectors safely.
        """

        if not self.closed:

            for detector in self.detectors.values():

                try:
                    detector.close()

                except Exception:
                    pass

            self.detectors.clear()

            self.closed = True

    def __enter__(self):

        if self.closed:

            raise RuntimeError(
                "This FaceDetector has already been closed."
            )

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        self.close()