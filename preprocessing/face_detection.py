
"""
FALSEFACET - Face Detection
===========================

PURPOSE:
    Detect faces in extracted video frames and return their
    bounding-box coordinates in the original image dimensions.

PIPELINE POSITION:
    extract_frame.py
        |
        v
    face_detection.py
        |
        v
    face_tracking.py
        |
        v
    roi.py
        |
        v
    rgb_extraction.py
        |
        v
    rppg.py

RESPONSIBILITIES:
    1. Detect multiple faces in an image.
    2. Support MediaPipe short-range and full-range models.
    3. Support optional multi-scale detection.
    4. Filter detections by confidence and minimum face size.
    5. Remove duplicate detections using Intersection over Union.
    6. Return bounding boxes in original-image coordinates.
    7. Sort faces from largest to smallest.
    8. Save optional detection results and statistics.

IMPORTANT:
    This class only detects faces.

    It does NOT track face identities across frames.
    FaceTracker handles target selection and tracking.

DETECTION OUTPUT:
    [
        {
            "bbox": [x, y, width, height],
            "confidence": 0.95,
            "detected": True
        }
    ]
"""

import cv2
import json
import math
import os
import re
import tempfile
import numpy as np
import mediapipe as mp

from pathlib import Path
from numbers import Real
from typing import List, Dict, Any, Optional, Tuple


class FaceDetector:
    """
    Multi-face MediaPipe detector for FALSEFACET.

    The detector can use one or both MediaPipe face-detection
    models and optionally process frames at different scales.

    Bounding boxes are always returned in the original image
    coordinate system:

        [x, y, width, height]

    Face tracking and identity selection are handled by
    FaceTracker, not by this class.
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.3,
        model_selection: int = 0,
        frame_scale: float = 1.0,
        max_faces: Optional[int] = None,
        use_dual_model: bool = False,
        duplicate_iou_threshold: float = 0.5,
        multi_scale_factors: Tuple[float, ...] = (1.0,),
        min_face_size: int = 12,
    ):

        # =================================================
        # 1. VALIDATE CONFIGURATION
        # =================================================

        if (
            isinstance(min_detection_confidence, bool)
            or not isinstance(min_detection_confidence, Real)
            or not math.isfinite(min_detection_confidence)
            or not 0.0 < min_detection_confidence <= 1.0
        ):
            raise ValueError(
                "min_detection_confidence must be finite "
                "and between 0 and 1."
            )

        if (
            isinstance(model_selection, bool)
            or not isinstance(model_selection, int)
            or model_selection not in (0, 1)
        ):
            raise ValueError(
                "model_selection must be 0 or 1."
            )

        if (
            isinstance(frame_scale, bool)
            or not isinstance(frame_scale, Real)
            or not math.isfinite(frame_scale)
            or frame_scale < 1.0
        ):
            raise ValueError(
                "frame_scale must be finite and >= 1.0."
            )

        if (
            isinstance(duplicate_iou_threshold, bool)
            or not isinstance(duplicate_iou_threshold, Real)
            or not math.isfinite(duplicate_iou_threshold)
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

        if not isinstance(use_dual_model, bool):

            raise TypeError(
                "use_dual_model must be a boolean."
            )

        if (
            isinstance(min_face_size, bool)
            or not isinstance(min_face_size, int)
            or min_face_size < 1
        ):
            raise ValueError(
                "min_face_size must be a positive integer."
            )

        if (
            not isinstance(multi_scale_factors, (list, tuple))
            or not multi_scale_factors
        ):
            raise ValueError(
                "multi_scale_factors must be a non-empty "
                "list or tuple."
            )

        validated_scales = []

        for scale in multi_scale_factors:

            if (
                isinstance(scale, bool)
                or not isinstance(scale, Real)
                or not math.isfinite(scale)
                or scale <= 0
            ):
                raise ValueError(
                    "Every multi-scale factor must be finite and > 0."
                )

            scale = float(scale)

            if scale not in validated_scales:
                validated_scales.append(scale)

        # =================================================
        # 2. SAVE CONFIGURATION
        # =================================================

        self.min_confidence = float(
            min_detection_confidence
        )

        self.model_selection = model_selection

        self.frame_scale = float(frame_scale)

        self.max_faces = max_faces

        self.use_dual_model = use_dual_model

        self.duplicate_iou_threshold = float(
            duplicate_iou_threshold
        )

        self.multi_scale_factors = tuple(
            validated_scales
        )

        self.min_face_size = min_face_size

        self.closed = False

        # Store initialized MediaPipe detectors here.
        self.detectors = {}

        self.image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
        }

        self.stats = self._new_stats()

        # =================================================
        # 3. INITIALIZE MEDIAPIPE
        # =================================================

        try:

            self.mp_face_detection = (
                mp.solutions.face_detection
            )

            model_selections = [
                self.model_selection
            ]

            if self.use_dual_model:

                other_model = 1 - self.model_selection

                model_selections.append(
                    other_model
                )

            for model in model_selections:

                self.detectors[model] = (
                    self.mp_face_detection.FaceDetection(
                        min_detection_confidence=self.min_confidence,
                        model_selection=model,
                    )
                )

        except Exception:

            # Close any models that were initialized
            # successfully before an initialization failure.

            self.close()

            raise

    # =====================================================
    # 4. STATISTICS
    # =====================================================

    @staticmethod
    def _new_stats() -> Dict[str, int]:
        """
        Create a fresh statistics dictionary.

        Statistics are updated when process_image() is used.
        """

        return {
            "images_attempted": 0,
            "valid_images": 0,
            "invalid_images": 0,
            "images_with_faces": 0,
            "images_without_faces": 0,
            "total_faces_detected": 0,
            "duplicate_faces_removed": 0,
        }

    # =====================================================
    # 5. VALIDATE IMAGE
    # =====================================================

    @staticmethod
    def _is_valid_image(image: Any) -> bool:
        """
        Check whether an image is suitable for MediaPipe.

        Expected input:
            NumPy array
            Three color channels
            uint8 image data
            Positive width and height
        """

        return (
            isinstance(image, np.ndarray)
            and image.ndim == 3
            and image.shape[2] == 3
            and image.shape[0] > 0
            and image.shape[1] > 0
            and image.size > 0
            and image.dtype == np.uint8
        )

    # =====================================================
    # 6. NATURAL FRAME SORTING
    # =====================================================

    @staticmethod
    def _natural_sort_key(path: Path):
        """
        Sort numbered frame filenames naturally.

        Example:

            frame_1.jpg
            frame_2.jpg
            frame_10.jpg

        Instead of:

            frame_1.jpg
            frame_10.jpg
            frame_2.jpg
        """

        parts = re.split(
            r"(\d+)",
            path.name,
        )

        return [
            int(part) if part.isdigit() else part.lower()
            for part in parts
        ]

    # =====================================================
    # 7. FACE GEOMETRY
    # =====================================================

    @staticmethod
    def _get_face_area(face: Dict[str, Any]) -> int:
        """
        Calculate the area of a detected face bounding box.
        """

        x, y, width, height = face["bbox"]

        return width * height

    @classmethod
    def _sort_faces_by_area(
        cls,
        faces: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Sort faces from largest to smallest.

        The largest face is returned first.
        """

        return sorted(
            faces,
            key=cls._get_face_area,
            reverse=True,
        )

    @staticmethod
    def _calculate_iou(
        bbox_a: List[int],
        bbox_b: List[int],
    ) -> float:
        """
        Calculate Intersection over Union (IoU).

        IoU measures the overlap between two bounding boxes.

        Range:
            0.0 = no overlap
            1.0 = identical boxes
        """

        ax, ay, aw, ah = bbox_a

        bx, by, bw, bh = bbox_b

        x1 = max(ax, bx)

        y1 = max(ay, by)

        x2 = min(
            ax + aw,
            bx + bw,
        )

        y2 = min(
            ay + ah,
            by + bh,
        )

        intersection_width = max(
            0,
            x2 - x1,
        )

        intersection_height = max(
            0,
            y2 - y1,
        )

        intersection_area = (
            intersection_width * intersection_height
        )

        area_a = aw * ah

        area_b = bw * bh

        union_area = (
            area_a + area_b - intersection_area
        )

        if union_area <= 0:
            return 0.0

        return intersection_area / union_area

    # =====================================================
    # 8. REMOVE DUPLICATE DETECTIONS
    # =====================================================

    def _remove_duplicate_faces(
        self,
        faces: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        Remove overlapping detections.

        The detection with the highest confidence is kept.

        This is important when:
            - Both MediaPipe models are enabled.
            - Multiple image scales are enabled.

        The same physical face may be detected more than once.

        A zero IoU threshold must not classify non-overlapping
        boxes as duplicates.
        """

        if not faces:
            return []

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

                # Require actual overlap.
                # This prevents IoU=0 from removing
                # completely separate faces.

                if (
                    iou > 0.0
                    and iou >= self.duplicate_iou_threshold
                ):

                    duplicate = True

                    self.stats[
                        "duplicate_faces_removed"
                    ] += 1

                    break

            if not duplicate:

                unique_faces.append(candidate)

        return unique_faces

    # =====================================================
    # 9. RUN ONE MEDIAPIPE MODEL
    # =====================================================

    def _run_model(
        self,
        detector,
        rgb_frame: np.ndarray,
        original_width: int,
        original_height: int,
    ) -> List[Dict[str, Any]]:
        """
        Run one MediaPipe model on an RGB image.

        MediaPipe returns normalized bounding boxes.

        This function converts them into pixel coordinates
        relative to the original image.
        """

        result = detector.process(rgb_frame)

        if not result.detections:
            return []

        detection_height, detection_width = (
            rgb_frame.shape[:2]
        )

        scale_x = (
            detection_width / original_width
        )

        scale_y = (
            detection_height / original_height
        )

        faces = []

        for detection in result.detections:

            # ---------------------------------------------
            # 9.1 CONFIDENCE
            # ---------------------------------------------

            if not detection.score:
                continue

            confidence = float(
                detection.score[0]
            )

            if (
                not math.isfinite(confidence)
                or confidence < self.min_confidence
            ):
                continue

            # ---------------------------------------------
            # 9.2 NORMALIZED BOUNDING BOX
            # ---------------------------------------------

            relative_bbox = (
                detection.location_data.relative_bounding_box
            )

            values = np.asarray(
                [
                    relative_bbox.xmin,
                    relative_bbox.ymin,
                    relative_bbox.width,
                    relative_bbox.height,
                ],
                dtype=np.float64,
            )

            if not np.isfinite(values).all():
                continue

            if (
                relative_bbox.width <= 0
                or relative_bbox.height <= 0
            ):
                continue

            # ---------------------------------------------
            # 9.3 CONVERT TO DETECTION IMAGE COORDINATES
            # ---------------------------------------------

            x1 = int(round(
                relative_bbox.xmin * detection_width
            ))

            y1 = int(round(
                relative_bbox.ymin * detection_height
            ))

            x2 = int(round(
                (
                    relative_bbox.xmin
                    + relative_bbox.width
                ) * detection_width
            ))

            y2 = int(round(
                (
                    relative_bbox.ymin
                    + relative_bbox.height
                ) * detection_height
            ))

            # Clamp to resized image boundaries.

            x1 = max(
                0,
                min(detection_width, x1),
            )

            y1 = max(
                0,
                min(detection_height, y1),
            )

            x2 = max(
                0,
                min(detection_width, x2),
            )

            y2 = max(
                0,
                min(detection_height, y2),
            )

            # ---------------------------------------------
            # 9.4 CONVERT TO ORIGINAL IMAGE COORDINATES
            # ---------------------------------------------

            x1 = int(round(x1 / scale_x))

            y1 = int(round(y1 / scale_y))

            x2 = int(round(x2 / scale_x))

            y2 = int(round(y2 / scale_y))

            # Clamp to original image boundaries.

            x1 = max(
                0,
                min(original_width, x1),
            )

            y1 = max(
                0,
                min(original_height, y1),
            )

            x2 = max(
                0,
                min(original_width, x2),
            )

            y2 = max(
                0,
                min(original_height, y2),
            )

            width = x2 - x1

            height = y2 - y1

            if width <= 0 or height <= 0:
                continue

            # ---------------------------------------------
            # 9.5 MINIMUM FACE SIZE
            # ---------------------------------------------

            if (
                width < self.min_face_size
                or height < self.min_face_size
            ):
                continue

            # ---------------------------------------------
            # 9.6 SAVE DETECTION
            # ---------------------------------------------

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

    # =====================================================
    # 10. DETECT FACES IN ONE FRAME
    # =====================================================

    def detect_frame(
        self,
        image: np.ndarray,
    ) -> List[Dict[str, Any]]:
        """
        Detect all valid faces in one BGR image.

        This is the main method used by face_extraction.py.

        Returns:
            A list of face detections sorted by area.

        Invalid images return an empty list.

        This method does not update process_image statistics.
        """

        if self.closed:

            raise RuntimeError(
                "FaceDetector has been closed."
            )

        if not self._is_valid_image(image):
            return []

        original_height, original_width = (
            image.shape[:2]
        )

        all_faces = []

        # ---------------------------------------------
        # 10.1 MULTI-SCALE DETECTION
        # ---------------------------------------------

        for scale_factor in self.multi_scale_factors:

            effective_scale = (
                self.frame_scale * scale_factor
            )

            if not math.isfinite(effective_scale):

                raise ValueError(
                    "Combined image scale is not finite."
                )

            if effective_scale <= 0:

                raise ValueError(
                    "Combined image scale must be positive."
                )

            if effective_scale == 1.0:

                detection_image = image

            else:

                new_width = max(
                    1,
                    int(round(
                        original_width * effective_scale
                    )),
                )

                new_height = max(
                    1,
                    int(round(
                        original_height * effective_scale
                    )),
                )

                # Use area interpolation for downscaling.
                # Use cubic interpolation for upscaling.

                interpolation = (
                    cv2.INTER_AREA
                    if effective_scale < 1.0
                    else cv2.INTER_CUBIC
                )

                detection_image = cv2.resize(
                    image,
                    (new_width, new_height),
                    interpolation=interpolation,
                )

            # -----------------------------------------
            # BGR -> RGB
            # -----------------------------------------

            rgb_image = cv2.cvtColor(
                detection_image,
                cv2.COLOR_BGR2RGB,
            )

            rgb_image = np.ascontiguousarray(
                rgb_image
            )

            # -----------------------------------------
            # 10.2 RUN AVAILABLE MEDIAPIPE MODELS
            # -----------------------------------------

            for detector in self.detectors.values():

                faces = self._run_model(
                    detector=detector,
                    rgb_frame=rgb_image,
                    original_width=original_width,
                    original_height=original_height,
                )

                all_faces.extend(faces)

        # ---------------------------------------------
        # 10.3 REMOVE DUPLICATES
        # ---------------------------------------------

        unique_faces = self._remove_duplicate_faces(
            all_faces
        )

        # ---------------------------------------------
        # 10.4 LARGEST FACE FIRST
        # ---------------------------------------------

        unique_faces = self._sort_faces_by_area(
            unique_faces
        )

        # ---------------------------------------------
        # 10.5 OPTIONAL FACE LIMIT
        # ---------------------------------------------

        if self.max_faces is not None:

            unique_faces = unique_faces[
                :self.max_faces
            ]

        return unique_faces

    # =====================================================
    # 11. PROCESS ONE IMAGE FILE
    # =====================================================

    def process_image(
        self,
        image_path: str,
    ) -> Dict[str, Any]:
        """
        Read and process one image file.

        Returns:
            {
                "image": "...",
                "detected": True,
                "bbox": [x, y, width, height],
                "confidence": 0.95,
                "faces": [...],
                "error": None
            }

        The bbox field contains the largest detected face.
        The faces field contains all returned detections.
        """

        if self.closed:

            raise RuntimeError(
                "FaceDetector has been closed."
            )

        image_path = Path(image_path)

        self.stats["images_attempted"] += 1

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if (
            image is None
            or not self._is_valid_image(image)
        ):

            self.stats["invalid_images"] += 1

            return {
                "image": str(image_path),
                "detected": False,
                "bbox": None,
                "confidence": 0.0,
                "faces": [],
                "error": "Image could not be read or is invalid.",
            }

        self.stats["valid_images"] += 1

        faces = self.detect_frame(image)

        if faces:

            self.stats["images_with_faces"] += 1

            self.stats["total_faces_detected"] += len(
                faces
            )

            largest_face = faces[0]

            return {
                "image": str(image_path),
                "detected": True,
                "bbox": largest_face["bbox"],
                "confidence": largest_face["confidence"],
                "faces": faces,
                "error": None,
            }

        self.stats["images_without_faces"] += 1

        return {
            "image": str(image_path),
            "detected": False,
            "bbox": None,
            "confidence": 0.0,
            "faces": [],
            "error": None,
        }

    # =====================================================
    # 12. SAVE JSON ATOMICALLY
    # =====================================================

    @staticmethod
    def _save_json_atomic(
        data: Any,
        output_path: Path,
    ) -> None:
        """
        Save JSON through a temporary file.

        os.replace() replaces the destination only after
        the temporary JSON file has been written successfully.

        This reduces the risk of incomplete JSON files if
        the process is interrupted during writing.
        """

        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        temporary_path = None

        try:

            with tempfile.NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=output_path.parent,
                suffix=".tmp",
                delete=False,
            ) as file:

                temporary_path = Path(
                    file.name
                )

                json.dump(
                    data,
                    file,
                    indent=4,
                    allow_nan=False,
                )

                file.flush()

                os.fsync(
                    file.fileno()
                )

            os.replace(
                temporary_path,
                output_path,
            )

        finally:

            if (
                temporary_path is not None
                and temporary_path.exists()
            ):

                temporary_path.unlink(
                    missing_ok=True
                )

    # =====================================================
    # 13. PROCESS EXTRACTED IMAGE DIRECTORY
    # =====================================================

    def process_directory(
        self,
        image_dir: str = "extracted_frames",
        output_filename: str = "detection_results.json",
        recursive: bool = True,
        verbose: bool = True,
    ) -> List[Dict[str, Any]]:
        """
        Process extracted frames inside a directory.

        Each sequence directory receives its own JSON file.

        Example:

            extracted_frames/
                train/
                    real/
                        video_001/
                            frame_000001.jpg
                            frame_000002.jpg
                            detection_results.json

        This method is primarily useful for standalone
        detection testing and debugging.

        The main face_extraction.py pipeline can call
        detect_frame() directly instead.
        """

        if self.closed:

            raise RuntimeError(
                "FaceDetector has been closed."
            )

        root = Path(image_dir)

        if not root.exists():

            raise FileNotFoundError(
                f"Directory not found: {root}"
            )

        if not root.is_dir():

            raise NotADirectoryError(
                f"Not a directory: {root}"
            )

        if (
            not isinstance(output_filename, str)
            or not output_filename.strip()
        ):

            raise ValueError(
                "output_filename must be a non-empty string."
            )

        # Reset statistics for this directory run.

        self.stats = self._new_stats()

        # ---------------------------------------------
        # 13.1 FIND IMAGE FILES
        # ---------------------------------------------

        if recursive:

            image_files = [
                path
                for path in root.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in self.image_extensions
                )
            ]

        else:

            image_files = [
                path
                for path in root.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in self.image_extensions
                )
            ]

        # Sort files naturally.

        image_files.sort(
            key=self._natural_sort_key
        )

        if not image_files:

            if verbose:

                print(
                    f"No extracted images found in: {root}"
                )

            return []

        # ---------------------------------------------
        # 13.2 GROUP IMAGES BY SEQUENCE DIRECTORY
        # ---------------------------------------------

        sequence_groups = {}

        for image_path in image_files:

            sequence_groups.setdefault(
                image_path.parent,
                [],
            ).append(image_path)

        all_results = []

        # ---------------------------------------------
        # 13.3 PROCESS EACH SEQUENCE
        # ---------------------------------------------

        for sequence_dir, sequence_images in sorted(
            sequence_groups.items(),
            key=lambda item: str(item[0]),
        ):

            if verbose:

                print(
                    f"\nProcessing sequence: {sequence_dir}"
                )

                print(
                    f"Images found: {len(sequence_images)}"
                )

            sequence_results = []

            for index, image_path in enumerate(
                sequence_images,
                start=1,
            ):

                result = self.process_image(
                    image_path
                )

                sequence_results.append(
                    result
                )

                all_results.append(
                    result
                )

                if verbose and (
                    index % 100 == 0
                    or index == len(sequence_images)
                ):

                    print(
                        f"Processed {index}/"
                        f"{len(sequence_images)} images"
                    )

            # -----------------------------------------
            # SAVE SEQUENCE RESULTS
            # -----------------------------------------

            output_path = (
                sequence_dir / output_filename
            )

            self._save_json_atomic(
                sequence_results,
                output_path,
            )

            if verbose:

                detected = sum(
                    result["detected"]
                    for result in sequence_results
                )

                print(
                    f"Faces detected: {detected}/"
                    f"{len(sequence_results)}"
                )

                print(
                    f"Saved: {output_path}"
                )

        # ---------------------------------------------
        # 13.4 SUMMARY
        # ---------------------------------------------

        total_images = self.stats["valid_images"]

        detected_images = (
            self.stats["images_with_faces"]
        )

        detection_rate = (
            detected_images / total_images * 100.0
            if total_images > 0
            else 0.0
        )

        if verbose:

            print(
                "\n========== FACE DETECTION SUMMARY =========="
            )

            for key, value in self.stats.items():

                print(
                    f"{key}: {value}"
                )

            print(
                f"Detection rate: {detection_rate:.2f}%"
            )

            print(
                "============================================="
            )

        return all_results

    # =====================================================
    # 14. CLOSE MEDIAPIPE RESOURCES
    # =====================================================

    def close(self) -> None:
        """
        Release all initialized MediaPipe detectors.

        Safe to call more than once.
        """

        if self.closed:
            return

        for detector in list(
            self.detectors.values()
        ):

            try:

                detector.close()

            except Exception:

                # Continue closing other detectors even
                # if one detector fails during cleanup.

                pass

        self.detectors.clear()

        self.closed = True

    # =====================================================
    # 15. CONTEXT MANAGER
    # =====================================================

    def __enter__(self):
        """
        Allow usage with:

            with FaceDetector() as detector:
                ...
        """

        if self.closed:

            raise RuntimeError(
                "FaceDetector has already been closed."
            )

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ):

        self.close()

        return False


# =========================================================
# 16. STANDALONE EXECUTION
# =========================================================

if __name__ == "__main__":

    # Start with one model and one image scale.
    #
    # This is the recommended starting configuration
    # for the current FALSEFACET preprocessing tests.

    with FaceDetector(
        min_detection_confidence=0.3,
        model_selection=0,
        frame_scale=1.0,
        max_faces=None,
        use_dual_model=False,
        duplicate_iou_threshold=0.5,
        multi_scale_factors=(1.0,),
        min_face_size=12,
    ) as detector:

        detector.process_directory(
            image_dir="extracted_frames",
            output_filename="detection_results.json",
            recursive=True,
            verbose=True,
        )
