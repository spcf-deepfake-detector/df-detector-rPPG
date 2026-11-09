import cv2
import json
import numpy as np
import mediapipe as mp

from pathlib import Path
from typing import List, Dict, Any, Optional


class FaceDetector:
    """
    Face detector for already-extracted face images.

    Workflow:
        1. Scan extracted_faces recursively.
        2. Read each extracted image.
        3. Detect faces using MediaPipe.
        4. Support multiple faces and optional dual models.
        5. Remove duplicate detections.
        6. Save detection information for every readable image.
        7. Save detection_results.json in each sequence folder.

    Original extracted images are never modified.

    Example:

        extracted_faces/
            test/
                fake/
                    video1/
                        frame_000000.jpg
                        frame_000001.jpg
                        metadata.json

    Output:

        extracted_faces/
            test/
                fake/
                    video1/
                        detection_results.json
    """

    def __init__(
        self,
        min_detection_confidence: float = 0.3,
        model_selection: int = 0,
        frame_scale: float = 1.0,
        max_faces: Optional[int] = None,
        use_dual_model: bool = False,
        duplicate_iou_threshold: float = 0.5,
    ):

        # -----------------------------------------------------
        # 1. Validate configuration
        # -----------------------------------------------------

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

        self.min_confidence = float(
            min_detection_confidence
        )

        self.model_selection = model_selection
        self.frame_scale = float(frame_scale)
        self.max_faces = max_faces
        self.use_dual_model = bool(use_dual_model)

        self.duplicate_iou_threshold = float(
            duplicate_iou_threshold
        )

        self.closed = False
        self.detectors = {}

        self.image_extensions = {
            ".jpg",
            ".jpeg",
            ".png",
        }

        # -----------------------------------------------------
        # 2. Initialize MediaPipe
        # -----------------------------------------------------

        try:

            self.mp_face_detection = (
                mp.solutions.face_detection
            )

            # Initialize the selected model.
            self.detectors[self.model_selection] = (
                self.mp_face_detection.FaceDetection(
                    min_detection_confidence=self.min_confidence,
                    model_selection=self.model_selection,
                )
            )

            # Optionally initialize the other model.
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
                "MediaPipe mp.solutions.face_detection is unavailable."
            ) from exc

        except Exception:

            self.close()
            raise

        self.stats = self._new_stats()

    # =====================================================
    # 3. Statistics
    # =====================================================

    @staticmethod
    def _new_stats():

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
    # 4. Validate image
    # =====================================================

    @staticmethod
    def _is_valid_image(image):

        return (
            isinstance(image, np.ndarray)
            and image.ndim == 3
            and image.shape[2] == 3
            and image.shape[0] > 0
            and image.shape[1] > 0
            and image.size > 0
        )

    # =====================================================
    # 5. Calculate face area
    # =====================================================

    @staticmethod
    def _get_face_area(face):

        x, y, width, height = face["bbox"]

        return max(0, width) * max(0, height)

    # =====================================================
    # 6. Sort faces by area
    # =====================================================

    @classmethod
    def _sort_faces_by_area(cls, faces):

        return sorted(
            faces,
            key=cls._get_face_area,
            reverse=True,
        )

    # =====================================================
    # 7. Calculate IoU
    # =====================================================

    @staticmethod
    def _calculate_iou(bbox_a, bbox_b):

        ax, ay, aw, ah = bbox_a
        bx, by, bw, bh = bbox_b

        intersection_x1 = max(ax, bx)
        intersection_y1 = max(ay, by)

        intersection_x2 = min(
            ax + aw,
            bx + bw,
        )

        intersection_y2 = min(
            ay + ah,
            by + bh,
        )

        intersection_width = max(
            0,
            intersection_x2 - intersection_x1,
        )

        intersection_height = max(
            0,
            intersection_y2 - intersection_y1,
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
    # 8. Remove duplicate detections
    # =====================================================

    def _remove_duplicate_faces(self, faces):

        if not faces:
            return []

        ordered = sorted(
            faces,
            key=lambda face: face["confidence"],
            reverse=True,
        )

        unique_faces = []

        for candidate in ordered:

            is_duplicate = False

            for existing in unique_faces:

                iou = self._calculate_iou(
                    candidate["bbox"],
                    existing["bbox"],
                )

                if iou >= self.duplicate_iou_threshold:

                    is_duplicate = True

                    self.stats[
                        "duplicate_faces_removed"
                    ] += 1

                    break

            if not is_duplicate:
                unique_faces.append(candidate)

        return unique_faces

    # =====================================================
    # 9. Run one MediaPipe model
    # =====================================================

    def _run_model(
        self,
        detector,
        rgb_frame,
        original_width,
        original_height,
    ):

        result = detector.process(rgb_frame)

        if not result.detections:
            return []

        detection_height, detection_width = (
            rgb_frame.shape[:2]
        )

        scale_x = detection_width / original_width
        scale_y = detection_height / original_height

        faces = []

        for detection in result.detections:

            # Confidence.
            if not detection.score:
                continue

            confidence = float(
                detection.score[0]
            )

            if (
                not np.isfinite(confidence)
                or confidence < self.min_confidence
            ):
                continue

            # Normalized bounding box.
            relative_bbox = (
                detection.location_data.relative_bounding_box
            )

            values = [
                relative_bbox.xmin,
                relative_bbox.ymin,
                relative_bbox.width,
                relative_bbox.height,
            ]

            if not np.isfinite(values).all():
                continue

            if (
                relative_bbox.width <= 0
                or relative_bbox.height <= 0
            ):
                continue

            # Convert normalized coordinates to resized pixels.
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

            # Clamp to resized image.
            x1 = max(0, min(detection_width, x1))
            y1 = max(0, min(detection_height, y1))

            x2 = max(0, min(detection_width, x2))
            y2 = max(0, min(detection_height, y2))

            # Convert to original image coordinates.
            x1 = int(round(x1 / scale_x))
            y1 = int(round(y1 / scale_y))

            x2 = int(round(x2 / scale_x))
            y2 = int(round(y2 / scale_y))

            # Clamp to original image.
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

    # =====================================================
    # 10. Detect faces in one extracted image
    # =====================================================

    def detect_frame(self, image):

        if self.closed:
            raise RuntimeError(
                "FaceDetector has been closed."
            )

        if not self._is_valid_image(image):
            return []

        original_height, original_width = image.shape[:2]

        # Resize image when configured.
        if self.frame_scale > 1.0:

            detection_image = cv2.resize(
                image,
                None,
                fx=self.frame_scale,
                fy=self.frame_scale,
                interpolation=cv2.INTER_CUBIC,
            )

        else:

            detection_image = image

        # Convert BGR to RGB.
        rgb_image = cv2.cvtColor(
            detection_image,
            cv2.COLOR_BGR2RGB,
        )

        all_faces = []

        # Run the configured MediaPipe models.
        for detector in self.detectors.values():

            model_faces = self._run_model(
                detector=detector,
                rgb_frame=rgb_image,
                original_width=original_width,
                original_height=original_height,
            )

            all_faces.extend(model_faces)

        # Remove duplicate detections.
        unique_faces = self._remove_duplicate_faces(
            all_faces
        )

        # Largest face first.
        unique_faces = self._sort_faces_by_area(
            unique_faces
        )

        # Apply optional limit.
        if self.max_faces is not None:

            unique_faces = unique_faces[
                :self.max_faces
            ]

        return unique_faces

    # =====================================================
    # 11. Process one image file
    # =====================================================

    def process_image(self, image_path):

        if self.closed:
            raise RuntimeError(
                "FaceDetector has been closed."
            )

        image_path = Path(image_path)

        image = cv2.imread(
            str(image_path)
        )

        self.stats["images_attempted"] += 1

        if image is None or not self._is_valid_image(image):

            self.stats["invalid_images"] += 1

            return {
                "image": str(image_path),
                "detected": False,
                "bbox": None,
                "confidence": 0.0,
                "faces": [],
                "error": "Image could not be read.",
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
    # 12. Process an extracted-images directory
    # =====================================================

    def process_directory(
        self,
        image_dir: str = "extracted_faces",
        output_filename: str = "detection_results.json",
        recursive: bool = True,
        verbose: bool = True,
    ):

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

        # Reset statistics for this directory.
        self.stats = self._new_stats()

        # Find extracted images.
        if recursive:

            image_files = sorted(
                path
                for path in root.rglob("*")
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in self.image_extensions
                )
            )

        else:

            image_files = sorted(
                path
                for path in root.iterdir()
                if (
                    path.is_file()
                    and path.suffix.lower()
                    in self.image_extensions
                )
            )

        if not image_files:

            print(
                f"No extracted images found in: {root}"
            )

            return []

        # Group images by their parent sequence folder.
        sequence_groups = {}

        for image_path in image_files:

            sequence_groups.setdefault(
                image_path.parent,
                [],
            ).append(image_path)

        all_results = []

        # Process each sequence separately.
        for sequence_dir, sequence_images in sorted(
            sequence_groups.items()
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

                sequence_results.append(result)
                all_results.append(result)

                if verbose and (
                    index % 100 == 0
                    or index == len(sequence_images)
                ):

                    print(
                        f"Processed {index}/"
                        f"{len(sequence_images)} images"
                    )

            # Save results beside the extracted images.
            output_path = (
                sequence_dir / output_filename
            )

            with open(
                output_path,
                "w",
                encoding="utf-8",
            ) as file:

                json.dump(
                    sequence_results,
                    file,
                    indent=4,
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

        # -------------------------------------------------
        # Overall statistics
        # -------------------------------------------------

        total_images = self.stats["valid_images"]

        detected_images = self.stats["images_with_faces"]

        detection_rate = (
            detected_images / total_images * 100.0
            if total_images > 0
            else 0.0
        )

        if verbose:

            print(
                "\n========== IMAGE DETECTION SUMMARY =========="
            )

            print(
                f"Images attempted: "
                f"{self.stats['images_attempted']}"
            )

            print(
                f"Valid images: "
                f"{self.stats['valid_images']}"
            )

            print(
                f"Invalid images: "
                f"{self.stats['invalid_images']}"
            )

            print(
                f"Images with faces: {detected_images}"
            )

            print(
                f"Images without faces: "
                f"{self.stats['images_without_faces']}"
            )

            print(
                f"Total faces detected: "
                f"{self.stats['total_faces_detected']}"
            )

            print(
                f"Duplicate detections removed: "
                f"{self.stats['duplicate_faces_removed']}"
            )

            print(
                f"Image detection rate: {detection_rate:.2f}%"
            )

            print(
                "=============================================="
            )

        return all_results

    # =====================================================
    # 13. Close MediaPipe resources
    # =====================================================

    def close(self):

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


# =========================================================
# 14. Standalone execution
# =========================================================

if __name__ == "__main__":

    with FaceDetector(
        min_detection_confidence=0.3,
        model_selection=0,
        frame_scale=1.0,
        max_faces=None,
        use_dual_model=False,
        duplicate_iou_threshold=0.5,
    ) as detector:

        detector.process_directory(
            image_dir="extracted_faces",
            output_filename="detection_results.json",
            recursive=True,
            verbose=True,
        )