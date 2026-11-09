import cv2
import json
import re
import numpy as np

from pathlib import Path
from typing import Optional, Dict, Any, List

from preprocessing.face_detection import FaceDetector


class FaceExtractor:
    """
    Detect and extract faces from previously extracted face images.

    Input:
        extracted_faces/
            train/real/video1/frame_000000.jpg
            train/fake/video2/frame_000000.jpg
            val/real/
            val/fake/
            test/real/
            test/fake/

    Output:
        detected_faces/
            train/real/video1/frame_000000.jpg
            train/fake/video2/frame_000000.jpg
            val/real/
            val/fake/
            test/real/
            test/fake/

    Workflow:
        1. Scan all six dataset categories.
        2. Find existing extracted face images.
        3. Read each image.
        4. Detect faces using FaceDetector.
        5. Select the largest detected face.
        6. Crop and save the detected face.
        7. Preserve frame indices and timestamps.
        8. Generate metadata for every image.

    This class does not process original videos.
    """

    REQUIRED_SPLITS = ("train", "val", "test")
    REQUIRED_CLASSES = ("real", "fake")

    IMAGE_EXTENSIONS = {
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp",
    }

    def __init__(
        self,
        input_dir: str = "extracted_faces",
        output_dir: str = "detected_faces",
        detection_confidence: float = 0.3,
        model_selection: int = 0,
        frame_scale: float = 1.0,
        padding_fraction: float = 0.20,
        image_format: str = "jpg",
        debug: bool = True,
        allow_existing_output: bool = False,
        max_faces: Optional[int] = None,
        use_dual_model: bool = False,
        multi_scale_factors=(1.0, 1.25, 1.5),
        duplicate_iou_threshold: float = 0.5,
        min_face_size: int = 12,
    ):

        # -------------------------------------------------
        # 1. Validate configuration
        # -------------------------------------------------

        if not np.isfinite(detection_confidence):
            raise ValueError(
                "detection_confidence must be finite."
            )

        if not 0.0 < detection_confidence <= 1.0:
            raise ValueError(
                "detection_confidence must be between 0 and 1."
            )

        if model_selection not in (0, 1):
            raise ValueError(
                "model_selection must be 0 or 1."
            )

        if not np.isfinite(frame_scale) or frame_scale < 1.0:
            raise ValueError(
                "frame_scale must be finite and >= 1."
            )

        if (
            not np.isfinite(padding_fraction)
            or not 0.0 <= padding_fraction <= 1.0
        ):
            raise ValueError(
                "padding_fraction must be between 0 and 1."
            )

        image_format = image_format.lower().lstrip(".")

        if image_format not in ("jpg", "png"):
            raise ValueError(
                "image_format must be jpg or png."
            )

        if max_faces is not None and max_faces < 1:
            raise ValueError(
                "max_faces must be None or >= 1."
            )

        if min_face_size < 1:
            raise ValueError(
                "min_face_size must be >= 1."
            )

        # -------------------------------------------------
        # 2. Store configuration
        # -------------------------------------------------

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        self.detection_confidence = float(
            detection_confidence
        )

        self.model_selection = model_selection
        self.frame_scale = float(frame_scale)

        self.padding_fraction = float(
            padding_fraction
        )

        self.image_format = image_format
        self.debug = debug

        self.allow_existing_output = allow_existing_output

        self.max_faces = max_faces
        self.use_dual_model = use_dual_model

        self.multi_scale_factors = tuple(
            multi_scale_factors
        )

        self.duplicate_iou_threshold = float(
            duplicate_iou_threshold
        )

        self.min_face_size = int(min_face_size)

    # =====================================================
    # 3. Initialize detector
    # =====================================================

    def _create_detector(self):

        return FaceDetector(
            min_detection_confidence=self.detection_confidence,
            model_selection=self.model_selection,
            frame_scale=self.frame_scale,
            max_faces=self.max_faces,
            use_dual_model=self.use_dual_model,
            duplicate_iou_threshold=self.duplicate_iou_threshold,
            multi_scale_factors=self.multi_scale_factors,
            min_face_size=self.min_face_size,
        )

    # =====================================================
    # 4. Validate bounding box
    # =====================================================

    @staticmethod
    def _validate_bbox(bbox):

        if bbox is None:
            return None

        try:

            values = np.asarray(
                bbox,
                dtype=np.float64,
            )

            if values.shape != (4,):
                return None

            if not np.isfinite(values).all():
                return None

            x, y, width, height = values

            if width <= 0 or height <= 0:
                return None

            return [
                int(round(x)),
                int(round(y)),
                int(round(width)),
                int(round(height)),
            ]

        except (TypeError, ValueError, OverflowError):

            return None

    # =====================================================
    # 5. Select largest face
    # =====================================================

    @classmethod
    def _select_largest_face(cls, detections):

        valid_detections = []

        for detection in detections:

            bbox = cls._validate_bbox(
                detection.get("bbox")
            )

            if bbox is None:
                continue

            area = bbox[2] * bbox[3]

            if area > 0:
                valid_detections.append(
                    (area, detection)
                )

        if not valid_detections:
            return None

        return max(
            valid_detections,
            key=lambda item: item[0],
        )[1]

    # =====================================================
    # 6. Crop detected face
    # =====================================================

    def _crop_face(self, image, bbox):

        bbox = self._validate_bbox(bbox)

        if bbox is None:
            return None

        x, y, width, height = bbox

        image_height, image_width = image.shape[:2]

        pad_x = int(
            round(width * self.padding_fraction)
        )

        pad_y = int(
            round(height * self.padding_fraction)
        )

        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)

        x2 = min(
            image_width,
            x + width + pad_x,
        )

        y2 = min(
            image_height,
            y + height + pad_y,
        )

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]

        if (
            crop.size == 0
            or crop.ndim != 3
            or crop.shape[2] != 3
        ):
            return None

        return crop

    # =====================================================
    # 7. Read original sequence metadata
    # =====================================================

    @staticmethod
    def _load_original_metadata(sequence_dir):

        metadata_path = sequence_dir / "metadata.json"

        if not metadata_path.is_file():
            return {}

        try:

            with open(
                metadata_path,
                "r",
                encoding="utf-8",
            ) as file:

                data = json.load(file)

            if isinstance(data, list):
                records = data

            elif isinstance(data, dict):
                records = (
                    data.get("metadata")
                    or data.get("frames")
                    or data.get("records")
                    or []
                )

            else:
                return {}

            return {
                int(row["frame_index"]): row
                for row in records
                if isinstance(row, dict)
                and "frame_index" in row
            }

        except (OSError, ValueError, TypeError, KeyError) as error:

            print(
                f"WARNING: Could not read {metadata_path}: {error}"
            )

            return {}

    # =====================================================
    # 8. Extract frame index from filename
    # =====================================================

    @staticmethod
    def _get_frame_index(image_path):

        match = re.search(
            r"frame_(\d+)",
            image_path.stem,
            re.IGNORECASE,
        )

        if match:
            return int(match.group(1))

        return None

    # =====================================================
    # 9. Create metadata record
    # =====================================================

    @staticmethod
    def _create_metadata(
        image_path,
        frame_index,
        timestamp,
        detected,
        bbox=None,
        confidence=0.0,
        output_image=None,
    ):

        return {
            "source_image": str(image_path),
            "frame_index": frame_index,
            "timestamp": timestamp,
            "detected": bool(detected),
            "bbox": bbox,
            "confidence": float(confidence),
            "image_path": (
                str(output_image)
                if output_image is not None
                else None
            ),
        }

    # =====================================================
    # 10. Process one extracted image
    # =====================================================

    def process_image(
        self,
        image_path,
        output_path,
        detector,
        original_metadata=None,
    ):

        image_path = Path(image_path)
        output_path = Path(output_path)

        frame_index = self._get_frame_index(
            image_path
        )

        original_record = {}

        if (
            original_metadata is not None
            and frame_index is not None
        ):

            original_record = original_metadata.get(
                frame_index,
                {},
            )

        timestamp = original_record.get("timestamp")

        # Read image.
        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:

            return {
                "metadata": self._create_metadata(
                    image_path=image_path,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    detected=False,
                ),
                "status": "unreadable",
            }

        # Detect faces in the extracted image.
        detections = detector.detect_frame(image)

        if not detections:

            return {
                "metadata": self._create_metadata(
                    image_path=image_path,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    detected=False,
                ),
                "status": "not_detected",
            }

        # Select largest detected face.
        best_detection = self._select_largest_face(
            detections
        )

        if best_detection is None:

            return {
                "metadata": self._create_metadata(
                    image_path=image_path,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    detected=False,
                ),
                "status": "invalid_detection",
            }

        bbox = self._validate_bbox(
            best_detection.get("bbox")
        )

        confidence = float(
            best_detection.get("confidence", 0.0)
        )

        # Crop the detected face.
        face_crop = self._crop_face(
            image,
            bbox,
        )

        if face_crop is None:

            return {
                "metadata": self._create_metadata(
                    image_path=image_path,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    detected=True,
                    bbox=bbox,
                    confidence=confidence,
                ),
                "status": "invalid_crop",
            }

        # Save only successful detections.
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True,
        )

        saved = cv2.imwrite(
            str(output_path),
            face_crop,
        )

        if not saved:

            return {
                "metadata": self._create_metadata(
                    image_path=image_path,
                    frame_index=frame_index,
                    timestamp=timestamp,
                    detected=True,
                    bbox=bbox,
                    confidence=confidence,
                ),
                "status": "save_failed",
            }

        return {
            "metadata": self._create_metadata(
                image_path=image_path,
                frame_index=frame_index,
                timestamp=timestamp,
                detected=True,
                bbox=bbox,
                confidence=confidence,
                output_image=output_path,
            ),
            "status": "saved",
        }

    # =====================================================
    # 11. Process one video sequence directory
    # =====================================================

    def process_sequence(
        self,
        sequence_dir,
        relative_sequence,
        detector,
    ):

        sequence_dir = Path(sequence_dir)

        output_sequence_dir = (
            self.output_dir / relative_sequence
        )

        if (
            output_sequence_dir.exists()
            and not self.allow_existing_output
        ):

            raise FileExistsError(
                f"Output directory already exists: "
                f"{output_sequence_dir}"
            )

        output_sequence_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Load timestamps from original metadata.
        original_metadata = self._load_original_metadata(
            sequence_dir
        )

        image_files = sorted(
            path
            for path in sequence_dir.iterdir()
            if (
                path.is_file()
                and path.suffix.lower() in self.IMAGE_EXTENSIONS
            )
        )

        if not image_files:
            return None

        records = []

        counts = {
            "total_images": 0,
            "detected_images": 0,
            "saved_images": 0,
            "missed_images": 0,
            "unreadable_images": 0,
            "invalid_crops": 0,
            "failed_saves": 0,
        }

        confidence_values = []

        for image_path in image_files:

            output_name = (
                image_path.stem + "." + self.image_format
            )

            output_path = (
                output_sequence_dir / output_name
            )

            result = self.process_image(
                image_path=image_path,
                output_path=output_path,
                detector=detector,
                original_metadata=original_metadata,
            )

            records.append(result["metadata"])

            counts["total_images"] += 1

            status = result["status"]

            if status == "saved":

                counts["detected_images"] += 1
                counts["saved_images"] += 1

                confidence_values.append(
                    result["metadata"]["confidence"]
                )

            elif status == "not_detected":
                counts["missed_images"] += 1

            elif status == "unreadable":
                counts["unreadable_images"] += 1

            elif status == "invalid_detection":
                counts["missed_images"] += 1

            elif status == "invalid_crop":

                counts["detected_images"] += 1
                counts["invalid_crops"] += 1

                confidence_values.append(
                    result["metadata"]["confidence"]
                )

            elif status == "save_failed":

                counts["detected_images"] += 1
                counts["failed_saves"] += 1

                confidence_values.append(
                    result["metadata"]["confidence"]
                )

        total = counts["total_images"]

        counts["detection_rate"] = (
            counts["detected_images"] / total * 100.0
            if total else 0.0
        )

        counts["extraction_rate"] = (
            counts["saved_images"] / total * 100.0
            if total else 0.0
        )

        counts["average_confidence"] = (
            float(np.mean(confidence_values))
            if confidence_values else 0.0
        )

        # Save metadata for this sequence.
        metadata_path = (
            output_sequence_dir / "metadata.json"
        )

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                records,
                file,
                indent=4,
            )

        counts["sequence"] = str(relative_sequence)
        counts["metadata_path"] = str(metadata_path)

        if self.debug:

            print(
                f"\nSequence: {relative_sequence}"
            )

            print(
                f"Images: {counts['total_images']}"
            )

            print(
                f"Detected: {counts['detected_images']}"
            )

            print(
                f"Saved: {counts['saved_images']}"
            )

            print(
                f"Detection rate: {counts['detection_rate']:.2f}%"
            )

            print(
                f"Extraction rate: {counts['extraction_rate']:.2f}%"
            )

        return counts

    # =====================================================
    # 12. Process the complete extracted dataset
    # =====================================================

    def extract_dataset(self):

        if not self.input_dir.is_dir():

            raise NotADirectoryError(
                f"Input directory not found: {self.input_dir}"
            )

        if (
            self.input_dir.resolve() == self.output_dir.resolve()
            or self.input_dir.resolve() in self.output_dir.resolve().parents
        ):

            raise ValueError(
                "Output directory must not be inside the input directory."
            )

        print(
            "\n========== EXTRACTED IMAGE DETECTION =========="
        )

        print(
            f"Input: {self.input_dir.resolve()}"
        )

        print(
            f"Output: {self.output_dir.resolve()}"
        )

        results = []

        detector = self._create_detector()

        try:

            for split in self.REQUIRED_SPLITS:

                for label in self.REQUIRED_CLASSES:

                    class_dir = (
                        self.input_dir / split / label
                    )

                    if not class_dir.is_dir():

                        print(
                            f"WARNING: Missing folder: {class_dir}"
                        )

                        continue

                    print(
                        f"\nProcessing category: {split}/{label}"
                    )

                    # Find sequence folders containing images.
                    sequence_dirs = sorted({
                        image_path.parent
                        for image_path in class_dir.rglob("*")
                        if (
                            image_path.is_file()
                            and image_path.suffix.lower()
                            in self.IMAGE_EXTENSIONS
                        )
                    })

                    for sequence_dir in sequence_dirs:

                        relative_sequence = (
                            sequence_dir.relative_to(
                                self.input_dir
                            )
                        )

                        try:

                            result = self.process_sequence(
                                sequence_dir=sequence_dir,
                                relative_sequence=relative_sequence,
                                detector=detector,
                            )

                            if result is not None:

                                result["split"] = split
                                result["label"] = label

                                results.append(result)

                        except Exception as error:

                            print(
                                f"ERROR processing {sequence_dir}: {error}"
                            )

                            results.append({
                                "sequence": str(relative_sequence),
                                "split": split,
                                "label": label,
                                "error": str(error),
                            })

        finally:

            detector.close()

        # -------------------------------------------------
        # Aggregate statistics
        # -------------------------------------------------

        successful = [
            result
            for result in results
            if "error" not in result
        ]

        failed = [
            result
            for result in results
            if "error" in result
        ]

        total_images = sum(
            item["total_images"]
            for item in successful
        )

        total_detected = sum(
            item["detected_images"]
            for item in successful
        )

        total_saved = sum(
            item["saved_images"]
            for item in successful
        )

        total_missed = sum(
            item["missed_images"]
            for item in successful
        )

        detection_rate = (
            total_detected / total_images * 100.0
            if total_images else 0.0
        )

        extraction_rate = (
            total_saved / total_images * 100.0
            if total_images else 0.0
        )

        print(
            "\n========== COMPLETE IMAGE DETECTION SUMMARY =========="
        )

        print(
            f"Sequences processed: {len(successful)}"
        )

        print(
            f"Failed sequences: {len(failed)}"
        )

        print(
            f"Total images: {total_images}"
        )

        print(
            f"Detected images: {total_detected}"
        )

        print(
            f"Saved face images: {total_saved}"
        )

        print(
            f"Missed images: {total_missed}"
        )

        print(
            f"Detection rate: {detection_rate:.2f}%"
        )

        print(
            f"Extraction rate: {extraction_rate:.2f}%"
        )

        print(
            f"Output directory: {self.output_dir.resolve()}"
        )

        print(
            "======================================================="
        )

        return results


# =========================================================
# 13. Standalone execution
# =========================================================

if __name__ == "__main__":

    extractor = FaceExtractor(
        input_dir="extracted_faces",
        output_dir="detected_faces",

        detection_confidence=0.3,
        model_selection=0,
        frame_scale=1.0,

        padding_fraction=0.20,
        image_format="jpg",

        debug=True,
        allow_existing_output=False,

        max_faces=None,
        use_dual_model=False,

        multi_scale_factors=(1.0, 1.25, 1.5),

        duplicate_iou_threshold=0.5,
        min_face_size=12,
    )

    extractor.extract_dataset()