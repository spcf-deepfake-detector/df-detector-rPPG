import cv2
import json
import numpy as np

from pathlib import Path
from typing import Optional, Dict, List, Any

from preprocessing.face_detection import FaceDetector
from preprocessing.face_tracking import FaceTracker


# ============================================================
# FACE EXTRACTION
# ============================================================

class FaceExtractor:
    """
    FALSEFACET face extraction pipeline.

    Pipeline:
        Extracted frames
            -> FaceDetector
            -> FaceTracker
            -> Selected face
            -> Face crop
            -> Saved face image
            -> metadata.json

    Important:
        - The largest face initializes the tracker.
        - The tracker follows the initially selected person.
        - Only actual detector-matched faces are cropped.
        - Predicted tracking positions are never treated as
          valid face detections.
        - A new tracker is created for every video sequence.
    """

    # ========================================================
    # INITIALIZATION
    # ========================================================

    def __init__(
        self,
        input_dir="extracted_frames",
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
        duplicate_iou_threshold=0.5,
    ):

        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)

        self.detection_confidence = detection_confidence
        self.model_selection = model_selection
        self.frame_scale = frame_scale
        self.padding_fraction = padding_fraction
        self.image_format = image_format.lower().lstrip(".")
        self.debug = debug
        self.allow_existing_output = allow_existing_output
        self.max_faces = max_faces
        self.use_dual_model = use_dual_model
        self.duplicate_iou_threshold = duplicate_iou_threshold

        self._validate_config()

        self.detector = None

    # ========================================================
    # CONFIGURATION VALIDATION
    # ========================================================

    def _validate_config(self):

        if not isinstance(self.input_dir, Path):
            raise TypeError("input_dir must be a Path.")

        if not isinstance(self.output_dir, Path):
            raise TypeError("output_dir must be a Path.")

        if not 0.0 < self.detection_confidence <= 1.0:
            raise ValueError(
                "detection_confidence must be between 0 and 1."
            )

        if self.model_selection not in (0, 1):
            raise ValueError(
                "model_selection must be 0 or 1."
            )

        if (
            not isinstance(self.frame_scale, (int, float))
            or isinstance(self.frame_scale, bool)
            or self.frame_scale <= 0
        ):
            raise ValueError(
                "frame_scale must be a positive number."
            )

        if (
            not isinstance(self.padding_fraction, (int, float))
            or isinstance(self.padding_fraction, bool)
            or self.padding_fraction < 0
        ):
            raise ValueError(
                "padding_fraction must be a non-negative number."
            )

        if self.image_format not in ("jpg", "jpeg", "png"):
            raise ValueError(
                "image_format must be jpg, jpeg, or png."
            )

        if self.max_faces is not None:
            if (
                not isinstance(self.max_faces, int)
                or isinstance(self.max_faces, bool)
                or self.max_faces < 1
            ):
                raise ValueError(
                    "max_faces must be a positive integer or None."
                )

        if not 0 <= self.duplicate_iou_threshold <= 1:
            raise ValueError(
                "duplicate_iou_threshold must be between 0 and 1."
            )

    # ========================================================
    # CREATE DETECTOR
    # ========================================================

    def _create_detector(self):

        return FaceDetector(
            min_detection_confidence=self.detection_confidence,
            model_selection=self.model_selection,
            frame_scale=self.frame_scale,
            max_faces=self.max_faces,
            use_dual_model=self.use_dual_model,
            duplicate_iou_threshold=self.duplicate_iou_threshold,
        )

    # ========================================================
    # CREATE TRACKER
    # ========================================================

    def _create_tracker(self):

        # A separate tracker is required for every video.
        return FaceTracker()

    # ========================================================
    # IMAGE VALIDATION
    # ========================================================

    @staticmethod
    def _is_valid_image(image):

        return (
            isinstance(image, np.ndarray)
            and image.ndim == 3
            and image.shape[2] == 3
            and image.shape[0] > 0
            and image.shape[1] > 0
            and image.dtype == np.uint8
        )

    # ========================================================
    # GET FRAME INDEX
    # ========================================================

    @staticmethod
    def _get_frame_index(image_path):

        import re

        match = re.search(
            r"frame_(\d+)",
            image_path.stem,
            flags=re.IGNORECASE
        )

        if match:
            return int(match.group(1))

        return None

    # ========================================================
    # LOAD ORIGINAL FRAME METADATA
    # ========================================================

    def _load_original_metadata(self, sequence_dir):

        metadata_path = sequence_dir / "metadata.json"

        if not metadata_path.exists():
            return {}

        try:

            with open(metadata_path, "r", encoding="utf-8") as file:
                metadata = json.load(file)

            if isinstance(metadata, list):
                rows = metadata

            elif isinstance(metadata, dict):
                rows = metadata.get("frames", [])

            else:
                return {}

            result = {}

            for row in rows:

                if not isinstance(row, dict):
                    continue

                frame_index = row.get("frame_index")

                if frame_index is None:
                    continue

                try:
                    frame_index = int(frame_index)
                except (ValueError, TypeError):
                    continue

                result[frame_index] = row

            return result

        except (OSError, json.JSONDecodeError) as error:

            if self.debug:
                print(
                    f"[WARNING] Could not load metadata: "
                    f"{metadata_path} ({error})"
                )

            return {}

    # ========================================================
    # GET TIMESTAMP
    # ========================================================

    @staticmethod
    def _get_timestamp(frame_index, original_metadata):

        row = original_metadata.get(frame_index, {})

        if not isinstance(row, dict):
            return None

        # Preserve the original timestamp field when available.
        return row.get(
            "timestamp",
            row.get("timestamp_seconds")
        )

    # ========================================================
    # PREPARE OUTPUT DIRECTORY
    # ========================================================

    def _prepare_output_directory(self, output_sequence):

        if output_sequence.exists():

            existing_images = [
                path
                for path in output_sequence.iterdir()
                if path.is_file()
                and path.suffix.lower() in (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
            ]

            metadata_path = output_sequence / "metadata.json"

            if existing_images and not self.allow_existing_output:

                raise FileExistsError(
                    f"Output already contains extracted faces: "
                    f"{output_sequence}\n"
                    f"Set allow_existing_output=True to overwrite."
                )

            if self.allow_existing_output:

                # Remove only previous generated face images.
                for image_path in existing_images:
                    image_path.unlink()

                # Remove previous metadata.
                if metadata_path.exists():
                    metadata_path.unlink()

        output_sequence.mkdir(
            parents=True,
            exist_ok=True
        )

    # ========================================================
    # CROP FACE
    # ========================================================

    def _crop_face(self, image, bbox):

        if not self._is_valid_image(image):
            return None

        if not isinstance(bbox, (list, tuple, np.ndarray)):
            return None

        if len(bbox) != 4:
            return None

        try:

            x, y, width, height = [
                float(value) for value in bbox
            ]

        except (TypeError, ValueError):
            return None

        if not np.all(
            np.isfinite([x, y, width, height])
        ):
            return None

        if width <= 0 or height <= 0:
            return None

        image_height, image_width = image.shape[:2]

        padding_x = width * self.padding_fraction
        padding_y = height * self.padding_fraction

        x1 = int(max(0, x - padding_x))
        y1 = int(max(0, y - padding_y))

        x2 = int(
            min(image_width, x + width + padding_x)
        )

        y2 = int(
            min(image_height, y + height + padding_y)
        )

        if x2 <= x1 or y2 <= y1:
            return None

        crop = image[y1:y2, x1:x2]

        if crop.size == 0:
            return None

        return crop

    # ========================================================
    # CREATE FRAME METADATA
    # ========================================================

    @staticmethod
    def _create_frame_metadata(
        source_image,
        frame_index,
        timestamp,
        detections=None,
        tracking_result=None,
        image_path=None,
        status="unknown",
        error=None,
    ):

        tracking_result = tracking_result or {}

        detected_bbox = tracking_result.get(
            "detected_bbox"
        )

        tracking_bbox = tracking_result.get(
            "bbox"
        )

        selected_face_index = tracking_result.get(
            "selected_face_index"
        )

        detected = (
            tracking_result.get("detected") is True
        )

        track_valid = (
            tracking_result.get("track_valid") is True
        )

        confidence = None

        if (
            detected
            and isinstance(selected_face_index, int)
            and detections is not None
            and 0 <= selected_face_index < len(detections)
        ):

            confidence = detections[
                selected_face_index
            ].get("confidence")

        return {
            "source_image": source_image,
            "frame_index": frame_index,
            "timestamp": timestamp,

            # Actual detector-matched face.
            "detected": detected,

            # Tracking information.
            "track_valid": track_valid,
            "tracking_status": tracking_result.get(
                "tracking_status"
            ),
            "selected_face_index": selected_face_index,

            # Original detector bounding box.
            "detected_bbox": detected_bbox,

            # Smoothed tracking bounding box.
            "tracking_bbox": tracking_bbox,

            "confidence": confidence,

            # Crop information.
            "crop_valid": image_path is not None,
            "image_path": image_path,
            "status": status,

            "error": error,
        }

    # ========================================================
    # PROCESS ONE FRAME
    # ========================================================

    def process_image(
        self,
        image_path,
        output_sequence,
        tracker,
        original_metadata,
        frame_position,
    ):

        image_path = Path(image_path)

        frame_index = self._get_frame_index(image_path)

        # Fallback is only for tracker ordering.
        # The original frame index remains None in metadata.
        tracker_frame_index = (
            frame_index
            if frame_index is not None
            else frame_position
        )

        timestamp = (
            self._get_timestamp(
                frame_index,
                original_metadata
            )
            if frame_index is not None
            else None
        )

        image = cv2.imread(
            str(image_path),
            cv2.IMREAD_COLOR
        )

        # ----------------------------------------------------
        # Handle unreadable images
        # ----------------------------------------------------

        if not self._is_valid_image(image):

            tracking_result = tracker.track_frame({
                "frame_idx": tracker_frame_index,
                "faces": [],
            })

            return self._create_frame_metadata(
                source_image=image_path.name,
                frame_index=frame_index,
                timestamp=timestamp,
                tracking_result=tracking_result,
                status="unreadable_image",
                error="Could not read a valid image.",
            )

        # ----------------------------------------------------
        # Detect faces
        # ----------------------------------------------------

        try:

            detections = self.detector.detect_frame(image)

            if not isinstance(detections, list):
                raise TypeError(
                    "FaceDetector.detect_frame() must return a list."
                )

        except Exception as error:

            # Count this frame as a missed detection.
            tracking_result = tracker.track_frame({
                "frame_idx": tracker_frame_index,
                "faces": [],
            })

            return self._create_frame_metadata(
                source_image=image_path.name,
                frame_index=frame_index,
                timestamp=timestamp,
                tracking_result=tracking_result,
                status="detection_error",
                error=str(error),
            )

        # ----------------------------------------------------
        # Track the selected face
        # ----------------------------------------------------

        tracking_result = tracker.track_frame({
            "frame_idx": tracker_frame_index,
            "faces": detections,
        })

        if not isinstance(tracking_result, dict):
            raise TypeError(
                "FaceTracker.track_frame() must return a dictionary."
            )

        # ----------------------------------------------------
        # Validate actual detection
        # ----------------------------------------------------

        detected = (
            tracking_result.get("detected") is True
        )

        track_valid = (
            tracking_result.get("track_valid") is True
        )

        selected_face_index = tracking_result.get(
            "selected_face_index"
        )

        # A predicted tracking position is NOT a valid detection.
        if not detected or not track_valid:

            return self._create_frame_metadata(
                source_image=image_path.name,
                frame_index=frame_index,
                timestamp=timestamp,
                detections=detections,
                tracking_result=tracking_result,
                status="no_valid_detection",
            )

        # ----------------------------------------------------
        # Validate selected detection
        # ----------------------------------------------------

        if (
            not isinstance(selected_face_index, int)
            or isinstance(selected_face_index, bool)
            or not 0 <= selected_face_index < len(detections)
        ):

            return self._create_frame_metadata(
                source_image=image_path.name,
                frame_index=frame_index,
                timestamp=timestamp,
                detections=detections,
                tracking_result=tracking_result,
                status="invalid_selected_face",
                error="Tracker returned an invalid face index.",
            )

        selected_detection = detections[
            selected_face_index
        ]

        if selected_detection.get("detected") is not True:

            return self._create_frame_metadata(
                source_image=image_path.name,
                frame_index=frame_index,
                timestamp=timestamp,
                detections=detections,
                tracking_result=tracking_result,
                status="selected_face_not_detected",
            )

        # ----------------------------------------------------
        # Validate bounding boxes
        # ----------------------------------------------------

        detected_bbox = tracking_result.get(
            "detected_bbox"
        )

        tracking_bbox = tracking_result.get(
            "bbox"
        )

        if detected_bbox is None or tracking_bbox is None:

            return self._create_frame_metadata(
                source_image=image_path.name,
                frame_index=frame_index,
                timestamp=timestamp,
                detections=detections,
                tracking_result=tracking_result,
                status="missing_bbox",
            )

        # ----------------------------------------------------
        # Crop using smoothed tracking bounding box
        # ----------------------------------------------------

        crop = self._crop_face(
            image,
            tracking_bbox
        )

        if crop is None:

            return self._create_frame_metadata(
                source_image=image_path.name,
                frame_index=frame_index,
                timestamp=timestamp,
                detections=detections,
                tracking_result=tracking_result,
                status="invalid_crop",
            )

        # ----------------------------------------------------
        # Save crop
        # ----------------------------------------------------

        output_name = (
            f"frame_{tracker_frame_index:05d}."
            f"{self.image_format}"
        )

        face_output_path = (
            output_sequence / output_name
        )

        saved = cv2.imwrite(
            str(face_output_path),
            crop
        )

        if not saved:

            return self._create_frame_metadata(
                source_image=image_path.name,
                frame_index=frame_index,
                timestamp=timestamp,
                detections=detections,
                tracking_result=tracking_result,
                status="save_failed",
                error="cv2.imwrite() returned False.",
            )

        return self._create_frame_metadata(
            source_image=image_path.name,
            frame_index=frame_index,
            timestamp=timestamp,
            detections=detections,
            tracking_result=tracking_result,
            image_path=output_name,
            status="success",
        )

    # ========================================================
    # PROCESS ONE VIDEO SEQUENCE
    # ========================================================

    def process_sequence(
        self,
        sequence_dir,
        output_sequence,
    ):

        sequence_dir = Path(sequence_dir)
        output_sequence = Path(output_sequence)

        if not sequence_dir.is_dir():
            raise FileNotFoundError(
                f"Sequence directory not found: {sequence_dir}"
            )

        frame_paths = sorted(
            [
                path
                for path in sequence_dir.iterdir()
                if path.is_file()
                and path.suffix.lower() in (
                    ".jpg",
                    ".jpeg",
                    ".png"
                )
                and path.name.lower().startswith("frame_")
            ]
        )

        if not frame_paths:
            raise ValueError(
                f"No extracted frames found: {sequence_dir}"
            )

        self._prepare_output_directory(
            output_sequence
        )

        original_metadata = self._load_original_metadata(
            sequence_dir
        )

        # A fresh tracker for every video sequence.
        tracker = self._create_tracker()

        frame_metadata = []

        successful = 0
        failed = 0
        detected_count = 0
        tracking_valid_count = 0

        print(
            f"\n[SEQUENCE] {sequence_dir.name}"
        )

        print(
            f"Frames found: {len(frame_paths)}"
        )

        for position, frame_path in enumerate(frame_paths):

            try:

                metadata = self.process_image(
                    image_path=frame_path,
                    output_sequence=output_sequence,
                    tracker=tracker,
                    original_metadata=original_metadata,
                    frame_position=position,
                )

            except Exception as error:

                # Unexpected frame-level error.
                # Do not silently treat it as a valid detection.
                metadata = self._create_frame_metadata(
                    source_image=frame_path.name,
                    frame_index=self._get_frame_index(frame_path),
                    timestamp=None,
                    status="processing_error",
                    error=str(error),
                )

                if self.debug:
                    print(
                        f"[ERROR] {frame_path.name}: {error}"
                    )

            frame_metadata.append(metadata)

            if metadata["status"] == "success":
                successful += 1
            else:
                failed += 1

            if metadata["detected"]:
                detected_count += 1

            if metadata["track_valid"]:
                tracking_valid_count += 1

            if self.debug and (
                position + 1
            ) % 50 == 0:

                print(
                    f"Processed {position + 1}/"
                    f"{len(frame_paths)} frames"
                )

        # ----------------------------------------------------
        # Save sequence metadata
        # ----------------------------------------------------

        metadata_path = (
            output_sequence / "metadata.json"
        )

        metadata_document = {
            "sequence": sequence_dir.name,
            "source_directory": str(sequence_dir),
            "total_frames": len(frame_paths),
            "successful_crops": successful,
            "failed_frames": failed,
            "detected_frames": detected_count,
            "tracking_valid_frames": tracking_valid_count,
            "frames": frame_metadata,
        }

        with open(
            metadata_path,
            "w",
            encoding="utf-8"
        ) as file:

            json.dump(
                metadata_document,
                file,
                indent=4,
                allow_nan=False
            )

        print("\n[SEQUENCE SUMMARY]")

        print(f"Total frames: {len(frame_paths)}")
        print(f"Successful crops: {successful}")
        print(f"Failed frames: {failed}")
        print(f"Actual detections: {detected_count}")
        print(f"Tracking-valid frames: {tracking_valid_count}")

        print(
            f"Metadata saved: {metadata_path}"
        )

        return metadata_document

    # ========================================================
    # PROCESS DATASET
    # ========================================================

    def extract_dataset(self):

        if not self.input_dir.is_dir():
            raise FileNotFoundError(
                f"Input directory not found: {self.input_dir}"
            )

        self.output_dir.mkdir(
            parents=True,
            exist_ok=True
        )

        self.detector = self._create_detector()

        total_sequences = 0
        successful_sequences = 0
        failed_sequences = 0

        try:

            for split in ("train", "val", "test"):

                split_input = self.input_dir / split
                split_output = self.output_dir / split

                if not split_input.is_dir():

                    if self.debug:
                        print(
                            f"[SKIPPED] Missing split: {split_input}"
                        )

                    continue

                for label in ("real", "fake"):

                    label_input = split_input / label
                    label_output = split_output / label

                    if not label_input.is_dir():

                        if self.debug:
                            print(
                                f"[SKIPPED] Missing directory: "
                                f"{label_input}"
                            )

                        continue

                    sequences = sorted(
                        [
                            path
                            for path in label_input.iterdir()
                            if path.is_dir()
                        ]
                    )

                    for sequence_dir in sequences:

                        total_sequences += 1

                        output_sequence = (
                            label_output / sequence_dir.name
                        )

                        print("\n================================")
                        print(
                            f"SPLIT: {split} | LABEL: {label}"
                        )
                        print(
                            f"VIDEO: {sequence_dir.name}"
                        )
                        print("================================")

                        try:

                            self.process_sequence(
                                sequence_dir=sequence_dir,
                                output_sequence=output_sequence,
                            )

                            successful_sequences += 1

                        except Exception as error:

                            failed_sequences += 1

                            print(
                                f"[FAILED] {sequence_dir}"
                            )

                            print(
                                f"Reason: {error}"
                            )

        finally:

            if self.detector is not None:

                self.detector.close()
                self.detector = None

        print("\n================================")
        print(" FACE EXTRACTION SUMMARY")
        print("================================")

        print(f"Total sequences: {total_sequences}")
        print(
            f"Successful sequences: {successful_sequences}"
        )
        print(
            f"Failed sequences: {failed_sequences}"
        )

        print(
            f"Output directory: {self.output_dir.resolve()}"
        )

        return {
            "total_sequences": total_sequences,
            "successful_sequences": successful_sequences,
            "failed_sequences": failed_sequences,
        }


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    extractor = FaceExtractor(
        input_dir="extracted_frames",
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
        duplicate_iou_threshold=0.5,
    )

    extractor.extract_dataset()


    #rsync -av --ignore-existing detected_faces/ extracted_faces/ --tranfring faces