import cv2
import json
import numpy as np

from pathlib import Path
from typing import Optional, Dict, Any

from preprocessing.face_detection import FaceDetector


class FaceExtractor:
    """
    Extract and save cropped face images from video frames.

    Workflow:
        1. Open video.
        2. Read frames sequentially.
        3. Detect faces using the existing FaceDetector.
        4. Select the face with the highest confidence.
        5. Crop the detected face with optional padding.
        6. Save the cropped face image.
        7. Save frame metadata, including frame number and timestamp.

    Bounding boxes use:
        [x, y, width, height]

    All bounding boxes refer to the original video frame.
    """

    def __init__(
        self,
        output_dir: str = "extracted_faces",
        detection_confidence: float = 0.3,
        model_selection: int = 1,
        frame_scale: float = 1.0,
        padding_fraction: float = 0.20,
        image_format: str = "jpg",
        debug: bool = True,
    ):

        # Validate configuration.
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
                "frame_scale must be finite and >= 1.0."
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
                "image_format must be 'jpg' or 'png'."
            )

        self.output_dir = Path(output_dir)

        self.detection_confidence = detection_confidence
        self.model_selection = model_selection
        self.frame_scale = frame_scale
        self.padding_fraction = padding_fraction
        self.image_format = image_format
        self.debug = debug

    # ---------------------------------------------------------
    # 1. Validate bounding box
    # ---------------------------------------------------------

    @staticmethod
    def _validate_bbox(bbox):

        if bbox is None:
            return None

        try:
            values = np.asarray(bbox, dtype=np.float64)

            if values.shape != (4,):
                return None

            if not np.isfinite(values).all():
                return None

            x, y, width, height = np.round(values).astype(int)

        except (TypeError, ValueError, OverflowError):
            return None

        if width <= 0 or height <= 0:
            return None

        return [x, y, width, height]

    # ---------------------------------------------------------
    # 2. Crop detected face
    # ---------------------------------------------------------

    def _crop_face(
        self,
        frame: np.ndarray,
        bbox,
    ) -> Optional[np.ndarray]:
        """
        Crop a face from the original video frame.

        Padding gives additional space around the detected face.
        """

        valid_bbox = self._validate_bbox(bbox)

        if valid_bbox is None:
            return None

        x, y, width, height = valid_bbox

        frame_h, frame_w = frame.shape[:2]

        # Calculate padding.
        pad_x = int(round(width * self.padding_fraction))
        pad_y = int(round(height * self.padding_fraction))

        # Calculate crop coordinates.
        x1 = max(0, x - pad_x)
        y1 = max(0, y - pad_y)

        x2 = min(frame_w, x + width + pad_x)
        y2 = min(frame_h, y + height + pad_y)

        # Make sure the crop is valid.
        if x2 <= x1 or y2 <= y1:
            return None

        # Extract face image.
        face_crop = frame[y1:y2, x1:x2]

        if (
            face_crop.size == 0
            or face_crop.ndim != 3
            or face_crop.shape[2] != 3
        ):
            return None

        return face_crop

    # ---------------------------------------------------------
    # 3. Extract faces from video
    # ---------------------------------------------------------

    def extract_video(
        self,
        video_path: str,
        output_name: Optional[str] = None,
    ) -> Dict[str, Any]:

        video_path = Path(video_path)

        if not video_path.is_file():
            raise FileNotFoundError(
                f"Video not found: {video_path}"
            )

        if output_name is None:
            output_name = video_path.stem

        # Create output folder.
        video_output_dir = self.output_dir / output_name

        video_output_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        # Open video.
        cap = cv2.VideoCapture(str(video_path))

        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open video: {video_path}"
            )

        # Get video information.
        fps = cap.get(cv2.CAP_PROP_FPS)

        if not np.isfinite(fps) or fps <= 0:
            fps = 30.0

        reported_total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        # Initialize your existing FaceDetector.
        detector = FaceDetector(
            min_detection_confidence=self.detection_confidence,
            model_selection=self.model_selection,
            frame_scale=self.frame_scale,
        )

        frame_index = 0
        saved_faces = 0
        missing_faces = 0
        failed_saves = 0

        metadata = []

        try:

            # Process video frame by frame.
            while True:

                success, frame = cap.read()

                if not success:
                    break

                timestamp = frame_index / fps

                # ---------------------------------------------
                # Detect faces in current frame.
                # ---------------------------------------------

                detections = detector.detect_frame(frame)

                if not detections:

                    missing_faces += 1

                    metadata.append({
                        "frame_index": frame_index,
                        "timestamp": timestamp,
                        "detected": False,
                        "bbox": None,
                        "confidence": 0.0,
                        "image_path": None,
                    })

                    frame_index += 1
                    continue

                # ---------------------------------------------
                # Select highest-confidence face.
                # ---------------------------------------------

                best_detection = max(
                    detections,
                    key=lambda item: item["confidence"],
                )

                bbox = best_detection["bbox"]

                confidence = float(
                    best_detection["confidence"]
                )

                # ---------------------------------------------
                # Crop face.
                # ---------------------------------------------

                face_crop = self._crop_face(
                    frame,
                    bbox,
                )

                if face_crop is None:

                    missing_faces += 1

                    metadata.append({
                        "frame_index": frame_index,
                        "timestamp": timestamp,
                        "detected": True,
                        "bbox": bbox,
                        "confidence": confidence,
                        "image_path": None,
                    })

                    frame_index += 1
                    continue

                # ---------------------------------------------
                # Save extracted face.
                # ---------------------------------------------

                filename = (
                    f"frame_{frame_index:06d}.{self.image_format}"
                )

                save_path = video_output_dir / filename

                saved = cv2.imwrite(
                    str(save_path),
                    face_crop,
                )

                if saved:

                    saved_faces += 1

                    metadata.append({
                        "frame_index": frame_index,
                        "timestamp": timestamp,
                        "detected": True,
                        "bbox": bbox,
                        "confidence": confidence,
                        "image_path": str(save_path),
                    })

                else:

                    failed_saves += 1

                    metadata.append({
                        "frame_index": frame_index,
                        "timestamp": timestamp,
                        "detected": True,
                        "bbox": bbox,
                        "confidence": confidence,
                        "image_path": None,
                    })

                frame_index += 1

        finally:

            cap.release()
            detector.close()

        # -----------------------------------------------------
        # 4. Save metadata
        # -----------------------------------------------------

        metadata_path = video_output_dir / "metadata.json"

        with open(
            metadata_path,
            "w",
            encoding="utf-8",
        ) as file:

            json.dump(
                metadata,
                file,
                indent=4,
            )

        # -----------------------------------------------------
        # 5. Generate summary
        # -----------------------------------------------------

        summary = {
            "video": str(video_path),
            "fps": float(fps),
            "total_frames": frame_index,
            "reported_total_frames": reported_total_frames,
            "saved_faces": saved_faces,
            "missing_faces": missing_faces,
            "failed_saves": failed_saves,
            "output_dir": str(video_output_dir),
            "metadata_path": str(metadata_path),
        }

        if self.debug:

            detection_rate = (
                saved_faces / frame_index * 100
                if frame_index > 0
                else 0.0
            )

            print(
                "\n========== FACE EXTRACTION SUMMARY =========="
            )

            print(f"Video: {video_path}")
            print(f"FPS: {fps:.2f}")
            print(f"Frames processed: {frame_index}")
            print(f"Faces saved: {saved_faces}")
            print(f"Missing faces: {missing_faces}")
            print(f"Failed saves: {failed_saves}")
            print(f"Successful extraction rate: {detection_rate:.2f}%")
            print(f"Output folder: {video_output_dir}")
            print(f"Metadata: {metadata_path}")

            print(
                "=============================================="
            )

        return summary

    # ---------------------------------------------------------
    # 6. Extract multiple videos
    # ---------------------------------------------------------

    def extract_multiple_videos(
        self,
        video_paths: list,
    ) -> list:

        results = []

        for video_path in video_paths:

            print(f"\nProcessing: {video_path}")

            result = self.extract_video(video_path)

            results.append(result)

        return results


# -------------------------------------------------------------
# Standalone testing
# -------------------------------------------------------------

if __name__ == "__main__":

    extractor = FaceExtractor(
        output_dir="extracted_faces",
        detection_confidence=0.3,
        model_selection=1,
        frame_scale=1.0,
        padding_fraction=0.20,
        image_format="jpg",
        debug=True,
    )

    video_path = "dataset/train/real/3213.mp4"

    extractor.extract_video(video_path)