import argparse
import shutil
import subprocess
import tempfile

from pathlib import Path

import cv2
import numpy as np
import torch

from scipy.signal import resample

from preprocessing.face_detection import FaceDetector
from preprocessing.face_tracking import FaceTracker
from preprocessing.rgb_extraction import RGBExtractor
from preprocessing.rppg import pos_method

from model.rppg_model import RPPG1DCNN


# =========================================================
# CONFIGURATION
# =========================================================

TARGET_LENGTH = 300

WINDOW_SECONDS = 10.0

MIN_VALID_RGB_FRACTION = 0.80

MIN_VALID_RGB_FRAMES = 10

REAL_LABEL = 0
FAKE_LABEL = 1

PREDICTION_THRESHOLD = 0.50


# =========================================================
# DEVICE
# =========================================================

def get_device():

    if torch.cuda.is_available():

        device = torch.device("cuda")

        print(
            f"[DEVICE] Using CUDA: "
            f"{torch.cuda.get_device_name(0)}"
        )

    else:

        device = torch.device("cpu")

        print("[DEVICE] CUDA unavailable. Using CPU.")

    return device


# =========================================================
# LOAD MODEL
# =========================================================

def load_model(checkpoint_path, device):

    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.is_file():

        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}"
        )

    print("\n[MODEL] Loading RPPG1DCNN...")

    model = RPPG1DCNN()

    try:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True,
        )

    except TypeError:

        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
        )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            state_dict = checkpoint["model_state_dict"]

        elif "state_dict" in checkpoint:

            state_dict = checkpoint["state_dict"]

        else:

            state_dict = checkpoint

    else:

        state_dict = checkpoint

    cleaned_state_dict = {}

    for key, value in state_dict.items():

        if key.startswith("module."):

            key = key.replace("module.", "", 1)

        cleaned_state_dict[key] = value

    model.load_state_dict(cleaned_state_dict)

    model.to(device)

    model.eval()

    print("[MODEL] Checkpoint loaded successfully.")

    return model


# =========================================================
# OPEN VIDEO WITH AUTOMATIC CODEC FALLBACK
# =========================================================

def open_video_with_fallback(video_path, temporary_directory):

    video_path = Path(video_path)

    print("\n[VIDEO] Opening original video...")

    cap = cv2.VideoCapture(str(video_path))

    if cap.isOpened():

        ret, first_frame = cap.read()

        if ret and first_frame is not None:

            print("[VIDEO] Original video decoded successfully.")

            return cap, first_frame, False

    cap.release()

    print(
        "[VIDEO WARNING] OpenCV could not decode "
        "the original video."
    )

    print("[VIDEO] Checking FFmpeg availability...")

    ffmpeg_path = shutil.which("ffmpeg")

    if ffmpeg_path is None:

        raise RuntimeError(
            "FFmpeg is not installed or not available in PATH. "
            "Install FFmpeg to enable automatic video conversion."
        )

    converted_path = (
        Path(temporary_directory)
        / "temporary_converted_video.mp4"
    )

    print("[VIDEO] Attempting automatic H.264 conversion...")

    command = [
        ffmpeg_path,
        "-hide_banner",
        "-loglevel", "error",
        "-y",
        "-hwaccel", "none",
        "-i", str(video_path),
        "-map", "0:v:0",
        "-an",
        "-c:v", "libx264",
        "-preset", "ultrafast",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        "-movflags", "+faststart",
        str(converted_path),
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:

        raise RuntimeError(
            "Automatic H.264 conversion failed.\n"
            f"FFmpeg error: {result.stderr[-2000:]}"
        )

    if not converted_path.is_file():

        raise RuntimeError(
            "FFmpeg did not produce a converted video."
        )

    if converted_path.stat().st_size == 0:

        raise RuntimeError(
            "The converted video is empty."
        )

    print("[VIDEO] H.264 conversion completed.")

    cap = cv2.VideoCapture(str(converted_path))

    if not cap.isOpened():

        cap.release()

        raise RuntimeError(
            "OpenCV could not open the converted H.264 video."
        )

    ret, first_frame = cap.read()

    if not ret or first_frame is None:

        cap.release()

        raise RuntimeError(
            "OpenCV could not read frames from the converted video."
        )

    print("[VIDEO] Converted video decoded successfully.")

    return cap, first_frame, True


# =========================================================
# RGB QUALITY CHECK
# =========================================================

def validate_rgb_window(rgb_window):

    rgb_window = np.asarray(
        rgb_window,
        dtype=np.float32,
    )

    if rgb_window.ndim != 2 or rgb_window.shape[1] != 3:

        return False, 0.0, "Invalid RGB shape"

    total_frames = len(rgb_window)

    if total_frames < MIN_VALID_RGB_FRAMES:

        return False, 0.0, "Insufficient frames"

    valid_mask = np.all(
        np.isfinite(rgb_window),
        axis=1,
    )

    valid_frames = int(
        np.count_nonzero(valid_mask)
    )

    valid_fraction = valid_frames / total_frames

    if valid_fraction < MIN_VALID_RGB_FRACTION:

        return (
            False,
            valid_fraction,
            "RGB validity below 80%",
        )

    return True, valid_fraction, "Valid"


# =========================================================
# RGB TO FIXED-LENGTH POS SIGNAL
# =========================================================

def prepare_pos_window(rgb_window, fps):

    valid, valid_fraction, reason = validate_rgb_window(
        rgb_window
    )

    if not valid:

        raise ValueError(
            f"RGB window rejected: {reason} "
            f"({valid_fraction:.2%} valid)"
        )

    pos_signal = pos_method(
        rgb_window,
        fps=fps,
    )

    pos_signal = np.asarray(
        pos_signal,
        dtype=np.float32,
    ).reshape(-1)

    if len(pos_signal) < MIN_VALID_RGB_FRAMES:

        raise ValueError(
            "POS signal is too short."
        )

    if not np.all(np.isfinite(pos_signal)):

        raise ValueError(
            "POS signal contains invalid values."
        )

    fixed_signal = resample(
        pos_signal,
        TARGET_LENGTH,
    ).astype(np.float32)

    if fixed_signal.shape != (TARGET_LENGTH,):

        raise ValueError(
            "Incorrect POS signal length."
        )

    if not np.all(np.isfinite(fixed_signal)):

        raise ValueError(
            "Resampled POS signal contains invalid values."
        )

    return fixed_signal, valid_fraction


# =========================================================
# MODEL PREDICTION
# =========================================================

def predict_window(model, signal, device):

    signal_tensor = torch.from_numpy(
        signal
    ).float().unsqueeze(0)

    signal_tensor = signal_tensor.to(device)

    with torch.inference_mode():

        logits = model(signal_tensor)

        probability = torch.sigmoid(
            logits
        ).item()

    predicted_label = (
        FAKE_LABEL
        if probability >= PREDICTION_THRESHOLD
        else REAL_LABEL
    )

    return predicted_label, probability


# =========================================================
# PROCESS ONE VIDEO
# =========================================================

def run_inference(
    video_path,
    checkpoint_path,
    window_seconds=WINDOW_SECONDS,
):

    video_path = Path(video_path)

    if not video_path.is_file():

        raise FileNotFoundError(
            f"Video not found: {video_path}"
        )

    # The model was trained using 10-second windows.
    # Do not silently change the inference duration.

    if not np.isclose(window_seconds, WINDOW_SECONDS):

        raise ValueError(
            "This checkpoint expects 10-second windows. "
            "Train and validate a different window duration "
            "before changing this setting."
        )

    print("\n========================================")
    print("       FALSEFACET rPPG INFERENCE")
    print("========================================")

    print(f"Video: {video_path}")

    # Temporary conversion directory.
    # Automatically deleted after inference.

    temporary_directory = tempfile.TemporaryDirectory(
        prefix="falsefacet_"
    )

    cap = None
    detector = None
    rgb_extractor = None

    try:

        # -------------------------------------------------
        # OPEN VIDEO
        # -------------------------------------------------

        cap, first_frame, converted = open_video_with_fallback(
            video_path,
            temporary_directory.name,
        )

        if converted:

            print(
                "[VIDEO] Using temporary H.264 copy."
            )

        # -------------------------------------------------
        # VIDEO INFORMATION
        # -------------------------------------------------

        fps = cap.get(
            cv2.CAP_PROP_FPS
        )

        if not np.isfinite(fps) or fps <= 0:

            raise ValueError(
                "Could not determine video FPS."
            )

        total_frames = int(
            cap.get(cv2.CAP_PROP_FRAME_COUNT)
        )

        frames_per_window = int(
            round(window_seconds * fps)
        )

        if frames_per_window < MIN_VALID_RGB_FRAMES:

            raise ValueError(
                "Window is too short for rPPG processing."
            )

        estimated_duration = (
            total_frames / fps
            if total_frames > 0
            else None
        )

        print("\n[VIDEO INFORMATION]")

        print(f"FPS: {fps:.3f}")

        print(f"Total frames: {total_frames}")

        print(
            f"Window duration: {window_seconds:.2f} seconds"
        )

        print(
            f"Frames per window: {frames_per_window}"
        )

        if estimated_duration is not None:

            print(
                f"Estimated video duration: "
                f"{estimated_duration:.2f} seconds"
            )

            # Reject short videos before loading the model
            # or running expensive face detection.

            if estimated_duration < window_seconds:

                print("\n[VIDEO REJECTED]")

                print(
                    f"This video is only "
                    f"{estimated_duration:.2f} seconds long."
                )

                print(
                    f"Your current rPPG model requires "
                    f"at least {window_seconds:.2f} seconds."
                )

                print(
                    "No model prediction was generated."
                )

                return {
                    "prediction": "UNAVAILABLE",
                    "reason": "VIDEO_TOO_SHORT",
                    "video_duration": estimated_duration,
                    "required_duration": window_seconds,
                    "windows": [],
                }

        # -------------------------------------------------
        # DEVICE AND MODEL
        # -------------------------------------------------

        device = get_device()

        model = load_model(
            checkpoint_path,
            device,
        )

        # -------------------------------------------------
        # INITIALIZE COMPONENTS
        # -------------------------------------------------

        detector = FaceDetector()

        tracker = FaceTracker(
            debug=False,
        )

        rgb_extractor = RGBExtractor(
            debug=False,
        )

        # -------------------------------------------------
        # TRACKING AND RGB EXTRACTION
        # -------------------------------------------------

        window_rgb = []

        window_results = []

        frame_idx = 0

        total_valid_rgb = 0

        total_invalid_rgb = 0

        print("\n[PROCESSING] Reading video frames...")

        # Process the first frame already read during
        # codec validation, then continue reading normally.

        pending_frame = first_frame

        while True:

            if pending_frame is not None:

                frame = pending_frame

                pending_frame = None

            else:

                ret, frame = cap.read()

                if not ret:

                    break

            # ---------------------------------------------
            # 1. DETECT FACES
            # ---------------------------------------------

            detected_faces = detector.detect_frame(
                frame
            )

            detection_record = {
                "frame_idx": frame_idx,
                "faces": detected_faces,
            }

            # ---------------------------------------------
            # 2. TRACK FACE
            # ---------------------------------------------

            tracking_record = tracker.track_frame(
                detection_record
            )

            # ---------------------------------------------
            # 3. EXTRACT RGB
            # ---------------------------------------------

            rgb = rgb_extractor.process_frame(
                frame,
                tracking_record,
            )

            rgb = np.asarray(
                rgb,
                dtype=np.float32,
            )

            if (
                rgb.shape == (3,)
                and np.all(np.isfinite(rgb))
            ):

                total_valid_rgb += 1

            else:

                total_invalid_rgb += 1

            # Preserve invalid frames as well.
            window_rgb.append(rgb)

            frame_idx += 1

            # ---------------------------------------------
            # 4. PROCESS COMPLETE WINDOW
            # ---------------------------------------------

            if len(window_rgb) == frames_per_window:

                current_window = np.asarray(
                    window_rgb,
                    dtype=np.float32,
                )

                window_number = len(window_results) + 1

                print(
                    f"\n[WINDOW {window_number}] "
                    f"Processing {len(current_window)} frames..."
                )

                try:

                    signal, valid_fraction = prepare_pos_window(
                        current_window,
                        fps,
                    )

                    predicted_label, probability = predict_window(
                        model,
                        signal,
                        device,
                    )

                    prediction_name = (
                        "FAKE"
                        if predicted_label == FAKE_LABEL
                        else "REAL"
                    )

                    result = {
                        "window": window_number,
                        "start_frame": (
                            frame_idx - frames_per_window
                        ),
                        "end_frame": frame_idx - 1,
                        "valid_rgb_fraction": valid_fraction,
                        "prediction": prediction_name,
                        "label": predicted_label,
                        "fake_probability": probability,
                    }

                    window_results.append(result)

                    print(
                        f"[WINDOW {window_number}] "
                        f"RGB validity: {valid_fraction:.2%}"
                    )

                    print(
                        f"[WINDOW {window_number}] "
                        f"Prediction: {prediction_name}"
                    )

                    print(
                        f"[WINDOW {window_number}] "
                        f"Fake probability: {probability:.4f}"
                    )

                except Exception as exc:

                    print(
                        f"[WINDOW {window_number}] "
                        f"REJECTED: {exc}"
                    )

                    window_results.append({
                        "window": window_number,
                        "prediction": "UNAVAILABLE",
                        "error": str(exc),
                    })

                window_rgb = []

                del current_window

        # -------------------------------------------------
        # FINAL FRAME INFORMATION
        # -------------------------------------------------

        remaining_frames = len(window_rgb)

        if remaining_frames > 0:

            print(
                f"\n[VIDEO] Ignored {remaining_frames} "
                f"remaining frames because they do not "
                f"form a complete 10-second window."
            )

        # -------------------------------------------------
        # FINAL RESULTS
        # -------------------------------------------------

        print("\n========================================")
        print("             INFERENCE RESULTS")
        print("========================================")

        print(f"Frames processed: {frame_idx}")

        print(f"Valid RGB frames: {total_valid_rgb}")

        print(f"Invalid RGB frames: {total_invalid_rgb}")

        if frame_idx:

            overall_rgb_validity = (
                total_valid_rgb / frame_idx
            )

            print(
                f"Overall RGB validity: "
                f"{overall_rgb_validity:.2%}"
            )

        valid_results = [
            result
            for result in window_results
            if result.get("prediction") in ("REAL", "FAKE")
        ]

        if not valid_results:

            print("\n[FINAL PREDICTION] UNAVAILABLE")

            print(
                "No complete window passed the quality checks."
            )

            return {
                "prediction": "UNAVAILABLE",
                "reason": "NO_VALID_WINDOWS",
                "windows": window_results,
            }

        # -------------------------------------------------
        # AGGREGATE PREDICTIONS
        # -------------------------------------------------

        probabilities = np.asarray(
            [
                result["fake_probability"]
                for result in valid_results
            ],
            dtype=np.float32,
        )

        mean_fake_probability = float(
            np.mean(probabilities)
        )

        overall_prediction = (
            "FAKE"
            if mean_fake_probability >= PREDICTION_THRESHOLD
            else "REAL"
        )

        # -------------------------------------------------
        # WINDOW RESULTS
        # -------------------------------------------------

        print("\n[WINDOW PREDICTIONS]")

        for result in window_results:

            if result.get("prediction") in ("REAL", "FAKE"):

                print(
                    f"Window {result['window']}: "
                    f"{result['prediction']} | "
                    f"Fake probability: "
                    f"{result['fake_probability']:.4f}"
                )

            else:

                print(
                    f"Window {result['window']}: "
                    f"UNAVAILABLE"
                )

        # -------------------------------------------------
        # FINAL PREDICTION
        # -------------------------------------------------

        print("\n[FINAL PREDICTION]")

        print(
            f"Result: {overall_prediction}"
        )

        print(
            f"Average fake probability: "
            f"{mean_fake_probability:.4f}"
        )

        print(
            f"Successfully evaluated windows: "
            f"{len(valid_results)}"
        )

        print("========================================")

        return {
            "prediction": overall_prediction,
            "fake_probability": mean_fake_probability,
            "evaluated_windows": len(valid_results),
            "windows": window_results,
        }

    finally:

        if cap is not None:

            cap.release()

        if detector is not None:

            detector.close()

        if rgb_extractor is not None:

            rgb_extractor.close()

        # Delete temporary converted video.
        temporary_directory.cleanup()

        print("\n[CLEANUP] Temporary video files removed.")


# =========================================================
# COMMAND LINE
# =========================================================

def main():

    parser = argparse.ArgumentParser(
        description="FALSEFACET rPPG video inference"
    )

    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to input video",
    )

    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model/checkpoints/best_rppg_model.pt",
        help="Path to trained RPPG1DCNN checkpoint",
    )

    parser.add_argument(
        "--window_seconds",
        type=float,
        default=WINDOW_SECONDS,
        help="Inference window duration (must be 10 seconds)",
    )

    args = parser.parse_args()

    run_inference(
        video_path=args.video,
        checkpoint_path=args.checkpoint,
        window_seconds=args.window_seconds,
    )


if __name__ == "__main__":

    main()