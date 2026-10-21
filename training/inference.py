
import os
import argparse

import numpy as np
import torch

from preprocessing.face_detection import FaceDetector
from preprocessing.face_tracking import FaceTracker
from preprocessing.rgb_extraction import RGBExtractor
from preprocessing.rppg import pos_method
from preprocessing.signal_quality import evaluate_signal_quality
from model.rppg_model import RPPG1DCNN


TARGET_LENGTH = 300
POS_WINDOW_SECONDS = 1.6

# Set this only after confirming that evaluate_signal_quality
# returns a score between 0 and 1 and validating a threshold.
QUALITY_THRESHOLD = None


def load_checkpoint(model, checkpoint_path, device):
    """Load a plain state_dict or a checkpoint containing model_state_dict."""
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device,
            weights_only=True
        )
    except TypeError:
        # Compatibility fallback for older PyTorch versions.
        # Only load checkpoints you trust.
        checkpoint = torch.load(
            checkpoint_path,
            map_location=device
        )

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    return model


def run_inference(video_path: str, checkpoint_path: str) -> dict:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(
            f"Video file not found: {video_path}"
        )

    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"Trained model checkpoint not found: {checkpoint_path}"
        )

    device = torch.device(
        "cuda:0" if torch.cuda.is_available() else "cpu"
    )
    print(f"Using device: {device}")

    # 1. Detect faces in the video.
    detector = FaceDetector()

    try:
        summary = detector.process_video(video_path)
    finally:
        detector.close()

    fps = float(summary.get("fps", 0.0))

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("The video has an invalid frame rate.")

    detections = summary.get("detections", [])

    if not detections:
        raise ValueError("No face detections were found in the video.")

    # 2. Track the face across frames.
    tracker = FaceTracker()
    tracked_dets = tracker.track(detections)

    if not tracked_dets:
        raise ValueError("No usable face track was found.")

    # 3. Extract RGB measurements from the tracked face.
    extractor = RGBExtractor()

    rgb_signals = np.asarray(
        extractor.process_video_frames(video_path, tracked_dets),
        dtype=np.float64
    )

    if (
        rgb_signals.ndim != 2
        or rgb_signals.shape[1] != 3
        or not np.all(np.isfinite(rgb_signals))
    ):
        raise ValueError(
            "RGB extraction must return finite values with shape (T, 3)."
        )

    # Check the minimum length needed for the POS window.
    # This assumes one RGB sample per video frame.
    min_samples = int(np.ceil(POS_WINDOW_SECONDS * fps))

    if len(rgb_signals) < min_samples:
        raise ValueError(
            f"Insufficient RGB samples: {len(rgb_signals)}. "
            f"Approximately {min_samples} samples are needed for a "
            f"{POS_WINDOW_SECONDS}-second POS window at {fps:.2f} FPS."
        )

    # 4. Generate the rPPG waveform.
    rppg_wave = np.asarray(
        pos_method(rgb_signals, fps=fps),
        dtype=np.float32
    )

    if (
        rppg_wave.ndim != 1
        or len(rppg_wave) < TARGET_LENGTH
        or not np.all(np.isfinite(rppg_wave))
    ):
        raise ValueError(
            f"Invalid or insufficient rPPG signal. Expected at least "
            f"{TARGET_LENGTH} finite samples, received {len(rppg_wave)}."
        )

    # Use the same first TARGET_LENGTH samples for all downstream steps.
    wave = rppg_wave[:TARGET_LENGTH]

    # 5. Evaluate signal quality.
    signal_quality = float(
        evaluate_signal_quality(wave, fps=fps)
    )

    if not np.isfinite(signal_quality):
        raise ValueError(
            "Signal-quality evaluation returned a non-finite value."
        )

    # Validate the range only if your quality function defines it as 0–1.
    if QUALITY_THRESHOLD is not None:
        if not 0.0 <= QUALITY_THRESHOLD <= 1.0:
            raise ValueError("QUALITY_THRESHOLD must be between 0 and 1.")

        if not 0.0 <= signal_quality <= 1.0:
            raise ValueError(
                "Signal quality is outside 0–1; verify "
                "evaluate_signal_quality() before using a threshold."
            )

    # 6. Estimate dominant frequency and heart rate.
    # These estimates are not reliable when the signal is poor.
    centered_wave = wave - np.mean(wave)

    freqs = np.fft.rfftfreq(
        len(centered_wave),
        d=1.0 / fps
    )
    fft_vals = np.abs(np.fft.rfft(centered_wave))

    phys_mask = (freqs >= 0.7) & (freqs <= 2.5)

    if np.any(phys_mask):
        band_freqs = freqs[phys_mask]
        band_magnitudes = fft_vals[phys_mask]

        dom_freq = float(
            band_freqs[np.argmax(band_magnitudes)]
        )
        estimated_hr = dom_freq * 60.0
    else:
        dom_freq = 0.0
        estimated_hr = 0.0

    # 7. Load the trained model.
    model = RPPG1DCNN().to(device)
    model = load_checkpoint(model, checkpoint_path, device)
    model.eval()

    # 8. Prepare the model input.
    input_tensor = torch.from_numpy(
        wave.copy()
    ).float().unsqueeze(0).to(device)

    # 9. Predict only when the signal passes a configured quality threshold.
    score = None

    if (
        QUALITY_THRESHOLD is not None
        and signal_quality < QUALITY_THRESHOLD
    ):
        prediction = "INCONCLUSIVE"
    else:
        with torch.no_grad():
            logits = model(input_tensor)
            score = torch.sigmoid(logits).item()

        prediction = "FAKE" if score >= 0.5 else "REAL"

    results = {
        "prediction": prediction,
        "score": score,
        "signal_quality": signal_quality,
        "dominant_frequency_hz": dom_freq,
        "estimated_hr_bpm": float(estimated_hr),
        "fps": fps,
        "device": str(device)
    }

    print("=" * 50)
    print("FALSEFACET rPPG INFERENCE")
    print("=" * 50)
    print(f"Prediction          : {prediction}")

    if score is not None:
        print(f"Fake-class score    : {score:.4f}")
    else:
        print("Fake-class score    : Not available")

    print(f"Signal quality      : {signal_quality:.4f}")
    print(f"Dominant frequency  : {dom_freq:.2f} Hz")
    print(f"Estimated heart rate: {estimated_hr:.1f} BPM")
    print("=" * 50)

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FALSEFACET - rPPG Video Inference"
    )
    parser.add_argument(
        "--video",
        type=str,
        required=True,
        help="Path to the input video file"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model/checkpoints/best_rppg_model.pt",
        help="Path to the trained model checkpoint"
    )

    args = parser.parse_args()

    run_inference(args.video, args.checkpoint)

'''python -m training.inference \
    --video dataset/test/fake/example.mp4 \
    --checkpoint model/checkpoints/best_rppg_model.pt'''