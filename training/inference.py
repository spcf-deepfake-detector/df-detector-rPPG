
import os
import argparse

import cv2
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


def run_inference(video_path: str, checkpoint_path: str) -> dict:
    if not os.path.isfile(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

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

    # 2. Track the face.
    tracker = FaceTracker()
    tracked_dets = tracker.track(detections)

    # 3. Extract RGB values from the tracked face.
    extractor = RGBExtractor()
    rgb_signals = np.asarray(
        extractor.process_video_frames(video_path, tracked_dets),
        dtype=np.float64
    )

    if rgb_signals.ndim != 2 or rgb_signals.shape[1] != 3:
        raise ValueError(
            "RGB extraction must return an array with shape (T, 3)."
        )

    # POS needs sufficient temporal data for its sliding window.
    min_frames = int(np.ceil(POS_WINDOW_SECONDS * fps))

    if len(rgb_signals) < min_frames:
        raise ValueError(
            f"Insufficient RGB samples: got {len(rgb_signals)}, "
            f"but approximately {min_frames} are needed for a "
            f"{POS_WINDOW_SECONDS}-second POS window at {fps:.2f} FPS."
        )

    # 4. Generate the rPPG waveform.
    rppg_wave = np.asarray(
        pos_method(rgb_signals, fps=fps),
        dtype=np.float32
    )

    if (
        rppg_wave.ndim != 1
        or len(rppg_wave) < 2
        or not np.all(np.isfinite(rppg_wave))
    ):
        raise ValueError("POS produced an invalid rPPG signal.")

    # 5. Evaluate signal quality.
    signal_quality = float(
        evaluate_signal_quality(rppg_wave, fps=fps)
    )

    if not np.isfinite(signal_quality):
        raise ValueError("Signal quality evaluation returned an invalid value.")

    # 6. Estimate dominant frequency and heart rate.
    # These estimates may be unreliable when the signal is noisy.
    centered_wave = rppg_wave - np.mean(rppg_wave)

    freqs = np.fft.rfftfreq(
        len(centered_wave),
        d=1.0 / fps
    )
    fft_vals = np.abs(np.fft.rfft(centered_wave))

    # Exclude the zero-frequency component.
    phys_mask = (freqs >= 0.7) & (freqs <= 2.5)

    if np.any(phys_mask):
        band_freqs = freqs[phys_mask]
        band_power = fft_vals[phys_mask]

        dom_freq = float(band_freqs[np.argmax(band_power)])
        estimated_hr = dom_freq * 60.0
    else:
        dom_freq = 0.0
        estimated_hr = 0.0

    # 7. Load the trained model.
    model = RPPG1DCNN().to(device)

    state_dict = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=True
    )
    model.load_state_dict(state_dict)
    model.eval()

    # 8. Pad or truncate to the model's expected input length.
    if len(rppg_wave) < TARGET_LENGTH:
        wave = np.pad(
            rppg_wave,
            (0, TARGET_LENGTH - len(rppg_wave)),
            mode="constant"
        )
    else:
        wave = rppg_wave[:TARGET_LENGTH]

    input_tensor = torch.from_numpy(
        wave.astype(np.float32, copy=False)
    ).unsqueeze(0).to(device)

    # 9. Predict the FAKE-class score.
    with torch.no_grad():
        logits = model(input_tensor)
        score = torch.sigmoid(logits).item()

    prediction = "FAKE" if score >= 0.5 else "REAL"

    results = {
        "prediction": prediction,
        "score": float(score),
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
    print(f"Fake-class score    : {score:.4f}")
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
        help="Path to the input video"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="model/checkpoints/best_rppg_model.pt",
        help="Path to the trained model checkpoint"
    )

    args = parser.parse_args()

    run_inference(args.video, args.checkpoint)
#python -m training.inference \
#    --video dataset/test/fake/example.mp4 \
#    --checkpoint model/checkpoints/best_rppg_model.pt