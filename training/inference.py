import os
import argparse
import torch
import numpy as np

from preprocessing.face_detection import FaceDetector
from preprocessing.face_tracking import FaceTracker
from preprocessing.rgb_extraction import RGBExtractor
from preprocessing.rppg import pos_method
from preprocessing.signal_quality import evaluate_signal_quality
from model.rppg_model import RPPG1DCNN

def run_inference(video_path: str, checkpoint_path: str) -> dict:
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video file not found: {video_path}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Preprocessing
    detector = FaceDetector()
    summary = detector.process_video(video_path)
    detector.close()

    tracker = FaceTracker()
    tracked_dets = tracker.track(summary["detections"])

    extractor = RGBExtractor()
    rgb_signals = extractor.process_video_frames(video_path, tracked_dets)

    fps = summary["fps"]
    if len(rgb_signals) < 10:
        rppg_wave = np.zeros(300, dtype=np.float32)
    else:
        rppg_wave = pos_method(rgb_signals, fps=fps)

    signal_quality = evaluate_signal_quality(rppg_wave, fps=fps)

    # Dominant Frequency & Heart Rate Calculation
    freqs = np.fft.rfftfreq(len(rppg_wave), d=1.0 / fps)
    fft_vals = np.abs(np.fft.rfft(rppg_wave))
    phys_mask = (freqs >= 0.7) & (freqs <= 2.5)
    
    if np.any(phys_mask):
        dom_freq = float(freqs[phys_mask][np.argmax(fft_vals[phys_mask])])
        estimated_hr = float(dom_freq * 60.0)
    else:
        dom_freq = 0.0
        estimated_hr = 0.0

    # Model Evaluation
    model = RPPG1DCNN().to(device)
    if os.path.exists(checkpoint_path):
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()

    if len(rppg_wave) < 300:
        wave_padded = np.pad(rppg_wave, (0, 300 - len(rppg_wave)), mode='constant')
    else:
        wave_padded = rppg_wave[:300]

    input_tensor = torch.tensor(wave_padded, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(input_tensor).squeeze()
        score = torch.sigmoid(logits).item()

    prediction = "FAKE" if score >= 0.5 else "REAL"

    results = {
        "prediction": prediction,
        "score": float(score),
        "signal_quality": float(signal_quality),
        "dominant_frequency_hz": float(dom_freq),
        "estimated_hr_bpm": float(estimated_hr)
    }

    print("=" * 50)
    print("INFERENCE RESULTS")
    print("=" * 50)
    print(f"Prediction         : {results['prediction']}")
    print(f"Fake Score         : {results['score']:.4f}")
    print(f"Signal Quality     : {results['signal_quality']:.4f}")
    print(f"Dominant Frequency : {results['dominant_frequency_hz']:.2f} Hz")
    print(f"Estimated Heart Rate: {results['estimated_hr_bpm']:.1f} BPM")
    print("=" * 50)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FALSE FACET - Video Inference")
    parser.add_argument("--video", type=str, required=True, help="Path to input video file")
    parser.add_argument("--checkpoint", type=str, default="model/checkpoints/best_rppg_model.pt")
    args = parser.parse_args()

    run_inference(args.video, args.checkpoint)