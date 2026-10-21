
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple


def plot_frequency_spectrum(
    rppg_signal: np.ndarray,
    fps: float = 30.0,
    save_path: str = "results/frequency/power_spectrum.png",
    show_plot: bool = False
) -> Tuple[float, float]:
    """
    Plot the normalized FFT power spectrum of an rPPG signal.

    Returns:
        (dominant_frequency_hz, estimated_heart_rate_bpm)

    Note:
        This estimates a dominant pulse-like frequency. It does not
        independently establish whether a video is real or fake.
    """
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a finite positive number.")

    rppg_arr = np.asarray(rppg_signal, dtype=np.float64).ravel()

    if len(rppg_arr) == 0:
        raise ValueError("The rPPG signal is empty.")

    if not np.all(np.isfinite(rppg_arr)):
        raise ValueError("The rPPG signal contains NaN or infinite values.")

    if len(rppg_arr) < 2:
        raise ValueError("At least two signal samples are required.")

    # Remove the mean to reduce the zero-frequency component.
    rppg_arr = rppg_arr - np.mean(rppg_arr)

    # Apply a Hann window to reduce spectral leakage.
    window = np.hanning(len(rppg_arr))
    windowed_signal = rppg_arr * window

    # Compute FFT and power spectrum.
    fft_coeffs = np.fft.rfft(windowed_signal)
    freqs = np.fft.rfftfreq(len(windowed_signal), d=1.0 / fps)
    power = np.abs(fft_coeffs) ** 2

    max_power = float(np.max(power))
    if max_power > 0:
        power = power / max_power

    # Physiological band: 0.7–2.5 Hz (42–150 BPM).
    phys_mask = (freqs >= 0.7) & (freqs <= 2.5)

    if np.any(phys_mask) and max_power > 0:
        band_power = power[phys_mask]
        band_freqs = freqs[phys_mask]

        peak_idx = int(np.argmax(band_power))
        dom_freq = float(band_freqs[peak_idx])
        estimated_hr = dom_freq * 60.0
        peak_power = float(band_power[peak_idx])
    else:
        dom_freq = 0.0
        estimated_hr = 0.0
        peak_power = 0.0

    # Ensure the output directory exists, including when the path
    # is just a filename with no directory component.
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    plt.figure(figsize=(9, 4.5))
    plt.plot(
        freqs, power,
        linewidth=1.8,
        label="Normalized FFT power"
    )

    plt.axvspan(
        0.7, 2.5,
        alpha=0.15,
        label="Physiological band (42–150 BPM)"
    )

    if dom_freq > 0:
        plt.axvline(
            dom_freq,
            linestyle="--",
            linewidth=1.5,
            label=f"Peak: {dom_freq:.2f} Hz ({estimated_hr:.1f} BPM)"
        )
        plt.scatter([dom_freq], [peak_power], s=50, zorder=5)

    plt.title("FalseFacet rPPG Frequency Spectrum")
    plt.xlabel("Frequency (Hz)")
    plt.ylabel("Normalized power")
    plt.xlim(0.0, min(4.0, fps / 2.0))
    plt.ylim(0.0, 1.05)
    plt.grid(True, linestyle=":", alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches="tight")
    print(f"[+] Frequency spectrum saved to: {save_path}")

    if show_plot:
        plt.show()

    plt.close()
    return dom_freq, estimated_hr


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FalseFacet rPPG frequency spectrum visualizer"
    )
    parser.add_argument(
        "--save_path",
        default="results/frequency/power_spectrum.png"
    )
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    # Generate a synthetic 1.2 Hz pulse (72 BPM) for a smoke test.
    sample_count = int(10 * args.fps) if np.isfinite(args.fps) and args.fps > 0 else 0
    if sample_count < 2:
        parser.error("--fps must be positive and produce at least two samples.")

    t = np.arange(sample_count) / args.fps
    rng = np.random.default_rng(42)
    synthetic_signal = (
        np.sin(2 * np.pi * 1.2 * t)
        + 0.3 * rng.normal(size=sample_count)
    )

    peak_hz, hr_bpm = plot_frequency_spectrum(
        synthetic_signal,
        fps=args.fps,
        save_path=args.save_path
    )

    print(f"[+] Dominant frequency: {peak_hz:.2f} Hz")
    print(f"[+] Estimated heart rate: {hr_bpm:.1f} BPM")

#python -m visual.frequency_plot