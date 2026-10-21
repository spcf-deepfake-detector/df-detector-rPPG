
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt


def plot_signals(
    raw_rgb: np.ndarray,
    rppg_wave: np.ndarray,
    fps: float = 30.0,
    save_path: str = "results/signals/rppg_signals.png",
    show_plot: bool = False
) -> None:
    """
    Plot raw RGB channel means and an extracted rPPG waveform.

    Args:
        raw_rgb: Array of shape (T, 3), containing RGB channel means.
        rppg_wave: One-dimensional array of T rPPG samples.
        fps: Actual sampling rate of these signal samples, in Hz.
        save_path: Path for the output image.
        show_plot: Whether to display the plot interactively.
    """
    # Validate the sampling rate.
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a finite positive number.")

    if not save_path or not save_path.strip():
        raise ValueError("save_path cannot be empty.")

    # Convert inputs to numeric arrays.
    try:
        rgb_arr = np.asarray(raw_rgb, dtype=np.float64)
        wave_arr = np.asarray(rppg_wave, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("Signals must contain numeric values.") from exc

    # Validate RGB shape.
    if rgb_arr.ndim != 2 or rgb_arr.shape[1] != 3:
        raise ValueError(
            f"Expected raw_rgb with shape (T, 3), got {rgb_arr.shape}."
        )

    # Accept only a 1D rPPG waveform.
    if wave_arr.ndim != 1:
        raise ValueError(
            f"Expected rppg_wave to be one-dimensional, got {wave_arr.shape}."
        )

    # Require matching sample counts.
    if len(rgb_arr) != len(wave_arr):
        raise ValueError(
            f"Signal length mismatch: raw_rgb has {len(rgb_arr)} samples, "
            f"but rppg_wave has {len(wave_arr)} samples."
        )

    if len(rgb_arr) < 2:
        raise ValueError("At least two signal samples are required.")

    # Reject invalid numbers rather than plotting misleading data.
    if not np.all(np.isfinite(rgb_arr)):
        raise ValueError("raw_rgb contains NaN or infinite values.")

    if not np.all(np.isfinite(wave_arr)):
        raise ValueError("rppg_wave contains NaN or infinite values.")

    # Build time axis from the actual sampling rate.
    num_samples = len(rgb_arr)
    time_axis = np.arange(num_samples, dtype=np.float64) / fps

    # Create the output directory if the path includes one.
    output_dir = os.path.dirname(save_path)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    fig, axes = plt.subplots(
        2, 1,
        figsize=(10, 6),
        sharex=True
    )

    try:
        # Panel 1: Raw RGB channel means.
        axes[0].plot(
            time_axis, rgb_arr[:, 0],
            color="crimson", linewidth=1.3, label="Red R(t)"
        )
        axes[0].plot(
            time_axis, rgb_arr[:, 1],
            color="mediumseagreen", linewidth=1.3, label="Green G(t)"
        )
        axes[0].plot(
            time_axis, rgb_arr[:, 2],
            color="royalblue", linewidth=1.3, label="Blue B(t)"
        )

        axes[0].set_title(
            "Raw RGB Mean Intensity Signals",
            fontsize=12, fontweight="bold"
        )
        axes[0].set_ylabel("Mean pixel intensity")
        axes[0].grid(True, linestyle=":", alpha=0.6)
        axes[0].legend(loc="upper right", fontsize=9)

        # Panel 2: Processed rPPG waveform.
        axes[1].plot(
            time_axis, wave_arr,
            color="purple", linewidth=1.5,
            label="Extracted rPPG waveform"
        )
        axes[1].axhline(
            0, color="gray", linestyle="--",
            linewidth=1.0, alpha=0.7
        )

        axes[1].set_title(
            "Extracted rPPG Signal",
            fontsize=12, fontweight="bold"
        )
        axes[1].set_xlabel("Time (seconds)")
        axes[1].set_ylabel("Signal amplitude")
        axes[1].grid(True, linestyle=":", alpha=0.6)
        axes[1].legend(loc="upper right", fontsize=9)

        fig.suptitle(
            "FalseFacet — rPPG Signal Visualization",
            fontsize=14, fontweight="bold"
        )
        fig.tight_layout()

        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        print(f"[+] Signal plot saved to: {save_path}")

        if show_plot:
            plt.show()

    finally:
        plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="FalseFacet rPPG Signal Visualizer"
    )
    parser.add_argument(
        "--save_path",
        default="results/signals/rppg_signals.png"
    )
    parser.add_argument("--fps", type=float, default=30.0)
    args = parser.parse_args()

    if not np.isfinite(args.fps) or args.fps <= 0:
        parser.error("--fps must be a finite positive number.")

    # Generate a reproducible synthetic 10-second signal.
    sample_count = int(round(10 * args.fps))
    if sample_count < 2:
        parser.error("--fps must produce at least two samples.")

    rng = np.random.default_rng(42)
    t = np.arange(sample_count) / args.fps

    pulse = 2.0 * np.sin(2 * np.pi * 1.2 * t)
    mock_rgb = np.column_stack([
        180 + pulse + rng.normal(0, 0.5, sample_count),
        140 + 1.5 * pulse + rng.normal(0, 0.5, sample_count),
        120 + 0.8 * pulse + rng.normal(0, 0.5, sample_count)
    ])

    std = np.std(pulse)
    mock_rppg = (
        (pulse - np.mean(pulse)) / std
        if std > 0
        else np.zeros_like(pulse)
    )

    print("[i] Testing signal visualization with synthetic data...")

    plot_signals(
        raw_rgb=mock_rgb,
        rppg_wave=mock_rppg,
        fps=args.fps,
        save_path=args.save_path
    )
#python -m visual.signal_plot