
import numpy as np
from sklearn.decomposition import FastICA

from preprocessing.signal_processing import (
    normalize_signal,
    detrend_signal,
    bandpass_filter
)


def prepare_rgb_signal(
    rgb_signal: np.ndarray,
    min_valid_fraction: float = 0.8
) -> np.ndarray:
    """
    Validate RGB measurements and interpolate missing samples.

    Input shape: (number_of_frames, 3)
    Channel order: R, G, B
    """

    rgb = np.asarray(rgb_signal, dtype=np.float64).copy()

    if (
        not np.isfinite(min_valid_fraction)
        or not 0.0 <= min_valid_fraction <= 1.0
    ):
        raise ValueError(
            "min_valid_fraction must be between 0 and 1."
        )

    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(
            "rgb_signal must have shape (number_of_frames, 3)."
        )

    if len(rgb) < 10:
        raise ValueError(
            f"Not enough frames for rPPG processing: {len(rgb)}."
        )

    # A frame is valid when all RGB channels are finite.
    valid_rows = np.all(np.isfinite(rgb), axis=1)

    valid_count = int(np.sum(valid_rows))
    total_count = len(rgb)
    missing_count = total_count - valid_count
    valid_fraction = valid_count / total_count

    # Provide diagnostic information when too many frames are missing.
    if valid_fraction < min_valid_fraction:
        raise ValueError(
            f"Too many missing RGB measurements: "
            f"{missing_count}/{total_count} frames missing, "
            f"{valid_fraction:.1%} valid. "
            f"Required: {min_valid_fraction:.1%} valid."
        )

    frame_indices = np.arange(len(rgb))

    # Interpolate missing values independently for each channel.
    for channel in range(3):

        valid = np.isfinite(rgb[:, channel])

        if np.sum(valid) < 2:
            raise ValueError(
                f"Insufficient valid samples in channel {channel}."
            )

        rgb[:, channel] = np.interp(
            frame_indices,
            frame_indices[valid],
            rgb[valid, channel]
        )

    if not np.all(np.isfinite(rgb)):
        raise ValueError(
            "RGB signal still contains invalid values."
        )

    return rgb


def green_method(
    rgb_signal: np.ndarray,
    fps: float = 30.0
) -> np.ndarray:
    """Extract an rPPG signal using the green channel."""

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    
    rgb = prepare_rgb_signal(rgb_signal)
    
    g = rgb[:, 1]

    g_det = detrend_signal(g)

    g_norm = normalize_signal(g_det)

    return bandpass_filter(g_norm, fps=fps)


def chrom_method(
    rgb_signal: np.ndarray,
    fps: float = 30.0
) -> np.ndarray:
    """Extract an rPPG signal using a CHROM-style method."""
    
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    
    rgb = prepare_rgb_signal(rgb_signal)

    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]

    eps = 1e-8

    # Normalize each channel by its temporal mean.
    r_mean = np.mean(r)
    g_mean = np.mean(g)
    b_mean = np.mean(b)

    if min(r_mean, g_mean, b_mean) <= eps:
        raise ValueError(
            "RGB channels have invalid or near-zero means."
        )
        
    if not np.all(np.isfinite([r_mean, g_mean, b_mean])):
        raise ValueError("RGB channel means must be finite.")

    r_n = r / r_mean
    g_n = g / g_mean
    b_n = b / b_mean

    # Chrominance projections.
    Xs = 3.0 * r_n - 2.0 * g_n

    Ys = 1.5 * r_n + g_n - 1.5 * b_n

    Xf = bandpass_filter(Xs, fps=fps)
    Yf = bandpass_filter(Ys, fps=fps)

    alpha = np.std(Xf) / (np.std(Yf) + eps)

    pulse_signal = Xf - alpha * Yf

    return normalize_signal(pulse_signal)


def pos_method(
    rgb_signal: np.ndarray,
    fps: float = 30.0,
    window_seconds: float = 1.6
) -> np.ndarray:
    """Extract an rPPG signal using a windowed POS method."""

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(" fps must be finite and positive")

    rgb = prepare_rgb_signal(rgb_signal)

    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be finite and positive.")

    n_frames = len(rgb)

    window_length = int(round(window_seconds * fps))

    if window_length < 2:
        raise ValueError(
            "POS window must contain at least two frames."
        )

    if n_frames < window_length:
        raise ValueError(
            "Video is too short for the configured POS window."
        )

    eps = 1e-8

    pulse_sum = np.zeros(n_frames, dtype=np.float64)
    pulse_count = np.zeros(n_frames, dtype=np.float64)

    # Process overlapping temporal windows.
    for start in range(n_frames - window_length + 1):

        end = start + window_length

        window = rgb[start:end, :]

        # Normalize the RGB channels within this window.
        channel_means = np.mean(window, axis=0)

        if not np.all(np.isfinite(channel_means)) or np.any(channel_means <= eps):
            continue

        Cn = window / channel_means

        # POS projections.
        S1 = Cn[:, 1] - Cn[:, 2]

        S2 = Cn[:, 1] + Cn[:, 2] - 2.0 * Cn[:, 0]

        # Calculate standard deviations across the window.
        alpha = np.std(S1) / (np.std(S2) + eps)

        h = S1 + alpha * S2

        # Remove the window's DC component.
        h = h - np.mean(h)

        # Overlap-add the window signals.
        pulse_sum[start:end] += h
        pulse_count[start:end] += 1.0

    valid = pulse_count > 0

    if not np.any(valid):
        raise ValueError(
            "POS could not produce a valid pulse signal."
        )

    pulse_signal = np.zeros(n_frames, dtype=np.float64)

    pulse_signal[valid] = (
        pulse_sum[valid] / pulse_count[valid]
    )

    pulse_signal = detrend_signal(pulse_signal)

    pulse_signal = normalize_signal(pulse_signal)

    return bandpass_filter(pulse_signal, fps=fps)


def ica_method(
    rgb_signal: np.ndarray,
    fps: float = 30.0
) -> np.ndarray:
    """Extract an rPPG candidate using Independent Component Analysis."""
    
    rgb = prepare_rgb_signal(rgb_signal)

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive.")

    # Use the green method as a fallback for short recordings.
    if len(rgb) < max(30, int(fps)):
        return green_method(rgb, fps=fps)

    # Standardize channels before ICA.
    channel_means = np.mean(rgb, axis=0)
    channel_stds = np.std(rgb, axis=0)

    if np.any(channel_stds < 1e-8):
        return green_method(rgb, fps=fps)

    rgb_standardized = (
        rgb - channel_means
    ) / channel_stds

    ica = FastICA(
        n_components=3,
        random_state=42,
        max_iter=1000,
        tol=0.001
    )

    try:
        components = ica.fit_transform(rgb_standardized)

    except (ValueError, FloatingPointError, np.linalg.LinAlgError):
        return green_method(rgb, fps=fps)

    # Select the component with the greatest spectral power
    # in the chosen physiological frequency range.
    best_component = None
    max_power = -1.0

    freqs = np.fft.rfftfreq(
        len(rgb),
        d=1.0 / fps
    )

    mask = (freqs >= 0.7) & (freqs <= 2.5)

    if not np.any(mask):
        return green_method(rgb, fps=fps)

    for component_index in range(components.shape[1]):

        component = components[:, component_index]

        component = detrend_signal(component)

        fft_values = np.abs(
            np.fft.rfft(component)
        ) ** 2

        band_power = float(
            np.sum(fft_values[mask])
        )

        if band_power > max_power:
            max_power = band_power
            best_component = component

    if best_component is None:
        return green_method(rgb, fps=fps)

    return bandpass_filter(
        normalize_signal(best_component),
        fps=fps
    )
    