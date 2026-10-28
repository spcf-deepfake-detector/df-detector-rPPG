import warnings

import numpy as np
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning

from preprocessing.signal_processing import (
    normalize_signal,
    detrend_signal,
    bandpass_filter,
)


# ============================================================
# 1. SHARED VALIDATION AND SIGNAL UTILITIES
# ============================================================

def _validate_fps(fps: float) -> float:
    """
    Validate the video's frames-per-second value.

    FPS must be a finite, positive number.
    """
    try:
        fps = float(fps)
    except (TypeError, ValueError) as exc:
        raise ValueError("fps must be a finite positive number.") from exc

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a finite positive number.")

    return fps


def _safe_detrend(signal: np.ndarray) -> np.ndarray:
    """
    Remove slow trends from a one-dimensional signal.

    Reject the result if it contains invalid values.
    """
    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if len(signal) < 2:
        raise ValueError("At least two samples are required to detrend.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("Cannot detrend a signal containing NaN or infinity.")

    result = np.asarray(detrend_signal(signal), dtype=np.float64)

    if result.shape != signal.shape:
        raise ValueError("Detrending changed the signal's shape.")

    if not np.all(np.isfinite(result)):
        raise ValueError("Detrending produced invalid values.")

    return result


def _safe_normalize(signal: np.ndarray) -> np.ndarray:
    """
    Normalize a signal while rejecting constant or invalid signals.

    A constant signal contains no useful temporal variation.
    """
    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if len(signal) < 2:
        raise ValueError("At least two samples are required to normalize.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("Cannot normalize a signal containing NaN or infinity.")

    signal_std = float(np.std(signal))

    if not np.isfinite(signal_std) or signal_std < 1e-8:
        raise ValueError(
            "Signal has insufficient variation for normalization."
        )

    result = np.asarray(normalize_signal(signal), dtype=np.float64)

    if result.shape != signal.shape:
        raise ValueError("Normalization changed the signal's shape.")

    if not np.all(np.isfinite(result)):
        raise ValueError("Normalization produced invalid values.")

    return result


def _safe_bandpass(
    signal: np.ndarray,
    fps: float
) -> np.ndarray:
    """
    Apply the project's bandpass filter and validate its output.

    The filter configuration itself is defined in signal_processing.py.
    """
    fps = _validate_fps(fps)
    signal = np.asarray(signal, dtype=np.float64)

    if signal.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if len(signal) < 2:
        raise ValueError("At least two samples are required for filtering.")

    if not np.all(np.isfinite(signal)):
        raise ValueError("Cannot filter a signal containing NaN or infinity.")

    try:
        result = np.asarray(
            bandpass_filter(signal, fps=fps),
            dtype=np.float64
        )
    except (ValueError, FloatingPointError, np.linalg.LinAlgError) as exc:
        raise ValueError(
            "Bandpass filtering failed. Check the signal length, FPS, "
            "and filter configuration in signal_processing.py."
        ) from exc

    if result.shape != signal.shape:
        raise ValueError("Bandpass filtering changed the signal's shape.")

    if not np.all(np.isfinite(result)):
        raise ValueError("Bandpass filtering produced invalid values.")

    return result


# ============================================================
# 2. RGB VALIDATION AND MISSING-VALUE INTERPOLATION
# ============================================================

def prepare_rgb_signal(
    rgb_signal: np.ndarray,
    min_valid_fraction: float = 0.8
) -> np.ndarray:
    """
    Validate RGB measurements and interpolate missing samples.

    Parameters
    ----------
    rgb_signal:
        Array with shape (number_of_frames, 3).
        Channel order must be R, G, B.

    min_valid_fraction:
        Minimum proportion of frames that must contain three finite
        RGB measurements. The default is 0.8, or 80%.

    Returns
    -------
    np.ndarray
        Valid RGB measurements with missing samples interpolated.

    Important
    ---------
    This function does not detect faces or extract RGB measurements.
    Those operations must already have happened earlier in the pipeline.
    """

    try:
        rgb = np.asarray(rgb_signal, dtype=np.float64).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "rgb_signal must contain numeric RGB measurements."
        ) from exc

    try:
        min_valid_fraction = float(min_valid_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "min_valid_fraction must be between 0 and 1."
        ) from exc

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

    total_count = len(rgb)

    if total_count < 10:
        raise ValueError(
            f"Not enough frames for rPPG processing: {total_count}. "
            "At least 10 frames are required by this validation step."
        )

    # A frame is valid only when all three channels are finite.
    valid_rows = np.all(np.isfinite(rgb), axis=1)

    valid_count = int(np.sum(valid_rows))
    missing_count = total_count - valid_count
    valid_fraction = valid_count / total_count

    print(
        f"[rPPG] RGB validation: "
        f"{valid_count}/{total_count} valid frames "
        f"({valid_fraction:.1%}); "
        f"{missing_count} frames contain missing or invalid values."
    )

    if valid_fraction < min_valid_fraction:
        raise ValueError(
            "Too many missing RGB measurements: "
            f"{missing_count}/{total_count} frames missing, "
            f"{valid_fraction:.1%} valid. "
            f"Required: {min_valid_fraction:.1%} valid. "
            "Investigate face detection, tracking, ROI extraction, "
            "and RGB extraction before lowering this threshold."
        )

    frame_indices = np.arange(total_count)

    # Interpolate each RGB channel independently.
    for channel in range(3):
        valid = np.isfinite(rgb[:, channel])
        valid_indices = frame_indices[valid]

        if len(valid_indices) < 2:
            channel_names = ("R", "G", "B")

            raise ValueError(
                f"Insufficient valid samples in the "
                f"{channel_names[channel]} channel."
            )

        # np.interp also fills missing values at the beginning or end
        # using the nearest valid sample. These are extrapolated edge
        # values, not genuine measurements.
        rgb[:, channel] = np.interp(
            frame_indices,
            valid_indices,
            rgb[valid, channel]
        )

    if not np.all(np.isfinite(rgb)):
        raise ValueError(
            "RGB signal still contains invalid values after interpolation."
        )

    return rgb


# ============================================================
# 3. GREEN-CHANNEL METHOD
# ============================================================

def green_method(
    rgb_signal: np.ndarray,
    fps: float = 30.0
) -> np.ndarray:
    """
    Extract an rPPG candidate from the green channel.

    Processing:
        RGB validation
        -> green channel
        -> detrending
        -> normalization
        -> bandpass filtering
    """

    fps = _validate_fps(fps)
    rgb = prepare_rgb_signal(rgb_signal)

    # RGB channel order is R=0, G=1, B=2.
    green = rgb[:, 1]

    green_detrended = _safe_detrend(green)
    green_normalized = _safe_normalize(green_detrended)

    return _safe_bandpass(green_normalized, fps=fps)


# ============================================================
# 4. CHROM METHOD
# ============================================================

def chrom_method(
    rgb_signal: np.ndarray,
    fps: float = 30.0
) -> np.ndarray:
    """
    Extract an rPPG candidate using a CHROM-style method.

    The method combines normalized red, green, and blue channels
    into two chrominance projections and combines their filtered
    signals.
    """

    fps = _validate_fps(fps)
    rgb = prepare_rgb_signal(rgb_signal)

    r = rgb[:, 0]
    g = rgb[:, 1]
    b = rgb[:, 2]

    eps = 1e-8

    # Calculate the temporal mean of each channel.
    channel_means = np.mean(rgb, axis=0)

    if not np.all(np.isfinite(channel_means)):
        raise ValueError("RGB channel means must be finite.")

    if np.any(channel_means <= eps):
        raise ValueError(
            "RGB channels have invalid or near-zero means."
        )

    r_mean, g_mean, b_mean = channel_means

    # Normalize each channel by its temporal mean.
    r_n = r / r_mean
    g_n = g / g_mean
    b_n = b / b_mean

    # CHROM-style chrominance projections.
    Xs = 3.0 * r_n - 2.0 * g_n
    Ys = 1.5 * r_n + g_n - 1.5 * b_n

    # Isolate variations within the configured physiological band.
    Xf = _safe_bandpass(Xs, fps=fps)
    Yf = _safe_bandpass(Ys, fps=fps)

    std_x = float(np.std(Xf))
    std_y = float(np.std(Yf))

    if not np.isfinite(std_x) or not np.isfinite(std_y):
        raise ValueError(
            "CHROM produced invalid projection statistics."
        )

    if std_y < eps:
        raise ValueError(
            "CHROM Y projection has insufficient variation."
        )

    alpha = std_x / (std_y + eps)

    pulse_signal = Xf - alpha * Yf

    pulse_signal = _safe_detrend(pulse_signal)

    return _safe_normalize(pulse_signal)


# ============================================================
# 5. POS METHOD
# ============================================================

def pos_method(
    rgb_signal: np.ndarray,
    fps: float = 30.0,
    window_seconds: float = 1.6
) -> np.ndarray:
    """
    Extract an rPPG candidate using a windowed POS-style method.

    Processing:
        RGB validation
        -> overlapping temporal windows
        -> channel normalization within each window
        -> POS projections
        -> overlap-add
        -> detrending and normalization
        -> bandpass filtering
    """

    fps = _validate_fps(fps)

    try:
        window_seconds = float(window_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "window_seconds must be a finite positive number."
        ) from exc

    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError(
            "window_seconds must be a finite positive number."
        )

    rgb = prepare_rgb_signal(rgb_signal)

    n_frames = len(rgb)
    window_length = int(round(window_seconds * fps))

    if window_length < 2:
        raise ValueError(
            "POS window must contain at least two frames."
        )

    if n_frames < window_length:
        raise ValueError(
            f"Video is too short for the configured POS window. "
            f"Frames: {n_frames}; required: {window_length}."
        )

    eps = 1e-8

    pulse_sum = np.zeros(n_frames, dtype=np.float64)
    pulse_count = np.zeros(n_frames, dtype=np.float64)

    windows_processed = 0
    windows_skipped = 0

    # Slide the window forward one frame at a time.
    for start in range(n_frames - window_length + 1):

        end = start + window_length
        window = rgb[start:end, :]

        # Normalize RGB channels by their means within this window.
        channel_means = np.mean(window, axis=0)

        if (
            not np.all(np.isfinite(channel_means))
            or np.any(channel_means <= eps)
        ):
            windows_skipped += 1
            continue

        Cn = window / channel_means

        # POS projections.
        S1 = Cn[:, 1] - Cn[:, 2]

        S2 = Cn[:, 1] + Cn[:, 2] - 2.0 * Cn[:, 0]

        std_s1 = float(np.std(S1))
        std_s2 = float(np.std(S2))

        if (
            not np.isfinite(std_s1)
            or not np.isfinite(std_s2)
            or std_s2 < eps
        ):
            windows_skipped += 1
            continue

        alpha = std_s1 / (std_s2 + eps)

        h = S1 + alpha * S2

        # Remove the window's mean/DC component.
        h = h - np.mean(h)

        if not np.all(np.isfinite(h)):
            windows_skipped += 1
            continue

        # Add this window's contribution to the full-length signal.
        pulse_sum[start:end] += h
        pulse_count[start:end] += 1.0

        windows_processed += 1

    print(
        f"[rPPG POS] Windows processed: {windows_processed}; "
        f"windows skipped: {windows_skipped}."
    )

    if windows_processed == 0:
        raise ValueError(
            "POS could not produce a valid pulse signal. "
            "All windows were unusable."
        )

    valid = pulse_count > 0
    valid_frame_count = int(np.sum(valid))

    if valid_frame_count < 2:
        raise ValueError(
            "POS produced fewer than two usable output samples."
        )

    # Average overlapping window contributions.
    pulse_signal = np.zeros(n_frames, dtype=np.float64)

    pulse_signal[valid] = (
        pulse_sum[valid] / pulse_count[valid]
    )

    # Normally, every frame is covered when the video is at least
    # as long as the configured window.
    if not np.all(valid):
        frame_indices = np.arange(n_frames)

        pulse_signal = np.interp(
            frame_indices,
            frame_indices[valid],
            pulse_signal[valid]
        )

    pulse_signal = _safe_detrend(pulse_signal)
    pulse_signal = _safe_normalize(pulse_signal)
    pulse_signal = _safe_bandpass(pulse_signal, fps=fps)

    print("[rPPG POS] Output shape:", pulse_signal.shape)
    print("[rPPG POS] All finite:", np.isfinite(pulse_signal).all())
    print("[rPPG POS] Standard deviation:", np.std(pulse_signal))

    return pulse_signal


# ============================================================
# 6. ICA METHOD
# ============================================================

def ica_method(
    rgb_signal: np.ndarray,
    fps: float = 30.0
) -> np.ndarray:
    """
    Extract an rPPG candidate using Independent Component Analysis.

    ICA separates the three RGB channels into statistically
    independent components. The component with the greatest
    spectral power in the selected frequency range is chosen.

    The green-channel method is used as a fallback when ICA
    cannot be used successfully.
    """

    fps = _validate_fps(fps)
    rgb = prepare_rgb_signal(rgb_signal)

    # Short recordings may not provide enough information for
    # stable ICA-based component selection.
    if len(rgb) < max(30, int(round(fps))):
        print(
            "[rPPG ICA] Recording is short; "
            "using the green-channel fallback."
        )

        return green_method(rgb, fps=fps)

    # Standardize each channel before ICA.
    channel_means = np.mean(rgb, axis=0)
    channel_stds = np.std(rgb, axis=0)

    if (
        not np.all(np.isfinite(channel_means))
        or not np.all(np.isfinite(channel_stds))
    ):
        print(
            "[rPPG ICA] Invalid channel statistics; "
            "using the green-channel fallback."
        )

        return green_method(rgb, fps=fps)

    if np.any(channel_stds < 1e-8):
        print(
            "[rPPG ICA] A channel has insufficient variation; "
            "using the green-channel fallback."
        )

        return green_method(rgb, fps=fps)

    rgb_standardized = (
        rgb - channel_means
    ) / channel_stds

    if not np.all(np.isfinite(rgb_standardized)):
        print(
            "[rPPG ICA] Standardization produced invalid values; "
            "using the green-channel fallback."
        )

        return green_method(rgb, fps=fps)

    # ICA requires enough samples and variation to estimate
    # independent components. Convergence is checked explicitly.
    ica = FastICA(
        n_components=3,
        random_state=42,
        max_iter=1000,
        tol=0.001,
        whiten="unit-variance"
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)

            components = ica.fit_transform(rgb_standardized)

    except (
        ValueError,
        FloatingPointError,
        np.linalg.LinAlgError,
        ConvergenceWarning
    ):
        print(
            "[rPPG ICA] ICA failed or did not converge; "
            "using the green-channel fallback."
        )

        return green_method(rgb, fps=fps)

    if (
        components.ndim != 2
        or components.shape[0] != len(rgb)
        or components.shape[1] != 3
        or not np.all(np.isfinite(components))
    ):
        print(
            "[rPPG ICA] ICA returned invalid components; "
            "using the green-channel fallback."
        )

        return green_method(rgb, fps=fps)

    # Frequency resolution depends on both FPS and recording length.
    freqs = np.fft.rfftfreq(
        len(rgb),
        d=1.0 / fps
    )

    # Candidate pulse-frequency band: 0.7-2.5 Hz.
    frequency_mask = (
        (freqs >= 0.7)
        & (freqs <= 2.5)
    )

    if not np.any(frequency_mask):
        print(
            "[rPPG ICA] No frequency bins in the selected band; "
            "using the green-channel fallback."
        )

        return green_method(rgb, fps=fps)

    best_component = None
    max_power = -1.0

    for component_index in range(components.shape[1]):

        component = components[:, component_index]

        try:
            component = _safe_detrend(component)
        except ValueError:
            continue

        # Calculate spectral power.
        fft_values = np.abs(
            np.fft.rfft(component)
        ) ** 2

        if not np.all(np.isfinite(fft_values)):
            continue

        band_power = float(
            np.sum(fft_values[frequency_mask])
        )

        if (
            np.isfinite(band_power)
            and band_power > max_power
        ):
            max_power = band_power
            best_component = component

    if best_component is None or max_power < 0:
        print(
            "[rPPG ICA] No usable component found; "
            "using the green-channel fallback."
        )

        return green_method(rgb, fps=fps)

    try:
        component_normalized = _safe_normalize(best_component)

        return _safe_bandpass(
            component_normalized,
            fps=fps
        )

    except ValueError:
        print(
            "[rPPG ICA] Selected component was unusable; "
            "using the green-channel fallback."
        )

        return green_method(rgb, fps=fps)