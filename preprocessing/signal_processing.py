import numpy as np
from scipy import signal


# ============================================================
# 1. NORMALIZE SIGNAL
# ============================================================

def normalize_signal(s: np.ndarray) -> np.ndarray:
    """
    Normalize a one-dimensional signal to approximately
    zero mean and unit variance.

    A constant signal returns zeros because it has no variation.
    The rPPG extraction methods should reject constant signals
    when a meaningful pulse signal is required.
    """

    s = np.asarray(s, dtype=np.float64)

    if s.ndim != 1:
        raise ValueError(
            "Signal must be one-dimensional."
        )

    if not np.all(np.isfinite(s)):
        raise ValueError(
            "Signal contains NaN or infinite values."
        )

    if len(s) == 0:
        return s.copy()

    mean = float(np.mean(s))
    std = float(np.std(s))

    if not np.isfinite(mean) or not np.isfinite(std):
        raise ValueError(
            "Signal mean or standard deviation is invalid."
        )

    # A constant or nearly constant signal cannot be meaningfully
    # standardized.
    if std < 1e-8:
        return np.zeros_like(s)

    normalized = (s - mean) / std

    if not np.all(np.isfinite(normalized)):
        raise ValueError(
            "Normalization produced invalid values."
        )

    return normalized


# ============================================================
# 2. DETREND SIGNAL
# ============================================================

def detrend_signal(s: np.ndarray) -> np.ndarray:
    """
    Remove a linear trend from a one-dimensional signal.

    Detrending helps reduce gradual changes in the signal,
    such as slow brightness or baseline drift.

    It does not remove every possible source of noise.
    """

    s = np.asarray(s, dtype=np.float64)

    if s.ndim != 1:
        raise ValueError(
            "Signal must be one-dimensional."
        )

    if not np.all(np.isfinite(s)):
        raise ValueError(
            "Signal contains NaN or infinite values."
        )

    if len(s) < 2:
        return s.copy()

    detrended = signal.detrend(
        s,
        type="linear"
    )

    if not np.all(np.isfinite(detrended)):
        raise ValueError(
            "Detrending produced invalid values."
        )

    return detrended


# ============================================================
# 3. BANDPASS FILTER
# ============================================================

def bandpass_filter(
    s: np.ndarray,
    fps: float = 30.0,
    low_hz: float = 0.7,
    high_hz: float = 2.5,
    order: int = 4
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter.

    Default passband:
        0.7-2.5 Hz
        Approximately 42-150 beats per minute.

    Long signals:
        Use zero-phase filtering with sosfiltfilt.

    Short signals:
        Use causal filtering with sosfilt because there may not
        be enough samples for zero-phase filtering.

    Note:
        The short-signal fallback can introduce phase delay
        and edge effects.
    """

    s = np.asarray(s, dtype=np.float64)

    # --------------------------------------------------------
    # Validate input signal
    # --------------------------------------------------------

    if s.ndim != 1:
        raise ValueError(
            "Signal must be one-dimensional."
        )

    if not np.all(np.isfinite(s)):
        raise ValueError(
            "Signal contains NaN or infinite values."
        )

    if len(s) == 0:
        return s.copy()

    # --------------------------------------------------------
    # Validate FPS
    # --------------------------------------------------------

    try:
        fps = float(fps)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "fps must be a finite positive number."
        ) from exc

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(
            "fps must be a finite positive number."
        )

    # --------------------------------------------------------
    # Validate cutoff frequencies
    # --------------------------------------------------------

    try:
        low_hz = float(low_hz)
        high_hz = float(high_hz)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Cutoff frequencies must be finite numbers."
        ) from exc

    if (
        not np.isfinite(low_hz)
        or not np.isfinite(high_hz)
    ):
        raise ValueError(
            "Cutoff frequencies must be finite."
        )

    if low_hz <= 0 or high_hz <= low_hz:
        raise ValueError(
            "Invalid bandpass cutoff frequencies. "
            "Require 0 < low_hz < high_hz."
        )

    # The Nyquist frequency is half the sampling rate.
    nyquist = fps / 2.0

    if high_hz >= nyquist:
        raise ValueError(
            f"High cutoff frequency ({high_hz} Hz) must be below "
            f"the Nyquist frequency ({nyquist} Hz). "
            f"Current FPS: {fps}."
        )

    # --------------------------------------------------------
    # Validate filter order
    # --------------------------------------------------------

    if (
        not isinstance(order, (int, np.integer))
        or isinstance(order, (bool, np.bool_))
        or order < 1
    ):
        raise ValueError(
            "Filter order must be a positive integer."
        )

    if len(s) < 2:
        return s.copy()

    # --------------------------------------------------------
    # Design the Butterworth filter
    # --------------------------------------------------------

    # Second-order sections are more numerically stable than
    # the traditional numerator/denominator representation.
    sos = signal.butter(
        int(order),
        [low_hz, high_hz],
        btype="bandpass",
        fs=fps,
        output="sos"
    )

    # --------------------------------------------------------
    # Determine whether zero-phase filtering is possible
    # --------------------------------------------------------

    # This follows SciPy's default padding-length calculation
    # for SOS zero-phase filtering.
    default_padlen = 3 * (
        2 * len(sos) + 1
        - min(
            int(np.sum(sos[:, 2] == 0)),
            int(np.sum(sos[:, 5] == 0))
        )
    )

    if len(s) > default_padlen:

        # Forward-backward filtering reduces phase distortion.
        filtered = signal.sosfiltfilt(
            sos,
            s
        )

    else:

        # Short recordings may not have enough samples for
        # forward-backward filtering.
        # This fallback is causal and may introduce delay.
        filtered = signal.sosfilt(
            sos,
            s
        )

    if not np.all(np.isfinite(filtered)):
        raise ValueError(
            "Bandpass filtering produced invalid values."
        )

    return filtered


# ============================================================
# 4. PROCESS RAW RGB
# ============================================================

def process_raw_rgb(
    rgb_signal: np.ndarray,
    fps: float = 30.0
) -> np.ndarray:
    """
    Detrend, normalize, and filter each RGB channel independently.

    Parameters
    ----------
    rgb_signal:
        Array of shape (number_of_frames, 3).

        Channel order:
            R = column 0
            G = column 1
            B = column 2

    fps:
        Video sampling rate in frames per second.

    Returns
    -------
    np.ndarray
        Processed RGB array with shape (number_of_frames, 3).

    Important
    ---------
    This function processes each channel independently.
    It does not replace the Green, CHROM, or POS algorithms.
    """

    rgb = np.asarray(
        rgb_signal,
        dtype=np.float64
    )

    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(
            "rgb_signal must have shape "
            "(number_of_frames, 3)."
        )

    if not np.all(np.isfinite(rgb)):
        raise ValueError(
            "RGB signal contains missing or invalid values. "
            "Handle missing measurements before processing."
        )

    if len(rgb) == 0:
        return np.empty(
            (0, 3),
            dtype=np.float64
        )

    # Validate FPS before processing any channels.
    try:
        fps = float(fps)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "fps must be a finite positive number."
        ) from exc

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError(
            "fps must be a finite positive number."
        )

    processed = np.zeros_like(rgb)

    # Process red, green, and blue separately.
    for channel in range(3):

        raw_channel = rgb[:, channel]

        # Remove slow linear drift.
        detrended = detrend_signal(
            raw_channel
        )

        # Standardize the channel.
        normalized = normalize_signal(
            detrended
        )

        # Keep temporal variations within the selected frequency band.
        filtered = bandpass_filter(
            normalized,
            fps=fps
        )

        processed[:, channel] = filtered

    if not np.all(np.isfinite(processed)):
        raise ValueError(
            "RGB processing produced invalid values."
        )

    return processed
