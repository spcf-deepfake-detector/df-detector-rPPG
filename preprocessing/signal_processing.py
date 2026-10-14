'''import numpy as np
from scipy import signal

def normalize_signal(s: np.ndarray) -> np.ndarray:
    """Normalizes array to zero mean and unit variance."""
    std = np.std(s)
    return (s - np.mean(s)) / (std + 1e-8)

def detrend_signal(s: np.ndarray) -> np.ndarray:
    """Removes slow linear and low-frequency drifts."""
    return signal.detrend(s, type='linear')

def bandpass_filter(s: np.ndarray, fps: float = 30.0, low_hz: float = 0.7, high_hz: float = 2.5, order: int = 4) -> np.ndarray:
    """Applies a Butterworth zero-phase bandpass filter corresponding to [42, 150] BPM."""
    nyq = 0.5 * fps
    if nyq <= high_hz:
        high_hz = nyq - 0.1
    low = low_hz / nyq
    high = high_hz / nyq
    
    if low <= 0 or high <= low:
        return s
        
    b, a = signal.butter(order, [low, high], btype='band')
    # Pad input length if array is small for filtfilt
    padlen = 3 * max(len(b), len(a))
    if len(s) <= padlen:
        return signal.lfilter(b, a, s)
    return signal.filtfilt(b, a, s)

def process_raw_rgb(rgb_signal: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """Full normalization, detrending, and filtering pipeline for RGB arrays (T, 3)."""
    processed = np.zeros_like(rgb_signal)
    for c in range(3):
        norm = normalize_signal(rgb_signal[:, c])
        det = detrend_signal(norm)
        filt = bandpass_filter(det, fps=fps)
        processed[:, c] = filt
    return processed'''
    
    
    

import numpy as np
from scipy import signal


def normalize_signal(s: np.ndarray) -> np.ndarray:
    """Normalize a 1D signal to zero mean and unit variance."""
    s = np.asarray(s, dtype=np.float64)

    if s.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if not np.all(np.isfinite(s)):
        raise ValueError("Signal contains NaN or infinite values.")

    if len(s) == 0:
        return s.copy()

    mean = np.mean(s)
    std = np.std(s)

    # A constant signal has no variation to normalize.
    if std < 1e-8:
        return np.zeros_like(s)

    return (s - mean) / std


def detrend_signal(s: np.ndarray) -> np.ndarray:
    """Remove a linear trend from a 1D signal."""
    s = np.asarray(s, dtype=np.float64)

    if s.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if not np.all(np.isfinite(s)):
        raise ValueError("Signal contains NaN or infinite values.")

    if len(s) < 2:
        return s.copy()

    return signal.detrend(s, type="linear")


def bandpass_filter(
    s: np.ndarray,
    fps: float = 30.0,
    low_hz: float = 0.7,
    high_hz: float = 2.5,
    order: int = 4
) -> np.ndarray:
    """
    Apply a Butterworth bandpass filter.

    Default passband: 0.7-2.5 Hz (42-150 BPM).
    Uses zero-phase filtering when the signal is long enough.
    """

    s = np.asarray(s, dtype=np.float64)

    if s.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if not np.all(np.isfinite(s)):
        raise ValueError("Signal contains NaN or infinite values.")

    if len(s) == 0:
        return s.copy()

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a finite positive number.")

    if not np.isfinite(low_hz) or not np.isfinite(high_hz):
        raise ValueError("Cutoff frequencies must be finite.")

    if low_hz <= 0 or high_hz <= low_hz:
        raise ValueError("Invalid bandpass cutoff frequencies.")

    if not isinstance(order, int) or order < 1:
        raise ValueError("Filter order must be a positive integer.")

    nyquist = fps / 2.0

    if high_hz >= nyquist:
        raise ValueError(
            "The high cutoff frequency must be below the "
            "Nyquist frequency (fps / 2)."
        )

    if len(s) < 2:
        return s.copy()

    # SOS representation is more numerically stable.
    sos = signal.butter(
        order,
        [low_hz, high_hz],
        btype="bandpass",
        fs=fps,
        output="sos"
    )

    # Use zero-phase filtering when enough samples are available.
    default_padlen = 3 * (
        2 * len(sos) + 1
        - min(
            (sos[:, 2] == 0).sum(),
            (sos[:, 5] == 0).sum()
        )
    )

    if len(s) > default_padlen:
        return signal.sosfiltfilt(sos, s)

    # Short signals cannot support zero-phase padding.
    # This causal fallback introduces phase delay and edge effects.
    return signal.sosfilt(sos, s)


def process_raw_rgb(
    rgb_signal: np.ndarray,
    fps: float = 30.0
) -> np.ndarray:
    """
    Detrend, normalize, and filter RGB signals.

    Input shape: (number_of_frames, 3)
    Channel order: R, G, B
    """

    rgb = np.asarray(rgb_signal, dtype=np.float64)

    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(
            "rgb_signal must have shape (number_of_frames, 3)."
        )

    if not np.all(np.isfinite(rgb)):
        raise ValueError(
            "RGB signal contains missing or invalid values. "
            "Handle missing measurements before processing."
        )

    if len(rgb) == 0:
        return np.empty((0, 3), dtype=np.float64)

    processed = np.zeros_like(rgb)

    for c in range(3):
        detrended = detrend_signal(rgb[:, c])
        normalized = normalize_signal(detrended)
        filtered = bandpass_filter(normalized, fps=fps)
        processed[:, c] = filtered

    return processed
