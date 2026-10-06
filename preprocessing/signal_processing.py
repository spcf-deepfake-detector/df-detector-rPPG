import numpy as np
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
    return processed