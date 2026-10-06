import numpy as np
from scipy import signal
from sklearn.decomposition import FastICA
from preprocessing.signal_processing import normalize_signal, detrend_signal, bandpass_filter

def green_method(rgb_signal: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """Extracts rPPG using Green Channel baseline."""
    g = rgb_signal[:, 1]
    g_det = detrend_signal(g)
    return bandpass_filter(normalize_signal(g_det), fps=fps)

def chrom_method(rgb_signal: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """Extracts rPPG using Chrominance-based method (CHROM)."""
    X = rgb_signal
    r, g, b = X[:, 0], X[:, 1], X[:, 2]

    # Normalize channels
    r_n = r / (np.mean(r) + 1e-8)
    g_n = g / (np.mean(g) + 1e-8)
    b_n = b / (np.mean(b) + 1e-8)

    Xs = 3.0 * r_n - 2.0 * g_n
    Ys = 1.5 * r_n + 1.0 * g_n - 1.5 * b_n

    Xf = bandpass_filter(Xs, fps=fps)
    Yf = bandpass_filter(Ys, fps=fps)

    alpha = np.std(Xf) / (np.std(Yf) + 1e-8)
    S = Xf - alpha * Yf
    return normalize_signal(S)

def pos_method(rgb_signal: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """Extracts rPPG using Plane-Orthogonal-to-Skin (POS) algorithm."""
    X = rgb_signal
    T = X.shape[0]
    e = np.zeros(T)

    for i in range(T):
        m = np.mean(X[max(0, i - 30):i + 1, :], axis=0) + 1e-8
        Cn = X[i, :] / m
        S1 = Cn[1] - Cn[2]
        S2 = Cn[1] + Cn[2] - 2.0 * Cn[0]
        h = S1 + (np.std(S1) / (np.std(S2) + 1e-8)) * S2
        e[i] = h

    return bandpass_filter(normalize_signal(e), fps=fps)

def ica_method(rgb_signal: np.ndarray, fps: float = 30.0) -> np.ndarray:
    """Extracts rPPG using Independent Component Analysis (ICA)."""
    if len(rgb_signal) < 10:
        return green_method(rgb_signal, fps)

    ica = FastICA(n_components=3, random_state=42, max_iter=500)
    try:
        components = ica.fit_transform(rgb_signal)
    except Exception:
        return green_method(rgb_signal, fps)

    # Select component with max power in physiological band [0.7, 2.5] Hz
    best_comp = components[:, 0]
    max_power = -1.0

    for c in range(3):
        comp = components[:, c]
        fft_vals = np.abs(np.fft.rfft(comp))
        freqs = np.fft.rfftfreq(len(comp), d=1.0 / fps)
        mask = (freqs >= 0.7) & (freqs <= 2.5)
        p_band = np.sum(fft_vals[mask] ** 2)
        if p_band > max_power:
            max_power = p_band
            best_comp = comp

    return bandpass_filter(normalize_signal(best_comp), fps=fps)