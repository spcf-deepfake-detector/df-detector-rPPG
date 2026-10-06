import numpy as np
from scipy import signal

def calculate_snr(rppg_signal: np.ndarray, fps: float = 30.0, hr_hz: float = 1.2) -> float:
    """Calculates Signal-to-Noise Ratio (SNR) in dB around estimated fundamental frequency."""
    N = len(rppg_signal)
    if N == 0:
        return -20.0

    fft_vals = np.abs(np.fft.rfft(rppg_signal)) ** 2
    freqs = np.fft.rfftfreq(N, d=1.0 / fps)

    # Fundamental & first harmonic windows (+/- 0.15 Hz)
    sig_mask = ((freqs >= hr_hz - 0.15) & (freqs <= hr_hz + 0.15)) | \
               ((freqs >= 2 * hr_hz - 0.15) & (freqs <= 2 * hr_hz + 0.15))
    noise_mask = (freqs >= 0.7) & (freqs <= 2.5) & (~sig_mask)

    p_sig = np.sum(fft_vals[sig_mask])
    p_noise = np.sum(fft_vals[noise_mask])

    if p_noise <= 0:
        return 20.0
    return float(10.0 * np.log10(p_sig / p_noise))

def calculate_spectral_concentration(rppg_signal: np.ndarray, fps: float = 30.0) -> float:
    """Calculates proportion of power centered around peak frequency."""
    N = len(rppg_signal)
    if N == 0:
        return 0.0

    fft_vals = np.abs(np.fft.rfft(rppg_signal)) ** 2
    freqs = np.fft.rfftfreq(N, d=1.0 / fps)

    phys_mask = (freqs >= 0.7) & (freqs <= 2.5)
    if not np.any(phys_mask):
        return 0.0

    phys_freqs = freqs[phys_mask]
    phys_fft = fft_vals[phys_mask]

    peak_idx = np.argmax(phys_fft)
    peak_freq = phys_freqs[peak_idx]

    peak_mask = (freqs >= peak_freq - 0.1) & (freqs <= peak_freq + 0.1)
    return float(np.sum(fft_vals[peak_mask]) / (np.sum(phys_fft) + 1e-8))

def evaluate_signal_quality(rppg_signal: np.ndarray, fps: float = 30.0) -> float:
    """Returns composite quality index Q in range [0.0, 1.0]."""
    if len(rppg_signal) == 0:
        return 0.0

    freqs = np.fft.rfftfreq(len(rppg_signal), d=1.0 / fps)
    fft_vals = np.abs(np.fft.rfft(rppg_signal))
    phys_mask = (freqs >= 0.7) & (freqs <= 2.5)

    if not np.any(phys_mask):
        return 0.0

    peak_freq = freqs[phys_mask][np.argmax(fft_vals[phys_mask])]
    snr = calculate_snr(rppg_signal, fps=fps, hr_hz=peak_freq)
    spec_conc = calculate_spectral_concentration(rppg_signal, fps=fps)

    # Map SNR [-10, 10] dB to [0, 1]
    snr_norm = np.clip((snr + 10.0) / 20.0, 0.0, 1.0)
    quality_score = float(0.6 * snr_norm + 0.4 * spec_conc)
    return float(np.clip(quality_score, 0.0, 1.0))