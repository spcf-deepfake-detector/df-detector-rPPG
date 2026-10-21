'''import numpy as np
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
    return float(np.clip(quality_score, 0.0, 1.0))'''
    

import numpy as np


LOW_HZ = 0.7
HIGH_HZ = 2.5
MIN_SIGNAL_SECONDS = 3.0


def _prepare_signal(
    rppg_signal: np.ndarray,
    fps: float
) -> np.ndarray:
    """Validate an rPPG signal before frequency analysis."""

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a positive finite number.")

    if HIGH_HZ >= fps / 2.0:
        raise ValueError(
            f"FPS is too low for the {HIGH_HZ} Hz upper cutoff. "
            "The upper cutoff must be below the Nyquist frequency (fps / 2)."
        )

    s = np.asarray(rppg_signal, dtype=np.float64)

    if s.ndim != 1:
        raise ValueError("rppg_signal must be a one-dimensional array.")

    if not np.all(np.isfinite(s)):
        raise ValueError(
            "rppg_signal contains NaN or infinite values. "
            "Clean the signal before quality evaluation."
        )

    return s


def _get_power_spectrum(s: np.ndarray, fps: float):
    """Calculate a windowed power spectrum."""

    window = np.hanning(len(s))
    centered = s - np.mean(s)

    fft_values = np.fft.rfft(centered * window)
    power = np.abs(fft_values) ** 2
    frequencies = np.fft.rfftfreq(len(s), d=1.0 / fps)

    return frequencies, power


def calculate_snr(
    rppg_signal: np.ndarray,
    fps: float = 30.0,
    hr_hz: float = 1.2
) -> float:
    """
    Estimate SNR in dB around the fundamental and first harmonic.
    Evaluates signal and noise power within the same physiological band.
    """

    s = _prepare_signal(rppg_signal, fps)

    if len(s) < int(MIN_SIGNAL_SECONDS * fps):
        return -20.0

    if not np.isfinite(hr_hz) or hr_hz <= 0:
        return -20.0

    frequencies, power = _get_power_spectrum(s, fps)

    physiological_mask = (
        (frequencies >= LOW_HZ) &
        (frequencies <= HIGH_HZ)
    )

    fundamental_mask = (
        (frequencies >= hr_hz - 0.15) &
        (frequencies <= hr_hz + 0.15)
    )

    harmonic_mask = (
        (frequencies >= 2.0 * hr_hz - 0.15) &
        (frequencies <= 2.0 * hr_hz + 0.15)
    )

    signal_mask = (
        (fundamental_mask | harmonic_mask) &
        physiological_mask
    )

    noise_mask = physiological_mask & ~signal_mask

    if not np.any(signal_mask) or not np.any(noise_mask):
        return -20.0

    signal_power = float(np.sum(power[signal_mask]))
    noise_power = float(np.sum(power[noise_mask]))

    if signal_power <= 1e-12:
        return -20.0

    if noise_power <= 1e-12:
        return 20.0

    snr_db = 10.0 * np.log10(signal_power / noise_power)

    return float(snr_db)


def calculate_spectral_concentration(
    rppg_signal: np.ndarray,
    fps: float = 30.0
) -> float:
    """
    Calculate the proportion of physiological-band power
    concentrated within +/- 0.1 Hz of the peak.
    """

    s = _prepare_signal(rppg_signal, fps)

    if len(s) < int(MIN_SIGNAL_SECONDS * fps):
        return 0.0

    frequencies, power = _get_power_spectrum(s, fps)

    physiological_mask = (
        (frequencies >= LOW_HZ) &
        (frequencies <= HIGH_HZ)
    )

    if not np.any(physiological_mask):
        return 0.0

    physiological_frequencies = frequencies[physiological_mask]
    physiological_power = power[physiological_mask]

    total_power = float(np.sum(physiological_power))

    if total_power <= 1e-12:
        return 0.0

    peak_index = int(np.argmax(physiological_power))
    peak_frequency = physiological_frequencies[peak_index]

    peak_mask = (
        (frequencies >= peak_frequency - 0.1) &
        (frequencies <= peak_frequency + 0.1) &
        physiological_mask
    )

    concentration = np.sum(power[peak_mask]) / total_power

    return float(np.clip(concentration, 0.0, 1.0))


def evaluate_signal_quality(
    rppg_signal: np.ndarray,
    fps: float = 30.0
) -> float:
    """
    Return a heuristic signal-quality score in [0, 1].
    This score does not directly determine whether a video is real or fake.
    """

    s = _prepare_signal(rppg_signal, fps)

    if len(s) < int(MIN_SIGNAL_SECONDS * fps):
        return 0.0

    if np.std(s) < 1e-8:
        return 0.0

    frequencies, power = _get_power_spectrum(s, fps)

    physiological_mask = (
        (frequencies >= LOW_HZ) &
        (frequencies <= HIGH_HZ)
    )

    if not np.any(physiological_mask):
        return 0.0

    physiological_power = power[physiological_mask]

    if np.sum(physiological_power) <= 1e-12:
        return 0.0

    peak_frequency = frequencies[physiological_mask][
        np.argmax(physiological_power)
    ]

    snr = calculate_snr(
        s,
        fps=fps,
        hr_hz=float(peak_frequency)
    )

    spectral_concentration = calculate_spectral_concentration(
        s,
        fps=fps
    )

    # Map SNR from [-10, 10] dB to [0, 1].
    snr_normalized = np.clip((snr + 10.0) / 20.0, 0.0, 1.0)

    quality_score = (
        0.6 * snr_normalized +
        0.4 * spectral_concentration
    )

    return float(np.clip(quality_score, 0.0, 1.0))

