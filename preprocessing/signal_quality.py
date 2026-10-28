import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

LOW_HZ = 0.7
HIGH_HZ = 2.5

MIN_SIGNAL_SECONDS = 3.0

# Frequency tolerance around the estimated pulse frequency.
FUNDAMENTAL_TOLERANCE_HZ = 0.15

# Width around the strongest frequency for spectral concentration.
CONCENTRATION_TOLERANCE_HZ = 0.1

# Numerical stability threshold.
EPSILON = 1e-12


# ============================================================
# 1. VALIDATE THE SIGNAL
# ============================================================

def _prepare_signal(
    rppg_signal: np.ndarray,
    fps: float
) -> np.ndarray:
    """
    Validate an rPPG signal before frequency analysis.

    Parameters
    ----------
    rppg_signal:
        One-dimensional extracted rPPG signal.

    fps:
        Sampling rate in frames per second.

    Returns
    -------
    np.ndarray
        Validated signal represented as float64.
    """

    # Validate FPS.
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

    # The upper frequency must be below the Nyquist frequency.
    nyquist = fps / 2.0

    if HIGH_HZ >= nyquist:
        raise ValueError(
            f"FPS is too low for the configured "
            f"{HIGH_HZ} Hz upper frequency. "
            f"The upper frequency must be below the "
            f"Nyquist frequency ({nyquist:.3f} Hz)."
        )

    # Convert the input to a numerical array.
    try:
        s = np.asarray(
            rppg_signal,
            dtype=np.float64
        )
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "rppg_signal must contain numeric values."
        ) from exc

    if s.ndim != 1:
        raise ValueError(
            "rppg_signal must be a one-dimensional array."
        )

    if not np.all(np.isfinite(s)):
        raise ValueError(
            "rppg_signal contains NaN or infinite values. "
            "Clean the signal before quality evaluation."
        )

    return s


# ============================================================
# 2. CHECK RECORDING DURATION
# ============================================================

def _has_sufficient_duration(
    s: np.ndarray,
    fps: float
) -> bool:
    """
    Check whether the signal meets the minimum duration.

    A recording shorter than MIN_SIGNAL_SECONDS is not
    considered long enough for this quality evaluation.
    """

    required_samples = int(
        np.ceil(MIN_SIGNAL_SECONDS * fps)
    )

    return len(s) >= required_samples


# ============================================================
# 3. CALCULATE THE POWER SPECTRUM
# ============================================================

def _get_power_spectrum(
    s: np.ndarray,
    fps: float
):
    """
    Calculate a Hann-windowed power spectrum.

    Returns
    -------
    frequencies:
        Frequencies corresponding to the FFT bins.

    power:
        Squared magnitude of the FFT.
    """

    if len(s) == 0:
        return (
            np.array([], dtype=np.float64),
            np.array([], dtype=np.float64)
        )

    # Remove the signal's mean before spectral analysis.
    centered = s - np.mean(s)

    # The Hann window reduces spectral leakage.
    window = np.hanning(len(centered))

    windowed_signal = centered * window

    fft_values = np.fft.rfft(
        windowed_signal
    )

    power = np.abs(fft_values) ** 2

    frequencies = np.fft.rfftfreq(
        len(s),
        d=1.0 / fps
    )

    return frequencies, power


# ============================================================
# 4. CALCULATE SIGNAL-TO-NOISE RATIO
# ============================================================

def calculate_snr(
    rppg_signal: np.ndarray,
    fps: float = 30.0,
    hr_hz: float = 1.2
) -> float:
    """
    Estimate signal-to-noise ratio (SNR) in decibels.

    Signal power is measured around:
        - The estimated fundamental frequency.
        - Its first harmonic, when it falls inside the
          configured physiological frequency band.

    Noise power is measured in the remaining part of the
    physiological frequency band.

    Returns
    -------
    float
        Estimated SNR in decibels.

    Notes
    -----
    This is a heuristic spectral SNR estimate, not a
    direct measurement of physiological accuracy.
    """

    s = _prepare_signal(
        rppg_signal,
        fps
    )

    # A short recording cannot provide sufficient frequency
    # resolution for this quality calculation.
    if not _has_sufficient_duration(s, fps):
        return -20.0

    # Validate the estimated fundamental frequency.
    try:
        hr_hz = float(hr_hz)
    except (TypeError, ValueError):
        return -20.0

    if (
        not np.isfinite(hr_hz)
        or hr_hz < LOW_HZ
        or hr_hz > HIGH_HZ
    ):
        return -20.0

    frequencies, power = _get_power_spectrum(
        s,
        fps
    )

    # Restrict evaluation to the configured physiological band.
    physiological_mask = (
        (frequencies >= LOW_HZ)
        & (frequencies <= HIGH_HZ)
    )

    if not np.any(physiological_mask):
        return -20.0

    # Frequency band around the fundamental.
    fundamental_mask = (
        (frequencies >= hr_hz - FUNDAMENTAL_TOLERANCE_HZ)
        & (frequencies <= hr_hz + FUNDAMENTAL_TOLERANCE_HZ)
    )

    # Frequency band around the first harmonic.
    harmonic_hz = 2.0 * hr_hz

    harmonic_mask = (
        (frequencies >= harmonic_hz - FUNDAMENTAL_TOLERANCE_HZ)
        & (frequencies <= harmonic_hz + FUNDAMENTAL_TOLERANCE_HZ)
    )

    # Only count frequencies inside the configured band.
    signal_mask = (
        (fundamental_mask | harmonic_mask)
        & physiological_mask
    )

    noise_mask = (
        physiological_mask
        & ~signal_mask
    )

    if not np.any(signal_mask) or not np.any(noise_mask):
        return -20.0

    signal_power = float(
        np.sum(power[signal_mask])
    )

    noise_power = float(
        np.sum(power[noise_mask])
    )

    if (
        not np.isfinite(signal_power)
        or not np.isfinite(noise_power)
    ):
        return -20.0

    # No meaningful signal power.
    if signal_power <= EPSILON:
        return -20.0

    # Avoid division by zero.
    # This value is a capped estimate, not proof of a perfect signal.
    if noise_power <= EPSILON:
        return 20.0

    snr_db = 10.0 * np.log10(
        signal_power / noise_power
    )

    if not np.isfinite(snr_db):
        return -20.0

    return float(snr_db)


# ============================================================
# 5. CALCULATE SPECTRAL CONCENTRATION
# ============================================================

def calculate_spectral_concentration(
    rppg_signal: np.ndarray,
    fps: float = 30.0
) -> float:
    """
    Calculate the proportion of physiological-band power
    concentrated within +/- 0.1 Hz of the strongest frequency.

    Returns
    -------
    float
        Score between 0.0 and 1.0.

    Interpretation
    --------------
    A higher score means more power is concentrated near
    one dominant frequency.

    It does not prove that the frequency originates from
    a genuine heartbeat.
    """

    s = _prepare_signal(
        rppg_signal,
        fps
    )

    if not _has_sufficient_duration(s, fps):
        return 0.0

    # A constant signal has no meaningful spectral concentration.
    if np.std(s) < 1e-8:
        return 0.0

    frequencies, power = _get_power_spectrum(
        s,
        fps
    )

    physiological_mask = (
        (frequencies >= LOW_HZ)
        & (frequencies <= HIGH_HZ)
    )

    if not np.any(physiological_mask):
        return 0.0

    physiological_frequencies = frequencies[
        physiological_mask
    ]

    physiological_power = power[
        physiological_mask
    ]

    total_power = float(
        np.sum(physiological_power)
    )

    if (
        not np.isfinite(total_power)
        or total_power <= EPSILON
    ):
        return 0.0

    # Find the strongest frequency within the physiological band.
    peak_index = int(
        np.argmax(physiological_power)
    )

    peak_frequency = float(
        physiological_frequencies[peak_index]
    )

    # Select the bins near the strongest frequency.
    peak_mask = (
        (frequencies >= peak_frequency - CONCENTRATION_TOLERANCE_HZ)
        & (frequencies <= peak_frequency + CONCENTRATION_TOLERANCE_HZ)
        & physiological_mask
    )

    peak_power = float(
        np.sum(power[peak_mask])
    )

    concentration = peak_power / total_power

    if not np.isfinite(concentration):
        return 0.0

    return float(
        np.clip(concentration, 0.0, 1.0)
    )


# ============================================================
# 6. EVALUATE OVERALL SIGNAL QUALITY
# ============================================================

def evaluate_signal_quality(
    rppg_signal: np.ndarray,
    fps: float = 30.0
) -> float:
    """
    Calculate a heuristic rPPG signal-quality score.

    Returns
    -------
    float
        Composite quality score between 0.0 and 1.0.

    Formula
    -------
    60% normalized SNR
    40% spectral concentration

    Important
    ---------
    This score estimates spectral signal quality only.

    It is not:
        - A probability that a video is real.
        - A probability that a video is fake.
        - Proof that a physiological signal is genuine.
    """

    s = _prepare_signal(
        rppg_signal,
        fps
    )

    # Reject recordings that are too short for this evaluation.
    if not _has_sufficient_duration(s, fps):
        return 0.0

    # Reject constant or nearly constant signals.
    if np.std(s) < 1e-8:
        return 0.0

    frequencies, power = _get_power_spectrum(
        s,
        fps
    )

    physiological_mask = (
        (frequencies >= LOW_HZ)
        & (frequencies <= HIGH_HZ)
    )

    if not np.any(physiological_mask):
        return 0.0

    physiological_power = power[
        physiological_mask
    ]

    total_physiological_power = float(
        np.sum(physiological_power)
    )

    if (
        not np.isfinite(total_physiological_power)
        or total_physiological_power <= EPSILON
    ):
        return 0.0

    # Estimate the dominant frequency in the physiological band.
    physiological_frequencies = frequencies[
        physiological_mask
    ]

    peak_index = int(
        np.argmax(physiological_power)
    )

    peak_frequency = float(
        physiological_frequencies[peak_index]
    )

    # Calculate the two quality indicators.
    snr = calculate_snr(
        s,
        fps=fps,
        hr_hz=peak_frequency
    )

    spectral_concentration = calculate_spectral_concentration(
        s,
        fps=fps
    )

    # Map SNR from [-10, 10] dB into [0, 1].
    snr_normalized = np.clip(
        (snr + 10.0) / 20.0,
        0.0,
        1.0
    )

    # Preserve the original weighting.
    quality_score = (
        0.6 * snr_normalized
        + 0.4 * spectral_concentration
    )

    return float(
        np.clip(quality_score, 0.0, 1.0)
    )
