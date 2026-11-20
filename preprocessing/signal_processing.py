
import numpy as np
import torch
from scipy import signal


# ============================================================
# 1. DEVICE MANAGEMENT AND VALIDATION
# ============================================================

def _get_device(device=None):
    """
    Select the processing device.

    CUDA is used when available if device is None or 'auto'.
    Explicit CUDA requests raise an error if CUDA is unavailable.
    """
    if device is None or str(device).lower() == "auto":
        return torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    try:
        selected = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError(
            "device must be 'auto', 'cuda', or 'cpu'."
        ) from exc

    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but CUDA is not available."
        )

    return selected


def _to_tensor(s, device=None):
    """
    Convert NumPy arrays or PyTorch tensors to float64 tensors.

    Returns:
        tensor
        was_tensor: whether the original input was a PyTorch tensor.
    """
    device = _get_device(device)

    was_tensor = isinstance(s, torch.Tensor)

    try:
        if was_tensor:
            tensor = s.to(
                device=device,
                dtype=torch.float64
            )
        else:
            tensor = torch.as_tensor(
                np.asarray(s, dtype=np.float64),
                dtype=torch.float64,
                device=device
            )
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(
            "Input must contain valid numeric values."
        ) from exc

    return tensor, was_tensor


def _restore_type(tensor, was_tensor):
    """
    Preserve the original input type.

    PyTorch input -> PyTorch output.
    NumPy input   -> NumPy output.
    """
    if was_tensor:
        return tensor

    return tensor.detach().cpu().numpy()


def _validate_signal(s):
    """Validate a one-dimensional finite signal."""
    if s.ndim != 1:
        raise ValueError(
            "Signal must be one-dimensional."
        )

    if not torch.isfinite(s).all().item():
        raise ValueError(
            "Signal contains NaN or infinite values."
        )


def _validate_fps(fps):
    """Validate a finite positive sampling rate."""
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

    return fps


# ============================================================
# 2. NORMALIZE SIGNAL
# ============================================================

def normalize_signal(s, device=None):
    """
    Normalize a one-dimensional signal using PyTorch.

    Formula:
        normalized = (signal - mean) / standard_deviation

    A constant signal returns zeros.
    """
    s, was_tensor = _to_tensor(s, device)

    _validate_signal(s)

    if s.numel() == 0:
        return _restore_type(s.clone(), was_tensor)

    mean = torch.mean(s)
    std = torch.std(s, unbiased=False)

    if not torch.isfinite(mean).item():
        raise ValueError(
            "Signal mean is invalid."
        )

    if not torch.isfinite(std).item():
        raise ValueError(
            "Signal standard deviation is invalid."
        )

    if std.item() < 1e-8:
        return _restore_type(
            torch.zeros_like(s),
            was_tensor
        )

    normalized = (s - mean) / std

    if not torch.isfinite(normalized).all().item():
        raise ValueError(
            "Normalization produced invalid values."
        )

    return _restore_type(normalized, was_tensor)


# ============================================================
# 3. DETREND SIGNAL
# ============================================================

def detrend_signal(s, device=None):
    """
    Remove a linear trend using least-squares regression.

    Fits:
        y = slope * x + intercept

    Then subtracts the fitted trend from the original signal.
    """
    s, was_tensor = _to_tensor(s, device)

    _validate_signal(s)

    n = s.numel()

    if n < 2:
        return _restore_type(s.clone(), was_tensor)

    x = torch.arange(
        n,
        device=s.device,
        dtype=s.dtype
    )

    x_mean = torch.mean(x)
    y_mean = torch.mean(s)

    centered_x = x - x_mean
    centered_y = s - y_mean

    denominator = torch.sum(centered_x ** 2)

    if denominator.item() < 1e-12:
        return _restore_type(
            s.clone(),
            was_tensor
        )

    slope = torch.sum(
        centered_x * centered_y
    ) / denominator

    trend = slope * centered_x + y_mean
    detrended = s - trend

    if not torch.isfinite(detrended).all().item():
        raise ValueError(
            "Detrending produced invalid values."
        )

    return _restore_type(detrended, was_tensor)


# ============================================================
# 4. BUTTERWORTH FFT BANDPASS FILTER
# ============================================================

def _butterworth_fft_filter(
    s,
    fps,
    low_hz,
    high_hz,
    order
):
    """
    Apply a Butterworth bandpass response using PyTorch FFT.

    SciPy:
        Designs the Butterworth SOS coefficients.

    PyTorch:
        Calculates the frequency response and filters the signal
        on the selected device.

    The squared magnitude response approximates forward-backward
    filtering with zero phase.

    Reflection padding reduces boundary artifacts.

    Note:
        This is an FFT-domain approximation of forward-backward
        filtering, not an exact replacement for scipy.signal.sosfiltfilt.
    """
    n = s.numel()

    if n < 2:
        return s.clone()

    # Design the Butterworth filter on CPU.
    try:
        sos = signal.butter(
            order,
            [low_hz, high_hz],
            btype="bandpass",
            fs=fps,
            output="sos"
        )
    except ValueError as exc:
        raise ValueError(
            f"Could not design Butterworth filter: {exc}"
        ) from exc

    sos = torch.as_tensor(
        sos,
        dtype=s.dtype,
        device=s.device
    )

    # Reflection padding.
    pad_length = min(
        3 * (2 * len(sos) + 1),
        n - 1
    )

    if pad_length > 0:
        padded = torch.cat([
            torch.flip(s[:pad_length], dims=[0]),
            s,
            torch.flip(s[-pad_length:], dims=[0])
        ])
    else:
        padded = s

    padded_length = padded.numel()

    # FFT.
    spectrum = torch.fft.rfft(padded)

    frequencies = torch.fft.rfftfreq(
        padded_length,
        d=1.0 / fps,
        device=s.device,
        dtype=s.dtype
    )

    omega = 2.0 * torch.pi * frequencies / fps

    z1 = torch.exp(-1j * omega)

    # Calculate the Butterworth frequency response.
    response = torch.ones(
        frequencies.shape,
        dtype=torch.complex128,
        device=s.device
    )

    for section in sos:
        b0, b1, b2, a0, a1, a2 = section

        numerator = (
            b0
            + b1 * z1
            + b2 * z1 ** 2
        )

        denominator = (
            a0
            + a1 * z1
            + a2 * z1 ** 2
        )

        if torch.any(torch.abs(denominator) < 1e-12).item():
            raise ValueError(
                "Invalid Butterworth filter denominator."
            )

        response = response * (
            numerator / denominator
        )

    # Approximate forward-backward response.
    magnitude_squared = torch.abs(response) ** 2

    filtered_spectrum = spectrum * magnitude_squared

    # Inverse FFT.
    filtered = torch.fft.irfft(
        filtered_spectrum,
        n=padded_length
    )

    # Remove reflection padding.
    if pad_length > 0:
        filtered = filtered[
            pad_length:pad_length + n
        ]

    if not torch.isfinite(filtered).all().item():
        raise ValueError(
            "Bandpass filtering produced invalid values."
        )

    return filtered


def bandpass_filter(
    s,
    fps=30.0,
    low_hz=0.7,
    high_hz=2.5,
    order=4,
    device=None
):
    """
    Apply Butterworth bandpass filtering using PyTorch FFT.

    Default frequency range:
        0.7–2.5 Hz
        Approximately 42–150 BPM.

    The result preserves the original input type.
    """
    s, was_tensor = _to_tensor(s, device)

    _validate_signal(s)

    if s.numel() < 2:
        return _restore_type(s.clone(), was_tensor)

    fps = _validate_fps(fps)

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

    nyquist = fps / 2.0

    if not (0.0 < low_hz < high_hz < nyquist):
        raise ValueError(
            "Require 0 < low_hz < high_hz < Nyquist "
            f"({nyquist:.3f} Hz)."
        )

    if (
        isinstance(order, (bool, np.bool_))
        or not isinstance(order, (int, np.integer))
        or order < 1
    ):
        raise ValueError(
            "Filter order must be a positive integer."
        )

    filtered = _butterworth_fft_filter(
        s,
        fps,
        low_hz,
        high_hz,
        int(order)
    )

    return _restore_type(filtered, was_tensor)


# ============================================================
# 5. PROCESS RAW RGB
# ============================================================

def process_raw_rgb(
    rgb_signal,
    fps=30.0,
    device=None
):
    """
    Process raw RGB measurements channel by channel.

    Input:
        Shape: (number_of_frames, 3)

    Channel order:
        Column 0 = Red
        Column 1 = Green
        Column 2 = Blue

    Processing for each channel:
        1. Detrend.
        2. Normalize.
        3. Apply bandpass filter.

    This function processes RGB channels independently.
    It does NOT calculate a POS pulse signal.

    Do not use this function as an additional processing step
    after pos_method() in rppg.py.
    """
    rgb, was_tensor = _to_tensor(
        rgb_signal,
        device
    )

    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError(
            "rgb_signal must have shape "
            "(number_of_frames, 3)."
        )

    if not torch.isfinite(rgb).all().item():
        raise ValueError(
            "RGB signal contains missing or invalid values."
        )

    if rgb.shape[0] == 0:
        empty = torch.empty(
            (0, 3),
            dtype=rgb.dtype,
            device=rgb.device
        )

        return _restore_type(empty, was_tensor)

    fps = _validate_fps(fps)

    processed_channels = []

    for channel in range(3):
        raw_channel = rgb[:, channel]

        detrended = detrend_signal(
            raw_channel,
            device=rgb.device
        )

        normalized = normalize_signal(
            detrended,
            device=rgb.device
        )

        filtered = bandpass_filter(
            normalized,
            fps=fps,
            device=rgb.device
        )

        processed_channels.append(
            filtered
        )

    processed = torch.stack(
        processed_channels,
        dim=1
    )

    if not torch.isfinite(processed).all().item():
        raise ValueError(
            "RGB processing produced invalid values."
        )

    return _restore_type(processed, was_tensor)
