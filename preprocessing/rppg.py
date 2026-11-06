import warnings

import numpy as np
import torch
from sklearn.decomposition import FastICA
from sklearn.exceptions import ConvergenceWarning


# ============================================================
# 1. DEVICE AND VALIDATION UTILITIES
# ============================================================

def _get_device(device=None):
    """Choose CUDA when available, otherwise use CPU."""
    if device is None or str(device).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        selected = torch.device(device)
    except (TypeError, RuntimeError) as exc:
        raise ValueError("device must be 'auto', 'cuda', or 'cpu'.") from exc

    if selected.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but PyTorch cannot access a CUDA device. "
            "Check your NVIDIA driver and PyTorch CUDA installation."
        )

    return selected


def _validate_fps(fps: float) -> float:
    try:
        fps = float(fps)
    except (TypeError, ValueError) as exc:
        raise ValueError("fps must be a finite positive number.") from exc

    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be a finite positive number.")

    return fps


def _to_tensor(signal, device):
    """Convert a one-dimensional signal to float32 on the selected device."""
    tensor = torch.as_tensor(signal, dtype=torch.float32, device=device)

    if tensor.ndim != 1:
        raise ValueError("Signal must be one-dimensional.")

    if tensor.numel() < 2:
        raise ValueError("At least two samples are required.")

    if not torch.isfinite(tensor).all().item():
        raise ValueError("Signal contains NaN or infinity.")

    return tensor


def _to_numpy(tensor):
    """Return a detached CPU NumPy array for compatibility with dataset.py."""
    return tensor.detach().cpu().numpy().astype(np.float32, copy=False)


def _safe_detrend(signal, device=None):
    """
    Remove a linear trend using least-squares regression on the selected
    device. This replaces scipy.signal.detrend for this module.
    """
    device = _get_device(device)
    x = _to_tensor(signal, device)

    n = x.numel()
    time = torch.arange(n, dtype=x.dtype, device=device)
    time = time - time.mean()

    denominator = torch.sum(time * time)
    if denominator <= 1e-12:
        raise ValueError("Signal is too short to detrend.")

    slope = torch.sum(time * (x - x.mean())) / denominator
    trend = x.mean() + slope * time
    result = x - trend

    if not torch.isfinite(result).all().item():
        raise ValueError("Detrending produced invalid values.")

    return result


def _safe_normalize(signal, device=None):
    """Zero-mean, unit-standard-deviation normalization on the device."""
    device = _get_device(device)
    x = _to_tensor(signal, device)

    std = torch.std(x, unbiased=False)
    if not torch.isfinite(std).item() or std.item() < 1e-8:
        raise ValueError("Signal has insufficient variation for normalization.")

    result = (x - torch.mean(x)) / std

    if not torch.isfinite(result).all().item():
        raise ValueError("Normalization produced invalid values.")

    return result


def _safe_bandpass(
    signal,
    fps: float,
    device=None,
    low_hz: float = 0.7,
    high_hz: float = 2.5,
    order: int = 4,
):
    """
    GPU-compatible frequency-domain Butterworth-style bandpass.

    This uses an FFT-domain zero-phase magnitude response. It is not
    numerically identical to scipy.signal.sosfiltfilt, but it preserves
    the requested passband and avoids transferring the signal to SciPy.
    """
    fps = _validate_fps(fps)
    device = _get_device(device)
    x = _to_tensor(signal, device)

    try:
        low_hz = float(low_hz)
        high_hz = float(high_hz)
        order = int(order)
    except (TypeError, ValueError) as exc:
        raise ValueError("Invalid bandpass filter configuration.") from exc

    nyquist = fps / 2.0
    if not (0.0 < low_hz < high_hz < nyquist):
        raise ValueError(
            f"Require 0 < low_hz < high_hz < Nyquist ({nyquist:.3f} Hz)."
        )
    if order < 1:
        raise ValueError("Filter order must be a positive integer.")

    n = x.numel()
    if n < 4:
        raise ValueError("At least four samples are required for bandpass filtering.")

    # Butterworth high-pass and low-pass magnitude responses.
    frequencies = torch.fft.rfftfreq(n, d=1.0 / fps, device=device)
    safe_frequencies = torch.clamp(frequencies, min=1e-8)

    highpass = 1.0 / torch.sqrt(
        1.0 + (low_hz / safe_frequencies) ** (2 * order)
    )
    lowpass = 1.0 / torch.sqrt(
        1.0 + (safe_frequencies / high_hz) ** (2 * order)
    )
    response = highpass * lowpass

    # Remove DC exactly.
    response[0] = 0.0

    spectrum = torch.fft.rfft(x)
    filtered = torch.fft.irfft(spectrum * response, n=n)

    if not torch.isfinite(filtered).all().item():
        raise ValueError("Bandpass filtering produced invalid values.")

    return filtered


# ============================================================
# 2. RGB VALIDATION AND MISSING-VALUE INTERPOLATION
# ============================================================

def prepare_rgb_signal(
    rgb_signal: np.ndarray,
    min_valid_fraction: float = 0.8,
) -> np.ndarray:
    """
    Validate RGB measurements and interpolate missing rows.

    RGB input and interpolation remain on CPU because the input comes
    from OpenCV/MediaPipe. The numerical rPPG algorithms then transfer
    the prepared array to CUDA.
    """
    try:
        rgb = np.asarray(rgb_signal, dtype=np.float32).copy()
    except (TypeError, ValueError) as exc:
        raise ValueError("rgb_signal must contain numeric RGB measurements.") from exc

    try:
        min_valid_fraction = float(min_valid_fraction)
    except (TypeError, ValueError) as exc:
        raise ValueError("min_valid_fraction must be between 0 and 1.") from exc

    if not np.isfinite(min_valid_fraction) or not 0.0 <= min_valid_fraction <= 1.0:
        raise ValueError("min_valid_fraction must be between 0 and 1.")

    if rgb.ndim != 2 or rgb.shape[1] != 3:
        raise ValueError("rgb_signal must have shape (number_of_frames, 3).")

    total_count = len(rgb)
    if total_count < 10:
        raise ValueError(
            f"Not enough frames for rPPG processing: {total_count}. "
            "At least 10 frames are required."
        )

    valid_rows = np.all(np.isfinite(rgb), axis=1)
    valid_count = int(np.sum(valid_rows))
    valid_fraction = valid_count / total_count
    missing_count = total_count - valid_count

    print(
        f"[rPPG] RGB validation: {valid_count}/{total_count} valid frames "
        f"({valid_fraction:.1%}); {missing_count} invalid frames."
    )

    if valid_fraction < min_valid_fraction:
        raise ValueError(
            f"Too many missing RGB measurements: {valid_fraction:.1%} valid; "
            f"{min_valid_fraction:.1%} required."
        )

    frame_indices = np.arange(total_count)

    for channel in range(3):
        valid = np.isfinite(rgb[:, channel])
        valid_indices = frame_indices[valid]

        if len(valid_indices) < 2:
            raise ValueError(
                f"Insufficient valid samples in channel {('R', 'G', 'B')[channel]}."
            )

        rgb[:, channel] = np.interp(
            frame_indices,
            valid_indices,
            rgb[valid, channel],
        )

    if not np.all(np.isfinite(rgb)):
        raise ValueError("RGB signal remains invalid after interpolation.")

    return rgb


def _prepare_rgb_tensor(rgb_signal, device):
    rgb = prepare_rgb_signal(rgb_signal)
    tensor = torch.as_tensor(rgb, dtype=torch.float32, device=device)

    if not torch.isfinite(tensor).all().item():
        raise ValueError("Prepared RGB tensor contains invalid values.")

    return tensor


# ============================================================
# 3. GREEN-CHANNEL METHOD
# ============================================================

def green_method(rgb_signal, fps=30.0, device=None):
    fps = _validate_fps(fps)
    device = _get_device(device)
    rgb = _prepare_rgb_tensor(rgb_signal, device)

    green = rgb[:, 1]
    green = _safe_detrend(green, device)
    green = _safe_normalize(green, device)
    green = _safe_bandpass(green, fps, device)

    return _to_numpy(green)


# ============================================================
# 4. CHROM METHOD
# ============================================================

def chrom_method(rgb_signal, fps=30.0, device=None):
    fps = _validate_fps(fps)
    device = _get_device(device)
    rgb = _prepare_rgb_tensor(rgb_signal, device)

    eps = 1e-8
    means = torch.mean(rgb, dim=0)

    if not torch.isfinite(means).all().item() or torch.any(means <= eps).item():
        raise ValueError("RGB channels have invalid or near-zero means.")

    r_n, g_n, b_n = (rgb[:, i] / means[i] for i in range(3))

    xs = 3.0 * r_n - 2.0 * g_n
    ys = 1.5 * r_n + g_n - 1.5 * b_n

    xf = _safe_bandpass(xs, fps, device)
    yf = _safe_bandpass(ys, fps, device)

    std_x = torch.std(xf, unbiased=False)
    std_y = torch.std(yf, unbiased=False)

    if not torch.isfinite(std_x).item() or not torch.isfinite(std_y).item():
        raise ValueError("CHROM produced invalid projection statistics.")
    if std_y.item() < eps:
        raise ValueError("CHROM Y projection has insufficient variation.")

    alpha = std_x / (std_y + eps)
    pulse = xf - alpha * yf
    pulse = _safe_detrend(pulse, device)
    pulse = _safe_normalize(pulse, device)

    return _to_numpy(pulse)


# ============================================================
# 5. POS METHOD
# ============================================================

def pos_method(
    rgb_signal: np.ndarray,
    fps: float = 30.0,
    window_seconds: float = 1.6,
    device=None,
) -> np.ndarray:
    """
    Windowed POS implemented with PyTorch.

    When CUDA is available, the RGB tensor, window normalization,
    projections, overlap-add, detrending, normalization, and bandpass
    filtering run on the GPU.

    The returned NumPy array is moved back to CPU because the existing
    RPPGDataset expects NumPy-compatible output.
    """
    fps = _validate_fps(fps)
    device = _get_device(device)

    try:
        window_seconds = float(window_seconds)
    except (TypeError, ValueError) as exc:
        raise ValueError("window_seconds must be finite and positive.") from exc

    if not np.isfinite(window_seconds) or window_seconds <= 0:
        raise ValueError("window_seconds must be finite and positive.")

    rgb = _prepare_rgb_tensor(rgb_signal, device)
    n_frames = rgb.shape[0]
    window_length = int(round(window_seconds * fps))

    if window_length < 2:
        raise ValueError("POS window must contain at least two frames.")
    if n_frames < window_length:
        raise ValueError(
            f"Video is too short for POS: {n_frames} frames; "
            f"{window_length} required."
        )

    eps = 1e-8

    # Build all overlapping windows at once: (num_windows, window, RGB).
    windows = rgb.unfold(0, window_length, 1).permute(0, 2, 1)
    channel_means = torch.mean(windows, dim=1)

    valid_mean = (
        torch.isfinite(channel_means).all(dim=1)
        & torch.all(channel_means > eps, dim=1)
    )

    safe_means = torch.clamp(channel_means, min=eps)
    normalized = windows / safe_means.unsqueeze(1)

    # POS projections for every window.
    s1 = normalized[:, :, 1] - normalized[:, :, 2]
    s2 = (
        normalized[:, :, 1]
        + normalized[:, :, 2]
        - 2.0 * normalized[:, :, 0]
    )

    std_s1 = torch.std(s1, dim=1, unbiased=False)
    std_s2 = torch.std(s2, dim=1, unbiased=False)

    valid_windows = (
        valid_mean
        & torch.isfinite(std_s1)
        & torch.isfinite(std_s2)
        & (std_s2 >= eps)
    )

    alpha = std_s1 / (std_s2 + eps)
    pulse_windows = s1 + alpha.unsqueeze(1) * s2
    pulse_windows = pulse_windows - torch.mean(pulse_windows, dim=1, keepdim=True)

    valid_windows = valid_windows & torch.isfinite(pulse_windows).all(dim=1)

    windows_processed = int(valid_windows.sum().item())
    windows_skipped = int(valid_windows.numel() - windows_processed)

    print(
        f"[rPPG POS] Device: {device}; windows processed: {windows_processed}; "
        f"windows skipped: {windows_skipped}."
    )

    if windows_processed == 0:
        raise ValueError("POS could not produce a valid pulse signal.")

    # Overlap-add. Each window contributes to its corresponding frame range.
    pulse_windows = pulse_windows * valid_windows.unsqueeze(1).to(rgb.dtype)

    pulse_sum = torch.zeros(n_frames, dtype=rgb.dtype, device=device)
    pulse_count = torch.zeros(n_frames, dtype=rgb.dtype, device=device)

    starts = torch.arange(
        window_length, device=device
    ).unsqueeze(0) + torch.arange(
        pulse_windows.shape[0], device=device
    ).unsqueeze(1)

    pulse_sum.scatter_add_(0, starts.reshape(-1), pulse_windows.reshape(-1))
    pulse_count.scatter_add_(
        0,
        starts.reshape(-1),
        valid_windows.to(rgb.dtype).unsqueeze(1).expand(-1, window_length).reshape(-1),
    )

    valid_frames = pulse_count > 0
    if int(valid_frames.sum().item()) < 2:
        raise ValueError("POS produced fewer than two usable output samples.")

    pulse = torch.zeros_like(pulse_sum)
    pulse[valid_frames] = pulse_sum[valid_frames] / pulse_count[valid_frames]

    # Fill any uncovered frames by linear interpolation on the GPU.
    if not torch.all(valid_frames).item():
        valid_indices = torch.nonzero(valid_frames, as_tuple=False).squeeze(1)
        missing_indices = torch.nonzero(~valid_frames, as_tuple=False).squeeze(1)

        right_positions = torch.searchsorted(valid_indices, missing_indices)
        right_positions = torch.clamp(right_positions, 0, valid_indices.numel() - 1)
        left_positions = torch.clamp(right_positions - 1, 0, valid_indices.numel() - 1)

        left_idx = valid_indices[left_positions]
        right_idx = valid_indices[right_positions]

        left_val = pulse[left_idx]
        right_val = pulse[right_idx]

        denominator = (right_idx - left_idx).to(rgb.dtype)
        fraction = torch.where(
            denominator > 0,
            (missing_indices - left_idx).to(rgb.dtype) / torch.clamp(denominator, min=1),
            torch.zeros_like(denominator),
        )
        pulse[missing_indices] = left_val + fraction * (right_val - left_val)

    pulse = _safe_detrend(pulse, device)
    pulse = _safe_normalize(pulse, device)
    pulse = _safe_bandpass(pulse, fps, device)

    print("[rPPG POS] Output shape:", tuple(pulse.shape))
    print("[rPPG POS] Device:", device)
    print("[rPPG POS] All finite:", torch.isfinite(pulse).all().item())
    print("[rPPG POS] Standard deviation:", torch.std(pulse, unbiased=False).item())

    return _to_numpy(pulse)


# ============================================================
# 6. ICA METHOD
# ============================================================

def ica_method(rgb_signal, fps=30.0, device=None):
    """
    ICA remains CPU-based because sklearn FastICA does not support CUDA.
    Its selected component is transferred to the requested device for
    detrending, normalization, spectral analysis, and filtering.
    """
    fps = _validate_fps(fps)
    device = _get_device(device)
    rgb = prepare_rgb_signal(rgb_signal)

    if len(rgb) < max(30, int(round(fps))):
        print("[rPPG ICA] Short recording; using green-channel fallback.")
        return green_method(rgb, fps=fps, device=device)

    means = np.mean(rgb, axis=0)
    stds = np.std(rgb, axis=0)

    if (
        not np.all(np.isfinite(means))
        or not np.all(np.isfinite(stds))
        or np.any(stds < 1e-8)
    ):
        print("[rPPG ICA] Invalid channel statistics; using green fallback.")
        return green_method(rgb, fps=fps, device=device)

    standardized = (rgb - means) / stds

    ica = FastICA(
        n_components=3,
        random_state=42,
        max_iter=1000,
        tol=0.001,
        whiten="unit-variance",
    )

    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            components = ica.fit_transform(standardized)
    except (
        ValueError,
        FloatingPointError,
        np.linalg.LinAlgError,
        ConvergenceWarning,
    ):
        print("[rPPG ICA] ICA failed; using green-channel fallback.")
        return green_method(rgb, fps=fps, device=device)

    if (
        components.ndim != 2
        or components.shape != (len(rgb), 3)
        or not np.all(np.isfinite(components))
    ):
        print("[rPPG ICA] Invalid components; using green fallback.")
        return green_method(rgb, fps=fps, device=device)

    # Spectral selection is performed on the selected device.
    component_tensor = torch.as_tensor(
        components, dtype=torch.float32, device=device
    )
    component_tensor = component_tensor - component_tensor.mean(dim=0, keepdim=True)

    frequencies = torch.fft.rfftfreq(
        len(rgb), d=1.0 / fps, device=device
    )
    frequency_mask = (frequencies >= 0.7) & (frequencies <= 2.5)

    if not frequency_mask.any().item():
        print("[rPPG ICA] No usable frequency bins; using green fallback.")
        return green_method(rgb, fps=fps, device=device)

    powers = []
    for index in range(3):
        component = _safe_detrend(component_tensor[:, index], device)
        spectrum = torch.fft.rfft(component)
        power = torch.abs(spectrum) ** 2
        powers.append(torch.sum(power[frequency_mask]))

    powers = torch.stack(powers)
    best_index = int(torch.argmax(powers).item())

    if not torch.isfinite(powers[best_index]).item():
        print("[rPPG ICA] No usable component; using green fallback.")
        return green_method(rgb, fps=fps, device=device)

    try:
        selected = _safe_detrend(component_tensor[:, best_index], device)
        selected = _safe_normalize(selected, device)
        selected = _safe_bandpass(selected, fps, device)
        return _to_numpy(selected)
    except ValueError:
        print("[rPPG ICA] Selected component unusable; using green fallback.")
        return green_method(rgb, fps=fps, device=device)
