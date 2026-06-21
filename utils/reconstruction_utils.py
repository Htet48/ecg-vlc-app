"""
Signal Reconstruction Methods
Implements 3 classical methods + combined approach
"""

import numpy as np
import pywt
from scipy import signal
from scipy.interpolate import interp1d, CubicSpline
from scipy.signal import savgol_filter


# =====================================
# Method 1: Interpolation
# =====================================

def interpolation_reconstruction(received, method='cubic'):
    """
    Reconstruct signal using interpolation
    Addresses: IM/DD clipping or occlusion → gaps in waveform
    
    Args:
        received: Received signal with clipping/gaps
        method: 'linear' or 'cubic'
    
    Returns:
        reconstructed: Interpolated signal
    """
    n = len(received)
    
    # Detect clipped regions (flat tops or abnormally low values)
    # Assume clipping threshold at 0.05 and 0.95 for normalized signals
    clipped_low = received < 0.05
    clipped_high = received > 0.95
    clipped = clipped_low | clipped_high
    
    # If too many clipped samples, use simple interpolation
    if clipped.sum() > 0.5 * n:
        # Fallback: use cubic spline on all points
        x = np.arange(n)
        if method == 'cubic':
            cs = CubicSpline(x, received)
            return cs(x)
        else:
            f = interp1d(x, received, kind='linear', fill_value='extrapolate')
            return f(x)
    
    # Get valid (non-clipped) samples
    valid_idx = np.where(~clipped)[0]
    valid_values = received[valid_idx]
    
    if len(valid_idx) < 4:
        # Not enough points, return original
        return received.copy()
    
    # Interpolate at clipped locations
    all_idx = np.arange(n)
    
    if method == 'cubic':
        cs = CubicSpline(valid_idx, valid_values, extrapolate=True)
        reconstructed = received.copy()
        reconstructed[clipped] = cs(all_idx[clipped])
    else:
        f = interp1d(valid_idx, valid_values, kind='linear', 
                    bounds_error=False, fill_value='extrapolate')
        reconstructed = received.copy()
        reconstructed[clipped] = f(all_idx[clipped])
    
    return reconstructed


# =====================================
# Method 2: Wavelet Denoising
# =====================================

def wavelet_denoising_reconstruction(received, wavelet='db4', level=4, threshold_mode='soft'):
    """
    Reconstruct signal using wavelet denoising
    Addresses: Thermal/shot noise → high-frequency contamination
    
    Args:
        received: Noisy received signal
        wavelet: Wavelet type ('db4', 'sym4', 'coif3')
        level: Decomposition level
        threshold_mode: 'soft' or 'hard'
    
    Returns:
        reconstructed: Denoised signal
    """
    # Perform wavelet decomposition
    coeffs = pywt.wavedec(received, wavelet, level=level)
    
    # Universal threshold (Donoho & Johnstone)
    # Estimate noise from finest detail coefficients
    sigma = np.median(np.abs(coeffs[-1])) / 0.6745
    threshold = sigma * np.sqrt(2 * np.log(len(received)))
    
    # Apply threshold to all detail coefficients
    coeffs_thresh = [coeffs[0]]  # Keep approximation
    for i in range(1, len(coeffs)):
        coeffs_thresh.append(pywt.threshold(coeffs[i], threshold, mode=threshold_mode))
    
    # Reconstruct
    reconstructed = pywt.waverec(coeffs_thresh, wavelet)
    
    # Handle length mismatch
    if len(reconstructed) > len(received):
        reconstructed = reconstructed[:len(received)]
    elif len(reconstructed) < len(received):
        # Pad with last value
        reconstructed = np.pad(reconstructed, (0, len(received) - len(reconstructed)), 
                              mode='edge')
    
    return reconstructed


# =====================================
# Method 3: OFDM Clipping Mitigation
# =====================================

def ofdm_clipping_mitigation(received, n_fft=64, cp_len=16, max_iter=5):
    """
    Mitigate OFDM clipping distortion using iterative clipping and filtering
    Addresses: OFDM PAPR → LED nonlinearity → severe clipping
    
    This implements a simplified version of iterative clipping mitigation:
    1. Detect clipped samples
    2. Estimate clipping noise
    3. Filter in frequency domain
    4. Iterate
    
    Args:
        received: Signal with OFDM clipping
        n_fft: FFT size
        cp_len: Cyclic prefix length
        max_iter: Maximum iterations
    
    Returns:
        reconstructed: Mitigated signal
    """
    n = len(received)
    reconstructed = received.copy()
    
    # Clipping threshold (assume normalized signal)
    clip_threshold_low = 0.05
    clip_threshold_high = 0.95
    
    for iteration in range(max_iter):
        # Detect clipped samples
        clipped_low = reconstructed < clip_threshold_low
        clipped_high = reconstructed > clip_threshold_high
        clipped = clipped_low | clipped_high
        
        if clipped.sum() == 0:
            break
        
        # Estimate clipping noise
        clipping_noise = np.zeros_like(reconstructed)
        clipping_noise[clipped_high] = reconstructed[clipped_high] - clip_threshold_high
        clipping_noise[clipped_low] = reconstructed[clipped_low] - clip_threshold_low
        
        # Transform to frequency domain (pad to n_fft if needed)
        if n > n_fft:
            n_fft = int(2 ** np.ceil(np.log2(n)))
        clipping_noise_padded = np.zeros(n_fft)
        clipping_noise_padded[:n] = clipping_noise
        freq_noise = np.fft.fft(clipping_noise_padded)
        
        # Filter: Attenuate out-of-band components
        # Keep only in-band frequencies (assume useful band is central 50%)
        band_start = n_fft // 4
        band_end = 3 * n_fft // 4
        freq_filtered = freq_noise.copy()
        freq_filtered[:band_start] = 0
        freq_filtered[band_end:] = 0
        
        # Transform back
        time_filtered = np.real(np.fft.ifft(freq_filtered))[:n]
        
        # Subtract filtered clipping noise
        reconstructed -= 0.5 * time_filtered  # Damping factor
        
        # Re-apply clipping to prevent amplification
        reconstructed = np.clip(reconstructed, 0, 1)
    
    return reconstructed


# =====================================
# Combined Reconstruction (Sequential)
# =====================================

def combined_classical_reconstruction(received, 
                                     apply_interpolation=True,
                                     apply_wavelet=True,
                                     apply_ofdm=True,
                                     wavelet='db4',
                                     ofdm_iter=3):
    """
    Apply all three classical methods sequentially
    
    Order: Interpolation → Wavelet Denoising → OFDM Mitigation
    
    Rationale:
    1. First restore missing/clipped samples (Interpolation)
    2. Then remove noise (Wavelet)
    3. Finally mitigate OFDM-specific distortion
    
    Args:
        received: Received signal
        apply_interpolation: Enable interpolation
        apply_wavelet: Enable wavelet denoising
        apply_ofdm: Enable OFDM clipping mitigation
        wavelet: Wavelet type for denoising
        ofdm_iter: Iterations for OFDM mitigation
    
    Returns:
        reconstructed: Final reconstructed signal
        intermediate: Dictionary of intermediate signals
    """
    intermediate = {'original': received.copy()}
    
    reconstructed = received.copy()
    
    # Step 1: Interpolation (fix clipping/gaps)
    if apply_interpolation:
        reconstructed = interpolation_reconstruction(reconstructed, method='cubic')
        intermediate['after_interpolation'] = reconstructed.copy()
    
    # Step 2: Wavelet Denoising (remove noise)
    if apply_wavelet:
        reconstructed = wavelet_denoising_reconstruction(reconstructed, wavelet=wavelet)
        intermediate['after_wavelet'] = reconstructed.copy()
    
    # Step 3: OFDM Clipping Mitigation (fix spectral distortion)
    if apply_ofdm:
        reconstructed = ofdm_clipping_mitigation(reconstructed, max_iter=ofdm_iter)
        intermediate['after_ofdm'] = reconstructed.copy()
    
    intermediate['final'] = reconstructed
    
    return reconstructed, intermediate


# =====================================
# Alternative Methods (for comparison)
# =====================================

def savitzky_golay_reconstruction(received, window_length=11, polyorder=3):
    """
    Savitzky-Golay filter for smoothing
    Good for preserving peaks while reducing noise
    
    Args:
        received: Received signal
        window_length: Filter window length (must be odd)
        polyorder: Polynomial order
    
    Returns:
        reconstructed: Smoothed signal
    """
    n = len(received)
    
    # Ensure odd window length
    if window_length % 2 == 0:
        window_length += 1
    
    # Ensure window fits
    if window_length > n:
        window_length = n if n % 2 == 1 else n - 1
    
    if window_length < polyorder + 2:
        return received.copy()
    
    reconstructed = savgol_filter(received, window_length, polyorder)
    
    return reconstructed


def wiener_reconstruction(received, noise_power=None, mysize=5):
    """
    Wiener filtering for optimal MMSE denoising
    
    Args:
        received: Noisy signal
        noise_power: Estimated noise power (if None, auto-estimate)
        mysize: Filter size
    
    Returns:
        reconstructed: Filtered signal
    """
    from scipy.signal import wiener
    
    if noise_power is None:
        # Estimate noise from high-frequency components
        smoothed = savgol_filter(received, 11, 3)
        noise_power = np.var(received - smoothed)
    
    reconstructed = wiener(received, mysize=mysize, noise=noise_power)
    
    return reconstructed


# =====================================
# Reconstruction Factory
# =====================================

def reconstruct_signal(received, method='combined', **kwargs):
    """
    Factory function for signal reconstruction
    
    Args:
        received: Received signal
        method: Reconstruction method
            - 'interpolation': Only interpolation
            - 'wavelet': Only wavelet denoising
            - 'ofdm': Only OFDM mitigation
            - 'combined': All three methods sequentially
            - 'savgol': Savitzky-Golay filter
            - 'wiener': Wiener filter
        **kwargs: Method-specific parameters
    
    Returns:
        reconstructed: Reconstructed signal
        info: Dictionary with method information
    """
    info = {'method': method}
    
    if method == 'interpolation':
        reconstructed = interpolation_reconstruction(received, 
                                                    method=kwargs.get('interp_method', 'cubic'))
        
    elif method == 'wavelet':
        reconstructed = wavelet_denoising_reconstruction(received,
                                                        wavelet=kwargs.get('wavelet', 'db4'),
                                                        level=kwargs.get('level', 4))
        
    elif method == 'ofdm':
        reconstructed = ofdm_clipping_mitigation(received,
                                                n_fft=kwargs.get('n_fft', 64),
                                                max_iter=kwargs.get('max_iter', 5))
        
    elif method == 'combined':
        reconstructed, intermediate = combined_classical_reconstruction(
            received,
            apply_interpolation=kwargs.get('apply_interpolation', True),
            apply_wavelet=kwargs.get('apply_wavelet', True),
            apply_ofdm=kwargs.get('apply_ofdm', True),
            wavelet=kwargs.get('wavelet', 'db4'),
            ofdm_iter=kwargs.get('ofdm_iter', 3)
        )
        info['intermediate'] = intermediate
        
    elif method == 'savgol':
        reconstructed = savitzky_golay_reconstruction(received,
                                                     window_length=kwargs.get('window_length', 11),
                                                     polyorder=kwargs.get('polyorder', 3))
        
    elif method == 'wiener':
        reconstructed = wiener_reconstruction(received,
                                             noise_power=kwargs.get('noise_power', None),
                                             mysize=kwargs.get('mysize', 5))
        
    else:
        raise ValueError(f"Unknown reconstruction method: {method}")
    
    return reconstructed, info
