"""
Performance Metrics for ECG Reconstruction
Three-level evaluation: Signal Fidelity, Morphology, Clinical Diagnostic
"""

import numpy as np
import pywt
from scipy import signal
from scipy.spatial.distance import euclidean
from scipy.stats import pearsonr


# =====================================
# Level 1: Signal Fidelity Metrics
# =====================================

def compute_mse(original, reconstructed):
    """Mean Squared Error"""
    return np.mean((original - reconstructed) ** 2)


def compute_rmse(original, reconstructed):
    """Root Mean Squared Error"""
    return np.sqrt(compute_mse(original, reconstructed))


def compute_prd(original, reconstructed):
    """
    Percent Root-mean-square Difference
    Lower is better (0% = perfect reconstruction)
    """
    numerator = np.sqrt(np.sum((original - reconstructed) ** 2))
    denominator = np.sqrt(np.sum(original ** 2))
    
    if denominator == 0:
        return np.inf
    
    prd = 100 * numerator / denominator
    return prd


def compute_snr(original, reconstructed):
    """
    Signal-to-Noise Ratio in dB
    Higher is better
    """
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - reconstructed) ** 2)
    
    if noise_power == 0:
        return np.inf
    
    snr_db = 10 * np.log10(signal_power / noise_power)
    return snr_db


def compute_correlation_coefficient(original, reconstructed):
    """
    Pearson Correlation Coefficient
    Range: [-1, 1], higher is better
    """
    if len(original) < 2:
        return 0.0
    
    corr, _ = pearsonr(original, reconstructed)
    return corr


def compute_nmse(original, reconstructed):
    """
    Normalized Mean Squared Error
    Lower is better
    """
    mse = compute_mse(original, reconstructed)
    signal_var = np.var(original)
    
    if signal_var == 0:
        return np.inf
    
    nmse = mse / signal_var
    return nmse


# =====================================
# Level 2: Morphology Preservation Metrics
# =====================================

def compute_dtw_distance(original, reconstructed):
    """
    Dynamic Time Warping distance
    Measures temporal alignment quality
    Lower is better
    """
    try:
        from fastdtw import fastdtw
        distance, _ = fastdtw(original, reconstructed, dist=euclidean)
        return distance
    except (ImportError, Exception):
        # Fallback: simple DTW implementation
        return compute_simple_dtw(original, reconstructed)


def compute_simple_dtw(x, y):
    """Simple DTW implementation (fallback)"""
    n, m = len(x), len(y)
    dtw_matrix = np.full((n + 1, m + 1), np.inf)
    dtw_matrix[0, 0] = 0
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            cost = abs(x[i-1] - y[j-1])
            dtw_matrix[i, j] = cost + min(
                dtw_matrix[i-1, j],      # insertion
                dtw_matrix[i, j-1],      # deletion
                dtw_matrix[i-1, j-1]     # match
            )
    
    return dtw_matrix[n, m]


def compute_wavelet_energy_preservation(original, reconstructed, wavelet='db4', level=4):
    """
    Wavelet energy preservation ratio
    Measures how well energy is preserved across scales
    Closer to 100% is better
    """
    # Decompose both signals
    coeffs_orig = pywt.wavedec(original, wavelet, level=level)
    coeffs_recon = pywt.wavedec(reconstructed, wavelet, level=level)
    
    # Compute total energy
    energy_orig = sum(np.sum(c ** 2) for c in coeffs_orig)
    energy_recon = sum(np.sum(c ** 2) for c in coeffs_recon)
    
    if energy_orig == 0:
        return 0.0
    
    preservation = 100 * energy_recon / energy_orig
    return preservation


def compute_frequency_domain_correlation(original, reconstructed):
    """
    Correlation in frequency domain
    Measures spectral similarity
    """
    # FFT
    fft_orig = np.fft.fft(original)
    fft_recon = np.fft.fft(reconstructed)
    
    # Magnitude spectrum
    mag_orig = np.abs(fft_orig)
    mag_recon = np.abs(fft_recon)
    
    # Correlation
    if len(mag_orig) < 2:
        return 0.0
    
    corr, _ = pearsonr(mag_orig, mag_recon)
    return corr


def detect_pqrst_points(ecg_signal, fs=360):
    """
    Simplified PQRST detection
    Returns approximate locations of P, Q, R, S, T peaks
    
    Args:
        ecg_signal: ECG signal
        fs: Sampling frequency
    
    Returns:
        pqrst: Dictionary with peak locations
    """
    from scipy.signal import find_peaks
    
    # Normalize
    norm_signal = (ecg_signal - ecg_signal.min()) / (ecg_signal.max() - ecg_signal.min() + 1e-12)
    
    # R peak (highest)
    r_peaks, _ = find_peaks(norm_signal, height=0.7 * np.max(norm_signal), distance=int(0.6 * fs))
    
    if len(r_peaks) == 0:
        return None
    
    # Use first R peak as reference
    r_idx = r_peaks[0]
    
    # Q and S are valleys around R
    search_window = int(0.05 * fs)  # 50ms window
    
    # Q: before R
    q_search_start = max(0, r_idx - search_window)
    q_segment = norm_signal[q_search_start:r_idx]
    if len(q_segment) > 0:
        q_idx = q_search_start + np.argmin(q_segment)
    else:
        q_idx = r_idx
    
    # S: after R
    s_search_end = min(len(norm_signal), r_idx + search_window)
    s_segment = norm_signal[r_idx:s_search_end]
    if len(s_segment) > 0:
        s_idx = r_idx + np.argmin(s_segment)
    else:
        s_idx = r_idx
    
    # P: before Q (smaller peak)
    p_search_start = max(0, q_idx - int(0.1 * fs))
    p_segment = norm_signal[p_search_start:q_idx]
    if len(p_segment) > 10:
        p_peaks_local, _ = find_peaks(p_segment, height=0.3 * np.max(norm_signal))
        if len(p_peaks_local) > 0:
            p_idx = p_search_start + p_peaks_local[-1]
        else:
            p_idx = q_idx - int(0.05 * fs)
    else:
        p_idx = q_idx
    
    # T: after S (smaller peak)
    t_search_end = min(len(norm_signal), s_idx + int(0.2 * fs))
    t_segment = norm_signal[s_idx:t_search_end]
    if len(t_segment) > 10:
        t_peaks_local, _ = find_peaks(t_segment, height=0.2 * np.max(norm_signal))
        if len(t_peaks_local) > 0:
            t_idx = s_idx + t_peaks_local[0]
        else:
            t_idx = s_idx + int(0.1 * fs)
    else:
        t_idx = s_idx
    
    return {
        'P': p_idx,
        'Q': q_idx,
        'R': r_idx,
        'S': s_idx,
        'T': t_idx
    }


def compute_pqrst_deviation(original, reconstructed, fs=360):
    """
    Compute deviations in PQRST peak locations
    Returns mean absolute deviation in samples
    """
    pqrst_orig = detect_pqrst_points(original, fs)
    pqrst_recon = detect_pqrst_points(reconstructed, fs)
    
    if pqrst_orig is None or pqrst_recon is None:
        return np.nan
    
    deviations = []
    for key in ['P', 'Q', 'R', 'S', 'T']:
        dev = abs(pqrst_orig[key] - pqrst_recon[key])
        deviations.append(dev)
    
    mean_dev = np.mean(deviations)
    return mean_dev


# =====================================
# Level 3: Clinical Diagnostic Metrics
# =====================================

def detect_qrs_complexes(ecg_signal, fs=360):
    """
    Detect QRS complexes (R peaks)
    
    Returns:
        r_peaks: Array of R peak indices
    """
    from scipy.signal import find_peaks
    
    # Normalize
    norm_signal = (ecg_signal - ecg_signal.min()) / (ecg_signal.max() - ecg_signal.min() + 1e-12)
    
    # Detect peaks
    r_peaks, _ = find_peaks(norm_signal, 
                           height=0.6 * np.max(norm_signal),
                           distance=int(0.5 * fs))
    
    return r_peaks


def compute_qrs_detection_metrics(original, reconstructed, fs=360, tolerance_ms=50):
    """
    Compute QRS detection accuracy metrics
    
    Args:
        original: Original ECG
        reconstructed: Reconstructed ECG
        fs: Sampling frequency
        tolerance_ms: Matching tolerance in milliseconds
    
    Returns:
        metrics: Dictionary with sensitivity, PPV, F1 score
    """
    r_peaks_orig = detect_qrs_complexes(original, fs)
    r_peaks_recon = detect_qrs_complexes(reconstructed, fs)
    
    if len(r_peaks_orig) == 0 or len(r_peaks_recon) == 0:
        return {
            'sensitivity': 0.0,
            'ppv': 0.0,
            'f1_score': 0.0,
            'num_orig': len(r_peaks_orig),
            'num_recon': len(r_peaks_recon)
        }
    
    # Match peaks within tolerance
    tolerance_samples = int(tolerance_ms * fs / 1000)
    
    tp = 0
    for peak_orig in r_peaks_orig:
        if np.any(np.abs(r_peaks_recon - peak_orig) <= tolerance_samples):
            tp += 1
    
    fn = len(r_peaks_orig) - tp
    fp = len(r_peaks_recon) - tp
    
    # Compute metrics
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    ppv = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    f1_score = 2 * (sensitivity * ppv) / (sensitivity + ppv) if (sensitivity + ppv) > 0 else 0.0
    
    return {
        'sensitivity': sensitivity * 100,  # as percentage
        'ppv': ppv * 100,
        'f1_score': f1_score,
        'num_orig': len(r_peaks_orig),
        'num_recon': len(r_peaks_recon),
        'tp': tp,
        'fp': fp,
        'fn': fn
    }


def compute_hrv_metrics(ecg_signal, fs=360):
    """
    Compute Heart Rate Variability metrics
    
    Returns:
        hrv_metrics: Dictionary with SDNN, RMSSD, mean_hr, std_hr
    """
    r_peaks = detect_qrs_complexes(ecg_signal, fs)
    
    if len(r_peaks) < 2:
        return {
            'sdnn_ms': np.nan,
            'rmssd_ms': np.nan,
            'mean_hr_bpm': np.nan,
            'std_hr_bpm': np.nan
        }
    
    # RR intervals in milliseconds
    rr_intervals = np.diff(r_peaks) / fs * 1000
    
    # SDNN: Standard deviation of NN intervals
    sdnn = np.std(rr_intervals)
    
    # RMSSD: Root mean square of successive differences
    rmssd = np.sqrt(np.mean(np.diff(rr_intervals) ** 2))
    
    # Heart rate statistics
    hr_bpm = 60000 / rr_intervals
    mean_hr = np.mean(hr_bpm)
    std_hr = np.std(hr_bpm)
    
    return {
        'sdnn_ms': sdnn,
        'rmssd_ms': rmssd,
        'mean_hr_bpm': mean_hr,
        'std_hr_bpm': std_hr
    }


def compute_hrv_preservation(original, reconstructed, fs=360):
    """
    Compute HRV preservation error
    
    Returns:
        errors: Dictionary with percentage errors for each HRV metric
    """
    hrv_orig = compute_hrv_metrics(original, fs)
    hrv_recon = compute_hrv_metrics(reconstructed, fs)
    
    errors = {}
    for key in ['sdnn_ms', 'rmssd_ms', 'mean_hr_bpm']:
        if not np.isnan(hrv_orig[key]) and hrv_orig[key] != 0:
            error_pct = 100 * abs(hrv_orig[key] - hrv_recon[key]) / hrv_orig[key]
            errors[f'{key}_error_pct'] = error_pct
        else:
            errors[f'{key}_error_pct'] = np.nan
    
    return errors


# =====================================
# Comprehensive Metrics Function
# =====================================

def compute_all_metrics(original, reconstructed, fs=360):
    """
    Compute all metrics (3 levels)
    
    Args:
        original: Original ECG signal
        reconstructed: Reconstructed ECG signal
        fs: Sampling frequency
    
    Returns:
        metrics: Dictionary with all metrics
    """
    # Ensure same length
    min_len = min(len(original), len(reconstructed))
    original = original[:min_len]
    reconstructed = reconstructed[:min_len]
    
    metrics = {}
    
    # === Level 1: Signal Fidelity ===
    metrics['MSE'] = compute_mse(original, reconstructed)
    metrics['RMSE'] = compute_rmse(original, reconstructed)
    metrics['PRD (%)'] = compute_prd(original, reconstructed)
    metrics['SNR (dB)'] = compute_snr(original, reconstructed)
    metrics['CC'] = compute_correlation_coefficient(original, reconstructed)
    metrics['NMSE'] = compute_nmse(original, reconstructed)
    
    # === Level 2: Morphology ===
    metrics['DTW'] = compute_dtw_distance(original, reconstructed)
    metrics['Wavelet_Energy (%)'] = compute_wavelet_energy_preservation(original, reconstructed)
    metrics['Freq_Correlation'] = compute_frequency_domain_correlation(original, reconstructed)
    metrics['PQRST_Deviation'] = compute_pqrst_deviation(original, reconstructed, fs)
    
    # === Level 3: Clinical Diagnostic ===
    qrs_metrics = compute_qrs_detection_metrics(original, reconstructed, fs)
    metrics.update({f'QRS_{k}': v for k, v in qrs_metrics.items()})
    
    hrv_errors = compute_hrv_preservation(original, reconstructed, fs)
    metrics.update(hrv_errors)
    
    return metrics


def format_metrics_table(metrics_dict):
    """
    Format metrics dictionary as readable table
    
    Args:
        metrics_dict: Dictionary of metrics
    
    Returns:
        formatted_str: Formatted string representation
    """
    lines = []
    lines.append("=" * 60)
    lines.append("PERFORMANCE METRICS")
    lines.append("=" * 60)
    
    # Level 1
    lines.append("\n=== SIGNAL FIDELITY ===")
    for key in ['MSE', 'RMSE', 'PRD (%)', 'SNR (dB)', 'CC', 'NMSE']:
        if key in metrics_dict:
            lines.append(f"{key:20s}: {metrics_dict[key]:.6f}")
    
    # Level 2
    lines.append("\n=== MORPHOLOGY PRESERVATION ===")
    for key in ['DTW', 'Wavelet_Energy (%)', 'Freq_Correlation', 'PQRST_Deviation']:
        if key in metrics_dict:
            val = metrics_dict[key]
            if not np.isnan(val):
                lines.append(f"{key:20s}: {val:.6f}")
    
    # Level 3
    lines.append("\n=== CLINICAL DIAGNOSTIC ===")
    for key in metrics_dict:
        if key.startswith('QRS_') or key.endswith('_error_pct'):
            val = metrics_dict[key]
            if not np.isnan(val):
                lines.append(f"{key:20s}: {val:.4f}")
    
    lines.append("=" * 60)
    
    return "\n".join(lines)


def analyze_r_peaks(original, reconstructed, fs=360):
    """
    Analyze R-peak preservation between original and reconstructed ECG
    
    Args:
        original: Original ECG signal
        reconstructed: Reconstructed ECG signal
        fs: Sampling frequency (default 360 Hz)
    
    Returns:
        dict with R-peak analysis metrics
    """
    try:
        import wfdb.processing
        
        # Detect R-peaks
        r_peaks_orig = wfdb.processing.xqrs_detect(original, fs, verbose=False)
        r_peaks_recon = wfdb.processing.xqrs_detect(reconstructed, fs, verbose=False)
        
        # Detection accuracy (match within 50ms window)
        window = int(0.05 * fs)  # 50ms tolerance
        matched = 0
        for peak_orig in r_peaks_orig:
            if any(abs(peak_recon - peak_orig) <= window for peak_recon in r_peaks_recon):
                matched += 1
        
        detection_accuracy = (matched / len(r_peaks_orig)) * 100 if len(r_peaks_orig) > 0 else 0
        
        # R-R interval analysis
        rr_orig = np.diff(r_peaks_orig) / fs * 1000  # Convert to ms
        rr_recon = np.diff(r_peaks_recon) / fs * 1000
        
        if len(rr_orig) > 0 and len(rr_recon) > 0:
            # Align lengths
            min_len = min(len(rr_orig), len(rr_recon))
            rr_mae = np.mean(np.abs(rr_orig[:min_len] - rr_recon[:min_len]))
            
            # Heart rate
            hr_orig = 60000 / np.mean(rr_orig)  # BPM
            hr_recon = 60000 / np.mean(rr_recon)
            hr_error = abs(hr_orig - hr_recon)
        else:
            rr_mae = np.nan
            hr_error = np.nan
        
        return {
            'r_peaks_original': r_peaks_orig,
            'r_peaks_reconstructed': r_peaks_recon,
            'R-Peak Detection (%)': detection_accuracy,
            'R-R Interval MAE (ms)': rr_mae,
            'Heart Rate Error (BPM)': hr_error,
            'Beats Original': len(r_peaks_orig),
            'Beats Reconstructed': len(r_peaks_recon)
        }
        
    except Exception as e:
        return {
            'R-Peak Detection (%)': np.nan,
            'R-R Interval MAE (ms)': np.nan,
            'Heart Rate Error (BPM)': np.nan,
            'error': str(e)
        }
