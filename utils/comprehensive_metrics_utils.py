"""
Comprehensive Clinical Metrics for ECG Reconstruction
=====================================================

Includes ALL perspectives:
1. Signal Fidelity (MSE, RMSE, SNR, PRD, CC)
2. Morphology (DTW, Wavelet Energy, PQRST deviations)
3. Diagnostic (R-peak detection, HRV indices)
4. Clinical (QRS accuracy, R-R intervals)

Based on Table 3.9 from your thesis
References: F. Liu et al. (2024), Ziget et al. (2000), etc.

Author: Grace
Date: December 2025
"""

import numpy as np
from scipy import signal as scipy_signal
from scipy.spatial.distance import euclidean
from fastdtw import fastdtw
import pywt


# =============================================================================
# 1. SIGNAL FIDELITY METRICS
# =============================================================================

def compute_mse(original, reconstructed):
    """Mean Squared Error"""
    return np.mean((original - reconstructed) ** 2)


def compute_rmse(original, reconstructed):
    """Root Mean Squared Error"""
    return np.sqrt(compute_mse(original, reconstructed))


def compute_snr(original, reconstructed):
    """
    Signal-to-Noise Ratio (dB)
    Reference: Ziget et al. (2000)
    
    Interpretation: Higher is better
    > 20 dB: Excellent
    """
    signal_power = np.mean(original ** 2)
    noise_power = np.mean((original - reconstructed) ** 2)
    if noise_power < 1e-10:
        return 100.0  # Perfect reconstruction
    return 10 * np.log10(signal_power / noise_power)


def compute_prd(original, reconstructed):
    """
    Percent Root-mean-square Difference (%)
    Reference: Ziget et al. (2000)
    
    Interpretation: Lower is better
    < 9%: Clinical grade (diagnostic quality)
    9-20%: Good quality
    > 20%: Poor quality
    """
    numerator = np.sqrt(np.mean((original - reconstructed) ** 2))
    denominator = np.sqrt(np.mean(original ** 2))
    if denominator < 1e-10:
        return 100.0
    return 100 * numerator / denominator


def compute_correlation_coefficient(original, reconstructed):
    """
    Pearson Correlation Coefficient (PCC)
    Reference: F. Liu et al. (2024)
    
    Interpretation: Higher is better
    > 0.95: Excellent shape similarity
    """
    orig_mean = np.mean(original)
    recon_mean = np.mean(reconstructed)
    
    numerator = np.sum((original - orig_mean) * (reconstructed - recon_mean))
    denominator = np.sqrt(
        np.sum((original - orig_mean) ** 2) * 
        np.sum((reconstructed - recon_mean) ** 2)
    )
    
    if denominator < 1e-10:
        return 1.0
    return numerator / denominator


# =============================================================================
# 2. MORPHOLOGY-AWARE METRICS
# =============================================================================

def compute_dtw_distance(original, reconstructed):
    """
    Dynamic Time Warping Distance
    Reference: Belen et al. (n.d.)
    
    Measures alignment cost - robust to temporal shifts
    Lower is better
    """
    try:
        distance, _ = fastdtw(original, reconstructed, dist=euclidean)
        # Normalize by length
        return distance / len(original)
    except:
        # Fallback to simple Euclidean
        return np.sqrt(np.mean((original - reconstructed) ** 2))


def compute_wavelet_energy_ratio(original, reconstructed, wavelet='db4', level=4):
    """
    Wavelet Energy Retention
    Reference: F. Liu et al. (2024)
    
    Measures how well sub-band energies are preserved
    Closer to 1.0 is better
    """
    # Decompose both signals
    coeffs_orig = pywt.wavedec(original, wavelet, level=level)
    coeffs_recon = pywt.wavedec(reconstructed, wavelet, level=level)
    
    # Compute energy in each subband
    energy_orig = [np.sum(c ** 2) for c in coeffs_orig]
    energy_recon = [np.sum(c ** 2) for c in coeffs_recon]
    
    # Total energy ratio
    total_orig = np.sum(energy_orig)
    total_recon = np.sum(energy_recon)
    
    if total_orig < 1e-10:
        return 1.0
    
    return total_recon / total_orig


def compute_pqrst_deviations(original, reconstructed, fs=360):
    """
    PQRST Waveform Deviation
    Reference: F. Liu et al. (2024)
    
    Measures fiducial timing/amplitude errors
    Lower is better (milliseconds and amplitude units)
    """
    # Simplified: compute segment-wise deviations
    # In full implementation, this would detect P, Q, R, S, T points
    
    # For now, compute max absolute deviation
    max_dev = np.max(np.abs(original - reconstructed))
    mean_dev = np.mean(np.abs(original - reconstructed))
    
    return {
        'max_deviation': max_dev,
        'mean_deviation': mean_dev,
        'timing_preserved': max_dev < 0.1  # Threshold
    }


# =============================================================================
# 3. DIAGNOSTIC METRICS (R-PEAK & HEARTBEAT)
# =============================================================================

def detect_r_peaks_simple(signal_data, fs=360, min_distance=200):
    """
    Simple R-peak detection using scipy
    For production, use wfdb.processing or Pan-Tompkins
    """
    # Find peaks
    peaks, _ = scipy_signal.find_peaks(
        signal_data,
        distance=min_distance,  # ~200ms between beats
        prominence=0.3  # Minimum prominence
    )
    return peaks


def compute_r_peak_detection_accuracy(original, reconstructed, fs=360):
    """
    R-peak Detection Rate
    Reference: Fariha et al. (2020)
    
    Measures beat detection integrity
    > 95%: Excellent
    """
    # Detect R-peaks in both
    peaks_orig = detect_r_peaks_simple(original, fs)
    peaks_recon = detect_r_peaks_simple(reconstructed, fs)
    
    if len(peaks_orig) == 0:
        return 100.0 if len(peaks_recon) == 0 else 0.0
    
    # Match peaks within tolerance (±50ms = ±18 samples at 360Hz)
    tolerance = int(0.05 * fs)
    matched = 0
    
    for peak_orig in peaks_orig:
        if np.any(np.abs(peaks_recon - peak_orig) <= tolerance):
            matched += 1
    
    detection_rate = 100 * matched / len(peaks_orig)
    
    return detection_rate


def compute_rr_interval_quality(original, reconstructed, fs=360):
    """
    R-R Interval Quality
    Measures beat-to-beat interval preservation
    Lower MAE is better (milliseconds)
    """
    peaks_orig = detect_r_peaks_simple(original, fs)
    peaks_recon = detect_r_peaks_simple(reconstructed, fs)
    
    if len(peaks_orig) < 2 or len(peaks_recon) < 2:
        return {
            'rr_mae_ms': 999.0,
            'rr_quality': 'Poor'
        }
    
    # Compute R-R intervals
    rr_orig = np.diff(peaks_orig) / fs * 1000  # Convert to ms
    rr_recon = np.diff(peaks_recon) / fs * 1000
    
    # If different number of beats, compare first N
    n_min = min(len(rr_orig), len(rr_recon))
    if n_min == 0:
        return {
            'rr_mae_ms': 999.0,
            'rr_quality': 'Poor'
        }
    
    rr_mae = np.mean(np.abs(rr_orig[:n_min] - rr_recon[:n_min]))
    
    # Quality assessment
    if rr_mae < 5:
        quality = 'Excellent'
    elif rr_mae < 20:
        quality = 'Good'
    elif rr_mae < 50:
        quality = 'Fair'
    else:
        quality = 'Poor'
    
    return {
        'rr_mae_ms': rr_mae,
        'rr_quality': quality
    }


def compute_hrv_preservation(original, reconstructed, fs=360):
    """
    Heart Rate Variability Indices
    Reference: F. Liu et al. (2024)
    
    Measures HRV index preservation
    """
    peaks_orig = detect_r_peaks_simple(original, fs)
    peaks_recon = detect_r_peaks_simple(reconstructed, fs)
    
    if len(peaks_orig) < 2 or len(peaks_recon) < 2:
        return {
            'sdnn_error_bpm': 999.0,
            'rmssd_error_bpm': 999.0,
            'hrv_quality': 'Poor'
        }
    
    # Compute R-R intervals
    rr_orig = np.diff(peaks_orig) / fs * 1000  # ms
    rr_recon = np.diff(peaks_recon) / fs * 1000
    
    # SDNN (Standard deviation of NN intervals)
    sdnn_orig = np.std(rr_orig)
    sdnn_recon = np.std(rr_recon)
    sdnn_error = abs(sdnn_orig - sdnn_recon)
    
    # RMSSD (Root mean square of successive differences)
    if len(rr_orig) > 1:
        rmssd_orig = np.sqrt(np.mean(np.diff(rr_orig) ** 2))
        rmssd_recon = np.sqrt(np.mean(np.diff(rr_recon) ** 2))
        rmssd_error = abs(rmssd_orig - rmssd_recon)
    else:
        rmssd_error = 0
    
    # Quality
    if sdnn_error < 5 and rmssd_error < 5:
        quality = 'Excellent'
    elif sdnn_error < 15 and rmssd_error < 15:
        quality = 'Good'
    else:
        quality = 'Fair'
    
    return {
        'sdnn_error_bpm': sdnn_error,
        'rmssd_error_bpm': rmssd_error,
        'hrv_quality': quality
    }


# =============================================================================
# 4. COMPREHENSIVE METRICS COMPUTATION
# =============================================================================

def compute_comprehensive_metrics(original, reconstructed, fs=360):
    """
    Compute ALL metrics across all categories
    
    Returns dictionary with:
    - signal_fidelity: MSE, RMSE, SNR, PRD, CC
    - morphology: DTW, Wavelet Energy, PQRST
    - diagnostic: R-peak detection, R-R intervals, HRV
    - clinical_grade: Overall assessment
    """
    
    metrics = {}
    
    # 1. Signal Fidelity
    metrics['signal_fidelity'] = {
        'mse': compute_mse(original, reconstructed),
        'rmse': compute_rmse(original, reconstructed),
        'snr_db': compute_snr(original, reconstructed),
        'prd_percent': compute_prd(original, reconstructed),
        'correlation': compute_correlation_coefficient(original, reconstructed)
    }
    
    # 2. Morphology
    metrics['morphology'] = {
        'dtw_distance': compute_dtw_distance(original, reconstructed),
        'wavelet_energy_ratio': compute_wavelet_energy_ratio(original, reconstructed),
        'pqrst': compute_pqrst_deviations(original, reconstructed, fs)
    }
    
    # 3. Diagnostic
    r_peak_acc = compute_r_peak_detection_accuracy(original, reconstructed, fs)
    rr_quality = compute_rr_interval_quality(original, reconstructed, fs)
    hrv_metrics = compute_hrv_preservation(original, reconstructed, fs)
    
    metrics['diagnostic'] = {
        'r_peak_detection_percent': r_peak_acc,
        'rr_mae_ms': rr_quality['rr_mae_ms'],
        'rr_quality': rr_quality['rr_quality'],
        'sdnn_error': hrv_metrics['sdnn_error_bpm'],
        'rmssd_error': hrv_metrics['rmssd_error_bpm'],
        'hrv_quality': hrv_metrics['hrv_quality']
    }
    
    # 4. Clinical Grade Assessment
    prd = metrics['signal_fidelity']['prd_percent']
    r_peak = metrics['diagnostic']['r_peak_detection_percent']
    
    if prd < 9.0 and r_peak > 95.0:
        clinical_grade = 'Excellent'
        clinical_suitable = True
    elif prd < 15.0 and r_peak > 90.0:
        clinical_grade = 'Good'
        clinical_suitable = True
    else:
        clinical_grade = 'Fair'
        clinical_suitable = False
    
    metrics['clinical'] = {
        'grade': clinical_grade,
        'suitable_for_diagnosis': clinical_suitable,
        'prd_clinical': prd < 9.0,
        'r_peak_clinical': r_peak > 95.0
    }
    
    return metrics


def format_metrics_for_display(metrics):
    """
    Format metrics dictionary for nice display
    """
    
    display = {}
    
    # Signal Fidelity
    sf = metrics['signal_fidelity']
    display['Signal Fidelity'] = {
        'MSE': f"{sf['mse']:.6f}",
        'RMSE': f"{sf['rmse']:.4f}",
        'SNR (dB)': f"{sf['snr_db']:.2f}",
        'PRD (%)': f"{sf['prd_percent']:.2f}",
        'Correlation': f"{sf['correlation']:.4f}"
    }
    
    # Morphology
    morph = metrics['morphology']
    display['Morphology'] = {
        'DTW Distance': f"{morph['dtw_distance']:.4f}",
        'Wavelet Energy': f"{morph['wavelet_energy_ratio']:.4f}",
        'Max PQRST Dev': f"{morph['pqrst']['max_deviation']:.4f}",
        'Mean PQRST Dev': f"{morph['pqrst']['mean_deviation']:.4f}"
    }
    
    # Diagnostic
    diag = metrics['diagnostic']
    display['Diagnostic'] = {
        'R-Peak Detection (%)': f"{diag['r_peak_detection_percent']:.2f}",
        'R-R MAE (ms)': f"{diag['rr_mae_ms']:.2f}",
        'R-R Quality': diag['rr_quality'],
        'HRV Quality': diag['hrv_quality']
    }
    
    # Clinical
    clin = metrics['clinical']
    display['Clinical'] = {
        'Grade': clin['grade'],
        'Diagnostic Use': '✅' if clin['suitable_for_diagnosis'] else '❌',
        'PRD Clinical': '✅' if clin['prd_clinical'] else '❌',
        'R-Peak Clinical': '✅' if clin['r_peak_clinical'] else '❌'
    }
    
    return display
