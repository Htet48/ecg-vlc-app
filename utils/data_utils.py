"""
Data Loading and Preprocessing Utilities
"""

import numpy as np
import wfdb
from scipy import signal
from scipy.interpolate import interp1d
import pandas as pd
from pathlib import Path

# Sampling frequency
FS = 360

def load_mitbih_record(record_name='100', duration=100, sampfrom=0):
    """
    Load ECG record from MIT-BIH database
    
    Args:
        record_name: Record ID (e.g., '100', '101')
        duration: Duration in seconds
        sampfrom: Starting sample
    
    Returns:
        ecg_signal: ECG signal array
        fs: Sampling frequency
        record: Full record object
    """
    sampto = sampfrom + int(duration * FS)
    
    try:
        record = wfdb.rdrecord(record_name, pn_dir='mitdb', 
                              sampfrom=sampfrom, sampto=sampto)
        ecg_signal = record.p_signal[:, 0]  # MLII lead
        return ecg_signal, record.fs, record
    except Exception as e:
        print(f"Error loading record {record_name}: {e}")
        # Return synthetic data as fallback
        return generate_synthetic_ecg(int(duration * FS), FS), FS, None


def generate_synthetic_ecg(n_samples, fs=360, hr_bpm=70):
    """
    Generate synthetic ECG signal for testing
    
    Args:
        n_samples: Number of samples
        fs: Sampling frequency
        hr_bpm: Heart rate in BPM
    
    Returns:
        ecg_signal: Synthetic ECG array
    """
    t = np.arange(n_samples) / fs
    rr = 60 / hr_bpm  # RR interval in seconds
    beat_samples = int(rr * fs)
    
    # Create one PQRST complex
    x = np.linspace(0, 1, beat_samples, endpoint=False)
    
    # P wave, Q, R, S, T components (Gaussian-like)
    components = [
        (0.18, 0.03, 0.12),   # P wave
        (0.38, 0.012, -0.35), # Q wave
        (0.40, 0.008, 0.95),  # R wave (peak)
        (0.42, 0.012, -0.40), # S wave
        (0.65, 0.05, 0.25)    # T wave
    ]
    
    beat = sum(amp * np.exp(-0.5 * ((x - center) / width) ** 2) 
               for center, width, amp in components)
    
    # Tile to create full signal
    full_signal = np.tile(beat, int(np.ceil(n_samples / beat_samples)))[:n_samples]
    
    # Add baseline wander
    baseline = 0.02 * np.sin(2 * np.pi * 0.2 * t)
    full_signal += baseline
    
    # Normalize
    full_signal = (full_signal - full_signal.min()) / (full_signal.max() - full_signal.min())
    
    return full_signal


def detect_r_peaks(ecg_signal, fs=360):
    """
    Detect R-peaks using Pan-Tompkins algorithm
    
    Args:
        ecg_signal: ECG signal array
        fs: Sampling frequency
    
    Returns:
        r_peaks: Array of R-peak indices
    """
    try:
        from wfdb import processing
        r_peaks = processing.xqrs_detect(ecg_signal, fs=fs, verbose=False)
        return r_peaks
    except:
        # Fallback: simple peak detection
        from scipy.signal import find_peaks
        # Normalize first
        norm_signal = (ecg_signal - ecg_signal.min()) / (ecg_signal.max() - ecg_signal.min())
        peaks, _ = find_peaks(norm_signal, 
                             height=0.6 * np.max(norm_signal),
                             distance=int(0.6 * fs))
        return peaks


def preprocess_ecg(ecg_signal, fs=360, target_window=None):
    """
    Complete ECG preprocessing pipeline
    
    Args:
        ecg_signal: Raw ECG signal
        fs: Sampling frequency  
        target_window: Target window size in samples (None = auto based on RR interval)
    
    Steps:
    1. Baseline wander removal (high-pass filter)
    2. R-peak detection
    3. Segment extraction (1.2 × RR intervals or target_window)
    4. Normalization to [0, 1]
    
    Args:
        ecg_signal: Raw ECG signal
        fs: Sampling frequency
    
    Returns:
        segments: Array of preprocessed ECG segments
        r_peaks: R-peak locations
        window_size: Size of each segment in samples
    """
    # 1. Baseline wander removal
    b, a = signal.butter(3, 0.5 / (fs / 2), 'highpass')
    ecg_filtered = signal.filtfilt(b, a, ecg_signal)
    
    # 2. R-peak detection
    r_peaks = detect_r_peaks(ecg_filtered, fs)
    
    if len(r_peaks) < 3:
        return np.array([]), r_peaks, 0
    
    # 3. Compute window size
    if target_window is not None:
        # Use user-specified window size
        window_size = target_window
    else:
        # Auto: 1.2 × median RR
        rr_intervals = np.diff(r_peaks) / fs  # in seconds
        median_rr = np.median(rr_intervals)
        window_duration = 1.2 * median_rr
        window_size = int(window_duration * fs)
    
    # 4. Extract segments
    segments = []
    valid_peaks = []
    
    for peak in r_peaks[1:-1]:  # Skip first and last
        start_idx = max(0, peak - window_size // 2)
        end_idx = min(len(ecg_filtered), peak + window_size // 2)
        segment = ecg_filtered[start_idx:end_idx]
        
        if len(segment) == window_size:
            # Normalize to [0, 1]
            seg_min, seg_max = segment.min(), segment.max()
            if seg_max > seg_min:
                seg_norm = (segment - seg_min) / (seg_max - seg_min)
                segments.append(seg_norm)
                valid_peaks.append(peak)
    
    return np.array(segments), np.array(valid_peaks), window_size


def load_imu_data(filepath, activity_label=3):
    """
    Load IMU data for specified activity
    
    Args:
        filepath: Path to IMU CSV file
        activity_label: 3=walking, 4=sitting, 5=standing
    
    Returns:
        imu_df: Filtered DataFrame for activity
    """
    try:
        imu_df = pd.read_csv(filepath)
        filtered = imu_df[imu_df['activity'] == activity_label].reset_index(drop=True)
        return filtered
    except Exception as e:
        print(f"Error loading IMU data: {e}")
        # Return synthetic IMU data
        n_samples = 1000
        return pd.DataFrame({
            'ax': np.random.randn(n_samples) * 1000,
            'ay': np.random.randn(n_samples) * 1000 + 14000,
            'az': np.random.randn(n_samples) * 1000 - 7000,
            'gx': np.random.randn(n_samples) * 2000,
            'gy': np.random.randn(n_samples) * 2000,
            'gz': np.random.randn(n_samples) * 3000,
            'activity': [activity_label] * n_samples
        })


def get_activity_statistics(segments, r_peaks, fs=360):
    """
    Compute statistics for ECG segments
    
    Args:
        segments: Array of ECG segments
        r_peaks: R-peak locations
        fs: Sampling frequency
    
    Returns:
        stats: Dictionary of statistics
    """
    stats = {
        'num_segments': len(segments),
        'segment_length': len(segments[0]) if len(segments) > 0 else 0,
        'num_r_peaks': len(r_peaks),
    }
    
    if len(r_peaks) > 1:
        rr_intervals = np.diff(r_peaks) / fs * 1000  # in ms
        stats['mean_hr_bpm'] = 60000 / np.mean(rr_intervals)
        stats['std_hr_bpm'] = 60000 * np.std(rr_intervals) / (np.mean(rr_intervals) ** 2)
        stats['median_rr_ms'] = np.median(rr_intervals)
        stats['sdnn_ms'] = np.std(rr_intervals)
    
    return stats
