"""
IMU Data Analysis and Markov Model Learning - ENHANCED PHYSICS-BASED VERSION
Extract motion patterns from IMU data and learn channel parameters

🎓 PHYSICS-BASED APPROACH:
This version implements proper signal processing algorithms based on biomechanics:
1. EMA (α=0.9) for gravity removal from accelerometer
2. Rolling Std (0.5s window) for gyroscope stability analysis
3. Dual-threshold state mapping (separate acc + gyro rules)

Author: Grace
Date: December 2025
"""

import numpy as np
import pandas as pd
from scipy import signal


# ============================================
# ALGORITHM 1: EMA GRAVITY REMOVAL (α=0.9)
# ============================================

def remove_gravity_with_ema(acc_data, alpha=0.9):
    """
    Remove gravity from accelerometer using Exponential Moving Average (EMA)
    
    🎓 WHY EMA WITH α=0.9?
    
    Physical Reality:
    - Accelerometer measures: Total = Gravity + Dynamic Motion
    - Gravity changes SLOWLY (only when body tilts)
    - Movement changes FAST (steps, arm swings)
    
    EMA acts as LOW-PASS FILTER:
    - Tracks slow changes (gravity) ✅
    - Ignores fast spikes (motion) ✅
    
    Why α=0.9 specifically?
    - α too high (0.99): Can't adapt to body orientation changes
    - α too low (0.5): Confuses movement with gravity
    - α=0.9: Perfect balance! Tracks orientation, filters motion
    
    Args:
        acc_data: DataFrame with ax, ay, az columns
        alpha: EMA smoothing factor (0.9 recommended)
    
    Returns:
        gravity_estimate: Estimated gravity components (ax, ay, az)
        dynamic_motion: Pure motion = total - gravity
    """
    # 🚨 CRITICAL FIX: Normalize raw sensor units to m/s²
    # UCI HAR dataset uses raw accelerometer values (±16g range, 16384 LSB/g)
    acc_values = acc_data[['ax', 'ay', 'az']].values
    max_val = np.max(np.abs(acc_values))
    
    if max_val > 100:
        # Convert from raw sensor units to m/s²
        # Formula: (raw / 16384) × 9.81
        acc_values = (acc_values / 16384.0) * 9.81
        acc_data = pd.DataFrame(acc_values, columns=['ax', 'ay', 'az'])
        print(f"   ℹ️ Normalized sensor units: max {max_val:.0f} → {np.max(np.abs(acc_values)):.2f} m/s²")
    
    n_samples = len(acc_data)
    
    # Initialize gravity estimate with first sample (assuming stationary start)
    gravity_x = np.zeros(n_samples)
    gravity_y = np.zeros(n_samples)
    gravity_z = np.zeros(n_samples)
    
    gravity_x[0] = acc_data['ax'].iloc[0]
    gravity_y[0] = acc_data['ay'].iloc[0]
    gravity_z[0] = acc_data['az'].iloc[0]
    
    # Apply EMA: gravity[t] = α × gravity[t-1] + (1-α) × acc[t]
    for i in range(1, n_samples):
        gravity_x[i] = alpha * gravity_x[i-1] + (1 - alpha) * acc_data['ax'].iloc[i]
        gravity_y[i] = alpha * gravity_y[i-1] + (1 - alpha) * acc_data['ay'].iloc[i]
        gravity_z[i] = alpha * gravity_z[i-1] + (1 - alpha) * acc_data['az'].iloc[i]
    
    # Dynamic motion = Total - Gravity
    dynamic_ax = acc_data['ax'].values - gravity_x
    dynamic_ay = acc_data['ay'].values - gravity_y
    dynamic_az = acc_data['az'].values - gravity_z
    
    # Compute magnitude of dynamic motion
    dynamic_magnitude = np.sqrt(dynamic_ax**2 + dynamic_ay**2 + dynamic_az**2)
    
    gravity_estimate = pd.DataFrame({
        'grav_x': gravity_x,
        'grav_y': gravity_y,
        'grav_z': gravity_z
    })
    
    dynamic_motion = pd.DataFrame({
        'dyn_ax': dynamic_ax,
        'dyn_ay': dynamic_ay,
        'dyn_az': dynamic_az,
        'dyn_magnitude': dynamic_magnitude
    })
    
    return gravity_estimate, dynamic_motion


# ============================================
# ALGORITHM 2: ROLLING STD FOR GYRO STABILITY
# ============================================

def compute_gyroscope_stability(gyro_data, window_sec=0.5, fs=50):
    """
    Compute gyroscope stability using rolling standard deviation
    
    🎓 WHY ROLLING STANDARD DEVIATION?
    
    Physical Meaning:
    - Standard Deviation = measure of "spread" or "variability"
    - Low Std → Stable orientation → LED points at detector → LoS ✅
    - High Std → Erratic turning → LED blocked by body → NLoS ✅
    
    Why 0.5 Second Window?
    - Too short (0.1s): Captures noise, not real motion
    - Too long (2.0s): Misses quick movements
    - Just right (0.5s): Matches human motion timescale!
      (Most body movements happen at ~1-2 Hz, so 0.5s captures one cycle)
    
    Args:
        gyro_data: DataFrame with gx, gy, gz columns
        window_sec: Rolling window duration (0.5s recommended)
        fs: Sampling frequency (Hz)
    
    Returns:
        gyro_magnitude: Magnitude of rotation
        gyro_stability: Rolling std (LOW = stable, HIGH = erratic)
    """
    # Compute gyroscope magnitude
    gyro_magnitude = np.sqrt(
        gyro_data['gx']**2 + 
        gyro_data['gy']**2 + 
        gyro_data['gz']**2
    )
    
    # Compute rolling standard deviation
    window_samples = int(window_sec * fs)
    
    # Use pandas rolling std
    gyro_stability = pd.Series(gyro_magnitude).rolling(
        window=window_samples, 
        min_periods=1,
        center=False
    ).std()
    
    # Fill NaN with forward fill
    gyro_stability = gyro_stability.ffill().fillna(0)
    
    return gyro_magnitude.values, gyro_stability.values


# ============================================
# ALGORITHM 3: DUAL-THRESHOLD STATE MAPPING
# ============================================

def map_motion_to_states_dual_threshold(dynamic_acc_mag, gyro_stability, activity_name):
    """
    Map motion to channel states using DUAL thresholds (acc + gyro)
    
    🎓 WHY PERCENTILES (ADAPTIVE THRESHOLDS)?
    
    Problem: Different activities have VERY different motion ranges
    - Standing: Acc range 12,000 - 15,000
    - Walking: Acc range 10,000 - 20,000
    - Running: Acc range 8,000 - 30,000
    
    Fixed thresholds DON'T work!
    
    Percentiles adapt to the DATA:
    - P20 = "20% of samples are below this"
    - P80 = "80% of samples are below this"
    
    Result: Works for ANY activity! ✅
    
    Why 20/80 specifically?
    - Divides data into: 20% LoS | 60% Partial | 20% Blocked
    - Reflects real motion patterns (Partial is most common)
    
    Logic:
    - Low acc AND low gyro std → LoS (stable, minimal motion)
    - High acc OR high gyro std → Blocked (dynamic motion or erratic rotation)
    - Otherwise → Partial
    
    Args:
        dynamic_acc_mag: Dynamic acceleration magnitude (gravity removed)
        gyro_stability: Gyroscope rolling std (variability measure)
        activity_name: 'walking', 'sitting', or 'standing'
    
    Returns:
        states: Array of channel states (0=LoS, 1=Partial, 2=Blocked)
    """
    # Activity-specific percentile thresholds
    # These reflect biomechanical differences between activities
    if activity_name == 'standing':
        # Standing: very stable, most time in LoS
        acc_percentiles = [50, 80]  # More tolerant to small movements
        gyro_percentiles = [60, 85]
    elif activity_name == 'sitting':
        # Sitting: moderately stable
        acc_percentiles = [40, 70]
        gyro_percentiles = [50, 75]
    elif activity_name == 'walking':
        # Walking: dynamic, more time in Partial/Blocked
        acc_percentiles = [25, 60]
        gyro_percentiles = [35, 65]
    else:
        # Default
        acc_percentiles = [33, 67]
        gyro_percentiles = [33, 67]
    
    # Compute thresholds
    acc_low = np.percentile(dynamic_acc_mag, acc_percentiles[0])
    acc_high = np.percentile(dynamic_acc_mag, acc_percentiles[1])
    
    gyro_low = np.percentile(gyro_stability, gyro_percentiles[0])
    gyro_high = np.percentile(gyro_stability, gyro_percentiles[1])
    
    # Initialize states
    n_samples = len(dynamic_acc_mag)
    states = np.ones(n_samples, dtype=int)  # Default: Partial (1)
    
    # Apply dual-threshold logic
    for i in range(n_samples):
        acc_val = dynamic_acc_mag[i]
        gyro_val = gyro_stability[i]
        
        # LoS: Both signals indicate stability
        if (acc_val < acc_low) and (gyro_val < gyro_low):
            states[i] = 0  # LoS
        
        # Blocked: Either signal indicates high motion/instability
        elif (acc_val > acc_high) or (gyro_val > gyro_high):
            states[i] = 2  # Blocked
        
        # Otherwise: Partial (already initialized to 1)
    
    return states


# ============================================
# MARKOV TRANSITION MATRIX LEARNING
# ============================================

def learn_markov_transition_matrix(states):
    """
    Learn first-order Markov transition matrix from state sequence
    
    🎓 HOW MARKOV GENERATES STATES FOR ECG:
    
    The Markov Process is "Smart Random":
    - NOT pure random: Probabilities come from YOUR real data
    - IS random: Which specific transition happens uses randomness
    
    Example:
    - Currently in LoS
    - Learned probabilities: LoS→LoS: 61%, LoS→Partial: 28%, LoS→Blocked: 11%
    - Roll dice with these probabilities
    - Result: Realistic motion sequence!
    
    Temporal Coherence:
    - Pure random: [LoS, Blocked, LoS, Blocked] ← Jumps wildly!
    - Markov chain: [LoS, LoS, Partial, Blocked] ← Smooth transitions!
    
    Args:
        states: Array of channel states (0, 1, 2)
    
    Returns:
        P: 3x3 transition probability matrix
        state_stats: Dictionary with state statistics
    """
    # Count transitions
    transition_counts = np.zeros((3, 3))
    
    for i in range(len(states) - 1):
        current_state = states[i]
        next_state = states[i + 1]
        transition_counts[current_state, next_state] += 1
    
    # Normalize to get probabilities
    P = np.zeros((3, 3))
    for i in range(3):
        row_sum = transition_counts[i].sum()
        if row_sum > 0:
            P[i] = transition_counts[i] / row_sum
        else:
            # If state never occurs, uniform distribution
            P[i] = np.array([1/3, 1/3, 1/3])
    
    # Compute state statistics
    unique, counts = np.unique(states, return_counts=True)
    state_distribution = {int(s): float(c / len(states)) for s, c in zip(unique, counts)}
    
    num_transitions = np.sum(np.diff(states) != 0)
    avg_duration = len(states) / (num_transitions + 1)
    
    state_stats = {
        'state_distribution': state_distribution,
        'num_transitions': int(num_transitions),
        'avg_state_duration': float(avg_duration),
        'total_samples': len(states)
    }
    
    return P, state_stats


# ============================================
# PARAMETER LEARNING FUNCTIONS
# ============================================

def learn_attenuation_parameters(dynamic_acc_mag, states):
    """
    Learn attenuation ranges for each state from motion data
    
    🎯 FIXED: Now uses ACTUAL motion statistics instead of hardcoded values
    
    Physical Reasoning:
    - Higher motion → More signal degradation → Higher attenuation
    - State 0 (LoS): Minimal attenuation, narrow range
    - State 1 (Partial): Moderate attenuation, medium range
    - State 2 (Blocked): High attenuation, wider range
    - Ranges are learned from percentiles of motion in each state
    
    Args:
        dynamic_acc_mag: Dynamic acceleration magnitude (after gravity removal)
        states: Array of channel states (0=LoS, 1=Partial, 2=Blocked)
    
    Returns:
        attenuation_dict: Dictionary with (min_dB, max_dB) for each state
    """
    # 🚨 SAFETY CHECK: Ensure proper units (should be 0-10 m/s² range)
    if np.max(np.abs(dynamic_acc_mag)) > 50:
        print(f"   ⚠️ WARNING: dynamic_acc_mag has large values (max={np.max(np.abs(dynamic_acc_mag)):.1f})")
        print(f"   ⚠️ Converting to m/s²...")
        dynamic_acc_mag = (dynamic_acc_mag / 16384.0) * 9.81
    
    attenuation_dict = {}
    
    for state in [0, 1, 2]:
        mask = states == state
        
        if mask.sum() > 10:  # Ensure sufficient samples for statistics
            # Get motion data for this state
            motion_in_state = dynamic_acc_mag[mask]
            
            # ✅ KEY FIX: Use percentiles of ACTUAL motion data
            motion_10th = np.percentile(motion_in_state, 10)
            motion_90th = np.percentile(motion_in_state, 90)
            motion_mean = np.mean(motion_in_state)
            
            # Empirical motion-to-attenuation mapping
            # Based on VLC physics: attenuation increases with motion intensity
            if state == 0:  # LoS - Minimal attenuation
                base_att = -0.25
                motion_factor = 0.15
            elif state == 1:  # Partial - Moderate attenuation
                base_att = -2.0
                motion_factor = 0.20
            else:  # Blocked (state == 2) - High attenuation
                base_att = -2.8
                motion_factor = 0.40
            
            # Calculate range based on ACTUAL motion statistics
            min_att = base_att - (motion_factor * (motion_90th - motion_mean))
            max_att = base_att + (motion_factor * (motion_mean - motion_10th))
            
            # Ensure lower bound ≤ upper bound (Reviewer 4 C#1 / Table 14 fix)
            lo = float(min(min_att, max_att))
            hi = float(max(min_att, max_att))
            attenuation_dict[state] = (lo, hi)
        else:
            # Fallback for insufficient data — ordered intervals guaranteed
            fallback = {0: (-0.5, -0.1), 1: (-2.5, -1.5), 2: (-3.5, -2.0)}
            attenuation_dict[state] = fallback[state]

    return attenuation_dict

def learn_jitter_parameter(gyro_stability):
    """
    Learn log-normal jitter parameter from motion variability
    
    Physical Reasoning:
    - Higher gyro variability → Larger jitter
    - Jitter represents motion-induced fading
    
    Args:
        gyro_stability: Gyroscope stability array (rolling std)
    
    Returns:
        sigma: Log-normal jitter parameter
    """
    # Compute variability using sliding window
    window_size = 50
    variability = []
    
    for i in range(0, len(gyro_stability) - window_size, window_size // 2):
        window = gyro_stability[i:i+window_size]
        variability.append(np.std(window))
    
    # Map variability to jitter parameter
    avg_variability = np.mean(variability)
    
    # Empirical mapping (tune based on your data)
    sigma = 0.05 + 0.15 * (avg_variability / (np.max(variability) + 1e-12))
    sigma = np.clip(sigma, 0.02, 0.20)
    
    return float(sigma)


def learn_diffuse_parameter(dynamic_acc_mag):
    """
    Learn diffuse path parameter from motion characteristics
    
    Physical Reasoning:
    - More motion → More diffuse scattering
    - Diffuse paths result from reflections off body/clothes
    
    Args:
        dynamic_acc_mag: Dynamic acceleration magnitude
    
    Returns:
        beta: Diffuse path weight parameter
    """
    # Higher overall motion → more diffuse scattering
    normalized_motion = dynamic_acc_mag / (np.max(dynamic_acc_mag) + 1e-12)
    avg_motion = np.mean(normalized_motion)
    
    # Empirical mapping
    beta = 0.02 + 0.15 * avg_motion
    beta = np.clip(beta, 0.01, 0.20)
    
    return float(beta)


# ============================================
# MAIN ANALYSIS FUNCTION
# ============================================

def analyze_single_activity(imu_data, activity_name):
    """
    Complete physics-based analysis for a single activity
    
    Pipeline:
    1. Remove gravity with EMA (α=0.9)
    2. Compute gyroscope stability (rolling std, 0.5s)
    3. Map to states using dual thresholds (percentiles)
    4. Learn Markov transition matrix
    5. Learn channel parameters (attenuation, jitter, diffuse)
    
    Args:
        imu_data: DataFrame with IMU measurements for one activity
        activity_name: Name of activity ('walking', 'sitting', 'standing')
    
    Returns:
        params: Dictionary with all learned parameters
    """
    print(f"\n{'='*70}")
    print(f"ANALYZING: {activity_name.upper()}")
    print(f"{'='*70}")
    
    # Extract accelerometer and gyroscope data
    acc_data = imu_data[['ax', 'ay', 'az']]
    gyro_data = imu_data[['gx', 'gy', 'gz']]
    
    # STEP 1: Remove gravity with EMA (α=0.9)
    print("\n[1/5] Removing gravity with EMA (α=0.9)...")
    gravity_estimate, dynamic_motion = remove_gravity_with_ema(acc_data, alpha=0.9)
    print(f"   ✓ Gravity removed. Dynamic motion range: {dynamic_motion['dyn_magnitude'].min():.1f} to {dynamic_motion['dyn_magnitude'].max():.1f}")
    
    # STEP 2: Compute gyroscope stability (rolling std)
    print("\n[2/5] Computing gyroscope stability (rolling std, 0.5s window)...")
    gyro_magnitude, gyro_stability = compute_gyroscope_stability(gyro_data, window_sec=0.5, fs=50)
    print(f"   ✓ Gyro stability computed. Range: {gyro_stability.min():.1f} to {gyro_stability.max():.1f}")
    
    # STEP 3: Map to channel states using dual thresholds
    print(f"\n[3/5] Mapping to channel states (activity-specific percentiles)...")
    states = map_motion_to_states_dual_threshold(
        dynamic_motion['dyn_magnitude'].values,
        gyro_stability,
        activity_name
    )
    
    # Count state distribution
    unique, counts = np.unique(states, return_counts=True)
    state_dist = {int(s): int(c) for s, c in zip(unique, counts)}
    print(f"   ✓ States mapped (revised terminology per reviewer comments):")
    print(f"      LoS-dominant (0):         {state_dist.get(0, 0)} samples ({state_dist.get(0, 0)/len(states)*100:.1f}%)")
    print(f"      Partially-obstructed (1): {state_dist.get(1, 0)} samples ({state_dist.get(1, 0)/len(states)*100:.1f}%)")
    print(f"      Diffuse-dominant (2):     {state_dist.get(2, 0)} samples ({state_dist.get(2, 0)/len(states)*100:.1f}%)")
    
    # STEP 4: Learn Markov transition matrix
    print("\n[4/5] Learning Markov transition matrix...")
    P, state_stats = learn_markov_transition_matrix(states)
    print(f"   ✓ Markov matrix learned. Num transitions: {state_stats['num_transitions']}")
    
    # STEP 5: Learn channel parameters
    print("\n[5/5] Learning channel parameters...")
    attenuation_dict = learn_attenuation_parameters(dynamic_motion['dyn_magnitude'].values, states)
    sigma = learn_jitter_parameter(gyro_stability)
    beta = learn_diffuse_parameter(dynamic_motion['dyn_magnitude'].values)
    print(f"   ✓ Parameters learned:")
    print(f"      Jitter (σ):  {sigma:.3f}")
    print(f"      Diffuse (β): {beta:.3f}")
    
    # Package results
    params = {
        'activity': activity_name,
        'markov_matrix': P,
        'state_statistics': state_stats,
        'attenuation_db': attenuation_dict,
        'sigma_jitter': sigma,
        'beta_diffuse': beta,
        'motion_stats': {
            'acc_dynamic_mean': float(np.mean(dynamic_motion['dyn_magnitude'])),
            'acc_dynamic_std': float(np.std(dynamic_motion['dyn_magnitude'])),
            'gyro_mag_mean': float(np.mean(gyro_magnitude)),
            'gyro_mag_std': float(np.std(gyro_magnitude)),
            'gyro_stability_mean': float(np.mean(gyro_stability)),
            'gyro_stability_std': float(np.std(gyro_stability))
        },
        'processing_info': {
            'gravity_removal': 'EMA with α=0.9',
            'gyro_analysis': 'Rolling std with 0.5s window',
            'state_mapping': 'Dual-threshold (acc + gyro) with activity-specific percentiles'
        }
    }
    
    print(f"\n{'='*70}")
    print(f"✓ ANALYSIS COMPLETE FOR {activity_name.upper()}")
    print(f"{'='*70}\n")
    
    return params


# ============================================
# LEGACY COMPATIBILITY FUNCTIONS
# ============================================

def load_imu_dataset(filepath, activity_labels={'walking': 3, 'sitting': 4, 'standing': 5}):
    """
    Load IMU dataset and separate by activity
    
    Args:
        filepath: Path to imu_dataset.csv
        activity_labels: Dictionary mapping activity names to labels
    
    Returns:
        imu_dict: Dictionary with separate DataFrames for each activity
    """
    df = pd.read_csv(filepath)
    
    imu_dict = {}
    for activity_name, label in activity_labels.items():
        imu_dict[activity_name] = df[df['activity'] == label].reset_index(drop=True)
    
    return imu_dict


def analyze_all_activities(imu_filepath):
    """
    Complete IMU analysis for all activities
    
    Args:
        imu_filepath: Path to imu_dataset.csv
    
    Returns:
        all_params: Dictionary with parameters for each activity
    """
    # Load IMU data
    imu_dict = load_imu_dataset(imu_filepath)
    
    all_params = {}
    
    for activity_name, imu_data in imu_dict.items():
        print(f"\n{'='*70}")
        print(f"Analyzing {activity_name}...")
        print(f"{'='*70}")
        params = analyze_single_activity(imu_data, activity_name)
        all_params[activity_name] = params
    
    return all_params


def print_analysis_summary(all_params):
    """
    Print human-readable summary of learned parameters
    
    Args:
        all_params: Dictionary from analyze_all_activities()
    """
    print("\n" + "="*70)
    print("IMU DATA ANALYSIS SUMMARY (PHYSICS-BASED)")
    print("="*70)
    
    for activity, params in all_params.items():
        print(f"\n{'='*70}")
        print(f"ACTIVITY: {activity.upper()}")
        print(f"{'='*70}")
        
        # State statistics (revised labels: LoS-dominant / partially-obstructed / diffuse-dominant)
        stats = params['state_statistics']
        print(f"\nState Distribution:")
        dist = stats['state_distribution']
        print(f"  LoS-dominant (0):         {dist.get(0, 0)*100:5.1f}%")
        print(f"  Partially-obstructed (1): {dist.get(1, 0)*100:5.1f}%")
        print(f"  Diffuse-dominant (2):     {dist.get(2, 0)*100:5.1f}%")
        print(f"  Num transitions:   {stats['num_transitions']}")
        print(f"  Avg state duration: {stats['avg_state_duration']:.1f} samples")
        
        # Markov matrix
        print(f"\nMarkov Transition Matrix P({activity}):")
        P = params['markov_matrix']
        print("                            →  LoS-dom  Part-obs  Diff-dom")
        for i, row in enumerate(P):
            state_name = ['LoS-dominant    ', 'Partially-obstr.', 'Diffuse-dominant'][i]
            print(f"  {state_name}  {row}")
        
        # Channel parameters
        print(f"\nChannel Parameters:")
        print(f"  Jitter (σ):  {params['sigma_jitter']:.3f}")
        print(f"  Diffuse (β): {params['beta_diffuse']:.3f}")
        
        print(f"\nAttenuation Ranges (dB) [effective surrogate parameters]:")
        att = params['attenuation_db']
        for state in [0, 1, 2]:
            state_name = ['LoS-dominant    ', 'Partially-obstr.', 'Diffuse-dominant'][state]
            lo, hi = att[state]
            print(f"  {state_name}: [{min(lo,hi):6.3f}, {max(lo,hi):6.3f}]")  # always lo ≤ hi
        
        # Processing info
        print(f"\nProcessing Methods Used:")
        proc = params['processing_info']
        print(f"  Gravity Removal:  {proc['gravity_removal']}")
        print(f"  Gyro Analysis:    {proc['gyro_analysis']}")
        print(f"  State Mapping:    {proc['state_mapping']}")
    
    print("\n" + "="*70)


# ============================================
# COMPATIBILITY WRAPPERS
# ============================================

def compute_motion_magnitude(imu_data):
    """
    Legacy wrapper for backward compatibility
    Now uses physics-based processing internally
    """
    # Use the new physics-based approach
    acc_data = imu_data[['ax', 'ay', 'az']]
    gyro_data = imu_data[['gx', 'gy', 'gz']]
    
    # Remove gravity
    _, dynamic_motion = remove_gravity_with_ema(acc_data, alpha=0.9)
    
    # Compute gyro stability
    gyro_mag, gyro_stab = compute_gyroscope_stability(gyro_data, window_sec=0.5, fs=50)
    
    # Return for compatibility
    acc_mag = np.sqrt(acc_data['ax']**2 + acc_data['ay']**2 + acc_data['az']**2)
    
    return acc_mag.values, gyro_mag, dynamic_motion['dyn_magnitude'].values


def map_motion_to_channel_states(motion_intensity, percentile_thresholds=[33, 67], activity_name=None):
    """
    Legacy wrapper - redirects to new dual-threshold method
    This is kept for backward compatibility but uses improved logic internally
    """
    # For legacy compatibility, we need to split this into acc and gyro
    # Since we only have combined motion_intensity, we'll use it for both
    # This is not ideal but maintains API compatibility
    
    if activity_name is None:
        activity_name = 'walking'
    
    # Use motion_intensity as proxy for both signals
    states = map_motion_to_states_dual_threshold(
        motion_intensity, 
        motion_intensity * 0.5,  # Scale down for gyro proxy
        activity_name
    )
    
    return states
