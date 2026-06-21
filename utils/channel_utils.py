"""
VLC Channel Modeling with IMU-Conditioned Markov Process
========================================================

This module implements the COMPLETE VLC channel model with ALL sub-stages:

Stage C (VLC Channel Effects):
  C1. State-Dependent Attenuation g(aₜ)
  C2. Log-Normal Jitter ξ(t)
  C3. Lambertian Channel H₀(t)
  C4. Direct Path Combination
  C5. Diffuse Path h_diff(t)
  C6. Signal Combination
  C7. LED Nonlinearity
  C8. Noise Addition

Author: Grace
Date: January 2026
"""

import numpy as np
from scipy import signal as sp_signal

# Default Markov transition matrices for each activity
MARKOV_MATRICES = {
    "walking": np.array([
        [0.75, 0.22, 0.03],
        [0.18, 0.62, 0.20],
        [0.06, 0.36, 0.58]
    ]),
    "sitting": np.array([
        [0.92, 0.07, 0.01],
        [0.10, 0.80, 0.10],
        [0.05, 0.35, 0.60]
    ]),
    "standing": np.array([
        [0.97, 0.03, 0.00],
        [0.25, 0.70, 0.05],
        [0.10, 0.40, 0.50]
    ])
}

# Attenuation ranges (dB) for each state and activity
ATTENUATION_DB = {
    "walking": {
        0: (-0.7, -0.2),  # LoS
        1: (-3.0, -1.4),  # Partial
        2: (-4.5, -3.2)   # NLoS
    },
    "sitting": {
        0: (-0.4, -0.1),
        1: (-2.2, -1.0),
        2: (-3.2, -2.4)
    },
    "standing": {
        0: (-0.2, 0.0),
        1: (-0.8, -0.3),
        2: (-1.4, -0.9)
    }
}

# Jitter parameters (log-normal standard deviation).
# Synced to the values LEARNED from the real IMU data by
# analyze_all_activities() — the same values used to generate the dataset
# (metadata.json). Keep these in sync if the IMU data or learning code changes.
SIGMA_JITTER = {
    "walking": 0.1478,
    "sitting": 0.0570,
    "standing": 0.0621,
}

# Diffuse path parameters (learned from real IMU motion; matches the dataset).
BETA_DIFFUSE = {
    "walking": 0.0461,
    "sitting": 0.0228,
    "standing": 0.0224,
}

# State-dependent direct/diffuse path weights (Reviewer 1 C#3, Reviewer 3 C#2).
# The channel equation becomes:
#   r(t) = LED[ η_dir[aₜ]·H0·g·ξ·s(t) + η_diff[aₜ]·β·h_eff*s(t) ] + n(t)
# Revised state terminology: 0=LoS-dominant, 1=partially-obstructed, 2=diffuse-dominant
ETA_DIR = {
    0: 0.95,   # LoS-dominant       — strong direct path
    1: 0.60,   # partially-obstructed — balanced
    2: 0.20,   # diffuse-dominant   — suppressed direct path
}
ETA_DIFF = {
    0: 0.05,   # LoS-dominant       — weak diffuse
    1: 0.40,   # partially-obstructed — balanced
    2: 0.80,   # diffuse-dominant   — dominant diffuse
}

# Revised state labels (LoS-dominant / partially-obstructed / diffuse-dominant)
STATE_NAMES = ['LoS-dominant', 'partially-obstructed', 'diffuse-dominant']


def generate_markov_states(P, n_samples, rng=None):
    """
    Generate state sequence using first-order Markov model
    
    Args:
        P: 3x3 transition probability matrix
        n_samples: Number of samples
        rng: Random number generator
    
    Returns:
        states: Array of states (0=LoS, 1=Partial, 2=NLoS)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Initial state distribution (equilibrium)
    p0 = np.ones(3) / 3
    
    states = np.empty(n_samples, dtype=int)
    states[0] = rng.choice(3, p=p0)
    
    for i in range(1, n_samples):
        states[i] = rng.choice(3, p=P[states[i-1]])
    
    return states


# =============================================================================
# C1: STATE-DEPENDENT ATTENUATION g(aₜ)
# =============================================================================

def compute_state_attenuation(states, activity, attenuation_db_dict, rng=None):
    """
    C1: Compute state-dependent attenuation g(aₜ)
    
    🎓 WHAT IS THIS?
    Channel attenuation depends on body motion state (LoS/Partial/NLoS).
    We learn these ranges from IMU motion intensity.
    
    📊 COMPUTATION PROCESS:
    1. For each sample i, check its Markov state aₜ ∈ {LoS, Partial, NLoS}
    2. Sample attenuation from state-specific range (uniform distribution)
    3. Convert dB to linear scale: g = 10^(AttdB/20)
    
    Args:
        states: State sequence array from Markov model
        activity: Activity type ('walking', 'sitting', 'standing')
        attenuation_db_dict: Learned attenuation ranges dictionary
        rng: Random number generator
    
    Returns:
        g_state: State-dependent attenuation (linear scale)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    n_samples = len(states)
    att_db = np.empty(n_samples)
    
    att_ranges = attenuation_db_dict[activity]
    
    # For each state, sample attenuation from learned range
    for state in (0, 1, 2):
        mask = states == state
        if mask.sum() > 0:
            lo, hi = att_ranges[state]
            att_db[mask] = rng.uniform(lo, hi, size=mask.sum())
    
    # Convert dB to linear scale
    g_state = 10.0 ** (att_db / 20.0)
    
    return g_state


# =============================================================================
# C2: LOG-NORMAL JITTER ξ(t)
# =============================================================================

def compute_lognormal_jitter(n_samples, sigma, rng=None):
    """
    C2: Compute log-normal jitter ξ(t)
    
    🎓 WHAT IS THIS?
    Random fluctuations due to body micro-movements (breathing, tremor).
    Log-normal distribution ensures ξ > 0 (cannot have negative amplitude).
    
    📊 COMPUTATION PROCESS:
    1. Sample from normal: ν ~ N(-σ²/2, σ²)
    2. Transform to log-normal: ξ = exp(ν)
    3. This ensures E[ξ] = 1 (zero mean in log domain)
    
    Args:
        n_samples: Number of samples
        sigma: Log-normal standard deviation (learned from gyro stability)
        rng: Random number generator
    
    Returns:
        xi: Log-normal jitter (E[ξ] = 1)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Log-normal with E[ξ] = 1
    nu = rng.normal(loc=-0.5 * sigma**2, scale=sigma, size=n_samples)
    xi = np.exp(nu)
    
    return xi


# =============================================================================
# C3: LAMBERTIAN CHANNEL H₀(t)
# =============================================================================

def compute_lambertian_channel(n_samples, fs, activity, rng=None):
    """
    C3: Compute time-varying Lambertian channel gain H₀(t)
    
    🎓 WHAT IS THIS?
    Optical channel gain based on LED radiation pattern (Lambertian model).
    Angles vary due to body motion (derived from IMU).
    
    📊 COMPUTATION PROCESS:
    1. Generate angle random walks from motion:
       - φ(t): transmitter angle (LED orientation)
       - ψ(t): receiver angle (photodetector orientation)
    2. Apply Lambertian formula:
       H₀ = (m+1)A/(2πd²) × cos^m(φ) × cos(ψ) × Ts × Gc
    3. Normalize to mean = 1
    
    Args:
        n_samples: Number of samples
        fs: Sampling frequency
        activity: Activity type (affects angle variability)
        rng: Random number generator
    
    Returns:
        H0: Lambertian channel gain (normalized to mean=1)
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    # Lambertian parameters
    m = 1.0  # Lambertian order
    A = 1.0  # Photodetector area (cm²)
    d = 0.25  # TX-RX distance (meters)
    Ts = 1.0  # Optical filter transmittance
    Gc = 1.0  # Concentrator gain
    
    # Angle random walk standard deviation (degrees) - from activity
    angle_std = {
        "walking": 0.12,
        "sitting": 0.06,
        "standing": 0.02
    }[activity]
    
    # Generate angle random walks
    phi = np.cumsum(rng.normal(0, np.deg2rad(angle_std), n_samples))
    psi = np.cumsum(rng.normal(0, np.deg2rad(angle_std), n_samples))
    
    # FOV constraint (60° field of view)
    FOV = np.deg2rad(60)
    psi = np.clip(psi, -FOV, FOV)
    
    # Lambertian formula
    H0 = ((m + 1) * A) / (2 * np.pi * d**2) * \
         np.cos(phi)**m * np.cos(psi) * Ts * Gc
    
    H0 = np.maximum(H0, 0.0)
    
    # Normalize to mean = 1
    H0 /= (H0.mean() + 1e-12)
    
    return H0


# =============================================================================
# C4: DIRECT PATH (combination with previous stages)
# =============================================================================

def compute_direct_path(signal, H0, g_state, xi):
    """
    C4: Compute direct path signal
    
    🎓 WHAT IS THIS?
    Combines all multiplicative channel effects on the signal.
    
    📊 COMPUTATION PROCESS:
    Direct path = H₀(t) × g(aₜ) × ξ(t) × s(t)
    
    Args:
        signal: Input signal s(t)
        H0: Lambertian channel gain
        g_state: State-dependent attenuation
        xi: Log-normal jitter
    
    Returns:
        direct_signal: Signal after direct path
    """
    direct_signal = H0 * g_state * xi * signal
    return direct_signal


# =============================================================================
# C5: DIFFUSE PATH h_diff(t)
# =============================================================================

def compute_diffuse_path(signal, beta, fs=360, tau_diff=0.015):
    """
    C5: Effective diffuse/memory path component.

    The exponential kernel h_eff(t) = exp(-t/τ_eff) represents an effective
    signal-level memory component that captures slow motion-induced received-
    power fluctuations and unresolved reflected energy in the surrogate channel
    model. It is NOT a physically resolved optical multipath delay-spread model:
    optical propagation over a 0.5–0.8 m chest-to-wrist link is on the ns scale,
    whereas τ_eff = 15 ms is orders of magnitude larger (Reviewer 3 C#7).

    The diffuse contribution is further weighted per channel state by η_diff[aₜ]
    in the C6 combination step, making it state-dependent as required by
    Reviewer 1 C#3 and Reviewer 3 C#2.

    Args:
        signal   : Input OFDM signal s(t).
        beta     : Activity-dependent diffuse weight (learned from IMU motion).
        fs       : Sampling frequency (Hz).
        tau_diff : Effective memory time constant (s); not a propagation delay.

    Returns:
        diffuse_signal: β × h_eff(t) * s(t)
    """
    # Exponential decay kernel
    K = int(4 * tau_diff * fs)
    t_kernel = np.arange(K) / fs
    h_diff = np.exp(-t_kernel / tau_diff)
    h_diff /= (h_diff.sum() + 1e-12)
    
    # Convolve
    diffuse_signal = beta * np.convolve(signal, h_diff, mode='same')
    
    return diffuse_signal


# =============================================================================
# C6: SIGNAL COMBINATION
# =============================================================================

def combine_direct_diffuse(direct_signal, diffuse_signal, states=None,
                           eta_dir=None, eta_diff=None):
    """
    C6: State-dependent combination of direct and diffuse paths.

    Revised model (Reviewer 1 C#3, Reviewer 3 C#2):
      combined(t) = η_dir[aₜ]·direct(t) + η_diff[aₜ]·diffuse(t)

    State weights vary per sample so that:
      LoS-dominant (0)        → mostly direct, little diffuse
      Partially-obstructed (1)→ balanced mix
      Diffuse-dominant (2)    → suppressed direct, dominant diffuse

    Args:
        direct_signal  : Direct path signal (after C4).
        diffuse_signal : Diffuse path signal (after C5).
        states         : Per-sample channel state array (0/1/2).
                         If None, falls back to equal weighting (legacy).
        eta_dir        : Dict {state: weight} for direct path.
                         Defaults to module-level ETA_DIR.
        eta_diff       : Dict {state: weight} for diffuse path.
                         Defaults to module-level ETA_DIFF.

    Returns:
        combined: State-weighted sum of direct + diffuse paths.
    """
    if states is None:
        # Legacy fallback: equal weighting (backward compatible)
        return direct_signal + diffuse_signal

    if eta_dir is None:
        eta_dir  = ETA_DIR
    if eta_diff is None:
        eta_diff = ETA_DIFF

    w_dir  = np.array([eta_dir [s] for s in states])
    w_diff = np.array([eta_diff[s] for s in states])
    combined = w_dir * direct_signal + w_diff * diffuse_signal
    return combined


# =============================================================================
# C7: LED NONLINEARITY
# =============================================================================

def apply_led_nonlinearity(signal, Pmax=1.2, a1=1.0, a3=-0.02, bias=0.0):
    """
    C7: Apply LED soft compression and hard clipping.

    The cubic coefficient a3 must be NEGATIVE to model saturation/compression
    at high drive levels (Reviewer 3 C#7). A positive value would expand the
    signal, which is physically incorrect for LED nonlinearity.

    Pipeline:
      1. Soft compression : y = a1·s + a3·s³  (a3 < 0 → saturation)
      2. Hard clipping     : y_clip = clip(y, 0, Pmax)

    Args:
        signal : Input signal.
        Pmax   : Maximum LED optical power (hard clip ceiling).
        a1     : Linear gain coefficient (default 1.0).
        a3     : Cubic compression coefficient; MUST be negative (default -0.02).
        bias   : DC operating-point offset.

    Returns:
        clipped_signal: LED output after compression and clipping.
    """
    nonlinear = a1 * signal + a3 * (signal ** 3) + bias
    clipped   = np.clip(nonlinear, 0.0, Pmax)
    return clipped


# =============================================================================
# C8: NOISE ADDITION
# =============================================================================

def add_noise(signal, ambient_mode='bright', rng=None):
    """
    C8: Add thermal and shot noise
    
    🎓 WHAT IS THIS?
    Receiver noise has two components:
    - Thermal noise: From electronics (constant, Gaussian)
    - Shot noise: From photon counting (signal-dependent, Gaussian approximation)
    
    📊 COMPUTATION PROCESS:
    1. Thermal: n_th ~ N(0, σ_th²)
    2. Shot: n_sh ~ N(0, σ_sh² × √signal)
    3. Total noise: n = n_th + n_sh
    
    FIXED: Reduced noise levels for realistic indoor VLC wearable:
    - Thermal noise reduced ~3x
    - Shot noise reduced ~4x
    This achieves BER ~2-6% instead of 40%+
    
    Args:
        signal: Input signal
        ambient_mode: 'bright' or 'dark'
        rng: Random number generator
    
    Returns:
        noisy_signal: Signal + noise
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    n_samples = len(signal)
    
    # FIXED: Reduced noise parameters for realistic BER
    if ambient_mode == 'bright':
        sigma_th = 0.0015  # Reduced 3.3x
        sigma_sh = 0.0025  # Reduced 4x
    else:  # dark
        sigma_th = 0.003   # Reduced 3.3x
        sigma_sh = 0.0015  # Reduced 2.7x
    
    # Thermal noise (Gaussian)
    n_thermal = rng.normal(0, sigma_th, size=n_samples)
    
    # Shot noise (signal-dependent Gaussian approximation)
    # FIXED: Generate N(0,1) samples, then scale by sqrt(signal)
    n_shot = rng.normal(0, 1, size=n_samples) * sigma_sh * np.sqrt(np.maximum(signal, 1e-12))
    
    noisy_signal = signal + n_thermal + n_shot
    
    return noisy_signal


# =============================================================================
# COMPLETE PIPELINE
# =============================================================================

def simulate_vlc_channel(signal, activity='walking', 
                        P=None, sigma=None, beta=None,
                        ambient='bright', fs=360, rng=None):
    """
    Complete VLC channel simulation pipeline
    
    Executes ALL sub-stages:
    Stage B → Markov states
    Stage C1 → State attenuation
    Stage C2 → Jitter
    Stage C3 → Lambertian
    Stage C4 → Direct path
    Stage C5 → Diffuse path
    Stage C6 → Combination
    Stage C7 → LED nonlinearity
    Stage C8 → Noise
    
    Args:
        signal: Input ECG signal (normalized)
        activity: Activity type ('walking', 'sitting', 'standing')
        P: Markov transition matrix (if None, uses default)
        sigma: Jitter parameter (if None, uses default)
        beta: Diffuse parameter (if None, uses default)
        ambient: Ambient light condition ('bright' or 'dark')
        fs: Sampling frequency
        rng: Random number generator
    
    Returns:
        received: Received signal r(t)
        states: State sequence
        components: Dictionary of ALL intermediate signals
    """
    if rng is None:
        rng = np.random.default_rng(42)
    
    n_samples = len(signal)
    
    # Use defaults if not provided
    if P is None:
        P = MARKOV_MATRICES[activity]
    if sigma is None:
        sigma = SIGMA_JITTER[activity]
    if beta is None:
        beta = BETA_DIFFUSE[activity]
    
    # Stage B: Generate Markov states
    # States: 0=LoS-dominant, 1=partially-obstructed, 2=diffuse-dominant
    states = generate_markov_states(P, n_samples, rng)

    # C1: State-dependent effective attenuation g(aₜ)
    g_state = compute_state_attenuation(states, activity, ATTENUATION_DB, rng)

    # C2: Log-normal jitter ξ(t) (motion-induced fading)
    xi = compute_lognormal_jitter(n_samples, sigma, rng)

    # C3: Lambertian geometric gain H0 (normalized; not a calibrated link budget)
    H0 = compute_lambertian_channel(n_samples, fs, activity, rng)

    # C4: Direct path = H0 · g(aₜ) · ξ(t) · s(t)
    direct_signal = compute_direct_path(signal, H0, g_state, xi)

    # C5: Diffuse/memory path.
    # τ_diff = 15 ms is an effective signal-level memory constant representing
    # slow motion-induced received-power fluctuations, NOT a physical optical
    # propagation delay (optical propagation over 0.5–0.8 m is on the ns scale).
    diffuse_signal = compute_diffuse_path(signal, beta, fs)

    # C6: State-dependent combination of direct + diffuse paths.
    # η_dir[state] and η_diff[state] weight each path per channel state so that
    # LoS-dominant, partially-obstructed, and diffuse-dominant conditions produce
    # different effective power-delay profiles (Reviewer 1 C#3, Reviewer 3 C#2).
    combined = combine_direct_diffuse(direct_signal, diffuse_signal, states=states)

    # C7: LED soft compression (a3=-0.02) followed by hard clipping at Pmax.
    clipped = apply_led_nonlinearity(combined)

    # C8: Thermal + shot noise (simulation-domain effective parameters).
    received = add_noise(clipped, ambient, rng)

    # Store ALL intermediate components for Streamlit visualization
    components = {
        'original':             signal,
        'c1_after_state_att':   g_state * signal,
        'c2_after_jitter':      g_state * xi * signal,
        'c3_after_lambertian':  H0 * g_state * xi * signal,
        'c4_direct_path':       direct_signal,
        'c5_diffuse_path':      diffuse_signal,
        'c6_combined':          combined,
        'c7_after_led':         clipped,
        'c8_received':          received,
        'states':               states,
        'state_names':          [STATE_NAMES[s] for s in states],
        'g_state':              g_state,
        'xi':                   xi,
        'H0':                   H0,
        # Per-sample path weights for inspection
        'eta_dir_applied':  np.array([ETA_DIR [s] for s in states]),
        'eta_diff_applied': np.array([ETA_DIFF[s] for s in states]),
    }

    return received, states, components


def get_state_statistics(states):
    """
    Compute state distribution statistics
    
    Args:
        states: State sequence array
    
    Returns:
        stats: Dictionary of statistics
    """
    unique, counts = np.unique(states, return_counts=True)
    total = len(states)
    
    stats = {
        # Revised state labels (Reviewer 3 C#2, Reviewer 3 C#4)
        'LoS_dominant_percent':        counts[unique == 0][0] / total * 100 if 0 in unique else 0.0,
        'partially_obstructed_percent': counts[unique == 1][0] / total * 100 if 1 in unique else 0.0,
        'diffuse_dominant_percent':    counts[unique == 2][0] / total * 100 if 2 in unique else 0.0,
        # Legacy keys kept for backward compatibility with existing Streamlit display code
        'LoS_percent':     counts[unique == 0][0] / total * 100 if 0 in unique else 0.0,
        'Partial_percent': counts[unique == 1][0] / total * 100 if 1 in unique else 0.0,
        'NLoS_percent':    counts[unique == 2][0] / total * 100 if 2 in unique else 0.0,
        'num_transitions':    np.sum(np.diff(states) != 0),
        'avg_state_duration': len(states) / (np.sum(np.diff(states) != 0) + 1)
    }

    return stats
