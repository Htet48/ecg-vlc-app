"""
VLC-ECG DATASET GENERATION PIPELINE - PROFESSIONAL VERSION
===========================================================

Complete demonstration of ECG signal reconstruction through VLC channel.
Shows each transformation step with essential mathematical explanations.

Author: Grace
Thesis: ECG Signal Reconstruction through VLC
Date: January 2026
"""

import streamlit as st
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path
import pandas as pd

sys.path.append(str(Path(__file__).parent.parent))

from utils.data_utils import load_mitbih_record, preprocess_ecg
from utils.aco_ofdm_vlc import ACO_OFDM_VLC
from utils.channel_utils import (
    MARKOV_MATRICES, SIGMA_JITTER, BETA_DIFFUSE, ATTENUATION_DB,
    ETA_DIR, ETA_DIFF, STATE_NAMES,
    generate_markov_states, compute_state_attenuation,
    compute_lognormal_jitter, compute_lambertian_channel,
    compute_diffuse_path, apply_led_nonlinearity, add_noise,
    combine_direct_diffuse, simulate_vlc_channel
)

def show_pipeline_explained():
    """Step-by-step mathematical pipeline — written for the reviewers."""
    with st.expander("📐 Pipeline Explained — step-by-step with equations", expanded=False):
        st.markdown(
            "Each ECG segment travels through eight stages. The equations and the "
            "exact numbers for one 3-second segment are shown below."
        )

        st.markdown("**Step 1 — Get & preprocess ECG (MIT-BIH)**")
        st.latex(r"x[n],\quad n=0,\dots,1079,\qquad 0 \le x[n] \le 1")
        st.caption("3 s × 360 Hz = 1080 samples, normalized to [0, 1].")

        st.markdown("**Step 2 — ECG → bits (8-bit quantization)**")
        st.latex(r"q[n] = \mathrm{round}\big(x[n]\cdot(2^{8}-1)\big)\;\Rightarrow\;"
                 r"1080\times 8 = 8640\ \text{bits}")

        st.markdown("**Step 3 — bits → 4-QAM symbols**")
        st.latex(r"8640\ \text{bits} \div 2 = 4320\ \text{symbols},\qquad "
                 r"s_m \in \tfrac{1}{\sqrt2}\{\pm1\pm j\}")

        st.markdown("**Step 4 — ACO-OFDM modulation (corrected N⁄4)**")
        st.latex(r"\text{independent subcarriers} = \tfrac{N}{4} = 64,\qquad "
                 r"\lceil 4320/64 \rceil = 68\ \text{frames}")
        st.latex(r"X[k]=X^{*}[N-k]\ \text{(Hermitian)},\quad "
                 r"s(t)=\max\!\big(\mathrm{IFFT}\{X\},\,0\big)\ \text{(ACO clip)}")
        st.caption("Data on positive-frequency odd carriers k = 1,3,…,127. "
                   "Add a 64-sample cyclic prefix → each frame = 320 samples.")

        st.markdown("**Step 5 — VLC pilot channel (C1–C8)**")
        st.latex(r"r(t)=\mathrm{LED}\!\big[\;\eta_{dir}[a_t]\,H_0\,g(a_t)\,\xi(t)\,s(t)"
                 r"\;+\;\eta_{diff}[a_t]\,\beta\,(h_{eff}\!*\!s)(t)\;\big]"
                 r"\;+\;n_{th}+n_{sh}")
        st.latex(r"\mathrm{LED}(c)=a_1 c + a_3 c^{3}\ (a_3=-0.02),\quad "
                 r"\text{clip to }[0,\,P_{max}{=}1.2]")
        st.caption("The diffuse convolution (h_eff, τ_eff = 15 ms) spreads energy "
                   "across samples → inter-symbol interference → bit errors. "
                   "More motion → larger η_diff → more errors.")

        st.markdown("**Step 6 — ACO-OFDM demodulation**")
        st.latex(r"\hat{X}[k]=\mathrm{FFT}\{r_{\text{no-CP}}\},\qquad "
                 r"\hat{s}_m = \hat{X}[1\!:\!N/2\!:\!2]\ \ (64\ \text{carriers})")

        st.markdown("**Step 7 — symbols → bits → BER**")
        st.latex(r"\mathrm{BER}=\frac{1}{N_b}\sum_{k=1}^{N_b}"
                 r"\mathbb{1}\!\left[b_k^{tx}\neq b_k^{rx}\right]")
        st.caption("Raw/uncoded BER — no FEC, no equalization. e.g. 0.068 = 6.8%.")

        st.markdown("**Step 8 — reconstruct ECG → SQ-SNR**")
        st.latex(r"\mathrm{SQ\text{-}SNR}=10\log_{10}"
                 r"\frac{\sum_n \hat{x}^2[n]}{\sum_n\big(x[n]-\hat{x}[n]\big)^2}\ \text{dB}")

        st.info(
            "**Full chain:** ECG (1080) → 8640 bits → 4320 symbols → 68 OFDM frames "
            "→ s(t) → [VLC channel] → r(t) → demod → 4320 symbols → 8640 bits → "
            "**BER**; reconstruct → **SQ-SNR**. The deep-learning models then recover "
            "the ECG from the distorted r(t) **without** equalization — that is the "
            "research contribution."
        )


def show_dataset_generation():
    st.markdown('<div class="main-header">📊 VLC-ECG Dataset Generation Pipeline</div>', unsafe_allow_html=True)

    st.markdown("""
    **Complete pipeline: ECG → ACO-OFDM → VLC Channel → Reconstruction**

    This demonstrates the novel approach of using IMU-learned Markov parameters
    for activity-specific VLC channel modeling in ECG transmission.
    """)

    # Step-by-step mathematical pipeline (for reviewers)
    show_pipeline_explained()

    # Show system diagram
    show_system_overview()

    st.markdown("---")

    # =============================================================================
    # STEP 1: ECG LOADING & PREPROCESSING
    # =============================================================================

    st.markdown("## 📥 Step 1: ECG Data Preparation")

    with st.expander("Load MIT-BIH ECG Record", expanded=False):
        col1, col2, col3 = st.columns([2, 1, 1])

        with col1:
            available_records = [
                '100', '101', '103', '105', '106', '108', '109',
                '111', '112', '113', '114', '116', '117',
                '118', '119', '121', '122', '123', '124',
                '200', '201', '202', '203', '205', '207',
                '208', '209', '210', '212', '213'
            ]
            record_id = st.selectbox("Record:", available_records, index=0)

        with col2:
            duration = st.slider("Duration (s):", 10, 300, 60, step=10)

        with col3:
            sampfrom = st.number_input("Start:", 0, 10000, 0, step=1000)

        if st.button("📥 Load ECG", type="primary"):
            with st.spinner("Loading..."):
                try:
                    ecg_raw, fs, _ = load_mitbih_record(record_id, duration, sampfrom)

                    if ecg_raw is not None and len(ecg_raw) > 0:
                        # Preprocess to 3-second segments (1080 samples @ 360 Hz)
                        segments, r_peaks, _ = preprocess_ecg(ecg_raw, fs=fs, target_window=1080)

                        if len(segments) > 0:
                            st.session_state['ecg_clean'] = segments[0]
                            st.session_state['record_id'] = record_id
                            st.session_state['all_segments'] = segments

                            # Calculate heart rate
                            if len(r_peaks) > 1:
                                rr_intervals = np.diff(r_peaks) / fs
                                hr = 60 / np.mean(rr_intervals)
                                st.session_state['heart_rate'] = hr

                            st.success(f"✅ Loaded: {len(segments)} segments × 1080 samples (3 sec each)")

                            # Visualize
                            fig = go.Figure()
                            fig.add_trace(go.Scatter(
                                y=segments[0], mode='lines',
                                line=dict(color='#2ecc71', width=1.5),
                                name='ECG'
                            ))
                            fig.update_layout(
                                title=f"Record {record_id}: Clean ECG Segment",
                                xaxis_title="Sample", yaxis_title="Amplitude",
                                height=300, template='plotly_dark', showlegend=False
                            )
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("No valid segments found")
                    else:
                        st.error("Failed to load ECG")
                except Exception as e:
                    st.error(f"Error: {e}")

    if 'ecg_clean' not in st.session_state:
        st.info("👆 Please load ECG data first")
        return

    st.markdown("---")

    # =============================================================================
    # STEP 2: ACTIVITY & PARAMETERS
    # =============================================================================

    st.markdown("## ⚙️ Step 2: Activity Selection & Parameters")

    col1, col2 = st.columns(2)

    with col1:
        activity = st.selectbox(
            "Activity:",
            ['walking', 'sitting', 'standing'],
            help="Determines IMU-learned channel parameters"
        )

    with col2:
        param_source = st.radio(
            "Parameter Source:",
            ['Learned (IMU)', 'Pre-calculated'],
            help="Use real IMU-learned or pre-calculated parameters"
        )

    # Get parameters
    if param_source == 'Learned (IMU)' and 'all_activity_results' in st.session_state:
        # Use learned parameters from IMU analysis
        params = st.session_state['all_activity_results'][activity]
        P = params['markov_matrix']
        sigma = params['sigma_jitter']
        beta = params['beta_diffuse']
        # ✅ FIX: Use LEARNED attenuation, wrapped in full dict format
        learned_att = params['attenuation_db']  # {0: (...), 1: (...), 2: (...)}
        attenuation_to_use = {activity: learned_att}  # Wrap: {'walking': {...}}
        param_used = "LEARNED from IMU"
    elif param_source == 'Learned (IMU)':
        # User selected "Learned" but IMU analysis not run yet
        st.warning("⚠️ Please run IMU analysis first (go to 'IMU & Markov Learning' page)")
        st.info("Using PRE-CALCULATED parameters for now")
        P = MARKOV_MATRICES[activity]
        sigma = SIGMA_JITTER[activity]
        beta = BETA_DIFFUSE[activity]
        attenuation_to_use = ATTENUATION_DB
        param_used = "PRE-CALCULATED"
    else:
        # User selected "Pre-calculated"
        P = MARKOV_MATRICES[activity]
        sigma = SIGMA_JITTER[activity]
        beta = BETA_DIFFUSE[activity]
        attenuation_to_use = ATTENUATION_DB  # Use default full dict
        param_used = "PRE-CALCULATED"

    # Show parameters compactly
    with st.expander("View Channel Parameters", expanded=False):
        # Show parameter source
        if param_used == "LEARNED from IMU":
            st.success(f"✅ Using **{param_used}** parameters")
        else:
            st.info(f"ℹ️ Using **{param_used}** parameters")

        col1, col2 = st.columns(2)

        with col1:
            st.markdown("**Markov Matrix P:**")
            _labels = ['LoS-dominant', 'Partially-obstructed', 'Diffuse-dominant']
            df_P = pd.DataFrame(P, columns=_labels, index=_labels)
            st.dataframe(df_P.style.format("{:.3f}"))

        with col2:
            st.markdown("**Parameters:**")
            # ✅ FIX: Extract activity-specific attenuation for display
            att_display = attenuation_to_use[activity] if activity in attenuation_to_use else attenuation_to_use
            st.code(f"""
σ_jitter: {sigma:.4f}
β_diffuse: {beta:.4f}

Attenuation (dB):
  LoS-dominant (0):        {att_display[0]}
  Partially-obstructed (1): {att_display[1]}
  Diffuse-dominant (2):    {att_display[2]}
            """)

    # Modulation parameters
    col1, col2, col3 = st.columns(3)
    with col1:
        fft_size = st.selectbox("FFT Size:", [128, 256, 512], index=1)
    with col2:
        cp_length = st.selectbox("CP Length:", [32, 64, 128], index=1)
    with col3:
        qam_order = st.selectbox("QAM Order:", [4, 16], index=0)

    st.markdown("---")

    # =============================================================================
    # STEP 3: EXECUTE PIPELINE
    # =============================================================================

    st.markdown("## 🚀 Step 3: Execute Transmission Pipeline")

    if st.button("▶️ Run Complete Pipeline", type="primary", use_container_width=True):

        ecg_clean = st.session_state['ecg_clean']
        aco = ACO_OFDM_VLC(N=fft_size, cp_len=cp_length, M=qam_order)
        rng = np.random.default_rng()
        results = {}

        # =====================================================================
        # STAGE A: ACO-OFDM MODULATION
        # =====================================================================

        with st.expander("📤 Stage A: ACO-OFDM Modulation", expanded=True):
            st.markdown("**Transmitter: ECG → Optical Signal**")

            # Modulation steps
            bits_tx = aco.ecg_to_bits(ecg_clean, bit_depth=8)
            symbols_tx = aco.bits_to_symbols(bits_tx)
            s_t, n_blocks = aco.aco_ofdm_modulate(symbols_tx)

            results['s_t'] = s_t
            results['bits_tx'] = bits_tx
            results['symbols_tx'] = symbols_tx

            st.code(f"""
Pipeline: ECG → Bits → QAM Symbols → ACO-OFDM

  1080 samples × 8 bits = {len(bits_tx)} bits
  {len(bits_tx)} bits ÷ {int(np.log2(qam_order))} = {len(symbols_tx)} symbols
  Independent data subcarriers per frame: N/4 = {fft_size // 4}
  {len(symbols_tx)} symbols → {n_blocks} OFDM blocks → {len(s_t)} samples s(t)

ACO-OFDM: Hermitian symmetry + clipping → real, non-negative signal
  CORRECTED: only the N/4 positive-frequency odd subcarriers carry data
  (the negative-frequency odd carriers are conjugate mirrors — no new data)
            """)

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=s_t[:1000], mode='lines',
                                    line=dict(color='#3498db', width=1)))
            fig.update_layout(
                title="Transmitted Signal s(t)",
                xaxis_title="Sample", yaxis_title="Amplitude",
                height=250, template='plotly_dark', showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # =====================================================================
        # STAGE B: MARKOV STATE GENERATION
        # =====================================================================

        with st.expander("🎯 Stage B: Generate Channel States (IMU-based)", expanded=True):
            st.markdown("**Key Innovation: IMU-learned Markov model**")

            # Generate states
            states = generate_markov_states(P, len(s_t), rng)
            results['states'] = states

            st.code(f"""
Markov Chain: P(st+1 | st) — IMU-informed motion-conditioned surrogate model

  Generated {len(states)} states:
    LoS-dominant (0):         {np.sum(states==0):5} ({np.sum(states==0)/len(states)*100:.1f}%)
    Partially-obstructed (1): {np.sum(states==1):5} ({np.sum(states==1)/len(states)*100:.1f}%)
    Diffuse-dominant (2):     {np.sum(states==2):5} ({np.sum(states==2)/len(states)*100:.1f}%)

  Path weights per state:
    State 0 — η_dir={ETA_DIR[0]:.2f}  η_diff={ETA_DIFF[0]:.2f}  (strong direct path)
    State 1 — η_dir={ETA_DIR[1]:.2f}  η_diff={ETA_DIFF[1]:.2f}  (balanced mix)
    State 2 — η_dir={ETA_DIR[2]:.2f}  η_diff={ETA_DIFF[2]:.2f}  (diffuse dominant)

  Transitions: {np.sum(np.diff(states) != 0)} state changes
            """)

            # Plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=states[:1000], mode='lines',
                                    line=dict(color='#e74c3c', width=1.5)))
            fig.update_layout(
                title="Markov State Sequence (0=LoS-dominant, 1=Partially-obstructed, 2=Diffuse-dominant)",
                xaxis_title="Sample", yaxis_title="State",
                height=250, template='plotly_dark',
                yaxis=dict(tickvals=[0,1,2]), showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # =====================================================================
        # STAGE C: VLC CHANNEL MODEL
        # =====================================================================

        with st.expander("📡 Stage C: VLC Channel Transformation", expanded=True):
            st.markdown("**Complete channel model: 8 components**")

            st.latex(
                r"r(t) = \left[\eta^{\mathrm{dir}}_{a_t} H_0 g(a_t)\xi(t)s(t)"
                r"+ \eta^{\mathrm{diff}}_{a_t}\beta\, h_{\mathrm{eff}}(t)*s(t)\right]_{\mathrm{LED}} + n(t)"
            )
            st.caption(
                "Revised model: state-dependent weights η_dir and η_diff vary the direct/diffuse "
                "balance per channel state (LoS-dominant → η_dir=0.95; Diffuse-dominant → η_diff=0.80). "
                "τ_eff = 15 ms is an effective signal-level memory constant, not a physical optical propagation delay."
            )

            # C1: State attenuation
            st.markdown("**C1. State-Dependent Attenuation g(aₜ)**")
            g_state = compute_state_attenuation(states, activity, attenuation_to_use, rng)
            results['g_state'] = g_state

            st.code(f"""
g(aₜ) = 10^(AttdB/20)
  where aₜ = Markov state (LoS/Partial/NLoS)
        AttdB = attenuation in dB, learned from IMU motion

  Result: g ∈ [{g_state.min():.4f}, {g_state.max():.4f}], mean={g_state.mean():.4f}
            """)

            # C2: Log-normal jitter
            st.markdown("**C2. Log-Normal Jitter ξ(t)**")
            xi = compute_lognormal_jitter(len(s_t), sigma, rng)
            results['xi'] = xi

            st.code(f"""
ξ(t) = exp(ν), ν ~ N(-σ²/2, σ²)
  where σ = {sigma:.4f} (jitter parameter from IMU)
        ν = log-normal random variable (zero-mean)

  Result: E[ξ]={xi.mean():.4f} ≈ 1.0 (unbiased), Std={xi.std():.4f}
            """)

            # C3: Lambertian channel
            st.markdown("**C3. Lambertian Channel H₀(t)**")
            H0 = compute_lambertian_channel(len(s_t), 360, activity, rng)
            results['H0'] = H0

            st.code(f"""
H₀(t) = (m+1)A/(2πd²) · cos^m(φ) · cos(ψ)
  where m = 1 (Lambertian order for standard LED)
        A = photodetector area, d = distance
        φ(t) = irradiance angle, ψ(t) = incidence angle
        (angles vary with body motion)

  Result: E[H₀]={H0.mean():.4f} ≈ 1.0 (normalized for stability)
            """)

            # C4-C6: State-dependent direct + diffuse combination
            st.markdown("**C4-C6. State-Dependent Direct + Diffuse Paths**")
            direct  = H0 * g_state * xi * s_t
            diffuse = compute_diffuse_path(s_t, beta, 360)
            combined = combine_direct_diffuse(direct, diffuse, states=states)
            results['direct']   = direct
            results['diffuse']  = diffuse
            results['combined'] = combined

            # Show η weights table
            eta_df = pd.DataFrame({
                'State': ['LoS-dominant (0)', 'Partially-obstructed (1)', 'Diffuse-dominant (2)'],
                'η_dir (direct weight)':  [ETA_DIR[0],  ETA_DIR[1],  ETA_DIR[2]],
                'η_diff (diffuse weight)': [ETA_DIFF[0], ETA_DIFF[1], ETA_DIFF[2]],
            })
            st.dataframe(eta_df.style.format({'η_dir (direct weight)': '{:.2f}',
                                              'η_diff (diffuse weight)': '{:.2f}'}),
                         use_container_width=True, hide_index=True)

            st.code(f"""
C4  Direct core : H₀ · g(aₜ) · ξ(t) · s(t)
C5  Diffuse core: β · h_eff(t) * s(t)
      β = {beta:.4f}  (learned from IMU motion intensity)
      τ_eff = 15 ms  (effective memory — NOT optical propagation delay)

C6  Combined (state-dependent):
      r_combined(t) = η_dir[aₜ] · direct + η_diff[aₜ] · diffuse
      → weights vary per sample according to Markov state aₜ
            """)

            # C7: LED nonlinearity
            st.markdown("**C7. LED Nonlinearity**")
            led_out = apply_led_nonlinearity(combined)
            results['led_out'] = led_out

            st.code(f"""
y_LED = clip( a₁·x + a₃·x³ ,  0, Pmax )
  where a₁ = 1.0   (linear gain)
        a₃ = -0.02  (cubic COMPRESSION — negative = saturation, not expansion)
        Pmax = 1.2  (hard clip at maximum optical power)

  Soft compression followed by hard clipping models LED saturation correctly.
  NOTE: a₃ must be NEGATIVE; a positive value would expand the signal (wrong).
            """)

            # C8: Noise
            st.markdown("**C8. Noise Addition**")
            r_t = add_noise(led_out, 'bright', rng)
            results['r_t'] = r_t

            # Calculate optical SNR (physical layer)
            signal_power = np.mean(led_out ** 2)
            noise_power = np.mean((r_t - led_out) ** 2)
            optical_snr_db = 10 * np.log10(signal_power / (noise_power + 1e-10))

            st.code(f"""
n(t) = n_thermal + n_shot
  where n_thermal = thermal noise from photodetector
        n_shot = shot noise from ambient light

  Optical SNR = {optical_snr_db:.2f} dB (physical layer, before demodulation)
  Note: This is NOT end-to-end SNR. After demodulation, effective
        SNR ≈ dB due to ACO-OFDM distortion & bit errors.
            """)

            # Plot received signal
            fig = go.Figure()
            fig.add_trace(go.Scatter(y=r_t[:1000], mode='lines',
                                    line=dict(color='#e67e22', width=1)))
            fig.update_layout(
                title="Received Signal r(t) after VLC Channel",
                xaxis_title="Sample", yaxis_title="Amplitude",
                height=250, template='plotly_dark', showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

        # =====================================================================
        # STAGE D: DEMODULATION
        # =====================================================================

        with st.expander("📥 Stage D: ACO-OFDM Demodulation", expanded=True):
            st.markdown("**Receiver: Optical Signal → Reconstructed ECG**")

            # Demodulation
            rx_symbols = aco.aco_ofdm_demodulate(r_t, n_blocks)
            rx_bits = aco.symbols_to_bits(rx_symbols[:len(symbols_tx)])
            ecg_recon = aco.bits_to_ecg(rx_bits, 1080)

            results['ecg_recon'] = ecg_recon

            # Calculate BER and PRD
            ber = np.sum(bits_tx[:len(rx_bits)] != rx_bits) / len(rx_bits)

            # PRD
            min_len = min(len(ecg_clean), len(ecg_recon))
            prd = 100 * np.sqrt(np.mean((ecg_clean[:min_len] - ecg_recon[:min_len])**2)) / \
                  np.sqrt(np.mean(ecg_clean[:min_len]**2))

            # Signal Quality SNR (Application Layer)
            ecg_rx_power = np.mean(ecg_recon[:min_len] ** 2)
            ecg_error_power = np.mean((ecg_clean[:min_len] - ecg_recon[:min_len]) ** 2)
            signal_quality_snr_db = 10 * np.log10(ecg_rx_power / (ecg_error_power + 1e-10))

            # Create metrics display with clear relationship
            st.markdown("#### 📊 Performance Metrics (Application Layer)")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Raw/Uncoded BER", f"{ber*100:.2f}%",
                         help="Raw BER after ACO-OFDM demodulation — no FEC or equalization applied. "
                              "High BER is expected and characterises channel severity. "
                              "(Prof's Response to Reviewer 3 C#1)")
            with col2:
                st.metric("PRD", f"{prd:.2f}%",
                         delta=f"Target: <9%",
                         delta_color="inverse" if prd > 9 else "normal",
                         help="Percent Root-mean-square Difference: ECG distortion")
            with col3:
                st.metric("Signal Quality SNR", f"{signal_quality_snr_db:.2f} dB",
                         help="Received signal quality: r(t) at RX end")

            st.code(f"""
*** APPLICATION LAYER: Signal Quality Assessment ***

Measures: Received ECG quality (r(t) at RX end)

 Raw BER = {ber*100:6.2f}%  ← Raw/uncoded BER after ACO-OFDM demodulation
             (high BER expected — characterises channel severity before FEC)
 PRD  = {prd:6.2f}%  ← Waveform distortion
 SNR  = {signal_quality_snr_db:6.2f} dB ← Received signal quality (r(t))

Formula:
  Signal power = mean(r(t)²)     = {ecg_rx_power:.6f}
  Error power  = mean((x-r(t))²) = {ecg_error_power:.6f}
  SNR = 10 × log₁₀(signal / error)

Relationship:
  High BER ({ber*100:.1f}%) → High PRD ({prd:.1f}%) → Low SNR ({signal_quality_snr_db:.1f} dB)


*** PHYSICAL LAYER: Optical Channel Quality ***

 Optical SNR = {optical_snr_db:.2f} dB (photodetector signal quality)

Why different from Signal Quality SNR?
  • Optical SNR: Physical layer (before demodulation)
  • Signal Quality SNR: Application layer (after full pipeline)
  • Gap = {optical_snr_db - signal_quality_snr_db:.2f} dB loss from OFDM + quantization + BER

Both metrics important:
  ✓ Optical SNR validates VLC channel model
  ✓ Signal Quality SNR evaluates clinical usability
            """)

            # Compare original vs reconstructed
            fig = make_subplots(
                rows=2, cols=1,
                subplot_titles=("Original ECG", "Received ECG (After VLC Channel)"),
                vertical_spacing=0.12
            )

            fig.add_trace(go.Scatter(y=ecg_clean, mode='lines',
                                    line=dict(color='#2ecc71', width=1.5),
                                    name='Original'), row=1, col=1)

            fig.add_trace(go.Scatter(y=ecg_recon, mode='lines',
                                    line=dict(color='#e67e22', width=1.5),
                                    name='Reconstructed'), row=2, col=1)

            fig.update_xaxes(title_text="Sample", row=2, col=1)
            fig.update_yaxes(title_text="Amplitude", row=1, col=1)
            fig.update_yaxes(title_text="Amplitude", row=2, col=1)

            fig.update_layout(height=500, template='plotly_dark', showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

            # Quality assessment (described as signal-fidelity proxy, not clinical validation)
            if prd < 9:
                st.success(f"✅ Signal fidelity proxy (PRD={prd:.2f}% < 9%) — diagnostic-preservation threshold")
            elif prd < 15:
                st.warning(f"⚠️ Moderate fidelity (PRD={prd:.2f}%, 9–15%)")
            else:
                st.error(f"❌ Low fidelity (PRD={prd:.2f}% > 15%) — high channel distortion")

        # =====================================================================
        # SENSITIVITY ANALYSIS — 3 ambient/occlusion scenarios
        # =====================================================================
        st.markdown("---")
        st.markdown("## 🔬 Sensitivity Analysis — Ambient / Occlusion Scenarios")

        # Explanation box — what the raw BER represents (post Hermitian/N-4 fix)
        st.info(
            "**About the reported BER:** "
            "The BER shown here is the **raw/uncoded BER** after ACO-OFDM demodulation — "
            "before any forward error correction (FEC) or equalization. "
            "Following the **Hermitian / N⁄4 subcarrier correction** (64 independent "
            "subcarriers, not 128), the BER now reflects only genuine channel impairments "
            "— noise, LED compression and motion-induced diffuse ISI — giving a realistic "
            "mean of ≈ 6.8% (walking ≈ 9.3%, sitting ≈ 6.5%, standing ≈ 3.8%). "
            "An earlier version double-counted the conjugate-mirror subcarriers, producing "
            "an artificial ≈ 25% BER floor. The deep learning models reconstruct the ECG "
            "from this distorted signal without equalization. "
            "*(Reviewer 3, Comments #1 & #3)*"
        )
        st.caption(
            "Three scenarios perturb the effective attenuation, jitter, and diffuse weights "
            "to emulate low, nominal, and severe ambient-light/occlusion conditions. "
            "Each scenario should produce a meaningfully different BER, showing robustness "
            "of the surrogate channel model (Reviewer 1 C#1,2 / Reviewer 2 C#4,5)."
        )
        st.success(
            "**What this table is for — the Ambient Robustness Scan.** "
            "Unlike the single-segment demo above (which varies per run) and the full-dataset "
            "per-activity statistics, this table fixes the activity mix (40% walking / 30% "
            "sitting / 30% standing) and instead **sweeps the ambient/occlusion condition** "
            "(low → nominal → severe) on a 200-segment subsample. It has two purposes: "
            "**(1)** it validates that the surrogate VLC channel behaves physically — BER rises "
            "monotonically as light/occlusion worsen (6.5% → 6.8% → 8.1%); and **(2)** it shows "
            "the *whole* low→severe ambient swing is only **≈ +1.6 pp**, far smaller than the "
            "**5.5 pp** activity swing (walking 9.3% → standing 3.8%). Ambient noise barely moves "
            "the BER while motion dominates → the channel is **motion-limited, not "
            "ambient-noise-limited**. (Values are a fixed canonical summary and do **not** change "
            "with the Activity selector — the per-activity breakdown lives on the Sensitivity page.)"
        )

        # ── Canonical sensitivity from metadata.json ──────────────────────────
        import json as _json
        _meta_path = Path('datasets/thesis_dataset/metadata.json')
        _w = {'walking': 0.4, 'sitting': 0.3, 'standing': 0.3}   # activity mix
        _lab = {'low_ambient': 'Low ambient',
                'nominal': 'Nominal (baseline)',
                'severe_ambient': 'Severe ambient'}
        _desc = {'low_ambient': 'Low ambient light / minimal occlusion',
                 'nominal': 'Indoor bright — surrogate channel baseline',
                 'severe_ambient': 'Strong sunlight / heavy occlusion (worst case)'}
        sens_rows = []
        if _meta_path.exists():
            _m = _json.load(open(_meta_path, encoding='utf-8'))
            _sa = _m.get('sensitivity_analysis', {})
            for _sc in ['low_ambient', 'nominal', 'severe_ambient']:
                cells = _sa.get(_sc)
                if not cells:
                    continue
                acts = [a for a in _w if a in cells]
                _ber = sum(_w[a] * cells[a]['ber'] for a in acts) * 100
                _snr = sum(_w[a] * cells[a]['sq_snr'] for a in acts)
                _prd = sum(_w[a] * cells[a].get('prd', 0.0) for a in acts)
                sens_rows.append({
                    'Scenario':    _lab[_sc],
                    'Conditions':  _desc[_sc],
                    'Raw BER (%)': f"{_ber:.2f}",
                    'SQ-SNR (dB)': f"{_snr:.2f}",
                    'PRD (%)':     f"{_prd:.2f}",
                })
            st.caption(
                "Values are read from `metadata.json` (canonical, full-dataset-aligned, "
                "activity-weighted 40/30/30) — identical to the Sensitivity Analysis page and "
                "the report. Re-run the generator to refresh."
            )
        else:
            st.warning("`metadata.json` not found — run the dataset generator to populate the "
                       "sensitivity analysis.")

        st.markdown(
            "**How each row is computed (40/30/30 activity weighting).** "
            "Every scenario is measured on **200 segments *per activity*** (walking, sitting, "
            "standing) under that ambient perturbation, then combined into one number using the "
            "dataset's activity mix:\n\n"
            "> `BER_scenario = 0.40 · BER_walking + 0.30 · BER_sitting + 0.30 · BER_standing`\n\n"
            "Example (low ambient): `0.40·8.81 + 0.30·6.28 + 0.30·3.63 = 6.50%`. "
            "The same weighting gives the SQ-SNR and PRD columns. So each table row **is** the "
            "200-segment result, summarized across the three activities — the per-activity "
            "200-segment breakdown is on the Sensitivity Analysis page."
        )
        df_sens = pd.DataFrame(sens_rows)
        st.dataframe(df_sens, use_container_width=True, hide_index=True)

        # Single BER bar chart
        if sens_rows:
            colors_s = ['#2ecc71', '#3498db', '#e74c3c'][:len(sens_rows)]
            fig_ber = go.Figure()
            fig_ber.add_trace(go.Bar(
                x=[r['Scenario'] for r in sens_rows],
                y=[float(r['Raw BER (%)']) for r in sens_rows],
                marker_color=colors_s,
                text=[f"{float(r['Raw BER (%)']):.1f}%" for r in sens_rows],
                textposition='outside'
            ))
            fig_ber.update_layout(
                title="Raw/Uncoded BER (%) per Scenario",
                yaxis_title='BER (%)', height=350,
                template='plotly_dark', showlegend=False,
                yaxis=dict(range=[0, 14])
            )
            st.plotly_chart(fig_ber, use_container_width=True)

        st.caption(
            "**Interpretation:** Raw BER increases from Low→Nominal→Severe as attenuation, "
            "jitter, and diffuse contribution increase. "
            "The surrogate channel model produces a range of BER values consistent with the "
            "expected activity-dependent impairment levels. "
            "Note: BER is raw/uncoded BER after ACO-OFDM demodulation. "
            "PRD and SQ-SNR are signal-fidelity proxies, not clinical validation metrics."
        )

        # Store results
        st.session_state['pipeline_results'] = results
        st.session_state['final_metrics'] = {
            'ber': ber,
            'prd': prd,
            'optical_snr_db': optical_snr_db,
            'signal_quality_snr_db': signal_quality_snr_db,
            'activity': activity
        }


def show_system_overview():
    """Show concise system diagram"""

    st.markdown("""
    ```
    *** VLC-ECG TRANSMISSION SYSTEM ***


  ECG(t) → ACO-OFDM → s(t) → [VLC Channel] → r(t) → Demod → ECG'
             TX              IMU-learned             RX
                             Markov model

  Channel Components:
    • State attenuation g(aₜ)  • Jitter ξ(t)
    • Lambertian H₀(t)         • Diffuse path
    • LED nonlinearity         • Noise n(t)

  Key Innovation: Activity-specific parameters from IMU data

    ```
    """)


if __name__ == '__main__':
    show_dataset_generation()
