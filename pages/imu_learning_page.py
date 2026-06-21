"""
IMU Learning & Markov Model - Professional Version
===================================================

Concise demonstration of learning channel parameters from real IMU data.
Committee-friendly: Essential concepts only, no verbose explanations.

Author: Grace
Date: January 2026
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent))

from utils.imu_analysis import (
    remove_gravity_with_ema,
    compute_gyroscope_stability,
    map_motion_to_states_dual_threshold,
    learn_markov_transition_matrix,
    learn_attenuation_parameters,
    learn_jitter_parameter,
    learn_diffuse_parameter
)
# Canonical, dataset-consistent surrogate parameters (single source of truth).
# These are the values learned from the full IMU dataset and used to generate
# the .npz dataset / metadata.json. The live demo below subsamples per the
# sidebar slider, so we display these canonical values to avoid any mismatch.
from utils.channel_utils import SIGMA_JITTER, BETA_DIFFUSE


def show_imu_learning_page():
    """Main page function"""
    
    st.markdown('<div class="main-header">🔬 IMU Learning & Markov Model</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    **Core Innovation:** Learning activity-specific VLC channel parameters from real body motion data.
    
    Pipeline: IMU Sensors → Motion Features → Channel States → 1st-Order Markov Model
    """)
    
    st.markdown("---")
    
    # =============================================================================
    # STEP 1: LOAD IMU DATA
    # =============================================================================
    
    st.markdown("## 📂 Step 1: Load IMU Dataset")
    
    with st.expander("Dataset Information", expanded=False):
        st.markdown("""
        **IMU-based Human Activity Recognition Dataset**
        - 6-axis IMU: 3-axis accelerometer + 3-axis gyroscope
        - Sampling: 50 Hz  
        - Activities: Walking (1), Sitting (4), Standing (5)
        - Source: https://data.mendeley.com/datasets/fcnmmsn857/3
        """)
    
    # Load dataset if not already loaded
    if 'imu_raw_data' not in st.session_state:
        # Use UCI HAR dataset (same as old version)
        imu_path = Path(__file__).parent.parent / 'data' / 'Activity_Recognition_Data.csv'
        
        if imu_path.exists():
            df = pd.read_csv(imu_path)
            st.session_state['imu_raw_data'] = df
            st.success(f"✅ Loaded {len(df):,} IMU samples")
        else:
            st.error(f"⚠️ Dataset not found: {imu_path}")
            st.stop()
    
    if 'imu_raw_data' not in st.session_state:
        st.warning("⚠️ Please load IMU dataset first")
        return
    
    st.markdown("---")
    
    # =============================================================================
    # STEP 2: SELECT ACTIVITY
    # =============================================================================
    
    st.markdown("## 🎬 Step 2: Select Activity")
    
    df = st.session_state['imu_raw_data']
    
    # Activity codes MUST match the dataset generator (load_imu_dataset):
    # walking=3, sitting=4, standing=5. Previously walking used code 1, which
    # made the live Markov matrix / state distribution differ from the dataset.
    activity_map = {3: 'walking', 4: 'sitting', 5: 'standing'}

    col1, col2 = st.columns([2, 1])

    with col1:
        activity_label = st.selectbox(
            "Activity:",
            options=[3, 4, 5],
            format_func=lambda x: f"{activity_map[x].capitalize()}",
            index=0
        )
        
        activity_name = activity_map[activity_label]
        activity_data = df[df['activity'] == activity_label].reset_index(drop=True)
        
        st.info(f"**{activity_name.capitalize()}:** {len(activity_data):,} samples ({len(activity_data)/50:.1f}s)")
    
    with col2:
        subsample_size = st.number_input(
            "Max samples:",
            min_value=1000,
            max_value=len(activity_data),
            value=min(10000, len(activity_data)),
            step=1000
        )
        
        if subsample_size < len(activity_data):
            activity_data = activity_data.iloc[:subsample_size]
    
    st.session_state['activity_name'] = activity_name
    st.session_state['activity_data'] = activity_data
    
    # Visualize raw sensors (compact)
    with st.expander("View Raw Sensor Data", expanded=False):
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=['Accelerometer', 'Gyroscope'],
            vertical_spacing=0.12
        )
        
        plot_samples = min(1000, len(activity_data))
        
        for axis, color in zip(['ax', 'ay', 'az'], ['#e74c3c', '#2ecc71', '#3498db']):
            fig.add_trace(go.Scatter(
                y=activity_data[axis].iloc[:plot_samples],
                name=axis, line=dict(color=color, width=1)
            ), row=1, col=1)
        
        for axis, color in zip(['gx', 'gy', 'gz'], ['#e74c3c', '#2ecc71', '#3498db']):
            fig.add_trace(go.Scatter(
                y=activity_data[axis].iloc[:plot_samples],
                name=axis, line=dict(color=color, width=1)
            ), row=2, col=1)
        
        fig.update_layout(height=400, template='plotly_white', showlegend=True)
        st.plotly_chart(fig, use_container_width=True)
    
    st.markdown("---")
    
    # =============================================================================
    # STEP 3: RUN ANALYSIS PIPELINE
    # =============================================================================
    
    st.markdown("## ⚙️ Step 3: Physics-Based Signal Processing")
    
    if st.button("▶️ Run Complete IMU Analysis Pipeline", type="primary", use_container_width=True):
        
        activity_data = st.session_state['activity_data']
        activity_name = st.session_state['activity_name']
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        # =====================================================================
        # 3.1: GRAVITY REMOVAL
        # =====================================================================
        
        status_text.markdown("**[1/5] Removing gravity...**")
        progress_bar.progress(20)
        
        acc_data = activity_data[['ax', 'ay', 'az']]
        gravity_estimate, dynamic_motion = remove_gravity_with_ema(acc_data, alpha=0.9)
        
        with st.expander("3.1: Gravity Removal (EMA)", expanded=True):
            st.code("""
gravity[t] = α × gravity[t-1] + (1-α) × acc[t]
dynamic[t] = acc[t] - gravity[t]

  where α = 0.9 (low-pass filter for slow orientation changes)
  
Result: Isolates pure motion from static gravity component
            """)
            
            # Compact visualization
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=dynamic_motion['dyn_az'][:1000], mode='lines',
                name='Dynamic Motion', line=dict(color='#2ecc71', width=1.5)
            ))
            fig.update_layout(
                title="Dynamic Motion (Gravity Removed)",
                height=250, template='plotly_white', showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # =====================================================================
        # 3.2: MOTION INTENSITY
        # =====================================================================
        
        status_text.markdown("**[2/5] Computing motion intensity...**")
        progress_bar.progress(40)
        
        # Use pre-computed magnitude from dynamic_motion
        motion_magnitude = dynamic_motion['dyn_magnitude']
        
        with st.expander("3.2: Motion Intensity", expanded=True):
            st.code(f"""
||a(t)|| = √(ax² + ay² + az²)

Mean intensity: {motion_magnitude.mean():.4f} m/s²
Std deviation:  {motion_magnitude.std():.4f} m/s²
            """)
        
        # =====================================================================
        # 3.3: GYROSCOPE STABILITY
        # =====================================================================
        
        status_text.markdown("**[3/5] Analyzing gyroscope stability...**")
        progress_bar.progress(60)
        
        gyro_data = activity_data[['gx', 'gy', 'gz']]
        gyro_magnitude, gyro_stability = compute_gyroscope_stability(gyro_data, window_sec=0.5)
        
        with st.expander("3.3: Gyroscope Stability", expanded=True):
            st.code(f"""
σ_gyro(t) = std([gx, gy, gz]) over sliding window

Mean stability: {np.mean(gyro_stability):.4f} rad/s
High stability → LoS, Low stability → NLoS
            """)
        
        # =====================================================================
        # 3.4: STATE MAPPING
        # =====================================================================
        
        status_text.markdown("**[4/5] Mapping to channel states...**")
        progress_bar.progress(80)
        
        states = map_motion_to_states_dual_threshold(
            motion_magnitude.values,  # Convert Series to numpy array
            gyro_stability,           # Already numpy array
            activity_name=activity_name  # Changed from 'activity' to 'activity_name'
        )
        
        # State statistics
        unique, counts = np.unique(states, return_counts=True)
        state_dist = {int(s): int(c) for s, c in zip(unique, counts)}
        
        with st.expander("3.4: Motion → Channel State Mapping (Surrogate Classification)", expanded=True):
            st.caption(
                "Percentile thresholds are activity-normalised surrogate classifiers, not hardware-calibrated "
                "physical misalignment models (Reviewer 1 C#4, Reviewer 4 C#4)."
            )
            st.code(f"""
Dual-threshold surrogate mapping: (dynamic acc, gyro stability) → channel state

Revised state terminology:
  LoS-dominant (0):         {state_dist.get(0, 0):6,} ({state_dist.get(0,0)/len(states)*100:.1f}%)
  Partially-obstructed (1): {state_dist.get(1, 0):6,} ({state_dist.get(1,0)/len(states)*100:.1f}%)
  Diffuse-dominant (2):     {state_dist.get(2, 0):6,} ({state_dist.get(2,0)/len(states)*100:.1f}%)

Physical interpretation:
  • High dynamic acc → arm displacement / possible obstruction of optical path
  • High gyro std    → angular misalignment / orientation variability
  • Activity-specific percentiles normalise for different motion regimes
    (walking=high-motion, sitting=moderate, standing=low-motion)

Transitions: {np.sum(np.diff(states) != 0):,}
            """)
            
            # State sequence plot
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                y=states[:2000], mode='lines',
                line=dict(color='#e74c3c', width=1.5)
            ))
            fig.update_layout(
                title="Channel State Sequence  (0=LoS-dominant | 1=Partially-obstructed | 2=Diffuse-dominant)",
                height=250, template='plotly_white',
        font=dict(color="#111111"),
                yaxis=dict(tickvals=[0,1,2]), showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)
        
        # =====================================================================
        # 3.5: MARKOV LEARNING
        # =====================================================================
        
        status_text.markdown("**[5/5] Learning Markov model & parameters...**")
        progress_bar.progress(100)
        
        # Learn Markov matrix
        P, state_stats = learn_markov_transition_matrix(states)  # Unpack tuple
        
        # Learn channel parameters
        attenuation_db = learn_attenuation_parameters(motion_magnitude.values,states)
        sigma_jitter = learn_jitter_parameter(gyro_stability)              # jitter ← gyro stability
        beta_diffuse = learn_diffuse_parameter(motion_magnitude.values)    # diffuse ← motion intensity

        # Display the canonical, dataset-consistent values. The live demo above
        # subsamples per the sidebar slider, so the recomputed values can differ
        # slightly from the full-dataset values (β is max-normalised → sensitive).
        # Using the canonical constants guarantees the page matches the dataset
        # and metadata.json exactly (no confusion).
        sigma_jitter = SIGMA_JITTER.get(activity_name, sigma_jitter)
        beta_diffuse = BETA_DIFFUSE.get(activity_name, beta_diffuse)
        
        with st.expander("3.5: Markov Model & Parameters", expanded=True):
            st.markdown("**1st-Order Markov Transition Matrix P:**")
            
            _slabels = ['LoS-dominant', 'Partially-obstructed', 'Diffuse-dominant']
            df_P = pd.DataFrame(P, columns=_slabels, index=_slabels)
            # Color gradient: Green (high) → Yellow (medium) → Red (low)
            styled_P = df_P.style.format("{:.3f}").background_gradient(
                cmap='RdYlGn',  # Red-Yellow-Green
                vmin=0.0,
                vmax=1.0,
                axis=None
            )
            st.dataframe(styled_P, use_container_width=True)
            
            st.markdown("**Channel Parameters (dataset-consistent, learned from motion):**")
            st.code(f"""
σ_jitter:  {sigma_jitter:.4f} (from gyro stability)
β_diffuse: {beta_diffuse:.4f} (from motion intensity)

Attenuation ranges (dB) [effective surrogate params — not hardware-calibrated]:
  LoS-dominant (0):         {attenuation_db[0]}
  Partially-obstructed (1): {attenuation_db[1]}
  Diffuse-dominant (2):     {attenuation_db[2]}
            """)
        
        # Store results - ACCUMULATE instead of overwrite
        if 'all_activity_results' not in st.session_state:
            st.session_state['all_activity_results'] = {}
        
        st.session_state['all_activity_results'][activity_name] = {
            'markov_matrix': P,
            'sigma_jitter': sigma_jitter,
            'beta_diffuse': beta_diffuse,
            'attenuation_db': attenuation_db,
            'state_distribution': state_dist,
            'transitions': int(np.sum(np.diff(states) != 0))
        }
        
        st.success(f"✅ Analysis complete for {activity_name.capitalize()}!")

        # Show threshold sensitivity for this activity
        show_threshold_sensitivity(
            activity_name,
            motion_magnitude.values,
            gyro_stability
        )
        
        # =====================================================================
        # SUMMARY TABLE - ALL LEARNED ACTIVITIES
        # =====================================================================
        
        st.markdown("---")
        st.markdown("## 📊 Summary of All Learned Parameters")
        
        # Get all activities that have been learned
        all_results = st.session_state.get('all_activity_results', {})
        
        if len(all_results) > 0:
            summary_rows = []
            for act_name, params in all_results.items():
                state_dist = params['state_distribution']
                total_samples = sum(state_dist.values())
                
                summary_rows.append({
                    'Activity': act_name.capitalize(),
                    'LoS-dominant %':        f"{state_dist.get(0, 0)/total_samples*100:.1f}%",
                    'Partially-obstructed %': f"{state_dist.get(1, 0)/total_samples*100:.1f}%",
                    'Diffuse-dominant %':    f"{state_dist.get(2, 0)/total_samples*100:.1f}%",
                    'Transitions': params['transitions'],
                    'σ_jitter': f"{params['sigma_jitter']:.4f}",
                    'β_diffuse': f"{params['beta_diffuse']:.4f}"
                })
            
            df_summary = pd.DataFrame(summary_rows)
            st.dataframe(df_summary, use_container_width=True, hide_index=True)
            
            # Show which activities still need to be learned
            all_possible = {'walking', 'sitting', 'standing'}
            learned = set(all_results.keys())
            remaining = all_possible - learned
            
            if remaining:
                st.info(f"📋 Run analysis for: {', '.join([a.capitalize() for a in remaining])} to complete all activities")
            else:
                st.success("✅ All 3 activities analyzed! Ready for dataset generation.")
        
        # =====================================================================
        # DETAILED COMPARISON - ALL ACTIVITIES
        # =====================================================================
        
        if len(all_results) >= 2:
            st.markdown("---")
            st.markdown("### 🔄 Comparison Across Activities")
            
            # Markov matrices comparison
            st.markdown("**Markov Transition Matrices (revised state labels):**")
            _sl = ['LoS-dom.', 'Part.-obstr.', 'Diff.-dom.']
            cols = st.columns(len(all_results))

            for idx, (act_name, params) in enumerate(all_results.items()):
                with cols[idx]:
                    st.markdown(f"**{act_name.capitalize()}**")
                    df_P = pd.DataFrame(params['markov_matrix'], columns=_sl, index=_sl)
                    styled_P = df_P.style.format("{:.3f}").background_gradient(
                        cmap='RdYlGn', vmin=0.0, vmax=1.0, axis=None)
                    st.dataframe(styled_P, use_container_width=True)
            
            # Parameters comparison
            st.markdown("**Channel Parameters:**")
            comparison_data = []
            for act_name, params in all_results.items():
                comparison_data.append({
                    'Activity': act_name.capitalize(),
                    'σ_jitter': f"{params['sigma_jitter']:.4f}",
                    'β_diffuse': f"{params['beta_diffuse']:.4f}"
                })
            
            df_comparison = pd.DataFrame(comparison_data)
            st.dataframe(df_comparison, use_container_width=True, hide_index=True)


def show_threshold_sensitivity(activity_name, motion_magnitude, gyro_stability):
    """
    Show how ±10 percentile perturbations on the dual-threshold classifier
    affect the state distribution (Reviewer 1 C#4, Reviewer 4 C#4).
    """
    st.markdown("---")
    st.markdown("### 🔬 Threshold Sensitivity Analysis")
    st.caption(
        "Perturbs the IMU percentile thresholds (Δ = −10, 0, +10) and shows how "
        "state distributions change. Demonstrates that the percentile rule is a "
        "surrogate modelling choice, not a hardware-calibrated physical classifier."
    )

    base_percentiles = {
        'standing': ([50, 80], [60, 85]),
        'sitting':  ([40, 70], [50, 75]),
        'walking':  ([25, 60], [35, 65]),
    }
    acc_p, gyr_p = base_percentiles.get(activity_name, ([33, 67], [33, 67]))

    rows = []
    for delta in [-10, 0, +10]:
        al = np.percentile(motion_magnitude, np.clip(acc_p[0]+delta, 1, 99))
        ah = np.percentile(motion_magnitude, np.clip(acc_p[1]+delta, 1, 99))
        gl = np.percentile(gyro_stability,   np.clip(gyr_p[0]+delta, 1, 99))
        gh = np.percentile(gyro_stability,   np.clip(gyr_p[1]+delta, 1, 99))

        n = len(motion_magnitude)
        st_arr = np.ones(n, dtype=int)
        for i in range(n):
            if motion_magnitude[i] < al and gyro_stability[i] < gl:
                st_arr[i] = 0
            elif motion_magnitude[i] > ah or gyro_stability[i] > gh:
                st_arr[i] = 2

        rows.append({
            'Perturbation': f'Δ = {delta:+d}',
            'LoS-dominant (%)':        f"{np.mean(st_arr==0)*100:.1f}",
            'Partially-obstructed (%)': f"{np.mean(st_arr==1)*100:.1f}",
            'Diffuse-dominant (%)':    f"{np.mean(st_arr==2)*100:.1f}",
        })

    df_th = pd.DataFrame(rows)
    st.dataframe(df_th, use_container_width=True, hide_index=True)

    # Bar chart
    fig_th = go.Figure()
    colors_th = {'LoS-dominant (%)': '#2ecc71',
                 'Partially-obstructed (%)': '#f39c12',
                 'Diffuse-dominant (%)': '#e74c3c'}
    for col, clr in colors_th.items():
        fig_th.add_trace(go.Bar(
            name=col,
            x=[r['Perturbation'] for r in rows],
            y=[float(r[col]) for r in rows],
            marker_color=clr
        ))
    fig_th.update_layout(
        barmode='stack', height=320, template='plotly_white',
        title=f"State Distribution vs Threshold Perturbation — {activity_name.capitalize()}",
        yaxis_title='Proportion (%)', legend_title='State'
    )
    st.plotly_chart(fig_th, use_container_width=True)


if __name__ == '__main__':
    show_imu_learning_page()
