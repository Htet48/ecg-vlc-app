"""
Home page for thesis demonstration application

Author: Htet@Grace
"""

import streamlit as st
import plotly.graph_objects as go


def show_home():
    """Display home page with thesis overview"""
    
    st.markdown('<div class="main-header">🫀 ECG Signal Reconstruction through VLC</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    ## Demonstrated by Htet@Grace (2026)
    
    Welcome to the interactive demonstration of the work on **Motion-Driven Markov Channel Modeling and Learning-Based Reconstruction for On-Body Optical Wireless ECG Transmission**.
    """)
    
    st.markdown("---")
    
    # Key contributions
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 🎯 Key Contributions
        
        **1. IMU-Learned Markov Channel Model**
        - First to use REAL body motion (IMU sensors) to learn VLC channel behavior
        - Activity-specific parameters (walking, sitting, standing)
        - 1st-order Markov captures temporal dynamics
        
        **2. Physics-Based Processing Pipeline**
        - EMA gravity removal for accelerometer
        - Rolling-window gyroscope stability analysis
        - Dual-threshold state classification
        
        **3. Complete Signal Flow**
        - MIT-BIH ECG → ACO-OFDM → VLC Channel → Reconstruction
        - All sub-stages visualized (C1-C8)
        - Realistic BER
        """)
    
    with col2:
        st.markdown("""
        **Our Approach:**
        - ✅ Dynamic channel from REAL IMU data
        - ✅ Activity-aware parameters
        - ✅ Realistic VLC
        - ✅ Advanced deep learning reconstruction
        
        **Result:**
        More realistic simulation of wearable ECG-VLC systems!
        """)
    
    st.markdown("---")
    
    # System overview
    st.markdown("### 🔄 Complete System Pipeline")
    
    pipeline_text = """
    ┌─────────────────────────────────────────────────────────────────┐
    │                    ECG SIGNAL SOURCE                            │
    │  MIT-BIH Database → 1080 samples (3 sec × 360 Hz)               │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │               STAGE A: ACO-OFDM MODULATION                      │
    │  A1. ECG → 8-bit Quantization                                   │
    │  A2. Bits → QAM Symbols (4-QAM)                                 │
    │  A3. ACO-OFDM (Odd subcarriers + IFFT + Cyclic Prefix)          │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓
                            s(t) [OFDM Signal]
                                    ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │         STAGE B: MARKOV STATE GENERATION (IMU-LEARNED)          │
    │  • Load real IMU data (accelerometer + gyroscope)               │
    │  • Physics-based processing (EMA, Rolling Std)                  │
    │  • Learn transition matrix P from motion patterns               │
    │  • Generate state sequence: a[t] ∈ {LoS, Partial, NLoS}         │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │            STAGE C: VLC CHANNEL EFFECTS (8 SUB-STAGES)          │
    │  C1. State-Dependent Attenuation g(aₜ)   ← Learned from IMU      |
    │  C2. Log-Normal Jitter ξ(t)             ← Learned from IMU      │
    │  C3. Lambertian Channel H₀(t)           ← Physics model         │
    │  C4. Direct Path Combination                                    │
    │  C5. Diffuse Path h_diff(t)             ← Learned from IMU      │
    │  C6. Signal Combination (Direct + Diffuse)                      │
    │  C7. LED Nonlinearity & Clipping                                │ 
    │  C8. Noise Addition (Thermal + Shot)                            │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓
                            r(t) [Received Signal]
                                    ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │          STAGE D: ACO-OFDM DEMODULATION & BER                   │
    │  • Remove Cyclic Prefix                                         │
    │  • FFT & Extract Odd Subcarriers                                │
    │  • QAM De-mapping → Bits                                        │
    │  • Bits → Reconstructed ECG                                     │
    │  • Calculate Bit Error Rate (BER)                               │
    └─────────────────────────────────────────────────────────────────┘
                                    ↓
    ┌─────────────────────────────────────────────────────────────────┐
    │          STAGE E: SIGNAL RECONSTRUCTION (OPTIONAL)              │
    │  Classical Methods:                                             │
    │    • Interpolation(1), Wavelet(2), OFDM Mitigation(3)           │
    │      and Combined(1+2+3)                                        │
    │  Deep Learning:                                                 │
    │    • CNN-BiLSTM (trained on degraded signals)                   │
    └─────────────────────────────────────────────────────────────────┘  
    """
    
    st.code(pipeline_text, language='text')
    
    st.markdown("---")
    
    # Quick start
    st.markdown("### 🚀 Quick Start Guide")
    
    st.markdown("""
    **Step 1:** 🔬 **IMU & Markov Learning**
    - Upload  IMU-based Human Activity Recognition Dataset (Activity_Recognition_Data.csv)
    - Select activity (walking/sitting/standing)
    - Run physics-based analysis to learn channel parameters
    - View learned Markov transition matrices
    
    **Step 2:** 📊 **Dataset Generation**
    - Load MIT-BIH ECG record
    - Configure ACO-OFDM parameters
    - Run complete TX → Channel → RX pipeline
    - View BER and all intermediate signals
    
    **Step 3:** 🧠 **Compare Reconstruction Methods**
    - Compare classical methods vs deep learning
    - Analyze reconstruction quality (SNR, correlation)
    - Understand which methods work best
    """)
    
    st.markdown("---")
    
    # Dataset info
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 📁 Required Datasets
        
        **1. MIT-BIH Arrhythmia Database**
        - Source: PhysioNet
        - Records: 100-124, 200-234
        - Sampling: 360 Hz
        - Duration: Variable (we use 3-sec windows)
        
        **2. IMU-based Human Activity Recognition Dataset**
        - Source: https://data.mendeley.com/datasets/fcnmmsn857/3
        - Sensors: 6-axis IMU (acc + gyro)
        - Sampling: 50 Hz
        - Activities: 6 (we use: walking, sitting, standing)
        - File: `Activity_Recognition_Data.csv`
        """)
    
    with col2:
        st.markdown("""
        ### 📈 Expected Results
        
        **Channel Parameters (Learned from IMU):**
        - Walking: Higher jitter (σ≈0.12), more transitions
        - Sitting: Medium stability (σ≈0.08)
        - Standing: Very stable (σ≈0.05), mostly LoS
        
        **BER Performance:**
        - Realistic: ~25% (with noise reduction)
        - Activity-dependent: Walking > Sitting > Standing
        
        **Reconstruction:**
        - Classical methods: Good for specific degradations
        - Deep learning: Excels at learning the complex, nonlinear patterns and temporal dependencies within heart activity that traditional methods miss
        """)
    
    st.markdown("---")
    
    # Citation
    st.markdown("""
    ### 📚 Citation
    
    If you use this work, please cite:
    ```
    Htet@Grace (2026). "Motion-Driven Markov Channel Modeling and Learning-Based Reconstruction for On-Body Optical Wireless ECG Transmission".
    ```
    """)
    
    # Contact
    st.info("""
    **Questions or Feedback?**  
      Please contact: htetag414@gmail.com
    """)


if __name__ == "__main__":
    show_home()
