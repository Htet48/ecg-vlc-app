"""
FINAL THESIS DEMONSTRATION APP
===============================

Complete ECG-VLC Thesis Demo showing:
1. ECG → ACO-OFDM Modulation
2. IMU-Learned Markov Channel Model
3. VLC Channel Effects (C1-C8)
4. ACO-OFDM Demodulation
5. Classical vs Deep Learning Reconstruction

Author: Grace
Date: January 2026
"""

import streamlit as st
import numpy as np
import sys
from pathlib import Path
import hmac

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# Page configuration (MUST be first Streamlit command)
st.set_page_config(
    page_title="ECG-VLC Thesis Demo - Grace",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# AUTHENTICATION SYSTEM
# =============================================================================

def check_password():
    """
    Returns True if the user has correct credentials.
    Supports multiple users with different access levels.
    """
    
    def login_form():
        """Display login form"""
        st.markdown("""
        <style>
            .login-container {
                max-width: 500px;
                margin: 100px auto;
                padding: 2rem;
                background-color: #262730;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
            }
            .login-title {
                color: #4a9eff;
                font-size: 2rem;
                text-align: center;
                margin-bottom: 1rem;
            }
            .login-subtitle {
                color: #ffffff;
                text-align: center;
                margin-bottom: 2rem;
            }
        </style>
        """, unsafe_allow_html=True)
        
        with st.container():
            col1, col2, col3 = st.columns([1, 2, 1])
            
            with col2:
                st.markdown('<div class="login-container">', unsafe_allow_html=True)
                
                st.markdown('<h1 class="login-title">🫀 ECG-VLC Thesis Demo</h1>', unsafe_allow_html=True)
                st.markdown('<p class="login-subtitle">ECG Signal Reconstruction through VLC</p>', unsafe_allow_html=True)
                
                with st.form("credentials_form"):
                    st.text_input("👤 Username", key="username", placeholder="Enter your username")
                    st.text_input("🔒 Password", type="password", key="password", placeholder="Enter your password")
                    
                    col_a, col_b, col_c = st.columns([1, 2, 1])
                    with col_b:
                        submitted = st.form_submit_button("🚀 Login", use_container_width=True)
                    
                    if submitted:
                        password_entered()
                
                st.markdown("---")
                
                # Information section
                st.info("""
                
                **Author:** Htet@Grace  
                **Topic:** Motion-Driven Markov Channel Modeling and Learning-Based Reconstruction for On-Body Optical Wireless ECG Transmission  
                **Innovation:** Motion-aware VLC channel with IMU-learned Markov models
                """)
                
                st.caption("📧 For access credentials, please contact: htetag414@gmail.com")
                
                st.markdown('</div>', unsafe_allow_html=True)

    def password_entered():
        """Check if username and password combination is correct"""
        username = st.session_state["username"]
        password = st.session_state["password"]
        
        # Check if username exists in secrets
        if "passwords" in st.secrets and username in st.secrets["passwords"]:
            # Compare password securely
            if hmac.compare_digest(password, st.secrets["passwords"][username]):
                st.session_state["password_correct"] = True
                st.session_state["logged_in_user"] = username
                
                # Get user's full name if available
                if "user_names" in st.secrets and username in st.secrets["user_names"]:
                    st.session_state["user_display_name"] = st.secrets["user_names"][username]
                else:
                    st.session_state["user_display_name"] = username
                
                # Clear password from session
                del st.session_state["password"]
                del st.session_state["username"]
            else:
                st.session_state["password_correct"] = False
        else:
            st.session_state["password_correct"] = False

    # Check if already authenticated
    if st.session_state.get("password_correct", False):
        return True

    # Show login form
    login_form()
    
    # Show error if credentials were incorrect
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.error("❌ Incorrect username or password. Please try again.")
    
    return False

# =============================================================================
# CHECK AUTHENTICATION
# =============================================================================

if not check_password():
    st.stop()  # Don't run anything else if not authenticated

# =============================================================================
# AUTHENTICATED SECTION - Only runs if login successful
# =============================================================================

# Custom CSS matching dark theme
st.markdown("""
<style>
    /* Dark theme */
    .stApp {
        background-color: #0e1117;
    }
    
    .main-header {
        font-size: 2rem;
        font-weight: bold;
        color: #ffffff;
        padding: 1rem;
        margin-bottom: 1rem;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background-color: #262730;
    }
    
    section[data-testid="stSidebar"] .element-container {
        color: #ffffff;
    }
    
    /* Info boxes */
    .info-box {
        background-color: #1e3a5f;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #4a9eff;
        margin: 1rem 0;
        color: #ffffff;
    }
    
    /* Metrics */
    [data-testid="stMetricValue"] {
        font-size: 1.8rem;
        color: #4a9eff;
    }
    
    /* User badge */
    .user-badge {
        background-color: #1e3a5f;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        color: #4a9eff;
        font-weight: bold;
        display: inline-block;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'activity' not in st.session_state:
    st.session_state['activity'] = 'walking'
if 'ambient' not in st.session_state:
    st.session_state['ambient'] = 'bright'

# =============================================================================
# SIDEBAR
# =============================================================================

st.sidebar.markdown("# 🫀 ECG-VLC System Interactive Demo")

# Show logged-in user
if "user_display_name" in st.session_state:
    st.sidebar.success(f"✅ Logged in as: **{st.session_state['user_display_name']}**")

# Logout button
if st.sidebar.button("🚪 Logout", use_container_width=True):
    # Clear all session state
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

st.sidebar.markdown("---")

# Navigation
page = st.sidebar.radio(
    "Navigation:",
    [
        "🏠 Home",
        "📊 Dataset Generation",
        "🔬 IMU & Markov Learning",
        "🧠 Classical vs Deep Learning",
        "🔭 Sensitivity Analysis"
    ]
)

st.sidebar.markdown("---")
st.sidebar.markdown("### ⚙️ Global Settings")

# Global parameters
activity = st.sidebar.selectbox(
    "Activity:",
    ["walking", "sitting", "standing"],
    help="Body motion activity affects channel"
)
st.session_state['activity'] = activity

ambient = st.sidebar.selectbox(
    "Ambient Light:",
    ["bright", "dark"]
)
st.session_state['ambient'] = ambient

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Research:** Motion-Driven Markov Channel Modeling and Learning-Based Reconstruction for On-Body Optical Wireless ECG Transmission

**Author:** Htet@Grace  
**Year:** 2026

**Key Innovation:**
Motion-aware VLC channel using IMU-learned Markov models

**Noise Model:** Physical photodetector characteristics  
(Thermal + shot noise based on ambient lighting)
""")

# =============================================================================
# PAGE ROUTING
# =============================================================================

if page == "🏠 Home":
    from pages import home_page
    home_page.show_home()

elif page == "📊 Dataset Generation":
    from pages import dataset_generation_complete
    dataset_generation_complete.show_dataset_generation()

elif page == "🔬 IMU & Markov Learning":
    from pages import imu_learning_page
    imu_learning_page.show_imu_learning_page()
    
elif page == "🧠 Classical vs Deep Learning":
    # Use YOUR actual page
    from pages import deep_learning_page
    deep_learning_page.show_deep_learning_page()

elif page == "🔭 Sensitivity Analysis":
    from pages import sensitivity_analysis_page
    sensitivity_analysis_page.show_sensitivity_analysis()
