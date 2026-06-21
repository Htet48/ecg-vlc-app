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
import matplotlib
import hmac
from pathlib import Path

# ── Force dark background for ALL matplotlib figures ──────────────────────
matplotlib.rcParams.update({
    'figure.facecolor':      '#0e1117',
    'axes.facecolor':        '#262730',
    'axes.edgecolor':        '#555555',
    'axes.labelcolor':       '#ffffff',
    'xtick.color':           '#aaaaaa',
    'ytick.color':           '#aaaaaa',
    'text.color':            '#ffffff',
    'grid.color':            '#444444',
    'grid.linestyle':        '--',
    'grid.linewidth':        0.6,
    'lines.linewidth':       1.8,
    'figure.dpi':            150,
    'savefig.dpi':           300,
    'savefig.facecolor':     '#0e1117',
    'savefig.edgecolor':     'none',
    'savefig.bbox':          'tight',
    'font.family':           'sans-serif',
    'font.size':             11,
    'axes.titlesize':        12,
    'axes.labelsize':        11,
    'legend.framealpha':     0.8,
    'legend.edgecolor':      '#555555',
    'legend.facecolor':      '#262730',
})

import matplotlib.pyplot as plt
import functools

def _force_dark_figure(fig):
    """Apply dark background to any matplotlib figure."""
    fig.patch.set_facecolor('#0e1117')
    for ax in fig.get_axes():
        ax.set_facecolor('#262730')
        ax.tick_params(colors='#aaaaaa', which='both', labelcolor='#aaaaaa')
        ax.xaxis.label.set_color('#ffffff')
        ax.yaxis.label.set_color('#ffffff')
        ax.title.set_color('#ffffff')
        for spine in ax.spines.values():
            spine.set_edgecolor('#555555')
        leg = ax.get_legend()
        if leg:
            leg.get_frame().set_facecolor('#262730')
            leg.get_frame().set_edgecolor('#555555')
            for txt in leg.get_texts():
                txt.set_color('#ffffff')
    return fig

# ── Patch plt.subplots so every new figure starts dark ────────────────────
_orig_subplots = plt.subplots
@functools.wraps(_orig_subplots)
def _dark_subplots(*args, **kwargs):
    fig, ax = _orig_subplots(*args, **kwargs)
    fig.patch.set_facecolor('#0e1117')
    if hasattr(ax, '__iter__'):
        for a in ax.flatten():
            a.set_facecolor('#262730')
    else:
        ax.set_facecolor('#262730')
    return fig, ax
plt.subplots = _dark_subplots

# ── Patch plt.figure so every new figure starts dark ──────────────────────
_orig_figure = plt.figure
@functools.wraps(_orig_figure)
def _dark_figure(*args, **kwargs):
    fig = _orig_figure(*args, **kwargs)
    fig.patch.set_facecolor('#0e1117')
    return fig
plt.figure = _dark_figure

# ── Patch st.pyplot to force dark just before rendering ───────────────────
_st_pyplot_orig = st.pyplot
@functools.wraps(_st_pyplot_orig)
def _dark_pyplot(fig=None, **kwargs):
    target = fig if fig is not None else plt.gcf()
    target = _force_dark_figure(target)
    return _st_pyplot_orig(target, **kwargs)
st.pyplot = _dark_pyplot

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# ── Page configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECG-VLC Research - Grace",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# =============================================================================
# AUTHENTICATION SYSTEM
# =============================================================================

def check_password():
    """Returns True if the user has correct credentials."""

    def login_form():
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
                st.markdown('<h1 class="login-title">🫀 ECG-VLC Research Demo</h1>', unsafe_allow_html=True)
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
                st.info("""
                **Author:** Htet@Grace
                **Topic:** Motion-Driven Markov Channel Modeling and Learning-Based Reconstruction for On-Body Optical Wireless ECG Transmission
                **Innovation:** Motion-aware VLC channel with IMU-learned Markov models
                """)
                st.caption("📧 For access credentials, please contact: htetag414@gmail.com")
                st.markdown('</div>', unsafe_allow_html=True)

    def password_entered():
        username = st.session_state["username"]
        password = st.session_state["password"]
        if "passwords" in st.secrets and username in st.secrets["passwords"]:
            if hmac.compare_digest(password, st.secrets["passwords"][username]):
                st.session_state["password_correct"] = True
                st.session_state["logged_in_user"] = username
                if "user_names" in st.secrets and username in st.secrets["user_names"]:
                    st.session_state["user_display_name"] = st.secrets["user_names"][username]
                else:
                    st.session_state["user_display_name"] = username
                del st.session_state["password"]
                del st.session_state["username"]
            else:
                st.session_state["password_correct"] = False
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    login_form()
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.error("❌ Incorrect username or password. Please try again.")
    return False

if not check_password():
    st.stop()

# ── Dark CSS theme — comprehensive override ───────────────────────────────
st.markdown("""
<style>
    /* Global dark background */
    .stApp, .main, .main .block-container,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background-color: #0e1117 !important;
        color: #ffffff !important;
    }

    /* Sidebar dark */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {
        background-color: #262730 !important;
        border-right: 1px solid #444444 !important;
    }
    section[data-testid="stSidebar"] * { color: #ffffff !important; }

    /* Code blocks */
    pre, code,
    .stCodeBlock, .stCodeBlock *,
    [data-testid="stCode"], [data-testid="stCode"] *,
    .highlight, .highlight * {
        background-color: #1e2130 !important;
        color: #e0e0e0 !important;
        border: 1px solid #444444 !important;
        border-radius: 6px !important;
    }

    /* Metric values */
    [data-testid="stMetricValue"] {
        color: #4a9eff !important;
        font-size: 1.8rem !important;
    }
    [data-testid="stMetricLabel"] { color: #aaaaaa !important; }

    /* Headings */
    .main-header, h1, h2, h3, h4, h5, h6 { color: #ffffff !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { color: #aaaaaa !important; }
    .stTabs [aria-selected="true"] {
        color: #4a9eff !important;
        border-bottom-color: #4a9eff !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #4a9eff !important;
        color: #ffffff !important;
        border: none !important;
        border-radius: 6px !important;
    }

    /* Tables */
    .dataframe, .dataframe * {
        background-color: #262730 !important;
        color: #ffffff !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #262730 !important;
        color: #ffffff !important;
    }

    /* Selectbox and radio */
    .stSelectbox *, .stRadio * { color: #ffffff !important; }

    /* Dividers */
    hr { border-color: #444444 !important; }

    /* General text */
    p, li, span, label { color: #e0e0e0 !important; }
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

st.sidebar.markdown("# 🫀 ECG-VLC Research Demo")

# Show logged-in user
if "user_display_name" in st.session_state:
    st.sidebar.success(f"✅ Logged in as: **{st.session_state['user_display_name']}**")

# Logout button
if st.sidebar.button("🚪 Logout", use_container_width=True):
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
        "📈 Sensitivity Analysis",
        "🧠 Classical vs Deep Learning"
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

# Ambient light is fixed to 'bright' — the dataset and paper are generated with
# the bright-indoor scenario; ambient variation is covered by the Sensitivity
# Analysis page (low / nominal / severe). Fixing it here keeps the live demo
# numbers consistent with the published results.
ambient = "bright"
st.session_state['ambient'] = ambient
st.sidebar.markdown("**Ambient Light:** `bright` (fixed)")
st.sidebar.caption(
    "Dataset uses the bright-indoor scenario. Ambient variation is analysed "
    "separately on the 📈 Sensitivity Analysis page."
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
**Research:** ECG Signal Reconstruction through VLC

**Author:** Grace
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

elif page == "📈 Sensitivity Analysis":
    from pages import sensitivity_analysis_page
    sensitivity_analysis_page.show_sensitivity_analysis()

elif page == "🧠 Classical vs Deep Learning":
    # Use YOUR actual page
    from pages import deep_learning_page
    deep_learning_page.show_deep_learning_page()

