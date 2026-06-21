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
from pathlib import Path

# ── Force white background for ALL matplotlib figures ─────────────────────
matplotlib.rcParams.update({
    'figure.facecolor':      'white',
    'axes.facecolor':        'white',
    'axes.edgecolor':        '#333333',
    'axes.labelcolor':       '#111111',
    'xtick.color':           '#111111',
    'ytick.color':           '#111111',
    'text.color':            '#111111',
    'grid.color':            '#DDDDDD',
    'grid.linestyle':        '--',
    'grid.linewidth':        0.6,
    'lines.linewidth':       1.8,
    'figure.dpi':            150,
    'savefig.dpi':           300,
    'savefig.facecolor':     'white',
    'savefig.edgecolor':     'none',
    'savefig.bbox':          'tight',
    'font.family':           'sans-serif',
    'font.size':             11,
    'axes.titlesize':        12,
    'axes.labelsize':        11,
    'legend.framealpha':     0.9,
    'legend.edgecolor':      '#CCCCCC',
})

# ── Deep intercept: block dark_background style + patch ALL figure creation ─
import matplotlib.pyplot as plt
import matplotlib.figure
import functools

def _force_white_figure(fig):
    """Strip dark background from any matplotlib figure — all axes."""
    fig.patch.set_facecolor('white')
    for ax in fig.get_axes():
        ax.set_facecolor('white')
        ax.tick_params(colors='#111111', which='both', labelcolor='#111111')
        ax.xaxis.label.set_color('#111111')
        ax.yaxis.label.set_color('#111111')
        ax.title.set_color('#111111')
        for spine in ax.spines.values():
            spine.set_edgecolor('#333333')
        for line in ax.get_lines():
            lc = line.get_color()
            if str(lc).lower() in ('white', '#ffffff', '#fff', 'w',
                                   (1.0, 1.0, 1.0, 1.0)):
                line.set_color('#111111')
        leg = ax.get_legend()
        if leg:
            leg.get_frame().set_facecolor('white')
            leg.get_frame().set_edgecolor('#CCCCCC')
            for txt in leg.get_texts():
                txt.set_color('#111111')
    return fig

# ── 1. Block plt.style.use('dark_background') ─────────────────────────────
_orig_style_use = plt.style.use
def _safe_style_use(style, *args, **kwargs):
    """Ignore any dark style requests."""
    if isinstance(style, str) and 'dark' in style.lower():
        return
    return _orig_style_use(style, *args, **kwargs)
plt.style.use = _safe_style_use

# ── 2. Patch plt.subplots so every new figure starts white ────────────────
_orig_subplots = plt.subplots
@functools.wraps(_orig_subplots)
def _white_subplots(*args, **kwargs):
    # Remove any dark facecolor kwarg
    if 'facecolor' in kwargs:
        kwargs.pop('facecolor')
    fig, ax = _orig_subplots(*args, **kwargs)
    fig.patch.set_facecolor('white')
    if hasattr(ax, '__iter__'):
        for a in ax.flatten():
            a.set_facecolor('white')
    else:
        ax.set_facecolor('white')
    return fig, ax
plt.subplots = _white_subplots

# ── 3. Patch plt.figure so every new figure starts white ──────────────────
_orig_figure = plt.figure
@functools.wraps(_orig_figure)
def _white_figure(*args, **kwargs):
    if 'facecolor' in kwargs:
        kwargs.pop('facecolor')
    fig = _orig_figure(*args, **kwargs)
    fig.patch.set_facecolor('white')
    return fig
plt.figure = _white_figure

# ── 4. Patch st.pyplot to force white just before rendering ───────────────
_st_pyplot_orig = st.pyplot
@functools.wraps(_st_pyplot_orig)
def _white_pyplot(fig=None, **kwargs):
    target = fig if fig is not None else plt.gcf()
    target = _force_white_figure(target)
    return _st_pyplot_orig(target, **kwargs)
st.pyplot = _white_pyplot

# Add project root to path
sys.path.append(str(Path(__file__).parent))

# ── Page configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="ECG-VLC Thesis Demo - Grace",
    page_icon="🫀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Light/white CSS theme — comprehensive override ────────────────────────
st.markdown("""
<style>
    /* Global white background */
    .stApp, .main, .main .block-container,
    [data-testid="stAppViewContainer"],
    [data-testid="stHeader"] {
        background-color: #FFFFFF !important;
        color: #1A1A1A !important;
    }

    /* Sidebar light grey */
    section[data-testid="stSidebar"],
    section[data-testid="stSidebar"] > div {
        background-color: #F5F7FA !important;
        border-right: 1px solid #E0E0E0 !important;
    }
    section[data-testid="stSidebar"] * { color: #1A1A1A !important; }

    /* Code blocks */
    pre, code,
    .stCodeBlock, .stCodeBlock *,
    [data-testid="stCode"], [data-testid="stCode"] *,
    .highlight, .highlight * {
        background-color: #F8F9FA !important;
        color: #1A1A1A !important;
        border: 1px solid #E0E0E0 !important;
        border-radius: 6px !important;
    }

    /* Metric values — IEEE blue */
    [data-testid="stMetricValue"] {
        color: #2E75B6 !important;
        font-size: 1.8rem !important;
    }
    [data-testid="stMetricLabel"] { color: #444444 !important; }

    /* Headings */
    .main-header, h1, h2, h3, h4, h5, h6 { color: #1A1A2E !important; }

    /* Tabs */
    .stTabs [data-baseweb="tab"] { color: #444444 !important; }
    .stTabs [aria-selected="true"] {
        color: #2E75B6 !important;
        border-bottom-color: #2E75B6 !important;
    }

    /* Buttons */
    .stButton > button {
        background-color: #2E75B6 !important;
        color: #FFFFFF !important;
        border: none !important;
        border-radius: 6px !important;
    }

    /* Tables */
    .dataframe, .dataframe * {
        background-color: #FFFFFF !important;
        color: #1A1A1A !important;
    }

    /* Expanders */
    .streamlit-expanderHeader {
        background-color: #F5F7FA !important;
        color: #1A1A2E !important;
    }

    /* Selectbox and radio */
    .stSelectbox *, .stRadio * { color: #1A1A1A !important; }

    /* Dividers */
    hr { border-color: #E0E0E0 !important; }
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

st.sidebar.markdown("# 🫀 ECG-VLC Thesis Demo")
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
**Thesis:** ECG Signal Reconstruction through VLC

**Student:** Grace  
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

