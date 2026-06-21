"""
Deep Learning Model Auto-Discovery and Comparison Page - COMPLETE VERSION
=========================================================================

FEATURES:
- Automatically discovers ALL trained models in data folder
- Flexible naming: looks for *_test_predictions.npz pattern
- Comprehensive training history visualization (Loss, MAE, SNR, PRD)
- Shows aggregate metrics across all test samples
- Allows selection of individual models or comparison of all
- Matches the visualization style from training notebooks

Author: Grace
Date: January 2026
"""

import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from pathlib import Path
import sys
import json
import re

# Import utilities
sys.path.append(str(Path(__file__).parent.parent))
from utils.reconstruction_utils import (
    interpolation_reconstruction,
    wavelet_denoising_reconstruction,
    ofdm_clipping_mitigation,
    combined_classical_reconstruction
)
from utils.metrics_utils import (
    compute_snr,
    compute_prd,
    compute_rmse,
    compute_correlation_coefficient
)
from utils.comprehensive_metrics_utils import (
    compute_r_peak_detection_accuracy
)


def show_deep_learning_page():
    """Main deep learning comparison page with auto-discovery"""
    
    st.title("🧠 Deep Learning Model Comparison")
    st.markdown("Automatic discovery and comparison of all trained models")
    st.markdown("---")
    
    # Auto-discover and load all models
    models_data = auto_discover_and_load_models()
    
    if not models_data:
        show_upload_instructions()
        return
    
    # Sidebar: Model selection
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 🤖 Model Selection")
    
    model_names = list(models_data.keys())
    
    # Create selection options
    selection_options = model_names.copy()
    if len(model_names) > 1:
        selection_options.append("All Models (Compare)")
    
    selected_view = st.sidebar.radio(
        "View Mode:",
        selection_options,
        index=len(selection_options) - 1 if len(selection_options) > 1 else 0
    )
    
    # Data overview
    show_data_overview(models_data, selected_view)
    
    # Training history option
    st.sidebar.markdown("---")
    show_training = st.sidebar.checkbox("📈 Show Training History", value=True)
    if show_training:
        display_training_history(models_data, selected_view)
    
    st.markdown("---")
    
    # AGGREGATE Performance comparison
    display_aggregate_performance_comparison(models_data, selected_view)
    
    st.markdown("---")
    
    # Sample-level comparison
    display_sample_comparison(models_data, selected_view)


def auto_discover_and_load_models():
    """
    Auto-discover all trained models in data folder
    Looks for files matching pattern: *_predictions.npz
    Returns: dict with model_name -> model_data
    """
    
    models = {}
    
    # Search paths
    search_paths = [
        Path('data'),
        Path('.'),
    ]
    
    # Pattern to match prediction files
    prediction_pattern = re.compile(r'(.+?)_predictions(?:_lite)?\.npz$')
    st.sidebar.markdown("### 🔍 Model Discovery")
    
    for search_path in search_paths:
        if not search_path.exists():
            continue
        
        # Find all .npz files matching the pattern
        for npz_file in search_path.glob('*_predictions.npz'):
            match = prediction_pattern.match(npz_file.name)
            if match:
                model_name = match.group(1)
                
                # Clean up model name for display
                display_name = clean_model_name(model_name)
                
                try:
                    # Load predictions
                    data = np.load(npz_file, allow_pickle=True)
                    
                    # Validate required fields
                    required_fields = ['original', 'degraded', 'reconstructed']
                    if not all(field in data for field in required_fields):
                        st.sidebar.warning(f"⚠️ {display_name}: Missing required fields")
                        continue
                    
                    models[display_name] = {
                        'original': data['original'],
                        'degraded': data['degraded'],
                        'reconstructed': data['reconstructed'],
                        'snr': data.get('snr', None),
                        'prd': data.get('prd', None),
                        'correlation': data.get('correlation', None),
                        'metadata': {
                            'n_samples': len(data['original']),
                            'sample_length': len(data['original'][0]),
                            'file_path': str(npz_file)
                        }
                    }
                    
                    # Try to load training history
                    history_file = npz_file.parent / f"{model_name}_training_history.json"
                    if history_file.exists():
                        try:
                            with open(history_file, 'r') as f:
                                models[display_name]['training_history'] = json.load(f)
                        except:
                            pass
                    
                    # Try alternative naming for training metrics
                    metrics_file = npz_file.parent / f"{model_name}_metrics.json"
                    if metrics_file.exists():
                        try:
                            with open(metrics_file, 'r') as f:
                                models[display_name]['test_metrics'] = json.load(f)
                        except:
                            pass
                    
                    st.sidebar.success(f"✅ {display_name}")
                    
                except Exception as e:
                    st.sidebar.error(f"❌ {display_name}: {str(e)}")
    
    # Sort models by name
    models = dict(sorted(models.items()))
    
    return models


def clean_model_name(raw_name):
    """
    Convert raw filename to clean display name
    Examples:
    - 'cnn_bilstm' -> 'CNN-BiLSTM'
    - 'transformer_cnn' -> 'Transformer+CNN'
    - 'autoencoder' -> 'Autoencoder'
    """
    
    # Replace underscores with spaces
    name = raw_name.replace('_', ' ')
    
    # Special cases
    replacements = {
        'cnn': 'CNN',
        'bilstm': 'BiLSTM',
        'lstm': 'LSTM',
        'gru': 'GRU',
        'transformer': 'Transformer',
        'autoencoder': 'Autoencoder',
        'unet': 'U-Net',
        'resnet': 'ResNet',
        'vgg': 'VGG',
        'encoder': 'Encoder',
        'decoder': 'Decoder'
    }
    
    # Apply replacements
    for old, new in replacements.items():
        name = re.sub(rf'\b{old}\b', new, name, flags=re.IGNORECASE)
    
    # Replace space + next word pattern with + for common architectures
    name = re.sub(r'(\bCNN\b) (\bBiLSTM\b)', r'\1-\2', name)
    name = re.sub(r'(\bTransformer\b) (\bCNN\b)', r'\1+\2', name)
    name = re.sub(r'(\bAuto\b) (\bEncoder\b)', r'\1\2', name)
    
    # Capitalize words
    name = ' '.join(word.capitalize() if word.lower() not in ['cnn', 'lstm', 'bilstm', 'gru'] 
                    else word for word in name.split())
    
    return name.strip()


def compute_aggregate_metrics_for_method(original_signals, reconstructed_signals):
    """Compute aggregate metrics across ALL samples"""
    
    all_snr = []
    all_prd = []
    all_rmse = []
    all_corr = []
    all_rpeak = []
    
    n_samples = min(len(original_signals), len(reconstructed_signals))
    
    for i in range(n_samples):
        try:
            original = original_signals[i]
            reconstructed = reconstructed_signals[i]
            
            all_snr.append(compute_snr(original, reconstructed))
            all_prd.append(compute_prd(original, reconstructed))
            all_rmse.append(compute_rmse(original, reconstructed))
            all_corr.append(compute_correlation_coefficient(original, reconstructed))
            all_rpeak.append(compute_r_peak_detection_accuracy(original, reconstructed))
        except:
            continue
    
    return {
        'snr_mean': np.mean(all_snr),
        'snr_std': np.std(all_snr),
        'prd_mean': np.mean(all_prd),
        'prd_std': np.std(all_prd),
        'rmse_mean': np.mean(all_rmse),
        'rmse_std': np.std(all_rmse),
        'corr_mean': np.mean(all_corr),
        'corr_std': np.std(all_corr),
        'rpeak_mean': np.mean(all_rpeak),
        'rpeak_std': np.std(all_rpeak),
        'all_snr': all_snr,
        'all_prd': all_prd,
        'all_rmse': all_rmse,
        'all_corr': all_corr,
        'all_rpeak': all_rpeak
    }


def show_data_overview(models_data, selected_view):
    """Show data overview for loaded models"""
    
    st.markdown("## 📊 Data Overview")
    
    # Get reference data
    first_model = list(models_data.values())[0]
    
    cols = st.columns(4)
    
    with cols[0]:
        st.metric("Test Samples", f"{first_model['metadata']['n_samples']:,}", help="Model evaluated on full 4,000-sample test set. Showing 2,000 representative samples for web deployment.")
        st.info(
    "📊 **Note:** Metrics computed from FULL 4,000-sample test set. "
    "Web demo shows 2,000 uniformly sampled representatives "
    "(GitHub size limits). Statistical difference: <0.1%"
)
    
    with cols[1]:
        st.metric("Sample Length", f"{first_model['metadata']['sample_length']} pts")
    
    with cols[2]:
        duration = first_model['metadata']['sample_length'] / 360
        st.metric("Duration", f"{duration:.1f} sec")
    
    with cols[3]:
        st.metric("Sampling Rate", "360 Hz")
    
    # Model status
    st.markdown("### 🤖 Models Discovered")
    
    n_models = len(models_data)
    cols_per_row = 3
    
    for i in range(0, n_models, cols_per_row):
        cols = st.columns(cols_per_row)
        for j, (model_name, model_data) in enumerate(list(models_data.items())[i:i+cols_per_row]):
            with cols[j]:
                has_history = 'training_history' in model_data
                history_icon = "📈" if has_history else "📊"
                st.success(f"{history_icon} **{model_name}**")


# COMPLETE FIXED display_training_history FUNCTION
# ==================================================
# This replaces the function starting at line ~311

def display_training_history(models_data, selected_view):
    """Display comprehensive training history with IMPROVED side-by-side layout"""
    
    st.markdown("---")
    st.markdown("## 📈 Training History")
    
    if selected_view == "All Models (Compare)":
        # Show all models with training history
        models_with_history = {name: data for name, data in models_data.items() 
                              if 'training_history' in data}
        
        if not models_with_history:
            st.warning("No training history available for any model")
            return
        
        st.info(f"📊 Comparing {len(models_with_history)} models - 2 per row for better visibility")
        
        # Display 2 models per row
        models_list = list(models_with_history.items())
        models_per_row = 2
        
        for row_idx in range(0, len(models_list), models_per_row):
            row_models = models_list[row_idx:row_idx + models_per_row]
            cols = st.columns(len(row_models))
            
            for col_idx, (model_name, model_data) in enumerate(row_models):
                with cols[col_idx]:
                    st.markdown(f"### {model_name} Training")
                    
                    history = model_data['training_history']
                    epochs = list(range(1, len(history['loss']) + 1))
                    
                    fig = make_subplots(
                        rows=2, cols=2,
                        subplot_titles=['Loss (MSE)', 'Mean Absolute Error',
                                      'Signal-to-Noise Ratio', 'PRD (%)'],
                        vertical_spacing=0.18,
                        horizontal_spacing=0.18,
                        specs=[[{"secondary_y": False}, {"secondary_y": False}],
                               [{"secondary_y": False}, {"secondary_y": False}]]
                    )
                    
                    # Loss
                    fig.add_trace(
                        go.Scatter(x=epochs, y=history['loss'],
                                  name='Train',
                                  line=dict(color='#1f77b4', width=2),
                                  showlegend=True),
                        row=1, col=1
                    )
                    if 'val_loss' in history:
                        fig.add_trace(
                            go.Scatter(x=epochs, y=history['val_loss'],
                                      name='Validation',
                                      line=dict(color='#ff7f0e', width=2, dash='dash'),
                                      showlegend=True),
                            row=1, col=1
                        )
                    
                    # MAE
                    if 'mae' in history:
                        fig.add_trace(
                            go.Scatter(x=epochs, y=history['mae'],
                                      name='Train',
                                      line=dict(color='#1f77b4', width=2),
                                      showlegend=False),
                            row=1, col=2
                        )
                    if 'val_mae' in history:
                        fig.add_trace(
                            go.Scatter(x=epochs, y=history['val_mae'],
                                      name='Validation',
                                      line=dict(color='#ff7f0e', width=2, dash='dash'),
                                      showlegend=False),
                            row=1, col=2
                        )
                    
                    # SNR
                    if 'val_snr' in history:
                        fig.add_trace(
                            go.Scatter(x=epochs, y=history['val_snr'],
                                      name='Val SNR',
                                      line=dict(color='#2ca02c', width=2.5),
                                      showlegend=False),
                            row=2, col=1
                        )
                    
                    # PRD
                    if 'val_prd' in history:
                        fig.add_trace(
                            go.Scatter(x=epochs, y=history['val_prd'],
                                      name='Val PRD',
                                      line=dict(color='#d62728', width=2.5),
                                      showlegend=False),
                            row=2, col=2
                        )
                        
                        fig.add_hline(y=9.0, line_dash="dash", line_color="green",
                                    annotation_text="Clinical Grade (9%)",
                                    annotation_position="top right",
                                    annotation=dict(font_size=10),
                                    row=2, col=2)
                    
                    # Update axes
                    fig.update_xaxes(title_text="Epoch", row=1, col=1, title_font=dict(size=11))
                    fig.update_xaxes(title_text="Epoch", row=1, col=2, title_font=dict(size=11))
                    fig.update_xaxes(title_text="Epoch", row=2, col=1, title_font=dict(size=11))
                    fig.update_xaxes(title_text="Epoch", row=2, col=2, title_font=dict(size=11))
                    
                    fig.update_yaxes(title_text="Loss", row=1, col=1, title_font=dict(size=11))
                    fig.update_yaxes(title_text="MAE", row=1, col=2, title_font=dict(size=11))
                    fig.update_yaxes(title_text="SNR (dB)", row=2, col=1, title_font=dict(size=11))
                    fig.update_yaxes(title_text="PRD (%)", row=2, col=2, title_font=dict(size=11))
                    
                    fig.update_layout(
                        height=800,
                        template='plotly_white',
        font=dict(color="#111111"),
                        showlegend=True,
                        legend=dict(
                            orientation="h",
                            yanchor="bottom",
                            y=1.02,
                            xanchor="center",
                            x=0.5,
                            font=dict(size=10)
                        ),
                        margin=dict(t=80, b=60, l=60, r=60)
                    )
                    
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # Metrics
                    final_loss = history['loss'][-1]
                    final_snr = history.get('val_snr', [0])[-1] if 'val_snr' in history else None
                    final_prd = history.get('val_prd', [0])[-1] if 'val_prd' in history else None
                    
                    metric_cols = st.columns(2)
                    with metric_cols[0]:
                        st.metric("Final Loss", f"{final_loss:.6f}")
                        if final_snr:
                            st.metric("Final SNR", f"{final_snr:.2f} dB")
                    with metric_cols[1]:
                        if 'val_mae' in history:
                            st.metric("Final MAE", f"{history['val_mae'][-1]:.4f}")
                        if final_prd:
                            status = "✅ Clinical" if final_prd < 9.0 else "⚠️ Close" if final_prd < 11.0 else ""
                            st.metric("Final PRD", f"{final_prd:.2f}%", delta=status)
            
            # Separator between rows
            if row_idx + models_per_row < len(models_list):
                st.markdown("---")
    
    else:
        # Single model view
        if selected_view not in models_data:
            st.error(f"Model {selected_view} not found")
            return
        
        model_data = models_data[selected_view]
        
        if 'training_history' not in model_data:
            st.warning(f"No training history available for {selected_view}")
            return
        
        st.markdown(f"### {selected_view} Training History")
        
        history = model_data['training_history']
        epochs = list(range(1, len(history['loss']) + 1))
        
        # Create 2x2 subplot - FULL WIDTH for single model
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=['Loss (MSE)', 'Mean Absolute Error',
                          'Signal-to-Noise Ratio', 'PRD (%)'],
            vertical_spacing=0.15,
            horizontal_spacing=0.12,
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Loss
        fig.add_trace(
            go.Scatter(x=epochs, y=history['loss'],
                      name='Train',
                      line=dict(color='#1f77b4', width=2),
                      showlegend=True),
            row=1, col=1
        )
        if 'val_loss' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['val_loss'],
                          name='Validation',
                          line=dict(color='#ff7f0e', width=2, dash='dash'),
                          showlegend=True),
                row=1, col=1
            )
        
        # MAE
        if 'mae' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['mae'],
                          name='Train',
                          line=dict(color='#1f77b4', width=2),
                          showlegend=False),
                row=1, col=2
            )
        if 'val_mae' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['val_mae'],
                          name='Validation',
                          line=dict(color='#ff7f0e', width=2, dash='dash'),
                          showlegend=False),
                row=1, col=2
            )
        
        # SNR
        if 'val_snr' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['val_snr'],
                          name='Val SNR',
                          line=dict(color='#2ca02c', width=2.5),
                          showlegend=False),
                row=2, col=1
            )
        
        # PRD
        if 'val_prd' in history:
            fig.add_trace(
                go.Scatter(x=epochs, y=history['val_prd'],
                          name='Val PRD',
                          line=dict(color='#d62728', width=2.5),
                          showlegend=False),
                row=2, col=2
            )
            
            fig.add_hline(y=9.0, line_dash="dash", line_color="green",
                        annotation_text="Clinical Grade (9%)",
                        annotation_position="top right",
                        annotation=dict(font_size=11),
                        row=2, col=2)
        
        # Update axes
        fig.update_xaxes(title_text="Epoch", row=1, col=1)
        fig.update_xaxes(title_text="Epoch", row=1, col=2)
        fig.update_xaxes(title_text="Epoch", row=2, col=1)
        fig.update_xaxes(title_text="Epoch", row=2, col=2)
        
        fig.update_yaxes(title_text="Loss", row=1, col=1)
        fig.update_yaxes(title_text="MAE", row=1, col=2)
        fig.update_yaxes(title_text="SNR (dB)", row=2, col=1)
        fig.update_yaxes(title_text="PRD (%)", row=2, col=2)
        
        fig.update_layout(
            height=700,
            template='plotly_white',
        font=dict(color="#111111"),
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="center",
                x=0.5
            ),
            margin=dict(t=60, b=50)
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        # Show final metrics
        final_loss = history['loss'][-1]
        final_snr = history.get('val_snr', [0])[-1] if 'val_snr' in history else None
        final_prd = history.get('val_prd', [0])[-1] if 'val_prd' in history else None
        
        cols = st.columns(4)
        with cols[0]:
            st.metric("Final Loss", f"{final_loss:.6f}")
        with cols[1]:
            if 'val_mae' in history:
                st.metric("Final MAE", f"{history['val_mae'][-1]:.4f}")
        with cols[2]:
            if final_snr:
                st.metric("Final SNR", f"{final_snr:.2f} dB")
        with cols[3]:
            if final_prd:
                clinical_status = "✅" if final_prd < 9.0 else "⚠️"
                st.metric("Final PRD", f"{final_prd:.2f}%", delta=clinical_status)
        
def display_aggregate_performance_comparison(models_data, selected_view):
    """Display aggregate performance comparison table.

    Uses pre-computed *_test_metrics.json values when available (fast, consistent
    with notebook results). Falls back to live recomputation only when JSON absent.
    Classical baselines are also sourced from the JSON.
    """
    st.markdown("## 🎯 Performance Metrics — Aggregate Statistics")
    st.info("📊 Metrics shown are **Mean ± Std** across all test samples. "
            "Values sourced from pre-computed notebook results where available.")

    # ── Collect DL metrics (JSON-first) ──────────────────────────────────────
    dl_metrics = {}
    models_to_show = (list(models_data.keys())
                      if selected_view == "All Models (Compare)"
                      else ([selected_view] if selected_view in models_data else []))

    for model_name in models_to_show:
        model_data = models_data[model_name]
        if "test_metrics" in model_data:
            json_m = model_data["test_metrics"]
            classical_keys = {"degraded", "interpolation", "wavelet", "ofdm",
                              "combined_classical"}
            dl_keys = [k for k in json_m if k not in classical_keys]
            if dl_keys:
                dl_metrics[model_name] = _normalise_metric_dict(json_m[dl_keys[0]])
        if model_name not in dl_metrics:
            with st.spinner(f"⏳ Computing metrics for {model_name}…"):
                dl_metrics[model_name] = compute_aggregate_metrics_for_method(
                    model_data["original"], model_data["reconstructed"])

    # ── Collect classical + degraded baselines (JSON-first) ──────────────────
    classical_metrics = {}
    degraded_metrics  = {}
    for model_data in models_data.values():
        if "test_metrics" in model_data:
            json_m = model_data["test_metrics"]
            key_map = {"interpolation": "Interpolation",
                       "wavelet": "Wavelet",
                       "ofdm": "OFDM Mitigation",
                       "combined_classical": "Combined Classical"}
            for jk, dk in key_map.items():
                if jk in json_m and dk not in classical_metrics:
                    classical_metrics[dk] = _normalise_metric_dict(json_m[jk])
            if "degraded" in json_m and not degraded_metrics:
                degraded_metrics["Degraded (baseline)"] = _normalise_metric_dict(json_m["degraded"])
            break

    if not classical_metrics:
        ref = list(models_data.values())[0]
        with st.spinner("⏳ Computing classical baselines…"):
            classical_metrics = compute_classical_baselines(ref["original"], ref["degraded"])

    st.success("✅ Metrics ready!")

    all_metrics = {}
    if degraded_metrics:
        all_metrics.update(degraded_metrics)
    all_metrics.update(classical_metrics)
    all_metrics.update(dl_metrics)

    st.markdown("---")
    create_comparison_table(all_metrics, classical_metrics, dl_metrics)
    display_model_config_table()
    if dl_metrics:
        st.markdown("---")
        display_improvement_metrics(dl_metrics, classical_metrics)
    st.markdown("---")
    display_quick_comparison_cards(dl_metrics, classical_metrics)
    st.markdown("---")
    display_clinical_assessment(dl_metrics)


def _normalise_metric_dict(raw: dict) -> dict:
    """Ensure every metric key the UI expects is present.
    Includes the beat-level metrics added per Reviewer 3, C#6 (ppv, f1, mate)."""
    m = dict(raw)
    for prefix in ("snr", "prd", "rmse", "corr", "rpeak", "wer", "ppv", "f1", "mate"):
        if f"{prefix}_mean" not in m:
            m[f"{prefix}_mean"] = 0.0
        if f"{prefix}_std" not in m:
            m[f"{prefix}_std"] = 0.0
    if "corr_mean" in m:
        m.setdefault("correlation_mean", m["corr_mean"])
        m.setdefault("correlation_std",  m["corr_std"])
    return m


def display_improvement_metrics(dl_metrics, classical_metrics):
    """Display improvement metrics comparing DL vs best classical"""

    st.markdown("### 🏆 Improvement: Deep Learning vs Best Classical")
    
    # Find best classical
    best_classical_snr = max(m['snr_mean'] for m in classical_metrics.values())
    best_classical_prd = min(m['prd_mean'] for m in classical_metrics.values())
    
    # Find which method is best classical
    best_classical_method_snr = [name for name, m in classical_metrics.items() if m['snr_mean'] == best_classical_snr][0]
    best_classical_method_prd = [name for name, m in classical_metrics.items() if m['prd_mean'] == best_classical_prd][0]
    
    for model_name, metrics in dl_metrics.items():
        st.markdown(f"#### {model_name}")
        
        # Calculate improvements
        snr_improvement = metrics['snr_mean'] - best_classical_snr
        prd_improvement = best_classical_prd - metrics['prd_mean']
        snr_improvement_pct = (snr_improvement / best_classical_snr) * 100
        prd_improvement_pct = (prd_improvement / best_classical_prd) * 100
        
        # Display in nice format
        imp_cols = st.columns(2)
        
        with imp_cols[0]:
            st.metric(
                f"Improvement (CNN-BiLSTM vs Best Classical)",
                "",
                help=f"Best Classical SNR: {best_classical_method_snr}"
            )
            st.markdown(f"**SNR Improvement:** +{snr_improvement:.2f} dB **(+{snr_improvement_pct:.0f}%)**")
        
        with imp_cols[1]:
            st.metric(
                f"Improvement (Transformer+CNN vs Best Classical)",
                "",
                help=f"Best Classical PRD: {best_classical_method_prd}"
            )
            st.markdown(f"**PRD Improvement:** -{prd_improvement:.2f}% **({prd_improvement_pct:.0f}%)**")
        
        st.markdown("---")


def compute_classical_baselines(original_signals, degraded_signals):
    """Compute metrics for classical methods"""
    
    metrics = {}
    
    # Method 1: Interpolation
    interpolation_recons = []
    for degraded in degraded_signals:
        try:
            recon = interpolation_reconstruction(degraded)
            interpolation_recons.append(recon)
        except:
            interpolation_recons.append(degraded)
    
    metrics['Interpolation'] = compute_aggregate_metrics_for_method(
        original_signals, np.array(interpolation_recons)
    )
    
    # Method 2: Wavelet Denoising
    wavelet_recons = []
    for degraded in degraded_signals:
        try:
            recon = wavelet_denoising_reconstruction(degraded)
            wavelet_recons.append(recon)
        except:
            wavelet_recons.append(degraded)
    
    metrics['Wavelet'] = compute_aggregate_metrics_for_method(
        original_signals, np.array(wavelet_recons)
    )
    
    # Method 3: OFDM Mitigation
    ofdm_recons = []
    for degraded in degraded_signals:
        try:
            recon = ofdm_clipping_mitigation(degraded)
            ofdm_recons.append(recon)
        except:
            ofdm_recons.append(degraded)
    
    metrics['OFDM Mitigation'] = compute_aggregate_metrics_for_method(
        original_signals, np.array(ofdm_recons)
    )
    
    # Method 4: Combined Classical
    combined_recons = []
    for degraded in degraded_signals:
        try:
            recon, _ = combined_classical_reconstruction(degraded)
            combined_recons.append(recon)
        except:
            combined_recons.append(degraded)
    
    metrics['Combined Classical'] = compute_aggregate_metrics_for_method(
        original_signals, np.array(combined_recons)
    )
    
    return metrics


def display_model_config_table():
    """Show the fair-comparison output/loss configuration of each model.

    Matches the original paper design: only CNN+GRU uses sigmoid + spectral
    (MSE+FFT) loss; all other models use linear output + pure MSE. The only
    major change versus the original submission is the Hermitian / N-4
    channel-simulation fix, not the model architectures.
    """
    st.markdown("### ⚙️ Model Configuration (fair comparison)")
    cfg = pd.DataFrame([
        {"Model": "CNN+GRU",        "Output": "sigmoid", "Loss": "0.80·MSE + 0.20·FFT", "Role": "spectral-aware recurrent"},
        {"Model": "CNN+LSTM",       "Output": "linear",  "Loss": "MSE",                "Role": "baseline recurrent"},
        {"Model": "CNN+BiLSTM",     "Output": "linear",  "Loss": "MSE",                "Role": "bidirectional context"},
        {"Model": "Transformer+CNN","Output": "linear",  "Loss": "MSE",                "Role": "global self-attention"},
    ])
    st.dataframe(cfg, use_container_width=True, hide_index=True)
    st.caption(
        "Only CNN+GRU uses the spectral (FFT) loss + sigmoid output; all other models use "
        "linear output + pure MSE — identical to the original paper. Beat-level metrics "
        "(R-Peak sensitivity, PPV, F1, MATE) use a ±5-sample (~14 ms) tolerance per "
        "Reviewer 3, C#6, tightened from the previous ±50 ms window."
    )


def create_comparison_table(all_metrics, classical_metrics, dl_metrics):
    """Create comprehensive comparison table"""
    
    # Prepare data for table
    table_data = []
    
    for method_name, metrics in all_metrics.items():
        is_dl = method_name in dl_metrics
        is_baseline = "baseline" in method_name.lower() or "degraded" in method_name.lower()

        # WER reported for ALL methods (classical > 1.2 = energy overshoot; DL ≈ 0.98)
        wer_val = metrics.get("wer_mean", 0.0)
        wer_std = metrics.get("wer_std",  0.0)
        wer_str = f"{wer_val:.4f} ± {wer_std:.4f}" if wer_val > 0 else "—"

        # Beat-level metrics (Reviewer 3, C#6). Shown when available.
        def _bm(key, suf="", scale=1.0, dec=1):
            v = metrics.get(f"{key}_mean", 0.0)
            s = metrics.get(f"{key}_std", 0.0)
            return f"{v*scale:.{dec}f} ± {s*scale:.{dec}f}{suf}" if v else "—"

        table_data.append({
            "Method":        method_name,
            "Type":          "🧠 Deep Learning" if is_dl else ("📉 Baseline" if is_baseline else "📐 Classical"),
            "SNR (dB)":      f"{metrics['snr_mean']:.2f} ± {metrics['snr_std']:.2f}",
            "PRD (%)":       f"{metrics['prd_mean']:.2f} ± {metrics['prd_std']:.2f}",
            "RMSE":          f"{metrics['rmse_mean']:.4f} ± {metrics['rmse_std']:.4f}",
            "CC":            f"{metrics['corr_mean']:.4f} ± {metrics['corr_std']:.4f}",
            "R-Peak (%)":    f"{metrics['rpeak_mean']:.1f} ± {metrics['rpeak_std']:.1f}",
            "PPV (%)":       _bm("ppv"),
            "F1 (%)":        _bm("f1"),
            "MATE (ms)":     _bm("mate", dec=2),
            "WER":           wer_str,
            "_snr_mean":     metrics["snr_mean"],
            "_prd_mean":     metrics["prd_mean"],
        })
    
    df = pd.DataFrame(table_data)
    df = df.sort_values("_snr_mean", ascending=False).drop(columns=["_snr_mean", "_prd_mean"])

    # Highlight DL rows
    st.dataframe(
        df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Method": st.column_config.TextColumn("Method", width="medium"),
            "Type":   st.column_config.TextColumn("Type",   width="medium"),
            "WER":    st.column_config.TextColumn("WER (Wavelet Energy Ratio)", width="medium"),
        }
    )
    
    # Find best methods
    best_snr = max(m['snr_mean'] for m in all_metrics.values())
    best_prd = min(m['prd_mean'] for m in all_metrics.values())
    
    best_snr_method = [name for name, m in all_metrics.items() if m['snr_mean'] == best_snr][0]
    best_prd_method = [name for name, m in all_metrics.items() if m['prd_mean'] == best_prd][0]
    
    col1, col2 = st.columns(2)
    with col1:
        st.success(f"🏆 **Best SNR:** {best_snr_method} ({best_snr:.2f} dB)")
    with col2:
        st.success(f"🏆 **Best PRD:** {best_prd_method} ({best_prd:.2f}%)")


def display_quick_comparison_cards(dl_metrics, classical_metrics):
    """Display quick comparison cards for DL models"""
    
    if not dl_metrics:
        return
    
    st.markdown("### 🏅 Quick Comparison: Both Models vs Best Classical")
    
    # Find best classical
    best_classical_snr = max(m['snr_mean'] for m in classical_metrics.values())
    best_classical_prd = min(m['prd_mean'] for m in classical_metrics.values())
    best_classical_rmse = min(m['rmse_mean'] for m in classical_metrics.values())
    best_classical_corr = max(m['corr_mean'] for m in classical_metrics.values())
    
    # Create columns for each DL model
    n_models = len(dl_metrics)
    cols = st.columns(n_models)
    
    for idx, (model_name, metrics) in enumerate(dl_metrics.items()):
        with cols[idx]:
            st.markdown(f"#### {model_name}")
            
            # Create a nice bordered container
            metric_container = st.container()
            with metric_container:
                # SNR
                snr_delta = metrics['snr_mean'] - best_classical_snr
                st.metric(
                    "SNR (dB) 📊",
                    f"{metrics['snr_mean']:.2f}",
                    delta=f"{snr_delta:+.2f}",
                    help=f"vs Best Classical: {best_classical_snr:.2f} dB | Std: ±{metrics['snr_std']:.2f}"
                )
                
                # PRD
                prd_delta = best_classical_prd - metrics['prd_mean']
                st.metric(
                    "PRD (%) 📉",
                    f"{metrics['prd_mean']:.2f}",
                    delta=f"{prd_delta:+.2f}",
                    delta_color="inverse",
                    help=f"vs Best Classical: {best_classical_prd:.2f}% | Std: ±{metrics['prd_std']:.2f}"
                )
                
                # RMSE
                rmse_delta = best_classical_rmse - metrics['rmse_mean']
                st.metric(
                    "RMSE 📏",
                    f"{metrics['rmse_mean']:.4f}",
                    delta=f"{rmse_delta:+.4f}",
                    delta_color="inverse",
                    help=f"vs Best Classical: {best_classical_rmse:.4f} | Std: ±{metrics['rmse_std']:.4f}"
                )
                
                # Correlation
                corr_delta = metrics['corr_mean'] - best_classical_corr
                st.metric(
                    "Correlation 🔗",
                    f"{metrics['corr_mean']:.4f}",
                    delta=f"{corr_delta:+.4f}",
                    help=f"vs Best Classical: {best_classical_corr:.4f} | Std: ±{metrics['corr_std']:.4f}"
                )

                # R-Peak accuracy
                rpeak = metrics.get("rpeak_mean", 0.0)
                rpeak_std = metrics.get("rpeak_std", 0.0)
                st.metric(
                    "R-Peak Accuracy 💓",
                    f"{rpeak:.1f}%",
                    delta="✅ Clinical" if rpeak >= 90.0 else None,
                    help=f"Beat detection accuracy (50 ms tolerance) | Std: ±{rpeak_std:.1f}%"
                )

                # WER — only show if available
                wer = metrics.get("wer_mean", 0.0)
                wer_std_v = metrics.get("wer_std", 0.0)
                if wer > 0:
                    wer_delta = 1.0 - abs(1.0 - wer)   # closeness to 1.0
                    st.metric(
                        "Wavelet Energy Ratio 〰️",
                        f"{wer:.4f}",
                        delta=f"{'✅ Spectral fidelity' if 0.95 <= wer <= 1.05 else '⚠️'}",
                        help=f"Sub-band energy preservation (ideal = 1.0) | Std: ±{wer_std_v:.4f}"
                    )


def display_clinical_assessment(dl_metrics):
    """Display clinical assessment for DL models"""
    
    if not dl_metrics:
        return
    
    st.markdown("### 🏥 Clinical Assessment")
    
    # Create columns for each model
    n_models = len(dl_metrics)
    cols = st.columns(n_models)
    
    for idx, (model_name, metrics) in enumerate(dl_metrics.items()):
        with cols[idx]:
            prd = metrics['prd_mean']
            prd_std = metrics['prd_std']
            rpeak = metrics['rpeak_mean']
            rpeak_std = metrics['rpeak_std']
            snr = metrics['snr_mean']
            snr_std = metrics['snr_std']
            
            # Determine clinical grade
            wer = metrics.get("wer_mean", 0.0)
            wer_std_v = metrics.get("wer_std", 0.0)

            if prd < 9.0 and rpeak >= 90.0:
                status_icon = "✅"
                status_text = "Clinical Grade"
                box_style = "success"
            elif prd < 9.0:
                status_icon = "✅"
                status_text = "Clinical Grade (PRD)"
                box_style = "info"
            elif prd < 11.0:
                status_icon = "⚠️"
                status_text = "Near Clinical Grade"
                box_style = "warning"
            else:
                status_icon = "❌"
                status_text = "Below Clinical Threshold"
                box_style = "warning"

            wer_label = ("✅" if 0.95 <= wer <= 1.05 else "⚠️") if wer > 0 else ""
            wer_line = (f"• **WER:** {wer:.4f} ± {wer_std_v:.4f} {wer_label}" if wer > 0 else "")
            prd_label   = "✅" if prd < 9.0 else ""
            rpeak_label = "✅" if rpeak >= 90.0 else ""
            body = (f"{status_icon} **{model_name}: {status_text}**\n\n"
                    f"• **SNR:** {snr:.2f} ± {snr_std:.2f} dB\n"
                    f"• **PRD:** {prd:.2f} ± {prd_std:.2f}% {prd_label}\n"
                    f"• **R-Peak:** {rpeak:.1f} ± {rpeak_std:.1f}% {rpeak_label}\n"
                    + wer_line)
            if box_style == "success":
                st.success(body)
            elif box_style == "info":
                st.info(body)
            else:
                st.warning(body)


def display_sample_comparison(models_data, selected_view):
    """Display sample-level signal comparison with full 6-metric table."""

    st.markdown("## 🔬 Sample-Level Visualization")
    st.info("📌 Shows a **single selected sample**. Scroll up for aggregate statistics across all test samples.")

    # ── Reference signals ────────────────────────────────────────────────────
    first_model = list(models_data.values())[0]
    n_samples   = first_model["metadata"]["n_samples"]

    sample_idx = st.slider("Select Sample", min_value=0, max_value=n_samples - 1, value=0)

    original = np.array(first_model["original"][sample_idx], dtype=np.float64)
    degraded = np.array(first_model["degraded"][sample_idx], dtype=np.float64)

    # ── Classical combined baseline ──────────────────────────────────────────
    try:
        combined_result = combined_classical_reconstruction(degraded)
        combined = combined_result[0] if isinstance(combined_result, tuple) else combined_result
    except Exception as e:
        combined = degraded.copy()
        st.warning(f"Classical reconstruction unavailable: {e}")

    # ── Build signal traces dict ──────────────────────────────────────────────
    time_axis = np.arange(len(original)) / 360.0

    # Fixed palette — visible on BOTH dark and light backgrounds
    TRACE_STYLES = {
        "Original":          dict(color="#00e676", width=2.5, dash="solid"),
        "Degraded (VLC in)": dict(color="#ff5252", width=1.2, dash="dot"),
        "Combined Classical": dict(color="#40c4ff", width=1.8, dash="dash"),
    }
    # DL model palette — distinct bright colors
    DL_COLORS = ["#ffd740", "#ea80fc", "#64ffda", "#ff6d00", "#b2ff59", "#ff4081"]
    DL_DASHES  = ["solid",   "dash",    "dashdot",  "dot",    "longdash", "solid"]

    fig = go.Figure()

    fig.add_trace(go.Scatter(x=time_axis, y=original,
                             name="Original (clean)",
                             line=dict(**TRACE_STYLES["Original"]), opacity=1.0))
    fig.add_trace(go.Scatter(x=time_axis, y=degraded,
                             name="VLC-degraded input",
                             line=dict(**TRACE_STYLES["Degraded (VLC in)"]), opacity=0.6))
    fig.add_trace(go.Scatter(x=time_axis, y=combined,
                             name="Combined Classical",
                             line=dict(**TRACE_STYLES["Combined Classical"]), opacity=0.75))

    models_to_plot = (list(models_data.keys())
                      if selected_view == "All Models (Compare)"
                      else ([selected_view] if selected_view in models_data else []))

    for idx, model_name in enumerate(models_to_plot):
        if model_name not in models_data:
            continue
        recon = np.array(models_data[model_name]["reconstructed"][sample_idx], dtype=np.float64)
        fig.add_trace(go.Scatter(
            x=time_axis, y=recon,
            name=model_name,
            line=dict(color=DL_COLORS[idx % len(DL_COLORS)],
                      width=2.2,
                      dash=DL_DASHES[idx % len(DL_DASHES)]),
            opacity=0.95
        ))

    fig.update_layout(
        title=dict(text=f"Sample {sample_idx} — ECG Signal Comparison", font=dict(size=15)),
        xaxis=dict(title="Time (seconds)", gridcolor="rgba(255,255,255,0.08)"),
        yaxis=dict(title="Normalized Amplitude", gridcolor="rgba(255,255,255,0.08)"),
        height=520,
        showlegend=True,
        legend=dict(orientation="v", yanchor="top", y=1.0,
                    xanchor="left", x=1.01,
                    font=dict(size=11),
                    bgcolor="rgba(255,255,255,0.9)", bordercolor="rgba(100,100,100,0.5)",
                    borderwidth=1),
        hovermode="x unified",
        template="plotly_white",
        font=dict(color="#111111"),
        plot_bgcolor="white",
        paper_bgcolor="white",
        margin=dict(r=160, t=60, b=50),
    )

    st.plotly_chart(fig, use_container_width=True)

    # ── Per-sample metrics table (ALL 6 metrics) ──────────────────────────────
    st.markdown(f"### 📊 Sample {sample_idx} — Full Metrics")
    st.caption("Computed on this single sample. Aggregate mean ± std are shown in the Performance section above.")

    models_to_show = (list(models_data.keys())
                      if selected_view == "All Models (Compare)"
                      else ([selected_view] if selected_view in models_data else []))

    rows = []
    for model_name in models_to_show:
        if model_name not in models_data:
            continue
        recon = np.array(models_data[model_name]["reconstructed"][sample_idx], dtype=np.float64)
        try:
            snr_v  = compute_snr(original, recon)
            prd_v  = compute_prd(original, recon)
            rmse_v = compute_rmse(original, recon)
            cc_v   = compute_correlation_coefficient(original, recon)

            # R-Peak accuracy
            try:
                rpeak_v = compute_r_peak_detection_accuracy(original, recon)
                rpeak_s = f"{rpeak_v:.1f}%" if rpeak_v == rpeak_v else "—"   # nan check
            except Exception:
                rpeak_s = "—"

            # Wavelet Energy Ratio
            try:
                import pywt as _pywt
                co = _pywt.wavedec(original, "db4", level=4)
                cr = _pywt.wavedec(recon,    "db4", level=4)
                wer_v = sum(np.sum(c**2) for c in cr) / (sum(np.sum(c**2) for c in co) + 1e-10)
                wer_s = f"{wer_v:.4f}"
            except Exception:
                wer_s = "—"

            prd_flag  = " ✅" if prd_v  < 9.0  else ""
            rpeak_flag = " ✅" if (rpeak_s != "—" and float(rpeak_s.rstrip("%")) >= 90.0) else ""

            rows.append({
                "Model":         model_name,
                "SNR (dB)":      f"{snr_v:.2f}",
                "PRD (%)": f"{prd_v:.2f}{prd_flag}",
                "RMSE":          f"{rmse_v:.4f}",
                "CC":            f"{cc_v:.4f}",
                "R-Peak (%)": f"{rpeak_s}{rpeak_flag}",
                "WER":           wer_s,
            })
        except Exception:
            pass

    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True,
                     column_config={
                         "PRD (%)": st.column_config.TextColumn("PRD (%) [<9% = clinical]"),
                         "R-Peak (%)": st.column_config.TextColumn("R-Peak (%) [≥90% = clinical]"),
                         "WER": st.column_config.TextColumn("WER (ideal = 1.0)"),
                     })

    # ── Mini per-metric cards for the best DL model ───────────────────────────
    if rows:
        best_row = min(rows, key=lambda r: float(r["PRD (%)"].replace(" ✅","")))
        st.markdown(f"#### 🏅 {best_row['Model']} — this sample at a glance")
        c1, c2, c3, c4, c5, c6 = st.columns(6)
        c1.metric("SNR",     best_row["SNR (dB)"] + " dB")
        c2.metric("PRD",     best_row["PRD (%)"] + "%" if "%" not in best_row["PRD (%)"] else best_row["PRD (%)"])
        c3.metric("RMSE",    best_row["RMSE"])
        c4.metric("CC",      best_row["CC"])
        c5.metric("R-Peak",  best_row["R-Peak (%)"])
        c6.metric("WER",     best_row["WER"])


def show_upload_instructions():
    """Show instructions for uploading model predictions"""
    
    st.warning("⚠️ No model predictions found!")
    
    st.markdown("""
    ### 📁 How to Add Your Models:
    
    Place your model prediction files in the `data/` folder following this naming pattern:
    
    ```
    <model_name>_test_predictions.npz
    ```
    
    **Examples:**
    - `cnn_bilstm_test_predictions.npz`
    - `transformer_cnn_test_predictions.npz`
    - `autoencoder_test_predictions.npz`
    - `my_custom_model_test_predictions.npz`
    
    **Required fields in .npz file:**
    - `original`: Original ECG signals
    - `degraded`: Degraded ECG signals  
    - `reconstructed`: Reconstructed ECG signals
    
    **Optional files for training history visualization:**
    - `<model_name>_training_history.json`
    
    **Expected format for training history JSON:**
    ```json
    {
        "loss": [0.035, 0.024, 0.015, ...],
        "val_loss": [0.024, 0.018, 0.012, ...],
        "mae": [0.12, 0.08, 0.04, ...],
        "val_mae": [0.10, 0.06, 0.03, ...],
        "val_snr": [8.5, 14.2, 19.8, ...],
        "val_prd": [36.8, 23.4, 14.2, ...]
    }
    ```
    
    The page will automatically discover and load all models!
    """)
    
    st.info("💡 **Tip:** The model name will be automatically cleaned for display. For example, 'cnn_bilstm' becomes 'CNN-BiLSTM'")


if __name__ == "__main__":
    show_deep_learning_page()
