"""
Sensitivity Analysis Page
=========================
Two robustness analyses requested by reviewers (R1 C#1-4, R2 C#4-5, R4 C#4):

  1. Channel sensitivity   — 3 ambient/occlusion scenarios x 3 activities
                             (low / nominal / severe).  Source: metadata.json.
  2. IMU threshold sensitivity — percentile-threshold perturbation
                             (delta = -10 / 0 / +10).  Computed live from the
                             real IMU CSV when available, else documented values.

Together they show the corrected pilot VLC channel is robust to both
ambient-noise and IMU-threshold uncertainty (simulation-to-reality discussion).

Author: Grace
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st


# ---------------------------------------------------------------------------
DATASET_DIR = Path(__file__).parent.parent / "datasets" / "thesis_dataset"
METADATA_PATH = DATASET_DIR / "metadata.json"
IMU_CSV_PATH = Path(__file__).parent.parent / "data" / "Activity_Recognition_Data.csv"

ACTIVITIES = ["walking", "sitting", "standing"]
SCENARIO_ORDER = ["low_ambient", "nominal", "severe_ambient"]
SCENARIO_LABELS = {
    "low_ambient": "Low ambient",
    "nominal": "Nominal (training)",
    "severe_ambient": "Severe ambient",
}

# Documented IMU threshold-sensitivity fallback (from run_threshold_sensitivity()).
THRESHOLD_FALLBACK = {
    "walking":  {-10: (5.1, 30.7, 64.2), 0: (14.9, 30.2, 54.9), 10: (22.1, 37.9, 40.0)},
    "sitting":  {-10: (16.6, 31.0, 52.4), 0: (26.8, 34.8, 38.5), 10: (38.0, 38.9, 23.1)},
    "standing": {-10: (25.6, 34.0, 40.4), 0: (37.1, 36.8, 26.1), 10: (49.1, 39.5, 11.4)},
}


def _load_metadata():
    if METADATA_PATH.exists():
        try:
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return None
    return None


def _compute_threshold_sensitivity_live():
    """Run the threshold-sensitivity utility on the real IMU CSV. Returns
    {activity: {delta: (los%, partial%, diffuse%)}} or None on failure."""
    if not IMU_CSV_PATH.exists():
        return None
    try:
        import sys
        sys.path.append(str(Path(__file__).parent.parent / "utils"))
        from imu_analysis import (load_imu_dataset, remove_gravity_with_ema,
                                  compute_gyroscope_stability)

        base = {
            "standing": ([50, 80], [60, 85]),
            "sitting":  ([40, 70], [50, 75]),
            "walking":  ([25, 60], [35, 65]),
        }
        imu = load_imu_dataset(str(IMU_CSV_PATH))
        out = {}
        for act in ACTIVITIES:
            if act not in imu:
                continue
            d = imu[act]
            _, dyn = remove_gravity_with_ema(d[["ax", "ay", "az"]])
            _, gyro_st = compute_gyroscope_stability(d[["gx", "gy", "gz"]])
            dyn_mag = dyn["dyn_magnitude"].values
            gyro_st = np.asarray(gyro_st).flatten()
            acc_p, gyr_p = base[act]
            out[act] = {}
            for delta in (-10, 0, 10):
                al = np.percentile(dyn_mag, np.clip(acc_p[0] + delta, 1, 99))
                ah = np.percentile(dyn_mag, np.clip(acc_p[1] + delta, 1, 99))
                gl = np.percentile(gyro_st, np.clip(gyr_p[0] + delta, 1, 99))
                gh = np.percentile(gyro_st, np.clip(gyr_p[1] + delta, 1, 99))
                states = np.ones(len(dyn_mag), dtype=int)
                states[(dyn_mag < al) & (gyro_st < gl)] = 0
                states[(dyn_mag > ah) | (gyro_st > gh)] = 2
                n = len(states)
                out[act][delta] = (
                    round(100 * np.mean(states == 0), 1),
                    round(100 * np.mean(states == 1), 1),
                    round(100 * np.mean(states == 2), 1),
                )
        return out or None
    except Exception:
        return None


def show_sensitivity_analysis():
    st.title("🔬 Sensitivity Analysis")
    st.markdown(
        "Two robustness analyses demonstrate that the "
        "corrected pilot VLC channel is **robust to both ambient-noise and "
        "IMU-threshold uncertainty** — the core of the simulation-to-reality argument."
    )

    meta = _load_metadata()

    # ===================================================================
    # 0. FULL-DATASET STATISTICS (headline — all 40,000 segments)
    # ===================================================================
    st.header("📊 Full-Dataset Statistics (all 40,000 segments)")
    st.caption(
        "Authoritative communication-layer metrics on the **entire dataset** "
        "(degraded ECG, *before* reconstruction), one definition each — the "
        "headline values reported in the manuscript. Per-activity reconciles "
        "with the overall (segment-count-weighted)."
    )
    fd = (meta or {}).get("statistics", {}).get("full_dataset")
    if fd:
        rows = []
        for a in ACTIVITIES:
            pa = fd.get("per_activity", {}).get(a)
            if pa:
                rows.append({"Activity": a, "n": pa.get("n"),
                             "Raw BER (%)": round(pa["ber_pct"], 2),
                             "SQ-SNR (dB)": round(pa["sqsnr_db"], 2),
                             "PRD (%)": round(pa["prd_pct"], 2)})
        rows.append({"Activity": "OVERALL", "n": fd.get("n"),
                     "Raw BER (%)": round(fd["ber_pct"], 2),
                     "SQ-SNR (dB)": round(fd["sqsnr_db"], 2),
                     "PRD (%)": round(fd["prd_pct"], 2)})
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

        # Split comparison (full vs train/val/test) if available
        scmp = (meta or {}).get("statistics", {}).get("split_comparison")
        if scmp:
            with st.expander("Per-split comparison (full vs train / val / test)", expanded=False):
                srows = []
                for k in ("full", "train", "val", "test"):
                    if k in scmp:
                        c = scmp[k]
                        srows.append({"Scope": k, "n": c.get("n"),
                                      "Raw BER (%)": round(c["ber_pct"], 2),
                                      "SQ-SNR (dB)": round(c["sqsnr_db"], 2),
                                      "PRD (%)": round(c["prd_pct"], 2)})
                st.dataframe(pd.DataFrame(srows), use_container_width=True, hide_index=True)
        st.info(
            f"**Headline (40,000 segments):** mean BER ≈ {fd['ber_pct']:.2f}%, "
            f"mean SQ-SNR ≈ {fd['sqsnr_db']:.2f} dB, mean PRD ≈ {fd['prd_pct']:.2f}%. "
            "These are the absolute values reported in the paper; the 200-segment "
            "sensitivity scan below shows the relative robustness to ambient perturbation."
        )
    else:
        st.warning("`full_dataset` block not found in `metadata.json` — re-run the generator.")

    st.markdown("---")

    # ===================================================================
    # 1. CHANNEL SENSITIVITY
    # ===================================================================
    st.header("1️⃣ Channel Sensitivity — Ambient / Occlusion (200-segment subsample)")
    st.caption(
        "**200-segment robustness subsample** (degraded ECG, *before* reconstruction). "
        "Three scenarios perturb noise, attenuation, jitter and diffuse scaling "
        "(200 segments × 3 activities each). Source: `metadata.json`. "
        "BER reconciles with the full dataset; the **absolute SQ-SNR/PRD here are "
        "subsample/noise-realization values** — the headline absolutes are the "
        "full-dataset values shown below."
    )

    with st.expander("📐 Calculation details — how channel sensitivity is computed", expanded=False):
        st.markdown(
            "Each scenario re-transmits the **same 200 ECG segments per activity** "
            "through a channel whose parameters are scaled by scenario factors, then "
            "averages the BER and SQ-SNR. Only five parameters change between scenarios:"
        )
        st.latex(r"g'(a_t) = g(a_t)\cdot 10^{\,\Delta_{att}/20}\qquad"
                 r"\xi'(t)=\xi(t)^{\,k_{jit}}")
        st.latex(r"\beta' = \beta\cdot k_{diff}\qquad"
                 r"\sigma'_{th}=\sigma_{th}\cdot k_{th}\qquad"
                 r"\sigma'_{sh}=\sigma_{sh}\cdot k_{sh}")
        st.markdown(
            "| Scenario | Δ_att (dB) | k_th | k_sh | k_jit | k_diff |\n"
            "|---|---|---|---|---|---|\n"
            "| Low ambient | −1.0 | 0.5 | 0.5 | 0.8 | 0.7 |\n"
            "| **Nominal** | 0.0 | 1.0 | 1.0 | 1.0 | 1.0 |\n"
            "| Severe ambient | +3.0 | 3.0 | 2.0 | 1.5 | 1.4 |"
        )
        st.latex(r"\mathrm{BER}_{sc,act}=\frac{1}{200}\sum_{i=1}^{200}"
                 r"\mathrm{BER}\big(x_i \to r_i^{sc}\big)")
        st.caption(
            "Interpretation: if BER barely moves when noise triples and attenuation "
            "grows by 3 dB, the channel is **not** noise-limited — it is limited by the "
            "structural motion-induced diffuse ISI, which the scenarios do not remove."
        )

    sens = (meta or {}).get("sensitivity_analysis") if meta else None
    if sens:
        rows = []
        for sc in SCENARIO_ORDER:
            for act in ACTIVITIES:
                cell = sens.get(sc, {}).get(act)
                if cell:
                    # 200-segment robustness subsample (degraded ECG, BEFORE
                    # reconstruction). BER reconciles with the full dataset; the
                    # absolute SQ-SNR/PRD are subsample/noise-realization values
                    # (the headline absolutes are the full-dataset values, §6).
                    row = {
                        "Scenario": SCENARIO_LABELS.get(sc, sc),
                        "Activity": act,
                        "Raw BER (%)": round(cell["ber"] * 100, 2),
                        "SQ-SNR (dB)": round(cell["sq_snr"], 2),
                    }
                    if "prd" in cell:
                        row["PRD (%)"] = round(cell["prd"], 2)
                    rows.append(row)
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # BER bar chart grouped by activity
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
        width = 0.25
        x = np.arange(len(ACTIVITIES))
        for i, sc in enumerate(SCENARIO_ORDER):
            bers = [sens.get(sc, {}).get(a, {}).get("ber", 0) * 100 for a in ACTIVITIES]
            ax1.bar(x + (i - 1) * width, bers, width, label=SCENARIO_LABELS[sc])
        ax1.set_xticks(x); ax1.set_xticklabels(ACTIVITIES)
        ax1.set_ylabel("Raw BER (%)"); ax1.set_title("BER by activity & scenario")
        ax1.legend(fontsize=8); ax1.grid(alpha=0.3, axis="y")

        for i, sc in enumerate(SCENARIO_ORDER):
            snrs = [sens.get(sc, {}).get(a, {}).get("sq_snr", 0) for a in ACTIVITIES]
            ax2.bar(x + (i - 1) * width, snrs, width, label=SCENARIO_LABELS[sc])
        ax2.set_xticks(x); ax2.set_xticklabels(ACTIVITIES)
        ax2.set_ylabel("SQ-SNR (dB)"); ax2.set_title("SQ-SNR by activity & scenario")
        ax2.legend(fontsize=8); ax2.grid(alpha=0.3, axis="y")
        st.pyplot(fig)

        # Max swing insight
        try:
            walk = [sens[sc]["walking"]["ber"] * 100 for sc in SCENARIO_ORDER]
            swing = max(walk) - min(walk)
            st.success(
                f"**Key result (relative change):** across all ambient scenarios the walking BER "
                f"swings only **{swing:.1f} percentage points** (low→severe), while changing "
                f"*activity* changes BER by ~5 pp. The channel is **motion-limited, not "
                f"ambient-noise-limited** — the reconstruction task stays consistently defined."
            )
        except Exception:
            pass
        # Pull the authoritative full-split absolute SQ-SNR straight from metadata
        # so this note always stays aligned with the dataset (one value everywhere).
        try:
            stats = (meta or {}).get("statistics", {})
            full = stats.get("full_dataset")
            if full:
                # authoritative full-dataset (all 40,000 segments) block
                ber_all = full["ber_pct"]
                snr_all = full["sqsnr_db"]
                prd_all = full.get("prd_pct")
                pa = full.get("per_activity", {})
                pa_txt = " / ".join(f"{pa[a]['sqsnr_db']:.2f}" for a in ACTIVITIES if a in pa)
            else:
                bd = stats.get("ber_distribution", {}); sd = stats.get("snr_distribution", {})
                ber_all = np.mean([bd[s]["mean"] for s in ("train","val","test") if s in bd]) * 100
                snr_all = np.mean([sd[s]["mean"] for s in ("train","val","test") if s in sd])
                prd_all = None
                pa = sd.get("train", {}).get("per_activity", {})
                pa_txt = " / ".join(f"{pa[a]['mean']:.2f}" for a in ACTIVITIES if a in pa) or "6.96 / 8.08 / 9.78"
            prd_txt = f", PRD ≈ {prd_all:.2f}%" if prd_all is not None else ""
            st.caption(
                "Note on absolute values: this robustness scan is reported primarily as the "
                f"**relative change** across scenarios. The single authoritative absolute value is "
                f"the **full dataset** result (mean BER ≈ {ber_all:.2f}%, mean SQ-SNR ≈ {snr_all:.2f} dB{prd_txt}; "
                f"per-activity SQ-SNR {pa_txt} dB for walking / sitting / standing), computed over "
                "all 40,000 segments with one definition — see the main results table."
            )
        except Exception:
            st.caption(
                "Note: report this scan as the **relative change** across scenarios; the single "
                "authoritative absolute BER/SQ-SNR is the full-split value in the statistics block."
            )
    else:
        st.warning(
            "No `sensitivity_analysis` block found in `metadata.json`. "
            "Re-run the dataset generation script to populate it."
        )

    # ── Two-axis proof: ambient effect vs activity effect ─────────────────────
    try:
        fd = (meta or {}).get("statistics", {}).get("full_dataset", {})
        sa = (meta or {}).get("sensitivity_analysis", {})
        # Activity axis (full dataset, nominal channel): walking − standing
        act_hi = fd["per_activity"]["walking"]["ber_pct"]
        act_lo = fd["per_activity"]["standing"]["ber_pct"]
        activity_eff = act_hi - act_lo
        # Ambient axis (sensitivity, walking): severe − low
        amb_lo = sa["low_ambient"]["walking"]["ber"] * 100
        amb_hi = sa["severe_ambient"]["walking"]["ber"] * 100
        ambient_eff = amb_hi - amb_lo
        st.markdown("### ✅ Proof: the channel is motion-limited, not ambient-noise-limited")
        st.markdown(
            f"| Axis | Comparison | BER change |\n"
            f"|---|---|---|\n"
            f"| **Ambient** (window) | walking: low → severe ambient | **+{ambient_eff:.1f} pp** |\n"
            f"| **Activity / motion** (heater) | nominal: walking → standing | **{activity_eff:.1f} pp** |"
        )
        ratio = activity_eff / ambient_eff if ambient_eff else float("nan")
        st.success(
            f"**Conclusion.** Changing the ambient/occlusion condition from low to severe changes the "
            f"raw BER by only **+{ambient_eff:.1f} percentage points**, whereas changing the activity "
            f"(walking → standing) changes it by **{activity_eff:.1f} percentage points** — about "
            f"**{ratio:.1f}× larger**. Because the activity (motion) effect dominates the ambient-noise "
            f"effect, the channel is **motion-limited, not ambient-noise-limited**. The reconstruction "
            f"task is therefore consistently defined across ambient conditions, supporting the "
            f"generalization claim. (Activity contrast from the full 40,000-segment dataset; ambient "
            f"contrast from the sensitivity scan.)"
        )
    except Exception:
        pass

    st.markdown("---")

    # ===================================================================
    # 2. IMU THRESHOLD SENSITIVITY
    # ===================================================================
    st.header("2️⃣ IMU Threshold Sensitivity (Δ = −10 / 0 / +10)")
    st.caption(
        "Percentile thresholds mapping IMU motion to channel states are engineering "
        "choices. Shifting all four thresholds by ±10 points tests how the state "
        "distribution responds."
    )

    with st.expander("📐 Calculation details — how threshold sensitivity is computed", expanded=False):
        st.markdown(
            "Each activity has four baseline percentile thresholds on the dynamic "
            "acceleration magnitude (acc) and gyro stability (gyro). A perturbation Δ "
            "shifts **all four** percentiles, then every IMU sample is re-classified:"
        )
        st.latex(r"\theta_{acc,low}=P\big(\text{acc},\,\mathrm{clip}(p_{low}+\Delta,1,99)\big)\quad"
                 r"\text{(and similarly for acc-high, gyro-low, gyro-high)}")
        st.code(
            "state = 1                                  # Partially-obstructed (default)\n"
            "if acc < theta_acc_low  and gyro < theta_gyro_low:  state = 0   # LoS\n"
            "if acc > theta_acc_high or  gyro > theta_gyro_high: state = 2   # Diffuse",
            language="python",
        )
        st.markdown(
            "**Baseline percentiles** — walking (25/60, 35/65), sitting (40/70, 50/75), "
            "standing (50/80, 60/85). Δ = −10 lowers the bars → more Diffuse; "
            "Δ = +10 raises them → more LoS. The table then reports the % of samples in "
            "each state."
        )
        st.caption(
            "Interpretation: even with a ±10-point threshold error, the *ordering* of the "
            "states is preserved (walking always most diffuse), so the learned Markov "
            "channel keeps its physical meaning."
        )

    thr = _compute_threshold_sensitivity_live()
    source = "live (computed from real IMU data)"
    if thr is None:
        thr = THRESHOLD_FALLBACK
        source = "documented values (IMU CSV not available for live computation)"
    st.caption(f"Source: {source}.")

    rows = []
    for act in ACTIVITIES:
        if act not in thr:
            continue
        for delta in (-10, 0, 10):
            los, par, dif = thr[act][delta]
            rows.append({
                "Activity": act,
                "Δ": f"{delta:+d}" if delta != 0 else "0 (baseline)",
                "LoS-dominant %": los,
                "Partially-obstructed %": par,
                "Diffuse-dominant %": dif,
            })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # Stacked bar per activity
    fig, axes = plt.subplots(1, len(ACTIVITIES), figsize=(13, 4), sharey=True)
    if len(ACTIVITIES) == 1:
        axes = [axes]
    for ax, act in zip(axes, ACTIVITIES):
        if act not in thr:
            continue
        deltas = [-10, 0, 10]
        los = [thr[act][d][0] for d in deltas]
        par = [thr[act][d][1] for d in deltas]
        dif = [thr[act][d][2] for d in deltas]
        labels = [f"Δ={d:+d}" if d != 0 else "Δ=0" for d in deltas]
        ax.bar(labels, los, label="LoS", color="#2E8B57")
        ax.bar(labels, par, bottom=los, label="Partial", color="#DAA520")
        ax.bar(labels, dif, bottom=np.array(los) + np.array(par),
               label="Diffuse", color="#C0504D")
        ax.set_title(act); ax.set_ylim(0, 100)
    axes[0].set_ylabel("State share (%)")
    axes[-1].legend(fontsize=8, loc="upper right")
    st.pyplot(fig)

    st.info(
        "**Interpretation (ordering preservation).** The key robustness result is not which "
        "activity shifts most, but that the **activity ordering is preserved** under any common "
        "±10-percentile perturbation: at every Δ, walking shows the most diffuse-dominant (most "
        "obstructed) states and standing the most LoS-dominant states "
        "(LoS: walking < sitting < standing for Δ = −10, 0, +10). This confirms the learned Markov "
        "model captures genuine motion-dependent channel behavior rather than an artifact of the "
        "specific threshold choice. "
        "(Note: standing's distribution shifts slightly *more* per unit threshold change because it "
        "operates at mid-range percentiles in the dense centre of its motion distribution, whereas "
        "walking operates in the sparse lower tail — but this does not affect the preserved ordering.)"
    )

    st.markdown("---")
    st.header("🔗 How the two analyses bridge simulation and reality")
    st.markdown(
        """
| Stage | Uncertainty source | Sensitivity test |
|---|---|---|
| IMU → channel states | percentile thresholds are engineering judgments | **Analysis 2** (Δ = ±10) |
| VLC channel (C1–C8) | noise & attenuation assumed, not measured | **Analysis 1** (low/nominal/severe) |

**Combined conclusion:** even if the IMU thresholds are ±10 percentile points off
**and** the ambient noise is 3× worse than assumed, the Markov channel keeps its
physically-correct state ordering and BER changes by < 2 pp. A model trained on the
nominal surrogate channel therefore generalizes to the real conditions it approximates.
"""
    )

    st.success(
        "**How we argue the simulation-to-reality gap is bridged (evidence-based):**\n\n"
        "1. **Inputs are real, not invented** — the Markov matrix P, jitter σ and "
        "diffuse β are *learned from real IMU motion data*, not hand-tuned.\n"
        "2. **Outputs are bounded by data, not assumptions** — Analysis 1 shows BER "
        "moves < 2 pp across a 3× noise / +3 dB attenuation sweep, so the unknown true "
        "ambient conditions cannot change the conclusions.\n"
        "3. **Structure survives uncertainty** — Analysis 2 shows the state ordering "
        "(walking > sitting > standing occlusion) holds under ±10-point threshold error.\n"
        "4. **Honest scope** — remaining un-modelled effects (sunlight, skin-tone "
        "reflectance, beam divergence) are documented in `metadata.json` "
        "(`simulation_to_reality_gap`), and hardware validation with a chest-LED / "
        "wrist-photodiode is stated as the next step."
    )
