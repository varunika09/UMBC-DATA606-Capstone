"""
ICU Sepsis Early Warning System
Streamlit Application — UMBC DATA 606 Capstone
Author: Varunika Bussa

Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import pickle
import json
import os
from pathlib import Path

# Page config 
st.set_page_config(
    page_title="ICU Sepsis Early Warning System",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    /* Main background */
    .stApp { background-color: #0f1117; }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1f2e 0%, #0f1117 100%);
        border-right: 1px solid #2d3748;
    }
    
    /* Cards */
    .metric-card {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    
    .risk-high {
        background: linear-gradient(135deg, #7f1d1d 0%, #991b1b 100%);
        border: 2px solid #ef4444;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    
    .risk-medium {
        background: linear-gradient(135deg, #78350f 0%, #92400e 100%);
        border: 2px solid #f59e0b;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    
    .risk-low {
        background: linear-gradient(135deg, #064e3b 0%, #065f46 100%);
        border: 2px solid #10b981;
        border-radius: 12px;
        padding: 24px;
        text-align: center;
    }
    
    .section-header {
        color: #60a5fa;
        font-size: 1.1rem;
        font-weight: 600;
        border-bottom: 1px solid #2d3748;
        padding-bottom: 8px;
        margin-bottom: 16px;
    }
    
    .info-box {
        background: #1e293b;
        border-left: 4px solid #3b82f6;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.9rem;
        color: #cbd5e1;
    }
    
    .warning-box {
        background: #1c1917;
        border-left: 4px solid #f59e0b;
        border-radius: 0 8px 8px 0;
        padding: 12px 16px;
        margin: 8px 0;
        font-size: 0.85rem;
        color: #fcd34d;
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    
    /* Metric values */
    [data-testid="metric-container"] {
        background: #1a1f2e;
        border: 1px solid #2d3748;
        border-radius: 10px;
        padding: 16px;
    }
    
    /* Header */
    .app-header {
        background: linear-gradient(90deg, #1e3a5f 0%, #1a1f2e 100%);
        border-bottom: 2px solid #3b82f6;
        padding: 16px 24px;
        margin: -24px -24px 24px -24px;
    }
</style>
""", unsafe_allow_html=True)

#Paths
OUTPUTS = Path("outputs")

# Load resources
@st.cache_resource
def load_model():
    with open(OUTPUTS / 'best_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open(OUTPUTS / 'feature_cols.json') as f:
        feat_cols = json.load(f)
    with open(OUTPUTS / 'model_metadata.json') as f:
        meta = json.load(f)
    return model, feat_cols, meta

@st.cache_data
def load_test_data():
    df = pd.read_parquet(OUTPUTS / 'test.parquet')
    return df

@st.cache_resource
def load_shap_explainer():
    shap_path = OUTPUTS / 'shap_explainer.pkl'
    if shap_path.exists():
        with open(shap_path, 'rb') as f:
            return pickle.load(f)
    return None

# Load everything
try:
    model, FEATURE_COLS, metadata = load_model()
    test_df = load_test_data()
    explainer = load_shap_explainer()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading model: {e}")
    st.stop()

# Sidebar 
with st.sidebar:
    st.markdown("### 🏥 Sepsis Warning System")
    st.markdown("*UMBC DATA 606 Capstone*")
    st.markdown("---")

    # Threshold selector
    st.markdown("#### ⚙️ Clinical Operating Mode")
    op_mode = st.radio(
        "",
        options=[
            "🎯 Balanced (Recommended)",
            "📡 High Sensitivity",
            "🔬 High Precision",
            "🔧 Custom Threshold"
        ],
        index=0,
        label_visibility="collapsed"
    )

    if op_mode == "📡 High Sensitivity":
        THRESHOLD = 0.20
        mode_label = "High Sensitivity"
        mode_desc = "Catches ~88% of cases. More false alarms."
    elif op_mode == "🔬 High Precision":
        THRESHOLD = 0.60
        mode_label = "High Precision"
        mode_desc = "Fewer false alarms. Misses more cases."
    elif op_mode == "🔧 Custom Threshold":
        THRESHOLD = st.slider("Threshold", 0.05, 0.95,
                              float(metadata['threshold']), 0.01)
        mode_label = "Custom"
        mode_desc = f"Manual threshold: {THRESHOLD:.2f}"
    else:
        THRESHOLD = float(metadata['threshold'])
        mode_label = "Balanced"
        mode_desc = "Best F1 tradeoff. Recommended for standard monitoring."

    st.markdown(f"""<div class="info-box">
        <strong>{mode_label}</strong><br>{mode_desc}
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Patient selector
    st.markdown("#### 👤 Patient Selection")
    
    # Filter options
    show_sepsis_only = st.checkbox("Show only sepsis patients", value=False)
    
    if show_sepsis_only:
        patient_ids = sorted(
            test_df[test_df['SepsisLabel'] == 1]['Patient_ID'].unique()
        )
        st.caption(f"{len(patient_ids)} sepsis patients in test set")
    else:
        patient_ids = sorted(test_df['Patient_ID'].unique())
        st.caption(f"{len(patient_ids)} total patients in test set")

    selected_pid = st.selectbox("Patient ID", patient_ids)

    # Hour selector
    patient_data = test_df[test_df['Patient_ID'] == selected_pid].sort_values('Hour')
    hours = patient_data['Hour'].tolist()
    selected_hour = st.select_slider("ICU Hour", options=hours,
                                     value=hours[-1] if hours else hours[0])

    st.markdown("---")

    # Model info
    st.markdown("#### 📊 Model Info")
    st.markdown(f"""
    - **Model:** {metadata['model_name'].replace(' (Val)', '')}
    - **PR-AUC:** {metadata['pr_auc_test']:.4f}
    - **ROC-AUC:** {metadata['roc_auc_test']:.4f}
    - **Features:** {metadata['n_features']}
    - **Target:** Sepsis within 6h
    """)

    st.markdown("---")
    st.markdown("""<div class="warning-box">
        ⚠️ Research prototype only.<br>
        Not for clinical use.
    </div>""", unsafe_allow_html=True)

# Main Header
st.markdown("""
<div style="background: linear-gradient(90deg, #1e3a5f 0%, #1a1f2e 100%);
     border-bottom: 2px solid #3b82f6; padding: 20px; margin-bottom: 24px;
     border-radius: 12px;">
    <h1 style="color: white; margin: 0; font-size: 1.8rem;">
        🏥 ICU Sepsis Early Warning System
    </h1>
    <p style="color: #94a3b8; margin: 4px 0 0 0; font-size: 0.95rem;">
        LightGBM · 106 Engineered Features · PhysioNet 2019 Dataset · 
        Predicts sepsis risk 6 hours before clinical onset
    </p>
</div>
""", unsafe_allow_html=True)

# Get data for selected patient/hour
row = patient_data[patient_data['Hour'] == selected_hour]

if row.empty:
    st.error("No data for this patient/hour combination.")
    st.stop()

X_row       = row[FEATURE_COLS]
true_sepsis = int(row['SepsisLabel'].values[0])
true_6h     = int(row['SepsisLabel_6h'].values[0])
risk_prob   = float(model.predict_proba(X_row)[:, 1][0])
is_alert    = risk_prob >= THRESHOLD

#  Top metrics row
col1, col2, col3, col4, col5 = st.columns([2.5, 1.2, 1.2, 1.2, 1.2])

with col1:
    if is_alert:
        bg = "risk-high"
        icon = "🚨"
        label = "HIGH RISK — SEPSIS ALERT"
        color = "#fca5a5"
    elif risk_prob >= 0.15:
        bg = "risk-medium"
        icon = "⚡"
        label = "MODERATE RISK — MONITOR CLOSELY"
        color = "#fde68a"
    else:
        bg = "risk-low"
        icon = "✅"
        label = "LOW RISK"
        color = "#6ee7b7"

    st.markdown(f"""
    <div class="{bg}">
        <div style="font-size: 2.5rem; margin-bottom: 8px;">{icon}</div>
        <div style="color: {color}; font-size: 1.4rem; font-weight: 700;">{risk_prob*100:.1f}%</div>
        <div style="color: {color}; font-size: 0.85rem; font-weight: 600;">{label}</div>
    </div>""", unsafe_allow_html=True)

with col2:
    delta_from_thresh = risk_prob - THRESHOLD
    st.metric("Risk Score", f"{risk_prob:.4f}",
              delta=f"{delta_from_thresh:+.4f} vs threshold",
              delta_color="inverse")

with col3:
    st.metric("Patient", f"{selected_pid}")
    st.metric("ICU Hour", f"{selected_hour}")

with col4:
    label_now = "🔴 Sepsis" if true_sepsis == 1 else "🟢 No Sepsis"
    label_6h  = "🔴 Yes" if true_6h == 1 else "🟢 No"
    st.metric("Current Sepsis", label_now)
    st.metric("Sepsis within 6h", label_6h)

with col5:
    correct = (is_alert == (true_6h == 1))
    st.metric("Threshold", f"{THRESHOLD:.3f}")
    st.metric("Prediction", "✅ Correct" if correct else "❌ Incorrect")

st.markdown("---")

# Row 2: Risk gauge + SHAP + Vitals
col_gauge, col_shap, col_vitals = st.columns([1, 1.5, 1.5])

# Gauge
with col_gauge:
    st.markdown('<div class="section-header">📊 Risk Gauge</div>', unsafe_allow_html=True)

    fig_g, ax_g = plt.subplots(figsize=(4, 3),
                                subplot_kw=dict(polar=True),
                                facecolor='#0f1117')
    ax_g.set_facecolor('#1a1f2e')

    theta = np.linspace(np.pi, 0, 300)
    r_in, r_out = 0.6, 1.0

    # Color zones
    ax_g.fill_between(theta[:100], r_in, r_out, alpha=0.85, color='#065f46')
    ax_g.fill_between(theta[100:200], r_in, r_out, alpha=0.85, color='#92400e')
    ax_g.fill_between(theta[200:], r_in, r_out, alpha=0.85, color='#7f1d1d')

    # Needle
    needle_angle = np.pi * (1 - risk_prob)
    ax_g.annotate("", xy=(needle_angle, 0.85), xytext=(0, 0),
                  arrowprops=dict(arrowstyle="->", color="white", lw=2.5))
    ax_g.scatter([0], [0], color='white', s=40, zorder=10)

    ax_g.set_ylim(0, 1)
    ax_g.set_yticks([])
    ax_g.set_xticks([np.pi, 3*np.pi/4, np.pi/2, np.pi/4, 0])
    ax_g.set_xticklabels(['0%', '25%', '50%', '75%', '100%'],
                          color='#94a3b8', fontsize=8)
    ax_g.set_thetamin(0); ax_g.set_thetamax(180)
    ax_g.spines['polar'].set_visible(False)

    ax_g.text(np.pi/2, 0.3, f"{risk_prob*100:.1f}%",
              ha='center', va='center', fontsize=16,
              fontweight='bold', color='white',
              transform=ax_g.transData)

    plt.tight_layout(pad=0.5)
    st.pyplot(fig_g, use_container_width=True)
    plt.close()

    # Threshold indicator
    st.markdown(f"""
    <div style="text-align:center; color: #94a3b8; font-size: 0.8rem; margin-top: 4px;">
        Alert threshold: <strong style="color: #f87171">{THRESHOLD:.3f}</strong> 
        ({mode_label})
    </div>""", unsafe_allow_html=True)

# SHAP Waterfall 
with col_shap:
    st.markdown('<div class="section-header">🔍 Why This Risk? (SHAP)</div>',
                unsafe_allow_html=True)

    if explainer is not None:
        try:
            import shap
            with st.spinner("Computing explanation..."):
                shap_exp = explainer(X_row)

                if hasattr(shap_exp, 'values') and len(np.array(shap_exp.values).shape) == 3:
                    import shap as shap_lib
                    shap_obj = shap_lib.Explanation(
                        values       = shap_exp.values[:, :, 1],
                        base_values  = shap_exp.base_values[:, 1],
                        data         = shap_exp.data,
                        feature_names= FEATURE_COLS
                    )
                else:
                    shap_obj = shap_exp

                fig_s, ax_s = plt.subplots(figsize=(6, 5), facecolor='#1a1f2e')
                ax_s.set_facecolor('#1a1f2e')
                shap.waterfall_plot(shap_obj[0], max_display=12, show=False)

                # Style the plot
                plt.gcf().set_facecolor('#1a1f2e')
                for ax in plt.gcf().axes:
                    ax.set_facecolor('#1a1f2e')
                    ax.tick_params(colors='#e2e8f0', labelsize=8)
                    ax.xaxis.label.set_color('#94a3b8')
                plt.title("Feature Contributions to Risk Score\n"
                          "→ Red = increased risk  |  ← Blue = decreased risk",
                          color='#e2e8f0', fontsize=9, pad=8)
                plt.tight_layout()
                st.pyplot(fig_s, use_container_width=True)
                plt.close()
        except Exception as e:
            st.info(f"SHAP explanation unavailable: {e}\n\nRun 04_shap_analysis_notebook.ipynb first.")
    else:
        st.info("Run 04_shap_analysis_notebook.ipynb first to enable SHAP explanations.")
        
        # Show top features as fallback
        st.markdown("**Top feature values for this patient:**")
        top_feats = ['HR', 'MAP', 'Resp', 'HR_mean_6h', 'HR_slope_6h',
                     'Lactate', 'sirs_score', 'shock_index']
        feat_vals = {f: float(X_row[f].values[0]) if f in X_row.columns else None
                     for f in top_feats}
        for f, v in feat_vals.items():
            if v is not None and not np.isnan(v):
                st.markdown(f"- **{f}**: {v:.3f}")

# Vital Signs
with col_vitals:
    st.markdown('<div class="section-header">📈 Vital Signs (Last 12h)</div>',
                unsafe_allow_html=True)

    window = patient_data[patient_data['Hour'] <= selected_hour].tail(12)

    vitals_config = [
        ('HR',    'Heart Rate',     'bpm',  90,   None, '#ef4444'),
        ('MAP',   'Mean Art. Pres', 'mmHg', None, 65,   '#3b82f6'),
        ('Resp',  'Resp Rate',      'br/m', 20,   None, '#10b981'),
        ('O2Sat', 'O2 Saturation',  '%',    None, 95,   '#f59e0b'),
    ]

    fig_v, axes_v = plt.subplots(2, 2, figsize=(6, 4.5), facecolor='#1a1f2e')
    fig_v.patch.set_facecolor('#1a1f2e')

    for ax_v, (col, title, unit, hi, lo, color) in zip(axes_v.flat, vitals_config):
        ax_v.set_facecolor('#0f1117')
        vals = window[col].values
        hrs  = window['Hour'].values

        ax_v.plot(hrs, vals, 'o-', color=color, linewidth=2, markersize=4,
                  markerfacecolor='white', markeredgecolor=color, markeredgewidth=1.5)

        if hi:
            ax_v.axhline(hi, color='#ef4444', linestyle='--', alpha=0.6,
                         linewidth=1, label=f'>{hi}')
        if lo:
            ax_v.axhline(lo, color='#ef4444', linestyle='--', alpha=0.6,
                         linewidth=1, label=f'<{lo}')

        # Mark current hour
        cur = window[window['Hour'] == selected_hour]
        if not cur.empty and not pd.isna(cur[col].values[0]):
            ax_v.scatter(cur['Hour'].values, cur[col].values,
                         color='white', s=60, zorder=5, marker='*')

        # Current value annotation
        if not pd.isna(vals[-1]):
            ax_v.annotate(f'{vals[-1]:.0f}', xy=(hrs[-1], vals[-1]),
                          xytext=(5, 5), textcoords='offset points',
                          color='white', fontsize=7)

        ax_v.set_title(f'{title} ({unit})', color='#e2e8f0', fontsize=8,
                       fontweight='bold')
        ax_v.tick_params(colors='#94a3b8', labelsize=7)
        ax_v.spines['bottom'].set_color('#2d3748')
        ax_v.spines['left'].set_color('#2d3748')
        ax_v.spines['top'].set_visible(False)
        ax_v.spines['right'].set_visible(False)
        if hi or lo:
            ax_v.legend(fontsize=6, framealpha=0.3, labelcolor='white')

    plt.tight_layout(pad=1.0)
    st.pyplot(fig_v, use_container_width=True)
    plt.close()

st.markdown("---")

# Risk Timeline
st.markdown('<div class="section-header">⏱️ Risk Score Timeline — Full ICU Stay</div>',
            unsafe_allow_html=True)

X_all  = patient_data[FEATURE_COLS]
probs  = model.predict_proba(X_all)[:, 1]
labels = patient_data['SepsisLabel'].values
hrs_   = patient_data['Hour'].values

fig_t, ax_t = plt.subplots(figsize=(14, 3.5), facecolor='#1a1f2e')
ax_t.set_facecolor('#0f1117')

# Area under the curve
ax_t.fill_between(hrs_, probs, alpha=0.15, color='#3b82f6')
ax_t.plot(hrs_, probs, '-', color='#3b82f6', linewidth=2, label='Predicted Risk')
ax_t.axhline(THRESHOLD, color='#ef4444', linestyle='--',
             linewidth=1.5, label=f'Alert Threshold ({THRESHOLD:.2f})')

# High risk zones
above = probs >= THRESHOLD
ax_t.fill_between(hrs_, probs, THRESHOLD, where=above,
                  alpha=0.3, color='#ef4444', interpolate=True, label='Alert Zone')

# Sepsis onset marker
if labels.any():
    onset_h = hrs_[labels == 1].min()
    ax_t.axvspan(onset_h, hrs_.max(), alpha=0.1, color='#f59e0b',
                 label=f'Actual Sepsis (h={onset_h})')
    ax_t.axvline(onset_h, color='#f59e0b', linestyle='-',
                 linewidth=2, alpha=0.8)
    ax_t.text(onset_h + 1, 0.92, f'Sepsis\nOnset\nh={onset_h}',
              color='#f59e0b', fontsize=8, va='top')

# Current hour
ax_t.axvline(selected_hour, color='white', linestyle=':',
             linewidth=2, alpha=0.8, label=f'Current Hour ({selected_hour})')

ax_t.set_xlabel('ICU Hour', color='#94a3b8', fontsize=10)
ax_t.set_ylabel('Predicted Sepsis Risk', color='#94a3b8', fontsize=10)
ax_t.set_title(f'Risk Trajectory — Patient {selected_pid} | '
               f'ICU Stay: {hrs_.min()}–{hrs_.max()} hours',
               color='#e2e8f0', fontsize=11, fontweight='bold')
ax_t.set_ylim(0, 1)
ax_t.tick_params(colors='#94a3b8', labelsize=9)
ax_t.spines['bottom'].set_color('#2d3748')
ax_t.spines['left'].set_color('#2d3748')
ax_t.spines['top'].set_visible(False)
ax_t.spines['right'].set_visible(False)
legend = ax_t.legend(fontsize=9, framealpha=0.3, labelcolor='white',
                     facecolor='#1a1f2e', edgecolor='#2d3748')

plt.tight_layout()
st.pyplot(fig_t, use_container_width=True)
plt.close()

st.markdown("---")

# Clinical Values + Engineered Features
col_lab, col_eng = st.columns(2)

with col_lab:
    st.markdown('<div class="section-header">🧪 Clinical Values at Hour {}</div>'.format(selected_hour),
                unsafe_allow_html=True)

    clinical_features = {
        'Vitals': ['HR', 'MAP', 'SBP', 'DBP', 'Resp', 'O2Sat', 'Temp'],
        'Key Labs': ['Lactate', 'WBC', 'Creatinine', 'Glucose', 'pH', 'Potassium'],
        'Clinical Scores': ['sirs_score', 'shock_index', 'map_low_flag', 'fever_flag']
    }

    for group, feats in clinical_features.items():
        st.markdown(f"**{group}**")
        cols = st.columns(len(feats))
        for i, feat in enumerate(feats):
            if feat in X_row.columns:
                val = float(X_row[feat].values[0])
                display = "—" if np.isnan(val) else f"{val:.2f}"
                cols[i].metric(feat, display)

with col_eng:
    st.markdown('<div class="section-header">⚙️ Key Engineered Features</div>',
                unsafe_allow_html=True)

    eng_features = {
        'Rolling Means (6h)': ['HR_mean_6h', 'MAP_mean_6h', 'Resp_mean_6h', 'O2Sat_mean_6h'],
        'Slopes (6h)': ['HR_slope_6h', 'MAP_slope_6h', 'Resp_slope_6h', 'Lactate_slope_6h'],
        'Missingness': ['Lactate_measured', 'WBC_measured', 'Lactate_hrs_since']
    }

    for group, feats in eng_features.items():
        st.markdown(f"**{group}**")
        cols = st.columns(len(feats))
        for i, feat in enumerate(feats):
            if feat in X_row.columns:
                val = float(X_row[feat].values[0])
                display = "—" if np.isnan(val) else f"{val:.3f}"
                cols[i].metric(feat, display)

# Footer 
st.markdown("---")
st.markdown("""
<div style="display: flex; justify-content: space-between; align-items: center; 
     padding: 16px; background: #1a1f2e; border-radius: 8px; margin-top: 16px;">
    <div style="color: #64748b; font-size: 0.8rem;">
        <strong style="color: #94a3b8;">Model:</strong> LightGBM Tuned (Optuna) &nbsp;|&nbsp;
        <strong style="color: #94a3b8;">Dataset:</strong> PhysioNet 2019 Sepsis Challenge &nbsp;|&nbsp;
        <strong style="color: #94a3b8;">PR-AUC:</strong> 0.1588 &nbsp;|&nbsp;
        <strong style="color: #94a3b8;">ROC-AUC:</strong> 0.8399
    </div>
    <div style="color: #ef4444; font-size: 0.75rem; font-weight: 600;">
        ⚠️ Research prototype — NOT for clinical use
    </div>
</div>
<div style="color: #374151; font-size: 0.75rem; text-align: center; margin-top: 8px;">
    UMBC DATA 606 Capstone · Varunika Bussa · Early Sepsis Risk Prediction
</div>
""", unsafe_allow_html=True)
