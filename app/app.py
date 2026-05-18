"""
ICU Sepsis Early Warning System
UMBC DATA 606 Capstone — Varunika Bussa
Run: streamlit run app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pickle
import json
from pathlib import Path

st.set_page_config(
    page_title="SepsisAlert ICU",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Force sidebar always visible — never collapse
st.markdown("""
<style>
/* Keep sidebar always open, prevent it from collapsing */
section[data-testid="stSidebar"] {
    width: 280px !important;
    min-width: 280px !important;
    transform: none !important;
    visibility: visible !important;
}
section[data-testid="stSidebar"][aria-expanded="false"] {
    width: 280px !important;
    min-width: 280px !important;
    transform: none !important;
    visibility: visible !important;
    display: block !important;
}
/* Hide the collapse arrow button but keep sidebar open */
[data-testid="collapsedControl"] {
    visibility: hidden !important;
    pointer-events: none !important;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Sans:wght@300;400;500;600;700&family=IBM+Plex+Mono:wght@400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; }

.stApp { background-color: #F0F4F8; font-family: 'IBM Plex Sans', sans-serif; }

[data-testid="stSidebar"] { background: #0A2540 !important; }
/* All sidebar text — bright white base */
[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
    font-family: 'IBM Plex Sans', sans-serif !important;
}
/* Section headings */
[data-testid="stSidebar"] h3 {
    color: #FFFFFF !important;
    font-size: 0.88rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.09em !important;
    text-transform: uppercase !important;
    margin-bottom: 6px !important;
}
/* Radio / checkbox labels — must be bright white */
[data-testid="stSidebar"] .stRadio label p,
[data-testid="stSidebar"] .stCheckbox label p,
[data-testid="stSidebar"] .stRadio div[role="radiogroup"] label span p,
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p,
[data-testid="stSidebar"] [data-baseweb="checkbox"] span {
    color: #FFFFFF !important;
    font-size: 0.85rem !important;
    font-weight: 500 !important;
}
/* Slider label and value */
[data-testid="stSidebar"] .stSlider p,
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMin"],
[data-testid="stSidebar"] .stSlider [data-testid="stTickBarMax"] {
    color: #E2E8F0 !important;
}
/* All p tags and span tags directly in sidebar — bright not grey */
[data-testid="stSidebar"] p { color: #D1DCE5 !important; font-size: 0.82rem !important; }
[data-testid="stSidebar"] small { color: #93C5FD !important; }
/* Selectbox — selected value text */
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] span,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[data-baseweb="select"] div,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[class*="ValueContainer"] span,
[data-testid="stSidebar"] [data-testid="stSelectbox"] div[class*="singleValue"],
[data-testid="stSidebar"] [data-testid="stSelectbox"] input {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}
/* Selectbox dropdown options */
[data-baseweb="popover"] li,
[data-baseweb="menu"] li {
    color: #0F172A !important;
    font-size: 0.85rem !important;
}
/* Divider lines */
[data-testid="stSidebar"] hr { border-color: #1E3A5F !important; margin: 12px 0 !important; }
/* Slider */
[data-testid="stSidebar"] .stSlider label { color: #E2E8F0 !important; }
/* Mode description captions */
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p { color: #93C5FD !important; font-size: 0.8rem !important; }

.main .block-container { padding: 1.2rem 1.8rem 2rem 1.8rem; max-width: 1440px; }

[data-testid="metric-container"] { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 10px; padding: 0.9rem 1rem; box-shadow: 0 1px 3px rgba(0,0,0,0.05); }
[data-testid="metric-container"] label { color: #64748B !important; font-size: 0.7rem !important; font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.09em !important; }
[data-testid="metric-container"] [data-testid="stMetricValue"] { color: #0F172A !important; font-size: 1.25rem !important; font-weight: 700 !important; font-family: 'IBM Plex Mono', monospace !important; }
[data-testid="metric-container"] [data-testid="stMetricDelta"] { font-size: 0.72rem !important; color: #64748B !important; }

.slabel { font-size: 0.68rem; font-weight: 700; letter-spacing: 0.11em; text-transform: uppercase; color: #64748B; margin-bottom: 10px; padding-bottom: 6px; border-bottom: 2px solid #E2E8F0; }

.card { background: #FFFFFF; border: 1px solid #E2E8F0; border-radius: 12px; padding: 1.2rem 1.4rem; box-shadow: 0 1px 4px rgba(0,0,0,0.05); }

.banner-high     { background:#FEF2F2; border:2px solid #DC2626; border-left:6px solid #DC2626; border-radius:10px; padding:1rem 1.4rem; }
.banner-moderate { background:#FFFBEB; border:2px solid #D97706; border-left:6px solid #D97706; border-radius:10px; padding:1rem 1.4rem; }
.banner-low      { background:#F0FDF4; border:2px solid #16A34A; border-left:6px solid #16A34A; border-radius:10px; padding:1rem 1.4rem; }
.banner-high .btitle     { color:#DC2626; font-size:1.4rem; font-weight:700; font-family:'IBM Plex Mono',monospace; }
.banner-moderate .btitle { color:#D97706; font-size:1.4rem; font-weight:700; font-family:'IBM Plex Mono',monospace; }
.banner-low .btitle      { color:#16A34A; font-size:1.4rem; font-weight:700; font-family:'IBM Plex Mono',monospace; }
.bsub { font-size:0.82rem; font-weight:500; color:#475569; margin-top:3px; }

.pstrip { background:#FFFFFF; border:1px solid #E2E8F0; border-radius:10px; padding:0.8rem 1.4rem; display:flex; gap:2rem; align-items:center; box-shadow:0 1px 3px rgba(0,0,0,0.04); margin:12px 0; flex-wrap:wrap; }
.pfield-lbl { font-size:0.62rem; font-weight:700; letter-spacing:0.1em; text-transform:uppercase; color:#94A3B8; }
.pfield-val { font-size:0.95rem; font-weight:700; color:#0F172A; font-family:'IBM Plex Mono',monospace; margin-top:1px; }
.divider { width:1px; height:32px; background:#E2E8F0; }

.stTabs [data-baseweb="tab-list"] { background:#F8FAFC; border-radius:8px; gap:3px; padding:4px; }
.stTabs [data-baseweb="tab"] { color:#475569 !important; font-weight:500 !important; font-size:0.85rem !important; border-radius:6px !important; padding: 6px 16px !important; }
.stTabs [aria-selected="true"] { background:#FFFFFF !important; color:#0A2540 !important; font-weight:700 !important; box-shadow:0 1px 3px rgba(0,0,0,0.1) !important; }

.footer { margin-top:2rem; padding:0.8rem 0 0; border-top:1px solid #E2E8F0; display:flex; justify-content:space-between; align-items:center; }
.fl { font-size:0.73rem; color:#94A3B8; }
.fr { font-size:0.72rem; color:#DC2626; font-weight:600; background:#FEF2F2; padding:3px 8px; border-radius:4px; border:1px solid #FECACA; }

.shap-guide { background:#F8FAFC; border:1px solid #E2E8F0; border-radius:10px; padding:1.2rem; font-size:0.84rem; color:#334155; line-height:1.75; }
.shap-guide strong { color:#0A2540; }

[data-testid="stSidebar"] [data-baseweb="select"] [class*="st-e5"] {
    color: #FFFFFF !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
}
.stMarkdown p { color:#334155 !important; font-size:0.88rem !important; }
</style>
""", unsafe_allow_html=True)

# ── Paths & load ──────────────────────────────────────────────────────────────
OUTPUTS = Path("outputs")

@st.cache_resource
def load_model():
    with open(OUTPUTS/'best_model.pkl','rb') as f: model = pickle.load(f)
    with open(OUTPUTS/'feature_cols.json') as f: feat_cols = json.load(f)
    with open(OUTPUTS/'model_metadata.json') as f: meta = json.load(f)
    return model, feat_cols, meta

@st.cache_data
def load_test(): return pd.read_parquet(OUTPUTS/'test.parquet')

@st.cache_resource
def load_shap():
    p = OUTPUTS/'shap_explainer.pkl'
    if p.exists():
        with open(p,'rb') as f: return pickle.load(f)
    return None

try:
    model, FEATURE_COLS, metadata = load_model()
    test_df  = load_test()
    explainer = load_shap()
except Exception as e:
    st.error(f"Error loading files: {e}"); st.stop()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div style="color:#FFFFFF;font-size:1.1rem;font-weight:700;letter-spacing:0.02em;">SepsisAlert</div>', unsafe_allow_html=True)
    st.markdown('<div style="color:#93C5FD;font-size:0.8rem;margin-bottom:4px;">ICU Early Warning System</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown('<div style="color:#FFFFFF;font-size:0.78rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">Alert Mode</div>', unsafe_allow_html=True)
    mode = st.radio("", ["High Sensitivity","Balanced","High Precision","Custom"],
                    index=1, label_visibility="collapsed")

    if mode == "Custom":
        THRESHOLD = st.slider("Threshold", 0.05, 0.95, float(metadata['threshold']), 0.01)
    elif mode == "High Sensitivity": THRESHOLD = 0.20
    elif mode == "High Precision":   THRESHOLD = 0.60
    else:                            THRESHOLD = float(metadata['threshold'])

    mode_desc = {
        "High Sensitivity": "Catches ~95% of cases. More false alarms. Best for triage.",
        "Balanced":         "Best balance of sensitivity and precision. Recommended.",
        "High Precision":   "Fewer false alarms. May miss some cases.",
        "Custom":           f"Manual threshold: {THRESHOLD:.2f}"
    }
    st.markdown(f'<div style="color:#93C5FD;font-size:0.79rem;margin:4px 0 8px 0;line-height:1.5;">{mode_desc[mode]}</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="color:#FFFFFF;font-size:0.78rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">Patient</div>', unsafe_allow_html=True)
    sepsis_only = st.checkbox("Sepsis patients only", False)
    pids = sorted(test_df[test_df['SepsisLabel']==1]['Patient_ID'].unique()) if sepsis_only \
           else sorted(test_df['Patient_ID'].unique())
    st.markdown(f'<div style="color:#93C5FD;font-size:0.78rem;margin-bottom:6px;">{len(pids):,} patients in test set</div>', unsafe_allow_html=True)

    pid = st.selectbox("Patient ID", pids, label_visibility="collapsed")
    pat = test_df[test_df['Patient_ID']==pid].sort_values('Hour')
    hrs = pat['Hour'].tolist()
    hour = st.select_slider("ICU Hour", options=hrs, value=hrs[-1],
                            label_visibility="collapsed")
    st.markdown(f'<div style="color:#93C5FD;font-size:0.78rem;margin-top:4px;">ICU stay: h{hrs[0]}–h{hrs[-1]} &nbsp;({len(hrs)} hours)</div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div style="color:#FFFFFF;font-size:0.78rem;font-weight:700;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:8px;">Model Performance</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div style="font-size:0.82rem; color:#BAC8D4; line-height:1.9;">
        <span style="color:#60A5FA; font-weight:600;">Algorithm:</span> XGBoost Tuned<br>
        <span style="color:#60A5FA; font-weight:600;">PR-AUC:</span> {metadata['pr_auc_test']:.4f}<br>
        <span style="color:#60A5FA; font-weight:600;">ROC-AUC:</span> {metadata['roc_auc_test']:.4f}<br>
        <span style="color:#60A5FA; font-weight:600;">Features:</span> {metadata['n_features']}<br>
        <span style="color:#60A5FA; font-weight:600;">Target:</span> Sepsis within next 6h<br>
        <span style="color:#60A5FA; font-weight:600;">Dataset:</span> PhysioNet 2019
    </div>
    """, unsafe_allow_html=True)

# ── Compute prediction ────────────────────────────────────────────────────────
row      = pat[pat['Hour']==hour]
X_row    = row[FEATURE_COLS]
true_sep = int(row['SepsisLabel'].values[0])
true_6h  = int(row['SepsisLabel_6h'].values[0])
risk     = float(model.predict_proba(X_row)[:,1][0])
is_alert = risk >= THRESHOLD
correct  = (is_alert == (true_6h == 1))

# ── Page header ───────────────────────────────────────────────────────────────
st.markdown("""
<div style="display:flex;justify-content:space-between;align-items:flex-end;margin-bottom:14px;">
  <div>
    <div style="font-size:1.55rem;font-weight:700;color:#0A2540;letter-spacing:-0.02em;">
      ICU Sepsis Early Warning
    </div>
    <div style="font-size:0.8rem;color:#64748B;margin-top:2px;">
      Predicts sepsis risk 6 hours before clinical onset &nbsp;·&nbsp; Test set patients only
    </div>
  </div>
  <div style="font-size:0.75rem;color:#94A3B8;text-align:right;">
    UMBC DATA 606 Capstone<br>
    <span style="color:#0A2540;font-weight:600;">Varunika Bussa</span>
  </div>
</div>
""", unsafe_allow_html=True)

# ── Patient info strip ────────────────────────────────────────────────────────
sep_val  = '<span style="color:#DC2626;font-weight:700;">SEPSIS</span>' if true_sep else '<span style="color:#16A34A;font-weight:700;">CLEAR</span>'
sixh_val = '<span style="color:#DC2626;font-weight:700;">YES</span>' if true_6h  else '<span style="color:#16A34A;font-weight:700;">NO</span>'
corr_val = '<span style="color:#16A34A;font-weight:700;">CORRECT</span>' if correct else '<span style="color:#DC2626;font-weight:700;">INCORRECT</span>'

st.markdown(f"""
<div class="pstrip">
  <div class="pfield-lbl">Patient ID</div>
  <div class="pfield-val">{pid}</div>
  <div class="divider"></div>
  <div class="pfield-lbl">ICU Hour</div>
  <div class="pfield-val">h{hour}</div>
  <div class="divider"></div>
  <div class="pfield-lbl">Risk Score</div>
  <div class="pfield-val">{risk:.4f}</div>
  <div class="divider"></div>
  <div class="pfield-lbl">Threshold</div>
  <div class="pfield-val">{THRESHOLD:.3f}</div>
  <div class="divider"></div>
  <div class="pfield-lbl">Current Sepsis</div>
  <div class="pfield-val">{sep_val}</div>
  <div class="divider"></div>
  <div class="pfield-lbl">Sepsis within 6h</div>
  <div class="pfield-val">{sixh_val}</div>
  <div class="divider"></div>
  <div class="pfield-lbl">Prediction</div>
  <div class="pfield-val">{corr_val}</div>
</div>
""", unsafe_allow_html=True)

# ── Risk banner ───────────────────────────────────────────────────────────────
if is_alert:
    cls, icon, title, sub = "banner-high",     "⚠",  f"{risk*100:.1f}%  —  HIGH RISK ALERT",    "Model predicts sepsis onset within 6 hours. Clinical assessment recommended immediately."
elif risk >= 0.15:
    cls, icon, title, sub = "banner-moderate", "◉",  f"{risk*100:.1f}%  —  ELEVATED RISK",      "Risk above baseline. Monitor vital signs closely and reassess in 1 hour."
else:
    cls, icon, title, sub = "banner-low",      "✓",  f"{risk*100:.1f}%  —  LOW RISK",           "No immediate sepsis concern predicted at this ICU hour."

st.markdown(f"""
<div class="{cls}">
  <div class="btitle">{icon}&nbsp; {title}</div>
  <div class="bsub">{sub}</div>
</div>
""", unsafe_allow_html=True)

st.markdown("<div style='height:18px'></div>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
t1, t2, t3, t4 = st.tabs([
    "  Overview  ",
    "  Risk Timeline  ",
    "  Clinical Values  ",
    "  SHAP Explanation  "
])

# ══════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ══════════════════════════════════════════════
with t1:
    cg, cv = st.columns([1, 2.2])

    with cg:
        st.markdown('<div class="slabel">Risk Gauge</div>', unsafe_allow_html=True)

        fig_g, ax_g = plt.subplots(figsize=(4,2.9), subplot_kw=dict(polar=True), facecolor='white')
        ax_g.set_facecolor('white')
        theta = np.linspace(np.pi, 0, 300)
        rin, rout = 0.52, 1.0

        ax_g.fill_between(theta[:100],   rin, rout, alpha=0.85, color='#DCFCE7')
        ax_g.fill_between(theta[100:200],rin, rout, alpha=0.85, color='#FEF9C3')
        ax_g.fill_between(theta[200:],   rin, rout, alpha=0.85, color='#FEE2E2')
        for t_ang in [theta[100], theta[200]]:
            ax_g.plot([t_ang,t_ang],[rin,rout], color='#CBD5E1', lw=1.2)

        na = np.pi * (1 - risk)
        ax_g.annotate("", xy=(na,0.80), xytext=(0,0),
                      arrowprops=dict(arrowstyle="->", color='#0A2540', lw=2.5))
        ax_g.scatter([0],[0], color='#0A2540', s=25, zorder=10)

        t_line = np.pi*(1-THRESHOLD)
        ax_g.plot([t_line,t_line],[rin,rout], color='#DC2626', lw=2, linestyle='--', alpha=0.7)

        ax_g.text(np.pi/2, 0.20, f"{risk*100:.1f}%",
                  ha='center', va='center', fontsize=19, fontweight='700',
                  color='#0A2540', fontfamily='monospace')
        ax_g.text(np.pi/2, -0.06, "RISK SCORE",
                  ha='center', va='center', fontsize=7.5,
                  color='#64748B', fontweight='600', fontfamily='monospace')

        ax_g.set_ylim(0,1); ax_g.set_yticks([])
        ax_g.set_xticks([np.pi, 3*np.pi/4, np.pi/2, np.pi/4, 0])
        ax_g.set_xticklabels(['0%','25%','50%','75%','100%'],
                              color='#475569', fontsize=8.5, fontfamily='monospace')
        ax_g.set_thetamin(0); ax_g.set_thetamax(180)
        ax_g.spines['polar'].set_visible(False)
        plt.tight_layout(pad=0.3)
        st.pyplot(fig_g, use_container_width=True)
        plt.close()

        st.markdown(f"""
        <div style="text-align:center;margin-top:2px;">
          <span style="font-size:0.72rem;color:#64748B;font-family:monospace;">
            RED DASHED = THRESHOLD &nbsp;{THRESHOLD:.3f}&nbsp; ({mode})
          </span>
        </div>
        """, unsafe_allow_html=True)

    with cv:
        st.markdown('<div class="slabel">Vital Signs — Last 12 Hours</div>',
                    unsafe_allow_html=True)
        window = pat[pat['Hour'] <= hour].tail(12)
        vcfg   = [
            ('HR','Heart Rate','bpm', 90,None,'#DC2626','#FEF2F2'),
            ('MAP','Mean Art. Press.','mmHg',None,65,'#2563EB','#EFF6FF'),
            ('Resp','Respiratory Rate','br/m',20,None,'#059669','#F0FDF4'),
            ('O2Sat','O₂ Saturation','%',None,94,'#D97706','#FFFBEB'),
        ]
        fig_v, axs = plt.subplots(1,4,figsize=(13,3.2),facecolor='white')
        fig_v.patch.set_facecolor('white')

        for ax_v,(col,ttl,unit,hi,lo,clr,bg) in zip(axs,vcfg):
            ax_v.set_facecolor(bg)
            for sp in ax_v.spines.values():
                sp.set_color('#E2E8F0'); sp.set_linewidth(0.8)
            vals = window[col].values; hrs_w = window['Hour'].values
            ax_v.plot(hrs_w, vals,'o-',color=clr,lw=2.2,markersize=5,
                      markerfacecolor='white',markeredgecolor=clr,markeredgewidth=2,zorder=3)
            if hi: ax_v.axhline(hi,color=clr,ls='--',alpha=0.45,lw=1.3)
            if lo: ax_v.axhline(lo,color=clr,ls='--',alpha=0.45,lw=1.3)
            cur = window[window['Hour']==hour]
            if not cur.empty and not pd.isna(cur[col].values[0]):
                ax_v.scatter(cur['Hour'].values, cur[col].values,
                             color=clr, s=90, zorder=6)
                ax_v.annotate(f"{cur[col].values[0]:.0f}",
                              (cur['Hour'].values[0],cur[col].values[0]),
                              textcoords='offset points',xytext=(0,9),
                              fontsize=9.5,fontweight='700',color=clr,ha='center')
            ax_v.set_title(f"{ttl}\n({unit})",fontsize=9,fontweight='600',
                           color='#1E293B',pad=5)
            ax_v.tick_params(labelsize=8,colors='#475569')
            ax_v.set_xlabel("ICU Hour",fontsize=7.5,color='#94A3B8')
            ax_v.grid(axis='y',alpha=0.3,color='#CBD5E1',lw=0.7)
            if hi or lo:
                ref = hi if hi else lo
                sym = '>' if hi else '<'
                ax_v.text(0.97,0.96,f"Alert: {sym}{ref}",transform=ax_v.transAxes,
                          fontsize=7,color=clr,ha='right',va='top',
                          fontweight='600',alpha=0.75)

        plt.tight_layout(pad=0.9)
        st.pyplot(fig_v, use_container_width=True)
        plt.close()

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    st.markdown('<div class="slabel">Key Clinical Indicators at Hour {}</div>'.format(hour),
                unsafe_allow_html=True)
    ki = ['HR','MAP','Resp','O2Sat','Temp','Lactate','WBC','sirs_score','shock_index']
    kc = st.columns(len(ki))
    for i,f in enumerate(ki):
        if f in X_row.columns:
            v = float(X_row[f].values[0])
            kc[i].metric(f, "—" if np.isnan(v) else f"{v:.2f}")

# ══════════════════════════════════════════════
# TAB 2 — RISK TIMELINE
# ══════════════════════════════════════════════
with t2:
    st.markdown('<div class="slabel">Predicted Risk Score — Full ICU Stay (Patient {})</div>'.format(pid),
                unsafe_allow_html=True)

    X_all  = pat[FEATURE_COLS]
    probs  = model.predict_proba(X_all)[:,1]
    labels = pat['SepsisLabel'].values
    hrs_   = pat['Hour'].values
    above  = probs >= THRESHOLD

    fig_t, ax_t = plt.subplots(figsize=(14,4.2),facecolor='white')
    ax_t.set_facecolor('white')

    ax_t.fill_between(hrs_, THRESHOLD, 1.0, alpha=0.04, color='#DC2626')
    ax_t.fill_between(hrs_, probs, alpha=0.1, color='#2563EB')
    ax_t.plot(hrs_, probs,'-',color='#2563EB',lw=2.2,label='Predicted Risk',zorder=3)
    ax_t.fill_between(hrs_, probs, THRESHOLD, where=above,
                      alpha=0.28, color='#DC2626', interpolate=True, label='Above Threshold',zorder=2)
    ax_t.axhline(THRESHOLD,color='#DC2626',ls='--',lw=2,
                 label=f'Alert Threshold ({THRESHOLD:.2f})',zorder=4)

    if labels.any():
        onset_h = hrs_[labels==1].min()
        ax_t.axvspan(onset_h,hrs_.max(),alpha=0.06,color='#D97706',
                     label=f'Sepsis Onset (h={onset_h})')
        ax_t.axvline(onset_h,color='#D97706',lw=2.2,alpha=0.9)
        ax_t.text(onset_h+1,0.96,f'Sepsis Onset\nh={onset_h}',
                  color='#92400E',fontsize=9,fontweight='700',va='top')

    ax_t.axvline(hour,color='#0A2540',lw=1.8,ls=':',alpha=0.9,
                 label=f'Current Hour (h={hour})')
    if hour in hrs_:
        cp = float(probs[hrs_==hour][0])
        ax_t.scatter([hour],[cp],color='#0A2540',s=80,zorder=7)

    ax_t.set_xlabel('ICU Hour',fontsize=11.5,color='#334155',fontweight='500')
    ax_t.set_ylabel('Predicted Sepsis Risk',fontsize=11.5,color='#334155',fontweight='500')
    ax_t.set_title(f'Risk Trajectory — Patient {pid}',
                   fontsize=13,fontweight='700',color='#0A2540',pad=10)
    ax_t.set_ylim(0,1.05)
    ax_t.tick_params(labelsize=10.5,colors='#475569')
    for sp in ax_t.spines.values(): sp.set_color('#E2E8F0')
    ax_t.grid(axis='y',alpha=0.35,color='#E2E8F0',lw=0.8)
    ax_t.legend(fontsize=10,frameon=True,fancybox=False,
                edgecolor='#E2E8F0',facecolor='white',labelcolor='#334155')
    plt.tight_layout()
    st.pyplot(fig_t, use_container_width=True)
    plt.close()

    st.markdown("<div style='height:10px'></div>", unsafe_allow_html=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Peak Risk Score",      f"{probs.max():.4f}")
    c2.metric("Hours Above Threshold",f"{int(above.sum())}")
    c3.metric("Current Risk Score",   f"{risk:.4f}")
    c4.metric("Total ICU Hours",      f"{len(hrs_)}")

# ══════════════════════════════════════════════
# TAB 3 — CLINICAL VALUES
# ══════════════════════════════════════════════
with t3:
    cl, cr = st.columns(2)
    with cl:
        st.markdown('<div class="slabel">Vital Signs</div>', unsafe_allow_html=True)
        v_feats = ['HR','MAP','SBP','DBP','Resp','O2Sat','Temp']
        v_units = {'HR':'bpm','MAP':'mmHg','SBP':'mmHg','DBP':'mmHg','Resp':'br/m','O2Sat':'%','Temp':'°C'}
        v_norms = {'HR':(60,100),'MAP':(70,105),'SBP':(90,140),'DBP':(60,90),'Resp':(12,20),'O2Sat':(95,100),'Temp':(36,38)}
        vc = st.columns(4)
        for i,f in enumerate(v_feats):
            if f in X_row.columns:
                v = float(X_row[f].values[0])
                lo,hi = v_norms.get(f,(None,None))
                flag = not np.isnan(v) and lo and hi and (v<lo or v>hi)
                vc[i%4].metric(f, "—" if np.isnan(v) else f"{v:.1f} {v_units.get(f,'')}",
                               "⚠ Outside normal" if flag else f"Normal: {lo}–{hi}",
                               delta_color="inverse" if flag else "off")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="slabel">Laboratory Values</div>', unsafe_allow_html=True)
        l_feats = ['Lactate','WBC','Creatinine','Glucose','pH','BUN','Hgb','Potassium']
        lc = st.columns(4)
        for i,f in enumerate(l_feats):
            if f in X_row.columns:
                v = float(X_row[f].values[0])
                lc[i%4].metric(f, "—" if np.isnan(v) else f"{v:.2f}")

    with cr:
        st.markdown('<div class="slabel">Clinical Risk Scores (SOFA-Inspired)</div>', unsafe_allow_html=True)
        scores = {
            'sirs_score':  ('SIRS Score','0–3 criteria met'),
            'shock_index': ('Shock Index','HR/SBP  ·  >1.0 = instability'),
            'map_low_flag':('Low MAP Flag','1 = MAP below 65 mmHg'),
            'fever_flag':  ('Fever Flag','1 = Temp outside 36–38°C'),
        }
        for feat,(label,desc) in scores.items():
            if feat in X_row.columns:
                v = float(X_row[feat].values[0])
                st.metric(label, "—" if np.isnan(v) else f"{v:.2f}", desc, delta_color="off")

        st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
        st.markdown('<div class="slabel">Engineered Temporal Features</div>', unsafe_allow_html=True)
        eng = {'HR_mean_6h':'HR Mean (6h)','HR_slope_6h':'HR Slope (6h)',
               'Resp_mean_6h':'Resp Mean (6h)','MAP_min_6h':'MAP Min (6h)',
               'Lactate_measured':'Lactate Drawn?','Lactate_hrs_since':'Lactate Age (h)'}
        ec = st.columns(3)
        for i,(f,lbl) in enumerate(eng.items()):
            if f in X_row.columns:
                v = float(X_row[f].values[0])
                ec[i%3].metric(lbl, "—" if np.isnan(v) else f"{v:.3f}")

# ══════════════════════════════════════════════
# TAB 4 — SHAP
# ══════════════════════════════════════════════
with t4:
    if explainer is not None:
        try:
            import shap
            with st.spinner("Computing SHAP explanation..."):
                shap_exp = explainer(X_row)
                if hasattr(shap_exp,'values') and len(np.array(shap_exp.values).shape)==3:
                    shap_obj = shap.Explanation(
                        values=shap_exp.values[:,:,1],
                        base_values=shap_exp.base_values[:,1],
                        data=shap_exp.data,
                        feature_names=FEATURE_COLS
                    )
                else: shap_obj = shap_exp

            cs, cg2 = st.columns([2.2, 1])
            with cs:
                st.markdown('<div class="slabel">What Drove This Prediction?</div>',
                            unsafe_allow_html=True)
                fig_s, _ = plt.subplots(figsize=(10,6.5), facecolor='white')
                plt.gcf().set_facecolor('white')
                shap.waterfall_plot(shap_obj[0], max_display=14, show=False)
                for ax in plt.gcf().axes:
                    ax.set_facecolor('white')
                    for sp in ax.spines.values(): sp.set_color('#E2E8F0')
                    ax.tick_params(colors='#334155', labelsize=9.5)
                plt.title(f"SHAP Waterfall — Patient {pid}, Hour {hour}  ·  Risk = {risk*100:.1f}%",
                          fontsize=11, fontweight='700', color='#0A2540', pad=12)
                plt.tight_layout()
                st.pyplot(fig_s, use_container_width=True)
                plt.close()

            with cg2:
                st.markdown('<div class="slabel">Reading This Chart</div>',
                            unsafe_allow_html=True)
                st.markdown(f"""
                <div class="shap-guide">
                <p><strong>What is SHAP?</strong><br>
                SHAP explains <em>why</em> this patient received a risk score of <strong>{risk*100:.1f}%</strong>. It shows which features pushed the score up or down.</p>

                <p><strong style="color:#DC2626;">Red bars →</strong><br>
                Feature values that <strong>increased</strong> this patient's risk above the baseline.</p>

                <p><strong style="color:#2563EB;">Blue bars ←</strong><br>
                Feature values that <strong>decreased</strong> this patient's risk.</p>

                <p><strong>f(x) = {risk:.4f}</strong><br>
                The final predicted risk score for this patient at this hour.</p>

                <p><strong>E[f(x)]</strong><br>
                The baseline — average prediction across all patients.</p>

                <p><strong>Bar length = influence strength.</strong><br>
                Longer bar = that feature had more impact on this specific prediction.</p>

                <p>The feature names show the actual value for this patient in brackets.</p>
                </div>
                """, unsafe_allow_html=True)

        except Exception as e:
            st.warning(f"SHAP unavailable: {e}")
    else:
        st.info("Run `04_shap_analysis_notebook.ipynb` first to enable SHAP explanations.")

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
  <div class="fl">UMBC DATA 606 Capstone &nbsp;·&nbsp; Varunika Bussa &nbsp;·&nbsp; XGBoost · PhysioNet 2019 · 105 Features</div>
  <div class="fr">⚠ Research prototype — Not for clinical use</div>
</div>
""", unsafe_allow_html=True)
