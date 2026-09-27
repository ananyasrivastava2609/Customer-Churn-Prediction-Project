"""
Customer Churn Risk Detection Dashboard — Redesigned UI v2 (Fixed Nav)
"""
import io
import json
import logging
import sys
import tempfile
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

Path(ROOT / "logs").mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.FileHandler(ROOT / "logs" / "app.log")],
)
logger = logging.getLogger(__name__)

st.set_page_config(
    page_title="ChurnGuard · Analytics",
    page_icon="🔮",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=DM+Sans:wght@300;400;500&display=swap');

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"], [data-testid="stMain"] {
    background: #07070f !important;
    color: #e8e6f0;
    font-family: 'DM Sans', sans-serif;
}
[data-testid="stMainBlockContainer"] { background: #07070f !important; }
[data-testid="stVerticalBlock"] { background: transparent !important; }

/* Hide sidebar and ALL its toggle buttons — nav lives in tabs now */
[data-testid="stSidebar"]                 { display: none !important; }
[data-testid="collapsedControl"]          { display: none !important; }
[data-testid="stSidebarCollapsedControl"] { display: none !important; }

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stDecoration"] { display: none; }

/* ── Tab navigation ── */
[data-testid="stTabs"] [role="tablist"] {
    background: #0b0b18 !important;
    border: 1px solid #1a1a2e !important;
    border-radius: 14px !important;
    padding: 0.3rem 0.4rem !important;
    gap: 0.2rem !important;
    margin-bottom: 1.6rem !important;
}
[data-testid="stTabs"] [role="tab"] {
    background: transparent !important;
    border: 1px solid transparent !important;
    border-radius: 10px !important;
    color: #5848a0 !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.84rem !important;
    font-weight: 600 !important;
    padding: 0.48rem 1.2rem !important;
    transition: background 0.15s, color 0.15s !important;
}
[data-testid="stTabs"] [role="tab"]:hover {
    background: rgba(100,60,220,0.1) !important;
    color: #a090d8 !important;
}
[data-testid="stTabs"] [role="tab"][aria-selected="true"] {
    background: rgba(100,60,220,0.18) !important;
    border-color: rgba(100,60,220,0.3) !important;
    color: #c0b0ff !important;
}
[data-testid="stTabs"] button[role="tab"] { border-bottom: none !important; }

/* ── Animations ── */
@keyframes drift {
    0%,100% { transform: translateY(0px) rotate(0deg); }
    50%      { transform: translateY(-8px) rotate(1deg); }
}
@keyframes fadeSlideUp {
    from { opacity:0; transform:translateY(14px); }
    to   { opacity:1; transform:translateY(0); }
}
@keyframes shimmer {
    0%   { background-position: -200% center; }
    100% { background-position: 200% center; }
}
@keyframes blink {
    0%,100% { opacity:1; }
    50%      { opacity:0.3; }
}

/* ── Header ── */
.churn-header {
    background: linear-gradient(140deg, #0c0c1e 0%, #110d2e 40%, #091825 100%);
    border: 1px solid #201840;
    border-radius: 20px;
    padding: 2.8rem 3.2rem;
    margin-bottom: 1.6rem;
    position: relative;
    overflow: hidden;
    animation: fadeSlideUp 0.6s ease both;
}
.churn-header::before {
    content: '';
    position: absolute; top: -80px; right: -80px;
    width: 300px; height: 300px;
    background: radial-gradient(circle, rgba(100,60,255,0.22) 0%, transparent 65%);
    border-radius: 50%;
    animation: drift 6s ease-in-out infinite;
}
.churn-header::after {
    content: '';
    position: absolute; bottom: -50px; left: 60px;
    width: 200px; height: 200px;
    background: radial-gradient(circle, rgba(0,160,255,0.12) 0%, transparent 65%);
    border-radius: 50%;
    animation: drift 8s ease-in-out infinite reverse;
}
.churn-header h1 {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem; font-weight: 800;
    letter-spacing: -0.04em; color: #fff;
    margin: 0 0 0.4rem 0; position: relative; z-index: 1;
    background: linear-gradient(90deg, #fff 25%, #c0a8ff 55%, #80c8ff 80%);
    background-size: 200% auto;
    -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    background-clip: text;
    animation: shimmer 5s linear infinite;
}
.churn-header p {
    font-family: 'DM Mono', monospace; font-size: 0.74rem;
    color: #6060a0; letter-spacing: 0.14em; text-transform: uppercase;
    margin: 0; position: relative; z-index: 1;
}
.header-row {
    display: flex; justify-content: space-between;
    align-items: flex-start; position: relative; z-index: 1;
}
.header-badge {
    display: inline-flex; align-items: center; gap: 0.5rem;
    background: rgba(100,60,255,0.1); border: 1px solid rgba(100,60,255,0.25);
    color: #8060d8; font-family: 'DM Mono', monospace;
    font-size: 0.66rem; letter-spacing: 0.14em;
    padding: 0.26rem 0.8rem; border-radius: 100px;
    margin-bottom: 1.2rem;
}
.live-dot {
    width: 6px; height: 6px; background: #7040ff;
    border-radius: 50%; display: inline-block;
    animation: blink 1.6s ease infinite;
}
.header-meta { display:flex; flex-direction:column; align-items:flex-end; gap:0.3rem; }
.header-meta-tag { font-family:'DM Mono',monospace; font-size:0.62rem; color:#383060; letter-spacing:0.1em; text-transform:uppercase; }

/* ── Upload section ── */
.upload-label {
    font-family: 'DM Mono', monospace; font-size: 0.64rem;
    font-weight: 500; letter-spacing: 0.14em; text-transform: uppercase;
    color: #5040a0; margin-bottom: 0.7rem;
    display: flex; align-items: center; gap: 0.4rem;
}
[data-testid="stFileUploader"] { background: transparent !important; }
[data-testid="stFileUploader"] > div {
    background: #0e0e1c !important;
    border: 1px solid #1e1e34 !important;
    border-radius: 12px !important;
}
[data-testid="stFileDropzone"] {
    background: #0b0b18 !important;
    border: 1px dashed #282848 !important;
    border-radius: 10px !important;
    transition: border-color 0.25s !important;
}
[data-testid="stFileDropzone"]:hover {
    border-color: #5030c0 !important;
    background: #0e0e22 !important;
}
[data-testid="stFileDropzone"] p,
[data-testid="stFileDropzone"] span { color: #5848a0 !important; }
[data-testid="stFileDropzone"] small { color: #30304a !important; }
[data-testid="stFileDropzone"] svg { fill: #4838a0 !important; opacity: 0.7; }
[data-testid="stFileDropzone"] button {
    background: #181830 !important; color: #8070c0 !important;
    border: 1px solid #2a2850 !important; border-radius: 7px !important;
}
[data-testid="stFileDropzone"] button:hover {
    background: #20204a !important; color: #a090e0 !important;
}

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg, #3a18a0 0%, #5535cc 100%) !important;
    color: #fff !important; border: none !important;
    border-radius: 10px !important;
    font-family: 'Syne', sans-serif !important;
    font-weight: 700 !important; font-size: 0.84rem !important;
    letter-spacing: 0.06em !important; padding: 0.65rem 1.5rem !important;
    transition: all 0.2s ease !important;
    box-shadow: 0 4px 20px rgba(80,48,200,0.35) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 30px rgba(80,48,200,0.5) !important;
}
.stButton > button:active { transform: translateY(0) !important; }
[data-testid="stBaseButton-secondary"] {
    background: #0e0e1c !important; border: 1px solid #242442 !important;
    color: #6858b0 !important; box-shadow: none !important;
}
[data-testid="stBaseButton-secondary"]:hover {
    border-color: #4830b0 !important; color: #9080d0 !important;
    transform: translateY(-2px) !important;
    box-shadow: 0 4px 16px rgba(60,30,160,0.2) !important;
}
.stDownloadButton > button {
    background: #0e0e1c !important; border: 1px solid #242442 !important;
    color: #6858b0 !important; font-family: 'DM Mono', monospace !important;
    font-size: 0.71rem !important; letter-spacing: 0.08em !important;
    box-shadow: none !important;
}

/* ── Cards ── */
.card {
    background: #0e0e1c; border: 1px solid #1c1c2e;
    border-radius: 14px; padding: 1.5rem; margin-bottom: 1rem;
    transition: border-color 0.2s, box-shadow 0.2s;
    animation: fadeSlideUp 0.5s ease both;
}
.card:hover { border-color: #2e2860; box-shadow: 0 4px 28px rgba(60,40,160,0.1); }
.card-title {
    font-family: 'Syne', sans-serif; font-size: 0.68rem;
    font-weight: 700; letter-spacing: 0.15em;
    text-transform: uppercase; color: #5040a0; margin-bottom: 1rem;
}

/* ── Risk display ── */
.risk-display {
    text-align: center; padding: 2.2rem 1rem;
    border-radius: 16px; position: relative; overflow: hidden;
    animation: fadeSlideUp 0.4s ease both;
}
.risk-high   { background: linear-gradient(150deg,#130808,#200c0c); border:1px solid #380f0f; }
.risk-medium { background: linear-gradient(150deg,#151000,#221800); border:1px solid #3a2800; }
.risk-low    { background: linear-gradient(150deg,#001410,#001d17); border:1px solid #003820; }
.risk-value  { font-family:'Syne',sans-serif; font-size:3.2rem; font-weight:800; line-height:1; margin-bottom:0.25rem; }
.risk-high   .risk-value { color:#ff3c3c; }
.risk-medium .risk-value { color:#ffa014; }
.risk-low    .risk-value { color:#24d468; }
.risk-label  { font-family:'DM Mono',monospace; font-size:0.66rem; letter-spacing:0.18em; text-transform:uppercase; opacity:0.5; }
.risk-prob   { font-family:'DM Mono',monospace; font-size:1.05rem; margin-top:0.5rem; opacity:0.72; }
.prob-bar-wrap { background:#181828; border-radius:100px; height:5px; margin:1.2rem 0 0; overflow:hidden; }
.prob-bar-fill-high   { background:linear-gradient(90deg,#c01818,#ff6060); border-radius:100px; height:100%; }
.prob-bar-fill-medium { background:linear-gradient(90deg,#d07000,#ffc040); border-radius:100px; height:100%; }
.prob-bar-fill-low    { background:linear-gradient(90deg,#00a040,#48ffaa); border-radius:100px; height:100%; }

/* ── Stat pills ── */
.stat-row  { display:flex; gap:0.6rem; flex-wrap:wrap; margin:0.8rem 0; }
.stat-pill { background:#101020; border:1px solid #1e1e38; border-radius:10px; padding:0.65rem 1rem; flex:1; min-width:88px; text-align:center; transition:border-color 0.2s; }
.stat-pill:hover  { border-color:#383070; }
.stat-pill-val    { font-family:'Syne',sans-serif; font-size:1.2rem; font-weight:700; color:#8860e0; }
.stat-pill-label  { font-family:'DM Mono',monospace; font-size:0.59rem; letter-spacing:0.1em; text-transform:uppercase; color:#383060; margin-top:0.15rem; }

/* ── Priority badges ── */
.priority-badge    { display:inline-block; font-family:'DM Mono',monospace; font-size:0.68rem; letter-spacing:0.1em; text-transform:uppercase; padding:0.28rem 0.85rem; border-radius:100px; font-weight:500; }
.priority-critical { background:rgba(240,30,30,0.1);  border:1px solid rgba(240,30,30,0.26);  color:#ff4848; }
.priority-high     { background:rgba(255,130,0,0.1);  border:1px solid rgba(255,130,0,0.26);  color:#ffa020; }
.priority-medium   { background:rgba(50,130,255,0.1); border:1px solid rgba(50,130,255,0.26); color:#60a0ff; }
.priority-low      { background:rgba(30,190,80,0.1);  border:1px solid rgba(30,190,80,0.26);  color:#40cc70; }

/* ── Decision box ── */
.decision-box { background:linear-gradient(140deg,#0d0d1c,#10102a); border:1px solid #222058; border-left:3px solid #6030f0; border-radius:12px; padding:1.2rem 1.4rem; display:flex; flex-direction:column; gap:0.75rem; }
.dec-row   { display:flex; flex-direction:column; gap:0.2rem; padding:0.6rem 0.8rem; background:rgba(80,40,180,0.07); border-radius:8px; border-left:2px solid rgba(100,60,220,0.3); }
.dec-key   { font-family:'DM Mono',monospace; font-size:0.6rem; letter-spacing:0.14em; text-transform:uppercase; color:#4838a0; }
.dec-val   { font-family:'DM Sans',sans-serif; font-size:0.9rem; color:#b8b0de; line-height:1.5; }
.dec-action{ color:#a090ff; font-weight:500; }

/* ── Explanation box ── */
.explanation-box { background:linear-gradient(140deg,#080812,#0c0c1e); border:1px solid #181830; border-radius:12px; padding:1.3rem 1.5rem; position:relative; overflow:hidden; }
.explanation-box::before { content:''; position:absolute; top:0; left:0; right:0; height:2px; background:linear-gradient(90deg,transparent,rgba(100,60,220,0.4),transparent); }
.explanation-text  { font-family:'DM Sans',sans-serif; font-size:0.88rem; line-height:1.85; color:#9888c0; font-style:italic; }
.explanation-quote { font-family:'DM Mono',monospace; font-size:1.4rem; color:rgba(100,60,220,0.3); line-height:1; margin-bottom:0.3rem; display:block; }

/* ── Feature bars ── */
.feat-row      { display:flex; align-items:center; margin-bottom:0.72rem; gap:0.8rem; }
.feat-name     { font-family:'DM Mono',monospace; font-size:0.66rem; color:#706898; width:155px; flex-shrink:0; text-transform:uppercase; letter-spacing:0.04em; }
.feat-bar-wrap { flex:1; background:#121228; border-radius:100px; height:5px; overflow:hidden; }
.feat-bar-fill { height:100%; border-radius:100px; background:linear-gradient(90deg,#3a18b0,#8858ff); }
.feat-val      { font-family:'DM Mono',monospace; font-size:0.64rem; color:#504880; width:42px; text-align:right; flex-shrink:0; }

/* ── Misc ── */
.section-divider { border:none; border-top:1px solid #131328; margin:1.8rem 0; }
.model-tag {
    display:inline-flex; align-items:center; gap:0.35rem;
    background:rgba(68,36,180,0.1); border:1px solid rgba(68,36,180,0.2);
    border-radius:6px; padding:0.2rem 0.58rem;
    font-family:'DM Mono',monospace; font-size:0.63rem;
    letter-spacing:0.08em; color:#7058c0;
    margin-right:0.32rem; margin-bottom:0.32rem;
}
.metric-card {
    background:#0e0e1c; border:1px solid #1c1c2e; border-radius:14px;
    padding:1.6rem 1.2rem; text-align:center;
    transition:transform 0.2s, border-color 0.2s, box-shadow 0.2s; cursor:default;
}
.metric-card:hover { transform:translateY(-3px); border-color:#2e2860; box-shadow:0 8px 32px rgba(60,30,200,0.12); }
.metric-val   { font-family:'Syne',sans-serif; font-size:2.2rem; font-weight:800; line-height:1; margin-bottom:0.4rem; }
.metric-label { font-family:'DM Mono',monospace; font-size:0.63rem; letter-spacing:0.12em; text-transform:uppercase; color:#383060; }

[data-testid="stSpinner"] { color:#7040ff !important; }
.stAlert { border-radius:12px !important; }
[data-testid="stCheckbox"] label { color:#8070b8 !important; font-size:0.87rem !important; }
</style>
""", unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class='churn-header'>
    <div class='header-row'>
        <div>
            <div class='header-badge'><span class='live-dot'></span> LIVE SYSTEM</div>
            <h1>Customer Churn Risk Detection</h1>
            <p>Kaggle Dataset &nbsp;·&nbsp; GradientBoosting &nbsp;·&nbsp; ROC-AUC 0.9967 &nbsp;·&nbsp; NLP Ticket Analysis</p>
        </div>
        <div class='header-meta'>
            <span class='header-meta-tag'>🔮 ChurnGuard</span>
            <span class='header-meta-tag'>Early Risk Detection v1.0</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── System info banner ─────────────────────────────────────────────────────────
model_info_path = ROOT / "models" / "churn_model_info.json"
if model_info_path.exists():
    with open(model_info_path) as f:
        _info = json.load(f)
    _roc  = _info.get("validation_scores", {}).get("roc_auc", 0)
    _feat = len(_info.get("selected_features", []))
    _name = _info.get("model_name", "—")
    st.markdown(f"""
    <div style='display:flex;gap:0.6rem;margin-bottom:0.5rem;flex-wrap:wrap;'>
        <span class='model-tag'>🤖 {_name}</span>
        <span class='model-tag'>📈 ROC-AUC {_roc:.4f}</span>
        <span class='model-tag'>🧩 {_feat} features</span>
    </div>
    """, unsafe_allow_html=True)

# ── Tab Navigation ─────────────────────────────────────────────────────────────
tab_analysis, tab_model, tab_about = st.tabs([
    "🏠  Analysis",
    "📊  Model Info",
    "ℹ️  About",
])


# ══════════════════════════════════════════════════════════════════════════════
# TAB: ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════
with tab_analysis:
    col_up1, col_up2 = st.columns(2)
    with col_up1:
        st.markdown("<div class='upload-label'><span>📁</span> Churn Dataset</div>", unsafe_allow_html=True)
        churn_file = st.file_uploader(
            "Upload churn CSV", type=["csv"],
            key="uploader_churn", label_visibility="collapsed",
        )
    with col_up2:
        st.markdown("<div class='upload-label'><span>🎫</span> Support Tickets</div>", unsafe_allow_html=True)
        ticket_file = st.file_uploader(
            "Upload ticket CSV", type=["csv"],
            key="uploader_ticket", label_visibility="collapsed",
        )

    st.markdown("<br>", unsafe_allow_html=True)
    col_opt, col_btn1, col_btn2, _ = st.columns([1.4, 1.2, 1.2, 3])
    with col_opt:
        use_demo = st.checkbox("Use demo data", value=True)
    with col_btn1:
        run_clicked = st.button("▶  Run Analysis", key="run_analysis", use_container_width=True)
    with col_btn2:
        demo_clicked = st.button("⚡  Demo Mode", key="demo_mode", use_container_width=True, type="secondary")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    if run_clicked or demo_clicked:
        try:
            with st.spinner("Running churn analysis pipeline..."):
                from src.data_processing import load_and_clean_churn
                from src.load_models import load_churn_model
                from src.decision_engine import decide
                from src.genai_explainer import explain
                import joblib

                if demo_clicked or use_demo:
                    churn_src  = str(ROOT / "data" / "sample_churn.csv")
                    ticket_src = str(ROOT / "data" / "sample_tickets.csv")
                else:
                    churn_src  = churn_file
                    ticket_src = ticket_file

                if churn_src is None:
                    st.error("Please upload a churn CSV or enable demo data.")
                    st.stop()

                if hasattr(churn_src, "read"):
                    with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as tmp:
                        tmp.write(churn_src.read())
                        tmp_path = tmp.name
                    try:
                        df_churn = load_and_clean_churn(tmp_path, use_saved_mapping=True)
                    finally:
                        Path(tmp_path).unlink(missing_ok=True)
                else:
                    df_churn = load_and_clean_churn(churn_src, use_saved_mapping=True)

                model, scaler, model_info = load_churn_model()
                enc_path = ROOT / "models" / "encoders.pkl"
                if not enc_path.exists():
                    st.error("Encoders not found. Run: `python -m src.train_churn`")
                    st.stop()

                enc_data      = joblib.load(enc_path)
                encoders      = enc_data["encoders"]
                num_cols      = enc_data["numeric_cols"]
                cat_cols      = enc_data["categorical_cols"]
                feature_names = enc_data["feature_names"]

                from src.train_churn import transform_churn_features
                X_full = transform_churn_features(df_churn, scaler, encoders, num_cols, cat_cols)
                X_one  = X_full[:1]
                proba  = float(model.predict_proba(X_one)[0, 1]) if hasattr(model, "predict_proba") else 0.5
                risk_label = "High" if proba >= 0.6 else "Medium" if proba >= 0.4 else "Low"

                ticket_priority = "medium"
                ticket_count    = 0
                if ticket_src is not None:
                    if hasattr(ticket_src, "read"):
                        ticket_src.seek(0)
                        tickets_df = pd.read_csv(io.BytesIO(ticket_src.read()))
                    else:
                        tickets_df = pd.read_csv(ticket_src)
                    ticket_count = len(tickets_df)
                    from src.data_processing import ensure_ticket_priority
                    tickets_df = ensure_ticket_priority(tickets_df)
                    if "ticket_priority" in tickets_df.columns and len(tickets_df["ticket_priority"].mode()):
                        ticket_priority = str(tickets_df["ticket_priority"].mode().iloc[0])

                decision = decide(proba, risk_label, ticket_priority)
                if model_info.get("feature_importances"):
                    imp = model_info["feature_importances"]
                    top_reasons = [f"{k}: {v:.3f}" for k, v in sorted(imp.items(), key=lambda x: -x[1])[:5]]
                else:
                    top_reasons = feature_names[:5] if feature_names else ["N/A"]

                explanation_text = explain(risk_label, proba, ticket_priority, top_reasons)

            # ── Results ──
            risk_class = risk_label.lower()
            bar_color  = "high" if risk_label == "High" else "medium" if risk_label == "Medium" else "low"
            bar_pct    = f"{proba * 100:.1f}%"
            roc = model_info.get("validation_scores", {}).get("roc_auc", 0)
            acc = model_info.get("validation_scores", {}).get("accuracy", 0)

            r1, r2, r3 = st.columns([1, 1.1, 1.8])
            with r1:
                st.markdown(f"""
                <div class='risk-display risk-{risk_class}'>
                    <div class='risk-value'>{risk_label}</div>
                    <div class='risk-label'>Churn Risk Level</div>
                    <div class='risk-prob'>{proba:.1%} probability</div>
                    <div class='prob-bar-wrap'>
                        <div class='prob-bar-fill-{bar_color}' style='width:{bar_pct}'></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with r2:
                priority_cls = ticket_priority.lower()
                st.markdown(f"""
                <div class='card' style='height:100%;'>
                    <div class='card-title'>Key Signals</div>
                    <div style='margin-bottom:1rem;'>
                        <div style='font-family:DM Mono,monospace;font-size:0.6rem;color:#383060;text-transform:uppercase;letter-spacing:0.1em;margin-bottom:0.4rem;'>Ticket Priority</div>
                        <span class='priority-badge priority-{priority_cls}'>{ticket_priority.upper()}</span>
                    </div>
                    <div class='stat-row'>
                        <div class='stat-pill'><div class='stat-pill-val'>{roc:.3f}</div><div class='stat-pill-label'>ROC-AUC</div></div>
                        <div class='stat-pill'><div class='stat-pill-val'>{acc:.3f}</div><div class='stat-pill-label'>Accuracy</div></div>
                    </div>
                    <div class='stat-row'>
                        <div class='stat-pill'><div class='stat-pill-val'>{len(df_churn):,}</div><div class='stat-pill-label'>Records</div></div>
                        <div class='stat-pill'><div class='stat-pill-val'>{ticket_count:,}</div><div class='stat-pill-label'>Tickets</div></div>
                    </div>
                </div>
                """, unsafe_allow_html=True)

            with r3:
                if isinstance(decision, dict):
                    dec_status  = decision.get("final_status", "")
                    dec_summary = decision.get("summary_signal", "")
                    dec_action  = decision.get("recommended_action", decision.get("action", ""))
                else:
                    dec_str = str(decision)
                    try:
                        import ast
                        dec_dict    = ast.literal_eval(dec_str)
                        dec_status  = dec_dict.get("final_status", "")
                        dec_summary = dec_dict.get("summary_signal", "")
                        dec_action  = dec_dict.get("recommended_action", dec_dict.get("action", ""))
                    except Exception:
                        dec_status  = ""
                        dec_summary = dec_str
                        dec_action  = ""

                dec_rows = ""
                if dec_status:
                    dec_rows += f"<div class='dec-row'><span class='dec-key'>Status</span><span class='dec-val'>{dec_status}</span></div>"
                if dec_summary:
                    dec_rows += f"<div class='dec-row'><span class='dec-key'>Signal</span><span class='dec-val'>{dec_summary}</span></div>"
                if dec_action:
                    dec_rows += f"<div class='dec-row'><span class='dec-key'>Action</span><span class='dec-val dec-action'>{dec_action}</span></div>"
                if not dec_rows:
                    dec_rows = f"<div style='color:#9080c0;font-size:0.9rem;'>{decision}</div>"

                st.markdown(f"""
                <div class='card' style='height:100%;'>
                    <div class='card-title'>Decision Engine Output</div>
                    <div class='decision-box'>{dec_rows}</div>
                </div>
                """, unsafe_allow_html=True)

            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
            feat_col, exp_col = st.columns([1, 1])

            with feat_col:
                st.markdown("<div class='card-title'>Top Feature Importances</div>", unsafe_allow_html=True)
                if model_info.get("feature_importances"):
                    imp     = model_info["feature_importances"]
                    top10   = sorted(imp.items(), key=lambda x: -x[1])[:8]
                    max_val = top10[0][1] if top10 else 1
                    bars_html = ""
                    for feat, val in top10:
                        pct   = (val / max_val) * 100
                        short = feat.replace("_", " ").title()[:22]
                        bars_html += f"""
                        <div class='feat-row'>
                            <div class='feat-name'>{short}</div>
                            <div class='feat-bar-wrap'><div class='feat-bar-fill' style='width:{pct:.1f}%'></div></div>
                            <div class='feat-val'>{val:.3f}</div>
                        </div>"""
                    st.markdown(bars_html, unsafe_allow_html=True)

            with exp_col:
                st.markdown("<div class='card-title'>AI Explanation</div>", unsafe_allow_html=True)
                st.markdown(
                    f"<div class='explanation-box'><span class='explanation-quote'>\u201c</span>"
                    f"<div class='explanation-text'>{explanation_text}</div></div>",
                    unsafe_allow_html=True,
                )
                st.markdown("<br>", unsafe_allow_html=True)
                model_name = model_info.get("model_name", "Unknown")
                st.markdown(f"""
                <span class='model-tag'>🤖 {model_name}</span>
                <span class='model-tag'>📅 {model_info.get('training_date','—')[:10]}</span>
                <span class='model-tag'>🌱 Seed {model_info.get('random_seed', 42)}</span>
                """, unsafe_allow_html=True)
                st.markdown("<br>", unsafe_allow_html=True)
                st.download_button(
                    "⬇  Download Model Info JSON",
                    data=json.dumps(model_info, indent=2),
                    file_name="churn_model_info.json",
                    mime="application/json",
                    use_container_width=True,
                )

            if "churn" in df_churn.columns:
                st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
                st.markdown("<div class='card-title'>Dataset Churn Distribution</div>", unsafe_allow_html=True)
                churn_counts = df_churn["churn"].value_counts().rename({0: "Retained", 1: "Churned"})
                chart_df = pd.DataFrame({"Count": churn_counts})
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.bar_chart(chart_df, color=["#7040ff"])
                with c2:
                    total      = churn_counts.sum()
                    churn_rate = churn_counts.get("Churned", 0) / total if total else 0
                    st.markdown(f"""
                    <div class='stat-pill' style='margin-top:1rem;'>
                        <div class='stat-pill-val'>{churn_rate:.1%}</div>
                        <div class='stat-pill-label'>Churn Rate</div>
                    </div>
                    <div class='stat-pill' style='margin-top:0.5rem;'>
                        <div class='stat-pill-val'>{total:,}</div>
                        <div class='stat-pill-label'>Total Records</div>
                    </div>
                    """, unsafe_allow_html=True)

        except FileNotFoundError as e:
            st.error(f"Model or data not found: {e}")
            st.info("Run `python -m src.train_churn` and `python -m src.train_nlp` first.")
        except Exception as e:
            st.error(f"Error: {e}")
            logger.exception("Run analysis")


# ══════════════════════════════════════════════════════════════════════════════
# TAB: MODEL INFO
# ══════════════════════════════════════════════════════════════════════════════
with tab_model:
    st.markdown("<div class='card-title'>Model Performance &amp; Configuration</div>", unsafe_allow_html=True)

    if not model_info_path.exists():
        st.warning("No trained model found. Run `python -m src.train_churn` first.")
    else:
        with open(model_info_path) as f:
            info = json.load(f)

        scores  = info.get("validation_scores", {})
        metrics = [
            ("ROC-AUC",   scores.get("roc_auc",  0), "#8050ff"),
            ("Accuracy",  scores.get("accuracy",  0), "#24cc60"),
            ("F1 Score",  scores.get("f1",        0), "#ffa014"),
            ("Precision", scores.get("precision", 0), "#34b0ff"),
        ]
        c1, c2, c3, c4 = st.columns(4)
        for col, (name, val, color) in zip([c1, c2, c3, c4], metrics):
            with col:
                st.markdown(f"""
                <div class='metric-card'>
                    <div class='metric-val' style='color:{color};'>{val:.4f}</div>
                    <div class='metric-label'>{name}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        col_a, col_b = st.columns(2)
        with col_a:
            st.markdown("<div class='card-title'>Model Details</div>", unsafe_allow_html=True)
            st.markdown(f"""
            <div class='card'>
                <div style='font-family:DM Mono,monospace;font-size:0.77rem;color:#706898;line-height:2.3;'>
                    <b style='color:#b0a8d8;'>Algorithm:</b> {info.get('model_name','—')}<br>
                    <b style='color:#b0a8d8;'>Training Date:</b> {info.get('training_date','—')[:19]}<br>
                    <b style='color:#b0a8d8;'>Random Seed:</b> {info.get('random_seed',42)}<br>
                    <b style='color:#b0a8d8;'>Version:</b> {info.get('version','1.0')}<br>
                    <b style='color:#b0a8d8;'>Features:</b> {len(info.get('selected_features',[]))}
                </div>
            </div>
            """, unsafe_allow_html=True)

        with col_b:
            st.markdown("<div class='card-title'>Feature Importances</div>", unsafe_allow_html=True)
            if info.get("feature_importances"):
                imp   = info["feature_importances"]
                top8  = sorted(imp.items(), key=lambda x: -x[1])[:8]
                max_v = top8[0][1] if top8 else 1
                bars  = ""
                for feat, val in top8:
                    pct  = (val / max_v) * 100
                    bars += f"""
                    <div class='feat-row'>
                        <div class='feat-name'>{feat.replace('_',' ').title()[:20]}</div>
                        <div class='feat-bar-wrap'><div class='feat-bar-fill' style='width:{pct:.1f}%'></div></div>
                        <div class='feat-val'>{val:.3f}</div>
                    </div>"""
                st.markdown(f"<div class='card'>{bars}</div>", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB: ABOUT
# ══════════════════════════════════════════════════════════════════════════════
with tab_about:
    st.markdown("""
    <div class='card'>
        <div class='card-title'>About This System</div>
        <div style='font-family:DM Sans,sans-serif;font-size:0.94rem;line-height:1.9;color:#8878b0;'>
            <b style='color:#c0b8e0;'>ChurnGuard</b> is an end-to-end early customer churn risk detection and
            explanation system built as an academic prototype for machine learning coursework.<br><br>
            The pipeline combines <b style='color:#c0b8e0;'>tabular churn prediction</b> (GradientBoosting,
            ROC-AUC 0.9967), <b style='color:#c0b8e0;'>NLP ticket analysis</b> (TF-IDF + classifier),
            a <b style='color:#c0b8e0;'>rule-based decision engine</b>, and a
            <b style='color:#c0b8e0;'>generative AI explainer</b>.
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("<div class='card-title'>Dataset Sources</div>", unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""
        <div class='card'>
            <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#fff;margin-bottom:0.6rem;'>Churn Dataset</div>
            <div style='font-family:DM Mono,monospace;font-size:0.69rem;color:#4838a0;line-height:2.2;'>
                muhammadshahidazeem/<br>customer-churn-dataset<br>
                <span style='color:#8878c0;'>442,212 records · 12 features</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class='card'>
            <div style='font-family:Syne,sans-serif;font-size:1rem;font-weight:700;color:#fff;margin-bottom:0.6rem;'>Ticket Dataset</div>
            <div style='font-family:DM Mono,monospace;font-size:0.69rem;color:#4838a0;line-height:2.2;'>
                suraj520/<br>customer-support-ticket-dataset<br>
                <span style='color:#8878c0;'>8,469 tickets · 4 features used</span>
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<div class='card-title'>Models Trained</div>", unsafe_allow_html=True)
    models_info = [
        ("Logistic Regression",  "0.9300", "Baseline linear classifier"),
        ("Gaussian Naive Bayes", "0.9661", "Probabilistic baseline"),
        ("Decision Tree",        "0.9939", "Interpretable tree-based"),
        ("K-Nearest Neighbours", "0.9698", "Instance-based learning"),
        ("Random Forest",        "0.9962", "Ensemble of trees"),
        ("Gradient Boosting ★",  "0.9967", "Best model · XGBoost"),
        ("Stacking Classifier",  "0.9955", "Meta-learner ensemble"),
        ("ANN (Keras)",          "0.9950", "Deep neural network"),
    ]
    for name, auc, desc in models_info:
        star = "★" in name
        st.markdown(f"""
        <div class='card' style='padding:0.88rem 1.3rem;margin-bottom:0.4rem;
            {"border-color:#281e60;background:linear-gradient(135deg,#0c0b1e,#0f0d24);" if star else ""}'>
            <div style='display:flex;justify-content:space-between;align-items:center;'>
                <div>
                    <span style='font-family:Syne,sans-serif;font-size:0.87rem;
                        font-weight:{"700" if star else "400"};
                        color:{"#c0b0ff" if star else "#8070b0"};'>{name}</span>
                    <span style='font-family:DM Sans,sans-serif;font-size:0.74rem;
                        color:#303050;margin-left:0.8rem;'>{desc}</span>
                </div>
                <span style='font-family:DM Mono,monospace;font-size:0.8rem;
                    color:{"#9060ff" if star else "#403870"};'>AUC {auc}</span>
            </div>
        </div>
        """, unsafe_allow_html=True)