"""
app_v2.py — Streamlit UI using the country-specific v2 pipeline.

Uses: agent_reasoner_v2 → risk_engine_v2.pkl + final_fused_dataset_v2.csv
Original app.py is UNTOUCHED.

Run with:  streamlit run app_v2.py
"""
import streamlit as st
import pandas as pd
from agent_reasoner_v2 import run_genai_agent

st.set_page_config(
    page_title="Supply Chain Risk | GenAI Agent v2",
    page_icon="🌐",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Global ── */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
}
section[data-testid="stSidebar"] {
    background: #1a1a2e;
    border-right: 1px solid #30305a;
}
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMarkdown p,
section[data-testid="stSidebar"] .stMarkdown li {
    color: #c5c6d0;
}

/* ── Metric cards ── */
div[data-testid="stMetric"] {
    background: rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 12px;
    padding: 20px 16px;
    text-align: center;
}
div[data-testid="stMetric"] label {
    color: #9ca3af !important;
    font-size: 0.85rem !important;
    letter-spacing: 0.05em;
    text-transform: uppercase;
}
div[data-testid="stMetric"] [data-testid="stMetricValue"] {
    font-size: 2rem !important;
    font-weight: 700 !important;
}

/* ── Briefing container ── */
.briefing-box {
    background: rgba(255,255,255,0.04);
    border: 1px solid rgba(255,255,255,0.08);
    border-radius: 14px;
    padding: 28px 32px;
    margin-top: 8px;
    line-height: 1.75;
    color: #e0e0e0;
}
.briefing-box h1, .briefing-box h2, .briefing-box h3,
.briefing-box h4, .briefing-box strong {
    color: #ffffff;
}
.briefing-box ul, .briefing-box ol {
    padding-left: 1.4em;
}
.briefing-box li {
    margin-bottom: 6px;
}

/* ── Risk badge colours ── */
.risk-high   { color: #ef4444; }
.risk-medium { color: #f59e0b; }
.risk-low    { color: #22c55e; }

/* ── Generate button ── */
.stButton > button {
    background: linear-gradient(90deg,#6366f1,#8b5cf6);
    color: #fff;
    border: none;
    border-radius: 8px;
    padding: 0.55em 2.4em;
    font-weight: 600;
    transition: transform 0.15s;
}
.stButton > button:hover {
    transform: scale(1.04);
    color: #fff;
}
</style>
""", unsafe_allow_html=True)


# ── Data ────────────────────────────────────────────────────────────────────
@st.cache_data
def load_dataframe():
    return pd.read_csv('final_fused_dataset_v2.csv')


def risk_level(prob: float):
    """Return a human label + CSS class for the risk probability."""
    if prob >= 0.65:
        return "High", "risk-high"
    elif prob >= 0.40:
        return "Medium", "risk-medium"
    return "Low", "risk-low"


# ── Main ────────────────────────────────────────────────────────────────────
def main():
    try:
        df = load_dataframe()
    except FileNotFoundError:
        st.error("Missing `final_fused_dataset_v2.csv`. Run `fusion_xb_boost_v2.py` first.")
        return

    countries = sorted(df['country'].dropna().unique())

    with st.sidebar:
        st.image("https://img.icons8.com/fluency/96/globe--v1.png", width=56)
        st.markdown("### 🌐 Supply Chain Risk")
        st.markdown("Select a country and generate a **GenAI** risk briefing.")
        st.markdown("---")
        country = st.selectbox("Country", countries)
        generate = st.button("🚀 Generate Briefing", use_container_width=True)
        st.markdown("---")
        st.caption(
            "Powered by **XGBoost** risk model, **GDELT** country-specific sentiment, "
            "and **LLaMA 3.3‑70B** via Groq."
        )

    # ── Header ──────────────────────────────────────────────────────────────
    st.markdown("# 🌐 Supply Chain Risk — GenAI Agent")
    st.markdown(
        "<p style='color:#9ca3af;margin-top:-12px;'>Real‑time risk intelligence powered by ML &amp; Large Language Models</p>",
        unsafe_allow_html=True,
    )

    if not generate:
        st.info("👈 Select a country from the sidebar and click **Generate Briefing** to begin.")
        return

    # ── Generate ────────────────────────────────────────────────────────────
    with st.spinner("Analysing risk factors and generating briefing …"):
        result = run_genai_agent(country)

    if not result.get('ok'):
        st.error(result.get('error'))
        return

    prob = result['risk_prob']
    label, css_cls = risk_level(prob)

    # ── KPI row ─────────────────────────────────────────────────────────────
    st.markdown(f"## Briefing — {country}")

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Risk Probability", f"{prob:.2%}")
    col2.metric("Risk Level", label)
    col3.metric("Sentiment Score", f"{result['sentiment']:.2f}")
    col4.metric("News Volume", f"{result['volume']:,.0f}")

    st.markdown("")

    # ── Agent Briefing (rendered as Markdown) ───────────────────────────────
    st.markdown("### 🤖 GenAI Agent Briefing")
    st.markdown(
        f"<div class='briefing-box'>{_md_to_html(result['ai_response'])}</div>",
        unsafe_allow_html=True,
    )

    # ── Footer ──────────────────────────────────────────────────────────────
    st.markdown("---")
    st.caption(
        "This dashboard uses a locally‑trained XGBoost model (`risk_engine_v2.pkl`), "
        "the country‑specific fused dataset (`final_fused_dataset_v2.csv`), and the "
        "Groq GenAI client configured in `agent_reasoner_v2.py`."
    )


def _md_to_html(md_text: str) -> str:
    """Convert markdown text to HTML for display in the briefing box."""
    try:
        import markdown
        return markdown.markdown(md_text, extensions=['extra', 'nl2br'])
    except ImportError:
        import re
        html = md_text
        html = re.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', html)
        html = html.replace('\n', '<br>')
        return html


if __name__ == "__main__":
    main()
