import streamlit as st
import pandas as pd
from agent_reasoner import run_genai_agent

st.set_page_config(page_title="Risk Agent UI", layout="wide")

@st.cache_data
def load_dataframe(path: str = 'final_fused_dataset.csv'):
    return pd.read_csv(path)

def main():
    st.title("Supply Chain Risk -- GenAI Agent")

    # Load dataset for country selection
    try:
        df = load_dataframe()
    except FileNotFoundError:
        st.error("Missing `final_fused_dataset.csv`. Please place it in the app folder.")
        return

    countries = sorted(df['country'].dropna().unique())
    country = st.sidebar.selectbox("Select country", countries)

    st.sidebar.markdown("---")
    st.sidebar.write("Choose a country and click 'Generate Briefing' to get a model + GenAI reasoning.")

    if st.button("Generate Briefing"):
        with st.spinner("Generating briefing..."):
            result = run_genai_agent(country)

        if not result.get('ok'):
            st.error(result.get('error'))
            return

        st.subheader(f"Briefing — {country}")
        col1, col2, col3 = st.columns(3)
        col1.metric("Risk Probability", f"{result['risk_prob']:.2%}")
        col2.metric("Sentiment Score", f"{result['sentiment']:.2f}")
        col3.metric("News Volume", f"{result['volume']:.4f}")

        st.markdown("**GenAI Agent Reasoning**")
        st.text_area("Agent Response", value=result['ai_response'], height=320)

        st.markdown("---")
        st.caption("This UI uses the local `risk_engine.pkl`, `final_fused_dataset.csv`, and the Groq GenAI client configured in `agent_reasoner.py`.")

if __name__ == "__main__":
    main()
