"""
agent_reasoner_v2.py — Same logic as agent_reasoner.py but uses the
country-specific v2 model and dataset.

Loads: risk_engine_v2.pkl, final_fused_dataset_v2.csv
Original agent_reasoner.py is UNTOUCHED.
"""
import os
import pandas as pd
import joblib
from groq import Groq

client = Groq(api_key=os.environ.get("GROQ_API_KEY"))


def run_genai_agent(country_name):
    """Generate a structured briefing for `country_name`.

    Returns a dict with keys: `ok` (bool), `error` (str, optional),
    `risk_prob`, `sentiment`, `volume`, `ai_response`.
    """
    try:
        df = pd.read_csv('final_fused_dataset_v2.csv')
        model = joblib.load('risk_engine_v2.pkl')
    except FileNotFoundError:
        return {"ok": False, "error": "Missing required files: final_fused_dataset_v2.csv or risk_engine_v2.pkl"}

    country_data = df[df['country'] == country_name].iloc[-1:]
    if country_data.empty:
        return {"ok": False, "error": f"Country '{country_name}' not found in dataset."}

    # Use the last row for model features (weather, etc.)
    features = country_data[['sentiment_score', 'news_volume', 'prcp', 'tavg', 'wspd']]

    try:
        risk_prob = float(model.predict_proba(features)[0][1])
    except Exception as e:
        return {"ok": False, "error": f"Model prediction failed: {e}"}

    # For display: use rolling average of last 6 months to avoid sparse-data zeros
    country_rows = df[df['country'] == country_name].tail(6)
    sentiment = float(country_rows['sentiment_score'].replace(0, pd.NA).mean(skipna=True))
    volume = float(country_rows['news_volume'].replace(0, pd.NA).mean(skipna=True))
    # If still NaN (all zeros), fall back to 0
    if pd.isna(sentiment):
        sentiment = 0.0
    if pd.isna(volume):
        volume = 0.0

    prompt = f"""You are an AI Supply Chain Risk Strategist.
Analyze the following technical data for {country_name}:

- **Predictive Risk Probability (XGBoost):** {risk_prob:.2%}
- **BERT News Sentiment Score:** {sentiment:.2f} (Scale: -1 to 1)
- **News Attention Volume:** {volume:.4f}

Write a concise, well-structured briefing for the Logistics Manager using Markdown formatting:
1. Start with a short executive summary (2-3 sentences).
2. Under a "## Key Findings" heading, explain each metric and what it means.
3. Under a "## Risk Correlation" heading, explain the correlation between news sentiment and risk probability.
4. Under a "## Mitigation Strategies" heading, suggest exactly two specific, actionable strategies (e.g., rerouting or safety stock).
5. Under a "## Recommendations" heading, provide 3 bullet-point recommendations.

Keep the tone professional and direct. Use bullet points and bold text for clarity. Do NOT include sign-offs, greetings, or classification labels."""

    try:
        completion = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.7,
            max_tokens=1024,
            top_p=1,
            stream=False
        )
        ai_text = completion.choices[0].message.content
    except Exception as e:
        ai_text = f"GenAI request failed: {e}"

    return {
        "ok": True,
        "risk_prob": risk_prob,
        "sentiment": sentiment,
        "volume": volume,
        "ai_response": ai_text,
    }


if __name__ == "__main__":
    out = run_genai_agent("India")
    if not out.get("ok"):
        print("Error:", out.get("error"))
    else:
        print(f"--- GENAI STRATEGIC BRIEFING FOR India ---")
        print(f"Predictive Risk Probability: {out['risk_prob']:.2%}")
        print(f"Sentiment: {out['sentiment']:.2f}")
        print(f"Volume: {out['volume']:.4f}")
        print("\n[AI AGENT RESPONSE]")
        print(out['ai_response'])
