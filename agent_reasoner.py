import pandas as pd
import joblib
from groq import Groq

# 1. Initialize Groq Client with your key
client = Groq(api_key="")

def run_genai_agent(country_name):
    """Generate a structured briefing for `country_name`.

    Returns a dict with keys: `ok` (bool), `error` (str, optional),
    `risk_prob`, `sentiment`, `volume`, `ai_response`.
    """
    # 2. Load the Fused Data and trained Model Engine
    try:
        df = pd.read_csv('final_fused_dataset.csv')
        model = joblib.load('risk_engine.pkl')
    except FileNotFoundError:
        return {"ok": False, "error": "Missing required files: final_fused_dataset.csv or risk_engine.pkl"}

    # 3. Get the latest data point for the selected country
    country_data = df[df['country'] == country_name].iloc[-1:]
    if country_data.empty:
        return {"ok": False, "error": f"Country '{country_name}' not found in dataset."}

    # Prepare features for the XGBoost model
    features = country_data[['sentiment_score', 'news_volume', 'prcp', 'tavg', 'wspd']]

    # Get traditional ML prediction
    try:
        risk_prob = float(model.predict_proba(features)[0][1])
    except Exception as e:
        return {"ok": False, "error": f"Model prediction failed: {e}"}

    sentiment = float(country_data['sentiment_score'].values[0])
    volume = float(country_data['news_volume'].values[0])

    # 4. Use Groq (GenAI) to reason about these numbers
    prompt = f"""
    You are an AI Supply Chain Risk Strategist.
    Analyze the following technical data for {country_name}:

    - Predictive Risk Probability (XGBoost): {risk_prob:.2%}
    - BERT News Sentiment Score: {sentiment:.2f} (Scale: -1 to 1)
    - News Attention Volume: {volume:.4f}

    Provide a professional briefing for the Logistics Manager.
    Explain the correlation between the negative news sentiment and the risk probability.
    Suggest two specific mitigation strategies (e.g., rerouting or safety stock).
    """

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
    # Example: Run for a specific country from your master_trade_data
    out = run_genai_agent("India")
    if not out.get("ok"):
        print("Error:", out.get("error"))
    else:
        print(f"--- GENAI STRATEGIC BRIEFING FOR India ---")
        print(f"Predictive Risk Probability: {out['risk_prob']:.2%}")
        print("\n[AI AGENT RESPONSE]")
        print(out['ai_response'])