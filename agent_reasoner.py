import pandas as pd
import joblib
from groq import Groq

# 1. Initialize Groq Client with your key
client = Groq(api_key="")

def run_genai_agent(country_name):
    print(f"--- GENAI STRATEGIC BRIEFING FOR {country_name.upper()} ---")
    
    # 2. Load the Fused Data and trained Model Engine
    try:
        df = pd.read_csv('final_fused_dataset.csv')
        model = joblib.load('risk_engine.pkl')
    except FileNotFoundError:
        print("❌ Error: Run '6_final_training.py' first to generate required files.")
        return

    # 3. Get the latest data point for the selected country
    country_data = df[df['country'] == country_name].iloc[-1:]
    if country_data.empty:
        print(f"❌ Country '{country_name}' not found in dataset.")
        return

    # Prepare features for the XGBoost model
    features = country_data[['sentiment_score', 'news_volume', 'prcp', 'tavg', 'wspd']]
    
    # Get traditional ML prediction
    risk_prob = model.predict_proba(features)[0][1]
    sentiment = country_data['sentiment_score'].values[0]
    volume = country_data['news_volume'].values[0]

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

    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile", # Groq's high-performance model
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=1024,
        top_p=1,
        stream=False
    )

    # 5. Output the result
    print("\n[AI AGENT RESPONSE]")
    print(completion.choices[0].message.content)

if __name__ == "__main__":
    # Example: Run for a specific country from your master_trade_data
    run_genai_agent("India")