import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

def run_final_training():
    print("--- FUSING REAL MULTI-MODAL DATA ---")
    
    # 1. Load your master datasets
    df_trade = pd.read_csv('master_trade_data.csv')
    df_news = pd.read_csv('processed_news_data.csv')
    df_weather = pd.read_csv('master_weather_data.csv')

    # Ensure all dates are aligned to the start of the month
    for df in [df_trade, df_news, df_weather]:
        df['date'] = pd.to_datetime(df['date']).dt.to_period('M').dt.to_timestamp()

    # 2. Merge Data (Fusion)
    # Trade (Country+Date) + Weather (Country+Date) + News (Date-Global)
    df_merged = pd.merge(df_trade, df_weather, on=['date', 'country'], how='left')
    df_merged = pd.merge(df_merged, df_news, on='date', how='left')
    
    # Fill gaps (e.g., if news or weather for a specific month is missing)
    df_merged = df_merged.ffill().fillna(0)

    # 3. Target Engineering (Disruption Label)
    # Target: Predict a >10% drop in trade volume (Risk)
    df_merged['prev_trade'] = df_merged.groupby('country')['trade_value_usd'].shift(1)
    df_merged['pct_change'] = (df_merged['trade_value_usd'] - df_merged['prev_trade']) / df_merged['prev_trade']
    df_merged['risk_label'] = (df_merged['pct_change'] < -0.10).astype(int)
    
    df_final = df_merged.dropna()

    # 4. Features Selection
    # Combined features: BERT sentiment, News Volume, Rain, Temp, Wind
    features = ['sentiment_score', 'news_volume', 'prcp', 'tavg', 'wspd']
    X = df_final[features]
    y = df_final['risk_label']

    # 5. Train XGBoost Model
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6)
    model.fit(X_train, y_train)

    # 6. Save Model for the Agent
    joblib.dump(model, 'risk_engine.pkl')
    df_final.to_csv('final_fused_dataset.csv', index=False)
    
    print(f"\n✅ Model Trained. Accuracy: {accuracy_score(y_test, model.predict(X_test)):.2%}")
    print(classification_report(y_test, model.predict(X_test)))

    # 7. Explainability (SHAP) - Part of your Epic 3
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print("\n✅ SHAP Explainability data generated.")
    
if __name__ == "__main__":
    run_final_training()