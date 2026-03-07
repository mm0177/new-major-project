"""
fusion_xb_boost_v2.py — Data fusion + XGBoost training using COUNTRY-SPECIFIC news.

Reads:
  - master_trade_data.csv     (country + date + trade_value_usd)
  - processed_news_data_v2.csv (country + date + sentiment_score + news_volume)
  - master_weather_data.csv    (country + date + weather features)

Outputs:
  - final_fused_dataset_v2.csv
  - risk_engine_v2.pkl

Original fusion_xb_boost.py is UNTOUCHED.
"""
import pandas as pd
import numpy as np
import xgboost as xgb
import shap
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score


def run_final_training_v2():
    print("=" * 60)
    print("FUSING MULTI-MODAL DATA (v2 — country-specific news)")
    print("=" * 60)

    # 1. Load master datasets
    df_trade   = pd.read_csv('master_trade_data.csv')
    df_news    = pd.read_csv('processed_news_data_v2.csv')
    df_weather = pd.read_csv('master_weather_data.csv')

    # Align all dates to start of month
    for df in [df_trade, df_news, df_weather]:
        df['date'] = pd.to_datetime(df['date']).dt.to_period('M').dt.to_timestamp()

    print(f"Trade rows:   {len(df_trade)}")
    print(f"News rows:    {len(df_news)}")
    print(f"Weather rows: {len(df_weather)}")

    # 2. Merge — Trade + Weather on (date, country), then + News on (date, country)
    df_merged = pd.merge(df_trade, df_weather, on=['date', 'country'], how='left')
    df_merged = pd.merge(df_merged, df_news, on=['date', 'country'], how='left')

    # Fill gaps
    df_merged = df_merged.sort_values(['country', 'date'])
    numeric_cols = df_merged.select_dtypes(include='number').columns
    df_merged[numeric_cols] = df_merged.groupby('country')[numeric_cols].ffill()
    df_merged = df_merged.fillna(0)

    print(f"Merged rows:  {len(df_merged)}")
    print(f"Countries:    {sorted(df_merged['country'].unique())}")

    # 3. Target Engineering — >10% drop in trade = risk
    df_merged['prev_trade'] = df_merged.groupby('country')['trade_value_usd'].shift(1)
    df_merged['pct_change'] = (
        (df_merged['trade_value_usd'] - df_merged['prev_trade']) / df_merged['prev_trade']
    )
    df_merged['risk_label'] = (df_merged['pct_change'] < -0.10).astype(int)

    df_final = df_merged.dropna()

    # Verify per-country sentiment variation
    summary = df_final.groupby('country')[['sentiment_score', 'news_volume']].mean()
    print("\n📊 Average sentiment & volume per country:")
    print(summary.to_string())

    # 4. Features
    features = ['sentiment_score', 'news_volume', 'prcp', 'tavg', 'wspd']
    X = df_final[features]
    y = df_final['risk_label']

    # 5. Train XGBoost
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=6)
    model.fit(X_train, y_train)

    # 6. Save new model + dataset
    joblib.dump(model, 'risk_engine_v2.pkl')
    df_final.to_csv('final_fused_dataset_v2.csv', index=False)

    acc = accuracy_score(y_test, model.predict(X_test))
    print(f"\n✅ Model v2 Trained.  Accuracy: {acc:.2%}")
    print(classification_report(y_test, model.predict(X_test)))

    # 7. SHAP Explainability
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    print("✅ SHAP Explainability data generated.")


if __name__ == "__main__":
    run_final_training_v2()
