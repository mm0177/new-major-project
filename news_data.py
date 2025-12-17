import pandas as pd
import numpy as np
from transformers import pipeline
import os

# 1. CLEAN ENVIRONMENT: Bypass expired tokens
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
if "HF_TOKEN" in os.environ: del os.environ["HF_TOKEN"]

def process_news_sentiment():
    print("--- STEP 5: BERT SENTIMENT EXTRACTION ---")
    
    # Load your real GDELT snippets
    try:
        # Use index_col=False to prevent column shifting issues
        df_snippets = pd.read_csv(r'C:\Users\DELL\OneDrive\Desktop\major_project\results-20251217132908.csv', index_col=False)
        print(f"Loaded {len(df_snippets)} snippets.")
    except FileNotFoundError:
        print("❌ File 'results-20251217132908.csv' not found!")
        return

    # 2. Initialize FinBERT
    print("Downloading and initializing FinBERT...")
    sentiment_task = pipeline(
        "sentiment-analysis", 
        model="ProsusAI/finbert",
        token=False
    )

    def get_sentiment(text):
        if pd.isna(text) or text == "": return 0
        try:
            result = sentiment_task(str(text)[:512])[0]
            label = result['label']
            score = result['score']
            return score if label == 'positive' else -score if label == 'negative' else 0
        except Exception:
            return 0

    print("Analyzing snippets (BERT Reasoning)...")
    df_snippets['sentiment_score'] = df_snippets['Snippet'].apply(get_sentiment)

    # 3. Aggregate to Monthly (FIXED: Added numeric_only=True)
    df_snippets['date'] = pd.to_datetime(df_snippets['MatchDateTime'])
    news_monthly = df_snippets.set_index('date').resample('MS').mean(numeric_only=True).reset_index()

    # 4. Merge with Volume Intensity (FIXED: Added numeric_only=True)
    try:
        df_volume = pd.read_csv(r'C:\Users\DELL\OneDrive\Desktop\major_project\results-20251217132843.csv', index_col=False)
        df_volume['date'] = pd.to_datetime(df_volume['Date'])
        vol_monthly = df_volume.set_index('date').resample('MS').mean(numeric_only=True).reset_index()
        
        final_news = pd.merge(news_monthly, vol_monthly[['date', 'Value']], on='date', how='inner')
        final_news.rename(columns={'Value': 'news_volume'}, inplace=True)
        
        final_news.to_csv('processed_news_data.csv', index=False)
        print("\n✅ SUCCESS: 'processed_news_data.csv' created.")
        print(final_news.head())
    except Exception as e:
        print(f"❌ Error merging volume: {e}")

if __name__ == "__main__":
    process_news_sentiment()