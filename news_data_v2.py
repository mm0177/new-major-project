"""
news_data_v2.py — Country-specific GDELT DOC news pipeline.

Fetches supply-chain sentiment (tone) + article volume per country
from the GDELT DOC API (global news articles), and outputs a single
'processed_news_data_v2.csv' with columns:
    date, country, sentiment_score, news_volume

Uses GDELT's built-in Average Tone (no FinBERT needed — much more data).
Original news_data.py is UNTOUCHED.
"""
import pandas as pd
import numpy as np
import requests
import io
import time
import os

os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
if "HF_TOKEN" in os.environ:
    del os.environ["HF_TOKEN"]

# ── Configuration ───────────────────────────────────────────────────────────
GDELT_DOC_URL = "https://api.gdeltproject.org/api/v2/doc/doc"
START_DT = "20230101000000"
END_DT   = "20241031000000"

COUNTRIES = ["China", "India", "Italy", "France", "Germany", "Finland", "USA"]

OUTPUT_CSV = "processed_news_data_v2.csv"

# GDELT rate limit: 1 request per 5 seconds
RATE_LIMIT_SECS = 6
MAX_RETRIES = 3


def _gdelt_get(params: dict, retries: int = MAX_RETRIES) -> requests.Response:
    """GET with automatic retry on 429 Too Many Requests."""
    for attempt in range(retries):
        resp = requests.get(GDELT_DOC_URL, params=params, timeout=60)
        if resp.status_code == 429:
            wait = RATE_LIMIT_SECS * (attempt + 2)
            print(f"   ⏳ Rate-limited, waiting {wait}s (attempt {attempt+1}/{retries})...")
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp
    # Final attempt — let it raise
    resp = requests.get(GDELT_DOC_URL, params=params, timeout=60)
    resp.raise_for_status()
    return resp


def fetch_tone(country: str) -> pd.DataFrame:
    """Fetch daily Average Tone for 'supply chain' + country from GDELT DOC API."""
    resp = _gdelt_get({
        "query": f'"supply chain" {country}',
        "mode": "TimelineTone",
        "format": "csv",
        "startdatetime": START_DT,
        "enddatetime": END_DT,
    })

    if len(resp.text) < 50:
        print(f"  ⚠️  No tone data returned for {country}")
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["Date", "Series", "Value"]
    df["country"] = country
    return df


def fetch_volume(country: str) -> pd.DataFrame:
    """Fetch daily raw article volume for 'supply chain' + country from GDELT DOC API."""
    resp = _gdelt_get({
        "query": f'"supply chain" {country}',
        "mode": "TimelineVolRaw",
        "format": "csv",
        "startdatetime": START_DT,
        "enddatetime": END_DT,
    })

    if len(resp.text) < 50:
        print(f"  ⚠️  No volume data returned for {country}")
        return pd.DataFrame()

    df = pd.read_csv(io.StringIO(resp.text))
    df.columns = ["Date", "Series", "Value"]
    df["country"] = country
    return df


def process_news_sentiment_v2():
    print("=" * 60)
    print("STEP 5-v2: COUNTRY-SPECIFIC GDELT DOC API — TONE + VOLUME")
    print("=" * 60)

    all_tone   = []
    all_volume = []

    for country in COUNTRIES:
        print(f"\n📡 Fetching GDELT DOC data for {country} ...")

        # Fetch tone
        tone = fetch_tone(country)
        if not tone.empty:
            all_tone.append(tone)
            print(f"   Tone days: {len(tone)}")
        time.sleep(RATE_LIMIT_SECS)

        # Fetch volume
        vol = fetch_volume(country)
        if not vol.empty:
            all_volume.append(vol)
            non_zero = vol[vol["Value"] > 0]
            print(f"   Volume days (non-zero): {len(non_zero)}")
        time.sleep(RATE_LIMIT_SECS)

    if not all_tone:
        print("❌ No tone data fetched for any country. Aborting.")
        return

    df_tone   = pd.concat(all_tone, ignore_index=True)
    print(f"\n📊 Total tone rows: {len(df_tone)}")

    # ── Aggregate tone → monthly sentiment per country ─────────────────────
    df_tone["date"] = pd.to_datetime(df_tone["Date"])
    # Normalise GDELT tone (roughly -10..+10) to -1..+1 scale
    df_tone["sentiment_score"] = df_tone["Value"] / 10.0

    tone_monthly = (
        df_tone
        .groupby("country")
        .resample("MS", on="date")["sentiment_score"]
        .mean()
        .reset_index()
    )

    # ── Aggregate volume → monthly average per country ─────────────────────
    if all_volume:
        df_volume = pd.concat(all_volume, ignore_index=True)
        print(f"📊 Total volume rows: {len(df_volume)}")
        df_volume["date"] = pd.to_datetime(df_volume["Date"])
        volume_monthly = (
            df_volume
            .groupby("country")
            .resample("MS", on="date")["Value"]
            .mean()
            .reset_index()
            .rename(columns={"Value": "news_volume"})
        )
    else:
        # If volume fetch failed, create placeholder
        print("⚠️  No volume data, using tone count as proxy")
        volume_monthly = (
            df_tone
            .groupby("country")
            .resample("MS", on="date")["Value"]
            .count()
            .reset_index()
            .rename(columns={"Value": "news_volume"})
        )

    # ── Merge tone + volume per (country, month) ──────────────────────────
    final = pd.merge(
        tone_monthly, volume_monthly,
        on=["date", "country"], how="outer",
    )
    final = final.sort_values(["country", "date"]).reset_index(drop=True)

    # ── Save ──────────────────────────────────────────────────────────────
    final.to_csv(OUTPUT_CSV, index=False)
    print(f"\n✅ SUCCESS: '{OUTPUT_CSV}' created with {len(final)} rows.")
    print(f"   Countries: {sorted(final['country'].unique())}")
    print(f"   Date range: {final['date'].min()} → {final['date'].max()}")
    print("\nSample per country (last row):")
    for c in sorted(final['country'].unique()):
        row = final[final['country'] == c].iloc[-1]
        print(f"   {c}: sentiment={row['sentiment_score']:.4f}, volume={row['news_volume']:.4f}")


if __name__ == "__main__":
    process_news_sentiment_v2()
