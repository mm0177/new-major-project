# Artificial Intelligence-Powered Supply Chain Risk Prediction System

A multi-modal data fusion platform that combines international trade data, global weather patterns, and real-time news sentiment analysis to predict supply chain disruption risks across seven major trading nations. The system uses Extreme Gradient Boosting (XGBoost) for risk classification, the Global Database of Events, Language, and Tone (GDELT) for country-specific news sentiment, and a Generative Artificial Intelligence agent powered by Large Language Model (LLaMA) 3.3-70B through Groq for strategic briefing generation — all delivered through an interactive Streamlit web dashboard.

---

## Table of Contents

1. [Project Overview](#project-overview)
2. [System Architecture](#system-architecture)
3. [Datasets Used](#datasets-used)
4. [Data Points and Features](#data-points-and-features)
5. [Application Flow](#application-flow)
6. [File-by-File Description](#file-by-file-description)
7. [Model Details](#model-details)
8. [Research Metrics and Evaluation](#research-metrics-and-evaluation)
9. [Technology Stack](#technology-stack)
10. [How to Run](#how-to-run)
11. [Directory Structure](#directory-structure)

---

## Project Overview

Global supply chains are vulnerable to disruptions caused by geopolitical tensions, adverse weather events, and shifts in public sentiment. This project builds an end-to-end pipeline that:

- **Ingests multi-modal data** from three distinct sources: international trade statistics, meteorological observations, and worldwide news articles.
- **Fuses the data** at a monthly, country-level granularity to create a unified analytical dataset.
- **Trains an Extreme Gradient Boosting (XGBoost) classifier** to predict whether a country will experience a significant trade disruption (defined as a greater-than-ten-percent month-over-month decline in trade value).
- **Generates strategic briefings** using a Generative Artificial Intelligence agent that interprets the model's predictions alongside sentiment and weather context.
- **Presents results** in a polished Streamlit web dashboard with interactive country selection, key performance indicator cards, and formatted Artificial Intelligence briefings.

### Countries Covered

| Country  | Representative Port City | Latitude  | Longitude  |
|----------|--------------------------|-----------|------------|
| China    | Shanghai                 | 31.2304   | 121.4737   |
| India    | Mumbai                   | 19.0760   | 72.8777    |
| Finland  | Helsinki                 | 60.1699   | 24.9384    |
| Germany  | Hamburg                  | 53.5511   | 9.9937     |
| United States of America | Los Angeles | 34.0522 | -118.2437  |
| Italy    | Genoa                    | 44.4056   | 8.9463     |
| France   | Le Havre                 | 49.4944   | 0.1079     |

### Time Period

All data covers the period from **January 2023 through October 2024** (twenty-two months).

---

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA INGESTION LAYER                         │
├──────────────────┬──────────────────┬───────────────────────────────┤
│  Trade Data      │  Weather Data    │  News Sentiment Data          │
│  (United Nations │  (Meteostat      │  (Global Database of Events,  │
│   Comtrade)      │   Open Weather)  │   Language, and Tone          │
│                  │                  │   Document Application        │
│  csv_combiner.py │  weather_data.py │   Programming Interface)      │
│                  │                  │                               │
│                  │                  │   news_data_v2.py             │
├──────────────────┴──────────────────┴───────────────────────────────┤
│                        DATA FUSION LAYER                            │
│                                                                     │
│  fusion_xb_boost_v2.py                                              │
│  - Merges all three data sources on (date, country)                 │
│  - Engineers target variable (>10% trade decline = risk)            │
│  - Trains XGBoost classifier                                        │
│  - Outputs: final_fused_dataset_v2.csv + risk_engine_v2.pkl        │
├─────────────────────────────────────────────────────────────────────┤
│                    INTELLIGENCE LAYER                                │
│                                                                     │
│  agent_reasoner_v2.py                                               │
│  - Loads trained model + fused dataset                              │
│  - Predicts risk probability per country                            │
│  - Calls Groq Large Language Model Application Programming          │
│    Interface (LLaMA 3.3-70B Versatile) for strategic briefing       │
├─────────────────────────────────────────────────────────────────────┤
│                    PRESENTATION LAYER                                │
│                                                                     │
│  app_v2.py (Streamlit Web Dashboard)                                │
│  - Country selector dropdown                                        │
│  - Key performance indicator cards (risk, sentiment, volume)        │
│  - Formatted Generative Artificial Intelligence briefing            │
│  - Custom cascading style sheets with glassmorphism design          │
└─────────────────────────────────────────────────────────────────────┘
```

---

## Datasets Used

### 1. International Trade Data (United Nations Comtrade)

- **Source:** United Nations Comtrade database — the world's largest repository of official international trade statistics.
- **Raw Files:** Three comma-separated values files downloaded manually from the United Nations Comtrade portal:
  - `TradeData_12_17_2025_11_45_0.csv`
  - `TradeData_12_17_2025_11_45_23.csv`
  - `TradeData_12_17_2025_11_45_53.csv`
- **Processing Script:** `csv_combiner.py`
- **Output File:** `master_trade_data.csv`
- **Records:** 208 rows (approximately 22 months × 7 countries, with some months having missing data depending on reporting schedules)
- **Granularity:** Monthly, per country
- **Key Columns:**
  - `date_code` — Year and month identifier in the format YYYYMM (for example, "202301" represents January 2023)
  - `country` — The reporting nation (China, Finland, France, Germany, India, Italy, or United States of America)
  - `trade_value_usd` — Total trade value in United States Dollars for that country in that month
  - `date` — Parsed timestamp representing the first day of the month (for example, 2023-01-01)
- **Filtering:** Only monthly frequency records are kept (annual aggregate records with frequency code "A" are excluded)

### 2. Weather Data (Meteostat Open Weather)

- **Source:** Meteostat — an open-source weather data interface that provides historical weather observations from thousands of stations worldwide.
- **Application Programming Interface:** Meteostat Python package (`meteostat.Monthly`)
- **Processing Script:** `weather_data.py`
- **Output File:** `master_weather_data.csv`
- **Records:** 231 rows (monthly observations for seven port cities)
- **Granularity:** Monthly, per country (using the country's primary trade port city as the weather observation point)
- **Key Columns:**
  - `date` — The last day of the observation month (for example, 2023-01-31)
  - `country` — Country name corresponding to the port city
  - `tavg` — Average temperature in degrees Celsius for the month (for example, 6.81 means an average of 6.81 degrees Celsius)
  - `prcp` — Average daily precipitation in millimeters (for example, 1.99 millimeters per day)
  - `wspd` — Average wind speed in kilometers per hour (for example, 13.03 kilometers per hour)
- **Collection Method:** The script queries Meteostat using latitude and longitude coordinates for each port city, retrieves daily observations, and resamples them to monthly averages.

### 3. News Sentiment Data (Global Database of Events, Language, and Tone — Document Application Programming Interface)

- **Source:** Global Database of Events, Language, and Tone (GDELT) Project — monitors news media from nearly every country in every language, identifying events, sentiments, and themes.
- **Application Programming Interface:** GDELT Document Application Programming Interface version 2 (`https://api.gdeltproject.org/api/v2/doc/doc`)
- **Processing Script:** `news_data_v2.py`
- **Output File:** `processed_news_data_v2.csv`
- **Records:** 154 rows (approximately 22 months × 7 countries)
- **Granularity:** Monthly, per country
- **Key Columns:**
  - `country` — The nation for which news sentiment was measured
  - `date` — First day of the month (for example, 2023-01-01)
  - `sentiment_score` — Normalized average news tone on a scale from negative one to positive one. Calculated by dividing the raw GDELT Average Tone (which ranges from approximately negative ten to positive ten) by ten. A positive value indicates generally favorable news coverage about that country's supply chain, while a negative value indicates unfavorable coverage.
  - `news_volume` — Average daily raw article count for that month. This represents how many news articles worldwide mentioned "supply chain" in conjunction with the given country during the measurement period.
- **Query:** For each country, the system searches for `"supply chain" {country_name}` across all global news sources monitored by the Global Database of Events, Language, and Tone.
- **Two Application Programming Interface Modes Used:**
  - `TimelineTone` — Returns daily Average Tone values (the overall sentiment of articles matching the query)
  - `TimelineVolRaw` — Returns daily raw article volume counts
- **Rate Limiting:** The Global Database of Events, Language, and Tone Application Programming Interface enforces a limit of one request every five seconds. The script uses six-second intervals between requests and includes automatic retry logic with exponential backoff for HTTP 429 (Too Many Requests) responses.

### 4. Fused Multi-Modal Dataset

- **Processing Script:** `fusion_xb_boost_v2.py`
- **Output File:** `final_fused_dataset_v2.csv`
- **Records:** 208 rows (one per country per month, matching trade data availability)
- **Granularity:** Monthly, per country
- **This dataset is the result of merging all three upstream sources and adding engineered target features.** All columns are described in detail in the next section.

---

## Data Points and Features

### Input Features Used by the Model

The Extreme Gradient Boosting classifier uses the following five features to predict supply chain risk:

| Feature Name       | Data Type | Source                | Description                                                                                                                                    | Scale / Range                          |
|--------------------|-----------|------------------------|------------------------------------------------------------------------------------------------------------------------------------------------|----------------------------------------|
| `sentiment_score`  | Float     | Global Database of Events, Language, and Tone Document Application Programming Interface | The normalized average tone of global news articles mentioning "supply chain" and the given country. Computed as the raw GDELT tone divided by ten. | Negative one to positive one            |
| `news_volume`      | Float     | Global Database of Events, Language, and Tone Document Application Programming Interface | The average daily count of news articles worldwide mentioning "supply chain" and the given country during the month.                            | Zero to approximately 135,000           |
| `tavg`             | Float     | Meteostat             | Average air temperature in degrees Celsius at the country's primary trade port city for the month.                                              | Approximately negative ten to forty     |
| `prcp`             | Float     | Meteostat             | Average daily precipitation in millimeters at the port city for the month.                                                                      | Zero to approximately fifteen           |
| `wspd`             | Float     | Meteostat             | Average wind speed in kilometers per hour at the port city for the month.                                                                       | Zero to approximately thirty            |

### Engineered Target Variable

| Feature Name       | Data Type | Description                                                                                                                                                                                |
|--------------------|-----------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| `prev_trade`       | Float     | The trade value in United States Dollars from the previous month for the same country (computed using a grouped shift of one).                                                              |
| `pct_change`       | Float     | The percentage change in trade value from the previous month: `(current_trade - previous_trade) / previous_trade`.                                                                          |
| `risk_label`       | Integer   | The binary target variable. Set to one if the percentage change is less than negative 0.10 (indicating a greater-than-ten-percent decline in trade value), otherwise set to zero.            |

### Additional Columns in the Fused Dataset

| Column Name        | Data Type | Description                                                                             |
|--------------------|-----------|-----------------------------------------------------------------------------------------|
| `date_code`        | String    | Year-month identifier in YYYYMM format (for example, "202302")                         |
| `country`          | String    | Country name (China, Finland, France, Germany, India, Italy, or United States of America)|
| `trade_value_usd`  | Float     | Total trade value in United States Dollars for the given country and month               |
| `date`             | Datetime  | Parsed timestamp for the first day of the month                                          |

---

## Application Flow

The system operates in a sequential pipeline with four major stages. Below is a step-by-step description of the entire data flow from raw data collection through to the final web dashboard output.

### Stage 1: Data Collection

#### Step 1.1 — Trade Data Aggregation (`csv_combiner.py`)

1. The script scans the working directory for all files matching the pattern `TradeData*.csv`. These files are manually downloaded from the United Nations Comtrade portal.
2. Each file is loaded with all columns read as string type to prevent numeric codes (like "01" for January) from being misinterpreted.
3. The script filters to keep only monthly frequency records (where `freqCode` equals "M"), discarding annual summaries.
4. Three columns are extracted and renamed:
   - `period` becomes `date_code` (for example, "202301")
   - `reporterDesc` becomes `country`
   - `primaryValue` becomes `trade_value_usd`
5. The `date_code` is parsed into a proper datetime object by appending "01" to create a YYYYMMDD string and parsing with the format `%Y%m%d`.
6. All individual files are concatenated into a single data frame and saved as `master_trade_data.csv`.

#### Step 1.2 — Weather Data Collection (`weather_data.py`)

1. For each of the seven countries, the script defines a geographic coordinate pair representing the country's primary international trade port city (for example, Shanghai at latitude 31.2304, longitude 121.4737 for China).
2. Using the Meteostat Python package, the script retrieves monthly weather observations for each location covering the period from January 2023 through September 2025.
3. The daily observations are resampled to monthly frequency by computing the mean of each variable.
4. Three weather variables are retained: average temperature (`tavg`), precipitation (`prcp`), and wind speed (`wspd`).
5. The country name is appended to each row, and all seven data frames are concatenated and saved as `master_weather_data.csv`.

#### Step 1.3 — News Sentiment Collection (`news_data_v2.py`)

1. For each of the seven countries, the script sends two requests to the Global Database of Events, Language, and Tone Document Application Programming Interface version 2:
   - A `TimelineTone` request to retrieve the daily average sentiment (tone) of all worldwide news articles matching the query `"supply chain" {country_name}`.
   - A `TimelineVolRaw` request to retrieve the daily raw count of matching articles.
2. The application programming interface returns comma-separated values data with three columns: `Date`, `Series` (always "Average Tone" or the volume series name), and `Value`.
3. Rate limiting is enforced by waiting six seconds between each request. If a request returns an HTTP 429 (Too Many Requests) status code, the script automatically retries up to three times with increasing wait intervals (twelve seconds, eighteen seconds, and twenty-four seconds).
4. The raw GDELT Average Tone value (which typically ranges from approximately negative ten to positive ten) is normalized to a negative-one-to-positive-one scale by dividing by ten.
5. Daily values are aggregated to monthly frequency: sentiment scores are averaged and article volumes are averaged per country per month.
6. The tone and volume data are merged on the combination of date and country, and the result is saved as `processed_news_data_v2.csv`.

### Stage 2: Data Fusion and Model Training (`fusion_xb_boost_v2.py`)

1. The three master datasets are loaded: `master_trade_data.csv`, `master_weather_data.csv`, and `processed_news_data_v2.csv`.
2. All date columns are aligned to the first day of each month to ensure consistent merging.
3. The datasets are merged using left joins on the composite key of `(date, country)`:
   - First, trade data is joined with weather data.
   - Then, the result is joined with news sentiment data.
4. Missing values are handled by forward-filling within each country group (so that if a country is missing news data for one month, the previous month's values are carried forward), and any remaining gaps are filled with zero.
5. **Target variable engineering:**
   - For each country, the previous month's trade value is computed using a grouped shift.
   - The percentage change from the previous month is calculated.
   - A binary risk label is created: one if the trade value dropped by more than ten percent compared to the previous month, zero otherwise.
6. Rows with missing values (primarily the first month for each country, which has no "previous month" to compare against) are dropped.
7. The dataset is split into training (eighty percent) and testing (twenty percent) subsets.
8. An Extreme Gradient Boosting classifier is trained with the following hyperparameters:
   - Number of estimators: 100
   - Learning rate: 0.05
   - Maximum depth: 6
9. The trained model is serialized and saved as `risk_engine_v2.pkl` using joblib.
10. The complete fused dataset (with all features, engineered columns, and target labels) is saved as `final_fused_dataset_v2.csv`.
11. Shapley Additive Explanations (SHAP) values are computed using a Tree Explainer for model interpretability.

### Stage 3: Generative Artificial Intelligence Agent Reasoning (`agent_reasoner_v2.py`)

1. When a country name is provided, the agent loads `final_fused_dataset_v2.csv` and `risk_engine_v2.pkl`.
2. The most recent data row for the selected country is extracted.
3. The five model features (`sentiment_score`, `news_volume`, `prcp`, `tavg`, `wspd`) are fed into the XGBoost model to obtain a predicted risk probability (the probability of class one, representing a disruption event).
4. For display purposes, the agent computes a rolling average of sentiment and news volume over the last six months for that country. This smooths out any individual months with sparse or missing data. Zero values are treated as missing to avoid dragging down the average.
5. A structured prompt is constructed containing the risk probability, sentiment score, and news volume, and is sent to the Groq Application Programming Interface using the LLaMA 3.3 70-Billion-Parameter Versatile model.
6. The prompt instructs the Large Language Model to generate a professional briefing in Markdown format with the following sections:
   - **Executive Summary** — two to three sentences providing an overview
   - **Key Findings** — explanation of each metric and its significance
   - **Risk Correlation** — analysis of how news sentiment relates to risk probability
   - **Mitigation Strategies** — exactly two specific, actionable recommendations (such as rerouting shipments or building safety stock)
   - **Recommendations** — three bullet-point action items
7. The function returns a dictionary containing the risk probability, display sentiment, display volume, and the complete Artificial Intelligence-generated briefing text.

### Stage 4: Web Dashboard (`app_v2.py`)

1. The Streamlit application loads the fused dataset on startup and caches it for performance.
2. Custom cascading style sheets are injected to create a modern interface with:
   - A gradient background
   - Glassmorphism-styled metric cards with frosted-glass appearance
   - A styled briefing container with left border accent
3. A sidebar provides a dropdown menu listing all seven countries.
4. When the user selects a country and clicks the "Generate Intelligence Report" button:
   - The `run_genai_agent()` function is called from `agent_reasoner_v2.py`.
   - Four key performance indicator cards are displayed in a row:
     - **Risk Probability** — percentage value from the XGBoost model
     - **Risk Level** — categorized as "High Risk" (greater than or equal to fifty percent), "Medium Risk" (greater than or equal to twenty percent), or "Low Risk" (less than twenty percent)
     - **Sentiment Score** — the rolling six-month average news sentiment for the selected country
     - **News Volume** — the rolling six-month average daily article count
   - The Artificial Intelligence-generated briefing is converted from Markdown to Hypertext Markup Language using the Python `markdown` library (with extra and newline-to-break extensions) and rendered in a styled container.

---

## File-by-File Description

### Core Pipeline (Version 2 — Country-Specific Sentiment)

| File                       | Purpose                                                                                                      |
|----------------------------|--------------------------------------------------------------------------------------------------------------|
| `csv_combiner.py`          | Combines multiple United Nations Comtrade trade data comma-separated values files into `master_trade_data.csv` |
| `weather_data.py`          | Fetches monthly weather data from Meteostat for seven port cities and outputs `master_weather_data.csv`        |
| `news_data_v2.py`          | Fetches country-specific news sentiment and volume from the GDELT Document Application Programming Interface and outputs `processed_news_data_v2.csv` |
| `fusion_xb_boost_v2.py`    | Merges trade, weather, and news data; engineers target variable; trains XGBoost model; outputs `final_fused_dataset_v2.csv` and `risk_engine_v2.pkl` |
| `agent_reasoner_v2.py`     | Loads trained model, predicts risk probability, calls Groq LLaMA 3.3 for strategic briefing generation         |
| `app_v2.py`                | Streamlit web dashboard with country selection, key performance indicator cards, and formatted briefings         |

### Original Pipeline (Version 1 — Global Sentiment)

These files represent the original implementation where news sentiment was not country-specific. They are preserved for reference and comparison.

| File                       | Purpose                                                                                                      |
|----------------------------|--------------------------------------------------------------------------------------------------------------|
| `news_data.py`             | Processes pre-downloaded GDELT Television snippets through FinBERT (Financial Bidirectional Encoder Representations from Transformers) for global sentiment analysis; outputs `processed_news_data.csv` |
| `fusion_xb_boost.py`       | Original data fusion and training using global (non-country-specific) news sentiment                          |
| `agent_reasoner.py`        | Original Generative Artificial Intelligence agent using the version 1 model                                   |
| `app.py`                   | Original Streamlit dashboard using version 1 pipeline                                                         |

### Research and Evaluation

| File                           | Purpose                                                                                                  |
|--------------------------------|----------------------------------------------------------------------------------------------------------|
| `model_metrics_evaluation.py`  | Comprehensive model evaluation: accuracy, precision, recall, F1-score, confusion matrix, receiver operating characteristic curve, area under the curve, precision-recall curve, five-fold cross-validation, feature importance, and Shapley Additive Explanations values |
| `model_metrics_improved.py`    | Improved evaluation pipeline using Synthetic Minority Over-sampling Technique (SMOTE) for class balancing, stratified splits, and scale positive weight adjustment |
| `shap_values.py`               | Generates Shapley Additive Explanations summary plots for model transparency and interpretability          |

### Data Files

| File                             | Records | Description                                                                                |
|----------------------------------|---------|--------------------------------------------------------------------------------------------|
| `master_trade_data.csv`          | 208     | Monthly trade values in United States Dollars for seven countries                           |
| `master_weather_data.csv`        | 231     | Monthly temperature, precipitation, and wind speed for seven port cities                    |
| `processed_news_data_v2.csv`     | 154     | Monthly country-specific news sentiment and volume from Global Database of Events, Language, and Tone |
| `final_fused_dataset_v2.csv`     | 208     | Merged multi-modal dataset with engineered risk labels (used by model and dashboard)        |
| `processed_news_data.csv`        | ~22     | Original global (non-country-specific) news sentiment from FinBERT analysis                  |
| `final_fused_dataset.csv`        | ~182    | Original merged dataset using global sentiment                                               |
| `risk_engine_v2.pkl`             | —       | Serialized XGBoost classifier (version 2, country-specific sentiment)                        |
| `risk_engine.pkl`                | —       | Serialized XGBoost classifier (version 1, global sentiment)                                  |

---

## Model Details

### Algorithm: Extreme Gradient Boosting (XGBoost) Classifier

- **Number of Estimators:** 100 boosting rounds
- **Learning Rate:** 0.05 (controls the step size at each boosting iteration)
- **Maximum Tree Depth:** 6 (limits the depth of each decision tree to prevent overfitting)
- **Train/Test Split:** Eighty percent training, twenty percent testing (random state 42 for reproducibility)

### Target Definition

A country-month is labeled as "at risk" (risk label equals one) if the trade value in United States Dollars declined by more than ten percent compared to the previous month for the same country. Otherwise, it is labeled as "not at risk" (risk label equals zero).

### Model Accuracy

- **Version 2 (country-specific sentiment):** 85.37 percent accuracy on the held-out test set

### Explainability

Shapley Additive Explanations (SHAP) Tree Explainer is used to compute feature contributions for each prediction, providing transparency into which features are driving the model's risk assessments.

---

## Research Metrics and Evaluation

The `research_metrics/` directory contains detailed evaluation results including:

### Baseline Model Evaluation

- Classification metrics summary (accuracy, precision, recall, F1-score)
- Detailed per-sample predictions
- Confusion matrix visualization
- Receiver Operating Characteristic curve with Area Under the Curve
- Precision-Recall curve
- Five-fold cross-validation scores
- Feature importance rankings
- Shapley Additive Explanations value distributions

### Improved Model with Synthetic Minority Over-sampling Technique

- Addresses class imbalance (far more "no risk" months than "risk" months)
- Uses Synthetic Minority Over-sampling Technique to generate synthetic minority class samples for balanced training
- Uses stratified train/test splits to maintain class proportions
- Applies XGBoost's built-in `scale_pos_weight` parameter
- Generates comparison visualizations of class distributions before and after oversampling

---

## Technology Stack

| Component                          | Technology                                                                                     |
|------------------------------------|-----------------------------------------------------------------------------------------------|
| Programming Language               | Python 3.11                                                                                    |
| Web Framework                      | Streamlit                                                                                      |
| Machine Learning Model             | XGBoost (Extreme Gradient Boosting Classifier)                                                |
| Generative Artificial Intelligence | Groq Application Programming Interface with Meta LLaMA 3.3 70-Billion-Parameter Versatile     |
| News Data Source                   | Global Database of Events, Language, and Tone Document Application Programming Interface v2     |
| Weather Data Source                | Meteostat Open Weather Data                                                                    |
| Trade Data Source                  | United Nations Comtrade International Trade Statistics                                          |
| Model Explainability               | Shapley Additive Explanations (SHAP)                                                           |
| Class Balancing                    | Synthetic Minority Over-sampling Technique (SMOTE) via imbalanced-learn                        |
| Sentiment Analysis (Version 1)     | FinBERT (ProsusAI/finbert via Hugging Face Transformers)                                       |
| Serialization                      | joblib                                                                                         |
| Markdown Rendering                 | Python markdown library with extra and newline-to-break extensions                             |

### Python Dependencies

- `streamlit` — Interactive web application framework
- `pandas` — Data manipulation and analysis
- `numpy` — Numerical computing
- `xgboost` — Gradient boosting machine learning library
- `scikit-learn` — Machine learning utilities (train/test split, metrics, cross-validation)
- `shap` — Shapley Additive Explanations model explainability
- `joblib` — Model serialization and deserialization
- `groq` — Groq Application Programming Interface client for Large Language Model inference
- `requests` — Hypertext Transfer Protocol requests for Global Database of Events, Language, and Tone Application Programming Interface
- `meteostat` — Weather data retrieval
- `matplotlib` — Plotting and visualization
- `seaborn` — Statistical data visualization
- `markdown` — Markdown to Hypertext Markup Language conversion
- `transformers` — Hugging Face library for FinBERT sentiment analysis (version 1 only)
- `imbalanced-learn` — Synthetic Minority Over-sampling Technique implementation

---

## How to Run

### Step 1: Data Collection (run once)

```bash
python csv_combiner.py
python weather_data.py
python news_data_v2.py
```

**Note:** The `news_data_v2.py` script takes approximately two to three minutes to complete due to rate limiting on the Global Database of Events, Language, and Tone Application Programming Interface (six-second intervals between requests for fourteen total requests across seven countries).

### Step 2: Model Training (run once, or re-run after new data)

```bash
python fusion_xb_boost_v2.py
```

### Step 3: Launch the Dashboard

```bash
streamlit run app_v2.py
```

The dashboard will open in your default web browser at `http://localhost:8501`. Select a country from the sidebar dropdown and click "Generate Intelligence Report" to view the risk assessment and Artificial Intelligence-generated strategic briefing.

### Optional: Generate Research Metrics

```bash
python model_metrics_evaluation.py
python model_metrics_improved.py
python shap_values.py
```

---

## Directory Structure

```
major_project/
│
├── app_v2.py                          # Streamlit dashboard (version 2 — country-specific)
├── app.py                             # Streamlit dashboard (version 1 — global sentiment)
├── agent_reasoner_v2.py               # Generative Artificial Intelligence agent (version 2)
├── agent_reasoner.py                  # Generative Artificial Intelligence agent (version 1)
├── news_data_v2.py                    # GDELT Document Application Programming Interface pipeline
├── news_data.py                       # FinBERT sentiment pipeline (version 1)
├── fusion_xb_boost_v2.py             # Data fusion + model training (version 2)
├── fusion_xb_boost.py                # Data fusion + model training (version 1)
├── csv_combiner.py                    # Trade data aggregator
├── weather_data.py                    # Weather data collector
├── model_metrics_evaluation.py        # Baseline model evaluation
├── model_metrics_improved.py          # Improved model with SMOTE evaluation
├── shap_values.py                     # SHAP explainability plots
│
├── master_trade_data.csv              # Aggregated monthly trade data
├── master_weather_data.csv            # Monthly weather observations
├── processed_news_data_v2.csv         # Country-specific news sentiment (GDELT)
├── processed_news_data.csv            # Global news sentiment (FinBERT)
├── final_fused_dataset_v2.csv         # Fused dataset with risk labels (version 2)
├── final_fused_dataset.csv            # Fused dataset with risk labels (version 1)
├── risk_engine_v2.pkl                 # Trained XGBoost model (version 2)
├── risk_engine.pkl                    # Trained XGBoost model (version 1)
│
├── TradeData_12_17_2025_11_45_0.csv   # Raw United Nations Comtrade export
├── TradeData_12_17_2025_11_45_23.csv  # Raw United Nations Comtrade export
├── TradeData_12_17_2025_11_45_53.csv  # Raw United Nations Comtrade export
│
├── research_metrics/                  # Evaluation results and visualizations
│   ├── README.md
│   ├── baseline_model/
│   │   ├── model_metrics_summary.csv
│   │   ├── detailed_predictions.csv
│   │   └── README.md
│   └── improved_model_smote/
│       ├── model_metrics_improved.csv
│       └── README.md
│
└── __pycache__/                       # Python bytecode cache
```
