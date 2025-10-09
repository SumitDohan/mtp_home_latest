# src/feature.py
import os
import glob
import pandas as pd
import mlflow
import subprocess
import sys
from datetime import datetime

# --- Environment setup ---
USE_DVC = os.getenv("USE_DVC", "true").lower() == "true"

# --- Define project paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data/processed")
os.makedirs(PROCESSED_PATH, exist_ok=True)

# --- MLflow setup ---
MLFLOW_TRACKING_URI = os.getenv(
    "MLFLOW_TRACKING_URI",
    f"file:///{os.path.join(PROJECT_ROOT, 'mlruns')}"
)
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("Financial_Sentiment_Pipeline")

# --- Load latest processed news sentiment file ---
news_files = sorted(glob.glob(os.path.join(PROCESSED_PATH, "processed_news_sentiment*.csv")))
if not news_files:
    raise FileNotFoundError("❌ No processed news sentiment CSV found in data/processed/")
news_file = news_files[-1]
print(f"📂 Using processed news sentiment file: {news_file}")

news_df = pd.read_csv(news_file, parse_dates=["published"])
print(f"✅ Loaded {len(news_df)} news articles")

# --- Validate columns ---
required_cols = {"sentiment_score", "sentiment_label", "published"}
missing = required_cols - set(news_df.columns)
if missing:
    raise ValueError(f"❌ Missing columns in {news_file}: {missing}")

# --- Clean and standardize date ---
news_df["published"] = pd.to_datetime(news_df["published"], errors='coerce')
news_df = news_df.dropna(subset=["published"])
news_df["published"] = news_df["published"].dt.date

# --- Aggregate daily sentiment metrics ---
daily_sentiment = news_df.groupby("published").agg(
    avg_sentiment=("sentiment_score", "mean"),
    percent_negative=("sentiment_label", lambda x: (x == "negative").mean() * 100)
).reset_index().rename(columns={"published": "Date"})

# --- Investment advice rules ---
def get_investment_advice(row):
    if row["percent_negative"] >= 50:
        return "HIGH RISK — AVOID INVESTING"
    elif row["percent_negative"] >= 35:
        return "NOT GOOD TO INVEST"
    else:
        return "NORMAL DAY"

daily_sentiment["investment_advice"] = daily_sentiment.apply(get_investment_advice, axis=1)

# --- Save engineered features safely ---
base_file = os.path.join(PROCESSED_PATH, "features.csv")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
features_file = base_file

# If file exists, create a timestamped backup
if os.path.exists(base_file):
    features_file = os.path.join(PROCESSED_PATH, f"features_{timestamp}.csv")

try:
    daily_sentiment.to_csv(features_file, index=False)
    print(f"✅ Feature-engineered dataset saved: {features_file}")
except PermissionError:
    features_file = os.path.join(PROCESSED_PATH, f"features_temp_{timestamp}.csv")
    daily_sentiment.to_csv(features_file, index=False)
    print(f"⚠️ Permission issue detected. Saved to temporary file: {features_file}")

# --- Optional: Track features using DVC ---
def track_with_dvc(file_path):
    if not USE_DVC:
        print(f"⚠️ Skipping DVC tracking for {file_path}")
        return

    try:
        subprocess.run([sys.executable, "-m", "dvc", "add", file_path], check=True)
        subprocess.run(["git", "add", f"{file_path}.dvc"], check=True)
        subprocess.run(["git", "commit", "-m", f"Track {os.path.basename(file_path)} with DVC"], check=False)
        print(f"✅ {file_path} tracked with DVC and Git commit attempted.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ DVC tracking failed for {file_path}: {e}")

track_with_dvc(features_file)

# --- Log with MLflow ---
with mlflow.start_run(run_name=f"feature_engineering_{timestamp}"):
    mlflow.log_artifact(features_file, artifact_path="features")
    mlflow.log_metric("num_days", len(daily_sentiment))
    mlflow.log_metric("avg_daily_negative_percent", daily_sentiment["percent_negative"].mean())

    # --- Identify and log high-negative days ---
    high_neg_days = daily_sentiment[daily_sentiment["percent_negative"] >= 35]
    high_neg_path = os.path.join(PROCESSED_PATH, f"high_negative_days_{timestamp}.txt")

    with open(high_neg_path, "w") as f:
        for _, row in high_neg_days.iterrows():
            f.write(f"{row['Date']} — {row['percent_negative']:.1f}% negative: {row['investment_advice']}\n")

    mlflow.log_artifact(high_neg_path, artifact_path="features")

print("✅ Feature engineering completed and logged to MLflow.")

# --- Print summary of high negative days ---
if not high_neg_days.empty:
    print("\n📉 Days with high negative sentiment:")
    for _, row in high_neg_days.iterrows():
        print(f"  - {row['Date']} — {row['percent_negative']:.1f}% negative: {row['investment_advice']}")
else:
    print("\n✅ No high-negative-sentiment days detected.")
