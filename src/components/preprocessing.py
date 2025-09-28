# src/preprocessing.py
import os
import glob
import pandas as pd
import mlflow

# --- MLflow Setup ---
mlflow.set_tracking_uri("file:/home/sweta/MTPnew/mlruns")
mlflow.set_experiment("Financial_Sentiment_Pipeline")

# --- Directory Setup ---
RAW_PATH = "data/raw"
PROCESSED_PATH = "data/processed"
os.makedirs(PROCESSED_PATH, exist_ok=True)

# --- Load latest news CSV ---
news_files = sorted(glob.glob(os.path.join(RAW_PATH, "news_*.csv")))
if not news_files:
    raise FileNotFoundError("❌ No news CSV files found in data/raw/")
news_file = news_files[-1]  # latest file
print(f"📂 Using news file: {news_file}")

# --- Load news data ---
news_df = pd.read_csv(news_file)

# --- Preprocessing (example: keep only relevant columns) ---
processed_df = news_df[["title", "link", "published", "summary"]].copy()

# --- Save processed CSV ---
processed_file = os.path.join(PROCESSED_PATH, f"processed_news.csv")
processed_df.to_csv(processed_file, index=False)
print(f"✅ Processed news saved to {processed_file}")

# --- Log to MLflow ---
with mlflow.start_run(run_name="news_preprocessing"):
    mlflow.log_artifact(processed_file, artifact_path="processed_news")
    mlflow.log_artifact(news_file, artifact_path="raw_news")
    mlflow.log_metric("num_news_articles", len(processed_df))

print("✅ News preprocessing complete and logged to MLflow.")
