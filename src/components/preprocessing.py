# src/preprocessing.py
import os
import glob
import pandas as pd
import mlflow
import subprocess
import sys

# --- MLflow Setup ---
mlflow.set_tracking_uri("file:/home/sweta/mtp_home_latest/mtp_home_latest/mlruns")
mlflow.set_experiment("Financial_Sentiment_Pipeline")

# --- Directory Setup ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(PROJECT_ROOT, "data/raw")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data/processed")
os.makedirs(PROCESSED_PATH, exist_ok=True)

# --- Load latest news CSV ---
news_files = sorted(glob.glob(os.path.join(RAW_PATH, "news_*.csv")))
if not news_files:
    raise FileNotFoundError("❌ No news CSV files found in data/raw/")
news_file = news_files[-1]  # latest file
print(f" Using news file: {news_file}")

# --- Load news data ---
news_df = pd.read_csv(news_file)

# --- Preprocessing (example: keep only relevant columns) ---
processed_df = news_df[["title", "link", "published", "summary"]].copy()

# --- Save processed CSV ---
processed_file = os.path.join(PROCESSED_PATH, f"processed_news.csv")
processed_df.to_csv(processed_file, index=False)
print(f"✅ Processed news saved to {processed_file}")

# --- Track processed file with DVC ---
def track_with_dvc(file_path):
    try:
        subprocess.run([sys.executable, "-m", "dvc", "add", file_path], check=True)
        subprocess.run(["git", "add", f"{file_path}.dvc"], check=True)
        subprocess.run(["git", "commit", "-m", f"Track {os.path.basename(file_path)} with DVC"], check=True)
        print(f"✅ {file_path} tracked with DVC and committed")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Failed to track {file_path} with DVC: {e}")

track_with_dvc(processed_file)

# --- Log to MLflow ---
with mlflow.start_run(run_name="news_preprocessing"):
    mlflow.log_artifact(processed_file, artifact_path="processed_news")
    mlflow.log_artifact(news_file, artifact_path="raw_news")
    mlflow.log_metric("num_news_articles", len(processed_df))

print("✅ News preprocessing complete and logged to MLflow.")
