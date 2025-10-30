# src/preprocessing.py
import os
import json
import feedparser
import pandas as pd
from datetime import datetime, timedelta
import mlflow
import subprocess
import sys

# =========================================================
# Environment Flags
# =========================================================
CI_MODE = os.getenv("CI", "false").lower() == "true"
DOCKER_MODE = os.getenv("DOCKER", "false").lower() == "true"
USE_DVC = os.getenv("USE_DVC", "true").lower() == "true" and not CI_MODE

# =========================================================
# Paths
# =========================================================
repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(repo_root, "data/raw")
PROCESSED_PATH = os.path.join(repo_root, "data/processed")
os.makedirs(PROCESSED_PATH, exist_ok=True)

# =========================================================
# MLflow Setup
# =========================================================
if DOCKER_MODE or CI_MODE:
    mlruns_dir = "/app/mlruns"
else:
    mlruns_dir = os.path.join(repo_root, "mlruns")

os.makedirs(mlruns_dir, exist_ok=True)
mlflow.set_tracking_uri(f"file://{mlruns_dir}")
mlflow.set_experiment("Financial_Sentiment_Pipeline")

print(f"ℹ️ MLflow tracking at: {mlruns_dir}")

# =========================================================
# Utility: Safe DVC Command
# =========================================================
def safe_dvc_command(cmd_list):
    if not USE_DVC or DOCKER_MODE or CI_MODE:
        print(f"⚠️ Skipping DVC command: {' '.join(cmd_list)}")
        return
    try:
        subprocess.run(cmd_list, check=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠️ DVC command failed: {e}")

# =========================================================
# Function to fetch news from Economic Times RSS
# =========================================================
def fetch_news_from_et(query="Nifty"):
    print(f"📰 Fetching news articles from Economic Times RSS for '{query}'")
    url = f"https://economictimes.indiatimes.com/archivelist.cms?keyword={query}"
    feed = feedparser.parse(f"https://economictimes.indiatimes.com/rssfeeds/2146842.cms")  # Nifty news RSS
    entries = feed.entries[:50]  # Take top 50
    news_list = []
    for e in entries:
        news_list.append({
            "title": e.title,
            "link": e.link,
            "published": e.published if "published" in e else str(datetime.now())
        })
    print(f"✅ Fetched {len(news_list)} news articles.")
    return pd.DataFrame(news_list)

# =========================================================
# Clean & preprocess news data
# =========================================================
def clean_news_data(news_df):
    print("🧹 Cleaning news data...")
    news_df = news_df.drop_duplicates(subset=["title"])
    news_df = news_df[news_df["title"].notnull() & (news_df["title"].str.strip() != "")]
    news_df["title"] = news_df["title"].str.replace(r"[^a-zA-Z0-9\s]", "", regex=True)
    return news_df.reset_index(drop=True)

# =========================================================
# Main Preprocessing Flow
# =========================================================
def main():
    start_date = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    end_date = datetime.now().strftime("%Y-%m-%d")

    # Fetch & clean news data
    news_df = fetch_news_from_et("Nifty")
    news_df = clean_news_data(news_df)

    # Save raw news
    raw_news_path = os.path.join(RAW_PATH, "news_NIFTY.csv")
    news_df.to_csv(raw_news_path, index=False)
    print(f"✅ Saved raw news data to {raw_news_path}")

    # Track raw data with DVC
    safe_dvc_command([sys.executable, "-m", "dvc", "add", raw_news_path])
    subprocess.run(["git", "add", f"{raw_news_path}.dvc"], check=False)
    subprocess.run(["git", "commit", "-m", "Track news_NIFTY.csv with DVC"], check=False)

    # Processed file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    processed_news_path = os.path.join(PROCESSED_PATH, f"processed_news_{timestamp}.csv")
    news_df.to_csv(processed_news_path, index=False)
    print(f"✅ Processed news data saved to {processed_news_path}")

    # Track processed file
    safe_dvc_command([sys.executable, "-m", "dvc", "add", processed_news_path])
    subprocess.run(["git", "add", f"{processed_news_path}.dvc"], check=False)
    subprocess.run(["git", "commit", "-m", f"Track processed_news_{timestamp}.csv with DVC"], check=False)

    # Summary JSON
    summary = {
        "query": "Nifty",
        "start_date": start_date,
        "end_date": end_date,
        "num_news_articles": len(news_df),
        "timestamp": timestamp
    }
    summary_path = os.path.join(RAW_PATH, "ingestion_summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=4)
    print(f"🗓️ Summary saved to {summary_path}")

    # Log to MLflow
    with mlflow.start_run(run_name=f"Preprocessing_{timestamp}"):
        mlflow.log_param("query", "Nifty")
        mlflow.log_param("num_news_articles", len(news_df))
        mlflow.log_param("start_date", start_date)
        mlflow.log_param("end_date", end_date)
        mlflow.log_artifact(raw_news_path, artifact_path="raw_data")
        mlflow.log_artifact(processed_news_path, artifact_path="processed_data")
        mlflow.log_artifact(summary_path, artifact_path="summary")

    print("✅ Preprocessing complete and logged to MLflow.")

if __name__ == "__main__":
    main()
