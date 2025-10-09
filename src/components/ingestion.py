# src/components/ingestion.py
import os
import json
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import feedparser
from datetime import date
import mlflow
import subprocess
import sys

# =========================================================
# DVC Control: disable inside Docker (if USE_DVC=false)
# =========================================================
USE_DVC = os.getenv("USE_DVC", "true").lower() == "true"

def safe_dvc_command(cmd_list):
    """Run DVC command safely (skip if USE_DVC=false)."""
    if not USE_DVC:
        print(f"⚠️ Skipping DVC command inside Docker: {' '.join(cmd_list)}")
        return
    try:
        subprocess.run(cmd_list, check=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠️ DVC command failed: {e}")

# =========================================================
# MLflow Setup (use user-writable directory)
# =========================================================
mlruns_path = os.getenv(
    "MLFLOW_TRACKING_URI",
    os.path.expanduser("~/mtp_home_latest/mtp_home_latest/mlruns")
)
mlflow.set_tracking_uri(f"file:///{mlruns_path}")
mlflow.set_experiment("Financial_Sentiment_Pipeline")
os.makedirs(os.path.expanduser(mlruns_path.replace("file://", "")), exist_ok=True)

# =========================================================
# Configuration
# =========================================================
ticker = "^NSEI"
query = "Nifty"
start_date = "2025-09-15"
end_date = date.today().isoformat()

# Save data under project folder
repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
raw_dir = os.path.join(repo_root, "data", "raw")
os.makedirs(raw_dir, exist_ok=True)

stock_csv_path = os.path.join(raw_dir, "stock.csv")
news_csv_path = os.path.join(raw_dir, "news_NIFTY.csv")
summary_path = os.path.join(raw_dir, "ingestion_summary.json")

# =========================================================
# Fetch stock data
# =========================================================
def fetch_stock_data():
    print(f"📥 Downloading stock data for {ticker} from {start_date} to {end_date}")
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        print("⚠️ No stock data returned. CSV will still be created as empty.")
    df.to_csv(stock_csv_path)
    print(f"✅ Stock data saved to {stock_csv_path}")
    return stock_csv_path

# =========================================================
# Fetch news data
# =========================================================
def fetch_news_data():
    print(f"📰 Fetching news articles from Economic Times RSS for '{query}'")
    RSS_URL = "https://economictimes.indiatimes.com/markets/stocks/news/rssfeeds/1977021501.cms"
    feed = feedparser.parse(RSS_URL)

    articles = []
    for entry in feed.entries[:50]:
        articles.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.published,
            "summary": entry.summary
        })

    df = pd.DataFrame(articles)
    df.to_csv(news_csv_path, index=False)
    print(f"✅ Saved {len(df)} news articles to {news_csv_path}")
    return news_csv_path

# =========================================================
# Track data files with DVC + Git
# =========================================================
def track_with_dvc(file_path):
    try:
        safe_dvc_command([sys.executable, "-m", "dvc", "add", file_path])

        dvc_file = f"{file_path}.dvc"
        if USE_DVC:
            subprocess.run(["git", "add", dvc_file], check=True)
            subprocess.run(["git", "commit", "-m", f"Track {os.path.basename(file_path)} with DVC"], check=False)
        print(f"✅ DVC tracking completed for {file_path}")
    except Exception as e:
        print(f"⚠️ Skipped or failed DVC tracking for {file_path}: {e}")

# =========================================================
# Log results with MLflow
# =========================================================
def log_with_mlflow(stock_path, news_path):
    with mlflow.start_run(run_name="data_ingestion"):
        mlflow.log_artifact(stock_path, artifact_path="raw_data")
        mlflow.log_artifact(news_path, artifact_path="raw_data")

        mlflow.log_param("ticker", ticker)
        mlflow.log_param("query", query)
        mlflow.log_param("start_date", start_date)
        mlflow.log_param("end_date", end_date)

        df_stock = pd.read_csv(stock_path)
        df_news = pd.read_csv(news_path)
        mlflow.log_metric("num_stock_records", len(df_stock))
        mlflow.log_metric("num_news_articles", len(df_news))

        summary = {
            "ticker": ticker,
            "query": query,
            "start_date": start_date,
            "end_date": end_date,
            "num_stock_records": len(df_stock),
            "num_news_articles": len(df_news)
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        mlflow.log_artifact(summary_path, artifact_path="raw_data")

        # Optional visualization
        try:
            if not df_stock.empty and "Close" in df_stock.columns:
                plt.figure(figsize=(8, 4))
                df_stock["Close"].plot(title="Closing Prices")
                plot_path = os.path.join(raw_dir, "stock_plot.png")
                plt.savefig(plot_path)
                plt.close()
                mlflow.log_artifact(plot_path, artifact_path="visuals")
            else:
                print("⚠️ Stock CSV empty or 'Close' column missing. Skipping plot.")
        except Exception as e:
            print(f"⚠️ Failed to plot closing prices: {e}")

        print("📦 Data artifacts, params, metrics, and summary logged to MLflow")

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    try:
        stock_path = fetch_stock_data()
        news_path = fetch_news_data()

        track_with_dvc(stock_path)
        track_with_dvc(news_path)

        log_with_mlflow(stock_path, news_path)
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
