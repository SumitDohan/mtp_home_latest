# src/components/ingestion.py
import os
import json
import yfinance as yf
import pandas as pd
import feedparser
from datetime import date
import mlflow
import subprocess
import sys
import argparse

# =========================================================
# CLI arguments for selective download
# =========================================================
parser = argparse.ArgumentParser()
parser.add_argument("--download-news-only", action="store_true", help="Only download news CSV")
parser.add_argument("--download-stock-only", action="store_true", help="Only download stock CSV")
args = parser.parse_args()

# =========================================================
# Environment flags
# =========================================================
CI_MODE = os.getenv("CI", "false").lower() == "true"
DOCKER_MODE = os.getenv("DOCKER", "false").lower() == "true"
USE_DVC = os.getenv("USE_DVC", "true").lower() == "true" and not CI_MODE

# =========================================================
# DVC helper
# =========================================================
def safe_dvc_command(cmd_list):
    if not USE_DVC:
        print(f"⚠️ Skipping DVC command: {' '.join(cmd_list)}")
        return
    try:
        subprocess.run(cmd_list, check=True)
    except subprocess.CalledProcessError as e:
        print(f"⚠️ DVC command failed: {e}")

# =========================================================
# MLflow setup (writable path)
# =========================================================
def get_mlruns_path():
    # Inside Docker or CI, use /app
    if DOCKER_MODE or CI_MODE:
        path = "/app/mlruns"
    else:
        # On host, use home directory
        home = os.path.expanduser("~")
        path = os.path.join(home, "mtp_home_latest", "mlruns")
    return path

mlruns_dir = get_mlruns_path()
os.makedirs(mlruns_dir, exist_ok=True)
mlflow.set_tracking_uri(f"file://{mlruns_dir}")
mlflow.set_experiment("Financial_Sentiment_Pipeline")
print(f"ℹ️ MLflow tracking at: {mlruns_dir}")

# =========================================================
# Config
# =========================================================
ticker = "^NSEI"
query = "Nifty"
start_date = "2025-10-07"
end_date = date.today().isoformat()

repo_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
raw_dir = os.path.join(repo_root, "data", "raw")
os.makedirs(raw_dir, exist_ok=True)

stock_csv_path = os.path.join(raw_dir, "stock.csv")
news_csv_path = os.path.join(raw_dir, "news_NIFTY.csv")
summary_path = os.path.join(raw_dir, "ingestion_summary.json")

# =========================================================
# Fetch stock
# =========================================================
def fetch_stock_data():
    print(f"📥 Downloading stock data for {ticker} from {start_date} to {end_date}")
    df = yf.download(ticker, start=start_date, end=end_date, auto_adjust=True, progress=False)
    if df.empty:
        print("⚠️ No stock data returned. CSV will still be created as empty.")
        df = pd.DataFrame(columns=["Open", "High", "Low", "Close", "Adj Close", "Volume"])
    df.to_csv(stock_csv_path)
    print(f"✅ Stock data saved to {stock_csv_path}")
    return stock_csv_path

# =========================================================
# Fetch news
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
# DVC tracking
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
# MLflow logging
# =========================================================
def log_with_mlflow(stock_path=None, news_path=None):
    with mlflow.start_run(run_name="data_ingestion"):
        if stock_path:
            mlflow.log_artifact(stock_path, artifact_path="raw_data")
        if news_path:
            mlflow.log_artifact(news_path, artifact_path="raw_data")

        mlflow.log_param("ticker", ticker)
        mlflow.log_param("query", query)
        mlflow.log_param("start_date", start_date)
        mlflow.log_param("end_date", end_date)

        df_stock = pd.read_csv(stock_path) if stock_path else pd.DataFrame()
        df_news = pd.read_csv(news_path) if news_path else pd.DataFrame()
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

        print("📦 Data artifacts, params, metrics, and summary logged to MLflow")

# =========================================================
# Main
# =========================================================
if __name__ == "__main__":
    try:
        stock_path, news_path = None, None

        if args.download_stock_only:
            stock_path = fetch_stock_data()
        elif args.download_news_only:
            news_path = fetch_news_data()
        else:
            stock_path = fetch_stock_data()
            news_path = fetch_news_data()

        if stock_path:
            track_with_dvc(stock_path)
        if news_path:
            track_with_dvc(news_path)

        log_with_mlflow(stock_path, news_path)

    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
