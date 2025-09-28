# src/ingestion.py
import os
import json
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
import feedparser
from datetime import date
import mlflow

# --- MLflow Setup ---
mlflow.set_tracking_uri(r"file:D:\mtp_home_latest\mtp_home_latest\mlruns")

mlflow.set_experiment("Financial_Sentiment_Pipeline")  # Set experiment name

# --- Configuration ---
ticker = "^NSEI"
query = "Nifty"  # used only for logging
start_date = "2025-09-15"
end_date = date.today().isoformat()

raw_dir = "data/raw"
os.makedirs(raw_dir, exist_ok=True)

stock_csv_path = os.path.join(raw_dir, "stock.csv")
news_csv_path = os.path.join(raw_dir, "news_NIFTY.csv")
summary_path = os.path.join(raw_dir, "ingestion_summary.json")

# --- Fetch stock data ---
def fetch_stock_data():
    print(f"📥 Downloading stock data for {ticker} from {start_date} to {end_date}")
    df = yf.download(ticker, start=start_date, end=end_date)
    if df.empty:
        raise ValueError("⚠️ No stock data returned.")
    df.to_csv(stock_csv_path)
    print(f"✅ Stock data saved to {stock_csv_path}")
    return stock_csv_path

# --- Fetch news data from Economic Times RSS ---
def fetch_news_data():
    print(f"📰 Fetching news articles from Economic Times RSS for '{query}'")
    RSS_URL = "https://economictimes.indiatimes.com/markets/stocks/news/rssfeeds/1977021501.cms"
    feed = feedparser.parse(RSS_URL)

    articles = []
    for entry in feed.entries[:50]:  # limit to top 50 articles
        articles.append({
            "title": entry.title,
            "link": entry.link,
            "published": entry.published,
            "summary": entry.summary
        })

    # Save as CSV
    df = pd.DataFrame(articles)
    df.to_csv(news_csv_path, index=False)
    print(f"✅ Saved {len(df)} news articles to {news_csv_path}")
    return news_csv_path

# --- MLflow logging ---
def log_with_mlflow(stock_path, news_path):
    with mlflow.start_run(run_name="data_ingestion"):
        # Log artifacts
        mlflow.log_artifact(stock_path, artifact_path="raw_data")
        mlflow.log_artifact(news_path, artifact_path="raw_data")

        # Log parameters
        mlflow.log_param("ticker", ticker)
        mlflow.log_param("query", query)
        mlflow.log_param("start_date", start_date)
        mlflow.log_param("end_date", end_date)

        # Log metrics
        df_stock = pd.read_csv(stock_path)
        df_news = pd.read_csv(news_path)

        mlflow.log_metric("num_stock_records", len(df_stock))
        mlflow.log_metric("num_news_articles", len(df_news))

        # Log tags
        mlflow.set_tag("phase", "data_ingestion")
        mlflow.set_tag("data_source", "yfinance_and_ET_RSS")

        # Log summary JSON
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

        # Optional: Log stock plot
        try:
            df_stock["Close"].plot(title="Closing Prices")
            plot_path = os.path.join(raw_dir, "stock_plot.png")
            plt.savefig(plot_path)
            plt.close()
            mlflow.log_artifact(plot_path, artifact_path="visuals")
        except Exception as e:
            print(f"⚠️ Failed to plot closing prices: {e}")

        print("📦 Data artifacts, params, metrics, and summary logged to MLflow")

# --- Main ---
if __name__ == "__main__":
    try:
        stock_path = fetch_stock_data()
        news_path = fetch_news_data()
        log_with_mlflow(stock_path, news_path)
    except Exception as e:
        print(f"❌ Ingestion failed: {e}")
