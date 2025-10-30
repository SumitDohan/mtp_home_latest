# src/api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
import glob
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# --- Paths ---
PROCESSED_PATH = "/home/sweta/mtp_home_latest/mtp_home_latest/src/data/processed"

# --- Initialize sentiment analyzer ---
sia = SentimentIntensityAnalyzer()

# --- Helper to get the latest processed file ---
def get_latest_processed_file():
    """Return the path of the most recently created processed CSV file."""
    csv_files = glob.glob(os.path.join(PROCESSED_PATH, "processed_news_*.csv"))
    if not csv_files:
        raise HTTPException(status_code=404, detail=f"No processed news files found in {PROCESSED_PATH}")
    latest_file = max(csv_files, key=os.path.getmtime)
    return latest_file


# --- Helper to compute sentiment ---
def compute_sentiment(text):
    if not isinstance(text, str) or text.strip() == "":
        return 0.0
    return sia.polarity_scores(text)["compound"]


# --- FastAPI app ---
app = FastAPI(
    title="Financial News Sentiment API",
    description="API to serve and analyze daily financial news sentiment data",
    version="1.2.0"
)

# --- Request models ---
class DateQuery(BaseModel):
    date: str  # format: 'YYYY-MM-DD'

class TextQuery(BaseModel):
    text: str  # text input for sentiment analysis


# --- Health check endpoint ---
@app.get("/health")
def health_check():
    return {"status": "running", "message": "FastAPI server is live and healthy."}


# --- Root endpoint ---
@app.get("/")
def root():
    """Return welcome message and preview of latest available data."""
    latest_file = get_latest_processed_file()

    try:
        df = pd.read_csv(latest_file, nrows=5, encoding="utf-8", on_bad_lines="skip")
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading file {latest_file}: {str(e)}")

    return {
        "message": "Welcome to the Financial News Sentiment API!",
        "latest_file": os.path.basename(latest_file),
        "available_endpoints": {
            "GET /health": "Check API health status",
            "GET /all_features": "Get all available processed data",
            "POST /features_by_date": "Get feature data for a specific date",
            "GET /latest": "Get the latest day's sentiment data",
            "POST /analyze_text": "Analyze sentiment of a custom text input"
        },
        "preview": df.head(3).to_dict(orient="records")
    }


# --- Endpoint: Get all processed data ---
@app.get("/all_features")
def get_all_features() -> list[dict]:
    """Return all available processed data with sentiment analysis."""
    latest_file = get_latest_processed_file()
    try:
        df = pd.read_csv(latest_file, encoding="utf-8", on_bad_lines="skip")
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading data: {str(e)}")

    # ✅ Compute sentiment for each row
    if "title" in df.columns:
        df["sentiment_score"] = df["title"].apply(compute_sentiment)
    elif "summary" in df.columns:
        df["sentiment_score"] = df["summary"].apply(compute_sentiment)

    df["sentiment_label"] = df["sentiment_score"].apply(
        lambda x: "Positive" if x > 0.05 else ("Negative" if x < -0.05 else "Neutral")
    )

    return df.to_dict(orient="records")


# --- Endpoint: Get data by specific date ---
@app.post("/features_by_date")
def get_features_by_date(query: DateQuery) -> list[dict]:
    """Return feature data for a specific date with sentiment scores."""
    latest_file = get_latest_processed_file()
    try:
        df = pd.read_csv(latest_file, encoding="utf-8", on_bad_lines="skip")
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading data: {str(e)}")

    if "Date" not in df.columns:
        raise HTTPException(status_code=400, detail="No 'Date' column found in the processed data.")

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce").dt.date
    query_date = pd.to_datetime(query.date).date()

    rows = df[df["Date"] == query_date]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {query.date}")

    # ✅ Add sentiment analysis
    if "title" in rows.columns:
        rows["sentiment_score"] = rows["title"].apply(compute_sentiment)
    elif "summary" in rows.columns:
        rows["sentiment_score"] = rows["summary"].apply(compute_sentiment)

    rows["sentiment_label"] = rows["sentiment_score"].apply(
        lambda x: "Positive" if x > 0.05 else ("Negative" if x < -0.05 else "Neutral")
    )

    return rows.to_dict(orient="records")


# --- Endpoint: Get the latest sentiment data ---
@app.get("/latest")
def get_latest_features():
    """Return the latest day's aggregated sentiment."""
    latest_file = get_latest_processed_file()

    try:
        df = pd.read_csv(latest_file, encoding="utf-8", on_bad_lines="skip")
        df = df.replace([np.inf, -np.inf], np.nan).fillna(0)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error reading data: {str(e)}")

    if df.empty:
        raise HTTPException(status_code=404, detail="Processed file is empty.")

    # ✅ Compute sentiment
    if "title" in df.columns:
        df["sentiment_score"] = df["title"].apply(compute_sentiment)
    elif "summary" in df.columns:
        df["sentiment_score"] = df["summary"].apply(compute_sentiment)
    else:
        raise HTTPException(status_code=400, detail="No suitable text column found for sentiment analysis.")

    avg_sentiment = df["sentiment_score"].mean()
    percent_positive = (df["sentiment_score"] > 0.05).mean() * 100
    percent_negative = (df["sentiment_score"] < -0.05).mean() * 100

    advice = (
        "BUY — Positive market outlook"
        if avg_sentiment > 0.05
        else "SELL — Negative market signals"
        if avg_sentiment < -0.05
        else "HOLD — Neutral sentiment"
    )

    return {
        "Date": str(pd.to_datetime("today").date()),
        "average_sentiment": round(avg_sentiment, 4),
        "percent_positive": round(percent_positive, 2),
        "percent_negative": round(percent_negative, 2),
        "investment_advice": advice,
    }


# --- ✅ NEW Endpoint: Analyze custom text sentiment ---
@app.post("/analyze_text")
def analyze_text(query: TextQuery):
    """Analyze sentiment of a user-provided string."""
    text = query.text.strip()
    if not text:
        raise HTTPException(status_code=400, detail="Input text cannot be empty.")

    score = compute_sentiment(text)
    label = "Positive" if score > 0.05 else "Negative" if score < -0.05 else "Neutral"

    return {
        "input_text": text,
        "sentiment_score": round(score, 4),
        "sentiment_label": label
    }
