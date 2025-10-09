# src/api/app.py
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import pandas as pd
import os

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data/processed")
FEATURES_FILE = os.path.join(PROCESSED_PATH, "features.csv")

# --- FastAPI app ---
app = FastAPI(title="Financial News Sentiment API",
              description="API to serve feature-engineered daily financial news sentiment data")

# --- Request model ---
class DateQuery(BaseModel):
    date: str  # format: 'YYYY-MM-DD'

# --- Endpoints ---
@app.get("/")
def root():
    """Return welcome message AND all feature data"""
    if not os.path.exists(FEATURES_FILE):
        raise HTTPException(status_code=404, detail="Features file not found")

    df = pd.read_csv(FEATURES_FILE)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df.sort_values("Date")
    
    return {
        "message": "Welcome to the Financial News Sentiment API.",
        "data": df.to_dict(orient="records"),
        "endpoints": {
            "GET /all_features": "Return all feature-engineered daily sentiment data, sorted by date.",
            "POST /features_by_date": "Return feature data for a specific date (send JSON with {'date': 'YYYY-MM-DD'}).",
            "GET /latest": "Return the latest day's feature-engineered sentiment data."
        }
    }

@app.get("/all_features")
def get_all_features() -> list[dict]:
    """Return all feature-engineered daily sentiment data, sorted by date."""
    if not os.path.exists(FEATURES_FILE):
        raise HTTPException(status_code=404, detail="Features file not found")
    
    df = pd.read_csv(FEATURES_FILE)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df.sort_values("Date")
    
    return df.to_dict(orient="records")

@app.post("/features_by_date")
def get_features_by_date(query: DateQuery) -> list[dict]:
    """Return feature data for a specific date."""
    if not os.path.exists(FEATURES_FILE):
        raise HTTPException(status_code=404, detail="Features file not found")
    
    df = pd.read_csv(FEATURES_FILE)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    
    try:
        query_date = pd.to_datetime(query.date).date()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid date format, use YYYY-MM-DD")
    
    rows = df[df["Date"] == query_date]
    if rows.empty:
        raise HTTPException(status_code=404, detail=f"No data found for {query.date}")
    
    return rows.to_dict(orient="records")  # always returns a list

@app.get("/latest")
def get_latest_features():
    """Return the latest day's feature-engineered sentiment data."""
    if not os.path.exists(FEATURES_FILE):
        raise HTTPException(status_code=404, detail="Features file not found")
    
    df = pd.read_csv(FEATURES_FILE)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    
    if df.empty:
        raise HTTPException(status_code=404, detail="No feature data available")
    
    latest_row = df.sort_values("Date", ascending=False).iloc[0]
    return latest_row.to_dict()
