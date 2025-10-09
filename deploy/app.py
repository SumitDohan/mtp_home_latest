# src/api/app.py
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
import pandas as pd
import os

# --- Paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # src/
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data/processed")
FEATURES_FILE = os.path.join(PROCESSED_PATH, "features.csv")

# --- FastAPI app ---
app = FastAPI(title="Financial News Sentiment API")

# --- Request model ---
class DateQuery(BaseModel):
    date: str  # format: 'YYYY-MM-DD'

# --- Utility to format HTML table ---
def df_to_html_table(df: pd.DataFrame) -> str:
    html = "<table border='1' style='border-collapse: collapse; text-align: center;'>"
    # Header
    html += "<tr style='background-color: #f2f2f2;'>"
    for col in df.columns:
        html += f"<th>{col}</th>"
    html += "</tr>"
    # Rows
    for _, row in df.iterrows():
        html += "<tr>"
        for col in df.columns:
            html += f"<td>{row[col]}</td>"
        html += "</tr>"
    html += "</table>"
    return html

# --- Endpoints ---
@app.get("/", response_class=HTMLResponse)
def root():
    return """
    <h2>Welcome to the Financial News Sentiment API</h2>
    <ul>
        <li>GET /all_features - Return all feature-engineered daily sentiment data, sorted by date.</li>
        <li>POST /features_by_date - Return feature data for a specific date (send JSON with {'date': 'YYYY-MM-DD'}).</li>
        <li>GET /latest - Return the latest day's feature-engineered sentiment data.</li>
    </ul>
    """

@app.get("/all_features", response_class=HTMLResponse)
def get_all_features():
    if not os.path.exists(FEATURES_FILE):
        raise HTTPException(status_code=404, detail="Features file not found")
    df = pd.read_csv(FEATURES_FILE)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    df = df.sort_values("Date", ascending=False)
    html_table = df_to_html_table(df)
    return f"<h3>All Feature-Engineered Daily Sentiment Data</h3>{html_table}"

@app.get("/latest", response_class=HTMLResponse)
def get_latest():
    if not os.path.exists(FEATURES_FILE):
        raise HTTPException(status_code=404, detail="Features file not found")
    df = pd.read_csv(FEATURES_FILE)
    df["Date"] = pd.to_datetime(df["Date"]).dt.date
    latest_row = df.sort_values("Date", ascending=False).iloc[0:1]
    html_table = df_to_html_table(latest_row)
    return f"<h3>Latest Day's Feature-Engineered Sentiment Data</h3>{html_table}"

@app.post("/features_by_date", response_class=HTMLResponse)
def get_features_by_date(query: DateQuery):
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
    
    html_table = df_to_html_table(rows)
    return f"<h3>Feature-Engineered Sentiment Data for {query_date}</h3>{html_table}"
