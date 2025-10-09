# src/news_sentiment.py
import os
import glob
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import mlflow
import subprocess
import sys

# --- Project root paths ---
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW_PATH = os.path.join(PROJECT_ROOT, "data/raw")
PROCESSED_PATH = os.path.join(PROJECT_ROOT, "data/processed")
os.makedirs(PROCESSED_PATH, exist_ok=True)

# --- Load latest news CSV ---
news_files = sorted(glob.glob(os.path.join(RAW_PATH, "news_*.csv")))
if not news_files:
    raise FileNotFoundError("❌ No news CSV files found in data/raw/")
news_file = news_files[-1]
print(f"📂 Using news file: {news_file}")

# --- Load news data ---
news_df = pd.read_csv(news_file)
if news_df.empty or "title" not in news_df.columns:
    raise ValueError("❌ News DataFrame is empty or missing 'title' column")

# --- Clean titles ---
news_df = news_df[news_df["title"].notnull() & (news_df["title"].str.strip() != "")]

# --- Load FinBERT ---
print("🔍 Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
model.eval()
labels = ['negative', 'neutral', 'positive']
print("✅ FinBERT model loaded.")

# --- Sentiment prediction functions ---
def predict_label(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return labels[torch.argmax(probs)]

def predict_score(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return probs[0][2].item() - probs[0][0].item()  # positive - negative

# --- Run sentiment analysis ---
print("🔬 Running sentiment analysis...")
news_df["sentiment_label"] = news_df["title"].apply(predict_label)
news_df["sentiment_score"] = news_df["title"].apply(predict_score)

# --- Metrics ---
avg_sentiment = news_df["sentiment_score"].mean()
label_counts = news_df["sentiment_label"].value_counts().to_dict()
total_articles = sum(label_counts.values())
percent_positive = (label_counts.get("positive", 0) / total_articles) * 100
percent_neutral = (label_counts.get("neutral", 0) / total_articles) * 100
percent_negative = (label_counts.get("negative", 0) / total_articles) * 100

# --- Investment advice ---
if 33 <= percent_positive <= 40 and 25 <= percent_neutral <= 30:
    investment_advice = "GOOD TIME TO BUY STOCKS"
elif 45 <= percent_neutral <= 50:
    investment_advice = "NORMAL DAY to do transactions"
elif percent_negative > 40:
    investment_advice = "NOT GOOD TO INVEST"
else:
    investment_advice = "MIXED SIGNALS — use caution"

# --- Save CSV ---
output_file = os.path.join(PROCESSED_PATH, "processed_news_sentiment.csv")
news_df.to_csv(output_file, index=False)
print(f"✅ Sentiment predictions saved to {output_file}")

# --- Track processed file with DVC ---
def track_with_dvc(file_path):
    try:
        subprocess.run([sys.executable, "-m", "dvc", "add", file_path], check=True)
        subprocess.run(["git", "add", f"{file_path}.dvc"], check=True)
        subprocess.run(["git", "commit", "-m", f"Track {os.path.basename(file_path)} with DVC"], check=True)
        print(f"✅ {file_path} tracked with DVC and committed")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Failed to track {file_path} with DVC: {e}")

track_with_dvc(output_file)

# --- Log to MLflow ---
mlflow.set_tracking_uri("file:/home/sweta/mtp_home_latest/mtp_home_latest/mlruns")
mlflow.set_experiment("Financial_Sentiment_Pipeline")

with mlflow.start_run(run_name="News_Sentiment_Analysis"):
    mlflow.log_param("num_articles", len(news_df))
    mlflow.log_param("investment_advice", investment_advice)
    mlflow.log_metric("avg_sentiment", avg_sentiment)
    mlflow.log_metric("percent_positive", percent_positive)
    mlflow.log_metric("percent_neutral", percent_neutral)
    mlflow.log_metric("percent_negative", percent_negative)
    for label, count in label_counts.items():
        mlflow.log_metric(f"count_{label}", count)
    mlflow.log_artifact(output_file, artifact_path="results")
    mlflow.log_artifact(news_file, artifact_path="raw_news")

print("✅ Sentiment analysis complete and logged to MLflow.")
