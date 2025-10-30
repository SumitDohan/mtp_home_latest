# src/model.py
import os
import glob
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import mlflow
import subprocess
import sys
from datetime import datetime

# =========================================================
# Environment Setup
# =========================================================
CI_MODE = os.getenv("CI", "false").lower() == "true"
DOCKER_MODE = os.getenv("DOCKER", "false").lower() == "true"
USE_DVC = os.getenv("USE_DVC", "true").lower() == "true" and not CI_MODE

# =========================================================
# Project Paths
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
# Utility: Safe DVC Commands
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
# Load latest processed news CSV
# =========================================================
news_files = sorted(glob.glob(os.path.join(PROCESSED_PATH, "processed_news*.csv")))
if not news_files:
    raise FileNotFoundError("❌ No processed news CSV files found in data/processed/")
news_file = news_files[-1]
print(f"📂 Using latest processed news file: {news_file}")

news_df = pd.read_csv(news_file)
if news_df.empty or "title" not in news_df.columns:
    raise ValueError("❌ Processed news DataFrame is empty or missing 'title' column")

news_df = news_df[news_df["title"].notnull() & (news_df["title"].str.strip() != "")]

# =========================================================
# Load FinBERT Model
# =========================================================
print("🔍 Loading FinBERT model...")
tokenizer = AutoTokenizer.from_pretrained("yiyanghkust/finbert-tone")
model = AutoModelForSequenceClassification.from_pretrained("yiyanghkust/finbert-tone")
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()
labels = ["negative", "neutral", "positive"]
print(f"✅ FinBERT model loaded on {device.upper()}.")

# =========================================================
# Sentiment prediction functions
# =========================================================
def predict_label(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return labels[torch.argmax(probs)]

def predict_score(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True, max_length=128).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
    probs = torch.nn.functional.softmax(outputs.logits, dim=1)
    return (probs[0][2] - probs[0][0]).item()  # positive - negative

# =========================================================
# Run Sentiment Analysis
# =========================================================
print("🔬 Running sentiment analysis on news headlines...")
news_df["sentiment_label"] = news_df["title"].apply(predict_label)
news_df["sentiment_score"] = news_df["title"].apply(predict_score)

# =========================================================
# Compute metrics & investment advice
# =========================================================
avg_sentiment = news_df["sentiment_score"].mean()
label_counts = news_df["sentiment_label"].value_counts().to_dict()
total_articles = sum(label_counts.values())
percent_positive = (label_counts.get("positive", 0) / total_articles) * 100
percent_neutral = (label_counts.get("neutral", 0) / total_articles) * 100
percent_negative = (label_counts.get("negative", 0) / total_articles) * 100

if 25 <= percent_positive and 25 <= percent_neutral:
    investment_advice = "GOOD TIME TO BUY STOCKS"
elif 45 <= percent_neutral <= 50:
    investment_advice = "NORMAL DAY TO DO TRANSACTIONS"
elif percent_negative > 40:
    investment_advice = "NOT GOOD TO INVEST"
else:
    investment_advice = "MIXED SIGNALS — USE CAUTION"

print(f"📈 Investment Advice: {investment_advice}")

# =========================================================
# Save timestamped sentiment file
# =========================================================
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_file = os.path.join(PROCESSED_PATH, f"processed_news_sentiment_{timestamp}.csv")
news_df.to_csv(output_file, index=False)
print(f"✅ Sentiment predictions saved to {output_file}")

# =========================================================
# DVC Tracking
# =========================================================
def track_with_dvc(file_path):
    if not USE_DVC or DOCKER_MODE or CI_MODE:
        print(f"⚠️ Skipping DVC tracking for {file_path} inside Docker/CI.")
        return
    try:
        safe_dvc_command([sys.executable, "-m", "dvc", "add", file_path])
        subprocess.run(["git", "add", f"{file_path}.dvc"], check=True)
        subprocess.run(["git", "commit", "-m", f"Track {os.path.basename(file_path)} with DVC"], check=False)
        print(f"✅ {file_path} tracked with DVC and Git commit attempted.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Failed to track {file_path} with DVC/Git: {e}")

track_with_dvc(output_file)

# =========================================================
# Optional: Push to DVC remote
# =========================================================
if USE_DVC and not (DOCKER_MODE or CI_MODE):
    try:
        subprocess.run([sys.executable, "-m", "dvc", "push"], check=True)
        print(f"📤 Pushed {output_file} to default DVC remote.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Failed to push {output_file} to DVC remote: {e}")

# =========================================================
# Log to MLflow
# =========================================================
with mlflow.start_run(run_name=f"News_Sentiment_Analysis_{timestamp}"):
    mlflow.log_param("num_articles", len(news_df))
    mlflow.log_param("investment_advice", investment_advice)
    mlflow.log_param("device", device)
    mlflow.log_metric("avg_sentiment", avg_sentiment)
    mlflow.log_metric("percent_positive", percent_positive)
    mlflow.log_metric("percent_neutral", percent_neutral)
    mlflow.log_metric("percent_negative", percent_negative)
    for label, count in label_counts.items():
        mlflow.log_metric(f"count_{label}", count)
    mlflow.log_artifact(news_file, artifact_path="input_news")
    mlflow.log_artifact(output_file, artifact_path="results")

print("✅ Sentiment analysis complete and logged to MLflow.")
