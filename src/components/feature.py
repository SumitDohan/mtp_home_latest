import os
import glob
import pandas as pd
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
# Load Latest News CSV (handles both timestamped and static names)
# =========================================================
pattern_list = [
    os.path.join(RAW_PATH, "news_*.csv"),   # e.g. news_2025-10-30.csv
    os.path.join(RAW_PATH, "news_NIFTY.csv")  # fallback static file
]

news_files = []
for pattern in pattern_list:
    news_files.extend(glob.glob(pattern))

if not news_files:
    raise FileNotFoundError(f"❌ No news CSV files found in {RAW_PATH}")

# Pick the most recently modified file
news_file = max(news_files, key=os.path.getmtime)
print(f"📰 Using latest news file: {news_file}")

# =========================================================
# Load and preprocess news data
# =========================================================
news_df = pd.read_csv(news_file)

# Ensure all expected columns exist
expected_cols = ["title", "link", "published", "summary"]
for col in expected_cols:
    if col not in news_df.columns:
        news_df[col] = ""

processed_df = news_df[expected_cols].copy()

# Add timestamp to processed filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
processed_file = os.path.join(PROCESSED_PATH, f"processed_news_{timestamp}.csv")
processed_df.to_csv(processed_file, index=False)
print(f"✅ Processed news saved to {processed_file}")

# =========================================================
# DVC Tracking
# =========================================================
def track_with_dvc(file_path):
    """Track file using DVC and commit to Git."""
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

track_with_dvc(processed_file)

# Optional: Push to DVC remote (skip inside Docker/CI)
if USE_DVC and not (DOCKER_MODE or CI_MODE):
    try:
        subprocess.run([sys.executable, "-m", "dvc", "push"], check=True)
        print(f"📤 Pushed {processed_file} to default DVC remote.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Failed to push {processed_file} to DVC remote: {e}")

# =========================================================
# Log to MLflow
# =========================================================
with mlflow.start_run(run_name=f"news_preprocessing_{timestamp}"):
    mlflow.log_artifact(news_file, artifact_path="raw_news")
    mlflow.log_artifact(processed_file, artifact_path="processed_news")
    mlflow.log_metric("num_news_articles", len(processed_df))

print("📦 News preprocessing complete and logged to MLflow.")
