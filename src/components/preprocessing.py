# src/preprocessing.py
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
RAW_PATH = os.path.join(repo_root, "data", "raw")
PROCESSED_PATH = os.path.join(repo_root, "data", "processed")
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
# Load Latest Stock CSV
# =========================================================
stock_files = sorted(glob.glob(os.path.join(RAW_PATH, "stock*.csv")))
if not stock_files:
    raise FileNotFoundError(f"❌ No stock CSV files found in {RAW_PATH}")
stock_file = stock_files[-1]
print(f"💹 Using latest stock file: {stock_file}")

# =========================================================
# Preprocess stock data
# =========================================================
stock_df = pd.read_csv(stock_file)

# Optional: Example preprocessing (fill missing values)
stock_df.fillna(method="ffill", inplace=True)

# Add timestamp to processed filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
processed_stock_file = os.path.join(PROCESSED_PATH, f"processed_stock_{timestamp}.csv")
stock_df.to_csv(processed_stock_file, index=False)
print(f"✅ Processed stock data saved to {processed_stock_file}")

# =========================================================
# DVC Tracking
# =========================================================
def track_with_dvc(file_path):
    if not USE_DVC or DOCKER_MODE or CI_MODE:
        print(f"⚠️ Skipping DVC tracking for {file_path} inside Docker/CI.")
        return
    try:
        safe_dvc_command([sys.executable, "-m", "dvc", "add", file_path])
        dvc_file = f"{file_path}.dvc"
        subprocess.run(["git", "add", dvc_file], check=True)
        subprocess.run(["git", "commit", "-m", f"Track {os.path.basename(file_path)} with DVC"], check=False)
        print(f"✅ {file_path} tracked with DVC and Git commit attempted.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Failed to track {file_path} with DVC/Git: {e}")

track_with_dvc(processed_stock_file)

# Optional: Push to DVC remote
if USE_DVC and not (DOCKER_MODE or CI_MODE):
    try:
        subprocess.run([sys.executable, "-m", "dvc", "push"], check=True)
        print(f"📤 Pushed {processed_stock_file} to default DVC remote.")
    except subprocess.CalledProcessError as e:
        print(f"⚠️ Failed to push {processed_stock_file} to DVC remote: {e}")

# =========================================================
# Log to MLflow
# =========================================================
with mlflow.start_run(run_name=f"stock_preprocessing_{timestamp}"):
    mlflow.log_artifact(stock_file, artifact_path="raw_stock")
    mlflow.log_artifact(processed_stock_file, artifact_path="processed_stock")
    mlflow.log_metric("num_stock_records", len(stock_df))

print("📦 Stock preprocessing complete and logged to MLflow.")
