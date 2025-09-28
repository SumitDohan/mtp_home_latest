# src/run_pipeline.py
from prefect import flow, task
import subprocess
import datetime

# --- Tasks for each component ---
@task
def run_data_ingestion():
    print("📥 Running Data Ingestion...")
    subprocess.run(["python", "components/ingestion.py"], check=True)

@task
def run_preprocessing():
    print("🔄 Running Preprocessing...")
    subprocess.run(["python", "components/preprocessing.py"], check=True)

@task
def run_sentiment_analysis():
    print("📝 Running Sentiment Analysis (FinBERT)...")
    subprocess.run(["python", "components/model.py"], check=True)

@task
def run_feature_engineering():
    print("📊 Running Feature Engineering...")
    subprocess.run(["python", "components/feature.py"], check=True)

# --- Flow ---
@flow(name="Financial News Sentiment Pipeline")
def financial_pipeline():
    run_data_ingestion()
    run_preprocessing()
    run_sentiment_analysis()
    run_feature_engineering()
    print(f"✅ Pipeline run completed at {datetime.datetime.now()}")

# --- Run locally ---
if __name__ == "__main__":
    financial_pipeline()
