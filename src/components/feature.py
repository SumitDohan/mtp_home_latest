# src/feature_engineering_component.py
import os
import glob
import pandas as pd
import mlflow

# --- Paths ---
PROCESSED_PATH = "data/processed"
os.makedirs(PROCESSED_PATH, exist_ok=True)

# --- MLflow Setup ---
mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("Financial_Sentiment_Pipeline")

# --- Load latest processed news sentiment CSV ---
news_files = sorted(glob.glob(os.path.join(PROCESSED_PATH, "processed_news_sentiment.csv")))
if not news_files:
    raise FileNotFoundError("❌ No processed news sentiment CSV found.")
news_file = news_files[-1]

news_df = pd.read_csv(news_file, parse_dates=["published"])
news_df["published"] = news_df["published"].dt.date  # convert to date
print(f"📂 Using news sentiment file: {news_file}")

# --- Aggregate news sentiment per day ---
daily_sentiment = news_df.groupby("published").agg(
    avg_sentiment=("sentiment_score", "mean"),
    percent_negative=("sentiment_label", lambda x: (x=="negative").sum() / len(x) * 100)
).reset_index().rename(columns={"published": "Date"})

# --- Investment advice based on negative sentiment ---
def get_investment_advice(row):
    if row["percent_negative"] >= 35:
        return "NOT GOOD TO INVEST"
    else:
        return "NORMAL DAY"

daily_sentiment["investment_advice"] = daily_sentiment.apply(get_investment_advice, axis=1)

# --- Save engineered features ---
features_file = os.path.join(PROCESSED_PATH, "features.csv")
daily_sentiment.to_csv(features_file, index=False)
print(f"✅ Feature-engineered dataset saved to {features_file}")

# --- Log to MLflow ---
with mlflow.start_run(run_name="feature_engineering"):
    mlflow.log_artifact(features_file, artifact_path="features")
    mlflow.log_metric("num_days", len(daily_sentiment))
    avg_neg = daily_sentiment["percent_negative"].mean()
    mlflow.log_metric("avg_daily_negative_percent", avg_neg)

print("✅ Feature engineering complete and logged to MLflow.")

# --- Print days with high negative sentiment ---
high_neg_days = daily_sentiment[daily_sentiment["percent_negative"] >= 35]
for idx, row in high_neg_days.iterrows():
    print(f"📉 {row['Date']} — {row['percent_negative']:.1f}% negative news: NOT GOOD TO INVEST")
