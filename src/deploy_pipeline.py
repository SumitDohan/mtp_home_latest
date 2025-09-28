# src/deploy_pipeline.py
from components.run_pipeline import financial_pipeline
from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import CronSchedule

# --- Create a daily deployment at 8:00 AM ---
deployment = Deployment.build_from_flow(
    flow=financial_pipeline,
    name="daily_financial_pipeline",
    work_pool_name="default",  # make sure this pool exists
    schedule=CronSchedule(cron="0 8 * * *", timezone="Asia/Kolkata"),
)

# --- Apply / register the deployment with Prefect Orion ---
if __name__ == "__main__":
    deployment.apply()
    print("✅ Deployment created successfully!")
