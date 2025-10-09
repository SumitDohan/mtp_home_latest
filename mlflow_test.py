import mlflow

mlflow.set_tracking_uri("/home/sweta/mtp_home_latest/mtp_home_latest/mlruns")
mlflow.set_experiment("Test_Experiment")

with mlflow.start_run(run_name="test_run"):
    mlflow.log_param("param1", 123)
    mlflow.log_metric("metric1", 0.99)
    mlflow.log_artifact("mlflow_test.py")
