import mlflow

mlflow.set_tracking_uri("file:///D:/tamil/mlruns")
mlflow.set_experiment("tamil-sentiment")

with mlflow.start_run(run_name="muril-binary-v4-fulldata"):
    mlflow.log_param("model", "google/muril-base-cased")
    mlflow.log_param("epochs", 5)
    mlflow.log_param("batch_size", 16)
    mlflow.log_param("learning_rate", 2e-5)
    mlflow.log_param("classes", "binary")
    mlflow.log_param("train_size", 55064)
    mlflow.log_param("dataset", "tamilmixsentiment + DravidianCodeMix Zenodo")
    mlflow.log_metric("accuracy", 0.9468)
    mlflow.log_metric("f1", 0.9470)
    mlflow.log_metric("positive_f1", 0.97)
    mlflow.log_metric("negative_f1", 0.85)
    print("Logged!")