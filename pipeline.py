import os
import subprocess
import sys
import mlflow
import mlflow.keras
from mlflow import MlflowClient

from src.model.padding_evaluation.evaluation import evaluate_padding

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
from src.model.stacked_lstm.validation import run_online_validation, run_cross_project_validation

sys.path.append(PROJECT_ROOT)
# Set MLflow tracking URI to the project root
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
MODEL_NAME = "Stacked-LSTM"

def training(tuner="ga", dataset_dir=None):
    """
    Run the full MLOps pipeline: data loading, training, and validation.
    """
    mlflow.set_experiment(MODEL_NAME)
    if mlflow.active_run() is not None: mlflow.end_run()
    with mlflow.start_run(run_name=MODEL_NAME):
        # Step: Model Training and Validation
        print("Running Online Validation and Selecting Bellwether...")
        bellwether_dataset, all_datasets, bellwether_model_uri = run_online_validation(tuner=tuner, dataset_dir=dataset_dir)

        print("\nRunning Cross-Project Validation with Selected Bellwether...")
        run_cross_project_validation(bellwether_dataset, all_datasets, bellwether_model_uri=bellwether_model_uri, tuner=tuner)

if __name__ == "__main__":
    dataset_dir =
    training(dataset_dir=dataset_dir)
