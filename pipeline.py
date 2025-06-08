# pipeline.py
import os
import random
import sys
import mlflow
import mlflow.keras
import numpy as np
import tensorflow as tf
import argparse
from dotenv import load_dotenv
from src.data.processing import get_dataset
from src.model.common.model_factory import ModelFactory
import dagshub
# dagshub.init(repo_owner='hoaiphuongdg26', repo_name='CI-build-failure-prediction-inference', mlflow=True)
# mlflow.set_tracking_uri(os.getenv("MLFLOW_S3_ENDPOINT_URL"))
model_names = ["Stacked-LSTM", "Stacked-BiLSTM", "ConvLSTM", "Padding"]

def set_seed(seed=42):
    os.environ["PYTHONHASHSEED"] = str(seed)
    tf.keras.utils.set_random_seed(seed)
    tf.config.experimental.enable_op_determinism()
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
sys.path.append(PROJECT_ROOT)

def training(model_name="Stacked-LSTM", tuner="ga", datasets=None):
    set_seed(42)
    """
    Run the full MLOps pipeline: data loading, training, and validation.
    Args:
        model_name (str): 'Stacked-LSTM', 'Stacked-LSTM' or 'Padding' to specify the model type.
        tuner (str): Hyperparameter tuning method (e.g., 'ga').
        datasets (dict): Dictionary of datasets for training and validation.
    """
    if datasets is None:
        raise ValueError("No datasets provided for online validation.")

    model_name = ModelFactory.get_model_name(model_name)
    mlflow.set_experiment(model_name)
    if mlflow.active_run() is not None:
        mlflow.end_run()

    with mlflow.start_run(run_name=f"{model_name}"):
        if model_name == "Padding":
            # Training PaddingModule độc lập
            ModelFactory.train_padding_module(
                model_name=model_name,
                datasets=datasets,
                input_dim=None,
                time_step=40,
                epochs=20,
                batch_size=32,
                r2_threshold=0.7,
                max_iterations=5
            )
        else:
            print("Running Online Validation and Selecting Bellwether...")
            bellwether_dataset, all_datasets, bellwether_model_uri = ModelFactory.run_online_validation(
                model_name=model_name,
                tuner=tuner,
                datasets=datasets
            )

            print("\nRunning Cross-Project Validation with Selected Bellwether...")
            ModelFactory.run_cross_project_validation(
                model_name=model_name,
                bellwether_dataset=bellwether_dataset,
                all_datasets=all_datasets,
                bellwether_model_uri=bellwether_model_uri,
                tuner=tuner
            )

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run the CI build failure prediction pipeline.")
    parser.add_argument("--model_name", type=str, default="Stacked-LSTM", choices=model_names,
                        help=f"Model type to use: {model_names}.")
    args = parser.parse_args()

    load_dotenv()
    datasets = get_dataset(
        repo_url=os.getenv("DAGSHUB_REPO"),
        data_path=os.getenv("DAGSHUB_DATA_PATH"),
        rev=os.getenv("DAGSHUB_BRANCH"),
        # file_list=["getsentry_sentry.csv"],
        dagshub_token=os.getenv("DAGSHUB_TOKEN"),
    )
    training(model_name=args.model_name, datasets=datasets)