import os
import sys
PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
from src.model.stacked_lstm.validation import run_online_validation, run_cross_project_validation

sys.path.append(PROJECT_ROOT)

def run_pipeline(tuner="ga", dataset_dir=os.path.join(PROJECT_ROOT, "data", "processed-local")):
    """
    Run the full MLOps pipeline: data loading, training, and validation.
    """
    # Step: Model Training and Validation
    print("Running Online Validation and Selecting Bellwether...")
    bellwether_dataset, all_datasets = run_online_validation(tuner=tuner, dataset_dir=dataset_dir)

    print("\nRunning Cross-Project Validation with Selected Bellwether...")
    run_cross_project_validation(bellwether_dataset, all_datasets, tuner=tuner)

if __name__ == "__main__":
    run_pipeline()