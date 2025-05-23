import os
import sys
import mlflow
import mlflow.keras
from dotenv import load_dotenv
from src.data.processing import get_dataset
from src.helpers import Utils
from src.model.stacked_lstm.preprocess import prepare_features

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))
from src.model.stacked_lstm.validation import run_online_validation, run_cross_project_validation, MODEL_DIR

sys.path.append(PROJECT_ROOT)
from src.model.padding_module import PaddingModule

# Set MLflow tracking URI to the project root
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])
MODEL_NAME = "Stacked-LSTM"
target_features = [
    "gh_project_name", "gh_build_started_at", "build_failed", "gh_num_issue_comments",
    "gh_num_pr_comments", "gh_team_size", "gh_sloc", "git_diff_src_churn",
    "git_diff_test_churn", "gh_diff_files_added", "gh_diff_files_deleted",
    "gh_diff_tests_added", "gh_diff_src_files", "gh_diff_doc_files",
    "gh_test_cases_per_kloc", "gh_is_pr", "gh_num_commit_comments", "tr_duration",
    "year_of_start", "month_of_start", "day_of_start", "same_committer",
    "proj_fail_rate_history", "proj_fail_rate_recent", "comm_fail_rate_history",
    "comm_fail_rate_recent", "comm_avg_experience", "no_config_edited",
    "num_files_edited", "num_distinct_authors", "prev_build_result", "day_week"
]

def training(tuner="ga", datasets=None):
    """
    Run the full MLOps pipeline: data loading, training, and validation.
    """
    if datasets is None:
        raise ValueError("No datasets provided for online validation.")

    mlflow.set_experiment(MODEL_NAME)
    if mlflow.active_run() is not None: mlflow.end_run()
    with mlflow.start_run(run_name=f"{MODEL_NAME}"):
        # Step: Padding Module Training
        with mlflow.start_run(run_name="Padding_Module", nested=True):
            print("Training PaddingModule...")
            sample_df = next(iter(datasets.values()))
            X, _ = prepare_features(sample_df, target_column='build_failed')
            input_dim = X.shape[1]
            padding_module = PaddingModule(input_dim=input_dim, time_step=40)
            padding_module.train(datasets, epochs=20, batch_size=32)
            padding_module.save_model("Padding-Module")
            print("Evaluating PaddingModule...")
            metrics = padding_module.evaluate(datasets, test_split=0.2)
            print("Evaluation Metrics:")
            for metric, value in metrics.items():
                print(f"{metric}: {value:.4f}")
            Utils.log_mlflow(metrics=metrics)

        # Step: Model Training and Validation
        print("Running Online Validation and Selecting Bellwether...")
        bellwether_dataset, all_datasets, bellwether_model_uri = run_online_validation(tuner=tuner, datasets=datasets)

        print("\nRunning Cross-Project Validation with Selected Bellwether...")
        run_cross_project_validation(bellwether_dataset, all_datasets, bellwether_model_uri=bellwether_model_uri, tuner=tuner)

if __name__ == "__main__":
    load_dotenv()
    datasets = get_dataset(
        repo_url=os.getenv("DAGSHUB_REPO"),
        data_path=os.getenv("DAGSHUB_DATA_PATH"),
        rev=os.getenv("DAGSHUB_BRANCH"),
        # file_list=["getsentry_sentry.csv"],
    )
    training(datasets=datasets)