# src/model/stacked_lstm/validation.py
import os
import pandas as pd
import mlflow
import mlflow.keras
from src.model.stacked_lstm.model import construct_lstm_model
from src.model.stacked_lstm.preprocess import test_preprocess, apply_smote, prepare_features
from src.model.stacked_lstm.tuners import evaluate_tuner, CONFIG
from src.helpers import Utils
from src.data.visualization import plot_class_distribution, plot_roc_curve, plot_training_history, plot_metrics
import numpy as np

# Define project root and centralize directories
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MLRUNS_DIR = os.path.join(PROJECT_ROOT, "mlruns")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "stacked_lstm")

os.makedirs(MODEL_DIR, exist_ok=True)

COLUMNS_RES = ["proj", "algo", "iter", "AUC", "accuracy", "F1", "exp"]
MODEL_NAME = "Stacked-LSTM"


def log_mlflow(params=None, metrics=None, history=None, prefix=""):
    """
    Log parameters and metrics to MLflow.

    Args:
        params (dict): Dictionary of parameters to log.
        metrics (dict): Dictionary of metrics to log.
        prefix (str): Optional prefix for metric keys (e.g., 'train_', 'test_').
    """
    if params:
        for key, value in params.items():
            mlflow.log_param(key, value)

    if metrics:
        for key, value in metrics.items():
            if isinstance(value, (int, float)):
                mlflow.log_metric(f"{prefix}{key}", value)

    if history:
        for metric_name, values in history.items():
            for epoch, value in enumerate(values, 1):
                mlflow.log_metric(f"{prefix}{metric_name}_{epoch}", value, step=epoch)

def run_online_validation(tuner="ga", datasets=None):
    all_train_entries = []
    all_test_entries = []
    if datasets is None:
        raise ValueError("No datasets provided for online validation.")

    with mlflow.start_run(run_name="Online_Validation_Main", nested=True):
        dataset_sizes = {file_name: len(df) for file_name, df in datasets.items()}
        top_10_files = sorted(dataset_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
        top_10_files = [f for f, _ in top_10_files]

        datasets = {f: datasets[f] for f in top_10_files}
        print(f"Selected top 10 datasets: {list(datasets.keys())}")

        best_models = {}
        for file_name, dataset in datasets.items():
            best_f1 = -1
            best_model = None
            best_params = None
            best_history = None
            best_X_test = None
            best_y_test = None
            best_fold_idx = None
            best_iteration = None
            best_entry_test = None
            best_threshold = None
            train_sets, test_sets = Utils.online_validation_folds(dataset)

            # Vẽ phân bố lớp trước khi cân bằng
            print(f"Plotting class distribution BEFORE balancing for {file_name}...")
            plot_class_distribution(train_sets, test_sets, proj_name=file_name)

            balanced_train_sets = []
            for fold_idx, (train_set, test_set) in enumerate(zip(train_sets, test_sets)):
                for iteration in range(1, CONFIG['NBR_REP'] + 1):
                    print(f"\n[Proj {file_name} | Fold {fold_idx + 1} | Iter {iteration}] Training...")
                    entry_train = evaluate_tuner(tuner, train_set, pretrained_model_path=None)

                    history = entry_train.get("history")
                    metrics = entry_train["entry"]
                    train_entry_flat = {
                        "iter": iteration,
                        "proj": f"proj{file_name}",
                        "exp": fold_idx + 1,
                        "algo": MODEL_NAME,
                        "AUC": metrics.get("AUC", 0.0),
                        "accuracy": metrics.get("accuracy", 0.0),
                        "F1": metrics.get("F1", 0.0)
                    }
                    all_train_entries.append(train_entry_flat)

                    # Prepare balanced data
                    X_train, y = prepare_features(train_set, target_column='build_failed')
                    feature_cols = X_train.columns.tolist()
                    X_smote, y_smote = apply_smote(X_train.values, y.values)
                    balanced_df = pd.DataFrame(X_smote, columns=feature_cols)
                    balanced_df['build_failed'] = y_smote

                    if iteration == 1:
                        balanced_train_sets.append(balanced_df)

                    current_model = entry_train["model"]
                    current_params = entry_train["params"]
                    X_test, y_test = test_preprocess(train_set, test_set, current_params["time_step"])
                    entry_test, threshold = Utils.predict_lstm(current_model, X_test, y_test)

                    entry_test.update({
                        "iter": iteration, "proj": file_name, "exp": fold_idx + 1, "algo": MODEL_NAME
                    })
                    if (not np.isnan(entry_test["AUC"])) and (entry_test["F1"] > best_f1):
                        best_f1 = entry_test["F1"]

                        best_model = current_model
                        best_params = current_params
                        best_history = history
                        best_X_test = X_test
                        best_y_test = y_test
                        best_fold_idx = fold_idx
                        best_iteration = iteration
                        best_entry_test = entry_test
                        best_threshold = threshold

                    print(f"Test metrics: {entry_test}")
                    all_test_entries.append(entry_test)

            if best_entry_test is None:
                print(f"Warning: No valid model found for {file_name}. Skipping MLflow logging...")
                continue

            with mlflow.start_run(run_name=f"{file_name}_best_model", nested=True) as run:
                log_mlflow(
                    params={
                        "project": file_name,
                        "fold": best_fold_idx + 1,
                        "interation": best_iteration,
                        "best_threshold": best_threshold,
                        **best_params
                        },
                   metrics=best_entry_test,
                   history=best_history.history if best_history else None,
                   prefix="test_"
                )

                best_models[file_name] = {
                    "model": best_model,
                    "history": best_history,
                    "X_test": best_X_test,
                    "y_test": best_y_test,
                    "fold_idx": best_fold_idx,
                    "entry_test": best_entry_test
                }

                # Vẽ các biểu đồ chỉ cho mô hình tốt nhất
                if best_history:
                    print(f"Plotting training history for best model of {file_name}...")
                    plot_training_history(best_history, file_name, best_fold_idx)

                y_pred_probs = best_model.predict(best_X_test).flatten()
                print(f"Plotting ROC curve for best model of {file_name}...")
                plot_roc_curve(best_y_test, y_pred_probs, file_name, best_fold_idx)

                # Vẽ phân bố lớp sau khi cân bằng (dùng balanced_train_sets từ lần lặp đầu tiên)
                print(f"Plotting class distribution AFTER balancing for {file_name}...")
                num_folds = min(len(balanced_train_sets), len(test_sets))
                plot_class_distribution(
                    balanced_train_sets[:num_folds],
                    test_sets[:num_folds],
                    proj_name=file_name + " (Balanced)"
                )

        test_df = pd.DataFrame(all_test_entries)
        proj_scores = test_df.groupby('proj')[['F1', 'AUC', 'accuracy']].mean()
        print("\nAverage Test Metrics by Project:")
        print(proj_scores)
        bellwether = proj_scores['F1'].idxmax()
        print(f"\nSelected Bellwether: {bellwether} (Best F1: {proj_scores.loc[bellwether, 'F1']:.4f})")
        bellwether_info = best_models[bellwether]

        # Save only the bellwether model
        with mlflow.start_run(run_name=f"bellwether_{bellwether}", nested=True):
            sanitized_bellwether = bellwether.replace('/', '_').replace('\\', '_')
            model_name = f"bellwether_model_{sanitized_bellwether}"
            mlflow.keras.log_model(bellwether_info["model"], artifact_path=model_name)
            bellwether_model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"
            log_mlflow(
                metrics=bellwether_info["entry_test"],
                history=bellwether_info["history"].history if bellwether_info["history"] else None,
                prefix="bellwether_"
            )

        train_df = pd.DataFrame(all_train_entries)
        test_df = pd.DataFrame(all_test_entries)
        print("Train columns:", train_df.columns.tolist())
        print("Test columns:", test_df.columns.tolist())

        # Plot the results
        plot_metrics(all_train_entries, all_test_entries, "Online Validation", COLUMNS_RES)
        return datasets[bellwether], datasets, bellwether_model_uri

def run_cross_project_validation(bellwether_dataset, all_datasets, bellwether_model_uri=None, tuner="ga", ):
    all_train_entries = []
    all_test_entries = []

    best_f1 = float('-inf')
    best_model = None
    best_threshold = None
    best_project = None
    best_iteration = None
    best_params = None
    best_metrics = None

    bellwether_model_paths = []

    with mlflow.start_run(run_name="Cross_Project_Validation_Main", nested=True):
        for iteration in range(1, CONFIG['NBR_REP'] + 1):
            print(f"[Cross-Project | Iter {iteration}] Training on Bellwether...")
            with mlflow.start_run(run_name=f"training_iter_{iteration}", nested=True):
                entry_train = evaluate_tuner(tuner, bellwether_dataset, bellwether_model_uri)
                current_model = entry_train["model"]
                current_params = entry_train["params"]

                # Ghi log metrics huấn luyện
                current_metrics = entry_train["entry"]
                train_entry_flat = {
                    "iter": iteration,
                    "proj": "bellwether",
                    "algo": MODEL_NAME,
                    "exp": 1,
                    "AUC": current_metrics.get("AUC", 0.0),
                    "accuracy": current_metrics.get("accuracy", 0.0),
                    "F1": current_metrics.get("F1", 0.0)
                }
                all_train_entries.append(train_entry_flat)

                if current_metrics.get("F1", 0.0) > best_f1:
                    model_name = f"bellwether_lstm_iter{iteration}"
                    mlflow.keras.log_model(current_model, artifact_path=model_name)

                    best_metric=current_metrics
                    best_params = current_params
                    bellwether_model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"
                    bellwether_model_paths.append((bellwether_model_uri, best_params, best_metric))

                log_mlflow(params=current_params, metrics=current_metrics, prefix="bellwether_train_")

        for iteration, (bellwether_model_path, best_params, best_metric) in enumerate(bellwether_model_paths, 1):
            for file_name, test_set in all_datasets.items():
                if test_set is not bellwether_dataset:
                    print(f"Testing on {file_name} with Transfer Learning...")
                    with mlflow.start_run(run_name=f"fine_tune_{file_name}", nested=True):
                        X_test, _ = test_preprocess(bellwether_dataset, test_set, best_params["time_step"])

                        # Fine-tune the model
                        fine_tune_params = best_params.copy()
                        fine_tune_params["nb_epochs"] = 5
                        entry_fine_tune = construct_lstm_model(fine_tune_params, test_set, pretrained_model_path=bellwether_model_path)
                        fine_tuned_model = entry_fine_tune["model"]

                        X_test, y_test = test_preprocess(bellwether_dataset, test_set, best_params["time_step"])
                        entry_test, threshold = Utils.predict_lstm(fine_tuned_model, X_test, y_test)
                        entry_test.update({
                            "iter": iteration, "proj": file_name, "exp": 1, "algo": MODEL_NAME
                        })
                        log_mlflow(params={"best_threshold": threshold, **best_params},
                                   metrics=entry_test,
                                   prefix="test_")

                        if (not np.isnan(entry_test["AUC"])) and (entry_test["F1"] > best_f1):
                            best_f1 = entry_test["F1"]
                            best_auc = entry_test["AUC"]
                            best_accuracy = entry_test["accuracy"]
                            best_recall = entry_test["recall"]
                            best_precision = entry_test["precision"]
                            best_time_step = best_params["time_step"]
                            best_model = fine_tuned_model
                            best_threshold = threshold
                            best_project = file_name
                            best_iteration = iteration

                        print(f"Test metrics for {file_name}: {entry_test}")
                        all_test_entries.append(entry_test)
        if best_model is not None:
            with mlflow.start_run(run_name=f"best_model_{best_project}", nested=True):
                best_model_name = f"best_stacked_lstm_cross_project_{best_project}_iter{best_iteration}"
                mlflow.keras.log_model(best_model, artifact_path=best_model_name)

                # Đăng ký mô hình vào Model Registry
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/{best_model_name}"
                mlflow.register_model(model_uri, MODEL_NAME)
                log_mlflow (params={"best_project": best_project, "best_iteration": best_iteration,
                           "best_threshold": best_threshold, "best_time_step": best_time_step},
                           metrics={"best_f1": best_f1, "best_auc": best_auc, "best_accuracy": best_accuracy,
                                    "best_recall": best_recall, "best_precision": best_precision},)

                print(f"Best cross-project model registered as 'BuildFailureModel', F1: {best_f1}, "
                      f"Project: {best_project}, Iteration: {best_iteration}, Threshold: {best_threshold}")
        plot_metrics(all_train_entries, all_test_entries, "Cross-Project Validation", COLUMNS_RES)