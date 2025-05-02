import json
import os
import pandas as pd
import mlflow
import mlflow.keras
from src.model.stacked_lstm.model import construct_lstm_model
from src.model.stacked_lstm.preprocess import test_preprocess, apply_smote
from src.model.stacked_lstm.tuners import evaluate_tuner, CONFIG
from src.helpers import Utils
from src.data.preprocess_data import preprocess_training
from src.data.visualization import plot_class_distribution, plot_roc_curve, plot_training_history, plot_metrics
import numpy as np
from dotenv import load_dotenv
load_dotenv()

# Define project root and centralize directories
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MLRUNS_DIR = os.path.join(PROJECT_ROOT, "mlruns")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "stacked_lstm")

# Set MLflow tracking URI to the project root
mlflow.set_tracking_uri(os.environ["MLFLOW_TRACKING_URI"])


os.makedirs(MODEL_DIR, exist_ok=True)

COLUMNS_RES = ["proj", "algo", "iter", "AUC", "accuracy", "F1", "exp"]
MODEL_NAME = "lstm"


def run_online_validation(tuner="ga", dataset_dir="../data/processed", experiment_name="Online_Validation"):
    all_train_entries = []
    all_test_entries = []

    mlflow.set_experiment(experiment_name)
    if mlflow.active_run() is not None: mlflow.end_run()
    with mlflow.start_run(run_name="Online_Validation_Main"):
        print(f"Loading datasets from {dataset_dir}...")
        datasets = preprocess_training(dataset_dir)

        dataset_sizes = {file_name: len(df) for file_name, df in datasets.items()}
        top_10_files = sorted(dataset_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
        top_10_files = [f for f, _ in top_10_files]

        datasets = {f: datasets[f] for f in top_10_files}
        print(f"Selected top 10 datasets: {list(datasets.keys())}")

        pretrained_model_path = None
        for file_name, dataset in datasets.items():
            best_f1 = -1
            best_model_path = None
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
                        # with mlflow.start_run(run_name=f"{file_name}_fold_{fold_idx + 1}_iter_{iteration}", nested=True):
                        entry_train = evaluate_tuner(tuner, train_set, experiment_name, pretrained_model_path=pretrained_model_path, fine_tune=True)

                        history = entry_train.get("history")
                        metrics = entry_train["entry"]  # Lấy dictionary chứa AUC, accuracy, F1
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

                        # Lấy dữ liệu đã cân bằng trước khi tạo chuỗi thời gian
                        feature_cols = [col for col in train_set.columns
                                        if col not in ['build_failed', 'gh_build_started_at', 'gh_project_name']
                                        and train_set[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
                        training_set = train_set[feature_cols].values
                        y = train_set['build_failed'].values
                        X_smote, y_smote = apply_smote(training_set, y)

                        # Tạo DataFrame từ dữ liệu cân bằng
                        balanced_df = pd.DataFrame(X_smote, columns=feature_cols)
                        balanced_df['build_failed'] = y_smote

                        if iteration == 1:  # Chỉ lưu ở lần lặp đầu tiên để tránh trùng lặp
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
                print(f"⚠️ Warning: No valid model found for {file_name}. Skipping MLflow logging...")
                continue

            # Sau khi tìm được mô hình tốt nhất, tạo một MLflow run duy nhất cho file này
            with mlflow.start_run(run_name=f"{file_name}_best_model", nested=True) as run:
                mlflow.log_param("project", file_name)
                mlflow.log_param("fold", best_fold_idx + 1)
                mlflow.log_param("iteration", best_iteration)
                mlflow.log_param("best_threshold", best_threshold)

                mlflow.log_param("time_step", best_params["time_step"])
                # mlflow.log_param("nb_units", best_params["nb_units"])
                # mlflow.log_param("nb_layers", best_params["nb_layers"])
                # mlflow.log_param("optimizer", best_params["optimizer"])
                # mlflow.log_param("nb_epochs", best_params["nb_epochs"])
                # mlflow.log_param("nb_batch", best_params["nb_batch"])
                # mlflow.log_param("drop_proba", best_params["drop_proba"])

                mlflow.log_metric("test_F1", best_entry_test["F1"])
                mlflow.log_metric("test_AUC", best_entry_test["AUC"])
                mlflow.log_metric("test_accuracy", best_entry_test["accuracy"])
                mlflow.log_metric(f"test_recall", best_entry_test["recall"])
                mlflow.log_metric(f"test_precision", best_entry_test["precision"])

                # Lưu mô hình tốt nhất
                model_name = f"best_stacked_lstm_{file_name}"
                mlflow.keras.log_model(best_model, artifact_path=model_name)

                if best_history:
                    history_path = os.path.join(MODEL_DIR, f"{file_name}_history.json")
                    with open(history_path, "w") as f:
                        json.dump(best_history.history, f)
                    mlflow.log_artifact(history_path, artifact_path=model_name)


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
                plot_class_distribution(balanced_train_sets[:num_folds], test_sets[:num_folds],
                                        proj_name=file_name + " (Balanced)")

            print(f"Best model for {file_name} saved at: {best_model_path}, F1: {best_f1}")

        test_df = pd.DataFrame(all_test_entries)
        proj_scores = test_df.groupby('proj')[['F1', 'AUC', 'accuracy']].mean()
        print("\nAverage Test Metrics by Project:")
        print(proj_scores)
        bellwether = proj_scores['F1'].idxmax()
        print(f"\nSelected Bellwether: {bellwether} (Best F1: {proj_scores.loc[bellwether, 'F1']:.4f})")

        train_df = pd.DataFrame(all_train_entries)
        test_df = pd.DataFrame(all_test_entries)
        print("Train columns:", train_df.columns.tolist())
        print("Test columns:", test_df.columns.tolist())

        # Plot the results
        plot_metrics(all_train_entries, all_test_entries, "Online Validation", COLUMNS_RES)
        return datasets[bellwether], datasets

def run_cross_project_validation(bellwether_dataset, all_datasets, tuner="ga", experiment_name="Cross_Project_Validation"):
    all_train_entries = []
    all_test_entries = []

    best_f1 = float('-inf')
    best_model = None
    best_threshold = None
    best_project = None
    best_iteration = None

    mlflow.set_experiment(experiment_name)
    if mlflow.active_run() is not None: mlflow.end_run()
    with mlflow.start_run(run_name="Cross_Project_Validation_Main"):
        bellwether_model_paths = []
        for iteration in range(1, CONFIG['NBR_REP'] + 1):
            print(f"[Cross-Project | Iter {iteration}] Training on Bellwether...")
            with mlflow.start_run(run_name=f"bellwether_training_iter_{iteration}", nested=True):
                entry_train = evaluate_tuner(tuner, bellwether_dataset, experiment_name)
                best_model = entry_train["model"]
                best_params = entry_train["params"]

                # Lưu mô hình trực tiếp vào file cục bộ
                model_name = f"bellwether_lstm_iter{iteration}"
                mlflow.keras.log_model(best_model, artifact_path=model_name)

                # Ghi log metrics huấn luyện
                metrics = entry_train["entry"]
                train_entry_flat = {
                    "iter": iteration,
                    "proj": "bellwether",
                    "algo": MODEL_NAME,
                    "exp": 1,
                    "AUC": metrics.get("AUC", 0.0),
                    "accuracy": metrics.get("accuracy", 0.0),
                    "F1": metrics.get("F1", 0.0)
                }
                all_train_entries.append(train_entry_flat)

                bellwether_model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"
                bellwether_model_paths.append((bellwether_model_uri, best_params))

        for iteration, (bellwether_model_path, best_params) in enumerate(bellwether_model_paths, 1):
            for file_name, test_set in all_datasets.items():
                if test_set is not bellwether_dataset:
                    print(f"Testing on {file_name} with Transfer Learning...")
                    with mlflow.start_run(run_name=f"fine_tune_{file_name}", nested=True):
                        X_test, _ = test_preprocess(bellwether_dataset, test_set, best_params["time_step"])

                        # Fine-tune the model
                        fine_tune_params = best_params.copy()
                        fine_tune_params["nb_epochs"] = 5
                        entry_fine_tune = construct_lstm_model(
                            fine_tune_params, test_set, pretrained_model_path=bellwether_model_path, fine_tune=True
                        )
                        fine_tuned_model = entry_fine_tune["model"]

                        X_test, y_test = test_preprocess(bellwether_dataset, test_set, best_params["time_step"])
                        entry_test, threshold = Utils.predict_lstm(fine_tuned_model, X_test, y_test)
                        entry_test.update({
                            "iter": iteration, "proj": file_name, "exp": 1, "algo": MODEL_NAME
                        })

                        mlflow.log_metric(f"test_F1", entry_test["F1"])
                        mlflow.log_metric(f"test_AUC", entry_test["AUC"])
                        mlflow.log_metric(f"test_accuracy", entry_test["accuracy"])
                        mlflow.log_metric(f"test_recall", entry_test["recall"])
                        mlflow.log_metric(f"test_precision", entry_test["precision"])
                        mlflow.log_param("best_threshold", threshold)

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
                mlflow.register_model(model_uri, "BuildFailureModel")

                mlflow.log_param("best_project", best_project)
                mlflow.log_param("best_iteration", best_iteration)
                mlflow.log_param("best_threshold", best_threshold)
                mlflow.log_param("best_time_step", best_time_step)
                mlflow.log_metric("best_f1", best_f1)
                mlflow.log_metric("best_auc", best_auc)
                mlflow.log_metric("best_accuracy", best_accuracy)
                mlflow.log_metric("best_recall", best_recall)
                mlflow.log_metric("best_precision", best_precision)

                print(f"Best cross-project model registered as 'BuildFailureModel', F1: {best_f1}, "
                      f"Project: {best_project}, Iteration: {best_iteration}, Threshold: {best_threshold}")
        plot_metrics(all_train_entries, all_test_entries, "Cross-Project Validation", COLUMNS_RES)