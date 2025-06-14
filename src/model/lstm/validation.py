# src/model/lstm/validation.py
import os
import random
import pandas as pd
import mlflow
import mlflow.keras
from src.model.lstm.model import construct_lstm_model
from src.model.common.preprocess import test_preprocess, apply_smote, prepare_features
from src.model.lstm.tuners import evaluate_tuner, CONFIG
from src.helpers import Utils

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MLRUNS_DIR = os.path.join(PROJECT_ROOT, "mlruns")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "stacked_lstm")
os.makedirs(MODEL_DIR, exist_ok=True)

COLUMNS_RES = ["proj", "algo", "iter", "AUC", "accuracy", "F1", "exp"]
MODEL_NAME = "Stacked-LSTM"

def run_online_validation(tuner="ga", datasets=None):
    if datasets is None:
        raise ValueError("No datasets provided for online validation.")

    all_test_entries = []

    with mlflow.start_run(run_name="Online_Validation_Main", nested=True):
        dataset_sizes = {file_name: len(df) for file_name, df in datasets.items()}
        top_10_files = sorted(dataset_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
        top_10_files = [f for f, _ in top_10_files]
        datasets = {f: datasets[f] for f in top_10_files}
        print(f"Selected top 10 datasets: {list(datasets.keys())}")

        best_models = {}
        for file_name, dataset in datasets.items():
            best_score = float('-inf')
            best_model_info = None
            train_sets, val_sets, test_sets = Utils.online_validation_folds(dataset)

            if not train_sets or not val_sets or not test_sets:
                print(f"Skipping project {file_name} due to insufficient data.")
                continue

            balanced_train_sets = []

            for fold_idx, (train_set, val_set, test_set) in enumerate(zip(train_sets, val_sets, test_sets)):
                if len(test_set) == 0 or len(val_set) == 0 or len(train_set) == 0:
                    print(f"Fold {fold_idx + 1} for {file_name} skipped due to empty set.")
                    continue

                for iteration in range(1, CONFIG['NBR_REP'] + 1):
                    print(f"\n[Proj {file_name} | Fold {fold_idx + 1} | Iter {iteration}] Training...")
                    entry_train = evaluate_tuner(tuner, train_set, val_set, pretrained_model_path=None)
                    current_model = entry_train["model"]
                    current_params = entry_train["params"]
                    current_history = entry_train.get("history")
                    current_metrics = entry_train["entry"]

                    X_train, y = prepare_features(train_set, target_column='build_failed')
                    X_smote, y_smote = apply_smote(X_train.values, y.values)
                    balanced_df = pd.DataFrame(X_smote, columns=X_train.columns)
                    balanced_df['build_failed'] = y_smote

                    if iteration == 1:
                        balanced_train_sets.append(balanced_df)

                    X_test, y_test = test_preprocess(train_set, test_set, current_params["time_step"])
                    if X_test.shape[0] == 0 or y_test.shape[0] == 0:
                        print(f"Test set for fold {fold_idx + 1} is empty, skipping iteration {iteration}.")
                        continue

                    entry_test, threshold = Utils.predict_lstm(current_model, X_test, y_test)
                    entry_test.update({
                        "iter": iteration,
                        "proj": file_name,
                        "exp": fold_idx + 1,
                        "algo": MODEL_NAME
                    })
                    score = Utils.calculate_weighted_score(entry_test)

                    # Update best model info if current score is higher
                    if score > best_score:
                        best_score = score
                        best_model_info = {
                            "model": current_model,
                            "params": {**current_params, "project": file_name, "fold": fold_idx + 1,
                                       "iteration": iteration, "threshold": threshold},
                            "history": current_history,
                            "X_test": X_test,
                            "y_test": y_test,
                            "entry_test": {**entry_test, "score": score}
                        }

                    all_test_entries.append(entry_test)
                    print(f"Test metrics: {entry_test}")

            if best_model_info is None:
                print(f"Warning: No valid model found for {file_name}. Skipping MLflow logging...")
                continue

            with mlflow.start_run(run_name=f"{file_name}_best_model", nested=True):
                log_params, log_metrics = Utils.build_log_entries(
                    params=best_model_info["params"],
                    metrics=best_model_info["entry_test"],
                    prefix="test_"
                )
                Utils.log_mlflow(params=log_params, metrics=log_metrics,
                                 history=best_model_info["history"].history if best_model_info["history"] else None)

                best_models[file_name] = best_model_info

        if not best_models:
            print("No valid models found for any project. Aborting validation.")
            return None, None, None

        test_df = pd.DataFrame(all_test_entries)
        proj_scores = test_df.groupby('proj')[['F1', 'AUC', 'accuracy']].mean()
        print("\nAverage Test Metrics by Project:")
        print(proj_scores)
        bellwether = proj_scores['F1'].idxmax()
        print(f"\nSelected Bellwether: {bellwether} (Best F1: {proj_scores.loc[bellwether, 'F1']:.4f})")
        bellwether_info = best_models[bellwether]

        sanitized_bellwether = bellwether.replace('/', '_').replace('\\', '_')
        model_name = f"bellwether_model_{sanitized_bellwether}"
        mlflow.keras.log_model(bellwether_info["model"], artifact_path=model_name)
        bellwether_model_uri = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"
        log_params, log_metrics = Utils.build_log_entries(
            params=bellwether_info["params"],
            metrics=bellwether_info["entry_test"],
            prefix="bellwether_"
        )
        Utils.log_mlflow(params=log_params, metrics=log_metrics,
                         history=bellwether_info["history"].history if bellwether_info["history"] else None)

        return datasets[bellwether], datasets, bellwether_model_uri

def run_cross_project_validation(bellwether_dataset, all_datasets, bellwether_model_uri=None, tuner="ga"):
    all_train_entries = []
    all_test_entries = []
    bellwether_model_paths = []

    with mlflow.start_run(run_name="Cross_Project_Validation_Main", nested=True):
        best_bellwether_info = None
        best_bellwether_score = float('-inf')

        with mlflow.start_run(run_name="Bellwether_Train", nested=True):
            train_sets, val_sets, _ = Utils.online_validation_folds(bellwether_dataset)
            if not train_sets or not val_sets:
                print(f"Skipping Bellwether training due to insufficient data in {bellwether_dataset}.")
                return

            for iteration in range(1, CONFIG['NBR_REP'] + 1):
                print(f"[Cross-Project | Iter {iteration}] Training on Bellwether...")
                with mlflow.start_run(run_name=f"training_iter_{iteration}", nested=True):
                    entry_train = evaluate_tuner(tuner, train_sets[0], val_sets[0], bellwether_model_uri)
                    current_model = entry_train["model"]
                    current_params = entry_train["params"]
                    current_metrics = entry_train["entry"]
                    print(f"$$$ entry_train = {entry_train}")

                    score = Utils.calculate_weighted_score(current_metrics)
                    log_params, log_metrics = Utils.build_log_entries(
                        params={**current_params, "project": "bellwether", "iteration": iteration},
                        metrics={**current_metrics, "score": score},
                        prefix="bellwether_train_"
                    )
                    Utils.log_mlflow(params=log_params, metrics=log_metrics)

                    if score > best_bellwether_score:
                        best_bellwether_score = score
                        best_bellwether_info = {
                            "model": current_model,
                            "model_name": f"bellwether_model_iter{iteration}",
                            "params": current_params,
                            "metrics": {**current_metrics, "score": score},
                            "iteration": iteration,
                        }

            if best_bellwether_info is not None:
                model_name = f"{best_bellwether_info['model_name']}"
                mlflow.keras.log_model(best_bellwether_info["model"], artifact_path=model_name)
                model_path = f"runs:/{mlflow.active_run().info.run_id}/{model_name}"
                log_params, log_metrics = Utils.build_log_entries(
                    params={**best_bellwether_info["params"], "model_path": model_path},
                    metrics=best_bellwether_info["metrics"],
                    prefix="best_bellwether_"
                )
                Utils.log_mlflow(params=log_params, metrics=log_metrics)
                bellwether_model_paths = [(model_path, best_bellwether_info["params"], best_bellwether_info["metrics"])]

        best_cross_project_info = None
        best_cross_project_score = float('-inf')

        with mlflow.start_run(run_name="Fine_Tune", nested=True):
            for iteration, (model_path, params, metrics) in enumerate(bellwether_model_paths, 1):
                for file_name, test_set in all_datasets.items():
                    if test_set is not bellwether_dataset:
                        print(f"Testing on {file_name} with Transfer Learning...")
                        with mlflow.start_run(run_name=f"fine_tune_{file_name}_iter{iteration}", nested=True):
                            train_sets, val_sets, test_sets = Utils.online_validation_folds(test_set)
                            if not train_sets or not val_sets or not test_sets:
                                print(f"Skipping {file_name} due to insufficient data.")
                                continue

                            fine_tune_params = params.copy()
                            fine_tune_params["nb_epochs"] = 5
                            print(f"$$$ fine_tune_params: {fine_tune_params}")
                            entry_fine_tune = construct_lstm_model(fine_tune_params, train_sets[0], val_sets[0],
                                                                   pretrained_model_path=model_path)
                            print(f"$$$ entry_fine_tune = {entry_fine_tune}")
                            fine_tuned_model = entry_fine_tune["model"]

                            X_test, y_test = test_preprocess(train_sets[0], test_sets[0], params["time_step"])
                            if X_test.shape[0] == 0 or y_test.shape[0] == 0:
                                print(f"Test set for {file_name} is empty, skipping.")
                                continue

                            entry_test, threshold = Utils.predict_lstm(fine_tuned_model, X_test, y_test)
                            entry_test.update({
                                "iter": iteration,
                                "proj": file_name,
                                "exp": 1,
                                "algo": MODEL_NAME
                            })
                            score = Utils.calculate_weighted_score(entry_test)

                            log_params, log_metrics = Utils.build_log_entries(
                                params={**fine_tune_params, "threshold": threshold, "project": file_name,
                                        "iteration": iteration, "time_step": params["time_step"],
                                        "input_dim": X_test.shape[2]},
                                metrics={**entry_test, "score": score},
                                prefix="test_"
                            )
                            Utils.log_mlflow(params=log_params, metrics=log_metrics)
                            print(f"Test metrics for {file_name}: {entry_test}")
                            all_test_entries.append(entry_test)

                            if score > best_cross_project_score:
                                best_cross_project_score = score
                                best_cross_project_info = {
                                    "model": fine_tuned_model,
                                    "params": {**entry_fine_tune["entry"],
                                               "project": file_name,
                                               "iteration": iteration,
                                               "threshold": threshold,
                                               "time_step": params["time_step"],
                                               "input_dim": X_test.shape[2]},
                                    "metrics": {
                                        "f1": entry_test["F1"],
                                        "auc": entry_test["AUC"],
                                        "accuracy": entry_test["accuracy"],
                                        "recall": entry_test["recall"],
                                        "precision": entry_test["precision"],
                                        "score": score
                                    }
                                }

            if best_cross_project_info is not None:
                best_project = best_cross_project_info["params"]["project"]
                best_iteration = best_cross_project_info["params"]["iteration"]
                best_model_name = f"best_stacked_lstm_cross_project_{best_project}_iter{best_iteration}"

                mlflow.keras.log_model(best_cross_project_info["model"], artifact_path=best_model_name)
                model_uri = f"runs:/{mlflow.active_run().info.run_id}/{best_model_name}"
                mlflow.register_model(model_uri, MODEL_NAME)

                log_params, log_metrics = Utils.build_log_entries(
                    params=best_cross_project_info["params"],
                    metrics=best_cross_project_info["metrics"]
                )
                Utils.log_mlflow(params=log_params, metrics=log_metrics)

                print(f"Best cross-project model registered as '{MODEL_NAME}', "
                      f"F1: {best_cross_project_info['metrics']['f1']}, "
                      f"Project: {best_project}, Iteration: {best_iteration}, "
                      f"Threshold: {best_cross_project_info['params']['threshold']}")