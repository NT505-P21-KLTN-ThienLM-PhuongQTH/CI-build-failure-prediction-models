import os
import pandas as pd
import matplotlib.pyplot as plt
from .model import construct_lstm_model
from .preprocess import test_preprocess, train_preprocess, apply_smote
from .tuners import evaluate_tuner, CONFIG
from src.helpers import Utils
from ...data.visualization import plot_class_distribution, plot_roc_curve, plot_training_history
import numpy as np

MODEL_DIR = "../models/stacked_lstm"
COLUMNS_RES = ["proj", "algo", "iter", "AUC", "accuracy", "F1", "exp"]
MODEL_NAME = "lstm"


def plot_metrics(train_entries, test_entries, title):
    train_df = pd.DataFrame(train_entries)[COLUMNS_RES]
    test_df = pd.DataFrame(test_entries)[COLUMNS_RES]

    print(f"\n{title} - Train Results:")
    print(train_df.groupby(['proj', 'exp']).mean())
    print(f"\n{title} - Test Results:")
    print(test_df.groupby(['proj', 'exp']).mean())

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for i, metric in enumerate(['AUC', 'accuracy', 'F1']):
        test_df.boxplot(column=metric, by='proj', ax=axes[i])
        axes[i].set_title(f"{metric} ({title})")
        axes[i].set_xlabel("Project")
        axes[i].set_ylabel(metric)
        axes[i].tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.show()

def run_online_validation(tuner="ga", dataset_dir="../data/processed"):
    # Run online validation and plot results.
    all_train_entries = []
    all_test_entries = []

    print(f"Loading datasets from {dataset_dir}...")
    dataset_sizes = {}

    # Get number of rows for each dataset
    for f in os.listdir(dataset_dir):
        try:
            df = Utils.get_dataset(f, dataset_dir)
            dataset_sizes[f] = len(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    # Sort and select top 10 largest datasets
    top_10_files = sorted(dataset_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
    top_10_files = [f for f, _ in top_10_files]

    # Load only the top 10 datasets
    datasets = {}
    for f in top_10_files:
        try:
            datasets[f] = Utils.get_dataset(f, dataset_dir)
            print(f"Loaded {f} with {len(datasets[f])} samples")
        except Exception as e:
            print(f"Error loading {f}: {e}")

    print(f"Selected top 10 datasets: {list(datasets.keys())}")
    if not datasets:
        raise ValueError(f"No datasets found in {dataset_dir}")

    for file_name, dataset in datasets.items():
        best_f1 = -1
        best_model_path = None
        train_sets, test_sets = Utils.online_validation_folds(dataset)

        # Vẽ phân bố lớp trước khi cân bằng
        print(f"Plotting class distribution BEFORE balancing for {file_name}...")
        plot_class_distribution(train_sets, test_sets, proj_name=file_name)
        # Lưu các train_sets đã cân bằng
        balanced_train_sets = []

        for fold_idx, (train_set, test_set) in enumerate(zip(train_sets, test_sets)):
            for iteration in range(1, CONFIG['NBR_REP'] + 1):
                print(f"\n[Proj {file_name} | Fold {fold_idx + 1} | Iter {iteration}] Training...")
                entry_train = evaluate_tuner(tuner, train_set)

                history = entry_train.get("history")
                print(f"History object: {history}")
                if history:
                    # Đường dẫn để lưu đồ thị loss và accuracy
                    save_path = os.path.join(MODEL_DIR,
                                             f"training_history_{file_name}_fold_{fold_idx + 1}_iter_{iteration}.png")
                    print(f"Plotting training history for {file_name} Fold {fold_idx + 1} Iter {iteration}...")
                    plot_training_history(history, file_name, fold_idx, save_path=None)
                else:
                    print("No training history available to plot.")

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

                entry_train.update({
                    "iter": iteration, "proj": f"proj{file_name}", "exp": fold_idx + 1, "algo": MODEL_NAME
                })
                all_train_entries.append(entry_train)

                best_model = entry_train["model"]
                best_params = entry_train["params"]
                X_test, y_test = test_preprocess(train_set, test_set, best_params["time_step"])
                entry_test = Utils.predict_lstm(best_model, X_test, y_test)

                y_pred_probs = best_model.predict(X_test).flatten()
                print(f"Plotting ROC curve for {file_name} Fold {fold_idx + 1}...")
                save_path = os.path.join(MODEL_DIR, f"roc_curve_{file_name}_fold_{fold_idx + 1}.png")
                plot_roc_curve(y_test, y_pred_probs, file_name, fold_idx, save_path=None)

                entry_test.update({
                    "iter": iteration, "proj": file_name, "exp": fold_idx + 1, "algo": MODEL_NAME
                })
                if entry_test["F1"] > best_f1:
                    best_f1 = entry_test["F1"]
                    best_model_path = os.path.join(MODEL_DIR, f"best_stacked_lstm_{file_name}.keras")
                    best_model.save(best_model_path)
                print(f"Test metrics: {entry_test}")
                all_test_entries.append(entry_test)

                # Vẽ phân bố lớp sau khi cân bằng
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
    plot_metrics(all_train_entries, all_test_entries, "Online Validation")
    return datasets[bellwether], datasets

def run_cross_project_validation(bellwether_dataset, all_datasets, tuner="ga"):
    # Run cross-project validation and plot results.
    all_train_entries = []
    all_test_entries = []

    for iteration in range(1, CONFIG['NBR_REP'] + 1):
        print(f"[Cross-Project | Iter {iteration}] Training on Bellwether...")
        entry_train = evaluate_tuner(tuner, bellwether_dataset)
        best_model = entry_train["model"]
        best_params = entry_train["params"]
        entry_train.update({
            "iter": iteration, "proj": "bellwether", "algo": MODEL_NAME, "exp": 1
        })
        all_train_entries.append(entry_train)

        for file_name, test_set in all_datasets.items():
            if test_set is not bellwether_dataset:
                best_f1 = -1
                best_model_path = None
                print(f"Testing on {file_name}...")
                X_test, y_test = test_preprocess(bellwether_dataset, test_set, best_params["time_step"])
                entry_test = Utils.predict_lstm(best_model, X_test, y_test)
                entry_test.update({
                    "iter": iteration, "proj": file_name, "exp": 1, "algo": MODEL_NAME
                })

                if entry_test["F1"] > best_f1:
                    best_f1 = entry_test["F1"]
                    best_model_path = os.path.join(MODEL_DIR,
                                                   f"best_stacked_lstm_{file_name}_cross_iter{iteration}.keras")
                    best_model.save(best_model_path)
                    print(f"Best model for {file_name} saved at: {best_model_path}, F1: {best_f1}")
                print(f"Test metrics: {entry_test}")
                all_test_entries.append(entry_test)

    plot_metrics(all_train_entries, all_test_entries, "Cross-Project Validation")