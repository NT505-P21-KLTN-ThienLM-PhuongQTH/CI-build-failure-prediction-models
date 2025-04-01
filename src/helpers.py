from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, roc_curve
import numpy as np
import pandas as pd
import warnings
import os

class Utils:
    # Move CONFIG to utils for now (can be moved to config.py later)
    CONFIG = {
        'NBR_GEN': 3,
        'NBR_SOL': 5,
        'WITH_SMOTE': True,
    }

    @staticmethod
    def get_dataset(file_name, dataset_dir="../data/processed"):
        # Load dataset
        file_path = os.path.join(dataset_dir, file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        dataset = pd.read_csv(file_path, parse_dates=['gh_build_started_at'], index_col="gh_build_started_at")
        dataset.sort_values(by=['gh_build_started_at'], inplace=True)
        return dataset

    @staticmethod
    def to_labels(pos_probs, threshold):
        # Convert probabilities to labels
        return (pos_probs >= threshold).astype(int)

    @staticmethod
    def get_best_threshold(y_true, y_pred):
        # Find the best threshold for the ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_pred)
        gmeans = np.sqrt(tpr * (1 - fpr)) # Geometric mean for balanced threshold
        ix = np.argmax(gmeans)
        return thresholds[ix]

    @staticmethod
    def get_entry(y_true, y_pred_probs, y_pred):
        # Get metrics for a single entry
        metrics = {}
        metrics["AUC"] = roc_auc_score(y_true, y_pred_probs)
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        try:
            metrics["precision"] = precision_score(y_true, y_pred)
            metrics["recall"] = recall_score(y_true, y_pred)
            metrics["F1"] = f1_score(y_true, y_pred)
        except ValueError as e:
            warnings.warn(f"Error computing precision/recall/F1: {e}")
            metrics["precision"] = 0.0
            metrics["recall"] = 0.0
            metrics["F1"] = 0.0
        return metrics

    @staticmethod
    def predict_lstm(model, X, y_true):
        y_pred_probs = model.predict(X, verbose=0) # Silence the output
        threshold = 0.5 if Utils.CONFIG.get('WITH_SMOTE', True) else Utils.get_best_threshold(y_true, y_pred_probs)
        # threshold = Utils.get_best_threshold(y_true, y_pred_probs)
        print(f"\nUsing threshold: {threshold}")
        y_pred = Utils.to_labels(y_pred_probs, threshold)
        return Utils.get_entry(y_true, y_pred_probs, y_pred)

    @staticmethod
    def is_int(n):
        return isinstance(n, int)

    @staticmethod
    def online_validation_folds(dataset, start_fold=6, end_fold=11, fold_ratio=0.1):
        if not isinstance(dataset, pd.DataFrame) or dataset.empty:
            raise ValueError("Dataset must be a non-empty pandas DataFrame")
        if start_fold >= end_fold or fold_ratio <= 0:
            raise ValueError("Invalid fold parameters: start_fold must be less than end_fold and fold_ratio must be positive")
        
        # Split the dataset into train and test sets for online validation (time-series)
        fold_size = int(len(dataset) * fold_ratio)
        if fold_size <= 0:
            raise ValueError("Fold size must be greater than 0")
        
        train_sets, test_sets = [], []
        for i in range(start_fold, end_fold):
            train_end = fold_size * (i - 1)
            test_end = fold_size * i
            if train_end >= len(dataset):
                break
            if test_end > len(dataset):
                test_end = len(dataset)
            train_sets.append(dataset.iloc[:train_end])
            test_sets.append(dataset.iloc[train_end:test_end])
            # if test_sets['build_failed'].sum() == 0:
            #     print(f"[Warning] Fold {i} has no failed builds. Skipping...")
            #     continue
            print(f"Fold {i}: Train {0}-{train_end}, Test {train_end}-{test_end}")
        return train_sets, test_sets