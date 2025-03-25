# src/utils/Utils.py
from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, roc_curve
import numpy as np
import pandas as pd
import warnings
import os


class Utils:
    # Move CONFIG to utils for now (can be moved to config.py later)
    CONFIG = {
        'NBR_REP': 6,
        'NBR_GEN': 5,
        'NBR_SOL': 5,
        'MAX_EVAL': 8,
        'WITH_SMOTE': True,
        'HYBRID_OPTION': True
    }
    if CONFIG['HYBRID_OPTION']:
        CONFIG['WITH_SMOTE'] = True

    @staticmethod
    def get_dataset(file_name):
        """
        Load dataset from a CSV file and sort by timestamp.

        Args:
            file_name: Name of the CSV file in 'data/processed/by_project/'

        Returns:
            Sorted DataFrame
        """
        file_path = os.path.join("data", "processed", "by_project", file_name)
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"File {file_path} not found")
        dataset = pd.read_csv(file_path, parse_dates=['gh_build_started_at'], index_col="gh_build_started_at")
        dataset.sort_values(by=['gh_build_started_at'], inplace=True)
        return dataset

    @staticmethod
    def to_labels(pos_probs, threshold):
        """
        Convert probabilities to binary labels based on threshold.

        Args:
            pos_probs: Predicted probabilities
            threshold: Decision threshold

        Returns:
            Binary labels (0 or 1)
        """
        if not isinstance(pos_probs, np.ndarray):
            pos_probs = np.array(pos_probs)
        return (pos_probs >= threshold).astype(int)

    @staticmethod
    def get_best_threshold(probs, y_train):
        """
        Find the best threshold using ROC curve.

        Args:
            probs: Predicted probabilities
            y_train: True labels

        Returns:
            Optimal threshold
        """
        fpr, tpr, thresholds = roc_curve(y_train, probs)
        j_scores = tpr - fpr
        return thresholds[np.argmax(j_scores)]

    @staticmethod
    def failure_info(dataset):
        """
        Calculate failure rate and dataset size.

        Args:
            dataset: DataFrame with 'build_failed' column

        Returns:
            Tuple (failure rate, dataset length)
        """
        if 'build_failed' not in dataset.columns:
            raise KeyError("'build_failed' column not found in dataset")
        failures = dataset['build_failed'] > 0
        return failures.mean(), len(dataset)

    @staticmethod
    def get_entry(y_true, y_pred):
        """
        Compute evaluation metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels

        Returns:
            Dictionary of metrics (AUC, accuracy, precision, recall, F1)
        """
        metrics = {}
        metrics["AUC"] = roc_auc_score(y_true, y_pred)
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
        """
        Predict and evaluate using an LSTM model.

        Args:
            model: Trained model
            X: Input data
            y_true: True labels
            threshold: Decision threshold

        Returns:
            Metrics dictionary
        """
        from sklearn.metrics import precision_recall_curve
        import numpy as np
        y_pred_prob = model.predict(X)
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_prob)

        # Tìm ngưỡng tối ưu (e.g., tối đa hóa F1)
        f1_scores = 2 * (precision * recall) / (precision + recall)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx]
        print(f"Ngưỡng tối ưu: {optimal_threshold}")

        y_pred_probs = model.predict(X, verbose=0)
        y_pred = Utils.to_labels(y_pred_probs, optimal_threshold)
        return Utils.get_entry(y_true, y_pred)

    @staticmethod
    def is_int(n):
        """
        Check if a value is an integer.

        Args:
            n: Value to check

        Returns:
            Boolean
        """
        return isinstance(n, int)

    @staticmethod
    def online_validation_folds(dataset, start_fold=6, end_fold=11, fold_ratio=0.1):
        """
        Split dataset into training and testing folds for online validation.

        Args:
            dataset: DataFrame to split
            start_fold: Starting fold number
            end_fold: Ending fold number
            fold_ratio: Ratio of data per fold

        Returns:
            Tuple (list of train sets, list of test sets)
        """
        fold_size = int(len(dataset) * fold_ratio)
        train_sets, test_sets = [], []

        for i in range(start_fold, end_fold):
            train_end = fold_size * (i - 1)
            test_end = fold_size * i
            if test_end > len(dataset):
                test_end = len(dataset)
            train_sets.append(dataset.iloc[:train_end])
            test_sets.append(dataset.iloc[train_end:test_end])
        return train_sets, test_sets

    @staticmethod
    def frange(start, stop=None, step=None):
        """
        Generate a range of floats.

        Args:
            start: Start value
            stop: Stop value (optional)
            step: Step size (optional)

        Returns:
            Numpy array of floats
        """
        if stop is None:
            stop = start
            start = 0.0
        if step is None:
            step = 1.0
        return np.arange(start, stop, step)

    @staticmethod
    def frange_int(start, stop=None, step=None):
        """
        Generate a range of integers.

        Args:
            start: Start value
            stop: Stop value (optional)
            step: Step size (optional)

        Returns:
            Range object of integers
        """
        if stop is None:
            stop = start
            start = 0
        if step is None:
            step = 1
        return range(start, stop, step)

    # Add save_model to match single_lstm.py's requirements
    @staticmethod
    def save_model(model, path):
        """
        Save the trained model to a file.

        Args:
            model: Keras model to save
            path: Path to save the model (including filename)
        """
        model.save(path)