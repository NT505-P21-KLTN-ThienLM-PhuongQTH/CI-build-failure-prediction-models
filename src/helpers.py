from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, roc_curve, \
    average_precision_score, precision_recall_curve
import numpy as np
import pandas as pd
import warnings
import os

class Utils:
    # Move CONFIG to utils for now (can be moved to config.py later)
    CONFIG = {
        'NBR_GEN': 2,
        'NBR_SOL': 2,
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
    def get_best_threshold(y_true, y_pred_probs, metric="f1"):
        """
        Tìm ngưỡng tối ưu dựa trên F1-score hoặc một metric khác.

        Parameters:
        - y_true: Nhãn thực tế (ground truth).
        - y_pred_probs: Xác suất dự đoán cho lớp positive (class 1).
        - metric: Metric để tối ưu hóa, mặc định là "f1".

        Returns:
        - best_threshold: Ngưỡng tối ưu.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_probs)

        # Tính F1-score cho từng ngưỡng
        f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)  # Thêm 1e-10 để tránh chia cho 0

        # Tìm ngưỡng có F1-score cao nhất
        best_idx = np.argmax(f1_scores)
        best_threshold = thresholds[best_idx]

        print(f"Best threshold (max F1): {best_threshold:.4f}, F1-score: {f1_scores[best_idx]:.4f}")
        return best_threshold

    @staticmethod
    def get_entry(y_true, y_pred_probs, y_pred):
        # Get metrics for a single entry
        metrics = {}
        metrics["AUC"] = roc_auc_score(y_true, y_pred_probs)
        metrics["accuracy"] = accuracy_score(y_true, y_pred)
        metrics["PR_AUC"] = average_precision_score(y_true, y_pred_probs)
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
        threshold = Utils.get_best_threshold(y_true, y_pred_probs)
        # threshold = Utils.get_best_threshold(y_true, y_pred_probs)
        print(f"\nUsing threshold: {threshold}")
        y_pred = Utils.to_labels(y_pred_probs, threshold)
        return Utils.get_entry(y_true, y_pred_probs, y_pred)

    @staticmethod
    def is_int(n):
        return isinstance(n, int)

    @staticmethod
    def online_validation_folds(dataset, n_folds=10, window_size=1000, step_size=500):
        """
        Chia dữ liệu time series thành các fold với cửa sổ trượt.

        Args:
            dataset: DataFrame chứa dữ liệu time series, đã được sắp xếp theo thời gian.
            n_folds: Số lượng fold.
            window_size: Kích thước cửa sổ huấn luyện.
            step_size: Bước trượt giữa các fold.

        Returns:
            train_sets, test_sets: List các tập huấn luyện và tập validation.
        """
        dataset = dataset.sort_values(by='gh_build_started_at')  # Đảm bảo dữ liệu được sắp xếp theo thời gian
        total_length = len(dataset)
        train_sets = []
        test_sets = []

        for fold in range(n_folds):
            start_idx = fold * step_size
            train_end_idx = start_idx + window_size
            test_end_idx = train_end_idx + (total_length // n_folds)

            if test_end_idx > total_length:
                break

            train_set = dataset.iloc[start_idx:train_end_idx]
            test_set = dataset.iloc[train_end_idx:test_end_idx]

            print(f"Fold {fold + 1}: Train {start_idx}-{train_end_idx}, Test {train_end_idx}-{test_end_idx}")
            train_sets.append(train_set)
            test_sets.append(test_set)

        return train_sets, test_sets