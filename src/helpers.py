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
    def get_best_threshold(y_true, y_pred_probs, priority=None, min_recall=None, min_precision=None):
        """
        Tìm ngưỡng tối ưu dựa trên metric được ưu tiên.

        Parameters:
        - y_true: Nhãn thực tế (ground truth).
        - y_pred_probs: Xác suất dự đoán cho lớp positive (class 1).
        - priority: Metric để tối ưu hóa ("f1" cho F1-score, "recall" để ưu tiên recall, "precision" để ưu tiên precision).
        - min_recall: Giá trị tối thiểu của recall (nếu có), chỉ áp dụng khi priority="recall".
        - min_precision: Giá trị tối thiểu của precision (nếu có), chỉ áp dụng khi priority="precision".

        Returns:
        - best_threshold: Ngưỡng tối ưu.
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_pred_probs)

        if priority == "f1":
            # Tính F1-score cho từng ngưỡng
            f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)
            best_idx = np.argmax(f1_scores)
            best_threshold = thresholds[best_idx]
            print(f"Best threshold (max F1): {best_threshold:.4f}, "
                  f"F1-score: {f1_scores[best_idx]:.4f}, "
                  f"Precision: {precision[best_idx]:.4f}, "
                  f"Recall: {recall[best_idx]:.4f}")

        elif priority == "recall":
            if min_recall is not None:
                valid_indices = np.where(recall >= min_recall)[0]
                if len(valid_indices) == 0:
                    print(f"No threshold satisfies min_recall={min_recall}. Using threshold with highest recall.")
                    best_idx = np.argmax(recall)
                else:
                    # Trong số các ngưỡng thỏa mãn, chọn ngưỡng có precision cao nhất
                    valid_precisions = precision[valid_indices]
                    best_idx = valid_indices[np.argmax(valid_precisions)]
            else:
                best_idx = np.argmax(recall)

            best_threshold = thresholds[best_idx]
            print(f"Best threshold (priority=recall, min_recall={min_recall}): {best_threshold:.4f}, "
                  f"Recall: {recall[best_idx]:.4f}, "
                  f"Precision: {precision[best_idx]:.4f}, "
                  f"F1-score: {2 * (precision[best_idx] * recall[best_idx]) / (precision[best_idx] + recall[best_idx] + 1e-10):.4f}")

        elif priority == "precision":
            if min_precision is not None:
                valid_indices = np.where(precision >= min_precision)[0]
                if len(valid_indices) == 0:
                    print(f"No threshold satisfies min_precision={min_precision}. Using threshold with highest precision.")
                    best_idx = np.argmax(precision)
                else:
                    # Trong số các ngưỡng thỏa mãn, chọn ngưỡng có recall cao nhất
                    valid_recalls = recall[valid_indices]
                    best_idx = valid_indices[np.argmax(valid_recalls)]
            else:
                # Nếu không có yêu cầu tối thiểu, chọn ngưỡng có precision cao nhất
                best_idx = np.argmax(precision)

            best_threshold = thresholds[best_idx]
            print(f"Best threshold (priority=precision, min_precision={min_precision}): {best_threshold:.4f}, "
                  f"Precision: {precision[best_idx]:.4f}, "
                  f"Recall: {recall[best_idx]:.4f}, "
                  f"F1-score: {2 * (precision[best_idx] * recall[best_idx]) / (precision[best_idx] + recall[best_idx] + 1e-10):.4f}")

        else:
            raise ValueError("Priority must be one of 'f1', 'recall', or 'precision'")

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
    def predict_lstm(model, X, y_true, threshold=None):
        y_pred_probs = model.predict(X, verbose=0)
        if y_true is not None:
            threshold = Utils.get_best_threshold(y_true, y_pred_probs, priority="f1")
            print(f"\nUsing threshold: {threshold}")
            y_pred = Utils.to_labels(y_pred_probs, threshold)
            metrics = Utils.get_entry(y_true, y_pred_probs, y_pred)
            return metrics, threshold
        else:
            if threshold is None:
                raise ValueError("Threshold must be provided for inference when y_true is not available.")
            y_pred = Utils.to_labels(y_pred_probs, threshold)
            return y_pred, y_pred_probs

    @staticmethod
    def is_int(n):
        return isinstance(n, int)

    @staticmethod
    def online_validation_folds(dataset, n_folds=10, window_size=1000, step_size=600):
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
        # dataset = dataset.sort_values(by='gh_build_started_at')  # Đã sort khi preprocess data
        total_length = len(dataset)
        train_sets = []
        test_sets = []

        for fold in range(n_folds):
            start_idx = fold * step_size
            train_end_idx = start_idx + window_size
            # test_end_idx = train_end_idx + (total_length // n_folds)
            test_size = int(window_size * 0.2)
            test_end_idx = train_end_idx + test_size
            if test_end_idx > total_length:
                break

            train_set = dataset.iloc[start_idx:train_end_idx]
            test_set = dataset.iloc[train_end_idx:test_end_idx]

            print(f"Fold {fold + 1}: Train {start_idx}-{train_end_idx}, Test {train_end_idx}-{test_end_idx}")
            train_sets.append(train_set)
            test_sets.append(test_set)

        return train_sets, test_sets