import sys
import os
import numpy as np
import pandas as pd
from keras.models import load_model
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.append(project_root)
import src.utils.Utils as Utils

# Đường dẫn mô hình đã lưu
MODEL_DIR = "models"
best_model_path = os.path.join(MODEL_DIR, "best_lstm_model.h5")

def test_preprocess(dataset_train, dataset_test, time_step):
    """
    Preprocess the testing dataset for LSTM.

    Args:
        dataset_train (DataFrame): Training dataset.
        dataset_test (DataFrame): Testing dataset.
        time_step (int): Number of time steps for LSTM.

    Returns:
        tuple: Processed X_test and y_test.
    """
    y_test = dataset_test.iloc[:, 0:1].values
    dataset_total = pd.concat((dataset_train['build_Failed'], dataset_test['build_Failed']), axis=0)
    inputs = dataset_total[len(dataset_total) - len(dataset_test) - time_step:].values.reshape(-1, 1)

    X_test = np.array([inputs[j - time_step:j, 0] for j in range(time_step, len(inputs))])

    return X_test.reshape(-1, time_step, 1), y_test

# Main script để test
if __name__ == "__main__":
    # Tải mô hình
    model = load_model(best_model_path)
    print(f"Loaded model from {best_model_path}")

    # Tải dataset và chia train/test (giả sử đã có từ trước)
    dataset = Utils.get_dataset("ruby.csv")
    train_sets, test_sets = Utils.online_validation_folds(dataset)

    # Giả sử bạn đã có entry_train_ga từ lần huấn luyện trước
    # Nếu không, cần lấy time_step từ file hoặc đặt thủ công
    time_step = 30  # Thay bằng entry_train_ga["params"]["time_step"] nếu có

    # Tiền xử lý dữ liệu test
    X_test, y_test = test_preprocess(train_sets[0], test_sets[0], time_step)
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    # Dự đoán
    predictions = model.predict(X_test)
    print(f"Predictions shape: {predictions.shape}")

    # Đánh giá
    entry_test = Utils.predict_lstm(model, X_test, y_test)
    print("Test Results:", entry_test)
