import os
import numpy as np
import pandas as pd
from keras.models import load_model
import src.utils.Utils as Utils
import pickle
import logging

# Thiết lập logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Đường dẫn mô hình và scaler đã lưu
MODEL_DIR = "models/single_lstm"
best_model_path = os.path.join(MODEL_DIR, "best_lstm_model.keras")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")  # Đường dẫn để lưu scaler từ train

def test_preprocess(dataset_train, dataset_test, time_step, scaler):
    """
    Preprocess the testing dataset for LSTM using the trained scaler.

    Args:
        dataset_train (DataFrame): Training dataset (to create sequence continuity).
        dataset_test (DataFrame): Testing dataset.
        time_step (int): Number of time steps for LSTM.
        scaler (MinMaxScaler): Scaler fitted from training phase.

    Returns:
        tuple: Processed X_test and y_test.
    """
    feature_cols = [col for col in dataset_train.columns 
                    if col not in [ 'gh_build_started_at', 'gh_project_name'] 
                    and dataset_train[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
    
    # Kiểm tra NA
    if dataset_train[feature_cols].isna().any().any() or dataset_test[feature_cols].isna().any().any():
        raise ValueError("Dataset contains NaN values in feature columns")

    # Chuẩn hóa dữ liệu bằng scaler đã fit
    train_scaled = scaler.transform(dataset_train[feature_cols])
    test_scaled = scaler.transform(dataset_test[feature_cols])
    dataset_total_scaled = np.vstack((train_scaled, test_scaled))
    y_test = dataset_test['build_failed'].values

    if len(dataset_total_scaled) < time_step + len(dataset_test):
        raise ValueError(f"Not enough data to create test sequences: total {len(dataset_total_scaled)}, required {time_step + len(dataset_test)}")

    # Tạo sequence
    inputs = dataset_total_scaled[len(dataset_total_scaled) - len(dataset_test) - time_step:]
    X_test = np.lib.stride_tricks.sliding_window_view(inputs, (time_step, inputs.shape[1]))[:-1]
    X_test = np.squeeze(X_test, axis=1)

    logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return X_test, y_test

if __name__ == "__main__":
    # Load model
    model = load_model(best_model_path)
    logging.info(f"Loaded model from {best_model_path}")

    # Load scaler (giả sử đã lưu từ giai đoạn train)
    try:
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        logging.info(f"Loaded scaler from {scaler_path}")
    except FileNotFoundError:
        logging.error(f"Scaler file {scaler_path} not found. Please ensure it was saved during training.")
        raise

    # Load dataset
    dataset = Utils.get_dataset("StackStorm_st2.csv")
    logging.info("Dataset Info:")
    logging.info(dataset.info())
    
    train_sets, test_sets = Utils.online_validation_folds(dataset)
    logging.info(f"Number of train folds: {len(train_sets)}, test folds: {len(test_sets)}")

    # Giả sử bạn có time_step từ huấn luyện (cần lưu trong file hoặc lấy từ cấu hình)
    time_step = 42  # Thay bằng giá trị thực tế từ entry_train_ga["params"]["time_step"] nếu có
    logging.info(f"Using time_step: {time_step}")

    # Tiền xử lý và dự đoán
    X_test, y_test = test_preprocess(train_sets[0], test_sets[0], time_step, scaler)
    
    predictions = model.predict(X_test)
    logging.info(f"Predictions shape: {predictions.shape}")

    entry_test = Utils.predict_lstm(model, X_test, y_test)
    logging.info("Test Results:")
    logging.info(entry_test)

    # Phân bố lớp trong test set
    # logging.info("Class distribution in test set:")
    # logging.info(pd.Series(y_test).value_counts(normalize=True))
    
    # Kiểm tra correlation để xem feature nào ảnh hưởng
    feature_cols = [col for col in dataset.columns if col not in ['build_failed', 'gh_build_started_at', 'gh_project_name']]
    correlation = dataset[feature_cols + ['build_failed']].corr()['build_failed'].sort_values()
    logging.info("Correlation with build_failed:")
    logging.info(correlation)