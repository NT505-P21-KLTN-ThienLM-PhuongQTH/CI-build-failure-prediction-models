import os
import numpy as np
import pandas as pd
from keras.models import load_model
import src.utils.Utils as Utils

# Đường dẫn mô hình đã lưu
MODEL_DIR = "models/single_lstm"
best_model_path = os.path.join(MODEL_DIR, "best_lstm_model.h5")

def test_preprocess(dataset_train, dataset_test, time_step):
    feature_cols = [col for col in dataset_train.columns 
                    if col not in ['build_failed', 'gh_build_started_at', 'gh_project_name'] 
                    and dataset_train[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    # Normalize data
    train_scaled = scaler.fit_transform(dataset_train[feature_cols])
    test_scaled = scaler.transform(dataset_test[feature_cols])
    dataset_total_scaled = np.vstack((train_scaled, test_scaled))
    y_test = dataset_test['build_failed'].values # Target column
        
    if len(dataset_total_scaled) < time_step + len(dataset_test):
        raise ValueError("Not enough data to create test sequences with given time_step")
    
    inputs = dataset_total_scaled[len(dataset_total_scaled) - len(dataset_test) - time_step:]
    X_test = np.lib.stride_tricks.sliding_window_view(inputs, (time_step, inputs.shape[1]))[:-1]
    X_test = np.squeeze(X_test, axis=1)

    return X_test, y_test

if __name__ == "__main__":
    # Load model
    model = load_model(best_model_path)
    print(f"Loaded model from {best_model_path}")

    # Load dataset
    dataset = Utils.get_dataset("CartoDB_cartodb.csv")
    train_sets, test_sets = Utils.online_validation_folds(dataset)

    # Giả sử bạn đã có entry_train_ga từ lần huấn luyện trước
    # Nếu không, cần lấy time_step từ file hoặc đặt thủ công
    time_step = 30  # Thay bằng entry_train_ga["params"]["time_step"] nếu có

    X_test, y_test = test_preprocess(train_sets[0], test_sets[0], time_step)
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")

    predictions = model.predict(X_test)
    print(f"Predictions shape: {predictions.shape}")

    entry_test = Utils.predict_lstm(model, X_test, y_test)
    print("Test Results:", entry_test)
