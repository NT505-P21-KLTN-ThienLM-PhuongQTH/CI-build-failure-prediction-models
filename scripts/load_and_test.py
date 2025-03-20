import os
import numpy as np
import pandas as pd
from keras.models import load_model
import src.utils.Utils as Utils

# Đường dẫn mô hình đã lưu
MODEL_DIR = "models/single_lstm"
best_model_path = os.path.join(MODEL_DIR, "best_lstm_model.h5")

def test_preprocess(dataset_train, dataset_test, time_step):
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()

    dataset_total = pd.concat((dataset_train.iloc[:, :19], dataset_test.iloc[:, :19]), axis=0)
    dataset_total_scaled = scaler.fit_transform(dataset_total)

    if len(dataset_total) < time_step + len(dataset_test):
        raise ValueError("Not enough data to create test sequences with given time_step")

    inputs = dataset_total_scaled[len(dataset_total) - len(dataset_test) - time_step:]
    X_test = np.array([inputs[j - time_step:j, :] for j in range(time_step, len(inputs))])
    y_test = dataset_test.iloc[:, 0].values

    return X_test, y_test

if __name__ == "__main__":
    # Load model
    model = load_model(best_model_path)
    print(f"Loaded model from {best_model_path}")

    # Load dataset
    dataset = Utils.get_dataset("jruby.csv")
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
