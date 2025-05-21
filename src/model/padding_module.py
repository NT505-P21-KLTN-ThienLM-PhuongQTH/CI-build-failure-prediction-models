import os
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import EarlyStopping
import mlflow
from src.model.stacked_lstm.preprocess import prepare_features

class PaddingModule:
    def __init__(self, input_dim, time_step, units=64, drop_proba=0.1):
        self.input_dim = input_dim
        self.time_step = time_step
        self.units = units
        self.drop_proba = drop_proba
        self.model = self._build_model()

    def _build_model(self):
        """Xây dựng mô hình LSTM để sinh dữ liệu padding."""
        model = Sequential()
        model.add(Input(shape=(None, self.input_dim)))  # None để chấp nhận chuỗi ngắn
        model.add(LSTM(units=self.units, return_sequences=True))
        model.add(Dropout(self.drop_proba))
        model.add(LSTM(units=self.units))
        model.add(Dropout(self.drop_proba))
        model.add(Dense(self.input_dim))  # Dự đoán vector đặc trưng cho mỗi timestep
        model.compile(optimizer='adam', loss='mse')
        return model

    def prepare_training_data(self, datasets, min_length=5):
        """
        Chuẩn bị dữ liệu huấn luyện cho PaddingModule.
        Args:
            datasets: Dictionary chứa các DataFrame của repository.
            min_length: Độ dài tối thiểu của chuỗi thực tế.
        Returns:
            X_train: Dữ liệu đầu vào (chuỗi ngắn).
            y_train: Dữ liệu mục tiêu (vector đặc trưng của timestep tiếp theo).
        """
        X_train = []
        y_train = []

        for repo_name, df in datasets.items():
            X, _ = prepare_features(df, target_column='build_failed')
            data = X.values
            if len(data) < self.time_step:
                continue  # Bỏ qua repository ngắn

            # Tạo các mẫu huấn luyện
            for i in range(min_length, len(data) - 1):
                # Chuỗi ngắn ngẫu nhiên từ 1 đến min(time_step, i)
                short_length = np.random.randint(1, min(self.time_step, i) + 1)
                short_seq = data[i - short_length:i]
                target = data[i]  # Vector đặc trưng của timestep tiếp theo
                X_train.append(short_seq)
                y_train.append(target)

        # Chuyển thành numpy array
        X_train = [np.array(seq) for seq in X_train]
        y_train = np.array(y_train)
        return X_train, y_train

    def train(self, datasets, epochs=20, batch_size=32):
        """Huấn luyện PaddingModule."""
        X_train, y_train = self.prepare_training_data(datasets)
        if len(X_train) == 0:
            raise ValueError("Không đủ dữ liệu để huấn luyện PaddingModule.")

        # Chuẩn bị dữ liệu cho LSTM (padding chuỗi ngắn)
        max_len = max(len(seq) for seq in X_train)
        X_padded = np.zeros((len(X_train), max_len, self.input_dim))
        for i, seq in enumerate(X_train):
            X_padded[i, -len(seq):, :] = seq

        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
        self.model.fit(X_padded, y_train, epochs=epochs, batch_size=batch_size,
                       validation_split=0.2, verbose=1, callbacks=[es])

    def pad_sequence(self, short_seq, target_length):
        """
        Sinh dữ liệu padding để đạt độ dài target_length.
        Args:
            short_seq: Chuỗi ngắn (numpy array shape: [short_len, input_dim]).
            target_length: Độ dài mong muốn (thường là time_step).
        Returns:
            Padded sequence (shape: [target_length, input_dim]).
        """
        if len(short_seq) >= target_length:
            return short_seq[:target_length]

        padded_seq = np.zeros((target_length, self.input_dim))
        padded_seq[-len(short_seq):] = short_seq  # Đặt chuỗi thực tế vào cuối

        # Sinh các timestep padding
        current_seq = short_seq
        for i in range(target_length - len(short_seq)):
            input_seq = current_seq[np.newaxis, :, :]  # Thêm batch dimension
            next_vector = self.model.predict(input_seq, verbose=0)
            padded_seq[target_length - len(short_seq) - 1 - i] = next_vector
            current_seq = np.vstack([next_vector, current_seq])  # Cập nhật chuỗi đầu vào
            if len(current_seq) > self.time_step:
                current_seq = current_seq[-self.time_step:]  # Giới hạn độ dài

        return padded_seq

    def save_model(self, model_name):
        """Đăng ký mô hình PaddingModule vào MLflow Model Registry."""
        if mlflow.active_run():
            mlflow.keras.log_model(self.model, artifact_path="padding_model")
            # Đăng ký mô hình vào Model Registry
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/padding_model"
            mlflow.register_model(model_uri, model_name)
            print(f"Mô hình đã được đăng ký với tên {model_name} tại {model_uri}")
        else:
            raise RuntimeError("Không có MLflow run đang hoạt động, không thể đăng ký mô hình.")

    def load_model(self, path):
        """Tải mô hình PaddingModule."""
        self.model = mlflow.keras.load_model(path)