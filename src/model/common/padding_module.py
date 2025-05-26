import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Flatten, Layer, Conv1D
from keras.callbacks import EarlyStopping
from keras.losses import MeanSquaredError
from keras.optimizers import Adam
from keras.initializers import GlorotUniform
import mlflow
import tensorflow as tf
from src.model.common.preprocess import prepare_features
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, median_absolute_error
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class TrainablePaddingLayer(Layer):
    def __init__(self, input_dim, kernel_size=3, **kwargs):
        super(TrainablePaddingLayer, self).__init__(**kwargs)
        self.input_dim = input_dim
        self.kernel_size = kernel_size
        self.conv = Conv1D(filters=input_dim, kernel_size=kernel_size, padding='valid',
                          kernel_initializer=GlorotUniform(seed=42))

    def call(self, inputs):
        left_pad = self.conv(inputs[:, :self.kernel_size, :])
        right_pad = self.conv(inputs[:, -self.kernel_size:, :])
        padded = tf.concat([left_pad, inputs, right_pad], axis=1)
        return padded

class PaddingModule:
    def __init__(self, input_dim, time_step, units=64, drop_proba=0.1):
        self.input_dim = input_dim
        self.time_step = time_step
        self.units = units
        self.drop_proba = drop_proba
        self.model = self._build_model()
        self.y_scaler = MinMaxScaler()
        self.zeroed_features = []  # Danh sách các feature đã bị gán bằng 0
        self.feature_names = None

    def _build_model(self):
        model = Sequential()
        model.add(TrainablePaddingLayer(input_dim=self.input_dim, kernel_size=3,
                                        input_shape=(self.time_step, self.input_dim)))
        model.add(Flatten())
        model.add(Dense(self.input_dim * 6))
        model.compile(optimizer=Adam(learning_rate=0.001), loss=MeanSquaredError())
        return model

    def prepare_training_data(self, datasets, max_samples=11000):
        X_train = []
        y_train = []

        for repo_name in sorted(datasets):
            df = datasets[repo_name]
            X, _ = prepare_features(df, target_column='build_failed')
            if not hasattr(self, "feature_names") or self.feature_names is None:
                self.feature_names = X.columns.tolist()
                print("Captured feature names:", self.feature_names)
            # Gán các feature trong zeroed_features bằng 0
            for feature in self.zeroed_features:
                if feature in X.columns:
                    X[feature] = 0
            data = X.values
            if len(data) <= self.time_step:
                continue

            for i in range(self.time_step, len(data)):
                window = data[i - self.time_step:i]
                T = np.concatenate([window[0:3], window[-3:]], axis=0)
                P = window
                X_train.append(P)
                y_train.append(T)
                if len(X_train) >= max_samples:
                    break
            if len(X_train) >= max_samples:
                break

        X_train = np.array(X_train)
        y_train = np.array(y_train)
        if y_train.ndim == 3:
            n_samples, border_len, input_dim = y_train.shape
            y_train = y_train.reshape((n_samples, border_len * input_dim))
        y_train = self.y_scaler.fit_transform(y_train)
        return X_train, y_train

    def train(self, datasets, epochs=10, batch_size=16, r2_threshold=0.5, max_iterations=5):
        iteration = 0
        best_r2 = -float('inf')
        best_metrics = None
        best_zeroed_features = []

        while iteration < max_iterations:
            print(f"\n🔄 Iteration {iteration + 1}/{max_iterations}")
            X_train, y_train = self.prepare_training_data(datasets)
            if len(X_train) == 0:
                raise ValueError("Không đủ dữ liệu để huấn luyện PaddingModule.")

            if X_train.ndim == 2:
                X_train = X_train[:, np.newaxis, :]

            if self.feature_names is None:
                raise ValueError("Feature names are not initialized. Ensure prepare_training_data is called first.")

            X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

            es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=5)
            self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                           validation_data=(X_val, y_val), verbose=1, callbacks=[es])

            # Đánh giá mô hình
            metrics = self.evaluate(datasets, test_split=0.2)
            current_r2 = metrics['R²']

            print(f"R² hiện tại: {current_r2:.4f}")
            if current_r2 > r2_threshold:
                print(f"✅ Đạt R² > {r2_threshold}. Dừng lại.")
                best_r2 = current_r2
                best_metrics = metrics
                best_zeroed_features = self.zeroed_features.copy()
                break

            # Tìm các feature có R² thấp
            X_test, y_test = self.prepare_training_data(datasets)
            n_test = int(len(X_test) * 0.2)
            X_test = X_test[-n_test:]
            y_test = y_test[-n_test:]
            y_pred = self.model.predict(X_test, verbose=0)
            y_pred = self.y_scaler.inverse_transform(np.clip(y_pred, 0.0, 1.0 - 1e-6))
            y_test = self.y_scaler.inverse_transform(y_test)

            output_feature_names = []
            prefixes = [f"head_{i}" for i in range(3)] + [f"tail_{i}" for i in range(3)]
            for prefix in prefixes:
                output_feature_names.extend([f"{prefix}:{name}" for name in self.feature_names])

            r2_scores = [r2_score(y_test[:, i], y_pred[:, i]) for i in range(y_test.shape[1])]
            r2_with_names = list(zip(output_feature_names, r2_scores))
            low_r2_features = [name for name, score in r2_with_names if score < -0.1]  # Ngưỡng R² âm

            if not low_r2_features:
                print("Không còn feature nào có R² quá thấp. Dừng lại.")
                best_r2 = current_r2
                best_metrics = metrics
                best_zeroed_features = self.zeroed_features.copy()
                break

            # Gán feature gốc tương ứng bằng 0
            for output_name in low_r2_features:
                feature_name = output_name.split(":")[1]  # Lấy tên feature gốc
                if feature_name not in self.zeroed_features:
                    self.zeroed_features.append(feature_name)
                    print(f"Feature {feature_name} đã được gán bằng 0.")

            iteration += 1
            # Rebuild model để đảm bảo input_dim không thay đổi
            self.model = self._build_model()

        print(f"\n🏁 Kết thúc sau {iteration + 1} lần lặp.")
        print(f"R² tốt nhất: {best_r2:.4f}")
        print(f"Features đã gán bằng 0: {best_zeroed_features}")
        print("Metrics tốt nhất:", best_metrics)
        return best_metrics, best_zeroed_features

    def evaluate(self, datasets, test_split=0.2):
        X_train, y_train = self.prepare_training_data(datasets)
        if len(X_train) == 0:
            raise ValueError("Không đủ dữ liệu để đánh giá PaddingModule.")

        n_samples = len(X_train)
        n_test = int(n_samples * test_split)
        X_test = X_train[-n_test:]
        y_test = y_train[-n_test:]

        y_pred = self.model.predict(X_test, verbose=0)
        mse_norm = mean_squared_error(y_test, y_pred)
        mae_norm = mean_absolute_error(y_test, y_pred)
        r2_norm = r2_score(y_test, y_pred)

        y_pred = np.clip(y_pred, 0.0, 1.0 - 1e-6)
        y_pred = self.y_scaler.inverse_transform(y_pred)
        y_test = self.y_scaler.inverse_transform(y_test)

        mse = mean_squared_error(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        medae = median_absolute_error(y_test, y_pred)

        print(f"Evaluation Metrics (Iteration):")
        print(f"MSE: {mse:.4f}, MAE: {mae:.4f}, R²: {r2:.4f}, MedAE: {medae:.4f}")

        if mlflow.active_run():
            mlflow.log_metric("test_mse", mse)
            mlflow.log_metric("test_mae", mae)
            mlflow.log_metric("test_r2", r2)
            mlflow.log_metric("test_medae", medae)

        return {
            "MSE": mse,
            "MAE": mae,
            "R²": r2,
            "MedAE": medae,
            "MSE_norm": mse_norm,
            "MAE_norm": mae_norm,
            "R²_norm": r2_norm
        }

    def save_model(self, model_name):
        if mlflow.active_run():
            mlflow.keras.log_model(self.model, artifact_path="padding_model")
            model_uri = f"runs:/{mlflow.active_run().info.run_id}/padding_model"
            mlflow.register_model(model_uri, model_name)
            print(f"Mô hình đã được đăng ký với tên {model_name} tại {model_uri}")
        else:
            raise RuntimeError("Không có MLflow run đang hoạt động, không thể đăng ký mô hình.")

    def load_model(self, path):
        self.model = mlflow.keras.load_model(path)