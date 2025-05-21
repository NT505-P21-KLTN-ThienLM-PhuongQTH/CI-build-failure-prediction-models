# src/model/stacked-lstm/preprocess.py
import random
import mlflow
import numpy as np
from imblearn.over_sampling import SMOTE
from src.helpers import Utils

def prepare_features(df, target_column='build_failed'):
    """Chuẩn bị dữ liệu cho phân tích feature importance.
    Cảnh báo nếu có cột không phải kiểu số."""
    numeric_types = ['int64', 'float64', 'int32', 'float32']

    excluded_columns = ["gh_build_started_at", "gh_project_name"]
    non_numeric_cols = [
        col for col in df.columns
        if df[col].dtype.name not in numeric_types and col not in excluded_columns
    ]
    if non_numeric_cols:
        print(f"[WARNING] The following columns are not numeric and will be discarded: {non_numeric_cols}")
    # Chuẩn bị dữ liệu đầu vào và nhãn
    X = df.select_dtypes(include=numeric_types).drop(columns=excluded_columns, errors='ignore')
    y = df[target_column]
    return X, y

def apply_smote(training_set, y):
    """
    Apply SMOTE to balance the dataset if configured to do so.

    Args:
        training_set (np.ndarray): Feature matrix for training.
        y (np.ndarray): Target labels.

    Returns:
        tuple: (balanced training_set, balanced y)
    """
    if not Utils.CONFIG['WITH_SMOTE'] or len(np.unique(y)) <= 1:
        print("SMOTE skipped: Config disabled or single class.")
        return training_set, y

    print("\nClass Distribution BEFORE SMOTE:")
    unique, counts = np.unique(y, return_counts=True)
    dist = dict(zip(unique, counts / len(y)))
    print(dist)

    print("\nApplying SMOTE...")
    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(training_set, y)

    print("Class Distribution AFTER SMOTE:")
    unique, counts = np.unique(y_smote, return_counts=True)
    dist = dict(zip(unique, counts / len(y_smote)))
    print(dist)
    return X_smote, y_smote

def train_preprocess(dataset_train, time_step, padding_module=None):
    X, y = prepare_features(dataset_train, target_column='build_failed')
    training_set = X.values

    # Padding nếu dữ liệu không đủ cho time_step
    if len(training_set) < time_step:
        if padding_module is None:
            raise ValueError(
                f"Dữ liệu huấn luyện quá ngắn ({len(training_set)} < {time_step}) và không có PaddingModule.")
        print(f"Padding dữ liệu huấn luyện từ {len(training_set)} lên {time_step}...")
        training_set = padding_module.pad_sequence(training_set, time_step)
        y = np.pad(y, (time_step - len(y), 0), mode='constant', constant_values=0)

    training_set, y_smote = apply_smote(training_set, y)

    try:
        X_train = np.lib.stride_tricks.sliding_window_view(
            training_set, (time_step, training_set.shape[1])
        )[:-1]
        X_train = np.squeeze(X_train, axis=1)
        y_train = y_smote[time_step:]
    except Exception as e:
        raise RuntimeError(f"Error during sliding window creation: {e}")
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    return X_train, y_train

def test_preprocess(dataset_train, dataset_test, time_step, padding_module=None, short_timestep=None):
    X_train, _ = prepare_features(dataset_train)
    X_test, y_test = prepare_features(dataset_test)

    print("X_train shape:", X_train.shape)
    print("X_train columns:", list(X_train.columns))
    print("\nX_test shape:", X_test.shape)
    print("X_test columns:", list(X_test.columns))

    dataset_total = np.vstack((X_train.values, X_test.values))

    if len(dataset_total) < time_step + len(dataset_test):
        if padding_module is None:
            raise ValueError("Không đủ dữ liệu cho test sequences và không có PaddingModule.")
        print(f"Padding dữ liệu test từ {len(dataset_total)} lên {time_step + len(dataset_test)}...")
        dataset_total = padding_module.pad_sequence(dataset_total, time_step + len(dataset_test))

    inputs = dataset_total[-len(dataset_test) - time_step:]
    X_test = np.lib.stride_tricks.sliding_window_view(inputs, (time_step, inputs.shape[1]))[:-1]
    X_test = np.squeeze(X_test, axis=1)
    y_test = y_test[-len(X_test):]

    # Nếu cần mô phỏng chuỗi thiếu
    if short_timestep is not None and short_timestep < time_step:
        print(f"Giả lập thiếu chuỗi: giữ {short_timestep} dòng cuối, rồi padding lại về {time_step}...")
        padded_sequences = []
        for seq in X_test:
            short_seq = seq[-short_timestep:]
            padded_seq = padding_module.pad_sequence(short_seq, time_step)
            padded_sequences.append(padded_seq)
        X_test = np.array(padded_sequences)
        assert len(X_test) == len(y_test), "Số lượng X_test và y_test không khớp!"

    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return X_test, y_test