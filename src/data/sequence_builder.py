# src/data/sequence_builder.py
import numpy as np
import pandas as pd
import logging
from sklearn.preprocessing import MinMaxScaler

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_feature_columns(dataset):
    """
    Lọc các cột đặc trưng từ dataset, loại bỏ các cột không phải đặc trưng.

    Args:
        dataset (pd.DataFrame): Dataset đầu vào.

    Returns:
        list: Danh sách các cột đặc trưng.
    """
    return [col for col in dataset.columns
            if col not in ['build_failed', 'gh_build_started_at', 'gh_project_name']
            and dataset[col].dtype in [np.float64, np.float32, np.int64, np.int32]]


def normalize_data(train_data, test_data=None):
    """
    Chuẩn hóa dữ liệu bằng MinMaxScaler.

    Args:
        train_data (np.ndarray): Dữ liệu huấn luyện.
        test_data (np.ndarray, optional): Dữ liệu kiểm tra. Nếu có, sẽ được chuẩn hóa bằng scaler từ train_data.

    Returns:
        tuple: (train_scaled, test_scaled, scaler) hoặc (train_scaled, None, scaler) nếu không có test_data.
    """
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train_data)
    logger.info(f"Min and Max after scaling (train): {train_scaled.min()}, {train_scaled.max()}")

    if test_data is not None:
        test_scaled = scaler.transform(test_data)
        logger.info(f"Min and Max after scaling (test): {test_scaled.min()}, {test_scaled.max()}")
        return train_scaled, test_scaled, scaler
    return train_scaled, None, scaler


def calculate_rows_cols(num_features):
    """
    Tính toán rows và cols dựa trên số đặc trưng.

    Args:
        num_features (int): Số đặc trưng.

    Returns:
        tuple: (rows, cols)
    """
    # Tìm cặp rows và cols sao cho rows * cols = num_features
    # Ưu tiên cols nhỏ (ví dụ: 2 hoặc 3)
    for cols in range(2, int(np.sqrt(num_features)) + 1):
        if num_features % cols == 0:
            rows = num_features // cols
            return rows, cols
    # Nếu không tìm thấy cặp chia hết, chọn cols=1
    return num_features, 1


def build_convlstm_sequences(data, time_step):
    """
    Tạo chuỗi dữ liệu 5D cho ConvLSTM: (samples, time_steps, rows, cols, channels).

    Args:
        data (np.ndarray): Dữ liệu đầu vào (đã chuẩn hóa).
        time_step (int): Số bước thời gian.

    Returns:
        np.ndarray: Dữ liệu đã được định dạng thành chuỗi 5D.
    """
    num_features = data.shape[1]
    logger.info(f"Number of features: {num_features}")

    # Tính toán rows và cols dựa trên num_features
    rows, cols = calculate_rows_cols(num_features)
    logger.info(f"Calculated rows={rows}, cols={cols} for {num_features} features")

    X_seq = np.lib.stride_tricks.sliding_window_view(data, (time_step, data.shape[1]))[:-1]
    X_seq = np.squeeze(X_seq, axis=1)

    samples = X_seq.shape[0]
    X = np.zeros((samples, time_step, rows, cols, 1))
    for i in range(samples):
        X[i] = X_seq[i].reshape(time_step, rows, cols, 1)
    return X


def preprocess_for_convlstm_train(dataset_train, time_step, with_smote=False):
    """
    Xử lý dữ liệu huấn luyện cho ConvLSTM.

    Args:
        dataset_train (pd.DataFrame): Dataset huấn luyện.
        time_step (int): Số bước thời gian.
        with_smote (bool): Có áp dụng SMOTE hay không.

    Returns:
        tuple: (X_train, y_train, scaler)
    """
    if not isinstance(dataset_train, pd.DataFrame):
        raise TypeError(f"Expected dataset_train to be a pandas DataFrame, got {type(dataset_train)}")

    if len(dataset_train) <= time_step:
        raise ValueError(f"Dataset size ({len(dataset_train)}) must be larger than time_step ({time_step})")

    # Lọc cột đặc trưng và tách nhãn
    feature_cols = get_feature_columns(dataset_train)
    training_set = dataset_train[feature_cols].values
    y = dataset_train['build_failed'].values

    # Chuẩn hóa dữ liệu
    training_set, _, scaler = normalize_data(training_set)

    # Tạo chuỗi dữ liệu
    X_train = build_convlstm_sequences(training_set, time_step)
    y_train = y[time_step:]

    logger.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    return X_train, y_train, scaler


def preprocess_for_convlstm_test(dataset_train, dataset_test, time_step, scaler):
    """
    Xử lý dữ liệu kiểm tra cho ConvLSTM.

    Args:
        dataset_train (pd.DataFrame): Dataset huấn luyện.
        dataset_test (pd.DataFrame): Dataset kiểm tra.
        time_step (int): Số bước thời gian.
        scaler (MinMaxScaler): Scaler đã được huấn luyện trên tập huấn luyện.

    Returns:
        tuple: (X_test, y_test)
    """
    if not isinstance(dataset_train, pd.DataFrame) or not isinstance(dataset_test, pd.DataFrame):
        raise TypeError("Both dataset_train and dataset_test must be pandas DataFrames")

    # Lọc cột đặc trưng
    feature_cols = get_feature_columns(dataset_train)
    train_data = dataset_train[feature_cols].values
    test_data = dataset_test[feature_cols].values
    y_test = dataset_test['build_failed'].values

    # Chuẩn hóa dữ liệu
    train_scaled = scaler.fit_transform(train_data)
    test_scaled = scaler.transform(test_data)
    dataset_total_scaled = np.vstack((train_scaled, test_scaled))

    if len(dataset_total_scaled) < time_step + len(dataset_test):
        raise ValueError("Not enough data to create test sequences with given time_step")

    # Tạo chuỗi dữ liệu
    inputs = dataset_total_scaled[len(dataset_total_scaled) - len(dataset_test) - time_step:]
    X_test = build_convlstm_sequences(inputs, time_step)

    logger.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return X_test, y_test