import os
from src.helpers import Utils
import pandas as pd
from typing import List, Tuple
from sklearn.preprocessing import StandardScaler

def preprocess_training(dataset_dir):
    """
    Load and validate datasets from the given directory (already preprocessed).

    Args:
        dataset_dir (str): Directory containing preprocessed datasets (e.g., data/processed-local).

    Returns:
        dict: Dictionary of loaded datasets with file names as keys.
    """
    if not os.path.exists(dataset_dir):
        raise FileNotFoundError(f"Dataset directory {dataset_dir} does not exist.")

    datasets = {}
    # List all CSV files in the dataset directory
    for file_name in os.listdir(dataset_dir):
        if not file_name.endswith(".csv"):
            continue

        print(f"Loading preprocessed dataset {file_name}...")
        try:
            # Step 1: Load the dataset using Utils.get_dataset
            dataset = Utils.get_dataset(file_name, dataset_dir)

            # Step 2: Validate the dataset
            # Ensure the target column 'build_failed' exists
            if 'build_failed' not in dataset.columns:
                raise ValueError(f"Target column 'build_failed' not found in {file_name}")

            # Ensure the timestamp column 'gh_build_started_at' exists
            if 'gh_build_started_at' not in dataset.index.name:
                raise ValueError(f"Index column 'gh_build_started_at' not set in {file_name}")

            # Step 3: Ensure data is sorted by timestamp (already handled in Utils.get_dataset, but double-check)
            # dataset.sort_values(by=['gh_build_started_at'], inplace=True)

            # Add to the datasets dictionary
            datasets[file_name] = dataset
            print(f"Loaded {file_name} with {len(dataset)} samples")

        except Exception as e:
            print(f"Error loading {file_name}: {e}")

    if not datasets:
        raise ValueError(f"No valid datasets found in {dataset_dir}")

    return datasets


def load_raw_data(file_path: str) -> pd.DataFrame:
    """
    Đọc dữ liệu thô từ file CSV.

    Args:
        file_path (str): Đường dẫn đến file CSV chứa dữ liệu build thực tế.

    Returns:
        pd.DataFrame: DataFrame chứa dữ liệu build.
    """
    try:
        df = pd.read_csv(file_path)
        return df
    except Exception as e:
        raise ValueError(f"Error loading data from {file_path}: {e}")


def preprocess_single_build(build_data: pd.DataFrame, feature_cols: List[str],
                            scaler: StandardScaler = None) -> pd.DataFrame:
    """
    Tiền xử lý dữ liệu build thực tế để khớp với định dạng huấn luyện.

    Args:
        build_data (pd.DataFrame): DataFrame chứa dữ liệu build.
        feature_cols (List[str]): Danh sách các cột đặc trưng mà mô hình sử dụng.
        scaler (StandardScaler, optional): Scaler đã được fit trên dữ liệu huấn luyện. Nếu None, sẽ tạo mới.

    Returns:
        pd.DataFrame: DataFrame đã được tiền xử lý.
    """
    # Kiểm tra các cột cần thiết
    missing_cols = [col for col in feature_cols if col not in build_data.columns]
    if missing_cols:
        raise ValueError(f"Missing required features: {missing_cols}")

    # Chọn các cột đặc trưng
    processed_data = build_data[feature_cols].copy()

    # Xử lý giá trị thiếu
    processed_data = processed_data.fillna(0)

    # Chuẩn hóa dữ liệu (nếu có scaler)
    if scaler is not None:
        processed_data[feature_cols] = scaler.transform(processed_data[feature_cols])
    else:
        # Nếu không có scaler, tạo mới và fit (chỉ dùng cho test, không khuyến khích)
        scaler = StandardScaler()
        processed_data[feature_cols] = scaler.fit_transform(processed_data[feature_cols])

    return processed_data


def prepare_sequence(build_data: pd.DataFrame, time_step: int, feature_cols: List[str]) -> Tuple[
    pd.DataFrame, List[str]]:
    """
    Chuẩn bị chuỗi dữ liệu cho dự đoán với mô hình LSTM.

    Args:
        build_data (pd.DataFrame): DataFrame chứa dữ liệu build thực tế.
        time_step (int): Số bước thời gian (time steps) mà mô hình yêu cầu.
        feature_cols (List[str]): Danh sách các cột đặc trưng.

    Returns:
        Tuple[pd.DataFrame, List[str]]: DataFrame đã được chuẩn bị và thông tin padding (nếu có).
    """
    # Sắp xếp dữ liệu theo thời gian (giả sử có cột gh_build_started_at)
    if 'gh_build_started_at' in build_data.columns:
        build_data = build_data.sort_values(by='gh_build_started_at').reset_index(drop=True)

    num_builds = len(build_data)
    padding_info = None

    # Padding nếu không đủ time_step
    if num_builds < time_step:
        padding_info = f"Input has {num_builds} builds, but {time_step} are required. Padded with {time_step - num_builds} zero-filled builds."
        padding_df = pd.DataFrame(
            [[0] * len(feature_cols)] * (time_step - num_builds),
            columns=feature_cols
        )
        build_data = pd.concat([padding_df, build_data], ignore_index=True)

    # Lấy time_step build cuối cùng nếu dữ liệu dài hơn time_step
    if len(build_data) > time_step:
        build_data = build_data.tail(time_step)

    return build_data, padding_info


def preprocess_for_prediction(file_path: str, time_step: int, feature_cols: List[str], scaler: StandardScaler = None) -> \
Tuple[pd.DataFrame, List[str]]:
    """
    Tiền xử lý dữ liệu thực tế từ file CSV để chuẩn bị cho dự đoán.

    Args:
        file_path (str): Đường dẫn đến file CSV chứa dữ liệu build.
        time_step (int): Số bước thời gian (time steps) mà mô hình yêu cầu.
        feature_cols (List[str]): Danh sách các cột đặc trưng.
        scaler (StandardScaler, optional): Scaler đã được fit trên dữ liệu huấn luyện.

    Returns:
        Tuple[pd.DataFrame, List[str]]: DataFrame đã được tiền xử lý và thông tin padding (nếu có).
    """
    # Đọc dữ liệu thô
    raw_data = load_raw_data(file_path)

    # Tiền xử lý dữ liệu
    processed_data = preprocess_single_build(raw_data, feature_cols, scaler)

    # Chuẩn bị chuỗi dữ liệu
    sequence_data, padding_info = prepare_sequence(processed_data, time_step, feature_cols)

    return sequence_data, padding_info