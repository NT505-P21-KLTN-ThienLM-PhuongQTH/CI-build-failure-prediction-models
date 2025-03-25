# src/data/dataset_loader.py
import os
import pandas as pd
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def get_dataset(filename, data_dir="data/processed/by_project"):
    """
    Tải dữ liệu từ một tệp CSV trong thư mục được chỉ định.

    Args:
        filename (str): Tên tệp CSV (ví dụ: "celluloid_celluloid.csv").
        data_dir (str): Thư mục chứa tệp dữ liệu (mặc định: "data/processed/by_project").

    Returns:
        pd.DataFrame: Dataset được tải dưới dạng DataFrame.

    Raises:
        FileNotFoundError: Nếu tệp không tồn tại.
        ValueError: Nếu dữ liệu rỗng.
    """
    # Xây dựng đường dẫn đầy đủ đến tệp
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    file_path = os.path.join(project_root, data_dir, filename)

    # Kiểm tra tệp có tồn tại không
    if not os.path.exists(file_path):
        logger.error(f"Dataset file not found: {file_path}")
        raise FileNotFoundError(f"Dataset file not found: {file_path}")

    # Tải dữ liệu
    logger.info(f"Loading dataset from {file_path}")
    dataset = pd.read_csv(file_path)

    # Kiểm tra dữ liệu rỗng
    if dataset.empty:
        logger.error(f"Dataset is empty: {file_path}")
        raise ValueError(f"Dataset is empty: {file_path}")

    # Chuyển đổi cột thời gian (nếu có) thành định dạng datetime
    if 'gh_build_started_at' in dataset.columns:
        dataset['gh_build_started_at'] = pd.to_datetime(dataset['gh_build_started_at'], errors='coerce')

    logger.info(f"Loaded dataset with shape: {dataset.shape}")
    return dataset


def list_available_datasets(data_dir="data/processed/by_project"):
    """
    Liệt kê tất cả các tệp dataset có sẵn trong thư mục.

    Args:
        data_dir (str): Thư mục chứa các tệp dữ liệu.

    Returns:
        list: Danh sách tên các tệp dataset.
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    data_path = os.path.join(project_root, data_dir)

    if not os.path.exists(data_path):
        logger.warning(f"Data directory not found: {data_path}")
        return []

    datasets = [f for f in os.listdir(data_path) if f.endswith('.csv')]
    logger.info(f"Found {len(datasets)} datasets in {data_path}: {datasets}")
    return datasets