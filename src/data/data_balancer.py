# src/data/data_balancer.py
import numpy as np
import logging
from imblearn.over_sampling import SMOTE
from sklearn.utils.class_weight import compute_class_weight

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def apply_smote(X, y, sampling_strategy=1.0, random_state=42):
    """
    Áp dụng SMOTE để cân bằng dữ liệu.

    Args:
        X (np.ndarray): Dữ liệu đầu vào.
        y (np.ndarray): Nhãn.
        sampling_strategy (float): Tỷ lệ cân bằng (mặc định 1.0 - cân bằng hoàn toàn).
        random_state (int): Seed cho tính ngẫu nhiên.

    Returns:
        tuple: (X_resampled, y_resampled)
    """
    logger.info("\nClass Distribution BEFORE SMOTE:")
    unique, counts = np.unique(y, return_counts=True)
    class_dist_before = dict(zip(unique, counts / len(y)))
    logger.info(class_dist_before)

    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_resampled, y_resampled = smote.fit_resample(X, y)

    logger.info("Class Distribution AFTER SMOTE:")
    unique, counts = np.unique(y_resampled, return_counts=True)
    class_dist_after = dict(zip(unique, counts / len(y_resampled)))
    logger.info(class_dist_after)

    return X_resampled, y_resampled

def compute_balanced_class_weights(y):
    """
    Tính class weights để xử lý dữ liệu không cân bằng.

    Args:
        y (np.ndarray): Nhãn.
        weight_adjustments (dict): Điều chỉnh trọng số cho từng lớp (mặc định: giảm lớp 0, tăng lớp 1).

    Returns:
        dict: Từ điển chứa trọng số cho từng lớp.
    """
    class_counts = np.bincount(y.astype(int))
    total_samples = len(y)
    n_classes = len(class_counts)
    class_weights = total_samples / (n_classes * class_counts)
    class_weight_dict = {i: class_weights[i] for i in range(n_classes)}
    logger.info(f"Computed class weights: {class_weight_dict}")
    return class_weight_dict