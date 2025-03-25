# src/data/data_splitter.py
import logging
from src.utils.Utils import Utils

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def split_train_test(dataset, method="online_validation"):
    """
    Chia dữ liệu thành tập huấn luyện và kiểm tra.

    Args:
        dataset (pd.DataFrame): Dataset đầu vào.
        method (str): Phương pháp chia dữ liệu (mặc định: online_validation).

    Returns:
        tuple: (train_sets, test_sets)
    """
    logger.info("\nClass Distribution in Dataset:")
    logger.info(dataset['build_failed'].value_counts(normalize=True))

    if method == "online_validation":
        train_sets, test_sets = Utils.online_validation_folds(dataset)
        logger.info(f"Split into {len(train_sets)} train sets and {len(test_sets)} test sets using online validation")
        return train_sets, test_sets
    else:
        raise ValueError(f"Unsupported split method: {method}")