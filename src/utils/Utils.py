from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler

# Hyperparameter tuning settings
NBR_REP = 6
NBR_GEN = 2
NBR_SOL = 10
MAX_EVAL = NBR_GEN * NBR_SOL

# Configuration for SMOTE and hybrid approach
WITH_SMOTE = False
HYBRID_OPTION = False
if HYBRID_OPTION:
    WITH_SMOTE = True

def get_dataset(file_name):
    """
    Load and preprocess dataset.

    Args:
        file_name (str): Name of the dataset file.

    Returns:
        pd.DataFrame: Preprocessed dataset sorted by 'gh_build_started_at'.
    """
    dataset = pd.read_csv(f"dataset/{file_name}", parse_dates=['gh_build_started_at'], index_col="gh_build_started_at")
    dataset.sort_values(by=['gh_build_started_at'], inplace=True)
    return dataset

def to_labels(pos_probs, threshold):
    """
    Convert probability predictions to binary labels based on a threshold.

    Args:
        pos_probs (np.ndarray): Array of probability predictions.
        threshold (float): Classification threshold.

    Returns:
        np.ndarray: Binary classification labels.
    """
    return (pos_probs >= threshold).astype(int)

def get_best_threshold(probs, y_train):
    """
    Determine the optimal threshold for classification based on AUC score.

    Args:
        probs (np.ndarray): Model predicted probabilities.
        y_train (np.ndarray): Ground truth labels.

    Returns:
        float: Best threshold value.
    """
    thresholds = np.arange(0, 1, 0.001)
    scores = [roc_auc_score(y_train, to_labels(probs, t)) for t in thresholds]
    return thresholds[np.argmax(scores)]  # Select threshold with max AUC

def failure_info(dataset):
    """
    Compute failure rate and dataset size.

    Args:
        dataset (pd.DataFrame): Input dataset.

    Returns:
        tuple: (failure_rate, dataset_size)
    """
    failures = dataset['build_Failed'] > 0
    return failures.mean(), len(dataset)

def get_entry(y_true, y_pred):
    """
    Compute evaluation metrics (AUC, accuracy, F1-score).

    Args:
        y_true (np.ndarray): True labels.
        y_pred (np.ndarray): Predicted labels.

    Returns:
        dict: Dictionary containing evaluation metrics.
    """
    return {
        "AUC": roc_auc_score(y_true, y_pred),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "F1": f1_score(y_true, y_pred)
    }

def predict_lstm(model, X, y_true):
    """
    Make predictions using LSTM model and adjust threshold dynamically.

    Args:
        model (keras.Model): Trained LSTM model.
        X (np.ndarray): Input features.
        y_true (np.ndarray): True labels.

    Returns:
        dict: Evaluation metrics of the prediction.
    """
    y_pred_probs = model.predict(X)

    # Use dynamic threshold selection if hybrid_option is enabled
    threshold = 0.5 if (WITH_SMOTE and not HYBRID_OPTION) else get_best_threshold(y_pred_probs, y_true)
    y_pred = (y_pred_probs >= threshold).astype(int)

    return get_entry(y_true, y_pred)

def is_int(n):
    """
    Check if a value can be converted to an integer.

    Args:
        n: Input value.

    Returns:
        bool: True if value is an integer, else False.
    """
    try:
        int(n)
        return True
    except ValueError:
        return False

def online_validation_folds(dataset, start_fold=6, end_fold=11, fold_ratio=0.1):
    """
    Split dataset into train/test folds for online validation.

    Args:
        dataset (pd.DataFrame): Input dataset.
        start_fold (int, optional): Starting fold index. Defaults to 6.
        end_fold (int, optional): Ending fold index. Defaults to 11.
        fold_ratio (float, optional): Percentage of data per fold. Defaults to 0.1.

    Returns:
        tuple: (train_sets, test_sets)
    """
    fold_size = int(len(dataset) * fold_ratio)
    train_sets = []
    test_sets = []
    
    for i in range(start_fold, end_fold):
        train_sets.append(dataset.iloc[:fold_size * (i - 1)])
        test_sets.append(dataset.iloc[fold_size * (i - 1): fold_size * i])

    return train_sets, test_sets

def frange(start, stop=None, step=None):
    """
    Generate a range of floating-point numbers.

    Args:
        start (float): Start value.
        stop (float, optional): Stop value.
        step (float, optional): Step size.

    Yields:
        float: Next value in range.
    """
    if stop is None:
        stop = start
        start = 0.0

    if step is None:
        step = 1.0

    while (step > 0 and start < stop) or (step < 0 and start > stop):
        yield round(start, 10)  # Avoid floating-point precision issues
        start += step

def frange_int(start, stop=None, step=None):
    """
    Generate a range of integers.

    Args:
        start (int): Start value.
        stop (int, optional): Stop value.
        step (int, optional): Step size.

    Yields:
        int: Next value in range.
    """
    if stop is None:
        stop = start
        start = 0

    if step is None:
        step = 1

    while (step > 0 and start < stop) or (step < 0 and start > stop):
        yield start
        start += step
