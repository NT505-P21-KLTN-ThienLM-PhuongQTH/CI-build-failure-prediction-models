from sklearn.metrics import f1_score, roc_auc_score, accuracy_score, precision_score, recall_score, roc_curve
import numpy as np
import pandas as pd
import warnings
import os

# Configuration
CONFIG = {
    'NBR_REP': 6,
    'NBR_GEN': 5,
    'NBR_SOL': 10,
    'MAX_EVAL': 8,
    'WITH_SMOTE': True,
    'HYBRID_OPTION': True
}
if CONFIG['HYBRID_OPTION']:
    CONFIG['WITH_SMOTE'] = True

def get_dataset(file_name):
    file_path = f"data/processed/by_project/{file_name}"
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File {file_path} not found")
    dataset = pd.read_csv(file_path, parse_dates=['gh_build_started_at'], index_col="gh_build_started_at")
    dataset.sort_values(by=['gh_build_started_at'], inplace=True)
    return dataset

def to_labels(pos_probs, threshold):
    if not isinstance(pos_probs, np.ndarray):
        pos_probs = np.array(pos_probs)
    return (pos_probs >= threshold).astype(int)

def get_best_threshold(probs, y_train):
    fpr, tpr, thresholds = roc_curve(y_train, probs)
    j_scores = tpr - fpr
    return thresholds[np.argmax(j_scores)]

def failure_info(dataset):
    if 'build_failed' not in dataset.columns:
        raise KeyError("'build_failed' column not found in dataset")
    failures = dataset['build_failed'] > 0
    return failures.mean(), len(dataset)

def get_entry(y_true, y_pred):
    metrics = {}
    metrics["AUC"] = roc_auc_score(y_true, y_pred)
    metrics["accuracy"] = accuracy_score(y_true, y_pred)
    try:
        metrics["precision"] = precision_score(y_true, y_pred)
        metrics["recall"] = recall_score(y_true, y_pred)
        metrics["F1"] = f1_score(y_true, y_pred)
    except ValueError as e:
        warnings.warn(f"Error computing precision/recall/F1: {e}")
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["F1"] = 0.0
    return metrics

def predict_lstm(model, X, y_true, threshold=0.5):
    y_pred_probs = model.predict(X)
    y_pred = (y_pred_probs >= threshold).astype(int)
    return get_entry(y_true, y_pred)

def is_int(n):
    return isinstance(n, int)

def online_validation_folds(dataset, start_fold=6, end_fold=11, fold_ratio=0.1):
    fold_size = int(len(dataset) * fold_ratio)
    train_sets, test_sets = [], []
    
    for i in range(start_fold, end_fold):
        train_end = fold_size * (i - 1)
        test_end = fold_size * i
        if test_end > len(dataset):
            test_end = len(dataset)
        train_sets.append(dataset.iloc[:train_end])
        test_sets.append(dataset.iloc[train_end:test_end])
    return train_sets, test_sets

def frange(start, stop=None, step=None):
    if stop is None:
        stop = start
        start = 0.0
    if step is None:
        step = 1.0
    return np.arange(start, stop, step)

def frange_int(start, stop=None, step=None):
    if stop is None:
        stop = start
        start = 0
    if step is None:
        step = 1
    return range(start, stop, step)