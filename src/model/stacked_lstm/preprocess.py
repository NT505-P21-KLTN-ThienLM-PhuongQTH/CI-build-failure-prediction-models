import numpy as np
from imblearn.over_sampling import SMOTE

from src.data.feature_analysis import prepare_features, scale_features
from src.helpers import Utils


def apply_smote(training_set, y):
    """
    Apply SMOTE to balance the dataset if configured to do so.

    Args:
        training_set (np.ndarray): Feature matrix for training.
        y (np.ndarray): Target labels.

    Returns:
        tuple: (balanced training_set, balanced y)
    """
    if Utils.CONFIG['WITH_SMOTE'] and len(np.unique(y)) > 1:
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
    else:
        return training_set, y

def train_preprocess(dataset_train, time_step):
    X, y = prepare_features(dataset_train, target_column='build_failed')
    # X_scaled, _ = scale_features(X)

    training_set = X.values

    # Limit time_step to the length of the training set
    if len(training_set) < time_step:
        print(f"Adjusting time_step from {time_step} to {len(training_set) - 1}")
        time_step = max(1, len(training_set) - 1)

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


def test_preprocess(dataset_train, dataset_test, time_step):
    X_train, _ = prepare_features(dataset_train)
    X_test, y_test = prepare_features(dataset_test)

    # X_train_scaled, scaler = scale_features(X_train)
    # X_test_scaled, _ = scale_features(X_test, scaler=scaler)

    dataset_total = np.vstack((X_train.values, X_test.values))

    if len(dataset_total) < time_step + len(dataset_test):
        raise ValueError("Not enough data for test sequences")

    inputs = dataset_total[-len(dataset_test) - time_step:]
    X_test = np.lib.stride_tricks.sliding_window_view(inputs, (time_step, inputs.shape[1]))[:-1]
    X_test = np.squeeze(X_test, axis=1)
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return X_test, y_test