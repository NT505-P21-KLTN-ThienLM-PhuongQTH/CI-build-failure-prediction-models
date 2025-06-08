# src/model/common/preprocess.py
import numpy as np
from imblearn.over_sampling import SMOTE
from src.helpers import Utils

feature_groups = {
    'project': ['gh_sloc', 'gh_test_cases_per_kloc', 'proj_fail_rate_history', 'proj_fail_rate_recent'],
    'code_change': ['git_diff_src_churn', 'git_diff_test_churn', 'gh_diff_files_added', 'gh_diff_files_deleted',
                    'gh_diff_tests_added', 'gh_diff_src_files', 'gh_diff_doc_files', 'num_files_edited'],
    'team': ['gh_team_size', 'gh_num_issue_comments', 'gh_num_pr_comments', 'gh_num_commit_comments',
             'same_committer', 'num_distinct_authors', 'comm_avg_experience', 'comm_fail_rate_history',
             'comm_fail_rate_recent'],
    'time': ['year_of_start', 'month_of_start', 'day_of_start', 'day_week', 'tr_duration', 'prev_build_result'],
    'config_pr': ['gh_is_pr', 'no_config_edited']
}

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

def train_preprocess(dataset_train, time_step):
    X, y = prepare_features(dataset_train, target_column='build_failed')
    training_set = X.values

    # Padding nếu dữ liệu không đủ cho time_step
    if len(training_set) < time_step:
        print(f"Padding dữ liệu huấn luyện từ {len(training_set)} lên {time_step} bằng chuỗi 0...")
        num_padding = time_step - len(training_set)
        padding = np.zeros((num_padding, training_set.shape[1]))
        training_set = np.vstack((padding, training_set))
        y = np.pad(y, (num_padding, 0), mode='constant', constant_values=0)

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

    print("X_train shape:", X_train.shape)
    print("X_train columns:", list(X_train.columns))
    print("\nX_test shape:", X_test.shape)
    print("X_test columns:", list(X_test.columns))

    dataset_total = np.vstack((X_train.values, X_test.values))

    if len(dataset_total) < time_step + len(dataset_test):
        print(f"Padding dữ liệu test từ {len(dataset_total)} lên {time_step + len(dataset_test)} bằng chuỗi 0...")
        num_padding = time_step + len(dataset_test) - len(dataset_total)
        padding = np.zeros((num_padding, dataset_total.shape[1]))
        dataset_total = np.vstack((padding, dataset_total))

    inputs = dataset_total[-len(dataset_test) - time_step:]
    X_test = np.lib.stride_tricks.sliding_window_view(inputs, (time_step, inputs.shape[1]))[:-1]
    X_test = np.squeeze(X_test, axis=1)
    y_test = y_test[-len(X_test):]

    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return X_test, y_test

def convlstm_train_preprocess(dataset_train, time_step):
    """
    Preprocess training data for ConvLSTM, grouping related features into a 5D format.
    """
    X, y = prepare_features(dataset_train, target_column='build_failed')
    feature_cols = X.columns.tolist()
    training_set = X.values

    if len(training_set) < time_step:
        print(f"Padding training data from {len(training_set)} to {time_step} with zeros...")
        num_padding = time_step - len(training_set)
        padding = np.zeros((num_padding, training_set.shape[1]))
        training_set = np.vstack((padding, training_set))
        y = np.pad(y, (num_padding, 0), mode='constant', constant_values=0)

    training_set, y_smote = apply_smote(training_set, y)

    try:
        X_train = np.lib.stride_tricks.sliding_window_view(
            training_set, (time_step, training_set.shape[1])
        )[:-1]
        X_train = np.squeeze(X_train, axis=1)
        group_indices = []
        for group, feats in feature_groups.items():
            indices = [feature_cols.index(f) for f in feats if f in feature_cols]
            group_indices.append(indices)
        max_cols = max(len(indices) for indices in group_indices)
        num_groups = len(group_indices)
        X_train_grouped = np.zeros((X_train.shape[0], time_step, num_groups, max_cols, 1))
        for i, indices in enumerate(group_indices):
            for j, idx in enumerate(indices):
                X_train_grouped[:, :, i, j, 0] = X_train[:, :, idx]
        y_train = y_smote[time_step:]
    except Exception as e:
        raise RuntimeError(f"Error during sliding window creation for ConvLSTM: {e}")
    print(f"X_train shape (ConvLSTM): {X_train_grouped.shape}, y_train shape: {y_train.shape}")
    return X_train_grouped, y_train

def convlstm_test_preprocess(dataset_train, dataset_test, time_step):
    """
    Preprocess test data for ConvLSTM, grouping related features into a 5D format.
    """
    X_train, _ = prepare_features(dataset_train)
    X_test, y_test = prepare_features(dataset_test)
    feature_cols = X_train.columns.tolist()

    print("X_train shape:", X_train.shape)
    print("X_train columns:", feature_cols)
    print("\nX_test shape:", X_test.shape)
    print("X_test columns:", list(X_test.columns))

    dataset_total = np.vstack((X_train.values, X_test.values))

    if len(dataset_total) < time_step + len(dataset_test):
        print(f"Padding test data from {len(dataset_total)} to {time_step + len(dataset_test)} with zeros...")
        num_padding = time_step + len(dataset_test) - len(dataset_total)
        padding = np.zeros((num_padding, dataset_total.shape[1]))
        dataset_total = np.vstack((padding, dataset_total))

    inputs = dataset_total[-len(dataset_test) - time_step:]
    X_test = np.lib.stride_tricks.sliding_window_view(inputs, (time_step, inputs.shape[1]))[:-1]
    X_test = np.squeeze(X_test, axis=1)

    group_indices = []
    for group, feats in feature_groups.items():
        indices = [feature_cols.index(f) for f in feats if f in feature_cols]
        group_indices.append(indices)
    max_cols = max(len(indices) for indices in group_indices)
    num_groups = len(group_indices)
    X_test_grouped = np.zeros((X_test.shape[0], time_step, num_groups, max_cols, 1))
    for i, indices in enumerate(group_indices):
        for j, idx in enumerate(indices):
            X_test_grouped[:, :, i, j, 0] = X_test[:, :, idx]
    y_test = y_test[-len(X_test):]

    print(f"X_test shape (ConvLSTM): {X_test_grouped.shape}, y_test shape: {y_test.shape}")
    return X_test_grouped, y_test