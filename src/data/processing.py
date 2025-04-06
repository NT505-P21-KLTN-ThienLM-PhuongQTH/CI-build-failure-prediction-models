import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import os
import glob
from collections import defaultdict

# Định nghĩa dtype cho các cột đặc biệt
DTYPE_SPEC = {
    "git_diff_src_churn": "float32",
    "gh_diff_files_modified": "float32",
    "gh_test_lines_per_kloc": "float32"
}

# Danh sách 25 cột mục tiêu
TARGET_COLUMNS = [
    "tr_status", "gh_project_name", "gh_is_pr", "gh_team_size", "git_num_commits",
    "gh_num_issue_comments", "gh_num_commit_comments", "gh_num_pr_comments",
    "git_diff_src_churn", "git_diff_test_churn", "gh_diff_files_added",
    "gh_diff_files_deleted", "gh_diff_files_modified", "gh_diff_tests_added",
    "gh_diff_tests_deleted", "gh_diff_src_files", "gh_diff_doc_files",
    "gh_diff_other_files", "gh_num_commits_on_files_touched", "gh_sloc",
    "gh_test_lines_per_kloc", "gh_test_cases_per_kloc", "gh_asserts_cases_per_kloc",
    "gh_by_core_team_member", "gh_build_started_at", "tr_log_num_tests_failed",
    "tr_duration", "tr_build_id", "tr_job_id"
]

# Ánh xạ tên cột cho từng nhóm
MAPPINGS = {
    "Group1": {
        "tr_status": "tr_status", "gh_project_name": "gh_project_name", "gh_is_pr": "gh_is_pr",
        "gh_team_size": "gh_team_size", "git_num_commits": "gh_num_commits_in_push",
        "gh_num_issue_comments": "gh_num_issue_comments", "gh_num_commit_comments": "gh_num_commit_comments",
        "gh_num_pr_comments": "gh_num_pr_comments", "git_diff_src_churn": "git_diff_src_churn",
        "git_diff_test_churn": "git_diff_test_churn", "gh_diff_files_added": "gh_diff_files_added",
        "gh_diff_files_deleted": "gh_diff_files_deleted", "gh_diff_files_modified": "gh_diff_files_modified",
        "gh_diff_tests_added": "gh_diff_tests_added", "gh_diff_tests_deleted": "gh_diff_tests_deleted",
        "gh_diff_src_files": "gh_diff_src_files", "gh_diff_doc_files": "gh_diff_doc_files",
        "gh_diff_other_files": "gh_diff_other_files", "gh_num_commits_on_files_touched": "gh_num_commits_on_files_touched",
        "gh_sloc": "gh_sloc", "gh_test_lines_per_kloc": "gh_test_lines_per_kloc",
        "gh_test_cases_per_kloc": "gh_test_cases_per_kloc", "gh_asserts_cases_per_kloc": "gh_asserts_cases_per_kloc",
        "gh_by_core_team_member": "gh_by_core_team_member", "gh_build_started_at": "gh_build_started_at",
        "tr_log_num_tests_failed": "tr_log_num_tests_failed", "tr_duration": "tr_duration",
        "tr_build_id": "tr_build_id", "tr_job_id": "tr_job_id",
    },
    "Group2": {
        "tr_status": "status", "gh_project_name": "project_name", "gh_is_pr": "is_pr",
        "gh_team_size": "team_size", "git_num_commits": "num_commits",
        "gh_num_issue_comments": "num_issue_comments", "gh_num_commit_comments": "num_commit_comments",
        "gh_num_pr_comments": "num_pr_comments", "git_diff_src_churn": "src_churn",
        "git_diff_test_churn": "test_churn", "gh_diff_files_added": "files_added",
        "gh_diff_files_deleted": "files_deleted", "gh_diff_files_modified": "files_modified",
        "gh_diff_tests_added": "tests_added", "gh_diff_tests_deleted": "tests_deleted",
        "gh_diff_src_files": "src_files", "gh_diff_doc_files": "doc_files",
        "gh_diff_other_files": "other_files", "gh_num_commits_on_files_touched": "commits_on_files_touched",
        "gh_sloc": "sloc", "gh_test_lines_per_kloc": "test_lines_per_kloc",
        "gh_test_cases_per_kloc": "test_cases_per_kloc", "gh_asserts_cases_per_kloc": "asserts_per_kloc",
        "gh_by_core_team_member": "main_team_member", "gh_build_started_at": "started_at",
        "tr_log_num_tests_failed": "failed", "tr_duration": "duration",
        "tr_build_id": "build_id", "tr_job_id": "job_id",
    },
    "Group3": {
        "tr_status": "tr_status", "gh_project_name": "gh_project_name", "gh_is_pr": "gh_is_pr",
        "gh_team_size": "gh_team_size", "git_num_commits": "git_num_commits",
        "gh_num_issue_comments": "gh_num_issue_comments", "gh_num_commit_comments": "gh_num_commit_comments",
        "gh_num_pr_comments": "gh_num_pr_comments", "git_diff_src_churn": "gh_src_churn",
        "git_diff_test_churn": "gh_test_churn", "gh_diff_files_added": "gh_files_added",
        "gh_diff_files_deleted": "gh_files_deleted", "gh_diff_files_modified": "gh_files_modified",
        "gh_diff_tests_added": "gh_tests_added", "gh_diff_tests_deleted": "gh_tests_deleted",
        "gh_diff_src_files": "gh_src_files", "gh_diff_doc_files": "gh_doc_files",
        "gh_diff_other_files": "gh_other_files", "gh_num_commits_on_files_touched": "gh_commits_on_files_touched",
        "gh_sloc": "gh_sloc", "gh_test_lines_per_kloc": "gh_test_lines_per_kloc",
        "gh_test_cases_per_kloc": "gh_test_cases_per_kloc", "gh_asserts_cases_per_kloc": "gh_asserts_cases_per_kloc",
        "gh_by_core_team_member": "gh_by_core_team_member", "gh_build_started_at": "tr_started_at",
        "tr_log_num_tests_failed": "tr_tests_failed", "tr_duration": "tr_duration",
        "tr_build_id": "tr_build_id", "tr_job_id": "tr_job_id",
    },
    "Group4": {
        "tr_status": "tr_status", "gh_project_name": "gh_project_name", "gh_is_pr": "gh_is_pr",
        "gh_team_size": "gh_team_size", "git_num_commits": "gh_num_commits_in_push",
        "gh_num_issue_comments": "gh_num_issue_comments", "gh_num_commit_comments": "gh_num_commit_comments",
        "gh_num_pr_comments": "gh_num_pr_comments", "git_diff_src_churn": "git_diff_src_churn",
        "git_diff_test_churn": "git_diff_test_churn", "gh_diff_files_added": "gh_diff_files_added",
        "gh_diff_files_deleted": "gh_diff_files_deleted", "gh_diff_files_modified": "gh_diff_files_modified",
        "gh_diff_tests_added": "gh_diff_tests_added", "gh_diff_tests_deleted": "gh_diff_tests_deleted",
        "gh_diff_src_files": "gh_diff_src_files", "gh_diff_doc_files": "gh_diff_doc_files",
        "gh_diff_other_files": "gh_diff_other_files", "gh_num_commits_on_files_touched": "gh_num_commits_on_files_touched",
        "gh_sloc": "gh_sloc", "gh_test_lines_per_kloc": "gh_test_lines_per_kloc",
        "gh_test_cases_per_kloc": "gh_test_cases_per_kloc", "gh_asserts_cases_per_kloc": "gh_asserts_cases_per_kloc",
        "gh_by_core_team_member": "gh_by_core_team_member", "gh_build_started_at": "gh_build_started_at",
        "tr_log_num_tests_failed": "tr_log_num_tests_failed", "tr_duration": "tr_duration",
        "tr_build_id": "tr_build_id", "tr_job_id": "tr_job_id",
    }
}

def group_files_by_columns(folder_path):
    """Phân loại file CSV theo cấu trúc cột."""
    csv_files = glob.glob(os.path.join(folder_path, "*.csv"))
    column_groups = defaultdict(list)
    for file_path in csv_files:
        try:
            df = pd.read_csv(file_path, nrows=0)
            columns = tuple(df.columns.tolist())
            column_groups[columns].append(file_path)
        except Exception as e:
            print(f"Lỗi đọc {file_path}: {e}")
    return list(column_groups.items())

def process_files(file_list, group_name, mapping):
    """Đọc và xử lý các file trong danh sách theo nhóm."""
    dfs = []
    for file in file_list:
        try:
            df = pd.read_csv(file, dtype=DTYPE_SPEC, low_memory=False)
            df = df.rename(columns={v: k for k, v in mapping[group_name].items()})
            available_columns = [col for col in TARGET_COLUMNS if col in df.columns]
            if available_columns:
                df = df[available_columns]
                dfs.append(df)
            else:
                print(f"Warning: No target columns found in {file}")
        except Exception as e:
            print(f"Error processing {file}: {e}")
    return dfs

def combine_datasets(folder_path, output_path="../data/combined/combined_travistorrent.csv"):
    """Kết hợp tất cả file CSV trong thư mục thành một file duy nhất."""
    groups_files = group_files_by_columns(folder_path)
    group_dfs = []
    for i, (columns, files) in enumerate(groups_files, start=1):
        group_name = f"Group{i}"
        dfs = process_files(files, group_name, MAPPINGS)
        group_dfs.append(dfs)
        print(f"Processed {len(dfs)} DataFrames for {group_name}")

    all_dfs = []
    for dfs in group_dfs:
        all_dfs.extend(dfs)

    all_dfs = [
        df.dropna(how='all', axis=1)
        for df in all_dfs
        if not df.empty and not df.isna().all().all()
    ]

    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        combined_df = combined_df[[col for col in TARGET_COLUMNS if col in combined_df.columns]]
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        combined_df.to_csv(output_path, index=False)
        print("Kích thước DataFrame cuối cùng:", combined_df.shape)
        print("Các cột trong DataFrame:", combined_df.columns.tolist())
        return combined_df
    else:
        print("Không có DataFrame nào được xử lý thành công.")
        return None

def load_data(file_path):
    """Tải dữ liệu từ CSV với dtype cụ thể."""
    df = pd.read_csv(file_path, dtype=DTYPE_SPEC, low_memory=False)
    df['tr_log_num_tests_failed'] = pd.to_numeric(df['tr_log_num_tests_failed'], errors='coerce').astype('float32')
    return df


def process_status(df: pd.DataFrame, status_column: str) -> pd.DataFrame:
    """
    Chuẩn hóa mọi cột trạng thái về build_failed (0/1) và loại bỏ giá trị lạ.

    - Mọi giá trị 0/1 (nếu đã numeric) hoặc 'passed'/'failed'/'errored'
      đều được map về 0/1.
    - Các giá trị khác thành NaN rồi bị drop.
    """
    df = df.copy()
    m = {0: 0, 1: 1, 'passed': 0, 'failed': 1, 'errored': 1}
    df['build_failed'] = df[status_column].map(m)
    df = df.dropna(subset=['build_failed'])
    df['build_failed'] = df['build_failed'].astype(int)
    return df

def summarize_projects(df: pd.DataFrame,
                       project_column: str = 'gh_project_name',
                       min_rows: int = 10000,
                       balance_threshold: float = 0.6) -> pd.DataFrame:
    """
    Giả sử df đã có cột 'build_failed' (0/1, không NaN).
    Tóm tắt per-project: số dòng, missing_ratio, passed/failed ratio.
    """
    rows = []
    for proj, g in df.groupby(project_column):
        n = len(g)
        if n < min_rows:
            continue
        miss = g.isna().mean().mean()
        vals = g['build_failed'].dropna().astype(int)
        if vals.empty:
            continue
        d = vals.value_counts(normalize=True)
        p, f = d.get(0, 0.0), d.get(1, 0.0)
        if max(p, f) > balance_threshold:
            continue
        rows.append((proj, n, round(miss, 4), round(p, 4), round(f, 4)))

    return (pd.DataFrame(rows,
             columns=['project', 'num_rows', 'missing_ratio', 'passed_ratio', 'failed_ratio'])
            .sort_values('num_rows', ascending=False))

def fill_nan_values(df):
    """Điền giá trị NaN theo logic đã định."""
    df = df.copy()
    groupby_project = df.groupby('gh_project_name')
    df['tr_log_num_tests_failed'] = groupby_project['tr_log_num_tests_failed'].transform(lambda x: x.fillna(0))
    df['tr_duration'] = groupby_project['tr_duration'].transform(
        lambda x: x.interpolate(method='linear', limit_direction='both').ffill().bfill().fillna(0))
    for col in ['git_num_commits', 'gh_num_issue_comments', 'gh_num_pr_comments', 'git_diff_src_churn', 
                'git_diff_test_churn', 'gh_diff_files_added', 'gh_diff_files_deleted', 'gh_diff_files_modified',
                'gh_diff_src_files', 'gh_diff_doc_files', 'gh_diff_other_files', 'gh_diff_tests_added', 
                'gh_diff_tests_deleted', 'gh_num_commits_on_files_touched', 'gh_test_lines_per_kloc', 
                'gh_test_cases_per_kloc', 'gh_asserts_cases_per_kloc']:
        df[col] = groupby_project[col].transform(lambda x: x.fillna(0))
    df['gh_by_core_team_member'] = groupby_project['gh_by_core_team_member'].transform(
        lambda x: x.fillna(x.mode()[0] if not x.mode().empty else 0))
    df['gh_team_size'] = groupby_project['gh_team_size'].transform(
        lambda x: x.fillna(x.mean() if not pd.isna(x.mean()) else 1))
    df['gh_sloc'] = groupby_project['gh_sloc'].transform(
        lambda x: x.fillna(x.mean() if not pd.isna(x.mean()) else 0))
    return df

def encode_categorical_columns(df, columns):
    """Mã hóa cột phân loại."""
    df_encoded = df.copy()
    encoders = {}
    for col in columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df_encoded[col].astype(str))
        encoders[col] = le
    return df_encoded, encoders

def normalize_numerical_columns(df, columns):
    """Chuẩn hóa cột số."""
    df_normalized = df.copy()
    scaler = MinMaxScaler()
    valid_columns = [col for col in columns if col in df.columns]
    df_normalized[valid_columns] = scaler.fit_transform(df_normalized[valid_columns])
    return df_normalized, scaler

def encode_cyclical_time_features(df, columns, periods):
    """Mã hóa các cột thời gian tuần hoàn."""
    df_encoded = df.copy()
    for col in columns:
        if col in df_encoded.columns and col in periods:
            period = periods[col]
            df_encoded[f'{col}_sin'] = np.sin(2 * np.pi * df_encoded[col] / period)
            df_encoded[f'{col}_cos'] = np.cos(2 * np.pi * df_encoded[col] / period)
    return df_encoded

def save_projects_to_files(df, output_dir="../data/processed", project_column='gh_project_name'):
    """Lưu từng dự án thành file CSV."""
    os.makedirs(output_dir, exist_ok=True)
    grouped = df.groupby(project_column)
    saved_files = {}
    for project, group_df in grouped:
        safe_project_name = project.replace('/', '_')
        file_path = os.path.join(output_dir, f"{safe_project_name}.csv")
        group_df.to_csv(file_path, index=False)
        saved_files[project] = file_path
    return saved_files