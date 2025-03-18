import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler, LabelEncoder

def split_data_by_project(df, output_path):
    """
    Divide data into folders by project name.

    :param df: DataFrame contains data that has been typed.
    :param output_path: Directory to save the split data.
    """
    projects = df['gh_project_name'].unique()

    for project in projects[:10]:
        file_name = str(project).replace("/", "_").replace(":", "_").replace(" ", "_") + ".csv"
        file_path = os.path.join(output_path, file_name)

        project_data = df[df['gh_project_name'] == project]
        # project_data = project_data.dropna().reset_index(drop=True)

        project_data.to_csv(file_path, index=False)


def preprocess_data(df):
    """
    Data preprocessing: remove null values, encode categorical variables, and normalize data.

    :param df: DataFrame needs preprocessing.
    :return: The DataFrame has been preprocessed.
    """
    df = df.copy()

    categorical_columns = ['gh_is_pr', 'gh_by_core_team_member']
    label_encoders = {}
    for col in categorical_columns:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    numerical_columns = [
        "git_num_all_built_commits",
        "gh_num_commits_on_files_touched",
        "git_diff_src_churn",
        "gh_diff_files_added",
        "gh_diff_files_deleted",
        "gh_diff_files_modified",
        "git_diff_test_churn",
        "gh_diff_tests_added",
        "gh_diff_tests_deleted",
        "gh_diff_src_files",
        "gh_diff_doc_files",
        "gh_diff_other_files",
        "gh_num_commit_comments",
        "gh_team_size",
        "gh_sloc",
        "gh_test_lines_per_kloc",
        "gh_test_cases_per_kloc",
        "gh_asserts_cases_per_kloc",
        "gh_repo_age",
        "gh_repo_num_commits"
    ]
    scaler = MinMaxScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    return df, label_encoders, scaler

if __name__ == "__main__":
    data_path = "data/raw/travistorrent.csv"
    output_path = "data/processed/"

    columns_to_keep = [
        "gh_project_name",
        "git_num_all_built_commits",
        "gh_num_commits_on_files_touched",
        "git_diff_src_churn",
        "gh_diff_files_added",
        "gh_diff_files_deleted",
        "gh_diff_files_modified",
        "git_diff_test_churn",
        "gh_diff_tests_added",
        "gh_diff_tests_deleted",
        "gh_diff_src_files",
        "gh_diff_doc_files",
        "gh_diff_other_files",
        "gh_num_commit_comments",
        "gh_by_core_team_member",
        "gh_team_size",
        "gh_is_pr",
        "gh_sloc",
        "gh_test_lines_per_kloc",
        "gh_test_cases_per_kloc",
        "gh_asserts_cases_per_kloc",
        "gh_first_commit_created_at",
        "gh_pushed_at",
        "gh_build_started_at",
        "gh_repo_age",
        "gh_repo_num_commits"
    ]

    dtype_spec = {
        "git_diff_src_churn": "float32",
        "gh_diff_files_modified": "float32",
        "gh_test_lines_per_kloc": "float32"
    }
    df = pd.read_csv(data_path, dtype=dtype_spec, low_memory=False)

    filtered_df = df[columns_to_keep].copy()

    preprocessed_df, label_encoders, scaler = preprocess_data(filtered_df)

    split_data_by_project(preprocessed_df, output_path)