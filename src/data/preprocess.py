import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
import logging


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)

def split_data_by_project(df, output_path, max_projects=None):
    """
    Divide processed data into folders by project name.

    :param df: Processed DataFrame.
    :param output_path: Directory to save split data.
    :param max_projects: Maximum number of projects to process (None for all).
    """
    logging.info("Starting split_data_by_project function")
    
    os.makedirs(output_path, exist_ok=True)
    logging.info(f"Output directory created or exists: {output_path}")
    
    projects = df['gh_project_name'].unique()
    logging.info(f"Found {len(projects)} unique projects")

    if max_projects:
        projects = projects[:max_projects]
        logging.info(f"Limited to {max_projects} projects")

    for i, project in enumerate(projects, 1):
        logging.info(f"Processing project {i}/{len(projects)}: {project}")
        
        file_name = str(project).replace("/", "_").replace(":", "_").replace(" ", "_") + ".csv"
        file_path = os.path.join(output_path, file_name)
        
        logging.info(f"Filtering data for project: {project}")
        project_data = df[df['gh_project_name'] == project]
        logging.info(f"Row count for {project}: {len(project_data)}")
        
        if not project_data.empty:
            logging.info(f"Saving {project} to {file_path}")
            project_data.to_csv(file_path, index=False)
            logging.info(f"Saved {project} with {len(project_data)} rows to {file_path}")
        else:
            logging.warning(f"No data to save for {project} (empty)")
    
    logging.info("Completed split_data_by_project function")

def preprocess_data(df, output_file=None, split_by_project=False, output_path=None, max_projects=None):
    """
    Preprocess raw data: handle nulls, encode categoricals, normalize numericals, and save.

    :param df: Input raw DataFrame.
    :param output_file: Path to save the fully processed DataFrame (e.g., 'processed.csv').
    :param split_by_project: If True, split data by project instead of saving one file.
    :param output_path: Directory to save split data if split_by_project=True.
    :param max_projects: Max projects to process if splitting.
    :return: Processed DataFrame, scaler, label_encoders
    """
    logging.info("Starting preprocess_data function")
    logging.info("Copying input DataFrame")
    df = df.copy()
    logging.info(f"Initial DataFrame shape: {df.shape}")
    
    # Rename and convert target column
    logging.info("Renaming 'tr_status' to 'build_failed' and converting values")
    if 'tr_status' in df.columns:
        df = df.rename(columns={'tr_status': 'build_failed'})
        df['build_failed'] = df['build_failed'].map({'passed': 0, 'failed': 1})
        if df['build_failed'].isnull().any():
            logging.warning("Some 'build_failed' values could not be mapped (not 'passed' or 'failed')")
    else:
        logging.error("'tr_status' column not found in DataFrame")
        raise ValueError("'tr_status' column not found in DataFrame")
    logging.info(f"Unique values in 'build_failed' after mapping: {df['build_failed'].unique()}")
    
    # Sort by time
    logging.info("Converting 'gh_build_started_at' to datetime and sorting")
    df['gh_build_started_at'] = pd.to_datetime(df['gh_build_started_at'])
    df = df.sort_values('gh_build_started_at').reset_index(drop=True)
    
    # Handle missing values
    logging.info("Handling missing values with dropna")
    initial_rows = len(df)
    df = df.dropna()
    logging.info(f"Rows before dropna: {initial_rows}, after dropna: {len(df)}")
    
    # Remove duplicates
    logging.info("Removing duplicate rows")
    initial_rows = len(df)
    df = df.drop_duplicates()
    logging.info(f"Rows before drop_duplicates: {initial_rows}, after drop_duplicates: {len(df)}")
    
    # Encode categorical columns
    categorical_columns = ['gh_is_pr', 'gh_by_core_team_member']
    label_encoders = {}
    logging.info("Encoding categorical columns")
    for col in categorical_columns:
        logging.info(f"Encoding column: {col}")
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col])
        label_encoders[col] = le

    # Normalize numerical columns
    numerical_columns = [
        "git_num_all_built_commits", "gh_num_commits_on_files_touched",
        "git_diff_src_churn", "gh_diff_files_added", "gh_diff_files_deleted",
        "gh_diff_files_modified", "git_diff_test_churn", "gh_diff_tests_added",
        "gh_diff_tests_deleted", "gh_diff_src_files", "gh_diff_doc_files",
        "gh_diff_other_files", "gh_num_commit_comments", "gh_team_size",
        "gh_sloc", "gh_test_lines_per_kloc", "gh_test_cases_per_kloc",
        "gh_asserts_cases_per_kloc", "gh_repo_age", "gh_repo_num_commits"
    ]
    logging.info("Normalizing numerical columns with MinMaxScaler")
    scaler = MinMaxScaler()
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    logging.info("Normalization completed")
    
    # Save processed data
    if output_file and not split_by_project:
        logging.info(f"Saving processed data to {output_file}")
        df.to_csv(output_file, index=False)
        logging.info(f"Saved processed data with shape {df.shape} to {output_file}")
    
    if split_by_project and output_path:
        logging.info("Splitting processed data by project")
        split_data_by_project(df, output_path, max_projects)
    
    logging.info("Completed preprocess_data function")
    return df, scaler, label_encoders

if __name__ == "__main__":
    data_path = "data/raw/travistorrent.csv"
    output_file = "data/processed/processed_all.csv"
    output_path = "data/processed/by_project/"

    columns_to_keep = [
        "tr_status", "gh_project_name", "git_num_all_built_commits",
        "gh_num_commits_on_files_touched", "git_diff_src_churn",
        "gh_diff_files_added", "gh_diff_files_deleted", "gh_diff_files_modified",
        "git_diff_test_churn", "gh_diff_tests_added", "gh_diff_tests_deleted",
        "gh_diff_src_files", "gh_diff_doc_files", "gh_diff_other_files",
        "gh_num_commit_comments", "gh_by_core_team_member", "gh_team_size",
        "gh_is_pr", "gh_sloc", "gh_test_lines_per_kloc", "gh_test_cases_per_kloc",
        "gh_asserts_cases_per_kloc", "gh_build_started_at", "gh_repo_age",
        "gh_repo_num_commits"
    ]

    dtype_spec = {
        "git_diff_src_churn": "float32",
        "gh_diff_files_modified": "float32",
        "gh_test_lines_per_kloc": "float32"
    }
    
    # Load raw data
    logging.info(f"Loading raw data from {data_path}")
    df = pd.read_csv(data_path, dtype=dtype_spec, low_memory=False)
    filtered_df = df[columns_to_keep].copy()
    
    # Preprocess and save data
    processed_df, scaler, label_encoders = preprocess_data(
        filtered_df,
        output_file=output_file,
        split_by_project=True, # Enable if you want to split by project or set to False to save one file
        output_path=output_path,
        max_projects=None
    )
    
    print(f"Processed DataFrame shape: {processed_df.shape}")