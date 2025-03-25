# import os
# import pandas as pd
# import numpy as np
# from sklearn.preprocessing import MinMaxScaler, LabelEncoder
# import logging

# # Set up logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler()]
# )

# def split_data_by_project(df, output_path, min_samples=5000):  # Giảm min_samples xuống 5000
#     logging.info("Starting split_data_by_project function")
    
#     os.makedirs(output_path, exist_ok=True)
#     logging.info(f"Output directory created or exists: {output_path}")
    
#     project_counts = df['gh_project_name'].value_counts()
#     projects = project_counts[project_counts >= min_samples].index
#     logging.info(f"Found {len(project_counts)} unique projects, {len(projects)} with >= {min_samples} samples")

#     for i, project in enumerate(projects, 1):
#         logging.info(f"Processing project {i}/{len(projects)}: {project}")
#         file_name = str(project).replace("/", "_").replace(":", "_").replace(" ", "_") + ".csv"
#         file_path = os.path.join(output_path, file_name)
        
#         project_data = df[df['gh_project_name'] == project]
#         logging.info(f"Row count for {project}: {len(project_data)}")
        
#         if not project_data.empty:
#             project_data.to_csv(file_path, index=False)
#             logging.info(f"Saved {project} with {len(project_data)} rows to {file_path}")
#         else:
#             logging.warning(f"No data to save for {project} (empty)")
    
#     logging.info("Completed split_data_by_project function")

# def preprocess_data(df, output_path=None, split_by_project=False, min_samples=5000):
#     logging.info("Starting preprocess_data function")
#     df = df.copy()
#     logging.info(f"Initial DataFrame shape: {df.shape}")
    
#     # Kiểm tra giá trị gốc của tr_status
#     logging.info("Checking unique values in 'tr_status' before mapping")
#     logging.info(f"Unique tr_status values: {df['tr_status'].unique()}")
    
#     # Rename and convert target column
#     logging.info("Renaming 'tr_status' to 'build_failed' and converting values")
#     df = df.rename(columns={'tr_status': 'build_failed'})
#     df['build_failed'] = df['build_failed'].map({'passed': 0, 'failed': 1, 'errored': 1, "canceled": 1})
#     logging.info(f"Unique values in 'build_failed': {df['build_failed'].unique()}")
    
#     # Sort by time
#     logging.info("Converting 'gh_build_started_at' to datetime and sorting")
#     df['gh_build_started_at'] = pd.to_datetime(df['gh_build_started_at'], errors='coerce')
#     df = df.sort_values('gh_build_started_at').reset_index(drop=True)
    
#     # Handle missing values chỉ ở cột quan trọng
#     logging.info("Handling missing values with dropna")
#     # key_columns = ['build_failed', 'gh_project_name', 'gh_build_started_at']
#     initial_rows = len(df)
#     df = df.dropna()
#     logging.info(f"Rows before dropna: {initial_rows}, after dropna: {len(df)}")
    
#     # Remove duplicates
#     logging.info("Removing duplicate rows")
#     initial_rows = len(df)
#     df = df.drop_duplicates()
#     logging.info(f"Rows before drop_duplicates: {initial_rows}, after drop_duplicates: {len(df)}")
    
#     # Encode categorical columns
#     categorical_columns = ['gh_is_pr', 'gh_by_core_team_member']
#     label_encoders = {}
#     logging.info("Encoding categorical columns")
#     for col in categorical_columns:
#         if col in df.columns:
#             logging.info(f"Encoding column: {col}")
#             le = LabelEncoder()
#             df[col] = le.fit_transform(df[col].astype(str))  # Xử lý cả NA
#             label_encoders[col] = le
#         else:
#             logging.warning(f"Column {col} not found, skipping")

#     # Normalize numerical columns
#     numerical_columns = [
#         "git_num_commits", "gh_num_commit_comments", "gh_src_churn",
#         "gh_test_churn", "gh_files_added", "gh_files_deleted", "gh_files_modified",
#         "gh_tests_added", "gh_tests_deleted", "gh_src_files", "gh_doc_files",
#         "gh_other_files", "gh_commits_on_files_touched", "gh_sloc",
#         "gh_test_lines_per_kloc", "gh_test_cases_per_kloc", "gh_asserts_cases_per_kloc",
#         "gh_team_size"
#     ]
#     logging.info("Normalizing numerical columns with MinMaxScaler")
#     scaler = MinMaxScaler()
#     available_numerical = [col for col in numerical_columns if col in df.columns]
#     df[available_numerical] = scaler.fit_transform(df[available_numerical].fillna(0))  # Fill missing values with 0
#     logging.info(f"Normalized columns: {available_numerical}")
    
#     # Save processed data
#     if split_by_project and output_path:
#         logging.info("Splitting processed data by project")
#         split_data_by_project(df, output_path, min_samples=min_samples)
    
#     logging.info("Completed preprocess_data function")
#     return df, scaler, label_encoders

# if __name__ == "__main__":
#     raw_data_dir = "data/raw"
#     output_path = "data/processed/by_project"
#     input_files = [
#         "travistorrent-2015.csv",
#         "travistorrent-2016.csv",
#         "travistorrent-2017.csv"
#     ]

#     columns_mapping_2015 = {
#         "status": "tr_status",
#         "project_name": "gh_project_name",
#         "is_pr": "gh_is_pr",
#         "main_team_member": "gh_by_core_team_member",
#         "team_size": "gh_team_size",
#         "sloc": "gh_sloc",
#         "started_at": "gh_build_started_at",
#         "num_commits": "git_num_commits",
#         "num_commit_comments": "gh_num_commit_comments",
#         "src_churn": "gh_src_churn",
#         "test_churn": "gh_test_churn",
#         "files_added": "gh_files_added",
#         "files_deleted": "gh_files_deleted",
#         "files_modified": "gh_files_modified",
#         "tests_added": "gh_tests_added",
#         "tests_deleted": "gh_tests_deleted",
#         "src_files": "gh_src_files",
#         "doc_files": "gh_doc_files",
#         "other_files": "gh_other_files",
#         "commits_on_files_touched": "gh_commits_on_files_touched",
#         "test_lines_per_kloc": "gh_test_lines_per_kloc",
#         "test_cases_per_kloc": "gh_test_cases_per_kloc",
#         "asserts_per_kloc": "gh_asserts_cases_per_kloc"
#     }

#     columns_mapping_2016 = {
#         "tr_status": "tr_status",
#         "gh_project_name": "gh_project_name",
#         "gh_is_pr": "gh_is_pr",
#         "gh_by_core_team_member": "gh_by_core_team_member",
#         "gh_team_size": "gh_team_size",
#         "gh_sloc": "gh_sloc",
#         "tr_started_at": "gh_build_started_at",
#         "git_num_commits": "git_num_commits",
#         "gh_num_commit_comments": "gh_num_commit_comments",
#         "gh_src_churn": "gh_src_churn",
#         "gh_test_churn": "gh_test_churn",
#         "gh_files_added": "gh_files_added",
#         "gh_files_deleted": "gh_files_deleted",
#         "gh_files_modified": "gh_files_modified",
#         "gh_tests_added": "gh_tests_added",
#         "gh_tests_deleted": "gh_tests_deleted",
#         "gh_src_files": "gh_src_files",
#         "gh_doc_files": "gh_doc_files",
#         "gh_other_files": "gh_other_files",
#         "gh_commits_on_files_touched": "gh_commits_on_files_touched",
#         "gh_test_lines_per_kloc": "gh_test_lines_per_kloc",
#         "gh_test_cases_per_kloc": "gh_test_cases_per_kloc",
#         "gh_asserts_cases_per_kloc": "gh_asserts_cases_per_kloc"
#     }

#     dtype_spec = {
#         "git_diff_src_churn": "float32",
#         "gh_diff_files_modified": "float32",
#         "gh_test_lines_per_kloc": "float32"
#     }
    
#     # Load and combine raw data
#     logging.info("Loading and combining raw data from multiple files")
#     dfs = []
#     for file in input_files:
#         file_path = os.path.join(raw_data_dir, file)
#         if os.path.exists(file_path):
#             logging.info(f"Loading {file_path}")
#             df_chunk = pd.read_csv(file_path, dtype=dtype_spec, low_memory=False)
#             logging.info(f"Columns in {file}: {list(df_chunk.columns)}")
            
#             if "2015" in file:
#                 df_chunk = df_chunk.rename(columns=columns_mapping_2015)
#             else:  # 2016 và 2017
#                 df_chunk = df_chunk.rename(columns=columns_mapping_2016)
            
#             columns_to_keep = list(set(columns_mapping_2015.values()) & set(columns_mapping_2016.values()))
#             available_columns = [col for col in columns_to_keep if col in df_chunk.columns]
#             filtered_df = df_chunk[available_columns].copy()
#             dfs.append(filtered_df)
#         else:
#             logging.warning(f"File {file_path} not found, skipping")
    
#     if not dfs:
#         logging.error("No input files found or loaded")
#         raise FileNotFoundError("No valid input files found in data/raw")
    
#     combined_df = pd.concat(dfs, ignore_index=True)
#     logging.info(f"Combined DataFrame shape: {combined_df.shape}")
    
#     processed_df, scaler, label_encoders = preprocess_data(
#         combined_df,
#         output_path=output_path,
#         split_by_project=True,
#         min_samples=5000
#     )
    
#     print(f"Processed DataFrame shape: {processed_df.shape}")

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

def process_chunk(chunk, columns_mapping, columns_to_keep):
    """Xử lý một chunk dữ liệu: chuẩn hóa tên cột, lọc cột, mapping build_failed."""
    chunk = chunk.rename(columns=columns_mapping)
    chunk = chunk[[col for col in columns_to_keep if col in chunk.columns]].copy()
    chunk['build_failed'] = chunk['tr_status'].map({'passed': 0, 'failed': 1, 'errored': 1, 'canceled': 1})
    chunk = chunk.drop(columns=['tr_status'])  # Xóa cột tr_status
    chunk['gh_build_started_at'] = pd.to_datetime(chunk['gh_build_started_at'], errors='coerce')
    return chunk

def split_data_by_project(df, output_path, min_samples=5000):
    os.makedirs(output_path, exist_ok=True)
    project_counts = df['gh_project_name'].value_counts()
    projects = project_counts[project_counts >= min_samples].index
    logging.info(f"Found {len(project_counts)} unique projects, {len(projects)} with >= {min_samples} samples")

    for i, project in enumerate(projects, 1):
        logging.info(f"Processing project {i}/{len(projects)}: {project}")
        file_name = str(project).replace("/", "_").replace(":", "_").replace(" ", "_") + ".csv"
        file_path = os.path.join(output_path, file_name)
        
        project_data = df[df['gh_project_name'] == project]
        project_data.to_csv(file_path, index=False)
        logging.info(f"Saved {project} with {len(project_data)} rows to {file_path}")

def preprocess_data(output_path, chunks, min_samples=5000):
    logging.info("Starting preprocess_data function")
    
    # Khởi tạo scaler và encoders
    scaler = MinMaxScaler()
    label_encoders = {}
    categorical_columns = ['gh_is_pr', 'gh_by_core_team_member']
    numerical_columns = [
        "git_num_commits", "gh_num_commit_comments", "gh_src_churn",
        "gh_test_churn", "gh_files_added", "gh_files_deleted", "gh_files_modified",
        "gh_tests_added", "gh_tests_deleted", "gh_src_files", "gh_doc_files",
        "gh_other_files", "gh_commits_on_files_touched", "gh_sloc",
        "gh_test_lines_per_kloc", "gh_test_cases_per_kloc", "gh_asserts_cases_per_kloc",
        "gh_team_size"
    ]
    
    # Xử lý từng chunk và gộp
    processed_chunks = []
    for chunk in chunks:
        logging.info(f"Processing chunk with shape: {chunk.shape}")
        
        # Loại NA ở cột quan trọng
        chunk = chunk.dropna()
        
        # Loại trùng lặp
        chunk = chunk.drop_duplicates()
        
        # Encode categorical
        for col in categorical_columns:
            if col in chunk.columns:
                if col not in label_encoders:
                    label_encoders[col] = LabelEncoder()
                chunk[col] = label_encoders[col].fit_transform(chunk[col].astype(str))
        
        # Điền NA bằng 0 cho numerical trong chunk
        available_numerical = [col for col in numerical_columns if col in chunk.columns]
        chunk[available_numerical] = chunk[available_numerical].fillna(0)
        
        processed_chunks.append(chunk)
    
    # Gộp các chunk
    df = pd.concat(processed_chunks, ignore_index=True)
    logging.info(f"Combined processed DataFrame shape: {df.shape}")
    
    # Kiểm tra NA trước khi chuẩn hóa
    logging.info(f"NA counts before final fillna: {df.isna().sum().to_dict()}")
    
    # Điền NA bằng 0 lần nữa và chuẩn hóa numerical
    df[numerical_columns] = df[numerical_columns].fillna(0)
    df[numerical_columns] = scaler.fit_transform(df[numerical_columns])
    logging.info(f"Normalized columns: {numerical_columns}")
    
    # Sắp xếp theo thời gian
    df = df.sort_values('gh_build_started_at').reset_index(drop=True)
    
    # Kiểm tra NA sau chuẩn hóa
    logging.info(f"NA counts after normalization: {df.isna().sum().to_dict()}")
    
    # Lưu theo project
    split_data_by_project(df, output_path, min_samples=min_samples)
    
    logging.info("Completed preprocess_data function")
    return df, scaler, label_encoders

if __name__ == "__main__":
    raw_data_dir = "data/raw"
    output_path = "data/processed/by_project"
    input_files = [
        "travistorrent-2015.csv",
        "travistorrent-2016.csv",
        "travistorrent-2017.csv"
    ]

    columns_mapping_2015 = {
        "status": "tr_status",
        "project_name": "gh_project_name",
        "is_pr": "gh_is_pr",
        "main_team_member": "gh_by_core_team_member",
        "team_size": "gh_team_size",
        "sloc": "gh_sloc",
        "started_at": "gh_build_started_at",
        "num_commits": "git_num_commits",
        "num_commit_comments": "gh_num_commit_comments",
        "src_churn": "gh_src_churn",
        "test_churn": "gh_test_churn",
        "files_added": "gh_files_added",
        "files_deleted": "gh_files_deleted",
        "files_modified": "gh_files_modified",
        "tests_added": "gh_tests_added",
        "tests_deleted": "gh_tests_deleted",
        "src_files": "gh_src_files",
        "doc_files": "gh_doc_files",
        "other_files": "gh_other_files",
        "commits_on_files_touched": "gh_commits_on_files_touched",
        "test_lines_per_kloc": "gh_test_lines_per_kloc",
        "test_cases_per_kloc": "gh_test_cases_per_kloc",
        "asserts_per_kloc": "gh_asserts_cases_per_kloc"
    }

    columns_mapping_2016 = {
        "tr_status": "tr_status",
        "gh_project_name": "gh_project_name",
        "gh_is_pr": "gh_is_pr",
        "gh_by_core_team_member": "gh_by_core_team_member",
        "gh_team_size": "gh_team_size",
        "gh_sloc": "gh_sloc",
        "tr_started_at": "gh_build_started_at",
        "git_num_commits": "git_num_commits",
        "gh_num_commit_comments": "gh_num_commit_comments",
        "gh_src_churn": "gh_src_churn",
        "gh_test_churn": "gh_test_churn",
        "gh_files_added": "gh_files_added",
        "gh_files_deleted": "gh_files_deleted",
        "gh_files_modified": "gh_files_modified",
        "gh_tests_added": "gh_tests_added",
        "gh_tests_deleted": "gh_tests_deleted",
        "gh_src_files": "gh_src_files",
        "gh_doc_files": "gh_doc_files",
        "gh_other_files": "gh_other_files",
        "gh_commits_on_files_touched": "gh_commits_on_files_touched",
        "gh_test_lines_per_kloc": "gh_test_lines_per_kloc",
        "gh_test_cases_per_kloc": "gh_test_cases_per_kloc",
        "gh_asserts_cases_per_kloc": "gh_asserts_cases_per_kloc"
    }

    dtype_spec = {
        "git_diff_src_churn": "float32",
        "gh_diff_files_modified": "float32",
        "gh_test_lines_per_kloc": "float32"
    }
    
    # Đọc và xử lý từng chunk
    chunks = []
    chunk_size = 500000
    columns_to_keep = list(set(columns_mapping_2015.values()) & set(columns_mapping_2016.values()))

    for file in input_files:
        file_path = os.path.join(raw_data_dir, file)
        if os.path.exists(file_path):
            logging.info(f"Loading {file_path} in chunks")
            for chunk in pd.read_csv(file_path, dtype=dtype_spec, low_memory=False, chunksize=chunk_size):
                if "2015" in file:
                    processed_chunk = process_chunk(chunk, columns_mapping_2015, columns_to_keep)
                else:  # 2016 và 2017
                    processed_chunk = process_chunk(chunk, columns_mapping_2016, columns_to_keep)
                chunks.append(processed_chunk)
        else:
            logging.warning(f"File {file_path} not found, skipping")
    
    if not chunks:
        logging.error("No input files found or loaded")
        raise FileNotFoundError("No valid input files found in data/raw")
    
    processed_df, scaler, label_encoders = preprocess_data(
        output_path=output_path,
        chunks=chunks,
        min_samples=5000
    )
    
    print(f"Processed DataFrame shape: {processed_df.shape}")