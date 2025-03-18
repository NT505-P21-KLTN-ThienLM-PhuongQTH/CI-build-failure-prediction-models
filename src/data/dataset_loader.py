import os
import numpy as np
import pandas as pd

def create_sequences(data, seq_length):
    """
    Create sequences of time series data.

    :param data: Time series data.
    :param seq_length: Length of time series.
    :return: Time series and corresponding labels.
    """
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        x = data.iloc[i:(i + seq_length)].values
        y = data.iloc[i + seq_length].values
        xs.append(x)
        ys.append(y)
    return np.array(xs), np.array(ys)

def save_sequences(sequences, labels, project_name, output_path):
    """
    Save the time series and labels to a numpy file.

    :param sequences: Time series.
    :param labels: Corresponding label.
    :param project_name: Project name.
    :param output_path: Directory to save data.
    """
    os.makedirs(output_path, exist_ok=True)

    sequences_file = os.path.join(output_path, f"{project_name}_sequences.npy")
    labels_file = os.path.join(output_path, f"{project_name}_labels.npy")

    np.save(sequences_file, sequences)
    np.save(labels_file, labels)

def process_project_data(project_file, seq_length, output_path):
    """
    Process data of a project: divide into time series and save to file.

    :param project_file: Path to the project CSV file.
    :param seq_length: Length of time series.
    :param output_path: Directory to save data.
    """
    df = pd.read_csv(project_file)

    sequences, labels = create_sequences(df, seq_length)

    project_name = os.path.splitext(os.path.basename(project_file))[0]
    save_sequences(sequences, labels, project_name, output_path)

if __name__ == "__main__":
    processed_data_path = "data/processed/"
    sequences_path = "data/sequences/"

    seq_length = 10

    for file_name in os.listdir(processed_data_path):
        if file_name.endswith(".csv"):
            project_file = os.path.join(processed_data_path, file_name)
            process_project_data(project_file, seq_length, sequences_path)