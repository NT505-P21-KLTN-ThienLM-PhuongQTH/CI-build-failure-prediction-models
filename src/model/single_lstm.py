# src/model/single_lstm.py
import os
import sys
import numpy as np
import pandas as pd
import logging
from timeit import default_timer as timer
from keras.models import Sequential
from keras.layers import Input, LSTM, Dense, Dropout
from keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils.class_weight import compute_class_weight
from ..utils import Utils as Utils
from ..optimization import GA_runner as GARunner
import pickle
import logging

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

MODEL_DIR = "models/single_lstm"
os.makedirs(MODEL_DIR, exist_ok=True)

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def train_preprocess(dataset_train, time_step):
    feature_cols = [col for col in dataset_train.columns 
                    if col not in ['gh_build_started_at', 'gh_project_name'] 
                    and dataset_train[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
    training_set = dataset_train[feature_cols].values
    y = dataset_train['build_failed'].values
    y = dataset_train['build_failed'].values

    if len(training_set) <= time_step:
        raise ValueError(f"Dataset size ({len(training_set)}) must be larger than time_step ({time_step})")

    scaler = MinMaxScaler()
    training_set = scaler.fit_transform(training_set)
    logger.info(f"Min and Max after scaling: {training_set.min()}, {training_set.max()}")

    logging.info("\nClass Distribution BEFORE SMOTE:")
    unique, counts = np.unique(y, return_counts=True)
    class_dist_before = dict(zip(unique, counts / len(y)))
    logging.info(class_dist_before)

    if Utils.CONFIG.get('WITH_SMOTE', True):
        logging.info("Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X, y_smote = smote.fit_resample(training_set, y)
        training_set = X
    else:
        y_smote = y

    logging.info("Class Distribution AFTER SMOTE:")
    unique, counts = np.unique(y_smote, return_counts=True)
    class_dist_after = dict(zip(unique, counts / len(y_smote)))
    logging.info(class_dist_after)

    X_train = np.lib.stride_tricks.sliding_window_view(training_set, (time_step, training_set.shape[1]))[:-1]
    X_train = np.squeeze(X_train, axis=1)
    y_train = y_smote[time_step:]

    logging.info(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    return X_train, y_train, scaler

def test_preprocess(dataset_train, dataset_test, time_step, scaler):
    feature_cols = [col for col in dataset_train.columns 
                    if col not in ['gh_build_started_at', 'gh_project_name'] 
                    and dataset_train[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
    
    train_scaled = scaler.transform(dataset_train[feature_cols])  # Use transform, not fit_transform for test set because we want to use the same scaler
    test_scaled = scaler.transform(dataset_test[feature_cols])
    dataset_total_scaled = np.vstack((train_scaled, test_scaled))
    y_test = dataset_test['build_failed'].values
    
    if len(dataset_total_scaled) < time_step + len(dataset_test):
        raise ValueError("Not enough data to create test sequences with given time_step")

    inputs = dataset_total_scaled[len(dataset_total_scaled) - len(dataset_test) - time_step:]
    X_test = np.lib.stride_tricks.sliding_window_view(inputs, (time_step, inputs.shape[1]))[:-1]
    X_test = np.squeeze(X_test, axis=1)

    logging.info(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return X_test, y_test

def construct_lstm_model(network_params, train_set):
    X_train, y_train, scaler = train_preprocess(train_set, network_params["time_step"])
    drop = network_params["drop_proba"]
    
    model = Sequential([
        Input(shape=(X_train.shape[1], X_train.shape[2])),
        LSTM(units=network_params["nb_units"], return_sequences=(network_params["nb_layers"] > 1)),
        Dropout(drop),
        *[LSTM(units=network_params["nb_units"], return_sequences=(i < network_params["nb_layers"] - 1)) 
          for i in range(1, network_params["nb_layers"])],
        Dropout(drop),
        Dense(units=1, activation='sigmoid')
    ])

    model.compile(optimizer=network_params["optimizer"], loss='binary_crossentropy', metrics=["accuracy"])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)  # Reduce overfitting
    
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
    
    history = model.fit(
        X_train, y_train, 
        epochs=network_params["nb_epochs"],
        batch_size=network_params["nb_batch"], 
        validation_split=0.2,
        verbose=1, 
        callbacks=[es], 
        class_weight=class_weight_dict
    )

    validation_loss = np.min(history.history['val_loss'])
    train_loss = np.min(history.history['loss'])
    logging.info(f"Train Loss: {train_loss}, Validation Loss: {validation_loss}")
    
    entry = Utils.predict_lstm(model, X_train, y_train)
    entry['validation_loss'] = validation_loss

    model_path = os.path.join(MODEL_DIR, f"lstm_{network_params['nb_units']}_{network_params['nb_layers']}.keras")
    model.save(model_path)
    logging.info(f"Model saved: {model_path}")

    scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
    with open(scaler_path, 'wb') as f:
        pickle.dump(scaler, f)
    logging.info(f"Scaler saved to {scaler_path}")

    return {'validation_loss': validation_loss, 'model': model, 'entry': entry, 'scaler': scaler}

def evaluate_tuner(tuner_option, train_set):
    global data
    data = train_set

    all_possible_params = {
        'drop_proba': list(np.linspace(0.01, 0.2, 20)),
        'nb_units': [32, 64, 128],
        'nb_epochs': [5, 10, 15],
        'nb_batch': [4, 8, 16, 32, 64],
        'nb_layers': [1, 2, 3, 4],
        'optimizer': ['adam', 'rmsprop'],
        'time_step': list(range(5, 31))
    }

    start = timer()

    if tuner_option == "ga":
        best_params, best_model, entry_train = GARunner.generate(all_possible_params, construct_lstm_model, data)
        scaler = construct_lstm_model(best_params, data)['scaler']  # Reconstruct the scaler

    end = timer()
    entry_train.update({"time": end - start, "params": best_params, "model": best_model})

    best_model_path = os.path.join(MODEL_DIR, "best_lstm_model.keras")
    best_model.save(best_model_path)
    logging.info(f"Best model saved at: {best_model_path}")

    return entry_train, scaler

if __name__ == "__main__":
    dataset = Utils.get_dataset("apache_jackrabbit-oak.csv")
    
    logging.info("Dataset Info:")
    logging.info(dataset.info())
    logging.info("\nDataset Head:")
    logging.info(dataset.head())
    
    train_sets, test_sets = Utils.online_validation_folds(dataset)
    logging.info(f"Number of train folds: {len(train_sets)}, test folds: {len(test_sets)}")
    
    # Evaluate GA tuner
    entry_train_ga, scaler = evaluate_tuner("ga", train_sets[0])
    
    # Test the best model
    X_test, y_test = test_preprocess(train_sets[0], test_sets[0], entry_train_ga["params"]["time_step"], scaler)
    entry_test = Utils.predict_lstm(entry_train_ga["model"], X_test, y_test)
    
    # Print results
    logging.info("Train Results:")
    logging.info(entry_train_ga)
    logging.info("Test Results:")
    logging.info(entry_test)

    logging.info(dataset['build_failed'].value_counts(normalize=True))

    # dataset_dir = "data/processed/by_project"
    # project_info = []

    # for filename in os.listdir(dataset_dir):
    #     if filename.endswith(".csv"):
    #         dataset = Utils.get_dataset(filename)
    #         if "build_failed" in dataset.columns:
    #             total_rows = len(dataset)
    #             class_ratio = dataset["build_failed"].value_counts(normalize=True).to_dict()
    #             class_absolute = dataset["build_failed"].value_counts().to_dict()

    #             project_info.append({
    #                 "project": filename,
    #                 "rows": total_rows,
    #                 "class_ratio": class_ratio,
    #                 "class_absolute": class_absolute
    #             })

    # top_projects = sorted(project_info, key=lambda x: x["rows"], reverse=True)[:20]

    # for project in top_projects:
    #     print(f"\n--- Project: {project['project']} ---")
    #     print(f"Total rows: {project['rows']}")
    #     print("Class distribution (ratio):")
    #     print(project["class_ratio"])
    #     print("Class distribution (absolute):")
    #     print(project["class_absolute"])
