import os
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from hyperopt import hp, Trials, STATUS_OK, fmin, tpe, rand
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
import optunity
import optunity.metrics
import src.utils.Utils as Utils
import src.optimization.GA_runner as GARunner
import ConfigSpace as CS
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB

# Ensure model directory exists
MODEL_DIR = "models/single_lstm"
os.makedirs(MODEL_DIR, exist_ok=True)

def train_preprocess(dataset_train, time_step):
    """
    Preprocess the training dataset for LSTM.

    Args:
        dataset_train (DataFrame): Training dataset.
        time_step (int): Number of time steps for LSTM.

    Returns:
        tuple: Processed X_train and y_train.
    """
    training_set = dataset_train.iloc[:, :19].values

    if len(training_set) <= time_step:
        raise ValueError(f"Dataset size ({len(training_set)}) must be larger than time_step ({time_step})")

    # Normalize data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    training_set = scaler.fit_transform(training_set)

    # Apply SMOTE if enabled
    if Utils.WITH_SMOTE:
        X, y = SMOTE().fit_resample(training_set, dataset_train.iloc[:, 0].values)
        training_set = X
    else:
        y = dataset_train.iloc[:, 0].values

    X_train = np.array([training_set[i - time_step:i, :] for i in range(time_step, len(training_set))])
    y_train = np.array([training_set[i, 0] for i in range(time_step, len(training_set))])

    return X_train, y_train

def test_preprocess(dataset_train, dataset_test, time_step):
    """
    Preprocess the testing dataset for LSTM.

    Args:
        dataset_train (DataFrame): Training dataset.
        dataset_test (DataFrame): Testing dataset.
        time_step (int): Number of time steps for LSTM.

    Returns:
        tuple: Processed X_test and y_test.
    """
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
    # Normalize data
    dataset_total = pd.concat((dataset_train.iloc[:, :19], dataset_test.iloc[:, :19]), axis=0)
    dataset_total_scaled = scaler.fit_transform(dataset_total)
    
    if len(dataset_total) < time_step + len(dataset_test):
        raise ValueError("Not enough data to create test sequences with given time_step")
    
    # dataset_total = pd.concat((dataset_train['build_Failed'], dataset_test['build_Failed']), axis=0)

    inputs = dataset_total_scaled[len(dataset_total) - len(dataset_test) - time_step:]
    X_test = np.array([inputs[j - time_step:j, :] for j in range(time_step, len(inputs))])
    y_test = dataset_test.iloc[:, 0].values

    return X_test, y_test

def construct_lstm_model(network_params, train_set):
    """
    Constructs and trains an LSTM model.

    Args:
        network_params (dict): Hyperparameters for the LSTM model.
        train_set (DataFrame): Training dataset.

    Returns:
        dict: Dictionary containing the model, validation loss, and metrics.
    """
    X_train, y_train = train_preprocess(train_set, network_params["time_step"])
    drop = round(network_params["drop_proba"])

    model = Sequential([
        LSTM(units=network_params["nb_units"], return_sequences=True, input_shape=(X_train.shape[1], 19)),
        Dropout(drop),
        *[LSTM(units=network_params["nb_units"], return_sequences=True) if i < network_params["nb_layers"] - 1 else
          LSTM(units=network_params["nb_units"]) for i in range(network_params["nb_layers"])],
        Dropout(drop),
        Dense(units=1, activation='sigmoid')
    ])

    model.compile(optimizer=network_params["optimizer"], loss='binary_crossentropy', metrics=["accuracy"])
    es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=10)

    history = model.fit(X_train, y_train, epochs=network_params["nb_epochs"],
                        batch_size=network_params["nb_batch"], verbose=0, callbacks=[es])

    validation_loss = np.amin(history.history['loss'])
    entry = Utils.predict_lstm(model, X_train, y_train)
    entry['validation_loss'] = validation_loss

    model_name = f"lstm_{network_params['nb_units']}_{network_params['nb_layers']}.h5"
    model_path = os.path.join(MODEL_DIR, model_name)
    model.save(model_path)
    print(f"Model saved: {model_path}")

    return {'validation_loss': validation_loss, 'model': model, 'entry': entry}

global data
global global_params
global global_model
global global_entry

def train_lstm_with_hyperopt(network_params):
    """
    Train LSTM using Hyperopt.

    Args:
        network_params (dict): Hyperparameters.

    Returns:
        dict: Loss and status for Hyperopt.
    """
    res = construct_lstm_model(network_params, data)
    return {'loss': res['validation_loss'], 'status': STATUS_OK}


def evaluate_tuner(tuner_option, train_set):
    """
    Evaluate hyperparameter tuning methods for LSTM.

    Args:
        tuner_option (str): Tuning method.
        train_set (DataFrame): Training dataset.

    Returns:
        dict: Best hyperparameters and metrics.
    """
    global data
    data = train_set

    # Define explicit parameter space for GA
    all_possible_params = {
        'drop_proba': list(np.linspace(0.01, 0.2, 20)),
        'nb_units': [32, 64],
        'nb_epochs': [4, 5, 6],
        'nb_batch': [4, 8, 16, 32, 64],
        'nb_layers': [1, 2, 3, 4],
        'optimizer': ['adam', 'rmsprop'],
        'time_step': list(range(30, 61))
    }

    start = timer()

    if tuner_option == "tpe":
        param_space = {k: hp.choice(k, v) for k, v in all_possible_params.items()}
        trials = Trials()
        fmin(train_lstm_with_hyperopt, param_space, algo=tpe.suggest, max_evals=Utils.MAX_EVAL, trials=trials)
        best_params, best_model, entry_train = global_params, global_model, global_entry

    elif tuner_option == "ga":
        best_params, best_model, entry_train = GARunner.generate(all_possible_params, construct_lstm_model, data)

    elif tuner_option == "default":
        best_params = {
            'nb_units': 64, 'nb_layers': 3, 'optimizer': 'adam', 'time_step': 30,
            'nb_epochs': 10, 'nb_batch': 64, 'drop_proba': 0.1
        }
        res = construct_lstm_model(best_params, data)
        entry_train, best_model = res["entry"], res["model"]

    end = timer()
    entry_train.update({"time": end - start, "params": best_params, "model": best_model})

    best_model_path = os.path.join(MODEL_DIR, "best_lstm_model.h5")
    if best_model:
        best_model.save(best_model_path)
        print(f"Best model saved at: {best_model_path}")

    return entry_train


if __name__ == "__main__":
    import hpbandster.core.nameserver as hpns

    NS = hpns.NameServer(run_id="LSTM", host='127.0.0.1', port=None)
    NS.start()

    dataset = Utils.get_dataset("sonarqube.csv")
    
    print("Dataset Type:", type(dataset))
    print("Dataset Shape:", dataset.shape)
    print("Dataset Columns:", dataset.columns)
    print(dataset.head())
    
    train_sets, test_sets = Utils.online_validation_folds(dataset)

    entry_train_ga = evaluate_tuner("ga", train_sets[0])
    X, y = test_preprocess(train_sets[0], test_sets[0], entry_train_ga["params"]["time_step"])
    entry_test = Utils.predict_lstm(entry_train_ga["model"], X, y)
    
    print(entry_train_ga)
    print(entry_test)
    
    NS.shutdown()

    