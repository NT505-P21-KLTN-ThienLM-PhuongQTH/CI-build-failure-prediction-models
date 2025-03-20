import os
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from hyperopt import hp, Trials, STATUS_OK, fmin, tpe, rand
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
# import src.utils.Utils as Utils
# import src.optimization.GA_runner as GARunner
from ..utils import Utils as Utils
from ..optimization import GA_runner as GARunner
import optunity
import ConfigSpace as CS
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB
import sys

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

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
    feature_cols = [col for col in dataset_train.columns 
                    if col not in ['build_failed', 'gh_build_started_at', 'gh_project_name'] 
                    and dataset_train[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
    training_set = dataset_train[feature_cols].values
    y = dataset_train['build_failed'].values # Target column

    if len(training_set) <= time_step:
        raise ValueError(f"Dataset size ({len(training_set)}) must be larger than time_step ({time_step})")

    # Normalize data
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    training_set = scaler.fit_transform(training_set)

    # Apply SMOTE if enabled
    if Utils.CONFIG['WITH_SMOTE']:
        X, y_smote = SMOTE().fit_resample(training_set, y)
        training_set = X
    else:
        y_smote = y

    # Create sequences for LSTM
    X_train = np.lib.stride_tricks.sliding_window_view(training_set, (time_step, training_set.shape[1]))[:-1]
    X_train = np.squeeze(X_train, axis=1)
    y_train = y_smote[time_step:]

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
    feature_cols = [col for col in dataset_train.columns 
                    if col not in ['build_failed', 'gh_build_started_at', 'gh_project_name'] 
                    and dataset_train[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
    from sklearn.preprocessing import MinMaxScaler
    scaler = MinMaxScaler()
    
    # Normalize data
    train_scaled = scaler.fit_transform(dataset_train[feature_cols])
    test_scaled = scaler.transform(dataset_test[feature_cols])
    dataset_total_scaled = np.vstack((train_scaled, test_scaled))
    y_test = dataset_test['build_failed'].values # Target column
    
    if len(dataset_total_scaled) < time_step + len(dataset_test):
        raise ValueError("Not enough data to create test sequences with given time_step")
    
    inputs = dataset_total_scaled[len(dataset_total_scaled) - len(dataset_test) - time_step:]
    X_test = np.lib.stride_tricks.sliding_window_view(inputs, (time_step, inputs.shape[1]))[:-1]
    X_test = np.squeeze(X_test, axis=1)

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
    drop = network_params["drop_proba"]
    
    model = Sequential([
        LSTM(units=network_params["nb_units"], return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2])),
        Dropout(drop),
        *[LSTM(units=network_params["nb_units"], return_sequences=True) if i < network_params["nb_layers"] - 1 else
          LSTM(units=network_params["nb_units"]) for i in range(network_params["nb_layers"])],
        Dropout(drop),
        Dense(units=1, activation='sigmoid')
    ])

    model.compile(optimizer=network_params["optimizer"], loss='binary_crossentropy', metrics=["accuracy"])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)
    
    history = model.fit(X_train, y_train, epochs=network_params["nb_epochs"],
                        batch_size=network_params["nb_batch"], validation_split=0.2,
                        verbose=0, callbacks=[es])

    validation_loss = np.amin(history.history['val_loss'])
    entry = Utils.predict_lstm(model, X_train, y_train)
    entry['validation_loss'] = validation_loss

    model_path = os.path.join(MODEL_DIR, f"lstm_{network_params['nb_units']}_{network_params['nb_layers']}.h5")
    model.save(model_path)
    print(f"Model saved: {model_path}")

    return {'validation_loss': validation_loss, 'model': model, 'entry': entry}

# global data
# global global_params
# global global_model
# global global_entry

def train_lstm_with_hyperopt(network_params):
    """
    Train LSTM using Hyperopt.

    Args:
        network_params (dict): Hyperparameters.

    Returns:
        dict: Loss and status for Hyperopt.
    """
    if 'data' not in globals():
        raise ValueError("Global 'data' not set. Ensure evaluate_tuner sets it correctly.")
    res = construct_lstm_model(network_params, globals()['data'])
    return {'loss': res['validation_loss'], 'status': STATUS_OK}

def convert_from_PSO(network_params):
    """
    Convert PSO params (e.g., optimizer from int to string).

    Args:
        network_params (dict): Parameters from PSO.

    Returns:
        dict: Converted parameters.
    """
    for key in network_params:
        if key == 'optimizer':
            network_params[key] = 'adam' if int(network_params[key]) == 1 else 'rmsprop'
        elif key == 'nb_layers':
            network_params[key] = int(network_params[key])
    return network_params

def fn_lstm_pso(drop_proba=0.01, nb_units=32, nb_epochs=2, nb_batch=4, nb_layers=1, optimizer=1, time_step=30):
    """
    Objective function for PSO.

    Args:
        Hyperparameters as individual arguments.

    Returns:
        float: 1 - validation_loss (maximize).
    """
    optimizer = 'adam' if int(optimizer) == 1 else 'rmsprop'
    network_params = {
        'nb_units': int(nb_units),
        'nb_layers': int(nb_layers),
        'optimizer': optimizer,
        'time_step': int(time_step),
        'nb_epochs': int(nb_epochs),
        'nb_batch': int(nb_batch),
        'drop_proba': drop_proba
    }
    if 'data' not in globals():
        raise ValueError("Global 'data' not set.")
    res = construct_lstm_model(network_params, globals()['data'])
    return 1 - float(res["validation_loss"])

class LSTMWorker(Worker):
    def __init__(self, train_set, **kwargs):
        super().__init__(**kwargs)
        self.train_set = train_set

    def compute(self, config, budget, **kwargs):
        res = construct_lstm_model(config, self.train_set)
        return {'loss': res['validation_loss'], 'info': {}}

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
        best = fmin(train_lstm_with_hyperopt, param_space, algo=tpe.suggest, max_evals=Utils.MAX_EVAL, trials=trials)
        best_params = {k: all_possible_params[k][v] for k, v in best.items()}
        res = construct_lstm_model(best_params, data)
        entry_train, best_model = res["entry"], res["model"]

    elif tuner_option == "ga":
        best_params, best_model, entry_train = GARunner.generate(all_possible_params, construct_lstm_model, data)

    elif tuner_option == "pso":
        params_PSO = {
            'nb_units': [all_possible_params['nb_units'][0], all_possible_params['nb_units'][-1]],
            'nb_layers': [all_possible_params['nb_layers'][0], all_possible_params['nb_layers'][-1]],
            'optimizer': [1, 2],  # 1: adam, 2: rmsprop
            'time_step': [all_possible_params['time_step'][0], all_possible_params['time_step'][-1]],
            'nb_epochs': [all_possible_params['nb_epochs'][0], all_possible_params['nb_epochs'][-1]],
            'nb_batch': [all_possible_params['nb_batch'][0], all_possible_params['nb_batch'][-1]],
            'drop_proba': [all_possible_params['drop_proba'][0], all_possible_params['drop_proba'][-1]]
        }
        best_params, _, _ = optunity.maximize_structured(fn_lstm_pso, params_PSO, num_evals=Utils.MAX_EVAL)
        best_params = convert_from_PSO(best_params)
        res = construct_lstm_model(best_params, data)
        entry_train, best_model = res["entry"], res["model"]
        
    elif tuner_option == "bohb":
        config_space = CS.ConfigurationSpace()
        config_space.add(CS.UniformIntegerHyperparameter('nb_units', lower=32, upper=64))
        config_space.add(CS.UniformIntegerHyperparameter('nb_layers', lower=1, upper=4))
        config_space.add(CS.CategoricalHyperparameter('optimizer', choices=['adam', 'rmsprop']))
        config_space.add(CS.UniformIntegerHyperparameter('time_step', lower=30, upper=60))
        config_space.add(CS.UniformIntegerHyperparameter('nb_epochs', lower=4, upper=6))
        config_space.add(CS.UniformIntegerHyperparameter('nb_batch', lower=4, upper=64))
        config_space.add(CS.UniformFloatHyperparameter('drop_proba', lower=0.01, upper=0.2))

        import hpbandster.core.nameserver as hpns
        NS = hpns.NameServer(run_id="LSTM", host='127.0.0.1', port=None)
        NS.start()
        w = LSTMWorker(train_set=data, nameserver='127.0.0.1', run_id="LSTM")
        w.run(background=True)
        bohb = BOHB(configspace=config_space, run_id="LSTM", nameserver='127.0.0.1', min_budget=1, max_budget=Utils.NBR_SOL)
        res = bohb.run(n_iterations=Utils.NBR_GEN)
        best = res.get_incumbent_id()
        best_params = res.get_id2config_mapping()[best]['config']
        res = construct_lstm_model(best_params, data)
        entry_train, best_model = res["entry"], res["model"]
        bohb.shutdown(shutdown_workers=True)
        NS.shutdown()

    elif tuner_option == "rs":
        param_space = {k: hp.choice(k, v) for k, v in all_possible_params.items()}
        trials = Trials()
        best = fmin(train_lstm_with_hyperopt, param_space, algo=rand.suggest, max_evals=Utils.MAX_EVAL, trials=trials)
        best_params = {k: all_possible_params[k][v] for k, v in best.items()}
        res = construct_lstm_model(best_params, data)
        entry_train, best_model = res["entry"], res["model"]

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
    best_model.save(best_model_path)
    print(f"Best model saved at: {best_model_path}")

    return entry_train

if __name__ == "__main__":
    dataset = Utils.get_dataset("ros_rosdistro.csv")
    
    # print("Dataset Info:")
    # print(dataset.info())
    # print("\nDataset Head:")
    # print(dataset.head())
    
    # train_sets, test_sets = Utils.online_validation_folds(dataset)

    # # Evaluate GA tuner
    # entry_train_ga = evaluate_tuner("ga", train_sets[0])
    # # Test the best model
    # X, y = test_preprocess(train_sets[0], test_sets[0], entry_train_ga["params"]["time_step"])
    # entry_test = Utils.predict_lstm(entry_train_ga["model"], X, y)
    
    # # Print results
    # print(entry_train_ga) # Best model from GA
    # print(entry_test) # Test results
    print(dataset['build_failed'].value_counts(normalize=True))