import os
import sys
import numpy as np
import pandas as pd
from timeit import default_timer as timer
from hyperopt import hp, Trials, STATUS_OK, fmin, tpe, rand
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
from imblearn.over_sampling import SMOTE
import optunity
import ConfigSpace as CS
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB
from sklearn.utils.class_weight import compute_class_weight

# Import custom modules
from ..utils import helpers
from ..optimization import GA_runner

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)
MODEL_DIR = "models/single_lstm"
os.makedirs(MODEL_DIR, exist_ok=True)

# Ensure model directory exists
def train_preprocess(dataset_train, time_step):
    # Preprocess the training dataset for LSTM.
    feature_cols = [col for col in dataset_train.columns 
                    if col not in ['build_failed', 'gh_build_started_at', 'gh_project_name'] 
                    and dataset_train[col].dtype in [np.float64, np.float32, np.int64, np.int32]]
    training_set = dataset_train[feature_cols].values
    y = dataset_train['build_failed'].values # Target column

    if len(training_set) <= time_step:
        raise ValueError(f"Dataset size ({len(training_set)}) must be larger than time_step ({time_step})")

    print("\nClass Distribution BEFORE SMOTE:")
    unique, counts = np.unique(y, return_counts=True)
    print(dict(zip(unique, counts / len(y))))

    print("\nClass Distribution BEFORE SMOTE:")
    unique, counts = np.unique(y, return_counts=True)
    class_dist_before = dict(zip(unique, counts / len(y)))
    print(class_dist_before)

    if helpers.CONFIG.get('WITH_SMOTE', True):
        print("Applying SMOTE...")
        smote = SMOTE(random_state=42)
        X, y_smote = smote.fit_resample(training_set, y)
        training_set = X
    else:
        y_smote = y

    print("Class Distribution AFTER SMOTE:")
    unique, counts = np.unique(y_smote, return_counts=True)
    print(dict(zip(unique, counts / len(y_smote))))

    # Create sequences for LSTM
    X_train = np.lib.stride_tricks.sliding_window_view(training_set, (time_step, training_set.shape[1]))[:-1]
    X_train = np.squeeze(X_train, axis=1)
    y_train = y_smote[time_step:]
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    return X_train, y_train

def test_preprocess(dataset_train, dataset_test, time_step):
    # Preprocess the test dataset for LSTM.
    feature_cols = [col for col in dataset_train.columns 
                    if col not in ['build_failed', 'gh_build_started_at', 'gh_project_name'] 
                    and dataset_train[col].dtype in [np.float64, np.float32, np.int64, np.int32]]

    train_data = dataset_train[feature_cols].values
    test_data = dataset_test[feature_cols].values
    dataset_total = np.vstack((train_data, test_data))
    y_test = dataset_test['build_failed'].values
    
    if len(dataset_total) < time_step + len(dataset_test):
        raise ValueError("Not enough data for test sequences")
    
    inputs = dataset_total[-len(dataset_test) - time_step:]
    X_test = np.lib.stride_tricks.sliding_window_view(inputs, (time_step, inputs.shape[1]))[:-1]
    X_test = np.squeeze(X_test, axis=1)
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    return X_test, y_test

def construct_lstm_model(network_params, train_set):
    # Construct and train the LSTM model.
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

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    history = model.fit(X_train, y_train, epochs=network_params["nb_epochs"],
                        batch_size=network_params["nb_batch"], validation_split=0.2,
                        verbose=0, callbacks=[es], class_weight=class_weight_dict)

    validation_loss = np.amin(history.history['val_loss'])
    entry = helpers.predict_lstm(model, X_train, y_train)
    entry['validation_loss'] = validation_loss

    model_path = os.path.join(MODEL_DIR, f"lstm_{network_params['nb_units']}_{network_params['nb_layers']}.keras")
    model.save(model_path)
    print(f"Model saved: {model_path}")
    return {'validation_loss': validation_loss, 'model': model, 'entry': entry}

def train_lstm_with_hyperopt(network_params):
    # Train LSTM with hyperopt.
    if 'data' not in globals():
        raise ValueError("Global 'data' not set. Ensure evaluate_tuner sets it correctly.")
    res = construct_lstm_model(network_params, globals()['data'])
    return {'loss': res['validation_loss'], 'status': STATUS_OK}

def convert_from_PSO(network_params):
    # Convert PSO parameters to appropriate types.
    for key in network_params:
        if key == 'optimizer':
            network_params[key] = 'adam' if int(network_params[key]) == 1 else 'rmsprop'
        elif key == 'nb_layers':
            network_params[key] = int(network_params[key])
    return network_params

def fn_lstm_pso(drop_proba=0.01, nb_units=32, nb_epochs=2, nb_batch=4, nb_layers=1, optimizer=1, time_step=30):
    # Function for PSO optimization.
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
    # Evaluate the specified tuner.
    global data
    data = train_set

    # Define explicit parameter space for GA
    all_possible_params = {
        'drop_proba': list(np.linspace(0.01, 0.2, 20)),
        'nb_units': [32, 64],
        'nb_epochs': [4, 5, 6],
        'nb_batch': [4, 8, 16, 32, 64], # Power of 2
        'nb_layers': [1, 2, 3, 4],
        'optimizer': ['adam', 'rmsprop'],
        'time_step': list(range(30, 61))
    }

    start = timer()

    if tuner_option == "ga":
        best_params, best_model, entry_train = GA_runner.generate(all_possible_params, construct_lstm_model, data)

    elif tuner_option == "tpe":
        param_space = {k: hp.choice(k, v) for k, v in all_possible_params.items()}
        trials = Trials()
        best = fmin(train_lstm_with_hyperopt, param_space, algo=tpe.suggest, max_evals=helpers.MAX_EVAL, trials=trials)
        best_params = {k: all_possible_params[k][v] for k, v in best.items()}
        res = construct_lstm_model(best_params, data)
        entry_train, best_model = res["entry"], res["model"]

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
        best_params, _, _ = optunity.maximize_structured(fn_lstm_pso, params_PSO, num_evals=helpers.MAX_EVAL)
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
        bohb = BOHB(configspace=config_space, run_id="LSTM", nameserver='127.0.0.1', min_budget=1, max_budget=helpers.NBR_SOL)
        res = bohb.run(n_iterations=helpers.NBR_GEN)
        best = res.get_incumbent_id()
        best_params = res.get_id2config_mapping()[best]['config']
        res = construct_lstm_model(best_params, data)
        entry_train, best_model = res["entry"], res["model"]
        bohb.shutdown(shutdown_workers=True)
        NS.shutdown()

    elif tuner_option == "rs":
        param_space = {k: hp.choice(k, v) for k, v in all_possible_params.items()}
        trials = Trials()
        best = fmin(train_lstm_with_hyperopt, param_space, algo=rand.suggest, max_evals=helpers.MAX_EVAL, trials=trials)
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
    best_model_path = os.path.join(MODEL_DIR, "best_lstm_model.keras")
    best_model.save(best_model_path)
    print(f"Best model saved at: {best_model_path}")
    return entry_train

if __name__ == "__main__":
    dataset = helpers.get_dataset("getsentry_sentry.csv")
    
    print("Dataset Info:")
    print(dataset.info())
    print("\nDataset Head:")
    print(dataset.head())
    
    train_sets, test_sets = helpers.online_validation_folds(dataset)

    # Evaluate GA tuner
    entry_train_ga = evaluate_tuner("ga", train_sets[0])
    # Test the best model
    X, y = test_preprocess(train_sets[0], test_sets[0], entry_train_ga["params"]["time_step"])
    entry_test = helpers.predict_lstm(entry_train_ga["model"], X, y)
    
    # Print results
    print(entry_train_ga) # Best model from GA
    print(entry_test) # Test results

    print(dataset['build_failed'].value_counts(normalize=True))


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