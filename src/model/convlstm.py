# src/model/convlstm.py
import os
import sys
import numpy as np
import pandas as pd
import logging
from timeit import default_timer as timer
from hyperopt import hp, Trials, STATUS_OK, fmin, tpe, rand
from keras.models import Sequential
from keras.layers import ConvLSTM2D, Dropout, Dense, Flatten, Input
from keras.callbacks import EarlyStopping
from keras.regularizers import l2
from src.utils.Utils import Utils
from src.optimization.GA_runner import GARunner
import optunity
import ConfigSpace as CS
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB
from src.data.dataset_loader import get_dataset

# Thêm các import mới
from src.data.sequence_builder import preprocess_for_convlstm_train, preprocess_for_convlstm_test
from src.data.data_balancer import apply_smote, compute_balanced_class_weights
from src.data.data_splitter import split_train_test

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
sys.path.append(project_root)

# Ensure model directory exists
MODEL_DIR = os.path.join(project_root, "models", "convlstm")
os.makedirs(MODEL_DIR, exist_ok=True)

def construct_convlstm_model(network_params, params_fn):
    """
    Constructs and trains a ConvLSTM model.

    Args:
        network_params (dict): Hyperparameters for the ConvLSTM model.
        params_fn (dict): Dictionary containing the training dataset under key 'train_set'.

    Returns:
        dict: Dictionary containing the model, validation loss, and metrics.
    """
    # Extract train_set from params_fn
    if not isinstance(params_fn, dict) or 'train_set' not in params_fn:
        raise ValueError("params_fn must be a dictionary containing 'train_set' key")

    train_set = params_fn['train_set']
    if not isinstance(train_set, pd.DataFrame):
        raise TypeError(f"Expected train_set to be a pandas DataFrame, got {type(train_set)}")

    # Xử lý dữ liệu huấn luyện
    X_train, y_train, scaler = preprocess_for_convlstm_train(
        train_set, network_params["time_step"], with_smote=Utils.CONFIG['WITH_SMOTE']
    )

    # Áp dụng SMOTE nếu được bật
    if Utils.CONFIG['WITH_SMOTE']:
        X_train_2d = X_train.reshape(X_train.shape[0], -1)  # Chuyển về 2D để áp dụng SMOTE
        X_train_2d, y_train = apply_smote(X_train_2d, y_train)
        X_train = X_train_2d.reshape(X_train_2d.shape[0], X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)

    drop = network_params["drop_proba"]

    # Build ConvLSTM model with Input layer
    model = Sequential()
    model.add(Input(shape=(X_train.shape[1], X_train.shape[2], X_train.shape[3], 1)))
    if network_params["nb_layers"] == 1:
        model.add(ConvLSTM2D(filters=network_params["nb_filters"], kernel_size=(3, 3),
                             padding='same', return_sequences=False, kernel_regularizer=l2(0.01)))
    else:
        model.add(ConvLSTM2D(filters=network_params["nb_filters"], kernel_size=(3, 3),
                             padding='same', return_sequences=True, kernel_regularizer=l2(0.01)))
        for _ in range(network_params["nb_layers"] - 2):
            model.add(ConvLSTM2D(filters=network_params["nb_filters"], kernel_size=(3, 3),
                                 padding='same', return_sequences=True, kernel_regularizer=l2(0.01)))
        model.add(ConvLSTM2D(filters=network_params["nb_filters"], kernel_size=(3, 3),
                             padding='same', return_sequences=False, kernel_regularizer=l2(0.01)))

    model.add(Dropout(drop))
    model.add(Flatten())
    model.add(Dense(units=1, activation='sigmoid'))

    model.compile(optimizer=network_params["optimizer"], loss='binary_crossentropy', metrics=["accuracy"])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=0, patience=10)

    # Tính class weights
    class_weight_dict = compute_balanced_class_weights(y_train)

    history = model.fit(X_train, y_train, epochs=network_params["nb_epochs"],
                        batch_size=network_params["nb_batch"], validation_split=0.2,
                        verbose=0, callbacks=[es], class_weight=class_weight_dict)

    validation_loss = np.amin(history.history['val_loss'])
    entry = Utils.predict_lstm(model, X_train, y_train)
    entry['validation_loss'] = validation_loss

    model_path = os.path.join(MODEL_DIR, f"convlstm_{network_params['nb_filters']}_{network_params['nb_layers']}.keras")
    Utils.save_model(model, model_path)
    logger.info(f"Model saved: {model_path}")

    return {'validation_loss': validation_loss, 'model': model, 'entry': entry, 'scaler': scaler}

# Các hàm còn lại (train_convlstm_with_hyperopt, convert_from_PSO, fn_convlstm_pso, ConvLSTMWorker, evaluate_tuner)
# giữ nguyên, chỉ cần cập nhật để sử dụng scaler nếu cần

def train_convlstm_with_hyperopt(network_params):
    if 'data' not in globals():
        raise ValueError("Global 'data' not set. Ensure evaluate_tuner sets it correctly.")
    res = construct_convlstm_model(network_params, {'train_set': globals()['data']})
    return {'loss': res['validation_loss'], 'status': STATUS_OK}

def convert_from_PSO(network_params):
    for key in network_params:
        if key == 'optimizer':
            network_params[key] = 'adam' if int(network_params[key]) == 1 else 'rmsprop'
        elif key == 'nb_layers':
            network_params[key] = int(network_params[key])
    return network_params

def fn_convlstm_pso(drop_proba=0.01, nb_filters=16, nb_epochs=2, nb_batch=4, nb_layers=1, optimizer=1, time_step=30):
    optimizer = 'adam' if int(optimizer) == 1 else 'rmsprop'
    network_params = {
        'nb_filters': int(nb_filters),
        'nb_layers': int(nb_layers),
        'optimizer': optimizer,
        'time_step': int(time_step),
        'nb_epochs': int(nb_epochs),
        'nb_batch': int(nb_batch),
        'drop_proba': drop_proba
    }
    if 'data' not in globals():
        raise ValueError("Global 'data' not set.")
    res = construct_convlstm_model(network_params, {'train_set': globals()['data']})
    return 1 - float(res["validation_loss"])

class ConvLSTMWorker(Worker):
    def __init__(self, train_set, **kwargs):
        super().__init__(**kwargs)
        self.train_set = train_set

    def compute(self, config, budget, **kwargs):
        res = construct_convlstm_model(config, {'train_set': self.train_set})
        return {'loss': res['validation_loss'], 'info': {}}

def evaluate_tuner(tuner_option, train_set):
    global data
    data = train_set

    all_possible_params = {
        'drop_proba': list(np.linspace(0.1, 0.3, 20)),
        'nb_filters': [8, 16, 32],
        'nb_epochs': [2, 3, 4],
        'nb_batch': [16, 32, 64, 128],
        'nb_layers': [1, 2],
        'optimizer': ['adam', 'rmsprop'],
        'time_step': [5, 10, 15]
    }

    start = timer()

    if tuner_option == "tpe":
        param_space = {k: hp.choice(k, v) for k, v in all_possible_params.items()}
        trials = Trials()
        best = fmin(train_convlstm_with_hyperopt, param_space, algo=tpe.suggest, max_evals=Utils.CONFIG['MAX_EVAL'],
                    trials=trials)
        best_params = {k: all_possible_params[k][v] for k, v in best.items()}
        res = construct_convlstm_model(best_params, {'train_set': data})
        entry_train, best_model, scaler = res["entry"], res["model"], res["scaler"]

    elif tuner_option == "ga":
        ga_params = {
            "population_size": 5,
            "max_generations": 5,
            "retain": 0.7,
            "random_select": 0.1,
            "mutate_chance": 0.1
        }
        ga_runner = GARunner(ga_params)
        best_params, best_model, entry_train = ga_runner.generate(
            all_possible_params=all_possible_params,
            fn_train=construct_convlstm_model,
            params_fn={'train_set': data}
        )
        res = construct_convlstm_model(best_params, {'train_set': data})
        entry_train, best_model, scaler = res["entry"], res["model"], res["scaler"]

    elif tuner_option == "pso":
        params_PSO = {
            'nb_filters': [all_possible_params['nb_filters'][0], all_possible_params['nb_filters'][-1]],
            'nb_layers': [all_possible_params['nb_layers'][0], all_possible_params['nb_layers'][-1]],
            'optimizer': [1, 2],
            'time_step': [all_possible_params['time_step'][0], all_possible_params['time_step'][-1]],
            'nb_epochs': [all_possible_params['nb_epochs'][0], all_possible_params['nb_epochs'][-1]],
            'nb_batch': [all_possible_params['nb_batch'][0], all_possible_params['nb_batch'][-1]],
            'drop_proba': [all_possible_params['drop_proba'][0], all_possible_params['drop_proba'][-1]]
        }
        best_params, _, _ = optunity.maximize_structured(fn_convlstm_pso, params_PSO,
                                                         num_evals=Utils.CONFIG['MAX_EVAL'])
        best_params = convert_from_PSO(best_params)
        res = construct_convlstm_model(best_params, {'train_set': data})
        entry_train, best_model, scaler = res["entry"], res["model"], res["scaler"]

    elif tuner_option == "bohb":
        config_space = CS.ConfigurationSpace()
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('nb_filters', lower=8, upper=32))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('nb_layers', lower=1, upper=2))
        config_space.add_hyperparameter(CS.CategoricalHyperparameter('optimizer', choices=['adam', 'rmsprop']))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('time_step', lower=5, upper=15))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('nb_epochs', lower=2, upper=4))
        config_space.add_hyperparameter(CS.UniformIntegerHyperparameter('nb_batch', lower=16, upper=128))
        config_space.add_hyperparameter(CS.UniformFloatHyperparameter('drop_proba', lower=0.1, upper=0.3))

        import hpbandster.core.nameserver as hpns
        NS = hpns.NameServer(run_id="ConvLSTM", host='127.0.0.1', port=None)
        NS.start()
        w = ConvLSTMWorker(train_set=data, nameserver='127.0.0.1', run_id="ConvLSTM")
        w.run(background=True)
        bohb = BOHB(configspace=config_space, run_id="ConvLSTM", nameserver='127.0.0.1', min_budget=1,
                    max_budget=Utils.CONFIG['NBR_SOL'])
        res = bohb.run(n_iterations=Utils.CONFIG['NBR_GEN'])
        best = res.get_incumbent_id()
        best_params = res.get_id2config_mapping()[best]['config']
        res = construct_convlstm_model(best_params, {'train_set': data})
        entry_train, best_model, scaler = res["entry"], res["model"], res["scaler"]
        bohb.shutdown(shutdown_workers=True)
        NS.shutdown()

    elif tuner_option == "rs":
        param_space = {k: hp.choice(k, v) for k, v in all_possible_params.items()}
        trials = Trials()
        best = fmin(train_convlstm_with_hyperopt, param_space, algo=rand.suggest, max_evals=Utils.CONFIG['MAX_EVAL'],
                    trials=trials)
        best_params = {k: all_possible_params[k][v] for k, v in best.items()}
        res = construct_convlstm_model(best_params, {'train_set': data})
        entry_train, best_model, scaler = res["entry"], res["model"], res["scaler"]

    elif tuner_option == "default":
        best_params = {
            'nb_filters': 16, 'nb_layers': 1, 'optimizer': 'adam', 'time_step': 10,
            'nb_epochs': 2, 'nb_batch': 64, 'drop_proba': 0.1
        }
        res = construct_convlstm_model(best_params, {'train_set': data})
        entry_train, best_model, scaler = res["entry"], res["model"], res["scaler"]

    end = timer()
    entry_train.update({"time": end - start, "params": best_params, "model": best_model})

    if best_model is None:
        raise ValueError("No valid model was trained. Check the GA optimization process for errors.")

    best_model_path = os.path.join(MODEL_DIR, "best_convlstm_model.keras")
    Utils.save_model(best_model, best_model_path)
    logger.info(f"Best model saved at: {best_model_path}")

    return entry_train, scaler

if __name__ == "__main__":

    # Tải dữ liệu
    dataset = get_dataset("ansible_ansible.csv")

    logger.info("Dataset Info:")
    logger.info(dataset.info())
    logger.info("\nDataset Head:")
    logger.info(dataset.head())

    # Chia dữ liệu
    train_sets, test_sets = split_train_test(dataset)

    # Huấn luyện và đánh giá
    entry_train_ga, scaler = evaluate_tuner("ga", train_sets[0])
    X, y = preprocess_for_convlstm_test(train_sets[0], test_sets[0], entry_train_ga["params"]["time_step"], scaler)
    entry_test = Utils.predict_lstm(entry_train_ga["model"], X, y)

    logger.info("Training Results (GA):")
    logger.info(entry_train_ga)
    logger.info("\nTest Results:")
    logger.info(entry_test)