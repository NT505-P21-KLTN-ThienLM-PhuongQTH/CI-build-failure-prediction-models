import numpy as np
import mlflow
import mlflow.keras
from hyperopt import hp, fmin, tpe, rand, Trials, STATUS_OK
import optunity
import ConfigSpace as CS
from hpbandster.core.worker import Worker
from hpbandster.optimizers import BOHB
from src.optimization.GA_runner import GARunner
from src.model.stacked_lstm.model import construct_lstm_model
from timeit import default_timer as timer
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "stacked_lstm")
os.makedirs(MODEL_DIR, exist_ok=True)

CONFIG = {'MAX_EVAL': 10, 'NBR_REP': 1}

class LSTMWorker(Worker):
    def __init__(self, train_set, **kwargs):
        super().__init__(**kwargs)
        self.train_set = train_set

    def compute(self, config, budget, **kwargs):
        res = construct_lstm_model(config, self.train_set)
        return {'loss': res['validation_loss'], 'info': {}}

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

def evaluate_tuner(tuner_option, train_set, pretrained_model_path=None):
    # Evaluate the specified tuner.
    global data
    data = train_set

    # mlflow.log_param("tuner", tuner_option)
    # Define explicit parameter space for GA
    all_possible_params = {
        'drop_proba': list(np.linspace(0.01, 0.21, 20)),
        'nb_units': [32, 64],
        'nb_epochs': [4, 5, 6],
        'nb_batch': [4, 8, 16, 32, 64],  # Power of 2
        'nb_layers': [1, 2, 3, 4],
        'optimizer': ['adam', 'rmsprop'],
        'time_step': list(range(30, 61))
    }

    start = timer()
    history = None

    if tuner_option == "ga":
        ga_runner = GARunner()
        best_params, best_model, entry_train, history = ga_runner.generate(
            all_possible_params, construct_lstm_model, data,
            pretrained_model_path=pretrained_model_path
        )

    elif tuner_option == "tpe":
        param_space = {k: hp.choice(k, v) for k, v in all_possible_params.items()}
        trials = Trials()
        best = fmin(train_lstm_with_hyperopt, param_space, algo=tpe.suggest, max_evals=CONFIG.get('MAX_EVAL'),
                    trials=trials)
        best_params = {k: all_possible_params[k][v] for k, v in best.items()}
        res = construct_lstm_model(best_params, data, pretrained_model_path=pretrained_model_path)
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
        best_params, _, _ = optunity.maximize_structured(fn_lstm_pso, params_PSO, num_evals=CONFIG.get('MAX_EVAL'))
        best_params = convert_from_PSO(best_params)
        res = construct_lstm_model(best_params, data, pretrained_model_path=pretrained_model_path)
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
        bohb = BOHB(configspace=config_space, run_id="LSTM", nameserver='127.0.0.1', min_budget=1,
                    max_budget=CONFIG.get('NBR_SOL'))
        res = bohb.run(n_iterations=CONFIG.get('NBR_GEN'))
        best = res.get_incumbent_id()
        best_params = res.get_id2config_mapping()[best]['config']
        res = construct_lstm_model(best_params, data, pretrained_model_path=pretrained_model_path)
        entry_train, best_model = res["entry"], res["model"]
        bohb.shutdown(shutdown_workers=True)
        NS.shutdown()

    elif tuner_option == "rs":
        param_space = {k: hp.choice(k, v) for k, v in all_possible_params.items()}
        trials = Trials()
        best = fmin(train_lstm_with_hyperopt, param_space, algo=rand.suggest,
                    max_evals=CONFIG.get('MAX_EVAL', trials=trials))
        best_params = {k: all_possible_params[k][v] for k, v in best.items()}
        res = construct_lstm_model(best_params, data, pretrained_model_path=pretrained_model_path)
        entry_train, best_model = res["entry"], res["model"]

    elif tuner_option == "default":
        best_params = {
            'nb_units': 64, 'nb_layers': 3, 'optimizer': 'adam', 'time_step': 30,
            'nb_epochs': 10, 'nb_batch': 64, 'drop_proba': 0.1
        }
        res = construct_lstm_model(best_params, data, pretrained_model_path=pretrained_model_path)
        entry_train, best_model = res["entry"], res["model"]

    end = timer()
    entry_train.update({"time": end - start, "params": best_params, "model": best_model})
    # best_model_path = os.path.join(MODEL_DIR, f"best_lstm_{proj_name}_fold{fold_idx}_iter{iter_idx}.keras")
    # best_model.save(best_model_path)
    # print(f"Best model saved at: {best_model_path}")

    # mlflow.log_params(best_params)
    # mlflow.log_metric("F1", entry_train["F1"])
    # mlflow.log_metric("AUC", entry_train["AUC"])
    # mlflow.log_metric("accuracy", entry_train["accuracy"])
    # mlflow.log_metric("training_time", end - start)

    # model_path = os.path.join(MODEL_DIR, f"best_lstm_model_{experiment_name}.keras")
    # best_model.save(model_path)
    # mlflow.log_artifact(model_path, artifact_path="best_lstm_model")

    history = history if tuner_option == "ga" else None
    return {"entry": entry_train, "params": best_params, "model": best_model, "history": history}
