# src/model/common/model_factory.py
import os
from src.model.common.padding_module import PaddingModule
from src.model.common.preprocess import train_preprocess as train_preprocess, prepare_features
from src.model.common.preprocess import test_preprocess
from src.model.common.preprocess import convlstm_test_preprocess
from src.model.common.preprocess import convlstm_train_preprocess
from src.model.lstm.model import construct_lstm_model
from src.model.lstm.tuners import evaluate_tuner as lstm_evaluate_tuner
from src.model.lstm.validation import run_online_validation as lstm_run_online_validation
from src.model.lstm.validation import run_cross_project_validation as lstm_run_cross_project_validation
from src.model.bilstm.model import construct_bilstm_model
from src.model.bilstm.tuners import evaluate_tuner as bilstm_evaluate_tuner
from src.model.bilstm.validation import run_online_validation as bilstm_run_online_validation
from src.model.bilstm.validation import run_cross_project_validation as bilstm_run_cross_project_validation
from src.model.conv_lstm.model import construct_convlstm_model
from src.model.conv_lstm.tuners import evaluate_tuner as convlstm_evaluate_tuner
from src.model.conv_lstm.validation import run_online_validation as convlstm_run_online_validation
from src.model.conv_lstm.validation import run_cross_project_validation as convlstm_run_cross_project_validation

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models")
os.makedirs(MODEL_DIR, exist_ok=True)


class ModelFactory:
    """Factory class to create and manage LSTM, BiLSTM, or ConvLSTM models."""

    @staticmethod
    def get_model_functions(model_name):
        """Return the appropriate model functions based on model_name."""
        if model_name == "Stacked-LSTM":
            return {
                "construct_model": construct_lstm_model,
                "train_preprocess": train_preprocess,
                "test_preprocess": test_preprocess,
                "evaluate_tuner": lstm_evaluate_tuner,
                "run_online_validation": lstm_run_online_validation,
                "run_cross_project_validation": lstm_run_cross_project_validation,
                "model_name": model_name
            }
        elif model_name == "Stacked-BiLSTM":
            return {
                "construct_model": construct_bilstm_model,
                "train_preprocess": train_preprocess,
                "test_preprocess": test_preprocess,
                "evaluate_tuner": bilstm_evaluate_tuner,
                "run_online_validation": bilstm_run_online_validation,
                "run_cross_project_validation": bilstm_run_cross_project_validation,
                "model_name": model_name
            }
        elif model_name == "ConvLSTM":
            return {
                "construct_model": construct_convlstm_model,
                "train_preprocess": convlstm_train_preprocess,
                "test_preprocess": convlstm_test_preprocess,
                "evaluate_tuner": convlstm_evaluate_tuner,
                "run_online_validation": convlstm_run_online_validation,
                "run_cross_project_validation": convlstm_run_cross_project_validation,
                "model_name": model_name
            }
        elif model_name == "Padding":
            return {
                "construct_model": None,
                "train_preprocess": None,
                "test_preprocess": None,
                "evaluate_tuner": None,
                "run_online_validation": None,
                "run_cross_project_validation": None,
                "model_name": model_name,
                "train_padding_module": ModelFactory.train_padding_module
            }
        else:
            raise ValueError(f"Unsupported model_name: {model_name}. Choose 'Stacked-LSTM', 'Stacked-BiLSTM', 'ConvLSTM', or 'Padding'.")

    @staticmethod
    def construct_model(model_name, network_params, train_set, pretrained_model_path=None):
        """Construct the specified model."""
        model_funcs = ModelFactory.get_model_functions(model_name)
        return model_funcs["construct_model"](network_params, train_set, pretrained_model_path)

    @staticmethod
    def preprocess_train(model_name, dataset_train, time_step):
        """Preprocess training data for the specified model."""
        model_funcs = ModelFactory.get_model_functions(model_name)
        return model_funcs["train_preprocess"](dataset_train, time_step)

    @staticmethod
    def preprocess_test(model_name, dataset_train, dataset_test, time_step, short_timestep=None):
        """Preprocess test data for the specified model."""
        model_funcs = ModelFactory.get_model_functions(model_name)
        return model_funcs["test_preprocess"](dataset_train, dataset_test, time_step, short_timestep)

    @staticmethod
    def evaluate_tuner(model_name, tuner_option, train_set, pretrained_model_path=None):
        """Run hyperparameter tuning for the specified model."""
        model_funcs = ModelFactory.get_model_functions(model_name)
        return model_funcs["evaluate_tuner"](tuner_option, train_set, pretrained_model_path)

    @staticmethod
    def run_online_validation(model_name, tuner="ga", datasets=None):
        """Run online validation for the specified model."""
        model_funcs = ModelFactory.get_model_functions(model_name)
        return model_funcs["run_online_validation"](tuner, datasets)

    @staticmethod
    def run_cross_project_validation(model_name, bellwether_dataset, all_datasets,
                                    bellwether_model_uri=None, tuner="ga"):
        """Run cross-project validation for the specified model."""
        model_funcs = ModelFactory.get_model_functions(model_name)
        return model_funcs["run_cross_project_validation"](bellwether_dataset, all_datasets, bellwether_model_uri, tuner)

    @staticmethod
    def get_model_name(model_name):
        """Get the model name for the specified model type."""
        model_funcs = ModelFactory.get_model_functions(model_name)
        return model_funcs["model_name"]

    @staticmethod
    def train_padding_module(model_name, datasets, input_dim=None, time_step=40, epochs=20, batch_size=32, r2_threshold=0.7,
                             max_iterations=5):
        """Train the PaddingModule independently."""
        print("Training PaddingModule...")
        sample_df = next(iter(datasets.values()))
        X, _ = prepare_features(sample_df, target_column='build_failed')
        input_dim = input_dim if input_dim is not None else X.shape[1]
        padding_module = PaddingModule(input_dim=input_dim, time_step=time_step)
        metrics, zeroed_features = padding_module.train(
            datasets,
            epochs=epochs,
            batch_size=batch_size,
            r2_threshold=r2_threshold,
            max_iterations=max_iterations
        )
        # Save the trained model
        padding_module.model.save(model_name)
        print("Evaluation Metrics:", metrics)
        print("Zeroed Features:", zeroed_features)
        return padding_module