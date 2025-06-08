# src/model/conv_lstm/model.py
import os
import mlflow
import mlflow.keras
from timeit import default_timer as timer
import numpy as np
from keras.models import Sequential
from keras.layers import ConvLSTM2D, Dense, Dropout, Input, Flatten
from keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from src.helpers import Utils
from src.model.common.preprocess import convlstm_train_preprocess, convlstm_test_preprocess, prepare_features

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "conv_lstm")
os.makedirs(MODEL_DIR, exist_ok=True)

def construct_convlstm_model(network_params, train_set, pretrained_model_path=None):
    """
    Construct and train a ConvLSTM model for build failure prediction.

    Args:
        network_params (dict): Hyperparameters for the model, including:
            - time_step: Number of time steps.
            - nb_filters: Number of filters in ConvLSTM layers.
            - nb_layers: Number of ConvLSTM layers.
            - optimizer: Optimizer for training.
            - nb_epochs: Number of training epochs.
            - nb_batch: Batch size.
            - drop_proba: Dropout probability.
        train_set (pd.DataFrame): Training dataset.
        pretrained_model_path (str, optional): Path to a pretrained model for fine-tuning.

    Returns:
        dict: Contains validation loss, trained model, metrics entry, and training history.
    """
    start_time = timer()

    # Preprocess training data for ConvLSTM
    X_train, y_train = convlstm_train_preprocess(train_set, network_params["time_step"])
    drop = network_params["drop_proba"]

    # Define feature groups to calculate input shape
    feature_groups = {
        'project': ['gh_sloc', 'gh_test_cases_per_kloc', 'proj_fail_rate_history', 'proj_fail_rate_recent'],
        'code_change': ['git_diff_src_churn', 'git_diff_test_churn', 'gh_diff_files_added', 'gh_diff_files_deleted',
                        'gh_diff_tests_added', 'gh_diff_src_files', 'gh_diff_doc_files', 'num_files_edited'],
        'team': ['gh_team_size', 'gh_num_issue_comments', 'gh_num_pr_comments', 'gh_num_commit_comments',
                 'same_committer', 'num_distinct_authors', 'comm_avg_experience', 'comm_fail_rate_history',
                 'comm_fail_rate_recent'],
        'time': ['year_of_start', 'month_of_start', 'day_of_start', 'day_week', 'tr_duration', 'prev_build_result'],
        'config_pr': ['gh_is_pr', 'no_config_edited']
    }
    X, _ = prepare_features(train_set, target_column='build_failed')
    feature_cols = X.columns.tolist()
    group_indices = []
    for group, feats in feature_groups.items():
        indices = [feature_cols.index(f) for f in feats if f in feature_cols]
        group_indices.append(indices)
    num_groups = len(group_indices)  # Number of groups (rows)
    max_cols = max(len(indices) for indices in group_indices)  # Max features in any group (cols)

    # Verify input shape
    expected_shape = (network_params["time_step"], num_groups, max_cols, 1)
    if X_train.shape[1:] != expected_shape:
        raise ValueError(f"X_train shape {X_train.shape[1:]} does not match expected shape {expected_shape}")

    # Construct the model
    if pretrained_model_path:
        print(f"Loading pretrained model from {pretrained_model_path}...")
        model = mlflow.keras.load_model(pretrained_model_path)
        print("Fine-tuning the pretrained model...")
        num_layers = len(model.layers)
        freeze_until = num_layers // 2
        for layer in model.layers[:freeze_until]:
            layer.trainable = False
        for layer in model.layers[freeze_until:]:
            layer.trainable = True
    else:
        model = Sequential()
        model.add(Input(shape=(network_params["time_step"], num_groups, max_cols, 1)))
        model.add(ConvLSTM2D(filters=network_params["nb_filters"],
                             kernel_size=(2, 2),
                             padding='same',
                             return_sequences=(network_params["nb_layers"] > 1)))
        model.add(Dropout(drop))
        for i in range(1, network_params["nb_layers"]):
            is_last = (i == network_params["nb_layers"] - 1)
            model.add(ConvLSTM2D(filters=network_params["nb_filters"],
                                 kernel_size=(2, 2),
                                 padding='same',
                                 return_sequences=not is_last))
            model.add(Dropout(drop))
        model.add(Flatten())
        model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer=network_params["optimizer"], loss='binary_crossentropy', metrics=["accuracy"])
    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=10)

    # Compute class weights
    class_weights = compute_class_weight('balanced', classes=np.array([0, 1]), y=y_train)
    class_weight_dict = {0: class_weights[0], 1: class_weights[1]}

    try:
        history = model.fit(X_train, y_train, epochs=network_params["nb_epochs"],
                            batch_size=network_params["nb_batch"], validation_split=0.2,
                            verbose=0, callbacks=[es], class_weight=class_weight_dict)
        validation_loss = np.amin(history.history['val_loss'])
    except Exception as e:
        print(f"Error during model training: {e}")
        return {"validation_loss": float('inf'), "model": None, "entry": {'F1': 0, 'validation_loss': float('inf')}}

    # Evaluate on training data
    entry, threshold = Utils.predict_convlstm(model, X_train, y_train)
    entry['validation_loss'] = validation_loss

    end_time = timer()
    training_time = end_time - start_time
    print(f"\nTraining time: {training_time:.2f} seconds")

    return {'validation_loss': validation_loss, 'model': model, 'entry': entry, 'history': history}