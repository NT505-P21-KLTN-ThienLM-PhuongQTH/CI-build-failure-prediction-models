import mlflow
import mlflow.keras
from timeit import default_timer as timer
import numpy as np
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout, Input
from keras.callbacks import EarlyStopping
from sklearn.utils.class_weight import compute_class_weight
from src.helpers import Utils
from src.model.stacked_lstm.preprocess import train_preprocess
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "stacked_lstm")
os.makedirs(MODEL_DIR, exist_ok=True)

def construct_lstm_model(network_params, train_set, pretrained_model_path=None, fine_tune=True, experiment_name="LSTM_Experiment"):
    start_time = timer()

    # Construct and train the LSTM model.
    X_train, y_train = train_preprocess(train_set, network_params["time_step"])
    drop = network_params["drop_proba"]

    if pretrained_model_path:
        print(f"Loading pretrained model from {pretrained_model_path}...")
        model = mlflow.keras.load_model(pretrained_model_path)
        if fine_tune:
            print("Fine-tuning the pretrained model...")
            # for layer in model.layers[:-1]:  # Freeze all layers except the last
            #     layer.trainable = False
            num_layers = len(model.layers)
            freeze_until = num_layers // 2  # Đóng băng nửa đầu các tầng
            for layer in model.layers[:freeze_until]:
                layer.trainable = False
            for layer in model.layers[freeze_until:]:
                layer.trainable = True
    else:
        model = Sequential()
        model.add(Input(shape=(X_train.shape[1], X_train.shape[2])))
        model.add(LSTM(units=network_params["nb_units"],
                       return_sequences=(network_params["nb_layers"] > 1)))
        model.add(Dropout(drop))
        for i in range(1, network_params["nb_layers"]):
            is_last = (i == network_params["nb_layers"] - 1)
            model.add(LSTM(units=network_params["nb_units"], return_sequences=not is_last))
            model.add(Dropout(drop))
        model.add(Dense(1, activation='sigmoid'))

    # print(f"Model summary:\n{model.summary()}")
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

    entry, threshold = Utils.predict_lstm(model, X_train, y_train)
    entry['validation_loss'] = validation_loss


    model_path = os.path.join(MODEL_DIR, f"lstm_model_{experiment_name}.keras")
    model.save(model_path)

    end_time = timer()
    training_time = end_time - start_time
    print(f"\nTraining time: {training_time:.2f} seconds")

    # model_path = os.path.join(MODEL_DIR, f"lstm_{network_params['nb_units']}_{network_params['nb_layers']}.keras")
    # model.save(model_path)
    # print(f"Model saved: {model_path}")

    return {'validation_loss': validation_loss, 'model': model, 'entry': entry, 'history': history}