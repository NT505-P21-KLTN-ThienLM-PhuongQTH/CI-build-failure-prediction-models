import os
import pandas as pd
from LSTM_Tuner import evaluate_tuner, test_preprocess
import Utils

COLUMNS_RES = ["proj", "algo", "iter", "AUC", "accuracy", "F1", "exp"]
MODEL_NAME = "LSTM"

def run_online_validation(tuner="ga", dataset_dir="dataset"):
    all_train_entries = []
    all_test_entries = []

    for file_name in os.listdir(dataset_dir):
        dataset = Utils.getDataset(file_name)
        train_sets, test_sets = Utils.online_validation_folds(dataset)

        for fold_idx, (train_set, test_set) in enumerate(zip(train_sets, test_sets)):
            for iteration in range(1, Utils.nbr_rep + 1):
                print(f"[{file_name} | Fold {fold_idx+1} | Iter {iteration}] Training...")

                entry_train = evaluate_tuner(tuner, train_set)
                entry_train.update({
                    "iter": iteration,
                    "proj": file_name,
                    "exp": fold_idx + 1,
                    "algo": MODEL_NAME
                })
                all_train_entries.append(entry_train)

                best_model = entry_train["model"]
                best_params = entry_train["params"]
                X_test, y_test = test_preprocess(train_set, test_set, best_params["time_step"])

                entry_test = Utils.predict_lstm(best_model, X_test, y_test)
                entry_test.update({
                    "iter": iteration,
                    "proj": file_name,
                    "exp": fold_idx + 1,
                    "algo": MODEL_NAME,
                    "best_params": best_params
                })
                all_test_entries.append(entry_test)

    results_train = pd.DataFrame(all_train_entries)[COLUMNS_RES]
    results_test = pd.DataFrame(all_test_entries)[COLUMNS_RES]

    prefix = f"hybrid{Utils.hybrid_option}_{Utils.with_smote}"
    results_train.to_excel(f"{prefix}_result_train_online_{MODEL_NAME}_{tuner}.xlsx", index=False)
    results_test.to_excel(f"{prefix}_result_test_online_{MODEL_NAME}_{tuner}.xlsx", index=False)


def run_cross_project_validation(tuner="ga", bellwether="jruby.csv", dataset_dir="dataset"):
    all_train_entries = []
    all_test_entries = []

    train_set = Utils.getDataset(bellwether)
    for iteration in range(1, Utils.nbr_rep + 1):
        print(f"[Cross-Project | Iter {iteration}] Training on {bellwether}...")
        entry_train = evaluate_tuner(tuner, train_set)
        best_model = entry_train["model"]
        best_params = entry_train["params"]

        entry_train.update({
            "iter": iteration,
            "proj": bellwether,
            "algo": MODEL_NAME
        })
        all_train_entries.append(entry_train)

        for file_name in os.listdir(dataset_dir):
            if file_name != bellwether:
                test_set = Utils.getDataset(file_name)
                X_test, y_test = test_preprocess(train_set, test_set, best_params["time_step"])
                entry_test = Utils.predict_lstm(best_model, X_test, y_test)
                entry_test.update({
                    "iter": iteration,
                    "proj": file_name,
                    "exp": 1,
                    "algo": MODEL_NAME
                })
                all_test_entries.append(entry_test)

    prefix = f"hybrid{Utils.hybrid_option}_{Utils.with_smote}"
    pd.DataFrame(all_train_entries)[COLUMNS_RES].to_excel(f"{prefix}_train_crossProj_{tuner}_{MODEL_NAME}.xlsx", index=False)
    pd.DataFrame(all_test_entries)[COLUMNS_RES].to_excel(f"{prefix}_test_crossProj_{tuner}_{MODEL_NAME}.xlsx", index=False)
