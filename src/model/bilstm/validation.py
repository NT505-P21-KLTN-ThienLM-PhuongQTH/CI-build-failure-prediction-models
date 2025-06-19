# src/model/bilstm/validation.py
import os
import pandas as pd
import mlflow
import mlflow.keras
from src.model.bilstm.model import construct_bilstm_model
from src.model.common.preprocess import test_preprocess, apply_smote, prepare_features
from src.model.bilstm.tuners import evaluate_tuner, CONFIG
from src.helpers import Utils
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
MLRUNS_DIR = os.path.join(PROJECT_ROOT, "mlruns")
MODEL_DIR = os.path.join(PROJECT_ROOT, "models", "stacked_bilstm")
os.makedirs(MODEL_DIR, exist_ok=True)

COLUMNS_RES = ["proj", "algo", "iter", "AUC", "accuracy", "F1", "PR_AUC", "exp"]
MODEL_NAME = "Stacked-BiLSTM"

def has_metric_above_threshold(entry, threshold=0.99):
    return any(entry.get(metric, 0) > threshold for metric in ["accuracy", "AUC", "recall", "precision", "PR_AUC"])

def run_online_validation(tuner="ga", datasets=None):
    if datasets is None:
        raise ValueError("No datasets provided for online validation.")

    all_test_entries = []

    # Main run for Online Validation
    with mlflow.start_run(run_name="Online_Validation_Pipeline", nested=True):
        main_run_id = mlflow.active_run().info.run_id
        print(f"🚀 Started Online Validation Pipeline - Run ID: {main_run_id}")

        # Log pipeline info
        mlflow.log_param("validation_type", "online")
        mlflow.log_param("tuner", tuner)
        mlflow.log_param("total_datasets", len(datasets))

        # Select top 10 datasets
        dataset_sizes = {file_name: len(df) for file_name, df in datasets.items()}
        top_10_pairs = sorted(dataset_sizes.items(), key=lambda x: x[1], reverse=True)[:10]
        top_10_files = [f for f, _ in top_10_pairs]
        datasets = {f: datasets[f] for f in top_10_files}

        # Log dataset selection info
        mlflow.log_param("selected_datasets", top_10_files)
        for i, (dataset_name, size) in enumerate(top_10_pairs):
            mlflow.log_param(f"dataset_{i + 1}_name", dataset_name)
            mlflow.log_param(f"dataset_{i + 1}_size", size)

        print(f"📊 Selected top 10 datasets: {list(datasets.keys())}")

        best_models = {}
        all_project_metrics = []

        # Process each dataset
        for dataset_idx, (file_name, dataset) in enumerate(datasets.items(), 1):
            print(f"\n{'=' * 60}")
            print(f"📁 Processing Dataset {dataset_idx}/{len(datasets)}: {file_name}")
            print(f"{'=' * 60}")

            # Run for each dataset
            with mlflow.start_run(run_name=f"Dataset_{dataset_idx}_{file_name}", nested=True):
                dataset_run_id = mlflow.active_run().info.run_id

                # Log dataset info
                mlflow.log_param("dataset_name", file_name)
                mlflow.log_param("dataset_size", len(dataset))
                mlflow.log_param("dataset_index", dataset_idx)

                best_f1 = float('-inf')
                best_model_info = None
                train_sets, test_sets = Utils.online_validation_folds(dataset)
                balanced_train_sets = []

                mlflow.log_param("num_folds", len(train_sets))
                mlflow.log_param("num_iterations", CONFIG['NBR_REP'])

                fold_metrics = []

                # Process each fold
                for fold_idx, (train_set, test_set) in enumerate(zip(train_sets, test_sets)):
                    print(f"\n📂 Fold {fold_idx + 1}/{len(train_sets)}")

                    # Run for each fold
                    with mlflow.start_run(run_name=f"Fold_{fold_idx + 1}", nested=True):
                        fold_run_id = mlflow.active_run().info.run_id

                        # Log fold info
                        mlflow.log_param("fold_index", fold_idx + 1)
                        mlflow.log_param("train_size", len(train_set))
                        mlflow.log_param("test_size", len(test_set))

                        iteration_metrics = []

                        previous_model_path = None
                        # Process each iteration
                        for iteration in range(1, CONFIG['NBR_REP'] + 1):
                            print(f"\n🔄 [Dataset: {file_name} | Fold: {fold_idx + 1} | Iter: {iteration}] Training...")

                            # Run for each iteration
                            with mlflow.start_run(run_name=f"Iteration_{iteration}", nested=True):
                                iter_run_id = mlflow.active_run().info.run_id

                                # Log iteration info
                                mlflow.log_param("iteration", iteration)
                                mlflow.log_param("dataset_name", file_name)
                                mlflow.log_param("fold_index", fold_idx + 1)

                                try:
                                    # Training phase
                                    print(f"  🏋️ Training model...")
                                    entry_train = evaluate_tuner(tuner, train_set, pretrained_model_path=previous_model_path)
                                    current_model = entry_train["model"]
                                    current_params = entry_train["params"]
                                    current_history = entry_train.get("history")
                                    current_metrics = entry_train["entry"]

                                    # Log training parameters
                                    train_log_params, train_log_metrics = Utils.build_log_entries(
                                        params=current_params,
                                        metrics=current_metrics,
                                        prefix="train_"
                                    )
                                    mlflow.log_params(train_log_params)
                                    mlflow.log_metrics(train_log_metrics)

                                    # Log training history if available
                                    if current_history and hasattr(current_history, 'history'):
                                        for epoch, loss in enumerate(current_history.history.get('loss', [])):
                                            mlflow.log_metric("train_loss", loss, step=epoch)
                                        for epoch, val_loss in enumerate(current_history.history.get('val_loss', [])):
                                            mlflow.log_metric("train_val_loss", val_loss, step=epoch)

                                    print(f"  🔄 Applying SMOTE...")
                                    X_train, y = prepare_features(train_set, target_column='build_failed')
                                    feature_cols = X_train.columns.tolist()
                                    X_smote, y_smote = apply_smote(X_train.values, y.values)
                                    balanced_df = pd.DataFrame(X_smote, columns=feature_cols)
                                    balanced_df['build_failed'] = y_smote
                                    balanced_train_sets.append(balanced_df)

                                    # Log SMOTE info
                                    mlflow.log_param("original_train_size", len(X_train))
                                    mlflow.log_param("smote_train_size", len(X_smote))
                                    mlflow.log_param("smote_positive_ratio", y_smote.sum() / len(y_smote))

                                    # Testing phase
                                    print(f"  🧪 Testing model...")
                                    X_test, y_test = test_preprocess(train_set, test_set, current_params["time_step"])
                                    entry_test, threshold = Utils.predict_lstm(current_model, X_test, y_test)

                                    # Prepare test entry
                                    entry_test.update({
                                        "iter": iteration,
                                        "proj": file_name,
                                        "exp": fold_idx + 1,
                                        "algo": MODEL_NAME,
                                        "threshold": threshold
                                    })

                                    # Log test parameters and metrics
                                    test_log_params = {
                                        "test_threshold": threshold,
                                        "test_input_shape": str(X_test.shape),
                                        "test_positive_samples": int(y_test.sum()),
                                        "test_negative_samples": int(len(y_test) - y_test.sum())
                                    }
                                    mlflow.log_params(test_log_params)

                                    test_log_params, test_log_metrics = Utils.build_log_entries(
                                        params={"threshold": threshold},
                                        metrics=entry_test,
                                        prefix="test_"
                                    )
                                    mlflow.log_metrics(test_log_metrics)

                                    f1 = entry_test["F1"]

                                    # Check for suspicious metrics
                                    if any(metric > 0.99 for metric in
                                           [entry_test.get("F1", 0), entry_test.get("AUC", 0),
                                            entry_test.get("accuracy", 0), entry_test.get("recall", 0),
                                            entry_test.get("precision", 0), entry_test.get("PR_AUC", 0)]):
                                        print("  ⚠️ Skipping model due to suspiciously high metrics (>0.99)")
                                        mlflow.log_param("status", "skipped_high_metrics")
                                        continue

                                    mlflow.log_param("status", "completed")

                                    # Update best model if current is better
                                    if (not has_metric_above_threshold(entry_test)) and (f1 > best_f1):
                                        best_f1 = f1
                                        best_model_info = {
                                            "model": current_model,
                                            "params": {**current_params, "project": file_name, "fold": fold_idx + 1,
                                                       "iteration": iteration, "threshold": threshold},
                                            "history": current_history,
                                            "X_test": X_test,
                                            "y_test": y_test,
                                            "entry_test": {**entry_test, "F1": f1},
                                            "run_id": iter_run_id
                                        }
                                        mlflow.log_param("is_best_model", True)
                                        print(f"  🎯 New best model! F1: {f1:.4f}, PR-AUC: {entry_test['PR_AUC']:.4f}")
                                        mlflow.keras.log_model(current_model, artifact_path=f"model_fold_{fold_idx}")
                                        previous_model_path = f"runs:/{iter_run_id}/model_fold_{fold_idx}"
                                    else:
                                        mlflow.log_param("is_best_model", False)

                                    # Store metrics
                                    iteration_metrics.append(entry_test)
                                    all_test_entries.append(entry_test)

                                    print(f"  📊 Test metrics: F1={entry_test['F1']:.4f}, AUC={entry_test['AUC']:.4f}, "
                                          f"PR-AUC={entry_test['PR_AUC']:.4f}, Acc={entry_test['accuracy']:.4f}")
                                    mlflow.log_param(f"status_iteration_{iteration}", "completed")
                                except Exception as e:
                                    print(f"  ❌ Error in iteration {iteration}: {str(e)}")
                                    mlflow.log_param("status", "failed")
                                    mlflow.log_param("error", str(e))

                        # Log fold summary
                        if iteration_metrics:
                            fold_avg_f1 = sum(m['F1'] for m in iteration_metrics) / len(iteration_metrics)
                            fold_avg_auc = sum(m['AUC'] for m in iteration_metrics) / len(iteration_metrics)
                            fold_avg_acc = sum(m['accuracy'] for m in iteration_metrics) / len(iteration_metrics)
                            fold_avg_pr_auc = sum(m['PR_AUC'] for m in iteration_metrics) / len(iteration_metrics)

                            mlflow.log_metric("fold_avg_f1", fold_avg_f1)
                            mlflow.log_metric("fold_avg_auc", fold_avg_auc)
                            mlflow.log_metric("fold_avg_accuracy", fold_avg_acc)
                            mlflow.log_metric("fold_avg_pr_auc", fold_avg_pr_auc)

                            fold_metrics.append({
                                'fold': fold_idx + 1,
                                'avg_f1': fold_avg_f1,
                                'avg_auc': fold_avg_auc,
                                'avg_accuracy': fold_avg_acc,
                                'avg_pr_auc': fold_avg_pr_auc
                            })

                # Log dataset summary
                if fold_metrics:
                    dataset_avg_f1 = sum(m['avg_f1'] for m in fold_metrics) / len(fold_metrics)
                    dataset_avg_auc = sum(m['avg_auc'] for m in fold_metrics) / len(fold_metrics)
                    dataset_avg_acc = sum(m['avg_accuracy'] for m in fold_metrics) / len(fold_metrics)
                    dataset_avg_pr_auc = sum(m['avg_pr_auc'] for m in fold_metrics) / len(fold_metrics)

                    mlflow.log_metric("dataset_avg_f1", dataset_avg_f1)
                    mlflow.log_metric("dataset_avg_auc", dataset_avg_auc)
                    mlflow.log_metric("dataset_avg_accuracy", dataset_avg_acc)
                    mlflow.log_metric("dataset_avg_pr_auc", dataset_avg_pr_auc)

                    all_project_metrics.append({
                        'project': file_name,
                        'avg_f1': dataset_avg_f1,
                        'avg_auc': dataset_avg_auc,
                        'avg_accuracy': dataset_avg_acc,
                        'avg_pr_auc': dataset_avg_pr_auc
                    })

                # Log best model for this dataset
                if best_model_info is None:
                    print(f"⚠️ Warning: No valid model found for {file_name}. Skipping model logging...")
                    mlflow.log_param("best_model_status", "not_found")
                else:
                    mlflow.log_param("best_model_status", "found")
                    mlflow.log_param("best_model_f1", best_model_info["entry_test"]["F1"])
                    mlflow.log_param("best_model_pr_auc", best_model_info["entry_test"]["PR_AUC"])
                    mlflow.log_param("best_model_fold", best_model_info["params"]["fold"])
                    mlflow.log_param("best_model_iteration", best_model_info["params"]["iteration"])

                    # Save best model
                    safe_name = file_name.replace('/', '_').replace('\\', '_')
                    model_name = f"best_model_{safe_name}"
                    mlflow.keras.log_model(best_model_info["model"], artifact_path=model_name)

                    best_models[file_name] = best_model_info
                    print(f"✅ Best model for {file_name}: F1={best_model_info['entry_test']['F1']:.4f}, "
                          f"PR-AUC={best_model_info['entry_test']['PR_AUC']:.4f}")

        # Pipeline summary
        print(f"\n{'=' * 80}")
        print("🏆 ONLINE VALIDATION SUMMARY")
        print(f"{'=' * 80}")

        if all_project_metrics:
            # Create summary DataFrame
            test_df = pd.DataFrame(all_test_entries)
            proj_scores = test_df.groupby('proj')[['F1', 'AUC', 'accuracy', 'PR_AUC']].mean()

            # Log overall pipeline metrics
            mlflow.log_metric("pipeline_avg_f1", proj_scores['F1'].mean())
            mlflow.log_metric("pipeline_avg_auc", proj_scores['AUC'].mean())
            mlflow.log_metric("pipeline_avg_accuracy", proj_scores['accuracy'].mean())
            mlflow.log_metric("pipeline_avg_pr_auc", proj_scores['PR_AUC'].mean())
            mlflow.log_metric("pipeline_std_f1", proj_scores['F1'].std())
            mlflow.log_metric("pipeline_std_auc", proj_scores['AUC'].std())
            mlflow.log_metric("pipeline_std_accuracy", proj_scores['accuracy'].std())
            mlflow.log_metric("pipeline_std_pr_auc", proj_scores['PR_AUC'].std())

            print("📊 Average Test Metrics by Project:")
            for idx, (proj, row) in enumerate(proj_scores.iterrows()):
                print(f"  {proj}: F1={row['F1']:.4f}, AUC={row['AUC']:.4f}, PR-AUC={row['PR_AUC']:.4f}, "
                      f"Acc={row['accuracy']:.4f}")
                mlflow.log_metric(f"project_{idx + 1}_f1", row['F1'])
                mlflow.log_metric(f"project_{idx + 1}_auc", row['AUC'])
                mlflow.log_metric(f"project_{idx + 1}_accuracy", row['accuracy'])
                mlflow.log_metric(f"project_{idx + 1}_pr_auc", row['PR_AUC'])

            # Select bellwether
            bellwether = proj_scores['F1'].idxmax()
            bellwether_f1 = proj_scores.loc[bellwether, 'F1']
            bellwether_pr_auc = proj_scores.loc[bellwether, 'PR_AUC']

            mlflow.log_param("bellwether_project", bellwether)
            mlflow.log_metric("bellwether_f1", bellwether_f1)
            mlflow.log_metric("bellwether_pr_auc", bellwether_pr_auc)

            print(f"\n🎯 Selected Bellwether: {bellwether}")
            print(f"   Best F1 Score: {bellwether_f1:.4f}")
            print(f"   Best PR-AUC: {bellwether_pr_auc:.4f}")

            # Log bellwether model
            bellwether_info = best_models[bellwether]
            sanitized_bellwether = bellwether.replace('/', '_').replace('\\', '_')
            bellwether_model_name = f"bellwether_model_{sanitized_bellwether}"
            mlflow.keras.log_model(bellwether_info["model"], artifact_path=bellwether_model_name)
            bellwether_model_uri = f"runs:/{main_run_id}/{bellwether_model_name}"

            print(f"💾 Bellwether model saved: {bellwether_model_uri}")

            return datasets[bellwether], datasets, bellwether_model_uri
        else:
            raise ValueError("No valid models found in online validation")

def run_cross_project_validation(bellwether_dataset, all_datasets, bellwether_model_uri=None, tuner="ga"):
    all_test_entries = []

    with mlflow.start_run(run_name="Cross_Project_Validation_Pipeline", nested=True):
        main_run_id = mlflow.active_run().info.run_id
        print(f"🚀 Started Cross-Project Validation Pipeline - Run ID: {main_run_id}")

        # Log pipeline info
        mlflow.log_param("validation_type", "cross_project")
        mlflow.log_param("tuner", tuner)
        mlflow.log_param("total_test_datasets", len(all_datasets) - 1)
        mlflow.log_param("bellwether_model_uri", bellwether_model_uri or "None")

        # Phase 1: Train on Bellwether
        print(f"\n{'=' * 60}")
        print("🎯 PHASE 1: BELLWETHER TRAINING")
        print(f"{'=' * 60}")

        best_bellwether_info = None
        best_bellwether_f1 = float('-inf')

        with mlflow.start_run(run_name="Phase1_Bellwether_Training", nested=True):
            phase1_run_id = mlflow.active_run().info.run_id
            mlflow.log_param("phase", "bellwether_training")
            mlflow.log_param("bellwether_dataset_size", len(bellwether_dataset))
            mlflow.log_param("num_iterations", CONFIG['NBR_REP'])

            iteration_metrics = []

            for iteration in range(1, CONFIG['NBR_REP'] + 1):
                with mlflow.start_run(run_name=f"Bellwether_Iteration_{iteration}", nested=True):
                    iter_run_id = mlflow.active_run().info.run_id
                    mlflow.log_param("iteration", iteration)
                    mlflow.log_param("training_type", "bellwether")

                    try:
                        print(f"  🏋️ Training bellwether model...")
                        entry_train = evaluate_tuner(tuner, bellwether_dataset, bellwether_model_uri)
                        current_model = entry_train["model"]
                        current_params = entry_train["params"]
                        current_metrics = entry_train["entry"]
                        current_history = entry_train.get("history")
                        current_threshold = entry_train.get("threshold", 0.5)
                        f1 = current_metrics["F1"]

                        train_log_params, train_log_metrics = Utils.build_log_entries(
                            params={**current_params, "project": "bellwether", "iteration": iteration, "threshold": current_threshold},
                            metrics=current_metrics,
                            prefix="bellwether_train_"
                        )
                        mlflow.log_params(train_log_params)
                        mlflow.log_metrics(train_log_metrics)

                        if current_history and hasattr(current_history, 'history'):
                            for epoch, loss in enumerate(current_history.history.get('loss', [])):
                                mlflow.log_metric("bellwether_train_loss", loss, step=epoch)
                            for epoch, val_loss in enumerate(current_history.history.get('val_loss', [])):
                                mlflow.log_metric("bellwether_val_loss", val_loss, step=epoch)

                        if any(metric > 0.99 for metric in [current_metrics.get("F1", 0), current_metrics.get("AUC", 0),
                                                            current_metrics.get("accuracy", 0), current_metrics.get("recall", 0),
                                                            current_metrics.get("precision", 0), current_metrics.get("PR_AUC", 0)]):
                            print("  ⚠️ Skipping bellwether model due to suspiciously high metrics (>0.99)")
                            mlflow.log_param("status", "skipped_high_metrics")
                            continue

                        mlflow.log_param("status", "completed")

                        if (not has_metric_above_threshold(current_metrics)) and (f1 > best_bellwether_f1):
                            best_bellwether_f1 = f1
                            best_bellwether_info = {
                                "model": current_model,
                                "model_name": f"bellwether_model_iter{iteration}",
                                "params": {**current_params, "threshold": current_threshold},
                                "metrics": {**current_metrics, "F1": f1},
                                "iteration": iteration,
                                "history": current_history,
                                "run_id": iter_run_id
                            }
                            mlflow.log_param("is_best_bellwether", True)
                            print(f"  🎯 New best bellwether! F1: {f1:.4f}")
                        else:
                            mlflow.log_param("is_best_bellwether", False)

                        iteration_metrics.append(current_metrics)
                        print(f"  📊 Bellwether metrics: F1={f1:.4f}, AUC={current_metrics['AUC']:.4f}, "
                              f"PR-AUC={current_metrics['PR_AUC']:.4f}")

                    except Exception as e:
                        print(f"  ❌ Error in bellwether iteration {iteration}: {str(e)}")
                        mlflow.log_param("status", "failed")
                        mlflow.log_param("error", str(e))

            if iteration_metrics:
                avg_f1 = sum(m['F1'] for m in iteration_metrics) / len(iteration_metrics)
                avg_auc = sum(m['AUC'] for m in iteration_metrics) / len(iteration_metrics)
                avg_acc = sum(m['accuracy'] for m in iteration_metrics) / len(iteration_metrics)
                avg_pr_auc = sum(m['PR_AUC'] for m in iteration_metrics) / len(iteration_metrics)

                mlflow.log_metric("phase1_avg_f1", avg_f1)
                mlflow.log_metric("phase1_avg_auc", avg_auc)
                mlflow.log_metric("phase1_avg_accuracy", avg_acc)
                mlflow.log_metric("phase1_avg_pr_auc", avg_pr_auc)

                print(f"  📊 Phase 1 Average: F1={avg_f1:.4f}, AUC={avg_auc:.4f}, PR-AUC={avg_pr_auc:.4f}, "
                      f"Acc={avg_acc:.4f}")

            if best_bellwether_info is None:
                raise ValueError("No valid bellwether model found")

            # Save and register best bellwether model
            model_name = f"{best_bellwether_info['model_name']}"
            mlflow.keras.log_model(best_bellwether_info["model"], artifact_path=model_name)
            model_uri = f"runs:/{phase1_run_id}/{model_name}"

            mlflow.log_param("best_bellwether_model_path", model_uri)
            mlflow.log_param("best_bellwether_f1", best_bellwether_info["metrics"]["F1"])
            mlflow.log_param("best_bellwether_iteration", best_bellwether_info["iteration"])

            try:
                registered_model = mlflow.register_model(model_uri, MODEL_NAME)
                mlflow.log_param("registered_model_name", MODEL_NAME)
                mlflow.log_param("registered_model_version", registered_model.version)
                print(f"✅ Bellwether model registered as '{MODEL_NAME}' version {registered_model.version}")
            except Exception as e:
                print(f"⚠️ Bellwether model registration failed: {str(e)}")
                mlflow.log_param("registration_error", str(e))

            print(f"  💾 Best bellwether model saved and registered: {model_uri}")
            print(f"  🏆 Best bellwether F1: {best_bellwether_info['metrics']['F1']:.4f}")

        # Phase 2: Cross-Project Validation
        print(f"\n{'=' * 60}")
        print("🔄 PHASE 2: CROSS-PROJECT VALIDATION")
        print(f"{'=' * 60}")

        with mlflow.start_run(run_name="Phase2_Cross_Project_Validation", nested=True):
            phase2_run_id = mlflow.active_run().info.run_id
            mlflow.log_param("phase", "cross_project_validation")

            test_datasets = {name: dataset for name, dataset in all_datasets.items() if dataset is not bellwether_dataset}
            mlflow.log_param("num_test_projects", len(test_datasets))
            mlflow.log_param("test_project_names", list(test_datasets.keys()))

            project_results = []

            # Load registered bellwether model and threshold
            bellwether_model = mlflow.keras.load_model(model_uri)
            fixed_threshold = best_bellwether_info["params"]["threshold"]
            params = best_bellwether_info["params"]

            for project_idx, (file_name, test_set) in enumerate(test_datasets.items(), 1):
                print(f"\n  🎯 Validating on Project {project_idx}/{len(test_datasets)}: {file_name}")

                with mlflow.start_run(run_name=f"Validate_Project_{file_name}", nested=True):
                    project_run_id = mlflow.active_run().info.run_id
                    mlflow.log_param("test_project", file_name)
                    mlflow.log_param("test_project_size", len(test_set))
                    mlflow.log_param("fixed_threshold", fixed_threshold)

                    try:
                        print(f"    🧪 Validating bellwether model with fixed threshold {fixed_threshold:.4f}...")
                        X_test, y_test = test_preprocess(bellwether_dataset, test_set, params["time_step"])
                        entry_test = Utils.predict_lstm(bellwether_model, X_test, y_test, fixed_threshold)

                        entry_test.update({
                            "iter": 1,
                            "proj": file_name,
                            "exp": 1,
                            "algo": MODEL_NAME,
                            "threshold": fixed_threshold
                        })
                        f1 = entry_test["F1"]

                        test_log_params = {
                            "test_threshold": fixed_threshold,
                            "test_input_shape": str(X_test.shape),
                            "test_positive_samples": int(y_test.sum()),
                            "test_negative_samples": int(len(y_test) - y_test.sum())
                        }
                        mlflow.log_params(test_log_params)

                        test_log_params, test_log_metrics = Utils.build_log_entries(
                            params={"threshold": fixed_threshold, "time_step": params["time_step"],
                                    "input_dim": X_test.shape[2]},
                            metrics=entry_test,
                            prefix="test_"
                        )
                        mlflow.log_metrics(test_log_metrics)

                        if any(metric > 0.99 for metric in [entry_test.get("F1", 0), entry_test.get("AUC", 0),
                                                            entry_test.get("accuracy", 0), entry_test.get("recall", 0),
                                                            entry_test.get("precision", 0), entry_test.get("PR_AUC", 0)]):
                            print(f"    ⚠️ Skipping validation on {file_name} due to suspiciously high metrics (>0.99)")
                            mlflow.log_param("status", "skipped_high_metrics")
                            continue

                        mlflow.log_param("status", "completed")
                        all_test_entries.append(entry_test)

                        project_results.append({
                            'project': file_name,
                            'f1': f1,
                            'auc': entry_test["AUC"],
                            'accuracy': entry_test["accuracy"],
                            'pr_auc': entry_test["PR_AUC"]
                        })

                        print(f"    📊 Validation results: F1={f1:.4f}, AUC={entry_test['AUC']:.4f}, "
                              f"PR-AUC={entry_test['PR_AUC']:.4f}, Acc={entry_test['accuracy']:.4f}")

                    except Exception as e:
                        print(f"    ❌ Error validating on {file_name}: {str(e)}")
                        mlflow.log_param("status", "failed")
                        mlflow.log_param("error", str(e))

            # Log phase 2 summary
            if project_results:
                phase2_avg_f1 = sum(r['f1'] for r in project_results) / len(project_results)
                phase2_avg_auc = sum(r['auc'] for r in project_results) / len(project_results)
                phase2_avg_acc = sum(r['accuracy'] for r in project_results) / len(project_results)
                phase2_avg_pr_auc = sum(r['pr_auc'] for r in project_results) / len(project_results)

                mlflow.log_metric("phase2_avg_f1", phase2_avg_f1)
                mlflow.log_metric("phase2_avg_auc", phase2_avg_auc)
                mlflow.log_metric("phase2_avg_accuracy", phase2_avg_acc)
                mlflow.log_metric("phase2_avg_pr_auc", phase2_avg_pr_auc)

                print(f"\n📊 Phase 2 Overall Average: F1={phase2_avg_f1:.4f}, "
                      f"AUC={phase2_avg_auc:.4f}, PR-AUC={phase2_avg_pr_auc:.4f}, Acc={phase2_avg_acc:.4f}")

        # Pipeline completion summary
        print(f"\n{'=' * 80}")
        print("🎉 CROSS-PROJECT VALIDATION COMPLETED")
        print(f"{'=' * 80}")

        if all_test_entries:
            pipeline_df = pd.DataFrame(all_test_entries)
            overall_avg_f1 = pipeline_df['F1'].mean()
            overall_avg_auc = pipeline_df['AUC'].mean()
            overall_avg_acc = pipeline_df['accuracy'].mean()
            overall_avg_pr_auc = pipeline_df['PR_AUC'].mean()
            overall_std_f1 = pipeline_df['F1'].std()
            overall_std_auc = pipeline_df['AUC'].std()
            overall_std_acc = pipeline_df['accuracy'].std()
            overall_std_pr_auc = pipeline_df['PR_AUC'].std()

            mlflow.log_metric("overall_avg_f1", overall_avg_f1)
            mlflow.log_metric("overall_avg_auc", overall_avg_auc)
            mlflow.log_metric("overall_avg_accuracy", overall_avg_acc)
            mlflow.log_metric("overall_avg_pr_auc", overall_avg_pr_auc)
            mlflow.log_metric("overall_std_f1", overall_std_f1)
            mlflow.log_metric("overall_std_auc", overall_std_auc)
            mlflow.log_metric("overall_std_accuracy", overall_std_acc)
            mlflow.log_metric("overall_std_pr_auc", overall_std_pr_auc)

            mlflow.log_param("total_experiments", len(all_test_entries))

            print(f"📊 Pipeline Results:")
            print(f"   Total Experiments: {len(all_test_entries)}")
            print(f"   Average F1: {overall_avg_f1:.4f} ± {overall_std_f1:.4f}")
            print(f"   Average AUC: {overall_avg_auc:.4f} ± {overall_std_auc:.4f}")
            print(f"   Average PR-AUC: {overall_avg_pr_auc:.4f} ± {overall_std_pr_auc:.4f}")
            print(f"   Average Accuracy: {overall_avg_acc:.4f} ± {overall_std_acc:.4f}")

        print(f"🔗 Main Run ID: {main_run_id}")
        print("✅ All results logged to MLflow successfully!")