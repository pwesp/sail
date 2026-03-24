#!/usr/bin/env python3
"""
Train elastic-net logistic regression probes on dense embeddings and sparse SAE features.

For every combination of foundation model, SAE configuration, and classification
target group (modality, sex, age group, slice orientation, anatomical structures),
trains logistic regression on the training split and evaluates on validation and
test splits. Saves per-target ROC-AUC metrics, feature coefficients, and
visualisations of top-activating samples.

Inputs:
  data/total_seg_{model}_encodings/   dense embeddings
  results/total_seg_sparse_features/{model}_matryoshka_sae/{config}/  sparse features

Outputs: results/linear_probes/{model}/{config}/

Run with: conda activate sail && python train_linear_probes.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import gc
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, log_loss, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from time import time

# Local imports
from src.dataloading import TotalSegmentatorLatentFeaturesDataset
from src.config_downstream import (
    BASE_MODEL_PATHS,
    RESULTS_BASE_DIR,
    TARGET_GROUPS,
    TOP_K_FEATURES,
    TOP_K_FEATURES_VIZ,
    N_SAMPLES_VIZ,
    MAX_TRAINING_SAMPLES,
    LOGISTIC_MAX_ITER,
    LOGISTIC_C,
    LOGISTIC_L1_RATIO,
)
from src.downstream_clf import (
    log,
    format_time,
    setup_job_logging,
    setup_target_group_logging,
    get_sae_configs,
    load_targets_for_group,
    subsample_training_data,
    compute_aggregate_metrics,
    compute_per_target_metrics,
    extract_feature_importance,
    count_nonzero_coefficients,
    check_target_group_complete,
    check_job_complete,
    save_training_artifacts,
    save_evaluation_metrics,
    plot_top_activating_samples,
)


# ==================================================
# CLI Arguments
# ==================================================

def parse_arguments():
    """Parse command-line arguments for filtering processing."""
    parser = argparse.ArgumentParser(
        description='Train elastic-net logistic regression on dense embeddings and sparse SAE features'
    )
    parser.add_argument(
        '--base-model',
        type=str,
        default=None,
        choices=['biomedparse', 'biomedparse_random', 'dinov3'],
        help='Filter to single base model (enables parallel processing by base model)'
    )
    parser.add_argument(
        '--sae-config',
        type=str,
        default=None,
        help='Filter to single SAE config, e.g., D_64_256_1024_4096_K_10_20_40_80'
    )
    parser.add_argument(
        '--target-group',
        type=str,
        default=None,
        help='Filter to single target group, e.g., modality'
    )
    return parser.parse_args()



def main():
    """Main processing workflow."""

    # Get CLI arguments
    args = parse_arguments()

    # Set hyperparameters
    C = LOGISTIC_C
    L1_RATIO = LOGISTIC_L1_RATIO

    # ==================================================
    # Select base model(s) to process
    # ==================================================
    log("Base models available:")
    for base_model_name in BASE_MODEL_PATHS.keys():
        log(f"  {base_model_name}")

    if args.base_model:
        base_models_to_process = {
            args.base_model: BASE_MODEL_PATHS[args.base_model]}
        log(f"Filtering to base model: {args.base_model}")
    else:
        base_models_to_process = BASE_MODEL_PATHS
        log(f"Processing all base models")

    based_models_to_process = list(base_models_to_process.items())
    n_base_models_to_process = len(based_models_to_process)

    for base_model_idx, (base_model_name, base_model_config) in enumerate(
            based_models_to_process, 1):

            log("=" * 80)
            log(f"BASE MODEL {base_model_idx}/{n_base_models_to_process}: {base_model_name.upper()}")
            log("=" * 80)

            # ==================================================
            # Load original dense embeddings (once per base model)
            # ==================================================

            # Extract numpy arrays and free torch tensors after loading to save memory

            log("Loading dense embeddings (one time per base model)...")
            data_load_start = time()

            dense_embedding_dataset_train = TotalSegmentatorLatentFeaturesDataset(
                feature_type='embeddings',
                set_type='split',
                parquet_path=base_model_config['dense_train'],
                shuffle=False,
                additional_metadata="all"
            )
            dense_latent_features_train = dense_embedding_dataset_train.features.numpy() # type: ignore
            del dense_embedding_dataset_train.features

            dense_embedding_dataset_val = TotalSegmentatorLatentFeaturesDataset(
                feature_type='embeddings',
                set_type='split',
                parquet_path=base_model_config['dense_val'],
                shuffle=False,
                additional_metadata="all"
            )
            dense_latent_features_val = dense_embedding_dataset_val.features.numpy() # type: ignore
            del dense_embedding_dataset_val.features

            dense_embedding_dataset_test = TotalSegmentatorLatentFeaturesDataset(
                feature_type='embeddings',
                set_type='split',
                parquet_path=base_model_config['dense_test'],
                shuffle=False,
                additional_metadata="all"
            )
            dense_latent_features_test = dense_embedding_dataset_test.features.numpy() # type: ignore
            del dense_embedding_dataset_test.features

            gc.collect()

            dense_load_time = time() - data_load_start
            log(f"  Dense embeddings loaded: {format_time(dense_load_time)}")

            # ==================================================
            # Loop over SAE configurations
            # ==================================================

            # Get all SAE configurations for this base model
            log(f"Getting all SAE configurations for current base model {base_model_name:s}...")
            sae_configs = get_sae_configs(base_model_config['sparse_base'])
            log(f"Found {len(sae_configs)} SAE configuration(s) to process")

            # Filter SAE configs based on CLI argument
            if args.sae_config:
                sae_configs = [c for c in sae_configs if c == args.sae_config]
                if len(sae_configs) == 0:
                    raise ValueError(f"WARNING: SAE config '{args.sae_config}' not found for base model '{base_model_name}'")
                log(f"Filtering to SAE config: {args.sae_config}")
            else:
                log(f"Processing all SAE configurations")

            sparse_features_train: np.ndarray | None = None
            sparse_features_val: np.ndarray | None = None
            sparse_features_test: np.ndarray | None = None

            n_sae_configs = len(sae_configs)

            for sae_config_idx, sae_config in enumerate(sae_configs, 1):

                sae_config_start = time()

                log("")
                log("=" * 80)
                log(f"SAE CONFIG {sae_config_idx}/{n_sae_configs}: {sae_config}")
                log("=" * 80)

                # ==================================================
                # Set up job-level logging
                # ==================================================

                job_log_path, log_dir = setup_job_logging(base_model_name, sae_config)
                log(f"Job log: {job_log_path}")
                log(f"Target group logs: {log_dir}/<target_group>.log")

                # Log to both console and job log file
                log("=" * 80, log_file=job_log_path)
                log(f"JOB: {base_model_name} / {sae_config}", log_file=job_log_path)
                log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file=job_log_path)
                log("=" * 80, log_file=job_log_path)

                # ==================================================
                # Check if job is complete (all target groups done)
                # ==================================================

                # Determine which target groups to check (respect CLI filter for debugging)
                if args.target_group:
                    if args.target_group in TARGET_GROUPS:
                        target_groups_to_check = {args.target_group: TARGET_GROUPS[args.target_group]}
                        log(f"Filtering to target group: {args.target_group}", log_file=job_log_path)
                    else:
                        raise ValueError(f"WARNING: Target group '{args.target_group}' not found. Skipping SAE config.")
                else:
                    target_groups_to_check = TARGET_GROUPS
                    log(f"Processing all target groups", log_file=job_log_path)

                # Check completion status
                is_complete, missing_groups = check_job_complete(
                    base_model_name, sae_config, target_groups_to_check, RESULTS_BASE_DIR
                )

                if is_complete:
                    log(f"Job already complete (all {len(target_groups_to_check)} target groups done). Skipping.", log_file=job_log_path)
                    log("=" * 80, log_file=job_log_path)
                    continue

                log(f"Processing {len(missing_groups)}/{len(target_groups_to_check)} target groups: {missing_groups}", log_file=job_log_path)
                log("", log_file=job_log_path)

                # ==================================================
                # Set up paths
                # ==================================================

                sparse_train_path = Path(base_model_config['sparse_base']) / sae_config / 'total_seg_sparse_features_train.parquet'
                sparse_val_path = Path(base_model_config['sparse_base']) / sae_config / 'total_seg_sparse_features_val.parquet'
                sparse_test_path = Path(base_model_config['sparse_base']) / sae_config / 'total_seg_sparse_features_test.parquet'
                results_dir = Path(RESULTS_BASE_DIR) / base_model_name / sae_config
                results_dir.mkdir(parents=True, exist_ok=True)

                # ==================================================
                # Load sparse features
                # ==================================================

                # Free previous sparse features before loading new ones
                if sparse_features_train is not None:
                    del sparse_features_train, sparse_features_val, sparse_features_test
                    gc.collect()

                log("Loading sparse features...")
                sparse_load_start = time()

                sparse_feature_dataset_train = TotalSegmentatorLatentFeaturesDataset(
                    feature_type='sparse_features',
                    set_type='split',
                    parquet_path=str(sparse_train_path),
                    shuffle=False,
                    additional_metadata=None
                )
                sparse_feature_dataset_val = TotalSegmentatorLatentFeaturesDataset(
                    feature_type='sparse_features',
                    set_type='split',
                    parquet_path=str(sparse_val_path),
                    shuffle=False,
                    additional_metadata=None
                )
                sparse_feature_dataset_test = TotalSegmentatorLatentFeaturesDataset(
                    feature_type='sparse_features',
                    set_type='split',
                    parquet_path=str(sparse_test_path),
                    shuffle=False,
                    additional_metadata=None
                )

                # Extract numpy arrays
                sparse_features_train = sparse_feature_dataset_train.features.numpy() # type: ignore
                sparse_features_val = sparse_feature_dataset_val.features.numpy() # type: ignore
                sparse_features_test = sparse_feature_dataset_test.features.numpy() # type: ignore

                del sparse_feature_dataset_train, sparse_feature_dataset_val, sparse_feature_dataset_test
                gc.collect()

                sparse_load_time = time() - sparse_load_start
                log(f"  Sparse features loaded: {format_time(sparse_load_time)} (shape: {sparse_features_train.shape})", log_file=job_log_path)

                # ==================================================
                # Loop over target groups
                # ==================================================

                # Process only missing target groups (already filtered by job-level check)
                target_groups_to_process = {name: target_groups_to_check[name] for name in missing_groups}
                target_groups_list = list(target_groups_to_process.items())
                total_target_groups = len(target_groups_list)

                for target_group_idx, (target_group_name, target_group_config) in enumerate(target_groups_list, 1):

                    target_group_start = time()

                    # ==================================================
                    # Set up target-group-level logging
                    # ==================================================

                    target_log_path = setup_target_group_logging(log_dir, target_group_name)

                    # Write to log
                    log("=" * 80, log_file=job_log_path, log_file2=target_log_path)
                    log(f"TARGET GROUP: {target_group_name}", log_file=job_log_path, log_file2=target_log_path)
                    log(f"Type: {target_group_config['type']}", log_file=job_log_path, log_file2=target_log_path)
                    log(f"Targets: {target_group_config['targets']}", log_file=job_log_path, log_file2=target_log_path)
                    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}", log_file=job_log_path, log_file2=target_log_path)
                    log("=" * 80, log_file=job_log_path, log_file2=target_log_path)

                    # Build labels for each split (and apply filtering for multiclass)
                    y_train = load_targets_for_group(target_group_config, dense_embedding_dataset_train)
                    y_val = load_targets_for_group(target_group_config, dense_embedding_dataset_val)
                    y_test = load_targets_for_group(target_group_config, dense_embedding_dataset_test)

                    # ==================================================
                    # Subsample training data
                    # ==================================================

                    log(f"Subsampling training data to {MAX_TRAINING_SAMPLES:d} samples...", log_file=job_log_path, log_file2=target_log_path)
                    X_dense_train_subsampled, y_train_subsampled = subsample_training_data(
                        dense_latent_features_train, y_train, max_samples=MAX_TRAINING_SAMPLES
                    )
                    X_sparse_train_subsampled, _ = subsample_training_data(
                        sparse_features_train, y_train, max_samples=MAX_TRAINING_SAMPLES
                    )
                    log(f"  Training samples: {len(y_train_subsampled)}", log_file=job_log_path, log_file2=target_log_path)

                    # ==================================================
                    # Train logistic regression models
                    # ==================================================

                    log("", log_file=job_log_path, log_file2=target_log_path)
                    log("Setting up logistic regression models...", log_file=job_log_path, log_file2=target_log_path)
                    log(f"Using hyperparameters: C={C}, l1_ratio={L1_RATIO}", log_file=job_log_path, log_file2=target_log_path)

                    # Determine if multilabel
                    is_multilabel_task = (y_train_subsampled.ndim == 2)

                    # Configure dense model
                    if L1_RATIO == 0.0:
                        base_dense = LogisticRegression(
                            penalty='l2', solver='lbfgs', C=C,
                            class_weight='balanced', max_iter=LOGISTIC_MAX_ITER, random_state=42
                        )
                    elif L1_RATIO == 1.0:
                        base_dense = LogisticRegression(
                            penalty='l1', solver='saga', C=C,
                            class_weight='balanced', max_iter=LOGISTIC_MAX_ITER, random_state=42
                        )
                    else:
                        base_dense = LogisticRegression(
                            penalty='elasticnet', solver='saga', C=C,
                            l1_ratio=L1_RATIO, class_weight='balanced',
                            max_iter=LOGISTIC_MAX_ITER, random_state=42
                        )

                    # Configure sparse model
                    if L1_RATIO == 0.0:
                        base_sparse = LogisticRegression(
                            penalty='l2', solver='lbfgs', C=C,
                            class_weight='balanced', max_iter=LOGISTIC_MAX_ITER, random_state=42
                        )
                    elif L1_RATIO == 1.0:
                        base_sparse = LogisticRegression(
                            penalty='l1', solver='saga', C=C,
                            class_weight='balanced', max_iter=LOGISTIC_MAX_ITER, random_state=42
                        )
                    else:
                        base_sparse = LogisticRegression(
                            penalty='elasticnet', solver='saga', C=C,
                            l1_ratio=L1_RATIO, class_weight='balanced',
                            max_iter=LOGISTIC_MAX_ITER, random_state=42
                        )

                    # Wrap with MultiOutputClassifier for multilabel tasks
                    if is_multilabel_task:
                        logistic_dense = MultiOutputClassifier(base_dense)
                        logistic_sparse = MultiOutputClassifier(base_sparse)
                    else:
                        logistic_dense = base_dense
                        logistic_sparse = base_sparse

                    # Train models
                    log("", log_file=job_log_path, log_file2=target_log_path)

                    log("Training model on dense features...", log_file=job_log_path, log_file2=target_log_path)
                    dense_train_start = time()
                    logistic_dense.fit(X_dense_train_subsampled, y_train_subsampled)
                    dense_train_time = time() - dense_train_start
                    n_nonzero_dense = count_nonzero_coefficients(logistic_dense)
                    n_features_dense = X_dense_train_subsampled.shape[1]
                    log(f"  Dense model trained: {format_time(dense_train_time)}, n_nonzero={n_nonzero_dense}/{n_features_dense}",
                        log_file=job_log_path, log_file2=target_log_path)

                    log("Training model on sparse features...", log_file=job_log_path, log_file2=target_log_path)
                    sparse_train_start = time()
                    logistic_sparse.fit(X_sparse_train_subsampled, y_train_subsampled)
                    sparse_train_time = time() - sparse_train_start
                    n_nonzero_sparse = count_nonzero_coefficients(logistic_sparse)
                    n_features_sparse = X_sparse_train_subsampled.shape[1]
                    log(f"  Sparse model trained: {format_time(sparse_train_time)}, n_nonzero={n_nonzero_sparse}/{n_features_sparse}",
                        log_file=job_log_path, log_file2=target_log_path)

                    # ==================================================
                    # Evaluate logistic regression models (unified loop)
                    # ==================================================

                    # Compute predictions and metrics for all splits ONCE
                    all_split_results = {}
                    train_metrics_dense: dict = {}
                    train_metrics_sparse: dict = {}

                    for split_name, X_dense_eval, X_sparse_eval, y_eval in [
                        ("train", X_dense_train_subsampled, X_sparse_train_subsampled, y_train_subsampled),
                        ("valid", dense_latent_features_val, sparse_features_val, y_val),
                        ("test", dense_latent_features_test, sparse_features_test, y_test),
                    ]:
                        # Filter out unlabeled samples for multiclass
                        if y_eval.ndim == 1:
                            eval_mask = y_eval >= 0
                        else:
                            eval_mask = slice(None)

                        y_eval_filtered = y_eval[eval_mask] if isinstance(eval_mask, np.ndarray) else y_eval

                        # Compute predictions
                        y_pred_dense = logistic_dense.predict(X_dense_eval[eval_mask])
                        y_proba_dense = logistic_dense.predict_proba(X_dense_eval[eval_mask])

                        y_pred_sparse = logistic_sparse.predict(X_sparse_eval[eval_mask])
                        y_proba_sparse = logistic_sparse.predict_proba(X_sparse_eval[eval_mask])

                        # Compute per-target metrics
                        per_target_metrics_dense = compute_per_target_metrics(
                            y_eval_filtered, y_pred_dense, y_proba_dense, target_group_config
                        )
                        per_target_metrics_sparse = compute_per_target_metrics(
                            y_eval_filtered, y_pred_sparse, y_proba_sparse, target_group_config
                        )

                        # Compute aggregate metrics for terminal display
                        aggregate_metrics_dense = compute_aggregate_metrics(
                            y_eval_filtered, y_pred_dense, y_proba_dense, target_group_config
                        )
                        aggregate_metrics_sparse = compute_aggregate_metrics(
                            y_eval_filtered, y_pred_sparse, y_proba_sparse, target_group_config
                        )

                        # Store results for this split
                        all_split_results[split_name] = {
                            'dense': {
                                'per_target': per_target_metrics_dense,
                                'aggregate': aggregate_metrics_dense
                            },
                            'sparse': {
                                'per_target': per_target_metrics_sparse,
                                'aggregate': aggregate_metrics_sparse
                            }
                        }

                        # Store train metrics for CSV files
                        if split_name == "train":
                            train_metrics_dense = per_target_metrics_dense
                            train_metrics_sparse = per_target_metrics_sparse

                    # Print all results to terminal
                    log("", log_file=job_log_path, log_file2=target_log_path)
                    log("Evaluating models...", log_file=job_log_path, log_file2=target_log_path)

                    for split_name in ["train", "valid", "test"]:
                        results = all_split_results[split_name]
                        agg_dense = results['dense']['aggregate']
                        agg_sparse = results['sparse']['aggregate']

                        log(f"  --- {split_name.capitalize()} ---", log_file=job_log_path, log_file2=target_log_path)

                        if target_group_config['type'] == 'multilabel':
                            log(f"    Dense  - F1(micro): {agg_dense['f1']:.3f}, Log-Loss: {agg_dense['log_loss']:.3f}, ROC-AUC(micro): {agg_dense['roc_auc']:.3f}",
                                log_file=job_log_path, log_file2=target_log_path)
                            log(f"    Sparse - F1(micro): {agg_sparse['f1']:.3f}, Log-Loss: {agg_sparse['log_loss']:.3f}, ROC-AUC(micro): {agg_sparse['roc_auc']:.3f}",
                                log_file=job_log_path, log_file2=target_log_path)
                        elif target_group_config['type'] == 'multiclass':
                            log(f"    Dense  - F1(weighted): {agg_dense['f1']:.3f}, Log-Loss: {agg_dense['log_loss']:.3f}, ROC-AUC(ovr): {agg_dense['roc_auc']:.3f}",
                                log_file=job_log_path, log_file2=target_log_path)
                            log(f"    Sparse - F1(weighted): {agg_sparse['f1']:.3f}, Log-Loss: {agg_sparse['log_loss']:.3f}, ROC-AUC(ovr): {agg_sparse['roc_auc']:.3f}",
                                log_file=job_log_path, log_file2=target_log_path)
                        else:  # binary
                            log(f"    Dense  - F1: {agg_dense['f1']:.3f}, Log-Loss: {agg_dense['log_loss']:.3f}, ROC-AUC: {agg_dense['roc_auc']:.3f}",
                                log_file=job_log_path, log_file2=target_log_path)
                            log(f"    Sparse - F1: {agg_sparse['f1']:.3f}, Log-Loss: {agg_sparse['log_loss']:.3f}, ROC-AUC: {agg_sparse['roc_auc']:.3f}",
                                log_file=job_log_path, log_file2=target_log_path)

                    # Store validation F1 for summary
                    val_f1_dense = all_split_results['valid']['dense']['aggregate']['f1']
                    val_f1_sparse = all_split_results['valid']['sparse']['aggregate']['f1']

                    # ==================================================
                    # Extract feature importance and save results
                    # ==================================================

                    log("", log_file=job_log_path, log_file2=target_log_path)
                    log("Extracting feature importance and saving results...", log_file=job_log_path, log_file2=target_log_path)

                    # Extract coefficients for both models
                    coef_dense_df = extract_feature_importance(logistic_dense, top_k=TOP_K_FEATURES)
                    coef_sparse_df = extract_feature_importance(logistic_sparse, top_k=TOP_K_FEATURES)

                    # Prepare coefficients dictionaries for each target
                    coefficients_dense = {}
                    coefficients_sparse = {}

                    for target in target_group_config['targets']:
                        coefficients_dense[target] = coef_dense_df
                        coefficients_sparse[target] = coef_sparse_df

                    # Prepare hyperparameter results DataFrame
                    # Both dense and sparse use fixed hyperparameters (no grid search)
                    grid_results_df = pd.DataFrame([
                        {'feature_type': 'dense', 'C': C, 'l1_ratio': L1_RATIO,
                         'n_nonzero': n_nonzero_dense, 'val_f1': val_f1_dense,
                         'method': 'fixed'},
                        {'feature_type': 'sparse', 'C': C, 'l1_ratio': L1_RATIO,
                         'n_nonzero': n_nonzero_sparse, 'val_f1': val_f1_sparse,
                         'method': 'fixed'}
                    ])

                    # Save training artifacts (coefficients, grid search)
                    save_training_artifacts(
                        results_dir=results_dir,
                        target_group=target_group_name,
                        coefficients_dense=coefficients_dense,
                        coefficients_sparse=coefficients_sparse,
                        grid_results=grid_results_df
                    )

                    # Save per-target metrics and visualizations for validation and test splits
                    for split_name in ['valid', 'test']:
                        # Get metadata dataset for this split
                        if split_name == 'valid':
                            metadata_dataset = dense_embedding_dataset_val
                            X_sparse_split = sparse_features_val
                        else:
                            metadata_dataset = dense_embedding_dataset_test
                            X_sparse_split = sparse_features_test

                        # Get pre-computed metrics for this split
                        per_target_metrics_dense = all_split_results[split_name]['dense']['per_target']
                        per_target_metrics_sparse = all_split_results[split_name]['sparse']['per_target']

                        # Save metrics for each target
                        for target in target_group_config['targets']:
                            # Dense model metrics
                            save_evaluation_metrics(
                                results_dir=results_dir,
                                split_name=split_name,
                                target_group=target_group_name,
                                target_name=target,
                                feature_type='dense',
                                model_config={
                                    'C': C,
                                    'l1_ratio': L1_RATIO,
                                    'n_features': n_features_dense,
                                    'n_nonzero': n_nonzero_dense,
                                    'fit_time': dense_train_time
                                },
                                train_metrics=train_metrics_dense.get(target, {'f1': np.nan, 'roc_auc': np.nan, 'balanced_acc': np.nan}),
                                eval_metrics={
                                    'f1': per_target_metrics_dense.get(target, {}).get('f1', np.nan),
                                    'roc_auc': per_target_metrics_dense.get(target, {}).get('roc_auc', np.nan),
                                    'balanced_acc': per_target_metrics_dense.get(target, {}).get('balanced_acc', np.nan),
                                    'log_loss': np.nan  # Not computed per-target
                                }
                            )

                            # Sparse model metrics
                            save_evaluation_metrics(
                                results_dir=results_dir,
                                split_name=split_name,
                                target_group=target_group_name,
                                target_name=target,
                                feature_type='sparse',
                                model_config={
                                    'C': C,
                                    'l1_ratio': L1_RATIO,
                                    'n_features': n_features_sparse,
                                    'n_nonzero': n_nonzero_sparse,
                                    'fit_time': sparse_train_time
                                },
                                train_metrics=train_metrics_sparse.get(target, {'f1': np.nan, 'roc_auc': np.nan, 'balanced_acc': np.nan}),
                                eval_metrics={
                                    'f1': per_target_metrics_sparse.get(target, {}).get('f1', np.nan),
                                    'roc_auc': per_target_metrics_sparse.get(target, {}).get('roc_auc', np.nan),
                                    'balanced_acc': per_target_metrics_sparse.get(target, {}).get('balanced_acc', np.nan),
                                    'log_loss': np.nan  # Not computed per-target
                                }
                            )

                        # Generate visualizations for sparse features (split-specific)
                        log(f"  Generating visualizations for {split_name} split...", log_file=job_log_path, log_file2=target_log_path)
                        for target in target_group_config['targets']:
                            plot_top_activating_samples(
                                model=logistic_sparse,
                                sparse_features=X_sparse_split,
                                metadata_dataset=metadata_dataset,
                                target_group=target_group_name,
                                target_name=target,
                                results_dir=results_dir / split_name,
                                top_k_features=TOP_K_FEATURES_VIZ,
                                n_samples=N_SAMPLES_VIZ
                            )

                    log("  Results saved successfully", log_file=job_log_path, log_file2=target_log_path)
                    log("", log_file=job_log_path, log_file2=target_log_path)

                    # End of evaluation loop - add target group summary
                    target_group_time = time() - target_group_start
                    sparsity_dense = 100 * (1 - n_nonzero_dense / n_features_dense)
                    sparsity_sparse = 100 * (1 - n_nonzero_sparse / n_features_sparse)

                    log("", log_file=job_log_path, log_file2=target_log_path)
                    log(f"TARGET GROUP '{target_group_name}' completed in {format_time(target_group_time)}",
                        log_file=job_log_path, log_file2=target_log_path)
                    if val_f1_dense is not None and val_f1_sparse is not None:
                        log(f"  Dense:  val_f1={val_f1_dense:.4f}, n_nonzero={n_nonzero_dense}/{n_features_dense} ({sparsity_dense:.1f}% sparse)",
                            log_file=job_log_path, log_file2=target_log_path)
                        log(f"  Sparse: val_f1={val_f1_sparse:.4f}, n_nonzero={n_nonzero_sparse}/{n_features_sparse} ({sparsity_sparse:.1f}% sparse)",
                            log_file=job_log_path, log_file2=target_log_path)

                # End of target group loop - add SAE config summary
                sae_config_time = time() - sae_config_start
                log("")
                log("=" * 80)
                log(f"SAE CONFIG '{sae_config}' completed in {format_time(sae_config_time)}")
                log(f"Processed {total_target_groups}/{total_target_groups} target groups")
                log("=" * 80)

                # End of SAE config loop
                log("")
                log(f"BASE MODEL '{base_model_name}' completed")
                log("=" * 80)

            # End of base model loop
            log("")
            log("=" * 80)
            log("ALL PROCESSING COMPLETE")
            log("=" * 80)

if __name__ == '__main__':
    main()