"""
Downstream classification utilities for elastic-net logistic regression.

This module contains all helper functions for training and evaluating logistic regression
models on dense embeddings and sparse SAE features for downstream classification tasks.
"""

# stdlib
from datetime import datetime
from pathlib import Path
from time import time

# third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, f1_score, log_loss, roc_auc_score, make_scorer
from sklearn.model_selection import train_test_split, GridSearchCV, PredefinedSplit
from sklearn.multioutput import MultiOutputClassifier

# local
from src.eval import create_binary_target_labels


# ==================================================
# Utility Functions
# ==================================================

def log(msg: str, log_file: Path | None = None, log_file2: Path | None = None) -> None:
    """
    Print message with timestamp to console and optionally to file(s).

    Args:
        msg: Message to log
        log_file: Optional path to first log file. If provided, appends message to file.
        log_file2: Optional path to second log file. If provided, appends message to file.
    """
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    formatted_msg = f"[{timestamp}] {msg}"
    print(formatted_msg, flush=True)  # Force immediate output

    if log_file is not None:
        with open(log_file, 'a') as f:
            f.write(formatted_msg + '\n')
            f.flush()  # Ensure file writes are immediate

    if log_file2 is not None:
        with open(log_file2, 'a') as f:
            f.write(formatted_msg + '\n')
            f.flush()  # Ensure file writes are immediate


def setup_job_logging(
    base_model: str,
    sae_config: str,
    log_base_dir: str = "logs/downstream_parallel"
) -> tuple[Path, Path]:
    """
    Create logging directory structure for a job.

    Structure:
        logs/downstream_parallel/{base_model}/{sae_config}/
            job.log                    # Main job orchestration log
            {target_group}.log         # Per-target-group detailed logs

    Args:
        base_model: Base model name (e.g., 'biomedparse')
        sae_config: SAE configuration name (e.g., 'D_128_512_2048_8192_K_30_60_120_240')
        log_base_dir: Base directory for logs

    Returns:
        job_log_path: Path to main job.log file
        log_dir: Directory for all logs for this job
    """
    log_dir = Path(log_base_dir) / base_model / sae_config
    log_dir.mkdir(parents=True, exist_ok=True)

    job_log_path = log_dir / "job.log"
    return job_log_path, log_dir


def setup_target_group_logging(log_dir: Path, target_group: str) -> Path:
    """
    Create per-target-group log file.

    Args:
        log_dir: Directory where target group logs are stored
        target_group: Target group name (e.g., 'age_group')

    Returns:
        target_log_path: Path to target group log file
    """
    target_log_path = log_dir / f"{target_group}.log"
    return target_log_path


def format_time(seconds: float) -> str:
    """Format seconds into human-readable time string."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        secs = int(seconds % 60)
        return f"{minutes}m {secs}s"
    else:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        return f"{hours}h {minutes}m"


# ==================================================
# Data Handling
# ==================================================

def get_sae_configs(embedding_model_base_path: str) -> list[str]:
    """
    Discover all SAE configurations in the path of an embedding model.

    Expected file structure:
    embedding_model_base_path/
    |-- sae_config_1/
    |   |-- total_seg_sparse_features_train.parquet
    |   |-- total_seg_sparse_features_val.parquet
    |   |-- total_seg_sparse_features_test.parquet
    |-- sae_config_2/
    |   ...
    """
    base_path = Path(embedding_model_base_path)
    if not base_path.exists():
        log(f"Warning: Embedding model base path does not exist: {base_path}")
        return []

    sae_configs = []
    for config_dir in sorted(base_path.iterdir()):
        if config_dir.is_dir():
            train_file = config_dir / 'total_seg_sparse_features_train.parquet'
            val_file = config_dir / 'total_seg_sparse_features_val.parquet'
            test_file = config_dir / 'total_seg_sparse_features_test.parquet'
            if train_file.exists() and val_file.exists() and test_file.exists():
                sae_configs.append(config_dir.name)

    return sae_configs


def load_targets_for_group(
    target_group_config: dict,
    dataset
) -> np.ndarray:
    """
    Generate targets for a classification group for use in scikit-learn.

    Returns:
        np.ndarray:
            - For multiclass: shape (n_samples,), integer class indices (unlabeled as -1)
            - For multilabel: shape (n_samples, n_labels), entries 0 or 1
    """
    group_type = str(target_group_config["type"])
    targets = list(target_group_config["targets"])
    n_samples = len(dataset)

    if group_type == "multiclass":
        masks: list[np.ndarray] = []
        for i, t in enumerate(targets):
            try:
                print(f"Creating binary target labels for class {t:s} = {i:d}")
                m, _ = create_binary_target_labels(t, dataset, verbose=False)
            except ValueError:
                print(f"Target {t} not present in this split. Filling with zeros.")
                m = np.zeros(n_samples, dtype=bool)
            masks.append(m.astype(bool))

        mask_stack = np.stack(masks, axis=1)  # (n_samples, n_targets)
        y = np.full(n_samples, -1, dtype=int)
        has_any = mask_stack.any(axis=1)
        # Encode samples with at least one active class, others stay -1
        # Argmax returns the index of the first active class
        if np.any(has_any):
            y[has_any] = mask_stack[has_any].argmax(axis=1).astype(int) # (n_samples,)

    elif group_type == "multilabel":
        cols: list[np.ndarray] = []
        for t in targets:
            try:
                print(f"Creating binary target labels for {t}")
                m, _ = create_binary_target_labels(t, dataset, verbose=False)
            except ValueError:
                print(f"Target {t} not present in this split. Filling with zeros.")
                m = np.zeros(n_samples, dtype=bool)
            cols.append(m.astype(int))
        y = np.stack(cols, axis=1).astype(int) # (n_samples, n_targets)
    else:
        raise ValueError(f"Unknown target group type: {group_type}")

    return y


def subsample_training_data(
    X_train: np.ndarray,
    y_train: np.ndarray,
    max_samples: int = 50000,
    random_state: int = 42
) -> tuple[np.ndarray, np.ndarray]:
    """
    Subsample training data using stratified sampling.

    Uses stratified sampling to preserve natural class proportions while limiting
    sample count. This is superior to balancing for elastic-net logistic regression
    because it preserves all class information and enables stable coefficient estimates.

    Use class_weight="balanced" in LogisticRegression to handle class imbalance.

    For multiclass: stratifies by class
    For multilabel: stratifies by label combination

    Args:
        X_train: Training features
        y_train: Training labels (1D for multiclass, 2D for multilabel)
        max_samples: Maximum samples to keep
        random_state: Random seed

    Returns:
        Subsampled (X_train, y_train)
    """
    # Filter out unlabeled samples for multiclass
    if y_train.ndim == 1:
        valid_mask = y_train >= 0
        X_train = X_train[valid_mask]
        y_train = y_train[valid_mask]

    # If already under limit, return as-is
    if len(X_train) <= max_samples:
        print(f"Training samples: {len(X_train)} (within limit, no subsampling needed)")
        return X_train, y_train

    # Stratified subsample
    if y_train.ndim == 1:
        # Multiclass: stratify by class
        print(f"Stratified subsampling: {len(X_train)} -> {max_samples} samples (multiclass)")
        X_sub, _, y_sub, _ = train_test_split(
            X_train, y_train,
            train_size=max_samples,
            stratify=y_train,
            random_state=random_state
        )
    else:
        # Multilabel: stratify by label combination
        print(f"Stratified subsampling: {len(X_train)} -> {max_samples} samples (multilabel)")
        # Convert to string representation for stratification
        label_combos = [''.join(map(str, row)) for row in y_train]
        X_sub, _, y_sub, _ = train_test_split(
            X_train, y_train,
            train_size=max_samples,
            stratify=label_combos,
            random_state=random_state
        )

    return X_sub, y_sub

# ==================================================
# Metrics and Feature Importance
# ==================================================

def compute_aggregate_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    target_group_config: dict
) -> dict:
    """
    Compute aggregate metrics (F1, log-loss, ROC-AUC) for terminal display.

    Args:
        y_true: Ground truth labels
        y_pred: Predictions
        y_proba: Prediction probabilities
        target_group_config: Target group configuration dict

    Returns:
        dict: {'f1': float, 'log_loss': float, 'roc_auc': float}
    """
    group_type = target_group_config['type']

    if group_type == 'multilabel':
        # Multilabel: micro-averaged metrics
        f1 = f1_score(y_true.astype(int), y_pred, average='micro')

        try:
            log_loss_val = log_loss(y_true.astype(int), y_proba)
        except (ValueError, IndexError):
            log_loss_val = np.nan

        try:
            roc_auc = roc_auc_score(y_true.astype(int), y_proba, average='micro')
        except (ValueError, IndexError):
            roc_auc = np.nan

    elif group_type == 'multiclass':
        # Multiclass: weighted-averaged metrics
        f1 = f1_score(y_true.astype(int), y_pred, average='weighted', zero_division=0)

        # Filter to only classes present in both y_true and y_proba
        n_proba_classes = y_proba.shape[1]
        unique_classes = np.unique(y_true)
        valid_mask = unique_classes < n_proba_classes

        if not valid_mask.all():
            # Some classes in y_true are not in y_proba (not learned by model)
            # Filter samples to only those with classes that were learned
            valid_samples_mask = np.isin(y_true, unique_classes[valid_mask])
            y_true_filtered = y_true[valid_samples_mask]
            y_pred_filtered = y_pred[valid_samples_mask]
            y_proba_filtered = y_proba[valid_samples_mask]
        else:
            y_true_filtered = y_true
            y_pred_filtered = y_pred
            y_proba_filtered = y_proba

        # Compute log_loss only on filtered samples
        try:
            if len(y_true_filtered) > 0:
                log_loss_val = log_loss(y_true_filtered.astype(int), y_proba_filtered)
            else:
                log_loss_val = np.nan
        except (ValueError, IndexError):
            log_loss_val = np.nan

        # Compute ROC-AUC
        try:
            if len(y_true_filtered) > 0:
                # sklearn treats 2-class problems as binary and expects 1D probabilities
                if y_proba_filtered.shape[1] == 2:
                    roc_auc = roc_auc_score(y_true_filtered.astype(int), y_proba_filtered[:, 1])
                else:
                    roc_auc = roc_auc_score(
                        y_true_filtered.astype(int), y_proba_filtered,
                        multi_class='ovr', average='weighted'
                    )
            else:
                roc_auc = np.nan
        except (ValueError, IndexError):
            roc_auc = np.nan

    else:
        # Binary classification
        f1 = f1_score(y_true.astype(int), y_pred, zero_division=0)

        try:
            log_loss_val = log_loss(y_true.astype(int), y_proba[:, 1])
        except (ValueError, IndexError):
            log_loss_val = np.nan

        try:
            roc_auc = roc_auc_score(y_true.astype(int), y_proba[:, 1])
        except (ValueError, IndexError):
            roc_auc = np.nan

    return {
        'f1': f1,
        'log_loss': log_loss_val,
        'roc_auc': roc_auc
    }


def compute_per_target_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    target_group_config: dict
) -> dict:
    """
    Compute F1, ROC-AUC, and balanced accuracy for each individual target.

    This function is called ONCE PER SPLIT (validation or test).
    The returned metrics are for that specific split only.

    Args:
        y_true: Ground truth labels for ONE split (val or test)
        y_pred: Predictions for that same split
        y_proba: Prediction probabilities for that same split
        target_group_config: Target group configuration dict

    Returns:
        dict[str, dict]: {target_name: {'f1': float, 'roc_auc': float, 'balanced_acc': float}}
    """
    targets = target_group_config['targets']
    group_type = target_group_config['type']
    metrics = {}

    if group_type == 'multiclass':
        # Check how many classes the model actually learned
        n_proba_classes = y_proba.shape[1] if y_proba.ndim > 1 else 1

        for i, target in enumerate(targets):
            # One-vs-rest for this class
            y_true_binary = (y_true == i).astype(int)
            y_pred_binary = (y_pred == i).astype(int)

            # Check if this class exists in y_true
            n_positive = y_true_binary.sum()
            n_negative = len(y_true_binary) - n_positive

            # Compute F1 and balanced accuracy (can handle single-class cases)
            f1 = f1_score(y_true_binary, y_pred_binary, zero_division=0)
            balanced_acc = balanced_accuracy_score(y_true_binary, y_pred_binary)

            # Compute ROC-AUC only if:
            # 1. Both classes are present in y_true
            # 2. The class index is within the range of y_proba columns
            if n_positive > 0 and n_negative > 0 and i < n_proba_classes:
                try:
                    roc_auc = roc_auc_score(y_true_binary, y_proba[:, i])
                except (ValueError, IndexError):
                    roc_auc = np.nan
            else:
                roc_auc = np.nan

            metrics[target] = {
                'f1': f1,
                'roc_auc': roc_auc,
                'balanced_acc': balanced_acc
            }

    elif group_type == 'multilabel':
        # Handle MultiOutputClassifier: predict_proba returns list of arrays
        # Each element is (n_samples, 2) for binary classification of that label
        # We want the probability of the positive class (index 1)
        if isinstance(y_proba, list):
            y_proba_positive = np.column_stack([proba[:, 1] for proba in y_proba])
        else:
            # Standard format (shouldn't happen for multilabel, but handle it)
            y_proba_positive = y_proba

        for i, target in enumerate(targets):
            y_true_target = y_true[:, i]
            y_pred_target = y_pred[:, i]

            # Check if both classes are present
            n_positive = y_true_target.sum()
            n_negative = len(y_true_target) - n_positive

            f1 = f1_score(y_true_target, y_pred_target, zero_division=0)
            balanced_acc = balanced_accuracy_score(y_true_target, y_pred_target)

            if n_positive > 0 and n_negative > 0:
                try:
                    roc_auc = roc_auc_score(y_true_target, y_proba_positive[:, i])
                except (ValueError, IndexError):
                    roc_auc = np.nan
            else:
                roc_auc = np.nan

            metrics[target] = {
                'f1': f1,
                'roc_auc': roc_auc,
                'balanced_acc': balanced_acc
            }

    return metrics


def extract_feature_importance(model, top_k: int = 100) -> pd.DataFrame:
    """
    Extract feature importance from logistic regression coefficients.

    For binary/multiclass: model.coef_ is (n_classes, n_features)
    For multilabel (MultiOutputClassifier): need to extract from estimators_

    Args:
        model: Trained LogisticRegression or MultiOutputClassifier model
        top_k: Number of top features to return

    Returns:
        pd.DataFrame with columns: rank, feature_idx, coefficient, abs_coefficient
    """
    # Handle MultiOutputClassifier (multilabel)
    if isinstance(model, MultiOutputClassifier):
        # Extract coefficients from each estimator and average
        coef_list = [estimator.coef_[0] for estimator in model.estimators_]
        coef = np.stack(coef_list, axis=0)  # (n_labels, n_features)
        # Use mean absolute coefficient across labels
        coef_vector = np.abs(coef).mean(axis=0)
    else:
        # Standard LogisticRegression
        coef = model.coef_

        if coef.shape[0] == 1:
            # Binary: use the single coefficient vector
            coef_vector = coef[0]
        else:
            # Multiclass: use mean absolute coefficient across classes
            coef_vector = np.abs(coef).mean(axis=0)

    # Get top-k by absolute value
    abs_coef = np.abs(coef_vector)
    top_indices = np.argsort(abs_coef)[::-1][:top_k]

    df = pd.DataFrame({
        'rank': range(1, len(top_indices) + 1),
        'feature_idx': top_indices,
        'coefficient': coef_vector[top_indices],
        'abs_coefficient': abs_coef[top_indices]
    })

    return df


def count_nonzero_coefficients(model) -> int:
    """
    Count number of non-zero coefficients (sparsity measure).

    Args:
        model: Trained LogisticRegression or MultiOutputClassifier model

    Returns:
        Number of non-zero coefficients
    """
    # Handle MultiOutputClassifier (multilabel)
    if isinstance(model, MultiOutputClassifier):
        # Extract coefficients from each estimator
        coef_list = [estimator.coef_[0] for estimator in model.estimators_]
        coef = np.stack(coef_list, axis=0)  # (n_labels, n_features)
        # Count features that are non-zero in ANY label
        return int(np.count_nonzero(np.any(coef != 0, axis=0)))
    else:
        # Standard LogisticRegression
        coef = model.coef_
        if coef.shape[0] == 1:
            return int(np.count_nonzero(coef[0]))
        else:
            # For multiclass: count features that are non-zero in ANY class
            return int(np.count_nonzero(np.any(coef != 0, axis=0)))


# ==================================================
# Results Saving
# ==================================================

def check_target_group_complete(results_dir: Path, target_group: str, targets: list[str]) -> bool:
    """
    Check if all result files exist for a target group.

    Checks:
    - Root: summary, grid search, coefficients (training artifacts)
    - valid/: per-target metrics (validation evaluation)
    - test/: per-target metrics (test evaluation)

    Args:
        results_dir: Base results directory for this SAE config
        target_group: Name of target group
        targets: List of individual target names

    Returns:
        True if all result files exist, False otherwise
    """
    # Check root-level training artifacts
    if not (results_dir / f"{target_group}_summary.csv").exists():
        return False
    if not (results_dir / f"{target_group}_grid_search.csv").exists():
        return False

    # Check coefficients for each target (both dense and sparse)
    for target in targets:
        if not (results_dir / f"{target_group}_{target}_coefficients_dense.csv").exists():
            return False
        if not (results_dir / f"{target_group}_{target}_coefficients_sparse.csv").exists():
            return False

    # Check split-specific evaluation metrics
    for split in ['valid', 'test']:
        split_dir = results_dir / split
        for target in targets:
            if not (split_dir / f"{target_group}_{target}_metrics.csv").exists():
                return False

    return True


def check_job_complete(
    base_model: str,
    sae_config: str,
    target_groups: dict,
    results_base_dir: str
) -> tuple[bool, list[str]]:
    """
    Check if all target groups are complete for a given job (base_model, SAE_config).

    Args:
        base_model: Base model name (e.g., 'biomedparse')
        sae_config: SAE configuration name (e.g., 'D_128_512_2048_8192_K_30_60_120_240')
        target_groups: Dictionary of target groups to check (from TARGET_GROUPS)
        results_base_dir: Base directory for results

    Returns:
        is_complete: True if all target groups have complete results
        missing_groups: List of target group names that need processing
    """
    results_dir = Path(results_base_dir) / base_model / sae_config

    missing = []
    for target_group_name, target_group_config in target_groups.items():
        targets = target_group_config['targets']
        if not check_target_group_complete(results_dir, target_group_name, targets):
            missing.append(target_group_name)

    is_complete = (len(missing) == 0)
    return is_complete, missing


def save_training_artifacts(
    results_dir: Path,
    target_group: str,
    coefficients_dense: dict,
    coefficients_sparse: dict,
    grid_results: pd.DataFrame
) -> None:
    """
    Save training artifacts (coefficients, grid search) to disk.

    These are model properties, not split-specific, so stored in results root.

    Args:
        results_dir: Base results directory for this SAE config
        target_group: Name of target group
        coefficients_dense: {target: {rank, feature_idx, coef, abs_coef}}
        coefficients_sparse: {target: {rank, feature_idx, coef, abs_coef}}
        grid_results: DataFrame with grid search results
    """
    # Save grid search results
    grid_path = results_dir / f"{target_group}_grid_search.csv"
    grid_results.to_csv(grid_path, index=False)
    print(f"  Saved grid search results: {grid_path}")

    # Save coefficients for each target
    for target, coef_data in coefficients_dense.items():
        coef_path = results_dir / f"{target_group}_{target}_coefficients_dense.csv"
        pd.DataFrame(coef_data).to_csv(coef_path, index=False)

    for target, coef_data in coefficients_sparse.items():
        coef_path = results_dir / f"{target_group}_{target}_coefficients_sparse.csv"
        pd.DataFrame(coef_data).to_csv(coef_path, index=False)


def save_evaluation_metrics(
    results_dir: Path,
    split_name: str,
    target_group: str,
    target_name: str,
    feature_type: str,
    model_config: dict,
    train_metrics: dict,
    eval_metrics: dict
) -> None:
    """
    Save evaluation metrics to split-specific directory.

    Args:
        results_dir: Base results directory for this SAE config
        split_name: 'valid' or 'test'
        target_group: Name of target group
        target_name: Individual target name
        feature_type: 'dense' or 'sparse'
        model_config: {'C': float, 'l1_ratio': float, 'n_features': int, 'n_nonzero': int, 'fit_time': float}
        train_metrics: {'f1': float, 'roc_auc': float, 'balanced_acc': float}
        eval_metrics: {'f1': float, 'roc_auc': float, 'balanced_acc': float, 'log_loss': float}
    """
    split_dir = results_dir / split_name
    split_dir.mkdir(exist_ok=True)

    metrics_path = split_dir / f"{target_group}_{target_name}_metrics.csv"

    # Compute sparsity
    sparsity = 1.0 - (model_config['n_nonzero'] / model_config['n_features'])

    # Construct row
    row = {
        'target': target_name,
        'feature_type': feature_type,
        'C': model_config['C'],
        'l1_ratio': model_config['l1_ratio'],
        'n_features': model_config['n_features'],
        'n_nonzero': model_config['n_nonzero'],
        'sparsity': sparsity,
        'train_f1': train_metrics['f1'],
        'train_roc_auc': train_metrics['roc_auc'],
        'train_balanced_acc': train_metrics['balanced_acc'],
        'eval_f1': eval_metrics['f1'],
        'eval_roc_auc': eval_metrics['roc_auc'],
        'eval_balanced_acc': eval_metrics['balanced_acc'],
        'eval_log_loss': eval_metrics['log_loss'],
        'fit_time': model_config['fit_time']
    }

    # Append or create new file
    df = pd.DataFrame([row])
    if metrics_path.exists():
        df_existing = pd.read_csv(metrics_path)
        df = pd.concat([df_existing, df], ignore_index=True)
    df.to_csv(metrics_path, index=False)


# ==================================================
# Visualization
# ==================================================

def plot_top_activating_samples(
    model,
    sparse_features: np.ndarray,
    metadata_dataset,
    target_group: str,
    target_name: str,
    results_dir: Path,
    top_k_features: int = 10,
    n_samples: int = 5
) -> None:
    """
    Plot top activating samples for top features based on coefficient magnitude.

    This visualization helps assess mono-semanticity of sparse features by showing
    which samples activate each feature most strongly.

    Args:
        model: Trained LogisticRegression or MultiOutputClassifier model
        sparse_features: Sparse feature matrix (n_samples, n_features)
        metadata_dataset: Dataset with metadata including image_path
        target_group: Name of target group
        target_name: Individual target name
        results_dir: Directory to save plots
        top_k_features: Number of top features to visualize
        n_samples: Number of top activating samples per feature
    """
    # Extract coefficients
    if isinstance(model, MultiOutputClassifier):
        # For multilabel: average absolute coefficients across labels
        coef_list = [estimator.coef_[0] for estimator in model.estimators_]
        coef = np.stack(coef_list, axis=0)
        coef_vector = np.abs(coef).mean(axis=0)
    else:
        coef = model.coef_
        if coef.shape[0] == 1:
            coef_vector = coef[0]
        else:
            # For multiclass: average absolute coefficients across classes
            coef_vector = np.abs(coef).mean(axis=0)

    # Get top features by coefficient magnitude
    abs_coef = np.abs(coef_vector)
    top_feature_indices = np.argsort(abs_coef)[::-1][:top_k_features]

    # Get metadata as DataFrame
    # Build from dataset attributes (image_ids) and metadata dictionary
    df_metadata = pd.DataFrame(metadata_dataset.metadata)
    df_metadata['image_id'] = metadata_dataset.image_ids
    df_metadata['slice_orientation'] = metadata_dataset.slice_orientations
    df_metadata['modality'] = metadata_dataset.modalities

    # Collect top samples per feature
    top_samples_per_feature = {}
    for feat_idx in top_feature_indices:
        feature_values = sparse_features[:, feat_idx]
        df_feat = df_metadata.copy()
        df_feat['feature_value'] = feature_values

        # Sort by feature value (descending) and deduplicate
        df_feat_sorted = df_feat.sort_values('feature_value', ascending=False)
        unique_rows = df_feat_sorted.drop_duplicates(
            subset=['image_id', 'slice_orientation', 'modality'], keep='first'
        )

        top_samples_per_feature[feat_idx] = {
            'sample_indices': unique_rows.head(n_samples).index.values,
            'feature_values': unique_rows.head(n_samples)['feature_value'].values,
            'coefficient': coef_vector[feat_idx]
        }

    # Plot images
    fig, axes = plt.subplots(top_k_features, n_samples, figsize=(20, 4 * top_k_features))

    # Handle single row case
    if top_k_features == 1:
        axes = axes.reshape(1, -1)

    for feat_rank, feat_idx in enumerate(top_feature_indices):
        sample_indices = top_samples_per_feature[feat_idx]['sample_indices']
        feature_values = top_samples_per_feature[feat_idx]['feature_values']
        coefficient = top_samples_per_feature[feat_idx]['coefficient']

        for i, sample_idx in enumerate(sample_indices):
            image_path = df_metadata.iloc[sample_idx]['image_path']
            try:
                img = Image.open(image_path).convert('L')
                ax = axes[feat_rank, i]
                ax.imshow(np.array(img), cmap='gray')
                ax.axis('off')

                # Add title with feature info on first image
                if i == 0:
                    ax.set_title(
                        f'Feature {feat_idx}\nCoef: {coefficient:.3f}\nVal: {feature_values[i]:.2f}',
                        fontsize=8, loc='left'
                    )
                else:
                    ax.set_title(f'{feature_values[i]:.2f}', fontsize=8)

            except (FileNotFoundError, OSError) as e:
                # Handle missing images
                ax = axes[feat_rank, i]
                ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')
                ax.axis('off')

    plt.suptitle(
        f'Top Activating Samples for {target_name.upper()} (Target Group: {target_group})',
        fontsize=14, y=0.995
    )
    plt.tight_layout()

    plot_path = results_dir / f"{target_group}_{target_name}_top_activating_samples.png"
    plt.savefig(plot_path, dpi=200, facecolor='white', bbox_inches='tight')
    plt.close()

    print(f"  Saved plot: {plot_path}")
