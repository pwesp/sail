#!/usr/bin/env python3
"""
Performance Recovery Analysis

Quantify how well sparse SAE features recover dense-model performance as a
function of feature count, and how efficiently performance approaches its sparse optimum.

This script analyzes the trade-off between feature count and model performance by
training logistic regression models on restricted feature sets (top-N features by
coefficient magnitude) and evaluating their performance recovery relative to dense baselines.
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
from sklearn.metrics import balanced_accuracy_score, f1_score, roc_auc_score
from sklearn.multioutput import MultiOutputClassifier
from time import time
import matplotlib.pyplot as plt
import matplotlib

# Use non-interactive backend for server environments
matplotlib.use('Agg')

# Local imports
from src.dataloading import TotalSegmentatorLatentFeaturesDataset
from src.config_downstream import BASE_MODEL_PATHS, TARGET_GROUPS, RESULTS_BASE_DIR
from src.downstream_clf import (
    log,
    load_targets_for_group,
    subsample_training_data,
    get_sae_configs,
)

# Constants
N_FEATURES_BASE_LIST = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100]
RECOVERY_THRESHOLDS = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
OUTPUT_DIR = "results/performance_recovery"
MAX_FEATURES_LIMIT = 100
MAX_TRAINING_SAMPLES = 5000  # Reduced from 10000 in `train_linear_probe.py` for faster runtime
DENSE_ROC_AUC_MIN = 0.55  # Filter out targets where dense model performs near-chance or worse

# Model hyperparameters (relaxed for performance recovery analysis)
LOGISTIC_C = 1.0
LOGISTIC_L1_RATIO = 0.5
LOGISTIC_MAX_ITER = 200  # Reduced from 1000 (10x speedup, sufficient for feature ranking)
RANDOM_STATE = 42


# ==================================================
# CLI Argument Parsing
# ==================================================

def parse_arguments() -> argparse.Namespace:
    """Parse command-line arguments for filtering processing."""
    parser = argparse.ArgumentParser(
        description='Analyze sparse feature performance recovery vs. dense baseline'
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


# ==================================================
# Helper Functions
# ==================================================

def create_masked_features(
    X_full: np.ndarray,
    feature_indices: np.ndarray
) -> np.ndarray:
    """
    Create masked feature matrix with only selected features active.

    Preserves full dimensionality - all features except feature_indices
    are set to 0.0. This maintains feature ID traceability.

    Args:
        X_full: Full sparse features (n_samples, n_features)
        feature_indices: Indices of features to keep active

    Returns:
        X_masked: Same shape as X_full, with all features except
                  feature_indices set to 0.0
    """
    X_masked = np.zeros_like(X_full)
    X_masked[:, feature_indices] = X_full[:, feature_indices]
    return X_masked


def load_coefficient_file(coef_path: Path) -> pd.DataFrame:
    """
    Load and validate coefficient file.

    Args:
        coef_path: Path to coefficient CSV file

    Returns:
        DataFrame with columns: rank, feature_idx, coefficient, abs_coefficient
        feature_idx maps to original sparse feature space (0 to n_features-1)
    """
    assert coef_path.exists(), f"Coefficient file not found: {coef_path}"

    coef_df = pd.read_csv(coef_path)

    # Validate expected columns
    required_cols = {'rank', 'feature_idx', 'coefficient', 'abs_coefficient'}
    assert required_cols.issubset(coef_df.columns), \
        f"Coefficient file missing required columns. Expected {required_cols}, got {set(coef_df.columns)}"

    # Validate feature indices are non-negative
    assert (coef_df['feature_idx'] >= 0).all(), \
        "Coefficient file contains negative feature indices"

    return coef_df


def load_baseline_metrics(
    results_dir: Path,
    target_group: str,
    target: str,
    split: str = 'valid'
) -> dict[str, float]:
    """
    Load pre-computed metrics for dense and sparse models.

    Reads CSV file and extracts metrics for both feature types.

    Args:
        results_dir: Base results directory (e.g., results/linear_probes/{BASE_MODEL}/{SAE_CONFIG})
        target_group: Target group name
        target: Target name
        split: Which split to load (default: 'valid')

    Returns:
        dict with keys:
            - dense_roc_auc, dense_f1, dense_balanced_acc
            - sparse_roc_auc, sparse_f1, sparse_balanced_acc
            - n_nonzero (int)
    """
    metrics_path = results_dir / split / f"{target_group}_{target}_metrics.csv"
    assert metrics_path.exists(), f"Metrics file not found: {metrics_path}"

    metrics_df = pd.read_csv(metrics_path)

    # Extract dense and sparse rows (take first occurrence if duplicates exist)
    dense_rows = metrics_df[metrics_df['feature_type'] == 'dense']
    sparse_rows = metrics_df[metrics_df['feature_type'] == 'sparse']

    assert len(dense_rows) >= 1, f"Expected at least 1 dense row, found {len(dense_rows)}"
    assert len(sparse_rows) >= 1, f"Expected at least 1 sparse row, found {len(sparse_rows)}"

    # Take first occurrence (handles duplicates from repeated runs)
    dense_row = dense_rows.iloc[0]
    sparse_row = sparse_rows.iloc[0]

    return {
        'dense_roc_auc': float(dense_row['eval_roc_auc']),
        'dense_f1': float(dense_row['eval_f1']),
        'dense_balanced_acc': float(dense_row['eval_balanced_acc']),
        'sparse_roc_auc': float(sparse_row['eval_roc_auc']),
        'sparse_f1': float(sparse_row['eval_f1']),
        'sparse_balanced_acc': float(sparse_row['eval_balanced_acc']),
        'n_nonzero': int(sparse_row['n_nonzero']),
    }


def determine_feature_counts_to_evaluate(
    n_nonzero: int,
    max_available: int,
    base_list: list[int] = N_FEATURES_BASE_LIST
) -> list[int]:
    """
    Build sorted list of feature counts to evaluate.

    Includes n_nonzero if not in base_list and <= 100.
    Clips to max_available features.

    Args:
        n_nonzero: Number of features used by original elastic-net model
        max_available: Maximum number of features in coefficient file
        base_list: Base list of feature counts to evaluate

    Returns:
        Sorted list of feature counts to evaluate
    """
    feature_counts = set(base_list)

    # Add n_nonzero if it's within limits and not already in list
    if n_nonzero <= MAX_FEATURES_LIMIT and n_nonzero not in feature_counts:
        feature_counts.add(n_nonzero)

    # Filter out counts exceeding available features
    feature_counts = {n for n in feature_counts if n <= max_available}

    # Return sorted list
    return sorted(feature_counts)


def check_target_complete(
    output_dir: Path,
    target_group: str,
    target: str
) -> bool:
    """
    Check if target already processed (resume capability).

    Args:
        output_dir: Output directory for results
        target_group: Target group name
        target: Target name

    Returns:
        True if target already processed, False otherwise
    """
    recovery_csv = output_dir / f"{target_group}_{target}_recovery.csv"
    recovery_png = output_dir / f"{target_group}_{target}_recovery.png"

    return recovery_csv.exists() and recovery_png.exists()



# ==================================================
# Core Analysis Functions
# ==================================================

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    task_type: str
) -> dict[str, float]:
    """
    Compute ROC-AUC, F1, and balanced accuracy for a given task type.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_proba: Predicted probabilities
        task_type: One of 'binary', 'multiclass', 'multilabel'

    Returns:
        dict with keys: roc_auc, f1, balanced_acc
    """
    metrics = {}

    try:
        if task_type == 'binary':
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba[:, 1])
            metrics['f1'] = f1_score(y_true, y_pred)
            metrics['balanced_acc'] = balanced_accuracy_score(y_true, y_pred)

        elif task_type == 'multiclass':
            metrics['roc_auc'] = roc_auc_score(
                y_true, y_proba, multi_class='ovr', average='weighted'
            )
            metrics['f1'] = f1_score(y_true, y_pred, average='weighted')
            metrics['balanced_acc'] = balanced_accuracy_score(y_true, y_pred)

        elif task_type == 'multilabel':
            metrics['roc_auc'] = roc_auc_score(y_true, y_proba, average='micro')
            metrics['f1'] = f1_score(y_true, y_pred, average='micro')
            # For multilabel, compute per-label balanced accuracy and average
            n_labels = y_true.shape[1]
            balanced_accs = []
            for i in range(n_labels):
                if len(np.unique(y_true[:, i])) > 1:  # Skip if only one class present
                    balanced_accs.append(balanced_accuracy_score(y_true[:, i], y_pred[:, i]))
            metrics['balanced_acc'] = np.mean(balanced_accs) if balanced_accs else np.nan

        else:
            raise ValueError(f"Unknown task type: {task_type}")

    except (ValueError, IndexError) as e:
        # Handle edge cases (e.g., only one class present, insufficient samples)
        log(f"Warning: Could not compute metrics: {e}")
        metrics['roc_auc'] = np.nan
        metrics['f1'] = np.nan
        metrics['balanced_acc'] = np.nan

    return metrics


def train_and_evaluate_restricted_model(
    X_train_full: np.ndarray,
    y_train: np.ndarray,
    X_val_full: np.ndarray,
    y_val: np.ndarray,
    X_test_full: np.ndarray,
    y_test: np.ndarray,
    feature_indices: np.ndarray,
    target_group_config: dict,
    C: float = LOGISTIC_C,
    l1_ratio: float = LOGISTIC_L1_RATIO,
    max_iter: int = LOGISTIC_MAX_ITER
) -> dict[str, float]:
    """
    Train logistic regression on masked features and evaluate.

    Creates masked versions of X_*_full where only features in
    feature_indices are non-zero. Maintains full dimensionality
    for feature ID traceability.

    Args:
        X_train_full: Full training features
        y_train: Training labels
        X_val_full: Full validation features
        y_val: Validation labels
        X_test_full: Full test features
        y_test: Test labels
        feature_indices: Indices of features to keep active
        target_group_config: Target group configuration dict
        C: Inverse regularization strength
        l1_ratio: Elastic-net mixing parameter
        max_iter: Maximum iterations for solver

    Returns:
        dict with metrics for all splits:
            train_roc_auc, train_f1, train_balanced_acc,
            valid_roc_auc, valid_f1, valid_balanced_acc,
            test_roc_auc, test_f1, test_balanced_acc
    """
    # Create masked features
    X_train_masked = create_masked_features(X_train_full, feature_indices)
    X_val_masked = create_masked_features(X_val_full, feature_indices)
    X_test_masked = create_masked_features(X_test_full, feature_indices)

    # Determine task type
    task_type = target_group_config['type']
    is_multilabel = (task_type == 'multilabel')

    # Configure logistic regression model
    if l1_ratio == 0.0:
        base_model = LogisticRegression(
            penalty='l2', solver='lbfgs', C=C,
            class_weight='balanced', max_iter=max_iter, random_state=RANDOM_STATE
        )
    elif l1_ratio == 1.0:
        base_model = LogisticRegression(
            penalty='l1', solver='saga', C=C,
            class_weight='balanced', max_iter=max_iter, random_state=RANDOM_STATE
        )
    else:
        base_model = LogisticRegression(
            penalty='elasticnet', solver='saga', C=C, l1_ratio=l1_ratio,
            class_weight='balanced', max_iter=max_iter, random_state=RANDOM_STATE
        )

    # Wrap with MultiOutputClassifier for multilabel tasks
    if is_multilabel:
        model = MultiOutputClassifier(base_model)
    else:
        model = base_model

    # Train model
    model.fit(X_train_masked, y_train)

    # Evaluate on all splits
    results = {}

    for split_name, X_masked, y_true in [
        ("train", X_train_masked, y_train),
        ("valid", X_val_masked, y_val),
        ("test", X_test_masked, y_test),
    ]:
        # Filter out unlabeled samples for multiclass
        if task_type == 'multiclass':
            eval_mask = y_true >= 0
            y_eval = y_true[eval_mask]
            X_eval = X_masked[eval_mask]
        else:
            y_eval = y_true
            X_eval = X_masked

        # Get predictions
        y_pred = model.predict(X_eval)
        y_proba = model.predict_proba(X_eval)

        # Compute metrics
        metrics = compute_metrics(y_eval, y_pred, y_proba, task_type)

        # Store with split prefix
        results[f"{split_name}_roc_auc"] = metrics['roc_auc']
        results[f"{split_name}_f1"] = metrics['f1']
        results[f"{split_name}_balanced_acc"] = metrics['balanced_acc']

    return results


def compute_recovery_thresholds(
    dense_baseline: float,
    sparse_results: list[dict],
    thresholds: list[float] = RECOVERY_THRESHOLDS
) -> dict[str, int]:
    """
    Find minimum N achieving each recovery threshold.

    Recovery = sparse_roc_auc(N) / dense_baseline_roc_auc

    Args:
        dense_baseline: Dense model ROC-AUC (from valid set)
        sparse_results: List of dicts with keys: n_features, valid_roc_auc
        thresholds: List of recovery percentages (e.g., [0.5, 0.6, ..., 1.0])

    Returns:
        dict mapping threshold (e.g., '50pct') to n_features (or np.nan if not reached)

    Note:
        For each threshold, finds the MINIMUM N where recovery >= threshold.
    """
    recovery_dict = {}

    for threshold in thresholds:
        threshold_pct = int(threshold * 100)
        threshold_key = f"n_features_{threshold_pct}pct"

        # Find minimum N where recovery >= threshold
        min_n = np.nan
        for result in sparse_results:
            n_features = result['n_features']
            valid_roc_auc = result['valid_roc_auc']

            # Skip if ROC-AUC is NaN
            if np.isnan(valid_roc_auc):
                continue

            # Compute recovery percentage
            recovery = valid_roc_auc / dense_baseline if dense_baseline > 0 else 0

            # Check if threshold met
            if recovery >= threshold:
                if np.isnan(min_n) or n_features < min_n:
                    min_n = n_features

        recovery_dict[threshold_key] = min_n

    return recovery_dict


def plot_recovery_curve(
    results: list[dict],
    dense_baseline: float,
    sparse_baseline: float,
    n_nonzero: int,
    output_path: Path,
    target_group: str,
    target: str
) -> None:
    """
    Generate ROC-AUC vs. feature count plot.

    Elements:
        - Lines: train, valid, test ROC-AUC vs. N
        - Horizontal line: dense baseline
        - Vertical line: n_nonzero
        - Marker: (n_nonzero, sparse_baseline) point

    Args:
        results: List of dicts with keys: n_features, train_roc_auc, valid_roc_auc, test_roc_auc
        dense_baseline: Dense model ROC-AUC (validation)
        sparse_baseline: Sparse model ROC-AUC at n_nonzero (validation)
        n_nonzero: Number of features used by original model
        output_path: Path to save plot
        target_group: Target group name (for title)
        target: Target name (for title)
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Extract data for plotting
    n_features_list = [r['n_features'] for r in results]
    train_roc_auc = [r['train_roc_auc'] for r in results]
    valid_roc_auc = [r['valid_roc_auc'] for r in results]
    test_roc_auc = [r['test_roc_auc'] for r in results]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Plot performance curves
    ax.plot(n_features_list, train_roc_auc, 'o-', label='train', alpha=0.7)
    ax.plot(n_features_list, valid_roc_auc, 's-', label='valid', alpha=0.7)
    ax.plot(n_features_list, test_roc_auc, '^-', label='test', alpha=0.7)

    # Add horizontal line for dense baseline
    ax.axhline(y=dense_baseline, color='gray', linestyle='--',
               label=f'dense baseline ({dense_baseline:.3f})', alpha=0.7)

    # Add vertical line at n_nonzero
    ax.axvline(x=n_nonzero, color='red', linestyle=':',
               label=f'n_nonzero ({n_nonzero})', alpha=0.5)

    # Mark the sparse baseline point
    ax.plot(n_nonzero, sparse_baseline, 'r*', markersize=15,
            label=f'sparse baseline ({sparse_baseline:.3f})', zorder=5)

    # Configure axes
    ax.set_xscale('log')
    ax.set_xlabel('number of features')
    ax.set_ylabel('roc-auc')
    ax.set_title(f'performance recovery: {target_group} - {target}')
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best')

    # Set y-axis limits to focus on the relevant range
    y_min = max(0.4, min(min(train_roc_auc), min(valid_roc_auc), min(test_roc_auc), dense_baseline) - 0.05)
    y_max = min(1.0, max(max(train_roc_auc), max(valid_roc_auc), max(test_roc_auc), dense_baseline) + 0.05)
    ax.set_ylim(y_min, y_max)

    # Save figure
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def process_target(
    base_model_name: str,
    sae_config: str,
    target_group_name: str,
    target: str,
    target_group_config: dict,
    X_sparse_train_full: np.ndarray,
    X_sparse_val_full: np.ndarray,
    X_sparse_test_full: np.ndarray,
    y_train: np.ndarray,
    y_val: np.ndarray,
    y_test: np.ndarray,
    results_dir: Path,
    output_dir: Path
) -> dict | None:
    """
    Orchestrate full analysis for one target.

    Receives full sparse feature matrices (no masking applied yet).
    For each N, creates masked versions and trains model.

    Args:
        base_model_name: Name of base model
        sae_config: SAE configuration name
        target_group_name: Target group name
        target: Target name
        target_group_config: Target group configuration dict
        X_sparse_train_full: Full training sparse features
        X_sparse_val_full: Full validation sparse features
        X_sparse_test_full: Full test sparse features
        y_train: Training labels
        y_val: Validation labels
        y_test: Test labels
        results_dir: Directory with pre-computed baseline metrics
        output_dir: Directory to save recovery analysis results

    Returns:
        Summary dict for aggregation
    """
    log(f"  Processing target: {target}")

    # Load coefficient file
    coef_path = results_dir / f"{target_group_name}_{target}_coefficients_sparse.csv"
    if not coef_path.exists():
        log(f"    WARNING: Coefficient file not found, skipping target: {coef_path}")
        raise FileNotFoundError(f"Coefficient file not found: {coef_path}")

    coef_df = load_coefficient_file(coef_path)
    max_available = len(coef_df)

    # Load baseline metrics
    try:
        # Always load baselines from the validation split: recovery is defined relative
        # to the dense model's validation performance, regardless of which split is evaluated.
        baseline_metrics = load_baseline_metrics(results_dir, target_group_name, target, split='valid')
    except (AssertionError, FileNotFoundError, KeyError) as e:
        log(f"    WARNING: Could not load baseline metrics, skipping target: {e}")
        raise ValueError(f"Could not load baseline metrics: {e}")

    dense_baseline = baseline_metrics['dense_roc_auc']
    sparse_baseline = baseline_metrics['sparse_roc_auc']
    n_nonzero = int(baseline_metrics['n_nonzero'])

    log(f"    Dense baseline ROC-AUC: {dense_baseline:.4f}")
    log(f"    Sparse baseline ROC-AUC: {sparse_baseline:.4f}")
    log(f"    n_nonzero: {n_nonzero}, max_available: {max_available}")

    # Filter out targets with poor dense baseline (same threshold as downstream classification)
    if dense_baseline < DENSE_ROC_AUC_MIN:
        log(f"    Skipping target '{target}': dense baseline ({dense_baseline:.4f}) < {DENSE_ROC_AUC_MIN}")
        log(f"    Recovery analysis requires dense model to perform above chance")
        return None

    # Determine feature counts to evaluate
    feature_counts = determine_feature_counts_to_evaluate(n_nonzero, max_available)
    log(f"    Evaluating {len(feature_counts)} feature counts: {feature_counts}")

    # Extract binary labels for this specific target
    # Uses same pattern as compute_per_target_metrics in src/downstream_clf.py
    task_type = target_group_config['type']
    if task_type == 'multiclass':
        target_idx = target_group_config['targets'].index(target)

        # Filter out unlabeled samples and create binary labels (one-vs-rest)
        train_mask = y_train >= 0
        y_train = (y_train[train_mask] == target_idx).astype(int)
        X_sparse_train_full = X_sparse_train_full[train_mask]

        val_mask = y_val >= 0
        y_val = (y_val[val_mask] == target_idx).astype(int)
        X_sparse_val_full = X_sparse_val_full[val_mask]

        test_mask = y_test >= 0
        y_test = (y_test[test_mask] == target_idx).astype(int)
        X_sparse_test_full = X_sparse_test_full[test_mask]

        # Update task type to binary for metric computation
        target_group_config = dict(target_group_config)  # Create a copy
        target_group_config['type'] = 'binary'

    elif task_type == 'multilabel':
        # Extract the specific target column from multilabel array
        target_idx = target_group_config['targets'].index(target)
        y_train = y_train[:, target_idx]
        y_val = y_val[:, target_idx]
        y_test = y_test[:, target_idx]

        # Update task type to binary for metric computation
        target_group_config = dict(target_group_config)  # Create a copy
        target_group_config['type'] = 'binary'

    # Check if both classes are present in training data (for all binary tasks)
    if target_group_config['type'] == 'binary':
        n_positive = y_train.sum()
        n_negative = len(y_train) - n_positive
        if n_positive == 0 or n_negative == 0:
            log(f"    WARNING: Target '{target}' has only one class in training data (positive={n_positive}, negative={n_negative}). Skipping.")
            return None

    # Train and evaluate for each feature count
    all_results = []

    for n_features in feature_counts:
        # Get top N features
        feature_indices = np.asarray(coef_df['feature_idx'])
        feature_indices = feature_indices[:n_features]

        # Validate feature indices
        n_features_in_full = X_sparse_train_full.shape[1]
        assert feature_indices.max() < n_features_in_full, \
            f"Feature index {feature_indices.max()} out of bounds for feature space of size {n_features_in_full}"

        # Train and evaluate
        metrics = train_and_evaluate_restricted_model(
            X_sparse_train_full, y_train,
            X_sparse_val_full, y_val,
            X_sparse_test_full, y_test,
            feature_indices,
            target_group_config
        )

        # Store results
        result: dict[str, int | float] = {'n_features': n_features}
        result.update(metrics)
        all_results.append(result)

        # Log progress
        log(f"      N={n_features:3d}: valid_roc_auc={metrics['valid_roc_auc']:.4f}")

    # Compute recovery thresholds
    recovery_thresholds = compute_recovery_thresholds(dense_baseline, all_results)

    # Find best performance
    valid_roc_aucs = [r['valid_roc_auc'] for r in all_results if not np.isnan(r['valid_roc_auc'])]
    if valid_roc_aucs:
        max_roc_auc = max(valid_roc_aucs)
        max_idx = [i for i, r in enumerate(all_results) if r['valid_roc_auc'] == max_roc_auc][0]
        n_at_max = all_results[max_idx]['n_features']
    else:
        max_roc_auc = np.nan
        n_at_max = np.nan

    # Save per-target results CSV
    recovery_csv = output_dir / f"{target_group_name}_{target}_recovery.csv"
    recovery_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(all_results).to_csv(recovery_csv, index=False)

    # Generate recovery plot
    recovery_png = output_dir / f"{target_group_name}_{target}_recovery.png"
    plot_recovery_curve(
        all_results, dense_baseline, sparse_baseline, n_nonzero,
        recovery_png, target_group_name, target
    )

    log(f"    Saved results to: {output_dir}")

    # Create summary record
    summary = {
        'base_model': base_model_name,
        'sae_config': sae_config,
        'target_group': target_group_name,
        'target': target,
        'dense_baseline_roc_auc': dense_baseline,
        'sparse_baseline_roc_auc': sparse_baseline,
        'n_nonzero': n_nonzero,
        'max_roc_auc': max_roc_auc,
        'n_at_max_roc_auc': n_at_max,
    }
    summary.update(recovery_thresholds)

    return summary



def main():
    """Main processing workflow."""

    args = parse_arguments()

    log("=" * 80)
    log("PERFORMANCE RECOVERY ANALYSIS")
    log("=" * 80)
    log(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("")

    # Select base models to process
    if args.base_model:
        base_models_to_process = {args.base_model: BASE_MODEL_PATHS[args.base_model]}
        log(f"Filtering to base model: {args.base_model}")
    else:
        base_models_to_process = BASE_MODEL_PATHS
        log(f"Processing all base models")

    log("")

    # Loop over base models
    for base_model_idx, (base_model_name, base_model_config) in enumerate(
        base_models_to_process.items(), 1
    ):
        log("=" * 80)
        log(f"BASE MODEL {base_model_idx}/{len(base_models_to_process)}: {base_model_name.upper()}")
        log("=" * 80)

        # Get SAE configs for this base model
        sae_configs = get_sae_configs(base_model_config['sparse_base'])
        log(f"Found {len(sae_configs)} SAE configuration(s)")

        # Filter SAE configs if specified
        if args.sae_config:
            sae_configs = [c for c in sae_configs if c == args.sae_config]
            if len(sae_configs) == 0:
                log(f"WARNING: SAE config '{args.sae_config}' not found, skipping base model")
                continue
            log(f"Filtering to SAE config: {args.sae_config}")

        # Loop over SAE configs
        for sae_config_idx, sae_config in enumerate(sae_configs, 1):
            log("")
            log("-" * 80)
            log(f"SAE CONFIG {sae_config_idx}/{len(sae_configs)}: {sae_config}")
            log("-" * 80)

            # Set up paths
            sparse_base = Path(base_model_config['sparse_base'])
            sparse_train_path = sparse_base / sae_config / 'total_seg_sparse_features_train.parquet'
            sparse_val_path = sparse_base / sae_config / 'total_seg_sparse_features_val.parquet'
            sparse_test_path = sparse_base / sae_config / 'total_seg_sparse_features_test.parquet'
            results_dir = Path(RESULTS_BASE_DIR) / base_model_name / sae_config
            output_dir = Path(OUTPUT_DIR) / base_model_name / sae_config

            # Validate paths exist
            if not sparse_train_path.exists():
                log(f"WARNING: Sparse features not found, skipping SAE config: {sparse_train_path}")
                continue

            # Load sparse features (once per SAE config)
            log("Loading sparse features...")
            load_start = time()

            sparse_train_dataset = TotalSegmentatorLatentFeaturesDataset(
                feature_type='sparse_features',
                set_type='split',
                parquet_path=str(sparse_train_path),
                shuffle=False,
                additional_metadata="all"
            )
            X_sparse_train_full = sparse_train_dataset.features.numpy()
            del sparse_train_dataset.features  # Free torch tensor

            sparse_val_dataset = TotalSegmentatorLatentFeaturesDataset(
                feature_type='sparse_features',
                set_type='split',
                parquet_path=str(sparse_val_path),
                shuffle=False,
                additional_metadata="all"
            )
            X_sparse_val_full = sparse_val_dataset.features.numpy()
            del sparse_val_dataset.features  # Free torch tensor

            sparse_test_dataset = TotalSegmentatorLatentFeaturesDataset(
                feature_type='sparse_features',
                set_type='split',
                parquet_path=str(sparse_test_path),
                shuffle=False,
                additional_metadata="all"
            )
            X_sparse_test_full = sparse_test_dataset.features.numpy()
            del sparse_test_dataset.features  # Free torch tensor

            gc.collect()  # Force garbage collection

            load_time = time() - load_start
            log(f"  Sparse features loaded in {load_time:.1f}s (shape: {X_sparse_train_full.shape})")

            # Determine which target groups to process
            if args.target_group:
                if args.target_group in TARGET_GROUPS:
                    target_groups_to_process = {args.target_group: TARGET_GROUPS[args.target_group]}
                    log(f"Filtering to target group: {args.target_group}")
                else:
                    log(f"WARNING: Target group '{args.target_group}' not found, skipping SAE config")
                    continue
            else:
                target_groups_to_process = TARGET_GROUPS
                log(f"Processing all target groups")

            # Process all target groups and collect summaries
            all_summaries = []

            for target_group_name, target_group_config in target_groups_to_process.items():
                log("")
                log(f"Target group: {target_group_name}")

                # Load targets for this group
                y_train = load_targets_for_group(target_group_config, sparse_train_dataset)
                y_val = load_targets_for_group(target_group_config, sparse_val_dataset)
                y_test = load_targets_for_group(target_group_config, sparse_test_dataset)

                # Subsample training data if needed
                if len(y_train) > MAX_TRAINING_SAMPLES:
                    log(f"  Subsampling training data: {len(y_train)} -> {MAX_TRAINING_SAMPLES}")
                    X_train_subsampled, y_train_subsampled = subsample_training_data(
                        X_sparse_train_full, y_train, max_samples=MAX_TRAINING_SAMPLES
                    )
                else:
                    X_train_subsampled = X_sparse_train_full
                    y_train_subsampled = y_train

                # Process each target in the group
                for target in target_group_config['targets']:
                    # Check if already complete
                    if check_target_complete(output_dir, target_group_name, target):
                        log(f"  Target {target} already complete, skipping")
                        continue

                    # Process target
                    summary = process_target(
                        base_model_name, sae_config,
                        target_group_name, target, target_group_config,
                        X_train_subsampled, X_sparse_val_full, X_sparse_test_full,
                        y_train_subsampled, y_val, y_test,
                        results_dir, output_dir
                    )

                    if summary is not None:
                        all_summaries.append(summary)

                # Free memory after each target group
                if len(y_train) > MAX_TRAINING_SAMPLES:
                    del X_train_subsampled, y_train_subsampled
                del y_train, y_val, y_test
                gc.collect()

            # Save summary table for this SAE config
            if all_summaries:
                summary_path = output_dir / "recovery_summary.csv"
                summary_path.parent.mkdir(parents=True, exist_ok=True)
                pd.DataFrame(all_summaries).to_csv(summary_path, index=False)
                log(f"\nSaved summary table: {summary_path}")
                log(f"Processed {len(all_summaries)} targets for SAE config {sae_config}")

            # Free memory
            del X_sparse_train_full, X_sparse_val_full, X_sparse_test_full
            del sparse_train_dataset, sparse_val_dataset, sparse_test_dataset
            gc.collect()

    log("")
    log("=" * 80)
    log("ANALYSIS COMPLETE")
    log(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log("=" * 80)


if __name__ == '__main__':
    main()
