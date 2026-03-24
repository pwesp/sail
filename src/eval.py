"""
SAE evaluation utilities: training metrics, plots, training curve visualisation,
and LLM-as-a-judge concept validation for autointerpretation.
"""

# stdlib
import json
import random
from pathlib import Path
from typing import Any, cast

# third-party
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from numpy.typing import NDArray
from scipy import stats

# local
from src.dataloading import TotalSegmentatorLatentFeaturesDataset, TotalSegmentatorFeaturesDataModule
from src.matryoshka_sae import MatryoshkaSAELightning

# ==================================================
# Scores and metrics
# ==================================================

def wilson_score_lower_bound(n_successes: np.ndarray, n_total: np.ndarray, confidence: float = 0.95) -> np.ndarray:
    """Compute the lower bound of the Wilson score interval."""
    
    # Handle zero denominators
    valid: NDArray[np.bool_] = np.asarray(n_total > 0, dtype=bool)
    result = np.zeros_like(n_successes, dtype=float)
    
    if not np.any(valid):
        return result
    
    # Calculate z-score once
    z = float(stats.norm.ppf((1 + confidence) / 2))
    z_squared = z ** 2
    
    # Vectorized calculations (only where valid)
    n_s = n_successes[valid].astype(float)
    n_t = n_total[valid].astype(float)
    
    p_hat = n_s / n_t
    
    denominator = 1 + z_squared / n_t
    center = (p_hat + z_squared / (2 * n_t)) / denominator
    margin = z * np.sqrt((p_hat * (1 - p_hat) / n_t + z_squared / (4 * n_t**2))) / denominator
    
    result[valid] = center - margin
    
    return result

def compute_composite_score(
    wilson_score: np.ndarray,
    recall: np.ndarray,
    lift: np.ndarray,
    n_active: np.ndarray,
    wilson_weight: float = 0.4,
    recall_weight: float = 0.3,
    lift_weight: float = 0.2,
    min_support: int = 10
    ) -> np.ndarray:
    """Combine multiple signals for better feature ranking."""
    
    # Normalize metrics to [0, 1]
    wilson_norm = (wilson_score - wilson_score.min()) / (wilson_score.max() - wilson_score.min() + 1e-10)
    recall_norm = recall  # Already in [0, 1]
    lift_norm = np.clip(lift / np.percentile(lift, 95), 0, 1)  # Cap outliers
    
    # Penalize features with very low support
    min_support = 10  # Require at least 10 activations
    support_penalty = np.clip(n_active / min_support, 0, 1)
    
    composite = (wilson_weight * wilson_norm + 
                 recall_weight * recall_norm + 
                 lift_weight * lift_norm)
    composite = composite * support_penalty
    
    return composite

# ==================================================
# Training evaluation
# ==================================================

def plot_matryoshka_sae_training_curves(log_dir: str, dictionary_sizes: list[int]) -> None:
    """
    Generate and display training diagnostics for Matryoshka SAE using logged metrics from PyTorch Lightning.

    Plots include:
    - Total and reconstruction loss curves for each hierarchical dictionary level
    - Sparsity (L0) trends across levels and epochs
    - Feature utilization rates and proportion of dead (unused) features
    - Evolution of the L0 norm for model interpretability

    Args:
        log_dir: Path to directory containing Lightning CSV metrics (metrics.csv)
        dictionary_sizes: List specifying the size of each dictionary level (for plotting and legend)
    """
    # Check if metrics.csv exists
    metrics_path = Path(log_dir) / "metrics.csv"
    if not metrics_path.exists():
        raise FileNotFoundError(f"metrics.csv not found at {metrics_path}")
    
    # Read metrics.csv
    df = pd.read_csv(metrics_path)
    
    # Extract training and validation metrics (use step-level data for smoother curves)
    train_metrics = df[df['train/loss'].notna()].copy()
    val_metrics = df[df['val/loss'].notna()].copy()
    
    # Extract epoch-level metrics (like correlation) separately
    # Train correlation metrics are logged at epoch end and merged with validation rows by pandas
    train_epoch_metrics = df[df['train/mean_abs_corr_level_0'].notna()].copy()
    val_epoch_metrics = val_metrics.copy()  # Val metrics are already epoch-level
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)
    
    fig.suptitle('Matryoshka SAE Training Curves', fontsize=16, fontweight='bold', y=0.995)
    
    # ------------------------------------------------------------------
    # Plot 1: Total Loss (Reconstruction + Sparsity)
    # ------------------------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    
    if 'epoch' in train_metrics.columns:
        ax1.plot(train_metrics['epoch'], train_metrics['train/loss'], 
                alpha=0.7, label='train', linewidth=2, color='cornflowerblue')
    
    if 'epoch' in val_metrics.columns and len(val_metrics) > 0:
        ax1.plot(val_metrics['epoch'], val_metrics['val/loss'], 
                marker='o', markersize=4, label='val', linewidth=2, color='orangered')
        
        val_loss_series = cast(pd.Series, val_metrics['val/loss'])
        best_idx = val_loss_series.idxmin()
        best_epoch = val_metrics.loc[best_idx, 'epoch']
        ax1.axvline(x=best_epoch, color='k', linestyle='--', alpha=0.4, linewidth=1.5, label=f'Best epoch: {int(best_epoch):d}')
    
    ax1.set_yscale('log')
    ax1.set_xlabel('epoch', fontsize=11)
    ax1.set_ylabel('loss', fontsize=11)
    ax1.set_title('Total Loss (Reconstruction + Sparsity)', fontsize=12, fontweight='bold')
    ax1.legend(fontsize=10, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    # ------------------------------------------------------------------
    # Plot 2: Reconstruction Loss
    # ------------------------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    
    if 'epoch' in train_metrics.columns and 'train/reconstruction_loss' in train_metrics.columns:
        ax2.plot(train_metrics['epoch'], train_metrics['train/reconstruction_loss'], 
        alpha=0.7, label='train', linewidth=2, color='cornflowerblue')
    if 'epoch' in val_metrics.columns and 'val/reconstruction_loss' in val_metrics.columns:
        ax2.plot(val_metrics['epoch'], val_metrics['val/reconstruction_loss'], 
        marker='o', markersize=4, label='val', linewidth=2, color='orangered')
    
    ax2.set_yscale('log')
    ax2.set_xlabel('epoch', fontsize=11)
    ax2.set_ylabel('MSE', fontsize=11)
    ax2.set_title('Reconstruction Loss (MSE)', fontsize=12, fontweight='bold')
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(True, alpha=0.3)
    
    # ------------------------------------------------------------------
    # Plot 3: Monosemanticity (Mean Absolute Correlation)
    # ------------------------------------------------------------------
    ax3 = fig.add_subplot(gs[0, 2])
    
    colors = matplotlib.colormaps['viridis'](np.linspace(0, 0.9, len(dictionary_sizes)))
    
    has_monosem_data = False
    
    # Plot training data (dashed lines) - use epoch-level metrics
    if 'epoch' in train_epoch_metrics.columns and len(train_epoch_metrics) > 0:
        for level_idx, dict_size in enumerate(dictionary_sizes):
            col_name = f'train/mean_abs_corr_level_{level_idx}'
            if col_name in train_epoch_metrics.columns:
                corr_data = train_epoch_metrics.loc[:, ['epoch', col_name]].dropna()
                if len(corr_data) > 0:
                    ax3.plot(corr_data['epoch'], corr_data[col_name], 
                            linestyle='--', alpha=0.7, label=f'Level {level_idx} (d={dict_size}) train', 
                            linewidth=2, color=colors[level_idx])
                    has_monosem_data = True
    
    # Plot validation data (solid lines with markers)
    if 'epoch' in val_epoch_metrics.columns and len(val_epoch_metrics) > 0:
        for level_idx, dict_size in enumerate(dictionary_sizes):
            col_name = f'val/mean_abs_corr_level_{level_idx}'
            if col_name in val_epoch_metrics.columns:
                corr_data = val_epoch_metrics.loc[:, ['epoch', col_name]].dropna()
                if len(corr_data) > 0:
                    ax3.plot(corr_data['epoch'], corr_data[col_name], 
                            marker='o', markersize=4, label=f'Level {level_idx} (d={dict_size}) val', 
                            linewidth=2, color=colors[level_idx])
                    has_monosem_data = True
    
    if not has_monosem_data:
        ax3.text(0.5, 0.5, 'No monosemanticity data',
                transform=ax3.transAxes, fontsize=14, fontweight='bold',
                ha='center', va='center', color='darkred')
        ax3.set_facecolor('#fff9e6')  # Light yellow background
    else:
        ax3.legend(fontsize=8, loc='best', ncol=1)
        ax3.set_ylim(bottom=0)  # Start y-axis at 0
    
    ax3.set_xlabel('epoch', fontsize=11)
    ax3.set_ylabel('Mean |Correlation|', fontsize=11)
    ax3.set_title('Monosemanticity (Lower = Better)', fontsize=12, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # ------------------------------------------------------------------
    # Plot 4: L0 Norm (Active Features per Sample)
    # ------------------------------------------------------------------
    ax4 = fig.add_subplot(gs[1, 0])
    
    if 'epoch' in val_metrics.columns:
        colors = matplotlib.colormaps['viridis'](np.linspace(0, 0.9, len(dictionary_sizes)))
        
        has_l0_data = False
        for level_idx, dict_size in enumerate(dictionary_sizes):
            col_name = f'val/l0_level_{level_idx}'
            if col_name in val_metrics.columns:
                col_data = val_metrics.loc[:, ['epoch', col_name]].dropna().copy()
                if len(col_data) > 0:
                    has_l0_data = True
                    ax4.plot(col_data['epoch'], col_data[col_name],
                            label=f'L{level_idx+1} ({dict_size:,})',
                            linewidth=2, color=colors[level_idx], 
                            marker='o', markersize=3, alpha=0.8)
        
        if has_l0_data:
            ax4.set_xlabel('epoch', fontsize=11)
            ax4.set_ylabel('L0 norm', fontsize=11)
            ax4.set_title('Active Features per Sample', fontsize=12, fontweight='bold')
            ax4.legend(fontsize=10, loc='best')
            ax4.grid(True, alpha=0.3)
        else:
            ax4.text(0.5, 0.5, 'L0 norm not logged',
                    transform=ax4.transAxes, fontsize=11,
                    ha='center', va='center', color='gray', style='italic')
            ax4.set_title('Active Features per Sample', fontsize=12, fontweight='bold')
    else:
        ax4.text(0.5, 0.5, 'No validation data',
                transform=ax4.transAxes, fontsize=11,
                ha='center', va='center', color='gray', style='italic')
        ax4.set_title('Active Features per Sample', fontsize=12, fontweight='bold')
    
    # ------------------------------------------------------------------
    # Plot 5: Feature Utilization (% Non-Dead Features)
    # ------------------------------------------------------------------
    ax5 = fig.add_subplot(gs[1, 1])
    
    if 'epoch' in val_metrics.columns:
        colors = matplotlib.colormaps['viridis'](np.linspace(0, 0.9, len(dictionary_sizes)))
        
        has_usage_data = False
        for level_idx, dict_size in enumerate(dictionary_sizes):
            col_name = f'val/feature_usage_level_{level_idx}'
            if col_name in val_metrics.columns:
                col_data = val_metrics.loc[:, ['epoch', col_name]].dropna().copy()
                if len(col_data) > 0:
                    has_usage_data = True
                    # Convert to percentage
                    usage_pct = col_data[col_name] * 100
                    ax5.plot(col_data['epoch'], usage_pct,
                            label=f'L{level_idx+1} ({dict_size:,})',
                            linewidth=2, color=colors[level_idx], 
                            marker='o', markersize=3, alpha=0.8)
        
        if has_usage_data:
            # Add danger zones with labels in legend
            ax5.axhline(y=50, color='orange', linestyle='--', alpha=0.5, linewidth=1.5, 
                       label='50%')
            ax5.axhline(y=25, color='red', linestyle='--', alpha=0.5, linewidth=1.5,
                       label='25%')
            ax5.set_ylim(0, 105)
            ax5.legend(fontsize=10, loc='best', ncol=2)
        else:
            ax5.text(0.5, 0.5, 'Feature usage not logged',
                    transform=ax5.transAxes, fontsize=11,
                    ha='center', va='center', color='gray', style='italic')
    
    ax5.set_xlabel('epoch', fontsize=11)
    ax5.set_ylabel('# active features / (%)', fontsize=11)
    ax5.set_title('Feature Utilization', fontsize=12, fontweight='bold')
    ax5.grid(True, alpha=0.3)
    
    # ------------------------------------------------------------------
    # Plot 6: Dead Features Count by Level
    # ------------------------------------------------------------------
    ax6 = fig.add_subplot(gs[1, 2])
    
    if 'epoch' in val_metrics.columns:
        colors = matplotlib.colormaps['viridis'](np.linspace(0, 0.9, len(dictionary_sizes)))
        
        has_dead_data = False
        for level_idx, dict_size in enumerate(dictionary_sizes):
            col_name = f'val/dead_features_level_{level_idx}'
            if col_name in val_metrics.columns:
                col_data = val_metrics.loc[:, ['epoch', col_name]].dropna().copy()
                if len(col_data) > 0:
                    has_dead_data = True
                    ax6.plot(col_data['epoch'], col_data[col_name],
                            label=f'L{level_idx+1} ({dict_size:,})',
                            linewidth=2, color=colors[level_idx], 
                            marker='o', markersize=3, alpha=0.8)
                    
                    # Add horizontal line showing half the dictionary size (50% dead)
                    ax6.axhline(y=dict_size*0.5, color=colors[level_idx], 
                              linestyle=':', alpha=0.3, linewidth=1)
        
        if has_dead_data:
            ax6.legend(fontsize=10, loc='best')
        else:
            ax6.text(0.5, 0.5, 'Dead features not logged',
                    transform=ax6.transAxes, fontsize=11,
                    ha='center', va='center', color='gray', style='italic')
    
    ax6.set_xlabel('epoch', fontsize=11)
    ax6.set_ylabel('# dead features', fontsize=11)
    ax6.set_title('Dead Features', fontsize=12, fontweight='bold')
    ax6.grid(True, alpha=0.3)

    # ------------------------------------------------------------------
    # Plot 7: Training Threshold Estimate
    # ------------------------------------------------------------------
    ax7 = fig.add_subplot(gs[2, 0])
    
    if 'train/threshold_estimate' in train_metrics.columns:
        # Get threshold estimate data (logged at step level, every 100 batches)
        threshold_data = train_metrics.loc[:, ['step', 'train/threshold_estimate']].dropna().copy()
        
        if len(threshold_data) > 0:
            ax7.plot(threshold_data['step'], threshold_data['train/threshold_estimate'],
                    linewidth=2, color='purple', marker='o', markersize=3, alpha=0.8)
            ax7.set_xlabel('step', fontsize=11)
            ax7.set_ylabel('threshold estimate', fontsize=11)
            ax7.set_title('Inference Threshold Estimate', fontsize=11, fontweight='bold')
            ax7.grid(True, alpha=0.3)

        else:
            ax7.text(0.5, 0.5, 'Threshold estimate not logged',
                    transform=ax7.transAxes, fontsize=11,
                    ha='center', va='center', color='gray', style='italic')
            ax7.set_title('Inference Threshold Estimate', fontsize=11, fontweight='bold')
    else:
        ax7.text(0.5, 0.5, 'Threshold estimate not logged',
                transform=ax7.transAxes, fontsize=11,
                ha='center', va='center', color='gray', style='italic')
        ax7.set_title('Inference Threshold Estimate', fontsize=11, fontweight='bold')
    
    ax7.grid(True, alpha=0.3)
    
    # ------------------------------------------------------------------
    # Plot 8: Final Validation R^2 Score
    # ------------------------------------------------------------------
    ax8 = fig.add_subplot(gs[2, 1])
    
    if 'epoch' in val_metrics.columns and len(val_metrics) > 0:
        colors = matplotlib.colormaps['viridis'](np.linspace(0, 0.9, len(dictionary_sizes)))
        
        # Get final validation R^2 per level
        final_r2_scores = []
        for level_idx in range(len(dictionary_sizes)):
            col_name = f'val/r2_level_{level_idx}'
            if col_name in val_metrics.columns:
                col_series = pd.Series(val_metrics[col_name])
                col_data = col_series.dropna()
                if len(col_data) > 0:
                    final_r2 = float(col_data.iloc[-1])
                    final_r2_scores.append(final_r2)
                else:
                    final_r2_scores.append(None)
            else:
                final_r2_scores.append(None)
        
        # Plot if we have data
        valid_r2 = [(i, r2) for i, r2 in enumerate(final_r2_scores) if r2 is not None]
        if valid_r2:
            indices, r2_vals = zip(*valid_r2)
            bars = ax8.bar(indices, r2_vals, color=[colors[i] for i in indices], 
                          alpha=0.7, edgecolor='black', linewidth=1.5)
            
            # Add quality thresholds
            ax8.axhline(y=0.90, color='orange', linestyle='--', alpha=0.4, linewidth=1.5,
                       label='0.90')
            ax8.axhline(y=0.95, color='green', linestyle='--', alpha=0.4, linewidth=1.5,
                       label='0.95')
            ax8.axhline(y=0.99, color='blue', linestyle='--', alpha=0.4, linewidth=1.5,
                       label='0.99')
            
            ax8.set_xlabel('dictionary level', fontsize=11)
            ax8.set_ylabel('R^2 Score', fontsize=11)
            ax8.set_title('Final Validation R^2 Score', fontsize=11, fontweight='bold')
            ax8.set_xticks(indices)
            ax8.set_xticklabels([f'L{i+1}\n{dictionary_sizes[i]:,}' for i in indices], 
                               fontsize=9)
            ax8.set_ylim((min(r2_vals) - 0.05, 1.0))  # Dynamic lower bound
            ax8.grid(True, alpha=0.3, axis='y')
            ax8.legend(fontsize=9, loc='best')
        else:
            ax8.text(0.5, 0.5, 'R^2 scores not logged',
                    transform=ax8.transAxes, fontsize=11,
                    ha='center', va='center', color='gray', style='italic')
            ax8.set_title('Final Validation R^2 Score', fontsize=11, fontweight='bold')
    else:
        ax8.text(0.5, 0.5, 'No validation data',
                transform=ax8.transAxes, fontsize=11,
                ha='center', va='center', color='gray', style='italic')
        ax8.set_title('Final Validation R^2 Score', fontsize=11, fontweight='bold')

    # ------------------------------------------------------------------
    # Plot 9: Validation Reconstruction Loss Evolution
    # ------------------------------------------------------------------
    ax9 = fig.add_subplot(gs[2, 2])
    
    if 'epoch' in val_metrics.columns:
        colors = matplotlib.colormaps['viridis'](np.linspace(0, 0.9, len(dictionary_sizes)))
        
        has_recon_data = False
        # Plot reconstruction loss evolution for each level
        for level_idx, dict_size in enumerate(dictionary_sizes):
            col_name = f'val/recon_loss_level_{level_idx}'
            if col_name in val_metrics.columns:
                col_data = val_metrics.loc[:, ['epoch', col_name]].dropna().copy()
                if len(col_data) > 0:
                    has_recon_data = True
                    ax9.plot(col_data['epoch'], col_data[col_name],
                            label=f'L{level_idx+1} ({dict_size:,})',
                            linewidth=2, color=colors[level_idx], 
                            marker='o', markersize=3, alpha=0.8)
        
        if has_recon_data:
            ax9.set_xlabel('epoch', fontsize=11)
            ax9.set_ylabel('MSE loss', fontsize=11)
            ax9.set_title('Validation Reconstruction Loss Evolution', fontsize=11, fontweight='bold')
            ax9.legend(fontsize=10, loc='best')
            ax9.set_yscale('log')
            ax9.grid(True, alpha=0.3)
        else:
            ax9.text(0.5, 0.5, 'Reconstruction loss not logged',
                    transform=ax9.transAxes, fontsize=11,
                    ha='center', va='center', color='gray', style='italic')
            ax9.set_title('Validation Reconstruction Loss Evolution', fontsize=11, fontweight='bold')
    else:
        ax9.text(0.5, 0.5, 'No validation data',
                transform=ax9.transAxes, fontsize=11,
                ha='center', va='center', color='gray', style='italic')
        ax9.set_title('Validation Reconstruction Loss Evolution', fontsize=11, fontweight='bold')

    # Save figure
    output_path = Path(log_dir) / "training_curves.png"
    plt.savefig(output_path, dpi=200, facecolor='white', bbox_inches='tight')
    print(f"\n[ok] Training curves saved to: {output_path}")
    plt.close()

# ==================================================
# Feature-target association
# ==================================================

def create_binary_target_labels(
    target_name: str,
    dataset: TotalSegmentatorLatentFeaturesDataset,
    verbose: bool = True
) -> tuple[np.ndarray, str]:    
    """
    Create binary target labels for a given target name.
    """
    # Modality
    if target_name in dataset.modalities:
        if verbose:
            print(f"Target {target_name} is a modality")
        target_mask = (dataset.modalities == target_name).astype(bool)
        target_type = 'modality'

    # KVP
    elif target_name in np.unique(dataset.metadata['kvp']).astype(str).tolist():
        if verbose:
            print(f"Target {target_name} is a kvp")
        target_mask = (dataset.metadata['kvp'] == float(target_name)).astype(bool)
        target_type = 'kvp'

    # Magnetic field strength
    elif target_name in np.unique(dataset.metadata['magnetic_field_strength']).tolist():
        if verbose:
            print(f"Target {target_name} is a magnetic field strength")
        target_mask = (dataset.metadata['magnetic_field_strength'] == target_name).astype(bool)
        target_type = 'magnetic_field_strength'

    # Manufacturer
    elif target_name in pd.Series(dataset.metadata['manufacturer']).dropna().unique().tolist():
        if verbose:
            print(f"Target {target_name} is a manufacturer")
        target_mask = (dataset.metadata['manufacturer'] == target_name).astype(bool)
        target_type = 'manufacturer'

    # Scanner model
    elif target_name in pd.Series(dataset.metadata['scanner_model']).dropna().unique().tolist():
        if verbose:
            print(f"Target {target_name} is a scanner model")
        target_mask = (dataset.metadata['scanner_model'] == target_name).astype(bool)
        target_type = 'scanner_model'

    # Institute
    elif target_name in pd.Series(dataset.metadata['institute']).dropna().unique().tolist():
        if verbose:
            print(f"Target {target_name} is an institute")
        target_mask = (dataset.metadata['institute'] == target_name).astype(bool)
        target_type = 'institute'

    # Slice orientation
    elif target_name in np.unique(dataset.slice_orientations).tolist():
        if verbose:
            print(f"Target {target_name} is a slice orientation")
        target_mask = (dataset.slice_orientations == target_name).astype(bool)
        target_type = 'slice_orientation'

    # Patient age
    elif target_name in np.unique(dataset.metadata['age_bin']).tolist():
        if verbose:
            print(f"Target {target_name} is a patient age bin")
        target_mask = (dataset.metadata['age_bin'] == target_name).astype(bool)
        target_type = 'age_bin'

    # Patient sex
    elif target_name in pd.Series(dataset.metadata['gender']).dropna().unique().tolist():
        if verbose:
            print(f"Target {target_name} is a patient sex")
        target_mask = (dataset.metadata['gender'] == target_name).astype(bool)
        target_type = 'gender'

    # Pathology
    elif target_name in pd.Series(dataset.metadata['pathology']).dropna().unique().tolist():
        if verbose:
            print(f"Target {target_name} is a pathology")
        target_mask = (dataset.metadata['pathology'] == target_name).astype(bool)
        target_type = 'pathology'

    # Special organ targets
    elif target_name == "clavicula":
        if verbose:
            print(f"Target {target_name} is a clavicula")
        clavicula_columns = [
            'clavicula_left', 'clavicula_right'
        ]
        clavicula_data = np.column_stack([dataset.metadata[col] for col in clavicula_columns])
        target_mask = np.any(clavicula_data, axis=1).astype(bool)
        target_type = 'organ'

    elif target_name == "kidney":
        print(f"Target {target_name} is a kidney")
        kidney_columns = [
            'kidney_cyst_left', 'kidney_cyst_right',
            'kidney_left', 'kidney_right'
        ]
        kidney_data = np.column_stack([dataset.metadata[col] for col in kidney_columns])
        target_mask = np.any(kidney_data, axis=1).astype(bool)
        target_type = 'organ'
    
    elif target_name == "lung":
        if verbose:
            print(f"Target {target_name} is a lung")
        lung_columns = [
            'lung_left', 'lung_lower_lobe_left', 'lung_lower_lobe_right',
            'lung_middle_lobe_right', 'lung_right', 'lung_upper_lobe_left',
            'lung_upper_lobe_right'
        ]
        lung_data = np.column_stack([dataset.metadata[col] for col in lung_columns])
        target_mask = np.any(lung_data, axis=1).astype(bool)
        target_type = 'organ'

    elif target_name == "vertebrae":
        if verbose:
            print(f"Target {target_name} is a vertebrae")
        vertebrae_columns = [
            'vertebrae', 'vertebrae_C1', 'vertebrae_C2', 'vertebrae_C3',
            'vertebrae_C4', 'vertebrae_C5', 'vertebrae_C6', 'vertebrae_C7',
            'vertebrae_L1', 'vertebrae_L2', 'vertebrae_L3', 'vertebrae_L4',
            'vertebrae_L5', 'vertebrae_S1', 'vertebrae_T1', 'vertebrae_T10',
            'vertebrae_T11', 'vertebrae_T12', 'vertebrae_T2', 'vertebrae_T3',
            'vertebrae_T4', 'vertebrae_T5', 'vertebrae_T6', 'vertebrae_T7',
            'vertebrae_T8', 'vertebrae_T9'
        ]
        vertebrae_data = np.column_stack([dataset.metadata[col] for col in vertebrae_columns])
        target_mask = np.any(vertebrae_data, axis=1).astype(bool)
        target_type = 'vertebrae'

    # Organs
    elif target_name in dataset.included_organs:
        if verbose:
            print(f"Target {target_name} is an organ")
        target_mask = dataset.metadata[target_name].astype(bool)
        target_type = 'organ'

    else:
        raise ValueError(f"Target {target_name} is not a valid target")

    return target_mask, target_type # type: ignore


# ==================================================
# Concept evaluation (LLM-as-a-judge)
# ==================================================

def sample_distractor_features(
    sparse_features: np.ndarray,
    target_feature_idx: int,
    always_active_features: set[int],
    n_distractors: int = 4,
    min_activations: int = 10,
    random_seed: int | None = None
) -> list[int]:
    """
    Sample random distractor features for contrastive evaluation.

    Distractors are features that:
    - Are not the target feature
    - Are not always-active
    - Have activation > 0 in at least min_activations samples
    - Are randomly selected

    Args:
        sparse_features: Sparse feature matrix (n_samples x n_features)
        target_feature_idx: Index of target feature to exclude
        always_active_features: Set of always-active feature indices to exclude
        n_distractors: Number of distractor features to sample (default: 4)
        min_activations: Minimum number of activations required (default: 10)
        random_seed: Random seed for reproducibility (optional)

    Returns:
        List of distractor feature indices

    Raises:
        ValueError: If not enough eligible features available
    """
    if random_seed is not None:
        random.seed(random_seed)

    n_features = sparse_features.shape[1]

    eligible_features = []

    for feat_idx in range(n_features):
        if feat_idx == target_feature_idx:
            continue
        if feat_idx in always_active_features:
            continue
        n_activations = (sparse_features[:, feat_idx] > 0).sum()
        if n_activations >= min_activations:
            eligible_features.append(feat_idx)

    if len(eligible_features) < n_distractors:
        raise ValueError(
            f"Not enough eligible features for distractors. "
            f"Need {n_distractors}, found {len(eligible_features)}. "
            f"Try reducing min_activations (current: {min_activations})"
        )

    return random.sample(eligible_features, n_distractors)


class ConceptCache:
    """
    Cache for generated concepts to avoid redundant VLM calls.

    When a feature appears as a distractor multiple times, we can reuse its concept.
    """

    def __init__(self, cache_path: Path | None = None):
        """
        Initialize concept cache.

        Args:
            cache_path: Path to cache file (optional, for persistence)
        """
        self.cache: dict[int, str] = {}
        self.cache_path = cache_path

        if cache_path and cache_path.exists():
            self.load()

    def get(self, feature_idx: int) -> str | None:
        """Get cached concept for feature, or None if not cached."""
        return self.cache.get(feature_idx)

    def set(self, feature_idx: int, concept: str):
        """Cache concept for feature."""
        self.cache[feature_idx] = concept

    def has(self, feature_idx: int) -> bool:
        """Check if feature concept is cached."""
        return feature_idx in self.cache

    def save(self):
        """Save cache to disk (if cache_path provided)."""
        if self.cache_path:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, 'w') as f:
                json.dump({str(k): v for k, v in self.cache.items()}, f, indent=2)

    def load(self):
        """Load cache from disk (if cache_path provided and exists)."""
        if self.cache_path and self.cache_path.exists():
            with open(self.cache_path, 'r') as f:
                data = json.load(f)
                self.cache = {int(k): v for k, v in data.items()}

    def __len__(self):
        """Return number of cached concepts."""
        return len(self.cache)


def compute_evaluation_metrics(results: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Compute aggregate evaluation metrics from individual results.

    Args:
        results: List of result dicts with 'true_concept_rank' and 'is_successful'

    Returns:
        Dict with evaluation metrics:
        - success_rate: Proportion of features with rank <= 2
        - mean_rank: Average rank position
        - median_rank: Median rank position
        - rank_distribution: Count of features at each rank (1-5)
        - n_evaluated: Number of features evaluated
        - n_parse_failures: Number of ranking parse failures
    """
    if not results:
        return {
            'success_rate': 0.0,
            'mean_rank': 0.0,
            'median_rank': 0.0,
            'rank_distribution': {1: 0, 2: 0, 3: 0, 4: 0, 5: 0},
            'n_evaluated': 0,
            'n_parse_failures': 0
        }

    ranks = [r['true_concept_rank'] for r in results]
    successes = [r['is_successful'] for r in results]
    parse_failures = [r for r in results if not r.get('parse_success', True)]

    rank_distribution = {i: 0 for i in range(1, 6)}
    for rank in ranks:
        if 1 <= rank <= 5:
            rank_distribution[rank] += 1

    return {
        'success_rate': sum(successes) / len(successes),
        'mean_rank': np.mean(ranks),
        'median_rank': np.median(ranks),
        'rank_distribution': rank_distribution,
        'n_evaluated': len(results),
        'n_parse_failures': len(parse_failures)
    }


def format_evaluation_summary(metrics: dict[str, Any]) -> str:
    """
    Format evaluation metrics as a readable summary string.

    Args:
        metrics: Dict from compute_evaluation_metrics()

    Returns:
        Formatted summary string
    """
    lines = [
        "=" * 70,
        "EVALUATION SUMMARY",
        "=" * 70,
        f"Features evaluated: {metrics['n_evaluated']}",
        f"Success rate (rank <= 2): {metrics['success_rate']:.1%}",
        f"Mean rank: {metrics['mean_rank']:.2f}",
        f"Median rank: {metrics['median_rank']:.1f}",
        "",
        "Rank distribution:",
    ]

    for rank in range(1, 6):
        count = metrics['rank_distribution'][rank]
        pct = count / metrics['n_evaluated'] * 100 if metrics['n_evaluated'] > 0 else 0
        bar = "#" * int(pct / 2)
        lines.append(f"  Rank {rank}: {count:3d} ({pct:5.1f}%) {bar}")

    if metrics['n_parse_failures'] > 0:
        lines.append("")
        lines.append(f"WARNING: Parse failures: {metrics['n_parse_failures']}")

    lines.append("=" * 70)

    return '\n'.join(lines)
