"""
Fingerprint Visualization

This module provides visualization functions for sparse and dense feature-based
image retrieval using fingerprints. Fingerprints are small sets of highly
discriminative features selected from sparse or dense feature spaces.

Key visualization types:
- **Development view**: 3Nx6 grid showing sparse retrievals, dense retrievals, and fingerprint histograms
- **Manuscript view**: 3x5 compact grid comparing sparse vs dense fingerprint retrievals with inset histograms
- **Comparison plots**: Scatter and box plots comparing retrieval quality metrics

Functions:
- visualize_fingerprint_results: Development visualization (3Nx6 layout)
- visualize_fingerprint_results_manuscript: Manuscript visualization (3x5 layout)
- visualize_sparse_vs_dense_comparison: Quality comparison plots
- _plot_histogram: Private helper for histogram rendering
"""

# stdlib
from pathlib import Path

# third-party
import matplotlib
import matplotlib.axes
import matplotlib.ticker
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from PIL import Image
from scipy import stats

# local
from src.dataloading import TotalSegmentatorLatentFeaturesDataset, get_image_path
from src.fingerprint_retrieval import extract_top_k_sparse_features, extract_top_k_dense_features


# ==================================================
# Private Helpers
# ==================================================

def _plot_histogram(
    ax: matplotlib.axes.Axes,
    feature_indices: np.ndarray,
    activations: np.ndarray,
    y_max: float
) -> None:
    """
    Plot a fingerprint histogram on given axes with shared y-axis limit.

    The histogram shows feature activations sorted by feature ID. This is a
    private helper function used by the main visualization functions.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes object to plot on
    feature_indices : np.ndarray
        Feature IDs (x-axis labels)
    activations : np.ndarray
        Activation values (bar heights)
    y_max : float
        Y-axis upper limit (for consistent scaling across subplots)

    Notes
    -----
    Features are sorted by ID (ascending) for display, not by activation magnitude.
    """
    # Sort features by ID for fingerprint visualization
    sorted_order = np.argsort(feature_indices)
    sorted_indices = feature_indices[sorted_order]
    sorted_activations = activations[sorted_order]

    ax.bar(
        range(len(sorted_indices)),
        sorted_activations,
        color='steelblue',
        edgecolor='black',
        linewidth=0.5
    )
    ax.set_xlabel("feature ID", fontsize=7)
    ax.set_ylabel("activation", fontsize=7)
    ax.set_ylim(0, y_max)
    ax.set_xticks(range(len(sorted_indices)))
    ax.set_xticklabels([f"{idx}" for idx in sorted_indices], fontsize=6, rotation=45, ha="right")
    ax.tick_params(axis='y', labelsize=6)
    ax.grid(axis='y', alpha=0.3)


# ==================================================
# Main Visualization Functions
# ==================================================

def visualize_fingerprint_results(
    references: list[tuple[str, str, int, int]],
    sparse_retrieval_results: dict[tuple[str, str, int], pd.DataFrame],
    dense_retrieval_results: dict[tuple[str, str, int], pd.DataFrame],
    feature_info: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]],
    metadata_dataset: TotalSegmentatorLatentFeaturesDataset,
    sparse_features: np.ndarray,
    always_active_features: set,
    output_path: Path,
    base_model: str,
    sae_config: str
) -> None:
    """
    Create 3Nx6 figure with sparse retrievals, dense retrievals, and fingerprints.

    This is the development visualization showing detailed retrieval results.
    Each reference image gets 3 rows:
    - Row 1: Reference + 5 sparse fingerprint retrievals
    - Row 2: Reference + 5 dense embedding retrievals
    - Row 3: Fingerprint histograms for reference and sparse retrievals

    Parameters
    ----------
    references : List[Tuple[str, str, int, int]]
        List of (modality, image_id, slice_idx, array_idx) tuples for reference images
    sparse_retrieval_results : Dict[Tuple[str, str, int], pd.DataFrame]
        Dict mapping (modality, image_id, slice_idx) -> DataFrame of sparse top-N similar images
    dense_retrieval_results : Dict[Tuple[str, str, int], pd.DataFrame]
        Dict mapping (modality, image_id, slice_idx) -> DataFrame of dense top-N similar images
    feature_info : Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray]]
        Dict mapping (modality, image_id, slice_idx) -> (feature_indices, activations) for sparse fingerprint
    metadata_dataset : TotalSegmentatorLatentFeaturesDataset
        Metadata dataset for retrieving image paths
    sparse_features : np.ndarray
        (N, D) array of sparse activations for all images
    always_active_features : set
        Set of always-active feature indices to exclude
    output_path : Path
        Path where the figure will be saved
    base_model : str
        Name of base model (for title)
    sae_config : str
        SAE configuration string (for title)

    Notes
    -----
    The reference fingerprint is used for all images in row 3 - this shows
    how well retrieved images match the reference's selected features.
    """
    n_refs = len(references)
    fig, axes = plt.subplots(3 * n_refs, 6, figsize=(18, n_refs * 9))

    # Handle single reference case (axes would be 1D)
    if n_refs == 1:
        axes = axes.reshape(3, -1)

    for ref_idx, (modality, ref_id, ref_slice, ref_array_idx) in enumerate(references):
        sparse_img_row = 3 * ref_idx
        dense_img_row = 3 * ref_idx + 1
        fp_row = 3 * ref_idx + 2

        # Load reference image once for reuse
        ref_path = get_image_path(metadata_dataset, ref_array_idx)
        ref_img = Image.open(ref_path).convert("L")
        ref_img_array = np.array(ref_img)

        key = (modality, ref_id, ref_slice)

        # Reference activations - use these feature indices for all images in this row
        ref_feature_indices, ref_activations = feature_info[key]
        all_activations_in_row = [ref_activations.max()]

        # Sparse retrieved activations - use the SAME feature indices as reference
        sparse_retrievals = sparse_retrieval_results[key]
        sparse_retrieved_fingerprints = []
        for _, row in sparse_retrievals.iterrows():
            ret_array_idx = int(row["array_idx"])
            # Extract activations for the reference's features (not the retrieved image's top-K)
            ret_activations = sparse_features[ret_array_idx, ref_feature_indices]
            sparse_retrieved_fingerprints.append(ret_activations)
            all_activations_in_row.append(ret_activations.max())

        # Dense retrievals (for visualization only, no fingerprints needed)
        dense_retrievals = dense_retrieval_results[key]

        # Compute shared y-axis limit (with 10% padding)
        y_max = max(all_activations_in_row) * 1.1

        # Row 1: Sparse retrieval results
        # Column 0: Reference image
        axes[sparse_img_row, 0].imshow(ref_img_array, cmap="gray")
        axes[sparse_img_row, 0].set_title(f"Reference\n{ref_id[:20]}\nslice {ref_slice}", fontsize=8)
        axes[sparse_img_row, 0].axis("off")

        # Columns 1-5: Sparse retrieved images
        for col_idx, (_, row) in enumerate(sparse_retrievals.iterrows(), start=1):
            ret_id = row["image_id"]
            ret_slice = int(row["slice_idx"])
            ret_array_idx = int(row["array_idx"])
            similarity = float(row["similarity"])

            # Retrieved image
            ret_path = get_image_path(metadata_dataset, ret_array_idx)
            ret_img = Image.open(ret_path).convert("L")
            axes[sparse_img_row, col_idx].imshow(np.array(ret_img), cmap="gray")
            axes[sparse_img_row, col_idx].set_title(
                f"Sparse Rank {col_idx}\n{ret_id[:20]}\nsim={similarity:.3f}",
                fontsize=8
            )
            axes[sparse_img_row, col_idx].axis("off")

        # Row 2: Dense retrieval results
        # Column 0: Reference image (reused)
        axes[dense_img_row, 0].imshow(ref_img_array, cmap="gray")
        axes[dense_img_row, 0].set_title(f"Reference\n{ref_id[:20]}\nslice {ref_slice}", fontsize=8)
        axes[dense_img_row, 0].axis("off")

        # Columns 1-5: Dense retrieved images
        for col_idx, (_, row) in enumerate(dense_retrievals.iterrows(), start=1):
            ret_id = row["image_id"]
            ret_slice = int(row["slice_idx"])
            ret_array_idx = int(row["array_idx"])
            similarity = float(row["similarity"])

            # Retrieved image
            ret_path = get_image_path(metadata_dataset, ret_array_idx)
            ret_img = Image.open(ret_path).convert("L")
            axes[dense_img_row, col_idx].imshow(np.array(ret_img), cmap="gray")
            axes[dense_img_row, col_idx].set_title(
                f"Dense Rank {col_idx}\n{ret_id[:20]}\nsim={similarity:.3f}",
                fontsize=8
            )
            axes[dense_img_row, col_idx].axis("off")

        # Row 3: Fingerprint histograms
        # Reference fingerprint
        _plot_histogram(axes[fp_row, 0], ref_feature_indices, ref_activations, y_max)

        # Retrieved fingerprints (for sparse retrievals)
        for col_idx, ret_activations in enumerate(sparse_retrieved_fingerprints, start=1):
            _plot_histogram(axes[fp_row, col_idx], ref_feature_indices, ret_activations, y_max)

    plt.suptitle(
        f"Sparse Fingerprint Results ({base_model} / {sae_config})",
        fontsize=12, y=0.998
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"\nSaved visualization to {output_path}")
    plt.close()


def visualize_fingerprint_results_manuscript(
    references: list[tuple[str, str, int, int]],
    sparse_retrieval_results: dict[tuple[str, str, int], pd.DataFrame],
    dense_fingerprint_retrieval_results: dict[tuple[str, str, int], pd.DataFrame],
    sparse_feature_info: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]],
    dense_feature_info: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]],
    metadata_dataset: TotalSegmentatorLatentFeaturesDataset,
    sparse_features: np.ndarray,
    dense_features: np.ndarray,
    always_active_features: set,
    output_path: Path,
    n_retrievals: int,
    top_k_features: int,
    colors: tuple[str, str] = ("coral", "steelblue")
) -> None:
    """
    Create 3x5 figure for manuscript showing sparse fingerprint retrievals.

    Compact visualization optimized for publication. Each reference image gets one row
    with 5 columns showing reference + 4 most similar images retrieved via sparse fingerprint.
    Each image has an inset histogram in the bottom-right corner showing its sparse fingerprint.

    Layout per reference (1 row):
    [Reference | Sparse Retrieved1 | Sparse Retrieved2 | Sparse Retrieved3 | Sparse Retrieved4]

    Histogram insets:
    - Reference: Sparse fingerprint (solid)
    - Sparse retrievals: Each image's own sparse top-K features

    Parameters
    ----------
    references : List[Tuple[str, str, int, int]]
        List of (modality, image_id, slice_idx, array_idx) tuples for reference images
    sparse_retrieval_results : Dict[Tuple[str, str, int], pd.DataFrame]
        Dict mapping (modality, image_id, slice_idx) -> sparse fingerprint top-N similar images
    dense_fingerprint_retrieval_results : Dict[Tuple[str, str, int], pd.DataFrame]
        Not used (kept for API compatibility)
    sparse_feature_info : Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray]]
        Dict mapping (modality, image_id, slice_idx) -> (sparse_feature_indices, activations)
    dense_feature_info : Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray]]
        Not used (kept for API compatibility)
    metadata_dataset : TotalSegmentatorLatentFeaturesDataset
        Metadata dataset for retrieving image paths
    sparse_features : np.ndarray
        (N, D) array of sparse activations for all images
    dense_features : np.ndarray
        Not used (kept for API compatibility)
    always_active_features : set
        Set of always-active sparse feature indices to exclude
    output_path : Path
        Path where the figure will be saved
    n_retrievals : int
        Number of sparse retrieved images to show (columns 1..n_retrievals)
    top_k_features : int
        Number of top features (K) used for fingerprints
    colors : Tuple[str, str], optional
        (dense_color, sparse_color) for histogram visualization, default: ("coral", "steelblue")

    Notes
    -----
    All histograms are normalized to [0, 1] range for visual consistency.
    """
    n_refs = len(references)
    n_sparse_retrievals = n_retrievals
    n_cols = 1 + n_sparse_retrievals  # reference + N sparse retrievals

    # Dynamic figure size: ~2 inches per column, ~2 inches per row
    fig_width = n_cols * 2.0
    fig_height = n_refs * 2.0

    fig, axes = plt.subplots(n_refs, n_cols, figsize=(fig_width, fig_height))

    # Handle single reference case (axes would be 1D)
    if n_refs == 1:
        axes = axes.reshape(1, -1)

    for ref_idx, (modality, ref_id, ref_slice, ref_array_idx) in enumerate(references):
        key = (modality, ref_id, ref_slice)

        # Extract reference's own top-K sparse features
        ref_sparse_indices, ref_sparse_vals = sparse_feature_info[key]

        # Collect all sparse activations for normalization
        all_sparse_activations = [ref_sparse_vals]

        # Collect fingerprints for all images in this row
        row_fingerprints = []  # List of (col_idx, fingerprint_data)

        # Reference fingerprints (col 0: sparse only)
        row_fingerprints.append((0, (ref_sparse_indices, ref_sparse_vals)))

        # Sparse retrievals - get each image's own sparse top-K (show 4 retrievals)
        sparse_retrievals = sparse_retrieval_results[key].head(n_sparse_retrievals)
        for col_offset, (_, row) in enumerate(sparse_retrievals.iterrows()):
            col_idx = 1 + col_offset
            ret_array_idx = int(row["array_idx"])
            # Extract this image's own sparse top-K features
            ret_sparse_indices, ret_sparse_vals = extract_top_k_sparse_features(
                sparse_features, ret_array_idx, top_k_features,
                always_active_features, None
            )
            row_fingerprints.append((col_idx, (ret_sparse_indices, ret_sparse_vals)))
            all_sparse_activations.append(ret_sparse_vals)

        # Compute normalization factor for sparse activations
        sparse_max = max([vals.max() for vals in all_sparse_activations])

        # Avoid division by zero
        sparse_scale = 1.0 / sparse_max if sparse_max > 0 else 1.0

        # All histograms will now be in [0, 1] range
        y_max = 1.0

        # Column 0: Reference image
        ref_path = get_image_path(metadata_dataset, ref_array_idx)
        ref_img = Image.open(ref_path).convert("L")
        axes[ref_idx, 0].imshow(np.array(ref_img), cmap="gray")
        axes[ref_idx, 0].axis("off")

        # Columns 1-N: Sparse retrievals
        sparse_retrievals_reset = sparse_retrieval_results[key].head(n_sparse_retrievals)
        for col_offset, (_, row) in enumerate(sparse_retrievals_reset.iterrows()):
            col_idx = 1 + col_offset
            ret_array_idx = int(row["array_idx"])

            # Retrieved image
            ret_path = get_image_path(metadata_dataset, ret_array_idx)
            ret_img = Image.open(ret_path).convert("L")
            axes[ref_idx, col_idx].imshow(np.array(ret_img), cmap="gray")
            axes[ref_idx, col_idx].axis("off")

        # Add inset histogram for each image (all show sparse fingerprints)
        for col_idx, fp_data in row_fingerprints:
            # Create inset axes at bottom-right corner
            inset_ax = inset_axes(
                axes[ref_idx, col_idx],
                width="25%",  # 25% of parent width
                height="25%",  # 25% of parent height
                loc='lower right',
                borderpad=0
            )

            # All images show sparse fingerprint (normalized, sorted by feature_id)
            indices, vals = fp_data

            # Sort by feature_id
            sorted_order = np.argsort(indices)
            sorted_vals = vals[sorted_order] * sparse_scale

            # Plot sparse fingerprint
            inset_ax.bar(
                range(len(indices)),
                sorted_vals,
                color=colors[1],  # Sparse color (steelblue by default)
                alpha=1.0,
                edgecolor='none'
            )

            # Set y-limit to shared max
            inset_ax.set_ylim(0, y_max)

            # Remove all labels and ticks
            inset_ax.set_xticks([])
            inset_ax.set_yticks([])
            inset_ax.set_xticklabels([])
            inset_ax.set_yticklabels([])

            # Add subtle border to distinguish from image
            for spine in inset_ax.spines.values():
                spine.set_edgecolor('white')
                spine.set_linewidth(0.5)

    plt.subplots_adjust(wspace=0.05, hspace=0.05)
    plt.savefig(output_path, dpi=150)
    print(f"\nSaved manuscript visualization to {output_path}")
    plt.close()


def visualize_sparse_vs_dense_comparison(
    comparison_results: list,
    output_path: Path,
    base_model: str,
    sae_config: str
) -> None:
    """
    Create visualization comparing sparse vs dense retrieval quality.

    Two-panel figure:
    - Panel A: Scatter plot showing sparse quality vs dense quality per reference
    - Panel B: Box plot showing distribution of sparse vs dense quality scores

    Includes statistical testing (Wilcoxon signed-rank test) if sample size >= 3.

    Parameters
    ----------
    comparison_results : list
        List of dicts, each with 'sparse_mean' and 'dense_mean' keys containing
        mean cosine similarity scores for a reference image
    output_path : Path
        Path where the figure will be saved
    base_model : str
        Name of base model (for title)
    sae_config : str
        SAE configuration string (for title)

    Notes
    -----
    The scatter plot includes a diagonal reference line (x=y) representing
    perfect parity between methods. Points above the line indicate sparse
    fingerprint retrieval outperforms dense retrieval for that reference.
    """
    # Extract data
    sparse_scores = [r["sparse_mean"] for r in comparison_results]
    dense_scores = [r["dense_mean"] for r in comparison_results]

    # Statistical test
    if len(sparse_scores) >= 3:
        stat_result = stats.wilcoxon(sparse_scores, dense_scores)
        p_value = stat_result.pvalue  # type: ignore[attr-defined]
    else:
        p_value = np.nan

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(10, 5))

    # Panel A: Scatter plot
    ax1 = axes[0]
    ax1.scatter(dense_scores, sparse_scores, alpha=0.6, s=100, edgecolors='black', linewidths=1)

    # Diagonal reference line (x=y, perfect parity)
    min_val = min(min(sparse_scores), min(dense_scores))
    max_val = max(max(sparse_scores), max(dense_scores))
    ax1.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, linewidth=1.5, label='Perfect parity')

    ax1.set_xlabel('mean cosine similarity (dense retrieval)', fontsize=11)
    ax1.set_ylabel('mean cosine similarity (sparse retrieval)', fontsize=11)
    ax1.set_title('Sparse Fingerprint vs. Dense Features Retrieval Quality', fontsize=12, fontweight='bold')
    ax1.legend(loc='lower right')
    ax1.grid(alpha=0.3)
    ax1.set_aspect('equal', adjustable='box')

    # Add text with statistics
    mean_sparse = np.mean(sparse_scores)
    mean_dense = np.mean(dense_scores)
    relative_perf = mean_sparse / mean_dense
    stats_text = f'Similarity sparse: {mean_sparse:.3f}\n' \
                 f'Similarity dense:  {mean_dense:.3f}\n' \
                 f'Relative Diff.:    {relative_perf:.1f}%'
    if not np.isnan(p_value):
        stats_text += f'\nWilcoxon p:        {p_value:.4f}'
    ax1.text(0.05, 0.95, stats_text, transform=ax1.transAxes,
             verticalalignment='top', fontsize=9,
             bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

    # Panel B: Box plot
    ax2 = axes[1]
    box_data = [sparse_scores, dense_scores]
    bp = ax2.boxplot(box_data, tick_labels=['Sparse', 'Dense'],
                      patch_artist=True, widths=0.6)

    # Color boxes
    box_colors = ['steelblue', 'coral']
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax2.set_ylabel('Mean Cosine Similarity', fontsize=11)
    ax2.set_title('Distribution Comparison', fontsize=12, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)

    plt.suptitle(f'Sparse Fingerprint vs. Dense Retrieval ({base_model} / {sae_config})',
                 fontsize=13, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\nSaved comparison visualization to {output_path}")
    plt.close()


def visualize_k_sweep_quality_curve(
    k_sweep_summary: pd.DataFrame,
    output_path: Path,
    base_model: str,
    sae_config: str
) -> None:
    """
    Plot retrieval quality vs K (fingerprint size).

    Shows how retrieval quality scales with the number of features used in
    the fingerprint. Plots mean quality +/- std for both sparse and dense
    fingerprints across different K values.

    Parameters
    ----------
    k_sweep_summary : pd.DataFrame
        Summary DataFrame with columns: k, mean_sparse_quality, std_sparse_quality,
        mean_dense_fp_quality, std_dense_fp_quality, n_references
    output_path : Path
        Where to save the visualization
    base_model : str
        Base model name (e.g., "dinov3")
    sae_config : str
        SAE configuration string
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    k_values = k_sweep_summary['k'].to_numpy()
    sparse_mean = k_sweep_summary['mean_sparse_quality'].to_numpy()
    sparse_std = k_sweep_summary['std_sparse_quality'].to_numpy()
    dense_fp_mean = k_sweep_summary['mean_dense_fp_quality'].to_numpy()
    dense_fp_std = k_sweep_summary['std_dense_fp_quality'].to_numpy()

    # Plot sparse fingerprint quality
    ax.errorbar(k_values, sparse_mean, yerr=sparse_std,
                marker='o', linewidth=2, markersize=8,
                label='Sparse Fingerprint', color='steelblue',
                capsize=5, capthick=2, alpha=0.9)

    # Plot dense fingerprint quality
    ax.errorbar(k_values, dense_fp_mean, yerr=dense_fp_std,
                marker='s', linewidth=2, markersize=8,
                label='Dense Fingerprint', color='coral',
                capsize=5, capthick=2, alpha=0.9)

    # Formatting
    ax.set_xlabel('Number of Features (K)', fontsize=12, fontweight='bold')
    ax.set_ylabel('Mean Cosine Similarity', fontsize=12, fontweight='bold')
    ax.set_title(f'Retrieval Quality vs Fingerprint Size\n({base_model} / {sae_config})',
                 fontsize=13, fontweight='bold')

    # Use log scale for x-axis if K values span orders of magnitude
    if max(k_values) / min(k_values) >= 10:
        ax.set_xscale('log')
        ax.set_xticks(k_values.astype(float))
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    ax.legend(fontsize=11, loc='lower right')
    ax.grid(True, alpha=0.3, linestyle='--')

    # Add horizontal line at sparse quality for K=max to show plateau
    if len(k_values) > 0:
        max_k_idx = k_values.argmax()
        ax.axhline(y=sparse_mean[max_k_idx], color='steelblue',
                   linestyle=':', alpha=0.5, linewidth=1)
        ax.axhline(y=dense_fp_mean[max_k_idx], color='coral',
                   linestyle=':', alpha=0.5, linewidth=1)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"[ok] Saved K-sweep quality curve to {output_path}")
    plt.close()


# ==================================================
# Shared Inset + Language-Retrieval Manuscript
# ==================================================

def add_fingerprint_inset(
    ax: matplotlib.axes.Axes,
    feature_indices: np.ndarray,
    activations: np.ndarray,
    color: str,
    loc: str = "lower right",
) -> None:
    """
    Overlay a compact normalised fingerprint bar-chart as an inset on ax.

    Features are sorted by index before plotting so the bar order is stable
    regardless of the order they were extracted.

    Args:
        ax: Host axes to attach the inset to
        feature_indices: Feature indices (used for sort order)
        activations: Corresponding activation values (normalised to [0, 1])
        color: Bar colour
        loc: Inset anchor location ('lower left', 'lower right', etc.)
    """
    inset = inset_axes(ax, width="30%", height="30%", loc=loc, borderpad=0)

    order = np.argsort(feature_indices)
    vals = activations[order].astype(float)
    if vals.max() > 0:
        vals = vals / vals.max()

    inset.bar(range(len(vals)), vals, color=color, edgecolor="none", alpha=0.9)
    inset.set_xlim(-0.5, len(vals) - 0.5)
    inset.set_ylim(0, 1)
    inset.set_xticks([])
    inset.set_yticks([])
    for spine in inset.spines.values():
        spine.set_edgecolor("white")
        spine.set_linewidth(0.5)


def visualize_query_retrieval_manuscript(
    feature_indices: np.ndarray,
    retrieved_df: pd.DataFrame,
    sparse_features: np.ndarray,
    metadata_dataset: TotalSegmentatorLatentFeaturesDataset,
    output_path: Path,
    color: str,
) -> None:
    """
    Create a compact 1x3 manuscript figure for a language-based retrieval query.

    Shows the top-3 retrieved images, each with a fingerprint inset (bottom right)
    displaying the query feature activations for that specific image.

    Args:
        feature_indices: Feature indices used for the query fingerprint
        retrieved_df: Retrieval results DataFrame with 'array_idx' column
        sparse_features: (N, D) array of sparse activations for all images
        metadata_dataset: Dataset providing image paths
        output_path: Path to save the PNG
        color: Inset bar colour
    """
    _ORI_LABELS = {"x": "sagittal", "y": "coronal", "z": "axial"}

    n_show = min(3, len(retrieved_df))

    fig, axes = plt.subplots(
        1, 2 * n_show,
        figsize=(n_show * 3.15, 2.2),
        gridspec_kw={"width_ratios": [3.0, 1.2] * n_show},
    )

    for i in range(n_show):
        row = retrieved_df.iloc[i]
        array_idx   = int(row["array_idx"])
        orientation = str(row["slice_orientation"])

        # -- Image panel ---------------------------------------------------
        ax_img = axes[i * 2]
        img_path = get_image_path(metadata_dataset, array_idx)
        img = np.array(Image.open(img_path).convert("L"))
        if orientation == "z":
            img = np.fliplr(img)
        ax_img.imshow(img, cmap="gray")
        ax_img.axis("off")
        activations = sparse_features[array_idx, feature_indices]
        add_fingerprint_inset(ax_img, feature_indices, activations, color, "lower left")

        # -- Metadata panel ------------------------------------------------
        ax_meta = axes[i * 2 + 1]
        ax_meta.set_xlim(0, 1)
        ax_meta.set_ylim(0, 1)
        ax_meta.axis("off")
        age_val = float(row["age"]) if "age" in row.index else float("nan")
        age_str = f"{age_val:.0f}y" if not pd.isna(age_val) else "?"
        lines = [
            str(row["modality"]).upper(),
            _ORI_LABELS.get(orientation, orientation),
            str(row.get("gender", "")),
            age_str,
        ]
        for j, line in enumerate(lines):
            ax_meta.text(
                0.01, 0.95 - j * 0.12, line,
                transform=ax_meta.transAxes,
                fontsize=13, va="top", ha="left", color="black",
            )

    fig.subplots_adjust(wspace=0.04)
    fig.savefig(output_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"  Saved manuscript visualization to {output_path.name}")
    plt.close(fig)
