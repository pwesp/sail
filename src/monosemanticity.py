"""Monosemanticity scoring for sparse SAE features."""

# stdlib
from pathlib import Path
from typing import cast

# third-party
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# local
from src.config_downstream import TARGET_GROUPS
from src.dataloading import TotalSegmentatorLatentFeaturesDataset


# ==================================================
# Config discovery
# ==================================================


def discover_all_configs(sparse_features_root: Path, sparse_features_dir_map: dict[str, str]) -> list[tuple[str, str]]:
    """Scan sparse_features_root two levels deep and return all (base_model, sae_config) pairs.

    Only base_model directories whose mapped name exists in sparse_features_root
    are included; everything else is silently skipped.
    """
    configs: list[tuple[str, str]] = []
    # Create reverse mapping: sparse_features_dir_name -> base_model
    reverse_map: dict[str, str] = {v: k for k, v in sparse_features_dir_map.items()}

    for sparse_features_dir in sorted(sparse_features_root.iterdir()):
        if not sparse_features_dir.is_dir():
            continue
        if sparse_features_dir.name not in reverse_map:
            continue
        base_model: str = reverse_map[sparse_features_dir.name]
        for sae_config_dir in sorted(sparse_features_dir.iterdir()):
            if not sae_config_dir.is_dir():
                continue
            configs.append((base_model, sae_config_dir.name))
    return configs


# ==================================================
# Feature-level helpers
# ==================================================


def get_dict_level(feature_idx: int, dict_sizes: list[int]) -> int:
    """Return the Matryoshka dictionary level for a feature index.

    Level boundaries are cumulative: level 0 is [0, dict_sizes[0]),
    level 1 is [dict_sizes[0], dict_sizes[1]), etc.
    """
    for level, boundary in enumerate(dict_sizes):
        if feature_idx < boundary:
            return level
    assert False, f"Feature index {feature_idx} >= largest dict size {dict_sizes[-1]}"


def get_top_activating_samples(
    feature_idx: int,
    sparse_features: np.ndarray,
    metadata_df: pd.DataFrame,
    dict_sizes: list[int],
    top_n: int,
    activation_threshold: float,
) -> dict[str, object]:
    """
    Retrieve top-N samples with highest activation for a feature.

    Returns dict with:
        - feature_idx: the feature index
        - dict_level: Matryoshka dictionary level (0 = most general)
        - n_nonzero_activations: how many samples have activation > threshold
        - n_retrieved: how many samples were actually retrieved (at most top_n, one per image)
        - sample_indices: indices into the original dataset
        - activations: corresponding activation values
        - metadata: DataFrame of metadata for the retrieved samples
    """
    activations: np.ndarray = sparse_features[:, feature_idx]

    n_nonzero = int((activations > activation_threshold).sum())

    # Sort descending by activation, then walk the list keeping only the first
    # (highest-activation) sample per image_id.  Neighboring slices in the same
    # CT/MRI volume can be near-identical; retaining more than one per image
    # would artificially inflate the apparent consistency of a feature.
    sorted_indices: np.ndarray = np.argsort(activations)[::-1]
    seen_images: set[object] = set()
    top_indices: list[int] = []
    for idx in sorted_indices:
        if activations[idx] <= activation_threshold:
            break  # remaining activations are all below threshold
        img_id = metadata_df.iloc[idx]["image_id"]
        if img_id not in seen_images:
            seen_images.add(img_id)
            top_indices.append(int(idx))
            if len(top_indices) == top_n:
                break

    top_indices_arr: np.ndarray = np.array(top_indices, dtype=np.intp)
    top_activations: np.ndarray = activations[top_indices_arr]

    top_metadata: pd.DataFrame = metadata_df.iloc[top_indices_arr].copy()
    top_metadata["activation"] = top_activations

    return {
        "feature_idx": feature_idx,
        "dict_level": get_dict_level(feature_idx, dict_sizes),
        "n_nonzero_activations": n_nonzero,
        "n_retrieved": len(top_indices),
        "sample_indices": top_indices,
        "activations": top_activations.tolist(),
        "metadata": top_metadata,
    }


# ==================================================
# Scoring helpers
# ==================================================


def compute_avg_pairwise_jaccard(organ_vectors: np.ndarray) -> float:
    """Average pairwise Jaccard similarity over N organ-presence vectors.

    organ_vectors: bool array, shape (N, n_organs).
    Convention: two all-zero rows (no organs in either sample) -> Jaccard 1.0.
    """
    n: int = organ_vectors.shape[0]
    assert n >= 2, f"Need at least 2 samples, got {n}"

    # Vectorised via matmul: intersection[i,j] = dot(O_i, O_j) = shared organs.
    P: np.ndarray = organ_vectors.astype(np.float32)
    intersection: np.ndarray = P @ P.T                                       # (n, n)
    sizes: np.ndarray = P.sum(axis=1)                                        # (n,)
    union: np.ndarray = sizes[:, None] + sizes[None, :] - intersection       # (n, n)

    # Masked division avoids RuntimeWarning where union == 0.
    # Where both samples have zero organs we set Jaccard to 1.0 (empty-set convention).
    jaccard: np.ndarray = np.ones((n, n), dtype=np.float64)
    nonzero: np.ndarray = union > 0
    jaccard[nonzero] = intersection[nonzero] / union[nonzero]

    # Mean over strict upper triangle (i < j); diagonal excluded.
    n_pairs: int = n * (n - 1) // 2
    return float(jaccard[np.triu_indices(n, k=1)].sum() / n_pairs)


def compute_null_baseline(organ_matrix: np.ndarray, n: int, n_draws: int, seed: int = 42) -> tuple[float, float]:
    """Monte Carlo estimate of avg pairwise Jaccard for n randomly-drawn samples.

    Draws n_draws independent subsets of size n from organ_matrix, computes
    avg pairwise Jaccard for each, and returns (mean, std).
    """
    rng: np.random.Generator = np.random.default_rng(seed)
    scores: list[float] = []
    for _ in range(n_draws):
        indices: np.ndarray = rng.choice(len(organ_matrix), size=n, replace=False)
        scores.append(compute_avg_pairwise_jaccard(organ_matrix[indices]))
    return float(np.mean(scores)), float(np.std(scores))


def compute_specificity_entropy(organ_vectors: np.ndarray, n_organs: int = 121) -> float:
    """Compute normalized entropy-based specificity score.

    Specificity measures how narrow the encoded concept is. A feature that
    activates on samples with only one organ (e.g., "liver") is more specific
    than a feature that activates on samples with many organs (e.g., "any
    abdominal organ").

    The specificity score is based on Shannon entropy over the distribution of
    organ presence counts in the top-N samples. High entropy (many organs
    distributed across samples) -> low specificity. Low entropy (few organs
    concentrated) -> high specificity.

    Args:
        organ_vectors: bool array, shape (N, n_organs). Each row is the binary
                       organ-presence vector for one sample.
        n_organs: total number of organ labels (121 for TotalSegmentator v2)

    Returns:
        Specificity score in [0, 1]. Higher = more specific (lower entropy).
        Returns 0.0 if no organs are present in any sample (edge case).
    """
    # Count how many of the top-N samples contain each organ
    organ_counts: np.ndarray = organ_vectors.sum(axis=0)  # (n_organs,)

    # Total organ occurrences across all samples
    total_organ_occurrences: int = int(organ_counts.sum())
    if total_organ_occurrences == 0:
        return 0.0  # No organs present in any sample

    # Normalize to probability distribution
    p: np.ndarray = organ_counts / total_organ_occurrences

    # Shannon entropy: H = -sum(p * log(p))
    # Convention: 0 * log(0) = 0 (limit as p->0)
    nonzero: np.ndarray = p > 0
    entropy: float = float(-1 * np.sum(p[nonzero] * np.log(p[nonzero])))

    # Normalize by maximum entropy: log(n_organs)
    # Maximum entropy occurs when all organs appear with equal probability
    max_entropy: float = np.log(n_organs)

    # Specificity = 1 - normalized_entropy
    # High entropy (uniform distribution over many organs) -> low specificity
    # Low entropy (concentrated in few organs) -> high specificity
    specificity: float = 1.0 - entropy / max_entropy

    return float(specificity)


# ==================================================
# Concept extraction
# ==================================================


def extract_shared_concepts(
    top_metadata: pd.DataFrame,
    n: int,
    threshold: float,
    exclude_columns: set[str],
    binary_group_labels: dict[str, str],
) -> list[str]:
    """Extract concepts shared by >= threshold fraction of the top-n samples.

    Iterates over every column in top_metadata (except exclude_columns) and
    applies a type-appropriate majority check against the full n denominator
    (NaN rows count as "not this concept", matching the binary organ convention):

      - Binary columns (all non-NaN values in {0, 1}): included as
        "group=column" when sum / n >= threshold.  The group label is looked up
        in binary_group_labels; columns absent from that dict fall back to the
        bare column name.
      - Categorical columns (object dtype): included as "column=value" when the
        most-frequent value's count / n >= threshold.
      - All other columns (continuous numerics, identifiers) are skipped.

    Return order: categorical concepts first (sorted), then binary concepts
    (sorted).  This surfaces modality / orientation / pathology before the
    long organ list.
    """
    subset: pd.DataFrame = top_metadata.iloc[:n]
    categorical_concepts: list[str] = []
    binary_concepts: list[str] = []

    for col in sorted(subset.columns):
        if col in exclude_columns:
            continue

        # Pandas stubs type `df[col]` as `Series | DataFrame | Unknown` because `col`
        # can be list-like; here `col` is always a `str`, so we cast to a Series.
        series: pd.Series = cast(pd.Series, subset[col]).dropna()
        if series.empty:
            continue

        # Binary columns (organ-style): all non-NaN values are 0 or 1
        if pd.api.types.is_numeric_dtype(series) and series.isin([0, 1]).all():
            if series.sum() / n >= threshold:
                group: str | None = binary_group_labels.get(col)
                binary_concepts.append(f"{group}={col}" if group else col)

        # Categorical columns: object dtype, check for a dominant value
        elif series.dtype == object:
            value_counts: pd.Series = series.value_counts()
            if value_counts.iloc[0] / n >= threshold:
                categorical_concepts.append(f"{col}={value_counts.index[0]}")

        # Numeric columns with values outside {0, 1}: skip (continuous)

    return categorical_concepts + binary_concepts


# ==================================================
# Diagnostic visualization
# ==================================================


def create_diagnostic_plots(
    feature_scores_df: pd.DataFrame,
    null_10_mean: float,
    null_50_mean: float,
    output_dir: Path,
    base_model: str,
    sae_config: str,
) -> None:
    """Generate score decay curves and 2D scatter plots for interpretability analysis.

    Creates two diagnostic plots:
    1. 6-panel score decay (coherence, specificity, monosemanticity at N=10 and N=50)
    2. 2-panel scatter (coherence vs specificity at N=10 and N=50)

    Plots are saved to output_dir with simplified filenames (no base_model/config
    needed since stored in subdirectory).
    """
    # --- Plot 1: 6-panel score decay ---
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    # Sort scores
    coherence_10_sorted: pd.Series = feature_scores_df["coherence_10"].sort_values(ascending=False)  # type: ignore[arg-type]
    coherence_10_sorted = coherence_10_sorted.reset_index(drop=True)  # type: ignore[assignment]
    
    coherence_50_sorted: pd.Series = feature_scores_df["coherence_50"].dropna().sort_values(ascending=False)  # type: ignore[arg-type]
    coherence_50_sorted = coherence_50_sorted.reset_index(drop=True)  # type: ignore[assignment]
    
    specificity_10_sorted: pd.Series = feature_scores_df["specificity_10"].sort_values(ascending=False)  # type: ignore[arg-type]
    specificity_10_sorted = specificity_10_sorted.reset_index(drop=True)  # type: ignore[assignment]
    
    specificity_50_sorted: pd.Series = feature_scores_df["specificity_50"].dropna().sort_values(ascending=False)  # type: ignore[arg-type]
    specificity_50_sorted = specificity_50_sorted.reset_index(drop=True)  # type: ignore[assignment]
    
    mono_10_sorted: pd.Series = feature_scores_df["monosemanticity_10"].sort_values(ascending=False)  # type: ignore[arg-type]
    mono_10_sorted = mono_10_sorted.reset_index(drop=True)  # type: ignore[assignment]

    mono_50_sorted: pd.Series = feature_scores_df["monosemanticity_50"].dropna().sort_values(ascending=False)  # type: ignore[arg-type]
    mono_50_sorted = mono_50_sorted.reset_index(drop=True)  # type: ignore[assignment]

    # Top row: coherence_10, specificity_10, monosemanticity_10
    ax = axes[0, 0]
    ax.plot(coherence_10_sorted.values, linewidth=2, color="steelblue")
    ax.axhline(y=null_10_mean, color="red", linestyle="--", linewidth=1, label="Null baseline (N=10)")
    ax.set_xlabel("Feature rank", fontsize=11)
    ax.set_ylabel("Coherence C(f)", fontsize=11)
    ax.set_title(f"Coherence (N=10) - {len(coherence_10_sorted)} features", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[0, 1]
    ax.plot(specificity_10_sorted.values, linewidth=2, color="darkorange")
    ax.set_xlabel("Feature rank", fontsize=11)
    ax.set_ylabel("Specificity S(f)", fontsize=11)
    ax.set_title(f"Specificity (N=10) - {len(specificity_10_sorted)} features", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 2]
    ax.plot(mono_10_sorted.values, linewidth=2, color="purple")
    ax.set_xlabel("Feature rank", fontsize=11)
    ax.set_ylabel("Monosemanticity M(f)", fontsize=11)
    ax.set_title(f"Monosemanticity (N=10) - {len(mono_10_sorted)} features", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    # Bottom row: coherence_50, specificity_50, monosemanticity_50
    ax = axes[1, 0]
    ax.plot(coherence_50_sorted.values, linewidth=2, color="steelblue")
    ax.axhline(y=null_50_mean, color="red", linestyle="--", linewidth=1, label="Null baseline (N=50)")
    ax.set_xlabel("Feature rank", fontsize=11)
    ax.set_ylabel("Coherence C(f)", fontsize=11)
    ax.set_title(f"Coherence (N=50) - {len(coherence_50_sorted)} features", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()

    ax = axes[1, 1]
    ax.plot(specificity_50_sorted.values, linewidth=2, color="darkorange")
    ax.set_xlabel("Feature rank", fontsize=11)
    ax.set_ylabel("Specificity S(f)", fontsize=11)
    ax.set_title(f"Specificity (N=50) - {len(specificity_50_sorted)} features", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 2]
    ax.plot(mono_50_sorted.values, linewidth=2, color="purple")
    ax.set_xlabel("Feature rank", fontsize=11)
    ax.set_ylabel("Monosemanticity M(f)", fontsize=11)
    ax.set_title(f"Monosemanticity (N=50) - {len(mono_50_sorted)} features", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)

    plt.suptitle(f"Score Decay Curves: {base_model} / {sae_config}",
                 fontsize=14, fontweight="bold", y=0.995)
    plt.tight_layout()

    decay_path = output_dir / "score_decay.png"
    plt.savefig(decay_path, dpi=150, bbox_inches="tight")
    print(f"  Saved score decay plot to {decay_path}")
    plt.close()

    # --- Plot 2: 2-panel coherence vs specificity scatter ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # N=10 scatter
    ax = axes[0]
    scatter = ax.scatter(
        feature_scores_df["coherence_10"],
        feature_scores_df["specificity_10"],
        c=feature_scores_df["dict_level"],
        cmap="viridis",
        alpha=0.6,
        s=20,
        edgecolors="none",
    )
    ax.axvline(x=null_10_mean, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Null baseline")
    ax.set_xlabel("Coherence (N=10)", fontsize=11)
    ax.set_ylabel("Specificity (N=10)", fontsize=11)
    ax.set_title(f"Coherence x Specificity (N=10) - {len(feature_scores_df)} features", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Dictionary Level", fontsize=10)

    # N=50 scatter
    df_50 = feature_scores_df.dropna(subset=["coherence_50", "specificity_50"])
    ax = axes[1]
    scatter = ax.scatter(
        df_50["coherence_50"],
        df_50["specificity_50"],
        c=df_50["dict_level"],
        cmap="viridis",
        alpha=0.6,
        s=20,
        edgecolors="none",
    )
    ax.axvline(x=null_50_mean, color="red", linestyle="--", linewidth=1, alpha=0.7, label="Null baseline")
    ax.set_xlabel("Coherence (N=50)", fontsize=11)
    ax.set_ylabel("Specificity (N=50)", fontsize=11)
    ax.set_title(f"Coherence x Specificity (N=50) - {len(df_50)} features", fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3)
    ax.legend()
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Dictionary Level", fontsize=10)

    plt.suptitle(f"2D Interpretability Space: {base_model} / {sae_config}",
                 fontsize=14, fontweight="bold", y=1.0)
    plt.tight_layout()

    scatter_path = output_dir / "coherence_vs_specificity.png"
    plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
    print(f"  Saved coherence vs specificity scatter to {scatter_path}")
    plt.close()


# ==================================================
# Per-config analysis runner
# ==================================================


def run_single_config(
    # Identity
    base_model: str,
    sae_config: str,
    # Paths / roots
    sparse_features_root: Path,
    sparse_features_dir_map: dict[str, str],
    split_dir: Path,
    # Split
    split: str,
    parquet_split: str,
    # Analysis parameters
    top_n_samples_per_feature: int,
    top_n_stability_check: int,
    min_activation_threshold: float,
    null_baseline_n_draws: int,
    concept_threshold: float,
    concept_exclude_columns: set[str],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[int, dict[str, object]]]:
    """Run the full interpretability analysis for one base_model x sae_config pair.

    Scores all features in the SAE dictionary (subject to minimal filtering) on two
    orthogonal axes: semantic coherence (consistency of activations) and semantic
    specificity (narrowness of encoded concept). Analysis is independent of downstream
    task performance.

    Returns:
        feature_scores_df: Per-feature summary (base_model + sae_config prepended).
        config_score_df:   Single-row config-level summary.
        feature_analysis:  Per-feature metadata dicts (for optional visualization by caller).
    """

    # --- Derived paths and values ----------------------------------------------
    sparse_features_dir: Path = sparse_features_root / sparse_features_dir_map[base_model] / sae_config
    output_dir:          Path = split_dir / base_model / sae_config

    # Parse dict sizes from config string: D_{d0}_{d1}_{d2}_{d3}_K_...
    # Cumulative Matryoshka level boundaries: [0, d0) = level 0, [d0, d1) = level 1, ...
    dict_sizes:          list[int] = [int(x) for x in sae_config.split("_")[1:5]]
    n_largest_dict_size: int       = dict_sizes[-1]

    assert sparse_features_dir.exists(), f"Sparse features directory does not exist: {sparse_features_dir}"

    print(f"SAE config: {sae_config}")
    print(f"  Dictionary sizes (level boundaries): {dict_sizes}")

    # --- Load data -------------------------------------------------------------
    # TotalSegmentatorLatentFeaturesDataset loads the full features tensor and all
    # metadata arrays into memory at construction time.
    dataset = TotalSegmentatorLatentFeaturesDataset(
        parquet_path=str(sparse_features_dir / f"total_seg_sparse_features_{parquet_split}.parquet"),
        feature_type="sparse_features",
        set_type="split",
        shuffle=False,
        additional_metadata="all",
    )

    sparse_features: np.ndarray = dataset.features.numpy()  # (n_samples, n_features)
    n_samples, n_features = sparse_features.shape

    print(f"Loaded sparse features: {n_samples} samples, {n_features} features")
    assert n_features == n_largest_dict_size, f"Expected {n_largest_dict_size} features, got {n_features}"

    # Build metadata DataFrame from dataset internals.
    # Core columns are stored as separate arrays; additional metadata (organs, patient
    # info, imaging parameters) is in dataset.metadata as {col_name: numpy array}.
    metadata_df: pd.DataFrame = pd.DataFrame({
        "modality":          dataset.modalities,
        "image_id":          dataset.image_ids,
        "slice_idx":         dataset.slice_indices,
        "slice_orientation": dataset.slice_orientations,
        **dataset.metadata,
    })

    organ_columns:         list[str]      = dataset.included_organs
    organ_presence_matrix: np.ndarray     = metadata_df[organ_columns].values.astype(bool)
    binary_group_labels:   dict[str, str] = {organ: "organ" for organ in organ_columns}

    print()
    print(f"Metadata: {len(metadata_df.columns)} columns")
    print(f"  Patient metadata (n = {len(dataset.included_patient_data)}): {dataset.included_patient_data[:5]}...")
    print(f"  Imaging metadata (n = {len(dataset.included_imaging_data)}): {dataset.included_imaging_data[:5]}...")
    print(f"  Organs metadata (n = {len(dataset.included_organs)}): {dataset.included_organs[:5]}...")

    # --- Initialize feature pool -----------------------------------------------
    # Interpretability is measured independently of downstream task performance.
    # Score all features in the SAE dictionary, subject to minimal filtering
    # (non-discriminative and insufficient-samples filters applied later).
    feature_pool: set[int] = set(range(n_features))

    print(f"\nInitial feature pool: {len(feature_pool)} features (all features in dictionary)")

    # Filter out features that activate on every single sample - these are not
    # discriminative and will produce meaningless interpretability scores.
    non_discriminative: set[int] = {
        idx for idx in feature_pool
        if int((sparse_features[:, idx] > min_activation_threshold).sum()) == n_samples
    }
    if non_discriminative:
        print(f"\nFiltering out {len(non_discriminative)} non-discriminative features "
              f"(activate on all {n_samples} samples)")
        if len(non_discriminative) <= 10:
            print(f"  Filtered features: {sorted(non_discriminative)}")
        else:
            print(f"  (showing first 10): {sorted(non_discriminative)[:10]}")
        feature_pool -= non_discriminative

    if len(feature_pool) == 0:
        raise ValueError("All features were filtered out as non-discriminative")

    print(f"Feature pool after non-discriminative filter: {len(feature_pool)} features")

    # Log warning for extreme sparsity configurations
    filter_rate: float = len(non_discriminative) / n_features
    if filter_rate > 0.9:
        print(f"  WARNING: {filter_rate*100:.1f}% of features filtered as non-discriminative")
        print(f"  This suggests extreme sparsity or pathological SAE behavior")

    print(f"  Unique feature indices in pool: {len(feature_pool)}")

    # --- Retrieve top-activating samples ---------------------------------------
    feature_analysis: dict[int, dict[str, object]] = {}
    for feature_idx in sorted(feature_pool):
        feature_analysis[feature_idx] = get_top_activating_samples(
            feature_idx, sparse_features, metadata_df, dict_sizes,
            top_n=top_n_stability_check, activation_threshold=min_activation_threshold,
        )

    print(f"\nRetrieved up to {top_n_stability_check} activating samples for {len(feature_analysis)} features")
    for feature_idx in sorted(feature_analysis):
        info = feature_analysis[feature_idx]
        max_act_str: str = f"{float(info['activations'][0]):.4f}" if info["activations"] else "-"  # type: ignore[index]
        print(f"  Feature {feature_idx} (level {info['dict_level']}): {info['n_nonzero_activations']} nonzero, "
              f"retrieved {info['n_retrieved']}, max activation = {max_act_str}")

    # Filter out features with fewer than top_n_samples_per_feature retrieved unique
    # images - they cannot be scored.  This can happen for very sparse features in
    # small dictionaries (e.g. D_16 or D_64 with low K).
    insufficient: set[int] = {
        idx for idx, info in feature_analysis.items()
        if int(info["n_retrieved"]) < top_n_samples_per_feature  # type: ignore[arg-type]
    }
    if insufficient:
        print(f"\nFiltering out {len(insufficient)} features with < {top_n_samples_per_feature} "
              f"retrieved samples: {sorted(insufficient)}")
        for idx in insufficient:
            del feature_analysis[idx]
    assert len(feature_analysis) > 0, "All features were filtered out (insufficient retrieved samples)"

    # --- Load or compute null baselines ----------------------------------------
    # Null baselines depend only on the dataset split, not on any SAE config.
    # Computed once and cached; subsequent configs load from file.
    null_baselines_path:    Path      = split_dir / "null_baselines.csv"
    null_baseline_n_values: list[int] = [5, 10, 20, 50]

    if null_baselines_path.exists():
        print(f"\nLoading null baselines from {null_baselines_path}")
        null_baselines_df: pd.DataFrame = pd.read_csv(null_baselines_path)
    else:
        print(f"\nComputing null baselines (cached at {null_baselines_path})")
        null_baselines_rows: list[dict[str, object]] = []
        for n_val in null_baseline_n_values:
            mean, std = compute_null_baseline(organ_presence_matrix, n_val, null_baseline_n_draws)
            null_baselines_rows.append({"N": n_val, "null_mean": round(mean, 6), "null_std": round(std, 6)})
        null_baselines_df = pd.DataFrame(null_baselines_rows)
        split_dir.mkdir(parents=True, exist_ok=True)
        null_baselines_df.to_csv(null_baselines_path, index=False)

    print(null_baselines_df.to_string(index=False))

    # Extract the two N values used in scoring; std is saved in the CSV for reference.
    null_lookup: dict[int, tuple[float, float]] = {
        int(row["N"]): (float(row["null_mean"]), float(row["null_std"]))
        for _, row in null_baselines_df.iterrows()
    }
    null_10_mean, _ = null_lookup[top_n_samples_per_feature]
    null_50_mean, _ = null_lookup[top_n_stability_check]

    # ==================================================
    # Two-Dimensional Interpretability Scoring
    # ==================================================
    # For each feature in the pool, compute two orthogonal interpretability axes:
    #
    # 1. COHERENCE C(f) (coherence_10, coherence_50):
    #    Does the feature consistently activate for the same kind of thing?
    #    Raw: average pairwise Jaccard over organ-presence vectors.
    #    Null-adjusted: max(0, raw - null_mean), removing chance-level similarity.
    #    High coherence = samples share the same organs (anatomical consistency).
    #
    # 2. SPECIFICITY S(f) (specificity_10, specificity_50):
    #    How narrow is the encoded concept?
    #    Measured via normalized Shannon entropy over organ distributions.
    #    High specificity = samples concentrate on few organs (narrow encoding).
    #
    # 3. MONOSEMANTICITY M(f) = C(f) x S(f) (monosemanticity_10, monosemanticity_50):
    #    Combined score. Uses null-adjusted C(f) so that features at chance level
    #    contribute zero regardless of their specificity score.
    #    M_config = mean M(f) of the top-10 features per configuration.
    #
    # These axes are INDEPENDENT and ORTHOGONAL:
    # - A feature can be coherent but not specific (broad anatomical region)
    # - A feature can be specific but not coherent (dataset artifact)
    # - Interpretable features should be BOTH coherent AND specific
    #
    # See context/interpretability_score.md for full methodology.
    for feature_idx, feature_data in feature_analysis.items():
        top_metadata: pd.DataFrame = feature_data["metadata"]  # type: ignore[assignment]
        organ_vectors: np.ndarray  = top_metadata[organ_columns].values.astype(bool)
        n_retrieved: int           = feature_data["n_retrieved"]  # type: ignore[arg-type]

        # --- Coherence: organ co-occurrence consistency (raw Jaccard) ---
        feature_analysis[feature_idx]["coherence_10"] = compute_avg_pairwise_jaccard(
            organ_vectors[:top_n_samples_per_feature]
        )
        feature_analysis[feature_idx]["coherence_50"] = (
            compute_avg_pairwise_jaccard(organ_vectors[:top_n_stability_check])
            if n_retrieved >= top_n_stability_check else None
        )

        # --- Specificity: concept distribution narrowness ---
        feature_analysis[feature_idx]["specificity_10"] = compute_specificity_entropy(
            organ_vectors[:top_n_samples_per_feature],
            n_organs=len(organ_columns)
        )
        feature_analysis[feature_idx]["specificity_50"] = (
            compute_specificity_entropy(
                organ_vectors[:top_n_stability_check],
                n_organs=len(organ_columns)
            )
            if n_retrieved >= top_n_stability_check else None
        )

        # --- Monosemanticity M(f) = C(f) x S(f), null-adjusted coherence ---
        # C(f) = max(0, raw_jaccard - null_mean): features at chance level get C=0.
        coherence_10_raw: float   = cast(float, feature_analysis[feature_idx]["coherence_10"])
        specificity_10_val: float = cast(float, feature_analysis[feature_idx]["specificity_10"])
        C_10: float = max(0.0, coherence_10_raw - null_10_mean)
        feature_analysis[feature_idx]["monosemanticity_10"] = C_10 * specificity_10_val

        coherence_50_raw: float | None   = cast(float | None, feature_analysis[feature_idx]["coherence_50"])
        specificity_50_val: float | None = cast(float | None, feature_analysis[feature_idx]["specificity_50"])
        if coherence_50_raw is not None and specificity_50_val is not None:
            C_50: float = max(0.0, coherence_50_raw - null_50_mean)
            feature_analysis[feature_idx]["monosemanticity_50"] = C_50 * specificity_50_val
        else:
            feature_analysis[feature_idx]["monosemanticity_50"] = None

        # Concepts shared by >= concept_threshold of the top-N samples.
        # All concepts use uniform key=value style: categoricals as col=value,
        # binary columns as group=col (group determined by binary_group_labels).
        # Continuous columns (age, kvp, ...) are skipped automatically.
        feature_analysis[feature_idx]["shared_concepts"] = extract_shared_concepts(
            top_metadata, top_n_samples_per_feature, concept_threshold,
            concept_exclude_columns, binary_group_labels,
        )

    # --- Aggregate, report, and save ------------------------------------------
    output_dir.mkdir(parents=True, exist_ok=True)

    # Per-feature summary table (one row per scored feature, sorted by coherence_10 descending).
    feature_scores_rows: list[dict[str, object]] = []
    for feature_idx in sorted(feature_analysis, key=lambda f: float(feature_analysis[f]["coherence_10"]), reverse=True):  # type: ignore[arg-type]
        info = feature_analysis[feature_idx]
        coherence_10:       float        = float(info["coherence_10"])        # type: ignore[arg-type]
        coherence_50:       float | None = info["coherence_50"]               # type: ignore[assignment]
        specificity_10:     float        = float(info["specificity_10"])      # type: ignore[arg-type]
        specificity_50:     float | None = info["specificity_50"]             # type: ignore[assignment]
        monosemanticity_10: float        = float(info["monosemanticity_10"])  # type: ignore[arg-type]
        monosemanticity_50: float | None = info["monosemanticity_50"]         # type: ignore[assignment]
        shared_concepts:    list[str]    = info["shared_concepts"]            # type: ignore[assignment]

        feature_scores_rows.append({
            "base_model":            base_model,
            "sae_config":            sae_config,
            "feature_idx":           feature_idx,
            "dict_level":            int(info["dict_level"]),  # type: ignore[arg-type]
            "coherence_10":          coherence_10,
            "coherence_50":          coherence_50,
            "specificity_10":        specificity_10,
            "specificity_50":        specificity_50,
            "monosemanticity_10":    monosemanticity_10,
            "monosemanticity_50":    monosemanticity_50,
            "pct_above_null":        (coherence_10 - null_10_mean) / null_10_mean * 100,
            "n_nonzero_activations": int(info["n_nonzero_activations"]),  # type: ignore[arg-type]
            "n_retrieved":           int(info["n_retrieved"]),            # type: ignore[arg-type]
            "shared_concepts":       ", ".join(shared_concepts) if shared_concepts else "",
        })

    feature_scores_df: pd.DataFrame = pd.DataFrame(feature_scores_rows)

    # Configuration-level interpretability scores (independent two-dimensional framework):
    #
    # All config-level metrics use top-N aggregation to ensure comparability across
    # configs with different feature pool sizes. We focus on the BEST features, not
    # the average across all (including uninformative) features.

    n_scored_features: int = len(feature_scores_df)

    # Coherence metrics (ranked by coherence_10)
    top10_coherence_features: pd.DataFrame = feature_scores_df.nlargest(10, "coherence_10")
    top10_coherence_mean: float = float(top10_coherence_features["coherence_10"].mean())

    top50_coherence_features: pd.DataFrame = feature_scores_df.nlargest(min(50, n_scored_features), "coherence_10")
    top50_coherence_mean: float = float(top50_coherence_features["coherence_10"].mean())

    # Specificity metrics (ranked by specificity_10)
    top10_specificity_features: pd.DataFrame = feature_scores_df.nlargest(10, "specificity_10")
    top10_specificity_mean: float = float(top10_specificity_features["specificity_10"].mean())

    top50_specificity_features: pd.DataFrame = feature_scores_df.nlargest(min(50, n_scored_features), "specificity_10")
    top50_specificity_mean: float = float(top50_specificity_features["specificity_10"].mean())

    # Monosemanticity metrics M(f) = C(f) x S(f), null-adjusted (ranked by monosemanticity_10)
    top10_mono_features: pd.DataFrame = feature_scores_df.nlargest(10, "monosemanticity_10")
    M_config: float = float(top10_mono_features["monosemanticity_10"].mean())

    top50_mono_features: pd.DataFrame = feature_scores_df.nlargest(min(50, n_scored_features), "monosemanticity_10")
    monosemanticity_top50_mean: float = float(top50_mono_features["monosemanticity_10"].mean())

    # All-features aggregate metrics (complement to top-N metrics above).
    # coherence_score: mean null-adjusted coherence across ALL scored features.
    # specificity_score: mean specificity across ALL scored features.
    # Note: pool size varies from dozens to thousands across configs, so these
    # are not directly comparable across configurations with different dict sizes.
    coherence_lifts: pd.Series = (feature_scores_df["coherence_10"] - null_10_mean).clip(lower=0.0)  # type: ignore[assignment]
    coherence_score: float = float(coherence_lifts.mean())
    n_above_null_coherence: int = int((coherence_lifts > 0).sum())
    specificity_score: float = float(feature_scores_df["specificity_10"].mean())  # type: ignore[attr-defined]

    # B (breadth score): Diagnostic metric for monosemanticity stability.
    # Mean of (monosemanticity_50 / monosemanticity_10) for features that have both scores.
    # Near 1 = interpretability persists broadly; near 0 = narrow, concentrated at top activations.
    has_mono_50: pd.Series = feature_scores_df["monosemanticity_50"].notna()  # type: ignore[assignment]
    breadth_score: float = float(
        (feature_scores_df.loc[has_mono_50, "monosemanticity_50"] / feature_scores_df.loc[has_mono_50, "monosemanticity_10"]).mean()
    ) if has_mono_50.any() else 0.0

    # Print per-feature table sorted by coherence_10 (descending).
    # C(10), C(50): raw Jaccard coherence at N=10, N=50
    # S(10), S(50): specificity at N=10, N=50 (normalized entropy over organ counts)
    # M(10), M(50): monosemanticity M(f) = C(f) x S(f) at N=10, N=50 (null-adjusted C)
    # >null%:       (coherence_10 - null_mean) / null_mean x 100; lift over baseline
    # Concept:      metadata shared by >= concept_threshold % of top-10 samples (modality, orientation, organs, etc.)
    # See context/interpretability_score.md
    print(f"\n--- Interpretability scores (null N=10: {null_10_mean:.4f} | null N=50: {null_50_mean:.4f}) ---")
    print(f"{'Feature':>8}  {'Lvl':>3}  {'C(10)':>6}  {'C(50)':>6}  {'S(10)':>6}  {'S(50)':>6}  {'M(10)':>6}  {'M(50)':>6}  {'>null%':>7}  Concept")
    print("-" * 150)
    for _, row in feature_scores_df.iterrows():
        concept_str: str = (str(row["shared_concepts"]).strip() or "-")
        mono_10: float = float(row["monosemanticity_10"])
        if bool(pd.isna(row["coherence_50"])):
            spec_50_str: str = "-" if pd.isna(row["specificity_50"]) else f"{row['specificity_50']:>6.3f}"  # type: ignore[arg-type]
            mono_50_str: str = "-" if pd.isna(row["monosemanticity_50"]) else f"{row['monosemanticity_50']:>6.3f}"  # type: ignore[arg-type]
            print(f"{int(row['feature_idx']):>8}  {int(row['dict_level']):>3}  {row['coherence_10']:>6.3f}  {'-':>6}  {row['specificity_10']:>6.3f}  {spec_50_str}  {mono_10:>6.3f}  {mono_50_str}  {'-':>7}  {concept_str}")
        else:
            spec_50_str = "-" if pd.isna(row["specificity_50"]) else f"{row['specificity_50']:>6.3f}"  # type: ignore[arg-type]
            mono_50_str = "-" if pd.isna(row["monosemanticity_50"]) else f"{row['monosemanticity_50']:>6.3f}"  # type: ignore[arg-type]
            print(f"{int(row['feature_idx']):>8}  {int(row['dict_level']):>3}  {row['coherence_10']:>6.3f}  {row['coherence_50']:>6.3f}  {row['specificity_10']:>6.3f}  {spec_50_str}  {mono_10:>6.3f}  {mono_50_str}  {row['pct_above_null']:>+6.0f}%  {concept_str}")

    print(f"\n--- Configuration-level scores: {base_model} / {sae_config} ---")
    print(f"  Top-10 coherence:       C(10) = {top10_coherence_mean:.4f}  (mean of top-10 features by coherence)")
    print(f"  Top-50 coherence:       C(50) = {top50_coherence_mean:.4f}  (mean of top-{min(50, n_scored_features)} features by coherence)")
    print(f"  Top-10 specificity:     S(10) = {top10_specificity_mean:.4f}  (mean of top-10 features by specificity)")
    print(f"  Top-50 specificity:     S(50) = {top50_specificity_mean:.4f}  (mean of top-{min(50, n_scored_features)} features by specificity)")
    print(f"  M_config (top-10 M(f)): M     = {M_config:.4f}  (mean of top-10 features by M(f) = C x S, null-adjusted)")
    print(f"  Top-50 monosemanticity:       = {monosemanticity_top50_mean:.4f}  (mean of top-{min(50, n_scored_features)} features by M(f))")
    print(f"  Breadth:                B     = {breadth_score:.4f}  (mean M(50)/M(10); interpretability stability)")
    print(f"\n  All-features coherence:   {coherence_score:.4f}  ({n_above_null_coherence}/{n_scored_features} above null)")
    print(f"  All-features specificity: {specificity_score:.4f}")

    feature_scores_path: Path = output_dir / "feature_scores.csv"
    feature_scores_df.to_csv(feature_scores_path, index=False)
    print(f"\nSaved feature scores to {feature_scores_path}")

    # Per-config summary (single row; aggregated across configs by the caller)
    config_score_df: pd.DataFrame = pd.DataFrame([{
        "base_model":              base_model,
        "sae_config":              sae_config,
        "split":                   split,
        "top10_coherence_mean":    top10_coherence_mean,
        "top50_coherence_mean":    top50_coherence_mean,
        "top10_specificity_mean":      top10_specificity_mean,
        "top50_specificity_mean":      top50_specificity_mean,
        "M_config":                    M_config,
        "monosemanticity_top50_mean":  monosemanticity_top50_mean,
        "breadth_score":               breadth_score,
        "n_features_scored":       len(feature_analysis),
        # All-features aggregate metrics (complement to top-N metrics above)
        "coherence_score":         coherence_score,
        "specificity_score":       specificity_score,
        "n_features_above_null":   n_above_null_coherence,
    }])
    config_score_path: Path = output_dir / "config_scores.csv"
    config_score_df.to_csv(config_score_path, index=False)
    print(f"Saved config score to {config_score_path}")

    # Generate diagnostic plots
    print(f"\nGenerating diagnostic plots...")
    create_diagnostic_plots(
        feature_scores_df=feature_scores_df,
        null_10_mean=null_10_mean,
        null_50_mean=null_50_mean,
        output_dir=output_dir,
        base_model=base_model,
        sae_config=sae_config,
    )

    return feature_scores_df, config_score_df, feature_analysis
