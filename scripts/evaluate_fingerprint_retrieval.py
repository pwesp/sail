#!/usr/bin/env python3
"""
Fingerprint-Based Image Retrieval

For a set of reference images, builds sparse and dense "fingerprints" from their
top-K activated features and retrieves the most similar images in the dataset.
Evaluates retrieval quality (cosine similarity in dense embedding space) across
a sweep of K values to produce the fingerprint retrieval curves in the manuscript.

Feature selection modes (FEATURE_SELECTION_MODE):
  all            Use all sparse features, excluding always-active ones.
  interpretable  Restrict to the top-N features by combined monosemanticity +
                 logistic regression task importance (via dataloading.load_interpretable_features).

Manuscript visualization:
  3 row x 5 column layout (one row per reference image)
  Columns: [Reference | Sparse Retrieved 1x2 | Dense Fingerprint Retrieved 1x2]

Outputs (results/fingerprint_retrieval/{model}/{config}/):
  k_sweep/k_sweep_per_reference.csv       per-reference quality at each K
  k_sweep/k_sweep_quality_summary.csv     mean quality aggregated over references
  sparse_vs_dense_comparison_*.csv        sparse vs dense quality per reference
  sparse_vs_dense_summary_*.csv           aggregate statistics
  reference_sparse/dense_features_*.csv   fingerprint feature indices per reference
  retrieved_sparse/dense_fingerprint_*.csv retrieved image metadata
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Tuple

# Local imports
from src.dataloading import (
    load_sparse_features_dataset,
    load_interpretable_features,
    TotalSegmentatorLatentFeaturesDataset,
)
from src.fingerprint_retrieval import (
    find_always_active_features,
    extract_top_k_sparse_features,
    extract_top_k_dense_features,
    retrieve_similar_images,
    retrieve_similar_images_dense,
    retrieve_similar_images_dense_fingerprint,
)
from src.fingerprint_visualization import (
    visualize_fingerprint_results,
    visualize_fingerprint_results_manuscript,
    visualize_sparse_vs_dense_comparison,
)
from src.fingerprint_retrieval import compute_retrieval_quality_scores
from src.retrieval_io import (
    save_reference_features_csv,
    save_retrieval_results_csv,
)

# ==================================================
# Configuration
# ==================================================

# Dataset and Model Configuration
SPLIT = "valid"         # used for monosemanticity results paths
PARQUET_SPLIT = "val"  # parquet filenames use "val"; monosemanticity paths use "valid"

SAE_CONFIGS = {
    "biomedparse":        "D_128_512_2048_8192_K_20_40_80_160",
    "dinov3":             "D_128_512_2048_8192_K_5_10_20_40",
    "biomedparse_random": "D_64_256_1024_4096_K_30_60_120_240",
}
BASE_MODEL = "biomedparse"  # "biomedparse" | "dinov3" | "biomedparse_random"
SAE_CONFIG = SAE_CONFIGS[BASE_MODEL]

# Analysis Parameters
TOP_K_FEATURES = 5      # Number of features per fingerprint (used when ENABLE_K_SWEEP=False)
TOP_N_RETRIEVALS = 5    # Similar images to retrieve per reference
N_RANDOM_REFS = 1000    # Random references for statistical analysis
N_REFS_PLOT = 10        # References to include in development plots
RANDOM_SEED = 0         # For reproducibility

# K-Sweep Analysis (multi-K retrieval quality analysis)
ENABLE_K_SWEEP = True   # Set to False to run only with TOP_K_FEATURES
K_VALUES = [1, 2, 3, 5, 10, 20, 50, 100]  # K values to test
VISUALIZATION_K = 5     # K value to use for visualizations

# Manuscript-Specific Settings
TOP_N_RETRIEVALS_MANUSCRIPT = 2  # Fewer retrievals for compact manuscript figure
N_REFS_PLOT_MANUSCRIPT = 3       # Fewer references for manuscript figure

# Manually selected references for manuscript (or None for random selection)
# Format: [(modality, image_id, slice_orientation, slice_idx), ...]
MANUSCRIPT_REFERENCE_IMAGES = [
    ("ct", "s0593", "z", 253),
    ("ct", "s0792", "z", 63),
    ("mri", "s0411", "x", 6)
]

# Feature Selection Strategy
FEATURE_SELECTION_MODE = "all"  # "all" or "interpretable"
TOP_N_INTERPRETABLE_FEATURES = 10  # Only used when mode="interpretable"

# Directory Paths
SPARSE_FEATURES_ROOT = Path("results/total_seg_sparse_features")
INTERPRETABILITY_ROOT = Path("results/monosemanticity")
OUTPUT_DIR = Path("results/fingerprint_retrieval")
DENSE_FEATURES_ROOT = Path("data")
LINEAR_PROBES_ROOT = Path("results/linear_probes")

# Derived Paths
SPARSE_FEATURES_PATH = (
    SPARSE_FEATURES_ROOT
    / f"{BASE_MODEL}_matryoshka_sae"
    / SAE_CONFIG
    / f"total_seg_sparse_features_{PARQUET_SPLIT}.parquet"
)

# ==================================================
# Data Loading
# ==================================================

def load_sparse_features() -> Tuple[np.ndarray, TotalSegmentatorLatentFeaturesDataset]:
    """Load sparse feature activations and metadata from parquet."""
    sparse_features, sparse_dataset = load_sparse_features_dataset(
        base_model=BASE_MODEL,
        sae_config=SAE_CONFIG,
        split=PARQUET_SPLIT,
        sparse_features_root=SPARSE_FEATURES_ROOT,
        as_numpy=True
    )
    print(f"Loaded {len(sparse_features)} images with {sparse_features.shape[1]} features")
    return sparse_features, sparse_dataset


def load_dense_features() -> Tuple[np.ndarray, TotalSegmentatorLatentFeaturesDataset]:
    """Load dense embeddings from the same dataset."""
    # Map sparse split names to dense split names ("val" -> "valid")
    split_mapping = {"val": "valid"}
    dense_split = split_mapping.get(PARQUET_SPLIT, PARQUET_SPLIT)

    # Construct dense feature path based on BASE_MODEL
    if BASE_MODEL == "biomedparse":
        dense_dir = "total_seg_biomedparse_encodings"
        dense_filename = f"total_seg_biomedparse_encodings_{dense_split}.parquet"
    elif BASE_MODEL == "biomedparse_random":
        dense_dir = "total_seg_biomedparse_random_encodings"
        dense_filename = f"total_seg_biomedparse_random_encodings_{dense_split}.parquet"
    elif BASE_MODEL == "dinov3":
        dense_dir = "total_seg_dinov3_encodings"
        dense_filename = f"total_seg_dinov3_encodings_{dense_split}.parquet"
    else:
        raise ValueError(f"Unknown BASE_MODEL: {BASE_MODEL}")

    dense_features_path = DENSE_FEATURES_ROOT / dense_dir / dense_filename

    print(f"\nLoading dense features from {dense_features_path}")

    # Load dense features
    dense_dataset = TotalSegmentatorLatentFeaturesDataset(
        feature_type='embeddings',
        set_type='split',
        parquet_path=str(dense_features_path),
        shuffle=False,
        additional_metadata="all"
    )

    # Extract numpy array
    dense_features = dense_dataset.features.numpy()  # type: ignore

    print(f"Loaded {len(dense_features)} images with {dense_features.shape[1]} dense features")
    return dense_features, dense_dataset




def select_random_references(
    metadata_dataset: TotalSegmentatorLatentFeaturesDataset,
    n: int,
    seed: int = 42
) -> List[Tuple[str, str, int, int]]:
    """
    Select N random images as references.

    Returns:
        List of (modality, image_id, slice_idx, array_idx) tuples
    """
    np.random.seed(seed)
    n_samples = len(metadata_dataset)
    random_indices = np.random.choice(n_samples, size=n, replace=False)

    references = []
    print(f"\nSelected {n} random reference images:")
    for idx in random_indices:
        modality = metadata_dataset.modalities[idx]
        img_id = metadata_dataset.image_ids[idx]
        slice_idx = int(metadata_dataset.slice_indices[idx])
        references.append((modality, img_id, slice_idx, idx))
        print(f"  {modality} {img_id} / slice {slice_idx} (array index {idx})")

    return references


def select_manual_references(
    metadata_dataset: TotalSegmentatorLatentFeaturesDataset,
    manual_specs: List[Tuple[str, str, str, int]]
) -> List[Tuple[str, str, int, int]]:
    """
    Select manually specified images as references.

    Args:
        metadata_dataset: Dataset with metadata
        manual_specs: List of (modality, image_id, slice_orientation, slice_idx) tuples
                     Example: [("ct", "s0001", "z", 150), ("mri", "s0100", "z", 75)]

    Returns:
        List of (modality, image_id, slice_idx, array_idx) tuples
    """
    references = []
    print(f"\nLooking up {len(manual_specs)} manually specified reference images:")

    for modality, img_id, orientation, slice_idx in manual_specs:
        # Find matching index in dataset
        found = False
        for idx in range(len(metadata_dataset)):
            dataset_modality = metadata_dataset.modalities[idx]
            dataset_img_id = metadata_dataset.image_ids[idx]
            dataset_slice_idx = int(metadata_dataset.slice_indices[idx])
            dataset_orientation = metadata_dataset.slice_orientations[idx]

            # Check if ALL fields match
            if (dataset_modality == modality and
                dataset_img_id == img_id and
                dataset_orientation == orientation and
                dataset_slice_idx == slice_idx):
                references.append((modality, img_id, slice_idx, idx))
                print(f"  [ok] Found: {modality} {img_id} / {orientation} / slice {slice_idx} (array index {idx})")
                found = True
                break

        if not found:
            print(f"  FAIL NOT FOUND: {modality} {img_id} / {orientation} / slice {slice_idx}")
            print(f"    Please verify: modality='{modality}', image_id='{img_id}', orientation='{orientation}', slice_idx={slice_idx}")

            # Provide debugging help: show a sample of what's in the dataset
            print(f"    Hint: Check if image '{img_id}' exists in the dataset with:")
            matching_ids = [i for i in range(len(metadata_dataset)) if metadata_dataset.image_ids[i] == img_id]
            if matching_ids:
                print(f"    Found {len(matching_ids)} slices for image_id '{img_id}':")
                for i in matching_ids[:5]:  # Show first 5 matches
                    print(f"      - {metadata_dataset.modalities[i]} / {metadata_dataset.slice_orientations[i]} / slice {metadata_dataset.slice_indices[i]} (idx {i})")
                if len(matching_ids) > 5:
                    print(f"      ... and {len(matching_ids) - 5} more")
            else:
                print(f"    No slices found for image_id '{img_id}' in the dataset")
                # Show a sample of available image_ids
                unique_ids = list(set(metadata_dataset.image_ids[:100]))[:5]
                print(f"    Sample image_ids in dataset: {unique_ids}")

    if len(references) < len(manual_specs):
        print(f"\nWARNING: Warning: Only found {len(references)} of {len(manual_specs)} specified references")

    return references


# ==================================================
# Main Workflow Helper Functions
# ==================================================

def _process_single_k(
    ref_id: str,
    ref_slice: int,
    ref_idx: int,
    k: int,
    n_retrievals: int,
    sparse_features: np.ndarray,
    dense_features: np.ndarray,
    metadata_dataset: TotalSegmentatorLatentFeaturesDataset,
    always_active_features: set,
    allowed_features: set | None
) -> Tuple[pd.DataFrame, pd.DataFrame, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], dict]:
    """
    Core processing logic for one reference image with one K value.

    Extracts top-K features, retrieves similar images, and computes quality scores.
    This is the shared logic used by both single-K and multi-K processing functions.

    Parameters
    ----------
    ref_id : str
        Reference image ID
    ref_slice : int
        Reference slice index
    ref_idx : int
        Reference array index in dataset
    k : int
        Number of features to use in fingerprint
    n_retrievals : int
        Number of similar images to retrieve
    sparse_features : np.ndarray
        Sparse feature activations
    dense_features : np.ndarray
        Dense embeddings
    metadata_dataset : TotalSegmentatorLatentFeaturesDataset
        Dataset with metadata
    always_active_features : set
        Features to exclude from fingerprint selection
    allowed_features : set | None
        If not None, only select from these features

    Returns
    -------
    Tuple containing:
        - sparse_results: DataFrame of sparse fingerprint retrieval results
        - dense_fp_results: DataFrame of dense fingerprint retrieval results
        - sparse_features_info: (indices, activations) for sparse fingerprint
        - dense_features_info: (indices, activations) for dense fingerprint
        - quality_metrics: Dict with quality scores
    """
    # Extract top-K sparse features for fingerprint
    sparse_feature_indices, sparse_activations = extract_top_k_sparse_features(
        sparse_features, ref_idx, k, always_active_features, allowed_features
    )

    # Extract top-K dense features for fingerprint
    dense_feature_indices, dense_activations = extract_top_k_dense_features(
        dense_features, ref_idx, k
    )

    # Sparse fingerprint retrieval
    sparse_retrievals = retrieve_similar_images(
        sparse_features, metadata_dataset, ref_idx, sparse_feature_indices, n_retrievals
    )

    # Dense fingerprint retrieval
    dense_fingerprint_retrievals = retrieve_similar_images_dense_fingerprint(
        dense_features, metadata_dataset, ref_idx, dense_feature_indices, n_retrievals
    )

    # Evaluate sparse retrieval quality in dense space
    sparse_quality = compute_retrieval_quality_scores(
        sparse_retrievals, ref_idx, dense_features
    )

    # Evaluate dense fingerprint retrieval quality
    dense_fp_quality = compute_retrieval_quality_scores(
        dense_fingerprint_retrievals, ref_idx, dense_features
    )

    # Store quality metrics
    quality_metrics = {
        "k": k,
        "reference_id": ref_id,
        "reference_slice": ref_slice,
        "sparse_mean": sparse_quality['mean_cosine_similarity'],
        "sparse_std": sparse_quality['std_cosine_similarity'],
        "sparse_min": sparse_quality['min_cosine_similarity'],
        "sparse_max": sparse_quality['max_cosine_similarity'],
        "dense_fp_mean": dense_fp_quality['mean_cosine_similarity'],
        "dense_fp_std": dense_fp_quality['std_cosine_similarity'],
        "dense_fp_min": dense_fp_quality['min_cosine_similarity'],
        "dense_fp_max": dense_fp_quality['max_cosine_similarity'],
    }

    return (
        sparse_retrievals,
        dense_fingerprint_retrievals,
        (sparse_feature_indices, sparse_activations),
        (dense_feature_indices, dense_activations),
        quality_metrics
    )


def _process_single_reference_multi_k(
    ref_id: str,
    ref_slice: int,
    ref_idx: int,
    k_values: List[int],
    n_retrievals: int,
    sparse_features: np.ndarray,
    dense_features: np.ndarray,
    metadata_dataset: TotalSegmentatorLatentFeaturesDataset,
    always_active_features: set,
    allowed_features: set | None,
) -> Tuple[Dict[int, Tuple[pd.DataFrame, pd.DataFrame, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], dict]], pd.DataFrame]:
    """
    Process a single reference image with multiple K values.

    Returns
    -------
    Tuple of:
        - results_by_k: Dict keyed by K value, each containing:
            - sparse_results: DataFrame of sparse fingerprint retrieval results
            - dense_fp_results: DataFrame of dense fingerprint retrieval results
            - sparse_features_info: (indices, activations) for sparse fingerprint
            - dense_features_info: (indices, activations) for dense fingerprint
            - quality_metrics: Dict with quality scores for this K
        - dense_retrievals: DataFrame of dense full retrieval (gold standard baseline)
    """
    print(f"\nProcessing reference: {ref_id} / slice {ref_slice} (index {ref_idx})")
    print(f"  Testing K values: {k_values}")

    # Compute dense full retrieval ONCE as gold standard baseline
    dense_retrievals = retrieve_similar_images_dense(
        dense_features, metadata_dataset, ref_idx, n_retrievals
    )
    dense_quality = compute_retrieval_quality_scores(
        dense_retrievals, ref_idx, dense_features
    )
    print(f"  Dense full (gold standard): {dense_quality['mean_cosine_similarity']:.3f}")

    results_by_k = {}

    for k in k_values:
        # Process this K value using core logic
        results_by_k[k] = _process_single_k(
            ref_id, ref_slice, ref_idx, k, n_retrievals,
            sparse_features, dense_features, metadata_dataset,
            always_active_features, allowed_features
        )

        # Print quality summary (compare against gold standard)
        quality_metrics = results_by_k[k][4]  # quality_metrics is 5th element
        relative_perf = quality_metrics['sparse_mean'] / dense_quality['mean_cosine_similarity']
        print(f"  K={k}: sparse_fp={quality_metrics['sparse_mean']:.3f} ({relative_perf:.1%} of full dense), "
              f"dense_fp={quality_metrics['dense_fp_mean']:.3f}")

    return results_by_k, dense_retrievals


def _process_single_reference(
    ref_id: str,
    ref_slice: int,
    ref_idx: int,
    row_idx: int,
    sparse_features: np.ndarray,
    dense_features: np.ndarray,
    metadata_dataset: TotalSegmentatorLatentFeaturesDataset,
    always_active_features: set,
    allowed_features: set | None,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Tuple[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray], dict, str]:
    """
    Process a single reference image: extract fingerprints, retrieve similar images, evaluate quality.

    This is the single-K mode function that also computes full dense retrieval for comparison.

    Returns
    -------
    Tuple containing:
        - sparse_results: DataFrame of sparse fingerprint retrieval results
        - dense_results: DataFrame of dense retrieval results
        - dense_fp_results: DataFrame of dense fingerprint retrieval results
        - sparse_features_info: (indices, activations) for sparse fingerprint
        - dense_features_info: (indices, activations) for dense fingerprint
        - comparison_result: Dict with comparison metrics
        - row_id: Unique identifier for this reference
    """
    row_id = f"row_{row_idx:04d}"
    print(f"\nProcessing reference: {ref_id} / slice {ref_slice} (index {ref_idx})")

    # Use core processing logic for K=TOP_K_FEATURES
    (sparse_retrievals, dense_fingerprint_retrievals,
     sparse_feat_info, dense_feat_info, quality_metrics) = _process_single_k(
        ref_id, ref_slice, ref_idx, TOP_K_FEATURES, TOP_N_RETRIEVALS,
        sparse_features, dense_features, metadata_dataset,
        always_active_features, allowed_features
    )

    sparse_feature_indices, sparse_activations = sparse_feat_info
    dense_feature_indices, dense_activations = dense_feat_info

    print(f"  Top-{TOP_K_FEATURES} sparse features: {sparse_feature_indices.tolist()}")
    print(f"  Sparse activations: {sparse_activations}")
    print(f"  Top-{TOP_K_FEATURES} dense features: {dense_feature_indices.tolist()}")
    print(f"  Dense activations: {dense_activations}")
    print(f"  Sparse retrieval quality: {quality_metrics['sparse_mean']:.3f} "
          f"+/- {quality_metrics['sparse_std']:.3f}")

    # Additional step for single-K mode: Dense retrieval (full dense vector)
    dense_retrievals = retrieve_similar_images_dense(
        dense_features, metadata_dataset, ref_idx, TOP_N_RETRIEVALS
    )

    # Evaluate dense retrieval quality
    dense_quality = compute_retrieval_quality_scores(
        dense_retrievals, ref_idx, dense_features
    )
    print(f"  Dense retrieval quality:  {dense_quality['mean_cosine_similarity']:.3f} "
          f"+/- {dense_quality['std_cosine_similarity']:.3f}")
    print(f"  Dense fingerprint quality: {quality_metrics['dense_fp_mean']:.3f} "
          f"+/- {quality_metrics['dense_fp_std']:.3f}")

    # Compute relative performance
    relative_perf = quality_metrics['sparse_mean'] / dense_quality['mean_cosine_similarity']
    relative_perf_fp = quality_metrics['sparse_mean'] / quality_metrics['dense_fp_mean']
    print(f"  Relative performance (sparse vs full dense): {relative_perf:.1%}")
    print(f"  Relative performance (sparse vs dense FP):   {relative_perf_fp:.1%}")

    # Prepare extended comparison results (includes dense full retrieval)
    comparison_result = {
        "reference_id": ref_id,
        "reference_slice": ref_slice,
        "sparse_mean": quality_metrics['sparse_mean'],
        "sparse_std": quality_metrics['sparse_std'],
        "sparse_min": quality_metrics['sparse_min'],
        "sparse_max": quality_metrics['sparse_max'],
        "dense_mean": dense_quality['mean_cosine_similarity'],
        "dense_std": dense_quality['std_cosine_similarity'],
        "dense_min": dense_quality['min_cosine_similarity'],
        "dense_max": dense_quality['max_cosine_similarity'],
        "dense_fp_mean": quality_metrics['dense_fp_mean'],
        "dense_fp_std": quality_metrics['dense_fp_std'],
        "dense_fp_min": quality_metrics['dense_fp_min'],
        "dense_fp_max": quality_metrics['dense_fp_max'],
        "relative_performance": relative_perf,
        "relative_performance_fp": relative_perf_fp
    }

    return (
        sparse_retrievals,
        dense_retrievals,
        dense_fingerprint_retrievals,
        (sparse_feature_indices, sparse_activations),
        (dense_feature_indices, dense_activations),
        comparison_result,
        row_id
    )



def main() -> None:
    # Create organized output directory structure
    config_output_dir = OUTPUT_DIR / BASE_MODEL / SAE_CONFIG
    k_sweep_dir = config_output_dir / "k_sweep"
    config_output_dir.mkdir(parents=True, exist_ok=True)
    k_sweep_dir.mkdir(parents=True, exist_ok=True)

    # ==================================================
    # PHASE 1: DATA LOADING
    # ==================================================
    print("\n" + "=" * 70)
    print("PHASE 1: DATA LOADING")
    print("=" * 70)

    sparse_features, metadata_dataset = load_sparse_features()
    dense_features, _ = load_dense_features()

    # Verify alignment
    assert len(sparse_features) == len(dense_features), \
        f"Mismatched lengths: sparse={len(sparse_features)}, dense={len(dense_features)}"
    print(f"\n[ok] Verified: sparse and dense features are aligned ({len(sparse_features)} samples)")

    # Find always-active features (exclude from fingerprint selection)
    always_active_features = find_always_active_features(sparse_features)

    # Load interpretable features if requested
    allowed_features = None
    if FEATURE_SELECTION_MODE == "interpretable":
        features_df = load_interpretable_features(
            base_model=BASE_MODEL,
            sae_config=SAE_CONFIG,
            top_n=TOP_N_INTERPRETABLE_FEATURES,
            always_active_features=always_active_features,
            interpretability_root=INTERPRETABILITY_ROOT,
            linear_probes_root=LINEAR_PROBES_ROOT,
        )
        allowed_features = set(features_df["feature_idx"].tolist())
        print(f"\nFeature selection mode: interpretable (top-{TOP_N_INTERPRETABLE_FEATURES})")
    else:
        print(f"\nFeature selection mode: all sparse features")

    # Select random references
    references = select_random_references(metadata_dataset, N_RANDOM_REFS, seed=RANDOM_SEED)

    # Select manuscript references if specified
    if MANUSCRIPT_REFERENCE_IMAGES is not None:
        print("\nSelecting manuscript-specific references...")
        manuscript_refs = select_manual_references(metadata_dataset, MANUSCRIPT_REFERENCE_IMAGES)
    else:
        manuscript_refs = references[:N_REFS_PLOT_MANUSCRIPT]

    # ==================================================
    # PHASE 2: QUANTITATIVE ANALYSIS
    # ==================================================
    print("\n" + "=" * 70)
    print("PHASE 2: QUANTITATIVE ANALYSIS")
    print("=" * 70)

    # Pre-declare variables that will be populated in the if/else branches
    all_sparse_results: Dict[Tuple[str, str, int], pd.DataFrame] = {}
    all_dense_results: Dict[Tuple[str, str, int], pd.DataFrame] = {}
    all_dense_fingerprint_results: Dict[Tuple[str, str, int], pd.DataFrame] = {}
    all_sparse_features: Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray]] = {}
    all_dense_features: Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray]] = {}
    comparison_results: List[dict] = []
    reference_row_ids: Dict[Tuple[str, str, int], str] = {}

    if ENABLE_K_SWEEP:
        print(f"\nK-Sweep Mode: Testing K values {K_VALUES}")
        print(f"Visualization will use K={VISUALIZATION_K}")

        # Storage for K-sweep results
        k_sweep_all_results: Dict[Tuple[str, str, int], Dict[int, Tuple]] = {}  # (modality, ref_id, ref_slice) -> {k -> results}
        k_sweep_quality_records: List[dict] = []  # All quality metrics for CSV

        # Process all references with K-sweep
        for row_idx, (modality, ref_id, ref_slice, ref_idx) in enumerate(references):
            results_by_k, dense_retrievals = _process_single_reference_multi_k(
                ref_id, ref_slice, ref_idx, K_VALUES, TOP_N_RETRIEVALS,
                sparse_features, dense_features, metadata_dataset,
                always_active_features, allowed_features,
            )
            k_sweep_all_results[(modality, ref_id, ref_slice)] = results_by_k
            all_dense_results[(modality, ref_id, ref_slice)] = dense_retrievals  # Store gold standard

            # Collect quality metrics for all K values
            for k, (_, _, _, _, quality_metrics) in results_by_k.items():
                k_sweep_quality_records.append(quality_metrics)

        # Process manuscript references with K-sweep if they weren't already processed
        k_sweep_manuscript_results: Dict[Tuple[str, str, int], Dict[int, Tuple]] = {}
        for modality, ref_id, ref_slice, ref_idx in manuscript_refs:
            key = (modality, ref_id, ref_slice)
            if key not in k_sweep_all_results:
                print(f"\nProcessing additional manuscript reference: {ref_id} / slice {ref_slice}")
                results_by_k, dense_retrievals = _process_single_reference_multi_k(
                    ref_id, ref_slice, ref_idx, K_VALUES, TOP_N_RETRIEVALS_MANUSCRIPT,
                    sparse_features, dense_features, metadata_dataset,
                    always_active_features, allowed_features,
                )
                k_sweep_manuscript_results[key] = results_by_k
                all_dense_results[key] = dense_retrievals  # Store gold standard
            else:
                k_sweep_manuscript_results[key] = k_sweep_all_results[key]

        # Extract results for VISUALIZATION_K for later visualization
        print(f"\nExtracting K={VISUALIZATION_K} results for visualization...")

        for (modality, ref_id, ref_slice), results_by_k in k_sweep_all_results.items():
            if VISUALIZATION_K in results_by_k:
                sparse_ret, dense_fp_ret, sparse_feat_info, dense_feat_info, quality_metrics = results_by_k[VISUALIZATION_K]
                all_sparse_results[(modality, ref_id, ref_slice)] = sparse_ret
                all_dense_fingerprint_results[(modality, ref_id, ref_slice)] = dense_fp_ret
                all_sparse_features[(modality, ref_id, ref_slice)] = sparse_feat_info
                all_dense_features[(modality, ref_id, ref_slice)] = dense_feat_info

                # Use gold standard dense retrieval (already computed once per reference)
                dense_retrievals = all_dense_results[(modality, ref_id, ref_slice)]

                # Find reference index for quality computation
                ref_idx = None
                for r_modality, r_id, r_slice, r_idx in references:
                    if r_modality == modality and r_id == ref_id and r_slice == ref_slice:
                        ref_idx = r_idx
                        break

                if ref_idx is not None:
                    dense_quality = compute_retrieval_quality_scores(
                        dense_retrievals, ref_idx, dense_features
                    )

                    comparison_results.append({
                        "reference_id": ref_id,
                        "reference_slice": ref_slice,
                        "sparse_mean": quality_metrics['sparse_mean'],
                        "sparse_std": quality_metrics['sparse_std'],
                        "sparse_min": quality_metrics['sparse_min'],
                        "sparse_max": quality_metrics['sparse_max'],
                        "dense_mean": dense_quality['mean_cosine_similarity'],
                        "dense_std": dense_quality['std_cosine_similarity'],
                        "dense_min": dense_quality['min_cosine_similarity'],
                        "dense_max": dense_quality['max_cosine_similarity'],
                        "dense_fp_mean": quality_metrics['dense_fp_mean'],
                        "dense_fp_std": quality_metrics['dense_fp_std'],
                        "dense_fp_min": quality_metrics['dense_fp_min'],
                        "dense_fp_max": quality_metrics['dense_fp_max'],
                        "relative_performance": quality_metrics['sparse_mean'] / dense_quality['mean_cosine_similarity'],
                        "relative_performance_fp": quality_metrics['sparse_mean'] / quality_metrics['dense_fp_mean']
                    })

        # Extract manuscript results for VISUALIZATION_K
        manuscript_sparse_results: Dict[Tuple[str, str, int], pd.DataFrame] = {}
        manuscript_dense_fp_results: Dict[Tuple[str, str, int], pd.DataFrame] = {}
        manuscript_sparse_features: Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray]] = {}
        manuscript_dense_features: Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray]] = {}

        for (modality, ref_id, ref_slice), results_by_k in k_sweep_manuscript_results.items():
            if VISUALIZATION_K in results_by_k:
                sparse_ret, dense_fp_ret, sparse_feat_info, dense_feat_info, _ = results_by_k[VISUALIZATION_K]
                manuscript_sparse_results[(modality, ref_id, ref_slice)] = sparse_ret
                manuscript_dense_fp_results[(modality, ref_id, ref_slice)] = dense_fp_ret
                manuscript_sparse_features[(modality, ref_id, ref_slice)] = sparse_feat_info
                manuscript_dense_features[(modality, ref_id, ref_slice)] = dense_feat_info

    else:
        print(f"\nSingle-K Mode: Using K={TOP_K_FEATURES}")

        # Original single-K processing
        for row_idx, (modality, ref_id, ref_slice, ref_idx) in enumerate(references):
            (sparse_ret, dense_ret, dense_fp_ret, sparse_feat_info, dense_feat_info,
             comparison_result, row_id) = _process_single_reference(
                ref_id, ref_slice, ref_idx, row_idx,
                sparse_features, dense_features, metadata_dataset,
                always_active_features, allowed_features,
            )

            reference_row_ids[(modality, ref_id, ref_slice)] = row_id
            all_sparse_results[(modality, ref_id, ref_slice)] = sparse_ret
            all_dense_results[(modality, ref_id, ref_slice)] = dense_ret
            all_dense_fingerprint_results[(modality, ref_id, ref_slice)] = dense_fp_ret
            all_sparse_features[(modality, ref_id, ref_slice)] = sparse_feat_info
            all_dense_features[(modality, ref_id, ref_slice)] = dense_feat_info
            comparison_results.append(comparison_result)

        # Process manuscript references separately (original logic)
        manuscript_sparse_results = {}
        manuscript_dense_fp_results = {}
        manuscript_sparse_features = {}
        manuscript_dense_features = {}

        for modality, ref_id, ref_slice, ref_idx in manuscript_refs:
            key = (modality, ref_id, ref_slice)
            if key not in all_sparse_features:
                print(f"\nProcessing additional manuscript reference: {ref_id} / slice {ref_slice}")

                sparse_feature_indices, sparse_activations = extract_top_k_sparse_features(
                    sparse_features, ref_idx, TOP_K_FEATURES, always_active_features, allowed_features
                )
                sparse_retrievals = retrieve_similar_images(
                    sparse_features, metadata_dataset, ref_idx, sparse_feature_indices, TOP_N_RETRIEVALS_MANUSCRIPT
                )

                dense_feature_indices, dense_activations = extract_top_k_dense_features(
                    dense_features, ref_idx, TOP_K_FEATURES
                )
                dense_fingerprint_retrievals = retrieve_similar_images_dense_fingerprint(
                    dense_features, metadata_dataset, ref_idx, dense_feature_indices, TOP_N_RETRIEVALS_MANUSCRIPT
                )

                manuscript_sparse_features[key] = (sparse_feature_indices, sparse_activations)
                manuscript_dense_features[key] = (dense_feature_indices, dense_activations)
                manuscript_sparse_results[key] = sparse_retrievals
                manuscript_dense_fp_results[key] = dense_fingerprint_retrievals
            else:
                manuscript_sparse_features[key] = all_sparse_features[key]
                manuscript_dense_features[key] = all_dense_features[key]
                manuscript_sparse_results[key] = all_sparse_results[key]
                manuscript_dense_fp_results[key] = all_dense_fingerprint_results[key]

    # ==================================================
    # PHASE 3: SAVE RESULTS
    # ==================================================
    print("\n" + "=" * 70)
    print("PHASE 3: SAVE RESULTS")
    print("=" * 70)

    # Generate row IDs for references (only if not already set in single-K mode)
    if not reference_row_ids:  # Empty dict evaluates to False
        for row_idx, (modality, ref_id, ref_slice, _) in enumerate(references):
            reference_row_ids[(modality, ref_id, ref_slice)] = f"row_{row_idx:04d}"

    manuscript_row_ids: Dict[Tuple[str, str, int], str] = {}
    for row_idx, (modality, ref_id, ref_slice, _) in enumerate(manuscript_refs):
        manuscript_row_ids[(modality, ref_id, ref_slice)] = f"manuscript_row_{row_idx:04d}"

    # Save K-sweep results if enabled
    if ENABLE_K_SWEEP:
        print("\nSaving K-sweep analysis results...")

        # Create per-reference CSV with all K values
        k_sweep_per_ref_df = pd.DataFrame(k_sweep_quality_records)
        k_sweep_per_ref_path = k_sweep_dir / "k_sweep_per_reference.csv"
        k_sweep_per_ref_df.to_csv(k_sweep_per_ref_path, index=False)
        print(f"[ok] Saved K-sweep per-reference results to {k_sweep_per_ref_path.name}")

        # Create summary by K value
        k_sweep_summary = k_sweep_per_ref_df.groupby('k').agg({
            'sparse_mean': ['mean', 'std'],
            'dense_fp_mean': ['mean', 'std'],
            'reference_id': 'count'
        }).reset_index()

        k_sweep_summary.columns = [
            'k',
            'mean_sparse_quality', 'std_sparse_quality',
            'mean_dense_fp_quality', 'std_dense_fp_quality',
            'n_references'
        ]

        # Add final row: "all features allowed" (full dense retrieval baseline)
        # Use k=9999 to keep column as integer while clearly marking this special case
        comparison_df_temp = pd.DataFrame(comparison_results)
        all_features_row = pd.DataFrame([{
            'k': 9999,
            'mean_sparse_quality': comparison_df_temp['dense_mean'].mean(),
            'std_sparse_quality': comparison_df_temp['dense_mean'].std(),
            'mean_dense_fp_quality': comparison_df_temp['dense_mean'].mean(),
            'std_dense_fp_quality': comparison_df_temp['dense_mean'].std(),
            'n_references': len(comparison_df_temp)
        }])
        k_sweep_summary = pd.concat([k_sweep_summary, all_features_row], ignore_index=True)

        k_sweep_summary_path = k_sweep_dir / "k_sweep_quality_summary.csv"
        k_sweep_summary.to_csv(k_sweep_summary_path, index=False)
        print(f"[ok] Saved K-sweep quality summary to {k_sweep_summary_path.name}")
        print(f"  (includes k=9999 row for full dense retrieval baseline)")

        print(f"\nK-Sweep Summary:")
        print(k_sweep_summary.to_string(index=False))

    # Save standard results (K=VISUALIZATION_K for K-sweep, or K=TOP_K_FEATURES for single-K)
    k_label = f"_k{VISUALIZATION_K}" if ENABLE_K_SWEEP else f"_k{TOP_K_FEATURES}"
    print(f"\nSaving standard results{k_label}...")

    sparse_feature_summary_path = config_output_dir / f"reference_sparse_features_{FEATURE_SELECTION_MODE}{k_label}.csv"
    save_reference_features_csv(
        all_sparse_features, reference_row_ids, metadata_dataset,
        sparse_feature_summary_path, feature_type="sparse"
    )

    dense_feature_summary_path = config_output_dir / f"reference_dense_features_{FEATURE_SELECTION_MODE}{k_label}.csv"
    save_reference_features_csv(
        all_dense_features, reference_row_ids, metadata_dataset,
        dense_feature_summary_path, feature_type="dense"
    )

    # Save retrieved samples summary (sparse fingerprint retrieval)
    sparse_retrieved_path = config_output_dir / f"retrieved_sparse_fingerprint_{FEATURE_SELECTION_MODE}{k_label}.csv"
    save_retrieval_results_csv(
        all_sparse_results, reference_row_ids, metadata_dataset,
        sparse_retrieved_path, retrieval_method="sparse_fingerprint"
    )

    # Save retrieved samples summary (dense fingerprint retrieval)
    dense_fp_retrieved_path = config_output_dir / f"retrieved_dense_fingerprint_{FEATURE_SELECTION_MODE}{k_label}.csv"
    save_retrieval_results_csv(
        all_dense_fingerprint_results, reference_row_ids, metadata_dataset,
        dense_fp_retrieved_path, retrieval_method="dense_fingerprint"
    )

    # Save comparison results
    comparison_df = pd.DataFrame(comparison_results)
    comparison_path = config_output_dir / f"sparse_vs_dense_comparison_{FEATURE_SELECTION_MODE}{k_label}.csv"
    comparison_df.to_csv(comparison_path, index=False)
    print(f"[ok] Saved comparison results to {comparison_path}")

    # Generate summary statistics
    summary_data = {
        "metric": [
            "mean_similarity",
            "std_similarity",
            "min_similarity",
            "max_similarity",
            "n_references"
        ],
        "sparse_retrieval": [
            np.mean(comparison_df['sparse_mean']),
            np.std(comparison_df['sparse_mean']),
            np.min(comparison_df['sparse_mean']),
            np.max(comparison_df['sparse_mean']),
            len(comparison_df)
        ],
        "dense_retrieval": [
            np.mean(comparison_df['dense_mean']) if 'dense_mean' in comparison_df.columns else 0,
            np.std(comparison_df['dense_mean']) if 'dense_mean' in comparison_df.columns else 0,
            np.min(comparison_df['dense_mean']) if 'dense_mean' in comparison_df.columns else 0,
            np.max(comparison_df['dense_mean']) if 'dense_mean' in comparison_df.columns else 0,
            len(comparison_df)
        ],
        "dense_fp_retrieval": [
            np.mean(comparison_df['dense_fp_mean']),
            np.std(comparison_df['dense_fp_mean']),
            np.min(comparison_df['dense_fp_mean']),
            np.max(comparison_df['dense_fp_mean']),
            len(comparison_df)
        ],
        "relative_performance": [
            np.mean(comparison_df['relative_performance']) if 'relative_performance' in comparison_df.columns else 0,
            np.std(comparison_df['relative_performance']) if 'relative_performance' in comparison_df.columns else 0,
            np.min(comparison_df['relative_performance']) if 'relative_performance' in comparison_df.columns else 0,
            np.max(comparison_df['relative_performance']) if 'relative_performance' in comparison_df.columns else 0,
            len(comparison_df)
        ],
        "relative_performance_fp": [
            np.mean(comparison_df['relative_performance_fp']) if 'relative_performance_fp' in comparison_df.columns else 0,
            np.std(comparison_df['relative_performance_fp']) if 'relative_performance_fp' in comparison_df.columns else 0,
            np.min(comparison_df['relative_performance_fp']) if 'relative_performance_fp' in comparison_df.columns else 0,
            np.max(comparison_df['relative_performance_fp']) if 'relative_performance_fp' in comparison_df.columns else 0,
            len(comparison_df)
        ]
    }
    summary_df = pd.DataFrame(summary_data)
    summary_path = config_output_dir / f"sparse_vs_dense_summary_{FEATURE_SELECTION_MODE}{k_label}.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"[ok] Saved summary statistics to {summary_path}")

    # Save manuscript-specific results
    print("\nSaving manuscript-specific results...")

    manuscript_sparse_summary_path = config_output_dir / f"reference_sparse_features_manuscript_{FEATURE_SELECTION_MODE}{k_label}.csv"
    save_reference_features_csv(
        manuscript_sparse_features, manuscript_row_ids, metadata_dataset,
        manuscript_sparse_summary_path, feature_type="manuscript sparse"
    )

    manuscript_dense_summary_path = config_output_dir / f"reference_dense_features_manuscript_{FEATURE_SELECTION_MODE}{k_label}.csv"
    save_reference_features_csv(
        manuscript_dense_features, manuscript_row_ids, metadata_dataset,
        manuscript_dense_summary_path, feature_type="manuscript dense"
    )

    manuscript_sparse_retrieved_path = config_output_dir / f"retrieved_sparse_fingerprint_manuscript_{FEATURE_SELECTION_MODE}{k_label}.csv"
    save_retrieval_results_csv(
        manuscript_sparse_results, manuscript_row_ids, metadata_dataset,
        manuscript_sparse_retrieved_path, retrieval_method="sparse_fingerprint"
    )

    manuscript_dense_fp_retrieved_path = config_output_dir / f"retrieved_dense_fingerprint_manuscript_{FEATURE_SELECTION_MODE}{k_label}.csv"
    save_retrieval_results_csv(
        manuscript_dense_fp_results, manuscript_row_ids, metadata_dataset,
        manuscript_dense_fp_retrieved_path, retrieval_method="dense_fingerprint"
    )

    # ==================================================
    # PHASE 4: GENERATE VISUALIZATIONS
    # ==================================================
    print("\n" + "=" * 70)
    print("PHASE 4: GENERATE VISUALIZATIONS")
    print("=" * 70)

    # Generate K-sweep quality curve if enabled
    if ENABLE_K_SWEEP:
        print("\nGenerating K-sweep quality curve...")
        from src.fingerprint_visualization import visualize_k_sweep_quality_curve

        k_sweep_curve_path = k_sweep_dir / "k_sweep_quality_curve.png"
        visualize_k_sweep_quality_curve(
            k_sweep_summary, k_sweep_curve_path, BASE_MODEL, SAE_CONFIG
        )

    # Generate comparison visualization
    print(f"\nGenerating sparse vs dense comparison plot (K={VISUALIZATION_K if ENABLE_K_SWEEP else TOP_K_FEATURES})...")
    comparison_viz_path = config_output_dir / f"sparse_vs_dense_comparison_{FEATURE_SELECTION_MODE}{k_label}.png"
    visualize_sparse_vs_dense_comparison(comparison_results, comparison_viz_path, BASE_MODEL, SAE_CONFIG)

    # Select subset of references for plotting
    plot_refs = references[:N_REFS_PLOT]
    print(f"\nGenerating development plot: {len(plot_refs)} of {len(references)} analyzed references")

    # Filter results to only include plotted references
    plot_sparse_results = {
        key: val for key, val in all_sparse_results.items()
        if any(key == (mod, ref_id, ref_slice) for mod, ref_id, ref_slice, _ in plot_refs)
    }
    plot_dense_results = {
        key: val for key, val in all_dense_results.items()
        if any(key == (mod, ref_id, ref_slice) for mod, ref_id, ref_slice, _ in plot_refs)
    }
    plot_features = {
        key: val for key, val in all_sparse_features.items()
        if any(key == (mod, ref_id, ref_slice) for mod, ref_id, ref_slice, _ in plot_refs)
    }

    viz_output_path = config_output_dir / f"sparse_fingerprint_results_{FEATURE_SELECTION_MODE}{k_label}.png"
    visualize_fingerprint_results(
        plot_refs, plot_sparse_results, plot_dense_results, plot_features, metadata_dataset,
        sparse_features, always_active_features, viz_output_path, BASE_MODEL, SAE_CONFIG
    )

    # Generate manuscript visualization
    print(f"\nGenerating manuscript visualization...")
    if len(manuscript_refs) == 0:
        print("WARNING: Warning: No manuscript references found, skipping manuscript visualization")
    else:
        k_for_viz = VISUALIZATION_K if ENABLE_K_SWEEP else TOP_K_FEATURES
        manuscript_viz_path = config_output_dir / f"sparse_fingerprint_manuscript_{FEATURE_SELECTION_MODE}{k_label}.png"
        visualize_fingerprint_results_manuscript(
            manuscript_refs,
            manuscript_sparse_results,
            manuscript_dense_fp_results,
            manuscript_sparse_features,
            manuscript_dense_features,
            metadata_dataset,
            sparse_features,
            dense_features,
            always_active_features,
            manuscript_viz_path,
            n_retrievals=TOP_N_RETRIEVALS_MANUSCRIPT,
            top_k_features=k_for_viz,
            colors=("magenta", "cyan")
        )

    # Print summary
    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"Output directory: {config_output_dir}")

    if ENABLE_K_SWEEP:
        print(f"\nMode: K-Sweep Analysis")
        print(f"  - K values tested: {K_VALUES}")
        print(f"  - Visualization K: {VISUALIZATION_K}")
        print(f"\nK-Sweep Results:")
        for _, row in k_sweep_summary.iterrows():
            k = int(row['k'])
            sparse_q = row['mean_sparse_quality']
            dense_fp_q = row['mean_dense_fp_quality']
            print(f"  K={k:3d}: Sparse={sparse_q:.3f}, Dense FP={dense_fp_q:.3f}")
    else:
        print(f"\nMode: Single-K Analysis (K={TOP_K_FEATURES})")

    print(f"\nStatistics for K={VISUALIZATION_K if ENABLE_K_SWEEP else TOP_K_FEATURES} (based on {len(references)} references):")
    print(f"  - Mean sparse FP quality:       {summary_data['sparse_retrieval'][0]:.3f} +/- {summary_data['sparse_retrieval'][1]:.3f}")
    print(f"  - Mean dense FP quality:        {summary_data['dense_fp_retrieval'][0]:.3f} +/- {summary_data['dense_fp_retrieval'][1]:.3f}")

    print(f"\nGenerated files:")
    if ENABLE_K_SWEEP:
        print(f"\nK-Sweep Analysis:")
        print(f"  - {k_sweep_dir}/k_sweep_quality_summary.csv")
        print(f"  - {k_sweep_dir}/k_sweep_per_reference.csv")
        print(f"  - {k_sweep_dir}/k_sweep_quality_curve.png")

    print(f"\nStandard Results (K={VISUALIZATION_K if ENABLE_K_SWEEP else TOP_K_FEATURES}):")
    print(f"  - {sparse_feature_summary_path.name}")
    print(f"  - {dense_feature_summary_path.name}")
    print(f"  - {sparse_retrieved_path.name}")
    print(f"  - {dense_fp_retrieved_path.name}")
    print(f"  - {comparison_path.name}")
    print(f"  - {summary_path.name}")

    print(f"\nManuscript Results:")
    print(f"  - {manuscript_sparse_summary_path.name}")
    print(f"  - {manuscript_dense_summary_path.name}")
    print(f"  - {manuscript_sparse_retrieved_path.name}")
    print(f"  - {manuscript_dense_fp_retrieved_path.name}")

    print(f"\nVisualizations:")
    print(f"  - {comparison_viz_path.name}")
    print(f"  - {viz_output_path.name}")
    if len(manuscript_refs) > 0:
        print(f"  - {manuscript_viz_path.name}")
    print("=" * 70)


if __name__ == "__main__":
    main()
