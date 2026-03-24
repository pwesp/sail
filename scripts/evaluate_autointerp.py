#!/usr/bin/env python3
"""
Automated interpretation of SAE sparse features using a vision-language model (VLM).

Pipeline (per feature):
  1. Retrieve diverse high-activation images via greedy max-dissimilarity sampling:
       top N_CANDIDATE_POOL images by activation -> pick N_SAMPLES_PER_FEATURE most dissimilar
  2. Feed images + activation context to VLM -> natural language concept description
  3. Evaluate concept quality with an LLM-as-judge setup (optional, ENABLE_EVALUATION):
       - Generate concept from TRAIN images
       - Sample N_DISTRACTORS distractor features and generate their concepts
       - Ask VLM to rank all (1 + N_DISTRACTORS) concepts against unseen VAL images
       - Success = true concept ranks <= 2
  4. Save per-feature visualisation and aggregate results CSV

Feature selection modes (FEATURE_SELECTION_MODE):
  top_n   Top-N features by combined monosemanticity + task importance score.
          Scores are computed on-the-fly from results/monosemanticity/ and
          results/linear_probes/.
  custom  Explicit list of feature indices (CUSTOM_FEATURE_IDS).
  all     Every feature active in >= 10 samples.

Outputs (results/auto_interpretation/{model}/{config}/{mode}/):
  feature_descriptions.csv    Concept + evaluation rank per feature
  figures/                    Per-feature visualisation PNGs
  evaluation_metrics.json     Aggregate success rates and rank distributions
  concept_cache.json          Cached distractor concepts (avoids recomputation on reruns)
  processing_log.txt          Full run log
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import datetime
import json
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Tuple

# Local imports
from src.dataloading import (
    load_sparse_features_dataset,
    load_interpretable_features,
    TotalSegmentatorLatentFeaturesDataset
)
from src.autointerp import (
    load_vlm_model,
    generate_feature_description,
    extract_metadata_statistics,
    rank_concepts_for_images
)
from src.feature_sampling import retrieve_diverse_samples
from src.autointerp_visualization import visualize_feature_interpretation
from src.eval import (
    sample_distractor_features,
    ConceptCache,
    compute_evaluation_metrics,
    format_evaluation_summary
)

# ==================================================
# Settings
# ==================================================

# -- Model ---------------------------------------------------------------------
SAE_CONFIGS = {
    "biomedparse":        "D_128_512_2048_8192_K_20_40_80_160",
    "dinov3":             "D_128_512_2048_8192_K_5_10_20_40",
    "biomedparse_random": "D_64_256_1024_4096_K_30_60_120_240",
}
BASE_MODEL = "biomedparse"  # "biomedparse" | "dinov3" | "biomedparse_random"
SAE_CONFIG = SAE_CONFIGS[BASE_MODEL]

# -- Feature selection ---------------------------------------------------------
FEATURE_SELECTION_MODE = "top_n"   # "top_n" | "custom" | "all"  (see module docstring)
TOP_N_FEATURES = 250               # (top_n) number of features to analyze
CUSTOM_FEATURE_IDS = [1544, 36, 1838, 425, 689, 111, 433, 28, 253, 1879, 457, 144, 64, 1789, 7041]
                                   # (custom) explicit feature indices to analyze

# -- Image sampling ------------------------------------------------------------
TOP_N_SAMPLES_PER_FEATURE = 5  # images passed to the VLM per feature
N_CANDIDATE_POOL = 20          # pre-selection pool size: take top-N by activation (unique
                               # image_ids only), then pick TOP_N_SAMPLES_PER_FEATURE via
                               # greedy max-dissimilarity on the full sparse feature vectors

# -- VLM - concept generation --------------------------------------------------
VLM_MODEL = "google/medgemma-27b-it"  # options: gemma-3-{1b,4b,7b}-it, medgemma-{4b,27b}-it
VLM_TEMPERATURE = 0.3  # generation temperature
VLM_MAX_TOKENS = 100   # max output tokens for the concept description
N_CONSISTENCY_RUNS = 1 # >1: run VLM multiple times and pick the most frequent output;
                       # rarely useful for free-form text (exact-match voting)

# -- Evaluation - LLM-as-judge -------------------------------------------------
ENABLE_EVALUATION = True   # False = concept generation only (faster, skips ranking)
N_DISTRACTORS = 4          # distractor features for contrastive ranking;
                           # VLM ranks 1 + N_DISTRACTORS concepts -> success if true rank <= 2
EVAL_MIN_ACTIVATIONS = 10  # minimum activation count for a feature to be eligible as distractor
RANKING_TEMPERATURE = 0.2  # lower = more deterministic ranking decisions

# -- Directories ---------------------------------------------------------------
SPARSE_FEATURES_ROOT     = Path("results/total_seg_sparse_features")
INTERPRETABILITY_ROOT    = Path("results/monosemanticity")
LINEAR_PROBES_ROOT = Path("results/linear_probes")
OUTPUT_DIR = (
    Path("results/auto_interpretation")
    / BASE_MODEL / SAE_CONFIG / FEATURE_SELECTION_MODE
)


# ==================================================
# Data Loading
# ==================================================

def load_sparse_features() -> Tuple[np.ndarray, TotalSegmentatorLatentFeaturesDataset]:
    """Load sparse feature activations and metadata from parquet (validation set)."""
    sparse_features, sparse_dataset = load_sparse_features_dataset(
        base_model=BASE_MODEL,
        sae_config=SAE_CONFIG,
        split="val",
        sparse_features_root=SPARSE_FEATURES_ROOT,
        as_numpy=True
    )
    print(f"Loaded {len(sparse_features)} images with {sparse_features.shape[1]} features")
    return sparse_features, sparse_dataset


def load_training_sparse_features() -> Tuple[np.ndarray, TotalSegmentatorLatentFeaturesDataset]:
    """Load sparse feature activations and metadata from parquet (training set)."""
    sparse_features, sparse_dataset = load_sparse_features_dataset(
        base_model=BASE_MODEL,
        sae_config=SAE_CONFIG,
        split="train",
        sparse_features_root=SPARSE_FEATURES_ROOT,
        as_numpy=True
    )
    print(f"Loaded {len(sparse_features)} training images with {sparse_features.shape[1]} features")
    return sparse_features, sparse_dataset


def find_always_active_features(sparse_features: np.ndarray) -> set:
    """
    Find features that are always active (>0) across all images.
    These should be excluded from feature selection.
    """
    print("\nChecking for always-active features...")
    always_active = set()

    for feat_idx in range(sparse_features.shape[1]):
        min_activation = sparse_features[:, feat_idx].min()
        if min_activation > 0:
            always_active.add(feat_idx)

    print(f"Found {len(always_active)} features with activation > 0 across all images")
    if len(always_active) > 0:
        print(f"Always-active features: {sorted(always_active)[:20]}...")
    return always_active



def main() -> None:
    # Validate mode
    valid_modes = ["top_n", "custom", "all"]
    if FEATURE_SELECTION_MODE not in valid_modes:
        raise ValueError(f"Invalid FEATURE_SELECTION_MODE: {FEATURE_SELECTION_MODE}. Must be one of {valid_modes}")

    # Create output directory
    figures_dir = OUTPUT_DIR / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # Start logging
    log_path = OUTPUT_DIR / "processing_log.txt"
    log_file = open(log_path, 'w')

    def log(message: str):
        print(message)
        log_file.write(message + '\n')
        log_file.flush()

    log("=" * 70)
    log("AUTOMATED FEATURE INTERPRETATION")
    log("=" * 70)
    log(f"Started: {datetime.datetime.now()}")
    log(f"Base model: {BASE_MODEL}")
    log(f"SAE config: {SAE_CONFIG}")
    log(f"Mode: {FEATURE_SELECTION_MODE}")

    if FEATURE_SELECTION_MODE == "top_n":
        log(f"Top-N features: {TOP_N_FEATURES}")
    elif FEATURE_SELECTION_MODE == "custom":
        log(f"Custom features: {CUSTOM_FEATURE_IDS}")
        log(f"Number of features: {len(CUSTOM_FEATURE_IDS)}")

    log(f"Samples per feature: {TOP_N_SAMPLES_PER_FEATURE}")
    log(f"Candidate pool: {N_CANDIDATE_POOL}")
    log(f"VLM model: {VLM_MODEL}")
    log(f"VLM temperature: {VLM_TEMPERATURE}")
    log(f"Self-consistency runs: {N_CONSISTENCY_RUNS}")
    log(f"Evaluation enabled: {ENABLE_EVALUATION}")
    if ENABLE_EVALUATION:
        log(f"  Number of distractors: {N_DISTRACTORS}")
        log(f"  Ranking temperature: {RANKING_TEMPERATURE}")

    # Phase 1: Load data
    log("\n" + "=" * 70)
    log("PHASE 1: LOADING DATA")
    log("=" * 70)

    # Load validation data (for evaluation or single-split mode)
    sparse_features, metadata_dataset = load_sparse_features()
    always_active_features = find_always_active_features(sparse_features)

    # Load training data (if evaluation enabled)
    train_sparse_features: np.ndarray | None = None
    train_metadata_dataset: TotalSegmentatorLatentFeaturesDataset | None = None
    concept_cache: ConceptCache | None = None
    
    if ENABLE_EVALUATION:
        train_sparse_features, train_metadata_dataset = load_training_sparse_features()
        # Concept cache for distractors
        concept_cache = ConceptCache(cache_path=OUTPUT_DIR / "concept_cache.json")
        log(f"[ok] Concept cache initialized ({len(concept_cache)} cached concepts)")

    # Phase 2: Select features
    log("\n" + "=" * 70)
    log("PHASE 2: FEATURE SELECTION")
    log("=" * 70)

    if FEATURE_SELECTION_MODE == "top_n":
        log(f"Using top-{TOP_N_FEATURES} selection based on interpretability + task importance")
        selected_features_df = load_interpretable_features(
            base_model=BASE_MODEL,
            sae_config=SAE_CONFIG,
            top_n=TOP_N_FEATURES,
            always_active_features=always_active_features,
            interpretability_root=INTERPRETABILITY_ROOT,
            linear_probes_root=LINEAR_PROBES_ROOT
        )
        log(f"[ok] Selected top {len(selected_features_df)} features for analysis")

    elif FEATURE_SELECTION_MODE == "custom":
        log(f"Using custom feature list: {CUSTOM_FEATURE_IDS}")
        filtered_custom_ids = [
            feat_idx for feat_idx in CUSTOM_FEATURE_IDS
            if feat_idx not in always_active_features
        ]
        if len(filtered_custom_ids) < len(CUSTOM_FEATURE_IDS):
            removed = set(CUSTOM_FEATURE_IDS) - set(filtered_custom_ids)
            log(f"WARNING: Removed {len(removed)} always-active features: {sorted(removed)}")
        selected_features_df = pd.DataFrame({
            'feature_idx': filtered_custom_ids,
            'selection_mode': 'custom',
        })
        log(f"[ok] Selected {len(selected_features_df)} custom features for analysis")

    elif FEATURE_SELECTION_MODE == "all":
        log("Finding all features active in >=10 samples...")
        n_features = sparse_features.shape[1]
        active_features = []
        min_samples = 10  # Minimum number of activations required

        for feat_idx in range(n_features):
            if feat_idx in always_active_features:
                continue
            n_activations = (sparse_features[:, feat_idx] > 0).sum()
            if n_activations >= min_samples:
                active_features.append(feat_idx)

        log(f"Found {len(active_features)} features active in >={min_samples} samples (excluding {len(always_active_features)} always-active)")
        selected_features_df = pd.DataFrame({
            'feature_idx': active_features,
            'selection_mode': 'all',
        })
        log(f"[ok] Selected {len(selected_features_df)} features for analysis")

    # Save selected features
    selected_features_path = OUTPUT_DIR / "selected_features.csv"
    selected_features_df.to_csv(selected_features_path, index=False)
    log(f"[ok] Saved selected features to {selected_features_path}")

    # Phase 3: Load VLM
    log("\n" + "=" * 70)
    log("PHASE 3: LOADING VLM")
    log("=" * 70)

    vlm_pipe = load_vlm_model(VLM_MODEL)

    # Phase 4: Process each feature
    log("\n" + "=" * 70)
    if ENABLE_EVALUATION:
        log("PHASE 4: CONCEPT GENERATION & EVALUATION")
    else:
        log("PHASE 4: GENERATING DESCRIPTIONS")
    log("=" * 70)

    results = []
    sample_metadata_records = []
    activation_stats_records = []

    # Pre-build metadata arrays for fast activation-stats computation
    _modalities = np.array([str(m).lower() for m in metadata_dataset.modalities])
    _genders    = np.array([str(g).lower() for g in metadata_dataset.metadata["gender"]])
    _ages       = np.array([float(a) for a in metadata_dataset.metadata["age"]])

    for position, (idx, row) in enumerate(selected_features_df.iterrows()):
        feature_idx = int(row['feature_idx'])
        log(f"\nProcessing feature {feature_idx} ({position + 1}/{len(selected_features_df)})...")

        # ===== TRAINING DATA: Concept generation =====
        if ENABLE_EVALUATION:
            # At this point, we know ENABLE_EVALUATION is True, so these are not None
            assert train_sparse_features is not None
            assert train_metadata_dataset is not None
            assert concept_cache is not None
            
            # Retrieve training samples for concept generation
            train_samples = retrieve_diverse_samples(
                sparse_features=train_sparse_features,
                metadata_dataset=train_metadata_dataset,
                feature_idx=feature_idx,
                n_candidates=N_CANDIDATE_POOL,
                n_samples=TOP_N_SAMPLES_PER_FEATURE
            )

            log(f"  [TRAIN] Retrieved {len(train_samples)} diverse samples")
            log(f"  [TRAIN] Activation range: {train_samples[0]['activation']:.3f} to {train_samples[-1]['activation']:.3f}")

            # Load training images
            train_images = []
            for sample in train_samples:
                img_path = sample['image_path']
                img = Image.open(img_path).convert('RGB')
                train_images.append(img)

            # Generate concept from training data
            log(f"  [TRAIN] Generating concept...")
            true_concept = generate_feature_description(
                vlm_pipe=vlm_pipe,
                images=train_images,
                samples=train_samples,
                temperature=VLM_TEMPERATURE,
                max_tokens=VLM_MAX_TOKENS,
                n_consistency_runs=N_CONSISTENCY_RUNS
            )

            log(f"  [TRAIN] [ok] True concept: {true_concept}")

            # Sample distractor features
            try:
                distractor_indices = sample_distractor_features(
                    sparse_features=train_sparse_features,
                    target_feature_idx=feature_idx,
                    always_active_features=always_active_features,
                    n_distractors=N_DISTRACTORS,
                    min_activations=EVAL_MIN_ACTIVATIONS
                )
                log(f"  [TRAIN] Sampled {len(distractor_indices)} distractor features: {distractor_indices}")
            except ValueError as e:
                log(f"  [TRAIN] WARNING: Failed to sample distractors: {e}")
                log(f"  [TRAIN] WARNING: Skipping evaluation for this feature")
                distractor_indices = []

            # Generate concepts for distractors
            distractor_concepts = []
            for dist_idx in distractor_indices:
                # Check cache first
                if concept_cache.has(dist_idx):
                    dist_concept = concept_cache.get(dist_idx)
                    log(f"  [TRAIN] Distractor {dist_idx}: {dist_concept} (cached)")
                else:
                    # Generate concept
                    dist_samples = retrieve_diverse_samples(
                        sparse_features=train_sparse_features,
                        metadata_dataset=train_metadata_dataset,
                        feature_idx=dist_idx,
                        n_candidates=N_CANDIDATE_POOL,
                        n_samples=TOP_N_SAMPLES_PER_FEATURE
                    )

                    dist_images = [Image.open(s['image_path']).convert('RGB') for s in dist_samples]

                    dist_concept = generate_feature_description(
                        vlm_pipe=vlm_pipe,
                        images=dist_images,
                        samples=dist_samples,
                        temperature=VLM_TEMPERATURE,
                        max_tokens=VLM_MAX_TOKENS,
                        n_consistency_runs=N_CONSISTENCY_RUNS
                    )

                    # Cache it
                    concept_cache.set(dist_idx, dist_concept)
                    log(f"  [TRAIN] Distractor {dist_idx}: {dist_concept}")

                distractor_concepts.append(dist_concept)

            # Prepare all concepts for ranking
            all_concepts = [true_concept] + distractor_concepts

        else:
            # No evaluation - use validation data for concept generation
            train_samples = None
            true_concept = None
            distractor_indices = []
            distractor_concepts = []
            all_concepts = []

        # ===== VALIDATION DATA: Retrieval for evaluation or visualization =====
        val_samples = retrieve_diverse_samples(
            sparse_features=sparse_features,
            metadata_dataset=metadata_dataset,
            feature_idx=feature_idx,
            n_candidates=N_CANDIDATE_POOL,
            n_samples=TOP_N_SAMPLES_PER_FEATURE
        )

        log(f"  [VAL] Retrieved {len(val_samples)} diverse samples")
        log(f"  [VAL] Activation range: {val_samples[0]['activation']:.3f} to {val_samples[-1]['activation']:.3f}")

        # Load validation images
        val_images = []
        for sample in val_samples:
            img_path = sample['image_path']
            img = Image.open(img_path).convert('RGB')
            val_images.append(img)

        # Generate description (if not using evaluation)
        description: str
        if not ENABLE_EVALUATION:
            log(f"  [VAL] Generating VLM description...")
            description = generate_feature_description(
                vlm_pipe=vlm_pipe,
                images=val_images,
                samples=val_samples,
                temperature=VLM_TEMPERATURE,
                max_tokens=VLM_MAX_TOKENS,
                n_consistency_runs=N_CONSISTENCY_RUNS
            )
            log(f"  [VAL] [ok] Description: {description}")
        else:
            # When ENABLE_EVALUATION is True, true_concept is guaranteed to be set
            assert true_concept is not None
            description = true_concept

        # ===== EVALUATION: Rank concepts on validation data =====
        if ENABLE_EVALUATION and len(all_concepts) == 1 + N_DISTRACTORS:
            log(f"  [VAL] Ranking {len(all_concepts)} concepts...")
            ranking_result = rank_concepts_for_images(
                vlm_pipe=vlm_pipe,
                images=val_images,
                concepts=all_concepts,
                samples=val_samples,
                temperature=RANKING_TEMPERATURE
            )

            # Get rank of true concept (which is at index 0)
            true_concept_rank = ranking_result['ranking'].index(0) + 1  # Convert to 1-indexed
            is_successful = true_concept_rank <= 2

            log(f"  [VAL] [ok] Ranking: {ranking_result['ranking']}")
            log(f"  [VAL] [ok] True concept rank: {true_concept_rank}/{1 + N_DISTRACTORS} {'[ok] SUCCESS' if is_successful else 'FAILURE'}")
            if not ranking_result['parse_success']:
                log(f"  [VAL] WARNING: Ranking parse failed, using fallback order")

        else:
            # No evaluation
            true_concept_rank = None
            is_successful = None
            ranking_result = None

        # Record sample metadata (image_id, slice_nr, modality, slice_orientation)
        for rank, sample in enumerate(val_samples, start=1):
            arr_idx = sample["array_idx"]
            sample_metadata_records.append({
                "feature_idx":       feature_idx,
                "sample_rank":       rank,
                "array_idx":         arr_idx,
                "image_id":          metadata_dataset.image_ids[arr_idx],
                "slice_nr":          int(metadata_dataset.slice_indices[arr_idx]),
                "modality":          sample["modality"],
                "slice_orientation": sample["slice_orientation"],
            })

        # Activation statistics over all val samples where feature is active
        act_mask = sparse_features[:, feature_idx] > 0
        activation_stats_records.append({
            "feature_idx":  feature_idx,
            "n_activating": int(act_mask.sum()),
            "n_ct":         int((_modalities[act_mask] == "ct").sum()),
            "n_mri":        int((_modalities[act_mask] == "mri").sum()),
            "n_male":       int((_genders[act_mask] == "m").sum()),
            "n_female":     int((_genders[act_mask] == "f").sum()),
            "mean_age":     float(np.nanmean(_ages[act_mask])) if act_mask.any() else float("nan"),
        })

        # Extract metadata statistics
        metadata_stats = extract_metadata_statistics(val_samples)

        # Visualize (use validation samples)
        figure_path = figures_dir / f"feature_{feature_idx:04d}_autointerp.png"
        visualize_feature_interpretation(
            feature_idx,
            description,
            val_samples,
            figure_path,
            true_concept_rank=true_concept_rank  # Pass rank if available
        )
        log(f"  [ok] Saved figure to {figure_path.name}")

        # Store results
        result = {
            'feature_idx': feature_idx,
            'description': description,
            'top_modalities': metadata_stats['top_modalities'],
            'top_orientations': metadata_stats['top_orientations'],
            'top_structures': metadata_stats['top_structures'],
            'mean_activation': metadata_stats['mean_activation']
        }

        # Add ranking info if available (only for top-N mode)
        if 'combined_rank' in row:
            result['combined_rank'] = row['combined_rank']
        if 'monosemanticity_50' in row:
            result['interp_score'] = row.get('monosemanticity_50', np.nan)

        # Add evaluation results
        if ENABLE_EVALUATION and ranking_result:
            result['true_concept_rank'] = true_concept_rank
            result['is_successful'] = is_successful
            result['distractor_features'] = distractor_indices
            result['distractor_concepts'] = distractor_concepts
            result['ranking_output'] = ranking_result['raw_output']
            result['parse_success'] = ranking_result['parse_success']
            result['train_activation_mean'] = train_samples[0]['activation'] if train_samples else np.nan
            result['val_activation_mean'] = val_samples[0]['activation']

        results.append(result)

    # Save concept cache if evaluation was enabled
    if ENABLE_EVALUATION and concept_cache:
        concept_cache.save()
        log(f"\n[ok] Saved concept cache ({len(concept_cache)} concepts)")

    # Phase 5: Export results
    log("\n" + "=" * 70)
    log("PHASE 5: EXPORTING RESULTS")
    log("=" * 70)

    results_df = pd.DataFrame(results)
    results_path = OUTPUT_DIR / "feature_descriptions.csv"
    results_df.to_csv(results_path, index=False)
    log(f"[ok] Saved feature descriptions to {results_path}")

    sample_metadata_path = OUTPUT_DIR / "feature_sample_metadata.csv"
    pd.DataFrame(sample_metadata_records).to_csv(sample_metadata_path, index=False)
    log(f"[ok] Saved feature sample metadata to {sample_metadata_path}")

    activation_stats_path = OUTPUT_DIR / "feature_activation_stats.csv"
    pd.DataFrame(activation_stats_records).to_csv(activation_stats_path, index=False)
    log(f"[ok] Saved feature activation stats to {activation_stats_path}")

    log(f"\n[ok] Generated {len(results)} feature descriptions")
    log(f"[ok] Created {len(results)} visualization figures")

    # Evaluation summary (if enabled)
    if ENABLE_EVALUATION:
        log("\n" + "=" * 70)
        log("PHASE 6: EVALUATION METRICS")
        log("=" * 70)

        # Filter results with evaluation data
        eval_results = [r for r in results if r.get('true_concept_rank') is not None]

        if len(eval_results) > 0:
            metrics = compute_evaluation_metrics(eval_results)
            summary = format_evaluation_summary(metrics)
            log(summary)

            # Save evaluation metrics
            eval_metrics_path = OUTPUT_DIR / "evaluation_metrics.json"
            with open(eval_metrics_path, 'w') as f:
                json.dump(metrics, f, indent=2)
            log(f"\n[ok] Saved evaluation metrics to {eval_metrics_path}")
        else:
            log("WARNING: No features with evaluation results")

    # Summary
    log("\n" + "=" * 70)
    log("ANALYSIS COMPLETE")
    log("=" * 70)
    log(f"Finished: {datetime.datetime.now()}")
    log(f"Output directory: {OUTPUT_DIR}")
    log(f"  - selected_features.csv: {len(selected_features_df)} features")
    log(f"  - feature_descriptions.csv: {len(results)} descriptions")
    log(f"  - figures/: {len(results)} visualization PNGs")
    if ENABLE_EVALUATION:
        log(f"  - evaluation_metrics.json: Evaluation summary")
        log(f"  - concept_cache.json: Cached distractor concepts")
    log("=" * 70)

    log_file.close()
    print(f"\n[ok] Processing log saved to {log_path}")


if __name__ == "__main__":
    main()
