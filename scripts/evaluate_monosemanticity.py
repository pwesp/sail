#!/usr/bin/env python3
"""
Independent two-dimensional interpretability analysis of sparse SAE features.

Goal: Score all sparse features in each SAE configuration on two orthogonal axes:
1. Semantic coherence (C): Does the feature consistently activate for the same kind of thing?
2. Semantic specificity (S): How narrow is the encoded concept?

This analysis is INDEPENDENT of downstream task performance. Features are scored purely
on what they encode, not on how useful they are for specific classification tasks.

Loops over all base_model x SAE_config pairs and produces per-config, per-model, and
global aggregated output CSVs.

Workflow (per config):
1. Score ALL features in the SAE dictionary (subject to minimal filtering)
2. For each feature:
   - Retrieve top-N activating samples from the validation set
   - Compute coherence: average pairwise Jaccard over organ-presence vectors
   - Compute specificity: normalized entropy over organ presence counts
3. Aggregate to configuration-level scores: C (coherence), S (specificity), M = C x S (monosemanticity)
4. Visualize top features sorted by coherence, specificity, or combined metric

Implementation details:
- Per-config analysis logic: src/monosemanticity.py:run_single_config
- This script is the loop driver: discover configs -> run each -> aggregate -> visualize
- See context/interpretability_score.md for full rationale and methodology
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# General imports
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from typing import Dict, List, Set, Tuple

# Local imports
from src.monosemanticity import discover_all_configs, run_single_config

# ==================================================
# Settings
# ==================================================

SPLIT = "valid"  # "train", "valid", or "test"
# Use "valid" (default): downstream scripts (evaluate_sae_configs.py, plot_sae_quality_and_perf_rec.py,
# evaluate_fingerprint_retrieval.py, evaluate_autointerp.py) all read from results/monosemanticity/valid/.

# Parquet filenames use "train" / "val" / "test"; the rest of the codebase uses "train" / "valid" / "test".
PARQUET_SPLIT: str = {"train": "train", "valid": "val", "test": "test"}[SPLIT]

SPARSE_FEATURES_ROOT = Path("results/total_seg_sparse_features")
ANALYSIS_ROOT        = Path("results/monosemanticity")
SPLIT_DIR            = ANALYSIS_ROOT / SPLIT  # null_baselines.csv and aggregated CSVs live here

SPARSE_FEATURES_DIR_MAP: Dict[str, str] = {
    "biomedparse":        "biomedparse_matryoshka_sae",
    "dinov3":             "dinov3_matryoshka_sae",
    "biomedparse_random": "biomedparse_random_matryoshka_sae",
}

# Analysis parameters
TOP_N_SAMPLES_PER_FEATURE     = 10     # N for coherence and specificity scores
TOP_N_STABILITY_CHECK         = 50     # N for stability check (see context/interpretability_score.md)
MIN_ACTIVATION_THRESHOLD      = 0.0    # Minimum activation value to consider
NULL_BASELINE_N_DRAWS: int    = 100    # Monte Carlo draws for null baseline
CONCEPT_THRESHOLD: float      = 0.75   # Fraction of top-N samples required for a concept to be "shared"
# Columns excluded from concept extraction: identifiers, paths, and values constant within a split.
CONCEPT_EXCLUDE_COLUMNS: Set[str] = {"image_id", "image_path", "split", "slice_idx", "activation"}

# Visualization
N_FEATURES_VIZ: int = 10     # Max features to visualize per config (capped by pool size)
N_SAMPLES_VIZ: int  = 5      # Samples per feature in visualization

# ==================================================
# Main
# ==================================================

def main() -> None:
    configs: List[Tuple[str, str]] = discover_all_configs(SPARSE_FEATURES_ROOT, SPARSE_FEATURES_DIR_MAP)
    print(f"Discovered {len(configs)} configs across {len(set(bm for bm, _ in configs))} base models")
    for bm, cfg in configs:
        print(f"  {bm} / {cfg}")

    all_feature_scores: List[pd.DataFrame] = []
    all_config_scores:  List[pd.DataFrame] = []

    for base_model, sae_config in configs:
        print(f"\n{'=' * 70}")
        print(f"  {base_model}  /  {sae_config}")
        print(f"{'=' * 70}")

        feature_scores_df, config_score_df, feature_analysis = run_single_config(
            base_model=base_model,
            sae_config=sae_config,
            sparse_features_root=SPARSE_FEATURES_ROOT,
            sparse_features_dir_map=SPARSE_FEATURES_DIR_MAP,
            split_dir=SPLIT_DIR,
            split=SPLIT,
            parquet_split=PARQUET_SPLIT,
            top_n_samples_per_feature=TOP_N_SAMPLES_PER_FEATURE,
            top_n_stability_check=TOP_N_STABILITY_CHECK,
            min_activation_threshold=MIN_ACTIVATION_THRESHOLD,
            null_baseline_n_draws=NULL_BASELINE_N_DRAWS,
            concept_threshold=CONCEPT_THRESHOLD,
            concept_exclude_columns=CONCEPT_EXCLUDE_COLUMNS,
        )

        all_feature_scores.append(feature_scores_df)
        all_config_scores.append(config_score_df)

        # --- Visualize top-activating samples ---------------------------------
        output_dir: Path = SPLIT_DIR / base_model / sae_config
        n_viz_features: int = min(N_FEATURES_VIZ, len(feature_analysis))

        # Create two visualizations: one sorted by coherence, one by specificity
        for sort_by, sort_key_fn in [
            ("coherence",   lambda f: float(feature_analysis[f]["coherence_10"])),   # type: ignore[arg-type]
            ("specificity", lambda f: float(feature_analysis[f]["specificity_10"])), # type: ignore[arg-type]
        ]:
            top_feature_indices: List[int] = sorted(
                feature_analysis,
                key=sort_key_fn,
                reverse=True,
            )[:n_viz_features]

            fig, axes = plt.subplots(
                n_viz_features, N_SAMPLES_VIZ + 1,
                figsize=(N_SAMPLES_VIZ * 3 + 2, n_viz_features * 3),
                gridspec_kw={"width_ratios": [1] * N_SAMPLES_VIZ + [0.55]},
            )
            # When n_viz_features == 1, subplots returns a 1-D array; normalise to 2-D.
            if n_viz_features == 1:
                axes = axes[np.newaxis, :]  # type: ignore[index]

            for row_idx, feature_idx in enumerate(top_feature_indices):
                info = feature_analysis[feature_idx]
                top_metadata: pd.DataFrame = info["metadata"]  # type: ignore[assignment]

                for col_idx in range(N_SAMPLES_VIZ):
                    ax = axes[row_idx, col_idx]
                    img_path: str = str(top_metadata.iloc[col_idx]["image_path"])
                    img: Image.Image = Image.open(img_path).convert("L")  # grayscale
                    ax.imshow(np.array(img), cmap="gray")
                    ax.axis("off")

                # Label column: all non-organ concepts + up to 3 organ concepts
                label_ax = axes[row_idx, N_SAMPLES_VIZ]
                label_ax.axis("off")
                shared_concepts: List[str] = info["shared_concepts"]  # type: ignore[assignment]
                organ_concepts: List[str]  = [c for c in shared_concepts if c.startswith("organ=")]
                other_concepts: List[str]  = [c for c in shared_concepts if not c.startswith("organ=")]
                viz_concepts:   List[str]  = other_concepts + organ_concepts[:3]
                concepts_str: str = ", ".join(viz_concepts) if viz_concepts else "-"
                label_ax.text(
                    0.05, 0.5,
                    f"Feature {feature_idx}\nShared labels: {concepts_str}",
                    transform=label_ax.transAxes,
                    fontsize=8, va="center", ha="left",
                )

            plt.suptitle(
                f"Interpretable Features (Model: {base_model}, SAE: {sae_config}, Sort: {sort_by})",
                fontsize=11,
            )
            plot_path: Path = output_dir / f"feature_samples_by_{sort_by}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches="tight")
            print(f"\nSaved {sort_by}-sorted visualization to {plot_path}")
            plt.close()

    # ----------------------------------------------------------
    # Aggregate and save
    # ----------------------------------------------------------

    # Per-base_model aggregation
    for model_name in sorted(set(bm for bm, _ in configs)):
        model_feature_dfs: List[pd.DataFrame] = [
            df for (bm, _), df in zip(configs, all_feature_scores) if bm == model_name
        ]
        model_config_dfs: List[pd.DataFrame] = [
            df for (bm, _), df in zip(configs, all_config_scores) if bm == model_name
        ]
        model_dir: Path = SPLIT_DIR / model_name
        model_dir.mkdir(parents=True, exist_ok=True)
        pd.concat(model_feature_dfs, ignore_index=True).to_csv(model_dir / "feature_scores_per_model.csv", index=False)
        pd.concat(model_config_dfs,  ignore_index=True).to_csv(model_dir / "config_scores_per_model.csv",  index=False)
        print(f"\nSaved per-model aggregates for {model_name} ({len(model_feature_dfs)} configs)")

    # Global aggregation
    SPLIT_DIR.mkdir(parents=True, exist_ok=True)
    pd.concat(all_feature_scores, ignore_index=True).to_csv(SPLIT_DIR / "feature_scores_all.csv", index=False)
    pd.concat(all_config_scores,  ignore_index=True).to_csv(SPLIT_DIR / "config_scores_all.csv",  index=False)
    print(f"\nSaved global aggregates: {len(all_feature_scores)} configs, "
          f"{sum(len(df) for df in all_feature_scores)} total feature rows")


if __name__ == "__main__":
    main()
