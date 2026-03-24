#!/usr/bin/env python3
"""
Retrospectively extract sample metadata for autointerp feature visualizations.

For each feature in feature_descriptions.csv, re-runs the deterministic
retrieve_diverse_samples() call (val split, same parameters as evaluate_autointerp.py)
and saves two CSVs to the same directory:

  feature_sample_metadata.csv
    One row per (feature, sample). Columns:
    feature_idx, sample_rank, array_idx, image_id, slice_nr, modality, slice_orientation

  feature_activation_stats.csv
    One row per feature. Aggregate statistics over ALL val samples where the
    feature activates (activation > 0). Columns:
    feature_idx, n_activating, n_ct, n_mri, n_male, n_female, mean_age
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from pathlib import Path
from typing import cast

from src.dataloading import load_sparse_features_dataset, TotalSegmentatorLatentFeaturesDataset
from src.feature_sampling import retrieve_diverse_samples

# -- Configs to process --------------------------------------------------------

CONFIGS = [
    {
        "base_model": "dinov3",
        "sae_config": "D_128_512_2048_8192_K_5_10_20_40",
        "mode": "top_n",
    },
    {
        "base_model": "biomedparse",
        "sae_config": "D_128_512_2048_8192_K_20_40_80_160",
        "mode": "top_n",
    },
]

# Must match evaluate_autointerp.py
N_CANDIDATE_POOL        = 20
N_SAMPLES_PER_FEATURE   = 5
SPARSE_FEATURES_ROOT    = Path("results/total_seg_sparse_features")
AUTO_INTERP_ROOT        = Path("results/auto_interpretation")


def compute_activation_stats(
    feature_indices: list,
    sparse_features: np.ndarray,
    metadata_dataset: TotalSegmentatorLatentFeaturesDataset,
) -> pd.DataFrame:
    """Compute population-level activation statistics for each feature over all val samples."""
    modalities = np.array([str(m).lower() for m in metadata_dataset.modalities])
    genders    = np.array([str(g).lower() for g in metadata_dataset.metadata["gender"]])
    ages       = np.array([float(a) for a in metadata_dataset.metadata["age"]])

    rows = []
    for feature_idx in feature_indices:
        mask = sparse_features[:, feature_idx] > 0
        rows.append({
            "feature_idx":  feature_idx,
            "n_activating": int(mask.sum()),
            "n_ct":         int((modalities[mask] == "ct").sum()),
            "n_mri":        int((modalities[mask] == "mri").sum()),
            "n_male":       int((genders[mask] == "m").sum()),
            "n_female":     int((genders[mask] == "f").sum()),
            "mean_age":     float(np.nanmean(ages[mask])) if mask.any() else float("nan"),
        })
    return pd.DataFrame(rows)


def extract_sample_metadata(base_model: str, sae_config: str, mode: str) -> None:
    output_dir = AUTO_INTERP_ROOT / base_model / sae_config / mode
    descriptions_path = output_dir / "feature_descriptions.csv"

    if not descriptions_path.exists():
        print(f"  FAIL {descriptions_path} not found, skipping")
        return

    feature_indices = cast(list[int], pd.read_csv(descriptions_path)["feature_idx"].tolist())
    print(f"  Found {len(feature_indices)} features in feature_descriptions.csv")

    # Load val sparse features (same split used for visualization)
    sparse_features, metadata_dataset = load_sparse_features_dataset(
        base_model=base_model,
        sae_config=sae_config,
        split="val",
        sparse_features_root=SPARSE_FEATURES_ROOT,
        as_numpy=True,
    )
    print(f"  Loaded sparse features: {sparse_features.shape}")

    # -- feature_sample_metadata.csv -------------------------------------------
    records = []
    for feature_idx in feature_indices:
        samples = retrieve_diverse_samples(
            sparse_features=sparse_features,
            metadata_dataset=metadata_dataset,
            feature_idx=feature_idx,
            n_candidates=N_CANDIDATE_POOL,
            n_samples=N_SAMPLES_PER_FEATURE,
        )
        for rank, sample in enumerate(samples, start=1):
            idx = sample["array_idx"]
            records.append({
                "feature_idx":       feature_idx,
                "sample_rank":       rank,
                "array_idx":         idx,
                "image_id":          metadata_dataset.image_ids[idx],
                "slice_nr":          int(metadata_dataset.slice_indices[idx]),
                "modality":          sample["modality"],
                "slice_orientation": sample["slice_orientation"],
            })

    out_path = output_dir / "feature_sample_metadata.csv"
    pd.DataFrame(records).to_csv(out_path, index=False)
    print(f"  [ok] Saved {len(records)} rows to {out_path}")

    # -- feature_activation_stats.csv ------------------------------------------
    stats_df = compute_activation_stats(feature_indices, sparse_features, metadata_dataset)
    stats_path = output_dir / "feature_activation_stats.csv"
    stats_df.to_csv(stats_path, index=False)
    print(f"  [ok] Saved {len(stats_df)} rows to {stats_path}")


def main() -> None:
    for cfg in CONFIGS:
        print(f"\n{cfg['base_model']} / {cfg['sae_config']}")
        extract_sample_metadata(**cfg)


if __name__ == "__main__":
    main()
