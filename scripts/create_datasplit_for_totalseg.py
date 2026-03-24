#!/usr/bin/env python3
"""
Create train/validation/test splits for TotalSegmentator embeddings.

Unique identifier:
    The unique unit of observation is the (modality, image_id) pair.
    Image IDs are only unique within a modality: image_id "0001" in CT and
    image_id "0001" in MRI refer to different images from patients from 
    different datasets.

Splitting strategy:
- Test set: held-out institutions (A, D, E)
- Train/val: remaining institutions, stratified by modality, gender (sex), and age
- Grouping: (modality, image_id) pairs ensure no scan leaks across splits
- Filtering: slices without any segmented anatomy are removed before splitting

Usage:
    conda activate sail
    python create_datasplit_for_totalseg.py --model biomedparse
    python create_datasplit_for_totalseg.py --model dinov3
    python create_datasplit_for_totalseg.py --model biomedparse_random
"""

import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit, StratifiedShuffleSplit
from typing import Optional


# ============================================================================
# Settings
# ============================================================================

DATA_ROOT = Path("data")
SPLIT_INDEX_PATH = DATA_ROOT / "totalseg_split_index.csv"

MODEL_CONFIGS = {
    "biomedparse": "total_seg_biomedparse_encodings",
    "biomedparse_random": "total_seg_biomedparse_random_encodings",
    "dinov3": "total_seg_dinov3_encodings",
}


# ============================================================================
# Helper functions for stratified splitting
# ============================================================================

def _process_age_column(group_df, age_bins, age_labels):
    """Bin age column and fill missing values with 'unknown'."""
    group_df = group_df.copy()
    group_df['age_bin'] = pd.cut(
        group_df['age'],
        bins=age_bins,
        labels=age_labels,
        include_lowest=True
    )
    group_df['age_bin'] = group_df['age_bin'].cat.add_categories(['unknown']).fillna('unknown')
    return group_df


def _identify_known_unknown_groups(group_df, stratify_cols_processed):
    """Return indices of rows with complete vs. missing stratification data."""
    mask_known = pd.Series([True] * len(group_df), index=group_df.index)
    for col in stratify_cols_processed:
        if col == 'age_bin':
            mask_known &= (group_df['age_bin'] != 'unknown')
        else:
            mask_known &= group_df[col].notna()
    known_indices = np.array(group_df[mask_known].index.tolist())
    unknown_indices = np.array(group_df[~mask_known].index.tolist())
    return known_indices, unknown_indices


def _distribute_unknown_groups(unknown_indices, split_proportion, random_state):
    """Randomly assign unknown-label groups proportionally to splits."""
    if len(unknown_indices) == 0:
        return np.array([]), unknown_indices
    rng = np.random.default_rng(random_state)
    shuffled = unknown_indices.copy()
    rng.shuffle(shuffled)
    n_first = int(len(shuffled) * split_proportion)
    return shuffled[:n_first], shuffled[n_first:]


def _stratified_split_with_unknowns(group_ids, stratify_labels, known_indices,
                                     unknown_indices, split_proportion, random_state):
    """
    Stratified split on known-label groups; unknown-label groups split randomly.
    Returns (first_idx, second_idx, n_first_unknown, n_second_unknown).
    """
    group_ids_known = group_ids[known_indices]
    sss = StratifiedShuffleSplit(n_splits=1, train_size=split_proportion,
                                  random_state=random_state)
    first_idx_known, second_idx_known = next(sss.split(group_ids_known, stratify_labels))

    first_idx_orig = known_indices[first_idx_known]
    second_idx_orig = known_indices[second_idx_known]

    first_unknown, second_unknown = _distribute_unknown_groups(
        unknown_indices, split_proportion, random_state
    )

    first_idx = (np.concatenate([first_idx_orig, first_unknown])
                 if len(first_unknown) > 0 else first_idx_orig)
    second_idx = (np.concatenate([second_idx_orig, second_unknown])
                  if len(second_unknown) > 0 else second_idx_orig)

    return first_idx, second_idx, len(first_unknown), len(second_unknown)


# ============================================================================
# Main split function
# ============================================================================

def create_grouped_train_val_test_split(
    df: pd.DataFrame,
    group_col: str = "image_id",
    test_institutions: Optional[list] = None,
    train_size: float = 0.8,
    val_size: float = 0.2,
    stratify_cols: Optional[list] = None,
    age_bins: Optional[list] = None,
    age_labels: Optional[list] = None,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train, val, and test sets.

    Phase 1 - institution-based test set: all scans from `test_institutions`
    go to the test split, preventing institution-level data leakage.

    Phase 2 - stratified train/val split: remaining data is split by
    (image_id, modality) groups with optional stratification on metadata columns.

    Args:
        df: DataFrame with columns including `image_id`, `modality`, `institute`.
        group_col: Column used for grouping (default: "image_id").
        test_institutions: Institution labels held out for testing.
        train_size: Fraction of non-test scans for training (default: 0.8).
        val_size: Fraction of non-test scans for validation (default: 0.2).
        stratify_cols: Metadata columns to stratify on (e.g. ["modality", "gender", "age"]).
        age_bins: Bin edges for age binning when "age" is in stratify_cols.
        age_labels: Labels for age bins.
        random_state: Random seed for reproducibility.

    Returns:
        Tuple of (train_df, val_df, test_df).
    """
    if test_institutions is None:
        test_institutions = ['A', 'D', 'E']
    if age_bins is None:
        age_bins = [0, 50, float('inf')]
    if age_labels is None:
        age_labels = ['<50', '>=50']

    assert np.isclose(train_size + val_size, 1.0), "train_size + val_size must equal 1.0"

    print(f"{'='*60}")
    print(f"PHASE 1: Institution-based test set")
    print(f"{'='*60}")
    print(f"Test institutions: {test_institutions}")
    print(f"Total samples: {len(df)}, unique {group_col}s: {df[group_col].nunique()}")

    # Composite key: (image_id, modality) - CT and MRI from same patient are separate scans
    df = df.copy()
    df['_scan_key'] = df[group_col].astype(str) + '_' + df['modality'].astype(str).str.lower()

    n_scans = df['_scan_key'].nunique()
    print(f"Unique (image_id, modality) scans: {n_scans}")

    test_scan_keys = df.loc[df['institute'].isin(test_institutions), '_scan_key'].tolist()

    # Detect any scan present in both test and non-test institutions
    test_set = set(df.loc[df['institute'].isin(test_institutions), '_scan_key'])
    other_set = set(df.loc[~df['institute'].isin(test_institutions), '_scan_key'])
    overlap = test_set & other_set
    if overlap:
        print(f"WARNING: {len(overlap)} scans appear in both test and non-test institutions "
              f"- assigning to test to prevent leakage.")

    test_df = df.loc[df['_scan_key'].isin(test_scan_keys)].reset_index(drop=True)
    trainval_df = df.loc[~df['_scan_key'].isin(test_scan_keys)].reset_index(drop=True)

    print(f"\nTest  : {len(test_df):6d} samples, {test_df['_scan_key'].nunique():4d} scans")
    print(f"TrainVal: {len(trainval_df):6d} samples, {trainval_df['_scan_key'].nunique():4d} scans")

    assert len(set(test_df['_scan_key']) & set(trainval_df['_scan_key'])) == 0

    print(f"\n{'='*60}")
    print(f"PHASE 2: Stratified train/val split (train={train_size:.0%}, val={val_size:.0%})")
    print(f"{'='*60}")

    group_df = trainval_df.groupby('_scan_key').first().reset_index()
    group_ids = group_df['_scan_key'].values

    if stratify_cols:
        stratify_cols_processed = []
        for col in stratify_cols:
            if col == 'age':
                group_df = _process_age_column(group_df, age_bins, age_labels)
                stratify_cols_processed.append('age_bin')
                print(f"Age binned: {age_labels}")
            else:
                stratify_cols_processed.append(col)

        known_indices, unknown_indices = _identify_known_unknown_groups(
            group_df.reset_index(drop=True), stratify_cols_processed
        )
        print(f"Scans with known stratification values: {len(known_indices)}")
        print(f"Scans with unknown stratification values: {len(unknown_indices)}")

        group_df_reset = group_df.reset_index(drop=True)
        group_df_reset['_strat_key'] = (
            group_df_reset[stratify_cols_processed].astype(str).agg('-'.join, axis=1)
        )
        stratify_labels_known = group_df_reset.loc[known_indices, '_strat_key'].values

        strat_counts = pd.Series(stratify_labels_known).value_counts()
        if strat_counts.min() < 2:
            print(f"WARNING: some stratification classes have <2 samples:\n{strat_counts[strat_counts < 2]}")

        group_ids_reset = group_df_reset['_scan_key'].values
        train_idx, val_idx, n_tr_unk, n_val_unk = _stratified_split_with_unknowns(
            group_ids_reset, stratify_labels_known, known_indices, unknown_indices,
            train_size, random_state
        )
        if n_tr_unk + n_val_unk > 0:
            print(f"Unknown groups: {n_tr_unk} -> train, {n_val_unk} -> val")
        train_scan_keys = group_ids_reset[train_idx].tolist()
        val_scan_keys = group_ids_reset[val_idx].tolist()
    else:
        gss = GroupShuffleSplit(n_splits=1, train_size=train_size, random_state=random_state)
        train_idx, val_idx = next(gss.split(group_ids, np.zeros(len(group_ids)), groups=group_ids))
        train_scan_keys = group_ids[train_idx].tolist()
        val_scan_keys = group_ids[val_idx].tolist()

    train_df = df.loc[df['_scan_key'].isin(train_scan_keys)].reset_index(drop=True)
    val_df = df.loc[df['_scan_key'].isin(val_scan_keys)].reset_index(drop=True)

    # Verify no overlap
    train_set = set(train_df['_scan_key'])
    val_set = set(val_df['_scan_key'])
    test_set2 = set(test_df['_scan_key'])
    assert len(train_set & val_set) == 0, "Train/val scan overlap!"
    assert len(train_set & test_set2) == 0, "Train/test scan overlap!"
    assert len(val_set & test_set2) == 0, "Val/test scan overlap!"

    print(f"\n{'='*60}")
    print(f"FINAL SPLIT")
    print(f"{'='*60}")
    total = df['_scan_key'].nunique()
    for name, split in [("Train", train_df), ("Val  ", val_df), ("Test ", test_df)]:
        scans = split['_scan_key'].nunique()
        print(f"{name}: {len(split):6d} samples, {scans:4d} scans "
              f"({len(split)/len(df)*100:.1f}% samples, {scans/total*100:.1f}% scans)")
    print(f"Test institutions: {test_institutions}")

    return (
        train_df.drop(columns=['_scan_key']),
        val_df.drop(columns=['_scan_key']),
        test_df.drop(columns=['_scan_key']),
    )


# ============================================================================
# Dataset filtering: remove slices outside the anatomy region
# ============================================================================

def filter_non_anatomy_slices(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each (image_id, modality, slice_orientation) view, keep only slices
    between the first and last slice that contains at least one segmented organ.
    This removes empty leading/trailing slices.
    """
    organ_cols = df.columns.to_list()[21:-3]
    print(f"Found {len(organ_cols)} organ columns.")

    df = df.copy()
    df['_view_id'] = (
        df['image_id'].astype(str) + '_' +
        df['modality'].astype(str) + '_' +
        df['slice_orientation'].astype(str)
    )

    rows_to_keep = []
    for view_id, view_df in df.groupby('_view_id'):
        view_df = view_df.sort_values('slice_idx').reset_index(drop=True)
        has_organ = (view_df[organ_cols].sum(axis=1) > 0).values
        organ_indices = np.where(has_organ)[0]
        if len(organ_indices) == 0:
            continue
        rows_to_keep.append(view_df.iloc[organ_indices[0]:organ_indices[-1] + 1])

    df_filtered = pd.concat(rows_to_keep, ignore_index=True).drop(columns=['_view_id'])
    removed = len(df) - len(df_filtered)
    print(f"Filtering: {len(df):,} -> {len(df_filtered):,} rows "
          f"({removed:,} non-anatomy slices removed, {removed/len(df)*100:.1f}%)")
    return df_filtered


# ============================================================================
# Split index: model-agnostic (modality, image_id) -> split mapping
# ============================================================================

def save_split_index(train_df, val_df, test_df, path: Path) -> pd.DataFrame:
    """Save a (modality, image_id) -> split mapping to CSV."""
    records = []
    for split_name, split_df in [("train", train_df), ("val", val_df), ("test", test_df)]:
        pairs = split_df[["modality", "image_id"]].drop_duplicates().copy()
        pairs["split"] = split_name
        records.append(pairs)
    index = pd.concat(records, ignore_index=True)
    index.to_csv(path, index=False)
    print(f"Split index saved to {path} ({len(index)} scans).")
    return index


def apply_split_index(df: pd.DataFrame, split_index: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Assign rows in df to train/val/test using a pre-computed split index.
    Rows whose (modality, image_id) pair is absent from the index are excluded
    with a warning.
    """
    df = df.merge(split_index, on=["modality", "image_id"], how="left")
    missing = df["split"].isna().sum()
    if missing:
        print(f"WARNING: {missing} rows have no split assignment and will be excluded.")
    train_df = df.loc[df["split"] == "train"].drop(columns=["split"]).reset_index(drop=True)
    val_df   = df.loc[df["split"] == "val"].drop(columns=["split"]).reset_index(drop=True)
    test_df  = df.loc[df["split"] == "test"].drop(columns=["split"]).reset_index(drop=True)
    return train_df, val_df, test_df


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Create train/val/test splits for TotalSegmentator embeddings."
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        required=True,
        help="Which embedding model to split.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT,
        help=f"Root data directory (default: {DATA_ROOT})",
    )
    parser.add_argument(
        "--train-size",
        type=float,
        default=0.8,
        help="Fraction of non-test scans for training (default: 0.8).",
    )
    parser.add_argument(
        "--random-state",
        type=int,
        default=42,
        help="Random seed (default: 42).",
    )
    parser.add_argument(
        "--no-save",
        action="store_true",
        help="Dry run - compute splits but do not write files.",
    )
    args = parser.parse_args()

    enc_dir = args.data_root / MODEL_CONFIGS[args.model]
    all_parquet = enc_dir / f"total_seg_{args.model}_encodings_all.parquet"
    train_path = enc_dir / f"total_seg_{args.model}_encodings_train.parquet"
    val_path = enc_dir / f"total_seg_{args.model}_encodings_valid.parquet"
    test_path = enc_dir / f"total_seg_{args.model}_encodings_test.parquet"
    split_index_path = args.data_root / "totalseg_split_index.csv"

    print(f"Loading {all_parquet} ...")
    df = pd.read_parquet(all_parquet)
    print(f"Loaded {len(df):,} rows.")

    df = filter_non_anatomy_slices(df)

    if split_index_path.exists():
        print(f"\nLoading existing split index from {split_index_path} ...")
        split_index = pd.read_csv(split_index_path)
        print(f"Split index: {len(split_index)} scans.")
        train_df, val_df, test_df = apply_split_index(df, split_index)
    else:
        print("\nNo split index found - creating new split ...")
        train_df, val_df, test_df = create_grouped_train_val_test_split(
            df,
            group_col="image_id",
            test_institutions=['A', 'D', 'E'],
            stratify_cols=["modality", "gender", "age"],
            train_size=args.train_size,
            val_size=1.0 - args.train_size,
            random_state=args.random_state,
            age_bins=[0, 40, 60, 80, float('inf')],
            age_labels=['<40', '40-60', '60-80', '>=80'],
        )
        if not args.no_save:
            save_split_index(train_df, val_df, test_df, split_index_path)

    if not args.no_save:
        print(f"\nSaving splits to {enc_dir} ...")
        train_df.to_parquet(train_path)
        val_df.to_parquet(val_path)
        test_df.to_parquet(test_path)
        train_df["image_id"].drop_duplicates().to_csv(
            str(train_path).replace(".parquet", "_image_ids.csv"), index=False
        )
        val_df["image_id"].drop_duplicates().to_csv(
            str(val_path).replace(".parquet", "_image_ids.csv"), index=False
        )
        test_df["image_id"].drop_duplicates().to_csv(
            str(test_path).replace(".parquet", "_image_ids.csv"), index=False
        )
        print("Done.")
    else:
        print("\nDry run - splits not saved.")


if __name__ == "__main__":
    main()
