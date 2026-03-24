#!/usr/bin/env python3
"""
Summarize dataset characteristics and write to results/dataset_summary.yaml.

A "sample" is one 2D image slice (one row in the parquet),
identified by (modality, image_id, slice_orientation, slice_idx).
CT images contribute slices in all 3 orientations (x, y, z);
MRI images contribute slices in a single orientation.
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow.parquet as pq
import yaml


PARQUET_PATH = Path("data/total_seg_biomedparse_encodings/total_seg_biomedparse_encodings_all.parquet")
OUTPUT_PATH = Path("results/dataset_summary.yaml")

# Columns excluded from metadata statistics: identifiers, split marker, file paths, feature blobs
_EXCLUDE_COLS = frozenset({
    "image_id", "slice_idx", "image_path", "split",
    "embedding", "embedding_shape", "embedding_dtype",
})

# Canonical organ column list (in order) for reference
_ORGAN_COLS_START = "adrenal_gland_left"


def load_df(parquet_path: Path) -> pd.DataFrame:
    """Load all non-feature-blob columns from the parquet file."""
    table = pq.read_table(parquet_path)

    # Normalize column name used in older files
    if "orientation" in table.column_names and "slice_orientation" not in table.column_names:
        table = table.rename_columns(
            [c if c != "orientation" else "slice_orientation" for c in table.column_names]
        )

    # Drop binary embedding blobs before converting to pandas (saves memory)
    blob_cols = [c for c in table.column_names if c in {"embedding", "embedding_shape", "embedding_dtype"}]
    if blob_cols:
        table = table.drop_columns(blob_cols)

    return table.to_pandas()


def metadata_keys(df: pd.DataFrame) -> list[str]:
    """Return sorted list of all metadata keys (excludes identifiers and split column)."""
    return sorted(c for c in df.columns if c not in _EXCLUDE_COLS)


def split_df(df: pd.DataFrame, split_name: str) -> pd.DataFrame:
    """Return rows for a given split name ('train', 'val', 'test')."""
    return cast(pd.DataFrame, df[df["split"] == split_name].copy())


def summarize(df: pd.DataFrame, keys: list[str]) -> dict[str, Any]:
    """Compute all summary statistics for a (sub-)dataframe."""
    n_samples = int(len(df))

    # image_ids: a modality-image_id pair is the fundamental image unit
    n_image_ids_per_modality: dict[str, int] = {}
    for modality in sorted(df["modality"].unique()):
        n_image_ids_per_modality[str(modality)] = int(
            df.loc[df["modality"] == modality, "image_id"].nunique()
        )
    n_image_ids_total = int(sum(n_image_ids_per_modality.values()))

    # Unique values per metadata key
    unique_per_key: dict[str, int] = {}
    for key in keys:
        unique_per_key[key] = int(df[key].nunique())

    total_unique = int(sum(unique_per_key.values()))

    return {
        "n_samples": n_samples,
        "n_image_ids": {
            "total": n_image_ids_total,
            "per_modality": n_image_ids_per_modality,
        },
        "n_metadata_keys": len(keys),
        "unique_values_per_metadata_key": unique_per_key,
        "total_unique_metadata_values": total_unique,
    }


def main() -> None:
    if not PARQUET_PATH.exists():
        print(f"ERROR: parquet not found: {PARQUET_PATH}", file=sys.stderr)
        sys.exit(1)

    print(f"Loading {PARQUET_PATH} ...")
    df = load_df(PARQUET_PATH)
    print(f"  {len(df):,} rows, {df['image_id'].nunique():,} unique image_ids")

    keys = metadata_keys(df)
    print(f"  {len(keys)} metadata keys")

    print("Computing overall statistics ...")
    result: dict[str, Any] = {
        "source_parquet": str(PARQUET_PATH),
        "overall": summarize(df, keys),
        "splits": {},
    }

    for split_name, yaml_key in [("train", "train"), ("val", "valid"), ("test", "test")]:
        sub = split_df(df, split_name)
        if sub.empty:
            print(f"  WARNING: split '{split_name}' is empty - skipping", file=sys.stderr)
            continue
        print(f"Computing statistics for split '{split_name}' ({len(sub):,} rows) ...")
        result["splits"][yaml_key] = summarize(sub, keys)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_PATH, "w") as f:
        yaml.dump(result, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    print(f"\nWritten to {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
