#!/usr/bin/env python3
"""
Rank all SAE configurations by combined monosemanticity and performance recovery.

Produces the configuration ranking shown in Table 1 of the manuscript.

Two dimensions are combined:
  1. Monosemanticity (M_config): mean M(f) of the top-10 most monosemantic
     features per configuration, as defined in the paper.
     Source: results/monosemanticity/valid/config_scores_all.csv
     (written by evaluate_monosemanticity.py)

  2. Performance recovery: mean number of sparse features required to reach
     each recovery threshold (50 %, 60 %, ..., 100 % of dense ROC-AUC),
     averaged across classification targets with a dense baseline > 0.6.
     Lower = fewer features needed = better.
     Source: results/performance_recovery/{model}/{config}/recovery_summary.csv
     (written by evaluate_performance_recovery.py)

Both dimensions are converted to within-model ranks, then averaged into a
combined rank. Configs are sorted by combined rank within each foundation model.

Output:
  results/config_ranking.csv   — one row per (base_model, sae_config)
"""

from pathlib import Path
from typing import cast

import pandas as pd


# ==================================================
# Configuration
# ==================================================

MONOSEMANTICITY_ROOT    = Path("results/monosemanticity")
PERFORMANCE_RECOVERY_ROOT = Path("results/performance_recovery")
OUTPUT_PATH             = Path("results/config_ranking.csv")

SPLIT = "valid"
DENSE_BASELINE_MIN = 0.6   # exclude near-chance targets from the performance dimension

RECOVERY_THRESHOLD_COLS = [
    "n_features_50pct",
    "n_features_60pct",
    "n_features_70pct",
    "n_features_80pct",
    "n_features_90pct",
    "n_features_100pct",
]

# Features not reached within the run are NaN; treat as the worst observed + 1
NAN_FILL_STRATEGY = "max_plus_one"


# ==================================================
# Data loading
# ==================================================

def load_monosemanticity_scores() -> pd.DataFrame:
    """
    Load config-level monosemanticity scores produced by evaluate_monosemanticity.py.

    Returns:
        DataFrame with columns: base_model, sae_config, dict_sizes, top_k_values, M_config
    """
    path = MONOSEMANTICITY_ROOT / SPLIT / "config_scores_all.csv"
    if not path.exists():
        raise FileNotFoundError(
            f"Config scores not found: {path}\n"
            f"Run evaluate_monosemanticity.py first."
        )

    df = pd.read_csv(path)

    required = {"base_model", "sae_config", "M_config"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"config_scores_all.csv is missing columns: {missing}")

    # Parse dict_sizes and top_k_values from sae_config string "D_{sizes}_K_{k}"
    parsed = df["sae_config"].str.extract(r"^D_(.+)_K_(.+)$")
    df["dict_sizes"]   = parsed[0]
    df["top_k_values"] = parsed[1]

    print(f"Monosemanticity: loaded {len(df)} configs from {path}")
    print(f"  M_config range: {df['M_config'].min():.4f} – {df['M_config'].max():.4f}")
    return cast(pd.DataFrame, df[["base_model", "sae_config", "dict_sizes", "top_k_values", "M_config"]])


def load_performance_recovery() -> pd.DataFrame:
    """
    Aggregate performance recovery across all configs.

    For each (base_model, sae_config):
      - Load recovery_summary.csv
      - Keep only targets with dense_baseline_roc_auc > DENSE_BASELINE_MIN
      - Fill unmet thresholds (NaN) with the column-wise max + 1
      - Average n_features_Xpct across thresholds and targets → mean_n_features

    Returns:
        DataFrame with columns: base_model, sae_config, mean_n_features, n_targets
    """
    records = []
    n_configs_loaded = 0

    for model_dir in sorted(PERFORMANCE_RECOVERY_ROOT.iterdir()):
        if not model_dir.is_dir():
            continue
        base_model = model_dir.name

        for config_dir in sorted(model_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            summary_path = config_dir / "recovery_summary.csv"
            if not summary_path.exists():
                print(f"  WARNING: missing {summary_path}, skipping")
                continue

            df = pd.read_csv(summary_path)

            # Filter to non-trivial targets
            df = cast(pd.DataFrame, df[df["dense_baseline_roc_auc"] > DENSE_BASELINE_MIN])
            if len(df) == 0:
                continue

            # Fill unreached thresholds
            available_cols = [c for c in RECOVERY_THRESHOLD_COLS if c in df.columns]
            df = df.copy()
            for col in available_cols:
                col_max: float = cast(pd.Series, df[col]).max(skipna=True)
                df[col] = df[col].fillna(col_max + 1 if pd.notna(col_max) else 100.0)

            mean_n = float(df[available_cols].to_numpy().mean())
            records.append({
                "base_model":    base_model,
                "sae_config":    config_dir.name,
                "mean_n_features": mean_n,
                "n_targets":     len(df),
            })
            n_configs_loaded += 1

    if not records:
        raise FileNotFoundError(
            f"No recovery_summary.csv files found under {PERFORMANCE_RECOVERY_ROOT}.\n"
            f"Run evaluate_performance_recovery.py first."
        )

    df_out = pd.DataFrame(records)
    print(f"Performance recovery: loaded {n_configs_loaded} configs")
    print(f"  mean_n_features range: {df_out['mean_n_features'].min():.2f} – "
          f"{df_out['mean_n_features'].max():.2f}")
    return df_out


# ==================================================
# Ranking
# ==================================================

def compute_rankings(mono_df: pd.DataFrame, perf_df: pd.DataFrame) -> pd.DataFrame:
    """
    Combine monosemanticity and performance recovery into a single config ranking.

    Strategy (per foundation model):
      1. Rank by M_config descending (higher monosemanticity = better = rank 1)
      2. Rank by mean_n_features ascending (fewer features needed = better = rank 1)
      3. Combined rank = average of the two ranks
      4. Final rank = rank configs by combined rank (ties broken by min)

    Args:
        mono_df: Output of load_monosemanticity_scores()
        perf_df: Output of load_performance_recovery()

    Returns:
        DataFrame sorted by (base_model, final_rank), columns:
        base_model, sae_config, dict_sizes, top_k_values,
        M_config, mean_n_features, n_targets,
        rank_mono, rank_perf, rank_combined, final_rank
    """
    df = mono_df.merge(perf_df, on=["base_model", "sae_config"], how="inner")

    n_mono_only = len(mono_df) - len(df)
    n_perf_only = len(perf_df) - len(df)
    if n_mono_only or n_perf_only:
        print(f"  WARNING: {n_mono_only} configs in monosemanticity but not performance recovery")
        print(f"  WARNING: {n_perf_only} configs in performance recovery but not monosemanticity")

    # Per-model ranks
    df["rank_mono"] = df.groupby("base_model")["M_config"].rank(
        ascending=False, method="min"
    ).astype(int)
    df["rank_perf"] = df.groupby("base_model")["mean_n_features"].rank(
        ascending=True, method="min"
    ).astype(int)

    df["rank_combined"] = (df["rank_mono"] + df["rank_perf"]) / 2
    df["final_rank"] = df.groupby("base_model")["rank_combined"].rank(
        method="min"
    ).astype(int)

    df = df.sort_values(["base_model", "final_rank"]).reset_index(drop=True)
    print(f"\nRanked {len(df)} configs across {df['base_model'].nunique()} foundation models")
    return df


# ==================================================
# Output
# ==================================================

def save_results(df: pd.DataFrame) -> None:
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved config ranking to {OUTPUT_PATH}")


def print_summary(df: pd.DataFrame, top_k: int = 5) -> None:
    print("\n" + "=" * 80)
    print(f"TOP-{top_k} SAE CONFIGURATIONS PER FOUNDATION MODEL")
    print("=" * 80)

    col_w = 46
    for bm in sorted(df["base_model"].unique()):
        sub = cast(pd.DataFrame, df[df["base_model"] == bm])
        n_total = len(sub)
        print(f"\n{bm.upper()}  ({n_total} configs total)")
        print("-" * 80)
        print(f"  {'#Comb':<7} {'#Mono':<7} {'#Perf':<7}  {'Config':<{col_w}}  {'M_config':<10}  mean_n_feat")
        print("-" * 80)

        for _, row in sub.head(top_k).iterrows():
            config_label = f"D_{row['dict_sizes']}_K_{row['top_k_values']}"
            print(
                f"  {int(row['final_rank']):<7} "
                f"{int(row['rank_mono']):<7} "
                f"{int(row['rank_perf']):<7}  "
                f"{config_label:<{col_w}}  "
                f"{row['M_config']:.4f}      "
                f"{row['mean_n_features']:.2f}"
            )
    print("=" * 80)


# ==================================================
# Main
# ==================================================

def main() -> None:
    print("=" * 80)
    print("SAE CONFIGURATION RANKING")
    print("=" * 80)
    print(f"Monosemanticity source : {MONOSEMANTICITY_ROOT / SPLIT / 'config_scores_all.csv'}")
    print(f"Performance source     : {PERFORMANCE_RECOVERY_ROOT}/{{model}}/{{config}}/recovery_summary.csv")
    print(f"Dense baseline filter  : > {DENSE_BASELINE_MIN}")
    print(f"Output                 : {OUTPUT_PATH}")
    print("=" * 80)

    mono_df = load_monosemanticity_scores()
    perf_df = load_performance_recovery()
    df      = compute_rankings(mono_df, perf_df)

    save_results(df)
    print_summary(df)

    print(f"\n[ok] Config ranking complete — {OUTPUT_PATH}")
    print("=" * 80)


if __name__ == "__main__":
    main()
