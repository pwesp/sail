#!/usr/bin/env python3
"""
plot_sae_quality_and_perf_rec.py

Produces Figure 2 of the manuscript: SAE quality and performance recovery across
all grid-search configurations, per foundation model.

Eight-panel layout (2 rows x 4 columns), saved as results/figure_sae_quality_and_perf_rec.pdf:

  Row 0 - SAE quality vs. L0 sparsity:
    A  Reconstruction fidelity   R^2 on the validation set
    B  Downstream classification mean ROC-AUC across tasks (sparse features vs. dense baseline)
    C  Feature activity          alive features at dictionary level 3
    D  Monosemanticity           mean monosemanticity of top-50 features (M(f) = C(f) x S(f))

  Row 1 - Performance recovery vs. L0 sparsity:
    E  @ N=1  feature   mean ROC-AUC using only the single most important feature
    F  @ N=3  features
    G  @ N=10 features
    H  @ N=50 features

Inputs (written by evaluate_monosemanticity.py, evaluate_performance_recovery.py,
train_linear_probes.py, train_matryoshka_sae.py):
  lightning_logs/                                       training metrics
  results/monosemanticity/valid/config_scores_all.csv   config-level monosemanticity
  results/linear_probes/{model}/{config}/valid/         downstream ROC-AUC metrics
  results/performance_recovery/{model}/{config}/        recovery_summary.csv + *_recovery.csv
"""

from pathlib import Path
from typing import cast

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.lines import Line2D
from matplotlib.ticker import StrMethodFormatter
import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LIGHTNING_LOG_DIR        = Path("lightning_logs")
CONFIG_SCORES_PATH       = Path("results/monosemanticity/valid/config_scores_all.csv")
LINEAR_PROBES_DIR  = Path("results/linear_probes")
PERFORMANCE_RECOVERY_DIR = Path("results/performance_recovery")
OUTPUT_DIR               = Path("results")

# ---------------------------------------------------------------------------
# Visual encoding
# ---------------------------------------------------------------------------
BASE_MODELS = ["biomedparse_random", "biomedparse", "dinov3"]

BASE_COLORS: dict[str, str] = {
    "biomedparse_random": "darkgrey",
    "biomedparse":        "darkorange",
    "dinov3":             "lightseagreen",
}

# biomedparse_random is a control -> de-emphasise
ALPHA: dict[str, float] = {
    "biomedparse_random": 0.8,
    "biomedparse":        1.0,
    "dinov3":             1.0,
}

DISPLAY_NAMES: dict[str, str] = {
    "biomedparse_random": "bmp random",
    "biomedparse":        "biomedparse",
    "dinov3":             "dinov3",
}

DICT_SIZE_FAMILIES = ["16_64_256_1024", "32_128_512_2048", "64_256_1024_4096", "128_512_2048_8192"]
LINE_STYLES        = ["-",              "--",               ":",               "-."]
MARKER_POOL        = ["o", "s", "D", "p", "v", "<", "X", "P"]

# Panel 3: targets where dense baseline is near-chance add noise, not signal
DENSE_ROC_AUC_MIN = 0.6

# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_l0_and_r2() -> pd.DataFrame:
    """Read val/l0_level_3, val/r2_level_3, and val/alive_features_level_3
    (last val row) from every lightning_logs/{model}_matryoshka_sae/{config}/metrics.csv."""
    records: list[dict[str, object]] = []

    for model_dir in sorted(LIGHTNING_LOG_DIR.iterdir()):
        if not model_dir.is_dir() or not model_dir.name.endswith("_matryoshka_sae"):
            continue
        base_model = model_dir.name.replace("_matryoshka_sae", "")
        if base_model not in BASE_COLORS:
            continue  # e.g. biomedparse_matryoshka_sae_v1

        for config_dir in sorted(model_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            metrics_path = config_dir / "metrics.csv"
            # File MUST exist - fail hard if missing
            df = pd.read_csv(metrics_path)
            val_rows = df[df["val/r2_level_3"].notna()]
            assert len(val_rows) > 0, f"No val rows in {metrics_path}"
            last = val_rows.iloc[-1]

            records.append({
                "base_model":        base_model,
                "sae_config":        config_dir.name,
                "l0":                float(last["val/l0_level_3"]),
                "r2_recon":          float(last["val/r2_level_3"]),
                "alive_features_l3": float(last["val/alive_features_level_3"]),
            })

    df_out = pd.DataFrame(records)
    assert len(df_out) > 0, "No training results loaded"
    return df_out


def load_config_scores() -> pd.DataFrame:
    """Load interpretability scores including coherence, specificity, and monosemanticity."""
    df = pd.read_csv(CONFIG_SCORES_PATH)
    return cast(pd.DataFrame, df[[
        "base_model", "sae_config",
        "coherence_score", "specificity_score",
        "top10_coherence_mean", "top50_coherence_mean",
        "top10_specificity_mean", "top50_specificity_mean",
        "M_config", "monosemanticity_top50_mean"
    ]])


def load_downstream_roc_auc() -> pd.DataFrame:
    """Read every *_metrics.csv, filter targets by dense baseline, aggregate per config."""
    records: list[dict[str, object]] = []

    for model_dir in sorted(LINEAR_PROBES_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        base_model: str = model_dir.name

        for config_dir in sorted(model_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            valid_dir = config_dir / "valid"
            assert valid_dir.is_dir(), f"Expected valid/ in {config_dir}"

            for mf in sorted(valid_dir.iterdir()):
                if not mf.name.endswith("_metrics.csv"):
                    continue
                df = pd.read_csv(mf)
                dense  = cast(pd.DataFrame, df[df["feature_type"] == "dense"])
                sparse = cast(pd.DataFrame, df[df["feature_type"] == "sparse"])
                assert len(dense)  >= 1, f"No dense row in {mf}"
                assert len(sparse) >= 1, f"No sparse row in {mf}"
                # Some files have duplicate rows (identical metrics, different fit_time); take first.

                records.append({
                    "base_model":         base_model,
                    "sae_config":         config_dir.name,
                    "target":             str(dense["target"].iloc[0]),
                    "dense_val_roc_auc":  float(dense["eval_roc_auc"].iloc[0]),
                    "sparse_val_roc_auc": float(sparse["eval_roc_auc"].iloc[0]),
                })

    df_all = pd.DataFrame(records)
    assert len(df_all) > 0, "No downstream results loaded"

    df_filtered = cast(pd.DataFrame, df_all[df_all["dense_val_roc_auc"] >= DENSE_ROC_AUC_MIN])
    print(f"ROC-AUC panel: {df_filtered['target'].nunique()}/{df_all['target'].nunique()} "
          f"targets kept (dense ROC-AUC >= {DENSE_ROC_AUC_MIN})")

    df_agg = cast(
        pd.DataFrame,
        df_filtered
        .groupby(["base_model", "sae_config"], as_index=False)
        .agg(
            mean_sparse_roc_auc=("sparse_val_roc_auc", "mean"),
            mean_dense_roc_auc =("dense_val_roc_auc",  "mean"),
            n_targets_kept     =("target",             "nunique"),
        )
    )
    return df_agg


def load_performance_recovery() -> pd.DataFrame:
    """Load performance recovery data from recovery_summary.csv files.

    Computes mean n_features needed to reach each recovery threshold (50%, 60%, ..., 100%)
    across all targets for each base_model/sae_config combination.

    Filters to only include targets where dense_baseline_roc_auc > 0.6 to exclude
    near-chance or worse tasks where recovery is meaningless.

    Treats unreached thresholds (NaN) as requiring 100 features - a conservative
    estimate reflecting that these targets need at least 100+ features to recover.
    """
    DENSE_ROC_AUC_MIN = 0.6
    records: list[dict[str, object]] = []
    total_targets_before = 0
    total_targets_after = 0

    for model_dir in sorted(PERFORMANCE_RECOVERY_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        base_model: str = model_dir.name

        for config_dir in sorted(model_dir.iterdir()):
            if not config_dir.is_dir():
                continue
            summary_file = config_dir / "recovery_summary.csv"
            # File MUST exist - fail hard if missing
            df = pd.read_csv(summary_file)
            total_targets_before += len(df)

            # Filter to only include targets where dense model performs above chance
            df_filtered = cast(pd.DataFrame, df[df["dense_baseline_roc_auc"] > DENSE_ROC_AUC_MIN])
            total_targets_after += len(df_filtered)

            if len(df_filtered) == 0:
                # Skip configs with no valid targets
                continue

            # Fill NaN values with 100 (threshold not reached within 100 features)
            # This treats unreached thresholds as requiring >=100 features
            recovery_cols = [
                "n_features_50pct",
                "n_features_60pct",
                "n_features_70pct",
                "n_features_80pct",
                "n_features_90pct",
                "n_features_100pct",
            ]
            df_filled = df_filtered.copy()
            for col in recovery_cols:
                df_filled[col] = df_filled[col].fillna(100.0)

            # Compute mean across all filtered targets for each threshold
            mean_record = {
                "base_model": base_model,
                "sae_config": config_dir.name,
                "mean_n_features_50pct":  df_filled["n_features_50pct"].mean(),
                "mean_n_features_60pct":  df_filled["n_features_60pct"].mean(),
                "mean_n_features_70pct":  df_filled["n_features_70pct"].mean(),
                "mean_n_features_80pct":  df_filled["n_features_80pct"].mean(),
                "mean_n_features_90pct":  df_filled["n_features_90pct"].mean(),
                "mean_n_features_100pct": df_filled["n_features_100pct"].mean(),
            }
            records.append(mean_record)

    df_out = pd.DataFrame(records)
    assert len(df_out) > 0, "No performance recovery data found - something is wrong!"
    print(f"Performance recovery: loaded {len(df_out)} configs")
    print(f"  Filtered targets: {total_targets_after}/{total_targets_before} "
          f"kept (dense ROC-AUC > {DENSE_ROC_AUC_MIN})")
    return df_out


def load_absolute_performance_at_n_features() -> pd.DataFrame:
    """Load absolute ROC-AUC achieved when using top-N features.

    For each base_model/sae_config, computes mean valid_roc_auc across all targets
    at specific feature counts (N = 1, 2, 3, 5, 10, 20, 30, 50, 75, 100).

    Filters to only include targets where dense_baseline_roc_auc > 0.6.
    """
    DENSE_ROC_AUC_MIN = 0.6
    FEATURE_COUNTS = [1, 2, 3, 5, 10, 20, 30, 50, 75, 100]
    records: list[dict[str, object]] = []
    total_targets_before = 0
    total_targets_after = 0

    for model_dir in sorted(PERFORMANCE_RECOVERY_DIR.iterdir()):
        if not model_dir.is_dir():
            continue
        base_model: str = model_dir.name

        for config_dir in sorted(model_dir.iterdir()):
            if not config_dir.is_dir():
                continue

            # Load recovery_summary.csv to get dense baselines
            summary_file = config_dir / "recovery_summary.csv"
            # File MUST exist - fail hard if missing
            df_summary = pd.read_csv(summary_file)
            total_targets_before += len(df_summary)

            # Filter by dense baseline
            df_summary_filtered = cast(pd.DataFrame, df_summary[df_summary["dense_baseline_roc_auc"] > DENSE_ROC_AUC_MIN])
            total_targets_after += len(df_summary_filtered)

            if len(df_summary_filtered) == 0:
                continue

            # Create a mapping of (target_group, target) -> dense_baseline
            dense_baseline_map: dict[tuple[str, str], float] = {}
            for _, row in df_summary_filtered.iterrows():
                key = (str(row['target_group']), str(row['target']))
                dense_baseline_map[key] = float(row['dense_baseline_roc_auc'])

            # Load all recovery CSV files for targets that passed the filter
            all_n_features_data: dict[int, list[float]] = {n: [] for n in FEATURE_COUNTS}
            for recovery_csv in sorted(config_dir.glob("*_recovery.csv")):
                # Parse target group and target from filename
                # Format: {target_group}_{target}_recovery.csv
                filename = recovery_csv.stem.replace("_recovery", "")

                # Find the last underscore to split target_group and target
                # This handles cases like "age_group_<40" correctly
                parts = filename.split("_")
                for i in range(len(parts) - 1, 0, -1):
                    target_group = "_".join(parts[:i])
                    target = "_".join(parts[i:])
                    if (target_group, target) in dense_baseline_map:
                        break
                else:
                    # Target not in filtered summary
                    continue

                # Load recovery data
                df_target = pd.read_csv(recovery_csv)

                # Extract ROC-AUC for each feature count
                for n_feat in FEATURE_COUNTS:
                    df_n = cast(pd.DataFrame, df_target[df_target['n_features'] == n_feat])
                    if len(df_n) > 0:
                        all_n_features_data[n_feat].append(float(df_n['valid_roc_auc'].iloc[0]))

            # Compute mean across targets for each feature count
            mean_record: dict[str, object] = {"base_model": base_model, "sae_config": config_dir.name}
            for n_feat in FEATURE_COUNTS:
                if len(all_n_features_data[n_feat]) > 0:
                    mean_roc = cast(float, pd.Series(all_n_features_data[n_feat]).mean())
                    mean_record[f'mean_roc_auc_n{n_feat}'] = mean_roc
                else:
                    mean_record[f'mean_roc_auc_n{n_feat}'] = float('nan')

            records.append(mean_record)

    df_out = pd.DataFrame(records)
    assert len(df_out) > 0, "No absolute performance data found - something is wrong!"
    print(f"Absolute performance at N features: loaded {len(df_out)} configs")
    print(f"  Filtered targets: {total_targets_after}/{total_targets_before} "
          f"kept (dense ROC-AUC > {DENSE_ROC_AUC_MIN})")
    return df_out


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------

def _plot_l0_panel(ax: Axes, df: pd.DataFrame, col: str,
                   top_k_to_marker: dict[str, str],
                   dict_size_to_ls: dict[str, str],
                   linewidth: float,
                   label_font_size: int,
                   tick_font_size: int) -> pd.DataFrame:
    """Draw connecting lines + shaped markers for every config onto one L0 axes.

    Filters to rows where *col* is not NaN so that panels with incomplete data
    still show everything they can.  Returns the filtered DataFrame so the
    caller can draw panel-specific reference lines on the same subset.
    """
    df_panel = cast(pd.DataFrame, df[df[col].notna()])

    for bm in BASE_MODELS:
        color    = BASE_COLORS[bm]
        alpha    = ALPHA[bm]
        df_model = cast(pd.DataFrame, df_panel[df_panel["base_model"] == bm])

        for ds in DICT_SIZE_FAMILIES:
            grp = cast(pd.DataFrame, df_model[df_model["dict_sizes"] == ds].sort_values(by="l0"))  # type: ignore[call-overload]
            if len(grp) == 0:
                continue

            ax.plot(
                grp["l0"].to_numpy(), grp[col].to_numpy(),
                color=color, linestyle=dict_size_to_ls[ds],
                linewidth=linewidth, alpha=alpha * 0.55, zorder=1,
            )

            for _, row in grp.iterrows():
                ax.scatter(
                    float(row["l0"]), float(row[col]),
                    color=color,
                    marker=top_k_to_marker[str(row["top_k_values"])],
                    s=80, edgecolors="white", linewidths=0.8,
                    alpha=alpha, zorder=3,
                )

    ax.set_xscale("log")
    ax.tick_params(axis='both', which='major', labelsize=tick_font_size, length=5, width=1.5)
    ax.set_xlabel("L0 (sparsity)", fontsize=label_font_size)
    ax.grid(True, alpha=0.15)
    return df_panel



def main() -> None:
    # =========================================================================
    # Style settings
    # =========================================================================

    linewidth = 2.0
    label_font_size = 16
    tick_font_size = 16
    legend_font_size = 16
    title_font_size = 16

    # =========================================================================
    # Load and merge data
    # =========================================================================
    df_master = load_l0_and_r2()
    assert len(df_master) > 0

    configs = [str(c).split("_K_", 1) for c in df_master["sae_config"]]
    df_master["dict_sizes"]   = [c[0][2:] for c in configs]   # strip 'D_'
    df_master["top_k_values"] = [c[1]     for c in configs]

    df = (
        df_master
        .merge(load_config_scores(),                      on=["base_model", "sae_config"], how="left")
        .merge(load_downstream_roc_auc(),                 on=["base_model", "sae_config"], how="left")
        .merge(load_performance_recovery(),               on=["base_model", "sae_config"], how="left")
        .merge(load_absolute_performance_at_n_features(), on=["base_model", "sae_config"], how="left")
    )

    # ----- Warn about missing data (per-panel; no global drop) -----
    for col, label in [("coherence_score",           "coherence"),
                       ("specificity_score",         "specificity"),
                       ("monosemanticity_top50_mean", "top-50 monosemanticity"),
                       ("mean_sparse_roc_auc",       "downstream ROC-AUC"),
                       ("mean_n_features_50pct",     "performance recovery")]:
        missing = cast(pd.DataFrame, df[df[col].isna()])
        if len(missing) > 0:
            missing_subset = cast(pd.DataFrame, missing[["base_model", "sae_config"]])
            print(f"WARNING - {len(missing)} configs missing {label} (skipped in that panel only):\n"
                  + missing_subset.to_string(index=False))

    assert len(df) >= 20, f"Only {len(df)} configs - too few to plot"

    # ----- Diagnostics -----
    print(f"\n{len(df)} configs total:")
    for bm in BASE_MODELS:
        sub = cast(pd.DataFrame, df[df["base_model"] == bm])
        if len(sub) == 0:
            continue
        dense_sub = cast(pd.Series, sub["mean_dense_roc_auc"]).dropna()
        dense_str = f"dense baseline = {dense_sub.mean():.3f}" if len(dense_sub) > 0 else "dense baseline = N/A"
        print(f"  {bm:25s} {len(sub):2d} configs  "
              f"L0 [{sub['l0'].min():6.1f} - {sub['l0'].max():6.1f}]  "
              f"{dense_str}")

    # Save merged table for inspection
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUTPUT_DIR / "sae_quality_and_perf_rec_data.csv", index=False)

    # ----- Prepare encoding maps -----
    dict_size_to_ls = dict(zip(DICT_SIZE_FAMILIES, LINE_STYLES))

    # Sort top_k by last element (~= L0 at level 3) so legend follows x-axis order
    all_top_k = sorted(
        df["top_k_values"].unique().tolist(),
        key=lambda tk: int(str(tk).split("_")[-1]),
    )
    assert len(all_top_k) <= len(MARKER_POOL), (
        f"{len(all_top_k)} distinct top_k patterns but only {len(MARKER_POOL)} markers"
    )
    top_k_to_marker = dict(zip(all_top_k, MARKER_POOL))

    # =========================================================================
    # Figure 1: Eight-panel L0 overview (2 rows x 4 columns)
    # =========================================================================
    fig, axes = plt.subplots(2, 4, figsize=(24, 12))

    # ----- Row 0, Column 0: Reconstruction fidelity -----
    _plot_l0_panel(
        axes[0, 0], df=df, col="r2_recon", top_k_to_marker=top_k_to_marker, dict_size_to_ls=dict_size_to_ls, 
        linewidth=linewidth, label_font_size=label_font_size, tick_font_size=tick_font_size)
    axes[0, 0].set_ylabel("R^2", fontsize=label_font_size)
    axes[0, 0].set_title("Reconstruction Fidelity", fontsize=title_font_size)

    # ----- Row 0, Column 1: Downstream classification (sparse curves + dense baselines) -----
    df_roc = _plot_l0_panel(axes[0, 1], df, "mean_sparse_roc_auc", top_k_to_marker, dict_size_to_ls, linewidth, label_font_size, tick_font_size)
    axes[0, 1].set_ylabel("ROC-AUC", fontsize=label_font_size)
    axes[0, 1].set_title("Downstream Classification", fontsize=title_font_size)

    for bm in BASE_MODELS:
        sub = cast(pd.DataFrame, df_roc[df_roc["base_model"] == bm])
        if len(sub) == 0:
            continue
        axes[0, 1].axhline(
            sub["mean_dense_roc_auc"].mean(),
            color=BASE_COLORS[bm], linestyle="--",
            linewidth=2.0, alpha=ALPHA[bm] * 0.8,
        )
    dense_handle = Line2D([0], [0], color="black", linestyle="--", linewidth=1.4)
    axes[0, 1].legend([dense_handle], ["dense baseline"], loc="upper left", fontsize=legend_font_size)

    # ----- Row 0, Column 2: Feature activity -----
    _plot_l0_panel(axes[0, 2], df, "alive_features_l3", top_k_to_marker, dict_size_to_ls, linewidth, label_font_size, tick_font_size)
    axes[0, 2].set_ylabel("# alive features", fontsize=label_font_size)
    axes[0, 2].set_title("Feature Activity", fontsize=title_font_size)

    # ----- Row 0, Column 3: Top-50 monosemanticity score -----
    _plot_l0_panel(axes[0, 3], df, "monosemanticity_top50_mean", top_k_to_marker, dict_size_to_ls, linewidth, label_font_size, tick_font_size)
    axes[0, 3].set_ylabel("monosemanticity score", fontsize=label_font_size)
    axes[0, 3].set_title("Monosemanticity", fontsize=title_font_size)

    # ----- Row 1, Column 0: Performance at 1 feature -----
    _plot_l0_panel(axes[1, 0], df, "mean_roc_auc_n1", top_k_to_marker, dict_size_to_ls, linewidth, label_font_size, tick_font_size)
    axes[1, 0].set_ylabel("ROC-AUC", fontsize=label_font_size)
    axes[1, 0].set_title("Performance Recovery @ 1 Feature", fontsize=title_font_size)
    axes[1, 0].yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
    axes[1, 0].set_ylim(0.54, 0.86)

    # ----- Row 1, Column 1: Performance at 3 features -----
    _plot_l0_panel(axes[1, 1], df, "mean_roc_auc_n3", top_k_to_marker, dict_size_to_ls, linewidth, label_font_size, tick_font_size)
    axes[1, 1].set_ylabel("ROC-AUC", fontsize=label_font_size)
    axes[1, 1].set_title("Performance Recovery @ 3 Features", fontsize=title_font_size)
    axes[1, 1].yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
    axes[1, 1].set_ylim(0.54, 0.86)

    # ----- Row 1, Column 2: Performance at 10 features -----
    _plot_l0_panel(axes[1, 2], df, "mean_roc_auc_n10", top_k_to_marker, dict_size_to_ls, linewidth, label_font_size, tick_font_size)
    axes[1, 2].set_ylabel("ROC-AUC", fontsize=label_font_size)
    axes[1, 2].set_title("Performance Recovery @ 10 Features", fontsize=title_font_size)
    axes[1, 2].yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
    axes[1, 2].set_ylim(0.54, 0.86)

    # ----- Row 1, Column 3: Performance at 50 features -----
    _plot_l0_panel(axes[1, 3], df, "mean_roc_auc_n50", top_k_to_marker, dict_size_to_ls, linewidth, label_font_size, tick_font_size)
    axes[1, 3].set_ylabel("ROC-AUC", fontsize=label_font_size)
    axes[1, 3].set_title("Performance Recovery @ 50 Features", fontsize=title_font_size)
    axes[1, 3].yaxis.set_major_formatter(StrMethodFormatter('{x:.2f}'))
    axes[1, 3].set_ylim(0.54, 0.86)

    # ----- Shared legend for Figure 1 (3 decoupled sections) -----
    handles: list[Line2D] = []
    labels:  list[str]    = []

    # Section 1: Base model -> colour (solid lines)
    for bm in BASE_MODELS[::-1]:
        handles.append(Line2D(
            [0], [0],
            color=BASE_COLORS[bm], linestyle="-",
            linewidth=2.5, alpha=ALPHA[bm],
        ))
        labels.append(DISPLAY_NAMES[bm])

    # Blank separator
    handles.append(Line2D([0], [0], color="none"))
    labels.append("")

    # Section 2: Dictionary-size family -> linestyle (neutral colour)
    for ds, ls in zip(DICT_SIZE_FAMILIES, LINE_STYLES):
        handles.append(Line2D(
            [0], [0],
            color="black", linestyle=ls,
            linewidth=1.5,
        ))
        labels.append(f"d={ds.replace("_", "-")}")

    # Blank separator
    handles.append(Line2D([0], [0], color="none"))
    labels.append("")

    # Section 3: Top_k markers
    for tk in all_top_k:
        handles.append(Line2D(
            [0], [0], marker=top_k_to_marker[tk], color="w",
            markerfacecolor="lightgray", markersize=9,
            markeredgecolor="black", markeredgewidth=0.8,
            linestyle="",
        ))
        labels.append(f"k={tk.replace("_", "-")}")

    plt.subplots_adjust(bottom=0.17, hspace=0.3, wspace=0.3)  # Allocate 20% of figure height for bottom space, increase space between rows (default is 0.2)
    fig.legend(handles, labels, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, 0.0), fontsize=legend_font_size)

    fig_path = OUTPUT_DIR / "figure_sae_quality_and_perf_rec.pdf"
    fig.savefig(fig_path, dpi=300, facecolor="white", bbox_inches="tight")
    print(f"\nFigure 2 saved to {fig_path}")
    plt.close(fig)

if __name__ == "__main__":
    main()