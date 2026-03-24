#!/usr/bin/env python3
"""
Zero-Shot Language-Driven Image Retrieval

Demonstrates language-to-image retrieval using sparse feature concepts as an
intermediate representation. An LLM maps a clinical text query to matching
feature descriptions, assembles a sparse fingerprint from their mean activations,
and performs cosine similarity search — with no reference image and no task-specific
training.

Workflow:
  1. Edit queries in src/config_language_based_image_retrieval.py
  2. Set RUN_MATCHING = True once to run VLM feature matching and save
     query_feature_matching.json. Only needed when queries or feature
     descriptions change; keep False for subsequent retrieval runs.
  3. Run retrieval on the test split using the saved matching.

Requires autointerp feature descriptions (evaluate_autointerp.py, top_n mode).

Outputs (results/language_retrieval/{BASE_MODEL}/{SAE_CONFIG}/):
  query_feature_matching.json       matched features per query (split-independent)
  {RETRIEVAL_SPLIT}/
    queries/{name}/
      retrieval_results.csv         retrieved images with similarity scores
      fingerprint.pdf               sparse fingerprint bar chart
      visualization.pdf             detailed retrieval results
      visualization_manuscript.pdf  compact manuscript figure
    all_queries_results.csv         all queries combined
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import json
import textwrap
from pathlib import Path
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.dataloading import (
    load_sparse_features_dataset,
    TotalSegmentatorLatentFeaturesDataset,
    get_image_path,
)
from src.fingerprint_retrieval import (
    create_query_fingerprint,
    retrieve_images_from_query,
)
from src.fingerprint_visualization import (
    visualize_query_retrieval_manuscript,
)
from src.autointerp import load_vlm_model, load_eligible_features, match_query_to_features
from src.config_language_based_image_retrieval import QUERIES


# -- Settings ------------------------------------------------------------------

SAE_CONFIGS = {
    "biomedparse":        "D_128_512_2048_8192_K_20_40_80_160",
    "dinov3":             "D_128_512_2048_8192_K_5_10_20_40",
    "biomedparse_random": "D_64_256_1024_4096_K_30_60_120_240",
}
BASE_MODEL = "dinov3"       # "biomedparse" | "dinov3" | "biomedparse_random"
                            # DINOv3 default: richer feature vocabulary reliably captures
                            # anatomy and modality; BiomedParse lacks modality-pure features
                            # for this task (see manuscript Sec. 3.3)
SAE_CONFIG = SAE_CONFIGS[BASE_MODEL]

N_RETRIEVALS    = 10      # images to retrieve per query
RETRIEVAL_SPLIT = "test"  # "val" or "test"

# -- Matching settings (only used when RUN_MATCHING = True) --------------------

RUN_MATCHING  = False                     # set True to re-run VLM feature matching
VLM_MODEL     = "google/medgemma-27b-it"  # options: gemma-3-{1b,4b,7b}-it, medgemma-{4b,27b}-it
N_FEATURES    = 3                         # features to select per query
TEMPERATURE   = 0.1                       # low temperature for deterministic matching
MAX_CONCEPT_RANK = 2                      # include features with true_concept_rank <= this

# -- Paths ---------------------------------------------------------------------

SPARSE_FEATURES_ROOT = Path("results/total_seg_sparse_features")
FEATURE_DESCRIPTIONS_PATH = Path(
    f"results/auto_interpretation/{BASE_MODEL}/{SAE_CONFIG}/top_n/feature_descriptions.csv"
)
_BASE_DIR     = Path(f"results/language_retrieval/{BASE_MODEL}/{SAE_CONFIG}")
MATCHING_JSON = _BASE_DIR / "query_feature_matching.json"  # split-independent
OUTPUT_DIR    = _BASE_DIR / RETRIEVAL_SPLIT                # retrieval results go here

INSET_COLOR = "darkorange" if BASE_MODEL == "biomedparse" else "lightseagreen"


def run_matching() -> None:
    """Run VLM feature matching for all queries and save query_feature_matching.json."""
    print("\n" + "=" * 70)
    print("QUERY -> FEATURE MATCHING")
    print("=" * 70)

    _BASE_DIR.mkdir(parents=True, exist_ok=True)
    eligible = load_eligible_features(FEATURE_DESCRIPTIONS_PATH, MAX_CONCEPT_RANK)
    vlm_pipe = load_vlm_model(model_name=VLM_MODEL)

    results = []
    for query in QUERIES:
        name = query["name"]
        text = query["text"]
        print(f'\nMatching: "{text}"')
        entry = match_query_to_features(
            vlm_pipe, name, text, eligible, N_FEATURES, TEMPERATURE, VLM_MODEL
        )
        if entry.get("reasoning"):
            print(f"  Reasoning: {entry['reasoning']}")
        print(f"  -> features: {entry['matched_feature_indices']}")
        for idx, desc in zip(entry["matched_feature_indices"], entry["matched_descriptions"]):
            print(f"       {idx}: {desc}")
        results.append(entry)

    with open(MATCHING_JSON, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n[ok] Saved {MATCHING_JSON}")


# -- Retrieval helpers (local) --------------------------------------------------

def load_feature_descriptions(csv_path: Path) -> pd.DataFrame:
    """Load feature descriptions CSV (needed for mean_activation lookup)."""
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} feature descriptions from {csv_path.name}")
    return df


def _enrich_with_metadata(
    df: pd.DataFrame,
    dataset: TotalSegmentatorLatentFeaturesDataset,
) -> pd.DataFrame:
    """Annotate retrieval results with per-image metadata columns.

    Adds: modality, slice_orientation, gender, age, organs (semicolon-separated).
    Uses array_idx directly to avoid a second search through the dataset.
    """
    rows = []
    for _, row in df.iterrows():
        idx = int(row["array_idx"])
        organs = [
            organ for organ in dataset.included_organs
            if dataset.metadata.get(organ) is not None and dataset.metadata[organ][idx]
        ]
        rows.append({
            "modality":          dataset.modalities[idx],
            "slice_orientation": dataset.slice_orientations[idx],
            "gender":            str(dataset.metadata["gender"][idx])
                                 if "gender" in dataset.metadata else "",
            "age":               float(dataset.metadata["age"][idx])
                                 if "age" in dataset.metadata else float("nan"),
            "organs":            "; ".join(organs),
        })
    return pd.concat([df, pd.DataFrame(rows)], axis=1)


# -- Visualization functions ----------------------------------------------------

def save_fingerprint_pdf(
    feature_indices: np.ndarray,
    query_vector: np.ndarray,
    output_path: Path,
    color: str,
) -> None:
    """Save a bare fingerprint bar chart (no labels, title, or legend) as PDF."""
    order = np.argsort(feature_indices)
    vals = query_vector[feature_indices[order]].astype(float)
    if vals.max() > 0:
        vals = vals / vals.max()

    # Sort feature indices and activations by index for stable plotting order
    feature_ids = feature_indices[order]

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.bar(range(len(vals)), vals, color=color, edgecolor="none")
    ax.set_xlim(-0.5, len(vals) - 0.5)
    ax.set_ylim(0, 1)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(feature_ids, rotation=45, ha="right", fontsize=18)
    ax.tick_params(axis="x", length=0)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.set_yticks([])
    ax.spines["bottom"].set_visible(True)
    fig.savefig(output_path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  Saved fingerprint to {output_path.name}")

def visualize_query_results(
    query_name: str,
    query_text: str,
    feature_indices: List[int],
    matched_descriptions: List[str],
    feature_descriptions: pd.DataFrame,
    retrieved_df: pd.DataFrame,
    metadata_dataset: TotalSegmentatorLatentFeaturesDataset,
    output_path: Path,
) -> None:
    """
    Visualize query and retrieved images (detailed, up to 8 images).

    Left panel shows selected concepts; right panels show retrieved images.
    """
    n_show = min(8, len(retrieved_df))
    n_cols = 4

    fig = plt.figure(figsize=(16, 9))
    fig.suptitle(f'Query: "{query_text}"', fontsize=14, fontweight="bold", y=0.98)

    from matplotlib.gridspec import GridSpec
    gs = GridSpec(
        2, n_cols + 1, figure=fig,
        width_ratios=[0.9, 1, 1, 1, 1],
        hspace=0.3, wspace=0.3, left=0.03, right=0.98, top=0.93, bottom=0.05,
    )

    ax_text = fig.add_subplot(gs[:, 0])
    ax_text.axis("off")

    concepts_lines = ["Selected Concepts:", ""]
    for i, (feat_idx, desc) in enumerate(zip(feature_indices, matched_descriptions), 1):
        feat_row = feature_descriptions[feature_descriptions["feature_idx"] == feat_idx]
        mean_act = float(feat_row.iloc[0]["mean_activation"]) if not feat_row.empty else float("nan")
        concepts_lines.append(f"{i}. Feature {feat_idx}")
        wrapped_desc = textwrap.fill(desc, width=30, initial_indent="   ", subsequent_indent="   ")
        concepts_lines.append(wrapped_desc)
        concepts_lines.append(f"   (mean_act: {mean_act:.4f})")
        concepts_lines.append("")

    ax_text.text(
        0.05, 0.95, "\n".join(concepts_lines),
        transform=ax_text.transAxes, fontsize=9,
        verticalalignment="top", horizontalalignment="left",
        fontfamily="monospace",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.3, pad=0.4),
    )

    for i in range(n_show):
        row = i // n_cols
        col = (i % n_cols) + 1

        ax = fig.add_subplot(gs[row, col])
        ret_row = retrieved_df.iloc[i]
        array_idx = int(ret_row["array_idx"])
        similarity = ret_row["similarity"]
        rank = ret_row["rank"]

        image_id = str(ret_row["image_id"])
        modality = ret_row["modality"]
        slice_orientation = ret_row["slice_orientation"]
        gender = ret_row["gender"]
        age_val = ret_row["age"]
        age_str = f"{int(age_val)}" if not pd.isna(age_val) else "?"
        ori_map = {"x": "sagittal", "y": "coronal", "z": "axial"}
        orientation_label = ori_map.get(slice_orientation, slice_orientation)

        img_path = get_image_path(metadata_dataset, array_idx)
        if img_path.exists():
            img = Image.open(img_path).convert("L")
            ax.imshow(np.array(img), cmap="gray")
        else:
            ax.text(0.5, 0.5, "Image\nNot Found", ha="center", va="center",
                    transform=ax.transAxes)

        ax.axis("off")
        ax.set_title(
            f"Rank {rank}: {image_id}\n"
            f"Sim score: {similarity:.3f}\n"
            f"{modality.upper()}, {orientation_label}, {gender}, {age_str}y",
            fontsize=8,
        )

    plt.savefig(output_path, dpi=150)
    print(f"  Saved visualization to {output_path.name}")
    plt.close()



def main() -> None:
    print("\n" + "=" * 70)
    print("LANGUAGE-BASED IMAGE RETRIEVAL ANALYSIS")
    print("=" * 70)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    (OUTPUT_DIR / "queries").mkdir(exist_ok=True)

    # -- Matching (optional) ---------------------------------------------------
    if RUN_MATCHING:
        run_matching()

    # -- Load sparse features (test split) -------------------------------------
    print("\nLoading sparse features (test split)...")
    sparse_features_raw, metadata_dataset = load_sparse_features_dataset(
        base_model=BASE_MODEL,
        sae_config=SAE_CONFIG,
        split=RETRIEVAL_SPLIT,
        sparse_features_root=SPARSE_FEATURES_ROOT,
        as_numpy=False,
    )
    if isinstance(sparse_features_raw, torch.Tensor):
        sparse_features: np.ndarray = sparse_features_raw.cpu().numpy()
    else:
        sparse_features = sparse_features_raw
    print(f"[ok] Sparse features: {sparse_features.shape}")
    print(f"[ok] Metadata: {len(metadata_dataset)} samples")

    n_features = sparse_features.shape[1]

    # -- Load feature descriptions (for mean_activation lookup) ----------------
    print("\nLoading feature descriptions...")
    feature_descriptions = load_feature_descriptions(FEATURE_DESCRIPTIONS_PATH)

    # -- Load matching JSON -----------------------------------------------------
    print(f"\nLoading query-feature matching from {MATCHING_JSON}...")
    with open(MATCHING_JSON) as f:
        matching: list = json.load(f)
    print(f"[ok] Loaded {len(matching)} matched queries")

    # -- Process each query -----------------------------------------------------
    all_results = []

    for entry in matching:
        query_name = entry["name"]
        query_text = entry["query_text"]
        feature_indices = np.array(entry["matched_feature_indices"])
        matched_descriptions = entry["matched_descriptions"]

        query_dir = OUTPUT_DIR / "queries" / query_name
        if query_dir.exists():
            print(f'\nSkipping query: {query_name} (output directory already exists)')
            continue

        print(f'\nProcessing query: {query_name}')
        print(f'  "{query_text}"')
        print(f'  features: {feature_indices.tolist()}')

        query_dir.mkdir(parents=True, exist_ok=True)

        # Build fingerprint
        query_vector = create_query_fingerprint(
            feature_indices, feature_descriptions, n_features
        )
        for feat_idx in feature_indices:
            print(f"    Feature {feat_idx}: mean_activation = {query_vector[feat_idx]:.6f}")

        # Retrieve
        print(f"  Retrieving top {N_RETRIEVALS} images...")
        retrieved_df = retrieve_images_from_query(
            query_vector, feature_indices, sparse_features, metadata_dataset, N_RETRIEVALS
        )
        print(f"  Top similarity: {retrieved_df.iloc[0]['similarity']:.4f}")

        # Enrich with metadata and save retrieval results
        retrieved_df = _enrich_with_metadata(retrieved_df, metadata_dataset)
        retrieved_df.to_csv(query_dir / "retrieval_results.csv", index=False)
        print(f"  [ok] Saved retrieval_results.csv")

        # Visualizations
        save_fingerprint_pdf(feature_indices, query_vector, query_dir / "fingerprint.pdf", INSET_COLOR)

        feature_indices_list: List[int] = [int(x) for x in feature_indices]

        visualize_query_results(
            query_name, query_text, feature_indices_list, matched_descriptions,
            feature_descriptions, retrieved_df, metadata_dataset,
            query_dir / "visualization.pdf",
        )

        visualize_query_retrieval_manuscript(
            feature_indices, retrieved_df, sparse_features, metadata_dataset,
            query_dir / "visualization_manuscript.pdf", INSET_COLOR,
        )

        # Collect for summary
        for _, row in retrieved_df.iterrows():
            all_results.append({
                "query_name":       query_name,
                "rank":             row["rank"],
                "image_id":         row["image_id"],
                "slice_idx":        row["slice_idx"],
                "similarity":       row["similarity"],
                "array_idx":        row["array_idx"],
                "modality":         row["modality"],
                "slice_orientation": row["slice_orientation"],
                "gender":           row["gender"],
                "age":              row["age"],
                "organs":           row["organs"],
            })

    # -- Save summary -----------------------------------------------------------
    pd.DataFrame(all_results).to_csv(OUTPUT_DIR / "all_queries_results.csv", index=False)
    print(f"\n[ok] Saved all_queries_results.csv")

    print("\n" + "=" * 70)
    print("ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
