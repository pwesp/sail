#!/usr/bin/env python3
"""
plot_fingerprint_retrieval.py

Figure 3: Sparse fingerprint retrieval for BiomedParse and DINOv3.
Retrieves the top-2 most similar images per model for five reference cases
spanning CT and MRI across multiple anatomical regions.
Saved as results/figure_fingerprint_retrieval.pdf.

5 row x 2*n_refs column layout.  Columns alternate: image | metadata.
  width_ratios = [3, 1] * n_refs  (metadata cols are 1/3 the width of image cols)

  Row 0    : Reference images
             - Inset bottom-left  - BiomedParse sparse fingerprint (orange)
             - Inset bottom-right - DINOv3 sparse fingerprint (teal)
  Rows 1-2 : BiomedParse sparse-fingerprint retrievals
             - Inset bottom-left - retrieved image's BiomedParse fingerprint (orange)
  Rows 3-4 : DINOv3 sparse-fingerprint retrievals
             - Inset bottom-right - retrieved image's DINOv3 fingerprint (teal)

  Each image column is followed by a narrow metadata column showing:
    modality - slice orientation - sex - age
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.axes
from PIL import Image

from src.dataloading import (
    load_sparse_features_dataset,
    TotalSegmentatorLatentFeaturesDataset,
    get_image_path,
)
from src.fingerprint_retrieval import (
    find_always_active_features,
    extract_top_k_sparse_features,
    retrieve_similar_images,
)
from src.fingerprint_visualization import add_fingerprint_inset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PARQUET_SPLIT  = "test"
TOP_K_FEATURES = 5
N_RETRIEVALS   = 2   # retrieved images per model

CONFIGS = {
    "biomedparse": "D_128_512_2048_8192_K_20_40_80_160",
    "dinov3":      "D_128_512_2048_8192_K_5_10_20_40",
}

MANUSCRIPT_REFERENCE_IMAGES = [
    ("ct",  "s0470", "z",  93),
    ("ct",  "s1132", "y", 123),
    ("ct",  "s0620", "z",  226),
    ("mri", "s0095", "z",  24),
    ("mri", "s0592", "x",  9),
    #("ct",  "s1371", "z", 178),
    #("ct",  "s0459", "z", 214),
    #("mri", "s0243", "y",  63),
]

SPARSE_FEATURES_ROOT = Path("results/total_seg_sparse_features")
OUTPUT_DIR           = Path("results")
FIGURE_NAME          = "figure_fingerprint_retrieval.pdf"

COLOR_BMP  = "darkorange"    # darkorange - BiomedParse
COLOR_DINO = "lightseagreen" # lightseagreen   - DINOv3


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_model_data(
    base_model: str,
) -> Tuple[np.ndarray, TotalSegmentatorLatentFeaturesDataset]:
    features, dataset = load_sparse_features_dataset(
        base_model=base_model,
        sae_config=CONFIGS[base_model],
        split=PARQUET_SPLIT,
        sparse_features_root=SPARSE_FEATURES_ROOT,
        as_numpy=True,
    )
    print(f"Loaded {base_model}: {len(features)} images, {features.shape[1]} features")
    return features, dataset


def find_reference_index(
    dataset: TotalSegmentatorLatentFeaturesDataset,
    modality: str,
    img_id: str,
    orientation: str,
    slice_idx: int,
) -> int:
    for i in range(len(dataset)):
        if (
            dataset.modalities[i]         == modality
            and dataset.image_ids[i]      == img_id
            and dataset.slice_orientations[i] == orientation
            and int(dataset.slice_indices[i]) == slice_idx
        ):
            return i
    raise ValueError(
        f"Reference not found: {modality} {img_id} {orientation} slice {slice_idx}"
    )


def _get_meta(
    dataset: TotalSegmentatorLatentFeaturesDataset,
    array_idx: int,
) -> Dict:
    """Return a flat metadata dict for one image (used for CSV export)."""
    gender = dataset.metadata.get("gender", None)
    age    = dataset.metadata.get("age",    None)
    return {
        "modality":          dataset.modalities[array_idx],
        "image_id":          dataset.image_ids[array_idx],
        "slice_nr":          int(dataset.slice_indices[array_idx]),
        "slice_orientation": dataset.slice_orientations[array_idx],
        "gender":            str(gender[array_idx]) if gender is not None else "unknown",
        "age":               float(age[array_idx])  if age    is not None else float("nan"),
    }


def _export_metadata_csv(
    all_meta_rows: List[Dict],
    output_dir: Path,
    stem: str,
) -> None:
    """
    Save metadata CSV - one row per image.

    Columns: type, reference_image_*, base_model, retrieved_image_*
    retrieved_image_* columns are None for reference rows.
    """
    def _ref_cols(m: Dict) -> Dict:
        return {
            "reference_image_modality":          m["modality"],
            "reference_image_id":                m["image_id"],
            "reference_image_slice_nr":          m["slice_nr"],
            "reference_image_slice_orientation": m["slice_orientation"],
            "reference_image_gender":            m["gender"],
            "reference_image_age":               m["age"],
        }

    def _ret_cols(m: Dict | None) -> Dict:
        if m is None:
            return {
                "retrieved_image_modality":          None,
                "retrieved_image_id":                None,
                "retrieved_image_slice_nr":          None,
                "retrieved_image_slice_orientation": None,
                "retrieved_image_gender":            None,
                "retrieved_image_age":               None,
            }
        return {
            "retrieved_image_modality":          m["modality"],
            "retrieved_image_id":                m["image_id"],
            "retrieved_image_slice_nr":          m["slice_nr"],
            "retrieved_image_slice_orientation": m["slice_orientation"],
            "retrieved_image_gender":            m["gender"],
            "retrieved_image_age":               m["age"],
        }

    records = []
    for entry in all_meta_rows:
        ref_meta = entry["reference"]
        records.append({"type": "reference",   "base_model": None,           **_ref_cols(ref_meta), **_ret_cols(None)})
        for meta in entry["biomedparse"]:
            records.append({"type": "retrieved", "base_model": "biomedparse", **_ref_cols(ref_meta), **_ret_cols(meta)})
        for meta in entry["dinov3"]:
            records.append({"type": "retrieved", "base_model": "dinov3",      **_ref_cols(ref_meta), **_ret_cols(meta)})

    df = pd.DataFrame(records)
    path = output_dir / f"{stem}_metadata.csv"
    df.to_csv(path, index=False)
    print(f"Metadata CSV saved to {path}")


def _export_features_csv(
    all_feat_rows: List[Dict],
    output_dir: Path,
    stem: str,
) -> None:
    """
    Save fingerprint features CSV.

    Columns: col, model, feature_rank, feature_idx, feature_value
    One row per (figure_col x model x top-k feature).
    """
    records = []
    for entry in all_feat_rows:
        col_idx = entry["col"]
        for model, (indices, values) in (
            ("biomedparse", entry["biomedparse"]),
            ("dinov3",      entry["dinov3"]),
        ):
            for rank, (idx, val) in enumerate(zip(indices, values), start=1):
                records.append({
                    "col":           col_idx,
                    "model":         model,
                    "feature_rank":  rank,
                    "feature_idx":   int(idx),
                    "feature_value": float(val),
                })
    df = pd.DataFrame(records)
    path = output_dir / f"{stem}_fingerprint_features.csv"
    df.to_csv(path, index=False)
    print(f"Fingerprint features CSV saved to {path}")


def _load_image(path: Path, orientation: str) -> np.ndarray:
    """Load a grayscale image, flipping axial slices horizontally (radiological convention)."""
    img = np.array(Image.open(path).convert("L"))
    # Axial slices: flip horizontally so patient right appears on the left (radiological convention)
    if orientation == "z":
        img = np.fliplr(img)
    # MRI s0602 additionally needs a horizontal flip (acquisition-specific orientation)
    if "s0602" in str(path) and "mri" in str(path):
        img = np.fliplr(img)
    return img


def _fill_metadata_ax(ax: matplotlib.axes.Axes, meta: Dict) -> None:
    """Display image metadata as text in a narrow axis (right of image)."""
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    ori_labels = {"x": "sagittal", "y": "coronal", "z": "axial"}
    ori = ori_labels.get(meta["slice_orientation"], meta["slice_orientation"])
    age_val = meta["age"]
    age = f"{age_val:.0f}y" if not np.isnan(age_val) else "?"

    lines = [meta["modality"].upper(), ori, meta["gender"], age]
    for i, line in enumerate(lines):
        ax.text(
            0.01, 0.95 - i * 0.12, line,
            transform=ax.transAxes,
            fontsize=13, va="top", ha="left", color="black",
        )



def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Load both models
    bmp_features,  bmp_dataset  = load_model_data("biomedparse")
    dino_features, dino_dataset = load_model_data("dinov3")

    bmp_always_active  = find_always_active_features(bmp_features)
    dino_always_active = find_always_active_features(dino_features)

    n_refs = len(MANUSCRIPT_REFERENCE_IMAGES)
    n_rows = 1 + N_RETRIEVALS * 2   # reference + BMP retrievals + DINOv3 retrievals
    fig, axes = plt.subplots(
        n_rows, 2 * n_refs,
        figsize=(n_refs * 3.15, n_rows * 2.2),
        gridspec_kw={"width_ratios": [3.0, 1.2] * n_refs},
    )

    # Collect data for CSV exports
    csv_meta_rows:  List[Dict] = []
    csv_feat_rows:  List[Dict] = []

    for col_idx, (modality, img_id, orientation, slice_idx) in enumerate(
        MANUSCRIPT_REFERENCE_IMAGES
    ):
        print(f"\nModality: {modality}, Image ID: {img_id}, Orientation: {orientation}, Slice Index: {slice_idx}")

        # -- look up each model's array index for this reference ----------
        bmp_ref_idx  = find_reference_index(bmp_dataset,  modality, img_id, orientation, slice_idx)
        dino_ref_idx = find_reference_index(dino_dataset, modality, img_id, orientation, slice_idx)

        # -- reference fingerprints ----------------------------------------
        bmp_ref_ind,  bmp_ref_val  = extract_top_k_sparse_features(
            bmp_features,  bmp_ref_idx,  TOP_K_FEATURES, bmp_always_active,  None)
        dino_ref_ind, dino_ref_val = extract_top_k_sparse_features(
            dino_features, dino_ref_idx, TOP_K_FEATURES, dino_always_active, None)

        # -- retrievals ----------------------------------------------------
        bmp_rets  = retrieve_similar_images(
            bmp_features,  bmp_dataset,  bmp_ref_idx,  bmp_ref_ind,  N_RETRIEVALS)
        dino_rets = retrieve_similar_images(
            dino_features, dino_dataset, dino_ref_idx, dino_ref_ind, N_RETRIEVALS)

        print(f"BMP retrievals:")
        print(bmp_rets)
        print(f"DINO retrievals:")
        print(dino_rets)

        # -- Row 0: reference + dual insets -------------------------------
        ref_img  = _load_image(get_image_path(bmp_dataset, bmp_ref_idx), orientation)
        ref_meta = _get_meta(bmp_dataset, bmp_ref_idx)
        axes[0, col_idx * 2].imshow(ref_img, cmap="gray")
        axes[0, col_idx * 2].axis("off")
        add_fingerprint_inset(axes[0, col_idx * 2], bmp_ref_ind,  bmp_ref_val,  COLOR_BMP,  "lower left")
        add_fingerprint_inset(axes[0, col_idx * 2], dino_ref_ind, dino_ref_val, COLOR_DINO, "lower right")
        _fill_metadata_ax(axes[0, col_idx * 2 + 1], ref_meta)

        # -- Rows 1-2: BiomedParse retrievals (turquoise inset, bottom-left) --
        bmp_ret_metas = []
        for ret_offset, (_, row) in enumerate(bmp_rets.iterrows()):
            ret_row_idx = 1 + ret_offset
            ret_arr_idx = int(row["array_idx"])
            ret_img     = _load_image(get_image_path(bmp_dataset, ret_arr_idx), bmp_dataset.slice_orientations[ret_arr_idx])
            axes[ret_row_idx, col_idx * 2].imshow(ret_img, cmap="gray")
            axes[ret_row_idx, col_idx * 2].axis("off")

            ret_ind, ret_val = extract_top_k_sparse_features(
                bmp_features, ret_arr_idx, TOP_K_FEATURES, bmp_always_active, None)
            add_fingerprint_inset(axes[ret_row_idx, col_idx * 2], ret_ind, ret_val, COLOR_BMP, "lower left")
            meta = _get_meta(bmp_dataset, ret_arr_idx)
            _fill_metadata_ax(axes[ret_row_idx, col_idx * 2 + 1], meta)
            bmp_ret_metas.append(meta)

        # -- Rows 3-4: DINOv3 retrievals (magenta inset, bottom-right) ---
        dino_ret_metas = []
        for ret_offset, (_, row) in enumerate(dino_rets.iterrows()):
            ret_row_idx = 1 + N_RETRIEVALS + ret_offset
            ret_arr_idx = int(row["array_idx"])
            ret_img     = _load_image(get_image_path(dino_dataset, ret_arr_idx), dino_dataset.slice_orientations[ret_arr_idx])
            axes[ret_row_idx, col_idx * 2].imshow(ret_img, cmap="gray")
            axes[ret_row_idx, col_idx * 2].axis("off")

            ret_ind, ret_val = extract_top_k_sparse_features(
                dino_features, ret_arr_idx, TOP_K_FEATURES, dino_always_active, None)
            add_fingerprint_inset(axes[ret_row_idx, col_idx * 2], ret_ind, ret_val, COLOR_DINO, "lower right")
            meta = _get_meta(dino_dataset, ret_arr_idx)
            _fill_metadata_ax(axes[ret_row_idx, col_idx * 2 + 1], meta)
            dino_ret_metas.append(meta)

        # -- Collect CSV data ----------------------------------------------
        csv_meta_rows.append({
            "col":        col_idx + 1,
            "reference":  ref_meta,
            "biomedparse": bmp_ret_metas,
            "dinov3":      dino_ret_metas,
        })
        csv_feat_rows.append({
            "col":        col_idx + 1,
            "biomedparse": (bmp_ref_ind,  bmp_ref_val),
            "dinov3":      (dino_ref_ind, dino_ref_val),
        })

    plt.subplots_adjust(wspace=0.04, hspace=0.05)

    # Export CSVs (before saving figure so any crash doesn't lose them)
    csv_stem = FIGURE_NAME.rsplit(".", 1)[0]
    _export_metadata_csv(csv_meta_rows, OUTPUT_DIR, csv_stem)
    _export_features_csv(csv_feat_rows, OUTPUT_DIR, csv_stem)

    # Save figure
    figure_path = OUTPUT_DIR / FIGURE_NAME
    fig.savefig(figure_path, dpi=300, bbox_inches="tight", facecolor="white")
    print(f"\nFigure saved to {figure_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
