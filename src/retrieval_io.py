"""
Retrieval I/O Utilities

CSV export functions for fingerprint-based image retrieval results.
Retrieval quality metrics live in src/fingerprint_retrieval.py.

Functions:
- save_reference_features_csv: Export reference fingerprints with metadata
- save_retrieval_results_csv: Export retrieval results with metadata
"""

# stdlib
from pathlib import Path
from typing import cast

# third-party
import numpy as np
import pandas as pd

# local
from src.dataloading import TotalSegmentatorLatentFeaturesDataset, get_reference_metadata


# ==================================================
# CSV Export
# ==================================================

def save_reference_features_csv(
    feature_info: dict[tuple[str, str, int], tuple[np.ndarray, np.ndarray]],
    reference_row_ids: dict[tuple[str, str, int], str],
    metadata_dataset: TotalSegmentatorLatentFeaturesDataset,
    output_path: Path,
    feature_type: str = "sparse"
) -> None:
    """
    Save reference image features to CSV with metadata.

    Exports a table where each row represents one feature from a reference image's
    fingerprint, including the feature index, activation value, and image metadata.

    Parameters
    ----------
    feature_info : Dict[Tuple[str, str, int], Tuple[np.ndarray, np.ndarray]]
        Dict mapping (modality, image_id, slice_idx) -> (feature_indices, activations)
    reference_row_ids : Dict[Tuple[str, str, int], str]
        Dict mapping (modality, image_id, slice_idx) -> unique row_id
    metadata_dataset : TotalSegmentatorLatentFeaturesDataset
        Dataset for retrieving image metadata
    output_path : Path
        Path where CSV file will be saved
    feature_type : str, optional
        Type of features ('sparse' or 'dense'), used for progress messages

    Notes
    -----
    Output CSV columns:
    - row_id: Unique identifier for reference image
    - reference_id: Image ID (e.g., 's0001')
    - reference_slice: Slice index
    - modality: CT or MRI
    - slice_orientation: x, y, or z
    - gender: Patient gender
    - age: Patient age
    - pathology: Pathology category
    - pathology_location: Anatomical location
    - organs_present: Semicolon-separated organ list
    - feature_idx: Feature index in dictionary
    - activation: Feature activation value
    """
    summary_rows = []

    for (modality, ref_id, ref_slice), (feature_indices, activations) in feature_info.items():
        row_id = reference_row_ids.get((modality, ref_id, ref_slice), "unknown")
        ref_metadata = get_reference_metadata(metadata_dataset, ref_id, ref_slice)

        for feat_idx, activation in zip(feature_indices, activations):
            summary_rows.append({
                "row_id": row_id,
                "reference_id": ref_id,
                "reference_slice": ref_slice,
                "modality": ref_metadata["modality"],
                "slice_orientation": ref_metadata["slice_orientation"],
                "gender": ref_metadata["gender"],
                "age": ref_metadata["age"],
                "pathology": ref_metadata["pathology"],
                "pathology_location": ref_metadata["pathology_location"],
                "organs_present": ref_metadata["organs_present"],
                "feature_idx": int(feat_idx),
                "activation": float(activation),
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path, index=False)
    print(f"[ok] Saved {feature_type} feature summary to {output_path}")


def save_retrieval_results_csv(
    retrieval_results: dict[tuple[str, str, int], pd.DataFrame],
    reference_row_ids: dict[tuple[str, str, int], str],
    metadata_dataset: TotalSegmentatorLatentFeaturesDataset,
    output_path: Path,
    retrieval_method: str
) -> None:
    """
    Save retrieval results to CSV with metadata for both reference and retrieved images.

    Exports a table where each row represents one retrieved image, including its
    rank, similarity score, and metadata for both the reference and retrieved images.

    Parameters
    ----------
    retrieval_results : Dict[Tuple[str, str, int], pd.DataFrame]
        Dict mapping (modality, image_id, slice_idx) -> DataFrame of retrieved images
        Each DataFrame must have columns: image_id, slice_idx, similarity
    reference_row_ids : Dict[Tuple[str, str, int], str]
        Dict mapping (modality, image_id, slice_idx) -> unique row_id
    metadata_dataset : TotalSegmentatorLatentFeaturesDataset
        Dataset for retrieving image metadata
    output_path : Path
        Path where CSV file will be saved
    retrieval_method : str
        Name of retrieval method (e.g., 'sparse_fingerprint', 'dense_full', 'dense_fingerprint')

    Notes
    -----
    Output CSV columns:
    - row_id: Unique identifier for reference image
    - reference_id: Reference image ID
    - reference_slice: Reference slice index
    - retrieval_method: Method used for retrieval
    - rank: Rank of retrieved image (1 = best match)
    - retrieved_image_id: Retrieved image ID
    - retrieved_slice_idx: Retrieved slice index
    - similarity: Similarity score (cosine similarity)
    - modality: Retrieved image modality
    - slice_orientation: Retrieved image orientation
    - gender: Retrieved image patient gender
    - age: Retrieved image patient age
    - pathology: Retrieved image pathology
    - pathology_location: Retrieved image pathology location
    - organs_present: Retrieved image organs list
    """
    summary_rows = []

    for (modality, ref_id, ref_slice), retrievals_df in retrieval_results.items():
        row_id = reference_row_ids.get((modality, ref_id, ref_slice), "unknown")

        for rank, (_, retrieval_row) in enumerate(retrievals_df.iterrows(), start=1):
            ret_img_id = cast(str, retrieval_row["image_id"])
            ret_slice_idx = int(retrieval_row["slice_idx"])
            similarity = float(retrieval_row["similarity"])

            ret_metadata = get_reference_metadata(metadata_dataset, ret_img_id, ret_slice_idx)

            summary_rows.append({
                "row_id": row_id,
                "reference_id": ref_id,
                "reference_slice": ref_slice,
                "retrieval_method": retrieval_method,
                "rank": rank,
                "retrieved_image_id": ret_img_id,
                "retrieved_slice_idx": ret_slice_idx,
                "similarity": similarity,
                "modality": ret_metadata["modality"],
                "slice_orientation": ret_metadata["slice_orientation"],
                "gender": ret_metadata["gender"],
                "age": ret_metadata["age"],
                "pathology": ret_metadata["pathology"],
                "pathology_location": ret_metadata["pathology_location"],
                "organs_present": ret_metadata["organs_present"],
            })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(output_path, index=False)
    print(f"[ok] Saved {retrieval_method} retrieved samples to {output_path}")
