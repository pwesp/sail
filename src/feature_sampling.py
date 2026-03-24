"""
Diversity-based sampling utilities for feature-based analyses.

Provides greedy max-dissimilarity sampling to select diverse high-activation samples.
This is a general-purpose utility that can be reused across multiple analyses:
- Auto-interpretation
- Feature visualization
- Feature ablation studies
- Concept discovery
"""

# stdlib
from pathlib import Path

# third-party
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# local
from src.dataloading import TotalSegmentatorLatentFeaturesDataset


def retrieve_diverse_samples(
    sparse_features: np.ndarray,
    metadata_dataset: TotalSegmentatorLatentFeaturesDataset,
    feature_idx: int,
    n_candidates: int = 20,
    n_samples: int = 6
) -> list[dict]:
    """
    Retrieve N diverse high-activation samples using greedy max-dissimilarity.

    Strategy:
    1. Get top-N_candidates samples with unique image_ids (by activation)
    2. Compute pairwise dissimilarity using full sparse feature vectors
    3. Greedily select n_samples that maximize diversity

    This reduces bias from:
    - Neighboring slices (same volume, minor spatial variation)
    - Single-patient concentration
    - Redundant near-duplicate slices

    And improves:
    - Semantic robustness of concept extraction
    - Generalization across patients and scan protocols
    - Quality of downstream interpretations

    Args:
        sparse_features: Full sparse features [N_samples, N_features]
        metadata_dataset: Dataset with metadata
        feature_idx: Feature to analyze
        n_candidates: Number of top activations to consider (default: 20)
        n_samples: Number of diverse samples to select (default: 6)

    Returns:
        List of sample dicts, ordered by selection (top activation first, then by diversity).
        Each dict contains:
        - array_idx: Index in dataset
        - activation: Feature activation value
        - image_path: Path to image file
        - age: Patient age
        - gender: Patient gender
        - modality: Imaging modality (CT/MRI)
        - pathology: Pathology information
        - slice_orientation: Slice orientation (axial/sagittal/coronal)
        - organs: List of organs present in slice
    """

    # 1. Get top-N candidates with unique image_ids
    activations = sparse_features[:, feature_idx]
    all_indices_sorted = np.argsort(activations)[::-1]

    candidates = []
    seen_image_ids = set()
    for idx in all_indices_sorted:
        img_id = metadata_dataset.image_ids[idx]
        if img_id in seen_image_ids:
            continue
        candidates.append(idx)
        seen_image_ids.add(img_id)
        if len(candidates) >= n_candidates:
            break

    # 2. Extract full sparse vectors for candidates
    candidate_vectors = sparse_features[candidates, :]  # [n_candidates, N_features]

    # 3. Compute pairwise cosine similarity
    similarity_matrix = cosine_similarity(candidate_vectors)
    dissimilarity_matrix = 1 - similarity_matrix

    # 4. Greedy max-dissimilarity sampling
    selected_candidate_indices = [0]  # Start with top activation
    for _ in range(n_samples - 1):
        # For each unselected candidate
        best_candidate: int | None = None
        best_min_dissim = -np.inf
        for cand_idx in range(len(candidates)):
            if cand_idx in selected_candidate_indices:
                continue
            # Min dissimilarity to already-selected set
            min_dissim = dissimilarity_matrix[cand_idx, selected_candidate_indices].min()
            if min_dissim > best_min_dissim:
                best_min_dissim = min_dissim
                best_candidate = cand_idx
        if best_candidate is not None:
            selected_candidate_indices.append(best_candidate)
        else:
            raise ValueError(f"No best candidate found at iteration {len(selected_candidate_indices)}")

    # 5. Map back to original dataset indices
    selected_indices = [candidates[i] for i in selected_candidate_indices]

    # 6. Extract metadata
    samples = []
    skip_cols = {'image_path', 'institute',
                 'kvp', 'echo_time', 'repetition_time', 'magnetic_field_strength',
                 'manufacturer', 'scanner_model', 'scanning_sequence',
                 'slice_thickness', 'source', 'split', 'study_type',
                 'pathology_location', 'age_bin'}

    for idx in selected_indices:
        sample_dict = {
            'array_idx': int(idx),
            'activation': float(activations[idx]),
            'image_path': Path(metadata_dataset.metadata['image_path'][idx]),
            'age': float(metadata_dataset.metadata['age'][idx]),
            'gender': str(metadata_dataset.metadata['gender'][idx]),
            'modality': str(metadata_dataset.modalities[idx]),
            'pathology': str(metadata_dataset.metadata['pathology'][idx]),
            'slice_orientation': str(metadata_dataset.slice_orientations[idx]),
        }

        # Extract organs
        organs_present = []
        for col_name in metadata_dataset.metadata.keys():
            if col_name in skip_cols:
                continue
            if metadata_dataset.metadata[col_name][idx]:
                organs_present.append(col_name)

        sample_dict['organs'] = organs_present
        samples.append(sample_dict)

    return samples
