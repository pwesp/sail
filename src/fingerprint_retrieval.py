"""
Fingerprint Extraction and Retrieval

This module provides core functionality for sparse and dense feature-based image retrieval
using "fingerprints" - small sets of highly discriminative features selected from sparse
or dense feature spaces.

Key concepts:
- **Sparse fingerprint**: Top-K most activated sparse features for an image
- **Dense fingerprint**: Top-K highest-magnitude dense features for an image
- **Similarity search**: Retrieve similar images using cosine similarity on fingerprints
- **Diversity constraint**: Only one slice per unique image_id in retrieval results

Functions:
- find_always_active_features: Identify features to exclude from fingerprints
- extract_top_k_sparse_features: Extract sparse fingerprint for an image
- extract_top_k_dense_features: Extract dense fingerprint for an image
- retrieve_similar_images: Retrieve images using sparse fingerprint
- retrieve_similar_images_dense: Retrieve images using full dense embedding
- retrieve_similar_images_dense_fingerprint: Retrieve images using dense fingerprint
- compute_retrieval_quality_scores: Evaluate retrieval quality in dense embedding space
- create_query_fingerprint: Build a synthetic query vector for language-driven retrieval
- retrieve_images_from_query: Retrieve images from a synthetic query fingerprint
"""

# stdlib

# third-party
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# local
from src.dataloading import TotalSegmentatorLatentFeaturesDataset


# ==================================================
# Feature Selection
# ==================================================

def find_always_active_features(sparse_features: np.ndarray) -> set:
    """
    Find features that are always active (>0) across all images.

    These features provide no discriminative power for similarity search
    and should be excluded from fingerprint selection.

    Parameters
    ----------
    sparse_features : np.ndarray
        (N, D) array of sparse activations where N is number of images
        and D is number of features

    Returns
    -------
    set
        Set of feature indices that are always active

    Examples
    --------
    >>> sparse_feats = np.array([[1.0, 0.0, 0.5], [1.0, 0.2, 0.0]])
    >>> always_active = find_always_active_features(sparse_feats)
    >>> print(always_active)
    {0}  # Feature 0 is always active (>0) in both images
    """
    print("\nChecking for always-active features...")
    always_active = set()

    for feat_idx in range(sparse_features.shape[1]):
        min_activation = sparse_features[:, feat_idx].min()
        if min_activation > 0:
            always_active.add(feat_idx)

    print(f"Found {len(always_active)} features that are always active (>0)")
    print(f"Always-active features: {sorted(always_active)}")
    return always_active


def extract_top_k_sparse_features(
    sparse_features: np.ndarray,
    array_idx: int,
    k: int,
    always_active_features: set,
    allowed_features: set | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract top-K most activated sparse features for a reference image.

    Features are selected by activation magnitude, excluding always-active features
    and optionally restricting to a subset of allowed features (e.g., interpretable features).

    Parameters
    ----------
    sparse_features : np.ndarray
        (N, D) array of sparse activations
    array_idx : int
        Index of the reference image in the array
    k : int
        Number of top features to extract
    always_active_features : set
        Set of feature indices to exclude (no discriminative power)
    allowed_features : set, optional
        If provided, only select from these features (e.g., interpretable features)
        If None, all features (except always-active) are candidates

    Returns
    -------
    feature_indices : np.ndarray
        Indices of selected features (length K), sorted by activation (descending)
    activation_values : np.ndarray
        Activation values for selected features (length K)

    Examples
    --------
    >>> sparse_feats = np.array([[0.0, 0.8, 0.3, 0.9]])
    >>> always_active = set()
    >>> indices, vals = extract_top_k_sparse_features(sparse_feats, 0, 2, always_active)
    >>> print(indices)  # Top 2 features by activation
    [3 1]  # Feature 3 (0.9) and Feature 1 (0.8)
    """
    # Get activations for this image
    activations = sparse_features[array_idx].copy()

    # Mask out always-active features by setting their activations to -inf
    for feat_idx in always_active_features:
        activations[feat_idx] = -np.inf

    # If allowed_features is specified, mask out features not in the allowed set
    if allowed_features is not None:
        for feat_idx in range(len(activations)):
            if feat_idx not in allowed_features:
                activations[feat_idx] = -np.inf

    # Get top-K indices (sorted descending)
    top_k_indices = np.argsort(activations)[-k:][::-1]
    top_k_vals = sparse_features[array_idx, top_k_indices]  # Get original values

    return top_k_indices, top_k_vals


def extract_top_k_dense_features(
    dense_features: np.ndarray,
    array_idx: int,
    k: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract top-K highest-magnitude dense features for a reference image.

    Unlike sparse features, dense features can be negative. Features are selected
    by absolute value magnitude.

    Parameters
    ----------
    dense_features : np.ndarray
        (N, D) array of dense embeddings
    array_idx : int
        Index of the reference image in the array
    k : int
        Number of top features to extract

    Returns
    -------
    feature_indices : np.ndarray
        Indices of selected features (length K), sorted by |value| (descending)
    activation_values : np.ndarray
        Original activation values for selected features (with sign preserved)

    Examples
    --------
    >>> dense_feats = np.array([[0.1, -0.8, 0.3, -0.9]])
    >>> indices, vals = extract_top_k_dense_features(dense_feats, 0, 2)
    >>> print(indices)  # Top 2 by absolute value
    [3 1]  # Feature 3 (|-0.9|=0.9) and Feature 1 (|-0.8|=0.8)
    >>> print(vals)
    [-0.9 -0.8]  # Original values (with sign)
    """
    # Get embedding for this image
    embedding = dense_features[array_idx].copy()

    # Get top-K indices by absolute value (sorted descending)
    abs_values = np.abs(embedding)
    top_k_indices = np.argsort(abs_values)[-k:][::-1]
    top_k_vals = embedding[top_k_indices]  # Get original values (with sign)

    return top_k_indices, top_k_vals


# ==================================================
# Similarity Search
# ==================================================

def compute_cosine_similarity(
    reference_vector: np.ndarray,
    dataset_vectors: np.ndarray
) -> np.ndarray:
    """
    Compute cosine similarity between reference and all dataset vectors.

    Cosine similarity ranges from -1 (opposite) to 1 (identical direction).
    For sparse features (all non-negative), similarity is in [0, 1].

    Parameters
    ----------
    reference_vector : np.ndarray
        (K,) array of reference feature values
    dataset_vectors : np.ndarray
        (N, K) array of dataset feature values

    Returns
    -------
    np.ndarray
        (N,) array of cosine similarity scores

    Notes
    -----
    Uses sklearn's cosine_similarity for numerical stability and efficiency.
    """
    # Reshape reference to (1, K) for sklearn
    ref_reshaped = reference_vector.reshape(1, -1)
    similarities = cosine_similarity(ref_reshaped, dataset_vectors)[0]
    return similarities


def _filter_one_slice_per_image(
    all_indices_sorted: np.ndarray,
    similarities: np.ndarray,
    metadata_dataset: TotalSegmentatorLatentFeaturesDataset,
    reference_idx: int | None,
    n: int
) -> pd.DataFrame:
    """
    Filter retrieval results to keep only one slice per unique image_id.

    This ensures diversity in retrieved images - we don't want all results
    to be different slices from the same 3D volume.

    Parameters
    ----------
    all_indices_sorted : np.ndarray
        Array indices sorted by similarity (descending)
    similarities : np.ndarray
        Similarity scores for all images
    metadata_dataset : TotalSegmentatorLatentFeaturesDataset
        Metadata dataset with image_id and slice_idx
    reference_idx : int or None
        Array index of reference image to exclude, or None for query-based retrieval
    n : int
        Number of unique images to retrieve

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: image_id, slice_idx, similarity, array_idx
        Contains top-N similar images (one slice per unique image_id)

    Notes
    -----
    This is a private helper function shared by all retrieval methods.
    """
    results = []
    seen_image_ids = set()
    if reference_idx is not None:
        ref_image_id = metadata_dataset.image_ids[reference_idx]
        seen_image_ids.add(ref_image_id)  # Exclude reference image_id

    for idx in all_indices_sorted:
        img_id = metadata_dataset.image_ids[idx]

        # Skip if we've already selected a slice from this image
        if img_id in seen_image_ids:
            continue

        # Add to results
        results.append({
            "image_id": img_id,
            "slice_idx": int(metadata_dataset.slice_indices[idx]),
            "similarity": float(similarities[idx]),
            "array_idx": int(idx)
        })
        seen_image_ids.add(img_id)

        # Stop when we have enough results
        if len(results) >= n:
            break

    return pd.DataFrame(results)


def retrieve_similar_images(
    sparse_features: np.ndarray,
    metadata_dataset: TotalSegmentatorLatentFeaturesDataset,
    reference_idx: int,
    feature_indices: np.ndarray,
    n: int
) -> pd.DataFrame:
    """
    Retrieve N most similar images using sparse fingerprint.

    Uses cosine similarity on selected sparse features (fingerprint) to find
    similar images. Only one slice per unique image_id is returned.

    Parameters
    ----------
    sparse_features : np.ndarray
        (N, D) array of sparse activations
    metadata_dataset : TotalSegmentatorLatentFeaturesDataset
        Metadata dataset with image_id and slice_idx
    reference_idx : int
        Array index of reference image
    feature_indices : np.ndarray
        Indices of features to use as fingerprint
    n : int
        Number of similar images to retrieve

    Returns
    -------
    pd.DataFrame
        DataFrame with top-N similar images, columns:
        - image_id: Unique image identifier
        - slice_idx: Slice index within the 3D volume
        - similarity: Similarity score
        - array_idx: Index in the feature array

    Notes
    -----
    Special case: When K=1 (single feature), similarity is based on absolute
    difference in activation: similarity = 1 / (1 + |diff|)
    This prevents division-by-zero issues with cosine similarity.

    Examples
    --------
    >>> # Retrieve 5 similar images using features [3, 7, 12]
    >>> results = retrieve_similar_images(sparse_feats, dataset, ref_idx=100,
    ...                                   feature_indices=np.array([3, 7, 12]), n=5)
    >>> print(results[['image_id', 'similarity']].head())
       image_id  similarity
    0     s0052    0.987
    1     s0123    0.982
    ...
    """
    # Extract reference fingerprint (K features)
    ref_vector = sparse_features[reference_idx, feature_indices]

    # Extract fingerprints for all images (N x K)
    dataset_vectors = sparse_features[:, feature_indices]

    # Compute similarities based on number of features
    if len(feature_indices) == 1:
        # Special case: single feature - use activation similarity
        # Similarity = 1 / (1 + |difference|), ranges from 0 to 1
        ref_activation = ref_vector[0]
        dataset_activations = dataset_vectors[:, 0]
        abs_diff = np.abs(dataset_activations - ref_activation)
        similarities = 1.0 / (1.0 + abs_diff)
    else:
        # Multiple features - use cosine similarity
        similarities = compute_cosine_similarity(ref_vector, dataset_vectors)

    # Set reference similarity to -inf so it's not selected
    similarities[reference_idx] = -np.inf

    # Get all indices sorted by similarity (descending)
    all_indices_sorted = np.argsort(similarities)[::-1]

    # Filter to keep only one slice per image_id
    return _filter_one_slice_per_image(
        all_indices_sorted, similarities, metadata_dataset, reference_idx, n
    )


def retrieve_similar_images_dense(
    dense_features: np.ndarray,
    metadata_dataset: TotalSegmentatorLatentFeaturesDataset,
    reference_idx: int,
    n: int
) -> pd.DataFrame:
    """
    Retrieve N most similar images using full dense embedding.

    Uses cosine similarity on the complete dense feature vector (not a fingerprint).
    This serves as a baseline for comparing against fingerprint-based retrieval.

    Parameters
    ----------
    dense_features : np.ndarray
        (N, D) array of dense embeddings
    metadata_dataset : TotalSegmentatorLatentFeaturesDataset
        Metadata dataset with image_id and slice_idx
    reference_idx : int
        Array index of reference image
    n : int
        Number of similar images to retrieve

    Returns
    -------
    pd.DataFrame
        DataFrame with top-N similar images (same format as retrieve_similar_images)

    Notes
    -----
    This retrieval uses ALL dense features (typically 1024 or 1536 dimensions)
    compared to fingerprint-based methods which use only K features (e.g., K=5).
    """
    # Extract reference embedding
    ref_vector = dense_features[reference_idx]

    # Compute cosine similarity with all images
    similarities = compute_cosine_similarity(ref_vector, dense_features)

    # Set reference similarity to -inf so it's not selected
    similarities[reference_idx] = -np.inf

    # Get all indices sorted by similarity (descending)
    all_indices_sorted = np.argsort(similarities)[::-1]

    # Filter to keep only one slice per image_id
    return _filter_one_slice_per_image(
        all_indices_sorted, similarities, metadata_dataset, reference_idx, n
    )


def retrieve_similar_images_dense_fingerprint(
    dense_features: np.ndarray,
    metadata_dataset: TotalSegmentatorLatentFeaturesDataset,
    reference_idx: int,
    feature_indices: np.ndarray,
    n: int
) -> pd.DataFrame:
    """
    Retrieve N most similar images using dense fingerprint.

    Uses cosine similarity on selected dense features (fingerprint) - analogous
    to sparse fingerprint retrieval but operating in dense embedding space.

    Parameters
    ----------
    dense_features : np.ndarray
        (N, D) array of dense embeddings
    metadata_dataset : TotalSegmentatorLatentFeaturesDataset
        Metadata dataset with image_id and slice_idx
    reference_idx : int
        Array index of reference image
    feature_indices : np.ndarray
        Indices of features to use as fingerprint
    n : int
        Number of similar images to retrieve

    Returns
    -------
    pd.DataFrame
        DataFrame with top-N similar images (same format as retrieve_similar_images)

    Notes
    -----
    Dense fingerprints select features by |magnitude| rather than activation value,
    since dense features can be negative. The fingerprint uses the original signed
    values for similarity computation.
    """
    # Extract reference fingerprint (K features)
    ref_vector = dense_features[reference_idx, feature_indices]

    # Extract fingerprints for all images (N x K)
    dataset_vectors = dense_features[:, feature_indices]

    # Compute cosine similarity
    similarities = compute_cosine_similarity(ref_vector, dataset_vectors)

    # Set reference similarity to -inf so it's not selected
    similarities[reference_idx] = -np.inf

    # Get all indices sorted by similarity (descending)
    all_indices_sorted = np.argsort(similarities)[::-1]

    # Filter to keep only one slice per image_id
    return _filter_one_slice_per_image(
        all_indices_sorted, similarities, metadata_dataset, reference_idx, n
    )


# ==================================================
# Retrieval Quality
# ==================================================

def compute_retrieval_quality_scores(
    retrieved_images_df: pd.DataFrame,
    reference_idx: int,
    dense_features: np.ndarray
) -> dict:
    """
    Compute quality metrics for retrieval results using dense embedding similarity.

    This measures how similar the retrieved images are to the reference image
    in the dense embedding space, regardless of which retrieval method was used.
    This provides a method-agnostic quality metric for comparing different
    retrieval approaches.

    Parameters
    ----------
    retrieved_images_df : pd.DataFrame
        DataFrame with retrieved images, must have 'array_idx' column containing
        array indices of retrieved images
    reference_idx : int
        Array index of reference image in the dense features array
    dense_features : np.ndarray
        (N, D) array of dense embeddings for all images

    Returns
    -------
    dict
        Dictionary with quality metrics:
        - mean_cosine_similarity: Average cosine similarity to reference
        - std_cosine_similarity: Standard deviation of similarities
        - min_cosine_similarity: Minimum similarity (worst match)
        - max_cosine_similarity: Maximum similarity (best match)
    """
    ref_embedding = dense_features[reference_idx]
    ret_indices = retrieved_images_df["array_idx"].values.astype(int)
    ret_embeddings = dense_features[ret_indices]
    similarities = cosine_similarity(ref_embedding.reshape(1, -1), ret_embeddings)[0]

    return {
        "mean_cosine_similarity": float(np.mean(similarities)),
        "std_cosine_similarity": float(np.std(similarities)),
        "min_cosine_similarity": float(np.min(similarities)),
        "max_cosine_similarity": float(np.max(similarities))
    }


# ==================================================
# Language-Driven Query Retrieval
# ==================================================

def create_query_fingerprint(
    feature_indices: np.ndarray,
    feature_descriptions: pd.DataFrame,
    n_features: int,
) -> np.ndarray:
    """
    Build a sparse query vector from the mean activations of selected features.

    Unlike image-based fingerprints (which use actual activations), this constructs
    a synthetic query vector by placing each feature's mean activation at its
    position - all other positions remain 0.

    Args:
        feature_indices: Indices of features selected for this query
        feature_descriptions: DataFrame with 'feature_idx' and 'mean_activation' columns
        n_features: Total number of features in the SAE (length of the output vector)

    Returns:
        1D float32 array of shape (n_features,)
    """
    query_vector = np.zeros(n_features, dtype=np.float32)
    for feat_idx in feature_indices:
        mean_act = feature_descriptions.loc[
            feature_descriptions["feature_idx"] == feat_idx, "mean_activation"
        ].values[0]
        query_vector[feat_idx] = mean_act
    return query_vector


def retrieve_images_from_query(
    query_vector: np.ndarray,
    feature_indices: np.ndarray,
    sparse_features: np.ndarray,
    metadata_dataset: TotalSegmentatorLatentFeaturesDataset,
    n: int,
) -> pd.DataFrame:
    """
    Retrieve images by cosine similarity to a pre-built query fingerprint.

    Analogous to retrieve_similar_images(), but operates on a synthetic query
    vector (from create_query_fingerprint) rather than a reference image index.
    Returns one row per unique image_id to avoid retrieving multiple slices from
    the same volume.

    Args:
        query_vector: Full-length sparse vector (n_features,) from create_query_fingerprint
        feature_indices: Feature positions used for similarity (sub-vector for comparison)
        sparse_features: (N, D) array of sparse activations for all images
        metadata_dataset: Dataset providing image_ids and slice_indices
        n: Number of images to retrieve

    Returns:
        DataFrame with columns: rank, image_id, slice_idx, similarity, array_idx
    """
    query_fingerprint = query_vector[feature_indices]
    dataset_fingerprints = sparse_features[:, feature_indices]

    if len(feature_indices) == 1:
        query_act = query_fingerprint[0]
        dataset_acts = dataset_fingerprints[:, 0]
        abs_diff = np.abs(dataset_acts - query_act)
        similarities = 1.0 / (1.0 + abs_diff)
    else:
        similarities = compute_cosine_similarity(query_fingerprint, dataset_fingerprints)

    all_indices_sorted = np.argsort(similarities)[::-1]

    df = _filter_one_slice_per_image(all_indices_sorted, similarities, metadata_dataset, None, n)
    df.insert(0, "rank", np.arange(1, len(df) + 1))
    return df
