"""
Data loading utilities for TotalSegmentator embeddings and sparse SAE features.

Provides dataset classes (TotalSegmentatorDataset, TotalSegmentatorLatentFeaturesDataset),
data modules for encoding and training, and helper functions for loading pre-computed
sparse feature activations and selecting interpretable features.
"""

# stdlib
import gc
from pathlib import Path
from typing import Any, cast, overload

# third-party
import lightning
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset

# ==================================================
# General functions
# ==================================================

def get_image_path(
    metadata_dataset: 'TotalSegmentatorLatentFeaturesDataset',
    array_idx: int
) -> Path:
    """
    Get image path from TotalSegmentator metadata dataset.

    Parameters
    ----------
    metadata_dataset : TotalSegmentatorLatentFeaturesDataset
        Metadata dataset with metadata dictionary containing image paths
    array_idx : int
        Index in the dataset array

    Returns
    -------
    Path
        Path to the image file

    Raises
    ------
    FileNotFoundError
        If the image file does not exist at the expected path
    """
    img_path = Path(metadata_dataset.metadata['image_path'][array_idx])
    if not img_path.exists():
        raise FileNotFoundError(f"Image not found: {img_path}")
    return img_path


def get_reference_metadata(
    metadata_dataset: 'TotalSegmentatorLatentFeaturesDataset',
    image_id: str,
    slice_idx: int
) -> dict[str, Any]:
    """
    Extract metadata for a reference image from TotalSegmentator dataset.

    Parameters
    ----------
    metadata_dataset : TotalSegmentatorLatentFeaturesDataset
        Dataset with metadata dictionary
    image_id : str
        Image identifier (e.g., 's0001')
    slice_idx : int
        Slice index within the 3D volume

    Returns
    -------
    dict
        Dictionary with metadata fields:
        - modality: Imaging modality (CT/MRI)
        - slice_orientation: Slice plane (x/y/z)
        - gender: Patient gender (m/f/unknown)
        - age: Patient age (float or None)
        - pathology: Pathology category
        - pathology_location: Anatomical location of pathology
        - organs_present: Semicolon-separated list of organs visible in slice

    Notes
    -----
    If the image/slice combination is not found, returns default 'unknown' values
    for all fields to prevent errors in downstream processing.
    """
    mask = (
        (metadata_dataset.image_ids == image_id) &
        (metadata_dataset.slice_indices == slice_idx)
    )
    matching_indices = np.where(mask)[0]

    if len(matching_indices) == 0:
        return {
            "modality": "unknown",
            "slice_orientation": "unknown",
            "gender": "unknown",
            "age": None,
            "pathology": "unknown",
            "pathology_location": "unknown",
            "organs_present": ""
        }

    idx = matching_indices[0]

    modality = metadata_dataset.modalities[idx]
    orientation = metadata_dataset.slice_orientations[idx]

    metadata = metadata_dataset.metadata
    gender = metadata['gender'][idx] if 'gender' in metadata else "unknown"
    age = metadata['age'][idx] if 'age' in metadata else None
    pathology = metadata['pathology'][idx] if 'pathology' in metadata else "unknown"
    pathology_location = metadata['pathology_location'][idx] if 'pathology_location' in metadata else "unknown"

    # Extract organs present (all organ columns that are True)
    organ_columns = [
        'adrenal_gland_left', 'adrenal_gland_right', 'aorta', 'atrial_appendage_left', 'autochthon_left', 'autochthon_right',
        'brachiocephalic_trunk', 'brachiocephalic_vein_left', 'brachiocephalic_vein_right', 'brain', 'clavicula_left', 'clavicula_right',
        'colon', 'common_carotid_artery_left', 'common_carotid_artery_right', 'costal_cartilages', 'duodenum', 'esophagus',
        'femur_left', 'femur_right', 'gallbladder', 'gluteus_maximus_left', 'gluteus_maximus_right', 'gluteus_medius_left',
        'gluteus_medius_right', 'gluteus_minimus_left', 'gluteus_minimus_right', 'heart', 'hip_left', 'hip_right',
        'humerus_left', 'humerus_right', 'iliac_artery_left', 'iliac_artery_right', 'iliac_vena_left', 'iliac_vena_right',
        'iliopsoas_left', 'iliopsoas_right', 'inferior_vena_cava', 'intervertebral_discs', 'kidney_cyst_left', 'kidney_cyst_right',
        'kidney_left', 'kidney_right', 'liver', 'lung_left', 'lung_lower_lobe_left', 'lung_lower_lobe_right',
        'lung_middle_lobe_right', 'lung_right', 'lung_upper_lobe_left', 'lung_upper_lobe_right', 'pancreas', 'portal_vein_and_splenic_vein',
        'prostate', 'pulmonary_vein', 'rib_left_1', 'rib_left_10', 'rib_left_11', 'rib_left_12',
        'rib_left_2', 'rib_left_3', 'rib_left_4', 'rib_left_5', 'rib_left_6', 'rib_left_7',
        'rib_left_8', 'rib_left_9', 'rib_right_1', 'rib_right_10', 'rib_right_11', 'rib_right_12',
        'rib_right_2', 'rib_right_3', 'rib_right_4', 'rib_right_5', 'rib_right_6', 'rib_right_7',
        'rib_right_8', 'rib_right_9', 'sacrum', 'scapula_left', 'scapula_right', 'skull',
        'small_bowel', 'spinal_cord', 'spleen', 'sternum', 'stomach', 'subclavian_artery_left',
        'subclavian_artery_right', 'superior_vena_cava', 'thyroid_gland', 'trachea', 'urinary_bladder', 'vertebrae',
        'vertebrae_C1', 'vertebrae_C2', 'vertebrae_C3', 'vertebrae_C4', 'vertebrae_C5', 'vertebrae_C6',
        'vertebrae_C7', 'vertebrae_L1', 'vertebrae_L2', 'vertebrae_L3', 'vertebrae_L4', 'vertebrae_L5',
        'vertebrae_S1', 'vertebrae_T1', 'vertebrae_T10', 'vertebrae_T11', 'vertebrae_T12', 'vertebrae_T2',
        'vertebrae_T3', 'vertebrae_T4', 'vertebrae_T5', 'vertebrae_T6', 'vertebrae_T7', 'vertebrae_T8',
        'vertebrae_T9'
    ]

    organs_present = []
    for organ in organ_columns:
        if organ in metadata and metadata[organ][idx]:
            organs_present.append(organ)

    return {
        "modality": modality,
        "slice_orientation": orientation,
        "gender": gender,
        "age": age,
        "pathology": pathology,
        "pathology_location": pathology_location,
        "organs_present": "; ".join(organs_present) if organs_present else ""
    }


@overload
def load_sparse_features_dataset(
    base_model: str,
    sae_config: str,
    split: str = "val",
    sparse_features_root: Path = Path("results/total_seg_sparse_features"),
    as_numpy: bool = True
) -> tuple[np.ndarray, 'TotalSegmentatorLatentFeaturesDataset']:
    ...


@overload
def load_sparse_features_dataset(
    base_model: str,
    sae_config: str,
    split: str = "val",
    sparse_features_root: Path = Path("results/total_seg_sparse_features"),
    as_numpy: bool = False
) -> tuple[torch.Tensor, 'TotalSegmentatorLatentFeaturesDataset']:
    ...


def load_sparse_features_dataset(
    base_model: str,
    sae_config: str,
    split: str = "val",
    sparse_features_root: Path = Path("results/total_seg_sparse_features"),
    as_numpy: bool = True
) -> tuple[np.ndarray | torch.Tensor, 'TotalSegmentatorLatentFeaturesDataset']:
    """
    Load sparse features for a given model configuration.

    This is a shared utility function used across multiple analysis scripts
    to consistently load sparse features and metadata.

    Parameters
    ----------
    base_model : str
        Base model name (e.g., "biomedparse", "dinov3")
    sae_config : str
        SAE configuration (e.g., "D_128_512_2048_8192_K_5_10_20_40")
    split : str, optional
        Dataset split to load (default: "val")
    sparse_features_root : Path, optional
        Root directory for sparse features (default: results/total_seg_sparse_features)
    as_numpy : bool, optional
        If True, return features as numpy array; if False, return as torch tensor (default: True)

    Returns
    -------
    features : np.ndarray or torch.Tensor
        Sparse features matrix of shape (n_samples, n_features)
    dataset : TotalSegmentatorLatentFeaturesDataset
        Dataset object containing metadata and access to individual samples

    Examples
    --------
    >>> sparse_features, metadata_dataset = load_sparse_features_dataset(
    ...     base_model="dinov3",
    ...     sae_config="D_128_512_2048_8192_K_5_10_20_40",
    ...     split="val"
    ... )
    >>> print(f"Loaded {sparse_features.shape[0]} samples with {sparse_features.shape[1]} features")
    """
    parquet_path = (
        sparse_features_root / f"{base_model}_matryoshka_sae" /
        sae_config / f"total_seg_sparse_features_{split}.parquet"
    )

    dataset = TotalSegmentatorLatentFeaturesDataset(
        feature_type='sparse_features',
        set_type='split',
        parquet_path=str(parquet_path),
        shuffle=False,
        additional_metadata="all"
    )

    features = dataset.features.numpy() if as_numpy else dataset.features
    return features, dataset

# ==================================================
# TotalSegmentator
# ==================================================

class TotalSegmentatorDataset(Dataset):
    
    def __init__(
        self,
        ct_data_dir: str = "data/total_seg_ct",
        mri_data_dir: str = "data/total_seg_mri",
        ct_metadata_csv: str = "data/total_seg_ct/metadata.csv",
        mri_metadata_csv: str = "data/total_seg_mri/metadata.csv"
    ):
        super().__init__()

        # Expected column structure constants
        self.CT_METADATA_COUNT = 14
        self.MRI_METADATA_COUNT = 17
        self.CT_ORGAN_COUNT = 117
        self.MRI_ORGAN_COUNT = 50
        
        self.ct_data_dir = Path(ct_data_dir)
        self.mri_data_dir = Path(mri_data_dir)

        # low_memory=False avoids mixed-dtype column warnings
        df_ct = pd.read_csv(ct_metadata_csv, low_memory=False)
        df_mri = pd.read_csv(mri_metadata_csv, low_memory=False)

        self.ct_metadata_names = self._extract_metadata_columns(df_ct, modality="ct")
        self.mri_metadata_names = self._extract_metadata_columns(df_mri, modality="mri")
        self.ct_organs = self._extract_organ_columns(df_ct, modality="ct")
        self.mri_organs = self._extract_organ_columns(df_mri, modality="mri")

        self.all_metadata_names: list[str] = self._compute_unified_metadata_names()
        self.all_organs: list[str] = self._compute_unified_organ_names()

        self.organ_to_id: dict[str, int] = {organ: idx for idx, organ in enumerate(self.all_organs)}

        self.df_combined_metadata = self._combine_and_normalize_metadata(df_ct, df_mri)

        self._log_dataset_stats(df_ct, df_mri)
    
    def __len__(self) -> int:
        return len(self.df_combined_metadata)
        
    def __getitem__(self, idx: int) -> dict:
        """Load image and metadata for a single slice."""
        row = self.df_combined_metadata.iloc[idx]

        img = Image.open(row['image_path'])
        if img.mode != 'RGB':
            img = img.convert('RGB')
        img = torch.from_numpy(np.array(img, dtype=np.float32))

        present_organs = [
            organ for organ in self.all_organs
            if organ in row and row[organ] == 1
        ]
        organ_ids = [self.organ_to_id[organ] for organ in present_organs]
        
        return {
            'image': img,
            'idx': idx,  # Original index in the dataset
            'metadata': row.to_dict(),  # Full metadata row as dict
            'modality': row['modality'],
            'slice_orientation': row['slice_orientation'],
            'slice_idx': row['slice_idx'],
            'organ_ids': organ_ids,
            'organ_names': present_organs,
        }
    
    def _extract_metadata_columns(self, df: pd.DataFrame, modality: str) -> pd.Index:
        """Extract metadata column names and validate structure."""
        n_metadata = self.CT_METADATA_COUNT if modality == "ct" else self.MRI_METADATA_COUNT
        metadata_names = df.columns[:n_metadata]

        assert metadata_names[0] == 'image_id', "First metadata column must be 'image_id'"
        if modality == "ct":
            assert metadata_names[-1] == 'pathology_location', "Last CT metadata column must be 'pathology_location'"
        else:
            assert metadata_names[-1] == 'source', "Last MRI metadata column must be 'source'"
        
        return metadata_names  # type: ignore[return-value]
    
    def _extract_organ_columns(self, df: pd.DataFrame, modality: str) -> pd.Index:
        """Extract organ column names and validate structure."""
        n_metadata = self.CT_METADATA_COUNT if modality == "ct" else self.MRI_METADATA_COUNT
        expected_count = self.CT_ORGAN_COUNT if modality == "ct" else self.MRI_ORGAN_COUNT
        
        organs = df.columns[n_metadata:]
        assert len(organs) == expected_count, f"Expected {expected_count} organ columns, got {len(organs)}"
        assert organs[0] == 'adrenal_gland_left', "First organ must be 'adrenal_gland_left'"
        
        if modality == "ct":
            assert organs[-1] == 'vertebrae_T9', "Last CT organ must be 'vertebrae_T9'"
        else:
            assert organs[-1] == 'vertebrae', "Last MRI organ must be 'vertebrae'"
        
        return organs  # type: ignore[return-value]
    
    def _compute_unified_metadata_names(self) -> list[str]:
        """Compute sorted list of all unique metadata column names."""
        all_names = np.unique(np.concatenate([
            np.array(self.ct_metadata_names),
            np.array(self.mri_metadata_names)
        ]))
        result: list[str] = [str(x) for x in all_names]
        return result
    
    def _compute_unified_organ_names(self) -> list[str]:
        """Compute sorted list of all unique organ names."""
        all_organs = np.unique(np.concatenate([
            np.array(self.ct_organs),
            np.array(self.mri_organs)
        ]))
        result: list[str] = [str(x) for x in all_organs]
        return result
    
    def _normalize_dataframe_columns(
        self, 
        df: pd.DataFrame, 
        modality: str,
        all_metadata: list[str],
        all_organs: list[str]
    ) -> pd.DataFrame:
        """
        Normalize dataframe to have all metadata and organ columns.
        
        Adds missing columns (None for metadata, 0 for organs) and ensures
        consistent column order. Returns a new dataframe (does not modify input).
        """
        df = df.copy()  # avoid modifying the caller's dataframe

        for col in all_metadata:
            if col not in df.columns:
                df[col] = pd.Series([None] * len(df), dtype='object')

        for col in all_organs:
            if col not in df.columns:
                df[col] = 0

        df['modality'] = modality

        column_order = all_metadata + ['modality'] + all_organs
        result = df[column_order]
        assert isinstance(result, pd.DataFrame)
        return result
    
    def _combine_and_normalize_metadata(
        self, 
        df_ct: pd.DataFrame, 
        df_mri: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Combine CT and MRI metadata with normalized column structure.
        
        Returns a single dataframe with all slices sorted by modality, image_id,
        slice_orientation, and slice_idx.
        """
        df_ct_normalized = self._normalize_dataframe_columns(
            df_ct, 'ct', self.all_metadata_names, self.all_organs
        )
        df_mri_normalized = self._normalize_dataframe_columns(
            df_mri, 'mri', self.all_metadata_names, self.all_organs
        )
        
        # Align dtypes so pandas concatenation doesn't produce unexpected object columns.
        # Columns that exist only in one modality are all-NA in the other; their dtype
        # must match the version that has real data.
        for col in self.ct_metadata_names:
            if col not in self.mri_metadata_names:
                df_mri_normalized[col] = df_mri_normalized[col].astype(df_ct_normalized[col].dtype)

        for col in self.mri_metadata_names:
            if col not in self.ct_metadata_names:
                df_ct_normalized[col] = df_ct_normalized[col].astype(df_mri_normalized[col].dtype)
        
        df_combined = pd.concat([df_ct_normalized, df_mri_normalized], ignore_index=True)
        df_combined = df_combined.sort_values(
            by=['modality', 'image_id', 'slice_orientation', 'slice_idx']
        )
        df_combined = df_combined.reset_index(drop=True)
        
        return df_combined
    
    def _log_dataset_stats(self, df_ct: pd.DataFrame, df_mri: pd.DataFrame) -> None:
        """Print dataset loading statistics."""
        print("CT data:")
        print(f"\tLoaded {len(df_ct)} slices from preprocessed TotalSegmentator CT dataset")
        print(f"\tNumber of organs found: {len(self.ct_organs)}")
        print("MRI data:")
        print(f"\tLoaded {len(df_mri)} slices from preprocessed TotalSegmentator MRI dataset")
        print(f"\tNumber of organs found: {len(self.mri_organs)}")
        print("Overall:")
        print(f"\tLoaded {len(self.df_combined_metadata)} slices")
        print(f"\tNumber of unique organs found: {len(self.all_organs)}")


class TotalSegmentatorEncodingDataModule(lightning.LightningDataModule):
    def __init__(
        self,
        totalseg_dataset: TotalSegmentatorDataset,
        batch_size: int = 8,
        num_workers: int = 1,
        start_idx: int = 0,
        pin_memory: bool = True,
        prefetch_factor: int = 2
    ):
        super().__init__()
        self.totalseg_dataset = totalseg_dataset
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.start_idx = start_idx
        self.pin_memory = pin_memory
        self.prefetch_factor = prefetch_factor
        
        self.dataset_subset: Dataset | None = None
    
    def setup(self, stage: str | None = None):
        """Setup dataset subset for encoding."""
        indices = list(range(self.start_idx, len(self.totalseg_dataset)))
        self.dataset_subset = torch.utils.data.Subset(self.totalseg_dataset, indices)
    
    def _collate_fn(self, batch: list[dict]) -> dict:
        """Custom collate function for encoding batches."""
        images = [item['image'] for item in batch]
        metadata = [item['metadata'] for item in batch]
        original_indices = [item['idx'] for item in batch]
        
        return {
            'images': images,
            'metadata': metadata,
            'original_indices': original_indices
        }
    
    def predict_dataloader(self):
        """Return dataloader for prediction/encoding."""
        if self.dataset_subset is None:
            self.setup()
        
        assert self.dataset_subset is not None, "Dataset not initialized"
        
        return torch.utils.data.DataLoader(
            self.dataset_subset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            persistent_workers=False,  # Disabled to prevent resource leaks in long-running jobs
            prefetch_factor=self.prefetch_factor if self.num_workers > 0 else None
        )


# ==================================================
# TotalSegmentator latent features
# ==================================================
class TotalSegmentatorLatentFeaturesDataset(Dataset):
    """
    PyTorch Dataset for loading TotalSegmentator latent features (embeddings or sparse features) for the TotalSegmentator dataset (biomedparse, biomedparse_random, dinov3) from parquet files.
    
    Loads latent features from pre-computed parquet files (train/val/test splits).
    Metadata is loaded into memory, latent features are read on-demand.
    """
    
    def __init__(
        self,
        parquet_path: str,
        feature_type: str,
        set_type: str = 'split',
        shuffle: bool = True,
        feature_ids: list[int] | None = None,
        random_seed: int = 42,
        additional_metadata: list[str] | str | None = None
    ):
        """
        Args:
            parquet_path: Path to parquet file containing embeddings or sparse features
            feature_type: Type of feature to load. Must be 'embeddings' or 'sparse_features'.
            set_type: Type of set to load. Must be 'split' or 'all'.
            shuffle: Whether to shuffle the dataset indices
            feature_ids: List of feature IDs to load. If None, all features are loaded.
            random_seed: Random seed for shuffling
            additional_metadata: If not None, the additional metadata columns are loaded and included in the return dictionary. Must be a list of column names or 'all' to load all metadata columns.
        """
        super().__init__()
        
        self.parquet_path = parquet_path
        self.feature_type = feature_type
        self.feature_ids = feature_ids
        self.additional_metadata = additional_metadata

        assert feature_type in ['sparse_features', 'embeddings'], "Invalid feature type. Must be 'sparse_features' or 'embeddings'."
        assert set_type in ['split', 'all'], "Invalid set type. Must be 'split' or 'all'."
        
        print(f"Loading latent features for TotalSegmentator dataset from {parquet_path}...")
        print(f"  Feature type: {feature_type}")
        table = pq.read_table(parquet_path)

        # "_image_modality_key" is an internal partitioning artifact; drop it
        if "_image_modality_key" in table.column_names:
            table = table.drop_columns(['_image_modality_key'])
        
        all_columns = set(table.column_names)

        core_columns = {'modality', 'image_id', 'slice_idx', 'slice_orientation'}

        if feature_type == 'sparse_features':
            feature_columns = {'sparse_features', 'sparse_features_shape', 'sparse_features_dtype'}
        elif feature_type == 'embeddings':
            feature_columns = {'embedding', 'embedding_shape', 'embedding_dtype'}
        else:
            raise ValueError(f"Invalid feature type: {feature_type}. Must be 'sparse_features' or 'embeddings'.")
        
        if set_type == 'split':
            patient_columns = {'age', 'gender'}
        elif set_type == 'all':
            patient_columns = {'age', 'gender'}
        else:
            raise ValueError(f"Invalid set type: {set_type}. Must be 'split' or 'all'.")

        diagnostic_columns = {'pathology', 'pathology_location'}
        imaging_columns = {'echo_time', 'image_path', 'institute', 'kvp', 'magnetic_field_strength', 'manufacturer','repetition_time', 'scanner_model', 'scanning_sequence', 'slice_thickness', 'source', 'study_type'}
        organ_columns = {
            'adrenal_gland_left', 'adrenal_gland_right', 'aorta', 'atrial_appendage_left', 'autochthon_left', 'autochthon_right',
            'brachiocephalic_trunk', 'brachiocephalic_vein_left', 'brachiocephalic_vein_right', 'brain', 'clavicula_left', 'clavicula_right',
            'colon', 'common_carotid_artery_left', 'common_carotid_artery_right', 'costal_cartilages', 'duodenum', 'esophagus',
            'femur_left', 'femur_right', 'gallbladder', 'gluteus_maximus_left', 'gluteus_maximus_right', 'gluteus_medius_left',
            'gluteus_medius_right', 'gluteus_minimus_left', 'gluteus_minimus_right', 'heart', 'hip_left', 'hip_right',
            'humerus_left', 'humerus_right', 'iliac_artery_left', 'iliac_artery_right', 'iliac_vena_left', 'iliac_vena_right',
            'iliopsoas_left', 'iliopsoas_right', 'inferior_vena_cava', 'intervertebral_discs', 'kidney_cyst_left', 'kidney_cyst_right',
            'kidney_left', 'kidney_right', 'liver', 'lung_left', 'lung_lower_lobe_left', 'lung_lower_lobe_right',
            'lung_middle_lobe_right', 'lung_right', 'lung_upper_lobe_left', 'lung_upper_lobe_right', 'pancreas', 'portal_vein_and_splenic_vein',
            'prostate', 'pulmonary_vein', 'rib_left_1', 'rib_left_10', 'rib_left_11', 'rib_left_12',
            'rib_left_2', 'rib_left_3', 'rib_left_4', 'rib_left_5', 'rib_left_6', 'rib_left_7',
            'rib_left_8', 'rib_left_9', 'rib_right_1', 'rib_right_10', 'rib_right_11', 'rib_right_12',
            'rib_right_2', 'rib_right_3', 'rib_right_4', 'rib_right_5', 'rib_right_6', 'rib_right_7',
            'rib_right_8', 'rib_right_9', 'sacrum', 'scapula_left', 'scapula_right', 'skull',
            'small_bowel', 'spinal_cord', 'spleen', 'sternum', 'stomach', 'subclavian_artery_left',
            'subclavian_artery_right', 'superior_vena_cava', 'thyroid_gland', 'trachea', 'urinary_bladder', 'vertebrae',
            'vertebrae_C1', 'vertebrae_C2', 'vertebrae_C3', 'vertebrae_C4', 'vertebrae_C5', 'vertebrae_C6',
            'vertebrae_C7', 'vertebrae_L1', 'vertebrae_L2', 'vertebrae_L3', 'vertebrae_L4', 'vertebrae_L5',
            'vertebrae_S1', 'vertebrae_T1', 'vertebrae_T10', 'vertebrae_T11', 'vertebrae_T12', 'vertebrae_T2',
            'vertebrae_T3', 'vertebrae_T4', 'vertebrae_T5', 'vertebrae_T6', 'vertebrae_T7', 'vertebrae_T8',
            'vertebrae_T9'
        }
        
        # Sanity check: all_columns should match the union of all known column sets
        expected_all_columns = core_columns | feature_columns | patient_columns | diagnostic_columns | imaging_columns | organ_columns
        if all_columns != expected_all_columns:
            missing = expected_all_columns - all_columns
            extra = all_columns - expected_all_columns
            error_msg = f"Column mismatch. Missing: {sorted(missing) if missing else 'none'}. Extra: {sorted(extra) if extra else 'none'}."
            raise AssertionError(error_msg)

        metadata_columns = patient_columns | diagnostic_columns | imaging_columns | organ_columns

        # Keep only valid metadata columns; core and feature columns are handled separately
        if isinstance(self.additional_metadata, list):
            valid_columns = [col for col in self.additional_metadata if col in metadata_columns]
            if valid_columns:
                self.additional_metadata = valid_columns
            else:
                print(f"  Warning: All requested additional_metadata columns are core or sparse feature columns. Ignoring additional_metadata.")
                self.additional_metadata = None
        elif isinstance(self.additional_metadata, str):
            if self.additional_metadata=="all":
                self.additional_metadata = metadata_columns
            else:
                print(f"  Warning: Invalid additional_metadata value. Ignoring additional_metadata.")
                self.additional_metadata = None

        self.modalities = table['modality'].to_numpy()
        self.image_ids = table['image_id'].to_numpy()
        self.slice_indices = table['slice_idx'].to_numpy()
        self.slice_orientations = table['slice_orientation'].to_numpy()

        # TODO: MRI s0602 is mislabelled as "y" (coronal) in the source data; it is actually "x" (sagittal).
        fix_mask = (self.image_ids == "s0602") & (self.modalities == "mri")
        self.slice_orientations[fix_mask] = "x"

        # Store features (as a tensor) in memory.
        # Don't use list->np.stack() — it creates a full temporary copy of the matrix,
        # which doubles peak memory usage (bad on shared-memory / UMA systems).
        if feature_type == 'sparse_features':
            feature_column = 'sparse_features'
        elif feature_type == 'embeddings':
            feature_column = 'embedding'
        else:
            raise ValueError(f"Invalid feature type: {feature_type}. Must be 'sparse_features' or 'embeddings'.")

        n_rows = len(self.image_ids)
        if n_rows == 0:
            raise ValueError(f"Parquet file contains 0 rows: {parquet_path}")

        # Determine feature dimensionality from the first row (float32 blob)
        first_blob = table[feature_column][0].as_py()
        first_vec = np.frombuffer(first_blob, dtype=np.float32)
        if feature_ids is not None:
            first_vec = first_vec[self.feature_ids]
        d = int(first_vec.shape[0])

        features_np = np.empty((n_rows, d), dtype=np.float32)
        for i in range(n_rows):
            blob = table[feature_column][i].as_py()
            vec = np.frombuffer(blob, dtype=np.float32)
            if feature_ids is not None:
                vec = vec[self.feature_ids]
            # Copy into preallocated array (avoids building a giant list)
            features_np[i] = vec
        self.features = torch.from_numpy(features_np)
        
        self.metadata = {}
        if self.additional_metadata:
                for col_name in sorted(self.additional_metadata):
                    self.metadata[col_name] = table[col_name].to_numpy()
                print(f"  Loaded {len(self.metadata)} additional metadata columns")

        self.included_patient_data = [patient for patient in patient_columns if patient in list(self.metadata.keys())]
        self.included_diagnostics = [diagnostic for diagnostic in diagnostic_columns if diagnostic in list(self.metadata.keys())]
        self.included_imaging_data = [imaging for imaging in imaging_columns if imaging in list(self.metadata.keys())]
        self.included_organs = [organ for organ in organ_columns if organ in list(self.metadata.keys())]

        # All requested data is loaded, table is no longer needed, free up memory
        del table
        gc.collect()
        # On some systems (especially UMA/shared-memory), releasing Python objects
        # does not immediately return memory to the OS. Best-effort trim helps.
        try:
            import ctypes
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:
            pass
        
        self.length = len(self.image_ids)

        self.indices = np.arange(self.length)
        if shuffle:
            np.random.seed(random_seed)
            np.random.shuffle(self.indices)
        
        print(f"  Total samples: {self.length}")
        print(f"  Sparse features tensor shape: {self.features.shape}")
        print(f"  Metadata columns: {list(self.metadata.keys())}")
        # Calculate total memory usage for all data stored in memory
        total_memory_mb = 0
        total_memory_mb += self.features.element_size() * self.features.nelement() / 1024**2
        total_memory_mb += self.modalities.nbytes / 1024**2
        total_memory_mb += self.image_ids.nbytes / 1024**2
        total_memory_mb += self.slice_indices.nbytes / 1024**2
        total_memory_mb += self.slice_orientations.nbytes / 1024**2
        total_memory_mb += self.indices.nbytes / 1024**2
        for col_data in self.metadata.values():
            total_memory_mb += col_data.nbytes / 1024**2
        print(f"  Memory usage: ~{total_memory_mb:.0f} MB")
    
    def __len__(self):
        return self.length
    
    def _convert_to_python_type(self, value):
        """
        Convert numpy/array types to Python native types.
        
        Args:
            value: Value that might be numpy scalar, array, or Python type
            
        Returns:
            Python native type (int, float, str, bool, None, or list/array if multi-dimensional)
        """
        if isinstance(value, np.ndarray):
            if value.ndim == 0:
                # Scalar array -> Python scalar
                return value.item()
            else:
                # Multi-dimensional array -> keep as numpy array
                return value
        elif isinstance(value, np.generic):
            # Numpy scalar type -> Python scalar
            return value.item()
        elif isinstance(value, (int, float, str, bool, type(None))):
            # Already Python native type
            return value
        else:
            # Fallback: convert to string for other types
            return str(value)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Get a single sample from the dataset. All values are converted to Python native types.
        
        Args:
            idx: Index of the sample to return
            
        Returns:
            dict: Dictionary containing:
                - 'sparse_features' or 'embedding': torch.Tensor (key depends on feature_type)
                - 'modality': str, the modality ('ct' or 'mri')
                - 'image_id': str, the image identifier
                - 'slice_idx': int, the slice index
                - 'slice_orientation': str, the slice orientation ('x', 'y', or 'z')
                - Additional metadata columns if return_all_metadata=True
        """
        actual_idx = self.indices[idx]

        if self.feature_type == 'sparse_features':
            feature_key = 'sparse_features'
            feature_value = self.features[actual_idx]
        elif self.feature_type == 'embeddings':
            feature_key = 'embedding'
            # L2-normalise to match the behaviour of TotalSegmentatorBmPEmbeddedDataset
            feature_value = F.normalize(self.features[actual_idx], p=2, dim=0)
        else:
            raise ValueError(f"Invalid feature_type: {self.feature_type}")

        result = {
            feature_key: feature_value,
            'modality': self._convert_to_python_type(self.modalities[actual_idx]),
            'image_id': self._convert_to_python_type(self.image_ids[actual_idx]),
            'slice_idx': int(self.slice_indices[actual_idx]),
            'slice_orientation': self._convert_to_python_type(self.slice_orientations[actual_idx])
        }

        if self.additional_metadata:
            for col_name, col_data in self.metadata.items():
                if isinstance(col_data, np.ndarray):
                    # Extract single value from numpy array
                    value = col_data[actual_idx]
                    result[col_name] = self._convert_to_python_type(value)
                else:
                    # List or other type - convert element
                    result[col_name] = self._convert_to_python_type(col_data[actual_idx])
        
        return result


class TotalSegmentatorFeaturesDataModule(lightning.LightningDataModule):
    """
    Lightning DataModule for TotalSegmentator BiomedParse sparse features.
    
    Loads pre-computed sparse features from train/val/test parquet files.
    
    Args:
        train_parquet: Path to training parquet file
        val_parquet: Path to validation parquet file
        test_parquet: Path to test parquet file (optional)
        batch_size: Batch size for the DataLoader (default: 2048)
        num_workers: Number of worker processes for data loading (default: 0)
        shuffle_train: Whether to shuffle training data (default: True)
        pin_memory: Whether to use pinned memory for faster GPU transfer (default: True)
        additional_metadata: If not None, the additional metadata columns are loaded and included in the return dictionary.
    """
    
    def __init__(
        self,
        feature_type: str,
        train_parquet: str = "data/total_seg_sparse_features/total_seg_sparse_features_train.parquet",
        val_parquet: str = "data/total_seg_sparse_features/total_seg_sparse_features_val.parquet",
        test_parquet: str | None = "data/total_seg_sparse_features/total_seg_sparse_features_val.parquet",
        additional_metadata: list[str] | None = None,
        feature_ids: list[int] | None = None,
        batch_size: int = 2048,
        num_workers: int = 0,
        shuffle_train: bool = True,
        pin_memory: bool = True
    ):
        super().__init__()
        
        self.save_hyperparameters()
        
        self.feature_type = feature_type
        self.train_parquet = train_parquet
        self.val_parquet = val_parquet
        self.test_parquet = test_parquet
        self.additional_metadata = additional_metadata
        self.feature_ids = feature_ids
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.shuffle_train = shuffle_train
        self.pin_memory = pin_memory
    
    def setup(self, stage: str | None = None):
        """Setup datasets for each stage. Only creates datasets if they don't already exist to avoid reloading data to GPU on every trainer.fit() call."""
        if stage == 'fit' or stage is None:
            if not hasattr(self, 'train_dataset'):
                self.train_dataset = TotalSegmentatorLatentFeaturesDataset(
                    feature_type=self.feature_type,
                    parquet_path=self.train_parquet,
                    shuffle=self.shuffle_train,
                    feature_ids=self.feature_ids,
                    additional_metadata=self.additional_metadata
                )
            if not hasattr(self, 'val_dataset'):
                self.val_dataset = TotalSegmentatorLatentFeaturesDataset(
                    feature_type=self.feature_type,
                    parquet_path=self.val_parquet,
                    shuffle=False,
                    feature_ids=self.feature_ids,
                    additional_metadata=self.additional_metadata
                )
        
        if stage == 'test' and self.test_parquet is not None:
            if not hasattr(self, 'test_dataset'):
                self.test_dataset = TotalSegmentatorLatentFeaturesDataset(
                    feature_type=self.feature_type,
                    parquet_path=self.test_parquet,
                    shuffle=False,
                    feature_ids=self.feature_ids,
                    additional_metadata=self.additional_metadata
                )
    
    def _collate_fn(self, batch: list[dict]) -> dict:
        """
        Collate function to handle batch of dicts with sparse features/embeddings and metadata.
        
        Args:
            batch: List of dicts from dataset __getitem__
            
        Returns:
            dict: Batched data with:
                - 'sparse_features' or 'embedding': torch.Tensor of shape (batch_size, d_hidden) (key depends on feature_type)
                - 'image_id': List[str] of image identifiers
                - 'slice_idx': List[int] of slice indices
                - 'orientation': List[str] of slice orientations
                - 'modality': List[str] of modalities
                - Additional metadata keys if return_all_metadata=True (as lists)
        """
        if self.feature_type == 'sparse_features':
            feature_key = 'sparse_features'
        elif self.feature_type == 'embeddings':
            feature_key = 'embedding'
        else:
            raise ValueError(f"Invalid feature_type: {self.feature_type}")

        result = {
            feature_key: torch.stack([item[feature_key] for item in batch]),
            'modality': [item['modality'] for item in batch],
            'image_id': [item['image_id'] for item in batch],
            'slice_idx': [item['slice_idx'] for item in batch],
            'slice_orientation': [item['slice_orientation'] for item in batch]
        }

        if self.additional_metadata:
            first_item_keys = set(batch[0].keys())
            core_keys = {feature_key, 'modality', 'image_id', 'slice_idx', 'slice_orientation'}
            additional_keys = first_item_keys - core_keys
            for key in additional_keys:
                result[key] = [item[key] for item in batch]        
        return result
    
    def train_dataloader(self):
        return torch.utils.data.DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,  # Dataset shuffles indices once at initialization
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )
    
    def val_dataloader(self):
        return torch.utils.data.DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )

    def test_dataloader(self):
        return torch.utils.data.DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            pin_memory=self.pin_memory and torch.cuda.is_available(),
            persistent_workers=self.num_workers > 0,
        )


# ==================================================
# Feature Selection Utilities
# ==================================================

def load_interpretable_features(
    base_model: str,
    sae_config: str,
    top_n: int,
    always_active_features: set,
    interpretability_root: Path,
    linear_probes_root: Path,
) -> pd.DataFrame:
    """
    Select the top-N features ranked by combined monosemanticity and task importance.

    Combines two dimensions:
      1. Monosemanticity: monosemanticity_50 score from results/monosemanticity/
      2. Task importance: average rank across logistic regression coefficients
         from results/linear_probes/

    Features are ranked by the average of their per-dimension ranks (lower = better).

    Args:
        base_model: Base model name (e.g., "biomedparse", "dinov3")
        sae_config: SAE configuration (e.g., "D_128_512_2048_8192_K_20_40_80_160")
        top_n: Number of top features to return
        always_active_features: Feature indices to exclude (always active, not interpretable)
        interpretability_root: Root of monosemanticity results
                               (expects .../valid/{base_model}/{sae_config}/feature_scores.csv)
        linear_probes_root: Root of linear probe results
                            (expects .../{base_model}/{sae_config}/*_coefficients_sparse.csv)

    Returns:
        DataFrame with top-N rows sorted by combined_rank, columns:
        feature_idx, monosem_rank, monosemanticity_50, coeff_rank, avg_coeff_rank, combined_rank
    """
    print(f"\nSelecting top-{top_n} features by combined monosemanticity + task importance...")

    # -- Monosemanticity dimension ------------------------------------------------
    scores_path = interpretability_root / "valid" / base_model / sae_config / "feature_scores.csv"
    if not scores_path.exists():
        raise FileNotFoundError(f"Monosemanticity scores not found: {scores_path}")
    monosem_df = cast(pd.DataFrame, pd.read_csv(scores_path)[["feature_idx", "monosemanticity_50", "n_nonzero_activations"]])
    monosem_df = monosem_df.sort_values("monosemanticity_50", ascending=False).reset_index(drop=True)
    monosem_df["monosem_rank"] = monosem_df.index + 1
    print(f"  Monosemanticity: {len(monosem_df)} features loaded from {scores_path}")

    # -- Task importance dimension (logistic regression coefficients) -------------
    coeff_dir = linear_probes_root / base_model / sae_config
    coeff_files = list(coeff_dir.glob("*_coefficients_sparse.csv"))
    if not coeff_files:
        raise FileNotFoundError(f"No coefficient files found in {coeff_dir}")
    print(f"  Task importance: {len(coeff_files)} task files loaded from {coeff_dir}")

    all_ranks: dict = {}
    for coeff_file in coeff_files:
        coeff_df = pd.read_csv(coeff_file)
        for _, row in coeff_df.iterrows():
            feat_idx = int(row["feature_idx"])
            all_ranks.setdefault(feat_idx, []).append(int(row["rank"]))

    coeff_summary = pd.DataFrame([
        {"feature_idx": feat_idx, "avg_coeff_rank": float(np.mean(ranks))}
        for feat_idx, ranks in all_ranks.items()
    ])
    coeff_summary = coeff_summary.sort_values("avg_coeff_rank").reset_index(drop=True)
    coeff_summary["coeff_rank"] = coeff_summary.index + 1

    # -- Combine ------------------------------------------------------------------
    combined_df = cast(pd.DataFrame, monosem_df[["feature_idx", "monosem_rank", "monosemanticity_50"]]).merge(
        cast(pd.DataFrame, coeff_summary[["feature_idx", "coeff_rank", "avg_coeff_rank"]]),
        on="feature_idx",
        how="outer",
    )
    max_rank = int(max(
        combined_df["monosem_rank"].max(skipna=True),
        combined_df["coeff_rank"].max(skipna=True),
    ))
    combined_df["monosem_rank"] = combined_df["monosem_rank"].fillna(max_rank + 1)
    combined_df["coeff_rank"]   = combined_df["coeff_rank"].fillna(max_rank + 1)
    combined_df["combined_rank"] = (combined_df["monosem_rank"] + combined_df["coeff_rank"]) / 2
    combined_df = combined_df.sort_values("combined_rank").reset_index(drop=True)

    # -- Exclude always-active and take top-N -------------------------------------
    combined_df = cast(pd.DataFrame, combined_df[~combined_df["feature_idx"].isin(list(always_active_features))])
    selected_df = cast(pd.DataFrame, combined_df.head(top_n).copy())

    print(f"  [ok] Selected {len(selected_df)} features")
    combined_rank_series = cast(pd.Series, selected_df['combined_rank'])
    print(f"    Combined rank range: {combined_rank_series.iloc[0]:.1f} (best) to "
          f"{combined_rank_series.iloc[-1]:.1f} (worst in top-{top_n})")

    return selected_df