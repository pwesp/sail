#!/usr/bin/env python3
"""
Process TotalSegmentator CT dataset into 2D slices for foundation model encoding.

For each CT scan, applies HU windowing (study-type-specific), extracts axial,
coronal, and sagittal slices, resizes them to 1024x1024 RGB (required by BiomedParse),
and records per-slice organ presence from the TotalSegmentator segmentation masks.

Inputs:  TotalSegmentator/CT_Dataset_v201/
Outputs: data/total_seg_ct/  (PNG slices + metadata.csv)

Run with: conda activate sail && python process_total_segmentator_ct_dataset.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from concurrent.futures import ProcessPoolExecutor, as_completed
from nibabel.nifti1 import Nifti1Image
from nibabel.loadsave import load as nib_load
from nibabel.orientations import aff2axcodes
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image
from scipy.ndimage import zoom
from skimage.exposure import rescale_intensity
from tqdm import tqdm
from typing import Any, cast

# Local imports
from src.preprocessing import get_organ_list

TOTALSEG_WINDOW_SETTINGS = {
    # Multi-region body scans
    'ct thorax-abdomen-pelvis':        {'center': -100, 'width': 1400},
    'ct neck-thorax-abdomen-pelvis':   {'center': -100, 'width': 1400},
    'ct abdomen-pelvis':               {'center': 50,   'width': 1000},
    'ct thorax-neck':                  {'center': -200, 'width': 1400},
    'ct polytrauma':                   {'center': -50,  'width': 1400},
    'ct thorax-abdomen':               {'center': -100, 'width': 1400},
    'ct whole body':                   {'center': -50,  'width': 1400},
    
    # Pelvis
    'ct pelvis':                       {'center': 80,   'width': 1400},
    
    # Thorax / Chest
    'ct thorax-chest':                 {'center': -200, 'width': 1400},
    'ct thorax':                       {'center': -200, 'width': 1400},
    'ct chest':                        {'center': -200, 'width': 1400},
    
    # Abdomen
    'ct abdomen':                      {'center': 50,   'width': 1000},
    
    # Neck
    'ct neck':                         {'center': 50,   'width': 1000},
    
    # Head / Brain
    'ct head':                         {'center': 40,   'width': 400},
    'ct polytrauma head':              {'center': 40,   'width': 400},
    'ct orbita':                       {'center': 60,   'width': 1000},
    'ct headbasis-felsenleg':          {'center': 120,  'width': 1200},
    'ct gesichtshead':                 {'center': 120,  'width': 1200},
    
    # CTA
    'ct angiography neck-thx-abd-pelvis-leg': {'center': 120, 'width': 1000},
    'ct angiography head':                     {'center': 100, 'width': 900},
    'ct angiography pelvis-leg':               {'center': 120, 'width': 1000},
    'ct angiography thorax-abdomen-pelvis':    {'center': 120, 'width': 1000},
    'ct angiography abdomen-pelvis-leg':       {'center': 120, 'width': 1000},
    'ct angiography abdomen-pelvis':           {'center': 120, 'width': 1000},
    'ct angiography thorax':                   {'center': 100, 'width': 1000},
    'ct angiography neck':                     {'center': 100, 'width': 900},
    
    # Cardiac
    'ct heart-thorakale aorta':        {'center': 120,  'width': 1000},
    'ct aortic valve':                 {'center': 120,  'width': 1000},
    'ct heart':                        {'center': 120,  'width': 1000},
    'ct left artrium':                 {'center': 120,  'width': 1000},
    'ct coronary arteries':            {'center': 120,  'width': 1000},
    
    # Spine
    'ct spine':                        {'center': 200,  'width': 2000},
    
    # Extremities
    'ct upper leg left':               {'center': 200,  'width': 2000},
    'ct hip right':                    {'center': 200,  'width': 2000},
    'ct upper limb both':              {'center': 200,  'width': 2000},
    'ct lower limb left':              {'center': 200,  'width': 2000},
    
    # Interventional
    'ct intervention':                {'center': 50,   'width': 1000},
    'ct operation':                   {'center': 50,   'width': 1000},
}


def get_window_for_study_type(study_type: str) -> tuple:
    """
    Get HU window center and width depending on the study type of the CT scan.
    
    Args:
        study_type: String describing the study type
        
    Returns:
        (center, width) tuple in Hounsfield Units
    """
    # Normalize: strip, lowercase, and collapse multiple spaces to single space
    study_type_clean = ' '.join(study_type.strip().lower().split())
    
    if study_type_clean not in TOTALSEG_WINDOW_SETTINGS:
        error_msg = (
            f"\n{'='*80}\n"
            f"ERROR: Unknown study type!\n"
            f"{'='*80}\n"
            f"  Original study type:     '{study_type}'\n"
            f"  Normalized study type:   '{study_type_clean}'\n"
            f"\n"
        ) 
        raise ValueError(error_msg)
    
    params = TOTALSEG_WINDOW_SETTINGS[study_type_clean]
    return params['center'], params['width']


def process_ct_scan_to_slices(
    totalseg_single_ct_scan_dir: Path,
    df_ct_metadata: pd.DataFrame,
    output_base_dir: Path,
    organs: list[str],
    target_size: tuple[int, int] = (1024, 1024),
    verbose: bool = True,
) -> dict[str, list]:
    """
    Process a CT scan: resize slices and compute organ presence per slice.
    
    Args:
        totalseg_single_ct_scan_dir: Directory containing ct.nii.gz and segmentations
        df_ct_metadata: DataFrame containing CT metadata
        organs: List of organ names (without .nii.gz extension)
        output_base_dir: Base directory for saving slices
        target_size: Target (width, height) for resized slices
        verbose: Whether to print progress
    
    Returns:
        Dict with image_id, slice_idx, slice_orientation, image_path, and organ presence vectors
    """
    ct_file = totalseg_single_ct_scan_dir / "ct.nii.gz"
    seg_dir = totalseg_single_ct_scan_dir / "segmentations"
    image_id = totalseg_single_ct_scan_dir.name
 
    if verbose:
        print(f"Processing CT '{image_id}' in file '{ct_file}'")
    
    # Load CT scan
    ct_nib = nib_load(ct_file)
    assert isinstance(ct_nib, Nifti1Image)
    ct_img = ct_nib.get_fdata()
    affine = ct_nib.affine
    
    # Get voxel spacing (in mm)
    voxel_spacing = ct_nib.header.get_zooms()[:3]  # (x, y, z) spacing in mm
    spacing_x, spacing_y, spacing_z = cast(tuple[float, float, float], voxel_spacing)

    # Check if voxel spacing is isotropic (within tolerance for floating-point comparison)
    tolerance = 1e-6  # 1 micrometer tolerance
    if not (np.isclose(spacing_x, spacing_y, atol=tolerance) and np.isclose(spacing_y, spacing_z, atol=tolerance)):
        raise ValueError(f"Voxel spacing is not isotropic. Voxel spacing: {spacing_x:.8f} x {spacing_y:.8f} x {spacing_z:.8f} mm")

    # Assess orientation from nib file
    original_orientation = aff2axcodes(affine)

    # Get study type
    meta_row = df_ct_metadata.loc[df_ct_metadata['image_id'] == image_id]
    study_type = meta_row['study_type'].values[0]

    if verbose:
        print(f"  Original shape: {ct_img.shape}")
        print(f"  Original orientation: {''.join(original_orientation)}")
        print(f"  Voxel spacing: {spacing_x:.3f} x {spacing_y:.3f} x {spacing_z:.3f} mm")
        print(f"  Physical volume per voxel: {spacing_x * spacing_y * spacing_z:.3f} mm^3")
        print(f"  Study type: {study_type:s}")
    
    # Window CT
    center, width = get_window_for_study_type(study_type)
    if verbose:
        print("  Windowing CT...")
        print(f"    Center/Window = {center}/{width} [HU]")

    lower = center - (width / 2)
    upper = center + (width / 2)
    ct_img = np.clip(ct_img, lower, upper)

    if verbose:
        print("  Sanity check: CT image values are within bounds...")
        print(f"    Min: {ct_img.min()} [HU]")
        print(f"    Max: {ct_img.max()} [HU]")

    ct_img = rescale_intensity(ct_img, out_range=cast(Any, (0, 255))).astype(np.uint8)
    
    # Create output directory for this CT scan
    output_dir = output_base_dir / image_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get dimensions
    x_dim, y_dim, z_dim = ct_img.shape
    
    # --------------------------------------------------
    # Calculate central crop indices for each view
    # --------------------------------------------------

    # Sagittal slices (x-direction): ct_img[x, :, :] -> shape (y_dim, z_dim)
    sagittal_square_size = min(y_dim, z_dim)
    sagittal_y_start = (y_dim - sagittal_square_size) // 2
    sagittal_y_end = sagittal_y_start + sagittal_square_size
    sagittal_z_start = (z_dim - sagittal_square_size) // 2
    sagittal_z_end = sagittal_z_start + sagittal_square_size
    
    # Coronal slices (y-direction): ct_img[:, y, :] -> shape (x_dim, z_dim)
    coronal_square_size = min(x_dim, z_dim)
    coronal_x_start = (x_dim - coronal_square_size) // 2
    coronal_x_end = coronal_x_start + coronal_square_size
    coronal_z_start = (z_dim - coronal_square_size) // 2
    coronal_z_end = coronal_z_start + coronal_square_size

    # Axial slices (z-direction): ct_img[:, :, z] -> shape (x_dim, y_dim)
    axial_square_size = min(x_dim, y_dim)
    axial_x_start = (x_dim - axial_square_size) // 2
    axial_x_end = axial_x_start + axial_square_size
    axial_y_start = (y_dim - axial_square_size) // 2
    axial_y_end = axial_y_start + axial_square_size
    
    if verbose:
        print(f"  Crop settings:")
        print(f"    Sagittal (x-slices): {sagittal_square_size}x{sagittal_square_size} "
              f"[y: {sagittal_y_start}:{sagittal_y_end}, z: {sagittal_z_start}:{sagittal_z_end}]")
        print(f"    Coronal (y-slices): {coronal_square_size}x{coronal_square_size} "
              f"[x: {coronal_x_start}:{coronal_x_end}, z: {coronal_z_start}:{coronal_z_end}]")
        print(f"    Axial (z-slices): {axial_square_size}x{axial_square_size} "
              f"[x: {axial_x_start}:{axial_x_end}, y: {axial_y_start}:{axial_y_end}]")

    # --------------------------------------------------
    # Process and save each CT slice
    # --------------------------------------------------

    # X-slices (sagittal)
    x_slice_paths: list[Path] = []
    for x_idx in range(x_dim):
        # Get square x-slice (sagittal)
        slice_2d = ct_img[x_idx, sagittal_y_start:sagittal_y_end, sagittal_z_start:sagittal_z_end]
        
        # Convert to RGB image and resize
        slice_2d = Image.fromarray(slice_2d)
        slice_2d = slice_2d.convert('RGB')
        slice_2d = slice_2d.resize(target_size, Image.Resampling.LANCZOS)
        
        # Apply transformation to get the correct orientation
        slice_2d = slice_2d.rotate(90, expand=True)
        slice_2d = slice_2d.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        
        # Save slice
        output_path = output_dir / f"x_slice_{x_idx:04d}.png"
        slice_2d.save(output_path)
        x_slice_paths.append(output_path)
    
    # Y-slices (coronal)
    y_slice_paths: list[Path] = []
    for y_idx in range(y_dim):
        # Get square y-slice (coronal)
        slice_2d = ct_img[coronal_x_start:coronal_x_end, y_idx, coronal_z_start:coronal_z_end]
        
        # Convert to RGB image and resize
        slice_2d = Image.fromarray(slice_2d)
        slice_2d = slice_2d.convert('RGB')
        slice_2d = slice_2d.resize(target_size, Image.Resampling.LANCZOS)
        
        # Apply transformation to get the correct orientation
        slice_2d = slice_2d.rotate(90, expand=True)
        slice_2d = slice_2d.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        
        # Save slice
        output_path = output_dir / f"y_slice_{y_idx:04d}.png"
        slice_2d.save(output_path)
        y_slice_paths.append(output_path)
    
    # Z-slices (axial)
    z_slice_paths: list[Path] = []
    for z_idx in range(z_dim):
        # Get square z-slice (axial)
        slice_2d = ct_img[axial_x_start:axial_x_end, axial_y_start:axial_y_end, z_idx]
        
        # Convert to RGB image and resize
        slice_2d = Image.fromarray(slice_2d)
        slice_2d = slice_2d.convert('RGB')
        slice_2d = slice_2d.resize(target_size, Image.Resampling.LANCZOS)
        
        # Apply transformation to get the correct orientation
        slice_2d = slice_2d.rotate(90, expand=True)
        
        # Save slice
        output_path = output_dir / f"z_slice_{z_idx:04d}.png"
        slice_2d.save(output_path)
        z_slice_paths.append(output_path)
    
    if verbose:
        print(f"  Saved {x_dim} x-slices, {y_dim} y-slices, {z_dim} z-slices to {output_dir}")

    # Initialize result dictionary
    result: dict[str, list] = {
        "image_id": [image_id] * (x_dim + y_dim + z_dim),
        "slice_idx": list(range(x_dim)) + list(range(y_dim)) + list(range(z_dim)),
        "slice_orientation": ["x"] * x_dim + ["y"] * y_dim + ["z"] * z_dim,
        "image_path": [str(p) for p in x_slice_paths] + [str(p) for p in y_slice_paths] + [str(p) for p in z_slice_paths]
    }

    # --------------------------------------------------
    # Process organ segmentations
    # --------------------------------------------------

    # Track if we've warned about dimension mismatch
    dimension_mismatch_warned = False

    for organ in organs:
        seg_file = seg_dir / f"{organ}.nii.gz"
        
        # Load segmentation mask
        seg_mask_nib = nib_load(seg_file)
        assert isinstance(seg_mask_nib, Nifti1Image)
        mask_data = seg_mask_nib.get_fdata()

        if verbose:
            print(f"  Processing organ: {organ:s}")
            print(f"  Segmentation mask file: {seg_file}")
            print(f"  Segmentation mask shape: {mask_data.shape}")
        
        # Check if mask dimensions match CT dimensions
        if mask_data.shape != (x_dim, y_dim, z_dim):
            # Warn once per CT scan
            if not dimension_mismatch_warned and verbose:
                print(f"  WARNING: Segmentation mask shape {mask_data.shape} differs from CT shape {(x_dim, y_dim, z_dim)}")
                print(f"           Resizing masks to match CT dimensions...")
                dimension_mismatch_warned = True
            
            # Resize mask to match CT dimensions
            zoom_factors = (
                x_dim / mask_data.shape[0],
                y_dim / mask_data.shape[1],
                z_dim / mask_data.shape[2]
            )
            mask_data = zoom(mask_data, zoom_factors, order=0, mode='nearest')
        
        # Convert to boolean for memory efficiency and faster .any() operations
        mask_data = mask_data.astype(bool)
        
        # No need to resize to target_size - we only care about presence
        # X-direction (sagittal): Check presence in cropped region
        mask_cropped_x = mask_data[:, sagittal_y_start:sagittal_y_end, sagittal_z_start:sagittal_z_end]
        x_presence = [int(mask_cropped_x[x_idx].any()) for x_idx in range(x_dim)]
        
        # Y-direction (coronal): Check presence in cropped region
        mask_cropped_y = mask_data[coronal_x_start:coronal_x_end, :, coronal_z_start:coronal_z_end]
        y_presence = [int(mask_cropped_y[:, y_idx, :].any()) for y_idx in range(y_dim)]
        
        # Z-direction (axial): Check presence in cropped region
        mask_cropped_z = mask_data[axial_x_start:axial_x_end, axial_y_start:axial_y_end, :]
        z_presence = [int(mask_cropped_z[:, :, z_idx].any()) for z_idx in range(z_dim)]

        # Combine presence vectors
        result[organ] = x_presence + y_presence + z_presence
        
        if verbose:
            print(f"  {organ}: {sum(x_presence)}/{x_dim} x-slices, "
                  f"{sum(y_presence)}/{y_dim} y-slices, "
                  f"{sum(z_presence)}/{z_dim} z-slices with organ presence")
    
    return result

def process_single_ct_wrapper(args):
    """
    Wrapper function for multiprocessing that unpacks arguments.
    
    Args:
        args: Tuple of (ct_dir, df_ct_metadata, organs, output_base_dir)
    
    Returns:
        Result dictionary from process_ct_scan_to_slices or None if error
    """
    ct_dir, df_ct_metadata, organs, output_base_dir = args
    
    try:
        result = process_ct_scan_to_slices(
            totalseg_single_ct_scan_dir=ct_dir,
            df_ct_metadata=df_ct_metadata,
            organs=organs,
            output_base_dir=output_base_dir,
            verbose=False  # Disable verbose to avoid messy output with multiprocessing
        )
        return result
    except Exception as e:
        print(f"ERROR processing {ct_dir.name}: {e}")
        return None

def main():
    # Directories and files
    CT_DIR = "TotalSegmentator/CT_Dataset_v201"
    REFERENCE_SEG_DIR = "TotalSegmentator/CT_Dataset_v201/s0000/segmentations"
    CT_META_CSV = "TotalSegmentator/CT_Dataset_v201/meta.csv"
    OUTPUT_BASE_DIR = "data/total_seg_ct"
    NUM_WORKERS = 32

    print("\n"+"="*50)
    print("PROCESSING TOTAL SEGMENTATOR CT DATASET")
    print("="*50+"\n")

    # Convert to Path objects
    ct_dir = Path(CT_DIR)
    reference_seg_dir = Path(REFERENCE_SEG_DIR)
    ct_meta_csv = Path(CT_META_CSV)
    output_base_dir = Path(OUTPUT_BASE_DIR)
    
    print(f"Original CT directory:            {ct_dir}")
    print(f"Reference segmentation directory: {reference_seg_dir}")
    print()

    # Create output directory
    print(f"Creating output directory: {output_base_dir}\n")
    output_base_dir.mkdir(parents=True, exist_ok=True)

    # Get all CT scan directories
    print("Searching for all CT scan directories...")
    ct_sample_dirs = [x for x in ct_dir.iterdir() if x.is_dir()]
    ct_sample_dirs.sort()
    print(f"Found {len(ct_sample_dirs):d} CT scan directories")
    print()

    # Get organ list
    print("Searching for all organs in the reference segmentation directory...")
    organs = get_organ_list(reference_seg_dir)
    print(f"Found {len(organs):d} organs")
    print()

    # Load metadata
    print("Loading CT metadata...")
    df_ct_metadata = pd.read_csv(ct_meta_csv, sep=";")
    print(f"Loaded {len(df_ct_metadata):d} metadata rows")
    print()
    
    # Process all CT scans with multiprocessing
    print("-"*50)
    print(f"Processing {len(ct_sample_dirs)} CT scans with {NUM_WORKERS} workers")
    print("-"*50+"\n")
    
    # Prepare arguments for each worker
    processing_args = [
        (ct_scan_dir, df_ct_metadata, organs, output_base_dir)
        for ct_scan_dir in ct_sample_dirs
    ]
    
    # Process with progress bar
    processing_results = []
    
    # Process with progress bar
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_ct_wrapper, args): args[0].name
            for args in processing_args
        }
        
        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing CTs"):
            ct_name = futures[future]
            try:
                result = future.result()
                if result is not None:
                    processing_results.append(result)
            except Exception as e:
                tqdm.write(f"ERROR with {ct_name}: {e}")

    print(f"\n[ok] Successfully processed {len(processing_results)}/{len(ct_sample_dirs)} CT scans")
   
    # Combine all results into a single dataframe
    print("\nCombining all results into a single dataframe...")
    df_ct_slices = pd.concat([pd.DataFrame(result) for result in processing_results], ignore_index=True)
    
    # Load and merge with metadata
    tot_seg_meta_csv = Path("TotalSegmentator/CT_Dataset_v201/meta.csv")
    df_tot_seg_meta = pd.read_csv(tot_seg_meta_csv, sep=";")
    df_ct_slices = df_ct_slices.merge(df_tot_seg_meta, on="image_id", how="left")
    print(f"  Merged with metadata from {tot_seg_meta_csv}")
    
    # Reorder columns: image_id, slice_idx, image_path, then metadata, then organs
    meta_cols = ["image_id", "slice_idx", "slice_orientation", "image_path"]
    tot_seg_meta_cols = cast(list[str], df_tot_seg_meta.columns.tolist()[1:])  # exclude image_id
    organ_cols = [col for col in organs if col in df_ct_slices.columns]

    # Reorder: meta cols, tot_seg_meta cols, then organ cols
    column_order = meta_cols + tot_seg_meta_cols + organ_cols
    df_ct_slices = df_ct_slices[column_order]

    # Save DataFrame
    output_csv = output_base_dir / "metadata.csv"
    df_ct_slices.to_csv(output_csv, index=False)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Results saved to:           {output_csv}")
    print(f"  Processed CT scans:         {len(processing_results)}")
    print(f"  Total slices (x, y, and z): {len(df_ct_slices)}")
    print(f"  Metadata columns:           {len(df_ct_slices.columns)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()