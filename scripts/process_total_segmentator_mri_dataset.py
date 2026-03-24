#!/usr/bin/env python3
"""
Process TotalSegmentator MRI dataset into 2D slices for foundation model encoding.

For each MRI scan, applies N4 bias field correction, normalises intensity,
extracts slices along the scan's native orientation, resizes them to 1024x1024 RGB
(required by BiomedParse), and records per-slice organ presence from the
TotalSegmentator segmentation masks.

Inputs:  TotalSegmentator/MRI_Dataset_v200/
Outputs: data/total_seg_mri/  (PNG slices + metadata.csv)

Run with: conda activate sail && python process_total_segmentator_mri_dataset.py
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
import SimpleITK as sitk
from skimage.exposure import rescale_intensity
from tqdm import tqdm
from typing import Any, cast

# Local imports
from src.preprocessing import get_organ_list

def apply_n4_bias_correction(
    mri_img: np.ndarray,
    mask: np.ndarray | None = None,
    verbose: bool = True
) -> np.ndarray:
    """
    Apply N4 bias field correction to MRI image.
    
    Args:
        mri_img: 3D MRI array
        mask: Optional binary mask (1 = tissue, 0 = background)
        max_threads: Maximum number of threads for N4 correction
        verbose: Whether to print progress
    
    Returns:
        Bias-corrected MRI image as numpy array
    """
    if verbose:
        print("  Applying N4 bias field correction...")
    
    try:

        # Convert numpy to SimpleITK
        sitk_img = sitk.GetImageFromArray(mri_img.astype(np.float32))
        
        # Create mask if not provided (simple Otsu thresholding)
        if mask is None:
            # Create mask: exclude background (assume background is low intensity)
            mask = (mri_img > np.percentile(mri_img, 5)).astype(np.uint8)
        
        sitk_mask = sitk.GetImageFromArray(mask.astype(np.uint8))
        
        # Initialize N4 bias correction
        corrector = sitk.N4BiasFieldCorrectionImageFilter()
        corrector.SetMaximumNumberOfIterations([50, 50, 50, 50])  # 4 resolution levels
        corrector.SetConvergenceThreshold(0.001)
        
        # Apply correction
        corrected_img = corrector.Execute(sitk_img, sitk_mask)
        corrected_array = sitk.GetArrayFromImage(corrected_img)
        
        if verbose:
            print("    [ok] Bias correction successful")
        
        return corrected_array
    
    except Exception as e:
        if verbose:
            print(f"    WARNING: Bias correction failed: {e}")
            print("    Continuing without correction...")
        return mri_img


def process_mri_scan_to_slices(
    totalseg_single_mri_scan_dir: Path,
    output_base_dir: Path,
    organs: list[str],
    target_size: tuple[int, int] = (1024, 1024),
    apply_bias_correction: bool = True,
    verbose: bool = True,
) -> dict[str, list]:
    """
    Process a MRI scan: resize slices and compute organ presence per slice.
    
    Args:
        totalseg_single_mri_scan_dir: Directory containing mri.nii.gz and segmentations
        output_base_dir: Base directory for saving slices
        organs: List of organ names (without .nii.gz extension)
        target_size: Target (width, height) for resized slices
        apply_bias_correction: Whether to apply N4 bias field correction
        verbose: Whether to print progress
    
    Returns:
        Dict with image_id, slice_idx, slice_orientation, image_path, and organ presence vectors
    """
    mri_file = totalseg_single_mri_scan_dir / "mri.nii.gz"
    seg_dir = totalseg_single_mri_scan_dir / "segmentations"
    image_id = totalseg_single_mri_scan_dir.name
    
    # Store resampling parameters if needed
    resampler_for_masks = None
    
    if verbose:
        print(f"Processing MRI '{image_id}' in file '{mri_file}'")
    
    # Load MRI scan
    mri_nib = nib_load(mri_file)
    assert isinstance(mri_nib, Nifti1Image)
    mri_img = mri_nib.get_fdata()
    affine = mri_nib.affine

    # Get voxel spacing (in mm)
    voxel_spacing = mri_nib.header.get_zooms()[:3]  # (x, y, z) spacing in mm
    spacing_x, spacing_y, spacing_z = cast(tuple[float, float, float], voxel_spacing)
    
    # Store original spacing as list for SimpleITK (needed for mask resampling)
    original_spacing = [float(spacing_x), float(spacing_y), float(spacing_z)]
    
    # Assess orientation from nib file
    original_orientation = aff2axcodes(affine)

    # Get dimensions
    x_dim, y_dim, z_dim = mri_img.shape

    # Find the shortest dimension (assumed to be the high-resolution slice direction)
    dims = {'x': x_dim, 'y': y_dim, 'z': z_dim}
    spacings = {'x': spacing_x, 'y': spacing_y, 'z': spacing_z}
    shortest_dim_name = min(dims, key=lambda k: dims[k])
    shortest_dim_size = dims[shortest_dim_name]
    
    # Get the two in-plane dimensions (perpendicular to slice direction)
    if shortest_dim_name == 'x':
        inplane_dims = ['y', 'z']
    elif shortest_dim_name == 'y':
        inplane_dims = ['x', 'z']
    elif shortest_dim_name == 'z':
        inplane_dims = ['x', 'y']
    else:
        raise ValueError(f"Invalid shortest dimension: {shortest_dim_name}")
    
    inplane_spacing_1 = spacings[inplane_dims[0]]
    inplane_spacing_2 = spacings[inplane_dims[1]]
    
    # Check if in-plane voxel spacing is isotropic
    tolerance = 1e-3  # 1 micrometer tolerance
    if not np.isclose(inplane_spacing_1, inplane_spacing_2, atol=tolerance):
        print(f"In-plane voxel spacing is not isotropic for high-res plane {shortest_dim_name.upper()}")
        print(f"Spacings: {inplane_dims[0]}={inplane_spacing_1:.8f} mm, {inplane_dims[1]}={inplane_spacing_2:.8f} mm")

        # --------------------------------------------------
        # Resample MRI to isotropic in-plane spacing
        # --------------------------------------------------

        print("Start resampling...")
        
        # Use the smaller spacing to maintain resolution
        target_inplane_spacing = min(inplane_spacing_1, inplane_spacing_2)
        
        # Create new spacing list with isotropic in-plane spacing (convert to list for SimpleITK)
        if shortest_dim_name == 'x':
            new_spacing = [float(spacings['x']), float(target_inplane_spacing), float(target_inplane_spacing)]
        elif shortest_dim_name == 'y':
            new_spacing = [float(target_inplane_spacing), float(spacings['y']), float(target_inplane_spacing)]
        else:  # shortest_dim_name == 'z'
            new_spacing = [float(target_inplane_spacing), float(target_inplane_spacing), float(spacings['z'])]
        
        print(f"  Target isotropic in-plane spacing: {target_inplane_spacing:.8f} mm")
        print(f"  New spacing: x={new_spacing[0]:.8f}, y={new_spacing[1]:.8f}, z={new_spacing[2]:.8f} mm")
        
        # Convert MRI to SimpleITK image
        sitk_mri = sitk.GetImageFromArray(mri_img.astype(np.float32).transpose(2, 1, 0))  # numpy (x,y,z) -> ITK (z,y,x)
        # Set original spacing
        sitk_mri.SetSpacing(original_spacing)
        
        # Create resampler for MRI
        resampler = sitk.ResampleImageFilter()
        resampler.SetInterpolator(sitk.sitkLinear)
        resampler.SetOutputSpacing(new_spacing)
        
        # Calculate new size based on spacing change
        original_size = sitk_mri.GetSize()
        new_size = [
            int(round(original_size[i] * original_spacing[i] / new_spacing[i]))
            for i in range(3)
        ]
        resampler.SetSize(new_size)
        resampler.SetOutputDirection(sitk_mri.GetDirection())
        resampler.SetOutputOrigin(sitk_mri.GetOrigin())
        resampler.SetDefaultPixelValue(0)
        
        # Resample MRI
        print(f"  Resampling MRI from {original_size} to {new_size}...")
        sitk_mri_resampled = resampler.Execute(sitk_mri)
        mri_img = sitk.GetArrayFromImage(sitk_mri_resampled).transpose(2, 1, 0)  # ITK (z,y,x) -> numpy (x,y,z)
        
        # Create a separate resampler for segmentation masks (using nearest neighbor interpolation)
        resampler_for_masks = sitk.ResampleImageFilter()
        resampler_for_masks.SetInterpolator(sitk.sitkNearestNeighbor)  # Use nearest neighbor for masks
        resampler_for_masks.SetOutputSpacing(new_spacing)
        resampler_for_masks.SetSize(new_size)
        resampler_for_masks.SetOutputDirection(sitk_mri.GetDirection())
        resampler_for_masks.SetOutputOrigin(sitk_mri.GetOrigin())
        resampler_for_masks.SetDefaultPixelValue(0)
        
        # Update voxel spacing and dimensions
        spacing_x, spacing_y, spacing_z = new_spacing
        x_dim, y_dim, z_dim = mri_img.shape
        
        print(f"  Resampling complete. New shape: {mri_img.shape}")
        print(f"  New voxel spacing: {spacing_x:.8f} x {spacing_y:.8f} x {spacing_z:.8f} mm")
        
        # Update dimensions dictionary for subsequent processing
        dims = {'x': x_dim, 'y': y_dim, 'z': z_dim}
        spacings = {'x': spacing_x, 'y': spacing_y, 'z': spacing_z}
        
        # Verify in-plane spacing is now isotropic
        inplane_spacing_1 = spacings[inplane_dims[0]]
        inplane_spacing_2 = spacings[inplane_dims[1]]
        if np.isclose(inplane_spacing_1, inplane_spacing_2, atol=tolerance):
            print(f"  [ok] In-plane spacing is now isotropic: {inplane_spacing_1:.8f} mm")
        
        
    if verbose:
        print(f"  Original shape: {mri_img.shape}")
        print(f"  Original orientation: {''.join(original_orientation)}")
        print(f"  Voxel spacing: {spacing_x:.3f} x {spacing_y:.3f} x {spacing_z:.3f} mm")
        print(f"  Physical volume per voxel: {spacing_x * spacing_y * spacing_z:.3f} mm^3")
        print(f"  High-resolution plane: {shortest_dim_name.upper()}-direction ({shortest_dim_size} slices)")
        print(f"  In-plane spacing ({inplane_dims[0]}, {inplane_dims[1]}): {inplane_spacing_1:.3f} x {inplane_spacing_2:.3f} mm [ok]")

    # Apply bias field correction BEFORE any other processing
    if apply_bias_correction:
        mri_img = apply_n4_bias_correction(mri_img, mask=None, verbose=verbose)

    # Clip to percentiles to handle outliers (MRI scans often have outliers)
    p_low, p_high = np.percentile(mri_img[mri_img > 0], [1, 99])  # Exclude background (0 values)
    mri_img = np.clip(mri_img, p_low, p_high)

    # Normalize to 0-255
    mri_img = rescale_intensity(mri_img, out_range=cast(Any, (0, 255))).astype(np.uint8)
    
    # Create output directory for this scan
    output_dir = output_base_dir / image_id
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------
    # Calculate central crop indices for each view
    # --------------------------------------------------

    # Sagittal slices (x-direction): mri_img[x, :, :] -> shape (y_dim, z_dim)
    sagittal_square_size = min(y_dim, z_dim)
    sagittal_y_start = (y_dim - sagittal_square_size) // 2
    sagittal_y_end = sagittal_y_start + sagittal_square_size
    sagittal_z_start = (z_dim - sagittal_square_size) // 2
    sagittal_z_end = sagittal_z_start + sagittal_square_size
    
    # Coronal slices (y-direction): mri_img[:, y, :] -> shape (x_dim, z_dim)
    coronal_square_size = min(x_dim, z_dim)
    coronal_x_start = (x_dim - coronal_square_size) // 2
    coronal_x_end = coronal_x_start + coronal_square_size
    coronal_z_start = (z_dim - coronal_square_size) // 2
    coronal_z_end = coronal_z_start + coronal_square_size

    # Axial slices (z-direction): mri_img[:, :, z] -> shape (x_dim, y_dim)
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
    # Process and save each MRI slice
    # --------------------------------------------------

    slice_paths: list[Path] = []
    for slice_idx in range(shortest_dim_size):
        # Get square slice from high-res plane
        if shortest_dim_name == 'x':
            slice_2d = mri_img[slice_idx, sagittal_y_start:sagittal_y_end, sagittal_z_start:sagittal_z_end]
        elif shortest_dim_name == 'y':
            slice_2d = mri_img[coronal_x_start:coronal_x_end, slice_idx, coronal_z_start:coronal_z_end]
        elif shortest_dim_name == 'z':
            slice_2d = mri_img[axial_x_start:axial_x_end, axial_y_start:axial_y_end, slice_idx]
        else:
            raise ValueError(f"Invalid shortest dimension: {shortest_dim_name}")
        
        # Convert to RGB image and resize
        slice_2d = Image.fromarray(slice_2d)
        slice_2d = slice_2d.convert('RGB')
        slice_2d = slice_2d.resize(target_size, Image.Resampling.LANCZOS)
        
        # Apply transformation to get the correct orientation (match CT script)
        if shortest_dim_name == 'x':
            # Sagittal transformation
            slice_2d = slice_2d.rotate(90, expand=True)
            slice_2d = slice_2d.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif shortest_dim_name == 'y':
            # Coronal transformation
            slice_2d = slice_2d.rotate(90, expand=True)
            slice_2d = slice_2d.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
        elif shortest_dim_name == 'z':
            # Axial transformation
            slice_2d = slice_2d.rotate(90, expand=True)

        # Save slice
        output_path = output_dir / f"slice_{slice_idx:04d}.png"
        slice_2d.save(output_path)
        slice_paths.append(output_path)
    
    if verbose:
        print(f"  Saved {shortest_dim_size} {shortest_dim_name}-slices to {output_dir}")

    # --------------------------------------------------
    # Process organ segmentations
    # --------------------------------------------------

    # Track if we've warned about dimension mismatch
    dimension_mismatch_warned = False

    # Initialize result dictionary
    result: dict[str, list] = {
        "image_id": [image_id] * shortest_dim_size,
        "slice_idx": list(range(shortest_dim_size)),
        "slice_orientation": [shortest_dim_name] * shortest_dim_size,
        "image_path": [str(p) for p in slice_paths],
    }
    
    # Process organ segmentations
    for organ in organs:
        seg_file = seg_dir / f"{organ}.nii.gz"
        
        # Load segmentation mask
        seg_mask_nib = nib_load(seg_file)
        assert isinstance(seg_mask_nib, Nifti1Image)
        mask_data = seg_mask_nib.get_fdata()

        if verbose:
            print(f"  Processing organ: {organ}")
            print(f"    Segmentation mask shape: {mask_data.shape}")

        # If MRI was resampled, apply the same resampling to masks
        if resampler_for_masks is not None:
            # Convert mask to SimpleITK image
            sitk_mask = sitk.GetImageFromArray(mask_data.astype(np.float32).transpose(2, 1, 0))
            # Set original spacing (before resampling)
            sitk_mask.SetSpacing(original_spacing)
            
            # Resample mask
            sitk_mask_resampled = resampler_for_masks.Execute(sitk_mask)
            mask_data = sitk.GetArrayFromImage(sitk_mask_resampled).transpose(2, 1, 0)

        # Check if mask dimensions match MRI dimensions
        if mask_data.shape != (x_dim, y_dim, z_dim):
            # Warn once per MRI scan
            if not dimension_mismatch_warned and verbose:
                print(f"  WARNING: Segmentation mask shape {mask_data.shape} differs from MRI shape {(x_dim, y_dim, z_dim)}")
                print(f"           Resizing masks to match MRI dimensions...")
                dimension_mismatch_warned = True
            
            # Resize mask to match MRI dimensions
            zoom_factors = (
                x_dim / mask_data.shape[0],
                y_dim / mask_data.shape[1],
                z_dim / mask_data.shape[2]
            )
            mask_data = zoom(mask_data, zoom_factors, order=0, mode='nearest')

        # Convert to boolean for memory efficiency and faster .any() operations
        mask_data = mask_data.astype(bool)

        # Check presence only in the high-res plane (no need to resize to target_size)
        if shortest_dim_name == 'x':
            # X-direction (sagittal): Check presence in cropped region
            mask_cropped = mask_data[:, sagittal_y_start:sagittal_y_end, sagittal_z_start:sagittal_z_end]
            organ_presence = [int(mask_cropped[idx].any()) for idx in range(x_dim)]
        elif shortest_dim_name == 'y':
            # Y-direction (coronal): Check presence in cropped region
            mask_cropped = mask_data[coronal_x_start:coronal_x_end, :, coronal_z_start:coronal_z_end]
            organ_presence = [int(mask_cropped[:, idx, :].any()) for idx in range(y_dim)]
        elif shortest_dim_name == 'z':
            # Z-direction (axial): Check presence in cropped region
            mask_cropped = mask_data[axial_x_start:axial_x_end, axial_y_start:axial_y_end, :]
            organ_presence = [int(mask_cropped[:, :, idx].any()) for idx in range(z_dim)]
        else:
            raise ValueError(f"Invalid shortest dimension: {shortest_dim_name}")

        # Add presence vector for this organ to result
        result[organ] = organ_presence
        
        if verbose:
            print(f"    {sum(organ_presence)}/{shortest_dim_size} slices with presence")
    
    return result


def process_single_mri_wrapper(args):
    """
    Wrapper function for multiprocessing that unpacks arguments.
    
    Args:
        args: Tuple of (mri_dir, organs, output_base_dir, bias_correction_max_threads)
    
    Returns:
        Result dictionary from process_mri_scan_to_slices or None if error
    """
    mri_dir, organs, output_base_dir, bias_field_correction = args
    
    try:
        result = process_mri_scan_to_slices(
            totalseg_single_mri_scan_dir=mri_dir,
            organs=organs,
            output_base_dir=output_base_dir,
            apply_bias_correction=bias_field_correction,
            verbose=False  # Disable verbose to avoid messy output with multiprocessing
        )
        return result
    except Exception as e:
        print(f"ERROR processing {mri_dir.name}: {e}")
        import traceback
        traceback.print_exc()  # Print full traceback for debugging
        return None


def main():
    # Directories and files
    MRI_DIR = "TotalSegmentator/MRI_Dataset_v200"
    REFERENCE_SEG_DIR = "TotalSegmentator/MRI_Dataset_v200/s0001/segmentations"
    MRI_META_CSV = "TotalSegmentator/MRI_Dataset_v200/meta.csv"
    OUTPUT_BASE_DIR = "data/total_seg_mri"
    NUM_WORKERS = 32
    BIAS_FIELD_CORRECTION = False

    print("\n"+"="*50)
    print("PROCESSING TOTAL SEGMENTATOR MRI DATASET")
    print("="*50+"\n")

    # Convert to Path objects
    mri_dir = Path(MRI_DIR)
    reference_seg_dir = Path(REFERENCE_SEG_DIR)
    mri_meta_csv = Path(MRI_META_CSV)
    output_base_dir = Path(OUTPUT_BASE_DIR)
    
    print(f"Original MRI directory:            {mri_dir}")
    print(f"Reference segmentation directory:  {reference_seg_dir}")
    print(f"Number of workers:                 {NUM_WORKERS}")
    print()
    
    # Create output directory
    print(f"Creating output directory: {output_base_dir}\n")
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Get all MRI scan directories
    print("Searching for all MRI scan directories...")
    mri_sample_dirs = [x for x in mri_dir.iterdir() if x.is_dir()]
    mri_sample_dirs.sort()
    print(f"Found {len(mri_sample_dirs):d} MRI scan directories")
    print()
    
    # Get organ list
    print("Searching for all organs in the reference segmentation directory...")
    organs = get_organ_list(reference_seg_dir)
    print(f"Found {len(organs):d} organs")
    print()
    
    # Load metadata
    print("Loading MRI metadata...")
    df_mri_metadata = pd.read_csv(mri_meta_csv, sep=";")
    print(f"Loaded {len(df_mri_metadata):d} metadata rows")
    print()
    
    # Process all MRI scans with multiprocessing
    print("-"*50)
    print(f"Processing {len(mri_sample_dirs)} MRI scans with {NUM_WORKERS} workers")
    print("-"*50+"\n")
    
    # Prepare arguments for each worker
    processing_args = [
        (mri_scan_dir, organs, output_base_dir, BIAS_FIELD_CORRECTION)
        for mri_scan_dir in mri_sample_dirs
    ]
    
    # Process with progress bar
    processing_results = []
    
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        futures = {
            executor.submit(process_single_mri_wrapper, args): args[0].name
            for args in processing_args
        }
        
        # Collect results with progress bar
        for future in tqdm(as_completed(futures), total=len(futures), desc="Processing MRIs"):
            mri_name = futures[future]
            try:
                result = future.result()
                if result is not None:
                    processing_results.append(result)
            except Exception as e:
                tqdm.write(f"ERROR with {mri_name}: {e}")
                # Don't shutdown executor - continue processing other MRIs

    print(f"\n[ok] Successfully processed {len(processing_results)}/{len(mri_sample_dirs)} MRI scans")

    # Check if any MRIs were processed
    if len(processing_results) == 0:
        print("\nERROR: ERROR: No MRI scans were successfully processed!")
        print("   Check the error messages above for details.")
        return

    # Combine all results into a single dataframe
    print("\nCombining all results into a single dataframe...")
    df_mri_slices = pd.concat([pd.DataFrame(result) for result in processing_results], ignore_index=True)
    
    # Load and merge with metadata
    tot_seg_meta_csv = Path("TotalSegmentator/MRI_Dataset_v200/meta.csv")
    df_tot_seg_meta = pd.read_csv(tot_seg_meta_csv, sep=";")
    df_mri_slices = df_mri_slices.merge(df_tot_seg_meta, on="image_id", how="left")
    print(f"  Merged with metadata from {tot_seg_meta_csv}")
    
    # Reorder columns: image_id, slice_idx, image_path, then metadata, then organs
    meta_cols = ["image_id", "slice_idx", "slice_orientation", "image_path"]
    tot_seg_meta_cols = cast(list[str], df_tot_seg_meta.columns.tolist()[1:])  # exclude image_id
    organ_cols = [col for col in organs if col in df_mri_slices.columns]
    
    # Reorder: meta cols, tot_seg_meta cols, then organ cols
    column_order = meta_cols + tot_seg_meta_cols + organ_cols
    df_mri_slices = df_mri_slices[column_order]
    
    # Save DataFrame
    output_csv = output_base_dir / "metadata.csv"
    df_mri_slices.to_csv(output_csv, index=False)
    
    print(f"\n{'='*60}")
    print(f"Processing complete!")
    print(f"  Results saved to:    {output_csv}")
    print(f"  Processed MRI scans: {len(processing_results)}")
    print(f"  Total slices:        {len(df_mri_slices)}")
    print(f"  Metadata columns:    {len(df_mri_slices.columns)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()