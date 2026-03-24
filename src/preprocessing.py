"""
TotalSegmentator dataset preprocessing utilities.
"""

from pathlib import Path


def get_organ_list(reference_seg_dir: Path) -> list[str]:
    """
    Get the list of all the unique organs in the TotalSegmentator datasetfrom a reference segmentation directory.
    
    Args:
        reference_seg_dir: Directory containing .nii.gz segmentation files for a single CT scan
    
    Returns:
        Sorted list of all the unique organ names in the TotalSegmentator dataset (without .nii.gz extension)
    """
    seg_files = [f for f in reference_seg_dir.glob("*") if f.name.endswith(".nii.gz")]
    seg_files.sort()
    organs = [f.name.replace(".nii.gz", "") for f in seg_files]
    organs.sort()
    return organs