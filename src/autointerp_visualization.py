"""
Visualization utilities for automated feature interpretation.

Provides functions to create publication-quality figures showing:
- Feature descriptions (VLM-generated text)
- Top-N activating images
- Per-image metadata (activation, modality, orientation)
"""

# stdlib
from pathlib import Path

# third-party
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def visualize_feature_interpretation(
    feature_idx: int,
    description: str,
    top_samples: list[dict],
    output_path: Path,
    true_concept_rank: int | None = None
) -> None:
    """
    Create publication-quality figure showing feature description + top-N images.

    Layout:
    +---------------------------------------------+
    |  Feature {idx}: {description text box}      |
    +---------------------------------------------+
    |  [Image 1]    [Image 2]    [Image 3]  ...   |
    |  act={val}    act={val}    act={val}        |
    |  {modality}   {modality}   {modality}       |
    |  {orient}     {orient}     {orient}         |
    +---------------------------------------------+

    Args:
        feature_idx: Feature index in SAE
        description: VLM-generated natural language description
        top_samples: List of sample dicts with metadata and image paths
        output_path: Path to save figure (PNG)
        true_concept_rank: Optional rank of true concept in evaluation (1-5)

    Returns:
        None (saves figure to disk)
    """
    n_samples = len(top_samples)

    # Create figure: 1 row for description + 1 row for images
    fig = plt.figure(figsize=(4 * n_samples, 6))
    gs = fig.add_gridspec(2, 1, height_ratios=[1, 4], hspace=0.3)

    # Top: Description text box
    ax_text = fig.add_subplot(gs[0, :])
    ax_text.axis('off')
    ax_text.text(
        0.5, 0.5,
        description,
        fontsize=12,
        verticalalignment='center',
        horizontalalignment='center',
        bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8, pad=1.0),
        transform=ax_text.transAxes,
        wrap=True
    )

    # Bottom: Image grid
    gs_images = gs[1, :].subgridspec(1, n_samples, wspace=0.2)

    for i, sample in enumerate(top_samples):
        ax = fig.add_subplot(gs_images[0, i])

        # Load and display image
        img_path = sample['image_path']
        if img_path.exists():
            img = Image.open(img_path).convert('L')
            ax.imshow(np.array(img), cmap='gray')
        else:
            ax.text(0.5, 0.5, 'Image\nNot Found', ha='center', va='center')

        # Title with metadata
        title_parts = [
            f"Activation: {sample['activation']:.3f}",
            f"{sample['modality']}",
            f"{sample['slice_orientation']}"
        ]
        ax.set_title('\n'.join(title_parts), fontsize=9)
        ax.axis('off')

    # Create title with optional rank
    if true_concept_rank is not None:
        title = f"Feature {feature_idx} (Rank: {true_concept_rank})"
    else:
        title = f"Feature {feature_idx}"

    plt.suptitle(title, fontsize=14, fontweight='bold', y=0.98)
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
