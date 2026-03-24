#!/usr/bin/env python3
"""
Encode TotalSegmentator CT and MRI images with a foundation model.

Supported models:
- biomedparse         Pretrained BiomedParse (1536-dim)
- biomedparse_random  Randomly initialized BiomedParse (1536-dim, random baseline)
- dinov3              Pretrained DINOv3 (1024-dim)

BiomedParse models require the 'biomedparse' conda environment.
DINOv3 requires the 'sail' conda environment.

Usage:
    conda activate biomedparse
    python encode_totalseg_embeddings.py --model biomedparse
    python encode_totalseg_embeddings.py --model biomedparse_random

    conda activate sail
    python encode_totalseg_embeddings.py --model dinov3
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import argparse
import numpy as np
from typing import Optional

from src.dataloading import TotalSegmentatorDataset, TotalSegmentatorEncodingDataModule
from src.encoding import encode_dataset, verify_parquet_file


# ============================================================================
# Settings
# ============================================================================

DATA_ROOT = Path("data")

CT_DATA_DIR     = DATA_ROOT / "total_seg_ct"
MRI_DATA_DIR    = DATA_ROOT / "total_seg_mri"
CT_METADATA_CSV = DATA_ROOT / "total_seg_ct" / "metadata.csv"
MRI_METADATA_CSV = DATA_ROOT / "total_seg_mri" / "metadata.csv"

# Per-model output paths and default batch sizes
MODEL_CONFIGS = {
    "biomedparse": {
        "output": DATA_ROOT / "total_seg_biomedparse_encodings" / "total_seg_biomedparse_encodings_all.parquet",
        "checkpoint_dir": DATA_ROOT / "total_seg_biomedparse_encodings_tmp",
        "batch_size": 32,
        "write_batch_size": 32 * 50,
        "num_workers": 8,
        "conda_env": "biomedparse",
    },
    "biomedparse_random": {
        "output": DATA_ROOT / "total_seg_biomedparse_random_encodings" / "total_seg_biomedparse_random_encodings_all.parquet",
        "checkpoint_dir": DATA_ROOT / "total_seg_biomedparse_random_encodings_tmp",
        "batch_size": 32,
        "write_batch_size": 32 * 50,
        "num_workers": 8,
        "conda_env": "biomedparse",
    },
    "dinov3": {
        "output": DATA_ROOT / "total_seg_dinov3_encodings" / "total_seg_dinov3_encodings_all.parquet",
        "checkpoint_dir": DATA_ROOT / "total_seg_dinov3_encodings_tmp",
        "batch_size": 64,
        "write_batch_size": 64 * 25,
        "num_workers": 4,
        "conda_env": "sail",
    },
}

DINOV3_MODEL_NAME = "facebook/dinov3-vitl16-pretrain-lvd1689m"
DINOV3_INPUT_SIZE = (1024, 1024)
BIOMEDPARSE_RANDOM_SEED = 42


# ============================================================================
# BiomedParse model loading (only imported when needed)
# ============================================================================

def load_biomedparse_model(use_pretrained: bool, random_seed: Optional[int] = None):
    """Load BiomedParse model, optionally with random initialization."""
    import torch
    from timm.layers.weight_init import trunc_normal_  # type: ignore

    biomedparse_path = Path("BiomedParse")
    sys.path.insert(0, str(biomedparse_path))

    from modeling import build_model  # type: ignore
    from modeling.BaseModel import BaseModel  # type: ignore
    from utilities.arguments import load_opt_from_config_files  # type: ignore
    from utilities.constants import BIOMED_CLASSES  # type: ignore
    from utilities.distributed import init_distributed  # type: ignore

    print("Loading BiomedParse model...")

    if not use_pretrained and random_seed is not None:
        print(f"  Setting random seed: {random_seed}")
        torch.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        np.random.seed(random_seed)

    config_path = biomedparse_path / "configs" / "biomedparse_inference.yaml"
    opt = load_opt_from_config_files([str(config_path)])
    opt = init_distributed(opt)

    if use_pretrained:
        model = BaseModel(opt, build_model(opt)).from_pretrained('hf_hub:microsoft/BiomedParse').eval()
        print("  Loaded pretrained weights from HuggingFace")
    else:
        print("  Using randomly initialized weights (no pretraining)")
        model = BaseModel(opt, build_model(opt))

        def _init_weights(m):
            if isinstance(m, torch.nn.Linear):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)
            elif isinstance(m, torch.nn.LayerNorm):
                torch.nn.init.constant_(m.bias, 0)
                torch.nn.init.constant_(m.weight, 1.0)
            elif isinstance(m, torch.nn.Conv2d):
                trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    torch.nn.init.constant_(m.bias, 0)

        model.apply(_init_weights)
        model = model.eval()

    if torch.cuda.is_available():
        model = model.cuda()

    with torch.no_grad():
        model.model.sem_seg_head.predictor.lang_encoder.get_text_embeddings(
            BIOMED_CLASSES + ["background"], is_eval=True
        )

    print("  BiomedParse model ready")
    return model


# ============================================================================
# Entry point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Encode TotalSegmentator images with a foundation model."
    )
    parser.add_argument(
        "--model",
        choices=list(MODEL_CONFIGS.keys()),
        required=True,
        help="Foundation model to use for encoding.",
    )
    parser.add_argument(
        "--num-images",
        type=int,
        default=0,
        help="Encode only the first N images (0 = all). Useful for testing.",
    )
    args = parser.parse_args()

    config = MODEL_CONFIGS[args.model]

    # Inform the user about the required conda environment
    print(f"INFO: Model '{args.model}' requires the '{config['conda_env']}' conda environment. "
          f"Make sure it is activated before running this script.")

    output_path = config["output"]
    checkpoint_dir = config["checkpoint_dir"]

    if args.num_images > 0:
        output_path = DATA_ROOT / f"total_seg_{args.model}_encodings_testing" / f"total_seg_{args.model}_encodings_testing.parquet"
        checkpoint_dir = DATA_ROOT / f"total_seg_{args.model}_encodings_testing_tmp"
        print(f"INFO: Test mode - encoding only {args.num_images} images.")

    # Load feature extractor
    if args.model in ("biomedparse", "biomedparse_random"):
        from src.feature_extractor import GlobalFeatureExtractor
        use_pretrained = (args.model == "biomedparse")
        model = load_biomedparse_model(use_pretrained=use_pretrained, random_seed=BIOMEDPARSE_RANDOM_SEED)
        feature_extractor = GlobalFeatureExtractor(model=model, backbone=model.model.backbone)
    else:
        from src.feature_extractor import DINOv3GlobalFeatureExtractor
        feature_extractor = DINOv3GlobalFeatureExtractor(
            model_name=DINOV3_MODEL_NAME,
            input_size=DINOV3_INPUT_SIZE,
            use_bfloat16=True,
        )

    # Load dataset
    print("\nLoading TotalSegmentator dataset...")
    dataset = TotalSegmentatorDataset(
        ct_data_dir=str(CT_DATA_DIR),
        mri_data_dir=str(MRI_DATA_DIR),
        ct_metadata_csv=str(CT_METADATA_CSV),
        mri_metadata_csv=str(MRI_METADATA_CSV),
    )
    print(f"  Dataset loaded: {len(dataset)} images")

    if args.num_images > 0:
        dataset.df_combined_metadata = dataset.df_combined_metadata.iloc[:args.num_images].copy()
        print(f"  Truncated to {len(dataset.df_combined_metadata)} images")

    # Resume from existing checkpoint batches
    checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)
    existing_batches = sorted(checkpoint_dir_path.glob("batch_*.parquet"))
    start_idx = len(existing_batches) * config["write_batch_size"] if existing_batches else 0

    datamodule = TotalSegmentatorEncodingDataModule(
        totalseg_dataset=dataset,
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        start_idx=start_idx,
        pin_memory=True,
        prefetch_factor=2,
    )

    encode_dataset(
        datamodule=datamodule,
        feature_extractor=feature_extractor,
        output_path_str=str(output_path),
        write_batch_size=config["write_batch_size"],
        checkpoint_dir=str(checkpoint_dir),
        total_images=len(dataset.df_combined_metadata),
    )

    verify_parquet_file(str(output_path))


if __name__ == "__main__":
    main()
