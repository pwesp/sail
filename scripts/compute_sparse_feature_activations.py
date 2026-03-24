#!/usr/bin/env python3
"""
Compute sparse feature activations for all train/val/test splits using trained
Matryoshka SAE checkpoints.

Discovers all trained SAE configurations under lightning_logs/, runs inference
on the TotalSegmentator embedding parquets, and writes sparse feature parquets
to results/total_seg_sparse_features/.

Run with: conda activate sail && python compute_sparse_feature_activations.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from collections import defaultdict
import numpy as np
import pandas as pd
from pathlib import Path
import torch
from tqdm import tqdm
import yaml

from src.matryoshka_sae import MatryoshkaSAELightning
from src.dataloading import TotalSegmentatorLatentFeaturesDataset


def load_pretrained_matryoshka_sae(checkpoint_path: str | Path, device: str) -> MatryoshkaSAELightning:
    """Load a trained Matryoshka SAE from a Lightning checkpoint."""
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    print(f"Loading SAE checkpoint: {checkpoint_path}")
    model = MatryoshkaSAELightning.load_from_checkpoint(checkpoint_path).eval()

    hp = model.hparams
    print(f"  Input dim: {hp.get('input_dim', 'N/A')}, "
          f"dictionary sizes: {hp.get('dictionary_sizes', 'N/A')}, "
          f"k values: {hp.get('k_values', 'N/A')}")

    if not model.model._threshold_estimated:
        print("  Inference threshold not estimated - calculating from state_dict...")
        model.model.calculate_threshold()
    print(f"  Inference threshold: {model.model.inference_threshold.item():.6f}")

    model = model.to('cuda' if device == 'cuda' and torch.cuda.is_available() else 'cpu')
    print(f"  SAE loaded on {'GPU' if device == 'cuda' else 'CPU'}")
    return model


def compute_sparse_features(
    sae: MatryoshkaSAELightning,
    dataset: TotalSegmentatorLatentFeaturesDataset,
    output_path: Path | str,
    batch_size: int = 2048,
    device: str = 'cuda',
) -> None:
    """
    Run SAE inference on a dataset and save sparse feature activations to parquet.

    Uses the largest dictionary level (last element of all_latents). Output rows
    contain all original metadata columns, with embedding columns replaced by:
    sparse_features, sparse_features_shape, sparse_features_dtype.

    Results are written in chunks and atomically renamed to avoid partial files.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"\nComputing sparse features: {len(dataset)} samples -> {output_path}")

    temp_dir = output_path.parent / (output_path.stem + '_chunks')
    temp_output_path = output_path.with_suffix(output_path.suffix + '.tmp')
    temp_dir.mkdir(exist_ok=True)

    # Clean up leftovers from any previous interrupted run
    if temp_output_path.exists():
        temp_output_path.unlink()
    for f in temp_dir.glob('chunk_*.parquet'):
        f.unlink()

    embedding_columns = {'embedding', 'embedding_shape', 'embedding_dtype'}
    row_buffer = []
    chunk_files = []
    total_written = 0

    for idx in tqdm(range(len(dataset)), desc="Computing sparse features"):
        sample = dataset[idx]
        embedding = sample['embedding'].unsqueeze(0)
        if device == 'cuda':
            embedding = embedding.cuda()

        with torch.no_grad():
            _, _, all_latents = sae(embedding)
            sparse = all_latents[-1].squeeze(0).cpu().numpy().astype(np.float32)

        row = {k: v for k, v in sample.items() if k not in embedding_columns}
        row['sparse_features'] = sparse.tobytes()
        row['sparse_features_shape'] = str(sparse.shape)
        row['sparse_features_dtype'] = str(sparse.dtype)
        row_buffer.append(row)

        if (idx + 1) % batch_size == 0:
            chunk_file = temp_dir / f"chunk_{len(chunk_files):06d}.parquet"
            pd.DataFrame(row_buffer).to_parquet(chunk_file, engine='pyarrow', index=False)
            chunk_files.append(chunk_file)
            total_written += len(row_buffer)
            row_buffer = []

    if row_buffer:
        chunk_file = temp_dir / f"chunk_{len(chunk_files):06d}.parquet"
        pd.DataFrame(row_buffer).to_parquet(chunk_file, engine='pyarrow', index=False)
        chunk_files.append(chunk_file)
        total_written += len(row_buffer)

    if not chunk_files:
        raise ValueError("No data written - all batches failed.")

    try:
        final_df = pd.concat(
            [pd.read_parquet(f) for f in chunk_files], ignore_index=True
        )
        assert len(final_df) == total_written, \
            f"Row count mismatch: expected {total_written}, got {len(final_df)}"

        final_df.to_parquet(temp_output_path, engine='pyarrow', index=False)
        for f in chunk_files:
            f.unlink()
        temp_dir.rmdir()
        temp_output_path.replace(output_path)
    except Exception:
        if temp_output_path.exists():
            temp_output_path.unlink()
        for f in chunk_files:
            if f.exists():
                f.unlink()
        if temp_dir.exists():
            try:
                temp_dir.rmdir()
            except OSError:
                pass
        raise

    print(f"  {len(final_df)} rows, {output_path.stat().st_size / (1024**2):.2f} MB")


def process_configuration(
    sae_checkpoint: str,
    train_parquet: str,
    val_parquet: str,
    test_parquet: str,
    output_dir: str,
    batch_size: int = 2048,
    device: str = 'cuda',
) -> None:
    """Compute sparse features for train/val/test splits of one SAE configuration."""
    if Path(output_dir).exists():
        print(f"  Output directory already exists - skipping: {output_dir}")
        return

    sae = load_pretrained_matryoshka_sae(sae_checkpoint, device=device)

    splits = [
        ('train', train_parquet, Path(output_dir) / "total_seg_sparse_features_train.parquet"),
        ('val',   val_parquet,   Path(output_dir) / "total_seg_sparse_features_val.parquet"),
        ('test',  test_parquet,  Path(output_dir) / "total_seg_sparse_features_test.parquet"),
    ]

    for split_name, parquet_path, output_path in splits:
        print(f"\n{'='*80}\nProcessing {split_name} split\n{'='*80}")
        dataset = TotalSegmentatorLatentFeaturesDataset(
            feature_type='embeddings',
            parquet_path=parquet_path,
            shuffle=False,
            additional_metadata="all",
        )
        compute_sparse_features(sae=sae, dataset=dataset, output_path=output_path,
                                batch_size=batch_size, device=device)

    print(f"\n  Output: {output_dir}")


def discover_sae_configurations(sae_directories: list[str]) -> list[dict]:
    """
    Discover all trained SAE configurations from the given log directories.

    Each subdirectory is expected to contain hparams.yaml and
    checkpoints/last.ckpt. Returns a list of config dicts with paths and
    hyperparameters needed for inference.
    """
    configurations = []

    for sae_dir in sae_directories:
        sae_dir_path = Path(sae_dir)
        if not sae_dir_path.exists():
            print(f"  Directory not found: {sae_dir}")
            continue

        dir_name = sae_dir_path.name
        if 'biomedparse_random' in dir_name:
            embedding_model = 'biomedparse_random'
        elif 'biomedparse' in dir_name:
            embedding_model = 'biomedparse'
        elif 'dinov3' in dir_name:
            embedding_model = 'dinov3'
        else:
            print(f"  Unknown embedding model in directory name: {dir_name}")
            continue

        for config_dir in sorted(sae_dir_path.iterdir()):
            if not config_dir.is_dir():
                continue

            hparams_file = config_dir / "hparams.yaml"
            checkpoint_file = config_dir / "checkpoints" / "last.ckpt"

            if not hparams_file.exists():
                print(f"  Missing hparams.yaml: {config_dir.name}")
                continue
            if not checkpoint_file.exists():
                print(f"  Missing checkpoint: {config_dir.name}")
                continue

            try:
                with open(hparams_file) as f:
                    hparams = yaml.safe_load(f)

                train_parquet = hparams.get('train_parquet')
                val_parquet = hparams.get('val_parquet')
                test_parquet = train_parquet.replace('train', 'test') if train_parquet else None

                config = {
                    'embedding_model': embedding_model,
                    'config_name': config_dir.name,
                    'log_subdir': dir_name,
                    'checkpoint_path': str(checkpoint_file),
                    'dictionary_sizes': hparams.get('dictionary_sizes'),
                    'k_values': hparams.get('k_values'),
                    'train_parquet': train_parquet,
                    'val_parquet': val_parquet,
                    'test_parquet': test_parquet,
                    'input_dim': hparams.get('input_dim'),
                }

                required = ['dictionary_sizes', 'k_values', 'train_parquet', 'val_parquet', 'input_dim']
                missing = [k for k in required if config.get(k) is None]
                if missing:
                    print(f"  Missing fields in {config_dir.name}: {missing}")
                    continue
                if test_parquet and not Path(test_parquet).exists():
                    print(f"  Test parquet not found for {config_dir.name}: {test_parquet}")
                    continue

                configurations.append(config)

            except Exception as e:
                print(f"  Error reading {hparams_file}: {e}")

    return configurations


def main():
    print("=" * 80)
    print("Compute Sparse Feature Activations - Matryoshka SAE / TotalSegmentator")
    print("=" * 80)

    SAE_DIRECTORIES = [
        "lightning_logs/biomedparse_matryoshka_sae",
        "lightning_logs/biomedparse_random_matryoshka_sae",
        "lightning_logs/dinov3_matryoshka_sae",
    ]
    BATCH_SIZE = 8192
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    TOP_OUTPUT_DIR = "results/total_seg_sparse_features"

    print("\nDiscovering trained SAE configurations...")
    configurations = discover_sae_configurations(SAE_DIRECTORIES)
    print(f"  Found {len(configurations)} valid configurations")

    by_model = defaultdict(list)
    for c in configurations:
        by_model[c['embedding_model']].append(c)
    for model, configs in by_model.items():
        print(f"  {model}: {len(configs)} configurations")

    print(f"\n{'='*80}\nProcessing SAE Configurations\n{'='*80}")

    for i, config in enumerate(tqdm(configurations, desc="Processing configurations"), 1):
        output_dir = f"{TOP_OUTPUT_DIR}/{config['log_subdir']}/{config['config_name']}"
        print(f"\n{'-'*80}")
        print(f"Config #{i}/{len(configurations)}: {config['config_name']}")
        print(f"  Model: {config['embedding_model']}, "
              f"dict sizes: {config['dictionary_sizes']}, "
              f"k values: {config['k_values']}")
        print(f"  Output: {output_dir}")

        process_configuration(
            sae_checkpoint=config['checkpoint_path'],
            train_parquet=config['train_parquet'],
            val_parquet=config['val_parquet'],
            test_parquet=config['test_parquet'],
            output_dir=output_dir,
            batch_size=BATCH_SIZE,
            device=DEVICE,
        )

    print(f"\n{'='*80}")
    print(f"Done. {len(configurations)} configurations processed.")
    print(f"Output: {TOP_OUTPUT_DIR}")


if __name__ == "__main__":
    main()
