#!/usr/bin/env python3
"""
Train a hierarchical Matryoshka Sparse Autoencoder (SAE) on pre-computed
foundation model embeddings (BiomedParse or DINOv3) from the TotalSegmentator
CT and MRI dataset using PyTorch Lightning.

Performs a grid search over dictionary sizes and sparsity levels (k values).
Run with: conda activate sail && python train_matryoshka_sae.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gc
import lightning as L
from lightning.pytorch.callbacks import ModelCheckpoint, LearningRateMonitor
from lightning.pytorch.loggers import CSVLogger
import os
from pathlib import Path
from sklearn.model_selection import ParameterGrid
import time
import torch
import torch.nn.functional as F
from tqdm import tqdm
from typing import List, Literal, Optional
import yaml

# local
from src.dataloading import TotalSegmentatorFeaturesDataModule
from src.eval import plot_matryoshka_sae_training_curves
from src.matryoshka_sae import MatryoshkaSAELightning


def print_memory_usage(label: str = "") -> None:
    """Print current system RAM and GPU memory usage."""
    import psutil
    process = psutil.Process(os.getpid())
    mem_gb = process.memory_info().rss / (1024 ** 3)
    vm = psutil.virtual_memory()
    print(f"\nMemory Usage: {label}")
    print(f"  Process RSS:        {mem_gb:.2f} GB")
    print(f"  System RAM Used:    {vm.used / (1024**3):.2f} GB / {vm.total / (1024**3):.2f} GB ({vm.percent:.1f}%)")
    print(f"  System RAM Avail:   {vm.available / (1024**3):.2f} GB")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / (1024**3)
            reserved  = torch.cuda.memory_reserved(i) / (1024**3)
            total     = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            print(f"  GPU {i} Allocated:  {allocated:.2f} GB")
            print(f"  GPU {i} Reserved:   {reserved:.2f} GB")
            print(f"  GPU {i} Total:      {total:.2f} GB")


# ============================================================================
# Main Training Script
# ============================================================================

def train_matryoshka_sae(
    dictionary_sizes: List[int],
    k_values: List[int],
    train_parquet: str = "data/total_seg_biomedparse_encodings/total_seg_biomedparse_encodings_train.parquet",
    val_parquet: str = "data/total_seg_biomedparse_encodings/total_seg_biomedparse_encodings_valid.parquet",
    test_parquet: Optional[str] = None,
    batch_size: int = 2048,
    num_workers: int = 0,
    d_input: int = 1536,
    tied_weights: bool = True,
    normalize_decoder: bool = True,
    device: str = 'cuda',
    learning_rate: float = 1e-4,
    l1_coefficient: float = 0.0,
    diversity_coefficient: float = 0.0,
    max_epochs: int = 100,
    gradient_clip_val: float = 0.5,
    precision: Literal['64', '32', '16', 'bf16', 'bf16-mixed', '16-mixed', '32-true', '16-true', 'bf16-true'] = 'bf16-mixed',
    log_dir: str = "lightning_logs",
    log_subdir: str = "totalseg_biomedparse_matryoshka_sae",
    config_name: str | None = None
) -> None:
    """
    Lightweight training wrapper for Matryoshka Sparse Autoencoder.

    Args:
        dictionary_sizes: list, nested dictionary sizes (e.g. [64, 256, ...])
        k_values: list, top-k per level (e.g. [20, 40, ...])
        train_parquet: train parquet path (default: '/.../train.parquet')
        val_parquet:   val parquet path (default: '/.../valid.parquet')
        test_parquet:  test parquet path (default: '/.../test.parquet')
        batch_size: mini-batch size (default: 2048)
        num_workers: dataloader workers (default: 0)
        d_input: input embedding dim (default: 1536)
        tied_weights: shared encoder/decoder (default: True)
        normalize_decoder: decoder columns unit norm (default: True)
        device: 'cuda' or 'cpu' (default: 'cuda')
        learning_rate: optimizer LR (default: 1e-4)
        l1_coefficient: L1 penalty weight (default: 0.0)
        diversity_coefficient: diversity penalty (default: 0.0)
        max_epochs: max training epochs (default: 100)
        gradient_clip_val: gradient clipping (default: 0.5)
        precision: precision mode (default: 'bf16-mixed')
        log_dir: logging directory (default: 'lightning_logs')
        log_subdir: experiment subdir (default: 'totalseg_biomedparse_matryoshka_sae')
        config_name: optional experiment name (default: None)
    """
    
    # ========================================================================
    # Check if config has already been trained
    # ========================================================================

    if Path(f"{log_dir:s}/{log_subdir:s}/{config_name:s}").exists():
        print(f"\n   !!! Logging directory already exists: {log_dir:s}/{log_subdir:s}/{config_name:s}")
        print(f"  -> Assuming the model config has already been trained")
        print(f"  -> Skipping training this config...\n")
        return None
    
    # ========================================================================
    # Create logging directory
    # ========================================================================
    os.makedirs(f"{log_dir:s}/{log_subdir:s}/{config_name:s}", exist_ok=True)
    
    # Initial memory check
    print_memory_usage("Start of Training")
    
    # ========================================================================
    # Validate Configuration
    # ========================================================================
    
    # Ensure k_values are compatible with dictionary_sizes
    if len(k_values) != len(dictionary_sizes):
        raise ValueError(
            f"k_values length ({len(k_values)}) must match dictionary_sizes length ({len(dictionary_sizes)})"
        )
    
    for i, (k, d) in enumerate(zip(k_values, dictionary_sizes)):
        if k > d:
            raise ValueError(
                f"k_values[{i}]={k} exceeds dictionary_sizes[{i}]={d}. "
                f"Top-K must be <= dictionary size for each level."
            )
    
    # ========================================================================
    # System Information
    # ========================================================================
    
    print("\n\nSystem Information")
    print("-" * 80)
    
    # CPU
    cpu_name = "Unknown"
    try:
        with open('/proc/cpuinfo', 'r') as f:
            for line in f:
                if line.startswith('model name'):
                    cpu_name = line.split(':', 1)[1].strip()
                    break
    except FileNotFoundError:
        pass
    
    cpu_count = "Unknown"
    try:
        cpu_count = os.cpu_count()
    except:
        pass

    print(f"  CPU: {cpu_name}")
    print(f"  CPU cores: {cpu_count}")
    
    # GPU
    print(f"  Specified device: {device:s}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  GPU: {gpu_name}")
        
        if "RTX" in gpu_name or "A100" in gpu_name:
            torch.set_float32_matmul_precision('high')
            print(f"  [ok] Enabled high-precision matmul for {gpu_name}")
    
    actual_device = 'cuda' if (torch.cuda.is_available() and device == 'cuda') else 'cpu'
    print(f"  Using device: {actual_device}")
    
    # ========================================================================
    # Datamodule Setup
    # ========================================================================
    
    print("\n\nDatamodule Setup")
    print("-" * 80)
    
    datamodule = TotalSegmentatorFeaturesDataModule(
        feature_type='embeddings',
        train_parquet=train_parquet,
        val_parquet=val_parquet,
        test_parquet=test_parquet,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle_train=True,
        pin_memory=False  # Disable to reduce memory pressure
    )
    
    datamodule.setup(stage='fit')
    train_dataset = datamodule.train_dataset
    val_dataset = datamodule.val_dataset
    
    n_train = len(train_dataset)
    n_val = len(val_dataset)
    train_batches = n_train // batch_size
    val_batches = n_val // batch_size
    
    print(f"  Train samples:  {n_train:,}")
    print(f"  Val samples:    {n_val:,}")
    print(f"  Batch size:     {batch_size}")
    print(f"  Train batches:  {train_batches}")
    print(f"  Val batches:    {val_batches}")
    
    # ========================================================================
    # Sanity Check: Load Sample Batch
    # ========================================================================
    
    print("\n\nSanity Check: Sample Batch")
    print("-" * 80)
    
    t0 = time.time()
    train_loader = datamodule.train_dataloader()
    sample_batch = next(iter(train_loader))
    t1 = time.time()
    
    embeddings = sample_batch['embedding']
    
    print(f"  Batch loading time: {t1-t0:.4f} seconds")
    print(f"  Batch properties:")
    print(f"    Shape:  {embeddings.shape}")
    print(f"    Dtype:  {embeddings.dtype}")
    print(f"    Device: {embeddings.device}")
    print(f"  Statistics:")
    print(f"    Mean:    {embeddings.mean().item():.4f}")
    print(f"    Std:     {embeddings.std().item():.4f}")
    print(f"    Min:     {embeddings.min().item():.4f}")
    print(f"    Max:     {embeddings.max().item():.4f}")
    print(f"    L2 norm: {embeddings.norm(dim=-1).mean().item():.4f}")
    
    assert embeddings.shape[-1] == d_input, \
        f"Expected dim {d_input}, got {embeddings.shape[-1]}"
    print(f"  [ok] Embedding dimension verified: {d_input}")
    
    print_memory_usage("After Data Loading")
    
    # ========================================================================
    # Model Creation
    # ========================================================================
    
    print("\n\nModel Architecture")
    print("-" * 80)
    
    model = MatryoshkaSAELightning(
        input_dim=d_input,
        dictionary_sizes=dictionary_sizes,
        tied_weights=tied_weights,
        normalize_decoder=normalize_decoder,
        l1_coefficient=l1_coefficient,
        diversity_coefficient=diversity_coefficient,
        learning_rate=learning_rate,
        k_values=k_values,
        correlation_sample_batches=20
    )
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"  Matryoshka SAE Configuration:")
    print(f"    Input dimension:       {d_input}")
    print(f"    Dictionary sizes:      {dictionary_sizes}")
    print(f"    Expansion ratios:      {[f'{d/d_input:.1f}x' for d in dictionary_sizes]}")
    print(f"    Top-K (all levels):    {k_values}")
    print(f"    Sparsity ratios:       {[f'{k/d*100:.2f}%' for k, d in zip(k_values, dictionary_sizes)]}")
    print(f"    Tied weights:          {tied_weights}")
    print(f"    Normalize decoder:     {normalize_decoder}")
    print(f"  Parameters:")
    print(f"    Total:                 {total_params:,}")
    print(f"    Trainable:             {trainable_params:,}")
    print(f"  Training Configuration:")
    print(f"    Learning rate:         {learning_rate}")
    print(f"    L1 coefficient:        {l1_coefficient}")
    print(f"    Diversity coefficient: {diversity_coefficient}")
    print(f"    Gradient clip:         {gradient_clip_val}")
    print(f"    Max epochs:            {max_epochs}")
    
    # ========================================================================
    # Sanity Check: Forward Pass
    # ========================================================================
    
    print("\n\nSanity Check: Forward Pass")
    print("-" * 80)
    
    model.eval()
    if actual_device == 'cuda':
        sample_batch = {k: v.cuda() if isinstance(v, torch.Tensor) else v 
                       for k, v in sample_batch.items()}
        model = model.cuda()
    
    with torch.no_grad():
        x_test = sample_batch['embedding']
        final_recon, all_recons, all_latents = model(x_test)
    
    print(f"  [ok] Forward pass successful!")
    print(f"  Input:  {x_test.shape}")
    print(f"  Output: {final_recon.shape}")
    print(f"  Levels: {len(all_recons)}")
    
    print(f"  Per-level Analysis:")
    for level_idx, (dict_size, recon, latents) in enumerate(
        zip(dictionary_sizes, all_recons, all_latents)
    ):
        l0 = (latents != 0).float().sum(-1).mean().item()
        mse = F.mse_loss(recon, x_test).item()
        print(f"    Level {level_idx+1} ({dict_size:,}): "
              f"L0={l0:.1f}, MSE={mse:.6f}")
    
    print_memory_usage("After Model Creation")
    
    # ========================================================================
    # Trainer Setup
    # ========================================================================
    
    print("\n\nTrainer Configuration")
    print("-" * 80)
    
    # Logger - use config_name as version
    log_every_n_steps = 50
    csv_logger = CSVLogger(
        save_dir=log_dir,
        name=log_subdir,
        version=config_name
    )
    
    checkpoint_dir = Path(csv_logger.log_dir) / "checkpoints"
    
    # Callbacks
    checkpoint_callback = ModelCheckpoint(
        dirpath=checkpoint_dir,
        filename='matryoshka-sae-{epoch:02d}-step-{step:06d}-val-loss-{val/loss:.6f}',
        every_n_epochs=10,
        save_last=True,
        verbose=True
    )
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    print(f"  Logging:")
    print(f"    Log directory:         {csv_logger.log_dir}")
    print(f"    Checkpoint directory:  {checkpoint_dir}")
    
    # Trainer
    trainer = L.Trainer(
        max_epochs=max_epochs,
        accelerator='gpu' if actual_device == 'cuda' else 'cpu',
        devices=1,
        precision=precision,
        logger=csv_logger,
        log_every_n_steps=log_every_n_steps,
        callbacks=[checkpoint_callback, lr_monitor],
        gradient_clip_val=gradient_clip_val,
        gradient_clip_algorithm='norm',
        enable_checkpointing=True,
        enable_progress_bar=True,
        enable_model_summary=True
    )
    
    print(f"  Trainer:")
    print(f"    Max epochs:            {max_epochs}")
    print(f"    Precision:             {precision}")
    print(f"    Gradient clip:         {gradient_clip_val}")
    print(f"    Log interval:          Every {log_every_n_steps:d} steps")
    
    # ========================================================================
    # Training
    # ========================================================================
    
    print("\n\nStarting Training")
    print("-" * 80)
    print()
    
    model.train()
    trainer.fit(model, datamodule)
    
    print("\n\nTraining Complete")
    print("-" * 80)
    print(f"  Checkpoints: {str(checkpoint_dir):s}")
    print(f"  Logs:        {csv_logger.log_dir:s}")

    # ========================================================================
    # Save hparams with data paths for compute_sparse_feature_activations.py
    # ========================================================================

    hparams_path = Path(csv_logger.log_dir) / "hparams.yaml"
    hparams_data = {
        "input_dim": d_input,
        "dictionary_sizes": dictionary_sizes,
        "k_values": k_values,
        "train_parquet": train_parquet,
        "val_parquet": val_parquet,
    }
    with open(hparams_path, "w") as f:
        yaml.dump(hparams_data, f, default_flow_style=False)
    print(f"  hparams:     {hparams_path}")

    # ========================================================================
    # Visualization
    # ========================================================================
    
    print("\n\nGenerating Visualizations")
    print("-" * 80)
    
    plot_matryoshka_sae_training_curves(csv_logger.log_dir, dictionary_sizes)

    # ========================================================================
    # Memory Cleanup
    # ========================================================================
    
    print("\n\nMemory Cleanup")
    print("-" * 80)
    
    # Clean up GPU memory
    if actual_device == 'cuda':
        del model
        del trainer
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        print("  [ok] GPU memory cleared")
    
    print_memory_usage("After Training Complete")

    print("\n\nTraining Pipeline Complete")
    print("-" * 80)


def main():
    """
    Main entry point for training with hyperparameter grid search.
    
    Trains all combinations of datasets, dictionary sizes, and k values using sklearn's ParameterGrid.
    """

    print("\n" + "="*80)
    print("TRAINING MATRYOSHKA SPARSE AUTOENCODER WITH HYPERPARAMETER GRID SEARCH")
    print("="*80)

    # Define embedding models with their input dimension and train/val paths
    EMBEDDING_MODELS = {
        'biomedparse': {
            'd_input': 1536,
            'train': "data/total_seg_biomedparse_encodings/total_seg_biomedparse_encodings_train.parquet",
            'val':   "data/total_seg_biomedparse_encodings/total_seg_biomedparse_encodings_valid.parquet",
        },
        'biomedparse_random': {
            'd_input': 1536,
            'train': "data/total_seg_biomedparse_random_encodings/total_seg_biomedparse_random_encodings_train.parquet",
            'val':   "data/total_seg_biomedparse_random_encodings/total_seg_biomedparse_random_encodings_valid.parquet",
        },
        'dinov3': {
            'd_input': 1024,
            'train': "data/total_seg_dinov3_encodings/total_seg_dinov3_encodings_train.parquet",
            'val':   "data/total_seg_dinov3_encodings/total_seg_dinov3_encodings_valid.parquet",
        },
    }
    
    # Define hyperparameter grid
    param_grid = {
        'embedding_model': list(EMBEDDING_MODELS.keys()),
        'dictionary_sizes': [
            [16, 64, 256, 1024],
            [32, 128, 512, 2048],
            [64, 256, 1024, 4096],
            [128, 512, 2048, 8192]
        ],
        'k_values': [
            [5, 5, 5, 5],
            [10, 10, 10, 10],
            [20, 20, 20, 20],
            [30, 30, 30, 30],
            [5, 10, 20, 40],
            [10, 20, 40, 80],
            [20, 40, 80, 160],
            [30, 60, 120, 240],
        ],
    }
    
    # Generate all combinations using sklearn's ParameterGrid
    grid = list(ParameterGrid(param_grid))
    total_configs = len(grid)
    
    print(f"\nParameter Grid ({total_configs:d} configurations):")
    for param in param_grid.keys():
        print(f"  {param:s}:")
        for value in param_grid[param]:
            print(f"    - {value}")
    
    for config_num, params in tqdm(enumerate(grid, 1), total=total_configs, desc="Training configurations"):
        embedding_model = params['embedding_model']
        dict_sizes = params['dictionary_sizes']
        k_vals = params['k_values']
        
        # Validate k_values against dictionary_sizes
        if any(k > d for k, d in zip(k_vals, dict_sizes)):
            print(f"\n    !!! Skipping invalid config #{config_num}/{total_configs}:")
            print(f"    Dictionary sizes: {dict_sizes}")
            print(f"    K values: {k_vals}")
            print(f"    Reason: k must be <= dictionary size for each level")
            continue

        model_config = EMBEDDING_MODELS[embedding_model]
        d_input = model_config['d_input']
        train_path = model_config['train']
        val_path = model_config['val']
        
        # Generate descriptive log and config name
        log_subdir = f"{embedding_model:s}_matryoshka_sae"
        config_name = f"D_{'_'.join(map(str, dict_sizes))}_K_{'_'.join(map(str, k_vals))}"
        
        print(f"\n\n{'='*80}")
        print(f"TRAINING CONFIG #{config_num:d}/{total_configs:d}")
        print(f"{'='*80}")
        print(f"  Config name: {config_name:s}")
        print(f"  Embedding model: {embedding_model:s}")
        print(f"  Dictionary sizes: {dict_sizes}")
        print(f"  K values: {k_vals}")
        print(f"  Input dimension: {d_input:d}")
        print(f"  Train path: {train_path:s}")
        print(f"  Val path: {val_path:s}")
        
        train_matryoshka_sae(
            dictionary_sizes=dict_sizes,
            k_values=k_vals,
            train_parquet=train_path,
            val_parquet=val_path,
            d_input=d_input,
            log_subdir=log_subdir,
            config_name=config_name
        )
        
        # Force garbage collection between configurations
        print(f"\n  [ok] Cleaned up memory after config #{config_num}")
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    print(f"\n\n{'='*80}")
    print(f"COMPLETED TRAINING ALL {total_configs} CONFIGURATIONS!")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()