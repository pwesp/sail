"""
Shared encoding utilities for writing embeddings to parquet.
"""

import gc
import numpy as np
import pandas as pd
from pathlib import Path
import psutil
from tqdm import tqdm
import torch


def encode_dataset(
    datamodule,
    feature_extractor,
    output_path_str: str,
    write_batch_size: int,
    checkpoint_dir: str | None = None,
    total_images: int = 0,
):
    """
    Encode all images using a DataModule and save to parquet file.

    Args:
        datamodule: TotalSegmentatorEncodingDataModule instance.
        feature_extractor: Feature extractor with an extract_batch_features() method.
        output_path_str: Path to save final parquet file.
        write_batch_size: Number of samples per checkpoint batch file.
        checkpoint_dir: Directory for checkpoint batch files (default: output_path.parent / "encoding_batches").
        total_images: Total number of images (for progress bar).
    """
    output_path = Path(output_path_str)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if checkpoint_dir is None:
        checkpoint_dir_path = output_path.parent / "encoding_batches"
    else:
        checkpoint_dir_path = Path(checkpoint_dir)
    checkpoint_dir_path.mkdir(parents=True, exist_ok=True)

    existing_batches = sorted(checkpoint_dir_path.glob("batch_*.parquet"))
    start_idx = datamodule.start_idx
    encoding_complete = (start_idx >= total_images)

    if existing_batches:
        print(f"\n  Found {len(existing_batches)} existing batches")
        if encoding_complete:
            print(f"  Encoding already complete ({start_idx}/{total_images} images), skipping to combine step")
        else:
            print(f"  Resuming from index {start_idx}/{total_images}")

    if not encoding_complete:
        print(f"\nEncoding {total_images} images...")
        print(f"  Batch size: {datamodule.batch_size}")
        print(f"  Write batch size: {write_batch_size}")
        print(f"  DataLoader workers: {datamodule.num_workers}")
        print(f"  Checkpoint directory: {checkpoint_dir_path}")
        print(f"  Final output: {output_path}")

        dataloader = datamodule.predict_dataloader()
        row_buffer = []
        batch_file_idx = len(existing_batches)
        processed_count = 0
        batches_since_write = 0

        pbar = tqdm(total=total_images, initial=start_idx, desc="Encoding images")

        for batch in dataloader:
            try:
                embeddings = feature_extractor.extract_batch_features(batch['images'])

                for metadata_dict, embedding in zip(batch['metadata'], embeddings):
                    row_dict = metadata_dict.copy()
                    row_dict['embedding'] = embedding.tobytes()
                    row_dict['embedding_shape'] = str(embedding.shape)
                    row_dict['embedding_dtype'] = str(embedding.dtype)
                    row_buffer.append(row_dict)

                processed_count += len(batch['images'])
                batches_since_write += 1

                del embeddings
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                if batches_since_write % 50 == 0:
                    gc.collect()

            except Exception as e:
                print(f"\n  WARNING: Error processing batch: {e}")
                print(f"     Batch indices: {batch['original_indices']}")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            pbar.update(len(batch['images']))

            if len(row_buffer) >= write_batch_size:
                gc.collect()
                batch_file = checkpoint_dir_path / f"batch_{batch_file_idx:04d}.parquet"
                pd.DataFrame(row_buffer[:write_batch_size]).to_parquet(batch_file, engine='pyarrow', index=False)
                row_buffer = row_buffer[write_batch_size:]
                batch_file_idx += 1
                batches_since_write = 0

                if batch_file_idx % 10 == 0:
                    mem = psutil.Process().memory_info()
                    pbar.write(f"  Batch {batch_file_idx:04d} written | RSS: {mem.rss / (1024**3):.2f} GB")

        pbar.close()

        if row_buffer:
            batch_file = checkpoint_dir_path / f"batch_{batch_file_idx:04d}.parquet"
            pd.DataFrame(row_buffer).to_parquet(batch_file, engine='pyarrow', index=False)

        print(f"\n  Encoded {processed_count} images")

    # Combine batch files into final output
    if output_path.exists():
        print(f"\n  Final parquet already exists: {output_path} ({output_path.stat().st_size / (1024**2):.2f} MB)")
        return

    print(f"\nCombining batch files...")
    all_batches = sorted(checkpoint_dir_path.glob("batch_*.parquet"))
    print(f"  Found {len(all_batches)} batch files")

    if not all_batches:
        print("  WARNING: No batch files found!")
        return

    final_df = pd.concat(
        [pd.read_parquet(f) for f in tqdm(all_batches, desc="Reading batches")],
        ignore_index=True
    )
    final_df.to_parquet(output_path, engine='pyarrow', index=False)

    sample = np.frombuffer(final_df['embedding'].iloc[0], dtype=np.float32)
    print(f"\n  Encoding complete: {len(final_df)} rows, {output_path.stat().st_size / (1024**2):.2f} MB")
    print(f"  Embedding shape: {sample.shape}, mean: {sample.mean():.4f}, L2: {np.linalg.norm(sample):.4f}")
    print(f"  Batch files kept in: {checkpoint_dir_path} (delete after verification)")


def verify_parquet_file(parquet_path: str):
    """Load and print a summary of an embedding parquet file."""
    path = Path(parquet_path)
    if not path.exists():
        print(f"ERROR: File does not exist: {parquet_path}")
        return

    df = pd.read_parquet(parquet_path)
    print(f"\nVerification: {parquet_path}")
    print(f"  Rows: {len(df)}, Columns: {len(df.columns)}, Size: {path.stat().st_size / (1024**2):.2f} MB")

    display_cols = [c for c in df.columns if c not in ('embedding', 'embedding_shape', 'embedding_dtype')][:10]
    print(df[display_cols].head(3).to_string())

    for idx in range(min(3, len(df))):
        emb = np.frombuffer(df['embedding'].iloc[idx], dtype=np.float32)
        print(f"  Row {idx}: shape={emb.shape}, mean={emb.mean():.4f}, L2={np.linalg.norm(emb):.4f}")

    missing = df.isnull().sum()
    missing = missing[missing > 0]
    if len(missing):
        print(f"  WARNING: Missing values: {missing.to_dict()}")
    else:
        print("  No missing values.")
