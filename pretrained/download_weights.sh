#!/usr/bin/env bash
# Download pretrained SAIL SAE weights from Hugging Face (pwesp/sail).
# Run from the repository root: bash pretrained/download_weights.sh
#
# Places checkpoints directly in the lightning_logs/ structure expected by
# compute_sparse_feature_activations.py, together with hparams.yaml files
# so the script can discover and load them automatically (Step 5).

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# ---------------------------------------------------------------------------
# Hugging Face URLs
# ---------------------------------------------------------------------------
BIOMEDPARSE_URL="https://huggingface.co/pwesp/sail/resolve/main/biomedparse_sae.ckpt"
DINOV3_URL="https://huggingface.co/pwesp/sail/resolve/main/dinov3_sae.ckpt"

# ---------------------------------------------------------------------------
# BiomedParse — optimal config D_128_512_2048_8192_K_20_40_80_160
# ---------------------------------------------------------------------------
BMP_LOG_DIR="$REPO_ROOT/lightning_logs/biomedparse_matryoshka_sae/D_128_512_2048_8192_K_20_40_80_160"
BMP_CKPT_DIR="$BMP_LOG_DIR/checkpoints"

mkdir -p "$BMP_CKPT_DIR"

cat > "$BMP_LOG_DIR/hparams.yaml" <<'EOF'
input_dim: 1536
dictionary_sizes: [128, 512, 2048, 8192]
k_values: [20, 40, 80, 160]
train_parquet: data/total_seg_biomedparse_encodings/total_seg_biomedparse_encodings_train.parquet
val_parquet: data/total_seg_biomedparse_encodings/total_seg_biomedparse_encodings_valid.parquet
EOF

echo "Downloading BiomedParse SAE weights..."
curl -L --progress-bar -o "$BMP_CKPT_DIR/last.ckpt" "$BIOMEDPARSE_URL"
echo "Saved to $BMP_CKPT_DIR/last.ckpt"

# ---------------------------------------------------------------------------
# DINOv3 — optimal config D_128_512_2048_8192_K_5_10_20_40
# ---------------------------------------------------------------------------
DINO_LOG_DIR="$REPO_ROOT/lightning_logs/dinov3_matryoshka_sae/D_128_512_2048_8192_K_5_10_20_40"
DINO_CKPT_DIR="$DINO_LOG_DIR/checkpoints"

mkdir -p "$DINO_CKPT_DIR"

cat > "$DINO_LOG_DIR/hparams.yaml" <<'EOF'
input_dim: 1024
dictionary_sizes: [128, 512, 2048, 8192]
k_values: [5, 10, 20, 40]
train_parquet: data/total_seg_dinov3_encodings/total_seg_dinov3_encodings_train.parquet
val_parquet: data/total_seg_dinov3_encodings/total_seg_dinov3_encodings_valid.parquet
EOF

echo "Downloading DINOv3 SAE weights..."
curl -L --progress-bar -o "$DINO_CKPT_DIR/last.ckpt" "$DINOV3_URL"
echo "Saved to $DINO_CKPT_DIR/last.ckpt"

echo ""
echo "Done. Pretrained weights placed in lightning_logs/:"
echo "  $BMP_CKPT_DIR/last.ckpt"
echo "  $DINO_CKPT_DIR/last.ckpt"
echo ""
echo "compute_sparse_feature_activations.py will discover thempwesp/sail