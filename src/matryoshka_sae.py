"""
Matryoshka Sparse Autoencoder (SAE) architecture and PyTorch Lightning training module.

Implements hierarchical nested dictionaries with BatchTopK activation during training
and a learnable JumpReLU threshold at inference. See Bussmann et al. (2025).
"""

# stdlib
from typing import Any

# third-party
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import r2_score
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR


class MatryoshkaSAE(nn.Module):
    """
    Matryoshka Sparse Autoencoder (SAE)
    
    This implementation trains multiple nested dictionaries simultaneously,
    where each nested sub-SAE reconstructs the input using only a subset
    of the total latents. This creates a hierarchy where early latents
    capture general features and later latents capture specific features.
    
    Uses BatchTopK during training for flexible per-sample sparsity, and
    threshold-based activation during inference (following BatchTopK paper).
    
    Based on: 
    - "Learning Multi-Level Features with Matryoshka Sparse Autoencoders"
      (Bussmann et al., 2025) - arXiv:2503.17547
    - "BatchTopK Sparse Autoencoders" 
      (Bussmann et al., 2024) - arXiv:2412.06410
    """
    
    def __init__(
        self,
        input_dim: int,
        dictionary_sizes: list[int],
        tied_weights: bool = True,
        normalize_decoder: bool = True,
        inference_threshold: float | torch.Tensor | None = None
    ):
        """
        Args:
            input_dim: Dimension of input activations
            dictionary_sizes: List of nested dictionary sizes (e.g., [512, 2048, 8192])
                             Must be in increasing order
            tied_weights: Whether to tie encoder and decoder weights
            normalize_decoder: Whether to normalize decoder columns to unit norm
            inference_threshold: Threshold for inference mode. If None, will be estimated
                               during training via running average
        """
        super().__init__()
        
        # Sanity check: dictionary sizes must be in increasing order
        assert all(dictionary_sizes[i] < dictionary_sizes[i+1] for i in range(len(dictionary_sizes)-1)), "Dictionary sizes must be in increasing order"
        
        self.input_dim = input_dim
        self.dictionary_sizes = dictionary_sizes
        self.max_dict_size = max(dictionary_sizes)
        self.tied_weights = tied_weights
        self.normalize_decoder = normalize_decoder

        # Inference threshold and its statistics
        self.inference_threshold: torch.Tensor
        self._threshold_sum: torch.Tensor
        self._threshold_count: torch.Tensor
        self._threshold_estimated = False
        
        # Register threshold and threshold statistics as buffer to persist in state_dict
        self.register_buffer('inference_threshold', torch.tensor(0.0))
        self.register_buffer('_threshold_sum', torch.tensor(0.0))
        self.register_buffer('_threshold_count', torch.tensor(0))
        
        # Use provided inference threshold if available
        if inference_threshold is not None:
            self.inference_threshold.data.fill_(inference_threshold)
            self._threshold_estimated = True

        # Encoder: maps input to latent codes
        self.encoder = nn.Linear(input_dim, self.max_dict_size, bias=True)
        
        # Decoder: reconstructs input from latent codes
        if tied_weights:
            # Tied weights: decoder is transpose of encoder
            self.decoder = None
        else:
            self.decoder = nn.Linear(self.max_dict_size, input_dim, bias=False)
        
        # Bias for reconstruction
        self.bias = nn.Parameter(torch.zeros(input_dim))
        
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize weights with Xavier uniform initialization"""
        nn.init.xavier_uniform_(self.encoder.weight)
        nn.init.zeros_(self.encoder.bias)
        
        if not self.tied_weights and self.decoder is not None:
            nn.init.xavier_uniform_(self.decoder.weight)
        
        if self.normalize_decoder:
            self._normalize_decoder_weights()
    
    def _normalize_decoder_weights(self):
        """Normalize decoder weight columns to unit norm"""
        with torch.no_grad():
            if self.tied_weights:
                # Normalize encoder weight rows (which become decoder columns)
                norms = torch.norm(self.encoder.weight.data, dim=1, keepdim=True)
                self.encoder.weight.data.div_(norms + 1e-8)
            elif self.decoder is not None:
                # Normalize decoder weight columns
                norms = torch.norm(self.decoder.weight.data, dim=0, keepdim=True)
                self.decoder.weight.data.div_(norms + 1e-8)
    
    def get_decoder_weight(self) -> torch.Tensor:
        """Get decoder weight matrix"""
        if self.tied_weights:
            return self.encoder.weight.data.t()
        elif self.decoder is not None:
            return self.decoder.weight.data
        else:
            raise RuntimeError("Decoder is None but tied_weights is False")
    
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Encode input to pre-activation latent codes
        
        Pre-activation codes are the raw, continuous outputs from the encoder
        before applying any activation function (ReLU, TopK, etc.). They represent
        the affinity/strength of each learned feature in the input, and can be
        positive, negative, or zero.
        
        These codes are computed once for the largest dictionary, then subsets
        are used for smaller nested dictionaries, which is more efficient than
        having separate encoders for each dictionary size.
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
               e.g., (32, 1532) for a batch of ViT features
            
        Returns:
            Pre-activation latent codes of shape (batch_size, max_dict_size)
            These are continuous values before sparsification
        """
        # Center the input by subtracting learned bias
        x_centered = x - self.bias
        
        # Apply linear transformation: z_pre = W_enc @ x_centered + b_enc
        # Each output dimension is a weighted sum of all input features
        latents = self.encoder(x_centered)
        
        return latents

    def batch_topk_activation(self, latents: torch.Tensor, k: int, update_threshold_stats: bool = False) -> torch.Tensor:
        """Keeps top-k activations on average per sample across the batch"""
        b, d = latents.shape

        # Number of top values to keep across the entire batch
        k_batch = k * b

        # Flatten along the batch dimension
        flat = latents.reshape(-1)  # shape (batch_size * dict_size,)

        # Get top-k values and indices
        topk_vals, topk_flat_idx = torch.topk(flat, k_batch, dim=0)

        # Track minimum of positive activations for JumpReLU inference
        if update_threshold_stats and self.training:
            positive_topk_vals = topk_vals[topk_vals > 0]
            if len(positive_topk_vals) > 0:
                min_val_kept = positive_topk_vals.min().detach() # Detach from computation graph to prevent memory accumulation
                self._threshold_sum += min_val_kept
                self._threshold_count += 1

                # Set the inference threshold to the running average of the minimum positive activation values
                self.inference_threshold.copy_(self._threshold_sum / self._threshold_count)
                
                # Mark threshold as estimated after accumulating enough statistics (100 batches for now)
                if self._threshold_count > 99:
                    if not self._threshold_estimated:
                        print(f"[ok] Inference threshold available after {self._threshold_count.item():d} batches. Current threshold: {self.inference_threshold.item():.6f}")
                    self._threshold_estimated = True

        # Create sparse tensor to store only top-k activations
        sparse_latents = torch.zeros_like(flat)

        # Fill the top-k values back to their original positions
        sparse_latents[topk_flat_idx] = topk_vals

        # Reshape back to (batch_size, dict_size)
        sparse_latents = sparse_latents.reshape(b, d)

        return sparse_latents

    def threshold_activation(self, latents: torch.Tensor, threshold: float) -> torch.Tensor:
        """Jump ReLU activation: zeros out activations below threshold"""
        return torch.clamp(latents - threshold, min=0.0)

    def per_sample_topk_activation(self, latents: torch.Tensor, k: int) -> torch.Tensor:
        """Keeps top-k activations per sample"""
        # Get top-k values and indices
        topk_values, topk_indices = torch.topk(latents, k, dim=-1)
        
        # Create sparse tensor to store only top-k activations
        sparse_latents = torch.zeros_like(latents)

        # Scatter the top-k values back to their original positions
        sparse_latents.scatter_(-1, topk_indices, topk_values)

        return sparse_latents

    def calculate_threshold(self) -> float:
        """
        Calculate the inference threshold from accumulated statistics.
        Should be called after training or before inference.
        """
        if self._threshold_count > 0:
            threshold = (self._threshold_sum / self._threshold_count).item()
            self.inference_threshold.data.fill_(threshold)
            self._threshold_estimated = True
            print(f"[ok] Calculated threshold: {threshold:.6f}")
            return threshold
        else:
            raise ValueError("No threshold statistics accumulated.")

    def decode(self, latents: torch.Tensor) -> torch.Tensor:
        """
        Decode latent codes back to input space
        
        Args:
            latents: Activated latent codes of shape (batch_size, max_dict_size)
            
        Returns:
            Reconstructed input of shape (batch_size, input_dim)
        """
        decoder_weight = self.get_decoder_weight()
        # decoder_weight shape: (input_dim, max_dict_size)
        # latents shape: (batch_size, max_dict_size)
        # We want: (batch_size, input_dim) = (batch_size, max_dict_size) @ (max_dict_size, input_dim)
        x_recon = F.linear(latents, decoder_weight) + self.bias
        
        return x_recon
    
    def forward(
        self, 
        x: torch.Tensor, 
        k_values: list[int] | None = None
    ) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """
        Forward pass with hierarchical nested reconstructions
        
        This implements the core Matryoshka SAE algorithm:
        1. Compute pre-activation codes once for the largest dictionary
        2. For each nested dictionary size, take a subset of features
        3. Apply sparsification (BatchTopK in training, threshold in inference)
        4. Reconstruct the input at each level
        
        Training mode: Uses BatchTopK for flexible per-sample sparsity
        Inference mode: Uses threshold-based activation (removes batch dependency).
                        Falls back to per-sample topk if threshold not estimated yet
                        (e.g., during sanity checks before training).
        
        Args:
            x: Input tensor of shape (batch_size, input_dim)
               e.g., (32, 1532) for a batch of ViT features
            k_values: List of k values (number of active features) for each 
                     nested dictionary level. Length must match len(dictionary_sizes).
                     If None, defaults to ~5% sparsity per level.
                     Only used in training mode.
                     
        Returns:
            Tuple of:
                - Final reconstruction using the largest dictionary (batch_size, input_dim)
                - List of nested reconstructions, one per dictionary size
                - List of nested sparse latent activations, one per level
        """
        batch_size = x.shape[0]
        
        # Default k values: use approximately 5% sparsity for each level
        if k_values is None:
            k_values = [max(1, int(0.05 * size)) for size in self.dictionary_sizes]
        
        # Compute pre-activation codes once for efficiency
        # Shape: (batch_size, max_dict_size) - continuous values before sparsification
        latents_pre = self.encode(x)
        
        reconstructions = []
        latent_activations = []
        
        # Process each nested dictionary level independently
        for dict_size, k in zip(self.dictionary_sizes, k_values):

            # Get the subset of latents for the current dictionary size
            latents_subset = latents_pre[:, :dict_size]
            
            if self.training:
                # Training: BatchTopK non-linear activation for flexible sparsity
                # Update threshold stats only for the largest dictionary (most representative)
                update_stats = (dict_size == self.max_dict_size)
                latents_active = self.batch_topk_activation(latents_subset, k, update_threshold_stats=update_stats)
            else:
                # Inference: Threshold-based JumpReLU non-linear activation
                if self._threshold_estimated:
                    latents_active = self.threshold_activation(latents_subset, self.inference_threshold.item())
                else:
                    # Fallback to per-sample topk if threshold not estimated yet
                    # This allows sanity checks and validation during early training
                    print("WARNING: No activation threshold available (yet). Falling back to per-sample topk")
                    latents_active = self.per_sample_topk_activation(latents_subset, k)
            
            # Pad with zeros to match max dictionary size for decoding
            # This allows us to use a single decoder for all levels
            if dict_size < self.max_dict_size:
                padding = torch.zeros(
                    batch_size, 
                    self.max_dict_size - dict_size,
                    device=latents_active.device
                )
                latents_active = torch.cat([latents_active, padding], dim=-1)
            
            # Reconstruct input from sparse activations
            x_recon = self.decode(latents_active)
            
            reconstructions.append(x_recon)
            latent_activations.append(latents_active)
        
        # Return the best reconstruction (from largest dictionary)
        final_reconstruction = reconstructions[-1]
        
        return final_reconstruction, reconstructions, latent_activations


class MatryoshkaSAELightning(L.LightningModule):
    """
    PyTorch Lightning wrapper for training Matryoshka Sparse Autoencoder.
    
    To reduce memory usage, correlation metrics are only computed on a random sample of batches at each epoch end,
    Since full correlation analysis otherwise consumes excessive GPU/CPU memory.
    
    Args:
        input_dim: Dimension of input activations
        dictionary_sizes: List of nested dictionary sizes
        tied_weights: Whether to tie encoder and decoder weights
        normalize_decoder: Whether to normalize decoder columns
        l1_coefficient: Weight for L1 sparsity penalty
        diversity_coefficient: Weight for feature diversity penalty
        learning_rate: Learning rate for optimizer
        k_values: List of k values for each dictionary level (None for auto)
        correlation_sample_batches: Number of batches to sample per epoch for correlation metrics (default: 20)
                                   Set to 0 to disable correlation metrics entirely.
    """
    def __init__(
        self,
        input_dim: int,
        dictionary_sizes: list[int],
        tied_weights: bool = True,
        normalize_decoder: bool = True,
        l1_coefficient: float = 0.0,
        diversity_coefficient: float = 0.0,
        learning_rate: float = 1e-3,
        k_values: list[int] | None = None,
        correlation_sample_batches: int = 20
    ):
        """
        Args:
            input_dim: Dimension of input activations
            dictionary_sizes: List of nested dictionary sizes
            tied_weights: Whether to tie encoder and decoder weights
            normalize_decoder: Whether to normalize decoder columns
            l1_coefficient: Weight for L1 sparsity penalty
            diversity_coefficient: Weight for feature diversity penalty
            learning_rate: Learning rate for optimizer
            k_values: List of k values for each dictionary level (None for auto)
            correlation_sample_batches: Number of batches to sample per epoch for correlation metrics (default: 20)
                                       Set to 0 to disable correlation metrics entirely.
        """
        super().__init__()
        self.save_hyperparameters()
        
        # Create the core SAE model
        self.model = MatryoshkaSAE(
            input_dim=input_dim,
            dictionary_sizes=dictionary_sizes,
            tied_weights=tied_weights,
            normalize_decoder=normalize_decoder
        )
        
        self.l1_coefficient = l1_coefficient
        self.diversity_coefficient = diversity_coefficient
        self.learning_rate = learning_rate
        self.k_values = k_values
        self.correlation_sample_batches = correlation_sample_batches
        
        # Buffers for accumulating latents for epoch-end correlation metrics
        self.train_latents_buffer = []
        self.val_latents_buffer = []
        
        # Track which batches to sample for correlation (computed at epoch start)
        self._train_sample_indices = set()
        self._val_sample_indices = set()
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, list[torch.Tensor], list[torch.Tensor]]:
        """Forward pass through the model"""
        return self.model(x, k_values=self.k_values)
    
    def _generate_correlation_sample_indices(self, num_batches: int) -> set:
        """Generate random batch indices to sample for correlation metrics."""
        if self.correlation_sample_batches == 0 or num_batches == 0:
            return set()
        
        import random
        # Sample up to correlation_sample_batches, but not more than total batches
        sample_size = min(self.correlation_sample_batches, num_batches)
        return set(random.sample(range(num_batches), sample_size))
    
    def compute_loss(
        self,
        x: torch.Tensor,
        reconstructions: list[torch.Tensor],
        latent_activations: list[torch.Tensor]
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """       
        The loss has two components for each nested level:
        1. Reconstruction loss (MSE)
        2. L1 sparsity penalty on activations
        3. Feature diversity loss (orthogonalize encoder weights)
        """
        total_recon_loss = torch.tensor(0.0, device=x.device)
        total_l1_loss = torch.tensor(0.0, device=x.device)
        diversity_loss = torch.tensor(0.0, device=x.device)

        # Compute loss for each nested level
        for i, (recon, latents) in enumerate(zip(reconstructions, latent_activations)):
            # Reconstruction loss (MSE)
            recon_loss = F.mse_loss(recon, x)
            total_recon_loss = total_recon_loss + recon_loss
            
            # L1 sparsity loss (average number of active features per sample)
            if self.l1_coefficient > 0.0:
                l1_loss = torch.abs(latents).sum(dim=-1).mean()
                total_l1_loss = total_l1_loss + l1_loss
            
        # Feature diversity loss (orthogonalize encoder weights)
        # Note: Outside of loop because encoder weights are shared across levels
        if self.diversity_coefficient > 0.0:
            encoder_weights = self.model.encoder.weight  # Get encoder weights
            W_norm = F.normalize(encoder_weights, p=2, dim=1)  # Normalize encoder weights
            similarity_matrix = torch.mm(W_norm, W_norm.t())  
            diversity_loss = (torch.triu(similarity_matrix, diagonal=1) ** 2).mean()  # Penalize high off-diagonal similarities using L2 penalty
        
        # Average over nested levels
        num_levels = len(reconstructions)
        avg_recon_loss = total_recon_loss / num_levels
        avg_l1_loss = total_l1_loss / num_levels

        # Total loss
        total_loss = avg_recon_loss + self.l1_coefficient * avg_l1_loss + self.diversity_coefficient * diversity_loss
        
        loss_dict = {
            'total_loss': total_loss,
            'reconstruction_loss': avg_recon_loss,
            'l1_loss': avg_l1_loss,
            'diversity_loss': diversity_loss
        }
        
        return total_loss, loss_dict
    
    def training_step(self, batch, batch_idx):
        """Training step"""
        # Extract embeddings from batch dict
        x = batch['embedding'] if isinstance(batch, dict) else batch
        batch_size = x.shape[0]
        
        # Check input statistics
        if self.global_step == 0:  # Only print on very first batch
            print("\n" + "="*80)
            print("INPUT STATISTICS (First Training Batch)")
            print("="*80)
            print(f"Input shape: {x.shape}")
            print(f"Input mean: {x.mean():.6f}")
            print(f"Input std: {x.std():.6f}")
            print(f"Input variance: {x.var():.6f}")
            print(f"Input min: {x.min():.6f}")
            print(f"Input max: {x.max():.6f}")
            print(f"Input median: {x.median():.6f}")
            print(f"% zeros: {(x == 0).float().mean():.2%}")
            print(f"% near-zero (< 0.001): {(x.abs() < 0.001).float().mean():.2%}")
            print(f"Current L1 coefficient: {self.l1_coefficient}")
            print("="*80 + "\n")

        # Forward pass
        final_recon, all_recons, all_latents = self(x)
        
        # Compute loss
        loss, loss_dict = self.compute_loss(x, all_recons, all_latents)
        
        # Log overall metrics
        self.log('train/loss', loss_dict['total_loss'], prog_bar=True, batch_size=batch_size)
        self.log('train/reconstruction_loss', loss_dict['reconstruction_loss'], batch_size=batch_size)
        self.log('train/l1_loss', loss_dict['l1_loss'], batch_size=batch_size)
        self.log('train/diversity_loss', loss_dict['diversity_loss'], batch_size=batch_size)
        self.log('train/threshold_estimate', self.model.inference_threshold.item(), batch_size=batch_size)
        
        # Log per-level metrics
        for i, (recon, latents) in enumerate(zip(all_recons, all_latents)):
            
            # -------------------------------------------------------------
            # 1. RECONSTRUCTION QUALITY
            # -------------------------------------------------------------

            # Reconstruction loss (MSE)
            recon_loss_level = F.mse_loss(recon, x)
            self.log(f'train/recon_loss_level_{i}', recon_loss_level, batch_size=batch_size)
            
            # Variance explained (primary metric used by Bussmann et al. in Matryoshka SAE paper)
            var_explained = 1 - ((x - recon).var() / x.var())
            self.log(f'train/var_explained_level_{i}', var_explained, batch_size=batch_size)

            # R^2 score
            x_flat = x.reshape(-1)
            recon_flat = recon.reshape(-1)
            ss_res = ((x_flat - recon_flat) ** 2).sum()
            ss_tot = ((x_flat - x_flat.mean()) ** 2).sum()
            r2_level = 1 - (ss_res / ss_tot)
            self.log(f'train/r2_level_{i}', r2_level, batch_size=batch_size)
        
            # -------------------------------------------------------------
            # 2. SPARSITY METRICS
            # -------------------------------------------------------------
            
            # L0 norm: average number of active features per sample
            l0 = (latents != 0).float().sum(dim=-1).mean()
            self.log(f'train/l0_level_{i}', l0, batch_size=batch_size)
            
            dict_size = self.model.dictionary_sizes[i]
            latents_subset = latents[:, :dict_size]
            
            # Alive feature count: features that activated at least once in this batch
            alive_count = (latents_subset > 1e-8).any(dim=0).sum().float()
            self.log(f'train/alive_features_level_{i}', alive_count, batch_size=batch_size)
            
            # Dead feature count: features that never activated in this batch
            dead_count = (latents_subset.abs() <= 1e-8).all(dim=0).sum().float()
            self.log(f'train/dead_features_level_{i}', dead_count, batch_size=batch_size)
        
            # Feature usage in this batch
            feature_usage = alive_count / dict_size
            self.log(f'train/feature_usage_level_{i}', feature_usage, batch_size=batch_size)
        
        # Store latents for epoch-end correlation computation (sampled batches only)
        # Only store for randomly sampled batches to reduce memory usage
        if self.correlation_sample_batches > 0 and batch_idx in self._train_sample_indices:
            # Detach to avoid memory accumulation
            self.train_latents_buffer.append([lat.detach() for lat in all_latents])
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        """Validation step"""
        # Extract embeddings from batch dict
        x = batch['embedding'] if isinstance(batch, dict) else batch
        batch_size = x.shape[0]

        # Check input statistics
        if self.global_step == 0:  # Only print on very first batch
            print("\n" + "="*80)
            print("INPUT STATISTICS (First Validation Batch)")
            print("="*80)
            print(f"Input shape: {x.shape}")
            print(f"Input mean: {x.mean():.6f}")
            print(f"Input std: {x.std():.6f}")
            print(f"Input variance: {x.var():.6f}")
            print(f"Input min: {x.min():.6f}")
            print(f"Input max: {x.max():.6f}")
            print(f"Input median: {x.median():.6f}")
            print(f"% zeros: {(x == 0).float().mean():.2%}")
            print(f"% near-zero (< 0.001): {(x.abs() < 0.001).float().mean():.2%}")
            print(f"Current L1 coefficient: {self.l1_coefficient}")
            print("="*80 + "\n")

        # Forward pass
        final_recon, all_recons, all_latents = self(x)
        
        # Compute loss
        loss, loss_dict = self.compute_loss(x, all_recons, all_latents)
        
        # Log overall metrics
        self.log('val/loss', loss_dict['total_loss'], prog_bar=True, batch_size=batch_size)
        self.log('val/reconstruction_loss', loss_dict['reconstruction_loss'], batch_size=batch_size)
        self.log('val/l1_loss', loss_dict['l1_loss'], batch_size=batch_size)
        self.log('val/diversity_loss', loss_dict['diversity_loss'], batch_size=batch_size)
        
        # Log per-level metrics
        for i, (recon, latents) in enumerate(zip(all_recons, all_latents)):
            
            # -------------------------------------------------------------
            # 1. RECONSTRUCTION QUALITY
            # -------------------------------------------------------------

            # Reconstruction loss (MSE)
            recon_loss_level = F.mse_loss(recon, x)
            self.log(f'val/recon_loss_level_{i}', recon_loss_level, batch_size=batch_size)
            
            # Variance explained (primary metric used by Bussmann et al. in Matryoshka SAE paper)
            var_explained = 1 - ((x - recon).var() / x.var())
            self.log(f'val/var_explained_level_{i}', var_explained, batch_size=batch_size)

            # R^2 score
            x_flat = x.reshape(-1)
            recon_flat = recon.reshape(-1)
            ss_res = ((x_flat - recon_flat) ** 2).sum()
            ss_tot = ((x_flat - x_flat.mean()) ** 2).sum()
            r2_level = 1 - (ss_res / ss_tot)
            self.log(f'val/r2_level_{i}', r2_level, batch_size=batch_size)

            # -------------------------------------------------------------
            # 2. SPARSITY METRICS
            # -------------------------------------------------------------
            
            # L0 norm: average number of active features per sample
            l0 = (latents != 0).float().sum(dim=-1).mean()
            self.log(f'val/l0_level_{i}', l0, batch_size=batch_size)
            
            dict_size = self.model.dictionary_sizes[i]
            latents_subset = latents[:, :dict_size]
            
            # Alive feature count: features that activated at least once in this batch
            alive_count = (latents_subset > 1e-8).any(dim=0).sum().float()
            self.log(f'val/alive_features_level_{i}', alive_count, batch_size=batch_size)
            
            # Dead feature count: features that never activated in this batch
            dead_count = (latents_subset.abs() <= 1e-8).all(dim=0).sum().float()
            self.log(f'val/dead_features_level_{i}', dead_count, batch_size=batch_size)
        
            # Feature usage in this batch
            feature_usage = alive_count / dict_size
            self.log(f'val/feature_usage_level_{i}', feature_usage, batch_size=batch_size)
        
        # Store latents for epoch-end correlation computation (sampled batches only)
        # Only store for randomly sampled batches to reduce memory usage
        if self.correlation_sample_batches > 0 and batch_idx in self._val_sample_indices:
            # Detach to avoid memory accumulation
            self.val_latents_buffer.append([lat.detach() for lat in all_latents])
        
        return loss
    
    def _log_correlation_metrics(self, latents_buffer: list[list[torch.Tensor]], prefix: str) -> None:
        """
        Compute monosemanticity metrics (feature correlations) from accumulated latents.
        
        Args:
            latents_buffer: List of batches, where each batch is a list of latent tensors per level
            prefix: Logging prefix ('train' or 'val')
        """
        if not latents_buffer:
            return
        
        # Process each dictionary level
        num_levels = len(latents_buffer[0])
        for i in range(num_levels):
            dict_size = self.model.dictionary_sizes[i]
            
            # Concatenate all batches for this level
            level_latents = torch.cat([batch[i][:, :dict_size] for batch in latents_buffer], dim=0)
            
            # Mean absolute correlation of active features
            active_mask = (level_latents.abs() > 1e-8).any(dim=0)  # Features active across epoch
            if active_mask.sum() > 1:  # Need at least 2 active features
                active_latents = level_latents[:, active_mask]  # (total_samples, num_active)
                
                # Pearson correlation matrix
                corr_matrix = torch.corrcoef(active_latents.T)  # (num_active, num_active)
                
                # Extract off-diagonal elements
                n_active = corr_matrix.shape[0]
                off_diag_mask = ~torch.eye(n_active, dtype=torch.bool, device=corr_matrix.device)
                off_diag_corrs = corr_matrix[off_diag_mask]
                
                mean_abs_corr = off_diag_corrs.abs().mean()
                self.log(f'{prefix}/mean_abs_corr_level_{i}', mean_abs_corr, sync_dist=True)
    
    def on_fit_start(self) -> None:
        """Capture and log trainer hyperparameters"""
        if self.trainer is not None:
            trainer_hparams = {}
            if hasattr(self.trainer, 'max_epochs') and self.trainer.max_epochs is not None:
                trainer_hparams['max_epochs'] = self.trainer.max_epochs
            if hasattr(self.trainer, 'gradient_clip_val') and self.trainer.gradient_clip_val is not None:
                trainer_hparams['gradient_clip_val'] = self.trainer.gradient_clip_val
            if trainer_hparams and self.logger is not None:
                self.logger.log_hyperparams(trainer_hparams)
        
        # Log correlation sampling configuration
        if self.correlation_sample_batches > 0:
            print(f"  - Correlation metrics: Sampling {self.correlation_sample_batches} batches/epoch (~{self.correlation_sample_batches*100//max(1, self.trainer.num_training_batches if hasattr(self.trainer, 'num_training_batches') else 1)}% of data)")
        else:
            print(f"  - Correlation metrics: Disabled")
    
    def on_train_epoch_start(self) -> None:
        """Generate random sample indices for correlation metrics at epoch start"""
        if self.trainer is not None and hasattr(self.trainer, 'num_training_batches'):
            num_batches = int(self.trainer.num_training_batches)
            self._train_sample_indices = self._generate_correlation_sample_indices(num_batches)
    
    def on_validation_epoch_start(self) -> None:
        """Generate random sample indices for correlation metrics at epoch start"""
        if self.trainer is not None and hasattr(self.trainer, 'num_val_batches'):
            # Get number of validation batches
            num_batches = int(self.trainer.num_val_batches[0]) if self.trainer.num_val_batches else 0
            self._val_sample_indices = self._generate_correlation_sample_indices(num_batches)

    def on_train_epoch_end(self) -> None:
        """Compute correlation metrics at the end of each training epoch"""
        if len(self.train_latents_buffer) > 0:
            self._log_correlation_metrics(self.train_latents_buffer, 'train')
        # Clear buffer to free memory
        self.train_latents_buffer.clear()
    
    def on_validation_epoch_end(self) -> None:
        """Compute correlation metrics at the end of each validation epoch"""
        if len(self.val_latents_buffer) > 0:
            self._log_correlation_metrics(self.val_latents_buffer, 'val')
        # Clear buffer to free memory
        self.val_latents_buffer.clear()
    
    def on_train_end(self) -> None:
        """
        Log final inference threshold after training completes
        """
        print("\n" + "="*80)
        print("ON TRAINING END")
        print("="*80 + "\n")
        print("Final inference threshold:")
        if self.model._threshold_estimated:
            print(f"[ok] Inference threshold estimated: {self.model.inference_threshold.item():.6f}")
            print(f"  (averaged over {self.model._threshold_count.item():.0f} batches)")
        else:
            print("WARNING: Inference threshold not estimated.")
            print("  Threshold will need to be estimated manually before inference.")
        print("\n" + "="*80)

    def configure_optimizers(self) -> Any:
        """Configure optimizer and learning rate scheduler"""
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        
        # Calculate from trainer
        estimated_steps = int(self.trainer.estimated_stepping_batches)
        warmup_steps = int(min(1000, estimated_steps // 10))  # Warmup for 10% or 1000 steps
        
        warmup_scheduler = LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=warmup_steps
        )
        
        cosine_scheduler = CosineAnnealingLR(
            optimizer,
            T_max=estimated_steps - warmup_steps,
            eta_min=1e-6
        )
        
        scheduler = SequentialLR(
            optimizer,
            schedulers=[warmup_scheduler, cosine_scheduler],
            milestones=[warmup_steps]
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'interval': 'step'
            }
        }