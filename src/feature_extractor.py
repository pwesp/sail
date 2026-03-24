"""
Foundation model feature extractors for BiomedParse and DINOv3.

Wraps each model's forward pass to produce a single global embedding vector
per image, as consumed by the SAE training and encoding pipeline.
"""

# stdlib
import numpy as np
import torch
import torch.nn.functional as F
from transformers import AutoImageProcessor, AutoModel


class GlobalFeatureExtractor():
    """
    Extract global feature vectors from BiomedParse backbone using pooling.
    
    Designed to work with TotalSegmentatorEncodingDataModule which provides
    batches of torch tensors (HWC, RGB, float32) from the dataset.
    
    Normalization parameters from BiomedParse configuration:
    - BiomedParse uses a Focal Transformer (FocalNet) backbone with ImageNet normalization.
    - From BiomedParse/configs/biomedparse_inference.yaml:
        INPUT:
          PIXEL_MEAN: [123.675, 116.280, 103.530]  # RGB order
          PIXEL_STD: [58.395, 57.120, 57.375]      # ImageNet std
    
    Note: BiomedParse is built on Detectron2 but uses standard ImageNet normalization,
    not Detectron1/MSRA models (which would have std=[1.0, 1.0, 1.0] with absorbed conv1 weights).
    """
    
    def __init__(
        self,
        model,
        backbone,
        input_size: tuple[int, int] = (1024, 1024),
        pooling: str = 'avg'
    ):
        self.model = model
        self.backbone = backbone
        self.input_size = input_size
        self.pooling = pooling 
        self.features = {}
        
        # Register hooks
        self._register_hooks()
        
    def _register_hooks(self) -> None:
        """Hook into the final norm layer to extract features."""
        def hook(module, input, output) -> None:
            self.features['norm3'] = output.detach()
        self.backbone.norm3.register_forward_hook(hook)
        print(f"[ok] Hooked into norm3")
    
    def _preprocess_batch(self, tensors: list[torch.Tensor]) -> torch.Tensor:
        """
        Convert batch of torch tensors (HWC, RGB, float32) to normalized batch.
        
        Args:
            tensors: List of torch tensors in HWC format, RGB, float32
            
        Returns:
            Normalized batch tensor in BCHW format, resized to input_size if needed
        """
        # Stack tensors and permute to BCHW
        batch_tensor = torch.stack(tensors, dim=0)  # (B, H, W, C)
        batch_tensor = batch_tensor.permute(0, 3, 1, 2)  # (B, H, W, C) -> (B, C, H, W)
        
        # Resize to expected input size if needed (before moving to GPU for efficiency)
        current_size = (batch_tensor.shape[2], batch_tensor.shape[3])  # (H, W)
        if current_size != self.input_size:
            batch_tensor = F.interpolate(
                batch_tensor, 
                size=self.input_size,
                mode='bilinear',
                align_corners=False,
                antialias=True  # Better quality downsampling
            )
        
        # Get normalization parameters
        pixel_mean = torch.tensor([123.675, 116.28, 103.53]).view(1, -1, 1, 1)
        pixel_std = torch.tensor([58.395, 57.12, 57.375]).view(1, -1, 1, 1)
        
        if torch.cuda.is_available():
            batch_tensor = batch_tensor.cuda()
            pixel_mean = pixel_mean.cuda()
            pixel_std = pixel_std.cuda()
        
        # Perform normalization
        batch_tensor = (batch_tensor - pixel_mean) / pixel_std
        
        return batch_tensor
    
    def _pool_features(self, feature_map: torch.Tensor) -> torch.Tensor:
        """
        Pool transformer features (B, L, C) -> (B, C).
        BiomedParse backbone outputs shape: (batch, num_patches, channels)
        Pool over the patch dimension (dim=1) to get global vector
        """
        if self.pooling == 'avg':
            return feature_map.mean(dim=1)
        elif self.pooling == 'max':
            return feature_map.amax(dim=1)
        else:
            raise ValueError(f"Invalid pooling method: {self.pooling}")

    def extract_batch_features(self, tensors: list[torch.Tensor]) -> np.ndarray:
        """
        Extract global feature vectors from batch of tensors.
        
        Args:
            tensors: List of torch Tensors (HWC, RGB, float32)
            
        Returns:
            numpy array (batch_size, dim) containing the global embeddings
        """
        self.features = {}
        
        # Preprocess batch
        batch_tensor = self._preprocess_batch(tensors)
        
        # Forward through backbone only (avoid segmentation head)
        # The registered hook captures the output of the final norm layer
        with torch.no_grad():
            _ = self.backbone(batch_tensor)
        
        # Get the hooked features
        feature_map = self.features['norm3']  # (batch_size, num_patches, channels)
        
        # Pool spatial dimensions -> vectors
        global_vecs = self._pool_features(feature_map)  # (batch_size, channels)
        
        # Convert to numpy (moves to CPU)
        global_vecs = global_vecs.cpu().numpy()
        
        # Clear intermediate tensors to free GPU memory
        del batch_tensor, feature_map
        self.features = {}
        
        return global_vecs



class DINOv3GlobalFeatureExtractor():
    """
    Extract global feature vectors from DINOv3 backbone using pooling.
    
    Designed to work with TotalSegmentatorEncodingDataModule which provides
    batches of torch tensors (HWC, RGB, float32) from the dataset.
    """
    
    def __init__(
        self,
        model_name: str,
        input_size: tuple[int, int] = (1024, 1024),
        use_bfloat16: bool = True
    ):
        self.model_name = model_name
        self.input_size = input_size
        self.use_bfloat16 = use_bfloat16

        # Load model with bfloat16 if requested
        if use_bfloat16 and torch.cuda.is_available():
            self.model = AutoModel.from_pretrained(
                model_name, 
                device_map="auto",
                dtype=torch.bfloat16
            )
            self.dtype = torch.bfloat16
            print(f"[ok] Loaded model with bfloat16 precision")
        else:
            self.model = AutoModel.from_pretrained(
                model_name,
                device_map="auto"
            )
            self.dtype = torch.float32
            print(f"[ok] Loaded model with float32 precision")

        # Load image processor
        self.image_processor = AutoImageProcessor.from_pretrained(model_name)
    
    def extract_batch_features(self, tensors: list[torch.Tensor]) -> np.ndarray:
        """
        Extract global feature vectors from batch of tensors.
        
        Args:
            tensors: List of torch Tensors (HWC, RGB, float32)
            
        Returns:
            numpy array (batch_size, dim) containing the global embeddings
        """

        # Preprocess batch
        batch_tensor = self.image_processor(
            images=tensors,
            return_tensors="pt",
            size={"height": self.input_size[0], "width": self.input_size[1]}
        ).to(self.model.device)
        
        # Convert to bfloat16 if enabled
        if self.use_bfloat16:
            batch_tensor['pixel_values'] = batch_tensor['pixel_values'].to(self.dtype)
        
        # Forward pass
        with torch.inference_mode():
            output = self.model(batch_tensor['pixel_values'])
                
        # Get global embeddings and convert to numpy (convert to float32 for numpy)
        global_vecs = output.pooler_output.float().cpu().numpy()
        
        # Clear intermediate tensors to free GPU memory
        del batch_tensor, output
        
        return global_vecs