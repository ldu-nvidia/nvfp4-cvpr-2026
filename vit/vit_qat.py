"""
Vision Transformer QAT Training Module
======================================

Self-contained module for Quantization-Aware Training (QAT) with Vision Transformer.
Designed for MRI per-pixel defect detection (LGG Brain Tumor Segmentation).

SUPPORTED RECIPES (9 total):
  BASE RECIPES:
    - 0:     Baseline (FP16/BF16/FP32, no quantization)
    - 1:  NVFP4 Full Quantization (forward + backward quantized)
    - 2: Forward-Only (backward uses original X, W, no gradient quantization)
    - 3: Chain Rule (backward reuses Q(X), Q(W) from forward, no gradient quantization)
  
  ADVANCED RECIPES:
    - 8:  NVFP4 Autograd + nvfp4 (autograd backend, nvfp4 casting)
    - 4:  NVFP4 2D Weights + RHT (2D blockwise + Random Hadamard Transform on WGRAD)
    - 5:  NVFP4 2D + RHT + SR (4 + Stochastic Rounding on gradients - best accuracy)
    - 6:  NVFP4 + SR Only (Stochastic Rounding without RHT - tests SR impact alone)
    - 7: Forward-Only + RHT (forward-only with Random Hadamard Transform)

MODEL SIZES:
    - pico:  ~500K params (CNN-equivalent for fair comparison)
    - nano:  ~1.5M params
    - micro: ~3M params
    - tiny:  ~6M params
    - small: ~22M params
    - base:  ~86M params
    - large: ~300M params

USAGE IN NOTEBOOK:
    import sys
    # NVFP4 QAT recipes require a compatible runtime environment.
    from vit_qat import (
        ViTQAT,
        create_vit_qat,
        run_vit_qat_training,
        SUPPORTED_RECIPES,
        MODEL_CONFIGS,
        print_supported_recipes,
        print_model_configs
    )
    
    # Print all supported recipes
    print_supported_recipes()
    
    # Print model configurations
    print_model_configs()
    
    # Create model
    model = create_vit_qat(
        model_size='pico',  # CNN-equivalent
        recipe_id=1,     # NVFP4 Full
        img_size=256,
        in_chans=3,
        num_classes=2
    )
    
    # Train with multiple recipes
    results = run_vit_qat_training(
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        model_size='pico',
        recipes=[0, 1, 2, 3],
        num_epochs=300
    )

Authors: Zijian Du and Oleg Rybakov
"""

import os
import sys
import copy
import time
import math
import warnings
from typing import Optional, Tuple, Dict, List, Any
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Import unified loss functions from common module
_COMMON_PATH = Path(__file__).parent.parent / "common"
if str(_COMMON_PATH) not in sys.path:
    sys.path.insert(0, str(_COMMON_PATH))
from loss import dice_loss, dice_coef_metric

# ═══════════════════════════════════════════════════════════════════════════════
# MODEL CONFIGURATIONS
# ═══════════════════════════════════════════════════════════════════════════════

# Predefined scale configurations (parallel to CNN's CNN_SCALES)
# MATCHED with CNN for fair comparison experiments
# NOTE: Smaller models to avoid AUC saturation (Jan 2026 revision)
VIT_SCALES = {
    # ═══════════════════════════════════════════════════════════════════════════
    # MATCHED SCALES (use these for CNN vs ViT comparison)
    # Revised Jan 2026: Smaller models to get more QAT recipe separation
    # ═══════════════════════════════════════════════════════════════════════════
    "small":  "matched_100k",   # ~100K params (matches CNN small)
    "medium": "matched_300k",   # ~300K params (matches CNN medium)
    "large":  "matched_1m",     # ~1M params (matches CNN large)
}

MODEL_CONFIGS = {
    # ═══════════════════════════════════════════════════════════════════════════
    # MATCHED CONFIGS (designed to match CNN CNN parameter counts exactly)
    # For fair ViT vs CNN architectural comparison at same param count
    # ═══════════════════════════════════════════════════════════════════════════
    'matched_500k': {
        'embed_dim': 64,
        'depth': 10,
        'num_heads': 2,
        'mlp_ratio': 3.0,
        'patch_size': 16,
        'description': '~530K params (matches CNN 4stg/64ch)',
    },
    'matched_4m': {
        'embed_dim': 192,
        'depth': 9,
        'num_heads': 6,
        'mlp_ratio': 3.0,
        'patch_size': 16,
        'description': '~3.7M params (matches CNN 6stg/128ch)',
    },
    'matched_15m': {
        'embed_dim': 448,
        'depth': 8,
        'num_heads': 14,
        'mlp_ratio': 2.0,
        'patch_size': 16,
        'description': '~13.7M params (matches CNN 7stg/224ch)',
    },
    
    # ═══════════════════════════════════════════════════════════════════════════
    # OLDER CONFIGS (kept for backward compatibility)
    # ═══════════════════════════════════════════════════════════════════════════
    'matched_100k': {
        'embed_dim': 40,
        'depth': 3,
        'num_heads': 2,
        'mlp_ratio': 2.0,
        'patch_size': 16,
        'description': '~85K params',
    },
    'matched_300k': {
        'embed_dim': 80,
        'depth': 4,
        'num_heads': 4,
        'mlp_ratio': 2.0,
        'patch_size': 16,
        'description': '~300K params',
    },
    'matched_1m': {
        'embed_dim': 128,
        'depth': 6,
        'num_heads': 4,
        'mlp_ratio': 3.0,
        'patch_size': 16,
        'description': '~1M params',
    },
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LEGACY CONFIGS (kept for backward compatibility / reference)
    # ═══════════════════════════════════════════════════════════════════════════
    'legacy_500k': {
        'embed_dim': 96,
        'depth': 4,
        'num_heads': 3,
        'mlp_ratio': 2.0,
        'patch_size': 16,
        'description': '~500K params (legacy small)',
    },
    'matched_3m': {
        'embed_dim': 192,
        'depth': 6,
        'num_heads': 6,
        'mlp_ratio': 3.0,
        'patch_size': 16,
        'description': '~3M params (legacy medium)',
    },
    'matched_10m': {
        'embed_dim': 256,
        'depth': 10,
        'num_heads': 8,
        'mlp_ratio': 4.0,
        'patch_size': 16,
        'description': '~10M params (legacy large)',
    },
    
    # ═══════════════════════════════════════════════════════════════════════════
    # STANDARD VIT CONFIGS (for reference / other experiments)
    # ═══════════════════════════════════════════════════════════════════════════
    'pico': {
        'embed_dim': 96,
        'depth': 4,
        'num_heads': 3,
        'mlp_ratio': 2.0,
        'patch_size': 16,
        'description': '~500K params',
    },
    'nano': {
        'embed_dim': 128,
        'depth': 6,
        'num_heads': 4,
        'mlp_ratio': 2.0,
        'patch_size': 16,
        'description': '~1.5M params',
    },
    'micro': {
        'embed_dim': 192,
        'depth': 6,
        'num_heads': 6,
        'mlp_ratio': 3.0,
        'patch_size': 16,
        'description': '~3M params',
    },
    'tiny': {
        'embed_dim': 192,
        'depth': 12,
        'num_heads': 3,
        'mlp_ratio': 4.0,
        'patch_size': 16,
        'description': '~6M params (ViT-Ti equivalent)',
    },
    'small_vit': {
        'embed_dim': 384,
        'depth': 12,
        'num_heads': 6,
        'mlp_ratio': 4.0,
        'patch_size': 8,
        'description': '~22M params (ViT-S equivalent)',
    },
    'base_vit': {
        'embed_dim': 768,
        'depth': 12,
        'num_heads': 12,
        'mlp_ratio': 4.0,
        'patch_size': 16,
        'description': '~86M params (ViT-B equivalent)',
    },
    'large_vit': {
        'embed_dim': 1024,
        'depth': 24,
        'num_heads': 16,
        'mlp_ratio': 4.0,
        'patch_size': 16,
        'description': '~300M params (ViT-L equivalent)',
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# SUPPORTED RECIPES
# ═══════════════════════════════════════════════════════════════════════════════

SUPPORTED_RECIPES = {
    # BASELINE AND QAT RECIPES
    0: {
        "name": "Baseline (FP16/BF16)",
        "description": "No quantization - standard mixed precision training",
        "forward_quantized": False,
        "backward_quantized": False,
        "requires_qat_runtime": False,
    },
    1: {
        "name": "NVFP4 Full Quantization",
        "description": "E2M1 quantization for forward (W, X) and backward (G, W, X)",
        "forward_quantized": True,
        "backward_quantized": True,
        "requires_qat_runtime": True,
    },
    2: {
        "name": "Forward-Only",
        "description": "E2M1 forward, BF16 backward (test backward quant impact)",
        "forward_quantized": True,
        "backward_quantized": False,
        "requires_qat_runtime": True,
    },
    3: {
        "name": "Chain Rule",
        "description": "Reuse forward quantized tensors in backward (memory optimization)",
        "forward_quantized": True,
        "backward_quantized": False,
        "requires_qat_runtime": True,
    },
    
    # ADVANCED QAT RECIPES
    8: {
        "name": "NVFP4 Autograd + nvfp4",
        "description": "E2M1 with autograd backend and nvfp4 casting (reference impl)",
        "forward_quantized": True,
        "backward_quantized": True,
        "requires_qat_runtime": True,
        "features": ["nvfp4_autograd", "nvfp4_cast"],
    },
    4: {
        "name": "NVFP4 2D Weights + RHT",
        "description": "2D blockwise (16x16) weight quantization + Random Hadamard Transform on WGRAD",
        "forward_quantized": True,
        "backward_quantized": True,
        "requires_qat_runtime": True,
        "features": ["2d_weights", "rht_wgrad"],
    },
    5: {
        "name": "NVFP4 2D + RHT + SR",
        "description": "4 + Stochastic Rounding on gradients (best accuracy recipe)",
        "forward_quantized": True,
        "backward_quantized": True,
        "requires_qat_runtime": True,
        "features": ["2d_weights", "rht_wgrad", "stochastic_rounding"],
    },
    6: {
        "name": "NVFP4 + SR Only",
        "description": "Stochastic Rounding only (no RHT, no 2D weights - isolates SR impact)",
        "forward_quantized": True,
        "backward_quantized": True,
        "requires_qat_runtime": True,
        "features": ["stochastic_rounding"],
    },
    7: {
        "name": "Forward-Only + RHT",
        "description": "Forward-only with Random Hadamard Transform on WGRAD",
        "forward_quantized": True,
        "backward_quantized": False,
        "requires_qat_runtime": True,
        "features": ["rht_wgrad", "forward_only"],
    },
}


# ═══════════════════════════════════════════════════════════════════════════════
# PUBLIC NVFP4 QAT RUNTIME COMPATIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

_QAT_RUNTIME_AVAILABLE = False
_QAT_RUNTIME_ADVANCED_AVAILABLE = False
_QAT_RUNTIME_QUIET = os.environ.get("QAT_RUNTIME_QUIET", "0") == "1"

PUBLIC_QAT_RUNTIME_MESSAGE = (
    "NVFP4 QAT recipes require a compatible NVFP4 quantization runtime. Vendor-specific runtime integration is intentionally omitted from this public release. Use recipe 0 for baseline training with the provided code."
)


def _qat_runtime_log(msg: str):
    """Print message only if not in quiet mode."""
    if not _QAT_RUNTIME_QUIET:
        print(msg)


def check_qat_runtime_available():
    """Raise a public-safe error for QAT recipes in this release."""
    raise RuntimeError(PUBLIC_QAT_RUNTIME_MESSAGE)


def detect_gpu_architecture():
    """Detect GPU architecture (compute capability)."""
    if not torch.cuda.is_available():
        return (0, 0, "No CUDA GPU")
    device = torch.cuda.current_device()
    props = torch.cuda.get_device_properties(device)
    return (props.major, props.minor, props.name)


def is_nvfp4_gpu_capable():
    """Check whether the active GPU generation can support NVFP4 workloads."""
    major, minor, name = detect_gpu_architecture()
    return major >= 9


def check_recipe_hardware_compatibility(recipe_id: int, verbose: bool = True):
    """Report public-release support for the selected recipe."""
    if recipe_id not in SUPPORTED_RECIPES:
        return False
    if recipe_id == 0:
        return True
    if verbose:
        warnings.warn(PUBLIC_QAT_RUNTIME_MESSAGE, RuntimeWarning)
    return False


def can_run_recipe(recipe_id: int) -> Tuple[bool, str]:
    """Check if a recipe can run in this public release."""
    if recipe_id == 0:
        return True, "Baseline recipe"
    return False, PUBLIC_QAT_RUNTIME_MESSAGE


class NVFPQuantizer:
    """
    Public placeholder for NVFP4 QAT integration.

    The paper's recipe metadata is provided for reproducibility context, while
    vendor/runtime-specific quantization integration is intentionally omitted.
    """

    def __init__(self, recipe_id: int):
        if recipe_id not in SUPPORTED_RECIPES:
            raise ValueError(f"Unsupported recipe_id: {recipe_id}. Supported: {list(SUPPORTED_RECIPES.keys())}")

        self.recipe_id = recipe_id
        self.enabled = recipe_id is not None and recipe_id != 0
        self.forward_only = recipe_id in [2, 7]
        self.chain_rule = recipe_id == 3
        self.has_rht = recipe_id in [4, 5, 7]
        self.has_sr = recipe_id in [5, 6]
        self.has_2d_weights = recipe_id in [4, 5, 7]
        self.uses_autograd = recipe_id == 8
        self.requires_qat_runtime = recipe_id != 0
        self.needs_advanced = recipe_id in [4, 5, 6, 7, 8]
        self.last_x_mse = None
        self.last_w_mse = None
        self.last_g_mse = None

        if self.enabled:
            check_qat_runtime_available()

    def quantize_activation(self, x: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            check_qat_runtime_available()
        return x

    def quantize_weight(self, w: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            check_qat_runtime_available()
        return w

    def quantize_gradient(self, g: torch.Tensor) -> torch.Tensor:
        if self.enabled:
            check_qat_runtime_available()
        return g


class QuantizedLinearFunction(torch.autograd.Function):
    """Public placeholder for the omitted NVFP4 Linear QAT path."""

    @staticmethod
    def forward(ctx, *args, **kwargs):
        check_qat_runtime_available()

    @staticmethod
    def backward(ctx, *grad_outputs):
        check_qat_runtime_available()


# ═══════════════════════════════════════════════════════════════════════════════
# MODEL COMPONENTS (Baseline)
# ═══════════════════════════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    """Patch Embedding (NOT quantized - input layer)."""
    
    def __init__(self, img_size: int, patch_size: int, in_chans: int, embed_dim: int):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.grid_size = img_size // patch_size
        
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        x = self.norm(x)
        return x


class DropPath(nn.Module):
    """Stochastic Depth."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class MultiHeadSelfAttention(nn.Module):
    """Multi-Head Self-Attention with optional quantization."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        qkv_bias: bool = True,
        quantizer: Optional[NVFPQuantizer] = None
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.quantizer = quantizer
        
        # Linear layers (potentially quantized)
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
    
    def _linear(self, x: torch.Tensor, linear: nn.Linear, quantize: bool = True) -> torch.Tensor:
        """Apply Linear - quantized or standard."""
        if self.quantizer is None or not self.quantizer.enabled or not quantize:
            return linear(x)
        else:
            return QuantizedLinearFunction.apply(x, linear.weight, linear.bias, self.quantizer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        qkv = self._linear(x, self.qkv).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, heads, N, head_dim]
        q, k, v = qkv[0], qkv[1], qkv[2]
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self._linear(x, self.proj)
        x = self.proj_drop(x)
        return x


class MLP(nn.Module):
    """Feed-Forward Network with optional quantization."""
    
    def __init__(
        self,
        in_features: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        quantizer: Optional[NVFPQuantizer] = None
    ):
        super().__init__()
        hidden_features = int(in_features * mlp_ratio)
        self.quantizer = quantizer
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, in_features)
        self.act = nn.GELU()
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
    
    def _linear(self, x: torch.Tensor, linear: nn.Linear, quantize: bool = True) -> torch.Tensor:
        """Apply Linear - quantized or standard."""
        if self.quantizer is None or not self.quantizer.enabled or not quantize:
            return linear(x)
        else:
            return QuantizedLinearFunction.apply(x, linear.weight, linear.bias, self.quantizer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._linear(x, self.fc1)
        x = self.act(x)
        x = self.drop1(x)
        x = self._linear(x, self.fc2)
        x = self.drop2(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer Encoder Block with optional quantization."""
    
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        quantizer: Optional[NVFPQuantizer] = None
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout, quantizer=quantizer)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio, dropout, quantizer=quantizer)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def _get_valid_num_groups(num_channels: int, max_groups: int = 8) -> int:
    """Find a valid number of groups that divides num_channels."""
    for g in range(min(max_groups, num_channels), 0, -1):
        if num_channels % g == 0:
            return g
    return 1


class SegmentationDecoder(nn.Module):
    """Segmentation Decoder (NOT quantized - output layer)."""
    
    def __init__(self, embed_dim: int, num_classes: int, num_upsamples: int = 4):
        super().__init__()
        
        channels = [embed_dim]
        for i in range(num_upsamples):
            channels.append(max(8, embed_dim // (2 ** (i + 1))))
        
        self.decoder_blocks = nn.ModuleList()
        for i in range(num_upsamples):
            num_groups = _get_valid_num_groups(channels[i+1])
            block = nn.Sequential(
                nn.ConvTranspose2d(channels[i], channels[i+1], kernel_size=2, stride=2),
                nn.GroupNorm(num_groups, channels[i+1]),
                nn.GELU(),
            )
            self.decoder_blocks.append(block)
        
        self.classifier = nn.Conv2d(channels[-1], num_classes, kernel_size=1)
    
    def forward(self, x: torch.Tensor, grid_size: int) -> torch.Tensor:
        B, N, D = x.shape
        x = x.transpose(1, 2).reshape(B, D, grid_size, grid_size)
        
        for block in self.decoder_blocks:
            x = block(x)
        
        return self.classifier(x)


# ═══════════════════════════════════════════════════════════════════════════════
# VIT QAT MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class ViTQAT(nn.Module):
    """
    Vision Transformer with Quantization-Aware Training support.
    
    Unified model supporting 8 quantization recipes with configurable sizes.
    
    Layer Quantization Pattern:
      - Patch embedding: NOT quantized (input layer)
      - Transformer blocks: Quantized (per recipe)
      - Decoder: NOT quantized (output layer)
    
    Model Sizes:
      - pico:  ~500K params (CNN-equivalent)
      - nano:  ~1.5M params
      - micro: ~3M params
      - tiny:  ~6M params (ViT-Ti equivalent)
      - small: ~22M params (ViT-S equivalent)
      - base:  ~86M params (ViT-B equivalent)
      - large: ~300M params (ViT-L equivalent)
    """
    
    def __init__(
        self,
        img_size: int = 256,
        patch_size: int = 16,
        in_chans: int = 3,
        num_classes: int = 1,  # Binary segmentation (same as CNN)
        embed_dim: int = 96,
        depth: int = 4,
        num_heads: int = 3,
        mlp_ratio: float = 2.0,
        dropout: float = 0.1,
        drop_path: float = 0.1,
        recipe_id: int = 0
    ):
        super().__init__()
        
        self.img_size = img_size
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.depth = depth
        self.num_classes = num_classes
        self.recipe_id = recipe_id
        
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size ** 2
        
        # Initialize quantizer (None for baseline)
        self.quantizer = None
        if recipe_id != 0:
            self.quantizer = NVFPQuantizer(recipe_id)
        
        # Patch embedding (NOT quantized)
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        
        # CLS token and position embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(dropout)
        
        # Stochastic depth schedule
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        
        # Transformer blocks (QUANTIZED)
        self.blocks = nn.ModuleList([
            TransformerBlock(embed_dim, num_heads, mlp_ratio, dropout, dpr[i], self.quantizer)
            for i in range(depth)
        ])
        
        self.norm = nn.LayerNorm(embed_dim)
        
        # Decoder (NOT quantized)
        num_upsamples = int(math.log2(patch_size))
        self.decoder = SegmentationDecoder(embed_dim, num_classes, num_upsamples)
        
        self._init_weights()
    
    def _init_weights(self):
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.shape[0]
        
        # Patch embedding (NOT quantized)
        x = self.patch_embed(x)
        
        # Add CLS token and position embedding
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        
        # Transformer blocks (QUANTIZED)
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        patch_tokens = x[:, 1:]  # Exclude CLS
        
        # Decode (NOT quantized)
        logits = self.decoder(patch_tokens, self.grid_size)
        
        # Ensure output matches input resolution
        if logits.shape[2:] != (self.img_size, self.img_size):
            logits = F.interpolate(logits, size=(self.img_size, self.img_size), mode='bilinear', align_corners=False)
        
        return logits
    
    def get_recipe_info(self) -> Dict[str, Any]:
        """Get information about the current recipe."""
        return SUPPORTED_RECIPES.get(self.recipe_id, {})


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def create_vit_qat(
    model_size: str = 'pico',
    recipe_id: int = 0,
    img_size: int = 256,
    in_chans: int = 3,
    num_classes: int = 1,  # Binary segmentation (same as CNN)
    dropout: float = 0.1,
    drop_path: float = 0.1,
    **kwargs
) -> ViTQAT:
    """
    Factory function to create ViTQAT model.
    
    Args:
        model_size: Size preset ('pico', 'nano', 'micro', 'tiny', 'small', 'base', 'large')
        recipe_id: NVFP4 QAT runtime recipe ID (0 for baseline, 1/2/3/etc. for QAT)
        img_size: Input image size (256 for LGG MRI)
        in_chans: Input channels (3 for LGG MRI)
        num_classes: Output classes (1 for binary segmentation, same as CNN)
        dropout: Dropout rate
        drop_path: Stochastic depth rate
    
    Returns:
        ViTQAT model configured for the specified size and recipe.
    
    Examples:
        # Baseline pico model
        model = create_vit_qat(model_size='pico', recipe_id=0)
        
        # NVFP4 Full Quantization
        model = create_vit_qat(model_size='pico', recipe_id=1)
        
        # Forward-Only
        model = create_vit_qat(model_size='pico', recipe_id=2)
    """
    if model_size not in MODEL_CONFIGS:
        raise ValueError(f"model_size must be one of {list(MODEL_CONFIGS.keys())}")
    
    config = MODEL_CONFIGS[model_size]
    
    return ViTQAT(
        img_size=img_size,
        patch_size=config['patch_size'],
        in_chans=in_chans,
        num_classes=num_classes,
        embed_dim=config['embed_dim'],
        depth=config['depth'],
        num_heads=config['num_heads'],
        mlp_ratio=config['mlp_ratio'],
        dropout=dropout,
        drop_path=drop_path,
        recipe_id=recipe_id,
        **kwargs
    )


def create_scalable_vit_qat(
    scale_name: str = "base",
    recipe_id: int = 0,
    img_size: int = 256,
    in_chans: int = 3,
    num_classes: int = 1,  # Binary segmentation (same as CNN)
    dropout: float = 0.1,
    drop_path: float = 0.1,
    **kwargs
) -> ViTQAT:
    """
    Factory function to create ViTQAT with predefined scale.
    
    This mirrors CNN's create_scalable_cnn_qat() for a consistent interface.
    
    Args:
        scale_name: Scale preset ("tiny", "base", "xlarge")
        recipe_id: NVFP4 QAT runtime recipe ID (0 for baseline, 1/2/3/etc. for QAT)
        img_size: Input image size (256 for LGG MRI)
        in_chans: Input channels (3 for LGG MRI)
        num_classes: Output classes (1 for binary segmentation, same as CNN)
        dropout: Dropout rate
        drop_path: Stochastic depth rate
    
    Returns:
        ViTQAT model configured for the specified scale and recipe.
    
    Scale Mapping:
        - "tiny":   pico config  (~500K params)
        - "base":   micro config (~3M params)
        - "xlarge": small config (~22M params)
    
    Examples:
        # Baseline tiny model
        model = create_scalable_vit_qat(scale_name="tiny", recipe_id=0)
        
        # NVFP4 Full Quantization with base scale
        model = create_scalable_vit_qat(scale_name="base", recipe_id=1)
    """
    if scale_name not in VIT_SCALES:
        raise ValueError(f"scale_name must be one of {list(VIT_SCALES.keys())}, got '{scale_name}'")
    
    # Map scale_name to model_size
    model_size = VIT_SCALES[scale_name]
    
    return create_vit_qat(
        model_size=model_size,
        recipe_id=recipe_id,
        img_size=img_size,
        in_chans=in_chans,
        num_classes=num_classes,
        dropout=dropout,
        drop_path=drop_path,
        **kwargs
    )


def get_vit_scale_info(scale_name: str) -> Dict[str, Any]:
    """
    Get information about a ViT scale configuration.
    
    Args:
        scale_name: Scale name ("tiny", "base", "xlarge")
        
    Returns:
        Dictionary with scale information including params estimate.
    """
    if scale_name not in VIT_SCALES:
        raise ValueError(f"Unknown scale_name: {scale_name}")
    
    model_size = VIT_SCALES[scale_name]
    config = MODEL_CONFIGS[model_size]
    
    return {
        "scale_name": scale_name,
        "model_size": model_size,
        "embed_dim": config["embed_dim"],
        "depth": config["depth"],
        "num_heads": config["num_heads"],
        "mlp_ratio": config["mlp_ratio"],
        "patch_size": config["patch_size"],
        "description": config["description"],
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES
# Loss functions imported from common/loss.py for consistency
# Available: dice_loss, dice_coef_metric
# ═══════════════════════════════════════════════════════════════════════════════

def train_one_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    loss_fn,
    device: torch.device,
    scaler: torch.cuda.amp.GradScaler,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    train_loss_sum = 0.0
    train_dice_sum = 0.0
    
    for images, masks in train_loader:
        images = images.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
            logits = model(images)
            loss = loss_fn(logits, masks)
        
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        
        train_loss_sum += loss.item()
        
        with torch.no_grad():
            probs = torch.softmax(logits.float(), dim=1)[:, 1]
            masks_m = (masks == 1).float()
            train_dice_sum += dice_coef_metric(probs, masks_m)
    
    return train_loss_sum / len(train_loader), train_dice_sum / len(train_loader)


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn,
    device: torch.device,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16
) -> Tuple[float, float]:
    """Validate model."""
    model.eval()
    val_loss_sum = 0.0
    val_dice_sum = 0.0
    
    with torch.no_grad():
        for images, masks in val_loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                logits = model(images)
                loss = loss_fn(logits, masks)
            
            val_loss_sum += loss.item()
            
            probs = torch.softmax(logits.float(), dim=1)[:, 1]
            masks_m = (masks == 1).float()
            val_dice_sum += dice_coef_metric(probs, masks_m)
    
    return val_loss_sum / len(val_loader), val_dice_sum / len(val_loader)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN QAT TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_vit_qat_training(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_size: str = 'pico',
    recipes: List[int] = [0, 1, 2, 3],
    num_epochs: int = 300,
    lr: float = 1e-3,
    device: torch.device = None,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    save_checkpoints: bool = True,
    save_telemetry: bool = True,
    output_dir: str = ".",
    verbose: bool = True,
    seed: int = 42,
    img_size: int = 256,
    in_chans: int = 3,
    num_classes: int = 2
) -> Dict[int, Dict[str, Any]]:
    """
    Train ViT with multiple QAT recipes and compare results.
    
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        model_size: Model size preset ('pico', 'nano', 'tiny', 'small', 'base', 'large')
        recipes: List of recipe IDs to train
        num_epochs: Number of training epochs
        lr: Learning rate
        device: CUDA device
        use_amp: Enable AMP
        amp_dtype: AMP dtype
        save_checkpoints: Save model checkpoints
        save_telemetry: Save training telemetry CSV
        output_dir: Base directory for outputs
        verbose: Print progress
        seed: Random seed
        img_size: Input image size
        in_chans: Input channels
        num_classes: Output classes
    
    Returns:
        Dictionary mapping recipe_id -> {model, history, training_time, ...}
    """
    import csv
    import random
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print info
    if verbose:
        major, minor, gpu_name = detect_gpu_architecture()
        print("=" * 80)
        print(f"🚀 ViT QAT Training - {model_size.upper()} Model")
        print("=" * 80)
        print(f"Hardware: {gpu_name} (Compute Capability {major}.{minor})")
        print(f"Model: {MODEL_CONFIGS[model_size]['description']}")
        print(f"Recipes: {recipes}")
        print("=" * 80)
    
    # Create output directories
    output_path = Path(output_dir)
    checkpoints_dir = output_path / "checkpoints"
    telemetry_dir = output_path / "telemetry"
    
    for d in [checkpoints_dir, telemetry_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Set seeds
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    
    results = {}
    baseline_weights = None
    
    for recipe_id in recipes:
        recipe_info = SUPPORTED_RECIPES.get(recipe_id, {})
        recipe_name = recipe_info.get("name", f"Recipe {recipe_id}")
        
        if verbose:
            print(f"\n{'='*60}")
            print(f"🧪 TRAINING: {recipe_name} (Recipe {recipe_id})")
            print(f"{'='*60}")
        
        # Create model
        model = create_vit_qat(
            model_size=model_size,
            recipe_id=recipe_id,
            img_size=img_size,
            in_chans=in_chans,
            num_classes=num_classes
        ).to(device)
        
        num_params = sum(p.numel() for p in model.parameters())
        if verbose:
            print(f"Parameters: {num_params:,}")
        
        # Copy baseline weights
        if recipe_id == 0:
            baseline_weights = copy.deepcopy(model.state_dict())
        elif baseline_weights is not None:
            model.load_state_dict(baseline_weights, strict=True)
        
        # Optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        history = {
            "train_loss": [], "val_loss": [],
            "train_dice": [], "val_dice": [],
            "epoch_time": []
        }
        
        # Telemetry file
        telemetry_path = None
        if save_telemetry:
            telemetry_path = telemetry_dir / f"vit_{model_size}_recipe_{recipe_id}_telemetry.csv"
            with open(telemetry_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["epoch", "train_loss", "val_loss", "train_dice", "val_dice", "epoch_time", "cumulative_time"])
        
        start_time = time.perf_counter()
        
        for epoch in range(num_epochs):
            epoch_start = time.perf_counter()
            
            train_loss, train_dice = train_one_epoch(
                model, train_loader, optimizer, dice_loss, device, scaler, use_amp, amp_dtype
            )
            val_loss, val_dice = validate(model, val_loader, dice_loss, device, use_amp, amp_dtype)
            
            epoch_time = time.perf_counter() - epoch_start
            cumulative_time = time.perf_counter() - start_time
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_dice"].append(train_dice)
            history["val_dice"].append(val_dice)
            history["epoch_time"].append(epoch_time)
            
            if save_telemetry and telemetry_path:
                with open(telemetry_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([epoch+1, train_loss, val_loss, train_dice, val_dice, epoch_time, cumulative_time])
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f}")
        
        total_time = time.perf_counter() - start_time
        
        if verbose:
            print(f"\n✅ Recipe {recipe_id} completed in {total_time:.1f}s")
            print(f"   Final Val Dice: {history['val_dice'][-1]:.4f}")
        
        # Save checkpoint
        checkpoint_path = None
        if save_checkpoints:
            checkpoint_path = checkpoints_dir / f"vit_{model_size}_recipe_{recipe_id}.pth"
            torch.save(model.state_dict(), checkpoint_path)
        
        results[recipe_id] = {
            "model": model,
            "history": history,
            "training_time": total_time,
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "telemetry": str(telemetry_path) if telemetry_path else None,
            "final_val_dice": history["val_dice"][-1],
            "final_val_loss": history["val_loss"][-1],
            "num_params": num_params,
        }
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_configs():
    """Print all model configurations."""
    print("=" * 80)
    print("📋 VIT MODEL CONFIGURATIONS")
    print("=" * 80)
    print(f"{'Size':<8} {'Embed':<8} {'Depth':<6} {'Heads':<6} {'MLP':<6} {'Patch':<6} {'Description':<30}")
    print("-" * 80)
    for name, cfg in MODEL_CONFIGS.items():
        print(f"{name:<8} {cfg['embed_dim']:<8} {cfg['depth']:<6} {cfg['num_heads']:<6} "
              f"{cfg['mlp_ratio']:<6} {cfg['patch_size']:<6} {cfg['description']:<30}")
    print("=" * 80)


def print_supported_recipes():
    """Print all supported recipes."""
    major, minor, gpu_name = detect_gpu_architecture()
    
    print("=" * 100)
    print("📋 SUPPORTED VIT QAT RECIPES")
    print("=" * 100)
    print(f"Current Hardware: {gpu_name} (Compute Capability {major}.{minor})")
    print(f"NVFP4 QAT runtime: {'✅' if _QAT_RUNTIME_AVAILABLE else '❌'} | Advanced (RHT/SR): {'✅' if _QAT_RUNTIME_ADVANCED_AVAILABLE else '❌'}")
    print("=" * 100)
    
    # Helper to check if recipe needs advanced features
    advanced_features = ["rht_wgrad", "stochastic_rounding", "2d_weights", "nvfp4_autograd", "nvfp4_cast"]
    def needs_advanced(info):
        features = info.get("features", [])
        return any(f in advanced_features for f in features)
    
    print("\nBASELINE AND QAT RECIPES:")
    print("-" * 100)
    for rid, info in sorted(SUPPORTED_RECIPES.items()):
        if not needs_advanced(info):
            print(f"  {rid:<8} {info['name']:<35} {info['description'][:55]}")
    
    print("\nADVANCED QAT RECIPES:")
    print("-" * 100)
    for rid, info in sorted(SUPPORTED_RECIPES.items()):
        if needs_advanced(info):
            compat = '✅' if _QAT_RUNTIME_ADVANCED_AVAILABLE else '❌'
            features = ', '.join(info.get('features', []))
            print(f"  {rid:<8} {info['name']:<35} {compat} [{features}]")
    print("=" * 100)


def print_recipe_comparison(results: Dict[int, Dict[str, Any]]):
    """Print comparison table of training results."""
    print("\n" + "=" * 80)
    print("📊 RECIPE COMPARISON")
    print("=" * 80)
    print(f"{'Recipe':<10} {'Name':<30} {'Val Dice':<12} {'Params':<15} {'Time (s)':<12}")
    print("-" * 80)
    
    for recipe_id, data in sorted(results.items(), key=lambda x: str(x[0])):
        name = SUPPORTED_RECIPES.get(recipe_id, {}).get("name", "Unknown")[:28]
        val_dice = data.get("final_val_dice", 0)
        num_params = data.get("num_params", 0)
        train_time = data.get("training_time", 0)
        print(f"{recipe_id:<10} {name:<30} {val_dice:<12.4f} {num_params:<15,} {train_time:<12.1f}")
    
    print("=" * 80)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN (Testing)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"NVFP4 QAT runtime Available: {_QAT_RUNTIME_AVAILABLE}")
    print(f"NVFP4 QAT runtime Advanced (RHT/SR/Autograd): {_QAT_RUNTIME_ADVANCED_AVAILABLE}")
    print()
    
    # Print configs
    print_model_configs()
    print()
    print_supported_recipes()
    
    # Test model creation for each size
    print("\n" + "=" * 80)
    print("Testing Model Creation (Baseline)")
    print("=" * 80)
    
    for model_size in ['pico', 'tiny', 'small']:
        print(f"\n--- {model_size.upper()} ---")
        
        # Baseline
        model = create_vit_qat(model_size=model_size, recipe_id=0).to(device)
        num_params = count_parameters(model)
        print(f"  Baseline: {num_params:,} params")
        
        x = torch.randn(1, 3, 256, 256).to(device)
        model.eval()
        with torch.no_grad():
            out = model(x)
        print(f"  Input: {list(x.shape)} -> Output: {list(out.shape)}")
    
    # Test ALL QAT recipes (if NVFP4 QAT runtime available)
    if _QAT_RUNTIME_AVAILABLE:
        print("\n" + "=" * 80)
        print("Testing ALL QAT Recipes")
        print("=" * 80)
        
        # Base recipes (baseline runtime group)
        base_recipes = [1, 2, 3]
        
        # Advanced recipes (NVFP4-capable runtime required)
        advanced_recipes = [8, 4, 5, 6, 7]
        
        # Test base recipes
        print("\n📋 BASE RECIPES (baseline runtime group):")
        for recipe_id in base_recipes:
            recipe_info = SUPPORTED_RECIPES[recipe_id]
            print(f"\n--- Recipe {recipe_id}: {recipe_info['name']} ---")
            
            try:
                model_q = create_vit_qat(model_size='pico', recipe_id=recipe_id).to(device)
                
                x = torch.randn(1, 3, 256, 256).to(device)
                model_q.train()
                out = model_q(x)
                print(f"  Forward pass: ✅")
                
                # Test backward
                loss = out.mean()
                loss.backward()
                print(f"  Backward pass: ✅")
                
                # Print quantizer info
                if model_q.quantizer:
                    print(f"  forward_only: {model_q.quantizer.forward_only}")
                    print(f"  chain_rule: {model_q.quantizer.chain_rule}")
                    print(f"  has_rht: {model_q.quantizer.has_rht}")
                    print(f"  has_sr: {model_q.quantizer.has_sr}")
                
            except Exception as e:
                print(f"  ❌ Error: {e}")
        
        # Test advanced recipes (only if NVFP4-capable runtime features available)
        if _QAT_RUNTIME_ADVANCED_AVAILABLE:
            print("\n📋 ADVANCED RECIPES (NVFP4-capable runtime required):")
            for recipe_id in advanced_recipes:
                recipe_info = SUPPORTED_RECIPES[recipe_id]
                print(f"\n--- Recipe {recipe_id}: {recipe_info['name']} ---")
                
                try:
                    model_q = create_vit_qat(model_size='pico', recipe_id=recipe_id).to(device)
                    
                    x = torch.randn(1, 3, 256, 256).to(device)
                    model_q.train()
                    out = model_q(x)
                    print(f"  Forward pass: ✅")
                    
                    # Test backward
                    loss = out.mean()
                    loss.backward()
                    print(f"  Backward pass: ✅")
                    
                    # Print quantizer info
                    if model_q.quantizer:
                        print(f"  forward_only: {model_q.quantizer.forward_only}")
                        print(f"  chain_rule: {model_q.quantizer.chain_rule}")
                        print(f"  has_rht: {model_q.quantizer.has_rht}")
                        print(f"  has_sr: {model_q.quantizer.has_sr}")
                        print(f"  has_2d_weights: {model_q.quantizer.has_2d_weights}")
                        print(f"  uses_autograd: {model_q.quantizer.uses_autograd}")
                    
                except Exception as e:
                    print(f"  ❌ Error: {e}")
        else:
            print("\n⚠️  Non-baseline recipes require a compatible NVFP4 QAT environment")
    
    # Summary
    print("\n" + "=" * 80)
    print("RECIPE ROUTING VERIFICATION")
    print("=" * 80)
    print("""
    Recipe Routing Logic (matches cnn_qat.py):
    
    ┌─────────────────────────────────────────────────────────────────────────┐
    │ Recipe │ backward_quant │ forward_only │ Backward Path                  │
    ├────────┼────────────────┼──────────────┼────────────────────────────────┤
    │ 0      │ N/A            │ N/A          │ Baseline (no quantization)     │
    │ 1   │ True           │ False        │ Full: Q(G), Q(X), Q(W)         │
    │ 8   │ True           │ False        │ Full: Q(G), Q(X), Q(W) + Auto  │
    │ 4   │ True           │ False        │ Full: Q(G), Q(X), Q(W) + RHT   │
    │ 5   │ True           │ False        │ Full: Q(G), Q(X), Q(W) + RHT+SR│
    │ 6   │ True           │ False        │ Full: Q(G), Q(X), Q(W) + SR    │
    │ 2  │ False          │ True         │ Forward-only: G, X_orig, W_orig│
    │ 3  │ False          │ False        │ Chain Rule: G, Q(X), Q(W)      │
    │ 7  │ False          │ True         │ Forward-only + RHT             │
    └─────────────────────────────────────────────────────────────────────────┘
    
    Runtime-specific quantization operator details are omitted from this public release.
    Non-baseline recipes require a compatible NVFP4 QAT environment.
    """)
    
    print("\n✅ All tests completed!")

