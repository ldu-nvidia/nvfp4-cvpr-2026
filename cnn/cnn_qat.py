"""
CNN QAT Training Module
===========================

Self-contained module for Quantization-Aware Training (QAT) with CNN.
Designed to be used directly in the cnn.ipynb notebook or standalone.

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

USAGE IN NOTEBOOK:
    # Add to imports
    import sys
    # NVFP4 QAT recipes require a compatible runtime environment.
    from cnn_qat import (
        CNNQAT,
        create_cnn_qat,
        run_qat_training_schemes,
        SUPPORTED_RECIPES,
        print_supported_recipes  # Helper to see all recipes
    )
    
    # Print all supported recipes
    print_supported_recipes()
    
    # Train base recipes (work on all hardware)
    results = run_qat_training_schemes(
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        recipes=[0, 1, 2, 3],
        num_epochs=300,
        device=device
    )
    
    # Train all recipes including advanced (requires NVFP4-capable runtime)
    results = run_qat_training_schemes(
        train_loader=train_dataloader,
        val_loader=val_dataloader,
        recipes=[0, 1, 8, 4, 5, 2, 3, 7],
        num_epochs=300,
        device=device
    )

Authors: Zijian Du and Oleg Rybakov
"""

import os
import sys
import copy
import time
import warnings
from typing import Optional, Tuple, Dict, List, Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pathlib import Path

# Import unified loss functions from common module
_COMMON_PATH = Path(__file__).parent.parent / "common"
if str(_COMMON_PATH) not in sys.path:
    sys.path.insert(0, str(_COMMON_PATH))
from loss import dice_loss, dice_coef_metric

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════

SUPPORTED_RECIPES = {
    # ═══════════════════════════════════════════════════════════════════════════════
    # BASELINE AND QAT RECIPES
    # ═══════════════════════════════════════════════════════════════════════════════
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
    60031: {
        "name": "NVFP4 Full (Skip Bottleneck)",
        "description": "Same as 1, but keep the bottleneck conv in BF16 (no quantization)",
        "forward_quantized": True,
        "backward_quantized": True,
        "requires_qat_runtime": True,
        "quantize_bottleneck_layer": False,
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
    
    # ═══════════════════════════════════════════════════════════════════════════════
    # ADVANCED QAT RECIPES
    # ═══════════════════════════════════════════════════════════════════════════════
    8: {
        "name": "NVFP4 Autograd + nvfp4",
        "description": "E2M1 with autograd backend and nvfp4 casting (reference impl)",
        "forward_quantized": True,
        "backward_quantized": True,  # Via autograd
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

PUBLIC_QAT_RUNTIME_MESSAGE = (
    "NVFP4 QAT recipes require a compatible NVFP4 quantization runtime. Vendor-specific runtime integration is intentionally omitted from this public release. Use recipe 0 for baseline training with the provided code."
)


def check_qat_runtime_available():
    """Raise a public-safe error for QAT recipes in this release."""
    raise RuntimeError(PUBLIC_QAT_RUNTIME_MESSAGE)


def detect_gpu_architecture():
    """
    Detect GPU architecture (compute capability).
    Returns tuple: (compute_capability_major, compute_capability_minor, gpu_name)
    """
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


class NVFPQuantizer:
    """
    Public placeholder for NVFP4 QAT integration.

    The paper's recipe metadata is provided for reproducibility context, while
    vendor/runtime-specific quantization integration is intentionally omitted.
    """

    def __init__(self, recipe_id):
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


class QuantizedConv2dFunction(torch.autograd.Function):
    """Public placeholder for the omitted NVFP4 Conv2d QAT path."""

    @staticmethod
    def forward(ctx, *args, **kwargs):
        check_qat_runtime_available()

    @staticmethod
    def backward(ctx, *grad_outputs):
        check_qat_runtime_available()


class QuantizedConvTranspose2dFunction(torch.autograd.Function):
    """Public placeholder for the omitted NVFP4 ConvTranspose2d QAT path."""

    @staticmethod
    def forward(ctx, *args, **kwargs):
        check_qat_runtime_available()

    @staticmethod
    def backward(ctx, *grad_outputs):
        check_qat_runtime_available()


# ═══════════════════════════════════════════════════════════════════════════════
# CNN QAT MODEL
# ═══════════════════════════════════════════════════════════════════════════════

class CNNQAT(nn.Module):
    """
    CNN with Quantization-Aware Training support.
    
    Unified model supporting 8 quantization recipes:
    
    BASE RECIPES:
      - Recipe 0:      Baseline (standard PyTorch)
      - Recipe 1:   NVFP4 Full Quantization
      - Recipe 2:  Forward-Only
      - Recipe 3:  Chain Rule
    
    ADVANCED RECIPES:
      - Recipe 8:   NVFP4 Autograd + nvfp4
      - Recipe 4:   NVFP4 2D Weights + RHT
      - Recipe 5:   NVFP4 2D + RHT + SR (best accuracy)
      - Recipe 7:  Forward-Only + RHT
    
    Layer Quantization Pattern:
      - First layer (e1): NOT quantized (preserves input quality)
      - Middle layers (e2-e4, d4-d2): Quantized (per recipe)
      - Last layer (d1): NOT quantized (preserves output/logits quality)
    
    Args:
        in_channels: Number of input channels (default 3 for RGB)
        num_classes: Number of output classes (default 1 for binary)
        base_channels: Network width (default 64)
        dropout_p: Dropout probability (default 0.5)
        use_leaky: Use LeakyReLU in first encoder block (default True)
        recipe_id: NVFP4 QAT runtime recipe ID (see SUPPORTED_RECIPES)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        dropout_p: float = 0.5,
        use_leaky: bool = True,
        recipe_id: int = 0,
        quantize_first_layer: Optional[bool] = None,
        quantize_last_layer: Optional[bool] = None,
        quantize_bottleneck_layer: Optional[bool] = None,
    ):
        """
        Initialize CNNQAT model.
        
        Args:
            in_channels: Number of input channels (default 3 for RGB)
            num_classes: Number of output classes (default 1 for binary)
            base_channels: Network width (default 64)
            dropout_p: Dropout probability (default 0.5)
            use_leaky: Use LeakyReLU in first encoder block (default True)
            recipe_id: NVFP4 QAT runtime recipe ID (0, 1, 2, 3, 8, 4, 5, 7)
                       Also supports variants: 900461, 900462, 900471, 900472
            quantize_first_layer: Override for e1 quantization (None = use recipe default)
            quantize_last_layer: Override for d1 quantization (None = use recipe default)
        """
        super().__init__()
        
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.dropout_p = dropout_p
        self.use_leaky = use_leaky
        
        # Parse recipe ID to handle variants (900461, 900462, etc.)
        # Variants encode: base_recipe * 10 + variant_digit
        # variant_digit 1 = quantize all layers, 2 = skip boundary layers
        base_recipe, variant_quantize_first_last = parse_recipe_id(recipe_id)
        
        self.recipe_id = recipe_id
        self.base_recipe = base_recipe  # The actual recipe used for quantization
        
        # Determine first/last layer quantization
        # Priority: explicit parameter > variant encoding > default (False)
        if quantize_first_layer is not None:
            self.quantize_first_layer = quantize_first_layer
        elif recipe_id != base_recipe:  # It's a variant
            self.quantize_first_layer = variant_quantize_first_last
        else:
            self.quantize_first_layer = False  # Default: e1 NOT quantized
        
        if quantize_last_layer is not None:
            self.quantize_last_layer = quantize_last_layer
        elif recipe_id != base_recipe:  # It's a variant
            self.quantize_last_layer = variant_quantize_first_last
        else:
            self.quantize_last_layer = False  # Default: d1 NOT quantized
        
        # Initialize quantizer (None for baseline)
        self.quantizer = None
        if base_recipe != 0:
            self.quantizer = NVFPQuantizer(base_recipe)

        # Whether to quantize the bottleneck conv (last encoder conv)
        # Default: True unless recipe explicitly disables it (e.g., 60031).
        if quantize_bottleneck_layer is not None:
            self.quantize_bottleneck_layer = quantize_bottleneck_layer
        else:
            self.quantize_bottleneck_layer = SUPPORTED_RECIPES.get(recipe_id, {}).get(
                "quantize_bottleneck_layer", True
            )
        
        # Activations
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        
        # Encoder (4 stages)
        self.e1 = nn.Conv2d(in_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.e2 = nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.e3 = nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.e4 = nn.Conv2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        
        # Decoder (4 stages)
        self.d4 = nn.ConvTranspose2d(base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.d3 = nn.ConvTranspose2d(2 * base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.d2 = nn.ConvTranspose2d(2 * base_channels, base_channels, kernel_size=4, stride=2, padding=1)
        self.d1 = nn.ConvTranspose2d(2 * base_channels, num_classes, kernel_size=4, stride=2, padding=1)
        
        self._init_weights()
    
    def _init_weights(self):
        """Kaiming initialization for all conv layers."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _conv2d(self, x: torch.Tensor, conv: nn.Conv2d, quantize: bool = True) -> torch.Tensor:
        """
        Apply Conv2d - quantized or standard based on recipe.
        
        Args:
            x: Input tensor
            conv: Conv2d layer
            quantize: Whether to apply quantization (False for first/last layers)
        """
        if self.quantizer is None or not quantize:
            return conv(x)
        else:
            return QuantizedConv2dFunction.apply(
                x, conv.weight, conv.bias, self.quantizer,
                conv.stride[0], conv.padding[0], conv.dilation[0], conv.groups
            )
    
    def _conv_transpose2d(self, x: torch.Tensor, conv: nn.ConvTranspose2d, quantize: bool = True) -> torch.Tensor:
        """
        Apply ConvTranspose2d - quantized or standard based on recipe.
        
        Args:
            x: Input tensor
            conv: ConvTranspose2d layer
            quantize: Whether to apply quantization (False for first/last layers)
        """
        if self.quantizer is None or not quantize:
            return conv(x)
        else:
            return QuantizedConvTranspose2dFunction.apply(
                x, conv.weight, conv.bias, self.quantizer,
                conv.stride[0], conv.padding[0], conv.output_padding[0],
                conv.dilation[0], conv.groups
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through encoder-decoder with skip connections.
        
        Note: First layer (e1) and last layer (d1) are NOT quantized to preserve
        input/output quality. Only intermediate layers are quantized.
        Activations (ReLU, LeakyReLU) and Dropout are never quantized.
        """
        
        # ═══════════════════════════════════════════════════════════════════
        # ENCODER
        # ═══════════════════════════════════════════════════════════════════
        
        # Stage 1: [B, in, H, W] -> [B, base, H/2, W/2]
        # First layer: Optionally quantized (default: False to preserve input quality)
        c1 = self._conv2d(x, self.e1, quantize=self.quantize_first_layer)
        c1 = self.leaky(c1) if self.use_leaky else self.relu(c1)
        c1 = F.dropout(c1, p=self.dropout_p, training=self.training)
        
        # Stage 2: [B, base, H/2, W/2] -> [B, base, H/4, W/4]
        c2 = self._conv2d(c1, self.e2, quantize=True)
        c2 = self.relu(c2)
        c2 = F.dropout(c2, p=self.dropout_p, training=self.training)
        
        # Stage 3: [B, base, H/4, W/4] -> [B, base, H/8, W/8]
        c3 = self._conv2d(c2, self.e3, quantize=True)
        c3 = self.relu(c3)
        c3 = F.dropout(c3, p=self.dropout_p, training=self.training)
        
        # Stage 4 (bottleneck): [B, base, H/8, W/8] -> [B, base, H/16, W/16]
        c4 = self._conv2d(c3, self.e4, quantize=True)
        c4 = self.relu(c4)
        c4 = F.dropout(c4, p=self.dropout_p, training=self.training)
        
        # ═══════════════════════════════════════════════════════════════════
        # DECODER with skip connections
        # ═══════════════════════════════════════════════════════════════════
        
        # Stage 4: [B, base, H/16, W/16] -> [B, base, H/8, W/8]
        u4 = self._conv_transpose2d(c4, self.d4, quantize=True)
        u4 = self.relu(u4)
        u4 = torch.cat([u4, c3], dim=1)
        u4 = F.dropout(u4, p=self.dropout_p, training=self.training)
        
        # Stage 3: [B, 2*base, H/8, W/8] -> [B, base, H/4, W/4]
        u3 = self._conv_transpose2d(u4, self.d3, quantize=True)
        u3 = self.relu(u3)
        u3 = torch.cat([u3, c2], dim=1)
        u3 = F.dropout(u3, p=self.dropout_p, training=self.training)
        
        # Stage 2: [B, 2*base, H/4, W/4] -> [B, base, H/2, W/2]
        u2 = self._conv_transpose2d(u3, self.d2, quantize=True)
        u2 = self.relu(u2)
        u2 = torch.cat([u2, c1], dim=1)
        
        # Stage 1 (output): [B, 2*base, H/2, W/2] -> [B, num_classes, H, W]
        # Last layer: Optionally quantized (default: False to preserve output/logits quality)
        logits = self._conv_transpose2d(u2, self.d1, quantize=self.quantize_last_layer)
        
        return logits
    
    def get_recipe_info(self) -> Dict[str, Any]:
        """Get information about the current recipe."""
        return SUPPORTED_RECIPES.get(self.recipe_id, {})


# ═══════════════════════════════════════════════════════════════════════════════
# FACTORY FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def create_cnn_qat(
    in_channels: int = 3,
    num_classes: int = 1,
    base_channels: int = 64,
    dropout_p: float = 0.5,
    use_leaky: bool = True,
    recipe_id: int = 0,
    quantize_first_layer: Optional[bool] = None,
    quantize_last_layer: Optional[bool] = None
) -> CNNQAT:
    """
    Factory function to create CNNQAT model.
    
    Args:
        in_channels: Number of input channels
        num_classes: Number of output classes
        base_channels: Network width
        dropout_p: Dropout probability
        use_leaky: Use LeakyReLU in first encoder block
        recipe_id: NVFP4 QAT runtime recipe ID (0, 1, 2, 3, 8, 4, 5, 7)
                   Also supports variants: 900461, 900462, 900471, 900472
        quantize_first_layer: Override for e1 quantization (None = use recipe default)
        quantize_last_layer: Override for d1 quantization (None = use recipe default)
    
    Returns:
        CNNQAT model configured for the specified recipe.
        
    Examples:
        # Baseline (no quantization)
        model = create_cnn_qat(recipe_id=0)
        
        # NVFP4 Full Quantization
        model = create_cnn_qat(recipe_id=1)
        
        # Forward-Only
        model = create_cnn_qat(recipe_id=2)
        
        # Forward-Only with ALL layers quantized (including e1, d1)
        model = create_cnn_qat(recipe_id=900461)
        
        # Explicit first/last layer control
        model = create_cnn_qat(recipe_id=1, quantize_first_layer=True)
    """
    return CNNQAT(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        dropout_p=dropout_p,
        use_leaky=use_leaky,
        recipe_id=recipe_id,
        quantize_first_layer=quantize_first_layer,
        quantize_last_layer=quantize_last_layer
    )


# ═══════════════════════════════════════════════════════════════════════════════
# SCALABLE CNN QAT
# ═══════════════════════════════════════════════════════════════════════════════

# Predefined scale configurations
# MATCHED with ViT for fair comparison experiments
# NOTE: Smaller models to avoid AUC saturation (Jan 2026 revision)
CNN_SCALES = {
    # name: (num_stages, width_multiplier, description)
    # ═══════════════════════════════════════════════════════════════════════════
    # PRIMARY SCALES FOR QAT EXPERIMENTS
    # "small" is the original CNN architecture that shows largest QAT differences
    # ═══════════════════════════════════════════════════════════════════════════
    "small":  (4, 1.0,  "4 stages, 64 channels - ~530K params (original CNN)"),
    "matched_500k": (4, 1.0,  "4 stages, 64 channels - ~530K params"),
    "matched_1m": (5, 1.25, "5 stages, 80 channels - ~1.13M params"),
    "matched_4m": (6, 2.0,  "6 stages, 128 channels - ~3.7M params"),
    "matched_10m": (7, 3.0,  "7 stages, 192 channels - ~10M params"),
    "matched_15m": (7, 3.5,  "7 stages, 224 channels - ~13.7M params"),
    "medium": (5, 1.5,  "5 stages, 96 channels - ~2M params"),
    "large":  (6, 2.0,  "6 stages, 128 channels - ~8M params"),
    
    # ═══════════════════════════════════════════════════════════════════════════
    # ADDITIONAL SCALES (for extended experiments)
    # ═══════════════════════════════════════════════════════════════════════════
    "tiny":   (2, 0.5,  "2 stages, 32 channels - ~35K params (minimal)"),
    "base":   (4, 1.0,  "4 stages, 64 channels - ~530K params (alias for small)"),
    "xlarge": (7, 3.0,  "7 stages, 192 channels - ~10M params (alias for matched_10m)"),
    
    # Legacy matched scales (pre-Jan 2026)
    "legacy_small":  (4, 1.0,  "4 stages, 64 channels - ~500K params"),
    "legacy_medium": (5, 2.0,  "5 stages, 128 channels - ~3M params"),
    "legacy_large":  (6, 2.5,  "6 stages, 160 channels - ~10M params"),
}


class ScalableCNNQAT(nn.Module):
    """
    Scalable CNN UNet with QAT support.
    
    This version supports variable depth (number of encoder/decoder stages) and
    width (channel multiplier) for flexible model scaling.
    
    Architecture:
    - N encoder stages: each halves spatial resolution (stride=2)
    - N decoder stages: each doubles spatial resolution (transposed conv)
    - BatchNorm2d after every conv (except final output layer)
    - Skip connections between corresponding encoder/decoder stages
    - First and last layers optionally kept in full precision
    
    Scaling Parameters:
    - num_stages: Number of encoder/decoder stages (2-8, default 4)
    - width_multiplier: Multiplier for base_channels (0.25-4.0, default 1.0)
    
    Predefined Scales (use scale_name parameter):
    - "tiny":   2 stages, 0.5x width  (32 channels)  - ~0.1M params
    - "small":  3 stages, 0.75x width (48 channels)  - ~0.3M params
    - "base":   4 stages, 1.0x width  (64 channels)  - ~0.5M params (original)
    - "medium": 5 stages, 1.5x width  (96 channels)  - ~2M params
    - "large":  6 stages, 2.0x width  (128 channels) - ~8M params
    - "xlarge": 7 stages, 3.0x width  (192 channels) - ~30M params
    
    Note: Input spatial size must be divisible by 2^num_stages.
    For num_stages=4 and input=256, bottleneck is 16x16.
    For num_stages=6 and input=256, bottleneck is 4x4.
    
    Args:
        in_channels: Number of input channels (default 3)
        num_classes: Number of output classes (default 1)
        base_channels: Base channel count before width multiplier (default 64)
        num_stages: Number of encoder/decoder stages (default 4)
        width_multiplier: Channel multiplier (default 1.0)
        scale_name: Predefined scale name (overrides num_stages and width_multiplier)
        dropout_p: Dropout probability (default 0.5)
        use_leaky: Use LeakyReLU in first encoder (default True)
        recipe_id: NVFP4 QAT runtime QAT recipe ID (default 0 = no quantization)
        quantize_first_layer: Override first layer quantization
        quantize_last_layer: Override last layer quantization
    
    Example:
        # Use predefined scale
        model = ScalableCNNQAT(scale_name="large", recipe_id=1)
        
        # Custom scale
        model = ScalableCNNQAT(num_stages=5, width_multiplier=1.5, recipe_id=1)
        
        # Compound scale factor (affects both width and depth)
        model = ScalableCNNQAT.from_scale_factor(scale=2.0, recipe_id=1)
    """
    
    def __init__(
        self,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        num_stages: int = 4,
        width_multiplier: float = 1.0,
        scale_name: Optional[str] = None,
        dropout_p: float = 0.5,
        use_leaky: bool = True,
        recipe_id: int = 0,
        quantize_first_layer: Optional[bool] = None,
        quantize_last_layer: Optional[bool] = None,
        quantize_bottleneck_layer: Optional[bool] = None,
    ):
        super().__init__()
        
        # Apply predefined scale if specified
        if scale_name is not None:
            if scale_name not in CNN_SCALES:
                raise ValueError(f"Unknown scale_name: {scale_name}. Available: {list(CNN_SCALES.keys())}")
            num_stages, width_multiplier, _ = CNN_SCALES[scale_name]
        
        # Validate parameters
        if num_stages < 2 or num_stages > 8:
            raise ValueError(f"num_stages must be 2-8, got {num_stages}")
        if width_multiplier < 0.25 or width_multiplier > 4.0:
            raise ValueError(f"width_multiplier must be 0.25-4.0, got {width_multiplier}")
        
        # Store configuration
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.base_channels = base_channels
        self.num_stages = num_stages
        self.width_multiplier = width_multiplier
        self.scale_name = scale_name
        self.dropout_p = dropout_p
        self.use_leaky = use_leaky
        
        # Compute actual channel count (must be divisible by 16 for NVFP4 QAT runtime quantization)
        self.channels = max(16, int(base_channels * width_multiplier // 16) * 16)
        
        # Parse recipe and set up quantization
        base_recipe, variant_quantize_first_last = parse_recipe_id(recipe_id)
        self.recipe_id = recipe_id
        self.base_recipe = base_recipe
        
        if quantize_first_layer is not None:
            self.quantize_first_layer = quantize_first_layer
        elif recipe_id != base_recipe:
            self.quantize_first_layer = variant_quantize_first_last
        else:
            self.quantize_first_layer = False
        
        if quantize_last_layer is not None:
            self.quantize_last_layer = quantize_last_layer
        elif recipe_id != base_recipe:
            self.quantize_last_layer = variant_quantize_first_last
        else:
            self.quantize_last_layer = False
        
        # Initialize quantizer
        self.quantizer = None
        if base_recipe != 0:
            self.quantizer = NVFPQuantizer(base_recipe)

        # Whether to quantize the bottleneck conv (last encoder conv)
        # Default: True unless recipe explicitly disables it (e.g., 60031).
        if quantize_bottleneck_layer is not None:
            self.quantize_bottleneck_layer = quantize_bottleneck_layer
        else:
            self.quantize_bottleneck_layer = SUPPORTED_RECIPES.get(recipe_id, {}).get(
                "quantize_bottleneck_layer", True
            )
        
        # Activations
        self.leaky = nn.LeakyReLU(0.2, inplace=True)
        self.relu = nn.ReLU(inplace=True)
        
        # Build encoder layers dynamically
        self.encoders = nn.ModuleList()
        self.encoder_norms = nn.ModuleList()
        for i in range(num_stages):
            if i == 0:
                # First encoder: in_channels -> channels
                self.encoders.append(
                    nn.Conv2d(in_channels, self.channels, kernel_size=4, stride=2, padding=1)
                )
            else:
                # Subsequent encoders: channels -> channels
                self.encoders.append(
                    nn.Conv2d(self.channels, self.channels, kernel_size=4, stride=2, padding=1)
                )
            # BatchNorm after every encoder conv
            self.encoder_norms.append(nn.BatchNorm2d(self.channels))
        
        # Build decoder layers dynamically
        self.decoders = nn.ModuleList()
        self.decoder_norms = nn.ModuleList()
        for i in range(num_stages):
            if i == 0:
                # First decoder (from bottleneck): channels -> channels
                self.decoders.append(
                    nn.ConvTranspose2d(self.channels, self.channels, kernel_size=4, stride=2, padding=1)
                )
                self.decoder_norms.append(nn.BatchNorm2d(self.channels))
            elif i == num_stages - 1:
                # Last decoder: 2*channels -> num_classes (no BatchNorm before final output)
                self.decoders.append(
                    nn.ConvTranspose2d(2 * self.channels, num_classes, kernel_size=4, stride=2, padding=1)
                )
                self.decoder_norms.append(nn.Identity())  # No norm on output layer
            else:
                # Middle decoders: 2*channels (with skip) -> channels
                self.decoders.append(
                    nn.ConvTranspose2d(2 * self.channels, self.channels, kernel_size=4, stride=2, padding=1)
                )
                self.decoder_norms.append(nn.BatchNorm2d(self.channels))
        
        self._init_weights()
    
    def _init_weights(self):
        """Kaiming initialization for conv layers, standard init for BatchNorm."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def _conv2d(self, x: torch.Tensor, conv: nn.Conv2d, quantize: bool = True) -> torch.Tensor:
        """Apply Conv2d with optional quantization."""
        if self.quantizer is None or not quantize:
            return conv(x)
        else:
            return QuantizedConv2dFunction.apply(
                x, conv.weight, conv.bias, self.quantizer,
                conv.stride[0], conv.padding[0], conv.dilation[0], conv.groups
            )
    
    def _conv_transpose2d(self, x: torch.Tensor, conv: nn.ConvTranspose2d, quantize: bool = True) -> torch.Tensor:
        """Apply ConvTranspose2d with optional quantization."""
        if self.quantizer is None or not quantize:
            return conv(x)
        else:
            return QuantizedConvTranspose2dFunction.apply(
                x, conv.weight, conv.bias, self.quantizer,
                conv.stride[0], conv.padding[0], conv.output_padding[0],
                conv.dilation[0], conv.groups
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through scalable encoder-decoder with skip connections.
        
        The number of stages is determined by self.num_stages.
        Skip connections connect encoder stage i with decoder stage (num_stages - 1 - i).
        """
        # ═══════════════════════════════════════════════════════════════════
        # ENCODER
        # ═══════════════════════════════════════════════════════════════════
        encoder_outputs = []
        h = x
        
        for i, encoder in enumerate(self.encoders):
            # First layer: optionally quantized
            # Last encoder layer and all middle layers: quantized
            quantize = (i > 0 or self.quantize_first_layer)
            if i == self.num_stages - 1 and not self.quantize_bottleneck_layer:
                quantize = False
            
            h = self._conv2d(h, encoder, quantize=quantize)
            h = self.encoder_norms[i](h)  # BatchNorm after conv
            
            # Activation
            if i == 0:
                h = self.leaky(h) if self.use_leaky else self.relu(h)
            else:
                h = self.relu(h)
            
            # Dropout
            h = F.dropout(h, p=self.dropout_p, training=self.training)
            
            # Save for skip connection (except last encoder = bottleneck)
            if i < self.num_stages - 1:
                encoder_outputs.append(h)
        
        # ═══════════════════════════════════════════════════════════════════
        # DECODER with skip connections
        # ═══════════════════════════════════════════════════════════════════
        for i, decoder in enumerate(self.decoders):
            # Last decoder layer: optionally quantized
            # All other decoder layers: quantized
            is_last = (i == self.num_stages - 1)
            quantize = (not is_last or self.quantize_last_layer)
            
            h = self._conv_transpose2d(h, decoder, quantize=quantize)
            
            if not is_last:
                # BatchNorm + Activation + Skip connection
                h = self.decoder_norms[i](h)
                h = self.relu(h)
                
                # Skip connection: concatenate with corresponding encoder output
                # Decoder i connects to encoder (num_stages - 2 - i)
                skip_idx = self.num_stages - 2 - i
                if skip_idx >= 0:
                    h = torch.cat([h, encoder_outputs[skip_idx]], dim=1)
                
                h = F.dropout(h, p=self.dropout_p, training=self.training)
        
        return h
    
    @classmethod
    def from_scale_factor(
        cls,
        scale: float = 1.0,
        in_channels: int = 3,
        num_classes: int = 1,
        base_channels: int = 64,
        dropout_p: float = 0.5,
        use_leaky: bool = True,
        recipe_id: int = 0,
        quantize_first_layer: Optional[bool] = None,
        quantize_last_layer: Optional[bool] = None
    ) -> 'ScalableCNNQAT':
        """
        Create model using compound scale factor.
        
        The scale factor affects both width and depth:
        - width_multiplier = scale^0.5 (square root scaling)
        - num_stages = 4 + round(log2(scale)) (logarithmic depth scaling)
        
        Scale examples:
        - scale=0.5:  ~3 stages, ~0.7x width
        - scale=1.0:  4 stages, 1.0x width (base)
        - scale=2.0:  5 stages, ~1.4x width
        - scale=4.0:  6 stages, 2.0x width
        - scale=8.0:  7 stages, ~2.8x width
        
        Args:
            scale: Compound scale factor (0.25 to 8.0)
            Other args: Same as __init__
            
        Returns:
            ScalableCNNQAT model
        """
        import math
        
        if scale < 0.25 or scale > 8.0:
            raise ValueError(f"scale must be 0.25-8.0, got {scale}")
        
        # Compound scaling formula
        width_multiplier = scale ** 0.5  # Square root for width
        num_stages = 4 + round(math.log2(scale)) if scale >= 1.0 else max(2, 4 + round(math.log2(scale)))
        num_stages = max(2, min(8, num_stages))  # Clamp to 2-8
        
        return cls(
            in_channels=in_channels,
            num_classes=num_classes,
            base_channels=base_channels,
            num_stages=num_stages,
            width_multiplier=width_multiplier,
            dropout_p=dropout_p,
            use_leaky=use_leaky,
            recipe_id=recipe_id,
            quantize_first_layer=quantize_first_layer,
            quantize_last_layer=quantize_last_layer
        )
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model configuration and statistics."""
        num_params = sum(p.numel() for p in self.parameters())
        num_trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            "scale_name": self.scale_name,
            "num_stages": self.num_stages,
            "width_multiplier": self.width_multiplier,
            "channels": self.channels,
            "base_channels": self.base_channels,
            "recipe_id": self.recipe_id,
            "base_recipe": self.base_recipe,
            "quantize_first_layer": self.quantize_first_layer,
            "quantize_last_layer": self.quantize_last_layer,
            "num_parameters": num_params,
            "num_trainable_parameters": num_trainable,
            "min_input_size": 2 ** self.num_stages,  # Minimum spatial size
        }
    
    def __repr__(self) -> str:
        info = self.get_model_info()
        return (
            f"ScalableCNNQAT(\n"
            f"  scale_name={info['scale_name']},\n"
            f"  num_stages={info['num_stages']},\n"
            f"  width_multiplier={info['width_multiplier']:.2f},\n"
            f"  channels={info['channels']},\n"
            f"  recipe_id={info['recipe_id']},\n"
            f"  parameters={info['num_parameters']:,}\n"
            f")"
        )


def create_scalable_cnn_qat(
    scale_name: str = "base",
    in_channels: int = 3,
    num_classes: int = 1,
    base_channels: int = 64,
    dropout_p: float = 0.5,
    use_leaky: bool = True,
    recipe_id: int = 0,
    quantize_first_layer: Optional[bool] = None,
    quantize_last_layer: Optional[bool] = None
) -> ScalableCNNQAT:
    """
    Factory function to create ScalableCNNQAT with predefined scale.
    
    Args:
        scale_name: One of "tiny", "small", "base", "medium", "large", "xlarge"
        Other args: Same as ScalableCNNQAT
        
    Returns:
        ScalableCNNQAT model
        
    Example:
        model = create_scalable_cnn_qat("large", recipe_id=1)
    """
    return ScalableCNNQAT(
        in_channels=in_channels,
        num_classes=num_classes,
        base_channels=base_channels,
        scale_name=scale_name,
        dropout_p=dropout_p,
        use_leaky=use_leaky,
        recipe_id=recipe_id,
        quantize_first_layer=quantize_first_layer,
        quantize_last_layer=quantize_last_layer
    )


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
    amp_dtype: torch.dtype = torch.float16,
    use_channels_last: bool = True
) -> Tuple[float, float]:
    """
    Train for one epoch (matching cnn.ipynb training loop).
    
    Returns:
        (avg_loss, avg_dice)
    """
    model.train()
    train_loss_sum = 0.0
    train_dice_sum = 0.0
    
    for images, masks in train_loader:
        if use_channels_last:
            images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
        else:
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
        
        # Dice metric
        with torch.no_grad():
            probs = torch.sigmoid(logits.float())
            masks_m = masks.unsqueeze(1) if masks.ndim == 3 else masks
            masks_m = masks_m.float()
            if masks_m.max() > 1:
                masks_m = masks_m / 255.0
            train_dice_sum += dice_coef_metric(probs, masks_m)
    
    avg_loss = train_loss_sum / len(train_loader)
    avg_dice = train_dice_sum / len(train_loader)
    return avg_loss, avg_dice


def validate(
    model: nn.Module,
    val_loader: DataLoader,
    loss_fn,
    device: torch.device,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    use_channels_last: bool = True
) -> Tuple[float, float]:
    """
    Validate model (matching cnn.ipynb validation loop).
    
    Returns:
        (avg_loss, avg_dice)
    """
    model.eval()
    val_loss_sum = 0.0
    val_dice_sum = 0.0
    
    with torch.no_grad():
        for images, masks in val_loader:
            if use_channels_last:
                images = images.to(device, non_blocking=True).to(memory_format=torch.channels_last)
            else:
                images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                logits = model(images)
                loss = loss_fn(logits, masks)
            
            val_loss_sum += loss.item()
            
            probs = torch.sigmoid(logits.float())
            masks_m = masks.unsqueeze(1) if masks.ndim == 3 else masks
            masks_m = masks_m.float()
            if masks_m.max() > 1:
                masks_m = masks_m / 255.0
            val_dice_sum += dice_coef_metric(probs, masks_m)
    
    avg_loss = val_loss_sum / len(val_loader)
    avg_dice = val_dice_sum / len(val_loader)
    return avg_loss, avg_dice


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN QAT TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def run_qat_training_schemes(
    train_loader: DataLoader,
    val_loader: DataLoader,
    recipes: List = [0, 1, 2, 3],
    num_epochs: int = 300,
    lr: float = 1e-3,
    device: torch.device = None,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16,
    use_channels_last: bool = True,
    save_checkpoints: bool = True,
    save_telemetry: bool = True,
    output_dir: str = ".",
    verbose: bool = True,
    seed: int = 42
) -> Dict[int, Dict[str, Any]]:
    """
    Train CNN with multiple QAT recipes and compare results.
    
    This is the main entry point for QAT experiments, designed to match
    the cnn.ipynb workflow.
    
    Args:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        recipes: List of recipe IDs to train. Supported recipes:
                 BASELINE/QAT: [0, 1, 2, 3, 900461, 900462, 900471, 900472]
                 ADVANCED (NVFP4-capable runtime): [8, 4, 5, 7]
                 Default: [0, 1, 2, 3]
        num_epochs: Number of training epochs (default: 300)
        lr: Learning rate (default: 1e-3, matching cnn.ipynb)
        device: CUDA device (default: auto-detect)
        use_amp: Enable AMP mixed precision (default: True)
        amp_dtype: AMP dtype (default: torch.float16)
        use_channels_last: Enable channels_last memory format (default: True)
        save_checkpoints: Save model checkpoints after training (default: True)
        save_telemetry: Save training telemetry CSV after each epoch (default: True)
        output_dir: Base directory for outputs (default: ".")
        verbose: Print progress (default: True)
        seed: Random seed for reproducibility (default: 42)
    
    Returns:
        Dictionary mapping recipe_id -> {
            "model": trained model,
            "history": training history,
            "training_time": total training time,
            "checkpoint": checkpoint path (if saved),
            "telemetry": telemetry CSV path (if saved)
        }
    
    Note:
        Non-baseline NVFP4 QAT recipes require a compatible NVFP4 training environment.
    
    Output Structure:
        {output_dir}/
        ├── checkpoints/     # Model checkpoint files (.pth)
        ├── telemetry/       # Training metrics CSV files
        └── plots/           # Generated plot images (created by plotting functions)
    """
    import csv
    from pathlib import Path
    
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Print hardware and recipe information
    if verbose:
        major, minor, gpu_name = detect_gpu_architecture()
        is_gpu_adv = is_nvfp4_gpu_capable()
        
        print("=" * 80)
        print("🚀 CNN QAT Training - Multi-Recipe Comparison")
        print("=" * 80)
        print(f"Hardware: {gpu_name} (Compute Capability {major}.{minor})")
        print(f"NVFP4-capable GPU: {'✅ Yes (supported NVFP4 GPU)' if is_gpu_adv else '❌ No (Advanced recipes will fail)'}")
        print(f"NVFP4 QAT runtime Available: {'✅ Yes' if _QAT_RUNTIME_AVAILABLE else '❌ No'}")
        print(f"NVFP4 QAT runtime Advanced: {'✅ Yes' if _QAT_RUNTIME_ADVANCED_AVAILABLE else '❌ No (RHT, SR, nvfp4)'}")
        print()
        print(f"Training Configuration:")
        print(f"  Epochs: {num_epochs}")
        print(f"  Learning Rate: {lr}")
        print(f"  AMP: {use_amp} ({'FP16' if amp_dtype == torch.float16 else 'BF16'})")
        print(f"  Channels Last: {use_channels_last}")
        print(f"  Random Seed: {seed}")
        print()
        
        # Categorize recipes
        base_recipes = []
        advanced_recipes = []
        invalid_recipes = []
        
        for recipe_id in recipes:
            if recipe_id not in SUPPORTED_RECIPES:
                invalid_recipes.append(recipe_id)
            elif SUPPORTED_RECIPES[recipe_id].get("requires_qat_runtime", False):
                advanced_recipes.append(recipe_id)
            else:
                base_recipes.append(recipe_id)
        
        if invalid_recipes:
            print(f"⚠️  Invalid Recipes: {invalid_recipes}")
            print(f"   Supported: {list(SUPPORTED_RECIPES.keys())}")
            print()
        
        if base_recipes:
            print(f"✅ Base Recipes (baseline runtime group): {base_recipes}")
            for rid in base_recipes:
                info = SUPPORTED_RECIPES[rid]
                print(f"   - {rid}: {info['name']}")
        
        if advanced_recipes:
            print()
            print(f"{'⚠️ ' if not is_gpu_adv else '✅'} Advanced Recipes (require NVFP4-capable runtime): {advanced_recipes}")
            for rid in advanced_recipes:
                info = SUPPORTED_RECIPES[rid]
                print(f"   - {rid}: {info['name']}")
                print(f"           Features: {', '.join(info.get('features', []))}")
            
            if not is_gpu_adv:
                print()
                print(f"{'='*80}")
                print(f"⚠️  WARNING: Advanced recipes require NVFP4-capable runtime (supported NVFP4 GPU) but running on {gpu_name}")
                print(f"{'='*80}")
                print(f"Advanced recipes will likely FAIL during training.")
                print(f"To use these recipes:")
                print(f"  1. Use a compatible NVFP4 QAT environment")
                print(f"  2. Run on supported NVFP4 GPU GPU")
                print(f"{'='*80}")
        
        print()
        print("=" * 80)
        print()
    
    # Create output directories
    output_path = Path(output_dir)
    checkpoints_dir = output_path / "checkpoints"
    telemetry_dir = output_path / "telemetry"
    plots_dir = output_path / "plots"
    
    for d in [checkpoints_dir, telemetry_dir, plots_dir]:
        d.mkdir(parents=True, exist_ok=True)
    
    # Set random seeds for reproducibility
    import random
    import os
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    if verbose:
        print(f"🎲 Random seed set to {seed} for reproducibility")
        print(f"📁 Output directories:")
        print(f"   Checkpoints: {checkpoints_dir}")
        print(f"   Telemetry:   {telemetry_dir}")
        print(f"   Plots:       {plots_dir}")
    
    results = {}
    
    # Create a baseline model for weight initialization
    baseline_weights = None
    
    for recipe_id in recipes:
        recipe_info = SUPPORTED_RECIPES.get(recipe_id, {})
        recipe_name = recipe_info.get("name", f"Recipe {recipe_id}")
        
        if verbose:
            print("\n" + "=" * 70)
            print(f"🧪 TRAINING: {recipe_name} (Recipe {recipe_id})")
            print("=" * 70)
            print(f"   Forward Quantized:  {recipe_info.get('forward_quantized', False)}")
            print(f"   Backward Quantized: {recipe_info.get('backward_quantized', False)}")
            print("=" * 70 + "\n")
        
        # Create model
        model = create_cnn_qat(
            in_channels=3,
            num_classes=1,
            recipe_id=recipe_id
        ).to(device)
        
        # Apply channels_last if enabled
        if use_channels_last:
            model = model.to(memory_format=torch.channels_last)
        
        # Copy baseline weights for fair comparison
        if recipe_id == 0:
            baseline_weights = copy.deepcopy(model.state_dict())
        elif baseline_weights is not None:
            model.load_state_dict(baseline_weights, strict=True)
        
        # Optimizer (Adamax, matching cnn.ipynb)
        optimizer = torch.optim.Adamax(model.parameters(), lr=lr)
        scaler = torch.cuda.amp.GradScaler(enabled=use_amp)
        
        # Training history
        history = {
            "train_loss": [], "val_loss": [],
            "train_dice": [], "val_dice": [],
            "epoch_time": [],
            "train_throughput": [], "val_throughput": []
        }
        
        # Get sample counts for throughput calculation
        train_samples = len(train_loader.dataset)
        val_samples = len(val_loader.dataset)
        
        # Setup telemetry file for this recipe
        telemetry_path = None
        if save_telemetry:
            telemetry_path = telemetry_dir / f"recipe_{recipe_id}_telemetry.csv"
            # Write CSV header
            with open(telemetry_path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "epoch", "train_loss", "val_loss", "train_dice", "val_dice",
                    "epoch_time", "train_throughput", "val_throughput", "cumulative_time"
                ])
        
        start_time = time.perf_counter()
        
        for epoch in range(num_epochs):
            epoch_start = time.perf_counter()
            
            # Train (measure time)
            train_start = time.perf_counter()
            train_loss, train_dice = train_one_epoch(
                model, train_loader, optimizer, dice_loss, device,
                scaler, use_amp, amp_dtype, use_channels_last
            )
            train_time = time.perf_counter() - train_start
            
            # Validate (measure time)
            val_start = time.perf_counter()
            val_loss, val_dice = validate(
                model, val_loader, dice_loss, device,
                use_amp, amp_dtype, use_channels_last
            )
            val_time = time.perf_counter() - val_start
            
            epoch_time = time.perf_counter() - epoch_start
            cumulative_time = time.perf_counter() - start_time
            
            # Calculate throughput (samples per second)
            train_throughput = train_samples / train_time if train_time > 0 else 0.0
            val_throughput = val_samples / val_time if val_time > 0 else 0.0
            
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            history["train_dice"].append(train_dice)
            history["val_dice"].append(val_dice)
            history["epoch_time"].append(epoch_time)
            history["train_throughput"].append(train_throughput)
            history["val_throughput"].append(val_throughput)
            
            # Save telemetry after each epoch (crash recovery)
            if save_telemetry and telemetry_path:
                with open(telemetry_path, "a", newline="") as f:
                    writer = csv.writer(f)
                    writer.writerow([
                        epoch + 1, train_loss, val_loss, train_dice, val_dice,
                        epoch_time, train_throughput, val_throughput, cumulative_time
                    ])
            
            if verbose and (epoch + 1) % 10 == 0:
                print(f"Epoch [{epoch+1}/{num_epochs}] "
                      f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | "
                      f"Train Dice: {train_dice:.4f} | Val Dice: {val_dice:.4f} | "
                      f"Train {train_time:.2f}s, {train_throughput:.1f} samp/s | "
                      f"Val {val_time:.2f}s, {val_throughput:.1f} samp/s")
        
        total_time = time.perf_counter() - start_time
        
        if verbose:
            print(f"\n✅ Recipe {recipe_id} completed in {total_time:.1f}s")
            print(f"   Final Val Dice: {history['val_dice'][-1]:.4f}")
        
        # Save checkpoint to checkpoints/ directory
        checkpoint_path = None
        if save_checkpoints:
            checkpoint_path = checkpoints_dir / f"cnn_recipe_{recipe_id}.pth"
            torch.save(model.state_dict(), checkpoint_path)
            if verbose:
                print(f"   Checkpoint: {checkpoint_path}")
        
        if save_telemetry and telemetry_path:
            if verbose:
                print(f"   Telemetry:  {telemetry_path}")
        
        results[recipe_id] = {
            "model": model,
            "history": history,
            "training_time": total_time,
            "checkpoint": str(checkpoint_path) if checkpoint_path else None,
            "telemetry": str(telemetry_path) if telemetry_path else None,
            "final_val_dice": history["val_dice"][-1],
            "final_val_loss": history["val_loss"][-1],
            "plots_dir": str(plots_dir),  # Pass plots dir for downstream use
        }
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def count_parameters(model: nn.Module) -> int:
    """Count trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_recipe_comparison(results: Dict[int, Dict[str, Any]]):
    """Print a comparison table of training results across recipes."""
    print("\n" + "=" * 70)
    print("📊 RECIPE COMPARISON")
    print("=" * 70)
    print(f"{'Recipe':<10} {'Name':<25} {'Val Dice':<12} {'Time (s)':<12}")
    print("-" * 70)
    
    for recipe_id, data in sorted(results.items(), key=lambda x: str(x[0])):
        name = SUPPORTED_RECIPES.get(recipe_id, {}).get("name", "Unknown")[:25]
        val_dice = data.get("final_val_dice", 0)
        train_time = data.get("training_time", 0)
        print(f"{recipe_id:<10} {name:<25} {val_dice:<12.4f} {train_time:<12.1f}")
    
    print("=" * 70)


def load_telemetry_from_csv(telemetry_dir: str = "telemetry") -> Dict[str, Dict[str, Any]]:
    """
    Load training telemetry from CSV files for crash recovery visualization.
    
    Args:
        telemetry_dir: Path to telemetry directory containing CSV files
        
    Returns:
        Dictionary mapping recipe_id -> {
            "history": {train_loss, val_loss, train_dice, val_dice, ...},
            "epochs_completed": int,
            "final_val_dice": float,
            "training_time": float
        }
    
    Usage:
        # After crash, load telemetry and visualize
        telemetry = load_telemetry_from_csv("telemetry/")
        for recipe_id, data in telemetry.items():
            print(f"Recipe {recipe_id}: {data['epochs_completed']} epochs, "
                  f"Val Dice: {data['final_val_dice']:.4f}")
    """
    import csv
    from pathlib import Path
    
    telemetry_path = Path(telemetry_dir)
    results = {}
    
    if not telemetry_path.exists():
        print(f"⚠️ Telemetry directory not found: {telemetry_dir}")
        return results
    
    for csv_file in telemetry_path.glob("recipe_*_telemetry.csv"):
        # Extract recipe_id from filename
        filename = csv_file.stem  # e.g., "recipe_90046_1_telemetry"
        recipe_id_str = filename.replace("recipe_", "").replace("_telemetry", "")
        
        # Try to parse as int, otherwise keep as string
        try:
            recipe_id = int(recipe_id_str)
        except ValueError:
            recipe_id = recipe_id_str
        
        history = {
            "train_loss": [], "val_loss": [],
            "train_dice": [], "val_dice": [],
            "epoch_time": [],
            "train_throughput": [], "val_throughput": []
        }
        
        try:
            with open(csv_file, "r", newline="") as f:
                reader = csv.DictReader(f)
                total_time = 0
                for row in reader:
                    history["train_loss"].append(float(row["train_loss"]))
                    history["val_loss"].append(float(row["val_loss"]))
                    history["train_dice"].append(float(row["train_dice"]))
                    history["val_dice"].append(float(row["val_dice"]))
                    history["epoch_time"].append(float(row["epoch_time"]))
                    history["train_throughput"].append(float(row["train_throughput"]))
                    history["val_throughput"].append(float(row["val_throughput"]))
                    total_time = float(row["cumulative_time"])
            
            if history["val_dice"]:
                results[recipe_id] = {
                    "history": history,
                    "epochs_completed": len(history["val_dice"]),
                    "final_val_dice": history["val_dice"][-1],
                    "final_val_loss": history["val_loss"][-1],
                    "training_time": total_time,
                    "telemetry_file": str(csv_file)
                }
                print(f"✅ Loaded: Recipe {recipe_id} ({len(history['val_dice'])} epochs)")
        except Exception as e:
            print(f"⚠️ Failed to load {csv_file}: {e}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def print_supported_recipes():
    """
    Print a formatted table of all supported recipes.
    Useful for reference when selecting recipes to train.
    """
    major, minor, gpu_name = detect_gpu_architecture()
    is_gpu_adv = is_nvfp4_gpu_capable()
    
    print("=" * 100)
    print("📋 SUPPORTED CNN QAT RECIPES")
    print("=" * 100)
    print(f"Current Hardware: {gpu_name} (Compute Capability {major}.{minor})")
    print(f"NVFP4-capable GPU: {'✅ Yes' if is_gpu_adv else '❌ No'}")
    print(f"NVFP4 QAT runtime Available: {'✅ Yes' if _QAT_RUNTIME_AVAILABLE else '❌ No'}")
    print(f"NVFP4 QAT runtime Advanced: {'✅ Yes' if _QAT_RUNTIME_ADVANCED_AVAILABLE else '❌ No'}")
    print("=" * 100)
    print()
    
    # Group recipes
    base_recipes = []
    advanced_recipes = []
    
    for recipe_id, info in SUPPORTED_RECIPES.items():
        if info.get("requires_qat_runtime", False):
            advanced_recipes.append((recipe_id, info))
        else:
            base_recipes.append((recipe_id, info))
    
    # Print base recipes
    print("BASELINE AND QAT RECIPES:")
    print("-" * 100)
    print(f"{'ID':<8} {'Name':<35} {'Description':<55}")
    print("-" * 100)
    for recipe_id, info in sorted(base_recipes):
        name = info['name'][:33]
        desc = info['description'][:53]
        print(f"{recipe_id:<8} {name:<35} {desc:<55}")
    print()
    
    # Print advanced recipes
    print("ADVANCED QAT RECIPES:")
    print("-" * 100)
    print(f"{'ID':<8} {'Name':<35} {'Features':<55}")
    print("-" * 100)
    for recipe_id, info in sorted(advanced_recipes):
        name = info['name'][:33]
        features = ', '.join(info.get('features', []))[:53]
        compatible = '✅' if is_gpu_adv else '❌'
        print(f"{recipe_id:<8} {name:<35} {features:<53} {compatible}")
    
    if advanced_recipes and not is_gpu_adv:
        print()
        print("⚠️  Non-baseline recipes require a compatible NVFP4 QAT environment.")
        print("    To use them: Run in a compatible NVFP4 QAT environment")
    
    print()
    print("=" * 100)
    print()
    print("USAGE:")
    print("  # Train base recipes only (works on all hardware)")
    print("  results = run_qat_training_schemes(")
    print("      recipes=[0, 1, 2, 3],")
    print("      num_epochs=300,")
    print("      ...)")
    print()
    print("  # Train all recipes (requires NVFP4-capable runtime for advanced)")
    print("  results = run_qat_training_schemes(")
    print("      recipes=[0, 1, 8, 4, 5, 6, 2, 3, 7],")
    print("      num_epochs=300,")
    print("      ...)")
    print("=" * 100)


# ═══════════════════════════════════════════════════════════════════════════════
# MAIN (Testing)
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"NVFP4 QAT runtime Available: {_QAT_RUNTIME_AVAILABLE}")
    print(f"NVFP4 QAT runtime Advanced: {_QAT_RUNTIME_ADVANCED_AVAILABLE}")
    print()
    
    # Print all supported recipes
    print_supported_recipes()
    
    # All supported recipes (base + advanced)
    base_recipes = [0, 1, 2, 3]
    advanced_recipes = [8, 4, 5, 6, 7]
    all_recipes = base_recipes + (advanced_recipes if _QAT_RUNTIME_ADVANCED_AVAILABLE else [])
    
    # Test model creation for each recipe
    for recipe_id in all_recipes:
        recipe_info = SUPPORTED_RECIPES.get(recipe_id, {})
        recipe_name = recipe_info.get("name", f"Recipe {recipe_id}")
        base_recipe, quant_first_last = parse_recipe_id(recipe_id)
        
        print(f"\n{'='*60}")
        print(f"Testing Recipe {recipe_id}: {recipe_name}")
        print(f"  Base Recipe: {base_recipe}")
        print(f"  Quantize First/Last: {quant_first_last}")
        print(f"{'='*60}")
        
        if base_recipe != 0 and not _QAT_RUNTIME_AVAILABLE:
            print("  ⚠️ Skipping (NVFP4 QAT runtime not available)")
            continue
        
        try:
            model = create_cnn_qat(recipe_id=recipe_id).to(device)
            print(f"  Parameters: {count_parameters(model):,}")
            print(f"  quantize_first_layer: {model.quantize_first_layer}")
            print(f"  quantize_last_layer: {model.quantize_last_layer}")
            
            x = torch.randn(2, 3, 256, 256).to(device)
            model.eval()
            with torch.no_grad():
                out = model(x)
            print(f"  Input: {list(x.shape)} -> Output: {list(out.shape)}")
            print("  ✅ Forward pass successful")
        except Exception as e:
            print(f"  ❌ Error: {e}")
    
    print("\n✅ All tests completed!")

