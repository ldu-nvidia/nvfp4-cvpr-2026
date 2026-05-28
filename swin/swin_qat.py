"""
Swin ViT (Windowed Attention) with QAT support.

Reconstructed from bytecode metadata. Imports shared components from vit_qat.py
and adds Swin-specific windowed multi-head self-attention.
"""

import sys
import math
from typing import Optional, Dict, Any
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

_VIT_PATH = str(Path(__file__).parent.parent / "vit")
if _VIT_PATH not in sys.path:
    sys.path.insert(0, _VIT_PATH)

from vit_qat import (
    NVFPQuantizer, QuantizedLinearFunction, SUPPORTED_RECIPES,
    _QAT_RUNTIME_AVAILABLE, _QAT_RUNTIME_ADVANCED_AVAILABLE,
    PatchEmbedding, DropPath, MLP, SegmentationDecoder,
)

SWIN_CONFIGS = {
    "matched_500k": {"embed_dim": 64, "depth": 10, "num_heads": 4, "mlp_ratio": 3.0, "patch_size": 16, "window_size": 4, "description": "~530K params, win=4 (local)"},
    "matched_4m": {"embed_dim": 192, "depth": 9, "num_heads": 6, "mlp_ratio": 3.0, "patch_size": 16, "window_size": 8, "description": "~3.7M params, win=8 (moderate)"},
    "matched_15m": {"embed_dim": 448, "depth": 8, "num_heads": 14, "mlp_ratio": 2.0, "patch_size": 16, "window_size": 16, "description": "~13.7M params, win=16 (global)"},
}

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size, window_size, W // window_size, window_size, C)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size * window_size, C)

def window_reverse(windows, window_size, H, W):
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.view(B, H // window_size, W // window_size, window_size, window_size, -1)
    return x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)

class WindowedMultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, shift_size, grid_size, dropout=0.0, qkv_bias=True, quantizer=None):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.window_size = window_size
        self.shift_size = shift_size
        self.grid_size = grid_size
        self.quantizer = quantizer
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=qkv_bias)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.attn_drop = nn.Dropout(dropout)
        self.proj_drop = nn.Dropout(dropout)
        self.relative_position_bias_table = nn.Parameter(torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads))
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w], indexing='ij'))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        self.register_buffer("relative_position_index", relative_coords.sum(-1))
        if shift_size > 0:
            H = W = grid_size
            img_mask = torch.zeros((1, H, W, 1))
            h_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            w_slices = (slice(0, -window_size), slice(-window_size, -shift_size), slice(-shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1
            mask_windows = window_partition(img_mask, window_size).view(-1, window_size * window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            self.register_buffer("attn_mask", attn_mask.masked_fill(attn_mask != 0, -100.0).masked_fill(attn_mask == 0, 0.0))
        else:
            self.attn_mask = None

    def _linear(self, x, linear, quantize=True):
        if self.quantizer and quantize:
            return QuantizedLinearFunction.apply(x, linear.weight, linear.bias, self.quantizer)
        return linear(x)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(math.sqrt(N))
        x = x.view(B, H, W, C)
        shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2)) if self.shift_size > 0 else x
        x_windows = window_partition(shifted_x, self.window_size)
        qkv = self._linear(x_windows, self.qkv)
        B_w = x_windows.shape[0]
        qkv = qkv.reshape(B_w, -1, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        rpb = self.relative_position_bias_table[self.relative_position_index.view(-1)].view(self.window_size**2, self.window_size**2, -1).permute(2, 0, 1).contiguous()
        attn = attn + rpb.unsqueeze(0)
        if self.attn_mask is not None:
            nW = self.attn_mask.shape[0]
            ws2 = self.window_size**2; attn = attn.view(B_w // nW, nW, self.num_heads, ws2, ws2) + self.attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, self.window_size**2, self.window_size**2)
        attn = self.attn_drop(F.softmax(attn, dim=-1))
        x_windows = (attn @ v).transpose(1, 2).reshape(B_w, -1, C)
        x_windows = self.proj_drop(self._linear(x_windows, self.proj))
        shifted_x = window_reverse(x_windows, self.window_size, H, W)
        x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2)) if self.shift_size > 0 else shifted_x
        return x.view(B, N, C)

class SwinTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, window_size, shift_size, grid_size, mlp_ratio=2.0, dropout=0.0, drop_path=0.0, quantizer=None):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = WindowedMultiHeadSelfAttention(embed_dim, num_heads, window_size, shift_size, grid_size, dropout=dropout, quantizer=quantizer)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = MLP(embed_dim, mlp_ratio=mlp_ratio, dropout=dropout, quantizer=quantizer)
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class SwinViTQAT(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, num_classes=1, embed_dim=64, depth=10, num_heads=4, mlp_ratio=2.0, window_size=4, dropout=0.1, drop_path=0.1, recipe_id=0):
        super().__init__()
        self.embed_dim = embed_dim
        self.recipe_id = recipe_id
        self.grid_size = img_size // patch_size
        self.quantizer = NVFPQuantizer(recipe_id) if recipe_id != 0 else None
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_chans, embed_dim)
        num_patches = self.grid_size ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout)
        dpr = [x.item() for x in torch.linspace(0, drop_path, depth)]
        self.blocks = nn.ModuleList([
            SwinTransformerBlock(embed_dim=embed_dim, num_heads=num_heads, window_size=window_size,
                shift_size=0 if (i % 2 == 0) else window_size // 2, grid_size=self.grid_size,
                mlp_ratio=mlp_ratio, dropout=dropout, drop_path=dpr[i], quantizer=self.quantizer)
            for i in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim)
        num_upsamples = int(math.log2(patch_size)); self.decoder = SegmentationDecoder(embed_dim, num_classes, num_upsamples)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        x = self.patch_embed(x)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.decoder(x, self.grid_size)

    def get_recipe_info(self):
        return SUPPORTED_RECIPES.get(self.recipe_id, {})

def create_swin_qat(model_size="matched_500k", recipe_id=0, img_size=256, in_chans=3, num_classes=1, dropout=0.1, drop_path=0.1):
    assert model_size in SWIN_CONFIGS, f"model_size must be one of {list(SWIN_CONFIGS.keys())}"
    cfg = SWIN_CONFIGS[model_size]
    return SwinViTQAT(img_size=img_size, patch_size=cfg["patch_size"], in_chans=in_chans, num_classes=num_classes,
        embed_dim=cfg["embed_dim"], depth=cfg["depth"], num_heads=cfg["num_heads"], mlp_ratio=cfg["mlp_ratio"],
        window_size=cfg["window_size"], dropout=dropout, drop_path=drop_path, recipe_id=recipe_id)

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def print_swin_configs():
    for name, cfg in SWIN_CONFIGS.items():
        model = create_swin_qat(model_size=name, recipe_id=0)
        n = count_parameters(model)
        print(f"  {name}: {cfg['description']} | embed_dim={cfg['embed_dim']}, depth={cfg['depth']}, num_heads={cfg['num_heads']}, window_size={cfg['window_size']} | {n:,} params")
        del model
