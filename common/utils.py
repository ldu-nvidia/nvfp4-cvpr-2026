"""
Shared QAT/QAD Training Utilities
==================================

Architecture-agnostic training infrastructure for Quantization-Aware Training (QAT)
and Quantization-Aware Distillation (QAD). Used by CNN, ViT, and Swin architectures
for brain tumor segmentation on LGG MRI data.

All model creation is done via `create_model_fn` callables, so this module has
zero architecture-specific imports and works with any model that takes
(model_size, recipe_id) and returns an nn.Module.

SECTIONS:
1. Random Seed & Reproducibility
2. Data Processing & Parsing (LGG MRI Dataset)
3. Visualization Utilities (sample images, diagnosis distribution)
4. Recipe Constants (colors, names)
5. Data Loading (dataset, dataloaders, transforms)
6. Checkpoint Loading & Telemetry Caching
7. KLD Weight Calibration
8. Metrics Report Saving (JSON + TXT)
9. Training (train_single_recipe, validation loss, early stopping)
10. Prediction Distribution Visualization
11. Config Fingerprinting & Experiment Tracking
12. Validation Inference (run_validation_inference, soft metrics)
13. Loss Functions (compute_qat_qad_loss)
14. Threshold Sweep Experiment
15. QAT vs QAD Comparison & KLD Weight Analysis
16. Training Curves Visualization
17. Ablation Study & Cross-Size Validation

Authors: Zijian Du and Oleg Rybakov
"""

import os
import sys
import copy
import time
import random
import re
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any, Callable
from dataclasses import dataclass, field
from itertools import product

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import roc_curve, auc

# Import unified loss functions and metrics (sibling module in common/)
from loss import (
    # Soft Metrics (no threshold - use for training monitoring & evaluation)
    soft_recall_metric, soft_f2_score_metric, soft_precision_metric, soft_iou_metric,
    dice_coef_metric,
    # Threshold-free curve metrics
    compute_auprc, compute_pr_curve,
    # Hard-thresholded metrics (for confusion matrix & final reporting only)
    recall_metric, f2_score_metric, precision_metric, iou_metric,
    compute_roc_auc, compute_roc_curve,
    # Comprehensive
    compute_all_metrics, print_metrics_summary,
)


# ═══════════════════════════════════════════════════════════════════════════════
# RANDOM SEED & REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    """Set random seeds for reproducibility across all libraries."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    try:
        # Use warn_only=True because some ops (e.g., nll_loss2d) have no deterministic impl
        torch.use_deterministic_algorithms(True, warn_only=True)
    except Exception:
        pass  # May not be available in all PyTorch versions


# ═══════════════════════════════════════════════════════════════════════════════
# DATA PROCESSING - LGG MRI DATASET
# ═══════════════════════════════════════════════════════════════════════════════

# ImageNet normalization constants
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


# Regex pattern for extracting numeric slice ID from filenames
# Matches: ..._45.tif -> id=45, ..._45_mask.tif -> id=45
SLICE_ID_PATTERN = re.compile(r'_(\d+)(?:_mask)?$', re.IGNORECASE)


def folder_and_id(path: str) -> Tuple[str, Optional[int]]:
    """
    Extract folder path and numeric slice ID from a file path.
    
    Uses regex to extract the trailing numeric ID, handling
    both image files (xxx_45.tif) and mask files (xxx_45_mask.tif).
    
    Returns:
        Tuple of (folder_path, slice_id) - slice_id is None if pattern doesn't match
    """
    path = str(path)
    stem = Path(path).stem
    folder = str(Path(path).parent)
    match = SLICE_ID_PATTERN.search(stem)
    return folder, (int(match.group(1)) if match else None)


def positive_negative_diagnosis(mask_path: str) -> int:
    """Determine diagnosis from mask: 1 if tumor present, 0 otherwise."""
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    return 1 if np.any(mask > 0) else 0


def align_images_and_masks(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Align image and mask paths by folder and numeric slice ID.
    
    Args:
        df: DataFrame with 'path' column containing image and mask paths
        
    Returns:
        Tuple of (image_paths, mask_paths) lists in aligned order
    """
    # Split into images and masks
    df_imgs = df[~df['path'].str.contains("mask", case=False, na=False)].copy()
    df_masks = df[df['path'].str.contains("mask", case=False, na=False)].copy()
    
    # Build lookups using (folder, numeric_id) as key
    img_lookup, mask_lookup = {}, {}
    
    for p in df_imgs['path'].tolist():
        f, i = folder_and_id(p)
        if i is not None:
            img_lookup[(f, i)] = p
    
    for p in df_masks['path'].tolist():
        f, i = folder_and_id(p)
        if i is not None:
            mask_lookup[(f, i)] = p
    
    # Find common keys and align
    common_keys = sorted(set(img_lookup.keys()).intersection(mask_lookup.keys()))
    if not common_keys:
        raise RuntimeError("No matched (folder, id) pairs found. Check filename patterns.")
    
    imgs = [img_lookup[k] for k in common_keys]
    masks = [mask_lookup[k] for k in common_keys]
    
    print(f"✅ Aligned {len(imgs)} image-mask pairs")
    return imgs, masks


# ═══════════════════════════════════════════════════════════════════════════════
# VISUALIZATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def denormalize_image(img: np.ndarray, mean: Tuple = IMAGENET_MEAN, std: Tuple = IMAGENET_STD) -> np.ndarray:
    """Denormalize image from ImageNet normalization."""
    img = img.copy()
    for c in range(3):
        img[..., c] = img[..., c] * std[c] + mean[c]
    return np.clip(img, 0, 1)


def plot_diagnosis_distribution(df: pd.DataFrame, save_path: Optional[str] = None, dpi: int = 300):
    """Plot distribution of positive/negative diagnoses."""
    fig, ax = plt.subplots(figsize=(10, 7), dpi=dpi)
    counts = df["diagnosis"].value_counts().sort_index()
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(["Negative (0)", "Positive (1)"], counts.values, color=colors)
    
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(count), ha='center', fontsize=18, fontweight='bold')
    
    ax.set_ylabel("Count", fontsize=18, fontweight='bold')
    ax.set_xlabel("Diagnosis", fontsize=18, fontweight='bold')
    ax.set_title("Diagnosis Distribution in Dataset", fontsize=22, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi)
        print(f"✅ Saved: {save_path}")
    
    plt.show()


def plot_sample_images(df: pd.DataFrame, n_samples: int = 6, save_path: Optional[str] = None, dpi: int = 300):
    """Plot sample images with their masks."""
    samples = df.sample(min(n_samples, len(df)))
    
    fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8), dpi=dpi)
    
    for i, (_, row) in enumerate(samples.iterrows()):
        if i >= n_samples:
            break
        img = cv2.cvtColor(cv2.imread(row["image_path"]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(row["mask_path"], 0)
        
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"ID: {row['patient'][:20]}", fontsize=16)
        axes[0, i].axis("off")
        
        axes[1, i].imshow(mask, cmap="gray")
        axes[1, i].set_title(f"Diag: {row['diagnosis']}", fontsize=16)
        axes[1, i].axis("off")
    
    plt.suptitle("Sample Images and Masks", fontsize=22, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi)
    
    plt.show()



# ═══════════════════════════════════════════════════════════════════════════════
# RECIPE COLOR SCHEME
# ═══════════════════════════════════════════════════════════════════════════════

RECIPE_COLORS = {
    0:     "#3498db",  # Blue - Baseline
    1:  "#e74c3c",  # Red - Full NVFP4
    60031: "#c0392b",  # Dark Red - NVFP4 Full (Skip Bottleneck)
    8:  "#9b59b6",  # Purple - Autograd
    4:  "#2ecc71",  # Green - 2D + RHT
    5:  "#f39c12",  # Orange - 2D + RHT + SR
    6:  "#1abc9c",  # Teal - SR Only
    2: "#34495e",  # Dark Gray - Forward-Only
    3: "#e67e22",  # Dark Orange - Chain Rule
    7: "#16a085",  # Dark Teal - Forward + RHT
}

RECIPE_NAMES = {
    0:     "Baseline",
    1:  "NVFP4 Full",
    60031: "NVFP4 Full (Skip Bottleneck)",
    8:  "Autograd",
    4:  "2D+RHT",
    5:  "2D+RHT+SR",
    6:  "SR Only",
    2: "Fwd-Only",
    3: "Chain Rule",
    7: "Fwd+RHT",
}


_LEGACY_MULTI_SCALE_REMOVED = True  # See cleanup audit

# ═══════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL DATA LOADING UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

import hashlib
import json
import pickle
import glob

import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from sklearn.model_selection import train_test_split


class BrainMriDataset(torch.utils.data.Dataset):
    """Brain MRI segmentation dataset for LGG data."""
    
    def __init__(self, df: pd.DataFrame, transforms):
        self.df = df
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = cv2.imread(self.df.iloc[idx, 1])
        mask = cv2.imread(self.df.iloc[idx, 2], 0)
        augmented = self.transforms(image=image, mask=mask)
        return augmented['image'], augmented['mask']


def get_transforms(
    patch_size: int = 256,
    train: bool = True,
    mean: Tuple = IMAGENET_MEAN,
    std: Tuple = IMAGENET_STD
) -> A.Compose:
    """Get albumentations transforms for training or validation."""
    if train:
        return A.Compose([
            A.Resize(patch_size, patch_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.3),
            A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.1, rotate_limit=15, p=0.3),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])
    else:
        return A.Compose([
            A.Resize(patch_size, patch_size),
            A.Normalize(mean=mean, std=std),
            ToTensorV2()
        ])


def load_lgg_dataset(
    data_path: str,
    seed: int = 2026,
    test_size: float = 0.2,
    val_ratio: float = 0.5
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load and split LGG MRI dataset.
    
    Args:
        data_path: Path to kaggle_3m directory
        seed: Random seed for reproducibility
        test_size: Fraction for test+val combined
        val_ratio: Fraction of test_size for validation
        
    Returns:
        (train_df, val_df, test_df) DataFrames with image_path, mask_path, diagnosis
    """
    # Build file list
    data_map = []
    for sub_dir_path in glob.glob(data_path + "*"):
        if os.path.isdir(sub_dir_path):
            dirname = sub_dir_path.split("/")[-1]
            for filename in os.listdir(sub_dir_path):
                image_path = sub_dir_path + "/" + filename
                data_map.extend([dirname, image_path])
    
    df = pd.DataFrame({"dirname": data_map[::2], "path": data_map[1::2]})
    
    # Align images and masks
    imgs, masks = align_images_and_masks(df)
    
    # Build final dataframe
    df = pd.DataFrame({
        "patient": [p.split("/")[-2] for p in imgs],
        "image_path": imgs,
        "mask_path": masks
    })
    df["diagnosis"] = df["mask_path"].apply(positive_negative_diagnosis)
    
    # Split
    train_df, temp_df = train_test_split(df, test_size=test_size, stratify=df.diagnosis, random_state=seed)
    val_df, test_df = train_test_split(temp_df, test_size=val_ratio, stratify=temp_df.diagnosis, random_state=seed)
    
    train_df = train_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)
    test_df = test_df.reset_index(drop=True)
    
    print(f"✅ Dataset loaded: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    print(f"   Positive samples: {df['diagnosis'].sum()}/{len(df)} ({100*df['diagnosis'].mean():.1f}%)")
    
    return train_df, val_df, test_df


def create_dataloaders(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    batch_size: int = 26,
    num_workers: int = 4,
    patch_size: int = 256
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test dataloaders."""
    train_transforms = get_transforms(patch_size, train=True)
    val_transforms = get_transforms(patch_size, train=False)
    
    train_loader = DataLoader(
        BrainMriDataset(train_df, train_transforms),
        batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    val_loader = DataLoader(
        BrainMriDataset(val_df, val_transforms),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    test_loader = DataLoader(
        BrainMriDataset(test_df, val_transforms),
        batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True
    )
    
    return train_loader, val_loader, test_loader


# ═══════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL CHECKPOINT LOADING
# ═══════════════════════════════════════════════════════════════════════════════

def _load_recipe_telemetry(
    ckpt_dir: Path,
    recipe_id: int,
    model_size: str,
    num_epochs: int,
    loss_type: str,
    config_hash: str,
    kld_weight: float = None
) -> Tuple[bool, str, Optional[Dict]]:
    """
    Load per-recipe telemetry file with strict config validation.
    
    Returns:
        (success: bool, message: str, telemetry_data: dict or None)
    """
    kld_suffix = f"_kld{kld_weight:.4f}" if kld_weight else ""
    telemetry_path = ckpt_dir / f"telemetry_recipe_{recipe_id}_{model_size}_{num_epochs}ep_{loss_type}{kld_suffix}_{config_hash}.pkl"
    
    if not telemetry_path.exists():
        return False, "Telemetry not found", None
    
    try:
        with open(telemetry_path, 'rb') as f:
            data = pickle.load(f)
        
        # ═══ STRICT VALIDATION: Ensure telemetry matches expected config ═══
        stored_hash = data.get("config_hash")
        if stored_hash != config_hash:
            return False, f"Hash mismatch: stored={stored_hash}, expected={config_hash}", None
        
        # Additional paranoia checks
        if data.get("recipe_id") != recipe_id:
            return False, f"Recipe mismatch: stored={data.get('recipe_id')}, expected={recipe_id}", None
        if data.get("model_size") != model_size:
            return False, f"Model size mismatch: stored={data.get('model_size')}, expected={model_size}", None
        if data.get("loss_type") != loss_type:
            return False, f"Loss type mismatch: stored={data.get('loss_type')}, expected={loss_type}", None
        if kld_weight is not None and data.get("kld_weight") != kld_weight:
            return False, f"KLD weight mismatch: stored={data.get('kld_weight')}, expected={kld_weight}", None
        
        return True, f"Loaded from {telemetry_path.name}", data
        
    except Exception as e:
        return False, f"Load error: {e}", None


def load_all_checkpoints(
    ckpt_dir: Path,
    loss_types: List[str],
    recipes: List[int],
    config_hashes: Dict[str, str],
    model_size: str,
    num_epochs: int,
    val_dataloader: DataLoader,
    device: torch.device,
    create_model_fn: Callable,
    force_revalidate: bool = False,
    get_config_fn: Callable = None,
    kld_weights: List[float] = None
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Load all checkpoints and populate results dictionary.
    
    PRIORITY ORDER:
    1. Per-recipe telemetry files (saved during training) - FASTEST, VALIDATED
    2. Checkpoint + live inference (fallback) - SLOWER, but creates telemetry
    
    Args:
        ckpt_dir: Checkpoint directory
        loss_types: List of loss types ["tversky", "qad_tversky"]
        recipes: List of recipe IDs
        config_hashes: Dict mapping loss_type -> config_hash (or nested for QAD)
        model_size: Model scale name
        num_epochs: Number of epochs
        val_dataloader: Validation dataloader
        device: Target device
        create_model_fn: Function to create model
        force_revalidate: Ignore cached telemetry and re-run inference
        get_config_fn: Function to get config dict for a loss type
        kld_weights: List of KLD weights for QAD (if nested structure)
        
    Returns:
        all_results[loss_type][recipe_id] = {...}  (or nested for QAD)
    """
    print("═" * 70)
    print("📂 LOADING CHECKPOINTS & TELEMETRY")
    print("═" * 70)
    
    # Initialize results structure
    all_results = {}
    for lt in loss_types:
        if "qad" in lt and kld_weights:
            all_results[lt] = {kw: {} for kw in kld_weights}
        else:
            all_results[lt] = {}
    
    # Identify QAT loss types (non-QAD) and QAD loss types
    qat_loss_types = [lt for lt in loss_types if "qad" not in lt]
    qad_loss_types = [lt for lt in loss_types if "qad" in lt]
    
    loaded_count = 0
    inference_count = 0
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 1: Load QAT results (any non-QAD loss type)
    # ═══════════════════════════════════════════════════════════════════════════
    for qat_lt in qat_loss_types:
        config_hash = config_hashes.get(qat_lt)
        print(f"\n🔄 {qat_lt.upper()} [hash: {config_hash}]")
        
        for recipe_id in recipes:
            if force_revalidate:
                success = False
                msg = "FORCE_REVALIDATE"
            else:
                success, msg, data = _load_recipe_telemetry(
                    ckpt_dir, recipe_id, model_size, num_epochs, qat_lt, config_hash
                )
            
            if success:
                # If telemetry has no history, try to get it from checkpoint
                if not data.get("history") or not data["history"].get("val_dice"):
                    ckpt_path = get_checkpoint_path(ckpt_dir, recipe_id, model_size, num_epochs, qat_lt, config_hash)
                    if ckpt_path.exists():
                        try:
                            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                            if ckpt.get("history"):
                                data["history"] = ckpt["history"]
                        except:
                            pass
                all_results[qat_lt][recipe_id] = data
                loaded_count += 1
                print(f"   ✅ {RECIPE_NAMES.get(recipe_id, recipe_id)}: Dice={data.get('best_dice', 0):.4f}, AUC={data.get('auc', 0):.4f} (cached)")
            else:
                # Fallback: Load checkpoint and run inference
                ckpt_path = get_checkpoint_path(ckpt_dir, recipe_id, model_size, num_epochs, qat_lt, config_hash)
                if not ckpt_path.exists():
                    # Try without hash (legacy)
                    ckpt_path = ckpt_dir / f"recipe_{recipe_id}_{model_size}_{num_epochs}ep_{qat_lt}.pt"
                
                if ckpt_path.exists():
                    try:
                        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                        
                        # Validate checkpoint config
                        stored_hash = ckpt.get("config_hash")
                        if stored_hash and stored_hash != config_hash:
                            print(f"   ⚠️  {RECIPE_NAMES.get(recipe_id, recipe_id)}: Config mismatch (ckpt={stored_hash}, expected={config_hash})")
                            continue
                        
                        model = create_model_fn(model_size=model_size, recipe_id=recipe_id).to(device)
                        model.load_state_dict(ckpt['state_dict'])
                        model.eval()
                        
                        inference_results = run_validation_inference(model, val_dataloader, device)
                        inference_count += 1
                        
                        result_data = {
                            "recipe_id": recipe_id,
                            "model_size": model_size,
                            "loss_type": qat_lt,
                            "config_hash": config_hash,
                            "best_dice": ckpt.get("best_dice", 0),
                            "history": ckpt.get("history", {}),
                            "checkpoint_path": str(ckpt_path),
                            "num_epochs": num_epochs,
                            "kld_weight": None,
                            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                            **inference_results,
                        }
                        all_results[qat_lt][recipe_id] = result_data
                        loaded_count += 1
                        print(f"   ✅ {RECIPE_NAMES.get(recipe_id, recipe_id)}: Dice={ckpt.get('best_dice', 0):.4f}, AUC={inference_results['auc']:.4f} (inference)")
                        
                        # Save telemetry for future caching
                        telemetry_path = ckpt_dir / f"telemetry_recipe_{recipe_id}_{model_size}_{num_epochs}ep_{qat_lt}_{config_hash}.pkl"
                        try:
                            with open(telemetry_path, 'wb') as f:
                                pickle.dump(result_data, f)
                        except:
                            pass  # Non-critical
                        
                        del model
                        torch.cuda.empty_cache()
                    except Exception as e:
                        print(f"   ✗ {RECIPE_NAMES.get(recipe_id, recipe_id)}: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # PHASE 2: Load QAD results (any QAD loss type, with KLD weight nesting)
    # ═══════════════════════════════════════════════════════════════════════════
    for qad_lt in qad_loss_types:
        qad_hashes = config_hashes.get(qad_lt, {})
        
        # Handle nested structure (kld_weights) or flat structure
        if isinstance(qad_hashes, dict) and kld_weights:
            # Nested: all_results[qad_lt][kld_weight][recipe_id]
            for kld_weight in kld_weights:
                config_hash = qad_hashes.get(kld_weight)
                if not config_hash:
                    continue
                    
                print(f"\n🔄 {qad_lt.upper()} [kld={kld_weight}, hash: {config_hash}]")
                
                for recipe_id in recipes:
                    if recipe_id == 0:  # Skip baseline for QAD
                        continue
                    
                    if force_revalidate:
                        success = False
                        msg = "FORCE_REVALIDATE"
                    else:
                        success, msg, data = _load_recipe_telemetry(
                            ckpt_dir, recipe_id, model_size, num_epochs, qad_lt, config_hash, kld_weight
                        )
                    
                    if success:
                        # If telemetry has no history, try to get it from checkpoint
                        if not data.get("history") or not data["history"].get("val_dice"):
                            ckpt_path = get_checkpoint_path(ckpt_dir, recipe_id, model_size, num_epochs, qad_lt, config_hash, kld_weight)
                            if ckpt_path.exists():
                                try:
                                    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                                    if ckpt.get("history"):
                                        data["history"] = ckpt["history"]
                                except:
                                    pass
                        all_results[qad_lt][kld_weight][recipe_id] = data
                        loaded_count += 1
                        print(f"   ✅ {RECIPE_NAMES.get(recipe_id, recipe_id)}: Dice={data.get('best_dice', 0):.4f}, AUC={data.get('auc', 0):.4f} (cached)")
                    else:
                        # Fallback: Load checkpoint and run inference
                        ckpt_path = get_checkpoint_path(ckpt_dir, recipe_id, model_size, num_epochs, qad_lt, config_hash, kld_weight)
                        if not ckpt_path.exists():
                            continue
                        
                        try:
                            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                            
                            stored_hash = ckpt.get("config_hash")
                            if stored_hash and stored_hash != config_hash:
                                print(f"   ⚠️  {RECIPE_NAMES.get(recipe_id, recipe_id)}: Config mismatch")
                                continue
                            
                            model = create_model_fn(model_size=model_size, recipe_id=recipe_id).to(device)
                            model.load_state_dict(ckpt['state_dict'])
                            model.eval()
                            
                            inference_results = run_validation_inference(model, val_dataloader, device)
                            inference_count += 1
                            
                            result_data = {
                                "recipe_id": recipe_id,
                                "model_size": model_size,
                                "loss_type": qad_lt,
                                "kld_weight": kld_weight,
                                "config_hash": config_hash,
                                "best_dice": ckpt.get("best_dice", 0),
                                "history": ckpt.get("history", {}),
                                "checkpoint_path": str(ckpt_path),
                                "num_epochs": num_epochs,
                                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                                **inference_results,
                            }
                            all_results[qad_lt][kld_weight][recipe_id] = result_data
                            loaded_count += 1
                            print(f"   ✅ {RECIPE_NAMES.get(recipe_id, recipe_id)}: Dice={ckpt.get('best_dice', 0):.4f}, AUC={inference_results['auc']:.4f} (inference)")
                            
                            # Save telemetry for future caching
                            telemetry_path = ckpt_dir / f"telemetry_recipe_{recipe_id}_{model_size}_{num_epochs}ep_{qad_lt}_kld{kld_weight:.4f}_{config_hash}.pkl"
                            try:
                                with open(telemetry_path, 'wb') as f:
                                    pickle.dump(result_data, f)
                            except:
                                pass  # Non-critical
                            
                            del model
                            torch.cuda.empty_cache()
                        except Exception as e:
                            print(f"   ✗ {RECIPE_NAMES.get(recipe_id, recipe_id)}: {e}")
        else:
            # Flat structure (legacy)
            config_hash = qad_hashes if isinstance(qad_hashes, str) else None
            if config_hash:
                print(f"\n🔄 {qad_lt.upper()} [hash: {config_hash}]")
                # ... similar logic for flat structure
    
    # ═══════════════════════════════════════════════════════════════════════════
    # COMPUTE KL FROM BASELINE (post-processing)
    # ═══════════════════════════════════════════════════════════════════════════
    # Get baseline predictions (recipe 0 from any QAT loss type)
    baseline_probs = None
    for qat_lt in qat_loss_types:
        if qat_lt in all_results and 0 in all_results[qat_lt]:
            baseline_probs = all_results[qat_lt][0].get("all_probs")
            if baseline_probs is not None:
                break
    
    if baseline_probs is not None:
        print(f"\n📊 Computing KL(baseline || student) with T=2.0...")
        
        # Compute KL for QAT recipes
        for qat_lt in qat_loss_types:
            for recipe_id, data in all_results.get(qat_lt, {}).items():
                if recipe_id == 0:
                    data["kl_from_baseline"] = 0.0
                    continue
                recipe_probs = data.get("all_probs")
                if recipe_probs is not None:
                    kl = _compute_kl_divergence(recipe_probs, baseline_probs, temperature=2.0)
                    data["kl_from_baseline"] = kl
        
        # Compute KL for QAD recipes (nested structure)
        for qad_lt in qad_loss_types:
            qad_results = all_results.get(qad_lt, {})
            if _is_nested_qad_results(qad_results):
                for kld_weight, recipes_data in qad_results.items():
                    for recipe_id, data in recipes_data.items():
                        recipe_probs = data.get("all_probs")
                        if recipe_probs is not None:
                            kl = _compute_kl_divergence(recipe_probs, baseline_probs, temperature=2.0)
                            data["kl_from_baseline"] = kl
            else:
                for recipe_id, data in qad_results.items():
                    if isinstance(recipe_id, int):
                        recipe_probs = data.get("all_probs")
                        if recipe_probs is not None:
                            kl = _compute_kl_divergence(recipe_probs, baseline_probs, temperature=2.0)
                            data["kl_from_baseline"] = kl
        
        print(f"   ✅ KL(baseline || student) computed for all recipes")
    else:
        print(f"\n⚠️  Baseline (recipe 0) not loaded - cannot compute KL divergence")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SUMMARY
    # ═══════════════════════════════════════════════════════════════════════════
    print(f"\n{'═'*70}")
    print(f"✅ Loaded: {loaded_count} results ({loaded_count - inference_count} cached, {inference_count} inference)")
    
    # Count per loss type (generic)
    for lt in loss_types:
        lt_data = all_results.get(lt, {})
        if "qad" in lt and isinstance(lt_data, dict):
            first_key = next(iter(lt_data.keys()), None)
            if isinstance(first_key, float):
                count = sum(len(v) for v in lt_data.values())
            else:
                count = len(lt_data)
            print(f"   {lt}: {count} configs")
        else:
            print(f"   {lt}: {len(lt_data)} recipes")
    print(f"{'═'*70}")
    
    return all_results


# ═══════════════════════════════════════════════════════════════════════════════
# KLD WEIGHT CALIBRATION
# ═══════════════════════════════════════════════════════════════════════════════

def calibrate_kld_weights(
    student_model: nn.Module,
    teacher_model: nn.Module,
    train_dataloader: DataLoader,
    device: torch.device,
    target_ratios: List[float],
    qad_temperature: float = 2.0,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7,
    num_calibration_batches: int = 10,
    verbose: bool = True,
) -> Dict[float, float]:
    """
    Calibrate KLD weights so that the KLD term contributes a target percentage
    of the total loss. This ensures scientifically meaningful comparisons across
    different distillation strengths.

    The QAD total loss is:
        total = task_weight * task_loss + kld_weight * T^2 * kld_loss

    For a target ratio r (e.g., 0.20 = 20%):
        r = kld_weight * T^2 * kld_loss / total
        => kld_weight = (r / (1-r)) * task_loss / (T^2 * kld_loss)

    Args:
        student_model: Student model (will be put in eval mode, NOT modified)
        teacher_model: Teacher model (already in eval mode)
        train_dataloader: Training dataloader for calibration batches
        device: Target device
        target_ratios: List of target KLD loss ratios (e.g., [0.05, 0.20, 0.40])
        qad_temperature: Temperature for KLD softening
        tversky_alpha: Tversky FP weight
        tversky_beta: Tversky FN weight
        num_calibration_batches: Number of batches to average over
        verbose: Print calibration details

    Returns:
        Dict mapping target_ratio -> calibrated kld_weight
        e.g., {0.05: 0.013, 0.20: 0.063, 0.40: 0.167}
    """
    student_model.eval()
    teacher_model.eval()

    task_loss_sum = 0.0
    kld_loss_sum = 0.0
    n_batches = 0
    T = qad_temperature

    with torch.no_grad():
        for images, masks in train_dataloader:
            if n_batches >= num_calibration_batches:
                break

            images = images.to(device)
            masks = masks.to(device)

            # Ensure mask shape and normalization
            if masks.ndim == 3:
                masks = masks.unsqueeze(1)
            masks = masks.float()
            if masks.max() > 1.0:
                masks = masks / 255.0

            with torch.cuda.amp.autocast():
                student_logits = student_model(images)
                teacher_logits = teacher_model(images)

            logits_f32 = student_logits.float().clamp(-20, 20)

            # Task loss: BCE + Tversky (same as compute_qat_qad_loss)
            bce_loss = F.binary_cross_entropy_with_logits(logits_f32, masks)
            probs = torch.sigmoid(logits_f32)
            eps = 1e-5
            TP = (probs * masks).sum(dim=(2, 3))
            FP = (probs * (1 - masks)).sum(dim=(2, 3))
            FN = ((1 - probs) * masks).sum(dim=(2, 3))
            tversky_index = (TP + eps) / (TP + tversky_alpha * FP + tversky_beta * FN + eps)
            tversky_loss_val = 1.0 - tversky_index.mean()
            task_loss = bce_loss + tversky_loss_val

            # KLD loss: BCE between temperature-softened student/teacher (unweighted)
            teacher_logits_f32 = teacher_logits.float().clamp(-20, 20)
            student_probs_T = torch.sigmoid(logits_f32 / T).clamp(1e-6, 1 - 1e-6)
            teacher_probs_T = torch.sigmoid(teacher_logits_f32 / T).clamp(1e-6, 1 - 1e-6)
            kld_loss = F.binary_cross_entropy(student_probs_T, teacher_probs_T.detach())

            task_loss_sum += task_loss.item()
            kld_loss_sum += kld_loss.item()
            n_batches += 1

    avg_task_loss = task_loss_sum / n_batches
    avg_kld_loss = kld_loss_sum / n_batches
    effective_kld = T ** 2 * avg_kld_loss  # T^2 scaling applied in actual loss

    if verbose:
        print(f"\n   📊 KLD Weight Calibration ({n_batches} batches)")
        print(f"      Avg task loss (BCE+Tversky): {avg_task_loss:.4f}")
        print(f"      Avg KLD loss (unweighted):   {avg_kld_loss:.4f}")
        print(f"      Effective KLD (T²={T**2:.0f} × KLD): {effective_kld:.4f}")

    calibrated_weights = {}
    for r in target_ratios:
        assert 0 < r < 1, f"Target ratio must be in (0, 1), got {r}"
        # Solve: r = w * T^2 * kld / (task + w * T^2 * kld)
        # => w = (r / (1-r)) * task / (T^2 * kld)
        w = (r / (1 - r)) * avg_task_loss / effective_kld
        calibrated_weights[r] = w

        if verbose:
            actual_kld_contrib = w * effective_kld
            actual_total = avg_task_loss + actual_kld_contrib
            actual_ratio = actual_kld_contrib / actual_total
            print(f"      Ratio {r*100:5.1f}% → kld_weight={w:.4f}"
                  f"  (verify: {actual_kld_contrib:.4f}/{actual_total:.4f} = {actual_ratio*100:.1f}%)")

    if verbose:
        print()

    return calibrated_weights


# ═══════════════════════════════════════════════════════════════════════════════
# METRICS REPORT SAVING
# ═══════════════════════════════════════════════════════════════════════════════

def _build_config_filename(
    prefix: str,
    model_size: str,
    num_epochs: int,
    loss_type: str,
    recipe_id: int = None,
    kld_weight: float = None,
    kld_ratio: float = None,
    config_hash: str = None,
    ext: str = "json",
) -> str:
    """Build an explicit, human-readable filename encoding the full training config."""
    parts = [prefix]
    if recipe_id is not None:
        name = RECIPE_NAMES.get(recipe_id, str(recipe_id)).replace(" ", "").replace("+", "")
        parts.append(f"r{recipe_id}_{name}")
    parts.append(model_size)
    parts.append(f"{num_epochs}ep")
    parts.append(loss_type)
    if kld_ratio is not None:
        parts.append(f"kld{kld_ratio*100:.0f}pct")
    elif kld_weight is not None:
        parts.append(f"kldw{kld_weight:.4f}")
    if config_hash:
        parts.append(config_hash)
    return "_".join(parts) + f".{ext}"


def save_metrics_report(
    metrics: Dict[str, Any],
    save_dir: Path,
    prefix: str,
    model_size: str,
    num_epochs: int,
    loss_type: str,
    recipe_id: int = None,
    kld_weight: float = None,
    kld_ratio: float = None,
    config_hash: str = None,
    config_dict: dict = None,
    extra_info: dict = None,
) -> Tuple[Path, Path]:
    """
    Save metrics as both JSON (machine-readable) and TXT (human-readable) files.

    Filenames explicitly encode the full training configuration to prevent
    any confusion between runs.

    Example filenames:
        metrics_r6003_NVFP4Full_matched_100k_100ep_tversky_04426ef6.json
        metrics_r6003_NVFP4Full_matched_100k_100ep_qad_tversky_kld20pct_a1b2c3d4.json
        summary_QAT_matched_100k_100ep_tversky_04426ef6.txt

    Args:
        metrics: Dictionary of metric values
        save_dir: Directory to save files in
        prefix: File prefix (e.g., "metrics", "summary", "calibration", "threshold_sweep")
        model_size: Model scale name
        num_epochs: Number of epochs
        loss_type: Loss type string
        recipe_id: Recipe ID (optional, for per-recipe files)
        kld_weight: KLD weight value (optional, for QAD)
        kld_ratio: KLD target ratio (optional, for QAD - preferred over kld_weight for naming)
        config_hash: Config hash (optional)
        config_dict: Full config dictionary (optional, included in JSON)
        extra_info: Additional key-value pairs to include in the report

    Returns:
        Tuple of (json_path, txt_path)
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    base_name = _build_config_filename(
        prefix, model_size, num_epochs, loss_type,
        recipe_id=recipe_id, kld_weight=kld_weight if kld_ratio is None else None,
        kld_ratio=kld_ratio, config_hash=config_hash
    )

    # ═══ JSON (machine-readable) ═══
    json_path = save_dir / base_name
    json_data = {
        "config": {
            "model_size": model_size,
            "num_epochs": num_epochs,
            "loss_type": loss_type,
            "config_hash": config_hash,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "metrics": {},
    }
    if recipe_id is not None:
        json_data["config"]["recipe_id"] = recipe_id
        json_data["config"]["recipe_name"] = RECIPE_NAMES.get(recipe_id, str(recipe_id))
    if kld_weight is not None:
        json_data["config"]["kld_weight"] = kld_weight
    if kld_ratio is not None:
        json_data["config"]["kld_target_ratio"] = kld_ratio
    if config_dict:
        json_data["config"]["full_config"] = config_dict
    if extra_info:
        json_data["extra"] = extra_info

    # Filter metrics to JSON-serializable values
    for k, v in metrics.items():
        if isinstance(v, (int, float, str, bool, type(None))):
            json_data["metrics"][k] = v
        elif isinstance(v, np.floating):
            json_data["metrics"][k] = float(v)
        elif isinstance(v, np.integer):
            json_data["metrics"][k] = int(v)
        elif isinstance(v, dict):
            # Nested dict (e.g., per-recipe results) - filter recursively
            sub = {}
            for sk, sv in v.items():
                if isinstance(sv, (int, float, str, bool, type(None))):
                    sub[str(sk)] = sv
                elif isinstance(sv, (np.floating, np.integer)):
                    sub[str(sk)] = float(sv)
            if sub:
                json_data["metrics"][k] = sub

    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)

    # ═══ TXT (human-readable) ═══
    txt_path = save_dir / base_name.replace(".json", ".txt")
    with open(txt_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"METRICS REPORT: {prefix.upper()}\n")
        f.write(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 80 + "\n\n")

        f.write("TRAINING CONFIGURATION\n")
        f.write("-" * 40 + "\n")
        f.write(f"  Model Size:    {model_size}\n")
        f.write(f"  Epochs:        {num_epochs}\n")
        f.write(f"  Loss Type:     {loss_type}\n")
        if recipe_id is not None:
            f.write(f"  Recipe:        {recipe_id} ({RECIPE_NAMES.get(recipe_id, 'Unknown')})\n")
        if kld_ratio is not None:
            f.write(f"  KLD Ratio:     {kld_ratio*100:.1f}%\n")
        if kld_weight is not None:
            f.write(f"  KLD Weight:    {kld_weight:.6f}\n")
        if config_hash:
            f.write(f"  Config Hash:   {config_hash}\n")
        f.write("\n")

        if config_dict:
            f.write("FULL CONFIG\n")
            f.write("-" * 40 + "\n")
            for k, v in sorted(config_dict.items()):
                f.write(f"  {k}: {v}\n")
            f.write("\n")

        f.write("METRICS\n")
        f.write("-" * 40 + "\n")
        for k, v in sorted(metrics.items()):
            if isinstance(v, (int, float, np.floating, np.integer)):
                f.write(f"  {k:<30s} {float(v):.6f}\n")
            elif isinstance(v, str):
                f.write(f"  {k:<30s} {v}\n")
            elif isinstance(v, dict):
                f.write(f"  {k}:\n")
                for sk, sv in sorted(v.items(), key=lambda x: str(x[0])):
                    if isinstance(sv, (int, float, np.floating, np.integer)):
                        f.write(f"    {str(sk):<28s} {float(sv):.6f}\n")
        f.write("\n")

        if extra_info:
            f.write("ADDITIONAL INFO\n")
            f.write("-" * 40 + "\n")
            for k, v in sorted(extra_info.items()):
                f.write(f"  {k}: {v}\n")
            f.write("\n")

        f.write("=" * 80 + "\n")

    return json_path, txt_path


def save_phase_summary(
    all_results: Dict,
    save_dir: Path,
    phase_name: str,
    model_size: str,
    num_epochs: int,
    loss_type: str,
    config_hash: str = None,
    config_dict: dict = None,
    kld_ratio: float = None,
    kld_weight: float = None,
    recipe_names: Dict[int, str] = None,
) -> Tuple[Path, Path]:
    """
    Save a summary of all recipes for a training phase (QAT or one QAD ratio).

    Args:
        all_results: Dict[recipe_id] -> result dict (for this phase)
        save_dir: Directory to save
        phase_name: e.g., "QAT", "QAD_5pct", "QAD_20pct"
        Others: config identifiers

    Returns:
        Tuple of (json_path, txt_path)
    """
    if recipe_names is None:
        recipe_names = RECIPE_NAMES

    summary_metrics = {}
    for recipe_id, result in sorted(all_results.items()):
        if not isinstance(recipe_id, int):
            continue
        name = recipe_names.get(recipe_id, str(recipe_id))
        summary_metrics[f"r{recipe_id}_{name}"] = {
            "recall": result.get("recall", 0),
            "f2_score": result.get("f2_score", 0),
            "dice": result.get("dice", result.get("best_dice", 0)),
            "auprc": result.get("auprc", 0),
            "precision": result.get("precision", 0),
            "iou": result.get("iou", 0),
            "auc": result.get("auc", 0),
            "kl_from_baseline": result.get("kl_from_baseline", 0),
            "ece": result.get("ece", 0),
            "best_epoch": result.get("best_epoch", 0),
            "actual_epochs": result.get("actual_epochs", 0),
            "early_stopped": result.get("early_stopped", False),
        }

    return save_metrics_report(
        metrics=summary_metrics,
        save_dir=save_dir,
        prefix=f"summary_{phase_name}",
        model_size=model_size,
        num_epochs=num_epochs,
        loss_type=loss_type,
        config_hash=config_hash,
        config_dict=config_dict,
        kld_ratio=kld_ratio,
        kld_weight=kld_weight,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# HIGH-LEVEL TRAINING FUNCTION
# ═══════════════════════════════════════════════════════════════════════════════

def train_single_recipe(
    recipe_id: int,
    loss_type: str,
    model_size: str,
    num_epochs: int,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
    create_model_fn: Callable,
    baseline_weights: dict,
    ckpt_dir: Path,
    config_hash: str,
    config_dict: dict,
    teacher_model: nn.Module = None,
    qad_params: Dict = None,
    target_dice: float = 0.5,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    optimizer_type: str = "adamw",  # "adamw" or "adamax"
    verbose: bool = True,
    # ═══ EARLY STOPPING & BEST CHECKPOINT SAVING ═══
    early_stopping_patience: int = 20,
    early_stopping_metric: str = "loss",  # "loss", "f2", "recall", or "dice"
    early_stopping_min_delta: float = 0.001,
    early_stopping_warmup: int = 0,  # Disable early stopping for first N epochs
) -> Dict[str, Any]:
    """
    Train a single recipe and return results.
    
    Args:
        recipe_id: Recipe ID to train
        loss_type: "tversky" (QAT) or "qad_tversky" (QAD)
        model_size: Model scale name
        num_epochs: Number of training epochs (max if early stopping)
        train_dataloader: Training dataloader
        val_dataloader: Validation dataloader
        device: Target device
        create_model_fn: Function to create model
        baseline_weights: Initial weights for fair comparison
        ckpt_dir: Checkpoint save directory
        config_hash: Config hash for checkpointing
        config_dict: Full config dictionary
        teacher_model: Teacher model for QAD (required if loss_type contains "qad")
        qad_params: Dict with task_weight, distill_weight, temperature
        target_dice: Target dice for convergence tracking
        learning_rate: Learning rate
        weight_decay: Weight decay
        optimizer_type: "adamw" (default, good for ViT) or "adamax" (good for CNN)
        verbose: Print training progress
        early_stopping_patience: Stop after N epochs without improvement (default: 20)
        early_stopping_metric: Metric to monitor - "loss", "f2", "recall", or "dice" (default: "loss")
        early_stopping_warmup: Disable early stopping for first N epochs (default: 0)
        early_stopping_min_delta: Minimum improvement to reset patience (default: 0.001)
        
    Returns:
        Results dictionary with history, metrics, probabilities
    """
    recipe_name = RECIPE_NAMES.get(recipe_id, f"Recipe {recipe_id}")
    
    # Create and initialize model
    model = create_model_fn(model_size=model_size, recipe_id=recipe_id).to(device)
    model.load_state_dict(baseline_weights, strict=False)
    
    # Optimizer and scheduler
    if optimizer_type == "adamax":
        optimizer = torch.optim.Adamax(model.parameters(), lr=learning_rate)
    else:  # "adamw" (default)
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=10)
    scaler = torch.amp.GradScaler('cuda')
    
    # QAD parameters
    qad_task_weight = qad_params.get("task_weight", 1.0) if qad_params else 1.0
    qad_distill_weight = qad_params.get("distill_weight", 1.0) if qad_params else 1.0
    qad_temperature = qad_params.get("temperature", 2.0) if qad_params else 2.0
    
    # Training history (includes F2 and Recall for early stopping)
    history = {
        "train_dice": [], "val_dice": [], "val_f2": [], "val_recall": [],
        "bce_loss": [], "dice_loss": [], "tversky_loss": [], "kld_loss": [], "total_loss": [],
        "val_loss": [],  # Validation loss (for early stopping)
        "grad_norm": [], "epoch_time": []
    }
    
    # ═══ BEST METRIC TRACKING & EARLY STOPPING ═══
    # Use -inf so the first epoch always counts as an improvement
    best_metric_value = float('-inf')
    best_val_dice = 0.0
    best_val_f2 = 0.0
    best_val_recall = 0.0
    best_epoch = 0
    patience_counter = 0
    best_state_dict = None  # Store best model weights
    
    epochs_to_target = None
    start_time = time.time()
    
    # Log file (include kld weight for QAD)
    kld_suffix = f"_kld{qad_distill_weight}" if "qad" in loss_type else ""
    log_path = ckpt_dir / f"recipe_{recipe_id}_{model_size}_{num_epochs}ep_{loss_type}{kld_suffix}.txt"
    with open(log_path, "w") as f:
        f.write(f"Recipe {recipe_id} ({recipe_name}) - {loss_type.upper()}\n")
        f.write(f"Model: {model_size}, Max Epochs: {num_epochs}, KLD Weight: {qad_distill_weight if 'qad' in loss_type else 'N/A'}\n")
        f.write(f"Early Stopping: metric={early_stopping_metric}, patience={early_stopping_patience}, min_delta={early_stopping_min_delta}\n")
        f.write("=" * 90 + "\n")
        f.write("Epoch,BCE,Dice,Tversky,KLD,Total,TrainDice,ValDice,ValF2,ValRecall,GradNorm,EpochTime\n")
    
    for epoch in range(num_epochs):
        epoch_start = time.time()
        model.train()
        
        train_loss, train_dice, grad_norm_sum = 0.0, 0.0, 0.0
        bce_sum, dice_sum, tversky_sum, kld_sum = 0.0, 0.0, 0.0, 0.0
        n_batches = 0
        
        for images, masks in train_dataloader:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            
            with torch.amp.autocast('cuda'):
                logits = model(images)
                loss, loss_components = compute_qat_qad_loss(
                    logits, masks, loss_type,
                    teacher_model, images,
                    qad_task_weight, qad_distill_weight, qad_temperature
                )
            
            if torch.isnan(loss) or torch.isinf(loss):
                continue
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            grad_norm_sum += grad_norm.item()
            scaler.step(optimizer)
            scaler.update()
            
            train_loss += loss_components["total"]
            bce_sum += loss_components["bce"]
            dice_sum += loss_components["dice"]
            tversky_sum += loss_components.get("tversky", 0.0)
            kld_sum += loss_components.get("kld", 0.0)
            
            with torch.no_grad():
                probs = torch.sigmoid(logits.float())
                m = masks.unsqueeze(1) if masks.ndim == 3 else masks
                if m.max() > 1.0:
                    m = m / 255.0
                train_dice += dice_coef_metric(probs, m.float())
            
            n_batches += 1
        
        if n_batches == 0:
            continue
        
        train_dice /= n_batches
        avg_bce = bce_sum / n_batches
        avg_dice_loss = dice_sum / n_batches
        avg_tversky = tversky_sum / n_batches
        avg_kld = kld_sum / n_batches
        avg_total = train_loss / n_batches
        avg_grad_norm = grad_norm_sum / n_batches
        
        # ═══════════════════════════════════════════════════════════════════════════
        # VALIDATION: Compute loss, Dice, F2, and Recall on validation set
        # ═══════════════════════════════════════════════════════════════════════════
        model.eval()
        all_val_probs = []
        all_val_targets = []
        val_dice = 0.0
        val_loss_sum = 0.0
        val_batches = 0
        
        with torch.no_grad():
            for images, masks in val_dataloader:
                images, masks = images.to(device), masks.to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(images)
                    # Compute validation loss (same loss function as training)
                    _, val_loss_components = compute_qat_qad_loss(
                        logits, masks, loss_type,
                        teacher_model, images,
                        qad_task_weight, qad_distill_weight, qad_temperature
                    )
                val_loss_sum += val_loss_components["total"]
                val_batches += 1
                
                probs = torch.sigmoid(logits.float())
                m = masks.unsqueeze(1) if masks.ndim == 3 else masks
                if m.max() > 1.0:
                    m = m / 255.0
                val_dice += dice_coef_metric(probs, m.float())
                
                # Collect for F2/Recall computation
                all_val_probs.append(probs.cpu())
                all_val_targets.append(m.float().cpu())
        
        val_dice /= len(val_dataloader)
        val_loss = val_loss_sum / val_batches
        
        # Compute soft F2 and Recall on full validation set (no threshold artifact)
        all_val_probs = torch.cat(all_val_probs, dim=0)
        all_val_targets = torch.cat(all_val_targets, dim=0)
        val_f2 = soft_f2_score_metric(all_val_probs, all_val_targets)
        val_recall = soft_recall_metric(all_val_probs, all_val_targets)
        
        # Scheduler step on the monitored metric
        if early_stopping_metric == "f2":
            scheduler.step(val_f2)
            current_metric = val_f2
        elif early_stopping_metric == "recall":
            scheduler.step(val_recall)
            current_metric = val_recall
        elif early_stopping_metric == "loss":
            # For loss: lower is better, so we negate for the "higher is better" logic
            scheduler.step(-val_loss)
            current_metric = -val_loss
        else:  # "dice"
            scheduler.step(val_dice)
            current_metric = val_dice
        epoch_time = time.time() - epoch_start
        
        # Record history
        history["train_dice"].append(train_dice)
        history["val_dice"].append(val_dice)
        history["val_f2"].append(val_f2)
        history["val_recall"].append(val_recall)
        history["bce_loss"].append(avg_bce)
        history["dice_loss"].append(avg_dice_loss)
        history["tversky_loss"].append(avg_tversky)
        history["kld_loss"].append(avg_kld)
        history["total_loss"].append(avg_total)
        history["val_loss"].append(val_loss)
        history["grad_norm"].append(avg_grad_norm)
        history["epoch_time"].append(epoch_time)
        
        # ═══════════════════════════════════════════════════════════════════════════
        # EARLY STOPPING & BEST CHECKPOINT SAVING
        # ═══════════════════════════════════════════════════════════════════════════
        improved = current_metric > best_metric_value + early_stopping_min_delta
        
        if improved:
            best_metric_value = current_metric
            best_val_dice = val_dice
            best_val_f2 = val_f2
            best_val_recall = val_recall
            best_epoch = epoch + 1
            patience_counter = 0
            # Save best model weights (deep copy)
            best_state_dict = copy.deepcopy(model.state_dict())
        else:
            patience_counter += 1
        
        if val_dice >= target_dice and epochs_to_target is None:
            epochs_to_target = epoch + 1
        
        # Log (with F2 and Recall)
        with open(log_path, "a") as f:
            f.write(f"{epoch+1},{avg_bce:.4f},{avg_dice_loss:.4f},{avg_tversky:.4f},{avg_kld:.4f},{avg_total:.4f},{train_dice:.4f},{val_dice:.4f},{val_f2:.4f},{val_recall:.4f},{avg_grad_norm:.4f},{epoch_time:.2f}\n")
        
        if verbose:
            # Build loss breakdown string based on loss type
            if "tversky" in loss_type:
                loss_str = f"BCE={avg_bce:.4f} Tvk={avg_tversky:.4f}"
            else:
                loss_str = f"BCE={avg_bce:.4f} Dice={avg_dice_loss:.4f}"
            if "qad" in loss_type:
                loss_str += f" KLD={avg_kld:.4f}"
            loss_str += f" Total={avg_total:.4f}"
            if (epoch + 1) <= early_stopping_warmup:
                early_stop_str = f" [W={epoch+1}/{early_stopping_warmup}]"  # Warmup phase
            elif patience_counter > 0:
                early_stop_str = f" [P={patience_counter}/{early_stopping_patience}]"
            else:
                early_stop_str = " ★"
            metric_str = f"ValLoss={val_loss:.4f} F2={val_f2:.4f} Dice={val_dice:.4f} Recall={val_recall:.4f}"
            print(f"   Ep {epoch+1:3d}: {loss_str} | {metric_str}{early_stop_str} | {epoch_time:.1f}s")
        
        # ═══ EARLY STOPPING CHECK (respects warmup) ═══
        if (epoch + 1) > early_stopping_warmup and patience_counter >= early_stopping_patience:
            if verbose:
                print(f"\n   ⏹️  Early stopping triggered at epoch {epoch+1} (no improvement for {early_stopping_patience} epochs)")
                print(f"   Best {early_stopping_metric.upper()}: {best_metric_value:.4f} at epoch {best_epoch}")
            break
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FINAL EVALUATION (on BEST checkpoint, not final epoch)
    # ═══════════════════════════════════════════════════════════════════════════
    actual_epochs = epoch + 1 if 'epoch' in dir() else num_epochs
    
    # Load best weights for final evaluation (if we have them)
    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
        if verbose:
            print(f"\n   📊 Final evaluation using best checkpoint (epoch {best_epoch})")
    
    inference_results = run_validation_inference(model, val_dataloader, device)
    
    results = {
        "history": history,
        "best_dice": best_val_dice,
        "best_f2": best_val_f2,
        "best_recall": best_val_recall,
        "best_epoch": best_epoch,
        "actual_epochs": actual_epochs,
        "early_stopped": patience_counter >= early_stopping_patience,
        "convergence": epochs_to_target or actual_epochs,
        "time": time.time() - start_time,
        **inference_results,
    }
    
    # Save BEST checkpoint (not final epoch)
    kld_weight = qad_distill_weight if "qad" in loss_type else None
    ckpt_path = get_checkpoint_path(ckpt_dir, recipe_id, model_size, num_epochs, loss_type, config_hash, kld_weight)
    
    # Use best_state_dict if available, otherwise current model
    save_state_dict = best_state_dict if best_state_dict is not None else model.state_dict()
    
    torch.save({
        "recipe_id": recipe_id,
        "model_size": model_size,
        "loss_type": loss_type,
        "config_hash": config_hash,
        "config": config_dict,
        "kld_weight": kld_weight,
        "state_dict": save_state_dict,  # BEST weights, not final
        "best_dice": best_val_dice,
        "best_f2": best_val_f2,
        "best_recall": best_val_recall,
        "best_epoch": best_epoch,
        "actual_epochs": actual_epochs,
        "early_stopped": patience_counter >= early_stopping_patience,
        "history": history,
    }, ckpt_path)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SAVE TELEMETRY IMMEDIATELY (avoids redundant inference on reload)
    # ═══════════════════════════════════════════════════════════════════════════
    # Telemetry file is per-recipe, per-config - contains FULL inference results
    # This ensures we NEVER mix up results from different training configs
    kld_suffix = f"_kld{kld_weight:.4f}" if kld_weight else ""
    telemetry_path = ckpt_dir / f"telemetry_recipe_{recipe_id}_{model_size}_{num_epochs}ep_{loss_type}{kld_suffix}_{config_hash}.pkl"
    
    telemetry_data = {
        # ═══ PROVENANCE: Exactly what training config produced these results ═══
        "recipe_id": recipe_id,
        "model_size": model_size,
        "num_epochs": num_epochs,
        "loss_type": loss_type,
        "kld_weight": kld_weight,
        "config_hash": config_hash,
        "config": config_dict,  # Full config for audit
        "checkpoint_path": str(ckpt_path),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        
        # ═══ TRAINING RESULTS (BEST CHECKPOINT) ═══
        "best_dice": best_val_dice,
        "best_f2": best_val_f2,
        "best_recall": best_val_recall,
        "best_epoch": best_epoch,
        "actual_epochs": actual_epochs,
        "early_stopped": patience_counter >= early_stopping_patience,
        "history": history,
        "convergence_epoch": epochs_to_target,
        "training_time": time.time() - start_time,
        
        # ═══ INFERENCE RESULTS (from run_validation_inference on BEST checkpoint) ═══
        **inference_results,
    }
    
    with open(telemetry_path, 'wb') as f:
        pickle.dump(telemetry_data, f)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SAVE HUMAN-READABLE METRICS REPORT (JSON + TXT)
    # ═══════════════════════════════════════════════════════════════════════════
    report_metrics = {
        # First Tier
        "recall": inference_results.get("recall", 0),
        "f2_score": inference_results.get("f2_score", 0),
        "dice": inference_results.get("dice", 0),
        "auprc": inference_results.get("auprc", 0),
        # Second Tier
        "precision": inference_results.get("precision", 0),
        "iou": inference_results.get("iou", 0),
        "auc": inference_results.get("auc", 0),
        # Calibration & Distribution
        "ece": inference_results.get("ece", 0),
        "logit_mean": inference_results.get("logit_mean", 0),
        "logit_std": inference_results.get("logit_std", 0),
        "prob_entropy": inference_results.get("prob_entropy", 0),
        # Confusion Matrix
        "tp": inference_results.get("tp", 0),
        "fp": inference_results.get("fp", 0),
        "fn": inference_results.get("fn", 0),
        "tn": inference_results.get("tn", 0),
        # Training Info
        "best_dice_training": best_val_dice,
        "best_f2_training": best_val_f2,
        "best_recall_training": best_val_recall,
        "best_epoch": best_epoch,
        "actual_epochs": actual_epochs,
        "early_stopped": patience_counter >= early_stopping_patience,
        "training_time_seconds": time.time() - start_time,
    }

    # Determine KLD ratio for filename (if available from config_dict)
    kld_ratio_for_name = None
    if "qad" in loss_type and config_dict:
        # Try to find the target ratio from the kld_weight
        # (will be set properly by the notebook)
        pass

    metrics_dir = ckpt_dir / "metrics"
    save_metrics_report(
        metrics=report_metrics,
        save_dir=metrics_dir,
        prefix="metrics",
        model_size=model_size,
        num_epochs=num_epochs,
        loss_type=loss_type,
        recipe_id=recipe_id,
        kld_weight=qad_distill_weight if "qad" in loss_type else None,
        config_hash=config_hash,
        config_dict=config_dict,
    )
    
    if verbose:
        early_str = f" (early stopped at {actual_epochs})" if patience_counter >= early_stopping_patience else ""
        print(f"   ✅ Best @ ep {best_epoch}: F2={best_val_f2:.4f} Recall={best_val_recall:.4f} Dice={best_val_dice:.4f}{early_str}")
        print(f"   📁 Saved: {ckpt_path.name}")
    
    del model
    torch.cuda.empty_cache()
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# PREDICTION DISTRIBUTION VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_prediction_distributions(
    val_df: pd.DataFrame,
    all_results: Dict[str, Dict[int, Dict]],
    ckpt_dir: Path,
    model_size: str,
    num_epochs: int,
    device: torch.device,
    create_model_fn: Callable,
    loss_types: List[str] = None,
    save_path: Optional[str] = None,
    use_log_scale: bool = True
):
    """
    Plot prediction distribution histograms for all recipes.
    
    Args:
        val_df: Validation dataframe
        all_results: Results dictionary
        ckpt_dir: Checkpoint directory
        model_size: Model scale
        num_epochs: Number of epochs
        device: Target device
        create_model_fn: Model factory function
        loss_types: Loss types to visualize (default: auto-discovered from all_results)
        save_path: Base path for saving (appends _{loss_type}.png)
        use_log_scale: Use log scale on y-axis (recommended for imbalanced data)
    """
    val_transforms = get_transforms(256, train=False)
    
    # Find sample with tumor
    sample_idx = None
    for idx, row in val_df.iterrows():
        if row['diagnosis'] == 1:
            sample_idx = idx
            break
    
    if sample_idx is None:
        print("No tumor image found")
        return
    
    # Load sample
    sample_image = cv2.imread(val_df.iloc[sample_idx, 1])
    sample_mask = cv2.imread(val_df.iloc[sample_idx, 2], 0)
    augmented = val_transforms(image=sample_image, mask=sample_mask)
    sample_tensor = augmented['image'].unsqueeze(0).to(device)
    
    # Auto-discover loss types if not provided
    if loss_types is None:
        loss_types = [k for k in all_results.keys() if isinstance(k, str)]
    
    # Get baseline prediction (find any recipe 0 checkpoint)
    baseline_pred = None
    for lt in loss_types:
        if "qad" in lt:
            continue
        baseline_ckpt = None
        for pattern in [f"recipe_0_{model_size}_{num_epochs}ep_{lt}_*.pt", f"recipe_0_{model_size}_{num_epochs}ep_{lt}.pt"]:
            matches = list(ckpt_dir.glob(pattern))
            if matches:
                baseline_ckpt = matches[0]
                break
        if baseline_ckpt and baseline_ckpt.exists():
            ckpt = torch.load(baseline_ckpt, map_location=device, weights_only=False)
            baseline_model = create_model_fn(model_size=model_size, recipe_id=0).to(device)
            baseline_model.load_state_dict(ckpt['state_dict'])
            baseline_model.eval()
            with torch.no_grad():
                baseline_pred = torch.sigmoid(baseline_model(sample_tensor)).squeeze().cpu().numpy()
            del baseline_model
            torch.cuda.empty_cache()
            break
    
    for loss_type in loss_types:
        results = all_results.get(loss_type, {})
        if not results:
            continue
        
        loss_label = "QAT" if "qad" not in loss_type else "QAD"
        predictions = {}
        
        for recipe_id in sorted(results.keys()):
            ckpt_path = ckpt_dir / f"recipe_{recipe_id}_{model_size}_{num_epochs}ep_{loss_type}.pt"
            if not ckpt_path.exists() and "qad" in loss_type and recipe_id == 0:
                # Fallback: teacher checkpoint may have been saved under QAT loss type
                ckpt_path = ckpt_dir / f"recipe_0_{model_size}_{num_epochs}ep_tversky.pt"
            if not ckpt_path.exists():
                continue
            
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            model = create_model_fn(model_size=model_size, recipe_id=recipe_id).to(device)
            model.load_state_dict(ckpt['state_dict'])
            model.eval()
            with torch.no_grad():
                predictions[recipe_id] = torch.sigmoid(model(sample_tensor)).squeeze().cpu().numpy()
            del model
            torch.cuda.empty_cache()
        
        if not predictions:
            continue
        
        # Plot with log scale to reveal tail distribution
        n = len(predictions)
        fig, axes = plt.subplots(2, (n + 1) // 2, figsize=(5 * ((n + 1) // 2), 10))
        axes = axes.flatten()
        
        for i, (rid, pred) in enumerate(sorted(predictions.items())):
            ax = axes[i]
            # Use more bins and log scale to see the full distribution
            counts, bins, _ = ax.hist(pred.flatten(), bins=100, alpha=0.7, color='steelblue', edgecolor='none')
            if baseline_pred is not None:
                ax.hist(baseline_pred.flatten(), bins=100, alpha=0.0, edgecolor='red', linewidth=1.5, histtype='step', label='Baseline')
            
            ax.set_title(RECIPE_NAMES.get(rid, str(rid)), fontsize=12, fontweight='bold')
            ax.set_xlabel('Probability')
            ax.set_xlim(0, 1)
            
            if use_log_scale:
                ax.set_yscale('log')
                ax.set_ylabel('Count (log)')
                ax.set_ylim(bottom=0.5)  # Avoid log(0)
            
            # Enhanced stats including percentiles
            p50 = np.percentile(pred, 50)
            p90 = np.percentile(pred, 90)
            p99 = np.percentile(pred, 99)
            stats = f"μ={pred.mean():.3f} σ={pred.std():.3f}\np50={p50:.3f} p90={p90:.3f}\np99={p99:.3f} >0.5:{100*(pred>0.5).mean():.1f}%"
            ax.text(0.97, 0.97, stats, transform=ax.transAxes, fontsize=8, va='top', ha='right',
                    bbox=dict(boxstyle='round', facecolor='white', alpha=0.9), family='monospace')
            
            if baseline_pred is not None:
                ax.legend(loc='upper center', fontsize=8)
        
        for i in range(n, len(axes)):
            axes[i].axis('off')
        
        scale_label = " (Log Scale)" if use_log_scale else ""
        plt.suptitle(f'Prediction Distribution ({loss_label}){scale_label}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            path = save_path.replace('.png', f'_{loss_type}.png')
            plt.savefig(path, dpi=150, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# CONFIG FINGERPRINTING & EXPERIMENT TRACKING
# ═══════════════════════════════════════════════════════════════════════════════


def compute_config_hash(config_dict: dict) -> str:
    """
    Create deterministic 8-character hash of training configuration.
    Used to detect config changes and invalidate cache.
    
    Args:
        config_dict: Dictionary of training hyperparameters
        
    Returns:
        8-character MD5 hash string
    """
    config_str = json.dumps(config_dict, sort_keys=True)
    return hashlib.md5(config_str.encode()).hexdigest()[:8]


def get_training_config(
    model_size: str,
    num_epochs: int,
    batch_size: int,
    learning_rate: float,
    weight_decay: float,
    loss_type: str,
    seed: int,
    qad_task_weight: float = 1.0,
    qad_distill_weight: float = 1.0,
    qad_temperature: float = 2.0
) -> dict:
    """
    Build training config dict for a specific loss type.
    
    Args:
        model_size: Model scale (e.g., "matched_300k")
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay
        loss_type: "tversky" (QAT) or "qad_tversky" (QAD)
        seed: Random seed
        qad_*: QAD-specific hyperparameters (only used if "qad" in loss_type)
        
    Returns:
        Configuration dictionary
    """
    config = {
        "model_size": model_size,
        "num_epochs": num_epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weight_decay": weight_decay,
        "loss_type": loss_type,
        "seed": seed,
    }
    if "qad" in loss_type:
        config["qad_task_weight"] = qad_task_weight
        config["qad_distill_weight"] = qad_distill_weight
        config["qad_temperature"] = qad_temperature
    return config


def get_checkpoint_path(
    ckpt_dir: Path,
    recipe_id: int,
    model_size: str,
    num_epochs: int,
    loss_type: str,
    config_hash: str = None,
    kld_weight: float = None
) -> Path:
    """
    Get checkpoint path with optional config hash suffix.
    
    Args:
        ckpt_dir: Checkpoint directory
        recipe_id: Recipe ID
        model_size: Model scale name
        num_epochs: Number of epochs
        loss_type: Loss type ("tversky" or "qad_tversky")
        config_hash: Optional config hash for versioned checkpoints
        kld_weight: KLD weight for QAD (included in filename for clarity)
        
    Returns:
        Path to checkpoint file
    """
    base_name = f"recipe_{recipe_id}_{model_size}_{num_epochs}ep_{loss_type}"
    # Add KLD weight to filename for QAD to make it human-readable (4 decimal places)
    if "qad" in loss_type and kld_weight is not None:
        base_name = f"{base_name}_kld{kld_weight:.4f}"
    if config_hash:
        return ckpt_dir / f"{base_name}_{config_hash}.pt"
    return ckpt_dir / f"{base_name}.pt"


def get_telemetry_cache_path(ckpt_dir: Path, loss_type: str, config_hash: str, kld_weight: float = None) -> Path:
    """Get telemetry cache path for a loss type and config hash."""
    if "qad" in loss_type and kld_weight is not None:
        return ckpt_dir / f"telemetry_cache_{loss_type}_kld{kld_weight:.4f}_{config_hash}.pkl"
    return ckpt_dir / f"telemetry_cache_{loss_type}_{config_hash}.pkl"


# ═══════════════════════════════════════════════════════════════════════════════
# HIERARCHICAL EXPERIMENT LOGGING SYSTEM
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class ExperimentPaths:
    """Container for all paths related to an experiment configuration."""
    base_dir: Path           # e.g., vit_ckpts/matched_300k
    checkpoint: Path         # e.g., vit_ckpts/matched_300k/qat/recipe_6003_100ep.pt
    telemetry: Path          # e.g., vit_ckpts/matched_300k/telemetry/recipe_6003_qat.pkl
    log: Path                # e.g., vit_ckpts/matched_300k/logs/recipe_6003_qat.txt
    
    def ensure_dirs(self):
        """Create all parent directories."""
        self.checkpoint.parent.mkdir(parents=True, exist_ok=True)
        self.telemetry.parent.mkdir(parents=True, exist_ok=True)
        self.log.parent.mkdir(parents=True, exist_ok=True)


def setup_experiment_dirs(base_ckpt_dir: Path, model_sizes: List[str] = None) -> Dict[str, Path]:
    """
    Create hierarchical directory structure for experiments.
    
    Args:
        base_ckpt_dir: Base checkpoint directory (e.g., vit_ckpts/)
        model_sizes: List of model sizes to create dirs for
        
    Returns:
        Dict mapping model_size to its base directory
    """
    if model_sizes is None:
        model_sizes = ["matched_100k", "matched_300k", "matched_1m"]
    
    dirs = {}
    for size in model_sizes:
        size_dir = base_ckpt_dir / size
        # Create subdirectories
        (size_dir / "qat").mkdir(parents=True, exist_ok=True)
        (size_dir / "qad").mkdir(parents=True, exist_ok=True)
        (size_dir / "telemetry").mkdir(parents=True, exist_ok=True)
        (size_dir / "logs").mkdir(parents=True, exist_ok=True)
        (size_dir / "plots").mkdir(parents=True, exist_ok=True)
        dirs[size] = size_dir
    
    # Create global files location
    (base_ckpt_dir / "summaries").mkdir(parents=True, exist_ok=True)
    
    return dirs


def get_experiment_paths(
    base_ckpt_dir: Path,
    model_size: str,
    recipe_id: int,
    loss_type: str,
    num_epochs: int,
    kld_weight: float = None,
    config_hash: str = None
) -> ExperimentPaths:
    """
    Get all paths for a specific experiment configuration.
    
    Args:
        base_ckpt_dir: Base checkpoint directory
        model_size: Model size (e.g., "matched_300k")
        recipe_id: Recipe ID
        loss_type: "tversky" (qat) or "qad_tversky" (qad)
        num_epochs: Number of epochs
        kld_weight: KLD weight for QAD
        config_hash: Config hash for versioning
        
    Returns:
        ExperimentPaths with all relevant paths
    """
    size_dir = base_ckpt_dir / model_size
    loss_dir = "qad" if "qad" in loss_type else "qat"
    
    # Build filename components (4 decimal places for KLD weight)
    base_name = f"recipe_{recipe_id}_{num_epochs}ep"
    if "qad" in loss_type and kld_weight is not None:
        base_name = f"{base_name}_kld{kld_weight:.4f}"
    if config_hash:
        base_name = f"{base_name}_{config_hash}"
    
    return ExperimentPaths(
        base_dir=size_dir,
        checkpoint=size_dir / loss_dir / f"{base_name}.pt",
        telemetry=size_dir / "telemetry" / f"{base_name}.pkl",
        log=size_dir / "logs" / f"{base_name}.txt"
    )


def get_plot_path(base_ckpt_dir: Path, model_size: str, plot_name: str) -> Path:
    """Get path for a plot file."""
    return base_ckpt_dir / model_size / "plots" / plot_name


def save_ablation_summary(
    results: Dict[str, Dict[str, Any]],
    base_ckpt_dir: Path,
    model_size: str,
    filename: str = "ablation_summary.csv"
) -> Path:
    """
    Save ablation study results to CSV.
    
    Args:
        results: Ablation results dict
        base_ckpt_dir: Base checkpoint directory
        model_size: Model size
        filename: Output filename
        
    Returns:
        Path to saved CSV
    """
    rows = []
    for config_key, r in results.items():
        rows.append({
            "config_key": config_key,
            "model_size": model_size,
            "loss_type": r.get("loss_type", ""),
            "recipe_id": r.get("recipe_id", 0),
            "kld_weight": r.get("kld_weight"),
            "best_dice": r.get("best_dice", 0),
            "auc": r.get("auc", 0),
            "kl_from_baseline": r.get("kl_from_baseline", 0),
            "ece": r.get("ece", 0),
            "logit_mean": r.get("logit_mean", 0),
            "logit_std": r.get("logit_std", 0),
            "entropy": r.get("prob_entropy", 0),
        })
    
    df = pd.DataFrame(rows)
    save_path = base_ckpt_dir / "summaries" / f"{model_size}_{filename}"
    save_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(save_path, index=False)
    print(f"✅ Saved ablation summary: {save_path}")
    return save_path


def load_ablation_summary(base_ckpt_dir: Path, model_size: str, filename: str = "ablation_summary.csv") -> pd.DataFrame:
    """Load ablation summary CSV."""
    path = base_ckpt_dir / "summaries" / f"{model_size}_{filename}"
    if path.exists():
        return pd.read_csv(path)
    return pd.DataFrame()


def get_checkpoint_path_v2(
    base_ckpt_dir: Path,
    model_size: str,
    recipe_id: int,
    num_epochs: int,
    loss_type: str,
    kld_weight: float = None,
    config_hash: str = None
) -> Path:
    """
    Get checkpoint path using hierarchical structure (v2).
    
    Directory structure:
        base_ckpt_dir/
        └── {model_size}/
            ├── qat/
            │   └── recipe_{id}_{epochs}ep[_{hash}].pt
            └── qad/
                └── recipe_{id}_{epochs}ep_kld{weight}[_{hash}].pt
    """
    paths = get_experiment_paths(
        base_ckpt_dir, model_size, recipe_id, loss_type, 
        num_epochs, kld_weight, config_hash
    )
    return paths.checkpoint


def get_telemetry_path_v2(
    base_ckpt_dir: Path,
    model_size: str,
    recipe_id: int,
    num_epochs: int,
    loss_type: str,
    kld_weight: float = None,
    config_hash: str = None
) -> Path:
    """Get telemetry cache path using hierarchical structure (v2)."""
    paths = get_experiment_paths(
        base_ckpt_dir, model_size, recipe_id, loss_type,
        num_epochs, kld_weight, config_hash
    )
    return paths.telemetry


def get_log_path_v2(
    base_ckpt_dir: Path,
    model_size: str,
    recipe_id: int,
    num_epochs: int,
    loss_type: str,
    kld_weight: float = None,
    config_hash: str = None
) -> Path:
    """Get training log path using hierarchical structure (v2)."""
    paths = get_experiment_paths(
        base_ckpt_dir, model_size, recipe_id, loss_type,
        num_epochs, kld_weight, config_hash
    )
    return paths.log


def print_experiment_structure(base_ckpt_dir: Path, model_size: str = None):
    """Print the experiment directory structure."""
    print(f"\n{'═'*60}")
    print(f"📁 EXPERIMENT DIRECTORY STRUCTURE")
    print(f"{'═'*60}")
    print(f"Base: {base_ckpt_dir}")
    
    sizes = [model_size] if model_size else ["matched_100k", "matched_300k", "matched_1m"]
    
    for size in sizes:
        size_dir = base_ckpt_dir / size
        if size_dir.exists():
            print(f"\n└── {size}/")
            for subdir in ["qat", "qad", "telemetry", "logs", "plots"]:
                sub_path = size_dir / subdir
                if sub_path.exists():
                    files = list(sub_path.glob("*"))
                    print(f"    ├── {subdir}/ ({len(files)} files)")
    
    summary_dir = base_ckpt_dir / "summaries"
    if summary_dir.exists():
        summaries = list(summary_dir.glob("*.csv"))
        print(f"\n└── summaries/ ({len(summaries)} files)")
    
    print(f"{'═'*60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# BACKWARD COMPATIBLE CHECKPOINT VALIDATION
# ═══════════════════════════════════════════════════════════════════════════════

def validate_checkpoint(ckpt_path: Path, expected_hash: str) -> Tuple[bool, str, Optional[dict]]:
    """
    Validate a checkpoint against expected config hash.
    
    Args:
        ckpt_path: Path to checkpoint
        expected_hash: Expected config hash
        
    Returns:
        (is_valid: bool, message: str, checkpoint_data: dict or None)
    """
    if not ckpt_path.exists():
        return False, "Checkpoint not found", None
    
    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        stored_hash = ckpt.get("config_hash", None)
        
        if stored_hash is None:
            return False, "Legacy checkpoint (no config hash)", ckpt
        
        if stored_hash != expected_hash:
            return False, f"Config mismatch: stored={stored_hash}, expected={expected_hash}", ckpt
        
        return True, "Valid checkpoint", ckpt
    except Exception as e:
        return False, f"Load error: {e}", None


def load_manifest(manifest_path: Path) -> dict:
    """Load experiment manifest from disk."""
    if manifest_path.exists():
        try:
            with open(manifest_path, 'r') as f:
                return json.load(f)
        except Exception:
            pass
    return {"experiments": {}, "version": "1.0"}


def save_manifest(manifest: dict, manifest_path: Path):
    """Save experiment manifest to disk."""
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)


def update_manifest(
    manifest_path: Path,
    recipe_id: int,
    loss_type: str,
    config_hash: str,
    metrics: dict,
    ckpt_path: str,
    model_size: str,
    num_epochs: int
):
    """Update manifest with new experiment entry."""
    manifest = load_manifest(manifest_path)
    key = f"{loss_type}_recipe_{recipe_id}"
    manifest["experiments"][key] = {
        "config_hash": config_hash,
        "checkpoint": str(ckpt_path),
        "metrics": metrics,
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "model_size": model_size,
        "num_epochs": num_epochs,
    }
    save_manifest(manifest, manifest_path)


def compute_ece(probs: np.ndarray, targets: np.ndarray, n_bins: int = 15) -> float:
    """
    Compute Expected Calibration Error.
    
    Args:
        probs: Predicted probabilities
        targets: Ground truth labels (0 or 1)
        n_bins: Number of calibration bins
        
    Returns:
        ECE value (lower is better)
    """
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (probs > bins[i]) & (probs <= bins[i+1])
        if mask.mean() > 0:
            ece += np.abs(targets[mask].mean() - probs[mask].mean()) * mask.mean()
    return ece


# ═══════════════════════════════════════════════════════════════════════════════
# TELEMETRY CACHING & VALIDATION INFERENCE
# ═══════════════════════════════════════════════════════════════════════════════

def load_telemetry_cache(
    cache_path: Path,
    expected_hash: str
) -> Tuple[bool, str, Optional[dict]]:
    """
    Load and validate telemetry cache.
    
    Args:
        cache_path: Path to cache file
        expected_hash: Expected config hash
        
    Returns:
        (is_valid: bool, reason: str, cached_results: dict or None)
    """
    if not cache_path.exists():
        return False, "Cache not found", None
    
    try:
        with open(cache_path, 'rb') as f:
            cached_data = pickle.load(f)
        
        stored_hash = cached_data.get("config_hash", None)
        if stored_hash != expected_hash:
            return False, f"Hash mismatch: stored={stored_hash}, expected={expected_hash}", None
        
        cached_results = cached_data.get("results", {})
        if not cached_results or "all_probs" not in list(cached_results.values())[0]:
            return False, "Missing inference data", None
        
        return True, f"Loaded {len(cached_results)} recipes", cached_results
        
    except Exception as e:
        return False, f"Load error: {e}", None


def save_telemetry_cache(
    cache_path: Path,
    config_hash: str,
    config: dict,
    results: dict
):
    """
    Save telemetry cache to disk.
    
    Args:
        cache_path: Path to save cache
        config_hash: Config hash for validation
        config: Full config dictionary
        results: Results dictionary to cache
    """
    with open(cache_path, 'wb') as f:
        pickle.dump({
            "config_hash": config_hash,
            "config": config,
            "results": results,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }, f)


def run_validation_inference(
    model: nn.Module,
    val_dataloader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    threshold: float = 0.5
) -> Dict[str, Any]:
    """
    Run inference on validation data and compute all telemetry metrics.
    
    Computes comprehensive metrics including:
    
    First Tier (Primary):
        - recall: Sensitivity (% of positives caught) - YOUR #1 METRIC
        - f2_score: Recall-weighted F-score (recall 2x more important)
        - dice: Dice coefficient (segmentation overlap)
        - auprc: Area Under Precision-Recall Curve (honest for imbalanced)
        
    Second Tier (Secondary):
        - precision: Positive Predictive Value
        - iou: Jaccard index
        - auc: ROC-AUC (can be inflated for imbalanced data)
    
    Curves (for visualization):
        - fpr, tpr: ROC curve data
        - pr_precision, pr_recall: PR curve data
    
    Args:
        model: Trained model (already loaded and in eval mode)
        val_dataloader: Validation dataloader
        device: Target device
        use_amp: Use automatic mixed precision
        threshold: Decision threshold for point metrics (default: 0.5)
        
    Returns:
        Dictionary with all metrics, curves, and raw data
    """
    model.eval()
    all_probs = []
    all_targets = []
    dice_sum = 0.0  # Per-image macro-average Dice (matches training loop)
    n_batches = 0
    
    with torch.no_grad():
        for images, masks in val_dataloader:
            images = images.to(device)
            
            with torch.cuda.amp.autocast(enabled=use_amp):
                logits = model(images)
            
            probs_batch = torch.sigmoid(logits.float()).cpu()
            
            # Normalize masks
            masks_float = masks.float()
            if masks_float.max() > 1.0:
                masks_float = masks_float / 255.0
            
            # Per-image macro-average Dice (consistent with training loop)
            dice_sum += dice_coef_metric(probs_batch, masks_float)
            n_batches += 1
            
            # Flatten for pixel-level metrics (recall, F2, AUPRC, ROC, etc.)
            all_probs.extend(probs_batch.numpy().flatten())
            all_targets.extend(masks_float.numpy().flatten())
    
    all_probs = np.array(all_probs)
    all_targets = np.array(all_targets)
    
    # Clip probabilities for numerical stability
    all_probs = np.clip(all_probs, 1e-7, 1 - 1e-7)
    
    # Convert to tensors for our metric functions
    probs_tensor = torch.tensor(all_probs)
    targets_tensor = torch.tensor(all_targets)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # FIRST TIER METRICS (Primary - All soft/threshold-free)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Soft Recall (probability-weighted, no threshold artifact)
    recall = soft_recall_metric(probs_tensor, targets_tensor)
    
    # Soft F2-Score (recall-weighted, no threshold artifact)
    f2 = soft_f2_score_metric(probs_tensor, targets_tensor)
    
    # Soft Dice (per-image macro-average, consistent with training loop)
    dice = dice_sum / n_batches
    
    # AUPRC (threshold-free: integrates over all thresholds)
    auprc = compute_auprc(probs_tensor, targets_tensor)
    
    # PR Curve data (for visualization - sweeps thresholds internally)
    pr_precision, pr_recall, pr_thresholds = compute_pr_curve(probs_tensor, targets_tensor)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # SECOND TIER METRICS (Secondary - All soft/threshold-free)
    # ═══════════════════════════════════════════════════════════════════════════
    
    # Soft Precision
    precision = soft_precision_metric(probs_tensor, targets_tensor)
    
    # Soft IoU (Jaccard)
    iou = soft_iou_metric(probs_tensor, targets_tensor)
    
    # ROC-AUC (standard but can be inflated)
    fpr, tpr, _ = roc_curve(all_targets, all_probs)
    auc_score = auc(fpr, tpr)
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CALIBRATION & DISTRIBUTION METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    
    # ECE (Expected Calibration Error)
    ece = compute_ece(all_probs, all_targets)
    
    # Logit statistics
    logits_np = np.log(all_probs / (1 - all_probs))
    logit_mean = np.mean(logits_np)
    logit_std = np.std(logits_np)
    
    # Entropy
    prob_entropy = -np.mean(
        all_probs * np.log(all_probs + 1e-10) + 
        (1 - all_probs) * np.log(1 - all_probs + 1e-10)
    )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # CONFUSION MATRIX COMPONENTS
    # ═══════════════════════════════════════════════════════════════════════════
    preds = (all_probs >= threshold).astype(np.float32)
    tp = np.sum((preds == 1) & (all_targets > 0.5))
    fp = np.sum((preds == 1) & (all_targets <= 0.5))
    fn = np.sum((preds == 0) & (all_targets > 0.5))
    tn = np.sum((preds == 0) & (all_targets <= 0.5))
    
    return {
        # Raw data
        "all_probs": all_probs,
        "all_targets": all_targets,
        
        # First Tier Metrics (Primary)
        "recall": recall,
        "f2_score": f2,
        "dice": dice,
        "auprc": auprc,
        
        # Second Tier Metrics (Secondary)
        "precision": precision,
        "iou": iou,
        "auc": auc_score,  # ROC-AUC (keep as 'auc' for backward compatibility)
        
        # Curves (for visualization)
        "fpr": fpr,
        "tpr": tpr,
        "pr_precision": pr_precision,
        "pr_recall": pr_recall,
        
        # Calibration & Distribution
        "ece": ece,
        "logit_mean": logit_mean,
        "logit_std": logit_std,
        "prob_entropy": prob_entropy,
        
        # Confusion Matrix
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": int(tn),
    }


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING UTILITIES - LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════

def compute_qat_qad_loss(
    student_logits: torch.Tensor,
    masks: torch.Tensor,
    loss_type: str,
    teacher_model: nn.Module = None,
    images: torch.Tensor = None,
    qad_task_weight: float = 1.0,
    qad_distill_weight: float = 1.0,
    qad_temperature: float = 2.0,
    tversky_alpha: float = 0.3,
    tversky_beta: float = 0.7
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Compute QAT or QAD loss with all components.
    
    Uses BCE + Tversky as the unified task loss. Tversky is a generalization
    of Dice: with α=β=0.5 it's equivalent to Dice, with α=0.3/β=0.7 it's
    recall-focused (penalizes FN 2.3x more than FP).
    
    Args:
        student_logits: Student model output logits
        masks: Ground truth masks
        loss_type: Loss type - one of:
            - "tversky": BCE + Tversky (QAT, configurable α/β)
            - "qad_tversky": BCE + Tversky + KLD distillation (QAD)
        teacher_model: Teacher model (required for QAD)
        images: Input images (required for QAD)
        qad_task_weight: Weight for task loss in QAD
        qad_distill_weight: Weight for distillation loss in QAD
        qad_temperature: Temperature for KLD softening
        tversky_alpha: Tversky FP weight (default: 0.3 = low FP penalty)
            Set α=β=0.5 for Dice-equivalent behavior.
        tversky_beta: Tversky FN weight (default: 0.7 = high FN penalty)
        
    Returns:
        (total_loss, loss_components_dict)
    """
    # Ensure mask shape matches logits
    if masks.ndim == 3:
        masks = masks.unsqueeze(1)
    masks = masks.float()
    
    # Normalize masks to [0, 1] if in [0, 255] range
    if masks.max() > 1.0:
        masks = masks / 255.0
    
    # Cast to float32 for numerical stability
    logits_f32 = student_logits.float().clamp(-20, 20)
    
    # BCE loss (used by all loss types)
    bce_loss = F.binary_cross_entropy_with_logits(logits_f32, masks)
    
    # Compute probabilities
    probs = torch.sigmoid(logits_f32)
    eps = 1e-5
    
    # Dice loss (balanced)
    intersection = (probs * masks).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + masks.sum(dim=(2, 3))
    dice_coef = (2. * intersection + eps) / (union + eps)
    dice_loss_val = 1.0 - dice_coef.mean()
    
    # Tversky loss (recall-focused) - computed for all types for logging
    TP = (probs * masks).sum(dim=(2, 3))
    FP = (probs * (1 - masks)).sum(dim=(2, 3))
    FN = ((1 - probs) * masks).sum(dim=(2, 3))
    tversky_index = (TP + eps) / (TP + tversky_alpha * FP + tversky_beta * FN + eps)
    tversky_loss_val = 1.0 - tversky_index.mean()
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LOSS TYPE: tversky (QAT - BCE + Tversky, configurable α/β)
    # Tversky is a generalization of Dice: α=β=0.5 → Dice, α=0.3/β=0.7 → recall-focused
    # ═══════════════════════════════════════════════════════════════════════════
    if "qad" not in loss_type:
        total_loss = bce_loss + tversky_loss_val
        return total_loss, {
            "bce": bce_loss.item(),
            "tversky": tversky_loss_val.item(),
            "dice": dice_loss_val.item(),  # Logged for reference (Tversky generalizes Dice)
            "total": total_loss.item()
        }
    
    # ═══════════════════════════════════════════════════════════════════════════
    # LOSS TYPE: qad_tversky (QAD - BCE + Tversky + KLD distillation)
    # ═══════════════════════════════════════════════════════════════════════════
    else:
        assert teacher_model is not None and images is not None, \
            "QAD requires teacher_model and images"
        
        task_loss = bce_loss + tversky_loss_val
        
        # Get teacher predictions
        with torch.no_grad():
            teacher_logits = teacher_model(images)
        
        # Distillation loss with temperature
        T = qad_temperature
        teacher_logits_f32 = teacher_logits.float().clamp(-20, 20)
        student_probs_T = torch.sigmoid(logits_f32 / T)
        teacher_probs_T = torch.sigmoid(teacher_logits_f32 / T)
        
        # Clamp for stability
        eps_kld = 1e-6
        student_probs_T = student_probs_T.clamp(eps_kld, 1 - eps_kld)
        teacher_probs_T = teacher_probs_T.clamp(eps_kld, 1 - eps_kld)
        
        # KLD via BCE
        with torch.amp.autocast('cuda', enabled=False):
            distill_loss = F.binary_cross_entropy(student_probs_T, teacher_probs_T.detach())
        
        # Combined loss (T^2 scaling for KLD)
        total_loss = qad_task_weight * task_loss + qad_distill_weight * (T ** 2) * distill_loss
        
        return total_loss, {
            "bce": bce_loss.item(),
            "tversky": tversky_loss_val.item(),
            "dice": dice_loss_val.item(),  # Logged for reference
            "kld": distill_loss.item(),
            "total": total_loss.item()
        }


def should_train_recipe(
    recipe_id: int,
    loss_type: str,
    config_hash: str,
    ckpt_dir: Path,
    model_size: str,
    num_epochs: int,
    force_retrain: bool = False,
    kld_weight: float = None
) -> Tuple[bool, str, Optional[Path]]:
    """
    Determine if training is needed for this recipe or can skip.
    
    Args:
        recipe_id: Recipe ID to check
        loss_type: Loss type
        config_hash: Current config hash
        ckpt_dir: Checkpoint directory
        model_size: Model scale name
        num_epochs: Number of epochs
        force_retrain: If True, always retrain
        kld_weight: KLD weight for QAD (required for correct filename matching)
        
    Returns:
        (should_train: bool, reason: str, existing_ckpt_path: Path or None)
    """
    if force_retrain:
        return True, "FORCE_RETRAIN=True", None
    
    # Check for checkpoint with matching config hash (includes kld_weight for QAD)
    ckpt_path_new = get_checkpoint_path(ckpt_dir, recipe_id, model_size, num_epochs, loss_type, config_hash, kld_weight)
    ckpt_path_legacy = get_checkpoint_path(ckpt_dir, recipe_id, model_size, num_epochs, loss_type, kld_weight=kld_weight)
    
    # Try new format first
    if ckpt_path_new.exists():
        is_valid, msg, _ = validate_checkpoint(ckpt_path_new, config_hash)
        if is_valid:
            return False, f"Valid checkpoint exists [{config_hash}]", ckpt_path_new
    
    # Try legacy format
    if ckpt_path_legacy.exists():
        is_valid, msg, _ = validate_checkpoint(ckpt_path_legacy, config_hash)
        if is_valid:
            return False, "Valid checkpoint exists (legacy)", ckpt_path_legacy
        elif "Legacy checkpoint" in msg:
            return True, "Legacy checkpoint (no hash) - retraining for safety", ckpt_path_legacy
        else:
            return True, f"Config mismatch: {msg}", ckpt_path_legacy
    
    return True, "No checkpoint found", None


# ═══════════════════════════════════════════════════════════════════════════════
# THRESHOLD SWEEP EXPERIMENT
# ═══════════════════════════════════════════════════════════════════════════════

def compute_f2_at_threshold(
    probs: np.ndarray,
    targets: np.ndarray,
    threshold: float
) -> float:
    """
    Compute F2 score at a specific probability threshold.
    
    F2 = 5PR / (4P + R), weights recall 2x more than precision.
    
    Args:
        probs: Predicted probabilities
        targets: Ground truth binary labels
        threshold: Decision threshold
        
    Returns:
        F2 score
    """
    pred = (probs > threshold).astype(np.float32)
    tp = (pred * targets).sum()
    fp = (pred * (1 - targets)).sum()
    fn = ((1 - pred) * targets).sum()
    eps = 1e-7
    precision = tp / (tp + fp + eps)
    recall = tp / (tp + fn + eps)
    beta = 2.0
    f2 = (1 + beta**2) * precision * recall / (beta**2 * precision + recall + eps)
    return f2


def run_threshold_sweep(
    all_results: Dict[str, Dict[int, Dict[str, Any]]],
    thresholds: np.ndarray = None,
    exclude_baseline: bool = True,
    kld_weight: float = 1.0
) -> Dict[str, Any]:
    """
    Run threshold sweep experiment comparing QAT vs QAD.
    
    Args:
        all_results: Results dict with QAT and QAD loss type keys
        thresholds: Array of thresholds to test (default: 0.05 to 0.95)
        exclude_baseline: Exclude recipe 0 from average calculation
        kld_weight: Which KLD weight to use for QAD comparison (default: 1.0)
        
    Returns:
        Dictionary with sweep results and statistics
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 19)
    
    # Discover QAT and QAD keys dynamically
    qat_key, qad_key = _discover_loss_keys(all_results)
    
    # Get QAD results for the specified kld_weight (handle nested structure)
    qad_results = _get_qad_results_for_kld(all_results.get(qad_key, {}), kld_weight) if qad_key else {}
    
    # Find common recipes between QAT and QAD
    qat_recipes = set(all_results.get(qat_key, {}).keys()) if qat_key else set()
    qad_recipes = set(qad_results.keys()) if qad_results else set()
    common = qat_recipes & qad_recipes
    quant_recipes = sorted([r for r in common if r != 0]) if exclude_baseline else sorted(common)
    
    if not quant_recipes:
        return {"error": "No common recipes found"}
    
    sweep_results = {"qat": {}, "qad": {}}
    
    # Compute F2 at each threshold for each recipe
    for recipe_id in quant_recipes:
        qat_result = all_results[qat_key].get(recipe_id, {})
        qad_result = qad_results.get(recipe_id, {})
        
        if "all_probs" not in qat_result or "all_probs" not in qad_result:
            continue
        
        qat_probs = qat_result["all_probs"]
        qad_probs = qad_result["all_probs"]
        targets = qat_result["all_targets"]
        
        sweep_results["qat"][recipe_id] = [
            compute_f2_at_threshold(qat_probs, targets, t) for t in thresholds
        ]
        sweep_results["qad"][recipe_id] = [
            compute_f2_at_threshold(qad_probs, targets, t) for t in thresholds
        ]
    
    # Compute averages
    avg_qat = np.zeros(len(thresholds))
    avg_qad = np.zeros(len(thresholds))
    n = 0
    
    for recipe_id in quant_recipes:
        if recipe_id in sweep_results["qat"]:
            avg_qat += np.array(sweep_results["qat"][recipe_id])
            avg_qad += np.array(sweep_results["qad"][recipe_id])
            n += 1
    
    if n > 0:
        avg_qat /= n
        avg_qad /= n
    
    # Find optimal thresholds
    best_qat_idx = np.argmax(avg_qat)
    best_qad_idx = np.argmax(avg_qad)
    
    # Find where QAD wins
    qad_wins = avg_qad > avg_qat
    
    return {
        "thresholds": thresholds,
        "avg_qat_f2": avg_qat,
        "avg_qad_f2": avg_qad,
        "per_recipe": sweep_results,
        "quant_recipes": quant_recipes,
        "best_qat_threshold": thresholds[best_qat_idx],
        "best_qad_threshold": thresholds[best_qad_idx],
        "best_qat_f2": avg_qat[best_qat_idx],
        "best_qad_f2": avg_qad[best_qad_idx],
        "qad_win_thresholds": thresholds[qad_wins].tolist() if qad_wins.any() else [],
    }


def plot_threshold_sweep(
    sweep_results: Dict[str, Any],
    recipe_names: Dict[int, str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 6),
    dpi: int = 150
):
    """
    Plot threshold sweep results.
    
    Args:
        sweep_results: Output from run_threshold_sweep()
        recipe_names: Optional mapping recipe_id -> display name
        save_path: Optional path to save figure
        figsize: Figure size
        dpi: Resolution
    """
    if recipe_names is None:
        recipe_names = RECIPE_NAMES
    
    thresholds = sweep_results["thresholds"]
    avg_qat = sweep_results["avg_qat_f2"]
    avg_qad = sweep_results["avg_qad_f2"]
    
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot 1: Average F2 vs Threshold
    ax = axes[0]
    ax.plot(thresholds, avg_qat, 'b-o', linewidth=2, markersize=6, label='QAT (BCE+Tversky)')
    ax.plot(thresholds, avg_qad, 'r-s', linewidth=2, markersize=6, label='QAD (Distillation)')
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7, label='Default (0.5)')
    ax.axvline(x=sweep_results["best_qat_threshold"], color='blue', linestyle=':', alpha=0.5)
    ax.axvline(x=sweep_results["best_qad_threshold"], color='red', linestyle=':', alpha=0.5)
    
    # Shade regions
    for i in range(len(thresholds) - 1):
        color = 'blue' if avg_qat[i] > avg_qad[i] else 'red'
        ax.axvspan(thresholds[i], thresholds[i+1], alpha=0.1, color=color)
    
    ax.set_xlabel('Decision Threshold', fontsize=12)
    ax.set_ylabel('F2 Score', fontsize=12)
    ax.set_title('Threshold Sweep: QAT vs QAD (F2)', fontsize=14)
    ax.legend(loc='lower center', fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    
    # Plot 2: F2 difference
    ax = axes[1]
    f2_diff = avg_qat - avg_qad
    colors = ['blue' if d > 0 else 'red' for d in f2_diff]
    ax.bar(thresholds, f2_diff, width=0.04, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(y=0, color='black', linewidth=1)
    ax.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
    ax.set_xlabel('Decision Threshold', fontsize=12)
    ax.set_ylabel('F2 Difference (QAT - QAD)', fontsize=12)
    ax.set_title('QAT Advantage (Blue = QAT wins, Red = QAD wins)', fontsize=14)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xlim([0, 1])
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else '.', exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.show()


def print_threshold_sweep_summary(sweep_results: Dict[str, Any], recipe_names: Dict[int, str] = None):
    """Print detailed threshold sweep summary table."""
    if recipe_names is None:
        recipe_names = RECIPE_NAMES
    
    thresholds = sweep_results["thresholds"]
    avg_qat = sweep_results["avg_qat_f2"]
    avg_qad = sweep_results["avg_qad_f2"]
    
    print("─" * 70)
    print(f"{'Threshold':<12} {'QAT F2':<12} {'QAD F2':<12} {'Δ (QAT-QAD)':<12} {'Winner':<10}")
    print("─" * 70)
    
    for i, t in enumerate(thresholds):
        delta = avg_qat[i] - avg_qad[i]
        winner = "QAT" if delta > 0 else ("QAD" if delta < 0 else "TIE")
        marker = " ← default" if abs(t - 0.5) < 0.01 else ""
        print(f"{t:<12.2f} {avg_qat[i]:<12.4f} {avg_qad[i]:<12.4f} {delta:<+12.4f} {winner:<10}{marker}")
    
    print("─" * 70)
    print(f"\nQAT: Best F2 = {sweep_results['best_qat_f2']:.4f} at θ = {sweep_results['best_qat_threshold']:.2f}")
    print(f"QAD: Best F2 = {sweep_results['best_qad_f2']:.4f} at θ = {sweep_results['best_qad_threshold']:.2f}")
    
    qad_wins = sweep_results["qad_win_thresholds"]
    if qad_wins:
        print(f"\n⚠️  QAD wins at thresholds: {[f'{t:.2f}' for t in qad_wins]}")
    else:
        print(f"\n✅ QAT wins at ALL thresholds tested")


def print_threshold_sweep_per_recipe(sweep_results: Dict[str, Any], recipe_names: Dict[int, str] = None):
    """
    Print per-recipe breakdown of threshold sweep results with conclusion.
    
    Args:
        sweep_results: Results from run_threshold_sweep()
        recipe_names: Optional mapping recipe_id -> display name
    """
    if recipe_names is None:
        recipe_names = RECIPE_NAMES
    
    if "error" in sweep_results:
        print(f"Error: {sweep_results['error']}")
        return
    
    thresholds = sweep_results["thresholds"]
    
    print(f"\n{'═'*90}")
    print("PER-RECIPE BREAKDOWN: Best F2 and Optimal Threshold")
    print(f"{'═'*90}")
    print(f"{'Recipe':<15} {'QAT Best F2':<12} {'QAT θ*':<10} {'QAD Best F2':<12} {'QAD θ*':<10} {'Winner':<10}")
    print(f"{'-'*90}")
    
    for recipe_id in sweep_results["quant_recipes"]:
        if recipe_id not in sweep_results["per_recipe"]["qat"]:
            continue
        qat_f2s = np.array(sweep_results["per_recipe"]["qat"][recipe_id])
        qad_f2s = np.array(sweep_results["per_recipe"]["qad"][recipe_id])
        qat_best, qad_best = qat_f2s.max(), qad_f2s.max()
        qat_opt, qad_opt = thresholds[qat_f2s.argmax()], thresholds[qad_f2s.argmax()]
        winner = "QAT" if qat_best > qad_best else "QAD"
        print(f"{recipe_names.get(recipe_id, str(recipe_id)):<15} {qat_best:<12.4f} {qat_opt:<10.2f} {qad_best:<12.4f} {qad_opt:<10.2f} {winner:<10}")
    
    print(f"{'═'*90}")
    
    # Conclusion
    print(f"\n{'═'*90}")
    print("CONCLUSION")
    print(f"{'═'*90}")
    if not sweep_results["qad_win_thresholds"]:
        print("• QAT outperforms QAD across ALL tested thresholds (0.05 to 0.95)")
        print("• The QAT advantage is NOT an artifact of the default 0.5 threshold")
    else:
        formatted_thresholds = [f"{t:.2f}" for t in sweep_results['qad_win_thresholds']]
        print(f"• QAD wins at {len(formatted_thresholds)} threshold(s): {formatted_thresholds}")
        print("• The optimal loss function depends on the chosen decision threshold")
    print(f"{'═'*90}")


# ═══════════════════════════════════════════════════════════════════════════════
# QAT VS QAD COMPARISON PLOTS
# ═══════════════════════════════════════════════════════════════════════════════

def _compute_kl_divergence(
    student_probs: np.ndarray, 
    baseline_probs: np.ndarray, 
    temperature: float = 2.0,
    epsilon: float = 1e-10
) -> float:
    """
    Compute KL divergence KL(baseline || student) with temperature scaling.
    
    Forward KL matches the training objective where we minimize KL(teacher || student).
    Baseline is on numerator, student on denominator.
    
    Args:
        student_probs: Student/model probability distribution (raw, T=1)
        baseline_probs: Baseline/teacher probability distribution (raw, T=1)
        temperature: Temperature for softening (default 2.0 to match training)
        epsilon: Small value for numerical stability
        
    Returns:
        KL divergence value (average per-pixel)
    """
    # Clip raw probabilities
    s_raw = np.clip(student_probs, epsilon, 1 - epsilon)
    b_raw = np.clip(baseline_probs, epsilon, 1 - epsilon)
    
    # Convert to logits and apply temperature scaling
    s_logits = np.log(s_raw / (1 - s_raw))
    b_logits = np.log(b_raw / (1 - b_raw))
    
    # Temperature-scaled probabilities (softened)
    s = 1 / (1 + np.exp(-s_logits / temperature))
    b = 1 / (1 + np.exp(-b_logits / temperature))
    
    # Clip again for stability
    s = np.clip(s, epsilon, 1 - epsilon)
    b = np.clip(b, epsilon, 1 - epsilon)
    
    # KL(B || S) = b * log(b/s) + (1-b) * log((1-b)/(1-s))
    kl = b * np.log(b / s) + (1 - b) * np.log((1 - b) / (1 - s))
    
    return float(np.mean(kl))


def compute_kl_from_baseline(
    results: Dict[int, Dict[str, Any]],
    baseline_probs: np.ndarray,
    temperature: float = 2.0
) -> Dict[int, float]:
    """
    Compute KL divergence KL(baseline || student) for all recipes in a results dict.
    
    Convenience wrapper around _compute_kl_divergence() that iterates over recipes.
    
    Args:
        results: Results dict {recipe_id: {"all_probs": array, ...}}
        baseline_probs: Baseline/teacher model probabilities (raw, T=1)
        temperature: Temperature for softening (default 2.0 to match training)
        
    Returns:
        Dict mapping recipe_id -> KL divergence
    """
    kl_values = {}
    for rid, data in results.items():
        if "all_probs" not in data:
            continue
        kl_values[rid] = _compute_kl_divergence(data["all_probs"], baseline_probs, temperature)
    return kl_values


def _is_nested_qad_results(qad_results: dict) -> bool:
    """Check if QAD results use nested kld_weight structure."""
    if not qad_results:
        return False
    first_key = next(iter(qad_results.keys()))
    # If first key is a float (0.5, 1.0, 2.0), it's nested by kld_weight
    return isinstance(first_key, float)


def _get_qad_results_for_kld(qad_results: dict, kld_weight: float = None) -> dict:
    """Extract QAD results for a specific kld_weight, handling both old and new formats."""
    if not qad_results:
        return {}
    
    if _is_nested_qad_results(qad_results):
        # New nested format: qad_results[kld_weight][recipe_id]
        if kld_weight is None:
            kld_weight = 1.0  # Default
        return qad_results.get(kld_weight, {})
    else:
        # Old flat format: qad_results[recipe_id]
        return qad_results


def _discover_loss_keys(all_results: dict) -> Tuple[Optional[str], Optional[str]]:
    """
    Discover QAT and QAD loss type keys from all_results dict.
    
    Works with any loss type names (tversky, bce_dice, qad_tversky, qad, etc.)
    by checking whether "qad" is in the key name.
    
    Returns:
        (qat_key, qad_key) - either may be None if not present
    """
    qat_key = next((k for k in all_results if isinstance(k, str) and "qad" not in k), None)
    qad_key = next((k for k in all_results if isinstance(k, str) and "qad" in k), None)
    return qat_key, qad_key


def plot_qat_vs_qad_comparison(
    all_results: Dict[str, Dict[int, Dict[str, Any]]],
    recipe_names: Dict[int, str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (16, 7),
    dpi: int = 150,
    kld_weight: float = None
):
    """
    Plot QAT vs QAD comparison with ROC curves and metrics bar chart.
    
    Args:
        all_results: Results dict with QAT and QAD loss type keys
                    QAD can be nested: all_results[qad_key][kld_weight][recipe_id]
                    or flat: all_results[qad_key][recipe_id]
        recipe_names: Optional mapping recipe_id -> display name
        save_path: Optional path to save figure
        figsize: Figure size
        dpi: Resolution
        kld_weight: Which KLD weight to plot for QAD (default: 1.0 if nested)
    """
    if recipe_names is None:
        recipe_names = RECIPE_NAMES
    
    qat_key, qad_key = _discover_loss_keys(all_results)
    loss_types_to_plot = [k for k in [qat_key, qad_key] if k is not None]
    
    for loss_type in loss_types_to_plot:
        raw_results = all_results.get(loss_type, {})
        if not raw_results:
            continue
        
        # Handle nested QAD structure
        if "qad" in loss_type:
            results = _get_qad_results_for_kld(raw_results, kld_weight)
            if _is_nested_qad_results(raw_results):
                actual_kld = kld_weight if kld_weight else 1.0
                loss_label = f"QAD (Distillation, KLD={actual_kld})"
            else:
                loss_label = "QAD (Distillation)"
        else:
            results = raw_results
            loss_label = f"QAT ({loss_type})"
        
        if not results:
            print(f"⚠️  No results for {loss_type}" + (f" kld_weight={kld_weight}" if "qad" in loss_type else ""))
            continue
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # Filter to only integer keys (recipe IDs)
        recipe_ids = sorted([k for k in results.keys() if isinstance(k, int)])
        if not recipe_ids:
            print(f"⚠️  No recipe results found for {loss_type}")
            plt.close(fig)
            continue
            
        colors = plt.cm.tab10(np.linspace(0, 1, len(recipe_ids)))
        
        # ROC Curve subplot
        ax = axes[0]
        for rid, color in zip(recipe_ids, colors):
            r = results[rid]
            if "fpr" not in r or "tpr" not in r:
                continue
            label = f"{recipe_names.get(rid, str(rid))} (AUC={r.get('auc', 0):.4f})"
            ax.plot(r['fpr'], r['tpr'], label=label, color=color, linewidth=2)
        
        ax.plot([0, 1], [0, 1], 'k--', alpha=0.5)
        ax.set_xlabel('False Positive Rate', fontsize=14)
        ax.set_ylabel('True Positive Rate', fontsize=14)
        ax.set_title(f'ROC Curves - {loss_label}', fontsize=16, fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.grid(True, alpha=0.3)
        
        # Metrics Bar Chart
        ax = axes[1]
        x = np.arange(len(recipe_ids))
        width = 0.25
        
        dice_vals = [results[rid].get('best_dice', 0) for rid in recipe_ids]
        auc_vals = [results[rid].get('auc', 0) for rid in recipe_ids]
        kld_vals = [results[rid].get('kl_from_baseline', 0) for rid in recipe_ids]
        
        ax.bar(x - width, dice_vals, width, label='Best Dice', color='steelblue')
        ax.bar(x, auc_vals, width, label='AUC', color='coral')
        ax.bar(x + width, [k*10 for k in kld_vals], width, label='KLD (×10)', color='seagreen')
        
        ax.set_xticks(x)
        ax.set_xticklabels([recipe_names.get(rid, str(rid)) for rid in recipe_ids], rotation=45, ha='right')
        ax.set_ylabel('Score', fontsize=14)
        ax.set_title(f'Metrics - {loss_label}', fontsize=16, fontweight='bold')
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            suffix = f"_{loss_type}"
            if "qad" in loss_type and _is_nested_qad_results(raw_results):
                suffix += f"_kld{kld_weight if kld_weight else 1.0}"
            path = save_path.replace('.png', f'{suffix}.png')
            plt.savefig(path, dpi=dpi, bbox_inches='tight')
            print(f"✅ Saved: {path}")
        
        plt.show()


def plot_kld_weight_comparison(
    all_results: Dict[str, Dict],
    recipe_id: int = None,
    recipe_names: Dict[int, str] = None,
    ratio_map: Dict[float, float] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5),
    dpi: int = 150
):
    """
    Compare QAD results across different KLD weights.
    
    Args:
        all_results: Results dict with nested QAD: all_results[qad_key][kld_weight][recipe_id]
        recipe_id: Specific recipe to compare (if None, shows average across recipes)
        recipe_names: Optional mapping recipe_id -> display name
        ratio_map: Optional mapping kld_weight -> target_ratio (e.g., {0.0333: 0.05})
                   If provided, x-axis labels show "5%" instead of raw weight
        save_path: Optional path to save figure
        figsize: Figure size
        dpi: Resolution
    """
    if recipe_names is None:
        recipe_names = RECIPE_NAMES
    
    _, qad_key = _discover_loss_keys(all_results)
    qad_results = all_results.get(qad_key, {}) if qad_key else {}
    if not _is_nested_qad_results(qad_results):
        print("⚠️  QAD results not in nested kld_weight format")
        return
    
    kld_weights = sorted(qad_results.keys())
    if not kld_weights:
        print("⚠️  No KLD weight results found")
        return
    
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    if recipe_id is not None:
        # Single recipe comparison
        dice_vals = [qad_results[kw].get(recipe_id, {}).get('best_dice', 0) for kw in kld_weights]
        auc_vals = [qad_results[kw].get(recipe_id, {}).get('auc', 0) for kw in kld_weights]
        kld_vals = [qad_results[kw].get(recipe_id, {}).get('kl_from_baseline', 0) for kw in kld_weights]
        title_suffix = f" - {recipe_names.get(recipe_id, str(recipe_id))}"
    else:
        # Average across all recipes
        dice_vals, auc_vals, kld_vals = [], [], []
        for kw in kld_weights:
            recipes = qad_results[kw]
            dice_vals.append(np.mean([r.get('best_dice', 0) for r in recipes.values()]))
            auc_vals.append(np.mean([r.get('auc', 0) for r in recipes.values()]))
            kld_vals.append(np.mean([r.get('kl_from_baseline', 0) for r in recipes.values()]))
        title_suffix = " (Average)"
    
    x = np.arange(len(kld_weights))
    
    # Build x-axis labels: show percentage if ratio_map provided, else rounded weight
    if ratio_map:
        # Invert: weight -> ratio for lookup
        weight_to_ratio = {w: r for r, w in ratio_map.items()} if ratio_map else {}
        x_labels = []
        for kw in kld_weights:
            ratio = weight_to_ratio.get(kw)
            if ratio is not None:
                x_labels.append(f"{ratio*100:.0f}%")
            else:
                x_labels.append(f"{kw:.4f}")
        x_axis_label = "KLD Loss Budget (%)"
    else:
        x_labels = [f"{kw:.4f}" for kw in kld_weights]
        x_axis_label = "KLD Weight"
    
    # Dice plot
    axes[0].bar(x, dice_vals, color='steelblue', edgecolor='black')
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(x_labels)
    axes[0].set_xlabel(x_axis_label, fontsize=12)
    axes[0].set_ylabel('Dice Score', fontsize=12)
    axes[0].set_title(f'Dice vs KLD Budget{title_suffix}', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='y')
    
    # AUC plot
    axes[1].bar(x, auc_vals, color='coral', edgecolor='black')
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(x_labels)
    axes[1].set_xlabel(x_axis_label, fontsize=12)
    axes[1].set_ylabel('AUC', fontsize=12)
    axes[1].set_title(f'AUC vs KLD Budget{title_suffix}', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # KLD from baseline plot
    axes[2].bar(x, kld_vals, color='seagreen', edgecolor='black')
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(x_labels)
    axes[2].set_xlabel(x_axis_label, fontsize=12)
    axes[2].set_ylabel('KLD from Baseline', fontsize=12)
    axes[2].set_title(f'KLD vs KLD Budget{title_suffix}', fontsize=14, fontweight='bold')
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.show()


def print_qat_vs_qad_summary(
    all_results: Dict[str, Dict[int, Dict[str, Any]]],
    recipe_names: Dict[int, str] = None,
    kld_weight: float = None
):
    """Print QAT vs QAD comparison summary table.
    
    Args:
        all_results: Results dict with QAT and QAD loss type keys
        recipe_names: Optional mapping recipe_id -> display name
        kld_weight: Which KLD weight to use for QAD (default: 1.0 if nested)
    """
    if recipe_names is None:
        recipe_names = RECIPE_NAMES
    
    qat_key, qad_key = _discover_loss_keys(all_results)
    qat_results = all_results.get(qat_key, {}) if qat_key else {}
    qad_results = _get_qad_results_for_kld(all_results.get(qad_key, {}), kld_weight) if qad_key else {}
    
    # Get common recipe IDs (integers only)
    qat_recipes = set(k for k in qat_results.keys() if isinstance(k, int))
    qad_recipes = set(k for k in qad_results.keys() if isinstance(k, int))
    common_recipes = sorted(qat_recipes & qad_recipes)
    
    if not common_recipes:
        print("No common recipes between QAT and QAD runs")
        # Show what we have
        if qat_recipes:
            print(f"  QAT recipes: {sorted(qat_recipes)}")
        if qad_recipes:
            print(f"  QAD recipes: {sorted(qad_recipes)}")
        elif qad_key and all_results.get(qad_key, {}):
            raw_qad = all_results.get(qad_key, {})
            if _is_nested_qad_results(raw_qad):
                print(f"  QAD kld_weights available: {sorted(raw_qad.keys())}")
        return
    
    # Determine KLD weight label
    raw_qad = all_results.get(qad_key, {}) if qad_key else {}
    kld_label = f" (kld={kld_weight if kld_weight else 1.0})" if _is_nested_qad_results(raw_qad) else ""
    
    print(f"\n{'Recipe':<15} {'QAT Dice':<12} {'QAD Dice':<12} {'QAT KLD':<12} {'QAD KLD':<12} {'QAT AUC':<10} {'QAD AUC':<10}")
    print(f"{'':>27}{kld_label}")
    print(f"{'-'*90}")
    
    for rid in common_recipes:
        qat = qat_results.get(rid, {})
        qad = qad_results.get(rid, {})
        
        print(f"{recipe_names.get(rid, str(rid)):<15} "
              f"{qat.get('best_dice', 0):<12.4f} {qad.get('best_dice', 0):<12.4f} "
              f"{qat.get('kl_from_baseline', 0):<12.6f} {qad.get('kl_from_baseline', 0):<12.6f} "
              f"{qat.get('auc', 0):<10.4f} {qad.get('auc', 0):<10.4f}")
    
    print(f"{'-'*90}")
    
    # Averages (excluding baseline)
    quant = [r for r in common_recipes if r != 0]
    if quant:
        avg_qat_dice = np.mean([qat_results[r].get("best_dice", 0) for r in quant])
        avg_qad_dice = np.mean([qad_results[r].get("best_dice", 0) for r in quant])
        avg_qat_kld = np.mean([qat_results[r].get("kl_from_baseline", 0) for r in quant])
        avg_qad_kld = np.mean([qad_results[r].get("kl_from_baseline", 0) for r in quant])
        avg_qat_auc = np.mean([qat_results[r].get("auc", 0) for r in quant])
        avg_qad_auc = np.mean([qad_results[r].get("auc", 0) for r in quant])
        
        print(f"{'AVG (Quant)':<15} {avg_qat_dice:<12.4f} {avg_qad_dice:<12.4f} "
              f"{avg_qat_kld:<12.6f} {avg_qad_kld:<12.6f} {avg_qat_auc:<10.4f} {avg_qad_auc:<10.4f}")


def print_metrics_table(
    results: Dict[int, Dict[str, Any]],
    loss_type: str,
    recipe_names: Dict[int, str] = None,
    kld_weight: float = None
):
    """
    Print detailed metrics table for a single loss type.
    
    Args:
        results: Results dict {recipe_id: {...metrics...}} or nested {kld_weight: {recipe_id: {...}}}
        loss_type: Loss type name for header (e.g. "tversky", "qad_tversky")
        recipe_names: Optional mapping recipe_id -> display name
        kld_weight: Which KLD weight to display for nested QAD results (default: 1.0)
    """
    if recipe_names is None:
        recipe_names = RECIPE_NAMES
    
    if not results:
        print(f"No results for {loss_type}")
        return
    
    # Handle nested QAD structure
    if "qad" in loss_type and _is_nested_qad_results(results):
        # Print table for each KLD weight
        kld_weights = sorted(results.keys())
        for kw in kld_weights:
            kw_results = results[kw]
            _print_metrics_table_flat(kw_results, f"QAD (kld={kw})", recipe_names)
    else:
        # Flat structure
        _print_metrics_table_flat(results, loss_type.upper(), recipe_names)


def _print_metrics_table_flat(
    results: Dict[int, Dict[str, Any]],
    title: str,
    recipe_names: Dict[int, str]
):
    """Print metrics table for flat results dict with first-tier and second-tier metrics."""
    if not results:
        print(f"No results for {title}")
        return
    
    # Filter to only integer keys (recipe IDs)
    recipe_ids = sorted([k for k in results.keys() if isinstance(k, int)])
    if not recipe_ids:
        print(f"No recipe results found for {title}")
        return
    
    print(f"\n{'='*115}")
    print(f"📊 {title} METRICS")
    print(f"{'='*115}")
    
    # First Tier (Primary)
    print(f"  {'Recipe':<15} {'Recall':<10} {'F2':<10} {'Dice':<10} {'AUPRC':<10} {'KL↓':<10} {'AUC':<8} {'ECE':<8}")
    print(f"  {'-'*110}")
    
    for rid in recipe_ids:
        r = results[rid]
        # Dice: prefer inference 'dice' over 'best_dice' (which may be from training loop)
        dice = r.get('dice', r.get('best_dice', 0))
        recall = r.get('recall', 0)
        f2 = r.get('f2_score', 0)
        auprc = r.get('auprc', 0)
        kl = r.get('kl_from_baseline', 0)
        kl_str = f"{kl:.6f}" if kl > 0 else "baseline"
        auc_val = r.get('auc', 0)
        ece = r.get('ece', 0)
        
        print(f"  {recipe_names.get(rid, str(rid)):<15} "
              f"{recall:<10.4f} {f2:<10.4f} {dice:<10.4f} {auprc:<10.4f} "
              f"{kl_str:<10} {auc_val:<8.4f} {ece:<8.4f}")
    
    print(f"{'='*115}")


# ═══════════════════════════════════════════════════════════════════════════════
# TRAINING CURVES VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_curves_6panel(
    results: Dict[int, Dict[str, Any]],
    loss_type: str,
    recipe_names: Dict[int, str] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (18, 12),
    dpi: int = 150,
    kld_weight: float = None
):
    """
    Plot 2x3 grid of training curves for a single loss type.
    
    Panels: Val Dice, BCE Loss, Tversky Loss, Total Loss, Grad Norm, KLD/Train Dice
    
    Args:
        results: Results dict for one loss type {recipe_id: {"history": {...}}}
        loss_type: Loss type name (e.g. "tversky", "qad_tversky")
        recipe_names: Optional mapping recipe_id -> display name
        save_path: Optional path to save figure
        figsize: Figure size
        dpi: Resolution
        kld_weight: Optional KLD weight for QAD title
    """
    if recipe_names is None:
        recipe_names = RECIPE_NAMES
    
    if "qad" not in loss_type:
        loss_label = f"QAT ({loss_type})"
    elif kld_weight is not None:
        loss_label = f"QAD ({loss_type}, kld={kld_weight})"
    else:
        loss_label = f"QAD ({loss_type})"
    
    fig, axes = plt.subplots(2, 3, figsize=figsize)
    recipe_ids = sorted(results.keys())
    colors = plt.cm.tab10(np.linspace(0, 1, len(recipe_ids)))
    
    # Debug: check if history exists
    has_any_history = False
    for rid in recipe_ids:
        h = results[rid].get("history", {})
        if h and any(h.get(m, []) for m in ["val_dice", "bce_loss", "total_loss"]):
            has_any_history = True
            break
    
    if not has_any_history:
        print(f"⚠️  WARNING: No training history found for {loss_label}!")
        print(f"   recipe_ids: {recipe_ids[:5]}{'...' if len(recipe_ids) > 5 else ''}")
        if recipe_ids:
            sample = results[recipe_ids[0]]
            print(f"   sample keys: {list(sample.keys())[:10]}")
            sample_h = sample.get("history", {})
            print(f"   history keys: {list(sample_h.keys()) if sample_h else 'EMPTY'}")
    
    metrics = [
        ("val_dice", "Validation Dice", 'lower right'),
        ("bce_loss", "BCE Loss", 'upper right'),
        ("dice_loss", "Dice Loss", 'upper right'),
        ("total_loss", "Total Loss", 'upper right'),
        ("grad_norm", "Gradient Norm", 'upper right'),
    ]
    
    for idx, (metric, title, legend_loc) in enumerate(metrics):
        ax = axes[idx // 3, idx % 3]
        for rid, color in zip(recipe_ids, colors):
            h = results[rid].get("history", {})
            if metric in h and h[metric]:
                ax.plot(h[metric], label=recipe_names.get(rid, str(rid)), color=color, linewidth=2)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title, fontweight='bold')
        ax.legend(loc=legend_loc, fontsize=8)
        ax.grid(True, alpha=0.3)
    
    # Last panel: KLD for QAD or Train Dice for QAT
    ax = axes[1, 2]
    if "qad" in loss_type:
        for rid, color in zip(recipe_ids, colors):
            h = results[rid].get("history", {})
            if "kld_loss" in h and h["kld_loss"] and any(v > 0 for v in h["kld_loss"]):
                ax.plot(h["kld_loss"], label=recipe_names.get(rid, str(rid)), color=color, linewidth=2)
        ax.set_ylabel('KLD Loss')
        ax.set_title('KLD Distillation Loss', fontweight='bold')
        ax.legend(loc='upper right', fontsize=8)
    else:
        for rid, color in zip(recipe_ids, colors):
            h = results[rid].get("history", {})
            if "train_dice" in h and h["train_dice"]:
                ax.plot(h["train_dice"], label=recipe_names.get(rid, str(rid)), color=color, linewidth=2)
        ax.set_ylabel('Train Dice')
        ax.set_title('Training Dice Score', fontweight='bold')
        ax.legend(loc='lower right', fontsize=8)
    ax.set_xlabel('Epoch')
    ax.grid(True, alpha=0.3)
    
    plt.suptitle(f'Training Curves - {loss_label}', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# ABLATION STUDY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════════


@dataclass
class AblationConfig:
    """Configuration for ablation study."""
    model_size: str = "matched_300k"
    num_epochs: int = 100
    batch_size: int = 26
    learning_rate: float = 1e-4
    weight_decay: float = 1e-4
    seed: int = 2026
    
    # Factors to sweep
    loss_types: List[str] = field(default_factory=lambda: ["tversky", "qad_tversky"])
    recipes: List[int] = field(default_factory=lambda: [0, 1, 2, 3, 4, 5, 6, 7])
    kld_weights: List[float] = field(default_factory=lambda: [0.5, 1.0, 2.0])
    
    # QAD fixed params
    qad_task_weight: float = 1.0
    qad_temperature: float = 2.0
    
    # Control
    force_retrain: bool = False
    
    def get_all_configs(self) -> List[Dict]:
        """Generate all experiment configurations."""
        configs = []
        
        # QAT loss types (non-QAD): just recipes (no KLD weight)
        qat_types = [lt for lt in self.loss_types if "qad" not in lt]
        for qat_lt in qat_types:
            for recipe in self.recipes:
                configs.append({
                    "loss_type": qat_lt,
                    "recipe_id": recipe,
                    "kld_weight": None,
                    "config_key": f"qat_r{recipe}"
                })
        
        # QAD loss types: recipes × kld_weights (skip recipe 0 - it's the teacher)
        qad_types = [lt for lt in self.loss_types if "qad" in lt]
        for qad_lt in qad_types:
            for recipe, kld_w in product(self.recipes, self.kld_weights):
                if recipe == 0:  # Skip baseline for QAD
                    continue
                configs.append({
                    "loss_type": qad_lt,
                    "recipe_id": recipe,
                    "kld_weight": kld_w,
                    "config_key": f"qad_r{recipe}_kld{kld_w}"
                })
        
        return configs
    
    def total_runs(self) -> int:
        """Count total number of runs."""
        return len(self.get_all_configs())


def run_ablation_study(
    ablation_config: AblationConfig,
    ckpt_dir: Path,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
    create_model_fn: Callable,
    baseline_weights: dict,
    verbose: bool = True,
    use_hierarchical: bool = True
) -> Dict[str, Dict[str, Any]]:
    """
    Run full ablation study across loss types, recipes, and KLD weights.
    
    Args:
        ablation_config: AblationConfig with experiment parameters
        ckpt_dir: Base directory to save checkpoints
        train_dataloader: Training data
        val_dataloader: Validation data
        device: Compute device
        create_model_fn: Model factory function
        baseline_weights: Initial weights for fair comparison
        verbose: Print progress
        use_hierarchical: Use new hierarchical directory structure (v2)
        
    Returns:
        Dict[config_key] = {metrics...}
        
    Directory structure when use_hierarchical=True:
        ckpt_dir/
        └── {model_size}/
            ├── qat/
            │   └── recipe_{id}_{epochs}ep[_{hash}].pt
            ├── qad/
            │   └── recipe_{id}_{epochs}ep_kld{weight}[_{hash}].pt
            ├── telemetry/
            ├── logs/
            └── plots/
    """
    results = {}
    all_configs = ablation_config.get_all_configs()
    total = len(all_configs)
    model_size = ablation_config.model_size
    num_epochs = ablation_config.num_epochs
    
    # Setup directories
    if use_hierarchical:
        setup_experiment_dirs(ckpt_dir, [model_size])
        size_dir = ckpt_dir / model_size
    else:
        size_dir = ckpt_dir
        size_dir.mkdir(exist_ok=True)
    
    # Always show header (concise)
    print(f"🔬 ABLATION STUDY: {total} configs | Model: {model_size} | Output: {size_dir}")
    if verbose:
        print(f"   Loss types: {ablation_config.loss_types}")
        print(f"   Recipes: {ablation_config.recipes}")
        print(f"   KLD weights: {ablation_config.kld_weights}")
    
    # Helper to get paths based on mode
    def get_ckpt_path(recipe_id, loss_type, kld_weight, config_hash):
        if use_hierarchical:
            return get_checkpoint_path_v2(
                ckpt_dir, model_size, recipe_id, num_epochs, loss_type, kld_weight, config_hash
            )
        else:
            return get_checkpoint_path(
                ckpt_dir, recipe_id, model_size, num_epochs, loss_type, config_hash, kld_weight
            )
    
    # Teacher model for QAD - will be loaded lazily when needed
    # (after QAT baseline is trained in the same run)
    teacher_model = None
    baseline_probs = None  # For KL computation
    
    # Try to pre-load teacher if it already exists
    if "qad" in ablation_config.loss_types:
        candidate_paths = []
        if use_hierarchical:
            qat_dir = ckpt_dir / model_size / "qat"
            if qat_dir.exists():
                for f in qat_dir.glob(f"recipe_0_{num_epochs}ep*.pt"):
                    candidate_paths.append(f)
        # Legacy paths (check both tversky and old bce_dice naming)
        for legacy_lt in ["tversky", "bce_dice"]:
            for f in ckpt_dir.glob(f"recipe_0_{model_size}_{num_epochs}ep_{legacy_lt}*.pt"):
                if f not in candidate_paths:
                    candidate_paths.append(f)
        
        for tpath in candidate_paths:
            if tpath.exists():
                ckpt = torch.load(tpath, map_location=device, weights_only=False)
                teacher_model = create_model_fn(model_size=model_size, recipe_id=0).to(device)
                teacher_model.load_state_dict(ckpt['state_dict'])
                teacher_model.eval()
                for p in teacher_model.parameters():
                    p.requires_grad = False
                print(f"✅ Teacher pre-loaded: {tpath.name} (Dice={ckpt.get('best_dice', 0):.4f})")
                
                # Also load baseline_probs for KL computation
                if 'all_probs' in ckpt:
                    baseline_probs = ckpt['all_probs']
                else:
                    # Run inference to get baseline probs
                    print(f"   📊 Computing baseline probs for KL divergence...")
                    inf_results = run_validation_inference(teacher_model, val_dataloader, device)
                    baseline_probs = inf_results.get('all_probs')
                break
        
        if teacher_model is None:
            print(f"ℹ️ Teacher not found yet - will load after QAT baseline is trained")
    
    # Run each configuration
    for i, cfg in enumerate(all_configs):
        config_key = cfg["config_key"]
        loss_type = cfg["loss_type"]
        recipe_id = cfg["recipe_id"]
        kld_weight = cfg["kld_weight"]
        
        # Progress indicator (always show, but concise when not verbose)
        if verbose:
            print(f"\n[{i+1}/{total}] {config_key}")
            print(f"{'─'*60}")
        else:
            # Inline progress
            print(f"\r[{i+1}/{total}] {config_key:<30}", end="", flush=True)
        
        # For QAD: try to load teacher if not already loaded (lazy loading)
        # This handles the case where recipe 0 QAT was just trained in this same run
        if "qad" in loss_type and teacher_model is None:
            # Build list of candidate paths for teacher checkpoint
            candidate_paths = []
            
            # 1. Hierarchical path with hash (most likely for fresh runs)
            if use_hierarchical:
                qat_dir = ckpt_dir / model_size / "qat"
                if qat_dir.exists():
                    # Find any recipe_0 checkpoint in qat directory
                    for f in qat_dir.glob(f"recipe_0_{num_epochs}ep*.pt"):
                        candidate_paths.append(f)
            
            # 2. Legacy flat paths (check both tversky and old bce_dice naming)
            for legacy_lt in ["tversky", "bce_dice"]:
                candidate_paths.append(ckpt_dir / f"recipe_0_{model_size}_{num_epochs}ep_{legacy_lt}.pt")
                for f in ckpt_dir.glob(f"recipe_0_{model_size}_{num_epochs}ep_{legacy_lt}*.pt"):
                    if f not in candidate_paths:
                        candidate_paths.append(f)
            
            # Try to load from any candidate
            for tpath in candidate_paths:
                if tpath.exists():
                    ckpt = torch.load(tpath, map_location=device, weights_only=False)
                    teacher_model = create_model_fn(model_size=model_size, recipe_id=0).to(device)
                    teacher_model.load_state_dict(ckpt['state_dict'])
                    teacher_model.eval()
                    for p in teacher_model.parameters():
                        p.requires_grad = False
                    print(f"\n   ✅ Teacher loaded: {tpath.name} (Dice={ckpt.get('best_dice', 0):.4f})")
                    # Also load baseline probs if not loaded
                    if baseline_probs is None:
                        if 'all_probs' not in ckpt:
                            model = create_model_fn(model_size=model_size, recipe_id=0).to(device)
                            model.load_state_dict(ckpt['state_dict'])
                            model.eval()
                            inf_results = run_validation_inference(model, val_dataloader, device)
                            baseline_probs = inf_results.get('all_probs')
                            del model
                            torch.cuda.empty_cache()
                        else:
                            baseline_probs = ckpt.get('all_probs')
                    break
            
            # If still no teacher, skip this config
            if teacher_model is None:
                print(f"\n   ⏭️ Skipped {config_key} (no teacher found)")
                continue
        
        # Build config hash
        training_config = get_training_config(
            model_size,
            num_epochs,
            ablation_config.batch_size,
            ablation_config.learning_rate,
            ablation_config.weight_decay,
            loss_type,
            ablation_config.seed,
            ablation_config.qad_task_weight,
            kld_weight or 1.0,
            ablation_config.qad_temperature
        )
        config_hash = compute_config_hash(training_config)
        
        # Check if already exists (check both hierarchical and legacy paths)
        ckpt_path = get_ckpt_path(recipe_id, loss_type, kld_weight, config_hash)
        ckpt_path_legacy = get_checkpoint_path(
            ckpt_dir, recipe_id, model_size, num_epochs, loss_type, config_hash, kld_weight
        )
        
        loaded_from = None
        for candidate in [ckpt_path, ckpt_path_legacy]:
            if candidate.exists() and not ablation_config.force_retrain:
                loaded_from = candidate
                break
        
        if loaded_from:
            if verbose:
                print(f"   ✅ Loading: {loaded_from.name}")
            ckpt = torch.load(loaded_from, map_location=device, weights_only=False)
            model = create_model_fn(model_size=model_size, recipe_id=recipe_id).to(device)
            model.load_state_dict(ckpt['state_dict'])
            model.eval()
            
            inference_results = run_validation_inference(model, val_dataloader, device)
            results[config_key] = {
                "loss_type": loss_type,
                "recipe_id": recipe_id,
                "kld_weight": kld_weight,
                "best_dice": ckpt.get("best_dice", 0),
                "history": ckpt.get("history", {}),
                **inference_results
            }
            del model
        else:
            if verbose:
                print(f"   📝 Training → {ckpt_path.parent.name}/{ckpt_path.name}")
            
            qad_params = {
                "task_weight": ablation_config.qad_task_weight,
                "distill_weight": kld_weight or 1.0,
                "temperature": ablation_config.qad_temperature
            }
            
            # Determine save directory
            train_ckpt_dir = ckpt_path.parent if use_hierarchical else ckpt_dir
            
            train_results = train_single_recipe(
                recipe_id=recipe_id,
                loss_type=loss_type,
                model_size=model_size,
                num_epochs=num_epochs,
                train_dataloader=train_dataloader,
                val_dataloader=val_dataloader,
                device=device,
                create_model_fn=create_model_fn,
                baseline_weights=baseline_weights,
                ckpt_dir=train_ckpt_dir,
                config_hash=config_hash,
                config_dict=training_config,
                teacher_model=teacher_model if "qad" in loss_type else None,
                qad_params=qad_params,
                target_dice=0.5,
                learning_rate=ablation_config.learning_rate,
                weight_decay=ablation_config.weight_decay,
                verbose=verbose
            )
            
            results[config_key] = {
                "loss_type": loss_type,
                "recipe_id": recipe_id,
                "kld_weight": kld_weight,
                **train_results
            }
        
        # Compute KLD from baseline
        if baseline_probs is not None and "all_probs" in results[config_key]:
            kl_val = compute_kl_from_baseline(
                {recipe_id: results[config_key]}, baseline_probs, 2.0
            ).get(recipe_id, 0)
            results[config_key]["kl_from_baseline"] = kl_val
        
        if verbose:
            r = results[config_key]
            print(f"   Dice={r.get('best_dice', 0):.4f} AUC={r.get('auc', 0):.4f} KLD={r.get('kl_from_baseline', 0):.6f}")
        
        torch.cuda.empty_cache()
    
    # Cleanup
    if teacher_model is not None:
        del teacher_model
        torch.cuda.empty_cache()
    
    # Save summary CSV
    if use_hierarchical and results:
        save_ablation_summary(results, ckpt_dir, model_size)
    
    # Always show completion summary
    if not verbose:
        print()  # Clear inline progress
    print(f"\n✅ Ablation complete: {len(results)} configs trained")
    if use_hierarchical:
        print(f"📁 Saved to: {size_dir}")
    
    if verbose:
        print_experiment_structure(ckpt_dir, model_size)
    
    return results


def rank_configs(
    results: Dict[str, Dict[str, Any]],
    metrics: List[str] = ["best_dice", "auc"],
    ascending: Dict[str, bool] = None
) -> pd.DataFrame:
    """
    Rank configurations by multiple metrics.
    
    Args:
        results: Ablation study results
        metrics: Metrics to rank by (first is primary)
        ascending: Dict specifying sort order per metric (default: descending for all)
        
    Returns:
        DataFrame with rankings
    """
    if ascending is None:
        ascending = {m: False for m in metrics}  # Higher is better by default
    ascending["kl_from_baseline"] = True  # Lower KLD is better
    ascending["ece"] = True  # Lower ECE is better
    
    rows = []
    for key, r in results.items():
        rows.append({
            "config": key,
            "loss_type": r.get("loss_type", ""),
            "recipe_id": r.get("recipe_id", 0),
            "kld_weight": r.get("kld_weight"),
            **{m: r.get(m, 0) for m in metrics + ["kl_from_baseline", "ece"]}
        })
    
    df = pd.DataFrame(rows)
    
    # Sort by primary metric
    primary = metrics[0]
    df = df.sort_values(primary, ascending=ascending.get(primary, False))
    df["rank"] = range(1, len(df) + 1)
    
    return df


def run_validation_across_sizes(
    top_configs: List[Dict],
    model_sizes: List[str],
    ckpt_dir: Path,
    train_dataloader: DataLoader,
    val_dataloader: DataLoader,
    device: torch.device,
    create_model_fn: Callable,
    num_epochs: int = 100,
    seed: int = 2026,
    force_retrain: bool = False,
    verbose: bool = True
) -> Dict[str, Dict[str, Dict[str, Any]]]:
    """
    Validate top configurations across different model sizes.
    
    Args:
        top_configs: List of config dicts with loss_type, recipe_id, kld_weight
        model_sizes: Model sizes to test
        ckpt_dir: Checkpoint directory
        train_dataloader: Training data
        val_dataloader: Validation data
        device: Compute device
        create_model_fn: Model factory
        num_epochs: Training epochs
        seed: Random seed
        force_retrain: Ignore existing checkpoints
        verbose: Print progress
        
    Returns:
        results[model_size][config_key] = {...}
    """
    results = {size: {} for size in model_sizes}
    
    if verbose:
        print(f"{'═'*80}")
        print(f"🔬 CROSS-SIZE VALIDATION")
        print(f"{'═'*80}")
        print(f"Model sizes: {model_sizes}")
        print(f"Configs to validate: {len(top_configs)}")
        print(f"{'═'*80}\n")
    
    for model_size in model_sizes:
        if verbose:
            print(f"\n{'─'*60}")
            print(f"📏 Model Size: {model_size}")
            print(f"{'─'*60}")
        
        # Create baseline weights for this size
        baseline = create_model_fn(model_size=model_size, recipe_id=0)
        baseline_weights = copy.deepcopy(baseline.state_dict())
        del baseline
        
        # Load teacher if needed
        teacher_model = None
        has_qad = any("qad" in c.get("loss_type", "") for c in top_configs)
        if has_qad:
            # Look for teacher checkpoint (try tversky first, then legacy bce_dice)
            teacher_ckpt = None
            for t_lt in ["tversky", "bce_dice"]:
                matches = list(ckpt_dir.glob(f"recipe_0_{model_size}_{num_epochs}ep_{t_lt}*.pt"))
                if matches:
                    teacher_ckpt = matches[0]
                    break
            if teacher_ckpt is None:
                teacher_ckpt = ckpt_dir / f"recipe_0_{model_size}_{num_epochs}ep_tversky.pt"  # Will not exist, handled below
            if teacher_ckpt.exists():
                ckpt = torch.load(teacher_ckpt, map_location=device, weights_only=False)
                teacher_model = create_model_fn(model_size=model_size, recipe_id=0).to(device)
                teacher_model.load_state_dict(ckpt['state_dict'])
                teacher_model.eval()
                for p in teacher_model.parameters():
                    p.requires_grad = False
        
        for cfg in top_configs:
            loss_type = cfg["loss_type"]
            recipe_id = cfg["recipe_id"]
            kld_weight = cfg.get("kld_weight")
            config_key = cfg.get("config_key", f"{loss_type}_r{recipe_id}")
            
            if verbose:
                print(f"\n   {config_key}...")
            
            if loss_type == "qad" and teacher_model is None:
                if verbose:
                    print(f"   ⏭️ Skipped (no teacher for {model_size})")
                continue
            
            training_config = get_training_config(
                model_size, num_epochs, 26, 1e-4, 1e-4,
                loss_type, seed, 1.0, kld_weight or 1.0, 2.0
            )
            config_hash = compute_config_hash(training_config)
            
            ckpt_path = get_checkpoint_path(
                ckpt_dir, recipe_id, model_size, num_epochs, loss_type, config_hash, kld_weight
            )
            
            if ckpt_path.exists() and not force_retrain:
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model = create_model_fn(model_size=model_size, recipe_id=recipe_id).to(device)
                model.load_state_dict(ckpt['state_dict'])
                model.eval()
                
                inference_results = run_validation_inference(model, val_dataloader, device)
                results[model_size][config_key] = {
                    "best_dice": ckpt.get("best_dice", 0),
                    **inference_results
                }
                del model
            else:
                qad_params = {"task_weight": 1.0, "distill_weight": kld_weight or 1.0, "temperature": 2.0}
                
                train_results = train_single_recipe(
                    recipe_id=recipe_id,
                    loss_type=loss_type,
                    model_size=model_size,
                    num_epochs=num_epochs,
                    train_dataloader=train_dataloader,
                    val_dataloader=val_dataloader,
                    device=device,
                    create_model_fn=create_model_fn,
                    baseline_weights=baseline_weights,
                    ckpt_dir=ckpt_dir,
                    config_hash=config_hash,
                    config_dict=training_config,
                    teacher_model=teacher_model if "qad" in loss_type else None,
                    qad_params=qad_params,
                    verbose=False
                )
                results[model_size][config_key] = train_results
            
            if verbose:
                r = results[model_size][config_key]
                print(f"   Dice={r.get('best_dice', 0):.4f} AUC={r.get('auc', 0):.4f}")
            
            torch.cuda.empty_cache()
        
        if teacher_model is not None:
            del teacher_model
            torch.cuda.empty_cache()
    
    return results


def plot_ablation_heatmap(
    results: Dict[str, Dict[str, Any]],
    metric: str = "best_dice",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8),
    dpi: int = 150
):
    """
    Plot ablation results as heatmap (recipes × kld_weights).
    
    Args:
        results: Ablation study results
        metric: Metric to visualize
        save_path: Path to save figure
        figsize: Figure size
        dpi: Resolution
    """
    # Separate QAT and QAD results
    qat_data = {}
    qad_data = {}  # qad_data[recipe][kld_weight] = value
    
    for key, r in results.items():
        recipe = r.get("recipe_id", 0)
        recipe_name = RECIPE_NAMES.get(recipe, str(recipe))
        
        if "qad" not in r.get("loss_type", ""):
            qat_data[recipe_name] = r.get(metric, 0)
        else:
            kld_w = r.get("kld_weight", 1.0)
            if recipe_name not in qad_data:
                qad_data[recipe_name] = {}
            qad_data[recipe_name][kld_w] = r.get(metric, 0)
    
    # Build QAD matrix
    recipes = sorted(qad_data.keys())
    kld_weights = sorted(set(kw for r in qad_data.values() for kw in r.keys()))
    
    qad_matrix = np.zeros((len(recipes), len(kld_weights)))
    for i, r in enumerate(recipes):
        for j, kw in enumerate(kld_weights):
            qad_matrix[i, j] = qad_data.get(r, {}).get(kw, 0)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # QAT bar chart
    ax1 = axes[0]
    qat_recipes = list(qat_data.keys())
    qat_values = [qat_data[r] for r in qat_recipes]
    bars = ax1.barh(qat_recipes, qat_values, color='steelblue', alpha=0.8)
    ax1.set_xlabel(metric.replace('_', ' ').title())
    ax1.set_title('QAT (BCE+Dice)', fontweight='bold')
    ax1.set_xlim(0, max(qat_values) * 1.1 if qat_values else 1)
    for bar, val in zip(bars, qat_values):
        ax1.text(val + 0.002, bar.get_y() + bar.get_height()/2, f'{val:.4f}', va='center', fontsize=9)
    
    # QAD heatmap
    ax2 = axes[1]
    im = ax2.imshow(qad_matrix, cmap='RdYlGn', aspect='auto')
    ax2.set_xticks(range(len(kld_weights)))
    ax2.set_xticklabels([f'KLD={kw}' for kw in kld_weights])
    ax2.set_yticks(range(len(recipes)))
    ax2.set_yticklabels(recipes)
    ax2.set_title('QAD (Distillation)', fontweight='bold')
    
    # Add values to heatmap
    for i in range(len(recipes)):
        for j in range(len(kld_weights)):
            ax2.text(j, i, f'{qad_matrix[i, j]:.3f}', ha='center', va='center', fontsize=9)
    
    plt.colorbar(im, ax=ax2, label=metric.replace('_', ' ').title())
    
    plt.suptitle(f'Ablation Study: {metric.replace("_", " ").title()}', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
        print(f"✅ Saved: {save_path}")
    
    plt.show()


def print_ablation_summary(
    results: Dict[str, Dict[str, Any]],
    metrics: List[str] = ["best_dice", "auc", "kl_from_baseline", "ece"]
):
    """Print formatted ablation study summary table."""
    print(f"\n{'═'*100}")
    print("ABLATION STUDY SUMMARY")
    print(f"{'═'*100}")
    
    # Header
    header = f"{'Config':<25} {'Loss':<10} {'Recipe':<12} {'KLD':<6}"
    for m in metrics:
        header += f" {m:<12}"
    print(header)
    print(f"{'─'*100}")
    
    # Sort by primary metric
    sorted_keys = sorted(results.keys(), key=lambda k: results[k].get(metrics[0], 0), reverse=True)
    
    for key in sorted_keys:
        r = results[key]
        recipe_name = RECIPE_NAMES.get(r.get("recipe_id", 0), str(r.get("recipe_id", 0)))[:11]
        kld = f"{r.get('kld_weight', '-')}" if r.get('kld_weight') else "-"
        
        row = f"{key:<25} {r.get('loss_type', ''):<10} {recipe_name:<12} {kld:<6}"
        for m in metrics:
            val = r.get(m, 0)
            row += f" {val:<12.4f}"
        print(row)
    
    print(f"{'═'*100}")
    
    # Top 3 summary
    print(f"\n🏆 TOP 3 by {metrics[0]}:")
    for i, key in enumerate(sorted_keys[:3]):
        r = results[key]
        print(f"   {i+1}. {key}: {metrics[0]}={r.get(metrics[0], 0):.4f}")


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("ViT QAT Utils Module")
    print("=" * 60)
    
    print("\n📁 DATA & SETUP:")
    print("  - set_seed(), build_file_dataframe(), align_images_and_masks()")
    
    print("\n🔐 CONFIG FINGERPRINTING:")
    print("  - compute_config_hash(), get_training_config()")
    print("  - get_checkpoint_path(), get_telemetry_cache_path()")
    print("  - validate_checkpoint(), should_train_recipe()")
    print("  - load_manifest(), save_manifest(), update_manifest()")
    
    print("\n💾 TELEMETRY CACHING:")
    print("  - load_telemetry_cache(), save_telemetry_cache()")
    print("  - run_validation_inference()")
    
    print("\n🎯 TRAINING:")
    print("  - compute_qat_qad_loss() - QAT/QAD loss computation")
    print("  - compute_ece() - Expected Calibration Error")
    print("  - compute_roc(), evaluate_on_test_set()")
    
    print("\n📊 THRESHOLD SWEEP:")
    print("  - compute_dice_at_threshold()")
    print("  - run_threshold_sweep()")
    print("  - plot_threshold_sweep(), print_threshold_sweep_summary()")
    
    print("\n📈 VISUALIZATION:")
    print("  - plot_qat_vs_qad_comparison(), print_qat_vs_qad_summary()")
    print("  - plot_training_curves_6panel()")
    print("  - plot_training_summary_2x3(), plot_multi_scale_roc_comparison()")
    print("  - compute_kl_from_baseline()")
    
    print("\n🔬 ABLATION STUDY:")
    print("  - run_ablation_study() - Full ablation sweep")
    print("  - run_validation_across_sizes() - Cross-size validation")
    print("  - rank_configs() - Rank by multiple metrics")
    print("  - plot_ablation_heatmap(), print_ablation_summary()")
    
    print("\n" + "=" * 60)
    print("✅ Module loaded successfully!")

