"""
Common Utilities Module for QAT Training Pipelines
===================================================

Shared utility functions for CNN, ViT, and Swin training pipelines.

SECTIONS:
1. Random Seed & Reproducibility
2. Data Processing & Parsing (LGG MRI Dataset)
3. Visualization Utilities  
4. Training Utilities (AMP, Loss, Metrics)
5. Evaluation Utilities (ROC, IoU, Test Metrics)
6. Multi-Scale Plotting
7. Inference Utilities

Authors: Zijian Du and Oleg Rybakov
"""

import os
import sys
import copy
import time
import random
import re
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List, Any, Callable

import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

# Force white background for all plots (override any dark theme)
plt.style.use('default')
plt.rcParams['figure.facecolor'] = 'white'
plt.rcParams['axes.facecolor'] = 'white'
plt.rcParams['savefig.facecolor'] = 'white'

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from sklearn.metrics import roc_curve, auc

# Import unified loss functions from centralized module
from loss import dice_loss, dice_coef_metric

# IPython imports (optional, for live plotting in notebooks)
try:
    from IPython.display import clear_output, display
    _IPYTHON_AVAILABLE = True
except ImportError:
    _IPYTHON_AVAILABLE = False


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 1: RANDOM SEED & REPRODUCIBILITY
# ═══════════════════════════════════════════════════════════════════════════════

def set_seed(seed: int = 42):
    """
    Set random seeds for reproducibility across NumPy, PyTorch, and CUDA.
    
    Args:
        seed: Integer seed value (default: 42)
    
    Note:
        Enables deterministic mode which may reduce performance.
        Sets CUBLAS_WORKSPACE_CONFIG for deterministic cuBLAS operations.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        try:
            torch.use_deterministic_algorithms(True, warn_only=True)
        except Exception:
            pass
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    
    os.environ["PYTHONHASHSEED"] = str(seed)


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: DATA PROCESSING & PARSING (LGG MRI Dataset)
# ═══════════════════════════════════════════════════════════════════════════════

# ImageNet normalization constants
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

# Regex pattern for extracting numeric slice ID from filenames
# Matches: ..._45.tif -> id=45, ..._45_mask.tif -> id=45
SLICE_ID_PATTERN = re.compile(r'_(\d+)(?:_mask)?$', re.IGNORECASE)


def folder_and_id(path: str) -> Tuple[str, Optional[int]]:
    """
    Extract folder path and numeric slice ID from a file path.
    
    Uses regex to robustly extract the trailing numeric ID, handling
    both image files (xxx_45.tif) and mask files (xxx_45_mask.tif).
    
    Args:
        path: Full path to image or mask file
        
    Returns:
        Tuple of (folder_path, slice_id)
        slice_id is None if the pattern doesn't match
    
    Example:
        >>> folder_and_id("/data/patient1/TCGA_xxx_45.tif")
        ('/data/patient1', 45)
        >>> folder_and_id("/data/patient1/TCGA_xxx_45_mask.tif")
        ('/data/patient1', 45)
    """
    path = str(path)
    stem = Path(path).stem
    folder = str(Path(path).parent)
    match = SLICE_ID_PATTERN.search(stem)
    return folder, (int(match.group(1)) if match else None)


def positive_negative_diagnosis(mask_path: str) -> int:
    """
    Determine if a mask contains any positive (tumor) pixels.
    
    Args:
        mask_path: Path to mask image file
        
    Returns:
        1 if any pixel > 0 (positive/tumor present)
        0 otherwise (negative/no tumor)
    """
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        return 0
    return 1 if np.any(mask > 0) else 0


def build_file_dataframe(data_path: str) -> pd.DataFrame:
    """
    Build a DataFrame of all files in the dataset directory.
    
    This is the RAW file listing - use align_images_and_masks() or
    build_final_dataframe() to get properly aligned image-mask pairs.
    
    Args:
        data_path: Root path containing patient subdirectories
        
    Returns:
        DataFrame with columns: 'dirname' (patient folder), 'path' (file path)
    """
    import glob
    
    # Ensure path ends with / for correct glob matching
    if not data_path.endswith('/'):
        data_path = data_path + '/'
    
    data_map = []
    for sub_dir_path in glob.glob(data_path + "*"):
        if os.path.isdir(sub_dir_path):
            dirname = sub_dir_path.split("/")[-1]
            for filename in os.listdir(sub_dir_path):
                image_path = sub_dir_path + "/" + filename
                data_map.extend([dirname, image_path])
    
    return pd.DataFrame({
        "dirname": data_map[::2],
        "path": data_map[1::2]
    })


def align_images_and_masks(df: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Align image and mask paths by folder and numeric slice ID.
    
    This is the CRITICAL function that ensures images are paired with
    their correct masks using (folder, numeric_id) matching.
    
    IMPORTANT: This properly handles lexicographic sorting issues
    (e.g., _1 vs _10) by using numeric ID matching, not string sorting.
    
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
    bad_imgs, bad_masks = [], []
    
    for p in df_imgs['path'].tolist():
        f, i = folder_and_id(p)
        if i is None:
            bad_imgs.append(p)
            continue
        img_lookup[(f, i)] = p
    
    for p in df_masks['path'].tolist():
        f, i = folder_and_id(p)
        if i is None:
            bad_masks.append(p)
            continue
        mask_lookup[(f, i)] = p
    
    # Log warnings
    if bad_imgs or bad_masks:
        print("⚠️ Warning: Could not parse numeric suffix for some files.")
        if bad_imgs:
            print(f"   Unparsed images: {len(bad_imgs)} files")
        if bad_masks:
            print(f"   Unparsed masks: {len(bad_masks)} files")
    
    # Find common keys and align
    common_keys = sorted(set(img_lookup.keys()).intersection(mask_lookup.keys()))
    if not common_keys:
        raise RuntimeError("No matched (folder, id) pairs found. Check filename patterns.")
    
    imgs = [img_lookup[k] for k in common_keys]
    masks = [mask_lookup[k] for k in common_keys]
    
    print(f"✅ Aligned {len(imgs)} image-mask pairs")
    return imgs, masks


def build_final_dataframe(df_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Build the final DataFrame with aligned image/mask paths and diagnosis labels.
    
    This is the MAIN function to use for creating training/validation/test splits.
    It combines build_file_dataframe + align_images_and_masks + diagnosis labeling.
    
    Args:
        df_raw: Raw DataFrame from build_file_dataframe()
        
    Returns:
        DataFrame with columns: patient, image_path, mask_path, diagnosis
    """
    # Get aligned paths
    imgs, masks = align_images_and_masks(df_raw)
    
    # Get patient names from image paths
    patients = [Path(p).parent.name for p in imgs]
    
    # Build DataFrame
    df = pd.DataFrame({
        "patient": patients,
        "image_path": imgs,
        "mask_path": masks
    })
    
    # Add diagnosis column
    df["diagnosis"] = df["mask_path"].apply(positive_negative_diagnosis)
    
    return df


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: VISUALIZATION UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def denormalize_image(img: np.ndarray, mean: np.ndarray = IMAGENET_MEAN, 
                      std: np.ndarray = IMAGENET_STD) -> np.ndarray:
    """
    Reverse ImageNet normalization for visualization.
    
    Args:
        img: Normalized image array [H, W, C]
        mean: Normalization mean
        std: Normalization std
        
    Returns:
        Denormalized image clipped to [0, 1]
    """
    return np.clip(img * std + mean, 0, 1)


def plot_diagnosis_distribution(df: pd.DataFrame, save_path: Optional[str] = None, 
                                dpi: int = 300):
    """
    Plot distribution of positive/negative diagnoses.
    
    Args:
        df: DataFrame with 'diagnosis' column
        save_path: Optional path to save figure
        dpi: Resolution
    """
    fig, ax = plt.subplots(figsize=(10, 7), dpi=dpi)
    counts = df["diagnosis"].value_counts().sort_index()
    colors = ["#2ecc71", "#e74c3c"]
    bars = ax.bar(["Negative (0)", "Positive (1)"], counts.values, color=colors)
    
    for bar, count in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10,
                str(count), ha='center', fontsize=18, fontweight='bold', color='black')
    
    ax.set_ylabel("Count", fontsize=18, fontweight='bold')
    ax.set_xlabel("Diagnosis", fontsize=18, fontweight='bold')
    ax.set_title("Diagnosis Distribution in Dataset", fontsize=22, fontweight='bold')
    ax.tick_params(axis='both', labelsize=16)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, facecolor='white')
        print(f"✅ Saved: {save_path}")
    
    plt.show()


def plot_sample_images(df: pd.DataFrame, n_samples: int = 6, 
                       save_path: Optional[str] = None, dpi: int = 300):
    """
    Plot sample images with their masks.
    
    Args:
        df: DataFrame with image_path, mask_path columns
        n_samples: Number of samples to show
        save_path: Optional path to save figure
        dpi: Resolution
    """
    samples = df.sample(min(n_samples, len(df)))
    
    fig, axes = plt.subplots(2, n_samples, figsize=(4*n_samples, 8), dpi=dpi)
    
    for i, (_, row) in enumerate(samples.iterrows()):
        if i >= n_samples:
            break
        img = cv2.cvtColor(cv2.imread(row["image_path"]), cv2.COLOR_BGR2RGB)
        mask = cv2.imread(row["mask_path"], 0)
        
        axes[0, i].imshow(img)
        axes[0, i].set_title(f"ID: {str(row.get('patient', row.get('id', '')))[:10]}", fontsize=16)
        axes[0, i].axis("off")
        
        axes[1, i].imshow(mask, cmap="gray")
        axes[1, i].set_title(f"Diag: {row['diagnosis']}", fontsize=16)
        axes[1, i].axis("off")
    
    plt.suptitle("Sample Images and Masks", fontsize=22, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, facecolor='white')
        print(f"✅ Saved: {save_path}")
    
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 4: TRAINING UTILITIES (AMP, Loss, Metrics)
# ═══════════════════════════════════════════════════════════════════════════════

def amp_dtype_from_str(dtype_str: str) -> torch.dtype:
    """
    Map string dtype to torch.dtype for AMP autocast.
    
    Args:
        dtype_str: "fp16", "bf16", "fp32", etc.
        
    Returns:
        Corresponding torch.dtype
    """
    mapping = {
        "float16": torch.float16,
        "fp16": torch.float16,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    return mapping.get((dtype_str or "").lower(), torch.float16)


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS FUNCTIONS - Imported from common/loss.py for consistency
# Available: dice_loss, dice_coef_metric
# ═══════════════════════════════════════════════════════════════════════════════


def warmup_lr_scheduler(optimizer: torch.optim.Optimizer, 
                        warmup_iters: int, 
                        warmup_factor: float = 0.1) -> torch.optim.lr_scheduler.LambdaLR:
    """
    Create a learning rate scheduler with linear warmup.
    
    Args:
        optimizer: PyTorch optimizer
        warmup_iters: Number of warmup iterations/epochs
        warmup_factor: Starting factor for warmup (e.g., 0.1)
        
    Returns:
        LambdaLR scheduler
    """
    def lr_lambda(x):
        if x >= warmup_iters:
            return 1.0
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def prepare_model_for_training(model: nn.Module, device: torch.device, 
                               use_channels_last: bool = True) -> nn.Module:
    """
    Prepare model for training with optional optimizations.
    
    Args:
        model: PyTorch model
        device: Target device
        use_channels_last: Enable channels_last memory format
        
    Returns:
        Prepared model
    """
    model = model.to(device)
    
    if use_channels_last and device.type == 'cuda':
        try:
            model = model.to(memory_format=torch.channels_last)
        except Exception:
            pass
    
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    
    return model


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: EVALUATION UTILITIES (ROC, IoU, Test Metrics)
# ═══════════════════════════════════════════════════════════════════════════════

def compute_roc(model: nn.Module, dataloader: DataLoader, device: torch.device,
                use_amp: bool = True, amp_dtype: torch.dtype = torch.float16) -> Dict[str, Any]:
    """
    Compute ROC curve and AUC for segmentation model.
    
    Args:
        model: Trained model
        dataloader: DataLoader for evaluation
        device: Target device
        use_amp: Use automatic mixed precision
        amp_dtype: AMP dtype
        
    Returns:
        Dictionary with fpr, tpr, thresholds, auc
    """
    model.eval()
    all_probs = []
    all_targets = []
    
    with torch.no_grad():
        for images, masks in dataloader:
            images = images.to(device)
            masks = masks.to(device)
            
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                logits = model(images)
            
            # Handle 2-class or 1-class output
            if logits.shape[1] == 2:
                probs = torch.softmax(logits.float(), dim=1)[:, 1]
            else:
                probs = torch.sigmoid(logits.float()).squeeze(1)
            
            all_probs.append(probs.cpu().numpy().flatten())
            all_targets.append(masks.cpu().numpy().flatten())
    
    all_probs = np.concatenate(all_probs)
    all_targets = np.concatenate(all_targets)
    
    fpr, tpr, thresholds = roc_curve(all_targets, all_probs)
    roc_auc = auc(fpr, tpr)
    
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": roc_auc
    }


def evaluate_on_test_set(
    model: nn.Module,
    test_loader: DataLoader,
    device: torch.device,
    loss_fn: Optional[Callable] = None,
    use_amp: bool = True,
    amp_dtype: torch.dtype = torch.float16
) -> Dict[str, float]:
    """
    Evaluate model on test set, returning Dice score and optionally loss.
    
    Args:
        model: Trained model
        test_loader: Test DataLoader
        device: Target device
        loss_fn: Optional loss function
        use_amp: Use automatic mixed precision
        amp_dtype: AMP dtype
        
    Returns:
        Dict with 'test_dice', 'test_loss' (if loss_fn provided)
    """
    model.eval()
    
    total_dice = 0.0
    total_loss = 0.0
    n_batches = 0
    
    with torch.no_grad():
        for imgs, masks in test_loader:
            imgs = imgs.to(device)
            masks = masks.float().unsqueeze(1).to(device)
            
            # Clamp mask values if needed (some masks are 0-255)
            if masks.max() > 1.0:
                masks = masks / 255.0
            
            with torch.cuda.amp.autocast(enabled=use_amp, dtype=amp_dtype):
                logits = model(imgs)
            
            # Compute Dice
            probs = torch.sigmoid(logits.float())
            dice = dice_coef_metric(probs, masks)
            total_dice += dice
            
            # Compute loss if provided
            if loss_fn is not None:
                loss = loss_fn(logits.float(), masks)
                total_loss += loss.item()
            
            n_batches += 1
    
    results = {'test_dice': total_dice / n_batches}
    
    if loss_fn is not None:
        results['test_loss'] = total_loss / n_batches
    
    return results


def compute_iou(pred: np.ndarray, target: np.ndarray, threshold: float = 0.5) -> float:
    """
    Compute Intersection over Union for binary masks.
    
    Args:
        pred: Predicted probabilities
        target: Ground truth mask
        threshold: Binarization threshold
        
    Returns:
        IoU score
    """
    pred_binary = (pred > threshold).astype(np.float32)
    target_binary = (target > 0).astype(np.float32)
    
    intersection = (pred_binary * target_binary).sum()
    union = pred_binary.sum() + target_binary.sum() - intersection
    
    return intersection / (union + 1e-6)


def compute_multi_scale_test_metrics(
    test_loader: DataLoader,
    output_dir: str,
    model_scales: List[str],
    recipes: List[int],
    device: torch.device,
    create_model_fn: Callable,
    loss_fn: Optional[Callable] = None,
    model_prefix: str = "model",
    verbose: bool = True
) -> Dict[str, Dict[int, Dict[str, float]]]:
    """
    Compute test metrics for all trained models across scales and recipes.
    
    Args:
        test_loader: Test DataLoader
        output_dir: Directory containing checkpoints
        model_scales: List of scale names (e.g., ["small", "medium", "large"])
        recipes: List of recipe IDs
        device: Target device
        create_model_fn: Factory function to create model
        loss_fn: Optional loss function for test loss
        model_prefix: Prefix for checkpoint files (e.g., "cnn", "vit")
        verbose: Print progress
        
    Returns:
        Nested dict: results[scale][recipe_id] = {"test_dice": float, "test_loss": float}
    """
    results = {}
    
    for scale_name in model_scales:
        results[scale_name] = {}
        
        for recipe_id in recipes:
            ckpt_dir = os.path.join(output_dir, scale_name, "checkpoints")
            ckpt_path = os.path.join(ckpt_dir, f"{model_prefix}_{scale_name}_recipe_{recipe_id}.pth")
            
            if not os.path.exists(ckpt_path):
                if verbose:
                    print(f"⚠️ {scale_name.upper()} Recipe {recipe_id}: No checkpoint found")
                continue
            
            try:
                model = create_model_fn(scale_name=scale_name, recipe_id=recipe_id)
                model.load_state_dict(torch.load(ckpt_path, map_location=device, weights_only=True))
                model.to(device)
                
                metrics = evaluate_on_test_set(model, test_loader, device, loss_fn)
                results[scale_name][recipe_id] = metrics
                
                if verbose:
                    dice = metrics['test_dice']
                    loss_str = f", Loss={metrics.get('test_loss', 0):.4f}" if 'test_loss' in metrics else ""
                    print(f"✅ {scale_name.upper()} Recipe {recipe_id}: Test Dice = {dice:.4f}{loss_str}")
                
                del model
                torch.cuda.empty_cache()
                
            except Exception as e:
                if verbose:
                    print(f"❌ {scale_name.upper()} Recipe {recipe_id}: Error - {e}")
    
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# RECIPE COLOR SCHEME & NAMES (Shared across models)
# ═══════════════════════════════════════════════════════════════════════════════

RECIPE_COLORS = {
    0:     "#4ECDC4",   # Baseline - Teal
    1:  "#FF6B6B",   # Full NVFP4 - Coral
    8:  "#9B59B6",   # Autograd - Purple
    4:  "#3498DB",   # 2D + RHT - Blue
    5:  "#2ECC71",   # 2D + RHT + SR - Green
    6:  "#F39C12",   # SR Only - Orange
    2: "#E74C3C",   # Forward-Only - Red
    3: "#1ABC9C",   # Chain Rule - Turquoise
    7: "#E91E63",   # Forward + RHT - Pink
}

RECIPE_NAMES = {
    0:     "Baseline",
    1:  "NVFP4 Full",
    8:  "Autograd",
    4:  "2D+RHT",
    5:  "2D+RHT+SR",
    6:  "SR Only",
    2: "Fwd-Only",
    3: "Chain Rule",
    7: "Fwd+RHT",
}


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: MULTI-SCALE PLOTTING
# ═══════════════════════════════════════════════════════════════════════════════

def plot_training_summary_2x3(
    results: Dict[str, Dict[int, Dict[str, Any]]],
    model_name: str = "Model",
    recipe_names: Optional[Dict[int, str]] = None,
    figsize: Tuple[int, int] = (42, 26),
    dpi: int = 300,
    save_path: Optional[str] = None,
    selected_recipes: Optional[List[int]] = None
):
    """
    Plot comprehensive training summary: 2×3 grid (Loss/Dice × Small/Medium/Large).
    
    Combines train and val curves with visual hierarchy:
    - Validation: Solid thick lines (prominent)
    - Training: Dashed thin semi-transparent lines (background)
    
    Args:
        results: Nested dict results[scale][recipe] = {"history": {...}}
        model_name: Name for title (e.g., "CNN", "ViT")
        recipe_names: Dict mapping recipe_id -> display name
        figsize: Figure size
        dpi: Resolution
        save_path: Optional path to save figure
        selected_recipes: Optional list of recipe IDs to show (None = all)
    """
    if recipe_names is None:
        recipe_names = RECIPE_NAMES
    
    scales = list(results.keys())
    n_scales = len(scales)
    
    # Get all recipes across all scales
    all_recipes = set()
    for scale_data in results.values():
        all_recipes.update(scale_data.keys())
    all_recipes = sorted(all_recipes)
    
    if selected_recipes is not None:
        all_recipes = [r for r in all_recipes if r in selected_recipes]
    
    # Create 2×3 grid: rows=(Loss, Dice), cols=(scales)
    fig, axes = plt.subplots(2, n_scales, figsize=figsize, dpi=dpi)
    
    if n_scales == 1:
        axes = axes.reshape(2, 1)
    
    metrics = [("loss", "val_loss", "train_loss", "Loss"),
               ("dice", "val_dice", "train_dice", "Dice")]
    
    for row, (metric_key, val_metric, train_metric, metric_label) in enumerate(metrics):
        for col, scale_name in enumerate(scales):
            ax = axes[row, col]
            scale_data = results.get(scale_name, {})
            
            for recipe_id in all_recipes:
                if recipe_id not in scale_data:
                    continue
                
                data = scale_data[recipe_id]
                if "history" not in data:
                    continue
                
                history = data["history"]
                color = RECIPE_COLORS.get(recipe_id, "#888888")
                name = recipe_names.get(recipe_id, f"R{recipe_id}")
                
                # Validation curves: PROMINENT (solid, thicker)
                if val_metric in history and history[val_metric]:
                    val_values = history[val_metric]
                    epochs = range(1, len(val_values) + 1)
                    ax.plot(epochs, val_values, 
                           color=color, linewidth=3, alpha=1.0,
                           linestyle='-', label=f"{name}")
                
                # Training curves: BACKGROUND (dashed, semi-transparent)
                if train_metric in history and history[train_metric]:
                    train_values = history[train_metric]
                    epochs = range(1, len(train_values) + 1)
                    ax.plot(epochs, train_values,
                           color=color, linewidth=2.5, alpha=0.5,
                           linestyle='--')
            
            ax.set_xlabel("Epoch", fontsize=28, fontweight='bold')
            ax.grid(True, alpha=0.3)
            ax.tick_params(axis='both', labelsize=24)
            
            if row == 0:
                ax.set_title(scale_name.upper(), fontsize=32, fontweight='bold', pad=10)
            
            if col == 0:
                ax.set_ylabel(metric_label, fontsize=28, fontweight='bold')
            
            if metric_key == "dice":
                ax.set_ylim(0, 1)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.82)
    
    fig.suptitle(f"{model_name} Train-Val Loss and Dice Score: all sizes x all recipes", 
                 fontsize=42, fontweight='bold', y=0.98)
    
    handles, labels = axes[0, 0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, 
                  loc='upper center', ncol=min(len(all_recipes), 8),
                  fontsize=34, framealpha=0.95,
                  bbox_to_anchor=(0.5, 0.93))
    
    fig.text(0.5, 0.86, "Solid = Validation | Dashed (faint) = Training", 
             ha='center', fontsize=32, style='italic', alpha=0.8)
    
    if save_path:
        plt.savefig(save_path, dpi=dpi, bbox_inches="tight", facecolor='white')
        print(f"✅ Saved: {save_path}")
    
    plt.show()
    return fig


def plot_multi_scale_dice_comparison(
    results: Dict[str, Dict[int, Dict[str, Any]]],
    model_name: str = "Model",
    recipe_names: Dict[int, str] = None,
    figsize: Tuple[int, int] = (24, 14),
    dpi: int = 300,
    save_path: Optional[str] = None,
    show_train: bool = True
):
    """
    Plot final Dice scores comparing all scales and recipes.
    
    Args:
        results: Dict[scale_name][recipe_id] = {"best_val_dice": float, ...}
        model_name: Name for title
        recipe_names: Optional mapping recipe_id -> display name
        figsize: Figure size
        dpi: Resolution
        save_path: Optional path to save figure
        show_train: If True, show train dice alongside val dice
    """
    if recipe_names is None:
        recipe_names = RECIPE_NAMES
    
    scales = list(results.keys())
    all_recipes = sorted(set().union(*[set(results[s].keys()) for s in scales]))
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    n_scales = len(scales)
    n_recipes = len(all_recipes)
    
    if show_train:
        group_width = 0.8 / n_recipes
        bar_width = group_width * 0.45
    else:
        group_width = 0.8 / n_recipes
        bar_width = group_width * 0.9
    
    for i, recipe_id in enumerate(all_recipes):
        val_scores = []
        train_scores = []
        for scale in scales:
            if recipe_id in results[scale]:
                val_scores.append(results[scale][recipe_id].get("best_val_dice", 0))
                history = results[scale][recipe_id].get("history", {})
                train_dice = history.get("train_dice", [])
                train_scores.append(max(train_dice) if train_dice else 0)
            else:
                val_scores.append(0)
                train_scores.append(0)
        
        color = RECIPE_COLORS.get(recipe_id, "#7f8c8d")
        label = recipe_names.get(recipe_id, f"R{recipe_id}")
        
        if show_train:
            group_x = np.arange(n_scales) + i * group_width - (n_recipes - 1) * group_width / 2
            ax.bar(group_x - bar_width/2, train_scores, bar_width, 
                   color=color, alpha=0.5, hatch='///', edgecolor='white', linewidth=0.5)
            ax.bar(group_x + bar_width/2, val_scores, bar_width, 
                   label=label, color=color, alpha=0.9, edgecolor='white', linewidth=0.5)
        else:
            x = np.arange(n_scales) + i * group_width - (n_recipes - 1) * group_width / 2
            ax.bar(x, val_scores, bar_width * 0.9, label=label, color=color, alpha=0.85)
    
    ax.set_xlabel("Model Scale", fontsize=28, fontweight='bold')
    ax.set_ylabel("Best Dice Score", fontsize=28, fontweight='bold')
    
    title = f"{model_name} QAT: Dice Score Comparison Across Scales"
    if show_train:
        title += "\n(Hatched = Train, Solid = Validation)"
    ax.set_title(title, fontsize=32, fontweight='bold')
    
    ax.set_xticks(np.arange(n_scales))
    ax.set_xticklabels([s.upper() for s in scales], fontsize=28)
    ax.tick_params(axis='both', labelsize=22)
    ax.set_ylim(0, 1.05)
    ax.grid(axis='y', alpha=0.3)
    
    ax.legend(loc="upper center", ncol=4, fontsize=20, bbox_to_anchor=(0.5, 1.0))
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✅ Saved: {save_path}")
    
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# ROC AUC UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def save_roc_results(roc_results: Dict[str, Dict[int, Dict[str, Any]]], save_path: str):
    """Save ROC results to JSON file for later plotting without recomputation."""
    serializable = {}
    for scale, recipes in roc_results.items():
        serializable[scale] = {}
        for recipe_id, data in recipes.items():
            serializable[scale][str(recipe_id)] = {
                "fpr": data["fpr"].tolist() if hasattr(data["fpr"], 'tolist') else data["fpr"],
                "tpr": data["tpr"].tolist() if hasattr(data["tpr"], 'tolist') else data["tpr"],
                "auc": float(data["auc"])
            }
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    with open(save_path, 'w') as f:
        json.dump(serializable, f, indent=2)
    print(f"✅ ROC data saved to: {save_path}")


def load_roc_results(load_path: str) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """Load ROC results from JSON file."""
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"ROC data file not found: {load_path}")
    
    with open(load_path, 'r') as f:
        data = json.load(f)
    
    roc_results = {}
    for scale, recipes in data.items():
        roc_results[scale] = {}
        for recipe_id_str, roc_data in recipes.items():
            recipe_id = int(recipe_id_str)
            roc_results[scale][recipe_id] = {
                "fpr": np.array(roc_data["fpr"]),
                "tpr": np.array(roc_data["tpr"]),
                "auc": roc_data["auc"]
            }
    
    print(f"✅ ROC data loaded from: {load_path}")
    return roc_results


def compute_multi_scale_roc(
    val_loader: DataLoader,
    output_dir: str,
    model_scales: List[str],
    recipes: List[int],
    device: torch.device,
    create_model_fn: Callable,
    model_prefix: str = "model",
    verbose: bool = True,
    save_path: Optional[str] = None,
    force_recompute: bool = False
) -> Dict[str, Dict[int, Dict[str, Any]]]:
    """
    Compute ROC curves for all model scales and recipes.
    
    Args:
        val_loader: Validation dataloader
        output_dir: Directory containing checkpoints
        model_scales: List of scales
        recipes: List of recipe IDs
        device: Torch device
        create_model_fn: Function to create model
        model_prefix: Prefix for checkpoint files (e.g., "cnn", "vit")
        verbose: Print progress
        save_path: Path to save/load ROC data
        force_recompute: If True, recompute even if saved data exists
        
    Returns:
        Dict[scale][recipe_id] = {"fpr": array, "tpr": array, "auc": float}
    """
    if save_path is None:
        save_path = os.path.join(output_dir, "roc_data.json")
    
    if os.path.exists(save_path) and not force_recompute:
        if verbose:
            print(f"📂 Loading cached ROC data from: {save_path}")
        return load_roc_results(save_path)
    
    if verbose:
        print("Computing ROC curves (this may take a few minutes)...")
    
    roc_results = {}
    
    for scale_name in model_scales:
        checkpoint_dir = os.path.join(output_dir, scale_name, "checkpoints")
        scale_roc = {}
        
        for recipe_id in recipes:
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_prefix}_{scale_name}_recipe_{recipe_id}.pth")
            
            if not os.path.exists(checkpoint_path):
                continue
            
            try:
                model = create_model_fn(scale_name=scale_name, recipe_id=recipe_id)
                model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
                model = model.to(device).eval()
                
                roc_data = compute_roc(model, val_loader, device)
                
                scale_roc[recipe_id] = {
                    "fpr": roc_data["fpr"],
                    "tpr": roc_data["tpr"],
                    "auc": roc_data["auc"]
                }
                
                if verbose:
                    print(f"✅ {scale_name.upper()} Recipe {recipe_id}: AUC = {roc_data['auc']:.4f}")
                
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                if verbose:
                    print(f"⚠️ Error loading {scale_name}/recipe_{recipe_id}: {e}")
        
        if scale_roc:
            roc_results[scale_name] = scale_roc
    
    if roc_results:
        save_roc_results(roc_results, save_path)
    
    return roc_results


def plot_multi_scale_roc_comparison(
    roc_results: Dict[str, Dict[int, Dict[str, Any]]],
    model_name: str = "Model",
    recipe_names: Optional[Dict[int, str]] = None,
    figsize: Tuple[int, int] = (42, 18),
    dpi: int = 300,
    save_path: Optional[str] = None
):
    """
    Plot ROC curves for all scales (one subplot per scale).
    
    Args:
        roc_results: Dict[scale][recipe_id] = {"fpr": array, "tpr": array, "auc": float}
        model_name: Name for title
        recipe_names: Optional mapping recipe_id -> display name
        figsize: Figure size
        dpi: Resolution
        save_path: Optional path to save
    """
    if recipe_names is None:
        recipe_names = RECIPE_NAMES
    
    scales = list(roc_results.keys())
    n_scales = len(scales)
    
    fig, axes = plt.subplots(1, n_scales, figsize=figsize, dpi=dpi)
    if n_scales == 1:
        axes = [axes]
    
    legend_handles = []
    legend_labels = []
    
    for ax, scale_name in zip(axes, scales):
        auc_table_lines = []
        
        for recipe_id, roc_data in sorted(roc_results[scale_name].items()):
            fpr = roc_data["fpr"]
            tpr = roc_data["tpr"]
            auc_val = roc_data["auc"]
            
            color = RECIPE_COLORS.get(recipe_id, "#888888")
            name = recipe_names.get(recipe_id, f"R{recipe_id}")
            
            line, = ax.plot(fpr, tpr, color=color, linewidth=3)
            
            if scale_name == scales[0]:
                legend_handles.append(line)
                legend_labels.append(f"{name}")
            
            short_name = name[:12] + ".." if len(name) > 14 else name
            auc_table_lines.append(f"{short_name}: {auc_val:.3f}")
        
        ax.plot([0, 1], [0, 1], '--', color='gray', linewidth=2, alpha=0.5)
        
        ax.set_xlabel("False Positive Rate", fontsize=32, fontweight='bold')
        ax.set_ylabel("True Positive Rate", fontsize=32, fontweight='bold')
        ax.set_title(f"{scale_name.upper()}", fontsize=36, fontweight='bold')
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.tick_params(axis='both', labelsize=32)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        
        auc_text = "AUC\n" + "\n".join(auc_table_lines)
        ax.text(0.97, 0.03, auc_text, transform=ax.transAxes,
                fontsize=24, verticalalignment='bottom', horizontalalignment='right',
                fontfamily='monospace', fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.6', facecolor='white', 
                         edgecolor='gray', alpha=0.95, linewidth=2))
    
    fig.legend(legend_handles, legend_labels,
              loc='upper center', ncol=min(len(legend_handles), 8),
              fontsize=34, framealpha=0.95,
              bbox_to_anchor=(0.5, 0.92))
    
    plt.suptitle(f"{model_name} ROC Curves: all scales x all recipes", 
                fontsize=42, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.subplots_adjust(top=0.82, bottom=0.12)
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✅ Saved: {save_path}")
    
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: INFERENCE UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def save_inference_results(
    all_predictions: Dict[str, Dict[int, np.ndarray]], 
    image_path: str,
    save_path: str
):
    """
    Save inference predictions to NPZ file for later plotting without recomputation.
    
    Args:
        all_predictions: Dict[scale][recipe_id] = probability_map (numpy array)
        image_path: Original image path (stored as metadata)
        save_path: Path to save .npz file
    """
    # Flatten nested dict for npz storage
    arrays_to_save = {"_image_path": np.array([image_path])}
    
    for scale, recipes in all_predictions.items():
        for recipe_id, pred in recipes.items():
            key = f"{scale}__recipe_{recipe_id}"
            arrays_to_save[key] = pred
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    np.savez_compressed(save_path, **arrays_to_save)
    print(f"✅ Inference data saved to: {save_path}")


def load_inference_results(load_path: str) -> Tuple[Dict[str, Dict[int, np.ndarray]], str]:
    """
    Load inference predictions from NPZ file.
    
    Args:
        load_path: Path to .npz file
        
    Returns:
        Tuple of (all_predictions dict, original_image_path)
    """
    if not os.path.exists(load_path):
        raise FileNotFoundError(f"Inference data file not found: {load_path}")
    
    data = np.load(load_path, allow_pickle=True)
    
    # Extract metadata
    image_path = str(data["_image_path"][0]) if "_image_path" in data else ""
    
    # Reconstruct nested dict
    all_predictions = {}
    for key in data.files:
        if key.startswith("_"):
            continue  # Skip metadata keys
        
        # Parse key: "small__recipe_6003" -> scale="small", recipe_id=1
        parts = key.split("__recipe_")
        if len(parts) == 2:
            scale = parts[0]
            recipe_id = int(parts[1])
            
            if scale not in all_predictions:
                all_predictions[scale] = {}
            all_predictions[scale][recipe_id] = data[key]
    
    print(f"✅ Inference data loaded from: {load_path}")
    return all_predictions, image_path


def run_multi_scale_inference(
    image_path: str,
    mask_path: str,
    output_dir: str,
    model_scales: List[str],
    recipes: List[int],
    recipe_names: Dict[int, str],
    device: torch.device,
    create_model_fn: Callable,
    model_prefix: str = "model",
    threshold: float = 0.5,
    save_path: Optional[str] = None,
    cache_path: Optional[str] = None,
    force_recompute: bool = False,
    verbose: bool = True
) -> Dict[str, Dict[int, np.ndarray]]:
    """
    Run inference across all model scales and recipes with optional caching.
    
    Args:
        image_path: Path to input image
        mask_path: Path to ground truth mask
        output_dir: Base output directory containing checkpoints
        model_scales: List of scales
        recipes: List of recipe IDs
        recipe_names: Dict mapping recipe_id -> display name
        device: Torch device
        create_model_fn: Function to create model
        model_prefix: Prefix for checkpoint files
        threshold: Binarization threshold for predictions
        save_path: Optional path to save visualization
        cache_path: Optional path to save/load cached predictions (.npz)
                   If None, defaults to output_dir/inference_cache.npz
        force_recompute: If True, recompute even if cache exists
        verbose: Print progress
        
    Returns:
        Dict[scale][recipe_id] = probability_map (numpy array)
    """
    # Set default cache path
    if cache_path is None:
        cache_path = os.path.join(output_dir, "inference_cache.npz")
    
    # Try to load from cache
    if os.path.exists(cache_path) and not force_recompute:
        try:
            all_predictions, cached_image_path = load_inference_results(cache_path)
            # Verify it's the same image (or close enough for re-plotting)
            if verbose:
                print(f"📂 Using cached predictions from: {cache_path}")
                if cached_image_path != image_path:
                    print(f"   ⚠️ Note: Cache was for different image, but predictions are loaded anyway.")
                    print(f"      Cached: {cached_image_path}")
                    print(f"      Current: {image_path}")
            
            # Plot if requested
            if all_predictions and save_path:
                plot_multi_scale_predictions_grid(
                    image_path=image_path,
                    mask_path=mask_path,
                    all_predictions=all_predictions,
                    recipe_names=recipe_names,
                    threshold=threshold,
                    save_path=save_path
                )
            return all_predictions
        except Exception as e:
            if verbose:
                print(f"⚠️ Failed to load cache: {e}. Recomputing...")
    
    import albumentations as A
    from albumentations.pytorch.transforms import ToTensorV2
    
    inference_transform = A.Compose([
        A.Resize(256, 256),
        A.Normalize(mean=IMAGENET_MEAN.tolist(), std=IMAGENET_STD.tolist()),
        ToTensorV2()
    ])
    
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # CRITICAL: Convert BGR to RGB!
    mask = cv2.imread(mask_path, 0)
    augmented = inference_transform(image=image, mask=mask)
    image_tensor = augmented['image'].unsqueeze(0).to(device)
    
    if verbose:
        print(f"📷 Computing inference for: {image_path}")
    
    all_predictions = {}
    
    for scale_name in model_scales:
        checkpoint_dir = os.path.join(output_dir, scale_name, "checkpoints")
        scale_predictions = {}
        
        for recipe_id in recipes:
            checkpoint_path = os.path.join(checkpoint_dir, f"{model_prefix}_{scale_name}_recipe_{recipe_id}.pth")
            
            if not os.path.exists(checkpoint_path):
                continue
            
            try:
                model = create_model_fn(scale_name=scale_name, recipe_id=recipe_id)
                model.load_state_dict(torch.load(checkpoint_path, map_location="cpu"))
                model = model.to(device).eval()
                
                with torch.no_grad():
                    logits = model(image_tensor)
                    if logits.shape[1] == 2:
                        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()[0]
                    else:
                        probs = torch.sigmoid(logits).cpu().numpy()[0, 0]
                
                scale_predictions[recipe_id] = probs
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                if verbose:
                    print(f"⚠️ Error loading {scale_name}/recipe_{recipe_id}: {e}")
        
        if scale_predictions:
            all_predictions[scale_name] = scale_predictions
            if verbose:
                print(f"✅ {scale_name.upper()}: Loaded {len(scale_predictions)} recipes")
    
    # Save to cache
    if all_predictions and cache_path:
        save_inference_results(all_predictions, image_path, cache_path)
    
    if all_predictions and save_path:
        plot_multi_scale_predictions_grid(
            image_path=image_path,
            mask_path=mask_path,
            all_predictions=all_predictions,
            recipe_names=recipe_names,
            threshold=threshold,
            save_path=save_path
        )
    
    return all_predictions


def plot_multi_scale_predictions_grid(
    image_path: str,
    mask_path: str,
    all_predictions: Dict[str, Dict[int, np.ndarray]],
    recipe_names: Dict[int, str],
    model_name: str = "Model",
    threshold: float = 0.5,
    figsize_per_cell: Tuple[float, float] = (4.5, 4.5),
    dpi: int = 300,
    save_path: Optional[str] = None
):
    """Plot predictions grid: rows = scales, cols = recipes."""
    orig_image = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2RGB)
    orig_image = cv2.resize(orig_image, (256, 256))
    orig_mask = cv2.imread(mask_path, 0)
    orig_mask = cv2.resize(orig_mask, (256, 256))
    
    scales = list(all_predictions.keys())
    recipe_ids = sorted(set().union(*[set(all_predictions[s].keys()) for s in scales]))
    n_scales = len(scales)
    n_cols = len(recipe_ids) + 2
    
    fig, axes = plt.subplots(
        n_scales, n_cols,
        figsize=(figsize_per_cell[0] * n_cols, figsize_per_cell[1] * n_scales),
        dpi=dpi
    )
    
    if n_scales == 1:
        axes = axes.reshape(1, -1)
    
    for row, scale_name in enumerate(scales):
        axes[row, 0].imshow(orig_image)
        axes[row, 0].set_title("Input" if row == 0 else "", fontsize=24, fontweight='bold')
        axes[row, 0].set_ylabel(scale_name.upper(), fontsize=26, fontweight='bold')
        axes[row, 0].axis('off')
        
        axes[row, 1].imshow(orig_mask, cmap='gray')
        axes[row, 1].set_title("Ground Truth" if row == 0 else "", fontsize=24, fontweight='bold')
        axes[row, 1].axis('off')
        
        for col, recipe_id in enumerate(recipe_ids):
            ax = axes[row, col + 2]
            
            if recipe_id in all_predictions.get(scale_name, {}):
                pred = all_predictions[scale_name][recipe_id]
                ax.imshow(pred > threshold, cmap='gray')
            else:
                ax.text(0.5, 0.5, 'N/A', ha='center', va='center',
                       transform=ax.transAxes, fontsize=24, color='gray')
                ax.set_facecolor('#f0f0f0')
            
            recipe_name = recipe_names.get(recipe_id, f"R{recipe_id}")
            if len(recipe_name) > 12:
                recipe_name = recipe_name[:10] + "..."
            ax.set_title(recipe_name if row == 0 else "", fontsize=22, fontweight='bold')
            ax.axis('off')
    
    plt.suptitle(f"{model_name} Segmentation Predictions: All Scales × All Recipes",
                fontsize=36, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"✅ Saved: {save_path}")
    
    plt.show()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: TELEMETRY LOADING & 4-PANEL VISUALIZATION
# ═══════════════════════════════════════════════════════════════════════════════

def load_telemetry_for_scale(
    scale_name: str,
    output_dir: str,
    model_prefix: str = "cnn"
) -> Dict[int, pd.DataFrame]:
    """
    Load all telemetry CSV files for a given scale.
    
    Args:
        scale_name: Model scale (e.g., "small", "medium", "base")
        output_dir: Base output directory containing scale subdirectories
        model_prefix: Model name prefix in telemetry filenames
        
    Returns:
        Dict mapping recipe_id -> DataFrame with telemetry data
    """
    import csv
    
    telemetry_dir = os.path.join(output_dir, scale_name, "telemetry")
    
    if not os.path.exists(telemetry_dir):
        print(f"❌ Telemetry directory not found: {telemetry_dir}")
        return {}
    
    data = {}
    for filename in sorted(os.listdir(telemetry_dir)):
        if filename.endswith('_telemetry.csv'):
            # Parse recipe ID from filename
            # Format: cnn_small_recipe_6003_telemetry.csv
            parts = filename.replace('_telemetry.csv', '').split('_recipe_')
            if len(parts) == 2:
                recipe_id = int(parts[1])
            else:
                # Fallback: try to find recipe ID
                parts = filename.replace('_telemetry.csv', '').split('_')
                try:
                    recipe_id = int(parts[-1])
                except ValueError:
                    continue
            
            filepath = os.path.join(telemetry_dir, filename)
            try:
                df = pd.read_csv(filepath)
                data[recipe_id] = df
                print(f"✅ Loaded Recipe {recipe_id} ({RECIPE_NAMES.get(recipe_id, '?')}): {len(df)} epochs")
            except Exception as e:
                print(f"⚠️ Failed to load {filename}: {e}")
    
    return data


def plot_4panel_train_val(
    data: Dict[int, pd.DataFrame],
    scale_name: str,
    scale_params_desc: str = "",
    save_path: Optional[str] = None,
    dpi: int = 150
):
    """
    Create 2x2 grid visualization with shared y-axes for train/val comparison.
    
    Layout:
        [Train Loss]  [Val Loss]    <- Same y-range
        [Train Dice]  [Val Dice]    <- Same y-range
    
    This allows visual comparison of train vs val performance and overfitting.
    
    Args:
        data: Dict mapping recipe_id -> DataFrame with telemetry data
        scale_name: Model scale name for title
        scale_params_desc: Parameter description (e.g., "~530K params")
        save_path: Optional path to save figure
        dpi: Resolution
        
    Returns:
        matplotlib Figure object
    """
    if not data:
        print("❌ No data to plot!")
        return None
    
    # Create figure
    fig, axes = plt.subplots(2, 2, figsize=(20, 14))
    fig.patch.set_facecolor('white')
    
    # Define panels: (row, col, metric_key, title, legend_loc)
    panels = [
        (0, 0, 'train_loss', 'Train Loss', 'upper right'),
        (0, 1, 'val_loss', 'Val Loss', 'upper right'),
        (1, 0, 'train_dice', 'Train Dice', 'lower right'),
        (1, 1, 'val_dice', 'Val Dice', 'lower right'),
    ]
    
    # First pass: collect all values to determine shared y-ranges
    all_loss_values = []
    all_dice_values = []
    
    for recipe_id, df in data.items():
        if 'train_loss' in df.columns:
            all_loss_values.extend(df['train_loss'].tolist())
        if 'val_loss' in df.columns:
            all_loss_values.extend(df['val_loss'].tolist())
        if 'train_dice' in df.columns:
            all_dice_values.extend(df['train_dice'].tolist())
        if 'val_dice' in df.columns:
            all_dice_values.extend(df['val_dice'].tolist())
    
    # Calculate shared ranges with padding
    if all_loss_values:
        loss_min = min(all_loss_values) * 0.95
        loss_max = max(all_loss_values) * 1.05
    else:
        loss_min, loss_max = 0, 1
        
    if all_dice_values:
        dice_min = max(0, min(all_dice_values) - 0.05)
        dice_max = min(1.0, max(all_dice_values) + 0.05)
    else:
        dice_min, dice_max = 0, 1
    
    print(f"\n📊 Y-axis ranges:")
    print(f"   Loss:  [{loss_min:.3f}, {loss_max:.3f}]")
    print(f"   Dice:  [{dice_min:.3f}, {dice_max:.3f}]")
    
    # Second pass: plot each panel
    for row, col, metric, title, legend_loc in panels:
        ax = axes[row, col]
        
        for recipe_id in sorted(data.keys()):
            df = data[recipe_id]
            
            if metric not in df.columns:
                continue
                
            color = RECIPE_COLORS.get(recipe_id, '#888888')
            name = RECIPE_NAMES.get(recipe_id, f'R{recipe_id}')
            
            epochs = df['epoch'].values
            values = df[metric].values
            
            # Get final value for legend
            final_val = values[-1]
            
            # Plot with final value in legend
            ax.plot(epochs, values, color=color, linewidth=2.5, alpha=0.85,
                   label=f'{name} ({final_val:.4f})')
        
        # Formatting
        ax.set_xlabel('Epoch', fontsize=14, fontweight='bold')
        ax.set_ylabel(title, fontsize=14, fontweight='bold')
        ax.set_title(title, fontsize=18, fontweight='bold')
        ax.legend(loc=legend_loc, fontsize=10, title='Recipe (Final)', title_fontsize=11)
        ax.grid(True, alpha=0.3)
        
        if data:
            max_epoch = max(df['epoch'].max() for df in data.values())
            ax.set_xlim(0, max_epoch)
        
        # Apply shared y-range
        if 'loss' in metric:
            ax.set_ylim(loss_min, loss_max)
        else:
            ax.set_ylim(dice_min, dice_max)
    
    # Main title
    title_text = f'CNN {scale_name.upper()}'
    if scale_params_desc:
        title_text += f' ({scale_params_desc})'
    title_text += ': QAT Recipe Comparison\nTrain vs Validation (Shared Y-Axes for Visual Comparison)'
    
    fig.suptitle(title_text, fontsize=20, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    plt.subplots_adjust(top=0.90)
    
    # Save
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=dpi, bbox_inches='tight', facecolor='white')
        print(f"\n✅ Saved: {save_path}")
    
    plt.show()
    return fig


def print_telemetry_summary_table(
    data: Dict[int, pd.DataFrame],
    scale_name: str,
    scale_params_desc: str = ""
):
    """
    Print a summary table of all recipes' final metrics and overfitting gap.
    
    Args:
        data: Dict mapping recipe_id -> DataFrame with telemetry data
        scale_name: Model scale name
        scale_params_desc: Parameter description
    """
    if not data:
        return
    
    print("\n" + "=" * 95)
    title = f"SUMMARY: CNN {scale_name.upper()}"
    if scale_params_desc:
        title += f" ({scale_params_desc})"
    print(title)
    print("=" * 95)
    print(f"{'Recipe':<15} {'Train Loss':<12} {'Val Loss':<12} {'Train Dice':<12} {'Val Dice':<12} {'Overfit Gap':<12}")
    print("-" * 95)
    
    for recipe_id in sorted(data.keys()):
        df = data[recipe_id]
        name = RECIPE_NAMES.get(recipe_id, f"Recipe {recipe_id}")
        
        train_loss = df['train_loss'].iloc[-1] if 'train_loss' in df.columns else 0
        val_loss = df['val_loss'].iloc[-1] if 'val_loss' in df.columns else 0
        train_dice = df['train_dice'].iloc[-1] if 'train_dice' in df.columns else 0
        val_dice = df['val_dice'].iloc[-1] if 'val_dice' in df.columns else 0
        overfit_gap = train_dice - val_dice  # Positive = overfitting
        
        print(f"{name:<15} {train_loss:<12.4f} {val_loss:<12.4f} {train_dice:<12.4f} {val_dice:<12.4f} {overfit_gap:<+12.4f}")
    
    print("=" * 95)
    print("Note: Overfit Gap = Train Dice - Val Dice (positive = overfitting, negative = underfitting)")
    print("=" * 95)


def visualize_scale_telemetry(
    scale_name: str,
    output_dir: str,
    model_prefix: str = "cnn",
    scale_params_desc: str = "",
    save_plot: bool = True
):
    """
    Convenience function to load telemetry and generate 4-panel visualization.
    
    Args:
        scale_name: Model scale (e.g., "small", "medium", "base")
        output_dir: Base output directory
        model_prefix: Model name prefix
        scale_params_desc: Parameter description for title
        save_plot: Whether to save the plot
        
    Returns:
        Dict mapping recipe_id -> DataFrame (the loaded telemetry data)
    """
    print(f"\n{'='*60}")
    print(f"📊 4-Panel Train/Val Comparison: {scale_name.upper()}")
    print(f"{'='*60}\n")
    
    # Load data
    data = load_telemetry_for_scale(scale_name, output_dir, model_prefix)
    
    if not data:
        print(f"\n❌ No telemetry found for scale '{scale_name}'")
        return {}
    
    # Print summary table
    print_telemetry_summary_table(data, scale_name, scale_params_desc)
    
    # Plot
    save_path = None
    if save_plot:
        plots_dir = os.path.join(output_dir, 'plots')
        os.makedirs(plots_dir, exist_ok=True)
        save_path = os.path.join(plots_dir, f'{scale_name}_4panel_train_val.png')
    
    plot_4panel_train_val(data, scale_name, scale_params_desc, save_path)
    
    return data


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE TEST
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 60)
    print("Common Utilities Module for QAT Training Pipelines")
    print("=" * 60)
    print("\nShared functions for CNN, ViT, and Swin:")
    print("\n📦 Data Processing:")
    print("  - set_seed()")
    print("  - folder_and_id()           # Robust regex-based ID extraction")
    print("  - build_file_dataframe()    # List all files")
    print("  - align_images_and_masks()  # CRITICAL: Proper (folder,id) matching")
    print("  - build_final_dataframe()   # Combined with diagnosis labels")
    print("\n📊 Loss Functions:")
    print("  - dice_loss()")
    print("  - dice_coef_metric()")
    print("\n📈 Metrics & Evaluation:")
    print("  - dice_coef_metric()")
    print("  - compute_roc()")
    print("  - compute_iou()")
    print("  - evaluate_on_test_set()")
    print("  - compute_multi_scale_test_metrics()")
    print("\n🎨 Plotting:")
    print("  - plot_training_summary_2x3()")
    print("  - plot_multi_scale_dice_comparison()")
    print("  - plot_multi_scale_roc_comparison()")
    print("  - plot_multi_scale_predictions_grid()")
    print("\n✅ Module loaded successfully!")
