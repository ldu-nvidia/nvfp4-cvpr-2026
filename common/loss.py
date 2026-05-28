"""
Unified Loss Functions & Evaluation Metrics for Brain Tumor Segmentation

This module provides a SINGLE SOURCE OF TRUTH for loss functions and evaluation
metrics used across all model architectures. All training, validation, and 
testing should import from here to guarantee consistency.

═══════════════════════════════════════════════════════════════════════════════
LOSS FUNCTIONS (for training)
═══════════════════════════════════════════════════════════════════════════════

    - dice_loss: Pure Dice loss (class-imbalance robust, macro-average)
    - bce_dice_loss: BCE + Dice combined (better probability calibration)
    - tversky_loss: Recall-focused loss (penalizes False Negatives more)
    - bce_tversky_loss: BCE + Tversky combined (stable gradients + recall-focused)

Loss Registry (use get_loss_fn to get by name):
    - "dice"        → dice_loss
    - "bce+dice"    → bce_dice_loss
    - "tversky"     → tversky_loss (recall-focused, α=0.3, β=0.7)
    - "bce+tversky" → bce_tversky_loss

═══════════════════════════════════════════════════════════════════════════════
EVALUATION METRICS (for testing/visualization)
═══════════════════════════════════════════════════════════════════════════════

First Tier (Primary) - Always Report:
    - recall_metric: Sensitivity, YOUR #1 METRIC - "% of tumors caught"
    - f2_score_metric: Recall-weighted F-score (recall 2x more important)
    - dice_coef_metric: Standard segmentation overlap
    - compute_auprc: Area Under PR Curve (honest for imbalanced data)
    - compute_pr_curve: Returns (precision, recall, thresholds) for plotting

Second Tier (Secondary) - Optional:
    - precision_metric: "% of predictions that are correct"
    - iou_metric: Jaccard index, stricter than Dice
    - compute_roc_auc: Area Under ROC Curve (can be inflated for imbalanced)
    - compute_roc_curve: Returns (fpr, tpr, thresholds) for plotting

Comprehensive:
    - compute_all_metrics: Returns dict with ALL metrics at once
    - print_metrics_summary: Pretty-print metrics summary

═══════════════════════════════════════════════════════════════════════════════
FORMULAS
═══════════════════════════════════════════════════════════════════════════════

Loss Functions:
    Dice_Loss = 1 - 2*TP / (2*TP + FP + FN)
    Tversky_Loss = 1 - TP / (TP + α*FP + β*FN)   [default: α=0.3, β=0.7]
    
Metrics:
    Recall = TP / (TP + FN)                      [Sensitivity]
    Precision = TP / (TP + FP)                   [PPV]
    F2 = 5*P*R / (4*P + R)                       [Recall-weighted]
    Dice = 2*TP / (2*TP + FP + FN)
    IoU = TP / (TP + FP + FN)                    [Jaccard]
    AUPRC = ∫ Precision d(Recall)               [No TN - honest for imbalance]
    ROC-AUC = ∫ TPR d(FPR)                       [Can be inflated by large TN]

═══════════════════════════════════════════════════════════════════════════════
USAGE EXAMPLES
═══════════════════════════════════════════════════════════════════════════════

Training:
    from common.loss import get_loss_fn, tversky_loss
    
    # Get loss by name
    loss_fn = get_loss_fn("bce+tversky")
    loss = loss_fn(logits, targets)
    
    # Or use directly with custom α, β
    loss = tversky_loss(logits, targets, alpha=0.3, beta=0.7)

Evaluation:
    from common.loss import (
        recall_metric, f2_score_metric, compute_auprc,
        compute_all_metrics, print_metrics_summary
    )
    
    # Individual metrics
    recall = recall_metric(probs, targets)
    auprc = compute_auprc(probs, targets)
    
    # All metrics at once
    metrics = compute_all_metrics(probs, targets)
    print_metrics_summary(metrics)

Visualization:
    from common.loss import compute_pr_curve, compute_roc_curve
    import matplotlib.pyplot as plt
    
    # PR Curve
    precisions, recalls, _ = compute_pr_curve(probs, targets)
    plt.plot(recalls, precisions)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    
    # ROC Curve
    fprs, tprs, _ = compute_roc_curve(probs, targets)
    plt.plot(fprs, tprs)
    plt.xlabel('FPR')
    plt.ylabel('TPR')
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


def dice_loss(logits: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """
    Pure Dice loss using MACRO-AVERAGE (per-image, then averaged).
    
    This is the PRIMARY LOSS FUNCTION for training. It is:
    - CLASS-IMBALANCE ROBUST: ignores true negatives, equal weight per image
    - CONSISTENT: same formula used in training, validation, and testing
    
    Formula:
        Dice(P, T) = 2 * sum(P * T) / (sum(P) + sum(T))
        Loss = 1 - mean(Dice across images in batch)
    
    Args:
        logits: Raw model output [B, 1, H, W] or [B, H, W]
                For 2-class output [B, 2, H, W], uses softmax on class 1
        targets: Binary mask [B, 1, H, W] or [B, H, W], values in {0,1} or {0,255}
        eps: Smoothing factor for numerical stability
        
    Returns:
        Dice loss scalar (1 - dice_coefficient)
    """
    # Handle 2-class output (legacy ViT) vs 1-class output (current)
    if logits.dim() == 4 and logits.shape[1] == 2:
        probs = torch.softmax(logits.float(), dim=1)[:, 1:2]  # [B, 1, H, W]
    else:
        probs = torch.sigmoid(logits.float())
    
    # Ensure probs is [B, 1, H, W]
    if probs.ndim == 3:
        probs = probs.unsqueeze(1)
    
    # Ensure targets is [B, 1, H, W] float in {0, 1}
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()
    if targets.max() > 1:
        targets = targets / 255.0
    
    # Per-image Dice (macro-average)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice_per_image = (2.0 * intersection + eps) / (union + eps)
    
    return 1.0 - dice_per_image.mean()


def bce_dice_loss(logits: torch.Tensor, targets: torch.Tensor, 
                  bce_weight: float = 1.0, dice_weight: float = 1.0,
                  eps: float = 1e-6) -> torch.Tensor:
    """
    Combined BCE + Dice loss for binary segmentation.
    
    This loss combines:
    - BCE: Pixel-wise binary cross entropy (good for probability calibration)
    - Dice: Region-based overlap metric (good for class imbalance)
    
    The combination often produces:
    - Better calibrated probabilities (smoother ROC curves)
    - Good segmentation overlap
    
    Formula:
        Loss = bce_weight * BCE(logits, targets) + dice_weight * Dice_Loss(logits, targets)
    
    Args:
        logits: Raw model output [B, 1, H, W] or [B, H, W]
        targets: Binary mask [B, 1, H, W] or [B, H, W], values in {0,1} or {0,255}
        bce_weight: Weight for BCE component (default: 1.0)
        dice_weight: Weight for Dice component (default: 1.0)
        eps: Smoothing factor for numerical stability
        
    Returns:
        Combined loss scalar
    """
    # Ensure targets is [B, 1, H, W] float in {0, 1}
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()
    if targets.max() > 1:
        targets = targets / 255.0
    
    # Ensure logits is [B, 1, H, W]
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    
    # Clamp logits to prevent extreme values
    logits = torch.clamp(logits.float(), min=-50.0, max=50.0)
    
    # BCE loss (pixel-wise)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
    
    # Dice loss (reuse the dice_loss function logic)
    probs = torch.sigmoid(logits)
    intersection = (probs * targets).sum(dim=(2, 3))
    union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
    dice_per_image = (2.0 * intersection + eps) / (union + eps)
    dice = 1.0 - dice_per_image.mean()
    
    return bce_weight * bce + dice_weight * dice


def tversky_loss(logits: torch.Tensor, targets: torch.Tensor,
                 alpha: float = 0.3, beta: float = 0.7,
                 eps: float = 1e-6) -> torch.Tensor:
    """
    Tversky Loss for RECALL-FOCUSED segmentation (prioritizes reducing False Negatives).
    
    This is a generalization of Dice Loss that allows asymmetric weighting of
    False Positives (FP) and False Negatives (FN). Use this when missing a 
    positive (e.g., tumor) is more costly than a false alarm.
    
    Formula:
        Tversky Index = TP / (TP + α*FP + β*FN)
        Tversky Loss = 1 - Tversky Index
    
    Special Cases:
        - α=0.5, β=0.5  →  Equivalent to Dice Loss
        - α=0.3, β=0.7  →  Recall-focused (penalizes FN 2.3x more than FP)
        - α=0.7, β=0.3  →  Precision-focused (penalizes FP more than FN)
    
    Soft/Differentiable Implementation:
        TP_soft = Σ(p * y)           # predicted prob × ground truth
        FP_soft = Σ(p * (1-y))       # predicted prob × (not ground truth)
        FN_soft = Σ((1-p) * y)       # (1 - predicted prob) × ground truth
    
    Args:
        logits: Raw model output [B, 1, H, W] or [B, H, W]
        targets: Binary mask [B, 1, H, W] or [B, H, W], values in {0,1} or {0,255}
        alpha: Weight for False Positives (default: 0.3 = low penalty for FP)
        beta: Weight for False Negatives (default: 0.7 = high penalty for FN)
        eps: Smoothing factor for numerical stability
        
    Returns:
        Tversky loss scalar (1 - tversky_index)
        
    Example:
        # Recall-focused (default): penalize missing tumors more
        loss = tversky_loss(logits, targets, alpha=0.3, beta=0.7)
        
        # Equivalent to Dice loss
        loss = tversky_loss(logits, targets, alpha=0.5, beta=0.5)
    """
    # Handle 2-class output (legacy ViT) vs 1-class output (current)
    if logits.dim() == 4 and logits.shape[1] == 2:
        probs = torch.softmax(logits.float(), dim=1)[:, 1:2]  # [B, 1, H, W]
    else:
        probs = torch.sigmoid(logits.float())
    
    # Ensure probs is [B, 1, H, W]
    if probs.ndim == 3:
        probs = probs.unsqueeze(1)
    
    # Ensure targets is [B, 1, H, W] float in {0, 1}
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()
    if targets.max() > 1:
        targets = targets / 255.0
    
    # Soft TP, FP, FN (fully differentiable)
    # TP: predicted positive AND actually positive
    TP = (probs * targets).sum(dim=(2, 3))
    # FP: predicted positive AND actually negative
    FP = (probs * (1 - targets)).sum(dim=(2, 3))
    # FN: predicted negative AND actually positive (MISSED!)
    FN = ((1 - probs) * targets).sum(dim=(2, 3))
    
    # Tversky Index (per-image, then macro-average)
    tversky_index = (TP + eps) / (TP + alpha * FP + beta * FN + eps)
    
    return 1.0 - tversky_index.mean()


def bce_tversky_loss(logits: torch.Tensor, targets: torch.Tensor,
                     alpha: float = 0.3, beta: float = 0.7,
                     bce_weight: float = 1.0, tversky_weight: float = 1.0,
                     eps: float = 1e-6) -> torch.Tensor:
    """
    Combined BCE + Tversky loss for RECALL-FOCUSED segmentation.
    
    Combines:
    - BCE: Pixel-wise loss for stable gradients and probability calibration
    - Tversky: Region-based loss that penalizes False Negatives more
    
    Formula:
        Loss = bce_weight * BCE + tversky_weight * Tversky_Loss
    
    Args:
        logits: Raw model output [B, 1, H, W] or [B, H, W]
        targets: Binary mask [B, 1, H, W] or [B, H, W], values in {0,1} or {0,255}
        alpha: Tversky weight for FP (default: 0.3)
        beta: Tversky weight for FN (default: 0.7)
        bce_weight: Weight for BCE component (default: 1.0)
        tversky_weight: Weight for Tversky component (default: 1.0)
        eps: Smoothing factor for numerical stability
        
    Returns:
        Combined loss scalar
    """
    # Ensure targets is [B, 1, H, W] float in {0, 1}
    if targets.ndim == 3:
        targets = targets.unsqueeze(1)
    targets = targets.float()
    if targets.max() > 1:
        targets = targets / 255.0
    
    # Ensure logits is [B, 1, H, W]
    if logits.ndim == 3:
        logits = logits.unsqueeze(1)
    
    # Clamp logits to prevent extreme values
    logits = torch.clamp(logits.float(), min=-50.0, max=50.0)
    
    # BCE loss (pixel-wise, stable gradients)
    bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='mean')
    
    # Tversky loss (recall-focused)
    probs = torch.sigmoid(logits)
    TP = (probs * targets).sum(dim=(2, 3))
    FP = (probs * (1 - targets)).sum(dim=(2, 3))
    FN = ((1 - probs) * targets).sum(dim=(2, 3))
    tversky_index = (TP + eps) / (TP + alpha * FP + beta * FN + eps)
    tversky = 1.0 - tversky_index.mean()
    
    return bce_weight * bce + tversky_weight * tversky


def dice_coef_metric(inputs: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Compute Dice coefficient for monitoring during training.
    
    Uses MACRO-AVERAGE: calculates per-image Dice, then averages.
    This matches the dice_loss formula exactly, so Loss + Dice ≈ 1.0
    
    Args:
        inputs: Probabilities in [0,1], shape [B, 1, H, W] or [B, H, W]
        target: Binary mask in {0,1}, shape [B, 1, H, W] or [B, H, W]
        eps: Smoothing factor
        
    Returns:
        Dice coefficient as float (macro-average across batch)
    """
    # Ensure inputs is [B, 1, H, W]
    if inputs.ndim == 3:
        inputs = inputs.unsqueeze(1)
    if target.ndim == 3:
        target = target.unsqueeze(1)
    
    inputs = inputs.float()
    target = target.float()
    if target.max() > 1:
        target = target / 255.0
    
    # Per-image Dice
    intersection = (inputs * target).sum(dim=(2, 3))
    union = inputs.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    dice_per_image = (2.0 * intersection + eps) / (union + eps)
    
    return dice_per_image.mean().item()


# ═══════════════════════════════════════════════════════════════════════════════
# SOFT EVALUATION METRICS (no threshold, uses probabilities directly)
# ═══════════════════════════════════════════════════════════════════════════════
#
# These metrics use soft (probability-weighted) computations instead of
# hard-thresholding at 0.5. This avoids threshold artifacts during training,
# especially on imbalanced data where the optimal threshold is far from 0.5.
#
# Soft TP = Σ(p · y),  Soft FP = Σ(p · (1-y)),  Soft FN = Σ((1-p) · y)
#
# ═══════════════════════════════════════════════════════════════════════════════


def soft_recall_metric(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Soft Recall: probability-weighted, no threshold.

    Soft_Recall = Σ(p · y) / Σ(y)

    Interpretation: average predicted probability on positive pixels.
    Range [0, 1]. Equivalent to hard recall as predictions become binary.
    """
    probs = probs.float().flatten()
    targets = targets.float().flatten()
    if targets.max() > 1:
        targets = targets / 255.0

    TP = (probs * targets).sum()
    FN = ((1 - probs) * targets).sum()
    return (TP / (TP + FN + eps)).item()


def soft_precision_metric(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Soft Precision: probability-weighted, no threshold.

    Soft_Precision = Σ(p · y) / Σ(p)

    Interpretation: what fraction of predicted probability mass lands on actual positives.
    Range [0, 1]. Equivalent to hard precision as predictions become binary.
    """
    probs = probs.float().flatten()
    targets = targets.float().flatten()
    if targets.max() > 1:
        targets = targets / 255.0

    TP = (probs * targets).sum()
    FP = (probs * (1 - targets)).sum()
    return (TP / (TP + FP + eps)).item()


def soft_f2_score_metric(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Soft F2-Score: probability-weighted, no threshold.

    F2 = 5 · P · R / (4P + R)  using soft precision and soft recall.

    Weights recall 2x more than precision. No threshold artifact.
    """
    P = soft_precision_metric(probs, targets, eps)
    R = soft_recall_metric(probs, targets, eps)
    beta = 2.0
    return (1 + beta**2) * P * R / (beta**2 * P + R + eps)


def soft_iou_metric(probs: torch.Tensor, targets: torch.Tensor, eps: float = 1e-6) -> float:
    """
    Soft IoU (Jaccard): probability-weighted, no threshold.

    Soft_IoU = Σ(p · y) / (Σ(p) + Σ(y) - Σ(p · y))

    Equivalent to hard IoU as predictions become binary.
    """
    probs = probs.float().flatten()
    targets = targets.float().flatten()
    if targets.max() > 1:
        targets = targets / 255.0

    intersection = (probs * targets).sum()
    union = probs.sum() + targets.sum() - intersection
    return (intersection / (union + eps)).item()


# ═══════════════════════════════════════════════════════════════════════════════
# HARD-THRESHOLDED METRICS (for confusion matrix, curve sweeps, final reporting)
# ═══════════════════════════════════════════════════════════════════════════════
#
# These use a fixed threshold (default 0.5) to binarize predictions.
# Use for: confusion matrix components (TP/FP/FN/TN), PR/ROC curves,
# and final reporting where a specific operating point is needed.
#
# For training monitoring, prefer the soft versions above.
#
# ═══════════════════════════════════════════════════════════════════════════════

import numpy as np
from typing import Tuple, Dict, Union, Optional


def _prepare_for_metrics(probs: torch.Tensor, targets: torch.Tensor, 
                         threshold: float = 0.5) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Prepare predictions and targets for metric computation.
    
    Args:
        probs: Probabilities in [0,1], any shape (will be flattened)
        targets: Binary mask in {0,1}, any shape (will be flattened)
        threshold: Threshold for binary prediction
        
    Returns:
        Tuple of (probs_flat, preds_flat, targets_flat) as numpy arrays
    """
    # Convert to numpy and flatten
    if isinstance(probs, torch.Tensor):
        probs_np = probs.detach().cpu().numpy().flatten()
    else:
        probs_np = np.array(probs).flatten()
    
    if isinstance(targets, torch.Tensor):
        targets_np = targets.detach().cpu().numpy().flatten()
    else:
        targets_np = np.array(targets).flatten()
    
    # Normalize targets to {0, 1}
    if targets_np.max() > 1:
        targets_np = targets_np / 255.0
    targets_np = (targets_np > 0.5).astype(np.float32)
    
    # Binary predictions at threshold
    preds_np = (probs_np >= threshold).astype(np.float32)
    
    return probs_np, preds_np, targets_np


def _compute_tp_fp_fn_tn(preds: np.ndarray, targets: np.ndarray) -> Tuple[int, int, int, int]:
    """
    Compute confusion matrix components.
    
    Args:
        preds: Binary predictions (0 or 1)
        targets: Binary ground truth (0 or 1)
        
    Returns:
        Tuple of (TP, FP, FN, TN)
    """
    TP = np.sum((preds == 1) & (targets == 1))
    FP = np.sum((preds == 1) & (targets == 0))
    FN = np.sum((preds == 0) & (targets == 1))
    TN = np.sum((preds == 0) & (targets == 0))
    return int(TP), int(FP), int(FN), int(TN)


# ═══════════════════════════════════════════════════════════════════════════════
# FIRST TIER METRICS (Primary - Always Report)
# ═══════════════════════════════════════════════════════════════════════════════

def recall_metric(probs: torch.Tensor, targets: torch.Tensor, 
                  threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    Compute Recall (Sensitivity, True Positive Rate).
    
    YOUR #1 METRIC for medical imaging - "What % of tumors did we catch?"
    
    Formula:
        Recall = TP / (TP + FN)
    
    Args:
        probs: Probabilities in [0,1]
        targets: Binary ground truth
        threshold: Decision threshold (default: 0.5)
        eps: Small value to prevent division by zero
        
    Returns:
        Recall as float in [0, 1]
    """
    _, preds, targets_np = _prepare_for_metrics(probs, targets, threshold)
    TP, FP, FN, TN = _compute_tp_fp_fn_tn(preds, targets_np)
    
    return TP / (TP + FN + eps)


def f2_score_metric(probs: torch.Tensor, targets: torch.Tensor,
                    threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    Compute F2-Score (Recall-weighted F-score).
    
    F2 weights Recall 2x more than Precision - perfect for medical imaging
    where missing a tumor (FN) is worse than a false alarm (FP).
    
    Formula:
        F2 = (1 + 2²) * (P * R) / (2² * P + R) = 5PR / (4P + R)
    
    Args:
        probs: Probabilities in [0,1]
        targets: Binary ground truth
        threshold: Decision threshold (default: 0.5)
        eps: Small value to prevent division by zero
        
    Returns:
        F2-score as float in [0, 1]
    """
    _, preds, targets_np = _prepare_for_metrics(probs, targets, threshold)
    TP, FP, FN, TN = _compute_tp_fp_fn_tn(preds, targets_np)
    
    precision = TP / (TP + FP + eps)
    recall = TP / (TP + FN + eps)
    
    beta = 2.0
    f2 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + eps)
    
    return f2


def compute_pr_curve(probs: torch.Tensor, targets: torch.Tensor,
                     num_thresholds: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute Precision-Recall curve by sweeping thresholds.
    
    Args:
        probs: Probabilities in [0,1]
        targets: Binary ground truth
        num_thresholds: Number of threshold points to evaluate
        
    Returns:
        Tuple of (precision_array, recall_array, thresholds_array)
        - Arrays are sorted by increasing threshold
        - Use for plotting: plt.plot(recall, precision)
    """
    probs_np, _, targets_np = _prepare_for_metrics(probs, targets, threshold=0.5)
    
    # Generate thresholds from 0 to 1
    thresholds = np.linspace(0, 1, num_thresholds)
    
    precisions = []
    recalls = []
    eps = 1e-6
    
    for thresh in thresholds:
        preds = (probs_np >= thresh).astype(np.float32)
        TP, FP, FN, TN = _compute_tp_fp_fn_tn(preds, targets_np)
        
        precision = TP / (TP + FP + eps)
        recall = TP / (TP + FN + eps)
        
        precisions.append(precision)
        recalls.append(recall)
    
    return np.array(precisions), np.array(recalls), thresholds


def compute_auprc(probs: torch.Tensor, targets: torch.Tensor,
                  num_thresholds: int = 100) -> float:
    """
    Compute Area Under Precision-Recall Curve (AUPRC).
    
    Uses sklearn.metrics.average_precision_score which considers all unique
    prediction values as thresholds, giving an exact result. Falls back to
    manual trapezoidal integration if sklearn is unavailable.
    
    Args:
        probs: Probabilities in [0,1]
        targets: Binary ground truth
        num_thresholds: Ignored when sklearn is available (kept for interface compatibility)
        
    Returns:
        AUPRC as float in [0, 1]
        - Random baseline = proportion of positive class (e.g., 0.015 for 1.5%)
        - Perfect = 1.0
    """
    probs_np = probs.detach().cpu().numpy().ravel() if isinstance(probs, torch.Tensor) else np.asarray(probs).ravel()
    targets_np = targets.detach().cpu().numpy().ravel() if isinstance(targets, torch.Tensor) else np.asarray(targets).ravel()
    
    try:
        from sklearn.metrics import average_precision_score
        return float(average_precision_score(targets_np, probs_np))
    except ImportError:
        precisions, recalls, _ = compute_pr_curve(probs, targets, num_thresholds)
        sorted_indices = np.argsort(recalls)
        auprc = np.trapz(precisions[sorted_indices], recalls[sorted_indices])
        return max(0.0, min(1.0, auprc))


# ═══════════════════════════════════════════════════════════════════════════════
# SECOND TIER METRICS (Secondary - Optional)
# ═══════════════════════════════════════════════════════════════════════════════

def precision_metric(probs: torch.Tensor, targets: torch.Tensor,
                     threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    Compute Precision (Positive Predictive Value).
    
    "Of what we predicted as tumor, what % is actually tumor?"
    
    Formula:
        Precision = TP / (TP + FP)
    
    Args:
        probs: Probabilities in [0,1]
        targets: Binary ground truth
        threshold: Decision threshold (default: 0.5)
        eps: Small value to prevent division by zero
        
    Returns:
        Precision as float in [0, 1]
    """
    _, preds, targets_np = _prepare_for_metrics(probs, targets, threshold)
    TP, FP, FN, TN = _compute_tp_fp_fn_tn(preds, targets_np)
    
    return TP / (TP + FP + eps)


def iou_metric(probs: torch.Tensor, targets: torch.Tensor,
               threshold: float = 0.5, eps: float = 1e-6) -> float:
    """
    Compute IoU (Intersection over Union, Jaccard Index).
    
    Stricter than Dice: IoU = Dice / (2 - Dice)
    
    Formula:
        IoU = TP / (TP + FP + FN)
    
    Args:
        probs: Probabilities in [0,1]
        targets: Binary ground truth
        threshold: Decision threshold (default: 0.5)
        eps: Small value to prevent division by zero
        
    Returns:
        IoU as float in [0, 1]
    """
    _, preds, targets_np = _prepare_for_metrics(probs, targets, threshold)
    TP, FP, FN, TN = _compute_tp_fp_fn_tn(preds, targets_np)
    
    return TP / (TP + FP + FN + eps)


def compute_roc_curve(probs: torch.Tensor, targets: torch.Tensor,
                      num_thresholds: int = 100) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve by sweeping thresholds.
    
    Args:
        probs: Probabilities in [0,1]
        targets: Binary ground truth
        num_thresholds: Number of threshold points to evaluate
        
    Returns:
        Tuple of (fpr_array, tpr_array, thresholds_array)
        - Use for plotting: plt.plot(fpr, tpr)
    """
    probs_np, _, targets_np = _prepare_for_metrics(probs, targets, threshold=0.5)
    
    # Generate thresholds from 0 to 1
    thresholds = np.linspace(0, 1, num_thresholds)
    
    fprs = []
    tprs = []
    eps = 1e-6
    
    for thresh in thresholds:
        preds = (probs_np >= thresh).astype(np.float32)
        TP, FP, FN, TN = _compute_tp_fp_fn_tn(preds, targets_np)
        
        tpr = TP / (TP + FN + eps)  # Sensitivity / Recall
        fpr = FP / (FP + TN + eps)  # 1 - Specificity
        
        tprs.append(tpr)
        fprs.append(fpr)
    
    return np.array(fprs), np.array(tprs), thresholds


def compute_roc_auc(probs: torch.Tensor, targets: torch.Tensor,
                    num_thresholds: int = 100) -> float:
    """
    Compute Area Under ROC Curve (ROC-AUC).
    
    Standard metric but CAN BE INFLATED for imbalanced data due to
    high True Negative count. Consider using AUPRC alongside.
    
    Uses trapezoidal integration over the ROC curve.
    
    Args:
        probs: Probabilities in [0,1]
        targets: Binary ground truth
        num_thresholds: Number of threshold points for integration
        
    Returns:
        ROC-AUC as float in [0, 1]
        - Random baseline = 0.5
        - Perfect = 1.0
    """
    fprs, tprs, _ = compute_roc_curve(probs, targets, num_thresholds)
    
    # Sort by FPR for proper integration
    sorted_indices = np.argsort(fprs)
    fprs_sorted = fprs[sorted_indices]
    tprs_sorted = tprs[sorted_indices]
    
    # Trapezoidal integration
    roc_auc = np.trapz(tprs_sorted, fprs_sorted)
    
    # Ensure in valid range
    return max(0.0, min(1.0, roc_auc))


# ═══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE METRICS COMPUTATION
# ═══════════════════════════════════════════════════════════════════════════════

def compute_all_metrics(probs: torch.Tensor, targets: torch.Tensor,
                        threshold: float = 0.5,
                        num_thresholds: int = 100) -> Dict[str, float]:
    """
    Compute ALL evaluation metrics at once.
    
    Returns a dictionary with:
    
    First Tier (Primary):
        - recall: Sensitivity (YOUR #1 METRIC)
        - f2_score: Recall-weighted F-score
        - dice: Standard segmentation overlap
        - auprc: Area Under PR Curve (honest for imbalanced)
    
    Second Tier (Secondary):
        - precision: Positive Predictive Value
        - roc_auc: Area Under ROC Curve
        - iou: Jaccard Index
    
    Confusion Matrix:
        - tp, fp, fn, tn: Raw counts
        
    Args:
        probs: Probabilities in [0,1]
        targets: Binary ground truth
        threshold: Decision threshold for point metrics
        num_thresholds: Resolution for curve-based metrics
        
    Returns:
        Dictionary with all metric values
    """
    # Prepare data
    probs_np, preds_np, targets_np = _prepare_for_metrics(probs, targets, threshold)
    TP, FP, FN, TN = _compute_tp_fp_fn_tn(preds_np, targets_np)
    
    eps = 1e-6
    
    # Point metrics
    recall = TP / (TP + FN + eps)
    precision = TP / (TP + FP + eps)
    
    # F2-score
    beta = 2.0
    f2 = (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall + eps)
    
    # Dice (soft, using probabilities)
    dice = dice_coef_metric(
        torch.tensor(probs_np).view(1, 1, -1, 1),
        torch.tensor(targets_np).view(1, 1, -1, 1)
    )
    
    # IoU
    iou = TP / (TP + FP + FN + eps)
    
    # Curve-based metrics
    auprc = compute_auprc(
        torch.tensor(probs_np), 
        torch.tensor(targets_np), 
        num_thresholds
    )
    roc_auc = compute_roc_auc(
        torch.tensor(probs_np), 
        torch.tensor(targets_np), 
        num_thresholds
    )
    
    return {
        # First Tier (Primary)
        'recall': recall,
        'f2_score': f2,
        'dice': dice,
        'auprc': auprc,
        
        # Second Tier (Secondary)
        'precision': precision,
        'roc_auc': roc_auc,
        'iou': iou,
        
        # Confusion Matrix
        'tp': TP,
        'fp': FP,
        'fn': FN,
        'tn': TN,
        
        # Derived rates
        'fpr': FP / (FP + TN + eps),  # False Positive Rate
        'fnr': FN / (TP + FN + eps),  # False Negative Rate = 1 - Recall
    }


def print_metrics_summary(metrics: Dict[str, float], title: str = "Evaluation Metrics") -> None:
    """
    Pretty-print metrics summary.
    
    Args:
        metrics: Dictionary from compute_all_metrics()
        title: Optional title for the summary
    """
    print(f"\n{'═' * 60}")
    print(f"  {title}")
    print(f"{'═' * 60}")
    
    print(f"\n  🥇 FIRST TIER (Primary)")
    print(f"  {'─' * 40}")
    print(f"    Recall (Sensitivity):  {metrics['recall']:.4f}  ← YOUR #1 METRIC")
    print(f"    F2-Score:              {metrics['f2_score']:.4f}  (recall-weighted)")
    print(f"    Dice Score:            {metrics['dice']:.4f}  (segmentation overlap)")
    print(f"    AUPRC:                 {metrics['auprc']:.4f}  (imbalance-honest)")
    
    print(f"\n  🥈 SECOND TIER (Secondary)")
    print(f"  {'─' * 40}")
    print(f"    Precision:             {metrics['precision']:.4f}")
    print(f"    ROC-AUC:               {metrics['roc_auc']:.4f}  (can be inflated)")
    print(f"    IoU (Jaccard):         {metrics['iou']:.4f}")
    
    print(f"\n  📊 Confusion Matrix")
    print(f"  {'─' * 40}")
    print(f"    TP: {metrics['tp']:,}  |  FP: {metrics['fp']:,}")
    print(f"    FN: {metrics['fn']:,}  |  TN: {metrics['tn']:,}")
    print(f"    FNR (miss rate): {metrics['fnr']:.4f}")
    
    print(f"\n{'═' * 60}\n")


# ═══════════════════════════════════════════════════════════════════════════════
# LOSS REGISTRY - Map names to loss functions
# ═══════════════════════════════════════════════════════════════════════════════

LOSS_REGISTRY = {
    # Pure Dice loss (class-imbalance robust, but linear ROC curves)
    "dice": dice_loss,
    
    # BCE + Dice loss (better probability calibration, smoother ROC curves)
    "bce+dice": bce_dice_loss,
    "bce_dice": bce_dice_loss,  # Alias
    
    # Tversky loss (recall-focused, penalizes False Negatives more)
    "tversky": tversky_loss,
    
    # BCE + Tversky loss (stable gradients + recall-focused)
    "bce+tversky": bce_tversky_loss,
    "bce_tversky": bce_tversky_loss,  # Alias
}


def get_loss_fn(name: str):
    """
    Get loss function by name from the registry.
    
    Args:
        name: Loss function name (e.g., "dice", "bce+dice")
        
    Returns:
        Loss function callable
        
    Raises:
        ValueError: If name not found in registry
        
    Example:
        loss_fn = get_loss_fn("bce+dice")
        loss = loss_fn(logits, targets)
    """
    name_lower = name.lower().strip()
    if name_lower not in LOSS_REGISTRY:
        available = list(LOSS_REGISTRY.keys())
        raise ValueError(f"Unknown loss '{name}'. Available: {available}")
    return LOSS_REGISTRY[name_lower]


def list_available_losses() -> dict:
    """
    List all available loss functions with descriptions.
    
    Returns:
        Dictionary mapping loss names to descriptions
    """
    descriptions = {
        "dice": "Pure Dice loss - class-imbalance robust, but linear ROC curves",
        "bce+dice": "BCE + Dice loss - better probability calibration, smoother ROC curves",
        "bce_dice": "(alias for bce+dice)",
        "tversky": "Tversky loss - recall-focused, penalizes FN 2.3x more than FP (α=0.3, β=0.7)",
        "bce+tversky": "BCE + Tversky loss - stable gradients + recall-focused",
        "bce_tversky": "(alias for bce+tversky)",
    }
    return descriptions


# ═══════════════════════════════════════════════════════════════════════════════
# MODULE INFO
# ═══════════════════════════════════════════════════════════════════════════════

__all__ = [
    # Loss functions
    'dice_loss',
    'bce_dice_loss',
    'tversky_loss',
    'bce_tversky_loss',
    
    # First Tier Metrics (Primary)
    'recall_metric',
    'f2_score_metric',
    'dice_coef_metric',
    'compute_auprc',
    'compute_pr_curve',
    
    # Second Tier Metrics (Secondary)
    'precision_metric',
    'iou_metric',
    'compute_roc_auc',
    'compute_roc_curve',
    
    # Comprehensive
    'compute_all_metrics',
    'print_metrics_summary',
    
    # Registry
    'LOSS_REGISTRY',
    'get_loss_fn',
    'list_available_losses',
]

if __name__ == "__main__":
    # Quick test
    print("=" * 60)
    print("UNIFIED LOSS FUNCTIONS & METRICS TEST")
    print("=" * 60)
    
    B, C, H, W = 4, 1, 64, 64
    logits = torch.randn(B, C, H, W)
    # Create imbalanced targets (like medical imaging with ~5% positive)
    targets = (torch.rand(B, C, H, W) > 0.95).float()
    
    # Test dice_loss
    d_loss = dice_loss(logits, targets)
    probs = torch.sigmoid(logits)
    metric = dice_coef_metric(probs, targets)
    
    # Test bce_dice_loss
    bd_loss = bce_dice_loss(logits, targets)
    
    # Test tversky_loss
    t_loss = tversky_loss(logits, targets, alpha=0.3, beta=0.7)
    t_loss_dice_equiv = tversky_loss(logits, targets, alpha=0.5, beta=0.5)
    
    # Test bce_tversky_loss
    bt_loss = bce_tversky_loss(logits, targets, alpha=0.3, beta=0.7)
    
    print(f"\nTest batch: {B} images, {H}x{W} pixels")
    print(f"Positive pixel ratio: {targets.mean().item():.2%} (imbalanced)")
    print(f"\n  dice_loss:        {d_loss.item():.6f}")
    print(f"  dice_coef_metric: {metric:.6f}")
    print(f"  Loss + Dice:      {d_loss.item() + metric:.6f} (should be ≈ 1.0)")
    print(f"\n  bce_dice_loss:    {bd_loss.item():.6f}")
    print(f"    (BCE component + Dice component)")
    
    print(f"\n" + "-" * 60)
    print("TVERSKY LOSS TESTS (Recall-Focused)")
    print("-" * 60)
    print(f"\n  tversky_loss (α=0.3, β=0.7): {t_loss.item():.6f}")
    print(f"    → Penalizes FN 2.3x more than FP")
    print(f"\n  tversky_loss (α=0.5, β=0.5): {t_loss_dice_equiv.item():.6f}")
    print(f"  dice_loss:                   {d_loss.item():.6f}")
    print(f"    → Should be similar (Tversky with α=β=0.5 ≈ Dice)")
    print(f"\n  bce_tversky_loss:            {bt_loss.item():.6f}")
    print(f"    (BCE + Tversky for stable gradients + recall focus)")
    
    # Test registry
    print("\n" + "=" * 60)
    print("LOSS REGISTRY TEST")
    print("=" * 60)
    print("\nAvailable losses:")
    for name, desc in list_available_losses().items():
        print(f"  '{name}': {desc}")
    
    # Test get_loss_fn
    print("\nTesting get_loss_fn:")
    for name in ["dice", "bce+dice", "tversky", "bce+tversky"]:
        fn = get_loss_fn(name)
        loss_val = fn(logits, targets)
        print(f"  get_loss_fn('{name}'): {loss_val.item():.6f}")
    
    # Test gradient flow (differentiability)
    print("\n" + "=" * 60)
    print("GRADIENT FLOW TEST (Differentiability)")
    print("=" * 60)
    logits_grad = torch.randn(B, C, H, W, requires_grad=True)
    loss = tversky_loss(logits_grad, targets)
    loss.backward()
    grad_norm = logits_grad.grad.norm().item()
    print(f"\n  Tversky loss backward pass: ✅")
    print(f"  Gradient norm: {grad_norm:.6f}")
    print(f"  Gradients flowing: {'✅ Yes' if grad_norm > 0 else '❌ No'}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # TEST EVALUATION METRICS
    # ═══════════════════════════════════════════════════════════════════════════
    print("\n" + "=" * 60)
    print("EVALUATION METRICS TEST")
    print("=" * 60)
    
    # Test individual metrics
    print(f"\n  First Tier (Primary) Metrics:")
    print(f"  {'─' * 40}")
    print(f"    recall_metric:    {recall_metric(probs, targets):.4f}")
    print(f"    f2_score_metric:  {f2_score_metric(probs, targets):.4f}")
    print(f"    dice_coef_metric: {dice_coef_metric(probs, targets):.4f}")
    print(f"    compute_auprc:    {compute_auprc(probs, targets):.4f}")
    
    print(f"\n  Second Tier (Secondary) Metrics:")
    print(f"  {'─' * 40}")
    print(f"    precision_metric: {precision_metric(probs, targets):.4f}")
    print(f"    iou_metric:       {iou_metric(probs, targets):.4f}")
    print(f"    compute_roc_auc:  {compute_roc_auc(probs, targets):.4f}")
    
    # Test curve computation
    print(f"\n  Curve Computation (for visualization):")
    print(f"  {'─' * 40}")
    precisions, recalls, pr_thresholds = compute_pr_curve(probs, targets)
    print(f"    PR Curve: {len(precisions)} points")
    print(f"      Precision range: [{precisions.min():.3f}, {precisions.max():.3f}]")
    print(f"      Recall range:    [{recalls.min():.3f}, {recalls.max():.3f}]")
    
    fprs, tprs, roc_thresholds = compute_roc_curve(probs, targets)
    print(f"    ROC Curve: {len(fprs)} points")
    print(f"      FPR range: [{fprs.min():.3f}, {fprs.max():.3f}]")
    print(f"      TPR range: [{tprs.min():.3f}, {tprs.max():.3f}]")
    
    # Test comprehensive metrics
    print("\n" + "=" * 60)
    print("COMPREHENSIVE METRICS (compute_all_metrics)")
    print("=" * 60)
    all_metrics = compute_all_metrics(probs, targets)
    print_metrics_summary(all_metrics, "Test Results (Imbalanced Data)")
    
    print(f"✅ All loss functions and metrics working!")
    
    # Print usage reminder
    print("\n" + "=" * 60)
    print("USAGE EXAMPLES")
    print("=" * 60)
    print("""
    # Import metrics
    from common.loss import (
        # First Tier (Primary)
        recall_metric, f2_score_metric, dice_coef_metric,
        compute_auprc, compute_pr_curve,
        
        # Second Tier (Secondary)
        precision_metric, iou_metric,
        compute_roc_auc, compute_roc_curve,
        
        # All at once
        compute_all_metrics, print_metrics_summary
    )
    
    # Compute individual metrics
    recall = recall_metric(probs, targets, threshold=0.5)
    auprc = compute_auprc(probs, targets)
    
    # Get curves for visualization
    precisions, recalls, thresholds = compute_pr_curve(probs, targets)
    plt.plot(recalls, precisions)  # PR Curve
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f'AUPRC = {auprc:.3f}')
    
    # Compute all metrics at once
    metrics = compute_all_metrics(probs, targets)
    print(f"Recall: {metrics['recall']:.4f}")
    print(f"AUPRC: {metrics['auprc']:.4f}")
    
    # Pretty print
    print_metrics_summary(metrics)
    """)
