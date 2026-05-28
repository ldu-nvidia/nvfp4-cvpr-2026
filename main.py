#!/usr/bin/env python3
"""
Unified NVFP4 QAT Experiment Runner
====================================

Single entry point for training, evaluating, and analyzing NVFP4 quantization-aware
training across CNN (U-Net), ViT, and Swin Transformer architectures.

Replaces the per-architecture Jupyter notebooks and run_multi_seed.py with a single
CLI-driven script that supports:
  - Patient-level data splitting (no leakage)
  - K-fold cross-validation (GroupKFold by patient)
  - Pluggable dataset registry
  - All architectures in one run
  - Multi-seed experiments
  - Post-hoc statistical analysis

Usage examples:
    # Single holdout run, all architectures, all sizes, seed 2026
    python main.py train --arch all --sizes all --dataset lgg --split patient-holdout

    # 5-fold CV on Swin + CNN at 4M scale
    python main.py train --arch swin cnn --sizes matched_4m --split patient-kfold --folds 5

    # Multi-seed for statistical validation
    python main.py train --arch swin cnn --sizes matched_4m --seeds 2026 2027 2028 2029 2030

    # Evaluate saved checkpoints
    python main.py evaluate --output-dir results/lgg/patient-holdout/single

    # Statistical analysis of results
    python main.py analyze --output-dir results/lgg/patient-holdout/single

    # Dry run to preview experiment plan
    python main.py train --arch all --sizes all --dry-run
"""

import sys
import os
import copy
import time
import json
import hashlib
import pickle
import argparse
import warnings
import glob as glob_module
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional, Callable
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", message=".*torch.cuda.amp.*")
# ═══════════════════════════════════════════════════════════════════════════════
# PATH SETUP
# ═══════════════════════════════════════════════════════════════════════════════
MAIN_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(MAIN_DIR))
sys.path.insert(0, str(MAIN_DIR / "swin"))
sys.path.insert(0, str(MAIN_DIR / "cnn"))
sys.path.insert(0, str(MAIN_DIR / "vit"))
sys.path.insert(0, str(MAIN_DIR / "common"))

# ═══════════════════════════════════════════════════════════════════════════════
# IMPORTS FROM EXISTING MODULES
# ═══════════════════════════════════════════════════════════════════════════════
import torch

from data_utils import set_seed, build_file_dataframe, align_images_and_masks, positive_negative_diagnosis
from loss import get_loss_fn, compute_all_metrics, compute_auprc, compute_pr_curve
from utils import (
    RECIPE_NAMES, RECIPE_COLORS,
    create_dataloaders, get_transforms,
    compute_config_hash, get_training_config, get_checkpoint_path,
    should_train_recipe, train_single_recipe, save_phase_summary,
    run_validation_inference, BrainMriDataset,
)

HAS_CNN = False
try:
    from cnn_qat import (
        ScalableCNNQAT, SUPPORTED_RECIPES as CNN_RECIPES, CNN_SCALES,
        _QAT_RUNTIME_AVAILABLE as _CNN_QAT_RUNTIME,
        _QAT_RUNTIME_ADVANCED_AVAILABLE as _CNN_QAT_RUNTIME_ADV,
    )
    HAS_CNN = True

    def create_cnn_qat(model_size="matched_500k", recipe_id=0, **kwargs):
        stages, width, _ = CNN_SCALES[model_size]
        return ScalableCNNQAT(
            num_stages=stages, width_multiplier=width, recipe_id=recipe_id,
            quantize_bottleneck_layer=False, **kwargs,
        )
except ImportError as e:
    print(f"CNN not available: {e}")

HAS_SWIN = False
try:
    from swin_qat import (
        create_swin_qat, SUPPORTED_RECIPES as SWIN_RECIPES, SWIN_CONFIGS,
        _QAT_RUNTIME_AVAILABLE as _SWIN_QAT_RUNTIME,
        _QAT_RUNTIME_ADVANCED_AVAILABLE as _SWIN_QAT_RUNTIME_ADV,
    )
    HAS_SWIN = True
except ImportError as e:
    print(f"Swin not available: {e}")

HAS_VIT = False
try:
    from vit_qat import (
        create_vit_qat, SUPPORTED_RECIPES as VIT_RECIPES,
        _QAT_RUNTIME_AVAILABLE as _VIT_QAT_RUNTIME,
        _QAT_RUNTIME_ADVANCED_AVAILABLE as _VIT_QAT_RUNTIME_ADV,
    )
    HAS_VIT = True
except ImportError as e:
    print(f"ViT not available: {e}")

# ═══════════════════════════════════════════════════════════════════════════════
# CONSTANTS
# ═══════════════════════════════════════════════════════════════════════════════
ALL_RECIPES = [0, 1, 2, 3, 4, 5, 6, 7]
ALL_SIZES = ["matched_500k", "matched_4m", "matched_15m"]
DEFAULT_SEEDS = [2026]
MULTI_SEEDS = [2026, 2027, 2028, 2029, 2030, 2031, 2032, 2033, 2034, 2035]

LOSS_TYPE = "tversky"
TVERSKY_ALPHA = 0.3
TVERSKY_BETA = 0.7
BATCH_SIZE = 32
NUM_EPOCHS = 100

EARLY_STOPPING_PATIENCE = 20
EARLY_STOPPING_METRIC = "loss"
EARLY_STOPPING_MIN_DELTA = 0.001
EARLY_STOPPING_WARMUP = 40

# Architecture-specific optimizer configs matching the original notebooks exactly.
# CNN uses Adamax (no weight decay); transformers use AdamW with regularization.
ARCH_CONFIGS = {
    "cnn": {
        "factory": "cnn",
        "optimizer": "adamax",
        "lr": 5e-4,
        "weight_decay": 0,
        "ckpt_prefix": "cnn",
        "available": lambda: HAS_CNN,
        "qat_runtime": lambda: HAS_CNN and _CNN_QAT_RUNTIME,
        "qat_runtime_adv": lambda: HAS_CNN and _CNN_QAT_RUNTIME_ADV,
    },
    "vit_adamw": {
        "factory": "vit",
        "optimizer": "adamw",
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "ckpt_prefix": "vit_adamw",
        "available": lambda: HAS_VIT,
        "qat_runtime": lambda: HAS_VIT and _VIT_QAT_RUNTIME,
        "qat_runtime_adv": lambda: HAS_VIT and _VIT_QAT_RUNTIME_ADV,
    },
    "vit_adamax": {
        "factory": "vit",
        "optimizer": "adamax",
        "lr": 5e-4,
        "weight_decay": 0,
        "ckpt_prefix": "vit_adamax",
        "available": lambda: HAS_VIT,
        "qat_runtime": lambda: HAS_VIT and _VIT_QAT_RUNTIME,
        "qat_runtime_adv": lambda: HAS_VIT and _VIT_QAT_RUNTIME_ADV,
    },
    "swin": {
        "factory": "swin",
        "optimizer": "adamw",
        "lr": 1e-4,
        "weight_decay": 1e-4,
        "ckpt_prefix": "swin",
        "available": lambda: HAS_SWIN,
        "qat_runtime": lambda: HAS_SWIN and _SWIN_QAT_RUNTIME,
        "qat_runtime_adv": lambda: HAS_SWIN and _SWIN_QAT_RUNTIME_ADV,
    },
}

ADVANCED_QAT_RECIPES = {4, 5, 6, 7, 8}


def _get_model_factory(arch_name: str) -> Callable:
    kind = ARCH_CONFIGS[arch_name]["factory"]
    if kind == "cnn":
        return create_cnn_qat
    elif kind == "vit":
        return create_vit_qat
    elif kind == "swin":
        return create_swin_qat
    raise ValueError(f"Unknown factory type: {kind}")


def can_run_recipe(arch_name: str, recipe_id: int) -> bool:
    if recipe_id == 0:
        return True
    cfg = ARCH_CONFIGS[arch_name]
    if not cfg["qat_runtime"]():
        return False
    if recipe_id in ADVANCED_QAT_RECIPES and not cfg["qat_runtime_adv"]():
        return False
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 2: DATASET REGISTRY (PLUGGABLE)
# ═══════════════════════════════════════════════════════════════════════════════

def load_lgg_dataset_patient_aware(data_path: str) -> pd.DataFrame:
    """
    Load LGG Brain MRI dataset with patient-level grouping.

    Returns DataFrame with columns: patient_id, image_path, mask_path, diagnosis
    where diagnosis is patient-level (1 if any slice is positive).
    """
    data_path = str(data_path)
    if not data_path.endswith("/"):
        data_path += "/"

    data_map = []
    for sub_dir_path in sorted(glob_module.glob(data_path + "*")):
        if os.path.isdir(sub_dir_path):
            dirname = sub_dir_path.split("/")[-1]
            for filename in os.listdir(sub_dir_path):
                image_path = sub_dir_path + "/" + filename
                data_map.extend([dirname, image_path])

    df_raw = pd.DataFrame({"dirname": data_map[::2], "path": data_map[1::2]})
    imgs, masks = align_images_and_masks(df_raw)

    patients = [Path(p).parent.name for p in imgs]
    df = pd.DataFrame({
        "patient_id": patients,
        "image_path": imgs,
        "mask_path": masks,
    })
    df["slice_diagnosis"] = df["mask_path"].apply(positive_negative_diagnosis)

    # Patient-level diagnosis: positive if ANY slice from that patient is positive
    patient_diag = df.groupby("patient_id")["slice_diagnosis"].max().reset_index()
    patient_diag.columns = ["patient_id", "diagnosis"]
    df = df.merge(patient_diag, on="patient_id")

    n_patients = df["patient_id"].nunique()
    n_pos = patient_diag["diagnosis"].sum()
    print(f"LGG dataset: {len(df)} slices from {n_patients} patients "
          f"({n_pos} positive, {n_patients - n_pos} negative)")
    return df


DATASET_REGISTRY: Dict[str, Callable] = {
    "lgg": load_lgg_dataset_patient_aware,
}


def register_dataset(name: str, loader_fn: Callable):
    """Register a new dataset loader. loader_fn(data_path) -> DataFrame with
    columns: patient_id, image_path, mask_path, diagnosis."""
    DATASET_REGISTRY[name] = loader_fn


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 3: PATIENT-LEVEL SPLITTING
# ═══════════════════════════════════════════════════════════════════════════════

def _patient_tumor_burden_bins(df: pd.DataFrame, n_bins: int = 3) -> pd.Series:
    """Bin patients by their positive-slice fraction so that stratification
    distributes light- and heavy-tumor-burden patients evenly across splits.
    Returns a Series indexed by patient_id with integer bin labels."""
    pos_frac = df.groupby("patient_id")["slice_diagnosis"].mean()
    bins = pd.qcut(pos_frac, q=n_bins, labels=False, duplicates="drop")
    return bins


def patient_holdout_split(
    df: pd.DataFrame,
    seed: int = 2026,
    test_size: float = 0.2,
    val_ratio: float = 0.5,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Patient-level stratified holdout split. No patient appears in more than
    one partition (train / val / test), eliminating data leakage.

    Stratification is by binned positive-slice fraction per patient, so
    patients with light and heavy tumor burden are balanced across splits.

    Returns a list with a single (train_df, val_df, test_df) tuple.
    """
    from sklearn.model_selection import train_test_split

    burden_bins = _patient_tumor_burden_bins(df)
    patient_ids = burden_bins.index.values
    strat_labels = burden_bins.values

    train_pids, temp_pids, train_strat, temp_strat = train_test_split(
        patient_ids, strat_labels, test_size=test_size, random_state=seed,
        stratify=strat_labels,
    )

    can_stratify_temp = len(np.unique(temp_strat)) > 1
    val_pids, test_pids = train_test_split(
        temp_pids, test_size=val_ratio, random_state=seed,
        stratify=temp_strat if can_stratify_temp else None,
    )

    train_df = df[df["patient_id"].isin(set(train_pids))].reset_index(drop=True)
    val_df = df[df["patient_id"].isin(set(val_pids))].reset_index(drop=True)
    test_df = df[df["patient_id"].isin(set(test_pids))].reset_index(drop=True)

    _print_split_stats("Holdout", train_df, val_df, test_df)
    return [(train_df, val_df, test_df)]


def patient_kfold_split(
    df: pd.DataFrame,
    n_folds: int = 5,
    seed: int = 2026,
    val_ratio: float = 0.5,
) -> List[Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]]:
    """
    Patient-level stratified k-fold CV. Within each fold the held-out
    patients are further split into val and test.

    Stratification is by binned positive-slice fraction per patient.
    """
    from sklearn.model_selection import StratifiedKFold, train_test_split

    burden_bins = _patient_tumor_burden_bins(df)
    patient_ids = burden_bins.index.values
    strat_labels = burden_bins.values

    skf = StratifiedKFold(n_splits=n_folds, shuffle=True, random_state=seed)

    folds = []
    for fold_idx, (train_pidx, heldout_pidx) in enumerate(
        skf.split(patient_ids, strat_labels)
    ):
        train_patients = set(patient_ids[train_pidx])
        heldout_pids = patient_ids[heldout_pidx]
        heldout_strat = strat_labels[heldout_pidx]

        can_stratify_held = len(np.unique(heldout_strat)) > 1
        val_pids, test_pids = train_test_split(
            heldout_pids, test_size=val_ratio, random_state=seed + fold_idx,
            stratify=heldout_strat if can_stratify_held else None,
        )

        train_df = df[df["patient_id"].isin(train_patients)].reset_index(drop=True)
        val_df = df[df["patient_id"].isin(set(val_pids))].reset_index(drop=True)
        test_df = df[df["patient_id"].isin(set(test_pids))].reset_index(drop=True)

        _print_split_stats(f"Fold {fold_idx}", train_df, val_df, test_df)
        folds.append((train_df, val_df, test_df))

    return folds


def _print_split_stats(label: str, train_df, val_df, test_df):
    def _stats(d):
        n = len(d)
        p = d["patient_id"].nunique()
        pos = d["slice_diagnosis"].sum() if "slice_diagnosis" in d.columns else "?"
        return f"{n} slices, {p} patients, {pos} pos"
    print(f"  {label}: train={_stats(train_df)} | val={_stats(val_df)} | test={_stats(test_df)}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 5: TRAINING ORCHESTRATION
# ═══════════════════════════════════════════════════════════════════════════════

def run_training(args):
    """Main training orchestration."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(args.gpu)}")

    # --- Load dataset ---
    if args.dataset not in DATASET_REGISTRY:
        print(f"Unknown dataset: {args.dataset}. Available: {list(DATASET_REGISTRY.keys())}")
        return
    df = DATASET_REGISTRY[args.dataset](args.data_path)

    # --- Generate splits ---
    if args.split == "patient-holdout":
        folds = patient_holdout_split(df, seed=args.split_seed)
    else:
        folds = patient_kfold_split(df, n_folds=args.folds, seed=args.split_seed)

    # --- Resolve arch / size / recipe lists ---
    archs = list(ARCH_CONFIGS.keys()) if "all" in args.arch else args.arch
    sizes = ALL_SIZES if "all" in args.sizes else args.sizes
    recipes = ALL_RECIPES if args.recipes is None or "all" in [str(r) for r in args.recipes] else [int(r) for r in args.recipes]

    # --- Experiment plan ---
    n_configs = len(folds) * len(archs) * len(sizes) * len(args.seeds) * len(recipes)
    print(f"\nExperiment plan: {len(folds)} fold(s) x {len(archs)} arch(s) x "
          f"{len(sizes)} size(s) x {len(args.seeds)} seed(s) x {len(recipes)} recipe(s) = {n_configs} configs")

    if args.dry_run:
        _print_dry_run(folds, archs, sizes, args.seeds, recipes, args)
        return

    # --- Run ---
    total_start = time.time()
    completed = 0
    skipped = 0
    failed = 0

    for fold_idx, (train_df, val_df, test_df) in enumerate(folds):
        fold_name = "single" if args.split == "patient-holdout" else f"fold_{fold_idx}"
        print(f"\n{'='*80}")
        print(f"FOLD: {fold_name}")
        print(f"{'='*80}")

        train_loader, val_loader, test_loader = create_dataloaders(
            train_df, val_df, test_df,
            batch_size=args.batch_size,
        )

        for arch_name in archs:
            cfg = ARCH_CONFIGS[arch_name]
            if not cfg["available"]():
                print(f"  Skipping {arch_name}: not available")
                continue

            factory = _get_model_factory(arch_name)

            for seed in args.seeds:
                for size in sizes:
                    # Shared baseline weights for fair comparison across recipes.
                    # "Same seed" = same RNG state for torch.manual_seed, producing
                    # identical initial parameters for a given (arch, size) pair.
                    # Different architectures have different parameter shapes, so
                    # the resulting weight tensors are NOT identical across archs.
                    set_seed(seed)
                    baseline_model = factory(model_size=size, recipe_id=0)
                    baseline_weights = copy.deepcopy(baseline_model.state_dict())
                    del baseline_model

                    ckpt_dir = Path(args.output_dir) / args.dataset / args.split / fold_name / cfg["ckpt_prefix"] / f"seed_{seed}"
                    ckpt_dir.mkdir(parents=True, exist_ok=True)

                    config_dict = get_training_config(
                        model_size=size,
                        num_epochs=args.epochs,
                        batch_size=args.batch_size,
                        learning_rate=cfg["lr"],
                        weight_decay=cfg["weight_decay"],
                        loss_type=LOSS_TYPE,
                        seed=seed,
                    )
                    config_hash = compute_config_hash(config_dict)

                    for recipe_id in recipes:
                        if not can_run_recipe(arch_name, recipe_id):
                            print(f"  [{arch_name}] Recipe {recipe_id} ({RECIPE_NAMES.get(recipe_id, '?')}): "
                                  f"skipped (NVFP4 QAT runtime not available)")
                            skipped += 1
                            continue

                        needs_train, reason, existing_ckpt = should_train_recipe(
                            recipe_id, LOSS_TYPE, config_hash, ckpt_dir,
                            size, args.epochs, force_retrain=args.force_retrain,
                        )

                        recipe_name = RECIPE_NAMES.get(recipe_id, f"R{recipe_id}")
                        tag = f"[{arch_name}|{size}|seed{seed}|{fold_name}]"

                        if not needs_train:
                            print(f"  {tag} Recipe {recipe_id} ({recipe_name}): skip ({reason})")
                            skipped += 1

                            _save_metrics_from_checkpoint(
                                existing_ckpt, ckpt_dir, recipe_id, recipe_name,
                                arch_name, size, seed, fold_name, args, config_hash,
                            )
                            continue

                        print(f"\n  {tag} Recipe {recipe_id} ({recipe_name}): TRAINING ({reason})")
                        set_seed(seed)

                        try:
                            results = train_single_recipe(
                                recipe_id=recipe_id,
                                loss_type=LOSS_TYPE,
                                model_size=size,
                                num_epochs=args.epochs,
                                train_dataloader=train_loader,
                                val_dataloader=val_loader,
                                device=device,
                                create_model_fn=factory,
                                baseline_weights=baseline_weights,
                                ckpt_dir=ckpt_dir,
                                config_hash=config_hash,
                                config_dict=config_dict,
                                learning_rate=cfg["lr"],
                                weight_decay=cfg["weight_decay"],
                                optimizer_type=cfg["optimizer"],
                                early_stopping_patience=EARLY_STOPPING_PATIENCE,
                                early_stopping_metric=EARLY_STOPPING_METRIC,
                                early_stopping_min_delta=EARLY_STOPPING_MIN_DELTA,
                                early_stopping_warmup=EARLY_STOPPING_WARMUP,
                            )

                            _save_metrics_json(
                                results, ckpt_dir, recipe_id, recipe_name,
                                arch_name, size, seed, fold_name, args, config_hash,
                            )

                            if args.eval_test:
                                _evaluate_on_test(
                                    factory, size, recipe_id, baseline_weights,
                                    ckpt_dir, config_hash, test_loader, device,
                                    arch_name, seed, fold_name, args,
                                )

                            completed += 1
                            print(f"  {tag} Recipe {recipe_id}: done "
                                  f"(AUPRC={results.get('auprc', 0):.4f}, "
                                  f"Dice={results.get('best_dice', 0):.4f}, "
                                  f"{results.get('actual_epochs', '?')} epochs)")

                        except Exception as e:
                            print(f"  {tag} Recipe {recipe_id}: FAILED - {e}")
                            failed += 1

    elapsed = time.time() - total_start
    print(f"\n{'='*80}")
    print(f"TRAINING COMPLETE: {completed} trained, {skipped} skipped, {failed} failed "
          f"in {elapsed/60:.1f} min")
    print(f"Output: {args.output_dir}")
    print(f"{'='*80}")


def _save_metrics_json(results, ckpt_dir, recipe_id, recipe_name, arch_name,
                       size, seed, fold_name, args, config_hash):
    """Save a standalone metrics JSON for the analysis pipeline."""
    metrics_dir = ckpt_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)

    metrics = {
        "recipe_id": recipe_id,
        "recipe_name": recipe_name,
        "arch": arch_name,
        "model_size": size,
        "seed": seed,
        "fold": fold_name,
        "dataset": args.dataset,
        "split": args.split,
        "config_hash": config_hash,
        "auprc": results.get("auprc", 0),
        "dice": results.get("best_dice", 0),
        "recall": results.get("recall", 0) if "recall" in results else results.get("best_recall", 0),
        "f2": results.get("f2_score", 0) if "f2_score" in results else results.get("best_f2", 0),
        "precision": results.get("precision", 0),
        "iou": results.get("iou", 0),
        "auc": results.get("auc", 0),
        "best_epoch": results.get("best_epoch", 0),
        "actual_epochs": results.get("actual_epochs", 0),
        "early_stopped": results.get("early_stopped", False),
        "training_time": results.get("time", 0),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
    }

    fname = f"metrics_r{recipe_id}_{recipe_name.replace('+', '')}_{size}_{args.epochs}ep_{config_hash}.json"
    with open(metrics_dir / fname, "w") as f:
        json.dump(metrics, f, indent=2)


def _save_metrics_from_checkpoint(ckpt_path, ckpt_dir, recipe_id, recipe_name,
                                  arch_name, size, seed, fold_name, args, config_hash):
    """Extract metrics from an existing checkpoint and save as JSON if not already present."""
    metrics_dir = ckpt_dir / "metrics"
    metrics_dir.mkdir(parents=True, exist_ok=True)
    fname = f"metrics_r{recipe_id}_{recipe_name.replace('+', '')}_{size}_{args.epochs}ep_{config_hash}.json"
    if (metrics_dir / fname).exists():
        return

    if ckpt_path is None or not Path(ckpt_path).exists():
        return

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        metrics = {
            "recipe_id": recipe_id,
            "recipe_name": recipe_name,
            "arch": arch_name,
            "model_size": size,
            "seed": seed,
            "fold": fold_name,
            "dataset": args.dataset,
            "split": args.split,
            "config_hash": config_hash,
            "auprc": 0,
            "dice": ckpt.get("best_dice", 0),
            "recall": ckpt.get("best_recall", 0),
            "f2": ckpt.get("best_f2", 0),
            "best_epoch": ckpt.get("best_epoch", 0),
            "actual_epochs": ckpt.get("actual_epochs", 0),
            "early_stopped": ckpt.get("early_stopped", False),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "source": "extracted_from_checkpoint",
        }
        with open(metrics_dir / fname, "w") as f:
            json.dump(metrics, f, indent=2)
    except Exception:
        pass


def _evaluate_on_test(factory, size, recipe_id, baseline_weights, ckpt_dir,
                      config_hash, test_loader, device, arch_name, seed,
                      fold_name, args):
    """Load best checkpoint and evaluate on the held-out test set."""
    ckpt_path = get_checkpoint_path(
        ckpt_dir, recipe_id, size, args.epochs, LOSS_TYPE, config_hash,
    )
    if not ckpt_path.exists():
        return

    try:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("state_dict", ckpt)

        model = factory(model_size=size, recipe_id=recipe_id).to(device)
        model.load_state_dict(state_dict, strict=False)

        test_results = run_validation_inference(model, test_loader, device)

        metrics_dir = ckpt_dir / "metrics"
        metrics_dir.mkdir(parents=True, exist_ok=True)
        recipe_name = RECIPE_NAMES.get(recipe_id, f"R{recipe_id}")
        fname = f"test_metrics_r{recipe_id}_{recipe_name.replace('+', '')}_{size}_{args.epochs}ep_{config_hash}.json"

        test_metrics = {
            "recipe_id": recipe_id,
            "recipe_name": recipe_name,
            "arch": arch_name,
            "model_size": size,
            "seed": seed,
            "fold": fold_name,
            "dataset": args.dataset,
            "split": args.split,
            "eval_set": "test",
            "auprc": test_results.get("auprc", 0),
            "dice": test_results.get("dice", 0),
            "recall": test_results.get("recall", 0),
            "f2": test_results.get("f2_score", 0),
            "precision": test_results.get("precision", 0),
            "auc": test_results.get("auc", 0),
            "iou": test_results.get("iou", 0),
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        with open(metrics_dir / fname, "w") as f:
            json.dump(test_metrics, f, indent=2)

        tag = f"[{arch_name}|{size}|seed{seed}|{fold_name}]"
        print(f"    {tag} TEST: AUPRC={test_metrics['auprc']:.4f}, "
              f"Dice={test_metrics['dice']:.4f}, Recall={test_metrics['recall']:.4f}")

        del model
        torch.cuda.empty_cache()

    except Exception as e:
        print(f"    Test evaluation failed: {e}")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 6: EVALUATION MODE
# ═══════════════════════════════════════════════════════════════════════════════

def run_evaluation(args):
    """Re-evaluate saved checkpoints (e.g. on a different split or test set)."""
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)

    if not output_dir.exists():
        print(f"Output dir not found: {output_dir}")
        return

    if args.dataset not in DATASET_REGISTRY:
        print(f"Unknown dataset: {args.dataset}")
        return
    df = DATASET_REGISTRY[args.dataset](args.data_path)

    if args.split == "patient-holdout":
        folds = patient_holdout_split(df, seed=args.split_seed)
    else:
        folds = patient_kfold_split(df, n_folds=args.folds, seed=args.split_seed)

    archs = list(ARCH_CONFIGS.keys()) if "all" in args.arch else args.arch

    evaluated = 0
    for fold_idx, (train_df, val_df, test_df) in enumerate(folds):
        fold_name = "single" if args.split == "patient-holdout" else f"fold_{fold_idx}"
        _, _, test_loader = create_dataloaders(train_df, val_df, test_df, batch_size=args.batch_size)

        for arch_name in archs:
            cfg = ARCH_CONFIGS[arch_name]
            if not cfg["available"]():
                continue
            factory = _get_model_factory(arch_name)

            for seed in args.seeds:
                ckpt_base = output_dir / args.dataset / args.split / fold_name / cfg["ckpt_prefix"] / f"seed_{seed}"
                if not ckpt_base.exists():
                    continue

                for ckpt_file in sorted(ckpt_base.glob("recipe_*.pt")):
                    try:
                        ckpt = torch.load(ckpt_file, map_location="cpu", weights_only=False)
                        rid = ckpt.get("recipe_id", 0)
                        msize = ckpt.get("model_size", "matched_4m")
                        state_dict = ckpt.get("state_dict", ckpt)

                        model = factory(model_size=msize, recipe_id=rid).to(device)
                        model.load_state_dict(state_dict, strict=False)
                        test_results = run_validation_inference(model, test_loader, device)

                        rname = RECIPE_NAMES.get(rid, f"R{rid}")
                        print(f"  [{arch_name}|{msize}|seed{seed}|{fold_name}] "
                              f"Recipe {rid} ({rname}): "
                              f"AUPRC={test_results.get('auprc', 0):.4f}")

                        del model
                        torch.cuda.empty_cache()
                        evaluated += 1
                    except Exception as e:
                        print(f"  Failed to evaluate {ckpt_file.name}: {e}")

    print(f"\nEvaluated {evaluated} checkpoints.")


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 7: ANALYSIS MODE
# ═══════════════════════════════════════════════════════════════════════════════

def run_analysis(args):
    """Aggregate metrics JSONs and produce summary statistics + pairwise tests."""
    from scipy import stats as scipy_stats

    output_dir = Path(args.output_dir)
    all_metrics = []

    for metrics_file in sorted(output_dir.rglob("metrics/metrics_r*.json")):
        try:
            with open(metrics_file) as f:
                m = json.load(f)
            all_metrics.append(m)
        except Exception:
            pass

    if not all_metrics:
        print(f"No metrics files found under {output_dir}")
        return

    df = pd.DataFrame(all_metrics)
    print(f"\nLoaded {len(df)} metric records from {output_dir}")
    print(f"  Architectures: {sorted(df['arch'].unique())}")
    print(f"  Sizes: {sorted(df['model_size'].unique())}")
    print(f"  Recipes: {sorted(df['recipe_id'].unique())}")
    print(f"  Seeds: {sorted(df['seed'].unique())}")
    if "fold" in df.columns:
        print(f"  Folds: {sorted(df['fold'].unique())}")

    # --- Raw results CSV ---
    analysis_dir = output_dir / "analysis"
    analysis_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(analysis_dir / "raw_results.csv", index=False)
    print(f"\nSaved: {analysis_dir / 'raw_results.csv'}")

    # --- Summary statistics ---
    primary_metric = "auprc"
    group_cols = ["arch", "model_size", "recipe_id", "recipe_name"]
    if "fold" in df.columns and df["fold"].nunique() > 1:
        group_cols.append("fold")

    summary_rows = []
    for keys, group in df.groupby(["arch", "model_size", "recipe_id"]):
        arch, msize, rid = keys
        vals = group[primary_metric].dropna().values
        n = len(vals)
        if n == 0:
            continue

        mean = np.mean(vals)
        std = np.std(vals, ddof=1) if n > 1 else 0
        ci95 = scipy_stats.t.ppf(0.975, n - 1) * std / np.sqrt(n) if n > 1 else 0
        rname = group["recipe_name"].iloc[0] if "recipe_name" in group.columns else RECIPE_NAMES.get(rid, f"R{rid}")

        summary_rows.append({
            "arch": arch,
            "model_size": msize,
            "recipe_id": rid,
            "recipe_name": rname,
            "n": n,
            f"{primary_metric}_mean": round(mean, 6),
            f"{primary_metric}_std": round(std, 6),
            f"{primary_metric}_ci95": round(ci95, 6),
            f"{primary_metric}_min": round(np.min(vals), 6),
            f"{primary_metric}_max": round(np.max(vals), 6),
        })

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(analysis_dir / "summary.csv", index=False)
    print(f"Saved: {analysis_dir / 'summary.csv'}")

    _print_summary_table(summary_df, primary_metric)

    # --- Pairwise tests ---
    if df["seed"].nunique() > 1 or (df.get("fold") is not None and df["fold"].nunique() > 1):
        pairwise_rows = _compute_pairwise_tests(df, primary_metric)
        if pairwise_rows:
            pw_df = pd.DataFrame(pairwise_rows)
            pw_df.to_csv(analysis_dir / "pairwise_tests.csv", index=False)
            print(f"Saved: {analysis_dir / 'pairwise_tests.csv'}")
    else:
        print("\nPairwise tests skipped (need multiple seeds or folds).")

    # --- Test metrics ---
    test_metrics = []
    for f in sorted(output_dir.rglob("metrics/test_metrics_r*.json")):
        try:
            with open(f) as fh:
                test_metrics.append(json.load(fh))
        except Exception:
            pass

    if test_metrics:
        test_df = pd.DataFrame(test_metrics)
        test_df.to_csv(analysis_dir / "test_results.csv", index=False)
        print(f"Saved: {analysis_dir / 'test_results.csv'} ({len(test_df)} records)")


def _print_summary_table(summary_df, metric):
    col_mean = f"{metric}_mean"
    col_std = f"{metric}_std"
    col_ci = f"{metric}_ci95"

    print(f"\n{'='*90}")
    print(f"{'Arch':<12} {'Size':<14} {'Recipe':<14} {'N':>3}  "
          f"{metric.upper()+' Mean':>12} {'Std':>8} {'95% CI':>8}")
    print(f"{'-'*90}")

    for _, row in summary_df.sort_values(["arch", "model_size", "recipe_id"]).iterrows():
        print(f"{row['arch']:<12} {row['model_size']:<14} "
              f"{row['recipe_name']:<14} {row['n']:>3}  "
              f"{row[col_mean]:>12.4f} {row[col_std]:>8.4f} {row[col_ci]:>8.4f}")
    print(f"{'='*90}")


def _compute_pairwise_tests(df, metric):
    """Wilcoxon signed-rank tests between recipe pairs, per (arch, size)."""
    from scipy.stats import wilcoxon
    from itertools import combinations

    rows = []
    for (arch, msize), group in df.groupby(["arch", "model_size"]):
        recipe_ids = sorted(group["recipe_id"].unique())
        if len(recipe_ids) < 2:
            continue

        pivot_col = "seed" if group["seed"].nunique() > 1 else "fold"

        for r1, r2 in combinations(recipe_ids, 2):
            d1 = group[group["recipe_id"] == r1].set_index(pivot_col)[metric]
            d2 = group[group["recipe_id"] == r2].set_index(pivot_col)[metric]
            common = d1.index.intersection(d2.index)
            if len(common) < 3:
                continue

            v1 = d1.loc[common].values
            v2 = d2.loc[common].values
            diff = v1 - v2

            if np.all(diff == 0):
                p = 1.0
                stat = 0
            else:
                stat, p = wilcoxon(v1, v2)

            pooled_std = np.std(diff, ddof=1)
            cohens_d = np.mean(diff) / pooled_std if pooled_std > 0 else 0

            rows.append({
                "arch": arch,
                "model_size": msize,
                "recipe_a": r1,
                "recipe_b": r2,
                "name_a": RECIPE_NAMES.get(r1, str(r1)),
                "name_b": RECIPE_NAMES.get(r2, str(r2)),
                "n_pairs": len(common),
                "mean_diff": round(np.mean(diff), 6),
                "wilcoxon_stat": round(stat, 4),
                "p_value": round(p, 6),
                "cohens_d": round(cohens_d, 4),
            })

    # Holm-Bonferroni correction
    if rows:
        pvals = [r["p_value"] for r in rows]
        corrected = _holm_bonferroni(pvals)
        for r, cp in zip(rows, corrected):
            r["p_corrected"] = round(cp, 6)
            r["significant"] = cp < 0.05

    return rows


def _holm_bonferroni(pvals):
    n = len(pvals)
    order = np.argsort(pvals)
    corrected = np.ones(n)
    for rank, idx in enumerate(order):
        corrected[idx] = min(pvals[idx] * (n - rank), 1.0)
    cummax = corrected[order].copy()
    for i in range(1, len(cummax)):
        cummax[i] = max(cummax[i], cummax[i - 1])
    corrected[order] = cummax
    return corrected.tolist()


# ═══════════════════════════════════════════════════════════════════════════════
# SECTION 8: UTILITIES
# ═══════════════════════════════════════════════════════════════════════════════

def _print_dry_run(folds, archs, sizes, seeds, recipes, args):
    print(f"\n{'='*80}")
    print("DRY RUN -- Experiment Plan")
    print(f"{'='*80}")
    print(f"  Dataset:    {args.dataset}")
    print(f"  Split:      {args.split}" + (f" ({args.folds} folds)" if args.split == "patient-kfold" else ""))
    print(f"  Folds:      {len(folds)}")
    print(f"  Archs:      {archs}")
    print(f"  Sizes:      {sizes}")
    print(f"  Seeds:      {seeds}")
    print(f"  Recipes:    {recipes}")
    print(f"  Epochs:     {args.epochs}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Output:     {args.output_dir}")

    total = 0
    runnable = 0
    for fold_idx in range(len(folds)):
        fold_name = "single" if args.split == "patient-holdout" else f"fold_{fold_idx}"
        for arch_name in archs:
            for seed in seeds:
                for size in sizes:
                    for recipe_id in recipes:
                        total += 1
                        ok = can_run_recipe(arch_name, recipe_id)
                        status = "RUN" if ok else "SKIP (no NVFP4 QAT runtime)"
                        rname = RECIPE_NAMES.get(recipe_id, f"R{recipe_id}")
                        if ok:
                            runnable += 1
                        print(f"    {fold_name} | {arch_name:<12} | {size:<14} | seed {seed} | "
                              f"recipe {recipe_id:>5} ({rname:<12}) | {status}")

    print(f"\nTotal: {total} configs, {runnable} runnable, {total - runnable} skipped")
    est_hours = runnable * 0.15
    print(f"Estimated time: ~{est_hours:.1f} hours (assuming ~9 min/config on RTX 6000)")


def save_run_manifest(args, output_dir: Path):
    """Save a JSON manifest of the run configuration for reproducibility."""
    manifest = {
        "command": " ".join(sys.argv),
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "args": vars(args),
        "gpu": torch.cuda.get_device_name(args.gpu) if torch.cuda.is_available() else "cpu",
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda if torch.cuda.is_available() else None,
    }
    manifest_path = output_dir / "run_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Manifest saved: {manifest_path}")


# ═══════════════════════════════════════════════════════════════════════════════
# CLI
# ═══════════════════════════════════════════════════════════════════════════════

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Unified NVFP4 QAT Experiment Runner",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    sub = p.add_subparsers(dest="mode", required=True)

    # --- train ---
    t = sub.add_parser("train", help="Train models")
    t.add_argument("--arch", nargs="+", default=["all"],
                    choices=["cnn", "vit_adamw", "vit_adamax", "swin", "all"],
                    help="Architecture(s) to train")
    t.add_argument("--sizes", nargs="+", default=["all"],
                    choices=ALL_SIZES + ["all"],
                    help="Model size(s)")
    t.add_argument("--dataset", default="lgg",
                    help=f"Dataset name (registered: {list(DATASET_REGISTRY.keys())})")
    t.add_argument("--data-path", default=str(MAIN_DIR / "lgg-mri-segmentation" / "kaggle_3m"),
                    help="Path to dataset root")
    t.add_argument("--split", default="patient-holdout",
                    choices=["patient-holdout", "patient-kfold"],
                    help="Split strategy")
    t.add_argument("--split-seed", type=int, default=2026,
                    help="Seed for data splitting (fixed across training seeds)")
    t.add_argument("--folds", type=int, default=5,
                    help="Number of folds for patient-kfold")
    t.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS,
                    help="Training seeds")
    t.add_argument("--recipes", nargs="+", default=["all"],
                    help="Recipe IDs or 'all'")
    t.add_argument("--epochs", type=int, default=NUM_EPOCHS)
    t.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    t.add_argument("--output-dir", default=str(MAIN_DIR / "results"),
                    help="Base output directory")
    t.add_argument("--force-retrain", action="store_true")
    t.add_argument("--dry-run", action="store_true")
    t.add_argument("--gpu", type=int, default=0)
    t.add_argument("--eval-test", action="store_true", default=True,
                    help="Evaluate on test set after training (default: True)")
    t.add_argument("--no-eval-test", dest="eval_test", action="store_false")

    # --- evaluate ---
    e = sub.add_parser("evaluate", help="Re-evaluate saved checkpoints")
    e.add_argument("--arch", nargs="+", default=["all"],
                    choices=["cnn", "vit_adamw", "vit_adamax", "swin", "all"])
    e.add_argument("--dataset", default="lgg")
    e.add_argument("--data-path", default=str(MAIN_DIR / "lgg-mri-segmentation" / "kaggle_3m"))
    e.add_argument("--split", default="patient-holdout",
                    choices=["patient-holdout", "patient-kfold"])
    e.add_argument("--split-seed", type=int, default=2026)
    e.add_argument("--folds", type=int, default=5)
    e.add_argument("--seeds", nargs="+", type=int, default=DEFAULT_SEEDS)
    e.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    e.add_argument("--output-dir", required=True)
    e.add_argument("--gpu", type=int, default=0)

    # --- analyze ---
    a = sub.add_parser("analyze", help="Statistical analysis of results")
    a.add_argument("--output-dir", required=True,
                    help="Directory containing metrics JSONs")

    return p


def main():
    parser = build_parser()
    args = parser.parse_args()

    print("=" * 80)
    print("NVFP4 QAT Unified Experiment Runner")
    print("=" * 80)
    print(f"Mode: {args.mode}")
    print(f"NVFP4 QAT runtime available: CNN={HAS_CNN}, Swin={HAS_SWIN}, ViT={HAS_VIT}")
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    if args.mode == "train":
        out = Path(args.output_dir)
        out.mkdir(parents=True, exist_ok=True)
        save_run_manifest(args, out)
        run_training(args)
    elif args.mode == "evaluate":
        run_evaluation(args)
    elif args.mode == "analyze":
        run_analysis(args)


if __name__ == "__main__":
    main()
