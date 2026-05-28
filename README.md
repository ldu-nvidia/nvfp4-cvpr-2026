# Not All NVFP4 QAT Recipes Are Equal

**How Architecture and Scale Shape Model Quality for Anomaly Segmentation**

*Zijian Du, Oleg Rybakov — NVIDIA*

**CVPR 2026 Workshop on Visual Anomaly and Novelty Detection (VAND)**

---

## Overview

This repository contains the code for our controlled study of NVFP4 quantization-aware training (QAT) across three vision architectures (CNN, ViT, Swin Transformer), three matched parameter scales (530K–13.7M), and eight QAT recipes on a recall-critical brain tumor segmentation task.

**Key findings:**
- Architecture choice determines FP4 quantization robustness — Swin is the most robust
- Advanced QAT recipes prevent two distinct failure modes: softmax attention discretization (small transformers) and gradient noise amplification (large CNNs)
- 4M parameters is the practical sweet spot; Swin is robust to recipe choice at this scale

## Note on NVFP4 QAT Runtime

NVFP4 quantization-aware training requires a compatible NVFP4 training environment. This public release provides the experiment orchestration, model definitions, evaluation metrics, and analysis scripts used for the paper. Vendor/runtime-specific quantization integration is intentionally omitted from the repository.

The provided code supports baseline training and can be adapted to other public quantization frameworks. Non-baseline NVFP4 QAT recipes are retained as experiment metadata and will report a clear runtime-compatibility message in this release.

## Repository Structure

```
nvfp4-cvpr-2026/
├── main.py                  # Unified experiment runner (train / evaluate / analyze)
├── common/
│   ├── loss.py              # Loss functions, AUPRC, evaluation metrics
│   ├── utils.py             # Training loop, data loading, inference
│   └── data_utils.py        # Dataset utilities
├── cnn/
│   └── cnn_qat.py           # Scalable encoder-decoder CNN with NVFP4 quantization
├── swin/
│   └── swin_qat.py          # Swin Transformer with NVFP4 quantization
├── vit/
│   └── vit_qat.py           # Vision Transformer with NVFP4 quantization
├── scripts/
│   ├── analyze_multi_seed.py           # Multi-seed statistical analysis
│   ├── generate_cv_figure.py           # Figure 7: CV robustness
│   └── plot_normalized_recipe_impact.py # Figure 6: QAT sensitivity
├── requirements.txt
└── README.md
```

## Dataset

We use the [LGG Brain MRI Segmentation](https://www.kaggle.com/datasets/mateuszbuda/lgg-mri-segmentation) dataset from Kaggle.

1. Download from Kaggle and extract to `lgg-mri-segmentation/kaggle_3m/`
2. The dataset contains 3,929 paired MRI slices and binary tumor masks from 110 patients

## Usage

### Training

Train all architectures at 4M scale with patient-level holdout split:

```bash
python main.py train \
  --arch all --sizes matched_4m \
  --dataset lgg --split patient-holdout \
  --seeds 2026 --recipes all --epochs 100
```

Train Swin and CNN with 5-fold patient-level cross-validation:

```bash
python main.py train \
  --arch swin cnn --sizes matched_4m \
  --dataset lgg --split patient-kfold --folds 5 \
  --seeds 2026 --recipes 0 5 --epochs 100
```

### Evaluation

```bash
python main.py evaluate --arch all --sizes all --dataset lgg
```

### Analysis

```bash
python main.py analyze --output-dir results/
```

### Key Arguments

| Argument | Options | Default | Description |
|----------|---------|---------|-------------|
| `mode` | `train`, `evaluate`, `analyze` | — | Operation mode |
| `--arch` | `cnn`, `vit_adamw`, `vit_adamax`, `swin`, `all` | `all` | Architecture(s) |
| `--sizes` | `matched_500k`, `matched_4m`, `matched_15m`, `all` | `all` | Model scale(s) |
| `--split` | `patient-holdout`, `patient-kfold` | `patient-holdout` | Data split strategy |
| `--folds` | int | `5` | Number of CV folds |
| `--seeds` | list of ints | `[2026]` | Random seeds |
| `--recipes` | list of ints or `all` | `all` | QAT recipe IDs |
| `--epochs` | int | `100` | Training epochs |
| `--gpu` | int | `0` | GPU device index |

### QAT Recipes

| ID | Recipe | 2D | RHT | SR | Description |
|----|--------|----|-----|-----|-------------|
| 0 | Baseline | — | — | — | FP16/BF16 (no quantization) |
| 1 | NVFP4 Full | | | | Base quantization, all passes |
| 2 | Fwd-Only | | | | Forward pass quantized only |
| 3 | Chain Rule | | | | Reuses forward quantized values in backward |
| 4 | 2D+RHT | ✓ | ✓ | | 2D block-scaling + RHT |
| 5 | 2D+RHT+SR | ✓ | ✓ | ✓ | All three advanced techniques |
| 6 | SR Only | | | ✓ | Stochastic rounding on gradients |
| 7 | Fwd+RHT | | ✓ | | Forward-only with RHT |

## Hardware

Experiments were run on two NVIDIA RTX PRO 6000 (Blackwell) GPUs with 96 GB memory each.

## Citation

```bibtex
@inproceedings{du2026nvfp4,
  title={Not All {NVFP4} {QAT} Recipes Are Equal: How Architecture and Scale Shape Model Quality for Anomaly Segmentation},
  author={Du, Zijian and Rybakov, Oleg},
  booktitle={CVPR 2026 Workshop on Visual Anomaly and Novelty Detection (VAND)},
  year={2026}
}
```

## License

Copyright (c) 2026 NVIDIA Corporation. All rights reserved.
