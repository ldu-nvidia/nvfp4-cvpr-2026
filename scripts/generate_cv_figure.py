"""
Generate Figure 6: 5-fold patient-level CV robustness figure.
Grouped by recipe (Baseline, 2D+RHT+SR) with Swin vs CNN side-by-side.
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

SWIN_COLOR = '#1565C0'
CNN_COLOR = '#E65100'

swin_base = np.array([0.7038, 0.8509, 0.7241, 0.8880, 0.7956])
swin_qat  = np.array([0.7671, 0.8347, 0.7055, 0.8855, 0.8047])
cnn_base  = np.array([0.6973, 0.8293, 0.7211, 0.6855, 0.8013])
cnn_qat   = np.array([0.5800, 0.7998, 0.7554, 0.6871, 0.8039])

groups = [
    (swin_base, cnn_base, 'Baseline (FP16)'),
    (swin_qat,  cnn_qat,  '2D+RHT+SR (FP4)'),
]

fig, ax = plt.subplots(figsize=(5.5, 4.2))

group_centers = [0.7, 2.1]
bar_w = 0.5
gap = 0.55

for gi, (swin_vals, cnn_vals, group_name) in enumerate(groups):
    cx = group_centers[gi]
    swin_pos = cx - gap / 2
    cnn_pos = cx + gap / 2

    for pos, vals, color in [(swin_pos, swin_vals, SWIN_COLOR),
                              (cnn_pos, cnn_vals, CNN_COLOR)]:
        mean = np.mean(vals)
        std = np.std(vals, ddof=1)

        ax.bar(pos, mean, width=bar_w, color=color, alpha=0.75,
               edgecolor=color, linewidth=1.8, zorder=2)

        ax.errorbar(pos, mean, yerr=std, fmt='none', color='#333333',
                    capsize=6, capthick=2, linewidth=2, zorder=3)

        ax.scatter([pos] * 5, vals, color='#333333', s=22, zorder=4,
                   edgecolors='white', linewidths=0.6, alpha=0.9)

        ax.text(pos, mean + std + 0.025, f'{mean:.3f}',
                ha='center', va='bottom', fontsize=11, fontweight='bold',
                color=color)

    ax.text(cx, -0.015, group_name, ha='center', va='top',
            fontsize=12, fontweight='bold',
            transform=ax.get_xaxis_transform())

ax.set_xlim(-0.1, 2.95)
ax.set_ylim(0.50, 0.98)
ax.set_xticks([])
ax.set_ylabel('Test AUPRC', fontsize=13, fontweight='bold')
ax.set_title('5-Fold Patient-Level Cross-Validation (4M Scale)',
             fontsize=14, fontweight='bold', pad=12)

ax.grid(axis='y', alpha=0.3, linestyle='--', zorder=0)
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

legend_elements = [
    mpatches.Patch(facecolor=SWIN_COLOR, alpha=0.75, edgecolor=SWIN_COLOR,
                   linewidth=1.5, label='Swin Transformer'),
    mpatches.Patch(facecolor=CNN_COLOR, alpha=0.75, edgecolor=CNN_COLOR,
                   linewidth=1.5, label='CNN'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#333333',
               markersize=7, label='Individual folds'),
]
ax.legend(handles=legend_elements, loc='upper center', fontsize=10,
          framealpha=0.9, edgecolor='#cccccc', ncol=3,
          bbox_to_anchor=(0.5, 1.0), handlelength=1.2, columnspacing=1.0)

plt.tight_layout()
out_path = os.path.join(os.path.dirname(__file__),
                        '..', 'cvpr26', 'figures', 'cv_robustness_5fold.png')
plt.savefig(out_path, dpi=300, bbox_inches='tight', facecolor='white')
print(f'Saved: {out_path}')
plt.close()
