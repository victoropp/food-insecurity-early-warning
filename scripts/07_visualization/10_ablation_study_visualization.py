#!/usr/bin/env python3
"""
Author: Victor Collins Oppon
MSc Data Science Dissertation
Middlesex University, 2025
"""

"""
Ablation Study Visualization - Feature Component Analysis
==========================================================
Creates publication-ready visualizations of ablation study results showing
the incremental contribution of each feature component.

Visualizations:
1. Feature component contribution (AUC improvements)
2. Performance metrics comparison across ablation models
3. Model complexity vs performance trade-off
4. Statistical significance heatmap

Date: December 27, 2025
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from pathlib import Path
from datetime import datetime
import json

# Add parent directory for config imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FIGURES_DIR, RESULTS_DIR

print("=" * 80)
print("ABLATION STUDY VISUALIZATION")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Set up paths
ABLATION_RESULTS = RESULTS_DIR / "baseline_comparison" / "ablation_studies"
ABLATION_FIGURES = FIGURES_DIR / "ablation_studies"
ABLATION_FIGURES.mkdir(parents=True, exist_ok=True)

# Plot settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

# Load data
print("Loading ablation study results...")
summary_df = pd.read_csv(ABLATION_RESULTS / "ablation_summary.csv")
comparisons_df = pd.read_csv(ABLATION_RESULTS / "ablation_comparisons.csv")
with open(ABLATION_RESULTS / "ablation_summary.json", 'r') as f:
    summary_json = json.load(f)

print(f"  Loaded {len(summary_df)} ablation models")
print(f"  Loaded {len(comparisons_df)} pairwise comparisons\n")

# Extract feature counts dynamically from the data
feature_counts = []
for desc in summary_df['features']:
    # Extract number from parentheses, e.g., "ratio + location (11)" -> 11
    count = int(desc.split('(')[1].split(')')[0])
    feature_counts.append(count)

# Define meaningful labels for all charts (using actual feature counts)
MODEL_LABELS_SHORT = [
    'Baseline:\nRatio + Loc',
    'Baseline\n+ Z-score',
    'Baseline + Z-score\n+ DMD',
    'Baseline + Z-score\n+ HMM',
    'Full Model:\nAll Features'
]

MODEL_LABELS_LONG = [
    f'Baseline\n(Ratio + Location)\n{feature_counts[0]} features',
    f'+ Z-score\n({feature_counts[1]} features)',
    f'+ DMD\n({feature_counts[2]} features)',
    f'+ HMM\n({feature_counts[3]} features)',
    f'Full Model\n(All Features)\n{feature_counts[4]} features'
]

MODEL_LABELS_VERY_SHORT = [
    'Ratio+Loc',
    '+Zscore',
    '+DMD',
    '+HMM',
    'Full'
]

# Define consistent, professional color scheme for all ablation models
# Using a muted, colorblind-friendly palette suitable for academic publications
MODEL_COLORS = [
    '#4C72B0',  # Muted blue - Baseline (Ratio+Loc)
    '#55A868',  # Muted green - +Zscore
    '#C44E52',  # Muted red - +DMD
    '#8172B3',  # Muted purple - +HMM
    '#CCB974'   # Muted gold - Full Model
]

# ============================================================================
# FIGURE 1: Feature Component Contribution (Waterfall Chart)
# ============================================================================
print("Creating Figure 1: Feature component contribution...")

fig, ax = plt.subplots(figsize=(12, 8))

# Extract AUC values
aucs = summary_df['mean_auc'].values
labels = MODEL_LABELS_LONG

# Calculate incremental gains
baseline = aucs[0]
gains = [0]  # Baseline has no gain
for i in range(1, len(aucs)):
    gains.append(aucs[i] - aucs[i-1])

# Create waterfall chart
x_pos = np.arange(len(labels))
colors = ['#2196F3' if g >= 0 else '#F44336' for g in gains]
colors[0] = '#757575'  # Baseline is gray

# Plot bars
bars = ax.bar(x_pos, aucs, color=colors, alpha=0.7, edgecolor='black', linewidth=1)

# Add value labels on bars
for i, (bar, auc, gain) in enumerate(zip(bars, aucs, gains)):
    # AUC value
    ax.text(bar.get_x() + bar.get_width()/2, auc + 0.002,
            f'{auc:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Gain/loss (skip baseline)
    if i > 0:
        sign = '+' if gain >= 0 else ''
        color = 'green' if gain >= 0 else 'red'
        ax.text(bar.get_x() + bar.get_width()/2, auc - 0.008,
                f'{sign}{gain:.4f}', ha='center', va='top',
                fontsize=8, color=color, style='italic')

# Add connecting lines to show progression
for i in range(len(aucs)-1):
    ax.plot([x_pos[i] + 0.4, x_pos[i+1] - 0.4],
            [aucs[i], aucs[i+1]],
            'k--', linewidth=1, alpha=0.3, zorder=0)

ax.set_xticks(x_pos)
ax.set_xticklabels(labels, fontsize=9)
ax.set_ylabel('AUC-ROC', fontsize=11, fontweight='bold')
ax.set_title('Ablation Study: Incremental Feature Component Contribution',
             fontsize=13, fontweight='bold', pad=20)
ax.set_ylim(0.87, max(aucs) * 1.02)
ax.grid(axis='y', alpha=0.3, linestyle='--')

# Add legend
legend_elements = [
    Patch(facecolor='#757575', label='Baseline (Ratio + Location)'),
    Patch(facecolor='#2196F3', label='Improvement'),
    Patch(facecolor='#F44336', label='Degradation')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

# Add summary text box with branching explanation
total_gain = aucs[-1] - aucs[0]
textstr = (
    f'Total Gain: +{total_gain:.4f} (+{total_gain/aucs[0]*100:.2f}%)\n\n'
    'Note: This chart shows sequential progression.\n'
    'See Figure 6 for branching ablation structure\n'
    '(HMM and DMD tested both independently\n'
    'and in combination).'
)
props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props, fontweight='normal')

plt.tight_layout()
plt.savefig(ABLATION_FIGURES / "01_feature_component_contribution.png",
            bbox_inches='tight', dpi=300)
plt.close()
print(f"  [OK] Saved: 01_feature_component_contribution.png\n")

# ============================================================================
# FIGURE 2: Multi-Metric Comparison
# ============================================================================
print("Creating Figure 2: Multi-metric comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

metrics = [
    ('mean_auc', 'AUC-ROC', axes[0, 0]),
    ('mean_pr_auc', 'PR-AUC', axes[0, 1]),
    ('mean_f1', 'F1-Score (Youden)', axes[1, 0]),
    ('mean_precision', 'Precision (Youden)', axes[1, 1])
]

x_pos = np.arange(len(summary_df))
width = 0.6

for metric_col, metric_name, ax in metrics:
    values = summary_df[metric_col].values

    # Use consistent color scheme
    bars = ax.bar(x_pos, values, width, color=MODEL_COLORS,
                   edgecolor='black', linewidth=1)

    # Add value labels
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width()/2, val + max(values)*0.01,
                f'{val:.4f}', ha='center', va='bottom', fontsize=8)

    # Formatting
    ax.set_xticks(x_pos)
    ax.set_xticklabels(MODEL_LABELS_VERY_SHORT, fontsize=9)
    ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
    ax.set_title(f'{metric_name} Across Feature Configurations', fontsize=11, fontweight='bold')
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.set_ylim(min(values) * 0.95, max(values) * 1.05)

plt.suptitle('Ablation Study: Multi-Metric Performance Comparison',
             fontsize=14, fontweight='bold', y=1.0)

# Add color-coded legend explaining labels (using actual feature counts)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=MODEL_COLORS[0], edgecolor='black', label=f'Ratio+Loc ({feature_counts[0]} features)'),
    Patch(facecolor=MODEL_COLORS[1], edgecolor='black', label=f'+Zscore ({feature_counts[1]} total)'),
    Patch(facecolor=MODEL_COLORS[2], edgecolor='black', label=f'+DMD ({feature_counts[2]} total)'),
    Patch(facecolor=MODEL_COLORS[3], edgecolor='black', label=f'+HMM ({feature_counts[3]} total)'),
    Patch(facecolor=MODEL_COLORS[4], edgecolor='black', label=f'Full ({feature_counts[4]} total)')
]
fig.legend(handles=legend_elements, loc='upper center', bbox_to_anchor=(0.5, -0.02),
           ncol=5, fontsize=9, framealpha=0.95, title='Feature Configurations (Branching Design: DMD & HMM tested independently + combined)',
           edgecolor='black', fancybox=False)

plt.tight_layout(rect=[0, 0.05, 1, 0.98])
plt.savefig(ABLATION_FIGURES / "02_multi_metric_comparison.png",
            bbox_inches='tight', dpi=300)
plt.close()
print(f"  [OK] Saved: 02_multi_metric_comparison.png\n")

# ============================================================================
# FIGURE 3: Complexity vs Performance Trade-off
# ============================================================================
print("Creating Figure 3: Complexity vs performance trade-off...")

fig, ax = plt.subplots(figsize=(10, 8))

# Use feature_counts already extracted at the top of the script
# Plot with consistent colors
scatter = ax.scatter(feature_counts, summary_df['mean_auc'],
                     s=300, c=MODEL_COLORS, alpha=0.7,
                     edgecolors='black', linewidth=2)

# Add error bars
ax.errorbar(feature_counts, summary_df['mean_auc'],
            yerr=summary_df['std_auc'], fmt='none',
            ecolor='gray', alpha=0.5, capsize=5)

# Add labels
for i, (x, y) in enumerate(zip(feature_counts, summary_df['mean_auc'])):
    ax.annotate(MODEL_LABELS_VERY_SHORT[i], (x, y),
                xytext=(10, 10), textcoords='offset points',
                fontsize=9, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

# Formatting
ax.set_xlabel('Number of Features', fontsize=11, fontweight='bold')
ax.set_ylabel('AUC-ROC', fontsize=11, fontweight='bold')
ax.set_title('Model Complexity vs Performance Trade-off',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--')

# Add trend line
z = np.polyfit(feature_counts, summary_df['mean_auc'], 2)
p = np.poly1d(z)
x_trend = np.linspace(min(feature_counts), max(feature_counts), 100)
ax.plot(x_trend, p(x_trend), 'r--', alpha=0.5, linewidth=2, label='Trend (quadratic)')

# Calculate marginal returns
marginal_returns = []
for i in range(1, len(feature_counts)):
    feature_diff = feature_counts[i] - feature_counts[i-1]
    auc_diff = summary_df['mean_auc'].values[i] - summary_df['mean_auc'].values[i-1]
    marginal_return = auc_diff / feature_diff if feature_diff > 0 else 0
    marginal_returns.append(marginal_return)

avg_marginal_return = np.mean(marginal_returns)
textstr = f'Avg Marginal Return:\n{avg_marginal_return:.6f} AUC/feature'
props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
ax.text(0.02, 0.98, textstr, transform=ax.transAxes, fontsize=9,
        verticalalignment='top', bbox=props)

# Add color-coded legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=MODEL_COLORS[0], edgecolor='black', label='Ratio+Loc'),
    Patch(facecolor=MODEL_COLORS[1], edgecolor='black', label='+Zscore'),
    Patch(facecolor=MODEL_COLORS[2], edgecolor='black', label='+DMD'),
    Patch(facecolor=MODEL_COLORS[3], edgecolor='black', label='+HMM'),
    Patch(facecolor=MODEL_COLORS[4], edgecolor='black', label='Full'),
    plt.Line2D([0], [0], color='r', linestyle='--', linewidth=2, label='Trend (quadratic)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9, framealpha=0.9)

plt.tight_layout()
plt.savefig(ABLATION_FIGURES / "03_complexity_vs_performance.png",
            bbox_inches='tight', dpi=300)
plt.close()
print(f"  [OK] Saved: 03_complexity_vs_performance.png\n")

# ============================================================================
# FIGURE 4: Statistical Significance Matrix
# ============================================================================
print("Creating Figure 4: Statistical significance matrix...")

fig, ax = plt.subplots(figsize=(10, 8))

# Create significance matrix
n_models = len(summary_df)
sig_matrix = np.zeros((n_models, n_models))
p_matrix = np.ones((n_models, n_models))

# Fill matrix from comparisons
for _, row in comparisons_df.iterrows():
    comparison = row['comparison']
    p_val = row['p_value']

    # Parse model indices from comparison name
    # Example: "Z-score Addition" compares Abl 1 vs Abl 2
    if 'Z-score Addition' in comparison:
        i, j = 0, 1
    elif 'DMD Addition (isolated)' in comparison:
        i, j = 1, 2
    elif 'HMM Addition (isolated)' in comparison:
        i, j = 1, 3
    elif 'HMM Addition (on top of DMD)' in comparison:
        i, j = 2, 4
    elif 'DMD Addition (on top of HMM)' in comparison:
        i, j = 3, 4
    elif 'Full Feature Set' in comparison:
        i, j = 0, 4
    else:
        continue

    p_matrix[i, j] = p_val
    p_matrix[j, i] = p_val

    # Significance levels: * p<0.05, ** p<0.01, *** p<0.001
    if p_val < 0.001:
        sig_matrix[i, j] = 3
        sig_matrix[j, i] = 3
    elif p_val < 0.01:
        sig_matrix[i, j] = 2
        sig_matrix[j, i] = 2
    elif p_val < 0.05:
        sig_matrix[i, j] = 1
        sig_matrix[j, i] = 1

# Plot heatmap
im = ax.imshow(p_matrix, cmap='RdYlGn_r', aspect='auto',
               vmin=0, vmax=0.1, interpolation='nearest')

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('p-value', fontsize=10, fontweight='bold')

# Add text annotations
for i in range(n_models):
    for j in range(n_models):
        if i == j:
            text = '–'
        else:
            p_val = p_matrix[i, j]
            sig_level = sig_matrix[i, j]

            if sig_level == 3:
                sig_str = '***'
            elif sig_level == 2:
                sig_str = '**'
            elif sig_level == 1:
                sig_str = '*'
            else:
                sig_str = 'ns'

            text = f'{p_val:.3f}\n{sig_str}'

        ax.text(j, i, text, ha='center', va='center',
                fontsize=8, fontweight='bold')

# Formatting
ax.set_xticks(np.arange(n_models))
ax.set_yticks(np.arange(n_models))
ax.set_xticklabels(MODEL_LABELS_VERY_SHORT, fontsize=9)
ax.set_yticklabels(MODEL_LABELS_VERY_SHORT, fontsize=9)
ax.set_title('Statistical Significance of AUC Differences\n(Paired t-tests)',
             fontsize=13, fontweight='bold', pad=15)

# Add grid
ax.set_xticks(np.arange(n_models) - 0.5, minor=True)
ax.set_yticks(np.arange(n_models) - 0.5, minor=True)
ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

# Add legends
sig_legend = '* p<0.05  ** p<0.01  *** p<0.001  ns = not significant'
ax.text(0.5, -0.12, sig_legend, transform=ax.transAxes,
        ha='center', fontsize=9, style='italic')

# Add color-coded feature configuration legend (using actual feature counts)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=MODEL_COLORS[0], edgecolor='black', label=f'Ratio+Loc ({feature_counts[0]})'),
    Patch(facecolor=MODEL_COLORS[1], edgecolor='black', label=f'+Zscore ({feature_counts[1]})'),
    Patch(facecolor=MODEL_COLORS[2], edgecolor='black', label=f'+DMD ({feature_counts[2]})'),
    Patch(facecolor=MODEL_COLORS[3], edgecolor='black', label=f'+HMM ({feature_counts[3]})'),
    Patch(facecolor=MODEL_COLORS[4], edgecolor='black', label=f'Full ({feature_counts[4]})')
]
ax.legend(handles=legend_elements, loc='center left', bbox_to_anchor=(1.05, 0.5),
          fontsize=9, framealpha=0.95, title='Feature Configurations\n(# features)',
          edgecolor='black', fancybox=False)

plt.tight_layout(rect=[0, 0, 0.85, 1])
plt.savefig(ABLATION_FIGURES / "04_statistical_significance.png",
            bbox_inches='tight', dpi=300)
plt.close()
print(f"  [OK] Saved: 04_statistical_significance.png\n")

# ============================================================================
# FIGURE 5: Precision-Recall Trade-off
# ============================================================================
print("Creating Figure 5: Precision-recall trade-off...")

fig, ax = plt.subplots(figsize=(10, 8))

# Plot precision vs recall with consistent colors
for i, row in summary_df.iterrows():
    ax.scatter(row['mean_recall'], row['mean_precision'],
               s=300, c=[MODEL_COLORS[i]], alpha=0.7,
               edgecolors='black', linewidth=2, zorder=3)

    # Add label with feature count
    feat_count = row["features"].split("(")[1].split(")")[0]
    ax.annotate(f'{MODEL_LABELS_VERY_SHORT[i]}\n({feat_count} feat)',
                (row['mean_recall'], row['mean_precision']),
                xytext=(15, 15), textcoords='offset points',
                fontsize=8, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.7),
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0'))

# Connect points in sequence
for i in range(len(summary_df)-1):
    ax.plot([summary_df.loc[i, 'mean_recall'], summary_df.loc[i+1, 'mean_recall']],
            [summary_df.loc[i, 'mean_precision'], summary_df.loc[i+1, 'mean_precision']],
            'k--', alpha=0.3, linewidth=1, zorder=1)

# Formatting
ax.set_xlabel('Recall (Youden Threshold)', fontsize=11, fontweight='bold')
ax.set_ylabel('Precision (Youden Threshold)', fontsize=11, fontweight='bold')
ax.set_title('Precision-Recall Trade-off Across Ablation Models',
             fontsize=13, fontweight='bold', pad=15)
ax.grid(True, alpha=0.3, linestyle='--')

# Add F1 iso-lines
f1_values = [0.3, 0.35, 0.4]
recall_range = np.linspace(0.1, 1.0, 100)
for f1 in f1_values:
    precision = f1 * recall_range / (2 * recall_range - f1)
    precision = np.where((precision >= 0) & (precision <= 1), precision, np.nan)
    ax.plot(recall_range, precision, ':', color='gray', alpha=0.5, linewidth=1)
    # Label
    valid_idx = np.where(~np.isnan(precision))[0]
    if len(valid_idx) > 0:
        mid_idx = valid_idx[len(valid_idx)//2]
        ax.text(recall_range[mid_idx], precision[mid_idx], f'F1={f1:.2f}',
                fontsize=7, color='gray', style='italic', rotation=-30)

# Add color-coded legend (using actual feature counts)
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor=MODEL_COLORS[0], edgecolor='black', label=f'Ratio+Loc ({feature_counts[0]})'),
    Patch(facecolor=MODEL_COLORS[1], edgecolor='black', label=f'+Zscore ({feature_counts[1]})'),
    Patch(facecolor=MODEL_COLORS[2], edgecolor='black', label=f'+DMD ({feature_counts[2]})'),
    Patch(facecolor=MODEL_COLORS[3], edgecolor='black', label=f'+HMM ({feature_counts[3]})'),
    Patch(facecolor=MODEL_COLORS[4], edgecolor='black', label=f'Full ({feature_counts[4]})')
]
ax.legend(handles=legend_elements, loc='lower left', fontsize=9, framealpha=0.9,
          title='Feature Configurations\n(# features)')

plt.tight_layout()
plt.savefig(ABLATION_FIGURES / "05_precision_recall_tradeoff.png",
            bbox_inches='tight', dpi=300)
plt.close()
print(f"  [OK] Saved: 05_precision_recall_tradeoff.png\n")

# ============================================================================
# Summary
# ============================================================================
print("=" * 80)
print("ABLATION STUDY VISUALIZATION COMPLETE")
print("=" * 80)
print(f"Output directory: {ABLATION_FIGURES}")
print("\nFigures created:")
print("  1. 01_feature_component_contribution.png - Incremental AUC gains")
print("  2. 02_multi_metric_comparison.png - AUC, PR-AUC, F1, Precision comparison")
print("  3. 03_complexity_vs_performance.png - Feature count vs AUC trade-off")
print("  4. 04_statistical_significance.png - Pairwise significance matrix")
print("  5. 05_precision_recall_tradeoff.png - Precision-recall space")
print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
