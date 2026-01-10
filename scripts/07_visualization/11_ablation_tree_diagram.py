#!/usr/bin/env python3
"""
Author: Victor Collins Oppon
MSc Data Science Dissertation
Middlesex University, 2025
"""

"""
Ablation Study Tree Diagram
============================
Creates a tree diagram showing the branching ablation study structure
to clarify how HMM and DMD were tested both in isolation and combination.

Date: December 27, 2025
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from pathlib import Path
from datetime import datetime

# Add parent directory for config imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FIGURES_DIR, RESULTS_DIR

print("=" * 80)
print("ABLATION STUDY TREE DIAGRAM")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Set up paths
ABLATION_RESULTS = RESULTS_DIR / "baseline_comparison" / "ablation_studies"
ABLATION_FIGURES = FIGURES_DIR / "ablation_studies"
ABLATION_FIGURES.mkdir(parents=True, exist_ok=True)

# Plot settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 9

# Load data
summary_df = pd.read_csv(ABLATION_RESULTS / "ablation_summary.csv")
comparisons_df = pd.read_csv(ABLATION_RESULTS / "ablation_comparisons.csv")

print("Creating ablation study tree diagram...\n")

# Create figure
fig, ax = plt.subplots(figsize=(14, 10))
ax.set_xlim(0, 14)
ax.set_ylim(0, 10)
ax.axis('off')

# Define node positions (x, y)
positions = {
    'baseline': (7, 8.5),
    'zscore': (7, 7),
    'dmd': (4, 5),
    'hmm': (10, 5),
    'full': (7, 3)
}

# Define colors
node_color = '#E8F4F8'
edge_color = '#2C3E50'
text_color = '#2C3E50'
highlight_color = '#27AE60'

# Helper function to draw a node
def draw_node(ax, pos, label, auc, features, color=node_color):
    box = FancyBboxPatch(
        (pos[0] - 1.2, pos[1] - 0.35), 2.4, 0.7,
        boxstyle="round,pad=0.1",
        facecolor=color,
        edgecolor=edge_color,
        linewidth=2
    )
    ax.add_patch(box)

    # Main label
    ax.text(pos[0], pos[1] + 0.15, label,
            ha='center', va='center', fontsize=10, fontweight='bold', color=text_color)

    # AUC and features
    ax.text(pos[0], pos[1] - 0.15, f'AUC: {auc:.4f} | {features} features',
            ha='center', va='center', fontsize=8, color=text_color)

# Helper function to draw an arrow with label
def draw_arrow(ax, start, end, label, delta, significance='', label_pos=None):
    arrow = FancyArrowPatch(
        start, end,
        arrowstyle='->',
        mutation_scale=20,
        linewidth=2.5 if significance == '*' else 2,
        color=highlight_color if significance == '*' else edge_color,
        zorder=1
    )
    ax.add_patch(arrow)

    # Use custom label position if provided, otherwise use midpoint with offset
    if label_pos is None:
        mid_x = (start[0] + end[0]) / 2
        mid_y = (start[1] + end[1]) / 2
        # Better offset logic based on direction
        offset_x = 0.8 if start[0] != end[0] else 0
        offset_y = 0 if start[1] != end[1] else 0
        label_x = mid_x + offset_x
        label_y = mid_y + offset_y
    else:
        label_x, label_y = label_pos

    label_text = f'{label}\n(Δ={delta:+.4f})'
    if significance:
        label_text += f'\n{significance}'

    ax.text(label_x, label_y, label_text,
            ha='center', va='center', fontsize=7,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                     edgecolor=highlight_color if significance == '*' else edge_color,
                     linewidth=1.5 if significance == '*' else 1),
            color=highlight_color if significance == '*' else text_color,
            fontweight='bold' if significance == '*' else 'normal')

# Draw nodes
draw_node(ax, positions['baseline'],
          'Baseline\nRatio + Location',
          summary_df.loc[0, 'mean_auc'], 11)

draw_node(ax, positions['zscore'],
          'Baseline + Z-score',
          summary_df.loc[1, 'mean_auc'], 20)

draw_node(ax, positions['dmd'],
          'Baseline + Z-score\n+ DMD',
          summary_df.loc[2, 'mean_auc'], 26)

draw_node(ax, positions['hmm'],
          'Baseline + Z-score\n+ HMM',
          summary_df.loc[3, 'mean_auc'], 32)

draw_node(ax, positions['full'],
          'Full Model\nAll Features',
          summary_df.loc[4, 'mean_auc'], 38, color='#D5F4E6')

# Draw arrows with deltas and custom label positions
# Baseline -> Zscore
delta = summary_df.loc[1, 'mean_auc'] - summary_df.loc[0, 'mean_auc']
draw_arrow(ax,
           (positions['baseline'][0], positions['baseline'][1] - 0.35),
           (positions['zscore'][0], positions['zscore'][1] + 0.35),
           'Add Z-score', delta, '', label_pos=(8.2, 7.75))

# Zscore -> DMD
delta = summary_df.loc[2, 'mean_auc'] - summary_df.loc[1, 'mean_auc']
draw_arrow(ax,
           (positions['zscore'][0] - 0.5, positions['zscore'][1] - 0.35),
           (positions['dmd'][0] + 0.5, positions['dmd'][1] + 0.35),
           'Add DMD\n(isolated)', delta, '', label_pos=(4.8, 6.0))

# Zscore -> HMM
delta = summary_df.loc[3, 'mean_auc'] - summary_df.loc[1, 'mean_auc']
draw_arrow(ax,
           (positions['zscore'][0] + 0.5, positions['zscore'][1] - 0.35),
           (positions['hmm'][0] - 0.5, positions['hmm'][1] + 0.35),
           'Add HMM\n(isolated)', delta, '', label_pos=(9.2, 6.0))

# DMD -> Full
delta = summary_df.loc[4, 'mean_auc'] - summary_df.loc[2, 'mean_auc']
sig = '*' if comparisons_df.loc[comparisons_df['comparison'] == 'HMM Addition (on top of DMD)', 'significance'].values[0] == '*' else ''
draw_arrow(ax,
           (positions['dmd'][0] + 0.5, positions['dmd'][1] - 0.35),
           (positions['full'][0] - 0.5, positions['full'][1] + 0.35),
           'Add HMM\n(on top of DMD)', delta, sig, label_pos=(4.8, 4.0))

# HMM -> Full
delta = summary_df.loc[4, 'mean_auc'] - summary_df.loc[3, 'mean_auc']
draw_arrow(ax,
           (positions['hmm'][0] - 0.5, positions['hmm'][1] - 0.35),
           (positions['full'][0] + 0.5, positions['full'][1] + 0.35),
           'Add DMD\n(on top of HMM)', delta, '', label_pos=(9.2, 4.0))

# Add title
ax.text(7, 9.5, 'Ablation Study: Branching Structure',
        ha='center', va='center', fontsize=14, fontweight='bold')

# Extract metrics from actual data (not hardcoded)
# HMM alone (Abl4 - Abl2)
hmm_alone_delta = summary_df.loc[3, 'mean_auc'] - summary_df.loc[1, 'mean_auc']
hmm_alone_comp = comparisons_df[comparisons_df['comparison'] == 'HMM Addition (isolated)'].iloc[0]
hmm_alone_pval = hmm_alone_comp['p_value']
hmm_alone_sig = hmm_alone_comp['significance']

# DMD alone (Abl3 - Abl2)
dmd_alone_delta = summary_df.loc[2, 'mean_auc'] - summary_df.loc[1, 'mean_auc']
dmd_alone_comp = comparisons_df[comparisons_df['comparison'] == 'DMD Addition (isolated)'].iloc[0]
dmd_alone_pval = dmd_alone_comp['p_value']
dmd_alone_sig = dmd_alone_comp['significance']

# HMM on top of DMD (Abl5 - Abl3)
hmm_on_dmd_delta = summary_df.loc[4, 'mean_auc'] - summary_df.loc[2, 'mean_auc']
hmm_on_dmd_comp = comparisons_df[comparisons_df['comparison'] == 'HMM Addition (on top of DMD)'].iloc[0]
hmm_on_dmd_pval = hmm_on_dmd_comp['p_value']
hmm_on_dmd_sig = hmm_on_dmd_comp['significance']

# DMD on top of HMM (Abl5 - Abl4)
dmd_on_hmm_delta = summary_df.loc[4, 'mean_auc'] - summary_df.loc[3, 'mean_auc']
dmd_on_hmm_comp = comparisons_df[comparisons_df['comparison'] == 'DMD Addition (on top of HMM)'].iloc[0]
dmd_on_hmm_pval = dmd_on_hmm_comp['p_value']
dmd_on_hmm_sig = dmd_on_hmm_comp['significance']

# Add legend with actual values
legend_text = (
    f'Key Findings:\n'
    f'• HMM alone: {hmm_alone_delta:+.4f} AUC (p={hmm_alone_pval:.3f}, {hmm_alone_sig})\n'
    f'• DMD alone: {dmd_alone_delta:+.4f} AUC (p={dmd_alone_pval:.3f}, {dmd_alone_sig})\n'
    f'• Adding HMM after DMD: {hmm_on_dmd_delta:+.4f} AUC\n'
    f'  (p={hmm_on_dmd_pval:.4f}, {hmm_on_dmd_sig} - SIGNIFICANT)\n'
    f'• Adding DMD after HMM: {dmd_on_hmm_delta:+.4f} AUC\n'
    f'  (p={dmd_on_hmm_pval:.3f}, {dmd_on_hmm_sig})\n\n'
    f'* = Statistically significant (p < 0.05)'
)
ax.text(0.5, 1.5, legend_text,  # Moved down to y=1.5 to avoid overlapping nodes
        ha='left', va='center', fontsize=7.5,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9E6',
                 edgecolor='#F39C12', linewidth=1.5))

# Add interpretation explaining why impacts differ
interp_text = (
    'Why Impacts Differ:\n'
    'Both paths reach the SAME full model\n'
    '(AUC=0.8884), but starting points differ:\n'
    f'• HMM alone: {summary_df.loc[3, "mean_auc"]:.4f} AUC (higher baseline)\n'
    f'• DMD alone: {summary_df.loc[2, "mean_auc"]:.4f} AUC (lower baseline)\n\n'
    'Adding features to a model that already\n'
    'performs well yields smaller marginal gains.\n'
    'Adding to a weaker model shows larger gains.'
)
ax.text(13.5, 1.5, interp_text,  # Moved down to y=1.5 to avoid overlapping nodes
        ha='right', va='center', fontsize=7.5,
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#E8F8F5',
                 edgecolor='#16A085', linewidth=1.5))

plt.tight_layout()
plt.savefig(ABLATION_FIGURES / "06_ablation_tree_structure.png",
            bbox_inches='tight', dpi=300)
plt.close()

print("  [OK] Saved: 06_ablation_tree_structure.png")

print("\n" + "=" * 80)
print("ABLATION TREE DIAGRAM COMPLETE")
print("=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
