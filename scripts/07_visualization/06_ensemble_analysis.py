#!/usr/bin/env python3
"""
Author: Victor Collins Oppon
MSc Data Science Dissertation
Middlesex University, 2025
"""

"""
Ensemble Analysis Visualizations
=================================
Publication-grade visualizations for the two-stage ensemble model.

Creates:
1. Weight optimization curve showing AUC vs alpha
2. Performance improvement comparison (Stage 1 vs Stage 2 vs Ensemble)
3. Metric comparison across all three approaches
4. Ensemble weight breakdown pie chart

Date: December 24, 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # MUST be before importing pyplot (non-interactive backend)
import matplotlib.pyplot as plt
# NOTE: seaborn removed - causes hanging on Windows
from datetime import datetime
import json

from config import RESULTS_DIR, FIGURES_DIR, VISUALIZATION_CONFIG

# Set publication style
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

print("=" * 80)
print("VISUALIZATION: ENSEMBLE ANALYSIS")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Output directory
OUTPUT_DIR = FIGURES_DIR / 'ensemble'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load ensemble results
ENSEMBLE_DIR = RESULTS_DIR / 'ensemble_stage1_stage2'
summary_file = ENSEMBLE_DIR / 'ensemble_summary.json'
weights_file = ENSEMBLE_DIR / 'weight_optimization.csv'

if not summary_file.exists():
    print(f"ERROR: Ensemble summary not found: {summary_file}")
    sys.exit(1)

print("Loading ensemble results...")
with open(summary_file, 'r') as f:
    summary = json.load(f)

weights_df = pd.read_csv(weights_file)
print(f"  Loaded {len(weights_df)} weight optimization points")
print(f"  Optimal alpha: {summary['optimization']['optimal_alpha']}")
print(f"  Optimal AUC: {summary['optimization']['optimal_auc']:.4f}")

# =============================================================================
# FIGURE 1: Weight Optimization Curve
# =============================================================================
print("\nCreating Figure 1: Weight optimization curve...")

fig, ax = plt.subplots(figsize=(12, 7))

# Plot AUC vs alpha
ax.plot(weights_df['alpha'], weights_df['auc'], 'o-', linewidth=2, markersize=4,
        color='steelblue', label='Ensemble AUC')

# Mark optimal point
optimal_alpha = summary['optimization']['optimal_alpha']
optimal_auc = summary['optimization']['optimal_auc']
ax.plot(optimal_alpha, optimal_auc, 'r*', markersize=20,
        label=f'Optimal (alpha={optimal_alpha:.2f}, AUC={optimal_auc:.4f})',
        zorder=5)

# NOTE: ensemble_summary.json contains Stage 1 performance on ENSEMBLE SUBSET only
# (1,383 common observations, AUC 0.8452). For FULL DATASET Stage 1 performance
# (20,722 observations, AUC 0.9075), use comparison_summary.json.

# Load FULL DATASET Stage 1 performance (not ensemble subset)
comparison_file = RESULTS_DIR / 'baseline_comparison' / 'comparison_summary.json'
with open(comparison_file, 'r') as f:
    comparison = json.load(f)
stage1_auc_full = comparison['comparison_table'][0]['auc_mean']  # 0.9075 (full dataset)

# Also load subset for context
stage1_auc_subset = summary['stage1_performance']['auc_roc']  # 0.8452 (ensemble subset only)
stage2_auc = summary['stage2_performance']['auc_roc']

# Plot both Stage 1 performance metrics with clear labels
ax.axhline(y=stage1_auc_full, color='orange', linestyle='--', linewidth=2,
           label=f'Stage 1 Full Dataset (AUC={stage1_auc_full:.4f})')
ax.axhline(y=stage1_auc_subset, color='orange', linestyle=':', linewidth=1.5, alpha=0.6,
           label=f'Stage 1 Ensemble Subset (AUC={stage1_auc_subset:.4f})')
ax.axhline(y=stage2_auc, color='green', linestyle='--', linewidth=2,
           label=f'Stage 2 Only (AUC={stage2_auc:.4f})')

# Mark equal weighting
equal_weight_auc = summary['optimization']['equal_weight_auc']
ax.plot(0.5, equal_weight_auc, 's', markersize=12, color='purple',
        label=f'Equal Weight (alpha=0.50, AUC={equal_weight_auc:.4f})')

ax.set_xlabel('Alpha (Stage 1 Weight)', fontsize=12, weight='bold')
ax.set_ylabel('AUC-ROC', fontsize=12, weight='bold')
ax.set_title('Ensemble Weight Optimization (h=8 months)', fontsize=14, weight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.grid(True, alpha=0.3)
ax.set_xlim(-0.05, 1.05)

# Add annotation
improvement_pct = summary['improvement']['auc_improvement_percent']
ax.text(0.05, 0.95,
        f'Improvement over best individual: +{improvement_pct:.2f}%\n'
        f'Common observations: {summary["data"]["common_obs"]:,}\n'
        f'Optimization metric: {summary["optimization"]["metric"]}',
        transform=ax.transAxes, fontsize=10,
        verticalalignment='top',
        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '01_weight_optimization_curve.png',
            dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
plt.close()
print(f"  Saved: 01_weight_optimization_curve.png")

# =============================================================================
# FIGURE 2: Performance Improvement Comparison
# =============================================================================
print("Creating Figure 2: Performance improvement comparison...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Metrics to compare
metrics = [
    ('auc_roc', 'AUC-ROC', 0, 1),
    ('pr_auc', 'PR-AUC', 0, 1),
    ('brier_score', 'Brier Score (lower is better)', 0, 0.15),
    ('f1', 'F1 Score', 0, 1)
]

for idx, (metric_key, metric_name, ymin, ymax) in enumerate(metrics):
    ax = axes[idx // 2, idx % 2]

    # Extract values
    stage1_val = summary['stage1_performance'][metric_key]
    stage2_val = summary['stage2_performance'][metric_key]
    ensemble_val = summary['ensemble_performance'][metric_key]

    # Create bar chart
    models = ['Stage 1\n(AR Baseline)', 'Stage 2\n(XGBoost Advanced)', 'Ensemble\n(Weighted Avg)']
    values = [stage1_val, stage2_val, ensemble_val]
    colors = ['orange', 'green', 'steelblue']

    bars = ax.bar(models, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)

    # Add value labels on bars
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{val:.4f}',
                ha='center', va='bottom', fontsize=10, weight='bold')

    ax.set_ylabel(metric_name, fontsize=11, weight='bold')
    ax.set_ylim(ymin, ymax)
    ax.grid(axis='y', alpha=0.3)

    # Highlight best performer (lowest for Brier, highest for others)
    if metric_key == 'brier_score':
        best_idx = np.argmin(values)
    else:
        best_idx = np.argmax(values)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)

plt.suptitle('Performance Comparison: Stage 1 vs Stage 2 vs Ensemble',
             fontsize=14, weight='bold', y=0.995)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / '02_performance_comparison.png',
            dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
plt.close()
print(f"  Saved: 02_performance_comparison.png")

# =============================================================================
# FIGURE 3: Ensemble Weight Breakdown
# =============================================================================
print("Creating Figure 3: Ensemble weight breakdown...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# Pie chart
stage1_weight = summary['optimization']['stage1_weight']
stage2_weight = summary['optimization']['stage2_weight']

sizes = [stage1_weight, stage2_weight]
labels = [f'Stage 1 (AR Baseline)\n{stage1_weight:.1%}',
          f'Stage 2 (XGBoost Advanced)\n{stage2_weight:.1%}']
colors = ['orange', 'green']
explode = (0.05, 0.05)

ax1.pie(sizes, explode=explode, labels=labels, colors=colors,
        autopct='', shadow=True, startangle=90, textprops={'fontsize': 12, 'weight': 'bold'})
ax1.set_title('Optimal Ensemble Weights', fontsize=13, weight='bold')

# Bar comparison of precision and recall
precision_vals = [
    summary['stage1_performance']['precision'],
    summary['stage2_performance']['precision'],
    summary['ensemble_performance']['precision']
]
recall_vals = [
    summary['stage1_performance']['recall'],
    summary['stage2_performance']['recall'],
    summary['ensemble_performance']['recall']
]

x = np.arange(3)
width = 0.35

bars1 = ax2.bar(x - width/2, precision_vals, width, label='Precision',
                color='skyblue', edgecolor='black', linewidth=1)
bars2 = ax2.bar(x + width/2, recall_vals, width, label='Recall',
                color='salmon', edgecolor='black', linewidth=1)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.3f}',
                ha='center', va='bottom', fontsize=9)

ax2.set_ylabel('Score', fontsize=11, weight='bold')
ax2.set_title('Precision vs Recall Comparison', fontsize=13, weight='bold')
ax2.set_xticks(x)
ax2.set_xticklabels(['Stage 1', 'Stage 2', 'Ensemble'])
ax2.legend(fontsize=10)
ax2.set_ylim(0, 1)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '03_weight_breakdown_and_metrics.png',
            dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
plt.close()
print(f"  Saved: 03_weight_breakdown_and_metrics.png")

# =============================================================================
# FIGURE 4: Data Coverage Comparison
# =============================================================================
print("Creating Figure 4: Data coverage comparison...")

fig, ax = plt.subplots(figsize=(10, 7))

# Data statistics
stage1_obs = summary['data']['stage1_only_obs']
stage2_obs = summary['data']['stage2_only_obs']
common_obs = summary['data']['common_obs']
crisis_events = summary['data']['crisis_events']
crisis_rate = summary['data']['crisis_rate']

# Venn diagram-style representation using bars
categories = ['Stage 1\nOnly', 'Common\nObservations', 'Stage 2\nOnly']
values = [stage1_obs, common_obs, stage2_obs]
colors_venn = ['orange', 'purple', 'green']

bars = ax.barh(categories, values, color=colors_venn, alpha=0.7, edgecolor='black', linewidth=1.5)

# Add value labels
for bar, val in zip(bars, values):
    width = bar.get_width()
    ax.text(width, bar.get_y() + bar.get_height()/2.,
            f'{val:,}',
            ha='left', va='center', fontsize=12, weight='bold',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

ax.set_xlabel('Number of Observations', fontsize=12, weight='bold')
ax.set_title(f'Data Coverage Comparison\n(Ensemble uses {common_obs:,} common observations with {crisis_events} crisis events, {crisis_rate:.2%} crisis rate)',
             fontsize=13, weight='bold')
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / '04_data_coverage.png',
            dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
plt.close()
print(f"  Saved: 04_data_coverage.png")

# Print summary
print("\n" + "=" * 80)
print("ENSEMBLE ANALYSIS VISUALIZATION COMPLETE")
print("=" * 80)
print(f"Output directory: {OUTPUT_DIR}")
print(f"Figures created: 4")
print(f"  1. Weight optimization curve")
print(f"  2. Performance comparison (AUC, PR-AUC, Brier, F1)")
print(f"  3. Weight breakdown and precision-recall comparison")
print(f"  4. Data coverage comparison")
print(f"\nEnd time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
