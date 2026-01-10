#!/usr/bin/env python3
"""
Author: Victor Collins Oppon
MSc Data Science Dissertation
Middlesex University, 2025
"""

"""
Pipeline Flow Diagram
=====================
Create comprehensive visualization showing the complete pipeline architecture
from raw data through to final ensemble predictions.

Shows:
- Stage 0: Literature Baseline (XGBoost on full dataset)
- Stage 1: AR Baseline (Spatial autoregressive model)
- Stage 2: Advanced ML (XGBoost with HMM/DMD features + WITH_AR_FILTER)
- Stage 3: Ensemble (Weighted probability averaging)
- Data flow, filtering logic, and model performance at each stage

Date: December 24, 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import json
from datetime import datetime

from config import RESULTS_DIR, FIGURES_DIR, VISUALIZATION_CONFIG

print("=" * 80)
print("VISUALIZATION: PIPELINE FLOW DIAGRAM")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Output directory
OUTPUT_DIR = FIGURES_DIR / 'pipeline'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load performance metrics from results - NO HARDCODED VALUES
print("Loading pipeline results...")

# Load ensemble metrics (for ensemble-specific data)
ensemble_file = RESULTS_DIR / 'ensemble_stage1_stage2' / 'ensemble_summary.json'
if not ensemble_file.exists():
    print(f"ERROR: Ensemble summary not found: {ensemble_file}")
    print("Cannot create pipeline diagram without actual results.")
    sys.exit(1)

with open(ensemble_file, 'r') as f:
    ensemble_summary = json.load(f)

ensemble_auc = ensemble_summary['ensemble_performance']['auc_roc']
common_obs = ensemble_summary['data']['common_obs']
stage1_obs = ensemble_summary['data']['stage1_only_obs']
stage2_obs = ensemble_summary['data']['stage2_only_obs']
crisis_events = ensemble_summary['data']['crisis_events']

print(f"  Loaded ensemble summary: {ensemble_file}")

# Load FULL DATASET metrics from comparison summary (NOT ensemble subset)
comparison_file = RESULTS_DIR / 'baseline_comparison' / 'comparison_summary.json'
if not comparison_file.exists():
    print(f"ERROR: Comparison summary not found: {comparison_file}")
    sys.exit(1)

with open(comparison_file, 'r') as f:
    comparison_data = json.load(f)

# Extract full-dataset AUC values (NOT the common-obs-only values from ensemble)
for model_entry in comparison_data['comparison_table']:
    if model_entry['model'] == 'Stage 1 AR Baseline':
        stage1_auc = model_entry['auc_mean']
        stage1_full_obs = int(model_entry['n_obs'])
    elif model_entry['model'] == 'Stage 2 XGBoost Advanced':
        stage2_auc = model_entry['auc_mean']
        stage2_full_obs = int(model_entry['n_obs'])

print(f"  Loaded comparison summary: {comparison_file}")

# Load Stage 0 baseline from actual results file
baseline_file = RESULTS_DIR / 'baseline_comparison' / 'literature_baseline' / 'literature_baseline_summary.json'
if not baseline_file.exists():
    # Try alternative location
    baseline_file = RESULTS_DIR / 'baseline_comparison' / 'comparison_summary.json'
    if baseline_file.exists():
        with open(baseline_file, 'r') as f:
            comparison_data = json.load(f)
        baseline_auc = comparison_data.get('literature_auc', None)
        print(f"  Loaded literature baseline from comparison: {baseline_file}")
    else:
        print(f"WARNING: Literature baseline not found at {baseline_file}")
        print("Pipeline diagram will exclude Stage 0.")
        baseline_auc = None
else:
    with open(baseline_file, 'r') as f:
        baseline_metrics = json.load(f)
    baseline_auc = baseline_metrics['mean_auc']  # No fallback - must exist
    print(f"  Loaded literature baseline: {baseline_file}")

# Validate all required metrics loaded successfully
if baseline_auc is None:
    print("  Stage 0 (Literature Baseline): NOT AVAILABLE")
else:
    print(f"  Stage 0 (Literature Baseline): AUC = {baseline_auc:.4f}")
print(f"  Stage 1 (AR Baseline): AUC = {stage1_auc:.4f}")
print(f"  Stage 2 (XGBoost Advanced): AUC = {stage2_auc:.4f}")
print(f"  Ensemble: AUC = {ensemble_auc:.4f}")

# =============================================================================
# CREATE PIPELINE FLOW DIAGRAM
# =============================================================================
print("\nCreating pipeline flow diagram...")

fig = plt.figure(figsize=(18, 14))
ax = fig.add_subplot(111)
ax.set_xlim(0, 10)
ax.set_ylim(0, 12)
ax.axis('off')

# Color scheme
color_data = '#E8F5E9'  # Light green
color_stage0 = '#FFF3E0'  # Light orange
color_stage1 = '#FFF9C4'  # Light yellow
color_stage2 = '#E1F5FE'  # Light blue
color_ensemble = '#F3E5F5'  # Light purple
color_filter = '#FFEBEE'  # Light red

# Font sizes
title_size = 12
text_size = 9
metric_size = 8

def draw_box(ax, x, y, width, height, text, color, title_size=title_size):
    """Draw a fancy box with text."""
    box = FancyBboxPatch((x, y), width, height,
                          boxstyle="round,pad=0.1",
                          edgecolor='black', facecolor=color,
                          linewidth=2)
    ax.add_patch(box)
    ax.text(x + width/2, y + height/2, text,
            ha='center', va='center', fontsize=title_size,
            weight='bold', multialignment='center')

def draw_arrow(ax, x1, y1, x2, y2, label='', style='->'):
    """Draw an arrow with optional label."""
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle=style, mutation_scale=20,
                           linewidth=2, color='black')
    ax.add_patch(arrow)
    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 0.1, mid_y, label,
                fontsize=text_size, ha='left',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Title
ax.text(5, 11.5, 'FINAL PIPELINE ARCHITECTURE',
        ha='center', fontsize=16, weight='bold')
ax.text(5, 11.1, 'Food Insecurity Early Warning System (h=8 months)',
        ha='center', fontsize=12, style='italic')

# ============= RAW DATA (Top) =============
draw_box(ax, 0.5, 10, 1.8, 0.6, 'IPC Data\n(Ground Truth)', color_data, text_size)
draw_box(ax, 2.5, 10, 1.8, 0.6, 'GDELT News\n(Features)', color_data, text_size)
draw_box(ax, 4.5, 10, 1.8, 0.6, 'Spatial Data\n(Boundaries)', color_data, text_size)
draw_box(ax, 6.5, 10, 1.8, 0.6, 'HMM Regimes\n(Phase 2)', color_data, text_size)
draw_box(ax, 8.5, 10, 1.3, 0.6, 'DMD Modes\n(Phase 2)', color_data, text_size)

# ============= STAGE 0: LITERATURE BASELINE =============
if baseline_auc is not None:
    y_stage0 = 8.7
    draw_box(ax, 0.5, y_stage0, 9.3, 1.0,
             f'STAGE 0: Literature Baseline (XGBoost on Full Dataset)\n'
             f'Features: GDELT Ratio Features Only | No AR Filter | No Geographic Encoding\n'
             f'AUC: {baseline_auc:.4f} | Training: All available data',
             color_stage0, text_size)
    draw_arrow(ax, 3.4, 10, 5, y_stage0 + 1, '')
    stage1_y_offset = 7.2
else:
    # Skip Stage 0 if baseline not available
    stage1_y_offset = 8.7

# ============= STAGE 1: AR BASELINE =============
y_stage1 = stage1_y_offset
draw_box(ax, 0.5, y_stage1, 4.3, 1.0,
         f'STAGE 1: Autoregressive (AR) Baseline\n'
         f'Method: Spatial AR (Ls + Lt) | 5-Fold Spatial CV\n'
         f'AUC: {stage1_auc:.4f} | Observations: {stage1_full_obs:,}',
         color_stage1, text_size)

draw_arrow(ax, 1.3, 10, 2, y_stage1 + 1, '')

# ============= FILTER LOGIC =============
y_filter = 6.5
draw_box(ax, 5.2, y_filter, 4.3, 0.6,
         'WITH_AR_FILTER: IPC <= 2 AND AR_pred = 0\n'
         f'Filters dataset to {stage2_obs + common_obs:,} observations',
         color_filter, text_size)

draw_arrow(ax, 2.7, y_stage1 + 0.5, 5.2, y_filter + 0.3, 'AR predictions')

# ============= STAGE 2: ADVANCED ML =============
y_stage2 = 4.8
draw_box(ax, 5.2, y_stage2, 4.3, 1.2,
         f'STAGE 2: XGBoost Advanced WITH_AR\n'
         f'Features: Ratio + Z-score + HMM + DMD + Country Encoding\n'
         f'Training: Filtered dataset only ({stage2_full_obs:,} obs)\n'
         f'5-Fold Spatial CV | Imbalanced handling\n'
         f'AUC: {stage2_auc:.4f}',
         color_stage2, text_size)

draw_arrow(ax, 7.4, y_filter, 7.4, y_stage2 + 1.2, 'Filtered data')
draw_arrow(ax, 7.5, 10, 7.5, y_stage2 + 1.2, '')  # HMM
draw_arrow(ax, 9.2, 10, 9.2, y_stage2 + 1.2, '')  # DMD

# ============= COMMON OBSERVATIONS =============
y_common = 3.8
draw_box(ax, 1.2, y_common, 3.0, 0.6,
         f'Common Observations\n{common_obs:,} obs | {ensemble_summary["data"]["crisis_events"]} crisis events',
         '#FFFDE7', text_size)

draw_arrow(ax, 2.7, y_stage1, 2.7, y_common + 0.6, '')
draw_arrow(ax, 7.4, y_stage2, 4.2, y_common + 0.3, '')

# ============= ENSEMBLE MODEL =============
y_ensemble = 2.2
draw_box(ax, 0.8, y_ensemble, 8.6, 1.1,
         f'ENSEMBLE: Weighted Probability Averaging\n'
         f'Optimal weights: Stage 1 ({ensemble_summary["optimization"]["stage1_weight"]:.0%}) + '
         f'Stage 2 ({ensemble_summary["optimization"]["stage2_weight"]:.0%})\n'
         f'Grid search optimization on common OOF predictions\n'
         f'Final AUC: {ensemble_auc:.4f} (+{ensemble_summary["improvement"]["auc_improvement_percent"]:.2f}% improvement)',
         color_ensemble, text_size)

draw_arrow(ax, 2.7, y_common, 3.5, y_ensemble + 1.1, '')

# ============= FINAL OUTPUT =============
y_output = 0.8
draw_box(ax, 2.5, y_output, 5.0, 0.9,
         f'FINAL PREDICTIONS\n'
         f'Crisis probability at h=8 months | Optimal threshold selection\n'
         f'PR-AUC: {ensemble_summary["ensemble_performance"]["pr_auc"]:.3f} | '
         f'F1: {ensemble_summary["ensemble_performance"]["f1"]:.3f} | '
         f'Brier: {ensemble_summary["ensemble_performance"]["brier_score"]:.4f}',
         '#E8F5E9', text_size)

draw_arrow(ax, 5, y_ensemble, 5, y_output + 0.9, '')

# ============= LEGEND =============
legend_y = 0.2
legend_elements = [
    mpatches.Patch(facecolor=color_data, edgecolor='black', label='Raw Data'),
    mpatches.Patch(facecolor=color_stage0, edgecolor='black', label='Stage 0: Baseline'),
    mpatches.Patch(facecolor=color_stage1, edgecolor='black', label='Stage 1: AR Model'),
    mpatches.Patch(facecolor=color_filter, edgecolor='black', label='Filtering Logic'),
    mpatches.Patch(facecolor=color_stage2, edgecolor='black', label='Stage 2: Advanced ML'),
    mpatches.Patch(facecolor=color_ensemble, edgecolor='black', label='Ensemble'),
]
ax.legend(handles=legend_elements, loc='lower center', ncol=6,
          fontsize=text_size, frameon=True, fancybox=True)

# Add timestamp
ax.text(9.8, 0.1, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M")}',
        ha='right', fontsize=7, style='italic')

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'pipeline_architecture.png',
            dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: pipeline_architecture.png")

print("\n" + "=" * 80)
print("PIPELINE FLOW DIAGRAM COMPLETE")
print("=" * 80)
print(f"Output: {OUTPUT_DIR / 'pipeline_architecture.png'}")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
