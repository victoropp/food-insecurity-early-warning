"""
================================================================================
PUBLICATION-GRADE VISUAL STORY: Complete Pipeline Performance
================================================================================
Creates a comprehensive visual narrative showing the progression from baseline
to ensemble across all modeling approaches.

Author: Victor Collins Oppon
Date: December 2025
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, ConnectionPatch
import seaborn as sns
from pathlib import Path
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# Import configuration
sys.path.append(str(Path(__file__).parent.parent))
from config import RESULTS_DIR, FIGURES_DIR

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = FIGURES_DIR / "publication"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

print("=" * 80)
print("PUBLICATION-GRADE PIPELINE STORY")
print("=" * 80)
print(f"\nOutput directory: {OUTPUT_DIR}\n")

# ============================================================================
# LOAD ALL MODEL RESULTS
# ============================================================================

print("Loading model results...")

# Load comparison results (already has all models)
comparison_file = RESULTS_DIR / 'baseline_comparison' / 'ar_vs_literature_comparison.csv'
comparison_df = pd.read_csv(comparison_file)

# Load ensemble details - UPDATED TO USE CORRECT FILE
ensemble_file = RESULTS_DIR / 'ensemble_stage1_stage2' / 'ensemble_summary.json'
with open(ensemble_file) as f:
    ensemble_data = json.load(f)

print(f"Loaded {len(comparison_df)} models\n")

# ============================================================================
# CREATE PUBLICATION-GRADE STORY VISUALIZATION
# ============================================================================

print("Creating publication-grade story visualization...")

# Create figure with 3 panels
fig = plt.figure(figsize=(16, 10), dpi=300)
gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.5, 0.8], width_ratios=[1, 1],
                      hspace=0.35, wspace=0.3)

# ============================================================================
# PANEL A: THE PIPELINE PROGRESSION (Top - Full Width)
# ============================================================================

ax_progression = fig.add_subplot(gs[0, :])

# Group models by category
# Use Literature Baseline WITH Location as the representative baseline
literature_with_loc = comparison_df[comparison_df['model'] == 'Literature Baseline (With Location)']
literature_no_loc = comparison_df[comparison_df['model'] == 'Literature Baseline (No Location)']
ar_baseline = comparison_df[comparison_df['model'] == 'Stage 1 AR Baseline'].iloc[0]
stage2_models = comparison_df[comparison_df['model'].str.contains('Stage 2')].copy()
ensemble = comparison_df[comparison_df['model'].str.contains('Ensemble')].iloc[0]

# Use with-location variant if available, otherwise fall back
if len(literature_with_loc) > 0:
    literature = literature_with_loc.iloc[0]
elif len(literature_no_loc) > 0:
    literature = literature_no_loc.iloc[0]
else:
    # Fallback for any Literature Baseline
    literature = comparison_df[comparison_df['model'].str.contains('Literature')].iloc[0]

# Create progression bars
categories = ['Literature\nBaseline', 'AR Baseline\n(Stage 1)', 'Best Stage 2\n(XGBoost)', 'Two-Stage\nEnsemble']
auc_values = [
    literature['auc_mean'],
    ar_baseline['auc_mean'],
    stage2_models['auc_mean'].max(),  # Best Stage 2
    ensemble['auc_mean']
]
# Professional 3-color scheme matching Panel B
colors_prog = ['#4A90E2', '#4A90E2', '#50C878', '#7B68BE']  # Blue, Blue, Green, Purple

bars = ax_progression.bar(range(len(categories)), auc_values, color=colors_prog,
                          alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels on bars
for i, (bar, val) in enumerate(zip(bars, auc_values)):
    height = bar.get_height()
    ax_progression.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                       f'{val:.3f}',
                       ha='center', va='bottom', fontsize=14, fontweight='bold')

# Add improvement arrows
for i in range(len(categories)-1):
    improvement = auc_values[i+1] - auc_values[i]
    pct_improvement = (improvement / auc_values[i]) * 100

    # Arrow
    ax_progression.annotate('', xy=(i+1, auc_values[i+1]-0.05),
                          xytext=(i, auc_values[i]+0.05),
                          arrowprops=dict(arrowstyle='->', lw=2, color='#34495E'))

    # Improvement text
    mid_x = i + 0.5
    mid_y = (auc_values[i] + auc_values[i+1]) / 2
    ax_progression.text(mid_x, mid_y + 0.1, f'+{improvement:.3f}\n({pct_improvement:+.1f}%)',
                       ha='center', va='center', fontsize=9,
                       bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                               edgecolor='#34495E', linewidth=1.5))

ax_progression.set_xticks(range(len(categories)))
ax_progression.set_xticklabels(categories, fontsize=12, fontweight='bold')
ax_progression.set_ylabel('AUC-ROC', fontsize=13, fontweight='bold')
ax_progression.set_title('A. Pipeline Progression: From Baseline to Ensemble',
                        fontsize=15, fontweight='bold', pad=20)
ax_progression.set_ylim(0, 1.05)
ax_progression.grid(axis='y', alpha=0.3, linestyle='--')

# Add horizontal reference lines
ax_progression.axhline(y=0.7, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax_progression.axhline(y=0.8, color='gray', linestyle=':', alpha=0.5, linewidth=1)
ax_progression.axhline(y=0.9, color='gray', linestyle=':', alpha=0.5, linewidth=1)

# Add legend for Panel A
legend_elements_prog = [
    mpatches.Patch(facecolor='#4A90E2', edgecolor='black', label='Baseline Models'),
    mpatches.Patch(facecolor='#50C878', edgecolor='black', label='Stage 2 Models'),
    mpatches.Patch(facecolor='#7B68BE', edgecolor='black', label='Ensemble')
]
ax_progression.legend(handles=legend_elements_prog, loc='upper left', frameon=True,
                     fontsize=10, edgecolor='black')

# ============================================================================
# PANEL B: ALL MODELS COMPARISON (Middle Left)
# ============================================================================

ax_all = fig.add_subplot(gs[1, 0])

# Sort by AUC descending
plot_df = comparison_df.sort_values('auc_mean', ascending=True).copy()

# Shorten model names for better display
plot_df['model_short'] = plot_df['model'].replace({
    'Two-Stage Ensemble (Stage 1 + 2)': 'Ensemble',
    'Stage 1 AR Baseline': 'AR Baseline',
    'Stage 2 XGBoost Advanced': 'XGB Advanced',
    'Stage 2 XGBoost Basic': 'XGB Basic',
    'Stage 2 Mixed-Effects Z-Score': 'ME Z-Score',
    'Stage 2 Mixed-Effects Ratio': 'ME Ratio',
    'Stage 2 Mixed-Effects Z-Score + HMM + DMD': 'ME Z + HMM/DMD',
    'Stage 2 Mixed-Effects Ratio + HMM + DMD': 'ME R + HMM/DMD',
    'Literature Baseline': 'Literature'
})

# Professional 3-color scheme: Blue=baselines, Green=Stage2, Purple=ensemble
colors_all = []
for model in plot_df['model']:
    if 'Ensemble' in model:
        colors_all.append('#7B68BE')  # Purple for ensemble
    elif 'Literature' in model or 'AR Baseline' in model:
        colors_all.append('#4A90E2')  # Blue for baselines
    else:  # All Stage 2 models
        colors_all.append('#50C878')  # Green for Stage 2

y_pos = np.arange(len(plot_df))
bars_all = ax_all.barh(y_pos, plot_df['auc_mean'],
                       color=colors_all, alpha=0.85, edgecolor='black', linewidth=0.8)

# Add value labels with AUC values only (no std to reduce clutter)
for i, (idx, row) in enumerate(plot_df.iterrows()):
    label_x = row['auc_mean'] + 0.01
    label = f"{row['auc_mean']:.3f}"
    ax_all.text(label_x, i, label, va='center', fontsize=9, fontweight='bold')

ax_all.set_yticks(y_pos)
ax_all.set_yticklabels(plot_df['model_short'], fontsize=9)
ax_all.set_xlabel('AUC-ROC', fontsize=11, fontweight='bold')
ax_all.set_title(f'B. Complete Model Comparison (All {len(plot_df)} Models)',
                fontsize=13, fontweight='bold', pad=15)
ax_all.set_xlim(0, max(plot_df['auc_mean']) * 1.12)
ax_all.grid(axis='x', alpha=0.3, linestyle='--')

# Add legend for Panel B
legend_elements_all = [
    mpatches.Patch(facecolor='#4A90E2', edgecolor='black', label='Baseline Models'),
    mpatches.Patch(facecolor='#50C878', edgecolor='black', label='Stage 2 Models'),
    mpatches.Patch(facecolor='#7B68BE', edgecolor='black', label='Ensemble')
]
ax_all.legend(handles=legend_elements_all, loc='lower right', frameon=True,
             fontsize=9, edgecolor='black')

# ============================================================================
# PANEL C: MODEL FAMILIES PERFORMANCE (Middle Right)
# ============================================================================

ax_families = fig.add_subplot(gs[1, 1])

# Group by model family - SEPARATE AR from Literature baselines
families = {
    'Literature\nBaseline': comparison_df[comparison_df['model'].str.contains('Literature')]['auc_mean'].values,
    'AR Baseline\n(Stage 1)': comparison_df[comparison_df['model'] == 'Stage 1 AR Baseline']['auc_mean'].values,
    'XGBoost\n(Stage 2)': comparison_df[comparison_df['model'].str.contains('XGBoost')]['auc_mean'].values,
    'Mixed-Effects\n(Stage 2)': comparison_df[comparison_df['model'].str.contains('Mixed-Effects')]['auc_mean'].values,
    'Ensemble': comparison_df[comparison_df['model'].str.contains('Ensemble')]['auc_mean'].values
}

family_means = {k: v.mean() for k, v in families.items()}
family_maxs = {k: v.max() for k, v in families.items()}
family_mins = {k: v.min() for k, v in families.items()}

x_fam = np.arange(len(families))
family_names = list(families.keys())
means = [family_means[f] for f in family_names]
maxs = [family_maxs[f] for f in family_names]
mins = [family_mins[f] for f in family_names]

# Color scheme: Light blue=Literature, Dark blue=AR, Green=Stage2, Purple=ensemble
colors_fam = ['#87CEEB', '#4A90E2', '#50C878', '#50C878', '#7B68BE']

# Plot ranges
for i, fname in enumerate(family_names):
    if len(families[fname]) > 1:
        ax_families.plot([i, i], [mins[i], maxs[i]], color=colors_fam[i],
                        linewidth=10, alpha=0.4, solid_capstyle='round')
        ax_families.scatter(i, means[i], color=colors_fam[i], s=180,
                          edgecolor='black', linewidth=2, zorder=3)
    else:
        ax_families.scatter(i, means[i], color=colors_fam[i], s=180,
                          edgecolor='black', linewidth=2, zorder=3)

# Add clean value labels - just the key values
for i, fname in enumerate(family_names):
    n_models = len(families[fname])
    if n_models > 1:
        # Only show max and min values, positioned clearly
        ax_families.text(i, maxs[i] + 0.02, f'{maxs[i]:.3f}', ha='center',
                        fontsize=9, fontweight='bold')
        ax_families.text(i, mins[i] - 0.02, f'{mins[i]:.3f}', ha='center',
                        fontsize=9, fontweight='bold')
    else:
        # Single value - show above the point
        ax_families.text(i, means[i] + 0.03, f'{means[i]:.3f}', ha='center',
                        fontsize=10, fontweight='bold')

ax_families.set_xticks(x_fam)
ax_families.set_xticklabels(family_names, fontsize=9, fontweight='bold')
ax_families.set_ylabel('AUC-ROC', fontsize=11, fontweight='bold')
ax_families.set_title('C. Model Family Performance Ranges',
                     fontsize=13, fontweight='bold', pad=15)
ax_families.set_ylim(0.4, 1.05)
ax_families.grid(axis='y', alpha=0.3, linestyle='--')

# Add legend for Panel C
legend_elements_fam = [
    mpatches.Patch(facecolor='#87CEEB', edgecolor='black', label='Literature Baseline'),
    mpatches.Patch(facecolor='#4A90E2', edgecolor='black', label='AR Baseline'),
    mpatches.Patch(facecolor='#50C878', edgecolor='black', label='Stage 2 Models'),
    mpatches.Patch(facecolor='#7B68BE', edgecolor='black', label='Ensemble')
]
ax_families.legend(handles=legend_elements_fam, loc='upper left', frameon=True,
                  fontsize=8, edgecolor='black')

# ============================================================================
# PANEL D: KEY FINDINGS (Bottom - Full Width)
# ============================================================================

ax_findings = fig.add_subplot(gs[2, :])
ax_findings.axis('off')

# Calculate key statistics from actual results
lit_auc = literature['auc_mean']
ar_auc = ar_baseline['auc_mean']
best_stage2_auc = stage2_models['auc_mean'].max()
ensemble_auc = ensemble['auc_mean']

# Get ensemble alpha from actual results
ensemble_alpha = ensemble_data.get('optimal_alpha', ensemble_data.get('alpha', 0.5))
ar_weight_pct = ensemble_alpha * 100
stage2_weight_pct = (1 - ensemble_alpha) * 100

improvement_ar_over_lit = ((ar_auc - lit_auc) / lit_auc) * 100
improvement_stage2_over_ar = ((best_stage2_auc - ar_auc) / ar_auc) * 100
improvement_ensemble_over_ar = ((ensemble_auc - ar_auc) / ar_auc) * 100
improvement_ensemble_over_stage2 = ((ensemble_auc - best_stage2_auc) / best_stage2_auc) * 100

# Get best XGBoost model name and feature count
best_xgb_model = stage2_models.loc[stage2_models['auc_mean'].idxmax()]
best_xgb_name = best_xgb_model['model']
best_xgb_features = best_xgb_model.get('n_features', 'N/A')

# Get ME model performance range
me_models = stage2_models[stage2_models['model'].str.contains('Mixed-Effects')]
me_min_auc = me_models['auc_mean'].min()
me_max_auc = me_models['auc_mean'].max()

# Get XGBoost Basic vs Advanced comparison
xgb_basic = comparison_df[comparison_df['model'] == 'Stage 2 XGBoost Basic']
xgb_advanced = comparison_df[comparison_df['model'] == 'Stage 2 XGBoost Advanced']
if len(xgb_basic) > 0 and len(xgb_advanced) > 0:
    xgb_basic_auc = xgb_basic['auc_mean'].values[0]
    xgb_advanced_auc = xgb_advanced['auc_mean'].values[0]
    hmm_dmd_impact = ((xgb_advanced_auc - xgb_basic_auc) / xgb_basic_auc) * 100
else:
    hmm_dmd_impact = 0.0

# Calculate XGBoost vs ME improvement
xgb_vs_me_improvement = ((best_stage2_auc - me_max_auc) / me_max_auc) * 100

findings_text = f"""
KEY FINDINGS:

1. BASELINE COMPARISON:
   • Literature Baseline: AUC = {lit_auc:.3f} (classic news-only approach)
   • AR Baseline: AUC = {ar_auc:.3f} (spatial-temporal autoregressive)
   • Improvement: {improvement_ar_over_lit:+.1f}% → AR baseline dominates naive approaches

2. STAGE 2 MODELS (News Features on AR-Filtered Data):
   • Best XGBoost: AUC = {best_stage2_auc:.3f} ({best_xgb_features} features)
   • Mixed-Effects: AUC range = {me_min_auc:.3f} - {me_max_auc:.3f}
   • XGBoost outperforms Mixed-Effects by {xgb_vs_me_improvement:.1f}%

3. ENSEMBLE ACHIEVES BEST PERFORMANCE:
   • Two-Stage Ensemble: AUC = {ensemble_auc:.3f} (α = {ensemble_alpha:.2f})
   • Improvement over AR baseline: {improvement_ensemble_over_ar:+.1f}%
   • Improvement over best Stage 2: {improvement_ensemble_over_stage2:+.1f}%
   • Optimal weighting: {ar_weight_pct:.0f}% AR + {stage2_weight_pct:.0f}% XGBoost

4. FEATURE ENGINEERING INSIGHTS:
   • HMM/DMD features in XGBoost: {hmm_dmd_impact:+.2f}% change
   • Mixed-Effects with HMM/DMD: Performance DECREASES (ensemble methods better for complex features)
   • Z-score features outperform ratio features in Mixed-Effects models
"""

ax_findings.text(0.05, 0.95, findings_text, transform=ax_findings.transAxes,
                fontsize=9, verticalalignment='top', family='monospace',
                bbox=dict(boxstyle='round,pad=1', facecolor='#F8F9FA',
                         edgecolor='#34495E', linewidth=2))

# ============================================================================
# OVERALL TITLE
# ============================================================================

fig.suptitle('From Baseline to Ensemble: A Complete Pipeline Story',
            fontsize=18, fontweight='bold', y=0.98)

plt.tight_layout(rect=[0, 0.01, 1, 0.97])

# ============================================================================
# SAVE FIGURE
# ============================================================================

output_file = OUTPUT_DIR / 'pipeline_story_complete.png'
plt.savefig(output_file, dpi=300, bbox_inches='tight', facecolor='white')
print(f"\n[OK] Saved: {output_file}")

plt.close()

print("\n" + "=" * 80)
print("PUBLICATION STORY COMPLETE")
print("=" * 80)
print(f"\nFigure saved to: {output_file}")
print("\nStory highlights:")
print(f"  - Literature -> AR: {improvement_ar_over_lit:+.1f}% improvement")
print(f"  - AR -> Best Stage 2: {improvement_stage2_over_ar:+.1f}% change")
print(f"  - Best Stage 2 -> Ensemble: {improvement_ensemble_over_stage2:+.1f}% improvement")
print(f"  - Overall (Literature -> Ensemble): {((ensemble_auc - lit_auc) / lit_auc * 100):+.1f}% improvement")
