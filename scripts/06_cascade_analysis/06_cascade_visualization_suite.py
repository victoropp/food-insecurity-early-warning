"""
Stage 3: Cascade Visualization Suite - Publication-Quality Figures
===================================================================
Generates publication-quality visualizations comparing AR baseline
with the improved cascade ensemble.

Figures:
1. Side-by-side confusion matrices
2. Precision-Recall comparison bar chart
3. Strategy comparison chart
4. Threshold optimization curve
5. Key saves by country (geographic analysis)

Author: Victor Collins Oppon
Date: December 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import sys
import warnings
warnings.filterwarnings('ignore')

import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns

# Set publication-quality defaults
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.family'] = 'sans-serif'

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import RESULTS_DIR

# Directories
INPUT_DIR = RESULTS_DIR / 'ensemble_improved'
FIGURES_DIR = RESULTS_DIR.parent / 'FIGURES' / 'cascade_comparison'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("CASCADE VISUALIZATION SUITE - PUBLICATION-QUALITY FIGURES")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output directory: {FIGURES_DIR}")
print()

# =============================================================================
# LOAD DATA
# =============================================================================

print("-" * 80)
print("Loading Data")
print("-" * 80)

with open(INPUT_DIR / 'improved_cascade_summary.json', 'r') as f:
    summary = json.load(f)

predictions = pd.read_csv(INPUT_DIR / 'improved_cascade_predictions.csv')
threshold_analysis = pd.read_csv(INPUT_DIR / 'threshold_cost_analysis.csv')

print(f"Loaded summary with {len(summary['all_strategies'])} strategies")
print()

# Extract key metrics
ar = summary['ar_baseline']
strategies = summary['all_strategies']

# =============================================================================
# FIGURE 1: Side-by-Side Confusion Matrices
# =============================================================================

print("-" * 80)
print("Figure 1: Confusion Matrices Comparison")
print("-" * 80)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Define confusion matrix data for each strategy
cm_data = [
    ('AR Baseline', [[ar['tn'], ar['fp']], [ar['fn'], ar['tp']]]),
    ('Adaptive Cascade', [
        [strategies['adaptive_threshold']['fn'] + strategies['adaptive_threshold']['n_overrides'] - (strategies['adaptive_threshold']['fn'] - ar['fn']),
         ar['fp'] + (strategies['adaptive_threshold']['n_overrides'] - (ar['fn'] - strategies['adaptive_threshold']['fn']))],
        [strategies['adaptive_threshold']['fn'],
         ar['tp'] + (ar['fn'] - strategies['adaptive_threshold']['fn'])]
    ]),
]

# Calculate actual confusion matrix values from strategy data
strategy_cms = {
    'AR Baseline': {
        'tn': ar['tn'], 'fp': ar['fp'], 'fn': ar['fn'], 'tp': ar['tp']
    },
    'Adaptive Cascade': {
        'tn': ar['tn'] - (strategies['adaptive_threshold']['n_overrides'] - (ar['fn'] - strategies['adaptive_threshold']['fn'])),
        'fp': ar['fp'] + (strategies['adaptive_threshold']['n_overrides'] - (ar['fn'] - strategies['adaptive_threshold']['fn'])),
        'fn': strategies['adaptive_threshold']['fn'],
        'tp': ar['tp'] + (ar['fn'] - strategies['adaptive_threshold']['fn'])
    },
    'Fixed Threshold (0.46)': {
        'tn': ar['tn'] - (strategies['fixed_low_threshold']['n_overrides'] - (ar['fn'] - strategies['fixed_low_threshold']['fn'])),
        'fp': ar['fp'] + (strategies['fixed_low_threshold']['n_overrides'] - (ar['fn'] - strategies['fixed_low_threshold']['fn'])),
        'fn': strategies['fixed_low_threshold']['fn'],
        'tp': ar['tp'] + (ar['fn'] - strategies['fixed_low_threshold']['fn'])
    }
}

# Plot confusion matrices
for idx, (name, cms) in enumerate(strategy_cms.items()):
    ax = axes[idx]
    cm = np.array([[cms['tn'], cms['fp']], [cms['fn'], cms['tp']]])

    # Custom color map
    cmap = sns.diverging_palette(220, 20, as_cmap=True)

    sns.heatmap(cm, annot=True, fmt=',d', cmap='Blues', ax=ax,
                xticklabels=['Pred: No Crisis', 'Pred: Crisis'],
                yticklabels=['Actual: No Crisis', 'Actual: Crisis'],
                annot_kws={'size': 14, 'weight': 'bold'})

    ax.set_title(f'{name}', fontsize=14, fontweight='bold')
    ax.set_xlabel('Predicted')
    ax.set_ylabel('Actual')

    # Highlight FN cell (missed crises)
    if name != 'AR Baseline':
        fn_change = ar['fn'] - cms['fn']
        if fn_change > 0:
            ax.text(0.5, 1.5, f'-{fn_change}', fontsize=10, color='green',
                    ha='center', va='top', transform=ax.transData)

plt.tight_layout()
fig.suptitle('Confusion Matrix Comparison: AR Baseline vs Cascade Ensembles',
             fontsize=14, fontweight='bold', y=1.02)

plt.savefig(FIGURES_DIR / 'fig1_confusion_matrices.png', bbox_inches='tight', dpi=300)
plt.savefig(FIGURES_DIR / 'fig1_confusion_matrices.pdf', bbox_inches='tight')
print(f"[OK] Saved: fig1_confusion_matrices.png/pdf")
plt.close()

# =============================================================================
# FIGURE 2: Precision-Recall-F1 Comparison Bar Chart
# =============================================================================

print("-" * 80)
print("Figure 2: Metrics Comparison Bar Chart")
print("-" * 80)

fig, ax = plt.subplots(figsize=(12, 6))

# Prepare data
metrics = ['Precision', 'Recall', 'F1 Score']
x = np.arange(len(metrics))
width = 0.25

# Values for each strategy
ar_values = [ar['precision'], ar['recall'], ar['f1']]
adaptive_values = [
    strategies['adaptive_threshold']['precision'],
    strategies['adaptive_threshold']['recall'],
    strategies['adaptive_threshold']['f1']
]
fixed_values = [
    strategies['fixed_low_threshold']['precision'],
    strategies['fixed_low_threshold']['recall'],
    strategies['fixed_low_threshold']['f1']
]

# Plot bars
bars1 = ax.bar(x - width, ar_values, width, label='AR Baseline', color='#2E86AB', edgecolor='black')
bars2 = ax.bar(x, adaptive_values, width, label='Adaptive Cascade', color='#28A745', edgecolor='black')
bars3 = ax.bar(x + width, fixed_values, width, label='Fixed Threshold (0.46)', color='#FFC107', edgecolor='black')

# Add value labels on bars with better formatting
def add_bar_labels(bars, values, color='black', fontweight='bold'):
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f'{val:.3f}',
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 5),
                    textcoords="offset points",
                    ha='center', va='bottom', fontsize=11, fontweight=fontweight,
                    color=color)

add_bar_labels(bars1, ar_values, color='#2E86AB')
add_bar_labels(bars2, adaptive_values, color='#28A745')
add_bar_labels(bars3, fixed_values, color='#B8860B')

ax.set_ylabel('Score', fontsize=12)
ax.set_title('Model Performance Comparison: Precision, Recall, and F1 Score',
             fontsize=14, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels(metrics, fontsize=12)
ax.legend(loc='upper right', fontsize=10)
ax.set_ylim(0, 1.0)
ax.axhline(y=0.7319, color='gray', linestyle='--', alpha=0.5, label='AR Baseline')
ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig2_metrics_comparison.png', bbox_inches='tight', dpi=300)
plt.savefig(FIGURES_DIR / 'fig2_metrics_comparison.pdf', bbox_inches='tight')
print(f"[OK] Saved: fig2_metrics_comparison.png/pdf")
plt.close()

# =============================================================================
# FIGURE 3: Strategy Comparison - FN Reduction vs Precision Trade-off
# =============================================================================

print("-" * 80)
print("Figure 3: Strategy Trade-off Analysis")
print("-" * 80)

fig, ax = plt.subplots(figsize=(10, 6))

# Strategy data
strategy_names = ['AR Baseline', 'Adaptive\nThreshold', 'Fixed\nThreshold (0.46)', 'Very High\nRecall']
fn_values = [ar['fn'], strategies['adaptive_threshold']['fn'],
             strategies['fixed_low_threshold']['fn'], strategies['very_high_recall']['fn']]
precision_values = [ar['precision'], strategies['adaptive_threshold']['precision'],
                    strategies['fixed_low_threshold']['precision'],
                    strategies['very_high_recall']['precision']]
overrides = [0, strategies['adaptive_threshold']['n_overrides'],
             strategies['fixed_low_threshold']['n_overrides'],
             strategies['very_high_recall']['n_overrides']]

# Create scatter plot
colors = ['#2E86AB', '#28A745', '#FFC107', '#DC3545']
sizes = [200 + o/5 for o in overrides]  # Size by overrides

for i, (name, fn, prec, over, color, size) in enumerate(zip(
    strategy_names, fn_values, precision_values, overrides, colors, sizes)):
    ax.scatter(fn, prec, s=size, c=color, alpha=0.8, edgecolor='black', linewidth=2)
    ax.annotate(name, (fn, prec), textcoords="offset points",
                xytext=(10, 10), ha='left', fontsize=10)

ax.set_xlabel('False Negatives (Missed Crises)', fontsize=12)
ax.set_ylabel('Precision', fontsize=12)
ax.set_title('Trade-off: False Negatives vs Precision\n(Circle size = number of overrides)',
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)

# Add arrow showing "better" direction
ax.annotate('', xy=(1200, 0.75), xytext=(1450, 0.68),
            arrowprops=dict(arrowstyle='->', color='green', lw=2))
ax.text(1325, 0.72, 'Better', fontsize=10, color='green', ha='center')

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig3_strategy_tradeoff.png', bbox_inches='tight', dpi=300)
plt.savefig(FIGURES_DIR / 'fig3_strategy_tradeoff.pdf', bbox_inches='tight')
print(f"[OK] Saved: fig3_strategy_tradeoff.png/pdf")
plt.close()

# =============================================================================
# FIGURE 4: Threshold Optimization Curve
# =============================================================================

print("-" * 80)
print("Figure 4: Threshold Optimization")
print("-" * 80)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Precision-Recall vs Threshold
ax1 = axes[0]
ax1.plot(threshold_analysis['threshold'], threshold_analysis['precision'],
         'b-', linewidth=2, label='Precision')
ax1.plot(threshold_analysis['threshold'], threshold_analysis['recall'],
         'r-', linewidth=2, label='Recall')
ax1.plot(threshold_analysis['threshold'], threshold_analysis['f1'],
         'g--', linewidth=2, label='F1 Score')

# Mark key thresholds
for thresh, name, color in [(0.46, 'Cost-Optimal', 'orange'),
                             (0.85, 'Original', 'purple')]:
    ax1.axvline(x=thresh, color=color, linestyle=':', alpha=0.7)
    ax1.text(thresh + 0.02, 0.9, name, rotation=90, fontsize=9, color=color)

ax1.axhline(y=ar['precision'], color='gray', linestyle='--', alpha=0.5)
ax1.text(0.1, ar['precision'] + 0.02, 'AR Baseline', fontsize=9, color='gray')

ax1.set_xlabel('Stage 2 Threshold', fontsize=12)
ax1.set_ylabel('Score', fontsize=12)
ax1.set_title('Precision/Recall vs Stage 2 Threshold', fontsize=13, fontweight='bold')
ax1.legend(loc='best')
ax1.grid(True, alpha=0.3)
ax1.set_xlim(0, 1)
ax1.set_ylim(0, 1)

# Right: Cost vs Threshold
ax2 = axes[1]
ax2.plot(threshold_analysis['threshold'], threshold_analysis['cost'],
         'k-', linewidth=2)
ax2.fill_between(threshold_analysis['threshold'], 0, threshold_analysis['cost'],
                 alpha=0.2)

# Mark optimal
optimal_idx = threshold_analysis['cost'].idxmin()
optimal_thresh = threshold_analysis.loc[optimal_idx, 'threshold']
optimal_cost = threshold_analysis.loc[optimal_idx, 'cost']
ax2.scatter([optimal_thresh], [optimal_cost], s=200, c='red', zorder=5,
            marker='*', edgecolor='black')
ax2.annotate(f'Optimal: {optimal_thresh:.2f}', (optimal_thresh, optimal_cost),
             textcoords="offset points", xytext=(20, 20), fontsize=10,
             arrowprops=dict(arrowstyle='->', color='red'))

ax2.set_xlabel('Stage 2 Threshold', fontsize=12)
ax2.set_ylabel('Cost (10×FN + 1×FP)', fontsize=12)
ax2.set_title('Humanitarian Cost Function\n(Missing crisis = 10× false alarm)',
              fontsize=13, fontweight='bold')
ax2.grid(True, alpha=0.3)
ax2.set_xlim(0, 1)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig4_threshold_optimization.png', bbox_inches='tight', dpi=300)
plt.savefig(FIGURES_DIR / 'fig4_threshold_optimization.pdf', bbox_inches='tight')
print(f"[OK] Saved: fig4_threshold_optimization.png/pdf")
plt.close()

# =============================================================================
# FIGURE 5: Key Saves by Country
# =============================================================================

print("-" * 80)
print("Figure 5: Key Saves by Country")
print("-" * 80)

# Load key saves
key_saves_by_country = summary['key_saves']['by_country']

fig, ax = plt.subplots(figsize=(12, 6))

countries = list(key_saves_by_country.keys())
counts = list(key_saves_by_country.values())

# Sort by count
sorted_pairs = sorted(zip(countries, counts), key=lambda x: x[1], reverse=True)
countries, counts = zip(*sorted_pairs)

# Create horizontal bar chart
colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(countries)))
bars = ax.barh(range(len(countries)), counts, color=colors, edgecolor='black')

ax.set_yticks(range(len(countries)))
ax.set_yticklabels(countries, fontsize=11)
ax.set_xlabel('Number of Crises Caught', fontsize=12)
ax.set_title('Additional Crises Detected by Cascade Ensemble\n(Cases AR Missed But Ensemble Caught)',
             fontsize=14, fontweight='bold')

# Add value labels with bold formatting
for bar, count in zip(bars, counts):
    ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
            f'{count}', ha='left', va='center', fontsize=12, fontweight='bold')

ax.set_xlim(0, max(counts) * 1.2)
ax.invert_yaxis()
ax.grid(axis='x', alpha=0.3)

plt.tight_layout()
plt.savefig(FIGURES_DIR / 'fig5_key_saves_by_country.png', bbox_inches='tight', dpi=300)
plt.savefig(FIGURES_DIR / 'fig5_key_saves_by_country.pdf', bbox_inches='tight')
print(f"[OK] Saved: fig5_key_saves_by_country.png/pdf")
plt.close()

# =============================================================================
# FIGURE 6: Summary Dashboard
# =============================================================================

print("-" * 80)
print("Figure 6: Summary Dashboard")
print("-" * 80)

fig = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

# Title
fig.suptitle('Cascade Ensemble Performance Summary\nFood Insecurity Early Warning System',
             fontsize=16, fontweight='bold', y=0.98)

# Panel 1: Key Metrics Comparison
ax1 = fig.add_subplot(gs[0, 0])
metrics = ['Precision', 'Recall', 'F1']
ar_vals = [ar['precision'], ar['recall'], ar['f1']]
adaptive_vals = [strategies['adaptive_threshold']['precision'],
                 strategies['adaptive_threshold']['recall'],
                 strategies['adaptive_threshold']['f1']]

x = np.arange(len(metrics))
width = 0.35
bars_ar = ax1.bar(x - width/2, ar_vals, width, label='AR Baseline', color='#2E86AB')
bars_ad = ax1.bar(x + width/2, adaptive_vals, width, label='Adaptive Cascade', color='#28A745')

# Add value labels
for bar, val in zip(bars_ar, ar_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#2E86AB')
for bar, val in zip(bars_ad, adaptive_vals):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold', color='#28A745')

ax1.set_ylabel('Score')
ax1.set_title('Metrics Comparison', fontweight='bold')
ax1.set_xticks(x)
ax1.set_xticklabels(metrics)
ax1.legend(loc='lower right', fontsize=8)
ax1.set_ylim(0.6, 0.85)
ax1.grid(axis='y', alpha=0.3)

# Panel 2: False Negatives Reduction
ax2 = fig.add_subplot(gs[0, 1])
fn_data = [ar['fn'], strategies['adaptive_threshold']['fn']]
bars_fn = ax2.bar(['AR Baseline', 'Adaptive Cascade'], fn_data,
               color=['#2E86AB', '#28A745'], edgecolor='black')
ax2.set_ylabel('False Negatives (Missed Crises)')
ax2.set_title('Missed Crises Reduction', fontweight='bold')

# Add value labels on bars
for bar, val in zip(bars_fn, fn_data):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 20,
             f'{val:,}', ha='center', va='bottom', fontsize=12, fontweight='bold')

reduction = ar['fn'] - strategies['adaptive_threshold']['fn']
ax2.annotate(f'-{reduction} crises\n({100*reduction/ar["fn"]:.1f}% reduction)',
             xy=(1, fn_data[1]), xytext=(1, fn_data[1] + 150),
             ha='center', fontsize=11, color='green', fontweight='bold',
             arrowprops=dict(arrowstyle='->', color='green', lw=1.5))
ax2.grid(axis='y', alpha=0.3)

# Panel 3: Override Analysis
ax3 = fig.add_subplot(gs[0, 2])
override_correct = ar['fn'] - strategies['adaptive_threshold']['fn']
override_incorrect = strategies['adaptive_threshold']['n_overrides'] - override_correct
ax3.pie([override_correct, override_incorrect],
        labels=[f'Correct\n({override_correct})', f'Incorrect\n({override_incorrect})'],
        colors=['#28A745', '#DC3545'], autopct='%1.1f%%',
        explode=[0.05, 0], startangle=90)
ax3.set_title(f'Override Quality\n({strategies["adaptive_threshold"]["n_overrides"]} total overrides)',
              fontweight='bold')

# Panel 4: Country Distribution
ax4 = fig.add_subplot(gs[1, 0:2])
top_countries = dict(list(sorted(key_saves_by_country.items(),
                                  key=lambda x: x[1], reverse=True))[:8])
bars_country = ax4.barh(list(top_countries.keys()), list(top_countries.values()),
         color=plt.cm.Blues(np.linspace(0.4, 0.8, len(top_countries))))

# Add value labels on bars
for bar, val in zip(bars_country, top_countries.values()):
    ax4.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
             f'{val}', ha='left', va='center', fontsize=11, fontweight='bold')

ax4.set_xlabel('Crises Caught')
ax4.set_title('Key Saves by Country (Top 8)', fontweight='bold')
ax4.invert_yaxis()
ax4.grid(axis='x', alpha=0.3)
ax4.set_xlim(0, max(top_countries.values()) * 1.15)

# Panel 5: Key Statistics Text
ax5 = fig.add_subplot(gs[1, 2])
ax5.axis('off')
stats_text = f"""
KEY STATISTICS

Total Observations: {summary['data']['total_observations']:,}
Crisis Events: {summary['data']['crisis_events']:,} ({100*summary['data']['crisis_rate']:.1f}%)

AR BASELINE:
  - Precision: {ar['precision']:.4f}
  - Recall: {ar['recall']:.4f}
  - F1 Score: {ar['f1']:.4f}
  - Missed Crises: {ar['fn']:,}

ADAPTIVE CASCADE:
  - Precision: {strategies['adaptive_threshold']['precision']:.4f}
  - Recall: {strategies['adaptive_threshold']['recall']:.4f}
  - F1 Score: {strategies['adaptive_threshold']['f1']:.4f}
  - Missed Crises: {strategies['adaptive_threshold']['fn']:,}
  - Overrides: {strategies['adaptive_threshold']['n_overrides']}

IMPROVEMENT:
  - Recall: +{100*(strategies['adaptive_threshold']['recall'] - ar['recall']):.2f}%
  - FN Reduction: {ar['fn'] - strategies['adaptive_threshold']['fn']:,}
  - Additional Detections: {ar['fn'] - strategies['adaptive_threshold']['fn']:,}
"""
ax5.text(0.1, 0.95, stats_text, transform=ax5.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.5))

plt.savefig(FIGURES_DIR / 'fig6_summary_dashboard.png', bbox_inches='tight', dpi=300)
plt.savefig(FIGURES_DIR / 'fig6_summary_dashboard.pdf', bbox_inches='tight')
print(f"[OK] Saved: fig6_summary_dashboard.png/pdf")
plt.close()

# =============================================================================
# SUMMARY
# =============================================================================

print()
print("=" * 80)
print("VISUALIZATION SUITE COMPLETE")
print("=" * 80)
print()
print(f"Generated 6 publication-quality figures:")
print(f"  1. Confusion matrices comparison")
print(f"  2. Precision-Recall-F1 bar chart")
print(f"  3. Strategy trade-off analysis")
print(f"  4. Threshold optimization curves")
print(f"  5. Key saves by country")
print(f"  6. Summary dashboard")
print()
print(f"All figures saved in PNG (300 DPI) and PDF formats")
print(f"Output directory: {FIGURES_DIR}")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
