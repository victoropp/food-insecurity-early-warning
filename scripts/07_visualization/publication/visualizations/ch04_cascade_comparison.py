"""
Figure 16: Cascade vs AR Baseline Performance Comparison
Publication-grade visualization showing precision-recall tradeoff and key saves

Date: 2026-01-05
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import json
from config import BASE_DIR

# Directories
BASE_DIR = Path(rstr(BASE_DIR))
DATA_DIR = BASE_DIR / "Dissertation Write Up" / "DATA_INVENTORIES"
OUTPUT_DIR = BASE_DIR / "Dissertation Write Up" / "LATEX_DISSERTATION" / "figures" / "ch04_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load cascade data from JSON
print("Loading cascade performance data...")
with open(DATA_DIR / "MASTER_METRICS_ALL_MODELS.json", 'r') as f:
    data = json.load(f)

cascade_prod = data['cascade']['production']
ar_baseline_h8 = data['ar_baseline']['h8']

# Extract metrics
ar_precision = cascade_prod['ar_precision']
ar_recall = cascade_prod['ar_recall']
ar_f1 = cascade_prod['ar_f1']
ar_tp = cascade_prod['ar_tp']
ar_fn = cascade_prod['ar_fn']

cascade_precision = cascade_prod['cascade_precision']
cascade_recall = cascade_prod['cascade_recall']
cascade_f1 = cascade_prod['cascade_f1']
cascade_tp = cascade_prod['cascade_tp']
cascade_fn = cascade_prod['cascade_fn']

key_saves = cascade_prod['key_saves']
key_save_rate = cascade_prod['key_save_rate']
recall_improvement = cascade_prod['recall_improvement']
precision_change = cascade_prod['precision_change']

print(f"\nAR Baseline: Precision={ar_precision:.4f}, Recall={ar_recall:.4f}, F1={ar_f1:.4f}")
print(f"Cascade: Precision={cascade_precision:.4f}, Recall={cascade_recall:.4f}, F1={cascade_f1:.4f}")
print(f"Key saves: {key_saves} ({key_save_rate:.1%})")
print(f"Recall improvement: +{recall_improvement:.4f} ({recall_improvement*100:.1f}pp)")
print(f"Precision change: {precision_change:.4f} ({precision_change*100:.1f}pp)")

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# Create figure with 2x2 layout
fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(14, 10))

# Panel 1: Precision-Recall Comparison
models = ['AR Baseline', 'Cascade']
precision_vals = [ar_precision, cascade_precision]
recall_vals = [ar_recall, cascade_recall]

x = np.arange(len(models))
width = 0.35

bars1 = ax1.bar(x - width/2, precision_vals, width, label='Precision',
                color='#3498DB', alpha=0.8, edgecolor='black', linewidth=2)
bars2 = ax1.bar(x + width/2, recall_vals, width, label='Recall',
                color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels
for bars in [bars1, bars2]:
    for bar in bars:
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.015,
                f'{height:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax1.set_xticks(x)
ax1.set_xticklabels(models, fontsize=12, fontweight='bold')
ax1.set_ylabel('Score', fontsize=12, fontweight='bold')
ax1.set_title('Panel A: Precision-Recall Tradeoff', fontsize=13, fontweight='bold', pad=10)
ax1.legend(fontsize=10, loc='upper left')
ax1.set_ylim(0, 0.9)
ax1.grid(axis='y', alpha=0.3, linestyle='--')

# Panel 2: Confusion Matrix Comparison
conf_data = {
    'AR Baseline': [ar_tp, ar_fn],
    'Cascade': [cascade_tp, cascade_fn]
}

x_pos = np.arange(len(models))
tp_bars = ax2.bar(x_pos - width/2, [ar_tp, cascade_tp], width, label='True Positives (TP)',
                  color='#27AE60', alpha=0.8, edgecolor='black', linewidth=2)
fn_bars = ax2.bar(x_pos + width/2, [ar_fn, cascade_fn], width, label='False Negatives (FN)',
                  color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=2)

# Add value labels
for bars in [tp_bars, fn_bars]:
    for bar in bars:
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 30,
                f'{int(height):,}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_xticks(x_pos)
ax2.set_xticklabels(models, fontsize=12, fontweight='bold')
ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
ax2.set_title('Panel B: True Positives vs False Negatives', fontsize=13, fontweight='bold', pad=10)
ax2.legend(fontsize=10, loc='center right')
ax2.grid(axis='y', alpha=0.3, linestyle='--')

# Panel 3: Key Saves Highlight
# Show the 249 key saves as the delta
delta_tp = cascade_tp - ar_tp
delta_fn = ar_fn - cascade_fn

categories = ['TP\nGain', 'FN\nReduction']
values = [delta_tp, delta_fn]
colors_delta = ['#27AE60', '#E74C3C']

bars = ax3.bar(categories, values, color=colors_delta, alpha=0.8,
               edgecolor='black', linewidth=2, width=0.6)

for bar, val in zip(bars, values):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height + 5,
            f'+{int(val):,}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax3.set_ylabel('Count', fontsize=12, fontweight='bold')
ax3.set_title(f'Panel C: Cascade Improvement ({key_saves} Key Saves)', fontsize=13, fontweight='bold', pad=10)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.set_ylim(0, max(values) * 1.2)

# Add annotation
ax3.text(0.5, 0.95, f'{key_save_rate:.1%} of AR failures rescued', transform=ax3.transAxes,
        ha='center', va='top', fontsize=11, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='gold', alpha=0.7, edgecolor='darkgoldenrod', linewidth=2))

# Panel 4: F1 Score Comparison
f1_vals = [ar_f1, cascade_f1]
bars = ax4.bar(models, f1_vals, color=['#3498DB', '#9B59B6'], alpha=0.8,
              edgecolor='black', linewidth=2, width=0.5)

for bar, val in zip(bars, f1_vals):
    height = bar.get_height()
    ax4.text(bar.get_x() + bar.get_width()/2., height + 0.01,
            f'{val:.3f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

ax4.set_ylabel('F1 Score', fontsize=12, fontweight='bold')
ax4.set_title('Panel D: F1 Score Comparison', fontsize=13, fontweight='bold', pad=10)
ax4.set_ylim(0, 0.8)
ax4.grid(axis='y', alpha=0.3, linestyle='--')

# Add change annotation
f1_change = cascade_f1 - ar_f1
change_text = f'F1 change: {f1_change:+.3f}\n({f1_change*100:+.1f}pp)'
ax4.text(0.5, 0.95, change_text, transform=ax4.transAxes,
        ha='center', va='top', fontsize=10, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.6', facecolor='lightblue', alpha=0.7, edgecolor='steelblue', linewidth=1.5))

# Overall title
fig.suptitle('Cascade vs AR Baseline Performance Comparison (h=8 months, n=20,722)',
            fontsize=15, fontweight='bold', y=0.995)

# Footer - emphasizing success
footer_text = (
    f"Cascade framework successfully rescues {key_saves} AR failures ({key_save_rate:.1%} rescue rate) "
    f"through targeted Stage 2 intervention on uncertain cases. "
    f"Recall improves +{recall_improvement*100:.1f}pp (AR: {ar_recall:.3f} → Cascade: {cascade_recall:.3f}), "
    f"capturing {delta_tp:,} additional crises 8 months in advance—rapid-onset shocks where persistence models fail. "
    f"Precision trades off {abs(precision_change)*100:.1f}pp (AR: {ar_precision:.3f} → Cascade: {cascade_precision:.3f}), "
    f"but these {key_saves} key saves target the most critical cases: "
    f"conflict escalations, economic collapses, displacement crises in Zimbabwe, Sudan, DRC. "
    f"Success on hardest cases where early warning matters most. "
    f"5-fold stratified spatial cross-validation, h=8 months forecast horizon."
)
fig.text(0.5, 0.01, footer_text, ha='center', fontsize=8, style='italic',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3),
         wrap=True)

plt.tight_layout(rect=[0, 0.06, 1, 0.99])

# Save
output_file = OUTPUT_DIR / "ch04_cascade_comparison.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file}")

output_file_png = OUTPUT_DIR / "ch04_cascade_comparison.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png}")

plt.close()

print("\n" + "="*80)
print("FIGURE 16 COMPLETE: Cascade vs AR Baseline Comparison")
print("="*80)
print(f"Key saves: {key_saves} ({key_save_rate:.1%})")
print(f"Recall improvement: +{recall_improvement*100:.1f}pp")
print(f"Precision change: {precision_change*100:.1f}pp")
