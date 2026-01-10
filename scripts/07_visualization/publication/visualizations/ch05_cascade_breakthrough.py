"""
Figure 23: Cascade Breakthrough - Success on Hardest Cases
REDESIGNED: Clean, readable flow diagram with NO overlapping text
Simple horizontal flow with clear separation

Date: 2026-01-05
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from config import BASE_DIR
warnings.filterwarnings('ignore')

# Directories
BASE_DIR = Path(rstr(BASE_DIR))
CASCADE_DIR = BASE_DIR / "RESULTS" / "cascade_optimized_production"
OUTPUT_DIR = BASE_DIR / "Dissertation Write Up" / "LATEX_DISSERTATION" / "figures" / "ch05_discussion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load cascade production data
print("Loading cascade data...")
df = pd.read_csv(CASCADE_DIR / "cascade_optimized_predictions.csv")

# Calculate key metrics
total_crises = int(df['y_true'].sum())
ar_tp = int(((df['ar_pred'] == 1) & (df['y_true'] == 1)).sum())
ar_fn = int(((df['ar_pred'] == 0) & (df['y_true'] == 1)).sum())
key_saves = int(((df['ar_pred'] == 0) & (df['y_true'] == 1) & (df['cascade_pred'] == 1)).sum())
still_missed = int(((df['ar_pred'] == 0) & (df['y_true'] == 1) & (df['cascade_pred'] == 0)).sum())

# Get top 3 countries
key_saves_df = df[(df['ar_pred'] == 0) & (df['y_true'] == 1) & (df['cascade_pred'] == 1)]
top_countries = key_saves_df.groupby('ipc_country').size().sort_values(ascending=False).head(3)

print(f"\nMetrics:")
print(f"  Total: {total_crises}, AR caught: {ar_tp}, AR failed: {ar_fn}")
print(f"  Cascade rescued: {key_saves}, Still missed: {still_missed}")

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11

# Create figure with clean layout
fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(111)

# Colors
color_total = '#E0E0E0'
color_ar_success = '#4CAF50'
color_ar_fail = '#FF9800'
color_cascade = '#FFD700'
color_missed = '#BDBDBD'

# Vertical positions - well separated
y_top = 0.75
y_middle = 0.5
y_bottom = 0.25

# Box dimensions
box_width = 0.18
box_height = 0.15

# LEVEL 1: Total Crises (left)
x1 = 0.05
rect1 = FancyBboxPatch((x1, y_middle - box_height/2), box_width, box_height,
                         boxstyle="round,pad=0.02", edgecolor='black',
                         facecolor=color_total, linewidth=2.5)
ax.add_patch(rect1)
ax.text(x1 + box_width/2, y_middle, 'TOTAL CRISES\n5,322',
        ha='center', va='center', fontsize=16, fontweight='bold')

# LEVEL 2: AR Results (middle) - TWO BOXES vertically separated
x2 = 0.32

# Top box: AR Success (GREEN)
rect2a = FancyBboxPatch((x2, y_top - box_height/2), box_width, box_height,
                         boxstyle="round,pad=0.02", edgecolor='black',
                         facecolor=color_ar_success, linewidth=2.5)
ax.add_patch(rect2a)
ax.text(x2 + box_width/2, y_top, f'AR CAUGHT\n3,895\n\n(73.2%)\nEasy: Persistent',
        ha='center', va='center', fontsize=13, fontweight='bold', color='black')

# Bottom box: AR Failed (ORANGE)
rect2b = FancyBboxPatch((x2, y_bottom - box_height/2), box_width, box_height,
                         boxstyle="round,pad=0.02", edgecolor='black',
                         facecolor=color_ar_fail, linewidth=2.5)
ax.add_patch(rect2b)
ax.text(x2 + box_width/2, y_bottom, f'AR FAILED\n1,427\n\n(26.8%)\nHard: Shocks',
        ha='center', va='center', fontsize=13, fontweight='bold', color='black')

# LEVEL 3: Cascade Results (right) - TWO BOXES vertically separated
x3 = 0.59

# Top box: CASCADE BREAKTHROUGH (GOLD - EMPHASIZED)
rect3a = FancyBboxPatch((x3, y_top - box_height/2), box_width, box_height,
                         boxstyle="round,pad=0.03", edgecolor='darkgoldenrod',
                         facecolor=color_cascade, linewidth=4)
ax.add_patch(rect3a)

# Add small star ABOVE the box (not overlapping text)
ax.scatter([x3 + box_width/2], [y_top + box_height/2 + 0.04], s=2000, marker='*',
          color='gold', edgecolors='darkgoldenrod', linewidths=3, zorder=10)

ax.text(x3 + box_width/2, y_top, f'CASCADE\nBREAKTHROUGH\n\n249\nRescued',
        ha='center', va='center', fontsize=14, fontweight='bold',
        color='black', zorder=11)

# Bottom box: Still Missed (GRAY - de-emphasized)
rect3b = FancyBboxPatch((x3, y_bottom - box_height/2), box_width, box_height,
                         boxstyle="round,pad=0.02", edgecolor='gray',
                         facecolor=color_missed, linewidth=2, alpha=0.6)
ax.add_patch(rect3b)
ax.text(x3 + box_width/2, y_bottom, f'Still Difficult\n1,178',
        ha='center', va='center', fontsize=12, color='black')

# ARROWS - clean paths, no overlaps
# Total → AR Success
arrow1 = FancyArrowPatch((x1 + box_width, y_middle), (x2, y_top),
                         arrowstyle='->', mutation_scale=35,
                         linewidth=4, color='green', alpha=0.7,
                         connectionstyle="arc3,rad=.3")
ax.add_patch(arrow1)

# Total → AR Failed
arrow2 = FancyArrowPatch((x1 + box_width, y_middle), (x2, y_bottom),
                         arrowstyle='->', mutation_scale=35,
                         linewidth=4, color='darkorange', alpha=0.7,
                         connectionstyle="arc3,rad=-.3")
ax.add_patch(arrow2)

# AR Failed → Cascade (EMPHASIZED - thicker gold arrow)
arrow3 = FancyArrowPatch((x2 + box_width, y_bottom), (x3, y_top),
                         arrowstyle='->', mutation_scale=45,
                         linewidth=6, color='darkgoldenrod', alpha=0.9,
                         connectionstyle="arc3,rad=.3")
ax.add_patch(arrow3)

# AR Failed → Still Missed (de-emphasized)
arrow4 = FancyArrowPatch((x2 + box_width, y_bottom), (x3, y_bottom),
                         arrowstyle='->', mutation_scale=25,
                         linewidth=3, color='gray', alpha=0.5)
ax.add_patch(arrow4)

# ANNOTATIONS - positioned in white space, NO overlaps

# Title at top
ax.text(0.5, 0.96,
        'Cascade Breakthrough: 249 Shock-Driven Crises Rescued Where Persistence Failed',
        ha='center', va='top', fontsize=17, fontweight='bold',
        transform=ax.transAxes)

# Percentage label above cascade box
ax.text(x3 + box_width/2, y_top + box_height/2 + 0.08,
        '17.4% of Hard Cases',
        ha='center', va='bottom', fontsize=12, fontweight='bold',
        color='darkgoldenrod')

# Country breakdown - BELOW cascade box in clean white space
countries_text = f'Top 3 Countries:\nZimbabwe: {top_countries.iloc[0]}  |  Sudan: {top_countries.iloc[1]}  |  DRC: {top_countries.iloc[2]}'
ax.text(x3 + box_width/2, y_top - box_height/2 - 0.10,
        countries_text,
        ha='center', va='top', fontsize=10, style='italic',
        color='black')

# Key insight box at bottom
insight_text = (
    "KEY INSIGHT: 249 crises = Conflict escalations (Sudan), economic collapse (Zimbabwe), displacement shocks (DRC)\n"
    "THE HARDEST CASES—invisible to persistence, caught 8 months in advance\n"
    "Real humanitarian impact where early warning matters most"
)
ax.text(0.5, 0.04,
        insight_text,
        ha='center', va='bottom', fontsize=10, style='italic',
        transform=ax.transAxes,
        bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow',
                  alpha=0.95, edgecolor='darkgoldenrod', linewidth=2.5))

# Set limits and remove axes
ax.set_xlim(0, 0.82)
ax.set_ylim(0, 1)
ax.axis('off')

plt.tight_layout()

# Save
output_file = OUTPUT_DIR / "ch05_cascade_breakthrough.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file}")

output_file_png = OUTPUT_DIR / "ch05_cascade_breakthrough.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png}")

plt.close()

print("\n" + "="*80)
print("FIGURE 23 COMPLETE: CASCADE BREAKTHROUGH (CLEAN REDESIGN)")
print("="*80)
