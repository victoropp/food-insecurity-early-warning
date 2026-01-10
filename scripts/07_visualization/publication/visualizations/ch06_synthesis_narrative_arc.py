"""
Chapter 6 Synthesis Figure: Complete Narrative Arc (REDESIGNED - CLEAN)
Publication-grade vertical flow showing the research story
Simplified, uncluttered design focusing on the core narrative

Date: 2026-01-05
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np
from pathlib import Path
from config import BASE_DIR

# Directories
BASE_DIR = Path(str(BASE_DIR))
OUTPUT_DIR = BASE_DIR / "Dissertation Write Up" / "LATEX_DISSERTATION" / "figures" / "ch06_conclusion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# VERIFIED metrics
TOTAL_CRISES = 5322
AR_CAUGHT = 3895
AR_FAILED = 1427
CASCADE_RESCUED = 249
ZIMBABWE_SAVES = 77
SUDAN_SAVES = 59
DRC_SAVES = 40

# Calculate percentages
AR_CATCH_RATE = 73.2
AR_FAIL_RATE = 26.8
CASCADE_RESCUE_RATE = 17.4

# STANDARD COLOR SCHEME
COLOR_AR_SUCCESS = '#27AE60'  # Green
COLOR_AR_FAIL = '#E67E22'     # Orange
COLOR_CASCADE = '#F39C12'     # Gold
COLOR_ZIMBABWE = '#E74C3C'    # Red
COLOR_SUDAN = '#3498DB'       # Blue
COLOR_DRC = '#9B59B6'         # Purple

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11

# Create figure - CLEANER layout
fig, ax = plt.subplots(figsize=(14, 10))

# Helper function - simplified boxes
def draw_box(ax, x, y, width, height, text, color, fontsize=14, alpha=0.9):
    rect = FancyBboxPatch((x - width/2, y - height/2), width, height,
                          boxstyle="round,pad=0.15",
                          facecolor=color, edgecolor='black',
                          linewidth=3, alpha=alpha)
    ax.add_patch(rect)
    ax.text(x, y, text, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='white')

# Helper function - clean arrows
def draw_arrow(ax, x1, y1, x2, y2, color, width=15, label=''):
    arrow = FancyArrowPatch((x1, y1), (x2, y2),
                           arrowstyle='->', mutation_scale=40,
                           lw=width, color=color, alpha=0.7,
                           zorder=1)
    ax.add_patch(arrow)

    if label:
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mid_x + 1.5, mid_y, label, fontsize=12, ha='left',
                fontweight='bold', color='black',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                         edgecolor='black', linewidth=2))

# VERTICAL FLOW - Much cleaner
y_positions = [8.5, 6.5, 4.5, 2, 0.3]

# LEVEL 1: Total (top)
draw_box(ax, 7, y_positions[0], 5, 1,
         f'5,322 Total Crises\n(20,722 observations, 18 countries)',
         '#95A5A6', fontsize=15)

# LEVEL 2: AR Split
draw_box(ax, 3, y_positions[1], 4, 1.2,
         f'AR CAUGHT\n3,895 (73.2%)',
         COLOR_AR_SUCCESS, fontsize=14)

draw_box(ax, 11, y_positions[1], 4, 1.2,
         f'AR FAILED\n1,427 (26.8%)',
         COLOR_AR_FAIL, fontsize=14)

# LEVEL 3: Cascade Result
draw_box(ax, 11, y_positions[2], 4.5, 1.2,
         f'CASCADE RESCUED\n249 (17.4% of failures)\nBREAKTHROUGH',
         COLOR_CASCADE, fontsize=14)

# LEVEL 4: Top 3 Countries - HORIZONTAL ROW
country_width = 3.5
draw_box(ax, 4, y_positions[3], country_width, 1,
         f'Zimbabwe\n77 saves (30.9%)',
         COLOR_ZIMBABWE, fontsize=13)

draw_box(ax, 8.5, y_positions[3], country_width, 1,
         f'Sudan\n59 saves (23.7%)',
         COLOR_SUDAN, fontsize=13)

draw_box(ax, 13, y_positions[3], country_width, 1,
         f'DRC\n40 saves (16.1%)',
         COLOR_DRC, fontsize=13)

# LEVEL 5: Summary box
summary_box = (
    '176 saves (70.7%) in top 3 conflict zones\n'
    'News features provide value where persistence fails'
)
ax.text(8.5, y_positions[4], summary_box, ha='center', va='center',
        fontsize=12, fontweight='bold',
        bbox=dict(boxstyle='round,pad=0.7', facecolor='#FFE699',
                 edgecolor='#F39C12', linewidth=3))

# ARROWS - Clean, proportional
# Total → AR Caught
draw_arrow(ax, 5, y_positions[0] - 0.6, 3, y_positions[1] + 0.7,
          COLOR_AR_SUCCESS, width=20)

# Total → AR Failed
draw_arrow(ax, 9, y_positions[0] - 0.6, 11, y_positions[1] + 0.7,
          COLOR_AR_FAIL, width=12)

# AR Failed → Cascade Rescued
draw_arrow(ax, 11, y_positions[1] - 0.7, 11, y_positions[2] + 0.7,
          COLOR_CASCADE, width=8,
          label='News features')

# Cascade → Top 3 (single arrow, then splits)
draw_arrow(ax, 11, y_positions[2] - 0.7, 8.5, y_positions[3] + 0.6,
          COLOR_CASCADE, width=6)

# Side annotations - MINIMAL
ax.text(0.5, y_positions[1], 'Persistence\nworks', ha='center', va='center',
        fontsize=11, style='italic', color=COLOR_AR_SUCCESS,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                 edgecolor=COLOR_AR_SUCCESS, linewidth=2))

ax.text(15.5, y_positions[1], 'Rapid\nshocks', ha='center', va='center',
        fontsize=11, style='italic', color=COLOR_AR_FAIL,
        bbox=dict(boxstyle='round,pad=0.4', facecolor='white',
                 edgecolor=COLOR_AR_FAIL, linewidth=2))

# Set axis properties
ax.set_xlim(-1, 17)
ax.set_ylim(-0.5, 9.5)
ax.axis('off')

# Title - SIMPLIFIED
ax.text(8, 9.3, 'Complete Narrative Arc: From Autocorrelation to Impact',
        ha='center', fontsize=18, fontweight='bold')

# Footer - CONCISE
footer = (
    'AR baseline (AUC=0.907) catches 73.2% via persistence → '
    'Cascade rescues 17.4% of AR failures (249 crises) → '
    '70.7% concentrated in Zimbabwe, Sudan, DRC conflict zones'
)
fig.text(0.5, 0.02, footer, ha='center', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))

plt.tight_layout(rect=[0, 0.05, 1, 0.98])

# Save
output_file = OUTPUT_DIR / "ch06_synthesis_narrative_arc.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file}")

output_file_png = OUTPUT_DIR / "ch06_synthesis_narrative_arc.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png}")

plt.close()

print("\n" + "="*80)
print("CHAPTER 6 SYNTHESIS FIGURE COMPLETE (REDESIGNED - CLEAN)")
print("="*80)
print(f"Layout: Vertical flow, minimal clutter")
print(f"Colors: Standard scheme (Zimbabwe=Red, Sudan=Blue, DRC=Purple)")
