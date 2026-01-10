"""
Figure 8: Feature Engineering Workflow
Publication-grade 4-stage pipeline showing transformation from raw GDELT to final 35 features

Date: 2026-01-04
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from config import BASE_DIR

# Directories
BASE_DIR = Path(str(BASE_DIR))
OUTPUT_DIR = BASE_DIR / "Dissertation Write Up" / "LATEX_DISSERTATION" / "figures" / "ch03_methods"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# Create figure
fig, ax = plt.subplots(figsize=(14, 9))

# Color scheme
COLOR_STAGE1 = '#3498DB'   # Blue - Raw
COLOR_STAGE2 = '#9B59B6'   # Purple - Transformed
COLOR_STAGE3 = '#E67E22'   # Orange - Advanced
COLOR_STAGE4 = '#27AE60'   # Green - Final

# Helper function to draw boxes
def draw_box(ax, x, y, width, height, text, color, fontsize=11, fontweight='normal'):
    rect = mpatches.FancyBboxPatch((x, y), width, height,
                                    boxstyle="round,pad=0.15",
                                    facecolor=color, edgecolor='black',
                                    linewidth=2.5, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, wrap=True)

# Helper function to draw arrows
def draw_arrow(ax, x1, y1, x2, y2, label='', fontsize=9):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    if label:
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax.text(mid_x + 0.3, mid_y, label, fontsize=fontsize, ha='center',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='yellow', alpha=0.6),
                fontweight='bold')

# STEP 1: Raw Article Counts (top)
draw_box(ax, 1.5, 6.5, 3, 1.3,
         'Raw Article Counts\n9 GDELT categories',
         COLOR_STAGE1, fontsize=13, fontweight='bold')

# Arrow down
draw_arrow(ax, 3, 6.5, 3, 5, '')

# STEP 2: Ratio + Z-score Transformations
draw_box(ax, 1.5, 3.5, 3, 1.3,
         'Ratio + Z-score\nTransformations\n18 features',
         COLOR_STAGE2, fontsize=13, fontweight='bold')

# Arrow down
draw_arrow(ax, 3, 3.5, 3, 2, '')

# STEP 3: HMM + DMD Advanced Features
draw_box(ax, 1.5, 0.5, 3, 1.3,
         'HMM + DMD\nAdvanced Features\n14 features',
         COLOR_STAGE3, fontsize=13, fontweight='bold')

# Arrow down
draw_arrow(ax, 3, 0.5, 3, -1, '')

# STEP 4: Final Feature Set
draw_box(ax, 1.5, -2.5, 3, 1.3,
         'Final Feature Set\n35 total features',
         COLOR_STAGE4, fontsize=13, fontweight='bold')

# Set axis properties - SIMPLIFIED, centered
ax.set_xlim(0, 6)
ax.set_ylim(-3.5, 8.5)
ax.axis('off')

# Title
ax.text(3, 8.2, 'Feature Engineering Pipeline',
        ha='center', fontsize=16, fontweight='bold')

# Footer
footer_text = (
    "Feature engineering pipeline transforms 7.6M GDELT articles into 35 engineered features for Stage 2 XGBoost models. "
    "Raw article counts (9 GDELT categories) → Ratio features (compositional dynamics) + Z-score features (anomaly detection) → "
    "HMM features (regime transitions) + DMD features (temporal dynamics) → Final feature set (3 location + 32 news-derived). "
    "Location features dominate importance (country_data_density 13.3%), while HMM transition risk ranks #5 (3.2% importance)."
)
fig.text(0.5, 0.02, footer_text, ha='center', fontsize=8, style='italic',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3),
         wrap=True)

plt.tight_layout(rect=[0, 0.08, 1, 1])

# Save
output_file = OUTPUT_DIR / "ch03_feature_engineering.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file}")

output_file_png = OUTPUT_DIR / "ch03_feature_engineering.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png}")

plt.close()

print("\n" + "="*80)
print("FIGURE 8 COMPLETE: Feature Engineering Workflow")
print("="*80)
print("Stage 1: 9 raw GDELT category counts")
print("Stage 2: 18 transformed (9 ratio + 9 z-score)")
print("Stage 3: 14 advanced (6 HMM + 8 DMD)")
print("Stage 4: 35 total features (3 location + 32 news-derived)")
print("Top features: Country_data_density (13.3%), Ratio_conflict (#1), HMM_transition (#5)")
