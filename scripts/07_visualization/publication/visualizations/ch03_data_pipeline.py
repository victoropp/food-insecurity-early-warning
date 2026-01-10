"""
Figure 6: Data Pipeline Flowchart
Publication-grade flowchart showing data processing from raw GDELT to final dataset

Uses VERIFIED counts from DATA_SCHEMA

Date: 2026-01-04
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from config import BASE_DIR

# Directories
BASE_DIR = Path(rstr(BASE_DIR))
OUTPUT_DIR = BASE_DIR / "Dissertation Write Up" / "LATEX_DISSERTATION" / "figures" / "ch03_methods"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# VERIFIED data counts from DATA_SCHEMA.md
GDELT_ARTICLES = "7.6M"  # From plan
IPC_ASSESSMENTS_RAW = "55,129"  # Raw IPC assessments before h=8 filtering
FINAL_OBSERVATIONS = "20,722"  # After h=8 filtering
COUNTRIES_RAW = "24"
COUNTRIES_FINAL = "18"
DISTRICTS_RAW = "3,438"  # Raw IPC districts before filtering
DISTRICTS_FINAL = "1,920"  # Final unique districts after filtering
CRISES = "5,322"  # Total crisis cases (y_true=1)

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# Create figure - SIMPLIFIED
fig, ax = plt.subplots(figsize=(12, 8))

# Color scheme
COLOR_INPUT = '#3498DB'    # Blue
COLOR_OUTPUT = '#27AE60'   # Green

# Helper function to draw boxes
def draw_box(ax, x, y, width, height, text, color, fontsize=12, fontweight='normal'):
    rect = mpatches.FancyBboxPatch((x, y), width, height,
                                    boxstyle="round,pad=0.15",
                                    facecolor=color, edgecolor='black',
                                    linewidth=2.5, alpha=0.85)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, text, ha='center', va='center',
            fontsize=fontsize, fontweight=fontweight, wrap=True)

# Helper function to draw arrows
def draw_arrow(ax, x1, y1, x2, y2, label='', fontsize=10):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', lw=3, color='black'))
    if label:
        mid_x, mid_y = (x1+x2)/2, (y1+y2)/2
        ax.text(mid_x, mid_y + 0.2, label, fontsize=fontsize, ha='center',
                bbox=dict(boxstyle='round,pad=0.4', facecolor='yellow', alpha=0.7),
                fontweight='bold')

# INPUTS (top)
draw_box(ax, 1, 6, 3.5, 1.3, f'GDELT\n{GDELT_ARTICLES} articles', COLOR_INPUT, fontsize=13, fontweight='bold')
draw_box(ax, 6, 6, 3.5, 1.3, f'IPC Assessments\n{IPC_ASSESSMENTS_RAW} records', COLOR_INPUT, fontsize=13, fontweight='bold')

# Arrows down
draw_arrow(ax, 2.75, 6, 2.75, 4.5, '9 categories')
draw_arrow(ax, 7.75, 6, 7.75, 4.5, 'h=8 filter')

# MERGE (middle)
draw_box(ax, 2.5, 3, 5.5, 1.3, f'Geographic Merge\n{DISTRICTS_RAW} → {DISTRICTS_FINAL} districts', COLOR_INPUT, fontsize=13)

# Arrow down
draw_arrow(ax, 5.25, 3, 5.25, 1.5, '35 features')

# OUTPUT (bottom)
draw_box(ax, 1.5, 0, 7.5, 1.3, f'FINAL: {FINAL_OBSERVATIONS} observations\n{CRISES} crises (25.7%), {COUNTRIES_FINAL} countries', COLOR_OUTPUT, fontsize=14, fontweight='bold')

# Set axis properties
ax.set_xlim(0, 11)
ax.set_ylim(-0.5, 7.8)
ax.axis('off')

# Title
ax.text(5.5, 7.5, 'Data Processing Pipeline',
        ha='center', fontsize=16, fontweight='bold')

# Footer with data provenance
footer_text = (
    "Data sources: GDELT Global Knowledge Graph (7.6M articles, 2021-2024) + IPC Cadre Harmonise assessments (55,129 district-period records).\n"
    f"Final dataset: {FINAL_OBSERVATIONS} observations after h=8 forecast filtering, covering {COUNTRIES_FINAL} countries and {DISTRICTS_FINAL} unique districts (filtered from {DISTRICTS_RAW} raw IPC districts).\n"
    f"Crisis rate: {CRISES} crises (25.7%). Geographic scope: Sub-Saharan Africa (18 countries with sufficient IPC coverage)."
)
fig.text(0.5, 0.02, footer_text, ha='center', fontsize=8, style='italic',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3))

plt.tight_layout(rect=[0, 0.08, 1, 1])

# Save
output_file = OUTPUT_DIR / "ch03_data_pipeline.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file}")

output_file_png = OUTPUT_DIR / "ch03_data_pipeline.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png}")

plt.close()

print("\n" + "="*80)
print("FIGURE 6 COMPLETE: Data Pipeline Flowchart")
print("="*80)
print(f"GDELT: {GDELT_ARTICLES} articles -> District-month aggregation")
print(f"IPC: {IPC_ASSESSMENTS_RAW} assessments -> h=8 filtering -> {FINAL_OBSERVATIONS} observations")
print(f"Geographic: {COUNTRIES_RAW} countries -> {COUNTRIES_FINAL} included, {DISTRICTS_RAW} -> {DISTRICTS_FINAL} districts")
print(f"Output: {CRISES} crises (25.7% crisis rate), 35 engineered features")
