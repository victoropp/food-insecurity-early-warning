"""
Figure 5: Literature Comparison Matrix
Publication-grade comparison showing methodological gaps in existing work

Uses color-coded table to show this study's advantages over literature

Date: 2026-01-04
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import matplotlib.patches as mpatches
from config import BASE_DIR

# Directories
BASE_DIR = Path(str(BASE_DIR))
OUTPUT_DIR = BASE_DIR / "Dissertation Write Up" / "LATEX_DISSERTATION" / "figures" / "ch02_background"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Literature comparison data
studies = [
    'Balashankar et al. (2023)',
    'Balashankar et al. (2021)',
    'Lentz et al. (2019)',
    'Busker et al. (2024)',
    'Nature Comm. (2024)',
    'This Work (2026)'
]

# Comparison criteria
criteria = [
    'AR Baseline\n(Lt + Ls)',
    'Spatial CV',
    'Two-Stage\nFramework',
    'Geographic\nHeterogeneity',
    'Dynamic\nFeatures'
]

# Data: 0 = No (red), 1 = Partial (yellow), 2 = Yes (green)
comparison_data = np.array([
    [0, 0, 0, 1, 0],  # Balashankar 2023
    [0, 0, 0, 0, 0],  # Balashankar 2021
    [1, 0, 0, 0, 0],  # Lentz 2019
    [0, 0, 0, 1, 0],  # Busker 2024
    [1, 0, 0, 0, 0],  # Nature Comm 2024
    [2, 2, 2, 2, 2],  # This Work
])

# Color mapping
colors = {
    0: '#E74C3C',  # Red - No
    1: '#F39C12',  # Orange/Yellow - Partial
    2: '#27AE60',  # Green - Yes
}

# Text labels
labels = {
    0: '✗',
    1: '~',
    2: '✓',
}

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11

# Create figure
fig, ax = plt.subplots(figsize=(14, 8))

# Create the heatmap
for i in range(len(studies)):
    for j in range(len(criteria)):
        value = comparison_data[i, j]
        color = colors[value]
        label = labels[value]

        # Draw cell background
        rect = mpatches.Rectangle((j, len(studies)-1-i), 1, 1,
                                   facecolor=color, edgecolor='black', linewidth=1.5)
        ax.add_patch(rect)

        # Add text
        fontsize = 24 if label == '✓' or label == '✗' else 20
        fontweight = 'bold' if i == len(studies)-1 else 'normal'
        ax.text(j + 0.5, len(studies)-1-i + 0.5, label,
                ha='center', va='center', fontsize=fontsize,
                fontweight=fontweight, color='white')

# Set axis properties
ax.set_xlim(0, len(criteria))
ax.set_ylim(0, len(studies))

# Y-axis: Studies
ax.set_yticks(np.arange(len(studies)) + 0.5)
ax.set_yticklabels(studies[::-1], fontsize=12)

# Highlight this work row
for tick_label in ax.get_yticklabels():
    if 'This Work' in tick_label.get_text():
        tick_label.set_fontweight('bold')
        tick_label.set_fontsize(13)

# X-axis: Criteria
ax.set_xticks(np.arange(len(criteria)) + 0.5)
ax.set_xticklabels(criteria, fontsize=11, fontweight='bold')

# Remove spines
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)
ax.spines['bottom'].set_visible(False)
ax.spines['left'].set_visible(False)

# Remove tick marks
ax.tick_params(axis='both', which='both', length=0)

# Title
ax.set_title('Literature Comparison: Methodological Gaps and This Work\'s Contributions',
             fontsize=14, fontweight='bold', pad=20)

# Add legend
legend_elements = [
    mpatches.Patch(facecolor=colors[2], edgecolor='black', label='✓ Yes (Complete)'),
    mpatches.Patch(facecolor=colors[1], edgecolor='black', label='~ Partial'),
    mpatches.Patch(facecolor=colors[0], edgecolor='black', label='✗ No (Absent)')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
          fontsize=11, frameon=True, edgecolor='black')

# Add footer annotation
footer_text = (
    "AR Baseline: Spatio-temporal autoregressive features (Lt + Ls) with zero external covariates\n"
    "Spatial CV: Leave-one-region-out or spatial block cross-validation preventing geographic leakage\n"
    "Two-Stage: Cascade framework (AR baseline → dynamic features for failures)\n"
    "Geographic Heterogeneity: Country-level analysis of where features provide value\n"
    "Dynamic Features: HMM regime transitions, DMD temporal modes, or z-score anomalies"
)
fig.text(0.5, 0.02, footer_text, ha='center', fontsize=9, style='italic',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', alpha=0.3))

plt.tight_layout(rect=[0, 0.12, 0.85, 1])

# Save
output_file = OUTPUT_DIR / "ch02_literature_comparison.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file}")

output_file_png = OUTPUT_DIR / "ch02_literature_comparison.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png}")

plt.close()

print("\n" + "="*80)
print("FIGURE 5 COMPLETE: Literature Comparison Matrix")
print("="*80)
print("Demonstrates this work addresses 5 systematic gaps:")
print("  1. AR baseline: AUC=0.907 with Lt + Ls (zero external covariates)")
print("  2. Spatial CV: Leave-one-country-out preventing leakage")
print("  3. Two-stage cascade: 249 key saves (17.4% rescue rate)")
print("  4. Geographic heterogeneity: 70.7% saves in Zimbabwe/Sudan/DRC")
print("  5. Dynamic features: HMM, DMD, z-scores evaluated via ablation")
