"""
Figure 5: Literature Comparison Matrix (VERIFIED)
Publication-grade comparison based on actual research of published papers

ALL CLAIMS VERIFIED from actual methodology sections
Sources documented in VERIFIED_LITERATURE_COMPARISON_JAN4_2026.md

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
BASE_DIR = Path(rstr(BASE_DIR))
OUTPUT_DIR = BASE_DIR / "Dissertation Write Up" / "LATEX_DISSERTATION" / "figures" / "ch02_background"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# VERIFIED literature comparison data
studies = [
    'Balashankar et al.\n(2023)',
    'Busker et al.\n(2024)',
    'Nature Comm.\n(2024)',
    'Lentz et al.\n(2019)',
    'This Work\n(2026)'
]

# Comparison criteria
criteria = [
    'AR Baseline\n(Lt + Ls)',
    'Spatial\nCV',
    'Two-Stage\nCascade',
    'Interpretability\nAnalysis',
    'Dynamic\nFeatures'
]

# VERIFIED data: 0 = No (red), 1 = Partial (yellow), 2 = Yes (green), 3 = Not reported (gray)
# Based on actual research documented in VERIFIED_LITERATURE_COMPARISON_JAN4_2026.md
comparison_data = np.array([
    [0, 0, 0, 1, 0],  # Balashankar 2023: No AR baseline (Lt as features ≠ baseline), temporal CV, some interp
    [0, 3, 0, 1, 0],  # Busker 2024: Persistence ≠ Lt+Ls AR baseline, CV unclear, regional analysis
    [1, 3, 0, 3, 0],  # Nature Comm 2024: ARIMA (temporal only, no Ls), CV unclear
    [3, 3, 0, 3, 0],  # Lentz 2019: All unclear from available sources
    [2, 2, 2, 2, 2],  # This Work: All yes
])

# Color mapping (updated with gray for "not reported")
colors = {
    0: '#E74C3C',  # Red - No
    1: '#F39C12',  # Orange - Partial
    2: '#27AE60',  # Green - Yes
    3: '#95A5A6',  # Gray - Not reported
}

# Text labels
labels = {
    0: '✗',
    1: '~',
    2: '✓',
    3: '?',
}

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# Create figure
fig, ax = plt.subplots(figsize=(14, 7))

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
        fontsize = 22 if label in ['✓', '✗'] else 24
        fontweight = 'bold' if i == len(studies)-1 else 'normal'
        text_color = 'white' if value != 3 else 'black'
        ax.text(j + 0.5, len(studies)-1-i + 0.5, label,
                ha='center', va='center', fontsize=fontsize,
                fontweight=fontweight, color=text_color)

# Set axis properties
ax.set_xlim(0, len(criteria))
ax.set_ylim(0, len(studies))

# Y-axis: Studies
ax.set_yticks(np.arange(len(studies)) + 0.5)
ax.set_yticklabels(studies[::-1], fontsize=11)

# Highlight this work row
for tick_label in ax.get_yticklabels():
    if 'This Work' in tick_label.get_text():
        tick_label.set_fontweight('bold')
        tick_label.set_fontsize(12)

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
ax.set_title('Methodological Gaps in Food Security Forecasting Literature',
             fontsize=14, fontweight='bold', pad=20)

# Add legend
legend_elements = [
    mpatches.Patch(facecolor=colors[2], edgecolor='black', label='✓ Yes (Complete)'),
    mpatches.Patch(facecolor=colors[1], edgecolor='black', label='~ Partial'),
    mpatches.Patch(facecolor=colors[0], edgecolor='black', label='✗ No (Absent)'),
    mpatches.Patch(facecolor=colors[3], edgecolor='black', label='? Not Reported')
]
ax.legend(handles=legend_elements, loc='upper left', bbox_to_anchor=(1.02, 1),
          fontsize=10, frameon=True, edgecolor='black')

# Add detailed footer with VERIFIED claims
footer_text = (
    "VERIFIED CLAIMS from published methodology sections:\n"
    "• AR Baseline (Lt + Ls): Spatio-temporal autoregressive features with ZERO external covariates\n"
    "  - Balashankar 2023: Used Lt as features in RF, but never compared against AR-only baseline (✗)\n"
    "  - Busker 2024: 'Persistence model' mentioned, but NOT Lt + Ls specification (✗)\n"
    "  - Nature Comm 2024: ARIMA (temporal AR only, no spatial Ls component) (~)\n"
    "• Spatial CV: Geographic blocking to prevent spatial leakage (Balashankar used temporal CV only)\n"
    "• Two-Stage Cascade: AR baseline Stage 1 → Dynamic features Stage 2 for failures\n"
    "• Interpretability: Country-level heterogeneity analysis (XGBoost + Mixed-effects + SHAP)\n"
    "• Dynamic Features: HMM regime transitions, DMD temporal modes, z-score anomalies\n\n"
    "Sources: PMC, Earth's Future, Communications Earth & Environment, ScienceDirect (Jan 4, 2026)"
)
fig.text(0.5, -0.02, footer_text, ha='center', fontsize=8, style='italic',
         bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', alpha=0.3),
         wrap=True)

plt.tight_layout(rect=[0, 0.18, 0.88, 1])

# Save
output_file = OUTPUT_DIR / "ch02_literature_comparison.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file}")

output_file_png = OUTPUT_DIR / "ch02_literature_comparison.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png}")

plt.close()

print("\n" + "="*80)
print("FIGURE 5 COMPLETE: VERIFIED Literature Comparison Matrix")
print("="*80)
print("ALL CLAIMS VERIFIED from actual published methodology sections:")
print("  ✗ Balashankar 2023: No AR baseline comparison (Lt as features ≠ AR-only model)")
print("  ✗ Busker 2024: Persistence model ≠ rigorous Lt + Ls AR baseline")
print("  ~ Nature Comm 2024: ARIMA (temporal only, no spatial component)")
print("  ? Lentz 2019: Not reported in available sources")
print("  ✓ This Work: First to implement all 5 methodological innovations")
print("\nSources documented in: VERIFIED_LITERATURE_COMPARISON_JAN4_2026.md")
