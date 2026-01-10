"""
Figure 2: Two-Stage Framework Architecture - SIMPLIFIED & CLEAN
Publication-grade diagram with MINIMAL text, LARGE fonts, CLEAR layout

Focus: Core cascade logic + key result (249 saves)
Remove: Excessive detail that won't be readable when scaled

Date: 2026-01-04
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import json
from pathlib import Path
from config import BASE_DIR

# Directories
BASE_DIR = Path(rstr(BASE_DIR))
DATA_DIR = BASE_DIR / "Dissertation Write Up" / "DATA_INVENTORIES"
OUTPUT_DIR = BASE_DIR / "Dissertation Write Up" / "LATEX_DISSERTATION" / "figures" / "ch01_introduction"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load JSON data
print("Loading data from MASTER_METRICS_ALL_MODELS.json...")
with open(DATA_DIR / "MASTER_METRICS_ALL_MODELS.json", 'r') as f:
    data = json.load(f)

# Extract VERIFIED metrics
cascade_prod = data['cascade']['production']

# AR Baseline metrics
ar_precision = cascade_prod['ar_precision']  # 0.7319
ar_recall = cascade_prod['ar_recall']  # 0.7319
ar_fn = cascade_prod['ar_fn']  # 1427 - AR missed crises

# Cascade metrics
cascade_precision = cascade_prod['cascade_precision']  # 0.5851
cascade_recall = cascade_prod['cascade_recall']  # 0.7787
key_saves = cascade_prod['key_saves']  # 249

# Total observations
total_obs = cascade_prod['total_obs']  # 20722

print(f"Verified metrics loaded:")
print(f"  AR: Precision=Recall={ar_precision:.3f}")
print(f"  Cascade: Precision={cascade_precision:.3f}, Recall={cascade_recall:.3f}")
print(f"  Key saves: {key_saves}")

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11  # LARGER base font

# Create figure - HORIZONTAL layout for better space usage
fig, ax = plt.subplots(figsize=(14, 6))

# Colors (colorblind-safe)
COLOR_AR = '#2E86AB'  # Blue
COLOR_STAGE2 = '#7B68BE'  # Purple
COLOR_CASCADE = '#F18F01'  # Orange

# ============================================================================
# HORIZONTAL 3-STAGE LAYOUT
# ============================================================================

# STAGE 1: AR Baseline (LEFT)
stage1_box = FancyBboxPatch(
    (0.5, 1.5), 3.5, 3,
    boxstyle="round,pad=0.15",
    edgecolor=COLOR_AR,
    facecolor=COLOR_AR,
    alpha=0.15,
    linewidth=3
)
ax.add_patch(stage1_box)

ax.text(2.25, 4.0, 'STAGE 1', fontsize=16, fontweight='bold', ha='center', color=COLOR_AR)
ax.text(2.25, 3.5, 'AR Baseline', fontsize=14, ha='center', color=COLOR_AR)
ax.text(2.25, 3.0, 'Lt: Past IPC (t-1,...,t-12)', fontsize=9, ha='center', style='italic')
ax.text(2.25, 2.6, 'Ls: Neighboring IPC (300km)', fontsize=9, ha='center', style='italic')
ax.text(2.25, 2.1, f'Precision = Recall = {ar_precision:.3f}', fontsize=11, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, pad=0.3))

# ARROW 1
arrow1 = FancyArrowPatch(
    (4.2, 3), (5.3, 3),
    arrowstyle='->', mutation_scale=40, linewidth=3, color='black'
)
ax.add_patch(arrow1)

# STAGE 2: XGBoost + News (MIDDLE)
stage2_box = FancyBboxPatch(
    (5.5, 1.5), 3.5, 3,
    boxstyle="round,pad=0.15",
    edgecolor=COLOR_STAGE2,
    facecolor=COLOR_STAGE2,
    alpha=0.15,
    linewidth=3
)
ax.add_patch(stage2_box)

ax.text(7.25, 4.0, 'STAGE 2', fontsize=16, fontweight='bold', ha='center', color=COLOR_STAGE2)
ax.text(7.25, 3.5, 'XGBoost + News', fontsize=14, ha='center', color=COLOR_STAGE2)
ax.text(7.25, 3.0, '35 Dynamic Features', fontsize=10, ha='center', style='italic')
ax.text(7.25, 2.4, 'Targets AR failures:', fontsize=10, ha='center')
ax.text(7.25, 2.0, f'{ar_fn:,} missed crises', fontsize=12, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.4))

# ARROW 2
arrow2 = FancyArrowPatch(
    (9.2, 3), (10.3, 3),
    arrowstyle='->', mutation_scale=40, linewidth=3, color='black'
)
ax.add_patch(arrow2)

# CASCADE OUTPUT (RIGHT)
cascade_box = FancyBboxPatch(
    (10.5, 1.5), 3.5, 3,
    boxstyle="round,pad=0.15",
    edgecolor=COLOR_CASCADE,
    facecolor=COLOR_CASCADE,
    alpha=0.15,
    linewidth=3
)
ax.add_patch(cascade_box)

ax.text(12.25, 4.0, 'CASCADE ENSEMBLE', fontsize=15, fontweight='bold', ha='center', color=COLOR_CASCADE)
ax.text(12.25, 3.5, f'Recall: {cascade_recall:.3f}', fontsize=13, ha='center', fontweight='bold')
ax.text(12.25, 2.9, f'(+{cascade_recall - ar_recall:.3f} improvement)', fontsize=10, ha='center', color='green')
ax.text(12.25, 2.3, f'{key_saves} CRISES', fontsize=14, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=COLOR_CASCADE, alpha=0.3, pad=0.5))
ax.text(12.25, 1.85, 'RESCUED', fontsize=14, ha='center', fontweight='bold',
        bbox=dict(boxstyle='round', facecolor=COLOR_CASCADE, alpha=0.3, pad=0.5))

# ============================================================================
# BOTTOM: Key countries
# ============================================================================
ax.text(7.25, 0.8, 'Top 3: Zimbabwe (77) · Sudan (59) · DRC (40)',
        fontsize=11, ha='center', style='italic', color='darkred')

# ============================================================================
# TITLE - Using actual dissertation topic
# ============================================================================
fig.suptitle('Two-Stage Framework: News Signals for Food Insecurity Early Warning',
        fontsize=17, fontweight='bold', y=0.98)

ax.text(7.25, 5.2, 'Selective deployment: AR for persistence, News for rapid-onset shocks',
        fontsize=12, ha='center', style='italic')

# Configure axes
ax.set_xlim(0, 14.5)
ax.set_ylim(0, 5.5)
ax.axis('off')

plt.tight_layout()

# Save
output_file = OUTPUT_DIR / "ch01_two_stage_framework.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file}")

output_file_png = OUTPUT_DIR / "ch01_two_stage_framework.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png}")

plt.close()

print("\n" + "="*80)
print("FIGURE 2 COMPLETE: Simplified Two-Stage Framework")
print("="*80)
print(f"CLEAN DESIGN - Large fonts, minimal text, clear layout")
print(f"  AR: Precision=Recall={ar_precision:.4f}")
print(f"  Cascade: Recall={cascade_recall:.4f} (+{cascade_recall-ar_recall:.4f})")
print(f"  Key saves: {key_saves}")
