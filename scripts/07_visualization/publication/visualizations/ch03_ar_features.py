"""
Figure 7: AR Feature Construction
Publication-grade diagram showing Lt (temporal) and Ls (spatial) autoregressive features

Uses correct AR terminology from FINAL_AR_TERMINOLOGY_VERIFICATION

Date: 2026-01-04
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
from pathlib import Path
from config import BASE_DIR

# Directories
BASE_DIR = Path(rstr(BASE_DIR))
OUTPUT_DIR = BASE_DIR / "Dissertation Write Up" / "LATEX_DISSERTATION" / "figures" / "ch03_methods"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11

# Create figure with 2 panels
fig = plt.figure(figsize=(14, 8))

# Panel A: Temporal Autoregressive Features (Lt)
ax1 = fig.add_subplot(2, 1, 1)

# Time series showing temporal lags
months = np.arange(0, 13)
# Simulated IPC values showing persistence
ipc_values = np.array([2.1, 2.3, 2.5, 2.8, 3.0, 3.2, 3.1, 3.3, 3.4, 3.5, 3.6, 3.7, 3.8])

# Plot time series
ax1.plot(months, ipc_values, marker='o', markersize=8, linewidth=2,
         color='#2E86AB', label='IPC Phase (example district)')

# Highlight forecast point
ax1.plot(12, ipc_values[12], marker='*', markersize=20, color='orange',
         markeredgecolor='black', markeredgewidth=2, zorder=10, label='Forecast target (t)')

# Show temporal lags
for i in range(12):
    ax1.annotate('', xy=(i, ipc_values[i]-0.15), xytext=(12, ipc_values[12]-0.15),
                 arrowprops=dict(arrowstyle='->', lw=1.5, color='gray', alpha=0.5))

# Add phase 3 threshold line
ax1.axhline(y=3.0, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Phase 3 (Crisis threshold)')

ax1.set_xlabel('Time (months before forecast)', fontsize=12, fontweight='bold')
ax1.set_ylabel('IPC Phase', fontsize=12, fontweight='bold')
ax1.set_title('Panel A: Temporal Autoregressive Features (Lt)', fontsize=13, fontweight='bold', pad=15)
ax1.set_xticks(months)
ax1.set_xticklabels(['t-12', 't-11', 't-10', 't-9', 't-8', 't-7', 't-6', 't-5', 't-4', 't-3', 't-2', 't-1', 't'])
ax1.set_ylim(1.8, 4.0)
ax1.grid(axis='y', alpha=0.3, linestyle='--')
ax1.legend(fontsize=10, loc='upper left')

# Add annotation
ax1.text(0.98, 0.05, 'Lt = {IPC(t-1), IPC(t-2), ..., IPC(t-12)}\n12 temporal lags of PAST IPC values',
         transform=ax1.transAxes, fontsize=10, ha='right', va='bottom',
         bbox=dict(boxstyle='round,pad=0.6', facecolor='yellow', alpha=0.6),
         fontweight='bold')

# Panel B: Spatial Autoregressive Features (Ls)
ax2 = fig.add_subplot(2, 1, 2)

# Draw central district
center = mpatches.Circle((0.5, 0.5), 0.08, facecolor='orange', edgecolor='black',
                         linewidth=2, zorder=10, label='Target district (IPC=?)')
ax2.add_patch(center)

# Draw neighboring districts at different distances
neighbors = [
    (0.25, 0.7, 100, 3.2),   # 100km, IPC=3.2
    (0.75, 0.7, 150, 3.5),   # 150km, IPC=3.5
    (0.2, 0.3, 200, 2.8),    # 200km, IPC=2.8
    (0.8, 0.3, 250, 3.0),    # 250km, IPC=3.0
]

weights_sum = 0
weighted_ipc_sum = 0

for x, y, dist, ipc in neighbors:
    # Draw neighbor
    neighbor = mpatches.Circle((x, y), 0.05, facecolor='#2E86AB', edgecolor='black', linewidth=1.5)
    ax2.add_patch(neighbor)

    # Draw distance line
    ax2.plot([0.5, x], [0.5, y], 'k--', linewidth=1, alpha=0.5)

    # Calculate inverse-distance weight
    weight = 1 / dist
    weights_sum += weight
    weighted_ipc_sum += weight * ipc

    # Simplified label - ONLY distance shown on diagram
    ax2.text(x, y - 0.07, f'{dist}km', fontsize=9, ha='center', va='top', fontweight='bold')

# Calculate weighted average
ls_value = weighted_ipc_sum / weights_sum

# Add formula - REPOSITIONED to RIGHT SIDE inside the plot area (not below)
formula_text = (
    r'$Ls = \frac{\sum_{j} \frac{1}{d_{ij}} \cdot IPC_j}{\sum_{j} \frac{1}{d_{ij}}}$' + '\n\n' +
    f'Inverse-distance weighted\naverage (300km radius)\n\n' +
    f'Example: Ls = {ls_value:.2f}'
)

# Position formula on the RIGHT side of the diagram, vertically centered
ax2.text(1.02, 0.5, formula_text, transform=ax2.transAxes, fontsize=9,
         ha='left', va='center',
         bbox=dict(boxstyle='round,pad=0.6', facecolor='lightblue', alpha=0.65),
         fontweight='normal')

ax2.set_xlim(0, 1)
ax2.set_ylim(0, 1)
ax2.set_aspect('equal')
ax2.axis('off')
ax2.set_title('Panel B: Spatial Autoregressive Features (Ls)', fontsize=13, fontweight='bold', pad=15)

# Overall title
fig.suptitle('AR Baseline Features: Lt (Temporal Lags) + Ls (Spatial Lag)',
             fontsize=15, fontweight='bold', y=0.98)

# Footer
footer_text = (
    "AR baseline uses ONLY autoregressive features (lagged dependent variable IPC): Lt (12 temporal lags of past IPC values at t-1 to t-12) "
    "and Ls (inverse-distance weighted average of neighboring districts' IPC within 300km radius). ZERO external covariates (no climate, conflict, news, prices). "
    "This enables measuring marginal value of any feature-based approach beyond pure spatio-temporal persistence. "
    "Performance: AUC-ROC=0.907, Precision=Recall=0.732 using Lt + Ls alone."
)
fig.text(0.5, 0.01, footer_text, ha='center', fontsize=8, style='italic',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3),
         wrap=True)

plt.tight_layout(rect=[0, 0.06, 1, 0.96])

# Save
output_file = OUTPUT_DIR / "ch03_ar_features.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file}")

output_file_png = OUTPUT_DIR / "ch03_ar_features.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png}")

plt.close()

print("\n" + "="*80)
print("FIGURE 7 COMPLETE: AR Feature Construction")
print("="*80)
print("Panel A: Lt = 12 temporal lags (t-1 to t-12) of past IPC values")
print("Panel B: Ls = Inverse-distance weighted neighboring IPC (300km radius)")
print("AR baseline: AUC=0.907 with Lt + Ls ONLY (zero external covariates)")
