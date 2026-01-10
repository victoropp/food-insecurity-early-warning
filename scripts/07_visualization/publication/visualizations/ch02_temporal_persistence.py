"""
Figure 4: Temporal Persistence in Food Security Crises
Publication-grade time series demonstrating strong temporal autocorrelation

DATA SOURCE: CASCADE_PRODUCTION predictions showing persistence patterns

Date: 2026-01-04
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
from config import BASE_DIR

# Directories
BASE_DIR = Path(rstr(BASE_DIR))
RESULTS_DIR = BASE_DIR / "RESULTS" / "cascade_optimized_production"
OUTPUT_DIR = BASE_DIR / "Dissertation Write Up" / "LATEX_DISSERTATION" / "figures" / "ch02_background"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load predictions data
print("Loading cascade predictions...")
preds = pd.read_csv(RESULTS_DIR / "cascade_optimized_predictions.csv")

print(f"Total observations: {len(preds)}")
print(f"Columns: {preds.columns.tolist()}")

# Compute temporal autocorrelation (ACF) for IPC outcomes
print("\nComputing temporal autocorrelation...")

# Group by district and compute lag-1 correlation
districts = preds.groupby('ipc_geographic_unit_full')

# Compute ACF at different lags
lags = range(1, 13)  # 1 to 12 months
acf_values = []

for lag in lags:
    correlations = []
    for district, group in districts:
        if len(group) > lag:
            # Sort by date
            group = group.sort_values('date')
            # Compute correlation between y(t) and y(t-lag)
            current = group['y_true'].values[lag:]
            lagged = group['y_true'].values[:-lag]
            if len(current) > 10:  # Require at least 10 observations
                corr = np.corrcoef(current, lagged)[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)

    acf_values.append(np.mean(correlations))
    print(f"Lag {lag}: ACF = {acf_values[-1]:.3f} (n={len(correlations)} districts)")

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 11

# Create figure
fig, ax = plt.subplots(figsize=(12, 6))

# Plot ACF
ax.plot(lags, acf_values, marker='o', linewidth=2.5, markersize=8,
        color='#2E86AB', label='Temporal ACF')
ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8, alpha=0.3)
ax.axhline(y=0.7, color='darkred', linestyle='--', linewidth=1.5, alpha=0.5,
           label='High persistence threshold (0.7)')

# Highlight h=8 (forecast horizon)
ax.axvline(x=8, color='orange', linestyle='--', linewidth=2, alpha=0.7,
           label='h=8 month forecast horizon')
ax.plot(8, acf_values[7], marker='*', markersize=20, color='orange',
        markeredgecolor='black', markeredgewidth=1.5, zorder=10)

# Annotate h=8 value
ax.text(8, acf_values[7] + 0.05, f'ACF(8) = {acf_values[7]:.3f}',
        fontsize=12, fontweight='bold', ha='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='yellow', alpha=0.6))

# Labels and formatting
ax.set_xlabel('Lag (months)', fontsize=13, fontweight='bold')
ax.set_ylabel('Autocorrelation Coefficient', fontsize=13, fontweight='bold')
ax.set_title('Temporal Persistence: Strong Autocorrelation Enables AR Baseline',
             fontsize=15, fontweight='bold', pad=15)

ax.set_xlim(0.5, 12.5)
ax.set_ylim(-0.1, 1.0)
ax.set_xticks(lags)
ax.grid(axis='y', alpha=0.3, linestyle='--')
ax.legend(fontsize=11, loc='upper right')

# Add interpretation text
interpretation = (
    f"Strong persistence at h=8: ACF={acf_values[7]:.3f}\n"
    f"\"Yesterday predicts today\" - AR baseline captures this pattern\n"
    f"Lag-1 ACF={acf_values[0]:.3f} shows very high short-term persistence"
)
ax.text(0.02, 0.02, interpretation, transform=ax.transAxes,
        fontsize=10, verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightgray', alpha=0.7))

plt.tight_layout()

# Save
output_file = OUTPUT_DIR / "ch02_temporal_persistence.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file}")

output_file_png = OUTPUT_DIR / "ch02_temporal_persistence.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png}")

plt.close()

print("\n" + "="*80)
print("FIGURE 4 COMPLETE: Temporal Persistence Patterns")
print("="*80)
print(f"ACF at h=8 months: {acf_values[7]:.4f}")
print(f"ACF at lag-1: {acf_values[0]:.4f}")
print(f"Demonstrates strong temporal autocorrelation enabling AR baseline")
