"""
Visualization: Feature Importance Plots
========================================
Create comprehensive feature importance visualizations across all models.

Date: December 21, 2025
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # MUST be before importing pyplot (non-interactive backend)
import matplotlib.pyplot as plt
# NOTE: seaborn removed - causes hanging on Windows
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))
from config import (BASE_DIR, RESULTS_DIR, FIGURES_DIR, VISUALIZATION_CONFIG)

OUTPUT_DIR = FIGURES_DIR / 'feature_importance'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("VISUALIZATION: FEATURE IMPORTANCE PLOTS")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Load consensus features
consensus_file = RESULTS_DIR / 'explainability' / 'integrated' / 'consensus_features.csv'

if not consensus_file.exists():
    print(f"ERROR: Consensus features not found: {consensus_file}")
    sys.exit(1)

consensus_df = pd.read_csv(consensus_file)

# Create comparison plot
print("Creating feature importance comparison plot...")

fig, ax = plt.subplots(figsize=(12, 10))

top_20 = consensus_df.head(20).copy()
top_20 = top_20.sort_values('mean_importance')

bars = ax.barh(range(len(top_20)), top_20['mean_importance'], color='steelblue', alpha=0.8)

# Color bars by n_sources
colors = plt.cm.viridis(top_20['n_sources'] / top_20['n_sources'].max())
for bar, color in zip(bars, colors):
    bar.set_color(color)

ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['feature'])
ax.set_xlabel('Mean Importance Score', fontsize=12)
ax.set_title('Top 20 Features by Consensus Importance', fontsize=14, weight='bold')
ax.grid(axis='x', alpha=0.3)

# Add colorbar
sm = plt.cm.ScalarMappable(cmap=plt.cm.viridis,
                           norm=plt.Normalize(vmin=top_20['n_sources'].min(),
                                            vmax=top_20['n_sources'].max()))
sm.set_array([])
cbar = plt.colorbar(sm, ax=ax)
cbar.set_label('Number of Sources', fontsize=10)

plt.tight_layout()
fig_file = OUTPUT_DIR / 'feature_importance_comparison.png'
plt.savefig(fig_file, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
plt.close()
print(f"Saved: {fig_file}")

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE PLOTS COMPLETE")
print("=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
