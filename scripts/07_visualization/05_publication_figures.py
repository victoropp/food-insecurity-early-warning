"""
Visualization: Publication-Ready Figures
=========================================
Generate all publication-ready figures with consistent style.

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

OUTPUT_DIR = FIGURES_DIR / 'publication_figures'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("VISUALIZATION: PUBLICATION-READY FIGURES")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Set publication style
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.family'] = 'serif'
plt.rcParams['font.size'] = 10

print("Creating publication figures...")

# Figure 1: Model Performance Comparison
print("  1. Model performance comparison...")

comparison_file = RESULTS_DIR / 'baseline_comparison' / 'model_comparison_summary.csv'
if comparison_file.exists():
    comparison_df = pd.read_csv(comparison_file)

    fig, ax = plt.subplots(figsize=(10, 6))

    models = comparison_df['model_name'].values
    aucs = comparison_df['mean_auc'].values
    auc_stds = comparison_df['std_auc'].values if 'std_auc' in comparison_df.columns else [0] * len(aucs)

    x = np.arange(len(models))
    ax.bar(x, aucs, yerr=auc_stds, capsize=5, alpha=0.8, color='steelblue')
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=45, ha='right')
    ax.set_ylabel('AUC-ROC')
    ax.set_title('Model Performance Comparison')
    ax.set_ylim([0.6, 0.9])
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'fig1_model_comparison.png',
                dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()

print("  All publication figures created")

print("\n" + "=" * 80)
print("PUBLICATION FIGURES COMPLETE")
print("=" * 80)
print(f"Output directory: {OUTPUT_DIR}")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
