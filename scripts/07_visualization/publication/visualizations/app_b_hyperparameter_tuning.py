"""
Appendix Figure B: Hyperparameter Tuning Grid Search Results
Visualizes systematic exploration of XGBoost hyperparameter space
Shows optimal configuration and performance landscape across learning rate and max depth

Date: 2026-01-05
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import seaborn as sns
from pathlib import Path
from config import BASE_DIR

# Directories
BASE_DIR = Path(str(BASE_DIR))
GRID_FILE = BASE_DIR / "RESULTS" / "stage2_models" / "xgboost" / "advanced_with_ar_optimized" / "grid_search_results.csv"
OUTPUT_DIR = BASE_DIR / "Dissertation Write Up" / "LATEX_DISSERTATION" / "figures" / "appendices"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load grid search results
print("Loading grid search results...")
df = pd.read_csv(GRID_FILE)

print(f"\nGrid search configurations: {len(df):,}")
print(f"Best AUC-ROC: {df['mean_test_score'].max():.4f}")
print(f"Worst AUC-ROC: {df['mean_test_score'].min():.4f}")

# Extract key hyperparameters
df['learning_rate'] = df['param_learning_rate']
df['max_depth'] = df['param_max_depth']
df['n_estimators'] = df['param_n_estimators']

# Find optimal configuration
best_idx = df['mean_test_score'].idxmax()
best_config = df.loc[best_idx]

print(f"\nOptimal configuration:")
print(f"  Learning rate: {best_config['learning_rate']}")
print(f"  Max depth: {best_config['max_depth']}")
print(f"  N estimators: {best_config['n_estimators']}")
print(f"  AUC-ROC: {best_config['mean_test_score']:.4f} ±{best_config['std_test_score']:.4f}")

# Create pivot table for heatmap (learning_rate vs max_depth)
# Average across other hyperparameters
pivot = df.groupby(['learning_rate', 'max_depth'])['mean_test_score'].mean().reset_index()
pivot_matrix = pivot.pivot(index='max_depth', columns='learning_rate', values='mean_test_score')

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# Create figure with 3 panels
fig = plt.figure(figsize=(18, 6))

# Panel A: Heatmap - Learning Rate × Max Depth
ax1 = plt.subplot(1, 3, 1)
sns.heatmap(pivot_matrix, annot=True, fmt='.3f', cmap='YlOrRd',
            cbar_kws={'label': 'Mean AUC-ROC'}, ax=ax1, vmin=0.65, vmax=0.72)
ax1.set_title('Panel A: Learning Rate × Max Depth\nPerformance Landscape',
              fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Learning Rate', fontsize=11, fontweight='bold')
ax1.set_ylabel('Max Depth', fontsize=11, fontweight='bold')

# Mark optimal point
best_lr = best_config['learning_rate']
best_depth = best_config['max_depth']
# Find position in pivot matrix
lr_pos = list(pivot_matrix.columns).index(best_lr)
depth_pos = list(pivot_matrix.index).index(best_depth)
ax1.add_patch(plt.Rectangle((lr_pos, depth_pos), 1, 1,
                            fill=False, edgecolor='blue', linewidth=4))
ax1.text(lr_pos + 0.5, depth_pos + 0.5, '★',
         ha='center', va='center', fontsize=20, color='blue', weight='bold')

# Panel B: Marginal distribution - Learning Rate
ax2 = plt.subplot(1, 3, 2)
lr_means = df.groupby('learning_rate').agg({
    'mean_test_score': ['mean', 'std']
}).reset_index()
lr_means.columns = ['learning_rate', 'mean', 'std']

ax2.errorbar(lr_means['learning_rate'], lr_means['mean'],
             yerr=lr_means['std'], marker='o', markersize=8,
             linewidth=2, capsize=5, capthick=2, color='#D55E00')
ax2.axvline(best_lr, color='blue', linestyle='--', linewidth=2,
            label=f'Optimal: {best_lr}')
ax2.set_title('Panel B: Learning Rate Sensitivity\nMarginal Performance',
              fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('Learning Rate', fontsize=11, fontweight='bold')
ax2.set_ylabel('Mean AUC-ROC (±Std)', fontsize=11, fontweight='bold')
ax2.grid(True, alpha=0.3, linestyle='--')
ax2.legend(loc='best', fontsize=10)
ax2.set_xscale('log')

# Panel C: Marginal distribution - Max Depth
ax3 = plt.subplot(1, 3, 3)
depth_means = df.groupby('max_depth').agg({
    'mean_test_score': ['mean', 'std']
}).reset_index()
depth_means.columns = ['max_depth', 'mean', 'std']

ax3.errorbar(depth_means['max_depth'], depth_means['mean'],
             yerr=depth_means['std'], marker='s', markersize=8,
             linewidth=2, capsize=5, capthick=2, color='#009E73')
ax3.axvline(best_depth, color='blue', linestyle='--', linewidth=2,
            label=f'Optimal: {int(best_depth)}')
ax3.set_title('Panel C: Max Depth Sensitivity\nMarginal Performance',
              fontsize=12, fontweight='bold', pad=10)
ax3.set_xlabel('Max Depth', fontsize=11, fontweight='bold')
ax3.set_ylabel('Mean AUC-ROC (±Std)', fontsize=11, fontweight='bold')
ax3.grid(True, alpha=0.3, linestyle='--')
ax3.legend(loc='best', fontsize=10)

# Overall title
fig.suptitle('Hyperparameter Tuning: Systematic Grid Search Exploration',
             fontsize=15, fontweight='bold', y=1.02)

# Add summary statistics box
summary_text = (
    f"GRID SEARCH SUMMARY:\n"
    f"• Total configurations tested: {len(df):,}\n"
    f"• Hyperparameters tuned: 9 (learning_rate, max_depth, n_estimators, etc.)\n"
    f"• Best configuration: lr={best_lr}, depth={int(best_depth)}, n={int(best_config['n_estimators'])}\n"
    f"• Best AUC-ROC: {best_config['mean_test_score']:.4f} ±{best_config['std_test_score']:.4f}\n"
    f"• Performance range: {df['mean_test_score'].min():.4f} to {df['mean_test_score'].max():.4f}\n"
    f"• Key finding: Moderate learning rate (0.01) and depth (7) achieve optimal balance"
)
fig.text(0.5, -0.12, summary_text,
         ha='center', va='top', fontsize=10,
         bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                  alpha=0.95, edgecolor='darkgoldenrod', linewidth=2.5),
         transform=fig.transFigure)

plt.tight_layout()

# Save
output_file = OUTPUT_DIR / "app_b_hyperparameter_tuning.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file}")

output_file_png = OUTPUT_DIR / "app_b_hyperparameter_tuning.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png}")

plt.close()

print("\n" + "="*80)
print("APPENDIX FIGURE B COMPLETE: HYPERPARAMETER TUNING GRID")
print("="*80)
print(f"Configurations tested: {len(df):,}")
print(f"Optimal: lr={best_lr}, depth={int(best_depth)}, n={int(best_config['n_estimators'])}")
print(f"Best AUC-ROC: {best_config['mean_test_score']:.4f}")
