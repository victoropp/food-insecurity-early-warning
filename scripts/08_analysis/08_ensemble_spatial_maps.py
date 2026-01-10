#!/usr/bin/env python3
"""
Ensemble Spatial Analysis: Where Does Each Model Excel?
========================================================
Creates maps showing complementary strengths of AR vs Stage 2 vs Ensemble.

Visualizations:
1. Map showing where AR wins vs Stage 2 wins
2. Performance improvement map (Stage 2 vs AR by district)
3. Ensemble benefit map (where ensemble adds most value)
4. Error rate comparison by geographic region

IMPORTANT: NO HARDCODED METRICS - All values computed from data
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR, PHASE4_RESULTS, FIGURES_DIR

# Directories
INPUT_DIR = PHASE4_RESULTS / 'ensemble_analysis'
OUTPUT_DIR = FIGURES_DIR / 'ensemble_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# PROFESSIONAL COLOR SCHEME (Consistent across all visualizations)
# =============================================================================
COLORS = {
    'ar': '#2C5F8D',           # Dark blue (AR Baseline)
    'stage2': '#50C878',       # Emerald green (Stage 2)
    'ensemble': '#7B68BE',     # Purple (Ensemble)
    'ar_light': '#4A90E2',     # Light blue
    'stage2_light': '#7FD99F', # Light green
    'ensemble_light': '#9B87D3',# Light purple
    'positive': '#27AE60',     # Green for improvements
    'negative': '#E74C3C',     # Red for degradations
    'neutral': '#95A5A6'       # Gray for neutral
}

# Configuration (data-driven, not hardcoded)
CONFIG = {
    'top_n_districts': None,  # Will be set dynamically (10% of total or min 20)
    'min_crises_threshold': None,  # Will be set dynamically (median or min 3)
    'max_label_length': 40,  # Truncate long district names
    'figure_dpi': 300
}

def truncate_label(label, max_length=40):
    """Truncate long district names for better visualization."""
    if len(label) <= max_length:
        return label
    # Try to truncate at comma or last word
    if ',' in label:
        parts = label.split(',')
        if len(parts[0]) <= max_length:
            return parts[0]
    # Truncate with ellipsis
    return label[:max_length-3] + '...'

print("=" * 80)
print("ENSEMBLE SPATIAL VISUALIZATION")
print("=" * 80)
print()

# Load data
obs_df = pd.read_csv(INPUT_DIR / 'observation_level_comparison.csv')
country_df = pd.read_csv(INPUT_DIR / 'country_comparative_performance.csv')

print(f"Loaded {len(obs_df):,} observations")
print(f"Loaded {len(country_df):,} geographic units")

# Set dynamic configuration based on data
n_districts = len(obs_df['geographic_unit'].unique())
# FIXED: Cap at 20 districts maximum for readability (was showing 94!)
CONFIG['top_n_districts'] = min(20, max(15, int(0.05 * n_districts)))  # 5% or 15-20 max

# Set minimum crisis threshold based on data distribution
crisis_counts = obs_df.groupby('geographic_unit')['y_true'].sum()
CONFIG['min_crises_threshold'] = max(3, int(crisis_counts.median()))  # Median or min 3

# Minimum observations per district for reliable estimates
# FIXED: Dataset has max 9 obs per district, so use median as threshold
obs_per_district = obs_df.groupby('geographic_unit').size()
CONFIG['min_obs_per_district'] = max(3, int(obs_per_district.quantile(0.25)))  # 25th percentile or min 3

print(f"\nConfiguration (data-driven):")
print(f"  Total districts: {n_districts}")
print(f"  Top N districts to show: {CONFIG['top_n_districts']} (capped at 20 for readability)")
print(f"  Min crises for analysis: {CONFIG['min_crises_threshold']}")
print(f"  Min observations per district: {CONFIG['min_obs_per_district']}")
print()

# =============================================================================
# VISUALIZATION 1: Winner Map (AR vs Stage 2 vs Ensemble)
# =============================================================================

print("Creating Visualization 1: Model Performance by District...")

# Aggregate to district level
district_summary = obs_df.groupby('geographic_unit').agg({
    'y_true': ['sum', 'count'],
    'ar_error': 'mean',
    's2_error': 'mean',
    'ens_error': 'mean',
    'ar_vs_s2_winner': lambda x: (x == 'AR Better').sum(),
    'stage1_prob': 'mean',
    'stage2_prob': 'mean',
    'ensemble_prob': 'mean'
})

district_summary.columns = ['n_crises', 'n_obs', 'ar_error', 's2_error',
                            'ens_error', 'n_ar_wins', 'ar_prob', 's2_prob', 'ens_prob']
district_summary['pct_ar_wins'] = 100 * district_summary['n_ar_wins'] / district_summary['n_obs']
district_summary['winner'] = np.where(
    district_summary['pct_ar_wins'] > 50, 'AR Better', 'Stage 2 Better'
)

# Create figure
fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('Ensemble Comparative Analysis: Spatial Patterns', fontsize=16, fontweight='bold')

# Plot 1: Distribution of winners
ax = axes[0, 0]
winner_counts = district_summary['winner'].value_counts()
# Use consistent professional colors
bar_colors = [COLORS['ar'] if 'AR' in label else COLORS['stage2']
              for label in winner_counts.index]
ax.bar(winner_counts.index, winner_counts.values, color=bar_colors,
       alpha=0.85, edgecolor='white', linewidth=2)
ax.set_title('Districts Where Each Model Excels\n(Based on Observation-Level Accuracy)',
             fontweight='bold', fontsize=12)
ax.set_ylabel('Number of Districts', fontsize=11)
ax.set_xlabel('Winner', fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# Add counts
for i, (label, count) in enumerate(winner_counts.items()):
    ax.text(i, count + 5, f'{count:,}\n({100*count/len(district_summary):.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: Error distribution
ax = axes[0, 1]
error_data = [
    district_summary['ar_error'].dropna(),
    district_summary['s2_error'].dropna(),
    district_summary['ens_error'].dropna()
]
bp = ax.boxplot(error_data, tick_labels=['AR Baseline', 'Stage 2', 'Ensemble'],
                patch_artist=True, widths=0.6)

# Use professional color scheme
colors_box = [COLORS['ar'], COLORS['stage2'], COLORS['ensemble']]
for patch, color in zip(bp['boxes'], colors_box):
    patch.set_facecolor(color)
    patch.set_alpha(0.8)
    patch.set_edgecolor('white')
    patch.set_linewidth(2)

# Style whiskers and medians
for whisker in bp['whiskers']:
    whisker.set(color='#34495E', linewidth=1.5)
for cap in bp['caps']:
    cap.set(color='#34495E', linewidth=1.5)
for median in bp['medians']:
    median.set(color='#34495E', linewidth=2)

ax.set_title('Prediction Error Distribution by Model\n(Mean Absolute Error per District)',
             fontweight='bold', fontsize=12)
ax.set_ylabel('Mean Absolute Error', fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# Add median values
medians = [np.median(d) for d in error_data]
for i, median in enumerate(medians):
    ax.text(i+1, median, f'{median:.3f}', ha='center', va='bottom',
            fontsize=9, fontweight='bold', bbox=dict(boxstyle='round,pad=0.4',
            facecolor='white', alpha=0.9, edgecolor='#34495E', linewidth=1))

# Plot 3: Performance improvement (Stage 2 vs AR)
ax = axes[1, 0]
district_summary['improvement'] = district_summary['ar_error'] - district_summary['s2_error']

# Filter: Only districts with sufficient data
filtered_districts = district_summary[district_summary['n_obs'] >= CONFIG['min_obs_per_district']].copy()

# Use dynamic top N (capped at 20)
top_n = CONFIG['top_n_districts']
improvement_sorted = filtered_districts.sort_values('improvement', ascending=False).head(top_n)

# Professional colors - green for positive, red for negative
colors_imp = [COLORS['positive'] if x > 0 else COLORS['negative']
              for x in improvement_sorted['improvement']]
ax.barh(range(len(improvement_sorted)), improvement_sorted['improvement'],
        color=colors_imp, alpha=0.85, edgecolor='white', linewidth=1.5)
ax.set_yticks(range(len(improvement_sorted)))
# Truncate labels for readability - INCREASED FONT SIZE
truncated_labels = [truncate_label(label, CONFIG['max_label_length'])
                    for label in improvement_sorted.index]
ax.set_yticklabels(truncated_labels, fontsize=9)  # Increased from 8
ax.set_title(f'Top {len(improvement_sorted)} Districts: Stage 2 Performance vs AR\n(Positive = Stage 2 Better, n≥{CONFIG["min_obs_per_district"]})',
             fontweight='bold', fontsize=12)
ax.set_xlabel('Error Reduction (AR error - Stage 2 error)', fontsize=11)
ax.axvline(0, color='#34495E', linestyle='--', linewidth=2, alpha=0.8)
ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)

# Plot 4: Ensemble benefit
ax = axes[1, 1]
# FIXED: Correct calculation of ensemble benefit
# Positive = ensemble is better (lower error)
district_summary['ensemble_benefit'] = (
    np.minimum(district_summary['ar_error'], district_summary['s2_error']) -
    district_summary['ens_error']
)

# Filter: Only districts with sufficient data
filtered_benefit = district_summary[district_summary['n_obs'] >= CONFIG['min_obs_per_district']].copy()

# Show only districts where ensemble provides benefit (positive values)
# Then take top N by magnitude
benefit_positive = filtered_benefit[filtered_benefit['ensemble_benefit'] > 0].copy()
benefit_sorted = benefit_positive.sort_values('ensemble_benefit', ascending=False).head(top_n)

# If we have fewer than top_n positive benefits, show what we have
if len(benefit_sorted) < top_n and len(filtered_benefit) > len(benefit_sorted):
    # Fall back to showing all (including negative) if needed
    benefit_sorted = filtered_benefit.sort_values('ensemble_benefit', ascending=False).head(top_n)

# Use professional colors - green for positive, red for negative
colors_benefit = [COLORS['positive'] if x > 0 else COLORS['negative']
                  for x in benefit_sorted['ensemble_benefit']]
ax.barh(range(len(benefit_sorted)), benefit_sorted['ensemble_benefit'],
        color=colors_benefit, alpha=0.85, edgecolor='white', linewidth=1.5)
ax.set_yticks(range(len(benefit_sorted)))
# Truncate labels for readability - INCREASED FONT SIZE
truncated_labels = [truncate_label(label, CONFIG['max_label_length'])
                    for label in benefit_sorted.index]
ax.set_yticklabels(truncated_labels, fontsize=9)  # Increased from 8

# Count positive vs negative
n_positive = (benefit_sorted['ensemble_benefit'] > 0).sum()
n_negative = len(benefit_sorted) - n_positive

# Updated title acknowledging the finding
ax.set_title(f'District-Level: Ensemble vs Best Individual Model\n(Weighted average: {n_negative} worse, {n_positive} better than winner-take-all, n≥{CONFIG["min_obs_per_district"]})',
             fontweight='bold', fontsize=10.5)
ax.set_xlabel('Error Difference (Best Individual - Ensemble)\nNegative = Ensemble worse than picking winner', fontsize=10)
ax.axvline(0, color='#34495E', linestyle='--', linewidth=2, alpha=0.8)
ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.5)

# Add annotation explaining the finding
ax.text(0.02, 0.98, 'Note: Ensemble uses weighted average (55% AR + 45% Stage 2).\nAt district level, selecting the winner is often better.',
        transform=ax.transAxes, fontsize=8, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='wheat', alpha=0.3))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'ensemble_spatial_patterns.png', dpi=CONFIG['figure_dpi'], bbox_inches='tight')
print(f"[OK] Saved: ensemble_spatial_patterns.png")
plt.close()

# =============================================================================
# VISUALIZATION 2: Crisis Detection Performance
# =============================================================================

print("Creating Visualization 2: Crisis Detection Performance...")

# Separate crisis vs non-crisis observations
crisis_districts = obs_df[obs_df['y_true'] == 1].groupby('geographic_unit').agg({
    'stage1_prob': 'mean',
    'stage2_prob': 'mean',
    'ensemble_prob': 'mean',
    'y_true': 'count'
}).rename(columns={'y_true': 'n_crises'})

# Use dynamic threshold
min_crises = CONFIG['min_crises_threshold']
crisis_districts = crisis_districts[crisis_districts['n_crises'] >= min_crises]

# Create figure
fig, axes = plt.subplots(1, 3, figsize=(18, 6))
fig.suptitle(f'Crisis Detection: Model Probability Distributions\n(Districts with ≥{min_crises} crisis events)',
             fontsize=14, fontweight='bold')

# AR probabilities for crises
ax = axes[0]
ax.hist(crisis_districts['stage1_prob'], bins=30, color=COLORS['ar'], alpha=0.8,
        edgecolor='white', linewidth=1.5)
ar_mean = crisis_districts['stage1_prob'].mean()
ax.axvline(ar_mean, color='#1A3A52', linestyle='--', linewidth=2.5,
           label=f'Mean = {ar_mean:.3f}')
ax.set_title('AR Baseline\nCrisis Probabilities', fontweight='bold', fontsize=12)
ax.set_xlabel('Predicted Probability', fontsize=11)
ax.set_ylabel('Number of Districts', fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

# Stage 2 probabilities for crises
ax = axes[1]
ax.hist(crisis_districts['stage2_prob'], bins=30, color=COLORS['stage2'], alpha=0.8,
        edgecolor='white', linewidth=1.5)
s2_mean = crisis_districts['stage2_prob'].mean()
ax.axvline(s2_mean, color='#2E7D5F', linestyle='--', linewidth=2.5,
           label=f'Mean = {s2_mean:.3f}')
ax.set_title('Stage 2 (XGBoost)\nCrisis Probabilities', fontweight='bold', fontsize=12)
ax.set_xlabel('Predicted Probability', fontsize=11)
ax.set_ylabel('Number of Districts', fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

# Ensemble probabilities for crises
ax = axes[2]
ax.hist(crisis_districts['ensemble_prob'], bins=30, color=COLORS['ensemble'], alpha=0.8,
        edgecolor='white', linewidth=1.5)
ens_mean = crisis_districts['ensemble_prob'].mean()
ax.axvline(ens_mean, color='#5B4E8B', linestyle='--', linewidth=2.5,
           label=f'Mean = {ens_mean:.3f}')
ax.set_title('Ensemble\nCrisis Probabilities', fontweight='bold', fontsize=12)
ax.set_xlabel('Predicted Probability', fontsize=11)
ax.set_ylabel('Number of Districts', fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'crisis_detection_comparison.png', dpi=CONFIG['figure_dpi'], bbox_inches='tight')
print(f"[OK] Saved: crisis_detection_comparison.png")
plt.close()

# =============================================================================
# VISUALIZATION 3: Model Performance Scatter
# =============================================================================

print("Creating Visualization 3: Model Performance Correlation...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Model Performance Correlation Across Districts', fontsize=14, fontweight='bold')

# AR vs Stage 2
ax = axes[0]
scatter = ax.scatter(district_summary['ar_error'], district_summary['s2_error'],
                    c=district_summary['n_crises'], cmap='YlOrRd',
                    alpha=0.7, edgecolor='white', s=60, linewidths=1.5)
ax.plot([0, 0.5], [0, 0.5], color='#34495E', linestyle='--', linewidth=2,
        alpha=0.7, label='Equal Performance')
ax.set_xlabel('AR Baseline Error', fontsize=11)
ax.set_ylabel('Stage 2 Error', fontsize=11)
ax.set_title('AR vs Stage 2 Performance\n(Color = Number of Crises)',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Crisis Events', fontsize=10)

# Best Individual vs Ensemble
ax = axes[1]
district_summary['best_individual_error'] = district_summary[['ar_error', 's2_error']].min(axis=1)
scatter = ax.scatter(district_summary['best_individual_error'],
                    district_summary['ens_error'],
                    c=district_summary['n_crises'], cmap='YlOrRd',
                    alpha=0.7, edgecolor='white', s=60, linewidths=1.5)
ax.plot([0, 0.5], [0, 0.5], color='#34495E', linestyle='--', linewidth=2,
        alpha=0.7, label='Equal Performance')
ax.set_xlabel('Best Individual Model Error', fontsize=11)
ax.set_ylabel('Ensemble Error', fontsize=11)
ax.set_title('Best Individual vs Ensemble\n(Color = Number of Crises)',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Crisis Events', fontsize=10)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'model_performance_correlation.png', dpi=CONFIG['figure_dpi'], bbox_inches='tight')
print(f"[OK] Saved: model_performance_correlation.png")
plt.close()

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("\nCreating summary table...")

summary_stats = pd.DataFrame({
    'Model': ['AR Baseline', 'Stage 2', 'Ensemble'],
    'Mean Error': [
        district_summary['ar_error'].mean(),
        district_summary['s2_error'].mean(),
        district_summary['ens_error'].mean()
    ],
    'Median Error': [
        district_summary['ar_error'].median(),
        district_summary['s2_error'].median(),
        district_summary['ens_error'].median()
    ],
    'Std Error': [
        district_summary['ar_error'].std(),
        district_summary['s2_error'].std(),
        district_summary['ens_error'].std()
    ],
    'Districts Where Best': [
        (district_summary['winner'] == 'AR Better').sum(),
        (district_summary['winner'] == 'Stage 2 Better').sum(),
        np.nan  # Ensemble not compared in this way
    ]
})

summary_stats.to_csv(OUTPUT_DIR / 'spatial_performance_summary.csv', index=False)
print(f"[OK] Saved: spatial_performance_summary.csv")

print()
print("=" * 80)
print("ENSEMBLE SPATIAL VISUALIZATION COMPLETE")
print("=" * 80)
print()
print("KEY FINDINGS:")
print(f"  Districts where AR performs better:     {(district_summary['winner'] == 'AR Better').sum():,}")
print(f"  Districts where Stage 2 performs better: {(district_summary['winner'] == 'Stage 2 Better').sum():,}")
print(f"  Mean error reduction (Ensemble vs best): {district_summary['ensemble_benefit'].mean():.4f}")
print()
print("=" * 80)
