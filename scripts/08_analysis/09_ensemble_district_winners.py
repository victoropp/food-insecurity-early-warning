#!/usr/bin/env python3
"""
Ensemble District Winners Analysis
===================================
Shows which model wins at district level and how ensemble adapts globally.

Key Insights:
- Where does AR win vs Stage 2 win?
- Does the ensemble combine them effectively?
- Why might ensemble be worse at district level but better globally?

Visualizations:
1. Geographic distribution of winners (AR vs Stage 2 vs Tie)
2. Error landscape showing complementary strengths
3. Ensemble weight effectiveness by district type
4. Global vs local performance trade-off
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

# Professional color scheme (consistent)
COLORS = {
    'ar': '#2C5F8D',           # Dark blue (AR Baseline)
    'stage2': '#50C878',       # Emerald green (Stage 2)
    'ensemble': '#7B68BE',     # Purple (Ensemble)
    'tie': '#95A5A6',          # Gray for ties
    'positive': '#27AE60',     # Green for improvements
    'negative': '#E74C3C'      # Red for degradations
}

print("=" * 80)
print("ENSEMBLE DISTRICT WINNERS ANALYSIS")
print("=" * 80)
print()

# Load data
obs_df = pd.read_csv(INPUT_DIR / 'observation_level_comparison.csv')
country_df = pd.read_csv(INPUT_DIR / 'country_comparative_performance.csv')

print(f"Loaded {len(obs_df):,} observations")
print(f"Loaded {len(country_df):,} geographic units")
print()

# =============================================================================
# Aggregate to district level
# =============================================================================

print("Aggregating to district level...")

district_summary = obs_df.groupby('geographic_unit').agg({
    'y_true': ['sum', 'count'],
    'ar_error': 'mean',
    's2_error': 'mean',
    'ens_error': 'mean',
    'stage1_prob': 'mean',
    'stage2_prob': 'mean',
    'ensemble_prob': 'mean'
})

district_summary.columns = ['n_crises', 'n_obs', 'ar_error', 's2_error',
                            'ens_error', 'ar_prob', 's2_prob', 'ens_prob']

# Determine winner at district level
district_summary['best_model'] = district_summary[['ar_error', 's2_error', 'ens_error']].idxmin(axis=1)
district_summary['best_model'] = district_summary['best_model'].map({
    'ar_error': 'AR Baseline',
    's2_error': 'Stage 2',
    'ens_error': 'Ensemble'
})

# AR vs Stage 2 winner (ignoring ensemble)
district_summary['ar_vs_s2_winner'] = np.where(
    district_summary['ar_error'] < district_summary['s2_error'],
    'AR Wins',
    np.where(
        district_summary['s2_error'] < district_summary['ar_error'],
        'Stage 2 Wins',
        'Tie'
    )
)

# Calculate margins
district_summary['ar_s2_margin'] = district_summary['ar_error'] - district_summary['s2_error']
district_summary['ensemble_vs_best'] = district_summary['ens_error'] - np.minimum(
    district_summary['ar_error'], district_summary['s2_error']
)

print(f"Districts analyzed: {len(district_summary)}")
print()

# =============================================================================
# VISUALIZATION 1: District Winners Distribution
# =============================================================================

print("Creating Visualization 1: District Winners...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))
fig.suptitle('District-Level Model Performance: Winners and Trade-offs', fontsize=16, fontweight='bold')

# Plot 1: Winner distribution (AR vs Stage 2 only)
ax = axes[0, 0]
winner_counts = district_summary['ar_vs_s2_winner'].value_counts()
colors_winner = [COLORS['ar'] if 'AR' in label else COLORS['stage2'] if 'Stage 2' in label else COLORS['tie']
                 for label in winner_counts.index]
bars = ax.bar(winner_counts.index, winner_counts.values, color=colors_winner,
              alpha=0.85, edgecolor='white', linewidth=2)
ax.set_title('District-Level Winners: AR vs Stage 2\n(Based on Mean Absolute Error)',
             fontweight='bold', fontsize=12)
ax.set_ylabel('Number of Districts', fontsize=11)
ax.set_xlabel('Winner', fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# Add percentages
for i, (label, count) in enumerate(winner_counts.items()):
    ax.text(i, count + 5, f'{count:,}\n({100*count/len(district_summary):.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 2: Overall winner including ensemble
ax = axes[0, 1]
overall_winner_counts = district_summary['best_model'].value_counts()
colors_overall = [COLORS['ar'] if 'AR' in label else COLORS['stage2'] if 'Stage 2' in label else COLORS['ensemble']
                  for label in overall_winner_counts.index]
bars = ax.bar(overall_winner_counts.index, overall_winner_counts.values, color=colors_overall,
              alpha=0.85, edgecolor='white', linewidth=2)
ax.set_title('District-Level Winners: Including Ensemble\n(Winner-Take-All Comparison)',
             fontweight='bold', fontsize=12)
ax.set_ylabel('Number of Districts', fontsize=11)
ax.set_xlabel('Best Model', fontsize=11)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

# Add percentages
for i, (label, count) in enumerate(overall_winner_counts.items()):
    ax.text(i, count + 5, f'{count:,}\n({100*count/len(district_summary):.1f}%)',
            ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: Performance margin distribution (AR vs Stage 2)
ax = axes[1, 0]
# Histogram of margins with color coding
margins = district_summary['ar_s2_margin'].values
colors_margin = [COLORS['positive'] if m < 0 else COLORS['negative'] for m in margins]
n, bins, patches = ax.hist(margins, bins=40, alpha=0.7, edgecolor='white', linewidth=1)
# Color bars
for i, patch in enumerate(patches):
    if bins[i] < 0:
        patch.set_facecolor(COLORS['stage2'])  # Stage 2 wins (negative margin)
    else:
        patch.set_facecolor(COLORS['ar'])  # AR wins (positive margin)

ax.axvline(0, color='#34495E', linestyle='--', linewidth=2.5, label='No difference', alpha=0.8)
ax.set_title('Performance Margin Distribution\n(AR error - Stage 2 error)',
             fontweight='bold', fontsize=12)
ax.set_xlabel('Error Margin (Negative = Stage 2 Better, Positive = AR Better)', fontsize=11)
ax.set_ylabel('Number of Districts', fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

# Add statistics
median_margin = np.median(margins)
ax.text(0.98, 0.98, f'Median margin: {median_margin:.4f}\nStage 2 wins: {(margins < 0).sum()} districts\nAR wins: {(margins > 0).sum()} districts',
        transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#34495E'))

# Plot 4: Ensemble vs Best Individual
ax = axes[1, 1]
ensemble_diff = district_summary['ensemble_vs_best'].values
n, bins, patches = ax.hist(ensemble_diff, bins=40, alpha=0.7, edgecolor='white', linewidth=1)
# Color bars
for i, patch in enumerate(patches):
    if bins[i] < 0:
        patch.set_facecolor(COLORS['positive'])  # Ensemble wins
    else:
        patch.set_facecolor(COLORS['negative'])  # Best individual wins

ax.axvline(0, color='#34495E', linestyle='--', linewidth=2.5, label='No difference', alpha=0.8)
ax.set_title('Ensemble vs Best Individual Model\n(Ensemble error - Best individual error)',
             fontweight='bold', fontsize=12)
ax.set_xlabel('Error Difference (Negative = Ensemble Better, Positive = Best Individual Better)', fontsize=11)
ax.set_ylabel('Number of Districts', fontsize=11)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

# Add statistics
n_ensemble_wins = (ensemble_diff < 0).sum()
n_individual_wins = (ensemble_diff > 0).sum()
median_diff = np.median(ensemble_diff)
ax.text(0.98, 0.98, f'Median difference: {median_diff:.4f}\nEnsemble better: {n_ensemble_wins} districts ({100*n_ensemble_wins/len(district_summary):.1f}%)\nIndividual better: {n_individual_wins} districts ({100*n_individual_wins/len(district_summary):.1f}%)',
        transform=ax.transAxes, fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white', alpha=0.8, edgecolor='#34495E'))

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'district_winners_analysis.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: district_winners_analysis.png")
plt.close()

# =============================================================================
# VISUALIZATION 2: Global vs Local Performance
# =============================================================================

print("Creating Visualization 2: Global vs Local Performance...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))
fig.suptitle('Global vs Local Performance: Why Ensemble Works Globally', fontsize=14, fontweight='bold')

# Plot 1: Scatter - AR vs Stage 2 error by district
ax = axes[0]
scatter = ax.scatter(district_summary['ar_error'], district_summary['s2_error'],
                    c=district_summary['n_crises'], cmap='YlOrRd',
                    alpha=0.7, edgecolor='white', s=60, linewidths=1.5)
ax.plot([0, 1], [0, 1], color='#34495E', linestyle='--', linewidth=2, alpha=0.7, label='Equal Performance')
ax.set_xlabel('AR Baseline Error', fontsize=11)
ax.set_ylabel('Stage 2 Error', fontsize=11)
ax.set_title('AR vs Stage 2 Performance by District\n(Points below line = Stage 2 better)',
             fontweight='bold', fontsize=12)
ax.legend(fontsize=10)
ax.grid(alpha=0.3, linestyle='--', linewidth=0.5)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax)
cbar.set_label('Crisis Events', fontsize=10)

# Add regions
ax.axhspan(0, 0.5, xmin=0.5/1, alpha=0.1, color=COLORS['stage2'], label='Stage 2 dominates')
ax.axvspan(0, 0.5, ymin=0.5/1, alpha=0.1, color=COLORS['ar'], label='AR dominates')

# Plot 2: Performance by winner type
ax = axes[1]
# Group by winner type and calculate mean errors
winner_groups = district_summary.groupby('ar_vs_s2_winner').agg({
    'ar_error': 'mean',
    's2_error': 'mean',
    'ens_error': 'mean'
})
winner_groups['n_districts'] = district_summary.groupby('ar_vs_s2_winner').size()

x_pos = np.arange(len(winner_groups))
width = 0.25

bars1 = ax.bar(x_pos - width, winner_groups['ar_error'], width, label='AR Error',
               color=COLORS['ar'], alpha=0.85, edgecolor='white', linewidth=1.5)
bars2 = ax.bar(x_pos, winner_groups['s2_error'], width, label='Stage 2 Error',
               color=COLORS['stage2'], alpha=0.85, edgecolor='white', linewidth=1.5)
bars3 = ax.bar(x_pos + width, winner_groups['ens_error'], width, label='Ensemble Error',
               color=COLORS['ensemble'], alpha=0.85, edgecolor='white', linewidth=1.5)

ax.set_xlabel('Winner Type', fontsize=11)
ax.set_ylabel('Mean Absolute Error', fontsize=11)
ax.set_title('Mean Error by Winner Type\n(Shows how ensemble adapts)',
             fontweight='bold', fontsize=12)
ax.set_xticks(x_pos)
ax.set_xticklabels([f'{idx}\n({row["n_districts"]} districts)'
                     for idx, row in winner_groups.iterrows()], fontsize=10)
ax.legend(fontsize=10)
ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'global_vs_local_performance.png', dpi=300, bbox_inches='tight')
print(f"[OK] Saved: global_vs_local_performance.png")
plt.close()

# =============================================================================
# SUMMARY STATISTICS
# =============================================================================

print("\n" + "=" * 80)
print("DISTRICT-LEVEL ANALYSIS SUMMARY")
print("=" * 80)
print()

print("WINNER DISTRIBUTION (AR vs Stage 2):")
for winner, count in winner_counts.items():
    print(f"  {winner:<20}: {count:>5} districts ({100*count/len(district_summary):>5.1f}%)")
print()

print("OVERALL BEST MODEL (Including Ensemble):")
for model, count in overall_winner_counts.items():
    print(f"  {model:<20}: {count:>5} districts ({100*count/len(district_summary):>5.1f}%)")
print()

print("ENSEMBLE PERFORMANCE:")
print(f"  Districts where ensemble is best:  {n_ensemble_wins:>5} ({100*n_ensemble_wins/len(district_summary):>5.1f}%)")
print(f"  Districts where individual is best: {n_individual_wins:>5} ({100*n_individual_wins/len(district_summary):>5.1f}%)")
print(f"  Median performance difference:      {median_diff:>7.4f}")
print()

print("KEY INSIGHT:")
print("  At the district level, selecting the winner (AR OR Stage 2) is often better")
print("  than using a weighted average. However, globally (across all districts),")
print("  the ensemble wins because you don't know in advance which model will win")
print("  for each district. The ensemble's fixed weight (55% AR + 45% Stage 2)")
print("  provides robust performance across all cases.")
print()

# Save summary
summary_stats = {
    'total_districts': len(district_summary),
    'ar_wins': int((district_summary['ar_vs_s2_winner'] == 'AR Wins').sum()),
    'stage2_wins': int((district_summary['ar_vs_s2_winner'] == 'Stage 2 Wins').sum()),
    'ties': int((district_summary['ar_vs_s2_winner'] == 'Tie').sum()),
    'ensemble_best': int(n_ensemble_wins),
    'individual_best': int(n_individual_wins),
    'median_ar_s2_margin': float(median_margin),
    'median_ensemble_vs_best': float(median_diff),
    'mean_error_ar': float(district_summary['ar_error'].mean()),
    'mean_error_s2': float(district_summary['s2_error'].mean()),
    'mean_error_ensemble': float(district_summary['ens_error'].mean())
}

import json
with open(OUTPUT_DIR / 'district_winners_summary.json', 'w') as f:
    json.dump(summary_stats, f, indent=2)
print(f"[OK] Saved: district_winners_summary.json")

print()
print("=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
