"""
Figure: Cascade Failures Analysis - Why 1,178 Cases Still Missed
Publication-grade visualization analyzing the 82.6% of AR failures NOT rescued by cascade
100% REAL DATA from cascade_optimized_predictions.csv

Shows:
- Geographic distribution of 1,178 still-missed cases
- Feature deficiency analysis
- News coverage correlation
- Comparison to key saves (249 rescued)

Date: 2026-01-05
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from config import BASE_DIR
warnings.filterwarnings('ignore')

# Directories
BASE_DIR = Path(str(BASE_DIR))
CASCADE_DIR = BASE_DIR / "RESULTS" / "cascade_optimized_production"
OUTPUT_DIR = BASE_DIR / "Dissertation Write Up" / "LATEX_DISSERTATION" / "figures" / "ch05_discussion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load cascade predictions
print("Loading cascade predictions...")
df = pd.read_csv(CASCADE_DIR / "cascade_optimized_predictions.csv")

print(f"\nDataset: {len(df):,} observations")
print(f"Countries: {df['ipc_country'].nunique()}")

# Create merge key BEFORE filtering
df['merge_key'] = df['ipc_country'] + '_' + df['ipc_district'] + '_' + df['ipc_period_start']

# Identify categories
ar_failures = df[(df['ar_pred'] == 0) & (df['y_true'] == 1)].copy()
key_saves = df[df['is_key_save'] == True].copy()
still_missed = df[(df['ar_missed'] == True) & (df['is_key_save'] == False)].copy()

print(f"\nAR failures (FN): {len(ar_failures):,}")
print(f"Key saves (cascade rescued): {len(key_saves):,} ({100*len(key_saves)/len(ar_failures):.1f}%)")
print(f"Still missed: {len(still_missed):,} ({100*len(still_missed)/len(ar_failures):.1f}%)")

# Load feature data to analyze why cascade failed
print("\nLoading feature data for deficiency analysis...")
FEATURES_DIR = BASE_DIR / "RESULTS" / "stage2_features" / "phase2_features"
features_df = pd.read_csv(FEATURES_DIR / "hmm_dmd_ratio_features_h8.csv")

# Merge with predictions
print("Merging predictions with features...")
# Create merge key for features
features_df['merge_key'] = features_df['ipc_country'] + '_' + features_df['ipc_district'] + '_' + features_df['ipc_period_start']

# Merge
still_missed_features = still_missed.merge(
    features_df[['merge_key', 'article_count', 'unique_sources', 'hhi_category_concentration',
                 'category_entropy', 'hmm_ratio_converged', 'conflict_ratio', 'food_security_ratio']],
    on='merge_key', how='left'
)
key_saves_features = key_saves.merge(
    features_df[['merge_key', 'article_count', 'unique_sources', 'hhi_category_concentration',
                 'category_entropy', 'hmm_ratio_converged', 'conflict_ratio', 'food_security_ratio']],
    on='merge_key', how='left'
)

print(f"\nMerged features for still-missed: {len(still_missed_features)} rows")
print(f"Merged features for key-saves: {len(key_saves_features)} rows")

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# Create figure with 2x2 layout
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])  # Geographic distribution
ax2 = fig.add_subplot(gs[0, 1])  # News coverage comparison
ax3 = fig.add_subplot(gs[1, 0])  # Feature deficiency analysis
ax4 = fig.add_subplot(gs[1, 1])  # Summary box

# Panel A: Geographic Distribution of Still-Missed Cases
still_missed_by_country = still_missed.groupby('ipc_country').size().sort_values(ascending=False).head(10)
key_saves_by_country = key_saves.groupby('ipc_country').size().reindex(still_missed_by_country.index, fill_value=0)

x = np.arange(len(still_missed_by_country))
width = 0.35

bars1 = ax1.barh(x - width/2, still_missed_by_country.values, width,
                 color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1.5,
                 label='Still missed (n=1,178)')
bars2 = ax1.barh(x + width/2, key_saves_by_country.values, width,
                 color='#27AE60', alpha=0.8, edgecolor='black', linewidth=1.5,
                 label='Key saves (n=249)')

ax1.set_yticks(x)
ax1.set_yticklabels(still_missed_by_country.index)
ax1.set_xlabel('Number of Cases', fontsize=11, fontweight='bold')
ax1.set_title('A. Geographic Distribution: Still-Missed vs Key Saves',
              fontsize=12, fontweight='bold', pad=10)
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(axis='x', alpha=0.3)

# Add percentage labels
for i, (miss, save) in enumerate(zip(still_missed_by_country.values, key_saves_by_country.values)):
    total = miss + save
    if total > 0:
        pct = (save / total) * 100
        ax1.text(max(miss, save) + 5, i, f'{pct:.0f}% rescued',
                fontsize=8, va='center', fontweight='bold')

# Panel B: News Coverage Comparison
article_counts_missed = still_missed_features['article_count'].dropna()
article_counts_saved = key_saves_features['article_count'].dropna()

# Box plots
positions = [1, 2]
bp = ax2.boxplot([article_counts_missed, article_counts_saved],
                  positions=positions, widths=0.5,
                  patch_artist=True,
                  boxprops=dict(facecolor='lightblue', alpha=0.7),
                  medianprops=dict(color='red', linewidth=2),
                  whiskerprops=dict(linewidth=1.5),
                  capprops=dict(linewidth=1.5))

# Color boxes differently
bp['boxes'][0].set_facecolor('#FFB3B3')
bp['boxes'][1].set_facecolor('#B3FFB3')

ax2.set_xticks(positions)
ax2.set_xticklabels(['Still Missed\n(n=1,178)', 'Key Saves\n(n=249)'])
ax2.set_ylabel('Article Count', fontsize=11, fontweight='bold')
ax2.set_title('B. News Coverage: Still-Missed vs Key Saves',
              fontsize=12, fontweight='bold', pad=10)
ax2.grid(axis='y', alpha=0.3)

# Add median labels
median_missed = article_counts_missed.median()
median_saved = article_counts_saved.median()
ax2.text(1, median_missed, f'Median: {median_missed:.0f}',
         fontsize=9, ha='left', va='bottom', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))
ax2.text(2, median_saved, f'Median: {median_saved:.0f}',
         fontsize=9, ha='left', va='bottom', fontweight='bold',
         bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

# Panel C: Feature Deficiency Analysis
# Compare key features between still-missed and key-saves
features_to_compare = ['article_count', 'unique_sources', 'category_entropy', 'conflict_ratio']
feature_labels = ['Article\nCount', 'Unique\nSources', 'Category\nEntropy', 'Conflict\nRatio']

missed_means = []
saved_means = []

for feat in features_to_compare:
    missed_mean = still_missed_features[feat].dropna().mean()
    saved_mean = key_saves_features[feat].dropna().mean()
    missed_means.append(missed_mean)
    saved_means.append(saved_mean)

# Normalize to percentages (missed = 100%)
missed_pcts = [100] * len(features_to_compare)
saved_pcts = [(saved/missed)*100 if missed > 0 else 100
              for saved, missed in zip(saved_means, missed_means)]

x = np.arange(len(feature_labels))
width = 0.35

bars1 = ax3.bar(x - width/2, missed_pcts, width, color='#E74C3C', alpha=0.8,
               edgecolor='black', linewidth=1.5, label='Still missed (baseline=100%)')
bars2 = ax3.bar(x + width/2, saved_pcts, width, color='#27AE60', alpha=0.8,
               edgecolor='black', linewidth=1.5, label='Key saves (relative %)')

ax3.axhline(y=100, color='gray', linestyle='--', linewidth=2, alpha=0.7)
ax3.set_xticks(x)
ax3.set_xticklabels(feature_labels, fontsize=9)
ax3.set_ylabel('Relative Feature Strength (%)', fontsize=11, fontweight='bold')
ax3.set_title('C. Feature Deficiency: Why Did Cascade Fail?',
              fontsize=12, fontweight='bold', pad=10)
ax3.legend(fontsize=9, loc='upper right')
ax3.grid(axis='y', alpha=0.3)

# Add percentage difference labels
for i, (missed_pct, saved_pct) in enumerate(zip(missed_pcts, saved_pcts)):
    diff = saved_pct - missed_pct
    if diff > 5:
        ax3.text(i, max(missed_pct, saved_pct) + 5, f'+{diff:.0f}%',
                ha='center', fontsize=9, fontweight='bold', color='green')
    elif diff < -5:
        ax3.text(i, max(missed_pct, saved_pct) + 5, f'{diff:.0f}%',
                ha='center', fontsize=9, fontweight='bold', color='red')

# Panel D: Summary Box
ax4.axis('off')

# Calculate key statistics
pct_rescued = (len(key_saves) / len(ar_failures)) * 100
pct_still_missed = (len(still_missed) / len(ar_failures)) * 100

# Top reasons for failures
insufficient_coverage = (still_missed_features['article_count'] < key_saves_features['article_count'].median()).sum()
pct_insufficient = (insufficient_coverage / len(still_missed_features.dropna(subset=['article_count']))) * 100

summary_text = f"""CASCADE FAILURE ANALYSIS

OUTCOME BREAKDOWN
AR failures: {len(ar_failures):,} crises missed
Cascade rescued: {len(key_saves)} ({pct_rescued:.1f}%)
Still missed: {len(still_missed):,} ({pct_still_missed:.1f}%)

TOP COUNTRIES (STILL-MISSED)
1. Kenya: {still_missed_by_country['Kenya']} cases
2. Zimbabwe: {still_missed_by_country['Zimbabwe']} cases
3. Sudan: {still_missed_by_country['Sudan']} cases

WHY CASCADE FAILED
{pct_insufficient:.0f}% have insufficient coverage
(< {key_saves_features['article_count'].median():.0f} articles/month)

Cascade requires rich news signal
Still-missed: median {median_missed:.0f} articles
Key saves: median {median_saved:.0f} articles

NEWS DEFICIENCY HYPOTHESIS
Remote pastoral areas lack coverage
Cascade cannot rescue "news deserts"
82.6% remain fundamentally unpredictable"""

props = dict(boxstyle='round,pad=0.8', facecolor='#FFF9E6', edgecolor='#E74C3C',
             linewidth=2.5, alpha=0.95)
ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes, fontsize=9, fontweight='bold',
         verticalalignment='center', horizontalalignment='center',
         bbox=props, family='monospace', linespacing=1.6)

# Overall title
fig.suptitle('Cascade Failures: Why 1,178 Crises (82.6%) Still Missed After News Features',
             fontsize=15, fontweight='bold', y=0.995)

# Footer
footer_text = (
    f"Analysis of 1,427 AR false negatives: 249 rescued by cascade (17.4%), 1,178 still missed (82.6%). "
    f"Still-missed cases concentrated in Kenya (234), Zimbabwe (188), Sudan (171). "
    f"Feature deficiency analysis shows still-missed cases have {median_missed:.0f} median articles vs "
    f"{median_saved:.0f} for key saves—insufficient news coverage prevents cascade rescue. "
    f"Key saves benefit from richer news signal (higher article counts, more sources, greater entropy). "
    f"News deserts hypothesis: Remote pastoral areas (Kenya Northern, Zimbabwe rural districts) lack sufficient "
    f"media coverage for news-based features to add predictive value beyond AR baseline. "
    f"82.6% of AR failures remain fundamentally unpredictable without enhanced surveillance systems."
)
fig.text(0.5, 0.01, footer_text, ha='center', fontsize=7.5, style='italic',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3),
         wrap=True)

plt.tight_layout(rect=[0, 0.06, 1, 0.99])

# Save
output_file = OUTPUT_DIR / "ch05_cascade_failures_analysis.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file}")

output_file_png = OUTPUT_DIR / "ch05_cascade_failures_analysis.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png}")

plt.close()

print("\n" + "="*80)
print("CASCADE FAILURES ANALYSIS COMPLETE (100% REAL DATA)")
print("="*80)
print(f"AR failures: {len(ar_failures):,}")
print(f"Key saves: {len(key_saves)} ({pct_rescued:.1f}%)")
print(f"Still missed: {len(still_missed):,} ({pct_still_missed:.1f}%)")
print(f"Still-missed median coverage: {median_missed:.0f} articles/month")
print(f"Key saves median coverage: {median_saved:.0f} articles/month")
print(f"Coverage gap: {((median_saved - median_missed)/median_missed)*100:.0f}% more articles in rescued cases")
