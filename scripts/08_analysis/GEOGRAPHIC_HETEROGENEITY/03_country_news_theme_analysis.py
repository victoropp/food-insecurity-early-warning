"""
Country-Specific News Theme Analysis
====================================
Analyze which news themes drive predictions in each country using SHAP values

Research Question:
Do different countries respond to different news themes?
- Zimbabwe: Economic themes?
- Sudan: Conflict themes?
- Ethiopia: Food security themes?

Data Source:
-----------
Uses observation-level SHAP values from XGBoost Advanced model:
- shap_features.csv: SHAP values for each observation (35 features)
- shap_metadata.csv: Country labels for each observation

Analysis Approach:
-----------------
1. Load SHAP values + country metadata
2. Group by country, compute mean |SHAP| for each feature
3. Aggregate features into theme categories
4. Rank themes by importance within each country
5. Compare country rankings to global rankings
6. Visualize country-specific theme signatures

Date: 2026-01-06
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import pandas as pd
import numpy as np
from pathlib import Path
import json
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
from config import BASE_DIR
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(rstr(BASE_DIR))
SHAP_DIR = BASE_DIR / "VISUALIZATIONS_PUBLICATION" / "academic_journal_submission" / "analysis_results"
ANALYSIS_OUTPUT = BASE_DIR / "Dissertation Write Up" / "GEOGRAPHIC_HETEROGENEITY_ANALYSIS"
FIGURES_OUTPUT = BASE_DIR / "Dissertation Write Up" / "LATEX_DISSERTATION" / "figures" / "ch05_discussion"

ANALYSIS_OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURES_OUTPUT.mkdir(parents=True, exist_ok=True)

print("="*80)
print("COUNTRY-SPECIFIC NEWS THEME ANALYSIS")
print("="*80)
print(f"\nData source: Observation-level SHAP values (n=23,039)")
print(f"Objective: Identify which news themes drive predictions in each country")
print()

# ============================================================================
# STEP 1: LOAD SHAP VALUES AND METADATA
# ============================================================================

print("-" * 80)
print("STEP 1: Loading SHAP values and country metadata")
print("-" * 80)

shap_features = pd.read_csv(SHAP_DIR / "shap_features.csv")
shap_metadata = pd.read_csv(SHAP_DIR / "shap_metadata.csv")

print(f"\n[OK] Loaded SHAP features: {shap_features.shape}")
print(f"[OK] Loaded metadata: {shap_metadata.shape}")
print(f"  Countries: {shap_metadata['ipc_country'].nunique()}")
print(f"  Features: {len(shap_features.columns)}")

# Combine
shap_data = pd.concat([shap_metadata, shap_features], axis=1)
print(f"\n[OK] Combined dataset: {shap_data.shape}")

# ============================================================================
# STEP 2: DEFINE THEME CATEGORIES
# ============================================================================

print("\n" + "-" * 80)
print("STEP 2: Mapping features to news themes")
print("-" * 80)

# Theme mapping based on feature names
THEME_MAPPING = {
    # Conflict theme
    'conflict_ratio': 'conflict',
    'conflict_zscore': 'conflict',

    # Displacement theme
    'displacement_ratio': 'displacement',
    'displacement_zscore': 'displacement',

    # Economic theme
    'economic_ratio': 'economic',
    'economic_zscore': 'economic',

    # Food security theme
    'food_security_ratio': 'food_security',
    'food_security_zscore': 'food_security',

    # Governance theme
    'governance_ratio': 'governance',
    'governance_zscore': 'governance',

    # Health theme
    'health_ratio': 'health',
    'health_zscore': 'health',

    # Humanitarian theme
    'humanitarian_ratio': 'humanitarian',
    'humanitarian_zscore': 'humanitarian',

    # Weather theme
    'weather_ratio': 'weather',
    'weather_zscore': 'weather',

    # Other theme
    'other_ratio': 'other',
    'other_zscore': 'other',

    # HMM features (regime detection - classify as 'other')
    'hmm_ratio_transition_risk': 'hmm',
    'hmm_ratio_crisis_prob': 'hmm',
    'hmm_zscore_transition_risk': 'hmm',
    'hmm_zscore_crisis_prob': 'hmm',
    'hmm_zscore_entropy': 'hmm',

    # DMD features (temporal evolution - classify as 'other')
    'dmd_ratio_mode_1': 'dmd',
    'dmd_ratio_mode_2': 'dmd',
    'dmd_zscore_mode_1': 'dmd',
    'dmd_zscore_mode_2': 'dmd',

    # Location metadata (not news themes)
    'country_data_density': 'location',
    'country_baseline_conflict': 'location',
    'country_baseline_food_security': 'location'
}

# Filter to news theme features only
news_theme_features = [f for f in shap_features.columns if f in THEME_MAPPING
                       and THEME_MAPPING[f] not in ['location', 'hmm', 'dmd']]

print(f"\n[OK] Mapped {len(news_theme_features)} features to news themes")
print(f"  Themes: {set(THEME_MAPPING[f] for f in news_theme_features)}")

# ============================================================================
# STEP 3: COMPUTE COUNTRY-LEVEL THEME IMPORTANCE
# ============================================================================

print("\n" + "-" * 80)
print("STEP 3: Computing country-level theme importance from SHAP")
print("-" * 80)

country_theme_importance = []

for country in sorted(shap_data['ipc_country'].unique()):
    country_data = shap_data[shap_data['ipc_country'] == country]

    print(f"\n  Processing: {country} (n={len(country_data)})")

    # Compute mean absolute SHAP value for each feature
    theme_scores = {}

    for feature in news_theme_features:
        theme = THEME_MAPPING[feature]
        mean_abs_shap = country_data[feature].abs().mean()

        if theme in theme_scores:
            theme_scores[theme] += mean_abs_shap
        else:
            theme_scores[theme] = mean_abs_shap

    # Normalize to percentages
    total_importance = sum(theme_scores.values())
    if total_importance > 0:
        theme_scores_pct = {theme: (score / total_importance) * 100
                           for theme, score in theme_scores.items()}
    else:
        theme_scores_pct = {theme: 0 for theme in theme_scores}

    # Store results
    for theme, importance in theme_scores_pct.items():
        country_theme_importance.append({
            'country': country,
            'theme': theme,
            'importance_pct': importance,
            'n_observations': len(country_data)
        })

# Convert to DataFrame
country_theme_df = pd.DataFrame(country_theme_importance)

# Compute ranks within each country
country_theme_df['rank'] = country_theme_df.groupby('country')['importance_pct'].rank(
    ascending=False, method='dense').astype(int)

print(f"\n[OK] Computed theme importance for {country_theme_df['country'].nunique()} countries")

# ============================================================================
# STEP 4: IDENTIFY COUNTRY-SPECIFIC THEME SIGNATURES
# ============================================================================

print("\n" + "-" * 80)
print("STEP 4: Country-specific theme signatures")
print("-" * 80)

print(f"\n{'Country':<35} {'#1 Theme':<18} {'#2 Theme':<18} {'#3 Theme':<18}")
print("-" * 90)

country_signatures = {}

for country in sorted(country_theme_df['country'].unique()):
    country_themes = country_theme_df[country_theme_df['country'] == country].sort_values(
        'importance_pct', ascending=False)
    top3 = country_themes.head(3)

    # Format output
    theme_strs = []
    for _, row in top3.iterrows():
        theme_strs.append(f"{row['theme']} ({row['importance_pct']:.1f}%)")

    print(f"{country:<35} {theme_strs[0]:<18} {theme_strs[1]:<18} {theme_strs[2]:<18}")

    # Store signature
    country_signatures[country] = top3[['theme', 'importance_pct', 'rank']].to_dict('records')

# ============================================================================
# STEP 5: COMPARE TO GLOBAL RANKINGS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 5: Global vs Country-Specific Theme Rankings")
print("-" * 80)

# Compute global theme importance
global_theme_importance = country_theme_df.groupby('theme')['importance_pct'].mean().sort_values(ascending=False)

print(f"\nGlobal Theme Rankings (average across all countries):")
print("-" * 50)
for rank, (theme, importance) in enumerate(global_theme_importance.items(), 1):
    print(f"  {rank}. {theme:<20} {importance:>6.2f}%")

# ============================================================================
# STEP 6: SAVE RESULTS
# ============================================================================

print("\n" + "-" * 80)
print("STEP 6: Saving analysis results")
print("-" * 80)

# Save theme importance table
output_file = ANALYSIS_OUTPUT / "country_theme_importance.csv"
country_theme_df.to_csv(output_file, index=False)
print(f"\n[OK] Saved: {output_file.name}")

# Save country signatures
signatures_file = ANALYSIS_OUTPUT / "country_theme_signatures.json"
with open(signatures_file, 'w') as f:
    json.dump(country_signatures, f, indent=2)
print(f"[OK] Saved: {signatures_file.name}")

# Save global rankings
global_rankings = {
    'global_theme_rankings': global_theme_importance.to_dict(),
    'country_count': int(country_theme_df['country'].nunique()),
    'total_observations': int(shap_data.shape[0])
}
global_file = ANALYSIS_OUTPUT / "global_theme_rankings.json"
with open(global_file, 'w') as f:
    json.dump(global_rankings, f, indent=2)
print(f"[OK] Saved: {global_file.name}")

# ============================================================================
# STEP 7: VISUALIZATION - THEME IMPORTANCE HEATMAP
# ============================================================================

print("\n" + "-" * 80)
print("STEP 7: Creating country-theme importance heatmap")
print("-" * 80)

# Pivot data for heatmap
heatmap_data = country_theme_df.pivot(index='country', columns='theme', values='importance_pct')
heatmap_data = heatmap_data.fillna(0)

# Sort countries by key saves (load from existing analysis)
geo_het_file = ANALYSIS_OUTPUT / "country_delta_auc_analysis.csv"
if geo_het_file.exists():
    geo_data = pd.read_csv(geo_het_file)
    country_order = geo_data.sort_values('key_saves', ascending=False)['country'].tolist()
    # Filter to countries in heatmap
    country_order = [c for c in country_order if c in heatmap_data.index]
else:
    country_order = heatmap_data.sum(axis=1).sort_values(ascending=False).index

# Sort themes by global importance
theme_order = global_theme_importance.index.tolist()
theme_order = [t for t in theme_order if t in heatmap_data.columns]

# Create figure
fig, ax = plt.subplots(figsize=(14, 11))

# Create heatmap with better color scheme (blue = more important)
sns.heatmap(heatmap_data.loc[country_order, theme_order],
            cmap='YlGnBu',  # Yellow-Green-Blue: neutral to positive
            annot=True,
            fmt='.1f',
            cbar_kws={'label': 'Theme Importance (% of Total SHAP)'},
            linewidths=0.5,
            vmin=0,
            vmax=20,
            ax=ax)

ax.set_title('Country-Specific News Theme Importance\\nWhich Themes Drive Cascade Predictions in Each Country?',
             fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('News Theme', fontsize=12, fontweight='bold')
ax.set_ylabel('Country (Sorted by Key Saves)', fontsize=12, fontweight='bold')

plt.tight_layout()

# Save
output_fig = FIGURES_OUTPUT / "fig5_country_theme_heatmap.pdf"
plt.savefig(output_fig, dpi=300, bbox_inches='tight')
plt.savefig(output_fig.with_suffix('.png'), dpi=300, bbox_inches='tight')
print(f"\n[OK] Saved: {output_fig.name}")
plt.close()

# ============================================================================
# STEP 8: COMPARISON VISUALIZATION - TOP 3 COUNTRIES
# ============================================================================

print("\n" + "-" * 80)
print("STEP 8: Creating theme comparison for top 3 countries")
print("-" * 80)

# Focus on Zimbabwe, Sudan, DRC (top 3 by key saves)
top3_countries = ['Zimbabwe', 'Sudan', 'Democratic Republic of the Congo']
top3_data = country_theme_df[country_theme_df['country'].isin(top3_countries)]

# Create bar chart comparison
fig, axes = plt.subplots(1, 3, figsize=(16, 6), sharey=False)

for idx, country in enumerate(top3_countries):
    country_data = top3_data[top3_data['country'] == country].sort_values('importance_pct', ascending=True)  # Ascending for horizontal bars (bottom to top)

    bars = axes[idx].barh(country_data['theme'], country_data['importance_pct'],
                          color='#3498DB', edgecolor='black', linewidth=1, alpha=0.85)  # Blue instead of red

    # Add value labels on bars
    for bar in bars:
        width = bar.get_width()
        axes[idx].text(width + 0.3, bar.get_y() + bar.get_height()/2,
                      f'{width:.1f}%',
                      ha='left', va='center', fontsize=9, fontweight='bold')

    axes[idx].set_xlabel('Importance (%)', fontsize=11, fontweight='bold')
    axes[idx].set_title(country, fontsize=12, fontweight='bold')
    axes[idx].grid(axis='x', alpha=0.3, linestyle='--')
    axes[idx].set_xlim(0, max(country_data['importance_pct']) * 1.15)  # Extra space for labels

    if idx == 0:
        axes[idx].set_ylabel('News Theme', fontsize=11, fontweight='bold')

plt.suptitle('Theme Signatures: Top 3 Countries by Key Saves', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()

output_fig2 = FIGURES_OUTPUT / "fig6_top3_theme_comparison.pdf"
plt.savefig(output_fig2, dpi=300, bbox_inches='tight')
plt.savefig(output_fig2.with_suffix('.png'), dpi=300, bbox_inches='tight')
print(f"[OK] Saved: {output_fig2.name}")
plt.close()

print("\n" + "="*80)
print("COUNTRY-SPECIFIC NEWS THEME ANALYSIS COMPLETE")
print("="*80)
print(f"\nKey findings:")
print(f"  - Analyzed {len(country_signatures)} countries (all available in SHAP data)")
print(f"  - Identified {len(global_theme_importance)} news themes")
print(f"  - Created 2 visualizations")
print(f"  - Data source: {shap_data.shape[0]:,} observation-level SHAP values")
print(f"\nData limitation:")
print(f"  - SHAP computed on subset: 13/18 countries from full cascade dataset")
print(f"  - Missing: Burkina Faso, Burundi, Cameroon, Chad, South Sudan")
print(f"  - These 5 countries had 0 key saves, so limited impact on analysis")
print()
