"""
Extract SHAP Theme Rankings - No Hardcoding
Aggregates SHAP values by news theme from raw SHAP feature CSV
Saves to JSON for use in visualizations

Date: 2026-01-05
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import pandas as pd
import numpy as np
import json
from pathlib import Path
from config import BASE_DIR

# Directories
BASE_DIR = Path(rstr(BASE_DIR))
SHAP_DIR = BASE_DIR / "VISUALIZATIONS_PUBLICATION" / "academic_journal_submission" / "analysis_results"
OUTPUT_DIR = BASE_DIR / "Dissertation Write Up" / "DATA_INVENTORIES"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("EXTRACTING SHAP THEME RANKINGS")
print("="*80)

# Load SHAP features CSV
print("\n[1/4] Loading SHAP features CSV...")
df_shap = pd.read_csv(SHAP_DIR / "shap_features.csv")
print(f"  Loaded: {df_shap.shape[0]:,} observations × {df_shap.shape[1]} features")

# Calculate mean absolute SHAP for each feature
print("\n[2/4] Calculating mean |SHAP| for all features...")
mean_abs_shap = df_shap.abs().mean().sort_values(ascending=False)

print(f"\n  Top 15 features by mean |SHAP|:")
for i, (feat, val) in enumerate(mean_abs_shap.head(15).items(), 1):
    print(f"    {i:2d}. {feat:40s}: {val:.4f}")

# Define the 9 news themes
themes = ['conflict', 'displacement', 'economic', 'food_security',
          'governance', 'health', 'humanitarian', 'other', 'weather']

# Extract theme-level SHAP rankings
print("\n[3/4] Aggregating SHAP values by theme...")

shap_theme_data = {
    'metadata': {
        'n_observations': int(df_shap.shape[0]),
        'n_features': int(df_shap.shape[1]),
        'source_file': 'shap_features.csv',
        'extraction_date': '2026-01-05',
        'description': 'Mean absolute SHAP values aggregated by news theme'
    },
    'themes': {}
}

for theme in themes:
    theme_lower = theme.lower()

    # Get both ratio and zscore SHAP values for this theme
    ratio_col = f"{theme_lower}_ratio"
    zscore_col = f"{theme_lower}_zscore"

    theme_data = {
        'theme_name': theme.replace('_', ' ').title(),
        'ratio_shap': None,
        'zscore_shap': None,
        'total_shap': 0.0
    }

    # Ratio SHAP
    if ratio_col in df_shap.columns:
        ratio_shap = df_shap[ratio_col].abs().mean()
        theme_data['ratio_shap'] = float(ratio_shap)
        theme_data['total_shap'] += float(ratio_shap)

    # Z-score SHAP
    if zscore_col in df_shap.columns:
        zscore_shap = df_shap[zscore_col].abs().mean()
        theme_data['zscore_shap'] = float(zscore_shap)
        theme_data['total_shap'] += float(zscore_shap)

    shap_theme_data['themes'][theme] = theme_data

# Sort themes by zscore SHAP (since z-scores dominate predictions)
sorted_themes = sorted(
    shap_theme_data['themes'].items(),
    key=lambda x: x[1]['zscore_shap'] if x[1]['zscore_shap'] else 0,
    reverse=True
)

print(f"\n  Theme Rankings by Z-Score SHAP:")
for rank, (theme, data) in enumerate(sorted_themes, 1):
    zscore = data['zscore_shap'] if data['zscore_shap'] else 0
    ratio = data['ratio_shap'] if data['ratio_shap'] else 0
    print(f"    {rank}. {data['theme_name']:20s} - Z-score: {zscore:.4f}, Ratio: {ratio:.4f}")

# Add ranking metadata
shap_theme_data['rankings'] = {
    'by_zscore': [
        {
            'rank': rank,
            'theme': theme,
            'theme_name': data['theme_name'],
            'zscore_shap': data['zscore_shap'],
            'ratio_shap': data['ratio_shap']
        }
        for rank, (theme, data) in enumerate(sorted_themes, 1)
    ]
}

# Calculate summary statistics
zscore_shap_values = [d['zscore_shap'] for d in shap_theme_data['themes'].values() if d['zscore_shap']]
shap_theme_data['summary'] = {
    'total_zscore_attribution': float(np.sum(zscore_shap_values)),
    'mean_zscore_attribution': float(np.mean(zscore_shap_values)),
    'top_theme': sorted_themes[0][1]['theme_name'],
    'top_theme_zscore': sorted_themes[0][1]['zscore_shap']
}

# Save to JSON
print("\n[4/4] Saving to JSON...")
output_file = OUTPUT_DIR / "SHAP_THEME_RANKINGS.json"
with open(output_file, 'w') as f:
    json.dump(shap_theme_data, f, indent=2)

print(f"\n[OK] Saved: {output_file}")

# Also save simplified version for easy loading
simplified = {
    theme: {
        'zscore': data['zscore_shap'],
        'ratio': data['ratio_shap']
    }
    for theme, data in shap_theme_data['themes'].items()
}

simplified_file = OUTPUT_DIR / "SHAP_THEME_RANKINGS_SIMPLE.json"
with open(simplified_file, 'w') as f:
    json.dump(simplified, f, indent=2)

print(f"[OK] Saved simplified: {simplified_file}")

print("\n" + "="*80)
print("EXTRACTION COMPLETE")
print("="*80)
print(f"Total themes: {len(themes)}")
print(f"Z-score features dominate: {shap_theme_data['summary']['total_zscore_attribution']:.2f} total attribution")
print(f"Top theme: {shap_theme_data['summary']['top_theme']} ({shap_theme_data['summary']['top_theme_zscore']:.4f})")
print(f"\nFiles created:")
print(f"  1. {output_file.name}")
print(f"  2. {simplified_file.name}")
