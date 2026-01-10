"""
Extract Mixed Effects Theme Coefficients - No Hardcoding
Extracts theme-level fixed effects coefficients from ALL_CSV_METRICS_EXTRACTED.json
Saves to simplified JSON for use in visualizations

Date: 2026-01-05
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import json
from pathlib import Path
from config import BASE_DIR

# Directories
BASE_DIR = Path(str(BASE_DIR))
DATA_DIR = BASE_DIR / "Dissertation Write Up" / "DATA_INVENTORIES"
OUTPUT_DIR = DATA_DIR
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("EXTRACTING MIXED EFFECTS THEME COEFFICIENTS")
print("="*80)

# Load ALL_CSV_METRICS_EXTRACTED.json
print("\n[1/3] Loading ALL_CSV_METRICS_EXTRACTED.json...")
with open(DATA_DIR / "ALL_CSV_METRICS_EXTRACTED.json", 'r') as f:
    data = json.load(f)

print("  [OK] Data loaded")

# Extract coefficients from pooled_ratio_with_ar_optimized (the main model)
print("\n[2/3] Extracting theme coefficients from pooled_ratio_with_ar_optimized...")
me_model = data['mixed_effects']['pooled_ratio_with_ar_optimized']
coefficients = me_model['fixed_effects']['coefficients']

# Define the 9 news themes
themes = ['conflict', 'displacement', 'economic', 'food_security',
          'governance', 'health', 'humanitarian', 'other', 'weather']

theme_coefficients = {
    'metadata': {
        'model': 'pooled_ratio_with_ar_optimized',
        'source_file': 'ALL_CSV_METRICS_EXTRACTED.json',
        'extraction_date': '2026-01-05',
        'description': 'Fixed effects coefficients for news theme ratio features'
    },
    'themes': {}
}

for coef_entry in coefficients:
    feature = coef_entry['feature']
    coefficient = coef_entry['coefficient']

    # Extract theme from feature name
    if '_ratio' in feature and feature != '(Intercept)':
        theme = feature.replace('_ratio', '')
        if theme in themes:
            theme_name = theme.replace('_', ' ').title()
            theme_coefficients['themes'][theme] = {
                'theme_name': theme_name,
                'coefficient': float(coefficient),
                'feature': feature
            }

# Sort by coefficient value
sorted_themes = sorted(
    theme_coefficients['themes'].items(),
    key=lambda x: x[1]['coefficient'],
    reverse=True
)

print(f"\n  Theme Rankings by Coefficient:")
for rank, (theme, data) in enumerate(sorted_themes, 1):
    print(f"    {rank}. {data['theme_name']:20s}: {data['coefficient']:+7.2f}")

# Add ranking metadata
theme_coefficients['rankings'] = [
    {
        'rank': rank,
        'theme': theme,
        'theme_name': data['theme_name'],
        'coefficient': data['coefficient']
    }
    for rank, (theme, data) in enumerate(sorted_themes, 1)
]

# Save to JSON
print("\n[3/3] Saving to JSON...")
output_file = OUTPUT_DIR / "MIXED_EFFECTS_THEME_COEFFICIENTS.json"
with open(output_file, 'w') as f:
    json.dump(theme_coefficients, f, indent=2)

print(f"\n[OK] Saved: {output_file}")

# Also save simplified version for easy loading
simplified = {
    theme: data['coefficient']
    for theme, data in theme_coefficients['themes'].items()
}

simplified_file = OUTPUT_DIR / "MIXED_EFFECTS_THEME_COEFFICIENTS_SIMPLE.json"
with open(simplified_file, 'w') as f:
    json.dump(simplified, f, indent=2)

print(f"[OK] Saved simplified: {simplified_file}")

print("\n" + "="*80)
print("EXTRACTION COMPLETE")
print("="*80)
print(f"Total themes: {len(theme_coefficients['themes'])}")
print(f"Top theme: {sorted_themes[0][1]['theme_name']} ({sorted_themes[0][1]['coefficient']:.2f})")
print(f"\nFiles created:")
print(f"  1. {output_file.name}")
print(f"  2. {simplified_file.name}")
