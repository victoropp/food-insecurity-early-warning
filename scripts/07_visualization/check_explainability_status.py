from pathlib import Path
from datetime import datetime, timedelta

figures_dir = Path('C:/GDELT_Africa_Extract/Scripts/district_pipeline/FINAL_PIPELINE/FIGURES')
cutoff = datetime.now() - timedelta(minutes=10)

categories = {
    'coefficients': 'Mixed-Effects Beta Coefficients',
    'regional_biases': 'Regional Biases (by country/location)',
    'spatial_bias': 'Spatial Bias Maps',
    'error_patterns': 'Error Patterns & Confusion Matrices',
    'shap': 'SHAP Analysis (XGBoost)'
}

print("\n" + "="*80)
print("EXPLAINABILITY FIGURES - STATUS CHECK")
print("="*80)
print(f"Checking for updates in last 10 minutes (since {cutoff.strftime('%H:%M')})\n")

for cat, description in categories.items():
    cat_dir = figures_dir / 'explainability' / cat
    if cat_dir.exists():
        files = list(cat_dir.glob('*.png'))
        if files:
            most_recent = max(files, key=lambda f: f.stat().st_mtime)
            mtime = datetime.fromtimestamp(most_recent.stat().st_mtime)
            status = 'UPDATED' if mtime > cutoff else 'OLD'
            print(f"{status:8} {cat:20} {len(files):3} files - Latest: {mtime.strftime('%Y-%m-%d %H:%M')}")
            print(f"         {description}")
        else:
            print(f"EMPTY    {cat:20}   0 files")
            print(f"         {description}")
    else:
        print(f"MISSING  {cat:20} (directory not found)")
        print(f"         {description}")
    print()

print("="*80)
