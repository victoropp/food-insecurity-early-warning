"""
Analysis: Error Taxonomy
=========================
Classify errors by type (spatial, temporal, feature-driven) and identify
systematic patterns.

Date: December 21, 2025
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))
from config import (BASE_DIR, RESULTS_DIR, STAGE2_MODELS_DIR)

OUTPUT_DIR = RESULTS_DIR / 'analysis' / 'error_taxonomy'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ANALYSIS: ERROR TAXONOMY")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Load predictions
pred_file = STAGE2_MODELS_DIR / 'xgboost' / 'basic_with_ar' / 'xgboost_basic_predictions.csv'

if not pred_file.exists():
    print(f"ERROR: Predictions not found: {pred_file}")
    sys.exit(1)

predictions = pd.read_csv(pred_file)

# Load spatial and temporal error analyses
spatial_file = RESULTS_DIR / 'explainability' / 'domain_audience' / 'spatial_errors' / 'spatial_errors.csv'
temporal_file = RESULTS_DIR / 'explainability' / 'domain_audience' / 'temporal_patterns' / 'temporal_patterns.csv'

# Classify errors
error_taxonomy = []

# False positives
fp_df = predictions[predictions['confusion_youden'] == 'FP']
error_taxonomy.append({
    'error_type': 'False Positive',
    'count': len(fp_df),
    'percentage': len(fp_df) / len(predictions) * 100,
    'description': 'Model predicted crisis but none occurred'
})

# False negatives
fn_df = predictions[predictions['confusion_youden'] == 'FN']
error_taxonomy.append({
    'error_type': 'False Negative',
    'count': len(fn_df),
    'percentage': len(fn_df) / len(predictions) * 100,
    'description': 'Model missed actual crisis'
})

# Save taxonomy
taxonomy_df = pd.DataFrame(error_taxonomy)
taxonomy_file = OUTPUT_DIR / 'error_taxonomy.csv'
taxonomy_df.to_csv(taxonomy_file, index=False)
print(f"Saved: {taxonomy_file}")

print("\nError taxonomy:")
print(taxonomy_df.to_string(index=False))

print("\n" + "=" * 80)
print("ERROR TAXONOMY ANALYSIS COMPLETE")
print("=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
