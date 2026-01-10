"""
XGBoost Pipeline - Phase 2: Combined Ratio + Zscore Feature Engineering
=======================================================================
Phase 2 XGBoost: Combine ratio and zscore features with location encoding.

This script merges the 9 macrocategory ratio features and 9 macrocategory
zscore features from the Simplified Pipeline to create a comprehensive
feature set for XGBoost models.

FEATURES INCLUDED:
1. 9 ratio features (macrocategory ratios only)
2. 9 zscore features (macrocategory zscores only)
3. 2 location features (country_encoded, district_encoded)
4. Spatial CV folds from Simplified Pipeline

Total: 20 features (9 ratio + 9 zscore + 2 location)

Author: Victor Collins Oppon
Date: December 2025
"""

import sys
from pathlib import Path

# Add FINAL_PIPELINE to path (3 levels up)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.preprocessing import LabelEncoder
import warnings

warnings.filterwarnings('ignore')

from config import BASE_DIR, STAGE2_FEATURES_DIR, RANDOM_STATE

# =============================================================================
# PATHS - XGBOOST PIPELINE (SELF-CONTAINED)
# =============================================================================

# Define Phase 2 and Phase 3 directories
PHASE2_RESULTS = STAGE2_FEATURES_DIR / 'phase2_features'
PHASE3_OUTPUT = STAGE2_FEATURES_DIR / 'phase3_combined'
PHASE3_OUTPUT.mkdir(parents=True, exist_ok=True)

# Input files from Phase 2
RATIO_FEATURES = PHASE2_RESULTS / "ratio_features_h8.csv"
ZSCORE_FEATURES = PHASE2_RESULTS / "zscore_features_h8.csv"

# Output files
COMBINED_OUTPUT_CSV = PHASE3_OUTPUT / "combined_basic_features_h8.csv"
COMBINED_OUTPUT_PARQUET = PHASE3_OUTPUT / "combined_basic_features_h8.parquet"

print("=" * 80)
print("XGBOOST PIPELINE - PHASE 2: COMBINED RATIO + ZSCORE FEATURES")
print("=" * 80)
print(f"\nStart time: {datetime.now()}\n")

# =============================================================================
# LOAD PHASE 2 FEATURES
# =============================================================================

print("-" * 40)
print("Loading XGBoost Pipeline Phase 2 features...")
print(f"   Ratio features: {RATIO_FEATURES}")
ratio_df = pd.read_csv(RATIO_FEATURES)
print(f"   Loaded: {len(ratio_df):,} observations")

print(f"   Zscore features: {ZSCORE_FEATURES}")
zscore_df = pd.read_csv(ZSCORE_FEATURES)
print(f"   Loaded: {len(zscore_df):,} observations")

# =============================================================================
# SELECT MACROCATEGORY FEATURES ONLY
# =============================================================================

print("\n" + "-" * 40)
print("Selecting macrocategory features...")

# 9 macrocategories
macro_categories = ['conflict', 'displacement', 'economic', 'food_security', 'governance',
                    'health', 'humanitarian', 'other', 'weather']

# Metadata columns to preserve - ALL geographic, temporal, and IPC data
METADATA_COLS = [
    # Geographic identifiers
    'ipc_country', 'ipc_country_code', 'ipc_fips_code',
    'ipc_district', 'ipc_region', 'ipc_geographic_unit',
    'ipc_geographic_unit_full', 'ipc_fewsnet_region', 'ipc_geographic_group',
    # Spatial coordinates
    'avg_latitude', 'avg_longitude', 'latitude_std', 'longitude_std',
    # Temporal identifiers
    'year_month', 'year', 'year_month_dt',
    'ipc_period_start', 'ipc_period_end',
    'ar_period_start', 'ar_period_end',
    # IPC classification data
    'ipc_value', 'ipc_value_filled',
    'ipc_binary_crisis', 'ipc_binary_crisis_filled',
    'ipc_future_crisis',
    # AR baseline predictions
    'ar_pred_optimal_filled', 'ar_prob_filled',
    # Other
    'african_country_count'
]

# Select only macrocategory ratio and zscore features
ratio_feature_cols = [f'{cat}_ratio' for cat in macro_categories]
zscore_feature_cols = [f'{cat}_zscore' for cat in macro_categories]

print(f"   Ratio features: {len(ratio_feature_cols)} (macrocategories only)")
print(f"   Zscore features: {len(zscore_feature_cols)} (macrocategories only)")

# Verify all features exist
missing_ratio = [col for col in ratio_feature_cols if col not in ratio_df.columns]
missing_zscore = [col for col in zscore_feature_cols if col not in zscore_df.columns]

if missing_ratio:
    print(f"   WARNING: Missing ratio features: {missing_ratio}")
if missing_zscore:
    print(f"   WARNING: Missing zscore features: {missing_zscore}")

# =============================================================================
# MERGE FEATURES
# =============================================================================

print("\n" + "-" * 40)
print("Merging ratio and zscore features...")

# Merge columns
merge_cols = ['ipc_country', 'ipc_geographic_unit_full', 'year_month']

# Filter metadata columns to only those that exist in ratio_df
available_metadata = [col for col in METADATA_COLS if col in ratio_df.columns]
print(f"   Preserving {len(available_metadata)} metadata columns")

# Prepare ratio columns (include fold for CV)
ratio_cols_to_merge = available_metadata + ratio_feature_cols
if 'fold' in ratio_df.columns:
    ratio_cols_to_merge.append('fold')
    print("   Including fold column for spatial CV")

# Merge
combined_df = ratio_df[ratio_cols_to_merge].merge(
    zscore_df[merge_cols + zscore_feature_cols],
    on=merge_cols,
    how='inner',
    validate='1:1'
)

print(f"   Merged observations: {len(combined_df):,}")
print(f"   Feature columns: {len(ratio_feature_cols) + len(zscore_feature_cols)}")

# =============================================================================
# ADD LOCATION ENCODING FOR XGBOOST
# =============================================================================

print("\n" + "-" * 40)
print("Adding location encoding for XGBoost...")

# Label encode country and district
country_encoder = LabelEncoder()
district_encoder = LabelEncoder()

combined_df['country_encoded'] = country_encoder.fit_transform(combined_df['ipc_country'])
combined_df['district_encoded'] = district_encoder.fit_transform(combined_df['ipc_geographic_unit_full'])

print(f"   Unique countries: {combined_df['country_encoded'].nunique()}")
print(f"   Unique districts: {combined_df['district_encoded'].nunique()}")

# =============================================================================
# DATA QUALITY CHECKS
# =============================================================================

print("\n" + "=" * 80)
print("DATA QUALITY CHECKS")
print("=" * 80)

all_feature_cols = ratio_feature_cols + zscore_feature_cols + ['country_encoded', 'district_encoded']

print(f"   Total observations: {len(combined_df):,}")
print(f"   Total features: {len(all_feature_cols)} (9 ratio + 9 zscore + 2 location)")
print(f"   Countries: {combined_df['ipc_country'].nunique()}")
print(f"   Districts: {combined_df['ipc_geographic_unit_full'].nunique()}")
print(f"   Crisis events: {combined_df['ipc_future_crisis'].sum():,} ({100*combined_df['ipc_future_crisis'].mean():.1f}%)")
print(f"   AR predictions available: {combined_df['ar_pred_optimal_filled'].notna().sum():,}")

if 'fold' in combined_df.columns:
    print(f"   Spatial CV folds: {combined_df['fold'].nunique()}")
    print(f"   Fold distribution:")
    for fold in range(combined_df['fold'].nunique()):
        fold_count = (combined_df['fold'] == fold).sum()
        print(f"      Fold {fold}: {fold_count:,} observations")

# Check feature completeness
print("\n   Feature completeness:")
for col in all_feature_cols:
    valid_count = combined_df[col].notna().sum()
    valid_pct = 100 * valid_count / len(combined_df)
    print(f"      {col}: {valid_pct:.1f}% complete")

# =============================================================================
# SAVE OUTPUTS
# =============================================================================

print("\n" + "-" * 40)
print("Saving combined features...")

# Save as CSV
combined_df.to_csv(COMBINED_OUTPUT_CSV, index=False)
print(f"   Saved: {COMBINED_OUTPUT_CSV}")

# Save as Parquet
combined_df.to_parquet(COMBINED_OUTPUT_PARQUET, index=False)
print(f"   Saved: {COMBINED_OUTPUT_PARQUET}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("XGBOOST PIPELINE PHASE 3 COMPLETE: COMBINED BASIC FEATURES")
print("=" * 80)
print(f"\nFeatures: {len(all_feature_cols)} (9 ratio + 9 zscore + 2 location)")
print(f"Observations: {len(combined_df):,}")
print(f"Districts: {combined_df['ipc_geographic_unit_full'].nunique()}")
print(f"Countries: {combined_df['ipc_country'].nunique()}")
print(f"\nOutput: {PHASE3_OUTPUT}")
print(f"End time: {datetime.now()}")
