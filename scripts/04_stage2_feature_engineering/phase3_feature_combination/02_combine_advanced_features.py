"""
Combined HMM+DMD Features - XGBoost Pipeline (REDESIGNED)
==========================================================
Phase 2, Step 6: Merge all feature sets (ratio + zscore + HMM + DMD) into a single dataset.

REDESIGN (December 24, 2025):
- HMM features reduced: 12 → 6 (3 ratio + 3 zscore)
- DMD features reduced: 42 → 8 (4 ratio + 4 zscore)
- Total feature reduction: 72 → 32 features (55% reduction)

This script combines:
1. Basic ratio features (9 macrocategories)
2. Basic zscore features (9 macrocategories)
3. HMM ratio features (3 outputs: crisis_prob, transition_risk, entropy)
4. HMM zscore features (3 outputs: crisis_prob, transition_risk, entropy)
5. DMD ratio features (4 crisis outputs)
6. DMD zscore features (4 crisis outputs)
7. Location encoding (country, district)
8. Target variables and metadata

Output: Comprehensive dataset with 32 features (18 base + 6 HMM + 8 DMD)

Author: Victor Collins Oppon, Claude Code
Date: December 2025 (Redesigned: Dec 24, 2025)
"""

import sys
from pathlib import Path

# Add FINAL_PIPELINE to path (3 levels up)
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime
import warnings

warnings.filterwarnings('ignore')

from config import BASE_DIR, STAGE2_FEATURES_DIR, RANDOM_STATE

# Define Phase 2 and Phase 3 directories
PHASE2_RESULTS = STAGE2_FEATURES_DIR / 'phase2_features'
PHASE3_OUTPUT = STAGE2_FEATURES_DIR / 'phase3_combined'
PHASE3_OUTPUT.mkdir(parents=True, exist_ok=True)

def ensure_directories():
    """Ensure output directories exist."""
    PHASE3_OUTPUT.mkdir(parents=True, exist_ok=True)

# Macro categories
MACRO_CATEGORIES = ['conflict', 'displacement', 'economic', 'food_security', 'governance',
                    'health', 'humanitarian', 'other', 'weather']

def ensure_directories():
    """Create output directories."""
    PHASE3_OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("XGBOOST PIPELINE - PHASE 2: COMBINED HMM+DMD FEATURES")
print("=" * 80)

# Feature selection for XGBoost
RATIO_FEATURES = [f'{cat}_ratio' for cat in MACRO_CATEGORIES]
ZSCORE_FEATURES = [f'{cat}_zscore' for cat in MACRO_CATEGORIES]

# REDESIGNED: HMM features reduced from 6 to 3 per type
HMM_RATIO_FEATURES = [
    'hmm_ratio_crisis_prob',      # P(Crisis-Prone state)
    'hmm_ratio_transition_risk',  # P(next_state=Crisis | current_state)
    'hmm_ratio_entropy',          # State uncertainty
]

HMM_ZSCORE_FEATURES = [
    'hmm_zscore_crisis_prob',
    'hmm_zscore_transition_risk',
    'hmm_zscore_entropy',
]

# REDESIGNED: DMD features reduced from 21 to 4 per type (crisis-focused)
DMD_RATIO_FEATURES = [
    'dmd_ratio_crisis_growth_rate',    # Dominant crisis mode growth
    'dmd_ratio_crisis_instability',    # Crisis-weighted sum of growing modes
    'dmd_ratio_crisis_frequency',      # Dominant crisis oscillation period
    'dmd_ratio_crisis_amplitude',      # Dominant crisis mode strength
]

DMD_ZSCORE_FEATURES = [
    'dmd_zscore_crisis_growth_rate',
    'dmd_zscore_crisis_instability',
    'dmd_zscore_crisis_frequency',
    'dmd_zscore_crisis_amplitude',
]

LOCATION_FEATURES = ['country_encoded', 'district_encoded']

TARGET_FEATURES = ['ipc_future_crisis', 'ipc_binary_crisis_filled']

AR_FEATURES = ['ar_pred_optimal_filled', 'ar_prob_filled']

IPC_FEATURES = ['ipc_value_filled']

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
    # Other
    'african_country_count',
    # Fold for CV
    'fold'
]


def load_dataset(file_path, name):
    """Load a feature dataset (parquet or CSV)."""
    print(f"\n   Loading {name}...")

    parquet_path = file_path.with_suffix('.parquet')
    if parquet_path.exists():
        df = pd.read_parquet(parquet_path)
        print(f"   Loaded from parquet: {len(df):,} rows, {len(df.columns)} columns")
    else:
        csv_path = file_path.with_suffix('.csv')
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            print(f"   Loaded from CSV: {len(df):,} rows, {len(df.columns)} columns")
        else:
            raise FileNotFoundError(f"Dataset not found: {file_path}")

    return df


def main():
    """Main execution function."""
    ensure_directories()

    print(f"\nStart time: {datetime.now()}")

    # ==========================================================================
    # LOAD ALL DATASETS
    # ==========================================================================
    print("\n" + "=" * 80)
    print("LOADING FEATURE DATASETS")
    print("=" * 80)

    # Base features from XGBoost Pipeline Phase 2
    ratio_base = load_dataset(PHASE2_RESULTS / 'ratio_features_h8', 'Ratio features (base)')
    zscore_base = load_dataset(PHASE2_RESULTS / 'zscore_features_h8', 'Zscore features (base)')

    # HMM features from XGBoost Pipeline Phase 2
    hmm_ratio = load_dataset(PHASE2_RESULTS / 'hmm_ratio_features_h8', 'HMM ratio features')
    hmm_zscore = load_dataset(PHASE2_RESULTS / 'hmm_zscore_features_h8', 'HMM zscore features')

    # DMD features from XGBoost Pipeline Phase 2
    dmd_ratio = load_dataset(PHASE2_RESULTS / 'hmm_dmd_ratio_features_h8', 'DMD ratio features')
    dmd_zscore = load_dataset(PHASE2_RESULTS / 'hmm_dmd_zscore_features_h8', 'DMD zscore features')

    # ==========================================================================
    # VERIFY COMMON KEY
    # ==========================================================================
    print("\n" + "-" * 40)
    print("Verifying common key...")

    key_cols = ['ipc_geographic_unit_full', 'year_month']

    # Check all datasets have the key columns
    for name, df in [('ratio_base', ratio_base), ('zscore_base', zscore_base),
                     ('hmm_ratio', hmm_ratio), ('hmm_zscore', hmm_zscore),
                     ('dmd_ratio', dmd_ratio), ('dmd_zscore', dmd_zscore)]:
        missing = [col for col in key_cols if col not in df.columns]
        if missing:
            raise ValueError(f"{name} missing key columns: {missing}")

    print(f"   All datasets have key columns: {key_cols}")

    # ==========================================================================
    # SELECT FEATURES FROM EACH DATASET
    # ==========================================================================
    print("\n" + "-" * 40)
    print("Selecting features from each dataset...")

    # Ratio base features (9 macro + location + metadata + target)
    ratio_cols_to_keep = (
        key_cols + RATIO_FEATURES + LOCATION_FEATURES +
        TARGET_FEATURES + AR_FEATURES + IPC_FEATURES + METADATA_COLS
    )
    # Remove duplicates from ratio_cols_to_keep
    ratio_cols_to_keep_unique = []
    for col in ratio_cols_to_keep:
        if col in ratio_base.columns and col not in ratio_cols_to_keep_unique:
            ratio_cols_to_keep_unique.append(col)
    ratio_selected = ratio_base[ratio_cols_to_keep_unique].copy()
    # Also ensure no duplicate columns in ratio_selected itself
    ratio_selected = ratio_selected.loc[:, ~ratio_selected.columns.duplicated()]
    print(f"   Ratio base: {len(ratio_selected.columns)} columns selected")

    # Zscore features (9 macro only - no keys, will merge on them)
    zscore_cols_to_keep = ZSCORE_FEATURES
    zscore_cols_to_keep = [c for c in zscore_cols_to_keep if c in zscore_base.columns]
    zscore_selected = zscore_base[key_cols + zscore_cols_to_keep].copy()
    print(f"   Zscore base: {len(zscore_cols_to_keep)} feature columns selected")

    # HMM ratio features (6 features - no keys, will merge on them)
    hmm_ratio_cols_to_keep = HMM_RATIO_FEATURES
    hmm_ratio_cols_to_keep = [c for c in hmm_ratio_cols_to_keep if c in hmm_ratio.columns]
    hmm_ratio_selected = hmm_ratio[key_cols + hmm_ratio_cols_to_keep].copy()
    print(f"   HMM ratio: {len(hmm_ratio_cols_to_keep)} feature columns selected")

    # HMM zscore features (6 features - no keys, will merge on them)
    hmm_zscore_cols_to_keep = HMM_ZSCORE_FEATURES
    hmm_zscore_cols_to_keep = [c for c in hmm_zscore_cols_to_keep if c in hmm_zscore.columns]
    hmm_zscore_selected = hmm_zscore[key_cols + hmm_zscore_cols_to_keep].copy()
    print(f"   HMM zscore: {len(hmm_zscore_cols_to_keep)} feature columns selected")

    # DMD ratio features (6 features - no keys, will merge on them)
    dmd_ratio_cols_to_keep = DMD_RATIO_FEATURES
    dmd_ratio_cols_to_keep = [c for c in dmd_ratio_cols_to_keep if c in dmd_ratio.columns]
    dmd_ratio_selected = dmd_ratio[key_cols + dmd_ratio_cols_to_keep].copy()
    print(f"   DMD ratio: {len(dmd_ratio_cols_to_keep)} feature columns selected")

    # DMD zscore features (6 features - no keys, will merge on them)
    dmd_zscore_cols_to_keep = DMD_ZSCORE_FEATURES
    dmd_zscore_cols_to_keep = [c for c in dmd_zscore_cols_to_keep if c in dmd_zscore.columns]
    dmd_zscore_selected = dmd_zscore[key_cols + dmd_zscore_cols_to_keep].copy()
    print(f"   DMD zscore: {len(dmd_zscore_cols_to_keep)} feature columns selected")

    # ==========================================================================
    # MERGE ALL DATASETS
    # ==========================================================================
    print("\n" + "-" * 40)
    print("Merging all feature datasets...")

    # Start with ratio features (has all metadata)
    df_combined = ratio_selected.copy()
    print(f"   Starting with ratio base: {len(df_combined):,} rows")

    # Merge zscore features
    df_combined = df_combined.merge(
        zscore_selected,
        on=key_cols,
        how='left',
        suffixes=('', '_dup')
    )
    print(f"   After merging zscore: {len(df_combined):,} rows, {len(df_combined.columns)} columns")

    # Merge HMM ratio features
    df_combined = df_combined.merge(
        hmm_ratio_selected,
        on=key_cols,
        how='left',
        suffixes=('', '_dup')
    )
    print(f"   After merging HMM ratio: {len(df_combined):,} rows, {len(df_combined.columns)} columns")

    # Merge HMM zscore features
    df_combined = df_combined.merge(
        hmm_zscore_selected,
        on=key_cols,
        how='left',
        suffixes=('', '_dup')
    )
    print(f"   After merging HMM zscore: {len(df_combined):,} rows, {len(df_combined.columns)} columns")

    # Merge DMD ratio features
    df_combined = df_combined.merge(
        dmd_ratio_selected,
        on=key_cols,
        how='left',
        suffixes=('', '_dup')
    )
    print(f"   After merging DMD ratio: {len(df_combined):,} rows, {len(df_combined.columns)} columns")

    # Merge DMD zscore features
    df_combined = df_combined.merge(
        dmd_zscore_selected,
        on=key_cols,
        how='left',
        suffixes=('', '_dup')
    )
    print(f"   After merging DMD zscore: {len(df_combined):,} rows, {len(df_combined.columns)} columns")

    # Remove duplicate columns (if any from merge)
    dup_cols = [c for c in df_combined.columns if c.endswith('_dup')]
    if dup_cols:
        df_combined = df_combined.drop(columns=dup_cols)
        print(f"   Removed {len(dup_cols)} duplicate columns")

    # ==========================================================================
    # ADD LOCATION ENCODING
    # ==========================================================================
    print("\n" + "-" * 40)
    print("Adding location encoding for XGBoost...")

    # Encode country and district for XGBoost
    from sklearn.preprocessing import LabelEncoder

    df_combined['country_encoded'] = LabelEncoder().fit_transform(df_combined['ipc_country'])
    df_combined['district_encoded'] = LabelEncoder().fit_transform(df_combined['ipc_geographic_unit_full'])

    print(f"   Unique countries: {df_combined['country_encoded'].nunique()}")
    print(f"   Unique districts: {df_combined['district_encoded'].nunique()}")

    # ==========================================================================
    # FINAL FEATURE COUNT
    # ==========================================================================
    print("\n" + "=" * 80)
    print("FEATURE COUNT SUMMARY")
    print("=" * 80)

    feature_counts = {
        'Ratio macros': len([c for c in df_combined.columns if c.endswith('_ratio')]),
        'Zscore macros': len([c for c in df_combined.columns if c.endswith('_zscore')]),
        'HMM ratio': len([c for c in df_combined.columns if c.startswith('hmm_ratio_')]),
        'HMM zscore': len([c for c in df_combined.columns if c.startswith('hmm_zscore_')]),
        'DMD ratio': len([c for c in df_combined.columns if c.startswith('dmd_ratio_')]),
        'DMD zscore': len([c for c in df_combined.columns if c.startswith('dmd_zscore_')]),
        'Location': len([c for c in df_combined.columns if c in LOCATION_FEATURES]),
        'Target': len([c for c in df_combined.columns if c in TARGET_FEATURES]),
        'AR baseline': len([c for c in df_combined.columns if c in AR_FEATURES]),
        'IPC': len([c for c in df_combined.columns if c in IPC_FEATURES]),
        'Metadata': len([c for c in df_combined.columns if c in METADATA_COLS]),
    }

    for cat, count in feature_counts.items():
        print(f"   {cat:<15}: {count:>3} features")

    total_features = sum(feature_counts.values())
    print(f"\n   TOTAL COLUMNS: {total_features}")
    print(f"   TOTAL ROWS: {len(df_combined):,}")

    # ==========================================================================
    # SAVE COMBINED DATASET
    # ==========================================================================
    print("\n" + "-" * 40)
    print("Saving combined HMM+DMD features...")

    # Parquet (preferred for large datasets)
    output_parquet = PHASE3_OUTPUT / 'combined_advanced_features_h8.parquet'
    df_combined.to_parquet(output_parquet, index=False)
    print(f"   Saved parquet: {output_parquet}")

    # CSV (for compatibility)
    output_csv = PHASE3_OUTPUT / 'combined_advanced_features_h8.csv'
    df_combined.to_csv(output_csv, index=False)
    print(f"   Saved CSV: {output_csv}")

    # ==========================================================================
    # SAVE SUMMARY
    # ==========================================================================
    summary = {
        'timestamp': datetime.now().isoformat(),
        'phase': 'Phase 2: Combined HMM+DMD Features - XGBoost Pipeline',
        'data': {
            'total_observations': len(df_combined),
            'total_features': total_features,
            'feature_breakdown': feature_counts,
            'n_countries': int(df_combined['ipc_country'].nunique()) if 'ipc_country' in df_combined.columns else 0,
            'n_districts': int(df_combined['ipc_geographic_unit_full'].nunique()),
        },
        'feature_list': {
            'ratio_macros': [c for c in df_combined.columns if c.endswith('_ratio')],
            'zscore_macros': [c for c in df_combined.columns if c.endswith('_zscore')],
            'hmm_ratio': [c for c in df_combined.columns if c.startswith('hmm_ratio_')],
            'hmm_zscore': [c for c in df_combined.columns if c.startswith('hmm_zscore_')],
            'dmd_ratio': [c for c in df_combined.columns if c.startswith('dmd_ratio_')],
            'dmd_zscore': [c for c in df_combined.columns if c.startswith('dmd_zscore_')],
        },
        'outputs': {
            'parquet': str(output_parquet),
            'csv': str(output_csv),
        }
    }

    summary_path = PHASE3_OUTPUT / 'combined_advanced_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   Saved summary: {summary_path}")

    # ==========================================================================
    # FINAL SUMMARY
    # ==========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2 STEP 6 COMPLETE: COMBINED HMM+DMD FEATURES")
    print("=" * 80)
    print(f"\n   Total features: {total_features}")
    print(f"   Total observations: {len(df_combined):,}")
    print(f"   Districts: {df_combined['ipc_geographic_unit_full'].nunique():,}")
    print(f"\n   Output files:")
    print(f"   - {output_parquet}")
    print(f"   - {output_csv}")
    print(f"\n   Next step: Phase 3 - XGBoost HMM+DMD Model Training")
    print(f"\nEnd time: {datetime.now()}")


if __name__ == '__main__':
    main()
