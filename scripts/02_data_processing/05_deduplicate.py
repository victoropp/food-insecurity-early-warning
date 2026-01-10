"""
Deduplicate IPC assessments - DISTRICT LEVEL
Aggregates to unique (ipc_geographic_unit_full, period) level.

KEY POINT: Each unique (geographic_unit_full, period) is ONE observation.
This script handles any duplicates from the GDELT matching process.

Author: Corrected for district-level analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from config import BASE_DIR

# Paths
BASE_DIR = Path(str(BASE_DIR.parent.parent.parent))

# District pipeline I/O (district_level subfolder)
DISTRICT_DATA_DIR = BASE_DIR / 'data' / 'district_level'
INPUT_FILE = DISTRICT_DATA_DIR / 'ml_dataset_complete.parquet'
OUTPUT_FILE = DISTRICT_DATA_DIR / 'ml_dataset_deduplicated.parquet'
OUTPUT_CSV = DISTRICT_DATA_DIR / 'ml_dataset_deduplicated.csv'


def weighted_mean(group, value_col, weight_col='location_mention_count'):
    """Compute weighted mean using location_mention_count as weights"""
    if weight_col not in group.columns:
        return group[value_col].mean()

    weights = group[weight_col]
    values = group[value_col]

    mask = values.notna() & weights.notna() & (weights > 0)

    if mask.sum() == 0:
        return np.nan

    return (values[mask] * weights[mask]).sum() / weights[mask].sum()


def main():
    print("=" * 80)
    print("IPC Assessment Deduplication - DISTRICT LEVEL")
    print("=" * 80)
    print(f"Start time: {datetime.now()}\n")

    # Load data
    print("1. Loading ML complete dataset...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"   Input rows: {len(df):,}")
    print(f"   Unique ipc_id: {df['ipc_id'].nunique():,}")
    print(f"   Unique ipc_geographic_unit_full: {df['ipc_geographic_unit_full'].nunique():,}")

    # Check for duplicates
    # The unique observation key is (ipc_geographic_unit_full, ipc_period_start)
    df['observation_key'] = (
        df['ipc_geographic_unit_full'].astype(str) + '_' +
        df['ipc_period_start'].astype(str)
    )

    n_unique = df['observation_key'].nunique()
    n_duplicates = len(df) - n_unique

    print(f"   Unique observations: {n_unique:,}")
    print(f"   Duplicate rows: {n_duplicates:,}")

    if n_duplicates == 0:
        print("\n   No duplicates found! Saving dataset as-is...")
        df = df.drop('observation_key', axis=1)
        df.to_parquet(OUTPUT_FILE, index=False)
        df.to_csv(OUTPUT_CSV, index=False)
        print(f"   Saved to {OUTPUT_FILE}")
        return

    print(f"\n2. Analyzing duplicates...")

    # Show sample duplicates
    dup_counts = df.groupby('observation_key').size()
    dup_keys = dup_counts[dup_counts > 1].head(3).index

    for key in dup_keys:
        rows = df[df['observation_key'] == key]
        print(f"\n   Duplicate: {key[:60]}...")
        print(f"   Rows: {len(rows)}")
        print(f"   IPC values: {rows['ipc_value'].unique()}")
        print(f"   Match levels: {rows['match_level'].unique() if 'match_level' in rows.columns else 'N/A'}")

    # Categorize columns for aggregation
    print("\n3. Categorizing columns for aggregation...")

    all_cols = df.columns.tolist()

    # IPC metadata columns (keep first - should be identical)
    ipc_meta_cols = [col for col in all_cols if col.startswith('ipc_')]
    print(f"   IPC metadata columns: {len(ipc_meta_cols)}")

    # Geographic columns (weighted mean)
    geo_cols = ['avg_latitude', 'avg_longitude', 'latitude_std', 'longitude_std']
    geo_cols = [c for c in geo_cols if c in all_cols]
    print(f"   Geographic columns (weighted mean): {len(geo_cols)}")

    # Count columns (sum)
    count_cols = [
        'article_count', 'location_mention_count', 'unique_location_names',
        'unique_cities', 'unique_days', 'unique_sources'
    ]
    count_cols = [c for c in count_cols if c in all_cols]
    print(f"   Count columns (sum): {len(count_cols)}")

    # Categorical columns (first)
    cat_cols = ['primary_gadm2', 'primary_gadm3', 'match_level', 'data_source',
                'has_articles', 'has_locations']
    cat_cols = [c for c in cat_cols if c in all_cols]
    print(f"   Categorical columns (first): {len(cat_cols)}")

    # Score columns (max)
    score_cols = ['match_score']
    score_cols = [c for c in score_cols if c in all_cols]

    # Date component columns (first)
    date_cols = ['year', 'month', 'quarter', 'day']
    date_cols = [c for c in date_cols if c in all_cols]

    # Numeric feature columns (sum) - everything else numeric
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_feature_cols = [
        c for c in numeric_cols
        if c not in ipc_meta_cols
        and c not in geo_cols
        and c not in count_cols
        and c not in score_cols
        and c not in date_cols
        and c not in ['observation_key']
    ]
    print(f"   Numeric feature columns (sum): {len(numeric_feature_cols)}")

    # Build aggregation dictionary
    print("\n4. Building aggregation dictionary...")

    agg_dict = {}

    # IPC metadata - keep first
    for col in ipc_meta_cols:
        if col not in ['observation_key']:
            agg_dict[col] = 'first'

    # Count columns - sum
    for col in count_cols:
        agg_dict[col] = 'sum'

    # Categorical - first
    for col in cat_cols:
        agg_dict[col] = 'first'

    # Score - max
    for col in score_cols:
        agg_dict[col] = 'max'

    # Date components - first
    for col in date_cols:
        agg_dict[col] = 'first'

    # Numeric features - sum
    for col in numeric_feature_cols:
        agg_dict[col] = 'sum'

    print(f"   Total columns to aggregate: {len(agg_dict)}")

    # Compute weighted averages for geographic columns
    print("\n5. Computing weighted averages for geographic columns...")

    weighted_avg_data = {}

    for col in geo_cols:
        print(f"   Computing weighted mean for {col}...", end='\r')
        weighted_avg_data[col] = df.groupby('observation_key').apply(
            lambda g: weighted_mean(g, col),
            include_groups=False
        )

    print(f"   Computed {len(weighted_avg_data)} weighted averages" + " " * 30)

    # Perform standard aggregation
    print("\n6. Aggregating by observation_key...")
    df_agg = df.groupby('observation_key').agg(agg_dict).reset_index()

    # Add weighted averages
    print("\n7. Adding weighted averages...")
    for col, values in weighted_avg_data.items():
        df_agg[col] = df_agg['observation_key'].map(values)
        print(f"   Added {col}")

    # Drop observation_key
    df_agg = df_agg.drop('observation_key', axis=1)

    # Validation
    print("\n8. Validation checks...")

    print(f"\n   Data reduction:")
    print(f"      Before: {len(df):,} rows")
    print(f"      After: {len(df_agg):,} rows")
    print(f"      Reduction: {len(df) - len(df_agg):,} rows ({(1 - len(df_agg)/len(df))*100:.1f}%)")

    # Verify uniqueness
    df_agg['check_key'] = (
        df_agg['ipc_geographic_unit_full'].astype(str) + '_' +
        df_agg['ipc_period_start'].astype(str)
    )
    is_unique = df_agg['check_key'].nunique() == len(df_agg)
    df_agg = df_agg.drop('check_key', axis=1)

    print(f"\n   Uniqueness check: {'PASSED' if is_unique else 'FAILED'}")

    # Check IPC values
    print(f"\n   IPC value summary:")
    print(f"      Unique ipc_id: {df_agg['ipc_id'].nunique():,}")
    print(f"      Missing ipc_value: {df_agg['ipc_value'].isna().sum()}")

    print(f"\n   Geographic coverage:")
    print(f"      Unique districts: {df_agg['ipc_district'].nunique():,}")
    print(f"      Unique geographic_unit_full: {df_agg['ipc_geographic_unit_full'].nunique():,}")
    print(f"      Countries: {df_agg['ipc_country'].nunique()}")

    # Save
    print(f"\n9. Saving deduplicated dataset...")
    print(f"   Parquet: {OUTPUT_FILE}")
    df_agg.to_parquet(OUTPUT_FILE, index=False)
    print("   [OK] Parquet saved")

    print(f"\n   CSV: {OUTPUT_CSV}")
    df_agg.to_csv(OUTPUT_CSV, index=False)
    print("   [OK] CSV saved")

    print("\n" + "=" * 80)
    print("Deduplication Complete - DISTRICT LEVEL")
    print("=" * 80)
    print(f"\nFinal dataset: {len(df_agg):,} unique district-period observations")
    print(f"\nMethodology note for dissertation:")
    print("'GDELT features were aggregated to unique IPC district-period observations")
    print("by summing event counts and computing weighted averages for geographic")
    print(f"coordinates, reducing from {len(df):,} matched records to {len(df_agg):,}")
    print("unique district-period observations.'")
    print(f"\nEnd time: {datetime.now()}")


if __name__ == "__main__":
    main()
