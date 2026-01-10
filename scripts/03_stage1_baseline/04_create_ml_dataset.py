"""
Create ML-Ready Dataset - DISTRICT LEVEL CORRECTED
Combines articles and locations into unified dataset at DISTRICT level.

KEY CORRECTIONS:
1. Uses district-level aggregated files
2. Merges on ipc_geographic_unit_full (unique district-period identifier)
3. Preserves district and region columns throughout

Author: Victor Collins Oppon
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))

# Import from config
from config import (
    BASE_DIR,
    STAGE1_DATA_DIR,
    STAGE1_RESULTS_DIR,
    STAGE2_FEATURES_DIR,
    STAGE2_MODELS_DIR,
    FIGURES_DIR,
    RANDOM_STATE
)
from datetime import datetime

# Paths
# BASE_DIR imported from config

# District pipeline I/O (district_level subfolder)
DISTRICT_DATA_DIR = BASE_DIR / 'data' / 'district_level'
ARTICLES_FILE = DISTRICT_DATA_DIR / 'articles_aggregated.parquet'
LOCATIONS_FILE = DISTRICT_DATA_DIR / 'locations_aggregated.parquet'
OUTPUT_COMPLETE = DISTRICT_DATA_DIR / 'ml_dataset_complete.parquet'
OUTPUT_COMPLETE_CSV = DISTRICT_DATA_DIR / 'ml_dataset_complete.csv'


def main():
    print("=" * 80)
    print("Creating ML-Ready Dataset - DISTRICT LEVEL")
    print("=" * 80)
    print(f"Start time: {datetime.now()}\n")

    # Load aggregated data
    print("1. Loading district-level aggregated articles...")
    articles = pd.read_parquet(ARTICLES_FILE)
    print(f"   Loaded {len(articles):,} article aggregations")
    print(f"   Unique districts: {articles['ipc_district'].nunique():,}")
    print(f"   Columns: {len(articles.columns)}")

    print("\n2. Loading district-level aggregated locations...")
    locations = pd.read_parquet(LOCATIONS_FILE)
    print(f"   Loaded {len(locations):,} location aggregations")
    print(f"   Unique districts: {locations['ipc_district'].nunique():,}")
    print(f"   Columns: {len(locations.columns)}")

    # Define merge keys - KEY CHANGE: Include district and region
    merge_keys = [
        'ipc_id',
        'ipc_country',
        'ipc_country_code',
        'ipc_fips_code',
        'ipc_district',  # NEW: Extracted district
        'ipc_region',  # NEW: Extracted region
        'ipc_geographic_unit',
        'ipc_geographic_unit_full',  # KEY: This is the unique identifier
        'ipc_period_start',
        'ipc_period_end',
        'ipc_period_length_days',
        'ipc_value',
        'ipc_description',
        'ipc_binary_crisis',
        'ipc_is_allowing_assistance',
        'ipc_fewsnet_region',
        'ipc_geographic_group',
        'ipc_scenario',
        'ipc_classification_scale',
        'ipc_reporting_date'
    ]

    # Verify merge keys exist in both datasets
    print("\n3. Verifying merge keys...")
    articles_keys = set(merge_keys) & set(articles.columns)
    locations_keys = set(merge_keys) & set(locations.columns)

    print(f"   Articles has {len(articles_keys)}/{len(merge_keys)} keys")
    print(f"   Locations has {len(locations_keys)}/{len(merge_keys)} keys")

    missing_from_articles = set(merge_keys) - articles_keys
    missing_from_locations = set(merge_keys) - locations_keys

    if missing_from_articles:
        print(f"   [WARNING] Missing from articles: {missing_from_articles}")
    if missing_from_locations:
        print(f"   [WARNING] Missing from locations: {missing_from_locations}")

    # Use common keys
    actual_merge_keys = list(articles_keys & locations_keys)
    print(f"\n   Using {len(actual_merge_keys)} merge keys")

    # Critical key validation
    critical_keys = ['ipc_id', 'ipc_geographic_unit_full', 'ipc_period_start', 'ipc_value']
    missing_critical = set(critical_keys) - set(actual_merge_keys)
    if missing_critical:
        raise ValueError(f"CRITICAL ERROR: Missing essential merge keys: {missing_critical}. "
                         "Pipeline cannot continue. Check upstream scripts.")

    # Ensure date columns have consistent types before merge
    print("\n   Converting date columns to consistent types...")
    date_cols = ['ipc_period_start', 'ipc_period_end', 'ipc_reporting_date']
    for col in date_cols:
        if col in articles.columns:
            articles[col] = pd.to_datetime(articles[col])
        if col in locations.columns:
            locations[col] = pd.to_datetime(locations[col])

    # Ensure ipc_id has consistent type (convert to string)
    print("   Converting ipc_id to consistent string type...")
    if 'ipc_id' in articles.columns:
        articles['ipc_id'] = articles['ipc_id'].astype(str)
    if 'ipc_id' in locations.columns:
        locations['ipc_id'] = locations['ipc_id'].astype(str)

    # Create COMPLETE dataset (inner join)
    print("\n4. Creating COMPLETE dataset (inner join)...")
    df_complete = articles.merge(
        locations,
        on=actual_merge_keys,
        how='inner',
        suffixes=('_articles', '_locations')
    )

    # Add data source flags
    df_complete['has_articles'] = True
    df_complete['has_locations'] = True
    df_complete['data_source'] = 'complete'

    print(f"   Result: {len(df_complete):,} rows")
    print(f"   Unique ipc_geographic_unit_full: {df_complete['ipc_geographic_unit_full'].nunique():,}")
    print(f"   Unique districts: {df_complete['ipc_district'].nunique():,}")
    print(f"   Columns: {len(df_complete.columns)}")

    # Handle match_level columns from both sources
    if 'match_level_articles' in df_complete.columns and 'match_level_locations' in df_complete.columns:
        # Prefer locations match level, fallback to articles
        df_complete['match_level'] = df_complete['match_level_locations'].fillna(
            df_complete['match_level_articles']
        )
        # Drop the separate columns
        df_complete = df_complete.drop(['match_level_articles', 'match_level_locations'], axis=1)

    # Extract date components for analysis
    print("\n5. Adding date components...")
    df_complete['ipc_period_start'] = pd.to_datetime(df_complete['ipc_period_start'])
    df_complete['ipc_period_end'] = pd.to_datetime(df_complete['ipc_period_end'])
    df_complete['year'] = df_complete['ipc_period_start'].dt.year
    df_complete['month'] = df_complete['ipc_period_start'].dt.month
    df_complete['quarter'] = df_complete['ipc_period_start'].dt.quarter

    # Summary statistics
    print("\n" + "=" * 80)
    print("Dataset Summary - DISTRICT LEVEL")
    print("=" * 80)

    print(f"\nTotal records: {len(df_complete):,}")
    print(f"Unique districts (ipc_district): {df_complete['ipc_district'].nunique():,}")
    print(f"Unique geographic_unit_full: {df_complete['ipc_geographic_unit_full'].nunique():,}")
    print(f"Unique IPC assessments (ipc_id): {df_complete['ipc_id'].nunique():,}")
    print(f"Date range: {df_complete['ipc_period_start'].min()} to {df_complete['ipc_period_end'].max()}")
    print(f"Countries: {df_complete['ipc_country'].nunique()}")

    print("\nDistricts per country:")
    district_counts = df_complete.groupby('ipc_country')['ipc_district'].nunique().sort_values(ascending=False)
    for country, count in district_counts.head(15).items():
        records = len(df_complete[df_complete['ipc_country'] == country])
        print(f"   {country}: {count} districts, {records:,} records")

    print("\nMatch level distribution:")
    if 'match_level' in df_complete.columns:
        print(df_complete['match_level'].value_counts())

    print("\nIPC Binary Crisis distribution:")
    print(df_complete['ipc_binary_crisis'].value_counts())

    print("\nIPC Value distribution:")
    print(df_complete['ipc_value'].value_counts().sort_index())

    # Verify district-level granularity
    print("\n" + "=" * 80)
    print("District-Level Verification")
    print("=" * 80)

    # Check if same (district, period) can have different IPC values
    # (This would indicate we're at the right level)
    grouped = df_complete.groupby(['ipc_district', 'ipc_period_start'])['ipc_value'].nunique()
    multi_value = (grouped > 1).sum()
    print(f"\n(district, period) combinations with varying IPC values: {multi_value}")
    print("(Expected: 0 if each district-period has one IPC value)")

    # Check uniqueness of (geographic_unit_full, period)
    unique_check = df_complete.groupby(['ipc_geographic_unit_full', 'ipc_period_start']).size()
    duplicates = (unique_check > 1).sum()
    print(f"(geographic_unit_full, period) duplicates: {duplicates}")
    print("(Expected: 0 - each observation should be unique)")

    if duplicates > 0:
        print("\nWARNING: Found duplicate observations. Deduplication may be needed.")
        # Show sample duplicates
        dup_idx = unique_check[unique_check > 1].head(3).index
        for idx in dup_idx:
            dup_rows = df_complete[
                (df_complete['ipc_geographic_unit_full'] == idx[0]) &
                (df_complete['ipc_period_start'] == idx[1])
            ]
            print(f"\n   Duplicate: {idx[0][:50]}..., {idx[1]}")
            print(f"   Rows: {len(dup_rows)}, IPC values: {dup_rows['ipc_value'].unique()}")

    # Save
    print(f"\n6. Saving COMPLETE dataset...")
    print(f"   Parquet: {OUTPUT_COMPLETE}")
    df_complete.to_parquet(OUTPUT_COMPLETE, index=False)
    print("   [OK] Parquet saved")

    print(f"   CSV: {OUTPUT_COMPLETE_CSV}")
    df_complete.to_csv(OUTPUT_COMPLETE_CSV, index=False)
    print("   [OK] CSV saved")

    # Print column summary
    print("\n" + "=" * 80)
    print("Column Summary")
    print("=" * 80)

    # Categorize columns
    ipc_cols = [c for c in df_complete.columns if c.startswith('ipc_')]
    geo_cols = ['ipc_district', 'ipc_region', 'primary_gadm2', 'primary_gadm3',
                'avg_latitude', 'avg_longitude', 'latitude_std', 'longitude_std']
    geo_cols = [c for c in geo_cols if c in df_complete.columns]
    article_cols = [c for c in df_complete.columns if 'article' in c.lower() or c.endswith('_articles')]
    location_cols = [c for c in df_complete.columns if 'location' in c.lower() or c.endswith('_locations')]
    time_cols = ['year', 'month', 'quarter']

    print(f"\nIPC metadata columns: {len(ipc_cols)}")
    print(f"Geographic columns: {len(geo_cols)}")
    print(f"Article-derived columns: {len(article_cols)}")
    print(f"Location-derived columns: {len(location_cols)}")
    print(f"Time columns: {len(time_cols)}")
    print(f"Total columns: {len(df_complete.columns)}")

    print("\n" + "=" * 80)
    print("ML Dataset Creation Complete - DISTRICT LEVEL")
    print("=" * 80)
    print(f"\nOutput: {OUTPUT_COMPLETE}")
    print(f"\nNext step: Run 05_deduplicate_district.py if duplicates found")
    print(f"Then: Run 06_stage1_feature_engineering_district.py")
    print(f"\nEnd time: {datetime.now()}")


if __name__ == "__main__":
    main()
