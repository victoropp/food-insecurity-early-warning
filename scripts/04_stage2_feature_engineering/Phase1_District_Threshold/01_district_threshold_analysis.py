"""
XGBoost Pipeline - Phase 1: District Threshold Analysis
=======================================================
Apply 200 articles/year threshold at DISTRICT level for XGBoost pipeline.

Self-contained script with XGBoost-specific paths.

Author: Victor Collins Oppon
Date: December 19, 2025
"""

import sys
import pandas as pd
import numpy as np
import json
from pathlib import Path

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent.parent))

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
from pathlib import Path
from datetime import datetime

# =============================================================================
# XGBOOST PIPELINE PATHS (SELF-CONTAINED)
# =============================================================================

# Use self-contained paths from config
from config import STAGE2_DATA_DIR, STAGE2_FEATURES_DIR

# Phase 1 output directory
PHASE1_OUTPUT = STAGE2_FEATURES_DIR / "phase1_district_threshold"
PHASE1_OUTPUT.mkdir(parents=True, exist_ok=True)

# Input from self-contained Stage 2 data
MONTHLY_DATA_FILE = STAGE2_DATA_DIR / "ml_dataset_monthly.parquet"

# Outputs
PHASE1_OUTPUT.mkdir(parents=True, exist_ok=True)
VALID_DISTRICTS_FILE = PHASE1_OUTPUT / "valid_districts.csv"
DISTRICT_STATS_FILE = PHASE1_OUTPUT / "district_statistics.csv"
SUMMARY_FILE = PHASE1_OUTPUT / "phase1_summary.json"

# =============================================================================
# CONFIGURATION
# =============================================================================

MIN_ARTICLES_PER_YEAR = 200  # District-level threshold
MIN_MONTHS_WITH_DATA = 12    # Minimum data availability

print("=" * 80)
print("XGBOOST PIPELINE - PHASE 1: DISTRICT THRESHOLD ANALYSIS")
print("=" * 80)
print(f"\nThreshold: {MIN_ARTICLES_PER_YEAR} articles/year per district")
print(f"Minimum months: {MIN_MONTHS_WITH_DATA}")
print(f"\nStart time: {datetime.now()}\n")

# =============================================================================
# LOAD DATA
# =============================================================================

print("-" * 40)
print("Loading monthly data...")
df = pd.read_parquet(MONTHLY_DATA_FILE)
print(f"   Loaded: {len(df):,} observations")
print(f"   Columns: {len(df.columns)}")

# Use canonical identifier
DISTRICT_COL = 'ipc_geographic_unit_full'
if DISTRICT_COL not in df.columns:
    raise ValueError(f"{DISTRICT_COL} not found!")
print(f"   District column: {DISTRICT_COL}")

# =============================================================================
# EXTRACT YEAR
# =============================================================================

print("\n" + "-" * 40)
print("Extracting year...")

if 'year' not in df.columns:
    if 'year_month' in df.columns:
        df['year'] = df['year_month'].str[:4].astype(int)
    else:
        df['year'] = pd.to_datetime(df['ipc_period_start']).dt.year

print(f"   Years: {sorted(df['year'].unique())}")

# =============================================================================
# COMPUTE ANNUAL ARTICLE COUNTS
# =============================================================================

print("\n" + "-" * 40)
print("Computing annual article counts...")

annual = df.groupby([DISTRICT_COL, 'year']).agg({
    'article_count': 'sum'
}).reset_index()
annual.columns = ['district', 'year', 'annual_articles']

print(f"   Unique districts: {annual['district'].nunique():,}")

# =============================================================================
# COMPUTE DISTRICT STATISTICS
# =============================================================================

print("\n" + "-" * 40)
print("Computing district statistics...")

stats = annual.groupby('district').agg({
    'annual_articles': ['mean', 'std', 'min', 'max', 'count']
}).reset_index()

stats.columns = ['district', 'mean_annual', 'std_annual', 'min_annual', 'max_annual', 'n_years']
stats['meets_threshold'] = stats['mean_annual'] >= MIN_ARTICLES_PER_YEAR

print(f"   Above threshold: {stats['meets_threshold'].sum():,}")
print(f"   Below threshold: {(~stats['meets_threshold']).sum():,}")

# =============================================================================
# CHECK MONTHS WITH DATA
# =============================================================================

print("\n" + "-" * 40)
print("Checking data availability...")

month_counts = df.groupby(DISTRICT_COL).size().reset_index(name='n_months')
month_counts.columns = ['district', 'n_months']

stats = stats.merge(month_counts, on='district', how='left')
stats['meets_months_req'] = stats['n_months'] >= MIN_MONTHS_WITH_DATA

print(f"   Districts with {MIN_MONTHS_WITH_DATA}+ months: {stats['meets_months_req'].sum():,}")

# =============================================================================
# IDENTIFY VALID DISTRICTS
# =============================================================================

print("\n" + "-" * 40)
print("Identifying valid districts...")

stats['is_valid'] = stats['meets_threshold'] & stats['meets_months_req']
valid_districts = stats[stats['is_valid']].copy()

print(f"   Valid districts: {len(valid_districts):,}")

# Add geographic metadata
district_meta = df[[DISTRICT_COL, 'ipc_country']].drop_duplicates()
district_meta.columns = ['district', 'ipc_country']

valid_districts = valid_districts.merge(district_meta, on='district', how='left')

# =============================================================================
# SAVE OUTPUTS
# =============================================================================

print("\n" + "-" * 40)
print("Saving outputs...")

# Valid districts
valid_output = valid_districts[['ipc_country', 'district', 'mean_annual', 'n_months']].copy()
valid_output.columns = ['ipc_country', 'ipc_geographic_unit_full', 'mean_annual_articles', 'n_months']

# CRITICAL: Strip whitespace from district identifiers to match Phase 2 processing
# (Source data has leading tabs that must be removed for consistency)
valid_output['ipc_geographic_unit_full'] = valid_output['ipc_geographic_unit_full'].str.strip()
valid_output['ipc_country'] = valid_output['ipc_country'].str.strip()

# After stripping, some districts may become duplicates - keep best performing one
pre_dedup = len(valid_output)
valid_output = valid_output.sort_values('mean_annual_articles', ascending=False).drop_duplicates(
    subset=['ipc_geographic_unit_full'], keep='first'
)
if len(valid_output) < pre_dedup:
    print(f"   Removed {pre_dedup - len(valid_output)} duplicates after whitespace stripping")

valid_output.to_csv(VALID_DISTRICTS_FILE, index=False)
print(f"   Saved: {VALID_DISTRICTS_FILE}")

# All district statistics
stats.to_csv(DISTRICT_STATS_FILE, index=False)
print(f"   Saved: {DISTRICT_STATS_FILE}")

# Summary JSON
summary = {
    'phase': 'Phase 1 - District Threshold',
    'pipeline': 'XGBoost_Pipeline',
    'threshold': {
        'min_articles_per_year': MIN_ARTICLES_PER_YEAR,
        'min_months_with_data': MIN_MONTHS_WITH_DATA
    },
    'results': {
        'total_districts': int(stats['district'].nunique()),
        'valid_districts': int(len(valid_districts)),
        'filtered_out': int(stats['district'].nunique() - len(valid_districts)),
        'countries': int(valid_districts['ipc_country'].nunique())
    },
    'timestamp': datetime.now().isoformat()
}

with open(SUMMARY_FILE, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"   Saved: {SUMMARY_FILE}")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("PHASE 1 COMPLETE: DISTRICT THRESHOLD ANALYSIS")
print("=" * 80)
print(f"\n   Valid districts: {len(valid_districts):,}")
print(f"   Countries: {valid_districts['ipc_country'].nunique()}")
print(f"   Output: {PHASE1_OUTPUT}")
print(f"\nEnd time: {datetime.now()}")
