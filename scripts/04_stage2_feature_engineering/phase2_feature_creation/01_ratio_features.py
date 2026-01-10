"""
Ratio Feature Engineering for XGBoost Pipeline (Stratified Spatial CV Variant)
===============================================================================
Phase 2, Step 1: Create ratio-based features for pooled models.

VARIANT: STRATIFIED SPATIAL CV
==============================
This variant uses:
1. STRATIFIED SPATIAL CV - Balanced folds by crisis rate AND geography
2. MEANINGFUL LOCATION FEATURES - Replace arbitrary label encoding with
   features that describe WHY a location has certain risk characteristics

KEY DIFFERENCES FROM KMEANS-ONLY PIPELINE:
- Uses stratified spatial CV (balanced crisis rates across folds)
- Adds country_historical_crisis_rate, district_historical_crisis_rate
- Adds country_baseline_conflict, country_baseline_food_security
- Adds district_crisis_volatility, country_crisis_trend
- Removes arbitrary country_encoded, district_encoded

FEATURES CREATED:
1. Category ratios (proportion of total articles)
2. Category concentrations (HHI, entropy)
3. Temporal trends (3-month and 6-month slopes)
4. Momentum features (delta)
5. MEANINGFUL LOCATION FEATURES (NEW)

Author: Victor Collins Oppon
Date: December 2025
"""

import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import from config
from config import (
    BASE_DIR,
    STAGE1_DATA_DIR,
    STAGE1_RESULTS_DIR,
    STAGE2_DATA_DIR,
    STAGE2_FEATURES_DIR,
    STAGE2_MODELS_DIR,
    FIGURES_DIR,
    RANDOM_STATE
)


import pandas as pd
import numpy as np
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Use self-contained paths from config (no hardcoded paths)
PHASE1_RESULTS = STAGE2_FEATURES_DIR / 'phase1_district_threshold'
PHASE2_RESULTS = STAGE2_FEATURES_DIR / 'phase2_features'

# Macro categories
MACRO_CATEGORIES = [
    'conflict_category', 'displacement_category', 'economic_category',
    'food_security_category', 'governance_category', 'health_category',
    'humanitarian_category', 'other_category', 'weather_category'
]

# Output files
OUTPUT_FILES = {
    'valid_districts': 'valid_districts.csv',
    'ratio_features': 'ratio_features_h8.csv',
}

def ensure_directories():
    PHASE2_RESULTS.mkdir(parents=True, exist_ok=True)
# Spatial CV configuration (self-contained)
N_FOLDS = 5
RANDOM_STATE = 42

def create_spatial_folds(df, district_col, n_folds=5, random_state=42):
    """
    Create spatial folds by district using KMeans clustering on coordinates.

    NOTE: This uses standard KMeans CV for compatibility with Mixed Effects models.
    XGBoost scripts will override with stratified spatial CV at training time.

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset containing district identifiers and coordinates
    district_col : str
        Column name for district identifier
    n_folds : int
        Number of spatial folds (default: 5)
    random_state : int
        Random seed for reproducibility (default: 42)

    Returns:
    --------
    pd.Series
        Fold assignments for each observation
    """
    from sklearn.cluster import KMeans
    import numpy as np

    # Get unique districts with their coordinates
    coord_cols = ['avg_latitude', 'avg_longitude']
    districts = df[[district_col] + coord_cols].drop_duplicates()
    districts_with_coords = districts.dropna(subset=coord_cols)

    print(f"   Creating {n_folds} spatial folds using KMeans clustering...")
    print(f"   Total districts: {len(districts)}")
    print(f"   Districts with valid coordinates: {len(districts_with_coords)}")

    # KMeans clustering on coordinates
    coords = districts_with_coords[coord_cols].values
    kmeans = KMeans(n_clusters=n_folds, random_state=random_state, n_init=10)
    districts_with_coords = districts_with_coords.copy()
    districts_with_coords['fold'] = kmeans.fit_predict(coords)

    # Print fold distribution
    print(f"   Fold distribution (KMeans spatial clustering):")
    for fold in range(n_folds):
        n_districts = (districts_with_coords['fold'] == fold).sum()
        print(f"      Fold {fold}: {n_districts} districts")

    # Map folds back to full dataset
    fold_map = dict(zip(districts_with_coords[district_col], districts_with_coords['fold']))
    return df[district_col].map(fold_map)

# =============================================================================
# PATHS
# =============================================================================

MONTHLY_DATA = STAGE2_DATA_DIR / 'ml_dataset_monthly.parquet'
IPC_REFERENCE = STAGE1_DATA_DIR / 'ipc_reference.parquet'
# AR predictions from Stage 1
AR_PREDICTIONS = STAGE1_RESULTS_DIR / 'predictions_h8_averaged.csv'

# =============================================================================
# METADATA COLUMNS TO PRESERVE
# =============================================================================

METADATA_COLUMNS = [
    # Geographic
    'ipc_country', 'ipc_country_code', 'ipc_district', 'ipc_region',
    'ipc_geographic_unit_full', 'ipc_fips_code',
    # Temporal
    'year_month', 'year', 'month',
    # Coordinates
    'avg_latitude', 'avg_longitude', 'latitude_std', 'longitude_std',
    # IPC
    'ipc_value', 'ipc_value_filled', 'ipc_binary_crisis', 'ipc_future_crisis',
    'ipc_period_start', 'ipc_period_end',
    # Volume
    'article_count', 'unique_sources',
    # CV
    'fold',
    # AR predictions
    'ar_pred_optimal_filled', 'ar_source_month', 'ar_target_month',
]

print("=" * 80)
print("XGBOOST PIPELINE - PHASE 2: RATIO FEATURE ENGINEERING")
print("=" * 80)


def load_valid_districts():
    """
    Load valid districts from Phase 1.

    UPDATED: Now loads ipc_geographic_unit_full (canonical identifier)
    instead of short district names to ensure consistent matching.
    """
    valid_path = PHASE1_RESULTS / OUTPUT_FILES['valid_districts']

    if not valid_path.exists():
        raise FileNotFoundError(
            f"Valid districts file not found: {valid_path}\n"
            "Please run Phase 1 scripts first."
        )

    valid_df = pd.read_csv(valid_path)

    # Use canonical identifier
    if 'ipc_geographic_unit_full' not in valid_df.columns:
        raise ValueError(
            "ipc_geographic_unit_full not found in Phase 1 output! "
            "Please re-run Phase 1 with updated script."
        )

    valid_districts = valid_df['ipc_geographic_unit_full'].tolist()
    print(f"   Loaded {len(valid_districts):,} valid districts from Phase 1")
    print(f"   Using canonical identifier: ipc_geographic_unit_full")

    return valid_districts


def load_and_filter_data(valid_districts):
    """
    Load monthly data and filter to valid districts.

    UPDATED: Now FORCES use of ipc_geographic_unit_full to match Phase 1.
    """
    print("\n" + "-" * 40)
    print("Loading and filtering data...")

    # Load monthly data
    df = pd.read_parquet(MONTHLY_DATA)
    print(f"   Loaded monthly data: {len(df):,} rows")

    # ALWAYS use canonical identifier
    district_col = 'ipc_geographic_unit_full'
    if district_col not in df.columns:
        raise ValueError(
            f"ipc_geographic_unit_full not found in monthly data! "
            f"Available columns: {df.columns.tolist()}"
        )

    print(f"   Using canonical identifier: {district_col}")

    # Filter to valid districts
    df_filtered = df[df[district_col].isin(valid_districts)].copy()
    print(f"   After filtering to valid districts: {len(df_filtered):,} rows")
    print(f"   Unique districts in filtered data: {df_filtered[district_col].nunique():,}")

    # Check country coverage
    if 'ipc_country' in df_filtered.columns:
        countries = df_filtered['ipc_country'].nunique()
        print(f"   Countries in filtered data: {countries}")

    return df_filtered, district_col


def compute_ratio_features(df):
    """
    Compute ratio-based features.

    Ratios = category_count / total_articles
    """
    print("\n" + "-" * 40)
    print("Computing ratio features...")

    # Identify category columns
    category_cols = [col for col in MACRO_CATEGORIES if col in df.columns]
    print(f"   Found {len(category_cols)} category columns")

    # Compute total articles (sum of all categories)
    df['total_category_articles'] = df[category_cols].sum(axis=1)

    # Compute ratios
    for col in category_cols:
        ratio_col = col.replace('_category', '_ratio')
        df[ratio_col] = df[col] / df['total_category_articles'].replace(0, np.nan)
        df[ratio_col] = df[ratio_col].fillna(0)

    print(f"   Created {len(category_cols)} ratio features")

    return df


def compute_concentration_metrics(df):
    """
    Compute concentration metrics.

    - HHI (Herfindahl-Hirschman Index)
    - Entropy (diversity measure)
    - Dominant category
    """
    print("\n" + "-" * 40)
    print("Computing concentration metrics...")

    ratio_cols = [col for col in df.columns if col.endswith('_ratio')]

    # HHI: sum of squared ratios
    df['hhi_category_concentration'] = df[ratio_cols].apply(
        lambda x: (x ** 2).sum(), axis=1
    )

    # Normalized HHI (0 to 1 scale)
    n_categories = len(ratio_cols)
    min_hhi = 1 / n_categories  # Perfect equality
    df['hhi_normalized'] = (df['hhi_category_concentration'] - min_hhi) / (1 - min_hhi)

    # Entropy: -sum(p * log(p))
    def compute_entropy(row):
        p = row[row > 0]  # Only positive values
        if len(p) == 0:
            return 0
        return -np.sum(p * np.log(p + 1e-10))

    df['category_entropy'] = df[ratio_cols].apply(compute_entropy, axis=1)

    # Dominant category
    df['dominant_category'] = df[ratio_cols].idxmax(axis=1)
    df['dominant_category_ratio'] = df[ratio_cols].max(axis=1)

    print("   Created: hhi_category_concentration, hhi_normalized, category_entropy")
    print("   Created: dominant_category, dominant_category_ratio")

    return df


def compute_temporal_trends(df, district_col):
    """
    Compute temporal trends using rolling windows.

    - 3-month trend (slope of linear regression)
    - 6-month trend
    """
    print("\n" + "-" * 40)
    print("Computing temporal trends...")

    ratio_cols = [col for col in df.columns if col.endswith('_ratio')]

    # Sort by district and time
    df = df.sort_values([district_col, 'year_month'])

    # 3-month trends
    def compute_slope(x):
        if len(x) < 3:
            return np.nan
        t = np.arange(len(x))
        try:
            slope = np.polyfit(t, x.values, 1)[0]
            return slope
        except:
            return np.nan

    print("   Computing 3-month trends...")
    for col in ratio_cols:
        trend_col = col.replace('_ratio', '_trend_3m')
        df[trend_col] = df.groupby(district_col)[col].transform(
            lambda x: x.rolling(3, min_periods=2).apply(compute_slope, raw=False)
        )

    print("   Computing 6-month trends...")
    for col in ratio_cols:
        trend_col = col.replace('_ratio', '_trend_6m')
        df[trend_col] = df.groupby(district_col)[col].transform(
            lambda x: x.rolling(6, min_periods=3).apply(compute_slope, raw=False)
        )

    print(f"   Created {len(ratio_cols) * 2} trend features")

    return df


def compute_momentum_features(df, district_col):
    """
    Compute momentum (delta) features.

    Delta = current - previous
    """
    print("\n" + "-" * 40)
    print("Computing momentum features...")

    ratio_cols = [col for col in df.columns if col.endswith('_ratio')]

    # Sort by district and time
    df = df.sort_values([district_col, 'year_month'])

    for col in ratio_cols:
        delta_col = col.replace('_ratio', '_delta')
        df[delta_col] = df.groupby(district_col)[col].diff()

    # Concentration delta
    df['hhi_delta'] = df.groupby(district_col)['hhi_category_concentration'].diff()
    df['entropy_delta'] = df.groupby(district_col)['category_entropy'].diff()

    print(f"   Created {len(ratio_cols) + 2} momentum features")

    return df


def load_and_merge_ar_predictions(df, district_col):
    """
    Load AR predictions and merge with features using FORWARD-FILL expansion.

    Uses FULL predictions file (all AR predictions, not just failures) to maximize coverage.

    AR predictions were made for IPC assessment periods (not individual months).
    We expand each AR prediction to ALL months within its assessment period,
    matching the same logic used for IPC classifications.
    """
    print("\n" + "-" * 40)
    print("Loading AR predictions...")

    if not AR_PREDICTIONS.exists():
        print(f"   Warning: AR predictions file not found: {AR_PREDICTIONS}")
        print("   AR prediction columns will not be available")
        return df

    # Load full AR predictions (CSV format)
    ar_df = pd.read_csv(AR_PREDICTIONS)
    print(f"   Loaded AR predictions: {len(ar_df):,} IPC assessment periods")

    # Strip whitespace from district identifier for alignment
    # (Source data has leading tabs that need to be removed)
    ar_df['ipc_geographic_unit_full'] = ar_df['ipc_geographic_unit_full'].str.strip()
    df[district_col] = df[district_col].str.strip()

    # Convert dates
    ar_df['ipc_period_start'] = pd.to_datetime(ar_df['ipc_period_start'])
    ar_df['ipc_period_end'] = pd.to_datetime(ar_df['ipc_period_end'])

    # FORWARD-FILL EXPANSION: Each IPC period is ~1 month, expand to exact month
    # This matches the IPC forward-fill logic for temporal consistency
    print("   Expanding AR predictions to monthly observations...")

    expanded_rows = []
    for _, row in ar_df.iterrows():
        # IPC periods are ~1 month (28-31 days)
        # Use period start month as the representative month
        month = row['ipc_period_start'].replace(day=1)
        expanded_rows.append({
            district_col: row['ipc_geographic_unit_full'],
            'year_month': month.strftime('%Y-%m'),
            'ar_pred_optimal': row['y_pred_optimal'],
            'ar_prob': row['pred_prob'],  # Column name from Stage 1 output
            'ar_pred_binary': row['y_pred'],
            'ar_period_start': row['ipc_period_start'],
            'ar_period_end': row['ipc_period_end']
        })

    ar_expanded = pd.DataFrame(expanded_rows)
    print(f"   Expanded to {len(ar_expanded):,} district-month observations")

    # Aggregate if multiple predictions per district-month (take max)
    ar_lookup = ar_expanded.groupby([district_col, 'year_month']).agg({
        'ar_pred_optimal': 'max',
        'ar_prob': 'max',
        'ar_pred_binary': 'max',
        'ar_period_start': 'min',
        'ar_period_end': 'max'
    }).reset_index()

    print(f"   Created AR lookup: {len(ar_lookup):,} unique district-month observations")

    # Rename columns to _filled suffix to indicate forward-fill methodology
    ar_lookup = ar_lookup.rename(columns={
        'ar_pred_optimal': 'ar_pred_optimal_filled',
        'ar_prob': 'ar_prob_filled',
        'ar_pred_binary': 'ar_pred_binary_filled'
    })

    # Merge with feature data
    merge_cols = [district_col, 'year_month']
    df = df.merge(ar_lookup, on=merge_cols, how='left', suffixes=('', '_ar'))

    # Report coverage
    ar_coverage = df['ar_pred_optimal_filled'].notna().sum()
    ar_coverage_pct = ar_coverage / len(df) * 100
    print(f"   AR predictions available: {ar_coverage:,} / {len(df):,} observations ({ar_coverage_pct:.1f}%)")

    return df


def create_target_variable(df, district_col):
    """
    Create target variable: ipc_future_crisis at t+h (h=8 months ahead).

    REPRODUCIBLE SEQUENTIAL PROCESS:
    1. Load IPC reference data
    2. Create district-month level IPC values
    3. Create future crisis variable by shifting IPC values forward
    4. Merge with feature data
    """
    print("\n" + "-" * 40)
    print("Creating target variable...")

    HORIZON = 8  # Prediction horizon in months

    # If ipc_future_crisis already exists, use it
    if 'ipc_future_crisis' in df.columns and df['ipc_future_crisis'].notna().sum() > 0:
        print("   Using existing ipc_future_crisis column")
    else:
        # Step 1: Load IPC reference data
        print(f"   Loading IPC reference data from: {IPC_REFERENCE}")

        if not IPC_REFERENCE.exists():
            print(f"   ERROR: IPC reference file not found: {IPC_REFERENCE}")
            print("   Target variable cannot be created")
            return df

        ipc_df = pd.read_parquet(IPC_REFERENCE)
        print(f"   Loaded {len(ipc_df):,} IPC observations")

        # Step 2: Create district identifier matching the feature data
        ipc_district_col = 'geographic_unit_full_name' if 'geographic_unit_full_name' in ipc_df.columns else 'district'

        # Convert dates
        ipc_df['projection_start'] = pd.to_datetime(ipc_df['projection_start'])
        ipc_df['projection_end'] = pd.to_datetime(ipc_df['projection_end'])

        # Create binary crisis indicator (IPC >= 3)
        ipc_df['ipc_binary_crisis'] = (ipc_df['ipc_value'] >= 3).astype(int)

        # Step 3: Expand IPC observations to ALL months covered by projection period
        # This is critical for monthly data - each IPC projection covers multiple months
        print("   Expanding IPC observations to all covered months...")

        expanded_rows = []
        for _, row in ipc_df.iterrows():
            # Generate all months from projection_start to projection_end
            months = pd.date_range(
                start=row['projection_start'].replace(day=1),
                end=row['projection_end'].replace(day=1),
                freq='MS'  # Month Start frequency
            )
            for month in months:
                expanded_rows.append({
                    ipc_district_col: row[ipc_district_col],
                    'year_month': month.strftime('%Y-%m'),
                    'ipc_value': row['ipc_value'],
                    'ipc_binary_crisis': row['ipc_binary_crisis'],
                    'ipc_period_start': row['projection_start'],
                    'ipc_period_end': row['projection_end']
                })

        ipc_expanded = pd.DataFrame(expanded_rows)
        print(f"   Expanded to {len(ipc_expanded):,} district-month observations")

        # Step 4: Create IPC lookup at district-month level
        # Take the maximum IPC value if multiple per district-month (overlapping projections)
        ipc_lookup = ipc_expanded.groupby([ipc_district_col, 'year_month']).agg({
            'ipc_value': 'max',
            'ipc_binary_crisis': 'max',
            'ipc_period_start': 'min',
            'ipc_period_end': 'max'
        }).reset_index()

        print(f"   Created IPC lookup: {len(ipc_lookup):,} unique district-month observations")

        # Step 5: Create complete timeline with forward-filling
        # Generate full timeline for all district-months in feature data
        print("   Creating complete timeline with forward-filling...")

        # Get unique districts and months from feature data
        feature_districts = df[district_col].unique()
        feature_months = sorted(df['year_month'].unique())

        # Create full grid of district-months from features
        full_grid = pd.DataFrame([
            {district_col: d, 'year_month': m}
            for d in feature_districts
            for m in feature_months
        ])

        # Rename IPC district column if needed
        if ipc_district_col != district_col:
            ipc_lookup = ipc_lookup.rename(columns={ipc_district_col: district_col})

        # Merge IPC lookup with full grid
        full_ipc = full_grid.merge(
            ipc_lookup[[district_col, 'year_month', 'ipc_value', 'ipc_binary_crisis', 'ipc_period_start', 'ipc_period_end']],
            on=[district_col, 'year_month'],
            how='left'
        )

        # Sort by district and time for forward filling
        full_ipc = full_ipc.sort_values([district_col, 'year_month'])

        # FIX ISSUE #2: TEMPORAL TARGET LEAKAGE
        # CRITICAL: Shift BEFORE forward-fill to prevent future values bleeding into past
        # Original (WRONG): ffill -> shift (future IPC values contaminate past)
        # Fixed (CORRECT): shift -> ffill (only observed values used for filling)

        # Step 6: Create future crisis variable by shifting FIRST
        # Shift crisis indicator backward (future value becomes current target)
        # This uses ONLY actually observed IPC values (no forward-filling yet)
        full_ipc['ipc_future_crisis'] = full_ipc.groupby(district_col)['ipc_binary_crisis'].shift(-HORIZON)

        # NOW forward-fill the shifted target (fills gaps with last OBSERVED future value)
        # This is safe because we're filling the target variable, not the predictor
        full_ipc['ipc_future_crisis'] = full_ipc.groupby(district_col)['ipc_future_crisis'].ffill()

        # Fill remaining NaN targets with 0 for:
        # 1. Last HORIZON months (no future to predict)
        # 2. Districts without any IPC data
        full_ipc['ipc_future_crisis'] = full_ipc['ipc_future_crisis'].fillna(0)

        # FIX ISSUE #8: FORWARD-FILLED IPC FOR TARGETS
        # Still create filled versions for current IPC status (used as features, not targets)
        # But document clearly these are CURRENT status, not future
        full_ipc['ipc_value_filled'] = full_ipc.groupby(district_col)['ipc_value'].ffill()
        full_ipc['ipc_binary_crisis_filled'] = full_ipc.groupby(district_col)['ipc_binary_crisis'].ffill()
        full_ipc['ipc_binary_crisis_filled'] = full_ipc['ipc_binary_crisis_filled'].fillna(0)

        print(f"   Created future crisis variable (h={HORIZON} months) - SHIFT THEN FILL")
        print(f"   Target coverage: {full_ipc['ipc_future_crisis'].notna().sum():,} / {len(full_ipc):,}")
        print(f"   Forward-filled IPC values: {full_ipc['ipc_value_filled'].notna().sum():,} / {len(full_ipc):,}")

        # Prepare IPC columns for merge
        ipc_lookup = full_ipc[[district_col, 'year_month', 'ipc_value', 'ipc_value_filled',
                               'ipc_binary_crisis', 'ipc_binary_crisis_filled',
                               'ipc_future_crisis', 'ipc_period_start', 'ipc_period_end']]

        # Step 7: Merge with feature data
        merge_cols = ['year_month', 'ipc_value', 'ipc_value_filled', 'ipc_binary_crisis',
                     'ipc_binary_crisis_filled', 'ipc_future_crisis', 'ipc_period_start', 'ipc_period_end']

        # Rename district column for merge if needed
        if ipc_district_col != district_col:
            ipc_lookup = ipc_lookup.rename(columns={ipc_district_col: district_col})

        df = df.merge(
            ipc_lookup[[district_col] + merge_cols].drop_duplicates(),
            on=[district_col, 'year_month'],
            how='left',
            suffixes=('', '_ipc')
        )

        print(f"   Merged IPC data with features")
        print(f"   Target coverage: {df['ipc_future_crisis'].notna().sum():,} / {len(df):,} rows")

    # Compute class balance
    if 'ipc_future_crisis' in df.columns:
        n_crisis = df['ipc_future_crisis'].sum()
        n_total = df['ipc_future_crisis'].notna().sum()
        prevalence = n_crisis / n_total if n_total > 0 else 0
        print(f"   Class balance: {n_crisis:,} crises / {n_total:,} total ({prevalence:.1%})")

    return df


def preserve_metadata(df):
    """Ensure all metadata columns are preserved."""
    print("\n" + "-" * 40)
    print("Preserving metadata columns...")

    available = [col for col in METADATA_COLUMNS if col in df.columns]
    missing = [col for col in METADATA_COLUMNS if col not in df.columns]

    print(f"   Available metadata columns: {len(available)}")
    if missing:
        print(f"   Missing metadata columns: {missing[:5]}...")

    return df


def main():
    """Main execution function."""
    ensure_directories()

    # Step 1: Load valid districts from Phase 1
    print("\n" + "-" * 40)
    print("Step 1: Loading valid districts...")
    valid_districts = load_valid_districts()

    # Step 2: Load and filter data
    print("\n" + "-" * 40)
    print("Step 2: Loading and filtering data...")
    df, district_col = load_and_filter_data(valid_districts)

    # Step 3: Compute ratio features
    print("\n" + "-" * 40)
    print("Step 3: Computing ratio features...")
    df = compute_ratio_features(df)

    # Step 4: Compute concentration metrics
    print("\n" + "-" * 40)
    print("Step 4: Computing concentration metrics...")
    df = compute_concentration_metrics(df)

    # Step 5: Compute temporal trends
    print("\n" + "-" * 40)
    print("Step 5: Computing temporal trends...")
    df = compute_temporal_trends(df, district_col)

    # Step 6: Compute momentum features
    print("\n" + "-" * 40)
    print("Step 6: Computing momentum features...")
    df = compute_momentum_features(df, district_col)

    # Step 7: Load AR predictions and spatial CV folds
    print("\n" + "-" * 40)
    print("Step 7: Loading AR predictions and CV folds...")
    df = load_and_merge_ar_predictions(df, district_col)

    # Step 8: Create spatial folds for ALL observations
    # Step 8: Create target variable (must be done BEFORE spatial folds)
    print("\n" + "-" * 40)
    print("Step 8: Creating target variable...")
    df = create_target_variable(df, district_col)

    # Step 9: Creating spatial CV folds for ALL observations
    # NOTE: Must be done AFTER IPC data is merged (which happens in create_target_variable)
    # Spatial folds are created fresh to ensure ALL observations have assignments,
    # not just AR failure cases. This is critical for proper train/test splits.
    print("\n" + "-" * 40)
    print("Step 9: Creating spatial CV folds for ALL observations...")
    if 'fold' in df.columns:
        df = df.drop(columns=['fold'])  # Remove any partial fold assignments
    df['fold'] = create_spatial_folds(df, 'ipc_geographic_unit_full', n_folds=N_FOLDS, random_state=RANDOM_STATE)

    # Step 10: Preserve metadata
    df = preserve_metadata(df)

    # Step 11: Deduplicate observations (Stage 1 methodology)
    print("\n" + "-" * 40)
    print("Step 11: Deduplicating observations...")
    print(f"   Before deduplication: {len(df):,} rows")

    # Following Stage 1 methodology: unique observation = (ipc_geographic_unit_full, year_month)
    df['observation_key'] = (
        df['ipc_geographic_unit_full'].astype(str) + '_' +
        df['year_month'].astype(str)
    )

    # Check for duplicates
    dupes = df.duplicated(subset=['observation_key']).sum()
    if dupes > 0:
        print(f"   Found {dupes:,} duplicate observations - aggregating...")

        # Identify columns to aggregate
        # Count columns: sum
        count_cols = ['article_count', 'location_mention_count', 'unique_sources']
        count_cols = [col for col in count_cols if col in df.columns]

        # Geographic coordinates: weighted mean by location_mention_count
        geo_cols = ['avg_latitude', 'avg_longitude', 'latitude_std', 'longitude_std']
        geo_cols = [col for col in geo_cols if col in df.columns]

        # All ratio features: mean
        ratio_cols = [col for col in df.columns if col.endswith('_ratio')]
        concentration_cols = [col for col in df.columns if col.endswith(('_hhi', '_entropy'))]
        trend_cols = [col for col in df.columns if 'trend' in col or 'slope' in col]
        delta_cols = [col for col in df.columns if 'delta' in col]

        # Metadata: first (should be identical)
        metadata_cols = ['ipc_geographic_unit_full', 'ipc_district', 'ipc_country',
                        'ipc_country_code', 'year_month', 'fold', 'ipc_future_crisis',
                        'ipc_value', 'ipc_value_filled', 'ipc_binary_crisis', 'ipc_binary_crisis_filled',
                        'ipc_period_start', 'ipc_period_end',
                        'ar_pred_optimal_filled', 'ar_prob_filled', 'ar_pred_binary_filled',
                        'ar_period_start', 'ar_period_end']
        metadata_cols = [col for col in metadata_cols if col in df.columns]

        # Build aggregation dict
        agg_dict = {}
        for col in count_cols:
            agg_dict[col] = 'sum'
        for col in geo_cols:
            if col in df.columns and 'location_mention_count' in df.columns:
                # Weighted mean
                agg_dict[col] = lambda x: np.average(
                    x,
                    weights=df.loc[x.index, 'location_mention_count']
                )
            else:
                agg_dict[col] = 'mean'
        for col in ratio_cols + concentration_cols + trend_cols + delta_cols:
            if col in df.columns:
                agg_dict[col] = 'mean'
        for col in metadata_cols:
            if col in df.columns:
                agg_dict[col] = 'first'

        # Aggregate
        df = df.groupby('observation_key').agg(agg_dict).reset_index()
        df = df.drop(columns=['observation_key'])

        print(f"   After deduplication: {len(df):,} rows")
    else:
        df = df.drop(columns=['observation_key'])
        print(f"   No duplicates found")

    # ==========================================================================
    # SAVE OUTPUT
    # ==========================================================================
    print("\n" + "-" * 40)
    print("Saving output...")

    output_path = PHASE2_RESULTS / OUTPUT_FILES['ratio_features']
    df.to_parquet(output_path, index=False)
    print(f"   Saved: {output_path}")

    # Also save CSV for easier inspection
    csv_path = PHASE2_RESULTS / 'ratio_features_h8.csv'
    df.to_csv(csv_path, index=False)
    print(f"   Saved: {csv_path}")

    # ==========================================================================
    # SUMMARY
    # ==========================================================================
    print("\n" + "=" * 80)
    print("PHASE 2 STEP 1 COMPLETE: Ratio Feature Engineering")
    print("=" * 80)

    # Count feature types
    ratio_cols = [col for col in df.columns if col.endswith('_ratio')]
    trend_cols = [col for col in df.columns if '_trend_' in col]
    delta_cols = [col for col in df.columns if col.endswith('_delta')]

    print(f"\n   Total observations: {len(df):,}")
    print(f"   Unique districts: {df[district_col].nunique():,}")
    print(f"   Unique countries: {df['ipc_country'].nunique() if 'ipc_country' in df.columns else 'N/A'}")
    print(f"\n   Features created:")
    print(f"      Ratio features: {len(ratio_cols)}")
    print(f"      Trend features: {len(trend_cols)}")
    print(f"      Momentum features: {len(delta_cols)}")
    print(f"      Concentration metrics: 4")
    print(f"      Total features: {len(df.columns)}")

    return df


if __name__ == '__main__':
    df = main()
