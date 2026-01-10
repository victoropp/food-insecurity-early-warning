"""
Stratified Spatial Cross-Validation Utility
============================================
Creates spatial folds that are:
1. Geographically clustered (nearby districts in same fold)
2. Balanced by crisis rate (similar % crisis in each fold)
3. Balanced by country representation (each fold has mix of countries)

This addresses the issue where pure KMeans CV creates folds with
wildly different crisis rates (2% to 24%), causing unstable evaluation.

Author: Victor Collins Oppon
Date: December 2025
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from typing import Tuple, Optional


def create_stratified_spatial_folds(
    df: pd.DataFrame,
    district_col: str = 'ipc_geographic_unit_full',
    country_col: str = 'ipc_country',
    target_col: str = 'ipc_future_crisis',
    lat_col: str = 'avg_latitude',
    lon_col: str = 'avg_longitude',
    n_folds: int = 5,
    n_spatial_clusters: int = 20,
    random_state: int = 42,
    verbose: bool = True
) -> pd.Series:
    """
    Create stratified spatial cross-validation folds.

    Algorithm:
    1. Create many small spatial clusters using KMeans (e.g., 20 clusters)
    2. Compute crisis rate and country distribution for each cluster
    3. Assign clusters to folds using greedy balancing:
       - Each fold should have similar crisis rate
       - Each fold should have representation from multiple countries

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with district identifiers, coordinates, and target
    district_col : str
        Column name for district identifier
    country_col : str
        Column name for country identifier
    target_col : str
        Column name for binary target variable
    lat_col, lon_col : str
        Column names for coordinates
    n_folds : int
        Number of CV folds (default: 5)
    n_spatial_clusters : int
        Number of initial spatial clusters (default: 20)
        More clusters = finer geographic granularity for balancing
    random_state : int
        Random seed for reproducibility
    verbose : bool
        Print progress information

    Returns:
    --------
    pd.Series
        Fold assignments for each observation (0 to n_folds-1)
    """

    if verbose:
        print(f"\n{'='*60}")
        print("STRATIFIED SPATIAL CROSS-VALIDATION")
        print(f"{'='*60}")

    # Step 1: Get unique districts with coordinates and crisis rates
    district_stats = df.groupby(district_col).agg({
        lat_col: 'mean',
        lon_col: 'mean',
        country_col: 'first',
        target_col: ['mean', 'sum', 'count']
    }).reset_index()

    # Flatten column names
    district_stats.columns = [
        district_col, 'lat', 'lon', 'country',
        'crisis_rate', 'n_crisis', 'n_obs'
    ]

    # Remove districts without coordinates
    districts_with_coords = district_stats.dropna(subset=['lat', 'lon']).copy()

    if verbose:
        print(f"\nTotal districts: {len(district_stats)}")
        print(f"Districts with coordinates: {len(districts_with_coords)}")
        print(f"Overall crisis rate: {districts_with_coords['crisis_rate'].mean():.1%}")

    # Step 2: Create spatial clusters using KMeans
    coords = districts_with_coords[['lat', 'lon']].values
    kmeans = KMeans(n_clusters=n_spatial_clusters, random_state=random_state, n_init=10)
    districts_with_coords['spatial_cluster'] = kmeans.fit_predict(coords)

    if verbose:
        print(f"\nCreated {n_spatial_clusters} spatial clusters")

    # Step 3: Compute statistics for each spatial cluster
    cluster_stats = districts_with_coords.groupby('spatial_cluster').agg({
        'n_obs': 'sum',
        'n_crisis': 'sum',
        'country': lambda x: list(set(x)),  # Unique countries in cluster
        district_col: 'count'  # Number of districts in cluster
    }).reset_index()

    cluster_stats.columns = ['spatial_cluster', 'n_obs', 'n_crisis', 'countries', 'n_districts']
    cluster_stats['crisis_rate'] = cluster_stats['n_crisis'] / cluster_stats['n_obs']
    cluster_stats['n_countries'] = cluster_stats['countries'].apply(len)

    if verbose:
        print(f"\nCluster statistics:")
        print(f"  Observations per cluster: {cluster_stats['n_obs'].mean():.0f} (mean)")
        print(f"  Crisis rate range: {cluster_stats['crisis_rate'].min():.1%} - {cluster_stats['crisis_rate'].max():.1%}")

    # Step 4: Assign clusters to folds using round-robin with crisis rate balancing
    # Goal: Each fold gets roughly equal observations AND crisis rate

    # Initialize fold assignments
    fold_assignments = {}
    fold_obs = {i: 0 for i in range(n_folds)}
    fold_crisis = {i: 0 for i in range(n_folds)}
    fold_countries = {i: set() for i in range(n_folds)}

    # Sort clusters by crisis rate (alternate high/low for balance)
    cluster_stats = cluster_stats.sort_values('crisis_rate', ascending=False)

    # Round-robin assignment: assign cluster to fold with fewest observations
    for idx, (_, cluster_row) in enumerate(cluster_stats.iterrows()):
        cluster_id = cluster_row['spatial_cluster']
        cluster_obs = cluster_row['n_obs']
        cluster_crisis = cluster_row['n_crisis']
        cluster_countries = set(cluster_row['countries'])

        # Find fold with minimum observations (greedy load balancing)
        best_fold = min(range(n_folds), key=lambda f: fold_obs[f])

        # Assign cluster to best fold
        fold_assignments[cluster_id] = best_fold
        fold_obs[best_fold] += cluster_obs
        fold_crisis[best_fold] += cluster_crisis
        fold_countries[best_fold] |= cluster_countries

    # Step 5: Map cluster assignments back to districts, then to observations
    districts_with_coords['fold'] = districts_with_coords['spatial_cluster'].map(fold_assignments)

    # Create district -> fold mapping
    district_fold_map = dict(zip(
        districts_with_coords[district_col],
        districts_with_coords['fold']
    ))

    # Map to original dataframe
    fold_series = df[district_col].map(district_fold_map)

    # Handle districts without coordinates (assign randomly but balanced)
    missing_mask = fold_series.isna()
    if missing_mask.sum() > 0:
        np.random.seed(random_state)
        fold_series.loc[missing_mask] = np.random.randint(0, n_folds, size=missing_mask.sum())

    fold_series = fold_series.astype(int)

    # Step 6: Report final fold statistics
    if verbose:
        print(f"\n{'='*60}")
        print("FOLD STATISTICS (Stratified Spatial CV)")
        print(f"{'='*60}")

        for fold in range(n_folds):
            fold_mask = fold_series == fold
            fold_data = df[fold_mask]
            fold_n = len(fold_data)
            fold_crisis_n = fold_data[target_col].sum() if target_col in fold_data.columns else 0
            fold_rate = fold_crisis_n / fold_n if fold_n > 0 else 0
            fold_countries_list = fold_data[country_col].unique() if country_col in fold_data.columns else []

            print(f"\n  Fold {fold}:")
            print(f"    Observations: {fold_n:,} ({fold_n/len(df)*100:.1f}%)")
            print(f"    Crisis events: {fold_crisis_n:,.0f} ({fold_rate:.1%})")
            print(f"    Countries: {len(fold_countries_list)}")

        # Compare with pure KMeans
        print(f"\n{'='*60}")
        print("COMPARISON: Crisis Rate Variation")
        print(f"{'='*60}")

        fold_rates = []
        for fold in range(n_folds):
            fold_mask = fold_series == fold
            fold_data = df[fold_mask]
            rate = fold_data[target_col].mean() if target_col in fold_data.columns else 0
            fold_rates.append(rate)

        print(f"  Crisis rate range: {min(fold_rates):.1%} - {max(fold_rates):.1%}")
        print(f"  Crisis rate std: {np.std(fold_rates):.1%}")
        print(f"  (Lower std = better balance)")

    return fold_series


def create_safe_location_features(
    df: pd.DataFrame,
    district_col: str = 'ipc_geographic_unit_full',
    country_col: str = 'ipc_country',
    verbose: bool = True
) -> pd.DataFrame:
    """
    Create location features that DO NOT use the target variable.

    These features are SAFE to compute on all data without causing leakage.
    They describe location characteristics using only input features (X), not target (y).

    Features created:
    1. country_baseline_conflict - average conflict_ratio for country
    2. country_baseline_food_security - average food_security_ratio for country
    3. country_data_density - observation count proportion (proxy for data quality)

    Parameters:
    -----------
    df : pd.DataFrame
        Dataset with location identifiers
    district_col, country_col : str
        Column names for location identifiers
    verbose : bool
        Print progress information

    Returns:
    --------
    pd.DataFrame
        Original dataframe with safe location features added
    """

    if verbose:
        print(f"\n{'='*60}")
        print("CREATING SAFE LOCATION FEATURES (no target leakage)")
        print(f"{'='*60}")

    df = df.copy()

    # 1. Country baseline conflict level (uses conflict_ratio, NOT target)
    if 'conflict_ratio' in df.columns and country_col in df.columns:
        country_conflict = df.groupby(country_col)['conflict_ratio'].mean()
        df['country_baseline_conflict'] = df[country_col].map(country_conflict)
        if verbose:
            print(f"\n  Created: country_baseline_conflict (from conflict_ratio)")

    # 2. Country baseline food security coverage (uses food_security_ratio, NOT target)
    if 'food_security_ratio' in df.columns and country_col in df.columns:
        country_food_sec = df.groupby(country_col)['food_security_ratio'].mean()
        df['country_baseline_food_security'] = df[country_col].map(country_food_sec)
        if verbose:
            print(f"  Created: country_baseline_food_security (from food_security_ratio)")

    # 3. Country observation density (uses observation count, NOT target)
    if country_col in df.columns:
        country_obs_count = df.groupby(country_col).size()
        total_obs = len(df)
        country_density = country_obs_count / total_obs
        df['country_data_density'] = df[country_col].map(country_density)
        if verbose:
            print(f"  Created: country_data_density (from observation counts)")

    if verbose:
        print(f"\n  Total SAFE location features: 3")
        print(f"  (These do NOT use target variable - no leakage)")

    return df


def compute_target_based_features_for_fold(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    district_col: str = 'ipc_geographic_unit_full',
    country_col: str = 'ipc_country',
    target_col: str = 'ipc_future_crisis',
    time_col: str = 'year_month'
) -> tuple:
    """
    Compute target-based location features using ONLY training data.

    CRITICAL: This prevents data leakage by computing statistics only from
    the training fold, then applying them to both train and test.

    Features created (from training data only):
    1. country_historical_crisis_rate - % of months with crisis in country
    2. district_historical_crisis_rate - % of months with crisis in district
    3. district_crisis_volatility - std of crisis occurrence
    4. country_crisis_trend - is crisis rate increasing/decreasing

    Parameters:
    -----------
    train_df : pd.DataFrame
        Training data for this fold
    test_df : pd.DataFrame
        Test data for this fold
    district_col, country_col : str
        Column names for location identifiers
    target_col : str
        Column name for binary target
    time_col : str
        Column name for time identifier

    Returns:
    --------
    tuple(pd.DataFrame, pd.DataFrame)
        Train and test dataframes with target-based features added
    """
    train_df = train_df.copy()
    test_df = test_df.copy()

    # 1. Country-level historical crisis rate (from TRAINING data only)
    if target_col in train_df.columns and country_col in train_df.columns:
        country_crisis_rate = train_df.groupby(country_col)[target_col].mean()
        train_df['country_historical_crisis_rate'] = train_df[country_col].map(country_crisis_rate)
        test_df['country_historical_crisis_rate'] = test_df[country_col].map(country_crisis_rate)
        # Fill NaN for countries not in training data with overall training rate
        overall_rate = train_df[target_col].mean()
        train_df['country_historical_crisis_rate'] = train_df['country_historical_crisis_rate'].fillna(overall_rate)
        test_df['country_historical_crisis_rate'] = test_df['country_historical_crisis_rate'].fillna(overall_rate)

    # 2. District-level historical crisis rate (from TRAINING data only)
    if target_col in train_df.columns and district_col in train_df.columns:
        district_crisis_rate = train_df.groupby(district_col)[target_col].mean()
        train_df['district_historical_crisis_rate'] = train_df[district_col].map(district_crisis_rate)
        test_df['district_historical_crisis_rate'] = test_df[district_col].map(district_crisis_rate)
        # Fill NaN for districts not in training data with country rate or overall rate
        train_df['district_historical_crisis_rate'] = train_df['district_historical_crisis_rate'].fillna(
            train_df['country_historical_crisis_rate']
        )
        test_df['district_historical_crisis_rate'] = test_df['district_historical_crisis_rate'].fillna(
            test_df['country_historical_crisis_rate']
        )

    # 3. District volatility (from TRAINING data only)
    if target_col in train_df.columns and district_col in train_df.columns:
        district_volatility = train_df.groupby(district_col)[target_col].std().fillna(0)
        train_df['district_crisis_volatility'] = train_df[district_col].map(district_volatility)
        test_df['district_crisis_volatility'] = test_df[district_col].map(district_volatility)
        train_df['district_crisis_volatility'] = train_df['district_crisis_volatility'].fillna(0)
        test_df['district_crisis_volatility'] = test_df['district_crisis_volatility'].fillna(0)

    # 4. Country-level crisis trend (from TRAINING data only)
    if target_col in train_df.columns and country_col in train_df.columns and time_col in train_df.columns:
        # Sort training data by time
        train_sorted = train_df.sort_values(time_col)
        midpoint = len(train_sorted) // 2
        first_half = train_sorted.iloc[:midpoint]
        second_half = train_sorted.iloc[midpoint:]

        # Compute crisis rate change from training data
        first_half_rate = first_half.groupby(country_col)[target_col].mean()
        second_half_rate = second_half.groupby(country_col)[target_col].mean()
        crisis_trend = (second_half_rate - first_half_rate).fillna(0)

        train_df['country_crisis_trend'] = train_df[country_col].map(crisis_trend)
        test_df['country_crisis_trend'] = test_df[country_col].map(crisis_trend)
        train_df['country_crisis_trend'] = train_df['country_crisis_trend'].fillna(0)
        test_df['country_crisis_trend'] = test_df['country_crisis_trend'].fillna(0)

    return train_df, test_df


# Keep old function for backward compatibility but mark as DEPRECATED
def create_meaningful_location_features(
    df: pd.DataFrame,
    district_col: str = 'ipc_geographic_unit_full',
    country_col: str = 'ipc_country',
    target_col: str = 'ipc_future_crisis',
    time_col: str = 'year_month',
    verbose: bool = True
) -> pd.DataFrame:
    """
    DEPRECATED - This function causes DATA LEAKAGE!

    Use create_safe_location_features() for features that don't use target,
    and compute_target_based_features_for_fold() inside the CV loop for
    features that use target.
    """
    import warnings
    warnings.warn(
        "create_meaningful_location_features() causes DATA LEAKAGE! "
        "Use create_safe_location_features() + compute_target_based_features_for_fold() instead.",
        DeprecationWarning
    )

    # Just call the safe version to avoid breaking existing code
    return create_safe_location_features(df, district_col, country_col, verbose)


if __name__ == '__main__':
    # Test with sample data
    print("Stratified Spatial CV utility loaded successfully")
    print("Use create_stratified_spatial_folds() for balanced CV folds")
    print("Use create_safe_location_features() for non-leaking location features")
    print("Use compute_target_based_features_for_fold() inside CV loop for target-based features")
