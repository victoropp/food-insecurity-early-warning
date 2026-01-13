"""
Stage 1: Autoregressive Feature Engineering - DISTRICT LEVEL CORRECTED
Creates Ls (spatial autoregressive) and Lt (temporal autoregressive) features.

KEY CORRECTIONS:
1. Groups by ipc_geographic_unit_full (unique district identifier)
2. Spatial neighbors based on district-level coordinates
3. Temporal lag within same district (not LHZ)

Author: Victor Collins Oppon
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
import warnings
from config import BASE_DIR
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(str(BASE_DIR.parent.parent.parent))

# District pipeline I/O (district_level subfolder)
DISTRICT_DATA_DIR = BASE_DIR / 'data' / 'district_level'
INPUT_FILE = DISTRICT_DATA_DIR / 'ml_dataset_deduplicated.parquet'
OUTPUT_FILE = DISTRICT_DATA_DIR / 'stage1_features.parquet'
OUTPUT_CSV = DISTRICT_DATA_DIR / 'stage1_features.csv'
SPATIAL_WEIGHTS_FILE = DISTRICT_DATA_DIR / 'spatial_weights.parquet'

# Parameters
RADIUS_KM = 300  # Spatial radius in kilometers


def haversine_distance(lat1, lon1, lat2, lon2):
    """Calculate great circle distance in kilometers"""
    R = 6371  # Earth radius in km

    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def build_spatial_weights_matrix(df, radius_km=300):
    """
    Build inverse distance weighted spatial matrix at DISTRICT level.
    W_ij = 1/d_ij if d_ij <= radius_km, else 0
    Row-normalized so each row sums to 1

    KEY CHANGE: Use ipc_geographic_unit_full as unique district identifier
    """
    print(f"\nBuilding spatial weights matrix (radius={radius_km}km)...")
    print("   Using ipc_geographic_unit_full as district identifier")

    # Get unique districts with their average coordinates
    # Each district should have consistent coordinates across time
    district_coords = df.groupby('ipc_geographic_unit_full').agg({
        'avg_latitude': 'mean',
        'avg_longitude': 'mean',
        'ipc_country': 'first',
        'ipc_district': 'first'
    }).reset_index()

    district_coords = district_coords.dropna(subset=['avg_latitude', 'avg_longitude'])
    district_coords = district_coords.reset_index(drop=True)

    n_districts = len(district_coords)
    print(f"   Computing distances for {n_districts} unique districts...")

    # Extract coordinates
    coords = district_coords[['avg_latitude', 'avg_longitude']].values

    # Compute distance matrix in chunks
    print(f"   Computing distance matrix...")
    coords_rad = np.radians(coords)
    chunk_size = 500
    n_chunks = (n_districts + chunk_size - 1) // chunk_size

    # Initialize sparse weights
    W_sparse = {}

    for chunk_idx in range(n_chunks):
        start_idx = chunk_idx * chunk_size
        end_idx = min((chunk_idx + 1) * chunk_size, n_districts)

        print(f"   Processing chunk {chunk_idx+1}/{n_chunks}...", end='\r')

        chunk_coords = coords_rad[start_idx:end_idx]

        lat1 = chunk_coords[:, 0:1]
        lon1 = chunk_coords[:, 1:2]
        lat2 = coords_rad[:, 0]
        lon2 = coords_rad[:, 1]

        dlat = lat2 - lat1
        dlon = lon2 - lon1

        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        distances = 6371 * c

        for i, global_i in enumerate(range(start_idx, end_idx)):
            valid_mask = (distances[i, :] > 0) & (distances[i, :] <= radius_km)

            if valid_mask.any():
                neighbor_indices = np.where(valid_mask)[0]
                neighbor_distances = distances[i, valid_mask]
                neighbor_weights = 1.0 / neighbor_distances

                weight_sum = neighbor_weights.sum()
                if weight_sum > 0:
                    neighbor_weights = neighbor_weights / weight_sum

                W_sparse[global_i] = dict(zip(neighbor_indices, neighbor_weights))

    print(f"\n   Computed {len(W_sparse)} districts with neighbors")

    # Convert to DataFrame
    W = np.zeros((n_districts, n_districts))
    for i, neighbors in W_sparse.items():
        for j, weight in neighbors.items():
            W[i, j] = weight

    W_df = pd.DataFrame(
        W,
        index=district_coords['ipc_geographic_unit_full'],
        columns=district_coords['ipc_geographic_unit_full']
    )

    # Add metadata
    W_df['country'] = district_coords['ipc_country'].values
    W_df['district'] = district_coords['ipc_district'].values

    print(f"   Spatial weights matrix: {W_df.shape}")
    print(f"   Avg neighbors per district: {(W > 0).sum(axis=1).mean():.1f}")
    print(f"   Districts with no neighbors: {(W.sum(axis=1) == 0).sum()}")

    return W_df, district_coords


def create_spatial_lag(df, W_df):
    """
    Compute spatial autoregressive feature Ls at DISTRICT level.
    Ls_it = sum_j(W_ij * IPC_jt) for neighboring districts j at same time t

    KEY CHANGE: Uses ipc_geographic_unit_full for district identification
    OPTIMIZED: O(n) using pre-built lookup dictionary instead of O(n²) filtering
    """
    print("\nCreating Ls (spatial lag) at DISTRICT level...")
    print("   Building (district, period) -> IPC lookup for O(n) performance...")

    # Get metadata columns to drop from weights
    meta_cols = ['country', 'district']
    meta_cols = [c for c in meta_cols if c in W_df.columns]

    # PRE-BUILD LOOKUP: (district_full, period_start) -> ipc_value
    # This makes neighbor lookups O(1) instead of O(n)
    ipc_lookup = {}
    for _, row in df.iterrows():
        key = (row['ipc_geographic_unit_full'], row['ipc_period_start'])
        ipc_lookup[key] = row['ipc_value']

    print(f"   Built lookup with {len(ipc_lookup):,} entries")

    # Pre-extract weights for each district (avoid repeated .loc calls)
    district_weights = {}
    district_neighbors = {}
    for district_full in W_df.index:
        weights = W_df.loc[district_full, :].drop(meta_cols, errors='ignore')
        neighbors = weights[weights > 0]
        if len(neighbors) > 0:
            district_weights[district_full] = neighbors.to_dict()
            district_neighbors[district_full] = list(neighbors.index)

    print(f"   Pre-computed weights for {len(district_weights):,} districts with neighbors")

    # Compute Ls for each row
    ls_values = []
    n_valid = 0

    for idx, row in df.iterrows():
        if idx % 5000 == 0:
            print(f"   Processing row {idx:,}/{len(df):,}...", end='\r')

        district_full = row['ipc_geographic_unit_full']
        period_start = row['ipc_period_start']

        # Check if district has neighbors
        if district_full not in district_neighbors:
            ls_values.append(np.nan)
            continue

        neighbors = district_neighbors[district_full]
        weights = district_weights[district_full]

        # Compute weighted average using O(1) lookups
        weighted_sum = 0
        weight_sum = 0

        for neighbor_full in neighbors:
            neighbor_key = (neighbor_full, period_start)
            if neighbor_key in ipc_lookup:
                neighbor_ipc = ipc_lookup[neighbor_key]
                w = weights[neighbor_full]
                weighted_sum += w * neighbor_ipc
                weight_sum += w

        if weight_sum > 0:
            ls_values.append(weighted_sum / weight_sum)
            n_valid += 1
        else:
            ls_values.append(np.nan)

    print(f"\n   Ls computed: {n_valid:,} valid")

    return ls_values


def main():
    print("=" * 80)
    print("Stage 1: Feature Engineering - DISTRICT LEVEL")
    print("=" * 80)
    print(f"Start time: {datetime.now()}\n")

    # Load data
    print("1. Loading deduplicated district-level dataset...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")

    # Convert dates
    print("\n2. Converting dates to datetime...")
    df['ipc_period_start'] = pd.to_datetime(df['ipc_period_start'])
    df['ipc_period_end'] = pd.to_datetime(df['ipc_period_end'])

    # Sort by district and time (CRITICAL for temporal lag)
    print("\n3. Sorting by district and time...")
    df = df.sort_values(['ipc_geographic_unit_full', 'ipc_period_start'])
    df = df.reset_index(drop=True)

    print(f"   Date range: {df['ipc_period_start'].min()} to {df['ipc_period_end'].max()}")
    print(f"   Countries: {df['ipc_country'].nunique()}")
    print(f"   Unique districts: {df['ipc_geographic_unit_full'].nunique():,}")

    # Build spatial weights matrix
    W_df, district_coords = build_spatial_weights_matrix(df, radius_km=RADIUS_KM)

    # Save spatial weights
    print(f"\n   Saving spatial weights to {SPATIAL_WEIGHTS_FILE}...")
    W_df.to_parquet(SPATIAL_WEIGHTS_FILE)

    # Create Lt (temporal lag)
    print("\n4. Creating Lt (temporal autoregressive) features...")
    print("   Lt = previous period IPC value (t-1) for SAME DISTRICT")
    print("   Grouped by: ipc_geographic_unit_full (unique district identifier)")

    # Group by district (geographic_unit_full) and get previous IPC
    df['Lt'] = df.groupby('ipc_geographic_unit_full')['ipc_value'].shift(1)

    lt_missing = df['Lt'].isna().sum()
    print(f"   Lt created: {lt_missing:,} missing (first observation per district)")

    # Create Ls (spatial lag)
    ls_values = create_spatial_lag(df, W_df)
    df['Ls'] = ls_values

    ls_missing = df['Ls'].isna().sum()
    print(f"   Ls created: {ls_missing:,} missing")

    # Create future crisis targets
    print("\n5. Creating future crisis targets...")
    print("   Building district -> periods lookup for O(n) performance...")

    # PRE-BUILD LOOKUP: district -> list of (period_start, ipc_value) sorted by date
    # This makes future observation lookups O(log n) instead of O(n)
    district_periods = {}
    for _, row in df.iterrows():
        district_full = row['ipc_geographic_unit_full']
        if district_full not in district_periods:
            district_periods[district_full] = []
        district_periods[district_full].append((row['ipc_period_start'], row['ipc_value']))

    # Sort each district's periods by date
    for district_full in district_periods:
        district_periods[district_full].sort(key=lambda x: x[0])

    print(f"   Built lookup for {len(district_periods):,} districts")

    for h in [4, 8, 12]:
        print(f"\n   Creating y_h{h} (crisis {h} months ahead)...")

        target_col = f'y_h{h}'
        target_values = []

        for idx, row in df.iterrows():
            if idx % 10000 == 0:
                print(f"      Processing row {idx:,}/{len(df):,}...", end='\r')

            district_full = row['ipc_geographic_unit_full']
            current_start = row['ipc_period_start']

            # Calculate target date window
            target_date_min = current_start + relativedelta(months=h)
            target_date_max = current_start + relativedelta(months=h+2)

            # Find future observation using pre-built lookup (O(k) where k = periods per district)
            periods = district_periods.get(district_full, [])
            future_ipc = None

            for period_start, ipc_value in periods:
                if period_start >= target_date_min:
                    if period_start <= target_date_max:
                        future_ipc = ipc_value
                        break
                    else:
                        # Past the window, no match
                        break

            if future_ipc is not None:
                target_values.append(1 if future_ipc >= 3 else 0)
            else:
                target_values.append(np.nan)

        df[target_col] = target_values

        valid = df[target_col].notna().sum()
        crisis = (df[target_col] == 1).sum()
        pct = (crisis/valid*100) if valid > 0 else 0
        print(f"\n      Valid: {valid:,}, Crisis: {crisis:,} ({pct:.1f}%)")

    # Validate no data leakage
    print("\n6. Validating no data leakage...")

    leakage_check = df.groupby('ipc_geographic_unit_full').apply(
        lambda g: (g['ipc_period_start'].shift(1) < g['ipc_period_start']).all(),
        include_groups=False
    )

    if leakage_check.all():
        print("   [OK] Lt validation passed: All temporal lags are from previous periods")
    else:
        failed = (~leakage_check).sum()
        print(f"   [WARNING] {failed} districts failed temporal validation")

    # Summary
    print("\n" + "=" * 80)
    print("Feature Engineering Summary - DISTRICT LEVEL")
    print("=" * 80)

    print(f"\nFinal dataset shape: {df.shape}")

    new_cols = ['Lt', 'Ls', 'y_h4', 'y_h8', 'y_h12']
    print(f"\nNew columns added:")
    for col in new_cols:
        if col in df.columns:
            valid = df[col].notna().sum()
            missing = df[col].isna().sum()
            print(f"  - {col}: {valid:,} valid ({missing:,} missing)")

    print(f"\nRows usable for training (complete features + label):")
    for h in [4, 8, 12]:
        usable = df[['Lt', 'Ls', f'y_h{h}']].notna().all(axis=1).sum()
        print(f"  h={h} months: {usable:,} rows ({usable/len(df)*100:.1f}%)")

    # Save
    print(f"\n7. Saving feature-engineered dataset...")
    print(f"   Parquet: {OUTPUT_FILE}")
    df.to_parquet(OUTPUT_FILE, index=False)
    print("   [OK] Parquet saved")

    print(f"\n   CSV: {OUTPUT_CSV}")
    df.to_csv(OUTPUT_CSV, index=False)
    print("   [OK] CSV saved")

    print("\n" + "=" * 80)
    print("Feature Engineering Complete - DISTRICT LEVEL")
    print("=" * 80)
    print(f"\nEnd time: {datetime.now()}")


if __name__ == "__main__":
    main()
