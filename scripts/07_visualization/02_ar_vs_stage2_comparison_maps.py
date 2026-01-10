#!/usr/bin/env python3
"""
AR Model Comparison Choropleth Maps
====================================
Phase 5, Step 6: Create 3-panel district-level choropleth maps comparing:
- Panel A: IPC Past Classification (Lt - ground truth from previous period)
- Panel B: AR Model Confusion Matrix (TP/TN/FP/FN)
- Panel C: ML Model Confusion Matrix (for AR Failures Only)

KEY FEATURES:
- Filled polygons using IPC district boundaries (matches Stage 1)
- Spatial join of predictions to IPC geometries
- Two visualization groups: Multiclass (IPC 1-4) and Binary (Crisis/No-Crisis)
- Temporal dimension: maps for each IPC classification date
- More granular than GADM - uses official FEWSNET food security zones

Author: Simplified Pipeline
Date: December 2024
"""

import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime
from typing import Dict, Optional, List, Tuple
import warnings
warnings.filterwarnings('ignore')

try:
    import geopandas as gpd
    from shapely.geometry import Point
    HAS_GEOPANDAS = True
except ImportError:
    HAS_GEOPANDAS = False
    print("Warning: geopandas not installed. Run: pip install geopandas")

from config import (
    BASE_DIR,
    FIGURES_DIR,
    RESULTS_DIR,
    IPC_BOUNDARIES_FILE,
    AFRICA_BASEMAP_FILE
)

# Stage 1 baseline for AR predictions (corrected path)
STAGE1_RESULTS = RESULTS_DIR / 'stage1_ar_baseline'

# Stage 2 models directory
STAGE2_MODELS = RESULTS_DIR / 'stage2_models'

# Phase 4 analysis results (for observation-level comparison)
PHASE4_RESULTS = RESULTS_DIR / 'analysis'

# Observation-level AR vs Stage 2 comparison file
OBS_COMPARISON_FILE = PHASE4_RESULTS / 'ensemble_analysis' / 'observation_level_comparison.csv'

# IPC boundaries path (matches Stage 1 approach)
IPC_BOUNDARIES = IPC_BOUNDARIES_FILE  # Self-contained: FINAL_PIPELINE/DATA/shapefiles/ipc_boundaries/

# Simple auto-detection functions (not using phase5_utils)
def auto_detect_models():
    """Auto-detect models from Stage 2 results (XGBoost + Mixed Effects)."""
    models = []
    stage2_dir = RESULTS_DIR / 'stage2_models'

    # Scan xgboost/ subdirectory
    xgboost_dir = stage2_dir / 'xgboost'
    if xgboost_dir.exists():
        for model_dir in xgboost_dir.iterdir():
            if model_dir.is_dir():
                # Check for predictions file
                pred_file = model_dir / f"{model_dir.name}_predictions.csv"
                if pred_file.exists():
                    models.append(f'xgboost/{model_dir.name}')

    # Scan mixed_effects/ subdirectory
    me_dir = stage2_dir / 'mixed_effects'
    if me_dir.exists():
        for model_dir in me_dir.iterdir():
            if model_dir.is_dir():
                # Check for predictions file
                pred_file = model_dir / f"{model_dir.name}_predictions.csv"
                if pred_file.exists():
                    models.append(f'mixed_effects/{model_dir.name}')

    return sorted(models)

def load_model_predictions(model_name):
    """
    Load predictions for a model.

    Args:
        model_name: Model name (e.g., 'xgboost/basic_with_ar' or 'mixed_effects/pooled_ratio_with_ar')

    Returns:
        DataFrame with predictions or None if not found
    """
    stage2_dir = RESULTS_DIR / 'stage2_models'

    # model_name format: 'category/name' (e.g., 'xgboost/basic_with_ar')
    model_path = stage2_dir / model_name

    # Extract just the model name (last part of path)
    just_model_name = Path(model_name).name
    pred_file = model_path / f"{just_model_name}_predictions.csv"

    if pred_file.exists():
        return pd.read_csv(pred_file)
    return None

print("=" * 80)
print("PHASE 5: AR MODEL COMPARISON CHOROPLETH MAPS")
print("=" * 80)

# =============================================================================
# CONFIGURATION
# =============================================================================

# IPC Classification Dates (from actual data)
IPC_DATES = [
    '2021-06', '2021-10',
    '2022-02', '2022-06', '2022-10',
    '2023-02', '2023-06', '2023-10',
    '2024-02',
]

# NOTE: Removed COUNTRY_ISO dictionary - now using IPC boundaries instead of GADM

# Color schemes
IPC_COLORS = {
    1: '#C7E9C0',  # Phase 1 - Minimal (light green)
    2: '#FFEDA0',  # Phase 2 - Stressed (yellow)
    3: '#FEB24C',  # Phase 3 - Crisis (orange)
    4: '#E31A1C',  # Phase 4 - Emergency (red)
    5: '#800026',  # Phase 5 - Famine (dark red)
}

BINARY_COLORS = {
    0: '#4CAF50',  # No crisis (green)
    1: '#E53935',  # Crisis (red)
}

CONFUSION_COLORS = {
    'TN': '#4CAF50',  # Green - True Negative
    'TP': '#2196F3',  # Blue - True Positive
    'FP': '#FF9800',  # Orange - False Positive
    'FN': '#E53935',  # Red - False Negative (missed crisis)
    'no_data': '#F5F5F5',  # Light gray
    'not_focus': '#E0E0E0',  # Gray - not in focus group
}

# Africa basemap path
AFRICA_BASEMAP = AFRICA_BASEMAP_FILE  # Self-contained: FINAL_PIPELINE/DATA/shapefiles/natural_earth/

# Map styling
plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
})


# =============================================================================
# DATA LOADING FUNCTIONS
# =============================================================================

def load_africa_basemap() -> Optional[gpd.GeoDataFrame]:
    """Load Africa basemap for background context."""
    if not HAS_GEOPANDAS:
        return None

    if AFRICA_BASEMAP.exists():
        gdf = gpd.read_file(AFRICA_BASEMAP)
        print(f"      Loaded Africa basemap: {len(gdf)} countries")
        return gdf

    print("      Warning: Africa basemap not found")
    return None


def load_ipc_boundaries() -> Optional[gpd.GeoDataFrame]:
    """
    Load official IPC district boundaries (same as Stage 1).

    Returns:
        GeoDataFrame with IPC district polygons and reset index for joining.
    """
    if not HAS_GEOPANDAS:
        return None

    if not IPC_BOUNDARIES.exists():
        print(f"      ERROR: IPC boundaries not found: {IPC_BOUNDARIES}")
        return None

    print("\n   Loading IPC district boundaries...")
    ipc_gdf = gpd.read_file(IPC_BOUNDARIES)

    # Reset index for consistent joining (matches Stage 1 approach)
    ipc_gdf = ipc_gdf.reset_index()

    print(f"      Loaded {len(ipc_gdf):,} IPC districts from {ipc_gdf['country_code'].nunique()} countries")
    return ipc_gdf


def load_ar_predictions() -> Optional[pd.DataFrame]:
    """Load AR model predictions from Stage 1 baseline."""
    ar_path = STAGE1_RESULTS / 'predictions_h8_averaged.csv'

    if not ar_path.exists():
        print(f"      ERROR: AR predictions not found at {ar_path}")
        return None

    df = pd.read_csv(ar_path)
    print(f"      Loaded {len(df):,} AR predictions")

    # Add year_month for filtering
    df['year_month'] = pd.to_datetime(df['ipc_period_start']).dt.strftime('%Y-%m')

    return df


def load_observation_comparison() -> Optional[pd.DataFrame]:
    """Load observation-level AR vs Stage 2 comparison data."""
    if not OBS_COMPARISON_FILE.exists():
        print(f"      WARNING: Observation comparison file not found at {OBS_COMPARISON_FILE}")
        return None

    df = pd.read_csv(OBS_COMPARISON_FILE)
    print(f"      Loaded {len(df):,} observation comparisons")

    # Add year_month for filtering
    df['year_month'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m')

    return df


def load_all_metric_data():
    """
    Load and align all datasets for metric-based maps.

    Returns:
        Tuple of (ar_df, s2_df, obs_comp_df) or (None, None, None) if any fails
    """
    print("\n   Loading metric data...")

    # Load AR predictions (24K rows, all periods, using h4 instead of h8)
    ar_path = STAGE1_RESULTS / 'predictions_h4_averaged.csv'
    if not ar_path.exists():
        print(f"      ERROR: AR predictions not found at {ar_path}")
        return None, None, None

    ar_df = pd.read_csv(ar_path)
    ar_df['year_month'] = pd.to_datetime(ar_df['ipc_period_start']).dt.strftime('%Y-%m')
    print(f"      Loaded {len(ar_df):,} AR predictions (h4)")

    # Load Stage 2 predictions (6.5K rows, AR-filtered)
    s2_path = STAGE2_MODELS / 'xgboost' / 'advanced_with_ar' / 'xgboost_hmm_dmd_with_ar_predictions.csv'
    if not s2_path.exists():
        print(f"      ERROR: Stage 2 predictions not found at {s2_path}")
        return None, None, None

    s2_df = pd.read_csv(s2_path)
    print(f"      Loaded {len(s2_df):,} Stage 2 predictions")

    # Load observation-level comparison
    obs_comp = load_observation_comparison()
    if obs_comp is None:
        print("      WARNING: Continuing without observation comparison data")

    # Align coordinates (use AR as reference for consistency)
    ar_coords = ar_df[['ipc_geographic_unit_full', 'avg_latitude', 'avg_longitude']].drop_duplicates()

    # Merge coordinates into S2 (in case they differ slightly)
    s2_df = s2_df.merge(
        ar_coords,
        on='ipc_geographic_unit_full',
        how='left',
        suffixes=('_orig', '')
    )

    # Fill missing coordinates with original S2 coordinates if AR doesn't have them
    if 'avg_latitude_orig' in s2_df.columns:
        s2_df['avg_latitude'] = s2_df['avg_latitude'].fillna(s2_df['avg_latitude_orig'])
        s2_df['avg_longitude'] = s2_df['avg_longitude'].fillna(s2_df['avg_longitude_orig'])

    print("      Coordinate alignment complete")

    return ar_df, s2_df, obs_comp


# NOTE: Removed spatial_join_to_gadm() and aggregate_predictions_to_districts()
# These functions are replaced by IPC-based panel functions that integrate
# spatial join and aggregation directly (matches Stage 1 approach)


# =============================================================================
# MAP HELPER FUNCTIONS
# =============================================================================

def draw_basemap_and_labels(ax, africa_basemap: Optional[gpd.GeoDataFrame]):
    """Draw Africa basemap with country labels."""
    if africa_basemap is None:
        return

    # Draw all African countries as light gray background
    africa_basemap.plot(ax=ax, color='#F8F9FA', edgecolor='#999999', linewidth=0.3)

    # Add country labels
    name_col = None
    for col in ['NAME', 'name', 'ADMIN', 'NAME_LONG']:
        if col in africa_basemap.columns:
            name_col = col
            break

    if name_col:
        for _, row in africa_basemap.iterrows():
            centroid = row.geometry.centroid
            # Only label if within map bounds
            if -20 <= centroid.x <= 55 and -35 <= centroid.y <= 40:
                ax.annotate(
                    row[name_col],
                    xy=(centroid.x, centroid.y),
                    ha='center', va='center',
                    fontsize=5, fontweight='bold',
                    color='#666666', alpha=0.7
                )


# =============================================================================
# CONFUSION CLASS COMPUTATION AND RAW METRICS
# =============================================================================

def compute_ar_confusion_class(df: pd.DataFrame) -> pd.Series:
    """Compute confusion class for AR model predictions."""
    # Detect target column dynamically
    y_true_col = None
    for col in ['ipc_future_crisis', 'y_true', 'future_crisis', 'target']:
        if col in df.columns:
            y_true_col = col
            break

    if y_true_col is None:
        raise ValueError(f"No target column found. Available: {df.columns.tolist()}")

    conditions = [
        (df[y_true_col] == 0) & (df['y_pred_optimal'] == 0),
        (df[y_true_col] == 1) & (df['y_pred_optimal'] == 1),
        (df[y_true_col] == 0) & (df['y_pred_optimal'] == 1),
        (df[y_true_col] == 1) & (df['y_pred_optimal'] == 0),
    ]
    choices = ['TN', 'TP', 'FP', 'FN']
    return pd.Series(np.select(conditions, choices, default='unknown'), index=df.index)


def compute_ml_confusion_class(df: pd.DataFrame) -> pd.Series:
    """
    Compute confusion class for ML model predictions.

    Uses same logic as compute_ar_confusion_class() but for ML model columns:
    - Uses 'future_crisis' instead of 'y_true'
    - Uses 'y_pred_youden' instead of 'y_pred_optimal'

    Returns:
        Series with values: 'TN', 'TP', 'FP', 'FN', 'unknown'
    """
    conditions = [
        (df['future_crisis'] == 0) & (df['y_pred_youden'] == 0),
        (df['future_crisis'] == 1) & (df['y_pred_youden'] == 1),
        (df['future_crisis'] == 0) & (df['y_pred_youden'] == 1),
        (df['future_crisis'] == 1) & (df['y_pred_youden'] == 0),
    ]
    choices = ['TN', 'TP', 'FP', 'FN']
    return pd.Series(np.select(conditions, choices, default='unknown'), index=df.index)


def compute_raw_metrics_ar(ar_period: pd.DataFrame) -> Dict[str, int]:
    """
    Compute confusion matrix metrics from raw (unaggregated) AR prediction data.

    This ensures metrics displayed in titles match the actual district-level predictions,
    not the aggregated GADM polygon values.

    Args:
        ar_period: DataFrame with AR predictions for a single time period

    Returns:
        Dictionary with keys: tn, tp, fp, fn
    """
    # Detect target column dynamically
    y_true_col = None
    for col in ['ipc_future_crisis', 'y_true', 'future_crisis', 'target']:
        if col in ar_period.columns:
            y_true_col = col
            break

    if y_true_col is None:
        raise ValueError(f"No target column found. Available: {ar_period.columns.tolist()}")

    y_true = ar_period[y_true_col].values
    y_pred = ar_period['y_pred_optimal'].values

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    return {'tn': tn, 'tp': tp, 'fp': fp, 'fn': fn}


def compute_raw_metrics_news(ml_period: pd.DataFrame) -> Dict[str, int]:
    """
    Compute confusion matrix metrics from raw (unaggregated) news model prediction data.

    This ensures metrics displayed in titles match the actual district-level predictions,
    not the aggregated GADM polygon values.

    Args:
        ml_period: DataFrame with news model predictions for a single time period

    Returns:
        Dictionary with keys: tn, tp, fp, fn
    """
    # Detect target column dynamically
    y_true_col = None
    for col in ['future_crisis', 'ipc_future_crisis', 'y_true', 'target']:
        if col in ml_period.columns:
            y_true_col = col
            break

    if y_true_col is None:
        raise ValueError(f"No target column found. Available: {ml_period.columns.tolist()}")

    # Detect prediction column dynamically
    y_pred_col = None
    for col in ['y_pred_youden', 'y_pred', 'pred', 'prediction']:
        if col in ml_period.columns:
            y_pred_col = col
            break

    if y_pred_col is None:
        raise ValueError(f"No prediction column found. Available: {ml_period.columns.tolist()}")

    y_true = ml_period[y_true_col].values
    y_pred = ml_period[y_pred_col].values

    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    return {'tn': tn, 'tp': tp, 'fp': fp, 'fn': fn}


# =============================================================================
# MAP CREATION FUNCTIONS
# =============================================================================

def create_three_panel_map(ipc_gdf: gpd.GeoDataFrame,
                            ar_df: pd.DataFrame,
                            ml_df: Optional[pd.DataFrame],
                            model_name: str,
                            year_month: str,
                            output_path: Path,
                            africa_basemap: Optional[gpd.GeoDataFrame] = None,
                            visualization_group: str = 'multiclass') -> bool:
    """
    Create 3-panel choropleth map using IPC boundaries (matches Stage 1).

    Panel A: IPC Past Classification (Lt)
    Panel B: AR Model Confusion Matrix
    Panel C: ML Model Confusion Matrix (AR Failures Only)
    """
    if not HAS_GEOPANDAS or ipc_gdf is None:
        return False

    # Filter data for this time period
    ar_period = ar_df[ar_df['year_month'] == year_month].copy()
    if ar_period.empty:
        print(f"      No AR data for {year_month}")
        return False

    if ml_df is not None:
        ml_period = ml_df[ml_df['year_month'] == year_month].copy()
    else:
        ml_period = None

    # Create figure with 3 panels
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # -------------------------------------------------------------------------
    # PANEL A: IPC Past Classification (Lt)
    # -------------------------------------------------------------------------
    ax = axes[0]

    # Draw Africa basemap first (all countries as light gray background)
    draw_basemap_and_labels(ax, africa_basemap)

    # Create points and spatial join (point WITHIN IPC polygon - Stage 1 approach)
    geometry = [Point(xy) for xy in zip(ar_period['avg_longitude'], ar_period['avg_latitude'])]
    ar_points = gpd.GeoDataFrame(ar_period, geometry=geometry, crs="EPSG:4326")
    joined = gpd.sjoin(ar_points, ipc_gdf, how='left', predicate='within')

    # Aggregate Lt values per IPC district
    # Note: sjoin with 'within' creates 'index' column from right GeoDataFrame (IPC index)
    index_col = 'index' if 'index' in joined.columns else 'index_right'
    valid = joined.dropna(subset=[index_col])

    if visualization_group == 'multiclass':
        # Use Lt directly (IPC 1-4) with MODE aggregation
        lt_agg = valid.groupby(index_col)['Lt'].apply(
            lambda x: int(x.mode().iloc[0]) if len(x.dropna()) > 0 and len(x.mode()) > 0 else np.nan
        )
        ipc_plot = ipc_gdf.copy()
        ipc_plot['Lt_value'] = ipc_plot['index'].map(lt_agg)

        # Plot with IPC colors
        for ipc_phase in [1, 2, 3, 4]:
            subset = ipc_plot[ipc_plot['Lt_value'] == ipc_phase]
            if not subset.empty:
                subset.plot(ax=ax, color=IPC_COLORS.get(ipc_phase, '#F5F5F5'),
                           edgecolor='white', linewidth=0.3)

        # No data districts
        no_data = ipc_plot[ipc_plot['Lt_value'].isna()]
        if not no_data.empty:
            no_data.plot(ax=ax, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.2)

        # Legend
        legend_elements = [
            mpatches.Patch(facecolor=IPC_COLORS[1], label='Phase 1 - Minimal'),
            mpatches.Patch(facecolor=IPC_COLORS[2], label='Phase 2 - Stressed'),
            mpatches.Patch(facecolor=IPC_COLORS[3], label='Phase 3 - Crisis'),
            mpatches.Patch(facecolor=IPC_COLORS[4], label='Phase 4 - Emergency'),
            mpatches.Patch(facecolor='#F5F5F5', label='No Data'),
        ]
        panel_a_title = 'IPC Past Classification\n(Multiclass: Phase 1-4)'

    else:  # binary
        # Convert Lt to binary (>=3 = crisis) using MAX aggregation (conservative)
        def compute_binary_lt_max(x):
            """
            Convert Lt values to binary using MAX aggregation.
            Conservative approach: if ANY point in IPC district has crisis (Lt>=3), district = crisis.
            Matches Stage 1 methodology.
            """
            valid = x.dropna()
            if len(valid) == 0:
                return np.nan
            # MAX aggregation: if any Lt >= 3, return 1 (crisis)
            return 1 if valid.max() >= 3 else 0

        lt_binary_agg = valid.groupby(index_col)['Lt'].apply(compute_binary_lt_max)
        ipc_plot = ipc_gdf.copy()
        ipc_plot['Lt_binary'] = ipc_plot['index'].map(lt_binary_agg)

        # Plot binary
        for crisis_val, color in [(0, BINARY_COLORS[0]), (1, BINARY_COLORS[1])]:
            subset = ipc_plot[ipc_plot['Lt_binary'] == crisis_val]
            if not subset.empty:
                subset.plot(ax=ax, color=color, edgecolor='white', linewidth=0.3)

        # No data
        no_data = ipc_plot[ipc_plot['Lt_binary'].isna()]
        if not no_data.empty:
            no_data.plot(ax=ax, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.2)

        legend_elements = [
            mpatches.Patch(facecolor=BINARY_COLORS[0], label='No Crisis (Phase 1-2)'),
            mpatches.Patch(facecolor=BINARY_COLORS[1], label='Crisis (Phase 3-4)'),
            mpatches.Patch(facecolor='#F5F5F5', label='No Data'),
        ]
        panel_a_title = 'Past IPC Status\n(h=8 months before prediction)'

    ax.legend(handles=legend_elements, loc='lower left', fontsize=8)
    ax.set_title(panel_a_title, fontweight='bold', fontsize=11)
    ax.set_xlim(-20, 55)
    ax.set_ylim(-35, 40)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # -------------------------------------------------------------------------
    # PANEL B: AR Model Confusion Matrix
    # -------------------------------------------------------------------------
    ax = axes[1]

    # Draw Africa basemap first
    draw_basemap_and_labels(ax, africa_basemap)

    # Compute raw metrics from unaggregated data for display in title
    ar_metrics = compute_raw_metrics_ar(ar_period)

    # Spatial join AR points to IPC polygons (point WITHIN polygon - Stage 1 approach)
    geometry = [Point(xy) for xy in zip(ar_period['avg_longitude'], ar_period['avg_latitude'])]
    ar_points = gpd.GeoDataFrame(ar_period, geometry=geometry, crs="EPSG:4326")
    joined = gpd.sjoin(ar_points, ipc_gdf, how='left', predicate='within')

    # Aggregate using MAX (conservative - matches Stage 1 methodology)
    index_col = 'index' if 'index' in joined.columns else 'index_right'
    valid = joined.dropna(subset=[index_col])

    agg = valid.groupby(index_col).agg({
        'ipc_future_crisis': 'max',           # Future ground truth (if ANY point has crisis, polygon = crisis)
        'y_pred_optimal': 'max',   # AR prediction (conservative)
        'country_code': 'first'
    }).reset_index()

    # Rename index column for merging
    agg = agg.rename(columns={index_col: 'ipc_index'})

    # Compute confusion class AFTER aggregation (critical for visual consistency)
    # Use 'ipc_future_crisis' which is the actual column name from aggregation
    agg['confusion_youden'] = 'no_data'
    agg.loc[(agg['ipc_future_crisis'] == 0) & (agg['y_pred_optimal'] == 0), 'confusion_youden'] = 'TN'
    agg.loc[(agg['ipc_future_crisis'] == 1) & (agg['y_pred_optimal'] == 1), 'confusion_youden'] = 'TP'
    agg.loc[(agg['ipc_future_crisis'] == 0) & (agg['y_pred_optimal'] == 1), 'confusion_youden'] = 'FP'
    agg.loc[(agg['ipc_future_crisis'] == 1) & (agg['y_pred_optimal'] == 0), 'confusion_youden'] = 'FN'

    # Merge back to IPC geometries
    ipc_plot = ipc_gdf.copy()
    ipc_plot = ipc_plot.merge(agg, left_on='index', right_on='ipc_index', how='left')
    ipc_plot['confusion'] = ipc_plot['confusion_youden'].fillna('no_data')

    # Plot by confusion class
    for conf_class in ['TN', 'TP', 'FP', 'FN', 'no_data']:
        subset = ipc_plot[ipc_plot['confusion'] == conf_class]
        if not subset.empty:
            color = CONFUSION_COLORS.get(conf_class, '#F5F5F5')
            subset.plot(ax=ax, color=color, edgecolor='white', linewidth=0.3)

    legend_elements = [
        mpatches.Patch(facecolor=CONFUSION_COLORS['TN'], label='TN (Correct No Crisis)'),
        mpatches.Patch(facecolor=CONFUSION_COLORS['TP'], label='TP (Correct Crisis)'),
        mpatches.Patch(facecolor=CONFUSION_COLORS['FP'], label='FP (False Alarm)'),
        mpatches.Patch(facecolor=CONFUSION_COLORS['FN'], label='FN (Missed Crisis)'),
        mpatches.Patch(facecolor=CONFUSION_COLORS['no_data'], label='No Data'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=8)

    # Title with raw metrics from unaggregated data
    ar_title = f'AR Model Prediction (h=8 months ahead)\n'
    ar_title += f'TP: {ar_metrics["tp"]}, FN: {ar_metrics["fn"]}, '
    ar_title += f'FP: {ar_metrics["fp"]}, TN: {ar_metrics["tn"]}'
    ax.set_title(ar_title, fontweight='bold', fontsize=11)
    ax.set_xlim(-20, 55)
    ax.set_ylim(-35, 40)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Store Panel B result for Panel C to use
    ipc_plot_panel_b = ipc_plot.copy()

    # -------------------------------------------------------------------------
    # PANEL C: ML Model Confusion Matrix (AR Failures Only)
    # -------------------------------------------------------------------------
    ax = axes[2]

    # Draw Africa basemap first
    draw_basemap_and_labels(ax, africa_basemap)

    if ml_period is not None and not ml_period.empty:
        # STEP 1: Compute raw metrics from unaggregated data for display in title
        ml_metrics = compute_raw_metrics_news(ml_period)

        # STEP 2: Spatial join ML points to IPC polygons (point WITHIN polygon - Stage 1 approach)
        # NOTE: We show ALL News model predictions (not filtering to AR failures)
        # Rationale: WITH_AR_FILTER dataset includes:
        #   - 394 observations where AR predicted 0 (all are False Negatives - AR failed)
        #   - 52,336 observations where AR made no prediction (different temporal/spatial coverage)
        # Showing all provides complete geographic view of News model performance, including both
        # areas where AR failed AND areas AR doesn't cover at all.
        geometry = [Point(xy) for xy in zip(ml_period['avg_longitude'], ml_period['avg_latitude'])]
        ml_points = gpd.GeoDataFrame(ml_period, geometry=geometry, crs="EPSG:4326")
        joined = gpd.sjoin(ml_points, ipc_gdf, how='left', predicate='within')

        # STEP 3: Aggregate precomputed confusion_class using MODE
        # News model predictions have precomputed confusion classes from Phase 3
        index_col = 'index' if 'index' in joined.columns else 'index_right'
        valid = joined.dropna(subset=[index_col])

        if 'confusion_youden' in valid.columns:
            ml_conf_agg = valid.groupby(index_col)['confusion_youden'].apply(
                lambda x: x.mode().iloc[0] if len(x.dropna()) > 0 and len(x.mode()) > 0 else 'no_data'
            )
        else:
            # Fallback: compute dynamically if precomputed column doesn't exist
            print(f"      Warning: confusion_class not found, computing dynamically")
            agg_ml = valid.groupby(index_col).agg({
                'future_crisis': 'max',
                'y_pred_youden': 'max'
            }).reset_index()

            agg_ml['confusion_youden'] = 'no_data'
            agg_ml.loc[(agg_ml['future_crisis'] == 0) & (agg_ml['y_pred_youden'] == 0), 'confusion_youden'] = 'TN'
            agg_ml.loc[(agg_ml['future_crisis'] == 1) & (agg_ml['y_pred_youden'] == 1), 'confusion_youden'] = 'TP'
            agg_ml.loc[(agg_ml['future_crisis'] == 0) & (agg_ml['y_pred_youden'] == 1), 'confusion_youden'] = 'FP'
            agg_ml.loc[(agg_ml['future_crisis'] == 1) & (agg_ml['y_pred_youden'] == 0), 'confusion_youden'] = 'FN'

            ml_conf_agg = agg_ml.set_index(index_col)['confusion_youden']

        # STEP 4: Map to IPC geometries (showing ALL predictions)
        ipc_plot_ml = ipc_gdf.copy()
        ipc_plot_ml['ml_confusion'] = ipc_plot_ml['index'].map(ml_conf_agg).fillna('no_data')

        # STEP 5: Plot (order matters - plot no_data first, then actual classes on top)
        for conf_class in ['no_data', 'TN', 'FP', 'FN', 'TP']:
            subset = ipc_plot_ml[ipc_plot_ml['ml_confusion'] == conf_class]
            if not subset.empty:
                color = CONFUSION_COLORS.get(conf_class, '#E0E0E0')
                subset.plot(ax=ax, color=color, edgecolor='white', linewidth=0.3)

        legend_elements = [
            mpatches.Patch(facecolor=CONFUSION_COLORS['TP'], label='TP (Correct Crisis Pred)'),
            mpatches.Patch(facecolor=CONFUSION_COLORS['FN'], label='FN (Missed Crisis)'),
            mpatches.Patch(facecolor=CONFUSION_COLORS['FP'], label='FP (False Alarm)'),
            mpatches.Patch(facecolor=CONFUSION_COLORS['TN'], label='TN (Correct No Crisis)'),
            mpatches.Patch(facecolor=CONFUSION_COLORS['no_data'], label='No ML Data'),
        ]

        # Count districts for reporting
        ml_districts = len(ipc_plot_ml[ipc_plot_ml['ml_confusion'] != 'no_data'])

        # Title with model name and raw metrics from unaggregated data
        ml_title = f'News Model: {model_name} (h=8)\n'
        ml_title += f'TP: {ml_metrics["tp"]}, FN: {ml_metrics["fn"]}, '
        ml_title += f'FP: {ml_metrics["fp"]}, TN: {ml_metrics["tn"]}\n'
        ml_title += f'Showing {ml_districts} IPC districts with News data'
    else:
        # No ML data - just show empty
        ipc_gdf.plot(ax=ax, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.2)
        legend_elements = [mpatches.Patch(facecolor='#F5F5F5', label='No ML Data')]
        ml_title = 'News Model Prediction\n(No ML Data)'

    ax.legend(handles=legend_elements, loc='lower left', fontsize=8)
    ax.set_title(ml_title, fontweight='bold', fontsize=11)
    ax.set_xlim(-20, 55)
    ax.set_ylim(-35, 40)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Main title
    group_label = 'Multiclass' if visualization_group == 'multiclass' else 'Binary'
    fig.suptitle(f'AR vs ML Model Comparison - {year_month} ({group_label})',
                 fontweight='bold', fontsize=14, y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return True


# =============================================================================
# METRIC COMPUTATION FUNCTIONS FOR AR STRENGTHS VS STAGE 2 IMPROVEMENTS
# =============================================================================

def compute_ar_district_metrics(ar_period_df):
    """
    Compute 3 AR strength metrics per district:
    1. Accuracy rate (% correct predictions)
    2. True positive rate (for crisis districts)
    3. Will be joined with obs_comp for dominance

    Args:
        ar_period_df: AR predictions for a single time period

    Returns:
        DataFrame with columns: ipc_geographic_unit_full, ar_accuracy, tp_rate, avg_latitude, avg_longitude
    """
    # Group by district
    district_groups = ar_period_df.groupby('ipc_geographic_unit_full')

    # Aggregate metrics
    district_agg = district_groups.agg({
        'correct': ['sum', 'count'],  # For accuracy
        'ipc_future_crisis': 'sum',   # Total crises in this district
        'avg_latitude': 'mean',
        'avg_longitude': 'mean'
    }).reset_index()

    # Flatten multi-level column names
    district_agg.columns = ['ipc_geographic_unit_full', 'correct_sum', 'correct_count',
                           'total_crises', 'avg_latitude', 'avg_longitude']

    # Compute accuracy rate
    district_agg['ar_accuracy'] = 100 * district_agg['correct_sum'] / district_agg['correct_count']

    # Compute TP count per district
    tp_by_district = ar_period_df[
        (ar_period_df['y_pred_optimal'] == 1) & (ar_period_df['ipc_future_crisis'] == 1)
    ].groupby('ipc_geographic_unit_full').size().reset_index(name='tp_count')

    district_agg = district_agg.merge(tp_by_district, on='ipc_geographic_unit_full', how='left')
    district_agg['tp_count'] = district_agg['tp_count'].fillna(0)

    # Compute TP rate (only for districts with crises)
    district_agg['tp_rate'] = np.where(
        district_agg['total_crises'] > 0,
        100 * district_agg['tp_count'] / district_agg['total_crises'],
        np.nan  # No crises = N/A
    )

    return district_agg[['ipc_geographic_unit_full', 'ar_accuracy', 'tp_rate',
                        'avg_latitude', 'avg_longitude']]


def compute_s2_improvement_metrics(s2_period_df, ar_period_df):
    """
    Compute 3 Stage 2 improvement metrics per district:
    1. Accuracy delta (S2 - AR)
    2. FN recovery rate (% AR failures caught by S2)
    3. Will be joined with obs_comp for dominance

    Args:
        s2_period_df: Stage 2 predictions for a single time period
        ar_period_df: AR predictions for the same time period

    Returns:
        DataFrame with columns: ipc_geographic_unit_full, accuracy_delta, fn_recovery_rate,
                               avg_latitude, avg_longitude
    """
    # Merge AR and S2 on geographic_unit + period
    # S2 already has year_month, AR should too
    merged = s2_period_df.merge(
        ar_period_df,
        on=['ipc_geographic_unit_full'],
        suffixes=('_s2', '_ar'),
        how='inner'  # Only keep observations that exist in both
    )

    # Compute correctness for S2
    merged['s2_correct'] = (merged['y_pred_youden'] == merged['ipc_future_crisis_s2']).astype(int)

    # AR correctness column may have _ar suffix after merge, or may be 'correct'
    ar_correct_col = 'correct_ar' if 'correct_ar' in merged.columns else 'correct'

    # Group by district
    district_agg = merged.groupby('ipc_geographic_unit_full').agg({
        's2_correct': 'mean',  # S2 accuracy
        ar_correct_col: 'mean',  # AR accuracy
        'avg_latitude_s2': 'mean',
        'avg_longitude_s2': 'mean'
    }).reset_index()

    # Rename AR accuracy column to standard name
    if ar_correct_col != 'ar_accuracy':
        district_agg.rename(columns={ar_correct_col: 'ar_accuracy'}, inplace=True)
    else:
        district_agg['ar_accuracy'] = district_agg[ar_correct_col]

    # Accuracy delta (S2 - AR, in percentage points)
    district_agg['accuracy_delta'] = 100 * (district_agg['s2_correct'] - district_agg['ar_accuracy'])

    # FN recovery: Find AR failures (FN cases)
    # AR failure column may have _ar suffix after merge
    ar_failure_col = 'ar_failure_ar' if 'ar_failure_ar' in merged.columns else 'ar_failure'
    ar_failures = merged[merged[ar_failure_col] == 1].copy() if ar_failure_col in merged.columns else pd.DataFrame()

    if len(ar_failures) > 0:
        # Among AR failures, what % did S2 catch?
        fn_recovery = ar_failures.groupby('ipc_geographic_unit_full').agg({
            'y_pred_youden': 'mean'  # % of AR failures that S2 predicted as crisis
        }).reset_index()
        fn_recovery['fn_recovery_rate'] = 100 * fn_recovery['y_pred_youden']

        # Merge back
        district_agg = district_agg.merge(
            fn_recovery[['ipc_geographic_unit_full', 'fn_recovery_rate']],
            on='ipc_geographic_unit_full',
            how='left'
        )
    else:
        district_agg['fn_recovery_rate'] = np.nan

    # Fill NaN recovery rates with 0 if no AR failures in district
    district_agg['fn_recovery_rate'] = district_agg['fn_recovery_rate'].fillna(0)

    return district_agg[['ipc_geographic_unit_full', 'accuracy_delta', 'fn_recovery_rate',
                        'avg_latitude_s2', 'avg_longitude_s2']].rename(columns={
                            'avg_latitude_s2': 'avg_latitude',
                            'avg_longitude_s2': 'avg_longitude'
                        })


def compute_dominance_metrics(obs_comp_period):
    """
    Aggregate observation-level winners to district percentages.

    Args:
        obs_comp_period: Observation comparison data for a single time period

    Returns:
        DataFrame with columns: ipc_geographic_unit_full, ar_dominance_pct, s2_dominance_pct
    """
    if obs_comp_period is None or len(obs_comp_period) == 0:
        return pd.DataFrame(columns=['ipc_geographic_unit_full', 'ar_dominance_pct', 's2_dominance_pct'])

    # Ensure we have the winner column
    if 'ar_vs_s2_winner' not in obs_comp_period.columns:
        print("      Warning: ar_vs_s2_winner column not found in observation comparison")
        return pd.DataFrame(columns=['ipc_geographic_unit_full', 'ar_dominance_pct', 's2_dominance_pct'])

    # Group by geographic unit (may be named differently)
    geo_col = 'ipc_geographic_unit_full' if 'ipc_geographic_unit_full' in obs_comp_period.columns else 'geographic_unit'

    dominance = obs_comp_period.groupby(geo_col).agg({
        'ar_vs_s2_winner': lambda x: 100 * (x == 'AR Better').sum() / len(x)  # % AR wins
    }).reset_index()

    dominance.columns = ['ipc_geographic_unit_full', 'ar_dominance_pct']
    dominance['s2_dominance_pct'] = 100 - dominance['ar_dominance_pct']

    return dominance


def aggregate_to_districts(metric_df, ipc_gdf, metric_column, aggregation='mean'):
    """
    Join point metrics to IPC district polygons using spatial join.

    Args:
        metric_df: DataFrame with metrics, avg_latitude, avg_longitude columns
        ipc_gdf: GeoDataFrame of IPC district polygons
        metric_column: Name of the metric column to aggregate
        aggregation: 'mean' for continuous metrics, 'mode' for categorical

    Returns:
        GeoDataFrame with metric values mapped to district polygons
    """
    # Create point geometries from lat/lon
    geometry = [Point(xy) for xy in zip(metric_df['avg_longitude'], metric_df['avg_latitude'])]
    metric_points = gpd.GeoDataFrame(metric_df, geometry=geometry, crs="EPSG:4326")

    # Ensure IPC gdf has a clean index
    ipc_with_index = ipc_gdf.copy()
    if ipc_with_index.index.name is None:
        ipc_with_index['ipc_index'] = ipc_with_index.index
    else:
        ipc_with_index['ipc_index'] = ipc_with_index.index

    # Spatial join: points within polygons
    joined = gpd.sjoin(metric_points, ipc_with_index, how='left', predicate='within')

    # Use ipc_index column for aggregation
    index_col = 'ipc_index'

    # Remove rows that didn't match any polygon
    valid = joined.dropna(subset=[index_col])

    # Aggregate to district level
    if aggregation == 'mean':
        metric_agg = valid.groupby(index_col)[metric_column].mean()
    elif aggregation == 'mode':
        metric_agg = valid.groupby(index_col)[metric_column].apply(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
        )
    else:
        raise ValueError(f"Unknown aggregation method: {aggregation}")

    # Map to IPC polygons
    ipc_plot = ipc_gdf.copy()
    ipc_plot[metric_column] = ipc_plot.index.map(metric_agg)

    return ipc_plot


# =============================================================================
# COLOR SCHEMES FOR METRIC MAPS
# =============================================================================

COLOR_SCHEMES = {
    'ar_accuracy': {
        'cmap': 'Greens',
        'vmin': 0,
        'vmax': 100,
        'label': 'AR Accuracy (%)',
        'type': 'sequential'
    },
    'tp_rate': {
        'type': 'categorical',
        'bins': [0, 20, 40, 70, 100],
        'colors': ['#E53935', '#FF9800', '#64B5F6', '#2196F3'],
        'labels': ['<20%', '20-40%', '40-70%', '>70%'],
        'label': 'AR True Positive Rate'
    },
    'ar_dominance_pct': {
        'cmap': 'RdYlGn_r',  # Reversed so green = AR wins
        'vmin': 0,
        'vmax': 100,
        'center': 50,
        'label': 'AR Win % (Green = AR Better)',
        'type': 'diverging'
    },
    'accuracy_delta': {
        'cmap': 'RdYlGn',
        'vmin': -30,
        'vmax': 30,
        'center': 0,
        'label': 'Accuracy Δ (Green = S2 Better)',
        'type': 'diverging'
    },
    'fn_recovery_rate': {
        'cmap': 'Purples',
        'vmin': 0,
        'vmax': 100,
        'label': 'FN Recovery Rate (%)',
        'type': 'sequential'
    },
    's2_dominance_pct': {
        'cmap': 'RdYlGn',
        'vmin': 0,
        'vmax': 100,
        'center': 50,
        'label': 'S2 Win % (Green = S2 Better)',
        'type': 'diverging'
    }
}


def create_metric_map(ipc_plot, ipc_ground_truth, metric_column, color_config,
                     title, year_month, output_path, africa_basemap=None):
    """
    Create 3-panel map: Ground Truth | Metric Visualization | Summary Statistics

    Args:
        ipc_plot: GeoDataFrame with metric values mapped to districts
        ipc_ground_truth: GeoDataFrame with ground truth mapped to IPC polygons
        metric_column: Name of the metric column to visualize
        color_config: Dictionary with colormap configuration
        title: Title for the map
        year_month: Time period (e.g., '2021-06')
        output_path: Path to save the figure
        africa_basemap: GeoDataFrame with Africa country boundaries for context
    """
    fig, axes = plt.subplots(1, 3, figsize=(20, 8), dpi=150)

    # Panel A: Ground Truth (using filled IPC polygons with same color scheme as predictions)
    ax = axes[0]

    # Add Africa basemap first (as background)
    if africa_basemap is not None:
        africa_basemap.plot(ax=ax, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.5, zorder=1)

    if ipc_ground_truth is not None and 'ground_truth_binary' in ipc_ground_truth.columns:
        # Use the same color scheme as the prediction panel
        if color_config['type'] == 'categorical':
            # For categorical schemes, map binary ground truth to categories
            # For TP rate: crisis districts get highest category color
            ipc_ground_truth['gt_display'] = ipc_ground_truth['ground_truth_binary']

            # Plot no data first
            no_data = ipc_ground_truth[ipc_ground_truth['ground_truth_binary'].isna()]
            if len(no_data) > 0:
                no_data.plot(ax=ax, color='#E0E0E0', edgecolor='#CCCCCC', linewidth=0.2, zorder=2)

            # Plot no crisis (use lowest category color - red)
            no_crisis = ipc_ground_truth[ipc_ground_truth['ground_truth_binary'] == 0]
            if len(no_crisis) > 0:
                no_crisis.plot(ax=ax, color=color_config['colors'][0], edgecolor='white', linewidth=0.3, zorder=2)

            # Plot crisis (use highest category color - blue)
            crisis = ipc_ground_truth[ipc_ground_truth['ground_truth_binary'] == 1]
            if len(crisis) > 0:
                crisis.plot(ax=ax, color=color_config['colors'][-1], edgecolor='white', linewidth=0.3, zorder=2)

            # Legend
            legend_elements = []
            if len(crisis) > 0:
                legend_elements.append(mpatches.Patch(facecolor=color_config['colors'][-1], label=f'Crisis ({len(crisis)})'))
            if len(no_crisis) > 0:
                legend_elements.append(mpatches.Patch(facecolor=color_config['colors'][0], label=f'No Crisis ({len(no_crisis)})'))
            if len(no_data) > 0:
                legend_elements.append(mpatches.Patch(facecolor='#E0E0E0', label=f'No Data ({len(no_data)})'))

            ax.legend(handles=legend_elements, loc='lower left', fontsize=8)
        else:
            # For sequential/diverging colormaps, plot using the colormap
            # Map binary values to colormap range
            ipc_ground_truth.plot(
                ax=ax,
                column='ground_truth_binary',
                cmap=color_config['cmap'],
                vmin=0,
                vmax=1,
                edgecolor='white',
                linewidth=0.3,
                legend=True,
                legend_kwds={'label': 'Ground Truth', 'orientation': 'horizontal', 'shrink': 0.8},
                zorder=2,
                missing_kwds={'color': '#E0E0E0', 'edgecolor': '#CCCCCC', 'linewidth': 0.2}
            )

        crisis_count = (ipc_ground_truth['ground_truth_binary'] == 1).sum()
        ax.set_title(f'Ground Truth\n{crisis_count} Crisis Districts', fontweight='bold', fontsize=11)
    else:
        ax.text(0.5, 0.5, 'No Ground Truth Data', ha='center', va='center', fontsize=12)
        ax.set_title('Ground Truth\n(No Data)', fontweight='bold', fontsize=11)

    # Add country labels to Panel A
    if africa_basemap is not None and 'COUNTRY' in africa_basemap.columns:
        for idx, row in africa_basemap.iterrows():
            if hasattr(row.geometry, 'centroid'):
                centroid = row.geometry.centroid
                ax.text(centroid.x, centroid.y, row['COUNTRY'],
                       fontsize=7, ha='center', va='center',
                       weight='bold', color='#333333', zorder=3)

    ax.set_xlim(-20, 55)
    ax.set_ylim(-35, 40)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Panel B: Metric Visualization
    ax = axes[1]

    # Add Africa basemap first (as background)
    if africa_basemap is not None:
        africa_basemap.plot(ax=ax, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.5, zorder=1)

    if color_config['type'] == 'categorical':
        # Categorical color scheme (e.g., for TP rate bins)
        bins = color_config['bins']
        colors = color_config['colors']
        labels = color_config['labels']

        # Bin the data
        ipc_plot['binned'] = pd.cut(ipc_plot[metric_column], bins=bins,
                                     labels=labels, include_lowest=True)

        # Plot each category
        legend_elements = []
        for label, color in zip(labels, colors):
            subset = ipc_plot[ipc_plot['binned'] == label]
            if not subset.empty:
                subset.plot(ax=ax, color=color, edgecolor='white', linewidth=0.3, zorder=2)
                legend_elements.append(mpatches.Patch(facecolor=color, label=label))

        # No data category
        no_data = ipc_plot[ipc_plot[metric_column].isna()]
        if not no_data.empty:
            no_data.plot(ax=ax, color='#E0E0E0', edgecolor='#CCCCCC', linewidth=0.2, zorder=2)
            legend_elements.append(mpatches.Patch(facecolor='#E0E0E0', label='No Data'))

        ax.legend(handles=legend_elements, loc='lower left', fontsize=8)

    else:
        # Sequential or diverging colormap
        ipc_plot.plot(
            ax=ax,
            column=metric_column,
            cmap=color_config['cmap'],
            vmin=color_config.get('vmin'),
            vmax=color_config.get('vmax'),
            edgecolor='white',
            linewidth=0.3,
            legend=True,
            legend_kwds={'label': color_config['label'], 'orientation': 'horizontal',
                        'shrink': 0.8},
            zorder=2
        )

    # Add country labels to Panel B
    if africa_basemap is not None and 'COUNTRY' in africa_basemap.columns:
        for idx, row in africa_basemap.iterrows():
            if hasattr(row.geometry, 'centroid'):
                centroid = row.geometry.centroid
                ax.text(centroid.x, centroid.y, row['COUNTRY'],
                       fontsize=7, ha='center', va='center',
                       weight='bold', color='#333333', zorder=3)

    # Count districts with data
    valid_districts = ipc_plot[metric_column].notna().sum()

    ax.set_title(f'{title}\n{valid_districts} Districts', fontweight='bold', fontsize=11)
    ax.set_xlim(-20, 55)
    ax.set_ylim(-35, 40)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    # Panel C: Additional context or comparison
    # For now, we'll show the metric distribution as a simple visualization
    ax = axes[2]

    # Show statistics
    metric_values = ipc_plot[metric_column].dropna()
    if len(metric_values) > 0:
        stats_text = f"Metric: {color_config['label']}\n\n"
        stats_text += f"Mean: {metric_values.mean():.2f}\n"
        stats_text += f"Median: {metric_values.median():.2f}\n"
        stats_text += f"Std: {metric_values.std():.2f}\n"
        stats_text += f"Min: {metric_values.min():.2f}\n"
        stats_text += f"Max: {metric_values.max():.2f}\n"
        stats_text += f"\nDistricts: {len(metric_values)}"

        ax.text(0.1, 0.5, stats_text, fontsize=10, family='monospace',
               verticalalignment='center')
        ax.axis('off')
        ax.set_title('Summary Statistics', fontweight='bold', fontsize=11)
    else:
        ax.text(0.5, 0.5, 'No Data', ha='center', va='center', fontsize=12)
        ax.axis('off')

    # Main title
    fig.suptitle(f'{title} - {year_month}', fontweight='bold', fontsize=14, y=0.98)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()

    return True


# =============================================================================
# METRIC MAP SUITE ORCHESTRATION
# =============================================================================

def create_metric_map_suite():
    """
    Generate all 6 map types across 9 time periods (54 total maps).

    Map types:
    1. AR Accuracy Heatmap
    2. AR True Positive Regions
    3. AR Dominance Map
    4. S2 Accuracy Improvement Delta
    5. S2 FN Recovery Map
    6. S2 Dominance Map
    """
    print("\n" + "=" * 80)
    print("6. GENERATING AR STRENGTHS VS STAGE 2 IMPROVEMENTS MAPS")
    print("=" * 80)

    # Load all data
    ar_df, s2_df, obs_comp = load_all_metric_data()
    if ar_df is None or s2_df is None:
        print("   ERROR: Failed to load metric data")
        return

    # Load IPC boundaries
    print("\n   Loading IPC boundaries...")
    ipc_gdf = load_ipc_boundaries()
    if ipc_gdf is None:
        print("   ERROR: Failed to load IPC boundaries")
        return

    # Load Africa basemap for country context
    print("\n   Loading Africa basemap...")
    africa_basemap = load_africa_basemap()
    if africa_basemap is None:
        print("   WARNING: Failed to load Africa basemap, maps will not have country boundaries")
    else:
        print(f"      Loaded {len(africa_basemap)} countries")

    # Define map configurations
    map_configs = [
        ('ar_accuracy', 'ar', 'ar_accuracy', 'AR Accuracy Heatmap', 'ar_strengths/accuracy_heatmap'),
        ('tp_rate', 'ar', 'tp_rate', 'AR True Positive Regions', 'ar_strengths/true_positive_regions'),
        ('ar_dominance_pct', 'dominance', 'ar_dominance_pct', 'AR Dominance Map', 'ar_strengths/ar_dominance'),
        ('accuracy_delta', 's2', 'accuracy_delta', 'S2 Accuracy Improvement', 's2_improvements/accuracy_delta'),
        ('fn_recovery_rate', 's2', 'fn_recovery_rate', 'S2 FN Recovery', 's2_improvements/fn_recovery'),
        ('s2_dominance_pct', 'dominance', 's2_dominance_pct', 'S2 Dominance Map', 's2_improvements/s2_dominance')
    ]

    maps_created = 0
    total_expected = len(IPC_DATES) * len(map_configs)

    # Iterate through time periods
    for year_month in IPC_DATES:
        print(f"\n   Processing {year_month}...")

        # Filter by period
        ar_period = ar_df[ar_df['year_month'] == year_month]
        s2_period = s2_df[s2_df['year_month'] == year_month] if 'year_month' in s2_df.columns else s2_df
        obs_period = obs_comp[obs_comp['year_month'] == year_month] if obs_comp is not None and 'year_month' in obs_comp.columns else None

        # Skip if no data
        if len(ar_period) == 0:
            print(f"      WARNING: No AR data for {year_month}, skipping...")
            continue

        # Compute metrics once per period
        print(f"      Computing metrics...")
        ar_metrics = compute_ar_district_metrics(ar_period)

        s2_metrics = None
        if len(s2_period) > 0:
            s2_metrics = compute_s2_improvement_metrics(s2_period, ar_period)

        dominance = None
        if obs_period is not None and len(obs_period) > 0:
            dominance = compute_dominance_metrics(obs_period)
            # Add coordinates from AR metrics
            if dominance is not None and len(dominance) > 0:
                coords = ar_metrics[['ipc_geographic_unit_full', 'avg_latitude', 'avg_longitude']]
                dominance = dominance.merge(coords, on='ipc_geographic_unit_full', how='left')

        # Create ground truth mapped to IPC polygons
        ground_truth_data = ar_period[['ipc_geographic_unit_full', 'avg_latitude', 'avg_longitude', 'ipc_future_crisis']].copy()

        # Aggregate ground truth to district level (mode for binary)
        gt_agg = ground_truth_data.groupby('ipc_geographic_unit_full').agg({
            'ipc_future_crisis': lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan,
            'avg_latitude': 'mean',
            'avg_longitude': 'mean'
        }).reset_index()
        gt_agg.rename(columns={'ipc_future_crisis': 'ground_truth_binary'}, inplace=True)

        # Spatial join to IPC polygons
        ipc_ground_truth = aggregate_to_districts(gt_agg, ipc_gdf, 'ground_truth_binary', aggregation='mode')

        # Create all 6 maps for this period
        for map_type, source, metric_col, title, subdirectory in map_configs:
            # Select appropriate metric dataframe
            if source == 'ar':
                metric_df = ar_metrics
            elif source == 's2':
                metric_df = s2_metrics
            elif source == 'dominance':
                metric_df = dominance
            else:
                continue

            # Skip if metric data unavailable
            if metric_df is None or len(metric_df) == 0:
                print(f"      [SKIP] {map_type} (no {source} data)")
                continue

            # Spatial join
            ipc_plot = aggregate_to_districts(metric_df, ipc_gdf, metric_col)

            # Create output directory
            output_dir = FIGURES_DIR / 'ar_strength_vs_s2_improvement' / subdirectory / year_month
            output_dir.mkdir(parents=True, exist_ok=True)
            output_path = output_dir / f'{map_type}_{year_month}.png'

            # Create map
            try:
                success = create_metric_map(
                    ipc_plot=ipc_plot,
                    ipc_ground_truth=ipc_ground_truth,
                    metric_column=metric_col,
                    color_config=COLOR_SCHEMES[metric_col],
                    title=title,
                    year_month=year_month,
                    output_path=output_path,
                    africa_basemap=africa_basemap
                )

                if success:
                    maps_created += 1
                    print(f"      [OK] {title}")

            except Exception as e:
                print(f"      [ERROR] {title}: {str(e)}")

    # Summary
    print("\n" + "=" * 80)
    print("METRIC MAPS COMPLETE")
    print("=" * 80)
    print(f"\n   Maps created: {maps_created} / {total_expected}")
    print(f"   Output directory: {FIGURES_DIR / 'ar_strength_vs_s2_improvement'}")
    print(f"   - AR Strengths: 3 map types × {len(IPC_DATES)} periods")
    print(f"   - S2 Improvements: 3 map types × {len(IPC_DATES)} periods")


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def ensure_directories():
    """Ensure output directories exist."""
    output_dir = FIGURES_DIR / 'ar_comparison_maps'
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / 'multiclass').mkdir(exist_ok=True)
    (output_dir / 'binary').mkdir(exist_ok=True)

def main():
    """Main execution function."""
    ensure_directories()

    if not HAS_GEOPANDAS:
        print("\n   ERROR: geopandas not installed.")
        print("   Install with: pip install geopandas")
        return

    # Output directories
    output_dir = FIGURES_DIR / 'ar_comparison_maps'
    multiclass_dir = output_dir / 'multiclass'
    binary_dir = output_dir / 'binary'

    # Load Africa basemap (for country context)
    print("\n1. Loading Africa basemap...")
    africa_basemap = load_africa_basemap()

    # Load IPC boundaries (replaces GADM - matches Stage 1)
    print("\n2. Loading IPC boundaries...")
    ipc_gdf = load_ipc_boundaries()
    if ipc_gdf is None:
        print("   ERROR: Failed to load IPC boundaries")
        return

    # Load AR predictions
    print("\n3. Loading AR model predictions...")
    ar_df = load_ar_predictions()
    if ar_df is None:
        print("   ERROR: Failed to load AR predictions")
        return

    # Auto-detect ML models
    print("\n4. Auto-detecting ML models...")
    all_models = auto_detect_models()
    print(f"   Found {len(all_models)} models")

    # Generate maps
    print("\n5. Generating maps...")
    maps_created = 0
    total_expected = len(IPC_DATES) * len(all_models) * 2  # multiclass + binary

    for viz_group in ['multiclass', 'binary']:
        group_dir = multiclass_dir if viz_group == 'multiclass' else binary_dir

        for year_month in IPC_DATES:
            # Create date subdirectory
            date_dir = group_dir / year_month
            date_dir.mkdir(parents=True, exist_ok=True)

            for model_name in all_models:
                print(f"   [{viz_group}] {year_month} - {model_name}...")

                # Load ML predictions for this model
                ml_df = load_model_predictions(model_name)
                if ml_df is not None:
                    # Add year_month if not present
                    if 'year_month' not in ml_df.columns:
                        if 'ipc_period_start' in ml_df.columns:
                            ml_df['year_month'] = pd.to_datetime(ml_df['ipc_period_start']).dt.strftime('%Y-%m')

                # Sanitize model name for filename (replace slashes with underscores)
                safe_model_name = model_name.replace('/', '_').replace('\\', '_')

                # Create map
                output_path = date_dir / f'ar_comparison_{viz_group}_{safe_model_name}_{year_month}.png'

                success = create_three_panel_map(
                    ipc_gdf=ipc_gdf,
                    ar_df=ar_df,
                    ml_df=ml_df,
                    model_name=model_name,
                    year_month=year_month,
                    output_path=output_path,
                    africa_basemap=africa_basemap,
                    visualization_group=viz_group
                )

                if success:
                    maps_created += 1

    # Summary
    print("\n" + "=" * 80)
    print("PHASE 5 STEP 6 COMPLETE: AR Comparison Choropleth Maps")
    print("=" * 80)
    print(f"\n   Maps created: {maps_created} / {total_expected}")
    print(f"   Output directory: {output_dir}")
    print(f"   - Multiclass maps: {multiclass_dir}")
    print(f"   - Binary maps: {binary_dir}")

    # Generate metric-based maps (AR strengths vs S2 improvements)
    create_metric_map_suite()


if __name__ == '__main__':
    main()
