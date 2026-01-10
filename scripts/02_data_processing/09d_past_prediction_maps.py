"""
09d_past_prediction_maps.py - Past Status + Prediction Outcome Maps

Creates 2-panel maps using IPC boundary polygons showing:
1. Previous period's actual IPC status (what the model saw)
2. Current period's prediction outcome (confusion matrix - 4 colors)

This visualizes the temporal relationship: Past -> Prediction -> Outcome

Author: Victor Collins Oppon
Date: 2025-12-02
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
import numpy as np
from datetime import datetime
from config import BASE_DIR

# Paths
BASE_DIR = Path(rstr(BASE_DIR.parent.parent.parent))
IPC_BOUNDARIES = BASE_DIR / 'data' / 'ipc_shapefiles' / 'ipc_africa_all_boundaries.geojson'
PRED_DIR = BASE_DIR / 'results' / 'district_level' / 'stage1_baseline'
OUTPUT_DIR = BASE_DIR / 'figures' / 'stage1_past_prediction'
NE_FILE = BASE_DIR / 'data' / 'natural_earth' / 'ne_50m_admin_0_countries_africa.shp'
STAGE1_FEATURES = BASE_DIR / 'data' / 'district_level' / 'stage1_features.parquet'

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Country label positions
COUNTRY_LABELS = {
    'BFA': ('Burkina\nFaso', -1.5, 12.5),
    'BDI': ('Burundi', 29.5, -3.5),
    'CMR': ('Cameroon', 12.5, 6.0),
    'CAF': ('CAR', 20.5, 6.5),
    'TCD': ('Chad', 18.5, 15.5),
    'COD': ('DRC', 23.5, -3.0),
    'ETH': ('Ethiopia', 39.5, 8.5),
    'KEN': ('Kenya', 38.0, 0.5),
    'MDG': ('Madagascar', 47.0, -19.0),
    'MWI': ('Malawi', 34.0, -13.5),
    'MLI': ('Mali', -4.0, 17.5),
    'MRT': ('Mauritania', -10.5, 20.5),
    'MOZ': ('Mozambique', 35.5, -18.5),
    'NER': ('Niger', 9.5, 17.5),
    'NGA': ('Nigeria', 8.0, 9.5),
    'RWA': ('Rwanda', 29.8, -2.0),
    'SEN': ('Senegal', -14.5, 14.5),
    'SOM': ('Somalia', 46.0, 5.0),
    'SSD': ('South\nSudan', 30.0, 7.5),
    'SDN': ('Sudan', 30.0, 16.0),
    'SWZ': ('Eswatini', 31.5, -26.5),
    'UGA': ('Uganda', 32.5, 1.5),
    'ZWE': ('Zimbabwe', 29.5, -19.5),
}


def add_country_labels(ax, fontsize=6, color='#666666'):
    """Add country name labels to the map."""
    for code, (name, lon, lat) in COUNTRY_LABELS.items():
        ax.annotate(
            name, xy=(lon, lat), fontsize=fontsize, ha='center', va='center',
            color=color, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.6, edgecolor='none')
        )


def load_africa_basemap():
    """Load Africa basemap."""
    if NE_FILE.exists():
        return gpd.read_file(NE_FILE)
    return None


def load_ipc_boundaries():
    """Load official IPC district boundaries."""
    if not IPC_BOUNDARIES.exists():
        print(f"IPC boundaries not found: {IPC_BOUNDARIES}")
        return None
    gdf = gpd.read_file(IPC_BOUNDARIES)
    print(f"Loaded {len(gdf)} IPC districts from {gdf['country_code'].nunique()} countries")
    return gdf


def load_predictions(horizon):
    """Load predictions for a specific horizon."""
    pred_file = PRED_DIR / f'predictions_h{horizon}_district_averaged.parquet'
    if not pred_file.exists():
        return None
    df = pd.read_parquet(pred_file)
    # Create year_month column from ipc_period_start
    df['year_month'] = pd.to_datetime(df['ipc_period_start']).dt.strftime('%Y-%m')
    return df


def load_raw_ipc_data():
    """Load raw IPC data from stage1_features for fallback when predictions don't exist."""
    if not STAGE1_FEATURES.exists():
        return None
    df = pd.read_parquet(STAGE1_FEATURES)
    df['year_month'] = pd.to_datetime(df['ipc_period_start']).dt.strftime('%Y-%m')
    return df


def get_raw_ipc_for_period(raw_data, period, ipc_gdf):
    """Get raw IPC data for a period and match to IPC boundaries."""
    period_data = raw_data[raw_data['year_month'] == period].copy()
    if len(period_data) == 0:
        return None

    # Create GeoDataFrame
    gdf = gpd.GeoDataFrame(
        period_data,
        geometry=gpd.points_from_xy(period_data['avg_longitude'], period_data['avg_latitude']),
        crs='EPSG:4326'
    )

    # Spatial join to IPC boundaries
    joined = gpd.sjoin(gdf, ipc_gdf, how='left', predicate='within')

    # Aggregate to IPC polygons - just need ipc_value for crisis status
    if 'index_right' not in joined.columns or joined['index_right'].isna().all():
        return None

    valid = joined.dropna(subset=['index_right'])
    agg = valid.groupby('index_right').agg({
        'ipc_value': 'max',
        'ipc_country': 'first'
    }).reset_index()

    # Create y_true for compatibility with aggregation function
    agg['y_true'] = (agg['ipc_value'] >= 3).astype(int)

    # Merge back to IPC geometries
    ipc_plot = ipc_gdf.copy().reset_index()
    ipc_plot = ipc_plot.merge(agg, left_on='index', right_on='index_right', how='left')

    return ipc_plot


def match_predictions_to_ipc(predictions, ipc_gdf):
    """Match predictions to IPC boundaries using spatial join on centroids."""
    pred_gdf = gpd.GeoDataFrame(
        predictions,
        geometry=gpd.points_from_xy(predictions['avg_longitude'], predictions['avg_latitude']),
        crs='EPSG:4326'
    )
    joined = gpd.sjoin(pred_gdf, ipc_gdf, how='left', predicate='within')
    matched = joined['index_right'].notna().sum()
    total = len(joined)
    print(f"   Matched {matched}/{total} predictions to IPC boundaries ({matched/total*100:.1f}%)")
    return joined


def aggregate_to_ipc(period_data, ipc_gdf):
    """Aggregate predictions to IPC polygons using optimal threshold."""
    if 'index_right' not in period_data.columns or period_data['index_right'].isna().all():
        return ipc_gdf.copy()

    valid_data = period_data.dropna(subset=['index_right'])

    agg = valid_data.groupby('index_right').agg({
        'y_true': 'max',
        'y_pred_optimal': 'max',
        'ipc_country': 'first'
    }).reset_index()

    # Compute confusion class
    agg['confusion_class'] = 'No Data'
    agg.loc[(agg['y_true'] == 0) & (agg['y_pred_optimal'] == 0), 'confusion_class'] = 'TN'
    agg.loc[(agg['y_true'] == 1) & (agg['y_pred_optimal'] == 1), 'confusion_class'] = 'TP'
    agg.loc[(agg['y_true'] == 0) & (agg['y_pred_optimal'] == 1), 'confusion_class'] = 'FP'
    agg.loc[(agg['y_true'] == 1) & (agg['y_pred_optimal'] == 0), 'confusion_class'] = 'FN'

    # Merge back to IPC geometries
    ipc_plot = ipc_gdf.copy().reset_index()
    ipc_plot = ipc_plot.merge(agg, left_on='index', right_on='index_right', how='left')

    return ipc_plot


def compute_metrics(period_data):
    """Compute metrics from prediction data using optimal threshold."""
    y_true = period_data['y_true'].values
    y_pred = period_data['y_pred_optimal'].values

    tn = ((y_true == 0) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    total = len(y_true)
    accuracy = (tn + tp) / total * 100 if total > 0 else 0
    precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0

    return {
        'total': total,
        'tn': int(tn), 'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
        'accuracy': accuracy, 'precision': precision, 'recall': recall
    }


def get_previous_period(current_period, horizon):
    """Calculate the previous period (h months before current)."""
    year, month = map(int, current_period.split('-'))
    prev_month = month - horizon
    prev_year = year
    while prev_month <= 0:
        prev_month += 12
        prev_year -= 1
    return f"{prev_year:04d}-{prev_month:02d}"


def create_past_prediction_map(current_ipc, prev_ipc, africa_gdf, metrics,
                                current_period, prev_period, horizon, output_dir):
    """
    Create 2-panel map using IPC boundary polygons:
    - Left: Previous period's actual IPC status
    - Right: Current period's prediction outcome (confusion matrix)
    """
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))

    # Main title
    fig.suptitle(
        f'Food Security Crisis Prediction: Past Status -> Prediction Outcome\n'
        f'Horizon: {horizon} months | Districts: {metrics["total"]:,} | '
        f'Precision: {metrics["precision"]:.1f}% | Recall: {metrics["recall"]:.1f}%',
        fontsize=14, fontweight='bold', y=0.98
    )

    # Colors
    past_colors = {'no_crisis': '#66BB6A', 'crisis': '#EF5350'}
    confusion_colors = {
        'TN': '#4CAF50',   # Green
        'TP': '#2196F3',   # Blue
        'FP': '#FF9800',   # Orange
        'FN': '#E53935',   # Red
        'No Data': '#E0E0E0'
    }

    # =========================================================================
    # Panel 1: Previous Period's Actual IPC Status
    # =========================================================================
    ax1 = axes[0]

    if prev_ipc is not None and 'y_true' in prev_ipc.columns:
        prev_crisis = (prev_ipc['y_true'] == 1).sum()
        prev_no_crisis = (prev_ipc['y_true'] == 0).sum()

        ax1.set_title(
            f'Previous Period: {prev_period}\n(Model Input - What the AR model saw)',
            fontsize=12, fontweight='bold', color='#333333'
        )

        # Plot basemap
        if africa_gdf is not None:
            africa_gdf.plot(ax=ax1, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.3)

        # Plot no crisis polygons (green)
        no_crisis = prev_ipc[prev_ipc['y_true'] == 0]
        if len(no_crisis) > 0:
            no_crisis.plot(ax=ax1, color=past_colors['no_crisis'], edgecolor='white', linewidth=0.2)

        # Plot crisis polygons (red)
        crisis = prev_ipc[prev_ipc['y_true'] == 1]
        if len(crisis) > 0:
            crisis.plot(ax=ax1, color=past_colors['crisis'], edgecolor='white', linewidth=0.2)

        # Legend
        legend1 = [
            mpatches.Patch(color=past_colors['no_crisis'], label=f'No Crisis (IPC 1-2): {prev_no_crisis:,}'),
            mpatches.Patch(color=past_colors['crisis'], label=f'Crisis (IPC 3+): {prev_crisis:,}')
        ]
        ax1.legend(handles=legend1, loc='lower left', fontsize=10,
                   title='Previous Period Status', title_fontsize=11)
    else:
        ax1.set_title(f'Previous Period: {prev_period}\n(No data available)',
                      fontsize=12, fontweight='bold', color='#999999')
        if africa_gdf is not None:
            africa_gdf.plot(ax=ax1, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.3)

    ax1.set_xlim(-25, 55)
    ax1.set_ylim(-40, 40)
    ax1.set_aspect('equal')
    ax1.axis('off')
    add_country_labels(ax1, fontsize=7)

    # =========================================================================
    # Panel 2: Current Period Prediction Outcome (Confusion Matrix)
    # =========================================================================
    ax2 = axes[1]
    ax2.set_title(
        f'Prediction Outcome: {current_period}\n'
        f'TN: {metrics["tn"]} | TP: {metrics["tp"]} | FP: {metrics["fp"]} | FN: {metrics["fn"]}',
        fontsize=12, fontweight='bold', color='#333333'
    )

    # Plot basemap
    if africa_gdf is not None:
        africa_gdf.plot(ax=ax2, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.3)

    # Plot each confusion class
    for cls in ['TN', 'TP', 'FP', 'FN']:
        subset = current_ipc[current_ipc['confusion_class'] == cls]
        if len(subset) > 0:
            edge_color = 'black' if cls == 'FN' else 'white'
            edge_width = 0.5 if cls == 'FN' else 0.2
            subset.plot(ax=ax2, color=confusion_colors[cls], edgecolor=edge_color, linewidth=edge_width)

    # Legend
    legend2 = [
        mpatches.Patch(color=confusion_colors['TN'], label=f'True Negative: {metrics["tn"]:,}'),
        mpatches.Patch(color=confusion_colors['TP'], label=f'True Positive: {metrics["tp"]:,}'),
        mpatches.Patch(color=confusion_colors['FP'], label=f'False Positive (Alarm): {metrics["fp"]:,}'),
        mpatches.Patch(color=confusion_colors['FN'], label=f'False Negative (MISSED): {metrics["fn"]:,}'),
    ]
    ax2.legend(handles=legend2, loc='lower left', fontsize=10,
               title='Prediction Outcome', title_fontsize=11)

    ax2.set_xlim(-25, 55)
    ax2.set_ylim(-40, 40)
    ax2.set_aspect('equal')
    ax2.axis('off')
    add_country_labels(ax2, fontsize=7)

    # Add arrow between panels
    fig.text(0.5, 0.5, '--->', fontsize=35, ha='center', va='center',
             color='#666666', fontweight='bold', transform=fig.transFigure)
    fig.text(0.5, 0.44, f'{horizon} months', fontsize=12, ha='center', va='center',
             color='#666666', transform=fig.transFigure)

    plt.tight_layout(rect=[0, 0, 1, 0.95])

    output_file = output_dir / f'past_prediction_{current_period}_h{horizon}.png'
    plt.savefig(output_file, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()

    # Force garbage collection to free memory
    import gc
    gc.collect()

    return output_file


def create_single_confusion_map(ipc_agg, africa_gdf, metrics,
                                 current_period, prev_period, horizon, output_dir):
    """Create single map showing confusion matrix with 4 colors using IPC boundaries."""
    fig, ax = plt.subplots(figsize=(12, 10))

    ax.set_title(
        f'Crisis Prediction Outcome: {prev_period} -> {current_period} (h={horizon} months)\n'
        f'Districts: {metrics["total"]:,} | Accuracy: {metrics["accuracy"]:.1f}% | '
        f'Precision: {metrics["precision"]:.1f}% | Recall: {metrics["recall"]:.1f}%',
        fontsize=14, fontweight='bold'
    )

    confusion_colors = {
        'TN': '#4CAF50',   # Green
        'TP': '#2196F3',   # Blue
        'FP': '#FF9800',   # Orange
        'FN': '#E53935',   # Red
        'No Data': '#E0E0E0'
    }

    # Plot basemap
    if africa_gdf is not None:
        africa_gdf.plot(ax=ax, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.3)

    # Plot each confusion class (FN last so it's on top)
    for cls in ['No Data', 'TN', 'TP', 'FP', 'FN']:
        subset = ipc_agg[ipc_agg['confusion_class'] == cls]
        if len(subset) > 0:
            edge_color = 'black' if cls == 'FN' else 'white'
            edge_width = 0.6 if cls == 'FN' else 0.2
            alpha = 0.3 if cls == 'No Data' else 1.0
            subset.plot(ax=ax, color=confusion_colors[cls], edgecolor=edge_color,
                       linewidth=edge_width, alpha=alpha)

    ax.set_xlim(-25, 55)
    ax.set_ylim(-40, 40)
    ax.set_aspect('equal')
    ax.axis('off')
    add_country_labels(ax, fontsize=8)

    # Legend with descriptions
    legend_elements = [
        mpatches.Patch(color=confusion_colors['TN'],
                       label=f'True Negative ({metrics["tn"]:,})\nNo crisis predicted, no crisis occurred'),
        mpatches.Patch(color=confusion_colors['TP'],
                       label=f'True Positive ({metrics["tp"]:,})\nCrisis predicted, crisis occurred'),
        mpatches.Patch(color=confusion_colors['FP'],
                       label=f'False Positive ({metrics["fp"]:,})\nCrisis predicted, but no crisis (false alarm)'),
        mpatches.Patch(color=confusion_colors['FN'],
                       label=f'False Negative ({metrics["fn"]:,})\nNo crisis predicted, but crisis occurred (MISSED)'),
    ]

    ax.legend(handles=legend_elements, loc='lower left', fontsize=10,
              title='Prediction Outcome (Confusion Matrix)', title_fontsize=11,
              framealpha=0.95)

    plt.tight_layout()

    output_file = output_dir / f'confusion_map_{current_period}_h{horizon}.png'
    plt.savefig(output_file, dpi=120, bbox_inches='tight', facecolor='white')
    plt.close()

    # Force garbage collection to free memory
    import gc
    gc.collect()

    return output_file


def main():
    print("=" * 80)
    print("Past Status + Prediction Outcome Maps (IPC Boundaries)")
    print("=" * 80)
    print(f"Start: {datetime.now()}\n")

    # Load basemap
    africa_gdf = load_africa_basemap()
    if africa_gdf is not None:
        print(f"Loaded Africa basemap: {len(africa_gdf)} countries")

    # Load IPC boundaries
    ipc_gdf = load_ipc_boundaries()
    if ipc_gdf is None:
        print("ERROR: Could not load IPC boundaries")
        return

    # Load raw IPC data for fallback when previous period not in predictions
    print("\n3. Loading raw IPC data for fallback...")
    raw_ipc_data = load_raw_ipc_data()
    if raw_ipc_data is not None:
        raw_periods = sorted(raw_ipc_data['year_month'].unique())
        print(f"   Loaded {len(raw_ipc_data):,} records across {len(raw_periods)} periods")
        print(f"   Range: {raw_periods[0]} to {raw_periods[-1]}")
    else:
        print("   WARNING: Could not load raw IPC data - fallback disabled")

    # Process each horizon
    for horizon in [4, 8, 12]:
        print(f"\n{'='*60}")
        print(f"h={horizon} months")
        print(f"{'='*60}")

        # Create horizon output directory
        horizon_dir = OUTPUT_DIR / f'h{horizon}'
        horizon_dir.mkdir(parents=True, exist_ok=True)

        # Load predictions
        predictions = load_predictions(horizon)
        if predictions is None:
            print(f"   No predictions found for h={horizon}")
            continue

        print(f"   Loaded {len(predictions):,} predictions")

        # Match predictions to IPC boundaries (once for all periods)
        joined = match_predictions_to_ipc(predictions, ipc_gdf)

        # Get unique periods
        periods = sorted(predictions['year_month'].unique())
        print(f"   {len(periods)} periods: {periods[0]} to {periods[-1]}")

        # Process each period
        for period in periods:
            # Get current period data
            current_data = joined[joined['year_month'] == period].copy()

            if len(current_data) == 0:
                continue

            # Aggregate to IPC polygons for current period
            current_ipc = aggregate_to_ipc(current_data, ipc_gdf)

            # Get previous period
            prev_period = get_previous_period(period, horizon)
            prev_data = joined[joined['year_month'] == prev_period].copy()

            # Aggregate previous period (if exists in predictions)
            if len(prev_data) > 0:
                prev_ipc = aggregate_to_ipc(prev_data, ipc_gdf)
            elif raw_ipc_data is not None:
                # Fallback to raw IPC data when prev_period not in predictions
                # This handles cases like 2021-02 which has data but no predictions
                prev_ipc = get_raw_ipc_for_period(raw_ipc_data, prev_period, ipc_gdf)
                if prev_ipc is not None:
                    print(f"      [Fallback] Using raw IPC data for {prev_period}")
            else:
                prev_ipc = None

            # Compute metrics for current period
            metrics = compute_metrics(current_data)

            # Create 2-panel map (past + prediction)
            output1 = create_past_prediction_map(
                current_ipc, prev_ipc, africa_gdf, metrics,
                period, prev_period, horizon, horizon_dir
            )
            print(f"   {period}: past_prediction map saved")

            # Create single confusion map
            if prev_ipc is not None:
                output2 = create_single_confusion_map(
                    current_ipc, africa_gdf, metrics,
                    period, prev_period, horizon, horizon_dir
                )
                print(f"   {period}: confusion map saved")

    print(f"\n{'='*80}")
    print("Complete")
    print(f"{'='*80}")
    print(f"\nSaved to: {OUTPUT_DIR}")
    print(f"End: {datetime.now()}")


if __name__ == '__main__':
    main()
