"""
09c_ipc_boundary_maps.py - Maps Using Official IPC Boundaries

Visualizes predictions using official IPC district boundaries from HDX/FEWS NET.
Matches predictions to IPC boundaries using spatial joins on centroids.

Author: Claude
Date: 2025-11-30
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
OUTPUT_DIR = BASE_DIR / 'figures' / 'stage1_ipc_boundaries'
NE_FILE = BASE_DIR / 'data' / 'natural_earth' / 'ne_50m_admin_0_countries_africa.shp'

# Country label positions (longitude, latitude) for 24 IPC countries
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
    'TGO': ('Togo', 1.0, 8.5),
    'UGA': ('Uganda', 32.5, 1.5),
    'ZWE': ('Zimbabwe', 29.5, -19.5),
}


def add_country_labels(ax, fontsize=7, color='#444444'):
    """Add country name labels to the map."""
    for code, (name, lon, lat) in COUNTRY_LABELS.items():
        ax.annotate(
            name,
            xy=(lon, lat),
            fontsize=fontsize,
            ha='center',
            va='center',
            color=color,
            fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7, edgecolor='none')
        )


def load_africa_basemap():
    """Load Africa basemap for context."""
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
        print(f"Predictions not found: {pred_file}")
        return None

    df = pd.read_parquet(pred_file)
    return df


def match_predictions_to_ipc(predictions, ipc_gdf):
    """
    Match predictions to IPC boundaries using spatial join on centroids.
    """
    # Create GeoDataFrame from predictions using centroids
    pred_gdf = gpd.GeoDataFrame(
        predictions,
        geometry=gpd.points_from_xy(predictions['avg_longitude'], predictions['avg_latitude']),
        crs='EPSG:4326'
    )

    # Spatial join to IPC boundaries
    joined = gpd.sjoin(pred_gdf, ipc_gdf, how='left', predicate='within')

    # Report match rate
    matched = joined['index_right'].notna().sum()
    total = len(joined)
    print(f"   Matched {matched}/{total} predictions to IPC boundaries ({matched/total*100:.1f}%)")

    return joined


def aggregate_to_ipc(period_data, ipc_gdf):
    """
    Aggregate predictions to IPC polygons.
    Uses max() for aggregation when multiple predictions fall in one polygon.
    CRITICAL: Compute ar_failure AFTER aggregation.
    """
    if 'index_right' not in period_data.columns or period_data['index_right'].isna().all():
        return ipc_gdf.copy()

    # Group by IPC polygon and aggregate using optimal threshold
    valid_data = period_data.dropna(subset=['index_right'])

    agg = valid_data.groupby('index_right').agg({
        'y_true': 'max',
        'y_pred_optimal': 'max',  # Use optimal threshold predictions
        'ipc_country': 'first'  # From predictions
    }).reset_index()

    # CRITICAL: Compute ar_failure AFTER aggregation using optimal predictions
    agg['ar_failure'] = ((agg['y_true'] == 1) & (agg['y_pred_optimal'] == 0)).astype(int)

    # Compute confusion class AFTER aggregation using optimal predictions
    agg['confusion_class'] = 'No Data'
    agg.loc[(agg['y_true'] == 0) & (agg['y_pred_optimal'] == 0), 'confusion_class'] = 'TN'
    agg.loc[(agg['y_true'] == 1) & (agg['y_pred_optimal'] == 1), 'confusion_class'] = 'TP'
    agg.loc[(agg['y_true'] == 0) & (agg['y_pred_optimal'] == 1), 'confusion_class'] = 'FP'
    agg.loc[(agg['y_true'] == 1) & (agg['y_pred_optimal'] == 0), 'confusion_class'] = 'FN'

    # Merge back to IPC geometries
    ipc_plot = ipc_gdf.copy().reset_index()
    ipc_plot = ipc_plot.merge(agg, left_on='index', right_on='index_right', how='left')

    return ipc_plot


def compute_raw_metrics(period_data):
    """Compute metrics from raw prediction data (not aggregated) using optimal threshold."""
    y_true = period_data['y_true'].values
    y_pred = period_data['y_pred_optimal'].values

    tn = ((y_true == 0) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()

    total = len(y_true)
    accuracy = (tn + tp) / total * 100 if total > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    ar_rate = fn / (tp + fn) * 100 if (tp + fn) > 0 else 0

    return {
        'total': total,
        'tn': int(tn), 'tp': int(tp), 'fp': int(fp), 'fn': int(fn),
        'accuracy': accuracy, 'recall': recall, 'ar_rate': ar_rate
    }


def create_comparison_map(ipc_agg, africa_gdf, period, horizon, metrics, output_dir):
    """Create 3-panel comparison map using IPC boundaries."""
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # Title with metrics
    fig.suptitle(
        f'IPC Food Security Crisis Prediction - {period}\n'
        f'h={horizon} months | Districts: {metrics["total"]:,} | '
        f'Accuracy: {metrics["accuracy"]:.1f}% | Recall: {metrics["recall"]:.1f}% | '
        f'AR Failures: {metrics["fn"]} ({metrics["ar_rate"]:.1f}%)',
        fontsize=14, fontweight='bold'
    )

    colors = {'no_crisis': '#4CAF50', 'crisis': '#E53935', 'missed': '#D32F2F', 'other': '#BDBDBD'}

    # Panel 1: Ground Truth
    ax1 = axes[0]
    ax1.set_title('Ground Truth\n(Actual IPC Status)', fontsize=12, fontweight='bold')

    if africa_gdf is not None:
        africa_gdf.plot(ax=ax1, color='#F5F5F5', edgecolor='#BDBDBD', linewidth=0.3)

    no_crisis = ipc_agg[ipc_agg['y_true'] == 0]
    crisis = ipc_agg[ipc_agg['y_true'] == 1]

    if len(no_crisis) > 0:
        no_crisis.plot(ax=ax1, color=colors['no_crisis'], edgecolor='white', linewidth=0.2)
    if len(crisis) > 0:
        crisis.plot(ax=ax1, color=colors['crisis'], edgecolor='white', linewidth=0.2)

    ax1.set_xlim(-25, 55)
    ax1.set_ylim(-40, 40)
    ax1.set_aspect('equal')
    ax1.axis('off')
    add_country_labels(ax1, fontsize=6)

    # Legend
    legend1 = [
        mpatches.Patch(color=colors['no_crisis'], label=f'No Crisis: {metrics["tn"] + metrics["fp"]:,}'),
        mpatches.Patch(color=colors['crisis'], label=f'Crisis: {metrics["tp"] + metrics["fn"]:,}')
    ]
    ax1.legend(handles=legend1, loc='lower left', fontsize=9)

    # Panel 2: Model Prediction
    ax2 = axes[1]
    ax2.set_title('Model Prediction', fontsize=12, fontweight='bold')

    if africa_gdf is not None:
        africa_gdf.plot(ax=ax2, color='#F5F5F5', edgecolor='#BDBDBD', linewidth=0.3)

    pred_no = ipc_agg[ipc_agg['y_pred_optimal'] == 0]
    pred_yes = ipc_agg[ipc_agg['y_pred_optimal'] == 1]

    if len(pred_no) > 0:
        pred_no.plot(ax=ax2, color=colors['no_crisis'], edgecolor='white', linewidth=0.2)
    if len(pred_yes) > 0:
        pred_yes.plot(ax=ax2, color=colors['crisis'], edgecolor='white', linewidth=0.2)

    ax2.set_xlim(-25, 55)
    ax2.set_ylim(-40, 40)
    ax2.set_aspect('equal')
    ax2.axis('off')
    add_country_labels(ax2, fontsize=6)

    legend2 = [
        mpatches.Patch(color=colors['no_crisis'], label=f'Pred No Crisis: {metrics["tn"] + metrics["fn"]:,}'),
        mpatches.Patch(color=colors['crisis'], label=f'Pred Crisis: {metrics["tp"] + metrics["fp"]:,}')
    ]
    ax2.legend(handles=legend2, loc='lower left', fontsize=9)

    # Panel 3: Missed Crises
    ax3 = axes[2]
    ax3.set_title(f'Missed Crises (AR Failures)\n{metrics["fn"]} missed ({metrics["ar_rate"]:.1f}%)',
                  fontsize=12, fontweight='bold', color='#D32F2F')

    if africa_gdf is not None:
        africa_gdf.plot(ax=ax3, color='#F5F5F5', edgecolor='#BDBDBD', linewidth=0.3)

    other = ipc_agg[ipc_agg['ar_failure'] != 1]
    missed = ipc_agg[ipc_agg['ar_failure'] == 1]

    if len(other) > 0:
        other.plot(ax=ax3, color=colors['other'], edgecolor='white', linewidth=0.2, alpha=0.5)
    if len(missed) > 0:
        missed.plot(ax=ax3, color=colors['missed'], edgecolor='black', linewidth=0.5)

    ax3.set_xlim(-25, 55)
    ax3.set_ylim(-40, 40)
    ax3.set_aspect('equal')
    ax3.axis('off')
    add_country_labels(ax3, fontsize=6)

    legend3 = [
        mpatches.Patch(color=colors['other'], alpha=0.5, label=f'Other: {metrics["total"] - metrics["fn"]:,}'),
        mpatches.Patch(color=colors['missed'], label=f'MISSED: {metrics["fn"]}')
    ]
    ax3.legend(handles=legend3, loc='lower left', fontsize=9)

    plt.tight_layout()

    output_file = output_dir / f'ipc_comparison_{period}_h{horizon}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_file


def create_confusion_map(ipc_agg, africa_gdf, period, horizon, metrics, output_dir):
    """Create confusion matrix map using IPC boundaries."""
    fig, ax = plt.subplots(figsize=(12, 10))

    ax.set_title(f'Model Performance - Prediction Outcomes\n{period} (h={horizon} months)',
                 fontsize=14, fontweight='bold')

    colors = {
        'TN': '#4CAF50',  # Green
        'TP': '#2196F3',  # Blue
        'FP': '#FF9800',  # Orange
        'FN': '#E53935',  # Red
        'No Data': '#E0E0E0'
    }

    if africa_gdf is not None:
        africa_gdf.plot(ax=ax, color='#F5F5F5', edgecolor='#BDBDBD', linewidth=0.3)

    # Plot each confusion class
    for cls, color in colors.items():
        subset = ipc_agg[ipc_agg['confusion_class'] == cls]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, edgecolor='white', linewidth=0.2)

    ax.set_xlim(-25, 55)
    ax.set_ylim(-40, 40)
    ax.set_aspect('equal')
    ax.axis('off')
    add_country_labels(ax, fontsize=7)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['TN'], label=f'True Negative: {metrics["tn"]:,}'),
        mpatches.Patch(facecolor=colors['TP'], label=f'True Positive: {metrics["tp"]:,}'),
        mpatches.Patch(facecolor=colors['FP'], label=f'False Positive: {metrics["fp"]:,}'),
        mpatches.Patch(facecolor=colors['FN'], label=f'False Negative (MISSED): {metrics["fn"]:,}'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10, title='Prediction Outcome')

    # Metrics box
    metrics_text = (f'Districts: {metrics["total"]:,}\nAccuracy: {metrics["accuracy"]:.1f}%\n'
                   f'Recall: {metrics["recall"]:.1f}%\nAR Failures: {metrics["fn"]} ({metrics["ar_rate"]:.1f}%)')
    ax.text(0.98, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()

    output_file = output_dir / f'ipc_confusion_{period}_h{horizon}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_file


def main():
    print("=" * 80)
    print("IPC Boundary Maps - Official IPC District Boundaries from HDX/FEWS NET")
    print("=" * 80)
    print(f"Start: {datetime.now()}\n")

    # Load Africa basemap
    africa_gdf = load_africa_basemap()
    if africa_gdf is not None:
        print(f"Loaded Africa basemap: {len(africa_gdf)} countries")

    # Load IPC boundaries
    ipc_gdf = load_ipc_boundaries()
    if ipc_gdf is None:
        print("ERROR: Could not load IPC boundaries")
        return

    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Process each horizon
    for horizon in [4, 8, 12]:
        print(f"\n{'='*60}")
        print(f"h={horizon} months")
        print(f"{'='*60}")

        h_output_dir = OUTPUT_DIR / f'h{horizon}'
        h_output_dir.mkdir(parents=True, exist_ok=True)

        # Load predictions
        preds = load_predictions(horizon)
        if preds is None:
            continue

        print(f"\nLoaded {len(preds):,} predictions")

        # Match predictions to IPC boundaries
        matched = match_predictions_to_ipc(preds, ipc_gdf)

        # Create period column
        matched['period'] = pd.to_datetime(matched['ipc_period_start']).dt.strftime('%Y-%m')

        # Get periods
        periods = sorted(matched['period'].unique())
        print(f"\n   {len(periods)} periods...")

        for period in periods:
            period_data = matched[matched['period'] == period].copy()

            if len(period_data) == 0:
                continue

            # Compute raw metrics
            metrics = compute_raw_metrics(period_data)

            print(f"\n   {period} ({metrics['total']:,} districts)")

            # Aggregate to IPC polygons
            ipc_agg = aggregate_to_ipc(period_data, ipc_gdf)

            # Create maps
            f1 = create_comparison_map(ipc_agg, africa_gdf, period, horizon, metrics, h_output_dir)
            print(f"      Saved: {f1}")

            f2 = create_confusion_map(ipc_agg, africa_gdf, period, horizon, metrics, h_output_dir)
            print(f"      Saved: {f2}")

    print(f"\n{'='*80}")
    print("Complete")
    print(f"{'='*80}")
    print(f"\nSaved to: {OUTPUT_DIR}")
    print(f"End: {datetime.now()}")


if __name__ == '__main__':
    main()
