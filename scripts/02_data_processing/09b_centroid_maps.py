"""
09b_centroid_maps.py - IPC District Centroid-Based Visualization

Visualizes predictions at full IPC district granularity using centroid points.
This avoids GADM aggregation entirely, showing 1 point per IPC district.

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
FEATURES_FILE = BASE_DIR / 'data' / 'stage1_features.parquet'
PRED_DIR = BASE_DIR / 'results' / 'district_level' / 'stage1_baseline'
OUTPUT_DIR = BASE_DIR / 'figures' / 'stage1_centroid_maps'
NE_FILE = BASE_DIR / 'data' / 'natural_earth' / 'ne_50m_admin_0_countries_africa.shp'

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

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


def load_predictions(horizon):
    """Load predictions for a specific horizon."""
    pred_file = PRED_DIR / f'predictions_h{horizon}_district_averaged.parquet'
    if not pred_file.exists():
        print(f"   Predictions not found: {pred_file}")
        return None

    df = pd.read_parquet(pred_file)
    return df


def load_features():
    """Load features with coordinates."""
    df = pd.read_parquet(FEATURES_FILE)
    # Keep only rows with valid coordinates
    df = df.dropna(subset=['avg_latitude', 'avg_longitude'])
    return df


def create_centroid_comparison_map(period_data, africa_gdf, period, horizon, output_dir):
    """
    Create 3-panel comparison using centroids instead of polygons.
    Shows full granular IPC data without GADM aggregation.
    """
    # Compute metrics from raw data using optimal threshold
    y_true = period_data['y_true'].values
    y_pred = period_data['y_pred_optimal'].values

    tn = ((y_true == 0) & (y_pred == 0)).sum()
    tp = ((y_true == 1) & (y_pred == 1)).sum()
    fp = ((y_true == 0) & (y_pred == 1)).sum()
    fn = ((y_true == 1) & (y_pred == 0)).sum()  # Missed crises

    total = len(y_true)
    accuracy = (tn + tp) / total * 100 if total > 0 else 0
    recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
    ar_rate = fn / (tp + fn) * 100 if (tp + fn) > 0 else 0

    # Create figure
    fig, axes = plt.subplots(1, 3, figsize=(20, 8))

    # Title
    fig.suptitle(f'IPC Food Security Crisis Prediction - {period}\n'
                 f'h={horizon} months | Districts: {total:,} | '
                 f'Accuracy: {accuracy:.1f}% | Recall: {recall:.1f}% | '
                 f'AR Failures: {fn} ({ar_rate:.1f}%)',
                 fontsize=14, fontweight='bold')

    # Colors
    colors = {
        'no_crisis': '#4CAF50',   # Green
        'crisis': '#E53935',       # Red
        'missed': '#D32F2F',       # Dark red
        'other': '#BDBDBD'         # Gray
    }

    # Point size
    point_size = 15

    # Panel 1: Ground Truth
    ax1 = axes[0]
    ax1.set_title('Ground Truth\n(Actual IPC Status)', fontsize=12, fontweight='bold')

    if africa_gdf is not None:
        africa_gdf.plot(ax=ax1, color='#F5F5F5', edgecolor='#BDBDBD', linewidth=0.3)

    # Plot no crisis (green)
    no_crisis = period_data[period_data['y_true'] == 0]
    if len(no_crisis) > 0:
        ax1.scatter(no_crisis['avg_longitude'], no_crisis['avg_latitude'],
                   c=colors['no_crisis'], s=point_size, alpha=0.7, label=f'No Crisis: {len(no_crisis):,}')

    # Plot crisis (red)
    crisis = period_data[period_data['y_true'] == 1]
    if len(crisis) > 0:
        ax1.scatter(crisis['avg_longitude'], crisis['avg_latitude'],
                   c=colors['crisis'], s=point_size, alpha=0.7, label=f'Crisis: {len(crisis):,}')

    ax1.set_xlim(-25, 55)
    ax1.set_ylim(-40, 40)
    ax1.set_aspect('equal')
    ax1.legend(loc='lower left', fontsize=9)
    ax1.axis('off')
    add_country_labels(ax1, fontsize=6)

    # Panel 2: Model Prediction
    ax2 = axes[1]
    ax2.set_title('Model Prediction', fontsize=12, fontweight='bold')

    if africa_gdf is not None:
        africa_gdf.plot(ax=ax2, color='#F5F5F5', edgecolor='#BDBDBD', linewidth=0.3)

    # Plot predicted no crisis (green)
    pred_no = period_data[period_data['y_pred_optimal'] == 0]
    if len(pred_no) > 0:
        ax2.scatter(pred_no['avg_longitude'], pred_no['avg_latitude'],
                   c=colors['no_crisis'], s=point_size, alpha=0.7, label=f'Pred No Crisis: {len(pred_no):,}')

    # Plot predicted crisis (red)
    pred_yes = period_data[period_data['y_pred_optimal'] == 1]
    if len(pred_yes) > 0:
        ax2.scatter(pred_yes['avg_longitude'], pred_yes['avg_latitude'],
                   c=colors['crisis'], s=point_size, alpha=0.7, label=f'Pred Crisis: {len(pred_yes):,}')

    ax2.set_xlim(-25, 55)
    ax2.set_ylim(-40, 40)
    ax2.set_aspect('equal')
    ax2.legend(loc='lower left', fontsize=9)
    ax2.axis('off')
    add_country_labels(ax2, fontsize=6)

    # Panel 3: Missed Crises (AR Failures)
    ax3 = axes[2]
    ax3.set_title(f'Missed Crises (AR Failures)\n{fn} missed ({ar_rate:.1f}%)',
                  fontsize=12, fontweight='bold', color='#D32F2F')

    if africa_gdf is not None:
        africa_gdf.plot(ax=ax3, color='#F5F5F5', edgecolor='#BDBDBD', linewidth=0.3)

    # Plot other districts faintly
    other = period_data[(period_data['y_true'] != 1) | (period_data['y_pred_optimal'] != 0)]
    if len(other) > 0:
        ax3.scatter(other['avg_longitude'], other['avg_latitude'],
                   c=colors['other'], s=point_size/2, alpha=0.3, label=f'Other: {len(other):,}')

    # Plot missed crises prominently
    missed = period_data[(period_data['y_true'] == 1) & (period_data['y_pred_optimal'] == 0)]
    if len(missed) > 0:
        ax3.scatter(missed['avg_longitude'], missed['avg_latitude'],
                   c=colors['missed'], s=point_size*2, alpha=0.9, marker='o',
                   edgecolors='black', linewidth=0.5, label=f'MISSED: {len(missed)}')

    ax3.set_xlim(-25, 55)
    ax3.set_ylim(-40, 40)
    ax3.set_aspect('equal')
    ax3.legend(loc='lower left', fontsize=9)
    ax3.axis('off')
    add_country_labels(ax3, fontsize=6)

    plt.tight_layout()

    # Save
    output_file = output_dir / f'centroid_comparison_{period}_h{horizon}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_file


def create_centroid_confusion_map(period_data, africa_gdf, period, horizon, output_dir):
    """
    Create confusion matrix visualization using centroids.
    Shows TN, TP, FP, FN at full IPC granularity.
    """
    # Compute metrics from raw data using optimal threshold
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

    # Assign confusion class using optimal threshold
    period_data = period_data.copy()
    period_data['confusion_class'] = 'TN'
    period_data.loc[(y_true == 0) & (y_pred == 0), 'confusion_class'] = 'TN'
    period_data.loc[(y_true == 1) & (y_pred == 1), 'confusion_class'] = 'TP'
    period_data.loc[(y_true == 0) & (y_pred == 1), 'confusion_class'] = 'FP'
    period_data.loc[(y_true == 1) & (y_pred == 0), 'confusion_class'] = 'FN'

    # Create figure
    fig, ax = plt.subplots(figsize=(12, 10))

    # Title
    ax.set_title(f'Model Performance - Prediction Outcomes\n{period} (h={horizon} months)',
                fontsize=14, fontweight='bold')

    # Colors
    colors = {
        'TN': '#4CAF50',   # Green - Correct No Crisis
        'TP': '#2196F3',   # Blue - Correct Crisis
        'FP': '#FF9800',   # Orange - False Alarm
        'FN': '#E53935'    # Red - Missed Crisis
    }

    # Point size
    point_size = 15

    # Plot basemap
    if africa_gdf is not None:
        africa_gdf.plot(ax=ax, color='#F5F5F5', edgecolor='#BDBDBD', linewidth=0.3)

    # Plot each class
    for cls, color in colors.items():
        subset = period_data[period_data['confusion_class'] == cls]
        if len(subset) > 0:
            ax.scatter(subset['avg_longitude'], subset['avg_latitude'],
                      c=color, s=point_size, alpha=0.7)

    ax.set_xlim(-25, 55)
    ax.set_ylim(-40, 40)
    ax.set_aspect('equal')
    ax.axis('off')
    add_country_labels(ax, fontsize=7)

    # Legend
    legend_elements = [
        mpatches.Patch(facecolor=colors['TN'], label=f'True Negative: {tn:,}'),
        mpatches.Patch(facecolor=colors['TP'], label=f'True Positive: {tp:,}'),
        mpatches.Patch(facecolor=colors['FP'], label=f'False Positive: {fp:,}'),
        mpatches.Patch(facecolor=colors['FN'], label=f'False Negative (MISSED): {fn:,}'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=10, title='Prediction Outcome')

    # Metrics box
    metrics_text = f'Districts: {total:,}\nAccuracy: {accuracy:.1f}%\nRecall: {recall:.1f}%\nAR Failures: {fn} ({ar_rate:.1f}%)'
    ax.text(0.98, 0.98, metrics_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.tight_layout()

    # Save
    output_file = output_dir / f'centroid_confusion_{period}_h{horizon}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_file


def main():
    print("=" * 80)
    print("Centroid-Based Maps - Full IPC Granularity (No GADM Aggregation)")
    print("=" * 80)
    print(f"Start: {datetime.now()}\n")

    # Load Africa basemap
    africa_gdf = load_africa_basemap()
    if africa_gdf is not None:
        print(f"Loaded Africa basemap: {len(africa_gdf)} countries")


    # Process each horizon
    for horizon in [4, 8, 12]:
        print(f"\n{'='*60}")
        print(f"h={horizon} months")
        print(f"{'='*60}")

        # Create output directory
        h_output_dir = OUTPUT_DIR / f'h{horizon}'
        h_output_dir.mkdir(parents=True, exist_ok=True)

        # Load predictions
        preds = load_predictions(horizon)
        if preds is None:
            continue

        print(f"\nLoaded {len(preds):,} predictions")

        # Predictions already have coordinates - no merge needed
        merged = preds.dropna(subset=['avg_latitude', 'avg_longitude'])
        print(f"Have coordinates for {len(merged):,}/{len(preds):,} predictions")

        # Create period column from ipc_period_start
        merged['period'] = pd.to_datetime(merged['ipc_period_start']).dt.strftime('%Y-%m')

        # Get periods
        periods = sorted(merged['period'].unique())
        print(f"\n   {len(periods)} periods...")

        for period in periods:
            period_data = merged[merged['period'] == period].copy()

            if len(period_data) == 0:
                continue

            print(f"\n   {period} ({len(period_data):,} districts)")

            # Create maps
            f1 = create_centroid_comparison_map(period_data, africa_gdf, period, horizon, h_output_dir)
            print(f"      Saved: {f1}")

            f2 = create_centroid_confusion_map(period_data, africa_gdf, period, horizon, h_output_dir)
            print(f"      Saved: {f2}")

    print(f"\n{'='*80}")
    print("Complete")
    print(f"{'='*80}")
    print(f"\nSaved to: {OUTPUT_DIR}")
    print(f"End: {datetime.now()}")


if __name__ == '__main__':
    main()
