"""
10_ar_failures_analysis.py - AR Failures (Missed Crises) Analysis for Stage 2

Visualizes all districts where the baseline AR model failed to predict crises.
These are the critical cases for Stage 2 dynamic news signal analysis.

AR Failure = y_true=1 (actual crisis) AND y_pred=0 (predicted no crisis)

UPDATED: Now uses optimal balanced threshold (P=R >= 0.6) instead of default 0.5

Author: Claude
Date: 2025-11-30, Updated 2025-12-02
"""

import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from pathlib import Path
import numpy as np
from datetime import datetime
from adjustText import adjust_text
from config import BASE_DIR

# Paths
BASE_DIR = Path(str(BASE_DIR.parent.parent.parent))
PRED_DIR = BASE_DIR / 'results' / 'district_level' / 'stage1_baseline'
OUTPUT_DIR = BASE_DIR / 'figures' / 'stage2_ar_failures'
NE_FILE = BASE_DIR / 'data' / 'natural_earth' / 'ne_50m_admin_0_countries_africa.shp'

# Create output directory
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Country label positions for 24 IPC countries
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


def add_country_labels(ax, fontsize=7, color='#666666'):
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
            bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, edgecolor='none')
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
    df['period'] = pd.to_datetime(df['ipc_period_start']).dt.strftime('%Y-%m')
    return df


def create_ar_failures_map(ar_failures, africa_gdf, period, horizon, output_dir, all_predictions=None):
    """
    Create a detailed map showing AR failures (missed crises) with district labels.
    Each missed district is labeled with its name for easy identification.
    """
    if len(ar_failures) == 0:
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 14))

    # Title
    ax.set_title(f'AR Failures (Missed Crises) - {period}\n'
                 f'h={horizon} months | {len(ar_failures)} Districts Missed\n'
                 f'Stage 2 Focus: Districts where baseline AR model failed',
                 fontsize=16, fontweight='bold', color='#D32F2F')

    # Plot basemap
    if africa_gdf is not None:
        africa_gdf.plot(ax=ax, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.3)

    # Plot all other districts faintly (if provided)
    if all_predictions is not None:
        other = all_predictions[all_predictions['ar_failure_optimal'] != 1]
        ax.scatter(other['avg_longitude'], other['avg_latitude'],
                   c='#E0E0E0', s=8, alpha=0.3, zorder=1)

    # Plot AR failures prominently
    ax.scatter(ar_failures['avg_longitude'], ar_failures['avg_latitude'],
               c='#D32F2F', s=100, alpha=0.9, marker='o',
               edgecolors='black', linewidth=1.5, zorder=3,
               label=f'Missed Crises: {len(ar_failures)}')

    # Add district labels using adjustText to avoid overlaps
    # Limit to 20 labels max per map to reduce clutter
    texts = []

    # Sort by longitude to spread labels geographically
    ar_failures_sorted = ar_failures.sort_values('avg_longitude')
    labels_to_show = ar_failures_sorted.head(20)

    for idx, row in labels_to_show.iterrows():
        t = ax.text(row['avg_longitude'], row['avg_latitude'], row['ipc_district'],
                   fontsize=7, fontweight='bold', color='#D32F2F',
                   bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.9,
                            edgecolor='#D32F2F', linewidth=0.5),
                   zorder=4)
        texts.append(t)

    # Use adjustText to reposition labels
    if texts:
        adjust_text(texts, ax=ax,
                   arrowprops=dict(arrowstyle='-', color='#666666', lw=0.8, alpha=0.7),
                   expand_points=(2.0, 2.0),
                   expand_text=(1.2, 1.2),
                   force_text=(0.8, 0.8),
                   force_points=(0.5, 0.5),
                   lim=500)

    ax.set_xlim(-25, 55)
    ax.set_ylim(-40, 40)
    ax.set_aspect('equal')
    ax.axis('off')

    # Add country labels (lighter to not compete)
    add_country_labels(ax, fontsize=8, color='#888888')

    # Legend
    legend_elements = [
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#D32F2F',
               markersize=12, markeredgecolor='black', markeredgewidth=1.5,
               label=f'Missed Crises (AR Failures): {len(ar_failures)}'),
        Line2D([0], [0], marker='o', color='w', markerfacecolor='#E0E0E0',
               markersize=8, label='Other Districts'),
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=11,
              title='District Status', title_fontsize=12)

    # Add info box with district list
    countries_affected = ar_failures['ipc_country'].nunique()
    country_counts = ar_failures.groupby('ipc_country').size().sort_values(ascending=False)

    info_text = f"Countries Affected: {countries_affected}\n"
    info_text += "-" * 25 + "\n"
    for country, count in country_counts.head(8).items():
        info_text += f"{country}: {count}\n"
    if len(country_counts) > 8:
        info_text += f"... and {len(country_counts) - 8} more"

    ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#D32F2F'))

    plt.tight_layout()

    # Save
    output_file = output_dir / f'ar_failures_{period}_h{horizon}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_file


def create_chronic_failures_map(all_ar_failures, africa_gdf, horizon, output_dir):
    """
    Create a map showing districts that are REPEATEDLY missed (chronic AR failures).
    Size of marker indicates how many times the district was missed.
    """
    # Count failures per district (by country + district name, NOT ipc_id)
    failure_counts = all_ar_failures.groupby(['ipc_country', 'ipc_district']).agg({
        'avg_latitude': 'first',
        'avg_longitude': 'first',
        'period': 'count'
    }).reset_index()
    failure_counts.columns = ['ipc_country', 'ipc_district', 'avg_latitude', 'avg_longitude', 'miss_count']

    # Filter to chronic failures (missed 2+ times)
    chronic = failure_counts[failure_counts['miss_count'] >= 2].copy()

    if len(chronic) == 0:
        print("   No chronic failures found")
        return None

    # Create figure
    fig, ax = plt.subplots(figsize=(16, 14))

    # Title
    total_periods = all_ar_failures['period'].nunique()
    ax.set_title(f'Chronic AR Failures - Districts Missed Multiple Times\n'
                 f'h={horizon} months | {len(chronic)} Districts with Repeated Failures\n'
                 f'(Out of {total_periods} IPC Periods)',
                 fontsize=16, fontweight='bold', color='#8B0000')

    # Plot basemap
    if africa_gdf is not None:
        africa_gdf.plot(ax=ax, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.3)

    # Size scale based on miss count
    sizes = chronic['miss_count'] * 40  # Scale factor

    # Color scale based on miss count
    colors = chronic['miss_count']

    # Plot chronic failures with size = miss count
    scatter = ax.scatter(chronic['avg_longitude'], chronic['avg_latitude'],
                        c=colors, s=sizes, cmap='YlOrRd', alpha=0.8,
                        edgecolors='black', linewidth=1, zorder=3,
                        vmin=2, vmax=chronic['miss_count'].max())

    # Add district labels for worst offenders using adjustText
    # Limit to top 20 and sort by longitude for better distribution
    top_chronic = chronic.nlargest(20, 'miss_count').sort_values('avg_longitude')
    texts = []

    for idx, row in top_chronic.iterrows():
        label = f"{row['ipc_district']} ({int(row['miss_count'])}x)"
        t = ax.text(row['avg_longitude'], row['avg_latitude'], label,
                   fontsize=7, fontweight='bold', color='#8B0000',
                   bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.9,
                            edgecolor='#8B0000', linewidth=0.5),
                   zorder=4)
        texts.append(t)

    # Use adjustText to reposition labels
    if texts:
        adjust_text(texts, ax=ax,
                   arrowprops=dict(arrowstyle='-', color='#666666', lw=0.8, alpha=0.7),
                   expand_points=(2.0, 2.0),
                   expand_text=(1.2, 1.2),
                   force_text=(0.8, 0.8),
                   force_points=(0.5, 0.5),
                   lim=500)

    ax.set_xlim(-25, 55)
    ax.set_ylim(-40, 40)
    ax.set_aspect('equal')
    ax.axis('off')

    # Add country labels
    add_country_labels(ax, fontsize=8, color='#888888')

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, aspect=20)
    cbar.set_label('Times Missed', fontsize=12)

    # Add summary statistics box
    stats_text = f"Chronic Failures Analysis\n"
    stats_text += "=" * 30 + "\n"
    stats_text += f"Districts missed 2+ times: {len(chronic)}\n"
    stats_text += f"Districts missed 3+ times: {len(chronic[chronic['miss_count'] >= 3])}\n"
    stats_text += f"Districts missed 5+ times: {len(chronic[chronic['miss_count'] >= 5])}\n"
    stats_text += f"Max times missed: {int(chronic['miss_count'].max())}\n"
    stats_text += "-" * 30 + "\n"
    stats_text += "Top 5 Most Missed:\n"
    for idx, row in chronic.nlargest(5, 'miss_count').iterrows():
        stats_text += f"  {row['ipc_district']}, {row['ipc_country'][:3]}: {int(row['miss_count'])}x\n"

    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='left',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#8B0000'))

    plt.tight_layout()

    # Save
    output_file = output_dir / f'chronic_ar_failures_h{horizon}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_file, chronic


def create_ar_failures_summary_chart(all_ar_failures, horizon, output_dir):
    """
    Create a summary bar chart showing AR failures by period and country.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    fig.suptitle(f'AR Failures Analysis Summary - h={horizon} months\n'
                 f'Stage 2 Focus Areas: {len(all_ar_failures)} Total Missed Crises',
                 fontsize=16, fontweight='bold')

    # Panel 1: AR Failures by Period
    ax1 = axes[0, 0]
    by_period = all_ar_failures.groupby('period').size().sort_index()
    bars = ax1.bar(range(len(by_period)), by_period.values, color='#D32F2F', alpha=0.8, edgecolor='black')
    ax1.set_xticks(range(len(by_period)))
    ax1.set_xticklabels(by_period.index, rotation=45, ha='right', fontsize=10)
    ax1.set_xlabel('IPC Period', fontsize=11)
    ax1.set_ylabel('Number of Missed Districts', fontsize=11)
    ax1.set_title('AR Failures by IPC Period', fontweight='bold', fontsize=12)
    ax1.tick_params(axis='y', labelsize=10)

    # Add value labels
    for bar, val in zip(bars, by_period.values):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                str(val), ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Panel 2: AR Failures by Country
    ax2 = axes[0, 1]
    by_country = all_ar_failures.groupby('ipc_country').size().sort_values(ascending=True)
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(by_country)))
    bars = ax2.barh(range(len(by_country)), by_country.values, color=colors, edgecolor='black')
    ax2.set_yticks(range(len(by_country)))
    ax2.set_yticklabels(by_country.index, fontsize=10)
    ax2.set_xlabel('Number of Missed Districts', fontsize=11)
    ax2.set_title('AR Failures by Country', fontweight='bold', fontsize=12)
    ax2.tick_params(axis='x', labelsize=10)

    # Add value labels
    for bar, val in zip(bars, by_country.values):
        ax2.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(val), ha='left', va='center', fontsize=9)

    # Panel 3: Top 20 Most Missed Districts
    ax3 = axes[1, 0]
    by_district = all_ar_failures.groupby(['ipc_country', 'ipc_district']).size().sort_values(ascending=True).tail(20)
    district_labels = [f"{d[1]}, {d[0][:3]}" for d in by_district.index]
    colors = plt.cm.OrRd(np.linspace(0.3, 0.9, len(by_district)))
    bars = ax3.barh(range(len(by_district)), by_district.values, color=colors, edgecolor='black')
    ax3.set_yticks(range(len(by_district)))
    ax3.set_yticklabels(district_labels, fontsize=9)
    ax3.set_xlabel('Times Missed', fontsize=11)
    ax3.set_title('Top 20 Most Frequently Missed Districts', fontweight='bold', fontsize=12)
    ax3.tick_params(axis='x', labelsize=10)

    for bar, val in zip(bars, by_district.values):
        ax3.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(val), ha='left', va='center', fontsize=9)

    # Panel 4: Heatmap of failures by country and period
    ax4 = axes[1, 1]
    pivot = all_ar_failures.groupby(['ipc_country', 'period']).size().unstack(fill_value=0)

    # Only show countries with at least 5 total failures
    country_totals = pivot.sum(axis=1)
    top_countries = country_totals[country_totals >= 5].index
    pivot_filtered = pivot.loc[top_countries]

    im = ax4.imshow(pivot_filtered.values, cmap='Reds', aspect='auto')
    ax4.set_xticks(range(len(pivot_filtered.columns)))
    ax4.set_xticklabels(pivot_filtered.columns, rotation=45, ha='right', fontsize=9)
    ax4.set_yticks(range(len(pivot_filtered.index)))
    ax4.set_yticklabels(pivot_filtered.index, fontsize=10)
    ax4.set_title('AR Failures Heatmap (Countries with 5+ failures)', fontweight='bold', fontsize=12)

    # Add text annotations
    for i in range(len(pivot_filtered.index)):
        for j in range(len(pivot_filtered.columns)):
            val = pivot_filtered.iloc[i, j]
            if val > 0:
                ax4.text(j, i, str(val), ha='center', va='center', fontsize=9,
                        color='white' if val > pivot_filtered.values.max()/2 else 'black')

    cbar = plt.colorbar(im, ax=ax4, shrink=0.8)
    cbar.set_label('Failures', fontsize=11)

    plt.tight_layout()

    # Save
    output_file = output_dir / f'ar_failures_summary_h{horizon}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_file


def create_ar_failures_table(all_ar_failures, horizon, output_dir):
    """
    Create a detailed CSV table of all AR failures for Stage 2 analysis.
    """
    # Full details
    full_table = all_ar_failures[['period', 'ipc_country', 'ipc_district',
                                   'avg_latitude', 'avg_longitude', 'ipc_id',
                                   'y_true', 'y_pred', 'y_pred_proba']].copy()
    full_table = full_table.sort_values(['period', 'ipc_country', 'ipc_district'])

    output_file = output_dir / f'ar_failures_detailed_h{horizon}.csv'
    full_table.to_csv(output_file, index=False)

    # Summary by district
    summary = all_ar_failures.groupby(['ipc_id', 'ipc_country', 'ipc_district',
                                       'avg_latitude', 'avg_longitude']).agg({
        'period': ['count', lambda x: ', '.join(sorted(x))]
    }).reset_index()
    summary.columns = ['ipc_id', 'ipc_country', 'ipc_district', 'avg_latitude',
                      'avg_longitude', 'times_missed', 'periods_missed']
    summary = summary.sort_values('times_missed', ascending=False)

    summary_file = output_dir / f'ar_failures_by_district_h{horizon}.csv'
    summary.to_csv(summary_file, index=False)

    return output_file, summary_file


def main():
    print("=" * 80)
    print("AR Failures Analysis - Stage 2 Focus Areas")
    print("Identifying Districts Where Baseline Model Failed to Predict Crises")
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

        # Filter to AR failures using OPTIMAL threshold (balanced P=R >= 0.6)
        # Use ar_failure_optimal instead of ar_failure (default 0.5 threshold)
        ar_failures = preds[preds['ar_failure_optimal'] == 1].copy()

        # Get optimal threshold for reporting
        opt_thresh = preds['optimal_threshold'].iloc[0] if 'optimal_threshold' in preds.columns else 'N/A'

        print(f"\nTotal predictions: {len(preds):,}")
        print(f"Optimal threshold: {opt_thresh:.3f}" if isinstance(opt_thresh, float) else f"Optimal threshold: {opt_thresh}")
        print(f"Total AR Failures (at optimal threshold): {len(ar_failures):,}")
        print(f"AR Failure Rate: {len(ar_failures)/len(preds)*100:.2f}%")

        # Get periods
        periods = sorted(ar_failures['period'].unique())
        print(f"\n{len(periods)} periods with AR failures")

        # Create map for each period
        print("\nGenerating per-period AR failure maps...")
        for period in periods:
            period_failures = ar_failures[ar_failures['period'] == period]
            period_all = preds[preds['period'] == period]

            f = create_ar_failures_map(period_failures, africa_gdf, period, horizon,
                                       h_output_dir, period_all)
            if f:
                print(f"   {period}: {len(period_failures)} failures -> {f.name}")

        # Create chronic failures map
        print("\nGenerating chronic failures map...")
        result = create_chronic_failures_map(ar_failures, africa_gdf, horizon, h_output_dir)
        if result:
            chronic_file, chronic_df = result
            print(f"   Chronic failures map: {chronic_file.name}")
            print(f"   Districts missed 2+ times: {len(chronic_df)}")

        # Create summary chart
        print("\nGenerating summary chart...")
        summary_file = create_ar_failures_summary_chart(ar_failures, horizon, h_output_dir)
        print(f"   Summary chart: {summary_file.name}")

        # Create CSV tables
        print("\nGenerating CSV tables for Stage 2...")
        detail_file, summary_csv = create_ar_failures_table(ar_failures, horizon, h_output_dir)
        print(f"   Detailed table: {detail_file.name}")
        print(f"   Summary by district: {summary_csv.name}")

    print(f"\n{'='*80}")
    print("AR Failures Analysis Complete")
    print(f"{'='*80}")
    print(f"\nOutput saved to: {OUTPUT_DIR}")
    print(f"End: {datetime.now()}")
    print("\n*** These districts are the Stage 2 focus areas ***")
    print("*** Use dynamic news signals to predict these AR failures ***")


if __name__ == '__main__':
    main()
