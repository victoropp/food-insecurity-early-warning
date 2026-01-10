"""
11_crisis_onset_analysis.py - Crisis Onset/Transition Analysis

Visualizes districts that transitioned from non-crisis (IPC ≤ 2) to crisis (IPC ≥ 3).
These are the cases the AR model cannot predict because it relies on persistence.

Stage 2 Goal: Use dynamic news signals to predict these transitions.

Transition Types:
- IPC 1 → 3+ : Sudden onset from minimal food insecurity
- IPC 2 → 3+ : Escalation from stressed to crisis

Author: Claude
Date: 2025-11-30
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
OUTPUT_DIR = BASE_DIR / 'figures' / 'stage2_crisis_onset'
NE_FILE = BASE_DIR / 'data' / 'natural_earth' / 'ne_50m_admin_0_countries_africa.shp'

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
    'TGO': ('Togo', 1.0, 8.5),
    'UGA': ('Uganda', 32.5, 1.5),
    'ZWE': ('Zimbabwe', 29.5, -19.5),
}


def add_country_labels(ax, fontsize=7, color='#666666'):
    """Add country name labels to the map."""
    for code, (name, lon, lat) in COUNTRY_LABELS.items():
        ax.annotate(name, xy=(lon, lat), fontsize=fontsize, ha='center', va='center',
                   color=color, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.6, edgecolor='none'))


def load_africa_basemap():
    """Load Africa basemap for context."""
    if NE_FILE.exists():
        return gpd.read_file(NE_FILE)
    return None


def load_predictions(horizon):
    """Load predictions for a specific horizon."""
    pred_file = PRED_DIR / f'predictions_h{horizon}_district_averaged.parquet'
    if not pred_file.exists():
        return None
    df = pd.read_parquet(pred_file)
    df['period'] = pd.to_datetime(df['ipc_period_start']).dt.strftime('%Y-%m')
    return df


def identify_crisis_onsets(df):
    """
    Identify crisis onset cases:
    - y_true = 1 (actual crisis at time t+δ)
    - Lt ≤ 2 (previous IPC was non-crisis)

    These are transitions from non-crisis to crisis.
    """
    # Crisis onsets: previous IPC ≤ 2, current actual IPC ≥ 3
    onsets = df[(df['y_true'] == 1) & (df['Lt'] <= 2)].copy()

    # Classify transition type
    onsets['transition_type'] = onsets['Lt'].map({
        1.0: 'IPC1→Crisis (Sudden)',
        2.0: 'IPC2→Crisis (Escalation)'
    })

    # Also identify if AR model caught it or missed it
    onsets['ar_caught'] = (onsets['y_pred_optimal'] == 1)
    onsets['ar_missed'] = (onsets['y_pred_optimal'] == 0)

    return onsets


def create_crisis_onset_map(onsets, africa_gdf, period, horizon, output_dir, all_data=None):
    """
    Create a map showing crisis onset cases for a specific period.
    Color-coded by transition type and whether AR caught it.
    """
    if len(onsets) == 0:
        return None

    fig, ax = plt.subplots(figsize=(16, 14))

    # Count by type
    sudden = onsets[onsets['Lt'] == 1]
    escalation = onsets[onsets['Lt'] == 2]
    ar_missed = onsets[onsets['ar_missed']]
    ar_caught = onsets[onsets['ar_caught']]

    ax.set_title(f'Crisis Onset Cases - {period} (h={horizon} months)\n'
                 f'Districts Transitioning from Non-Crisis to Crisis\n'
                 f'Total: {len(onsets)} | Sudden (IPC1→3+): {len(sudden)} | Escalation (IPC2→3+): {len(escalation)}',
                 fontsize=16, fontweight='bold')

    # Plot basemap
    if africa_gdf is not None:
        africa_gdf.plot(ax=ax, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.3)

    # Plot other districts faintly
    if all_data is not None:
        other = all_data[~all_data.index.isin(onsets.index)]
        ax.scatter(other['avg_longitude'], other['avg_latitude'],
                   c='#E0E0E0', s=8, alpha=0.2, zorder=1)

    # Colors for transition types
    colors = {
        'sudden_missed': '#8B0000',    # Dark red - sudden onset, AR missed
        'sudden_caught': '#FF6B6B',    # Light red - sudden onset, AR caught
        'escalation_missed': '#00008B', # Dark blue - escalation, AR missed
        'escalation_caught': '#6B6BFF'  # Light blue - escalation, AR caught
    }

    # Plot sudden onsets (IPC 1 → 3+) - triangles
    sudden_missed = sudden[sudden['ar_missed']]
    sudden_caught = sudden[sudden['ar_caught']]

    if len(sudden_missed) > 0:
        ax.scatter(sudden_missed['avg_longitude'], sudden_missed['avg_latitude'],
                   c=colors['sudden_missed'], s=120, marker='^', alpha=0.9,
                   edgecolors='black', linewidth=1.5, zorder=4,
                   label=f'IPC1→Crisis, AR Missed: {len(sudden_missed)}')

    if len(sudden_caught) > 0:
        ax.scatter(sudden_caught['avg_longitude'], sudden_caught['avg_latitude'],
                   c=colors['sudden_caught'], s=80, marker='^', alpha=0.8,
                   edgecolors='black', linewidth=1, zorder=3,
                   label=f'IPC1→Crisis, AR Caught: {len(sudden_caught)}')

    # Plot escalations (IPC 2 → 3+) - circles
    escalation_missed = escalation[escalation['ar_missed']]
    escalation_caught = escalation[escalation['ar_caught']]

    if len(escalation_missed) > 0:
        ax.scatter(escalation_missed['avg_longitude'], escalation_missed['avg_latitude'],
                   c=colors['escalation_missed'], s=120, marker='o', alpha=0.9,
                   edgecolors='black', linewidth=1.5, zorder=4,
                   label=f'IPC2→Crisis, AR Missed: {len(escalation_missed)}')

    if len(escalation_caught) > 0:
        ax.scatter(escalation_caught['avg_longitude'], escalation_caught['avg_latitude'],
                   c=colors['escalation_caught'], s=80, marker='o', alpha=0.8,
                   edgecolors='black', linewidth=1, zorder=3,
                   label=f'IPC2→Crisis, AR Caught: {len(escalation_caught)}')

    # Add labels for AR-missed cases using adjustText to avoid overlaps
    # Label ALL sudden onset misses (critical - 100% miss rate)
    # Label limited escalation misses to reduce clutter
    texts = []

    # Sudden onset misses - always label (they're few and critical)
    for idx, row in sudden_missed.iterrows():
        t = ax.text(row['avg_longitude'], row['avg_latitude'], row['ipc_district'],
                   fontsize=8, fontweight='bold', color='#8B0000',
                   bbox=dict(boxstyle='round,pad=0.15', facecolor='white', alpha=0.9, edgecolor='#8B0000', linewidth=0.5),
                   zorder=5)
        texts.append(t)

    # Escalation misses - limit to 10 max per map to reduce clutter
    # Sort by longitude to spread labels geographically
    escalation_sorted = escalation_missed.sort_values('avg_longitude')
    escalation_to_label = escalation_sorted.head(10)
    for idx, row in escalation_to_label.iterrows():
        t = ax.text(row['avg_longitude'], row['avg_latitude'], row['ipc_district'],
                   fontsize=6, fontweight='bold', color='#00008B',
                   bbox=dict(boxstyle='round,pad=0.12', facecolor='white', alpha=0.9, edgecolor='#00008B', linewidth=0.5),
                   zorder=5)
        texts.append(t)

    # Use adjustText to automatically reposition labels to avoid overlaps
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
    add_country_labels(ax, fontsize=8, color='#888888')

    # Legend
    ax.legend(loc='lower left', fontsize=11, title='Transition Type & AR Performance', title_fontsize=11)

    # Info box
    info_text = f"Crisis Onset Analysis\n"
    info_text += "=" * 25 + "\n"
    info_text += f"Total onsets: {len(onsets)}\n"
    info_text += f"  Sudden (IPC1→3+): {len(sudden)}\n"
    info_text += f"  Escalation (IPC2→3+): {len(escalation)}\n"
    info_text += "-" * 25 + "\n"
    info_text += f"AR Model Performance:\n"
    info_text += f"  Caught: {len(ar_caught)} ({len(ar_caught)/len(onsets)*100:.1f}%)\n"
    info_text += f"  Missed: {len(ar_missed)} ({len(ar_missed)/len(onsets)*100:.1f}%)\n"

    ax.text(0.98, 0.98, info_text, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor='#333'))

    plt.tight_layout()

    output_file = output_dir / f'crisis_onset_{period}_h{horizon}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_file


def create_onset_summary_chart(all_onsets, horizon, output_dir):
    """
    Create summary charts for crisis onset analysis.
    """
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # Calculate totals
    total = len(all_onsets)
    sudden = len(all_onsets[all_onsets['Lt'] == 1])
    escalation = len(all_onsets[all_onsets['Lt'] == 2])
    ar_missed = len(all_onsets[all_onsets['ar_missed']])
    ar_caught = len(all_onsets[all_onsets['ar_caught']])

    fig.suptitle(f'Crisis Onset Analysis Summary - h={horizon} months\n'
                 f'Total Transitions to Crisis: {total} | AR Missed: {ar_missed} ({ar_missed/total*100:.1f}%)',
                 fontsize=16, fontweight='bold')

    # Panel 1: Onsets by period, stacked by transition type
    ax1 = axes[0, 0]
    by_period_type = all_onsets.groupby(['period', 'Lt']).size().unstack(fill_value=0)
    by_period_type.columns = ['IPC1→Crisis', 'IPC2→Crisis']
    by_period_type.plot(kind='bar', stacked=True, ax=ax1, color=['#D32F2F', '#1976D2'], edgecolor='black')
    ax1.set_xlabel('IPC Period', fontsize=11)
    ax1.set_ylabel('Number of Crisis Onsets', fontsize=11)
    ax1.set_title('Crisis Onsets by Period & Transition Type', fontweight='bold', fontsize=12)
    ax1.legend(title='Transition', fontsize=10, title_fontsize=10)
    ax1.tick_params(axis='x', rotation=45, labelsize=10)
    ax1.tick_params(axis='y', labelsize=10)

    # Add totals on bars
    for i, period in enumerate(by_period_type.index):
        total_val = by_period_type.loc[period].sum()
        ax1.text(i, total_val + 1, str(int(total_val)), ha='center', fontsize=10, fontweight='bold')

    # Panel 2: AR performance on onset cases
    ax2 = axes[0, 1]
    ar_perf = all_onsets.groupby(['Lt', 'ar_missed']).size().unstack(fill_value=0)
    ar_perf.index = ['IPC1→Crisis', 'IPC2→Crisis']
    ar_perf.columns = ['AR Caught', 'AR Missed']
    ar_perf.plot(kind='bar', ax=ax2, color=['#4CAF50', '#D32F2F'], edgecolor='black')
    ax2.set_xlabel('Transition Type', fontsize=11)
    ax2.set_ylabel('Number of Cases', fontsize=11)
    ax2.set_title('AR Model Performance on Crisis Onsets', fontweight='bold', fontsize=12)
    ax2.legend(title='Result', fontsize=10, title_fontsize=10)
    ax2.tick_params(axis='x', rotation=0, labelsize=10)
    ax2.tick_params(axis='y', labelsize=10)

    # Add values on bars
    for container in ax2.containers:
        ax2.bar_label(container, fontsize=10, fontweight='bold')

    # Panel 3: Countries with most onsets
    ax3 = axes[1, 0]
    by_country = all_onsets.groupby('ipc_country').size().sort_values(ascending=True)
    colors = plt.cm.OrRd(np.linspace(0.3, 0.9, len(by_country)))
    bars = ax3.barh(range(len(by_country)), by_country.values, color=colors, edgecolor='black')
    ax3.set_yticks(range(len(by_country)))
    ax3.set_yticklabels(by_country.index, fontsize=10)
    ax3.set_xlabel('Number of Crisis Onsets', fontsize=11)
    ax3.set_title('Crisis Onsets by Country', fontweight='bold', fontsize=12)
    ax3.tick_params(axis='x', labelsize=10)
    for bar, val in zip(bars, by_country.values):
        ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                str(val), ha='left', va='center', fontsize=9)

    # Panel 4: Districts with multiple onsets (chronic transition areas)
    ax4 = axes[1, 1]
    by_district = all_onsets.groupby(['ipc_country', 'ipc_district']).size().sort_values(ascending=True).tail(20)
    district_labels = [f"{d[1]}, {d[0][:3]}" for d in by_district.index]
    colors = plt.cm.Reds(np.linspace(0.3, 0.9, len(by_district)))
    bars = ax4.barh(range(len(by_district)), by_district.values, color=colors, edgecolor='black')
    ax4.set_yticks(range(len(by_district)))
    ax4.set_yticklabels(district_labels, fontsize=9)
    ax4.set_xlabel('Number of Crisis Onsets', fontsize=11)
    ax4.set_title('Top 20 Districts with Repeated Crisis Onsets\n(Chronic Transition Zones)', fontweight='bold', fontsize=12)
    ax4.tick_params(axis='x', labelsize=10)
    for bar, val in zip(bars, by_district.values):
        ax4.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                str(val), ha='left', va='center', fontsize=9)

    plt.tight_layout()

    output_file = output_dir / f'crisis_onset_summary_h{horizon}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_file


def create_ar_performance_on_onsets(all_onsets, horizon, output_dir):
    """
    Create detailed analysis of AR model performance on crisis onset cases.
    """
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    fig.suptitle(f'AR Model Performance on Crisis Onset Detection - h={horizon} months\n'
                 f'Key Question: Which onset types does the AR model miss most?',
                 fontsize=16, fontweight='bold')

    # Panel 1: Overall AR performance pie
    ax1 = axes[0]
    ar_missed = len(all_onsets[all_onsets['ar_missed']])
    ar_caught = len(all_onsets[all_onsets['ar_caught']])
    sizes = [ar_caught, ar_missed]
    labels = [f'AR Caught\n{ar_caught} ({ar_caught/(ar_caught+ar_missed)*100:.1f}%)',
              f'AR Missed\n{ar_missed} ({ar_missed/(ar_caught+ar_missed)*100:.1f}%)']
    colors = ['#4CAF50', '#D32F2F']
    ax1.pie(sizes, labels=labels, colors=colors, autopct='', startangle=90,
            wedgeprops=dict(edgecolor='black', linewidth=1.5),
            textprops={'fontsize': 11})
    ax1.set_title('Overall AR Performance\non Crisis Onsets', fontweight='bold', fontsize=12)

    # Panel 2: Performance by transition type
    ax2 = axes[1]
    perf_by_type = all_onsets.groupby('Lt').agg({
        'ar_caught': 'sum',
        'ar_missed': 'sum'
    })
    perf_by_type['total'] = perf_by_type['ar_caught'] + perf_by_type['ar_missed']
    perf_by_type['miss_rate'] = perf_by_type['ar_missed'] / perf_by_type['total'] * 100

    x = [0, 1]
    width = 0.35
    ax2.bar([i - width/2 for i in x], perf_by_type['ar_caught'], width,
            label='AR Caught', color='#4CAF50', edgecolor='black')
    ax2.bar([i + width/2 for i in x], perf_by_type['ar_missed'], width,
            label='AR Missed', color='#D32F2F', edgecolor='black')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['IPC1→Crisis\n(Sudden Onset)', 'IPC2→Crisis\n(Escalation)'], fontsize=11)
    ax2.set_ylabel('Number of Cases', fontsize=11)
    ax2.set_title('AR Performance by Transition Type', fontweight='bold', fontsize=12)
    ax2.legend(fontsize=10)
    ax2.tick_params(axis='y', labelsize=10)

    # Add miss rate labels
    for i, (idx, row) in enumerate(perf_by_type.iterrows()):
        ax2.text(i, row['total'] + 2, f'Miss Rate: {row["miss_rate"]:.1f}%',
                ha='center', fontsize=11, fontweight='bold', color='#D32F2F')

    # Panel 3: Miss rate by country (for countries with 10+ onsets)
    ax3 = axes[2]
    by_country = all_onsets.groupby('ipc_country').agg({
        'ar_caught': 'sum',
        'ar_missed': 'sum'
    })
    by_country['total'] = by_country['ar_caught'] + by_country['ar_missed']
    by_country['miss_rate'] = by_country['ar_missed'] / by_country['total'] * 100
    by_country = by_country[by_country['total'] >= 10].sort_values('miss_rate', ascending=True)

    colors = plt.cm.RdYlGn_r(by_country['miss_rate'] / 100)
    bars = ax3.barh(range(len(by_country)), by_country['miss_rate'], color=colors, edgecolor='black')
    ax3.set_yticks(range(len(by_country)))
    ax3.set_yticklabels(by_country.index, fontsize=10)
    ax3.set_xlabel('AR Miss Rate (%)', fontsize=11)
    ax3.set_title('AR Miss Rate by Country\n(Countries with 10+ onsets)', fontweight='bold', fontsize=12)
    ax3.axvline(x=50, color='gray', linestyle='--', alpha=0.5)
    ax3.tick_params(axis='x', labelsize=10)

    for bar, (idx, row) in zip(bars, by_country.iterrows()):
        ax3.text(bar.get_width() + 1, bar.get_y() + bar.get_height()/2,
                f'{row["miss_rate"]:.0f}% ({int(row["ar_missed"])}/{int(row["total"])})',
                ha='left', va='center', fontsize=9)

    plt.tight_layout()

    output_file = output_dir / f'ar_performance_on_onsets_h{horizon}.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_file


def create_onset_csv(all_onsets, horizon, output_dir):
    """Create CSV tables for crisis onset analysis."""
    # Detailed table
    detail = all_onsets[['period', 'ipc_country', 'ipc_district', 'Lt', 'ipc_value',
                         'avg_latitude', 'avg_longitude', 'y_pred_proba',
                         'ar_caught', 'ar_missed', 'transition_type']].copy()
    detail = detail.sort_values(['period', 'ipc_country', 'ipc_district'])
    detail.to_csv(output_dir / f'crisis_onsets_detailed_h{horizon}.csv', index=False)

    # Summary by district
    summary = all_onsets.groupby(['ipc_country', 'ipc_district']).agg({
        'avg_latitude': 'first',
        'avg_longitude': 'first',
        'period': 'count',
        'ar_missed': 'sum',
        'ar_caught': 'sum'
    }).reset_index()
    summary.columns = ['country', 'district', 'lat', 'lon', 'total_onsets', 'ar_missed', 'ar_caught']
    summary['miss_rate'] = summary['ar_missed'] / summary['total_onsets'] * 100
    summary = summary.sort_values('total_onsets', ascending=False)
    summary.to_csv(output_dir / f'crisis_onsets_by_district_h{horizon}.csv', index=False)

    return detail, summary


def main():
    print("=" * 80)
    print("Crisis Onset Analysis - Transitions from Non-Crisis to Crisis")
    print("Stage 2 Target: Districts where AR model fails due to lack of persistence signal")
    print("=" * 80)
    print(f"Start: {datetime.now()}\n")

    africa_gdf = load_africa_basemap()
    if africa_gdf is not None:
        print(f"Loaded Africa basemap: {len(africa_gdf)} countries")

    for horizon in [4, 8, 12]:
        print(f"\n{'='*60}")
        print(f"h={horizon} months")
        print(f"{'='*60}")

        h_output_dir = OUTPUT_DIR / f'h{horizon}'
        h_output_dir.mkdir(parents=True, exist_ok=True)

        preds = load_predictions(horizon)
        if preds is None:
            continue

        # Identify crisis onset cases
        onsets = identify_crisis_onsets(preds)

        print(f"\nTotal predictions: {len(preds):,}")
        print(f"Crisis onset cases (IPC<=2 -> IPC>=3): {len(onsets):,}")
        print(f"  - Sudden (IPC1->Crisis): {len(onsets[onsets['Lt']==1]):,}")
        print(f"  - Escalation (IPC2->Crisis): {len(onsets[onsets['Lt']==2]):,}")
        print(f"AR Performance on onsets:")
        print(f"  - Caught: {onsets['ar_caught'].sum():,}")
        print(f"  - Missed: {onsets['ar_missed'].sum():,}")

        # Create per-period maps
        periods = sorted(onsets['period'].unique())
        print(f"\n{len(periods)} periods with crisis onsets")

        print("\nGenerating per-period onset maps...")
        for period in periods:
            period_onsets = onsets[onsets['period'] == period]
            period_all = preds[preds['period'] == period]

            f = create_crisis_onset_map(period_onsets, africa_gdf, period, horizon,
                                        h_output_dir, period_all)
            if f:
                print(f"   {period}: {len(period_onsets)} onsets -> {f.name}")

        # Create summary chart
        print("\nGenerating summary charts...")
        f1 = create_onset_summary_chart(onsets, horizon, h_output_dir)
        print(f"   Summary chart: {f1.name}")

        f2 = create_ar_performance_on_onsets(onsets, horizon, h_output_dir)
        print(f"   AR performance chart: {f2.name}")

        # Create CSV tables
        print("\nGenerating CSV tables...")
        create_onset_csv(onsets, horizon, h_output_dir)
        print(f"   CSV tables saved")

    print(f"\n{'='*80}")
    print("Crisis Onset Analysis Complete")
    print(f"{'='*80}")
    print(f"\nOutput saved to: {OUTPUT_DIR}")
    print(f"End: {datetime.now()}")
    print("\n*** These transition cases are the primary Stage 2 target ***")
    print("*** Dynamic news signals should predict IPC 1/2 -> 3+ transitions ***")


if __name__ == '__main__':
    main()
