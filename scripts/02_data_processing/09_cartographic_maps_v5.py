"""
Publication-Grade Cartographic Maps V5 - District Level IPC Crisis Predictions
Creates proper filled polygon choropleths using spatial joins with GADM boundaries.

Key fixes over V4:
- ar_failure computed AFTER GADM aggregation (not before) to ensure visual consistency
- Missed crises = polygons where aggregated(ground_truth)=crisis AND aggregated(predicted)=no_crisis
- All 3 panels perfectly aligned
- All metrics 100% data-driven from actual prediction results

Author: District-level analysis cartographic visualization V5
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path
from datetime import datetime
import warnings
from config import BASE_DIR
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(rstr(BASE_DIR.parent.parent.parent))
GADM_DIR = BASE_DIR / 'data' / 'gadm'
RESULTS_DIR = BASE_DIR / 'results' / 'district_level' / 'stage1_baseline'
FIGURES_DIR = BASE_DIR / 'figures' / 'stage1_cartographic_v5'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Official IPC Color Scheme
COLORS = {
    'no_crisis': '#6ABD45',
    'crisis': '#E31E24',
    'no_data': '#F5F5F5',
    'background': '#E0E0E0',
    'ocean': '#B8D4E8',
    'correct_no_crisis': '#6ABD45',
    'correct_crisis': '#2166AC',
    'false_alarm': '#FDB863',
    'ar_failure': '#B2182B',
    'border': '#666666',
}

plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'

IPC_COUNTRIES = {
    'BFA': 'Burkina Faso', 'BDI': 'Burundi', 'CMR': 'Cameroon',
    'CAF': 'CAR', 'TCD': 'Chad', 'COD': 'DRC',
    'ETH': 'Ethiopia', 'KEN': 'Kenya', 'MDG': 'Madagascar', 'MWI': 'Malawi',
    'MLI': 'Mali', 'MRT': 'Mauritania', 'MOZ': 'Mozambique', 'NER': 'Niger',
    'NGA': 'Nigeria', 'RWA': 'Rwanda', 'SEN': 'Senegal', 'SOM': 'Somalia',
    'SSD': 'South Sudan', 'SDN': 'Sudan', 'SWZ': 'Eswatini', 'TGO': 'Togo',
    'UGA': 'Uganda', 'ZWE': 'Zimbabwe'
}

COUNTRY_CENTROIDS = {
    'BFA': (-1.5, 12.3, 'Burkina\nFaso'), 'BDI': (29.9, -3.4, 'Burundi'),
    'CMR': (12.4, 6.0, 'Cameroon'), 'CAF': (20.9, 6.6, 'CAR'),
    'TCD': (18.7, 15.5, 'Chad'), 'COD': (23.5, -2.5, 'DRC'),
    'ETH': (40.0, 9.0, 'Ethiopia'), 'KEN': (37.9, 0.5, 'Kenya'),
    'MDG': (46.9, -19.5, 'Madagascar'), 'MWI': (34.3, -13.3, 'Malawi'),
    'MLI': (-4.0, 17.6, 'Mali'), 'MRT': (-10.5, 20.5, 'Mauritania'),
    'MOZ': (35.5, -17.5, 'Mozambique'), 'NER': (9.5, 17.0, 'Niger'),
    'NGA': (8.0, 9.5, 'Nigeria'), 'RWA': (29.9, -2.0, 'Rwanda'),
    'SEN': (-14.5, 14.5, 'Senegal'), 'SOM': (46.0, 6.0, 'Somalia'),
    'SSD': (30.0, 7.5, 'South\nSudan'), 'SDN': (30.0, 15.5, 'Sudan'),
    'SWZ': (31.5, -26.5, 'Eswatini'), 'TGO': (1.0, 8.5, 'Togo'),
    'UGA': (32.5, 1.5, 'Uganda'), 'ZWE': (29.5, -19.5, 'Zimbabwe')
}


def load_gadm_boundaries():
    """Load GADM boundaries at finest available level."""
    print("\nLoading GADM boundaries...")
    all_boundaries = []
    adm3_countries = ['ETH', 'KEN', 'UGA', 'RWA', 'MDG', 'MWI', 'SEN', 'MLI',
                      'BFA', 'NER', 'TCD', 'CMR', 'NGA', 'MOZ', 'SDN', 'SSD', 'SOM', 'CAF']

    for iso3 in IPC_COUNTRIES.keys():
        if iso3 in adm3_countries:
            f = GADM_DIR / f'gadm41_{iso3}_3.shp'
            if f.exists():
                gdf = gpd.read_file(f)
                gdf['country_iso3'] = iso3
                all_boundaries.append(gdf)
                print(f"   {iso3}: ADM3 ({len(gdf)})")
                continue
        f = GADM_DIR / f'gadm41_{iso3}_2.shp'
        if f.exists():
            gdf = gpd.read_file(f)
            gdf['country_iso3'] = iso3
            all_boundaries.append(gdf)
            print(f"   {iso3}: ADM2 ({len(gdf)})")

    combined = pd.concat(all_boundaries, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, crs=all_boundaries[0].crs)
    print(f"   Total: {len(combined)}")
    return combined


def load_africa_basemap():
    """Load ALL African country boundaries for full continent basemap."""
    print("\nLoading full Africa basemap...")

    # Try Natural Earth data first (all 54 African countries)
    ne_file = BASE_DIR / 'data' / 'natural_earth' / 'ne_50m_admin_0_countries_africa.shp'
    if ne_file.exists():
        africa = gpd.read_file(ne_file)
        print(f"   Loaded Natural Earth: {len(africa)} African countries")
        return africa

    # Fallback to GADM (only IPC countries)
    print("   Natural Earth not found, using GADM...")
    country_files = list(GADM_DIR.glob('gadm41_*_0.shp'))
    all_countries = []
    for f in country_files:
        try:
            gdf = gpd.read_file(f)
            all_countries.append(gdf)
        except:
            pass
    combined = pd.concat(all_countries, ignore_index=True)
    combined = gpd.GeoDataFrame(combined, crs=all_countries[0].crs)
    print(f"   Loaded {len(combined)} countries (GADM fallback)")
    return combined


def load_predictions(horizon=4):
    """Load prediction results."""
    print(f"\nLoading predictions h={horizon}...")
    pred_file = RESULTS_DIR / f'predictions_h{horizon}_district_averaged.parquet'
    df = pd.read_parquet(pred_file)
    df['ipc_period_start'] = pd.to_datetime(df['ipc_period_start'])
    df['period_label'] = df['ipc_period_start'].dt.strftime('%Y-%m')
    print(f"   {len(df):,} observations, {len(df['period_label'].unique())} periods")
    return df


def compute_raw_metrics(period_data):
    """Compute metrics from RAW prediction data."""
    tn = ((period_data['y_true'] == 0) & (period_data['y_pred_optimal'] == 0)).sum()
    tp = ((period_data['y_true'] == 1) & (period_data['y_pred_optimal'] == 1)).sum()
    fp = ((period_data['y_true'] == 0) & (period_data['y_pred_optimal'] == 1)).sum()
    fn = ((period_data['y_true'] == 1) & (period_data['y_pred_optimal'] == 0)).sum()

    total = tn + tp + fp + fn
    accuracy = (tn + tp) / total * 100 if total > 0 else 0
    actual_crisis = tp + fn
    recall = tp / actual_crisis * 100 if actual_crisis > 0 else 0
    ar_rate = fn / actual_crisis * 100 if actual_crisis > 0 else 0

    return {
        'tn': tn, 'tp': tp, 'fp': fp, 'fn': fn,
        'total': total, 'accuracy': accuracy,
        'recall': recall, 'ar_rate': ar_rate,
        'n_no_crisis': tn + fp, 'n_crisis': tp + fn
    }


def spatial_join_predictions(predictions, gadm_boundaries):
    """Spatially join predictions to GADM."""
    print("\nSpatial join...")
    pred_valid = predictions.dropna(subset=['avg_latitude', 'avg_longitude']).copy()
    pred_gdf = gpd.GeoDataFrame(
        pred_valid,
        geometry=gpd.points_from_xy(pred_valid['avg_longitude'], pred_valid['avg_latitude']),
        crs='EPSG:4326'
    )
    gadm_boundaries = gadm_boundaries.to_crs('EPSG:4326')
    joined = gpd.sjoin(pred_gdf, gadm_boundaries, how='left', predicate='within')
    matched = joined['index_right'].notna().sum()
    print(f"   Matched {matched:,}/{len(joined):,} ({matched/len(joined)*100:.1f}%)")
    return joined, gadm_boundaries


def aggregate_to_gadm(period_data, gadm_gdf):
    """
    Aggregate predictions to GADM polygons.
    CRITICAL: Compute ar_failure AFTER aggregation, not before.
    UPDATED: Uses y_pred_optimal (balanced P=R threshold) instead of y_pred (default 0.5)
    """
    if 'index_right' not in period_data.columns:
        return gadm_gdf.copy()

    # Aggregate y_true and y_pred_optimal using max
    agg = period_data.groupby('index_right').agg({
        'y_true': 'max',
        'y_pred_optimal': 'max',  # Use optimal threshold predictions
        'ipc_country': 'first'
    }).reset_index()

    # CRITICAL: Compute ar_failure AFTER aggregation using optimal predictions
    # ar_failure = 1 only if aggregated ground_truth=crisis AND aggregated prediction=no_crisis
    agg['ar_failure'] = ((agg['y_true'] == 1) & (agg['y_pred_optimal'] == 0)).astype(int)

    # Compute confusion class AFTER aggregation using optimal predictions
    agg['confusion_class'] = 'TN'
    agg.loc[(agg['y_true'] == 0) & (agg['y_pred_optimal'] == 0), 'confusion_class'] = 'TN'
    agg.loc[(agg['y_true'] == 1) & (agg['y_pred_optimal'] == 1), 'confusion_class'] = 'TP'
    agg.loc[(agg['y_true'] == 0) & (agg['y_pred_optimal'] == 1), 'confusion_class'] = 'FP'
    agg.loc[(agg['y_true'] == 1) & (agg['y_pred_optimal'] == 0), 'confusion_class'] = 'FN'

    # Merge back to GADM geometries
    gadm_plot = gadm_gdf.copy().reset_index()
    gadm_plot = gadm_plot.merge(agg, left_on='index', right_on='index_right', how='left')

    return gadm_plot


def add_country_labels(ax):
    """Add country labels."""
    for iso3, (lon, lat, name) in COUNTRY_CENTROIDS.items():
        ax.text(lon, lat, name, fontsize=7, ha='center', va='center',
               fontweight='bold', color='#333333',
               bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                        alpha=0.7, edgecolor='none'))


def create_3panel_comparison(gadm_gdf, predictions_joined, period_label,
                              output_path, africa_basemap, horizon, raw_predictions):
    """Create 3-panel comparison with CORRECT ar_failure alignment."""

    period_data = predictions_joined[predictions_joined['period_label'] == period_label].copy()
    raw_period = raw_predictions[raw_predictions['period_label'] == period_label]
    metrics = compute_raw_metrics(raw_period)

    if len(period_data) == 0:
        print(f"      No data for {period_label}")
        return

    # Aggregate to GADM with CORRECT ar_failure computation
    gadm_plot = aggregate_to_gadm(period_data, gadm_gdf)

    fig, axes = plt.subplots(1, 3, figsize=(24, 10))

    # === Panel 1: Ground Truth ===
    ax = axes[0]
    ax.set_facecolor(COLORS['ocean'])
    if africa_basemap is not None:
        africa_basemap.plot(ax=ax, color=COLORS['background'], edgecolor=COLORS['border'], linewidth=0.5)

    no_data = gadm_plot[gadm_plot['y_true'].isna()]
    if len(no_data) > 0:
        no_data.plot(ax=ax, color=COLORS['no_data'], edgecolor='#CCC', linewidth=0.1)

    no_crisis = gadm_plot[gadm_plot['y_true'] == 0]
    if len(no_crisis) > 0:
        no_crisis.plot(ax=ax, color=COLORS['no_crisis'], edgecolor='#2d5016', linewidth=0.3)

    crisis = gadm_plot[gadm_plot['y_true'] == 1]
    if len(crisis) > 0:
        crisis.plot(ax=ax, color=COLORS['crisis'], edgecolor='#8b0000', linewidth=0.3)

    ax.set_xlim(-20, 55)
    ax.set_ylim(-37, 38)
    ax.set_aspect('equal')
    ax.axis('off')
    add_country_labels(ax)
    ax.set_title('Ground Truth\n(Actual IPC Status)', fontsize=12, fontweight='bold')

    legend_elements = [
        mpatches.Patch(facecolor=COLORS['no_crisis'], edgecolor='#2d5016',
                      label=f'No Crisis: {metrics["n_no_crisis"]:,}'),
        mpatches.Patch(facecolor=COLORS['crisis'], edgecolor='#8b0000',
                      label=f'Crisis: {metrics["n_crisis"]:,} ({metrics["n_crisis"]/metrics["total"]*100:.1f}%)')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9, frameon=True)

    # === Panel 2: Model Prediction ===
    ax = axes[1]
    ax.set_facecolor(COLORS['ocean'])
    if africa_basemap is not None:
        africa_basemap.plot(ax=ax, color=COLORS['background'], edgecolor=COLORS['border'], linewidth=0.5)

    no_data = gadm_plot[gadm_plot['y_pred_optimal'].isna()]
    if len(no_data) > 0:
        no_data.plot(ax=ax, color=COLORS['no_data'], edgecolor='#CCC', linewidth=0.1)

    pred_no = gadm_plot[gadm_plot['y_pred_optimal'] == 0]
    if len(pred_no) > 0:
        pred_no.plot(ax=ax, color=COLORS['no_crisis'], edgecolor='#2d5016', linewidth=0.3)

    pred_yes = gadm_plot[gadm_plot['y_pred_optimal'] == 1]
    if len(pred_yes) > 0:
        pred_yes.plot(ax=ax, color=COLORS['crisis'], edgecolor='#8b0000', linewidth=0.3)

    ax.set_xlim(-20, 55)
    ax.set_ylim(-37, 38)
    ax.set_aspect('equal')
    ax.axis('off')
    add_country_labels(ax)
    ax.set_title('Model Prediction', fontsize=12, fontweight='bold')

    n_pred_no = metrics['tn'] + metrics['fn']
    n_pred_yes = metrics['tp'] + metrics['fp']
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['no_crisis'], edgecolor='#2d5016',
                      label=f'Pred No Crisis: {n_pred_no:,}'),
        mpatches.Patch(facecolor=COLORS['crisis'], edgecolor='#8b0000',
                      label=f'Pred Crisis: {n_pred_yes:,}')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9, frameon=True)

    # === Panel 3: Missed Crises ===
    ax = axes[2]
    ax.set_facecolor(COLORS['ocean'])
    if africa_basemap is not None:
        africa_basemap.plot(ax=ax, color=COLORS['background'], edgecolor=COLORS['border'], linewidth=0.5)

    # Other districts (not missed)
    other = gadm_plot[(gadm_plot['ar_failure'].notna()) & (gadm_plot['ar_failure'] == 0)]
    if len(other) > 0:
        other.plot(ax=ax, color='#D0D0D0', edgecolor='#999', linewidth=0.2)

    # AR Failures (missed crises) - computed AFTER aggregation
    failures = gadm_plot[gadm_plot['ar_failure'] == 1]
    n_gadm_failures = len(failures)
    if len(failures) > 0:
        failures.plot(ax=ax, color=COLORS['ar_failure'], edgecolor='black', linewidth=0.8)

    ax.set_xlim(-20, 55)
    ax.set_ylim(-37, 38)
    ax.set_aspect('equal')
    ax.axis('off')
    add_country_labels(ax)

    ax.set_title(f'Missed Crises (AR Failures)\n{metrics["fn"]:,} missed ({metrics["ar_rate"]:.1f}%)',
                fontsize=12, fontweight='bold', color=COLORS['ar_failure'])

    legend_elements = [
        mpatches.Patch(facecolor=COLORS['ar_failure'], edgecolor='black',
                      label=f'MISSED: {metrics["fn"]:,} (raw)'),
        mpatches.Patch(facecolor='#D0D0D0', edgecolor='#999',
                      label=f'Other')
    ]
    ax.legend(handles=legend_elements, loc='lower left', fontsize=9, frameon=True)

    # Main title
    period_date = pd.to_datetime(period_label + '-01')
    period_name = period_date.strftime('%B %Y')

    fig.suptitle(
        f'IPC Food Security Crisis Prediction - {period_name}\n'
        f'h={horizon} months | Districts: {metrics["total"]:,} | '
        f'Accuracy: {metrics["accuracy"]:.1f}% | Recall: {metrics["recall"]:.1f}% | '
        f'AR Failures: {metrics["fn"]:,} ({metrics["ar_rate"]:.1f}%)',
        fontsize=14, fontweight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"      Saved: {output_path}")


def create_confusion_choropleth(gadm_gdf, predictions_joined, period_label,
                                 output_path, africa_basemap, horizon, raw_predictions):
    """Create confusion matrix choropleth."""

    period_data = predictions_joined[predictions_joined['period_label'] == period_label].copy()
    raw_period = raw_predictions[raw_predictions['period_label'] == period_label]
    metrics = compute_raw_metrics(raw_period)

    if len(period_data) == 0:
        return

    gadm_plot = aggregate_to_gadm(period_data, gadm_gdf)

    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.set_facecolor(COLORS['ocean'])

    if africa_basemap is not None:
        africa_basemap.plot(ax=ax, color=COLORS['background'], edgecolor=COLORS['border'], linewidth=0.5)

    confusion_colors = {
        'TN': COLORS['correct_no_crisis'],
        'TP': COLORS['correct_crisis'],
        'FP': COLORS['false_alarm'],
        'FN': COLORS['ar_failure']
    }

    # No data
    no_data = gadm_plot[gadm_plot['confusion_class'].isna()]
    if len(no_data) > 0:
        no_data.plot(ax=ax, color=COLORS['no_data'], edgecolor='#CCC', linewidth=0.1)

    # Plot each confusion class
    for cls, color in confusion_colors.items():
        subset = gadm_plot[gadm_plot['confusion_class'] == cls]
        if len(subset) > 0:
            subset.plot(ax=ax, color=color, edgecolor='#444', linewidth=0.3)

    ax.set_xlim(-20, 55)
    ax.set_ylim(-37, 38)
    ax.set_aspect('equal')
    ax.axis('off')
    add_country_labels(ax)

    period_date = pd.to_datetime(period_label + '-01')
    period_name = period_date.strftime('%B %Y')

    ax.set_title(f'Model Performance - Prediction Outcomes\n{period_name} (h={horizon} months)',
                 fontsize=14, fontweight='bold', pad=15)

    legend_elements = [
        mpatches.Patch(facecolor=confusion_colors['TN'], edgecolor='#444',
                      label=f'True Negative: {metrics["tn"]:,}'),
        mpatches.Patch(facecolor=confusion_colors['TP'], edgecolor='#444',
                      label=f'True Positive: {metrics["tp"]:,}'),
        mpatches.Patch(facecolor=confusion_colors['FP'], edgecolor='#444',
                      label=f'False Positive: {metrics["fp"]:,}'),
        mpatches.Patch(facecolor=confusion_colors['FN'], edgecolor='#444',
                      label=f'False Negative (MISSED): {metrics["fn"]:,}'),
        mpatches.Patch(facecolor=COLORS['no_data'], edgecolor='#CCC', label='No IPC Data')
    ]

    ax.legend(handles=legend_elements, loc='lower left', fontsize=10,
              frameon=True, fancybox=True, shadow=True,
              title='Prediction Outcome', title_fontsize=11)

    stats_text = (f'Districts: {metrics["total"]:,}\n'
                  f'Accuracy: {metrics["accuracy"]:.1f}%\n'
                  f'Recall: {metrics["recall"]:.1f}%\n'
                  f'AR Failures: {metrics["fn"]:,} ({metrics["ar_rate"]:.1f}%)')

    ax.text(0.98, 0.98, stats_text, transform=ax.transAxes, fontsize=11,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.9))

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"      Saved: {output_path}")


def create_ground_truth_choropleth(gadm_gdf, predictions_joined, period_label,
                                    output_path, africa_basemap, horizon, raw_predictions):
    """Create ground truth choropleth."""

    period_data = predictions_joined[predictions_joined['period_label'] == period_label].copy()
    raw_period = raw_predictions[raw_predictions['period_label'] == period_label]
    metrics = compute_raw_metrics(raw_period)

    if len(period_data) == 0:
        return

    gadm_plot = aggregate_to_gadm(period_data, gadm_gdf)

    fig, ax = plt.subplots(1, 1, figsize=(16, 16))
    ax.set_facecolor(COLORS['ocean'])

    if africa_basemap is not None:
        africa_basemap.plot(ax=ax, color=COLORS['background'], edgecolor=COLORS['border'], linewidth=0.5)

    no_data = gadm_plot[gadm_plot['y_true'].isna()]
    if len(no_data) > 0:
        no_data.plot(ax=ax, color=COLORS['no_data'], edgecolor='#CCC', linewidth=0.1)

    no_crisis = gadm_plot[gadm_plot['y_true'] == 0]
    if len(no_crisis) > 0:
        no_crisis.plot(ax=ax, color=COLORS['no_crisis'], edgecolor='#2d5016', linewidth=0.3)

    crisis = gadm_plot[gadm_plot['y_true'] == 1]
    if len(crisis) > 0:
        crisis.plot(ax=ax, color=COLORS['crisis'], edgecolor='#8b0000', linewidth=0.3)

    ax.set_xlim(-20, 55)
    ax.set_ylim(-37, 38)
    ax.set_aspect('equal')
    ax.axis('off')
    add_country_labels(ax)

    period_date = pd.to_datetime(period_label + '-01')
    period_name = period_date.strftime('%B %Y')

    ax.set_title(f'IPC Food Security Classification - Ground Truth\n{period_name}',
                 fontsize=14, fontweight='bold', pad=15)

    crisis_pct = metrics['n_crisis'] / metrics['total'] * 100
    legend_elements = [
        mpatches.Patch(facecolor=COLORS['no_crisis'], edgecolor='#2d5016',
                      label=f'No Crisis (IPC 1-2): {metrics["n_no_crisis"]:,}'),
        mpatches.Patch(facecolor=COLORS['crisis'], edgecolor='#8b0000',
                      label=f'Crisis (IPC 3+): {metrics["n_crisis"]:,} ({crisis_pct:.1f}%)'),
        mpatches.Patch(facecolor=COLORS['no_data'], edgecolor='#CCC', label='No IPC Data')
    ]

    ax.legend(handles=legend_elements, loc='lower left', fontsize=11,
              frameon=True, fancybox=True, shadow=True,
              title='Food Security Status', title_fontsize=12)

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"      Saved: {output_path}")


def main():
    print("=" * 80)
    print("Cartographic Maps V5 - Correct AR Failure Alignment")
    print("=" * 80)
    print(f"Start: {datetime.now()}\n")

    gadm_boundaries = load_gadm_boundaries()
    africa_basemap = load_africa_basemap()

    for horizon in [4, 8, 12]:
        print(f"\n{'='*60}")
        print(f"h={horizon} months")
        print(f"{'='*60}")

        try:
            raw_predictions = load_predictions(horizon)

            predictions_joined, gadm_matched = spatial_join_predictions(
                raw_predictions, gadm_boundaries
            )

            horizon_dir = FIGURES_DIR / f'h{horizon}'
            horizon_dir.mkdir(parents=True, exist_ok=True)

            periods = sorted(predictions_joined['period_label'].unique())
            print(f"\n   {len(periods)} periods...")

            for period in periods:
                print(f"\n   {period}")

                create_3panel_comparison(
                    gadm_matched, predictions_joined, period,
                    horizon_dir / f'comparison_{period}_h{horizon}.png',
                    africa_basemap, horizon, raw_predictions
                )

                create_confusion_choropleth(
                    gadm_matched, predictions_joined, period,
                    horizon_dir / f'confusion_{period}_h{horizon}.png',
                    africa_basemap, horizon, raw_predictions
                )

                create_ground_truth_choropleth(
                    gadm_matched, predictions_joined, period,
                    horizon_dir / f'ground_truth_{period}_h{horizon}.png',
                    africa_basemap, horizon, raw_predictions
                )

        except Exception as e:
            print(f"   Error: {e}")
            import traceback
            traceback.print_exc()

    print(f"\n{'='*80}")
    print("Complete")
    print(f"{'='*80}")
    print(f"\nSaved to: {FIGURES_DIR}")
    print(f"End: {datetime.now()}")


if __name__ == "__main__":
    main()
