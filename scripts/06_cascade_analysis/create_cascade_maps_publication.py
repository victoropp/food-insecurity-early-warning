"""
Publication-Grade Choropleth Maps for Cascade Ensemble Story
============================================================

Creates state-of-the-art geographic visualizations showing:
1. Key saves by IPC period (temporal breakdown)
2. Overall key saves summary across all periods
3. Geographic distribution of cascade ensemble improvements

Uses official IPC boundaries and GADM country boundaries.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap, BoundaryNorm
from pathlib import Path
from shapely.geometry import Point
import warnings
from config import BASE_DIR
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(str(BASE_DIR))
IPC_SHAPEFILE_DIR = Path(r"C:\GDELT_Africa_Extract\Data\ipc_shapefiles")
AFRICA_BASEMAP_FILE = Path(r"C:\GDELT_Africa_Extract\data\natural_earth\ne_50m_admin_0_countries_africa.shp")
RESULTS_DIR = BASE_DIR / "RESULTS"
FIGURES_DIR = BASE_DIR / "FIGURES"

# Africa extent (from working examples in codebase)
AFRICA_EXTENT = [-20, 55, -35, 40]  # [west, east, south, north]

# Publication-quality settings
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 13
plt.rcParams['legend.fontsize'] = 9
plt.rcParams['figure.dpi'] = 150
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Color schemes
CONFUSION_COLORS = {
    'TP': '#2196F3',    # Blue - Correct crisis prediction
    'TN': '#4CAF50',    # Green - Correct no-crisis
    'FP': '#FF9800',    # Orange - False alarm
    'FN': '#E53935',    # Red - Missed crisis
    'no_data': '#EEEEEE'  # Light gray
}

KEY_SAVE_COLORS = {
    'key_save': '#FFD700',     # Gold - AR missed, Ensemble caught
    'still_missed': '#8B0000',  # Dark red - Both missed
    'ar_caught': '#90EE90',     # Light green - AR caught
    'no_crisis': '#F5F5F5'      # Very light gray
}

# Country name mapping (ISO3 to full names)
COUNTRY_NAMES = {
    'BDI': 'Burundi', 'BFA': 'Burkina Faso', 'CAF': 'CAR',
    'CMR': 'Cameroon', 'COD': 'DRC', 'ETH': 'Ethiopia',
    'KEN': 'Kenya', 'LSO': 'Lesotho', 'MDG': 'Madagascar',
    'MLI': 'Mali', 'MOZ': 'Mozambique', 'MRT': 'Mauritania',
    'MWI': 'Malawi', 'NER': 'Niger', 'NGA': 'Nigeria',
    'RWA': 'Rwanda', 'SDN': 'Sudan', 'SEN': 'Senegal',
    'SOM': 'Somalia', 'SSD': 'South Sudan', 'SWZ': 'Eswatini',
    'TCD': 'Chad', 'UGA': 'Uganda', 'ZWE': 'Zimbabwe'
}

def load_africa_basemap():
    """Load complete Africa basemap with all countries including North Africa."""
    print("  - Loading Africa basemap...")

    if not AFRICA_BASEMAP_FILE.exists():
        print(f"    Warning: Africa basemap not found at {AFRICA_BASEMAP_FILE}")
        return None

    africa = gpd.read_file(AFRICA_BASEMAP_FILE)

    # Ensure CRS is WGS84
    if africa.crs.to_epsg() != 4326:
        africa = africa.to_crs('EPSG:4326')

    print(f"    Loaded {len(africa)} African countries")
    return africa

def load_ipc_boundaries():
    """Load IPC district boundaries."""
    ipc_path = IPC_SHAPEFILE_DIR / 'ipc_africa_all_boundaries.geojson'

    if ipc_path.exists():
        ipc_gdf = gpd.read_file(ipc_path)
        if ipc_gdf.crs.to_epsg() != 4326:
            ipc_gdf = ipc_gdf.to_crs('EPSG:4326')
        return ipc_gdf

    print(f"Error: IPC boundaries not found at {ipc_path}")
    return None

def predictions_to_geopoints(predictions_df):
    """Convert predictions DataFrame to GeoDataFrame with points."""
    geometry = [Point(xy) for xy in zip(
        predictions_df['avg_longitude'],
        predictions_df['avg_latitude']
    )]

    gdf = gpd.GeoDataFrame(predictions_df, geometry=geometry, crs="EPSG:4326")
    return gdf

def compute_confusion_class(y_true, y_pred):
    """Compute confusion matrix class for each prediction."""
    classes = []
    for true, pred in zip(y_true, y_pred):
        if true == 1 and pred == 1:
            classes.append('TP')
        elif true == 0 and pred == 0:
            classes.append('TN')
        elif true == 0 and pred == 1:
            classes.append('FP')
        elif true == 1 and pred == 0:
            classes.append('FN')
        else:
            classes.append('no_data')
    return classes

def spatial_join_and_aggregate(pred_points, ipc_gdf, period_filter=None):
    """
    Spatial join predictions to IPC polygons and aggregate.
    Optionally filter by IPC period.
    Uses MAX aggregation (conservative - if any crisis in district, mark as crisis).
    """
    # Filter by period if specified
    if period_filter is not None:
        pred_points = pred_points[
            (pred_points['ipc_period_start'] == period_filter[0]) &
            (pred_points['ipc_period_end'] == period_filter[1])
        ].copy()

    if len(pred_points) == 0:
        print(f"    Warning: No predictions for period {period_filter}")
        return ipc_gdf.copy()

    # Perform spatial join: match each point to the IPC polygon it falls within
    joined = gpd.sjoin(pred_points, ipc_gdf, how='inner', predicate='within')

    if period_filter:
        print(f"    Period {period_filter[0]}: Matched {len(joined):,} predictions")

    # Aggregate by IPC district
    agg_dict = {
        'y_true': 'max',
        'ar_pred': 'max',
        'ar_prob': 'mean',
        'cascade_pred': 'max',
        'is_key_save': 'max',
    }

    if 'ipc_country' in joined.columns:
        agg_dict['ipc_country'] = 'first'
    if 'ipc_district' in joined.columns:
        agg_dict['ipc_district'] = 'first'

    agg = joined.groupby('index_right').agg(agg_dict).reset_index()

    # Compute confusion classes
    agg['ar_confusion'] = compute_confusion_class(
        agg['y_true'].values,
        agg['ar_pred'].values
    )
    agg['cascade_confusion'] = compute_confusion_class(
        agg['y_true'].values,
        agg['cascade_pred'].values
    )

    # Merge back to polygons
    ipc_plot = ipc_gdf.copy()
    ipc_plot = ipc_plot.reset_index()
    ipc_plot = ipc_plot.merge(agg, left_on='index', right_on='index_right', how='left')

    # Fill NaN confusion classes with 'no_data'
    ipc_plot['ar_confusion'] = ipc_plot['ar_confusion'].fillna('no_data')
    ipc_plot['cascade_confusion'] = ipc_plot['cascade_confusion'].fillna('no_data')
    ipc_plot['is_key_save'] = ipc_plot['is_key_save'].fillna(0)

    return ipc_plot

def draw_basemap_background(ax, africa_basemap):
    """Draw Africa basemap as light gray background."""
    if africa_basemap is None:
        return

    # Draw all African countries as light gray background
    africa_basemap.plot(
        ax=ax,
        color='#F8F9FA',
        edgecolor='#CCCCCC',
        linewidth=0.3,
        zorder=1
    )

def draw_basemap_overlay(ax, africa_basemap):
    """Draw country boundaries on top with thick black lines."""
    if africa_basemap is None:
        return

    # Plot country boundaries on top with thick black lines
    africa_basemap.boundary.plot(
        ax=ax,
        linewidth=2.0,
        edgecolor='#000000',
        facecolor='none',
        zorder=100,
        alpha=0.9
    )

def add_country_labels(ax, africa_basemap):
    """Add country name labels."""
    if africa_basemap is None:
        return

    # Find the name column
    name_col = None
    for col in ['NAME', 'name', 'ADMIN', 'NAME_LONG', 'COUNTRY']:
        if col in africa_basemap.columns:
            name_col = col
            break

    if name_col is None:
        return

    for idx, row in africa_basemap.iterrows():
        centroid = row.geometry.centroid
        # Only label if within map bounds
        if AFRICA_EXTENT[0] <= centroid.x <= AFRICA_EXTENT[1] and AFRICA_EXTENT[2] <= centroid.y <= AFRICA_EXTENT[3]:
            ax.text(
                centroid.x, centroid.y,
                row[name_col],
                fontsize=8,
                ha='center',
                va='center',
                color='#000000',
                weight='bold',
                zorder=150,
                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.85, edgecolor='black', linewidth=0.8)
            )

def create_key_saves_period_map(ipc_data, africa_basemap, model_name, period_str, n_saves_total, output_path):
    """Create map showing key saves for a specific IPC period.

    Args:
        n_saves_total: Total number of key save predictions (for consistency with performance charts)
    """

    fig, ax = plt.subplots(1, 1, figsize=(20, 14))

    # Draw Africa basemap background
    draw_basemap_background(ax, africa_basemap)

    # Assign key save categories
    ipc_data_plot = ipc_data.copy()
    ipc_data_plot['category'] = 'no_data'

    # No crisis districts (y_true = 0)
    ipc_data_plot.loc[ipc_data_plot['y_true'] == 0, 'category'] = 'no_crisis'

    # AR caught (TP in AR)
    ipc_data_plot.loc[ipc_data_plot['ar_confusion'] == 'TP', 'category'] = 'ar_caught'

    # Both models missed (FN in both)
    both_missed = (ipc_data_plot['ar_confusion'] == 'FN') & (ipc_data_plot['cascade_confusion'] == 'FN')
    ipc_data_plot.loc[both_missed, 'category'] = 'still_missed'

    # Key saves (AR missed, Ensemble caught)
    ipc_data_plot.loc[ipc_data_plot['is_key_save'] == 1, 'category'] = 'key_save'

    # Plot each category
    for category in ['no_crisis', 'ar_caught', 'still_missed', 'key_save']:
        subset = ipc_data_plot[ipc_data_plot['category'] == category]
        if len(subset) > 0:
            subset.plot(
                ax=ax,
                color=KEY_SAVE_COLORS[category],
                edgecolor='#333333' if category == 'key_save' else '#AAAAAA',
                linewidth=2 if category == 'key_save' else 0.3,
                zorder=5 if category == 'key_save' else 2
            )

    # Draw country boundaries on top
    draw_basemap_overlay(ax, africa_basemap)

    # Add country labels
    add_country_labels(ax, africa_basemap)

    # Set Africa extent
    ax.set_xlim(AFRICA_EXTENT[0], AFRICA_EXTENT[1])
    ax.set_ylim(AFRICA_EXTENT[2], AFRICA_EXTENT[3])
    ax.set_aspect('equal')
    ax.axis('off')

    # Title - use total predictions count (for consistency with performance charts)
    ax.set_title(
        f'{model_name}: {n_saves_total} Key Saves in {period_str}\n' +
        'Vulnerable Populations: AR Missed, Cascade Ensemble Caught',
        fontsize=16, weight='bold', pad=20
    )

    # Legend - simple and clear (no district counts - less confusing)
    legend_elements = [
        mpatches.Patch(color=KEY_SAVE_COLORS['key_save'],
                      label='KEY SAVE: AR missed, Ensemble caught',
                      edgecolor='#333333', linewidth=2),
        mpatches.Patch(color=KEY_SAVE_COLORS['still_missed'],
                      label='Still missed by both models'),
        mpatches.Patch(color=KEY_SAVE_COLORS['ar_caught'],
                      label='AR already caught'),
        mpatches.Patch(color=KEY_SAVE_COLORS['no_crisis'],
                      label='No crisis (true negative)')
    ]

    ax.legend(handles=legend_elements, loc='lower left', frameon=True,
             fancybox=True, shadow=True, fontsize=10)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved: {output_path}")

def create_three_panel_comparison_map(ipc_data, africa_basemap, model_name, period_str, n_saves_total, output_path):
    """Create 3-panel comparison: AR Baseline | Cascade Ensemble | Key Saves with confusion matrices."""

    fig, axes = plt.subplots(1, 3, figsize=(30, 10))

    # Common setup for all panels
    for ax in axes:
        draw_basemap_background(ax, africa_basemap)

    ipc_data_plot = ipc_data.copy()

    # =========================================================================
    # PANEL A: AR BASELINE CONFUSION MATRIX
    # =========================================================================
    ax = axes[0]

    # Plot confusion matrix for AR
    for confusion_type in ['TN', 'TP', 'FP', 'FN']:
        subset = ipc_data_plot[ipc_data_plot['ar_confusion'] == confusion_type]
        if len(subset) > 0:
            subset.plot(
                ax=ax,
                color=CONFUSION_COLORS[confusion_type],
                edgecolor='#666666',
                linewidth=0.3,
                zorder=2
            )

    draw_basemap_overlay(ax, africa_basemap)
    add_country_labels(ax, africa_basemap)

    ax.set_xlim(AFRICA_EXTENT[0], AFRICA_EXTENT[1])
    ax.set_ylim(AFRICA_EXTENT[2], AFRICA_EXTENT[3])
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title(
        f'A. AR Baseline Performance\n{period_str}',
        fontsize=14, weight='bold', pad=15
    )

    # Legend for Panel A
    legend_a = [
        mpatches.Patch(color=CONFUSION_COLORS['TP'], label='Correct: Crisis (TP)'),
        mpatches.Patch(color=CONFUSION_COLORS['TN'], label='Correct: No Crisis (TN)'),
        mpatches.Patch(color=CONFUSION_COLORS['FP'], label='False Alarm (FP)'),
        mpatches.Patch(color=CONFUSION_COLORS['FN'], label='Missed Crisis (FN)')
    ]
    ax.legend(handles=legend_a, loc='lower left', frameon=True, fancybox=True, shadow=True, fontsize=9)

    # =========================================================================
    # PANEL B: CASCADE ENSEMBLE CONFUSION MATRIX
    # =========================================================================
    ax = axes[1]

    # Plot confusion matrix for Cascade
    for confusion_type in ['TN', 'TP', 'FP', 'FN']:
        subset = ipc_data_plot[ipc_data_plot['cascade_confusion'] == confusion_type]
        if len(subset) > 0:
            subset.plot(
                ax=ax,
                color=CONFUSION_COLORS[confusion_type],
                edgecolor='#666666',
                linewidth=0.3,
                zorder=2
            )

    draw_basemap_overlay(ax, africa_basemap)
    add_country_labels(ax, africa_basemap)

    ax.set_xlim(AFRICA_EXTENT[0], AFRICA_EXTENT[1])
    ax.set_ylim(AFRICA_EXTENT[2], AFRICA_EXTENT[3])
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title(
        f'B. Cascade Ensemble Performance\n{period_str}',
        fontsize=14, weight='bold', pad=15
    )

    # Legend for Panel B
    legend_b = [
        mpatches.Patch(color=CONFUSION_COLORS['TP'], label='Correct: Crisis (TP)'),
        mpatches.Patch(color=CONFUSION_COLORS['TN'], label='Correct: No Crisis (TN)'),
        mpatches.Patch(color=CONFUSION_COLORS['FP'], label='False Alarm (FP)'),
        mpatches.Patch(color=CONFUSION_COLORS['FN'], label='Missed Crisis (FN)')
    ]
    ax.legend(handles=legend_b, loc='lower left', frameon=True, fancybox=True, shadow=True, fontsize=9)

    # =========================================================================
    # PANEL C: KEY SAVES (AR MISSED, ENSEMBLE CAUGHT)
    # =========================================================================
    ax = axes[2]

    # Assign key save categories
    ipc_data_plot['category'] = 'no_data'
    ipc_data_plot.loc[ipc_data_plot['y_true'] == 0, 'category'] = 'no_crisis'
    ipc_data_plot.loc[ipc_data_plot['ar_confusion'] == 'TP', 'category'] = 'ar_caught'

    # Both models missed
    both_missed = (ipc_data_plot['ar_confusion'] == 'FN') & (ipc_data_plot['cascade_confusion'] == 'FN')
    ipc_data_plot.loc[both_missed, 'category'] = 'still_missed'

    # Key saves
    ipc_data_plot.loc[ipc_data_plot['is_key_save'] == 1, 'category'] = 'key_save'

    # Plot each category
    for category in ['no_crisis', 'ar_caught', 'still_missed', 'key_save']:
        subset = ipc_data_plot[ipc_data_plot['category'] == category]
        if len(subset) > 0:
            subset.plot(
                ax=ax,
                color=KEY_SAVE_COLORS[category],
                edgecolor='#333333' if category == 'key_save' else '#AAAAAA',
                linewidth=2 if category == 'key_save' else 0.3,
                zorder=5 if category == 'key_save' else 2
            )

    draw_basemap_overlay(ax, africa_basemap)
    add_country_labels(ax, africa_basemap)

    ax.set_xlim(AFRICA_EXTENT[0], AFRICA_EXTENT[1])
    ax.set_ylim(AFRICA_EXTENT[2], AFRICA_EXTENT[3])
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title(
        f'C. Key Saves: {n_saves_total}\n{period_str}',
        fontsize=14, weight='bold', pad=15
    )

    # Legend for Panel C
    legend_c = [
        mpatches.Patch(color=KEY_SAVE_COLORS['key_save'],
                      label='KEY SAVE: AR missed, Ensemble caught',
                      edgecolor='#333333', linewidth=2),
        mpatches.Patch(color=KEY_SAVE_COLORS['still_missed'],
                      label='Still missed by both models'),
        mpatches.Patch(color=KEY_SAVE_COLORS['ar_caught'],
                      label='AR already caught'),
        mpatches.Patch(color=KEY_SAVE_COLORS['no_crisis'],
                      label='No crisis (true negative)')
    ]
    ax.legend(handles=legend_c, loc='lower left', frameon=True, fancybox=True, shadow=True, fontsize=9)

    # Main title
    fig.suptitle(
        f'{model_name}: Performance Comparison Across Africa\n' +
        'AR Baseline vs Cascade Ensemble vs Key Saves',
        fontsize=18, weight='bold', y=0.98
    )

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved: {output_path}")

def create_overall_summary_map(all_periods_data, africa_basemap, model_name, total_saves, output_path):
    """Create overall summary map aggregating all periods."""

    fig, ax = plt.subplots(1, 1, figsize=(24, 16))

    # Draw Africa basemap background
    draw_basemap_background(ax, africa_basemap)

    # Count key saves per district across all periods
    key_save_counts = all_periods_data[all_periods_data['is_key_save'] == 1].groupby('district_name').size()

    # Merge counts back
    summary_data = all_periods_data.copy()
    summary_data['key_save_count'] = summary_data['district_name'].map(key_save_counts).fillna(0)

    # Categories
    summary_data['category'] = 'no_data'
    summary_data.loc[summary_data['key_save_count'] == 0, 'category'] = 'no_saves'
    summary_data.loc[summary_data['key_save_count'] > 0, 'category'] = 'has_saves'

    # Plot districts with no key saves (light gray)
    no_saves = summary_data[summary_data['category'] == 'no_saves']
    if len(no_saves) > 0:
        no_saves.plot(
            ax=ax,
            color='#F5F5F5',
            edgecolor='#CCCCCC',
            linewidth=0.3,
            zorder=2
        )

    # Plot districts with key saves (graduated colors by count)
    has_saves = summary_data[summary_data['category'] == 'has_saves']
    if len(has_saves) > 0:
        # Create color map from light gold to dark gold
        has_saves.plot(
            ax=ax,
            column='key_save_count',
            cmap='YlOrRd',
            edgecolor='#333333',
            linewidth=1.5,
            legend=True,
            legend_kwds={'label': 'Number of Key Saves', 'shrink': 0.6},
            vmin=0,
            vmax=has_saves['key_save_count'].max(),
            zorder=5
        )

    # Draw country boundaries on top
    draw_basemap_overlay(ax, africa_basemap)

    # Add country labels
    add_country_labels(ax, africa_basemap)

    # Set Africa extent
    ax.set_xlim(AFRICA_EXTENT[0], AFRICA_EXTENT[1])
    ax.set_ylim(AFRICA_EXTENT[2], AFRICA_EXTENT[3])
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    ax.set_title(
        f'{model_name}: {total_saves} Total Key Saves Across All IPC Periods\n' +
        'Geographic Distribution of Vulnerable Populations Rescued by Cascade Ensemble',
        fontsize=18, weight='bold', pad=20
    )

    # Summary box
    if len(has_saves) > 0:
        # Country breakdown
        country_counts = has_saves.groupby('ipc_country')['key_save_count'].sum().sort_values(ascending=False)
        top_5 = country_counts.head(5)

        summary_text = f"IMPACT: {total_saves} vulnerable populations\nnow receive early warning\n\nTop countries:\n"
        for country, count in top_5.items():
            summary_text += f"  {country}: {int(count)}\n"

        ax.text(
            0.98, 0.98, summary_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment='top',
            horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='gold', alpha=0.9, edgecolor='darkred', linewidth=3),
            weight='bold',
            color='darkred'
        )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  [OK] Saved: {output_path}")

def generate_maps_for_model(model_dir, model_name):
    """Generate all maps for a single model."""

    print(f"\n{'='*80}")
    print(f"Generating maps for: {model_name}")
    print(f"{'='*80}\n")

    # Load predictions
    pred_file = model_dir / 'cascade_optimized_predictions.csv'
    if not pred_file.exists():
        print(f"Error: Predictions file not found: {pred_file}")
        return

    predictions = pd.read_csv(pred_file)
    print(f"  - Loaded {len(predictions):,} predictions")

    # Load shapefiles
    print("  - Loading shapefiles...")
    africa_basemap = load_africa_basemap()
    ipc_gdf = load_ipc_boundaries()

    if ipc_gdf is None:
        print("Error: Could not load IPC boundaries")
        return

    print(f"  - Loaded {len(ipc_gdf):,} IPC districts")

    # Convert predictions to geographic points
    print("  - Converting predictions to geographic points...")
    pred_points = predictions_to_geopoints(predictions)

    # Get unique IPC periods
    periods = predictions[['ipc_period_start', 'ipc_period_end']].drop_duplicates().sort_values('ipc_period_start')
    print(f"\n  - Found {len(periods)} IPC periods")

    # Create output directory
    output_dir = FIGURES_DIR / model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')
    output_dir.mkdir(parents=True, exist_ok=True)

    # Generate map for each period
    print("\n  Generating period-specific maps...")
    all_period_data = []

    for idx, (_, period) in enumerate(periods.iterrows(), 1):
        period_start = period['ipc_period_start']
        period_end = period['ipc_period_end']

        # Filter predictions for this period
        period_preds = predictions[
            (predictions['ipc_period_start'] == period_start) &
            (predictions['ipc_period_end'] == period_end)
        ]

        n_saves = period_preds['is_key_save'].sum()

        if n_saves > 0:
            # Spatial join and aggregate for this period
            period_data = spatial_join_and_aggregate(
                pred_points, ipc_gdf,
                period_filter=(period_start, period_end)
            )

            # Create map for this period
            period_str = f"{period_start[:7]} to {period_end[:7]}"
            map_filename = f"period_{idx:02d}_{period_start[:7].replace('-', '')}_key_saves.png"

            # Create single-panel key saves map
            create_key_saves_period_map(
                period_data, africa_basemap, model_name,
                period_str, int(n_saves),
                output_dir / map_filename
            )

            # Create 3-panel comparison map (AR | Ensemble | Key Saves)
            comparison_filename = f"comparison_{idx:02d}_{period_start[:7].replace('-', '')}_three_panel.png"
            create_three_panel_comparison_map(
                period_data, africa_basemap, model_name,
                period_str, int(n_saves),
                output_dir / comparison_filename
            )

            all_period_data.append(period_data)

    # Create overall summary map
    print("\n  Generating overall summary map...")
    if all_period_data:
        # Combine all periods
        combined_data = pd.concat(all_period_data, ignore_index=True)
        total_saves = predictions['is_key_save'].sum()

        create_overall_summary_map(
            combined_data, africa_basemap, model_name,
            int(total_saves),
            output_dir / 'summary_all_periods_key_saves.png'
        )

    print(f"\n{'='*80}")
    print(f"All maps saved to: {output_dir}")
    print(f"{'='*80}\n")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("PUBLICATION-GRADE CHOROPLETH MAPS FOR CASCADE ENSEMBLE")
    print("="*80)

    # Model 1: Ablation Best
    model1_dir = RESULTS_DIR / 'cascade_ablation_best'
    generate_maps_for_model(model1_dir, "Ablation Model (Ratio + Location Features)")

    # Model 2: Advanced
    model2_dir = RESULTS_DIR / 'cascade_optimized_production'
    generate_maps_for_model(model2_dir, "Advanced Model (All Features)")

    print("\n" + "="*80)
    print("MAP GENERATION COMPLETE")
    print("="*80)
    print("\nGenerated maps:")
    print("  - Period-specific maps showing key saves for each IPC period")
    print("  - Overall summary map aggregating all periods")
    print("\nAll maps use:")
    print("  - Official IPC district boundaries")
    print("  - GADM country boundaries with labels")
    print("  - Publication-quality 300 DPI resolution")
    print("="*80 + "\n")
