"""
Financial Times-Style Narrative Visualizations for Cascade Ensemble Story
==========================================================================

Creates compelling FT-style visualizations with:
- Proportional bubble maps showing key saves by country
- Temporal evolution showing crisis patterns over time
- Annotation-rich narrative maps with callouts
- Globe insets and professional styling
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, FancyArrowPatch
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
import warnings
from config import BASE_DIR
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(rstr(BASE_DIR))
RESULTS_DIR = BASE_DIR / "RESULTS"
FIGURES_DIR = BASE_DIR / "FIGURES"
AFRICA_BASEMAP_FILE = Path(r"C:\GDELT_Africa_Extract\data\natural_earth\ne_50m_admin_0_countries_africa.shp")

# Africa extent
AFRICA_EXTENT = [-20, 55, -35, 40]

# FT Color Palette
FT_COLORS = {
    'background': '#FFF9F5',
    'map_bg': '#FFF1E0',
    'beige': '#F2DFCE',
    'teal': '#0D7680',
    'blue_grey': '#678096',
    'burnt_orange': '#CD5733',
    'dark_red': '#A12A19',
    'text_dark': '#333333',
    'text_light': '#666666',
    'border': '#999999'
}

# Publication-quality settings
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Georgia', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1,
    'savefig.facecolor': 'white',
    'pdf.fonttype': 42,
    'ps.fonttype': 42,
})

def load_africa_basemap():
    """Load complete Africa basemap."""
    if not AFRICA_BASEMAP_FILE.exists():
        print(f"Warning: Africa basemap not found")
        return None

    africa = gpd.read_file(AFRICA_BASEMAP_FILE)
    if africa.crs.to_epsg() != 4326:
        africa = africa.to_crs('EPSG:4326')

    print(f"  - Loaded {len(africa)} African countries")
    return africa

def add_country_labels(ax, africa_basemap, extent):
    """Add country names to the map - ALL African countries."""
    x_min, x_max, y_min, y_max = extent[0], extent[1], extent[2], extent[3]

    for idx, row in africa_basemap.iterrows():
        # Get country name - try different column names
        country_name = None
        for col in ['NAME', 'ADMIN', 'name', 'admin', 'NAME_EN']:
            if col in africa_basemap.columns:
                country_name = row[col]
                break

        if country_name is None:
            continue

        # Get centroid
        centroid = row.geometry.centroid
        x, y = centroid.x, centroid.y

        # Check if within extent
        if x_min <= x <= x_max and y_min <= y <= y_max:
            # Adjust font size based on country size
            geom_area = row.geometry.area
            if geom_area < 5:  # Small countries
                fontsize = 6
            elif geom_area < 20:  # Medium countries
                fontsize = 7
            else:  # Large countries
                fontsize = 8

            ax.text(
                x, y, country_name,
                fontsize=fontsize,
                fontweight='bold',
                color=FT_COLORS['text_dark'],
                ha='center',
                va='center',
                alpha=0.8,
                zorder=150
            )

def create_globe_inset(ax_main, africa_basemap):
    """Create globe inset showing Africa."""
    ax_inset = inset_axes(
        ax_main,
        width="12%",
        height="12%",
        loc='lower left',
        borderpad=2
    )

    ax_inset.set_facecolor('white')
    ax_inset.set_aspect('equal')

    af_bounds = africa_basemap.total_bounds
    af_x_min, af_y_min, af_x_max, af_y_max = af_bounds
    center_x = (af_x_min + af_x_max) / 2
    center_y = (af_y_min + af_y_max) / 2

    from matplotlib.patches import Circle
    globe_radius = 50

    circle = Circle(
        (center_x, center_y), globe_radius,
        facecolor='white',
        edgecolor='#666666',
        linewidth=1.5,
        zorder=1
    )
    ax_inset.add_patch(circle)

    africa_basemap.plot(
        ax=ax_inset,
        facecolor='#BBBBBB',
        edgecolor='#666666',
        linewidth=0.4,
        zorder=2
    )

    ax_inset.set_xlim(center_x - globe_radius - 2, center_x + globe_radius + 2)
    ax_inset.set_ylim(center_y - globe_radius - 2, center_y + globe_radius + 2)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])

    for spine in ax_inset.spines.values():
        spine.set_visible(False)

    return ax_inset

def create_proportional_bubble_map(predictions_df, africa_basemap, model_name, output_path):
    """
    Create FT-style bubble map showing key saves by country.
    Bubble size = number of key saves.
    NO HARDCODED METRICS.
    """

    print(f"\n  Creating proportional bubble map for {model_name}...")

    # Calculate key saves by country - NO HARDCODING
    country_stats = predictions_df[predictions_df['is_key_save'] == 1].groupby('ipc_country').agg({
        'is_key_save': 'sum',
        'avg_latitude': 'mean',
        'avg_longitude': 'mean'
    }).reset_index()

    country_stats.columns = ['country', 'n_key_saves', 'lat', 'lon']

    # Also get AR caught for comparison
    ar_caught = predictions_df[predictions_df['confusion_ar'] == 'TP'].groupby('ipc_country').size().reset_index()
    ar_caught.columns = ['country', 'ar_caught_count']

    country_stats = country_stats.merge(ar_caught, on='country', how='left')
    country_stats['ar_caught_count'] = country_stats['ar_caught_count'].fillna(0)

    # Sort by key saves
    country_stats = country_stats.sort_values('n_key_saves', ascending=False)

    # Create figure
    fig = plt.figure(figsize=(20, 14), facecolor=FT_COLORS['background'])
    ax = fig.add_axes([0.05, 0.12, 0.9, 0.78])
    ax.set_facecolor(FT_COLORS['map_bg'])

    # Plot Africa basemap
    africa_basemap.plot(
        ax=ax,
        facecolor='#E8E8E8',
        edgecolor=FT_COLORS['text_dark'],
        linewidth=0.8,
        alpha=0.3,
        zorder=1
    )

    africa_basemap.boundary.plot(
        ax=ax,
        linewidth=1.2,
        edgecolor=FT_COLORS['text_dark'],
        alpha=0.6,
        zorder=100
    )

    # Plot bubbles for key saves (orange)
    max_saves = country_stats['n_key_saves'].max()
    for _, row in country_stats.iterrows():
        if row['n_key_saves'] > 0:
            # Bubble size proportional to saves
            size = (row['n_key_saves'] / max_saves) * 3000 + 200

            ax.scatter(
                row['lon'], row['lat'],
                s=size,
                color='#FF6600',
                alpha=0.6,
                edgecolor=FT_COLORS['dark_red'],
                linewidth=2,
                zorder=10,
                label='_nolegend_'
            )

    # Plot smaller bubbles for AR caught (green)
    max_ar = country_stats['ar_caught_count'].max()
    for _, row in country_stats[country_stats['ar_caught_count'] > 0].iterrows():
        size_ar = (row['ar_caught_count'] / max_ar) * 1500 + 100

        ax.scatter(
            row['lon'] + 2, row['lat'] - 2,  # Slight offset
            s=size_ar,
            color='#66CC66',
            alpha=0.4,
            edgecolor='#339933',
            linewidth=1,
            zorder=5,
            label='_nolegend_'
        )

    # Add annotations for top 3 countries
    top_3 = country_stats.head(3)
    for idx, row in top_3.iterrows():
        ax.annotate(
            f"{row['country']}\n{int(row['n_key_saves'])} key saves",
            xy=(row['lon'], row['lat']),
            xytext=(20, 20),
            textcoords='offset points',
            fontsize=9,
            fontweight='bold',
            color=FT_COLORS['text_dark'],
            bbox=dict(
                boxstyle='round,pad=0.5',
                facecolor=FT_COLORS['beige'],
                edgecolor=FT_COLORS['burnt_orange'],
                linewidth=1.5,
                alpha=0.9
            ),
            arrowprops=dict(
                arrowstyle='->',
                connectionstyle='arc3,rad=0.3',
                color=FT_COLORS['burnt_orange'],
                linewidth=2
            ),
            zorder=200
        )

    # Add country labels
    add_country_labels(ax, africa_basemap, AFRICA_EXTENT)

    # Globe inset
    create_globe_inset(ax, africa_basemap)

    # Set extent
    ax.set_xlim(AFRICA_EXTENT[0], AFRICA_EXTENT[1])
    ax.set_ylim(AFRICA_EXTENT[2], AFRICA_EXTENT[3])
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    total_key_saves = int(country_stats['n_key_saves'].sum())
    n_countries = len(country_stats[country_stats['n_key_saves'] > 0])

    fig.text(
        0.5, 0.94,
        f'{model_name}: Geographic Distribution of {total_key_saves} Key Saves',
        fontsize=18, weight='bold', ha='center', va='top',
        color=FT_COLORS['text_dark']
    )

    fig.text(
        0.5, 0.91,
        f'Cascade ensemble catches crises AR missed across {n_countries} countries',
        fontsize=12, ha='center', va='top',
        color=FT_COLORS['text_light'], style='italic'
    )

    # Enhanced legend with bubble size reference (positioned to avoid Madagascar)
    legend_ax = fig.add_axes([0.05, 0.15, 0.28, 0.25], facecolor='none')
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis('off')

    # Title
    legend_ax.text(0.5, 0.95, 'Legend', fontsize=11, ha='center', weight='bold', color=FT_COLORS['text_dark'])

    # Draw example bubbles with sizes
    # Large bubble
    legend_ax.scatter(0.12, 0.75, s=2000, color='#FF6600', alpha=0.6, edgecolor=FT_COLORS['dark_red'], linewidth=2)
    legend_ax.text(0.28, 0.75, f'Large: ≈{int(max_saves)} key saves', fontsize=9, va='center', color=FT_COLORS['text_dark'])

    # Medium bubble
    legend_ax.scatter(0.12, 0.55, s=1200, color='#FF6600', alpha=0.6, edgecolor=FT_COLORS['dark_red'], linewidth=2)
    legend_ax.text(0.28, 0.55, f'Medium: ≈{int(max_saves/2)} key saves', fontsize=9, va='center', color=FT_COLORS['text_dark'])

    # Small bubble
    legend_ax.scatter(0.12, 0.35, s=500, color='#FF6600', alpha=0.6, edgecolor=FT_COLORS['dark_red'], linewidth=2)
    legend_ax.text(0.28, 0.35, 'Small: Few key saves', fontsize=9, va='center', color=FT_COLORS['text_dark'])

    # AR caught reference
    legend_ax.scatter(0.12, 0.15, s=800, color='#66CC66', alpha=0.4, edgecolor='#339933', linewidth=1)
    legend_ax.text(0.28, 0.15, 'AR already caught (green)', fontsize=9, va='center', color=FT_COLORS['text_dark'])

    legend_ax.text(0.5, 0.02, 'Bubble size = number of predictions', fontsize=8, ha='center',
                  color=FT_COLORS['text_light'], style='italic', weight='bold')

    # Source
    fig.text(
        0.02, 0.01,
        'Source: IPC, GDELT | Analysis: Cascade Ensemble Model',
        fontsize=8, color=FT_COLORS['text_light']
    )

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=FT_COLORS['background'])
    plt.close()

    print(f"  [OK] Saved: {output_path}")

def create_temporal_evolution_map(predictions_df, africa_basemap, model_name, output_path):
    """
    Create temporal evolution map showing how key saves evolved over time.
    Shows temporal patterns with arrows and annotations.
    NO HARDCODED METRICS.
    """

    print(f"\n  Creating temporal evolution map for {model_name}...")

    # Get key saves by period and country
    period_stats = predictions_df[predictions_df['is_key_save'] == 1].groupby(
        ['ipc_period_start', 'ipc_country']
    ).agg({
        'is_key_save': 'sum',
        'avg_latitude': 'mean',
        'avg_longitude': 'mean'
    }).reset_index()

    period_stats.columns = ['period', 'country', 'n_saves', 'lat', 'lon']
    period_stats['period_date'] = pd.to_datetime(period_stats['period'])
    period_stats = period_stats.sort_values('period_date')

    # Divide into early, mid, late periods
    periods = period_stats['period'].unique()
    n_periods = len(periods)

    early_periods = periods[:n_periods//3]
    mid_periods = periods[n_periods//3:2*n_periods//3]
    late_periods = periods[2*n_periods//3:]

    period_stats['era'] = 'Late'
    period_stats.loc[period_stats['period'].isin(early_periods), 'era'] = 'Early'
    period_stats.loc[period_stats['period'].isin(mid_periods), 'era'] = 'Mid'

    # Create figure
    fig = plt.figure(figsize=(22, 14), facecolor=FT_COLORS['background'])
    ax = fig.add_axes([0.05, 0.12, 0.9, 0.78])
    ax.set_facecolor(FT_COLORS['map_bg'])

    # Plot Africa
    africa_basemap.plot(
        ax=ax,
        facecolor='#E8E8E8',
        edgecolor=FT_COLORS['text_dark'],
        linewidth=0.8,
        alpha=0.3,
        zorder=1
    )

    africa_basemap.boundary.plot(
        ax=ax,
        linewidth=1.2,
        edgecolor=FT_COLORS['text_dark'],
        alpha=0.6,
        zorder=100
    )

    # Plot points by era with different colors
    era_colors = {
        'Early': '#9966CC',  # Purple
        'Mid': '#FF9933',    # Orange
        'Late': '#CC0000'    # Red
    }

    for era, color in era_colors.items():
        era_data = period_stats[period_stats['era'] == era]
        if len(era_data) > 0:
            for _, row in era_data.iterrows():
                size = row['n_saves'] * 50 + 100
                ax.scatter(
                    row['lon'], row['lat'],
                    s=size,
                    color=color,
                    alpha=0.6,
                    edgecolor='black',
                    linewidth=1.5,
                    zorder=10
                )

    # Add country labels
    add_country_labels(ax, africa_basemap, AFRICA_EXTENT)

    # Globe inset
    create_globe_inset(ax, africa_basemap)

    # Set extent
    ax.set_xlim(AFRICA_EXTENT[0], AFRICA_EXTENT[1])
    ax.set_ylim(AFRICA_EXTENT[2], AFRICA_EXTENT[3])
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    total_saves = int(period_stats['n_saves'].sum())

    fig.text(
        0.5, 0.94,
        f'{model_name}: Temporal Evolution of {total_saves} Key Saves',
        fontsize=18, weight='bold', ha='center', va='top',
        color=FT_COLORS['text_dark']
    )

    # Get actual period dates for subtitle
    start_date = period_stats['period_date'].min().strftime('%Y-%m')
    end_date = period_stats['period_date'].max().strftime('%Y-%m')

    fig.text(
        0.5, 0.91,
        f'Crisis patterns from {start_date} to {end_date}',
        fontsize=12, ha='center', va='top',
        color=FT_COLORS['text_light'], style='italic'
    )

    # Enhanced legend with bubble size explanation
    # Time period legend
    legend_elements = [
        mpatches.Patch(color='#9966CC', label=f'Early period ({early_periods[0][:7]} to {early_periods[-1][:7]})', alpha=0.6),
        mpatches.Patch(color='#FF9933', label=f'Mid period ({mid_periods[0][:7]} to {mid_periods[-1][:7]})', alpha=0.6),
        mpatches.Patch(color='#CC0000', label=f'Late period ({late_periods[0][:7]} to {late_periods[-1][:7]})', alpha=0.6)
    ]

    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.08),
        ncol=3,
        frameon=True,
        fancybox=True,
        fontsize=10,
        edgecolor=FT_COLORS['border'],
        title='Time Periods (bubble size = number of key saves per country-period)',
        title_fontsize=9
    )

    # Source
    fig.text(
        0.02, 0.01,
        'Source: IPC, GDELT | Analysis: Cascade Ensemble Model',
        fontsize=8, color=FT_COLORS['text_light']
    )

    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=FT_COLORS['background'])
    plt.close()

    print(f"  [OK] Saved: {output_path}")

def generate_narrative_visualizations(model_dir, model_name):
    """Generate FT-style narrative visualizations."""

    print(f"\n{'='*80}")
    print(f"Generating FT-STYLE NARRATIVE VISUALIZATIONS for: {model_name}")
    print(f"{'='*80}\n")

    pred_file = model_dir / 'cascade_optimized_predictions.csv'
    if not pred_file.exists():
        print(f"Error: Predictions file not found")
        return

    predictions = pd.read_csv(pred_file)
    print(f"  - Loaded {len(predictions):,} predictions")

    africa_basemap = load_africa_basemap()
    if africa_basemap is None:
        return

    output_dir = FIGURES_DIR / model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Proportional bubble map
    bubble_output = output_dir / 'ft_narrative_bubble_map.png'
    create_proportional_bubble_map(predictions, africa_basemap, model_name, bubble_output)

    # 2. Temporal evolution map
    temporal_output = output_dir / 'ft_narrative_temporal_evolution.png'
    create_temporal_evolution_map(predictions, africa_basemap, model_name, temporal_output)

    print(f"\n{'='*80}")
    print(f"Narrative visualizations saved to: {output_dir}")
    print(f"{'='*80}\n")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("FT-STYLE NARRATIVE VISUALIZATIONS FOR CASCADE ENSEMBLE")
    print("="*80)

    model1_dir = RESULTS_DIR / 'cascade_ablation_best'
    generate_narrative_visualizations(model1_dir, "Ablation Model (Ratio + Location Features)")

    model2_dir = RESULTS_DIR / 'cascade_optimized_production'
    generate_narrative_visualizations(model2_dir, "Advanced Model (All Features)")

    print("\n" + "="*80)
    print("NARRATIVE VISUALIZATION GENERATION COMPLETE")
    print("="*80 + "\n")
