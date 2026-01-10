"""
Enhanced FT-Style Publication-Grade Choropleth Maps for Cascade Ensemble
=========================================================================

Creates Financial Times-style maps with:
- Beige/cream backgrounds
- Inset locator maps
- Numbered narrative boxes
- Curved arrows and annotations
- Professional typography
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
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

# Africa extent
AFRICA_EXTENT = [-20, 55, -35, 40]

# Financial Times Color Palette
FT_COLORS = {
    'background': '#FFF9F5',      # Very light cream
    'map_bg': '#FFF1E0',          # Old lace (map background)
    'beige': '#F2DFCE',           # Champagne pink (boxes)
    'teal': '#0D7680',            # FT accent teal
    'blue_grey': '#678096',       # Primary data color
    'light_blue': '#ACC2CF',      # Secondary data
    'burnt_orange': '#CD5733',    # Highlight/accent
    'olive': '#979461',           # Neutral
    'dark_red': '#A12A19',        # Strong accent
    'text_dark': '#333333',       # Text
    'text_light': '#666666',      # Caption text
    'border': '#999999'           # Borders
}

# Confusion matrix colors - Highly distinct colors
CONFUSION_COLORS_FT = {
    'TP': '#0066CC',    # Strong blue - Correct crisis (TRUE POSITIVE)
    'TN': '#66CC66',    # Green - Correct no crisis (TRUE NEGATIVE)
    'FP': '#FFD700',    # Gold/Yellow - False alarm (FALSE POSITIVE)
    'FN': '#CC0000',    # Red - Missed crisis (FALSE NEGATIVE)
    'no_data': '#E8E8E8'
}

# Key saves colors - Maximum distinction (different color families)
KEY_SAVE_COLORS_FT = {
    'key_save': '#9400D3',      # Purple/Violet - Key saves (ensemble caught, AR missed)
    'still_missed': '#CC0000',  # Red - Still missed by both (matches FN)
    'ar_caught': '#66CC66',     # Green - AR already caught
    'no_crisis': '#F0F0F0'      # Very light gray
}

# Publication-quality typography settings
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
    'axes.titlesize': 14,
    'axes.titleweight': 'bold',
    'axes.labelsize': 11,
    'axes.linewidth': 0.8,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'legend.frameon': True,
    'legend.framealpha': 0.95,
    'legend.edgecolor': FT_COLORS['border'],
    'legend.fancybox': True,
    'lines.linewidth': 1.5,
    'grid.linewidth': 0.5,
    'grid.alpha': 0.3
})

def load_africa_basemap():
    """Load complete Africa basemap."""
    if not AFRICA_BASEMAP_FILE.exists():
        print(f"    Warning: Africa basemap not found")
        return None

    africa = gpd.read_file(AFRICA_BASEMAP_FILE)
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

    print(f"Error: IPC boundaries not found")
    return None

def predictions_to_geopoints(predictions_df):
    """Convert predictions DataFrame to GeoDataFrame."""
    geometry = [Point(xy) for xy in zip(
        predictions_df['avg_longitude'],
        predictions_df['avg_latitude']
    )]
    return gpd.GeoDataFrame(predictions_df, geometry=geometry, crs="EPSG:4326")

def compute_confusion_class(y_true, y_pred):
    """Compute confusion matrix class."""
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
    """Spatial join and aggregate predictions to districts."""
    if period_filter is not None:
        pred_points = pred_points[
            (pred_points['ipc_period_start'] == period_filter[0]) &
            (pred_points['ipc_period_end'] == period_filter[1])
        ].copy()

    if len(pred_points) == 0:
        return ipc_gdf.copy()

    joined = gpd.sjoin(pred_points, ipc_gdf, how='inner', predicate='within')

    agg_dict = {
        'y_true': 'max',
        'ar_pred': 'max',
        'cascade_pred': 'max',
        'is_key_save': 'max',
    }

    agg = joined.groupby('index_right').agg(agg_dict).reset_index()

    agg['ar_confusion'] = compute_confusion_class(
        agg['y_true'].values,
        agg['ar_pred'].values
    )
    agg['cascade_confusion'] = compute_confusion_class(
        agg['y_true'].values,
        agg['cascade_pred'].values
    )

    ipc_plot = ipc_gdf.copy().reset_index()
    ipc_plot = ipc_plot.merge(agg, left_on='index', right_on='index_right', how='left')
    ipc_plot['ar_confusion'] = ipc_plot['ar_confusion'].fillna('no_data')
    ipc_plot['cascade_confusion'] = ipc_plot['cascade_confusion'].fillna('no_data')
    ipc_plot['is_key_save'] = ipc_plot['is_key_save'].fillna(0)

    return ipc_plot

def create_inset_locator(ax_main, africa_basemap, region_bounds):
    """Create globe inset showing Africa outline with red box highlighting region."""
    ax_inset = inset_axes(
        ax_main,
        width="15%",
        height="15%",
        loc='lower left',
        borderpad=2
    )

    # Set background
    ax_inset.set_facecolor('white')
    ax_inset.set_aspect('equal')

    # Get Africa bounds for centering
    af_bounds = africa_basemap.total_bounds
    af_x_min, af_y_min, af_x_max, af_y_max = af_bounds
    center_x = (af_x_min + af_x_max) / 2
    center_y = (af_y_min + af_y_max) / 2

    # Draw a circle to represent the globe (centered on Africa)
    from matplotlib.patches import Circle, Rectangle
    globe_radius = 50  # Adjusted for proper globe size

    circle = Circle(
        (center_x, center_y), globe_radius,
        facecolor='white',
        edgecolor='#666666',
        linewidth=1.5,
        zorder=1
    )
    ax_inset.add_patch(circle)

    # Plot Africa as simple gray outline within the globe
    africa_basemap.plot(
        ax=ax_inset,
        facecolor='#BBBBBB',
        edgecolor='#666666',
        linewidth=0.4,
        zorder=2
    )

    # Draw red box around the data region (not all of Africa)
    x_min, y_min, x_max, y_max = region_bounds

    rect = Rectangle(
        (x_min, y_min),
        (x_max - x_min),
        (y_max - y_min),
        linewidth=2,
        edgecolor=FT_COLORS['dark_red'],
        facecolor='none',
        zorder=3
    )
    ax_inset.add_patch(rect)

    # Set extent to show globe
    ax_inset.set_xlim(center_x - globe_radius - 2, center_x + globe_radius + 2)
    ax_inset.set_ylim(center_y - globe_radius - 2, center_y + globe_radius + 2)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])

    # Remove spines for clean globe look
    for spine in ax_inset.spines.values():
        spine.set_visible(False)

    return ax_inset

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
            # Adjust font size based on country size (smaller countries get smaller text)
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

def add_narrative_box(ax, number, text, position, width=0.28, height=0.11):
    """Add numbered narrative text box in FT style."""
    x, y = position

    # Create box
    box = FancyBboxPatch(
        (x, y), width, height,
        boxstyle='round,pad=0.01',
        transform=ax.transAxes,
        facecolor=FT_COLORS['beige'],
        edgecolor=FT_COLORS['blue_grey'],
        linewidth=1.2,
        alpha=0.95,
        zorder=200
    )
    ax.add_patch(box)

    # Add number circle
    ax.text(
        x + 0.025, y + height/2,
        str(number),
        transform=ax.transAxes,
        fontsize=13,
        fontweight='bold',
        color='white',
        bbox=dict(
            boxstyle='circle,pad=0.35',
            facecolor=FT_COLORS['burnt_orange'],
            edgecolor='none'
        ),
        verticalalignment='center',
        horizontalalignment='center',
        zorder=201
    )

    # Add text
    ax.text(
        x + 0.06, y + height/2,
        text,
        transform=ax.transAxes,
        fontsize=9,
        color=FT_COLORS['text_dark'],
        verticalalignment='center',
        horizontalalignment='left',
        zorder=201,
        wrap=True
    )

def create_enhanced_three_panel_map(ipc_data, africa_basemap, model_name, period_str,
                                   n_saves_total, n_key_save_districts, output_path):
    """Create enhanced FT-style 3-panel comparison map with narrative."""

    fig = plt.figure(figsize=(32, 11), facecolor=FT_COLORS['background'])

    # Create 3 panels
    gs = fig.add_gridspec(1, 3, left=0.05, right=0.98, bottom=0.12, top=0.85,
                          wspace=0.08, hspace=0.05)

    axes = [fig.add_subplot(gs[0, i]) for i in range(3)]

    ipc_data_plot = ipc_data.copy()

    # Get region bounds for inset
    valid_geom = ipc_data_plot[ipc_data_plot.geometry.notnull()]
    if len(valid_geom) > 0:
        region_bounds = valid_geom.total_bounds
    else:
        region_bounds = AFRICA_EXTENT

    # =========================================================================
    # PANEL A: AR BASELINE
    # =========================================================================
    ax = axes[0]
    ax.set_facecolor(FT_COLORS['map_bg'])

    for confusion_type in ['TN', 'TP', 'FP', 'FN']:
        subset = ipc_data_plot[ipc_data_plot['ar_confusion'] == confusion_type]
        if len(subset) > 0:
            subset.plot(
                ax=ax,
                color=CONFUSION_COLORS_FT[confusion_type],
                edgecolor='white',
                linewidth=0.2,
                alpha=0.85,
                zorder=2
            )

    # Country boundaries
    africa_basemap.boundary.plot(
        ax=ax, linewidth=1.5, edgecolor=FT_COLORS['text_dark'],
        alpha=0.6, zorder=100
    )

    # Add country labels
    add_country_labels(ax, africa_basemap, AFRICA_EXTENT)

    # Inset locator
    create_inset_locator(ax, africa_basemap, region_bounds)

    ax.set_xlim(AFRICA_EXTENT[0], AFRICA_EXTENT[1])
    ax.set_ylim(AFRICA_EXTENT[2], AFRICA_EXTENT[3])
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('A. AR Baseline',
                fontsize=13, weight='bold', pad=15, color=FT_COLORS['text_dark'])

    # =========================================================================
    # PANEL B: CASCADE ENSEMBLE
    # =========================================================================
    ax = axes[1]
    ax.set_facecolor(FT_COLORS['map_bg'])

    for confusion_type in ['TN', 'TP', 'FP', 'FN']:
        subset = ipc_data_plot[ipc_data_plot['cascade_confusion'] == confusion_type]
        if len(subset) > 0:
            subset.plot(
                ax=ax,
                color=CONFUSION_COLORS_FT[confusion_type],
                edgecolor='white',
                linewidth=0.2,
                alpha=0.85,
                zorder=2
            )

    africa_basemap.boundary.plot(
        ax=ax, linewidth=1.5, edgecolor=FT_COLORS['text_dark'],
        alpha=0.6, zorder=100
    )

    # Add country labels
    add_country_labels(ax, africa_basemap, AFRICA_EXTENT)

    ax.set_xlim(AFRICA_EXTENT[0], AFRICA_EXTENT[1])
    ax.set_ylim(AFRICA_EXTENT[2], AFRICA_EXTENT[3])
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title('B. Cascade Ensemble',
                fontsize=13, weight='bold', pad=15, color=FT_COLORS['text_dark'])

    # =========================================================================
    # PANEL C: KEY SAVES
    # =========================================================================
    ax = axes[2]
    ax.set_facecolor(FT_COLORS['map_bg'])

    ipc_data_plot['category'] = 'no_data'
    ipc_data_plot.loc[ipc_data_plot['y_true'] == 0, 'category'] = 'no_crisis'
    ipc_data_plot.loc[ipc_data_plot['ar_confusion'] == 'TP', 'category'] = 'ar_caught'

    both_missed = (ipc_data_plot['ar_confusion'] == 'FN') & (ipc_data_plot['cascade_confusion'] == 'FN')
    ipc_data_plot.loc[both_missed, 'category'] = 'still_missed'
    ipc_data_plot.loc[ipc_data_plot['is_key_save'] == 1, 'category'] = 'key_save'

    for category in ['no_crisis', 'ar_caught', 'still_missed', 'key_save']:
        subset = ipc_data_plot[ipc_data_plot['category'] == category]
        if len(subset) > 0:
            subset.plot(
                ax=ax,
                color=KEY_SAVE_COLORS_FT[category],
                edgecolor='white' if category != 'key_save' else FT_COLORS['text_dark'],
                linewidth=0.2 if category != 'key_save' else 1.5,
                alpha=0.85,
                zorder=5 if category == 'key_save' else 2
            )

    africa_basemap.boundary.plot(
        ax=ax, linewidth=1.5, edgecolor=FT_COLORS['text_dark'],
        alpha=0.6, zorder=100
    )

    # Add country labels
    add_country_labels(ax, africa_basemap, AFRICA_EXTENT)

    ax.set_xlim(AFRICA_EXTENT[0], AFRICA_EXTENT[1])
    ax.set_ylim(AFRICA_EXTENT[2], AFRICA_EXTENT[3])
    ax.set_aspect('equal')
    ax.axis('off')

    ax.set_title(f'C. Key Saves: {n_saves_total}',
                fontsize=13, weight='bold', pad=15, color=FT_COLORS['text_dark'])

    # =========================================================================
    # NARRATIVE BOXES (Bottom of figure)
    # =========================================================================
    narrative_ax = fig.add_axes([0, 0, 1, 1], facecolor='none', zorder=300)
    narrative_ax.set_xlim(0, 1)
    narrative_ax.set_ylim(0, 1)
    narrative_ax.axis('off')

    add_narrative_box(
        narrative_ax, 1,
        'AR baseline misses critical food\ninsecurity crises (shown in red)',
        (0.05, 0.02), width=0.25, height=0.08
    )

    add_narrative_box(
        narrative_ax, 2,
        'Cascade ensemble verifies AR\npredictions with Stage 2 model',
        (0.35, 0.02), width=0.25, height=0.08
    )

    add_narrative_box(
        narrative_ax, 3,
        f'{n_saves_total} vulnerable populations now\nreceive early warning ({n_key_save_districts} districts)',
        (0.65, 0.02), width=0.3, height=0.08
    )

    # Main title - simplified
    fig.text(
        0.5, 0.93,
        f'{model_name}: {period_str}',
        fontsize=16, weight='bold', ha='center', va='top',
        color=FT_COLORS['text_dark']
    )

    # Subtitle - simplified
    fig.text(
        0.5, 0.90,
        'Cascade Ensemble vs AR Baseline',
        fontsize=11, ha='center', va='top',
        color=FT_COLORS['text_light'], style='italic'
    )

    # Legend (centered bottom)
    legend_elements = [
        mpatches.Patch(color=CONFUSION_COLORS_FT['TP'], label='Correct Crisis (TP)', alpha=0.85),
        mpatches.Patch(color=CONFUSION_COLORS_FT['TN'], label='Correct No Crisis (TN)', alpha=0.85),
        mpatches.Patch(color=CONFUSION_COLORS_FT['FP'], label='False Alarm (FP)', alpha=0.85),
        mpatches.Patch(color=CONFUSION_COLORS_FT['FN'], label='Missed Crisis (FN)', alpha=0.85),
        mpatches.Patch(color='white', label=''),  # Spacer
        mpatches.Patch(color=KEY_SAVE_COLORS_FT['key_save'], label='KEY SAVE',
                      edgecolor=FT_COLORS['text_dark'], linewidth=1.5, alpha=0.85)
    ]

    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.11),
        ncol=6,
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=9,
        edgecolor=FT_COLORS['border']
    )

    # Source attribution
    fig.text(
        0.02, 0.01,
        'Source: IPC, GDELT | Analysis: Cascade Ensemble Model',
        fontsize=8, color=FT_COLORS['text_light']
    )

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor=FT_COLORS['background'], edgecolor='none')
    plt.close()

    print(f"  [OK] Saved: {output_path}")

def create_single_key_saves_map_ft_style(ipc_data, africa_basemap, model_name,
                                         period_str, n_saves_total, output_path):
    """Create single-panel FT-style map showing only key saves."""

    fig = plt.figure(figsize=(18, 14), facecolor=FT_COLORS['background'])

    # Main map axis
    ax = fig.add_axes([0.05, 0.15, 0.9, 0.75])
    ax.set_facecolor(FT_COLORS['map_bg'])

    ipc_data_plot = ipc_data.copy()

    # Assign categories
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
                color=KEY_SAVE_COLORS_FT[category],
                edgecolor=FT_COLORS['text_dark'] if category == 'key_save' else 'white',
                linewidth=1.5 if category == 'key_save' else 0.3,
                alpha=0.85,
                zorder=5 if category == 'key_save' else 2
            )

    # Country boundaries
    africa_basemap.boundary.plot(
        ax=ax, linewidth=1.5, edgecolor=FT_COLORS['text_dark'],
        alpha=0.6, zorder=100
    )

    # Add country labels
    add_country_labels(ax, africa_basemap, AFRICA_EXTENT)

    # Get region bounds for inset
    valid_geom = ipc_data_plot[ipc_data_plot.geometry.notnull()]
    if len(valid_geom) > 0:
        region_bounds = valid_geom.total_bounds
    else:
        region_bounds = AFRICA_EXTENT

    # Inset locator
    create_inset_locator(ax, africa_basemap, region_bounds)

    ax.set_xlim(AFRICA_EXTENT[0], AFRICA_EXTENT[1])
    ax.set_ylim(AFRICA_EXTENT[2], AFRICA_EXTENT[3])
    ax.set_aspect('equal')
    ax.axis('off')

    # Title - simplified
    fig.text(
        0.5, 0.94,
        f'{model_name}: {n_saves_total} Key Saves',
        fontsize=18, weight='bold', ha='center', va='top',
        color=FT_COLORS['text_dark']
    )

    # Subtitle
    fig.text(
        0.5, 0.91,
        f'{period_str} | Vulnerable Populations: AR Missed, Cascade Ensemble Caught',
        fontsize=12, ha='center', va='top',
        color=FT_COLORS['text_light'], style='italic'
    )

    # Legend
    legend_elements = [
        mpatches.Patch(color=KEY_SAVE_COLORS_FT['key_save'],
                      label='KEY SAVE: AR missed, Ensemble caught',
                      edgecolor=FT_COLORS['text_dark'], linewidth=1.5, alpha=0.85),
        mpatches.Patch(color=KEY_SAVE_COLORS_FT['still_missed'],
                      label='Still missed by both models', alpha=0.85),
        mpatches.Patch(color=KEY_SAVE_COLORS_FT['ar_caught'],
                      label='AR already caught', alpha=0.85),
        mpatches.Patch(color=KEY_SAVE_COLORS_FT['no_crisis'],
                      label='No crisis (true negative)', alpha=0.85)
    ]

    fig.legend(
        handles=legend_elements,
        loc='lower center',
        bbox_to_anchor=(0.5, 0.08),
        ncol=4,
        frameon=True,
        fancybox=True,
        shadow=False,
        fontsize=10,
        edgecolor=FT_COLORS['border']
    )

    # Source attribution
    fig.text(
        0.02, 0.01,
        'Source: IPC, GDELT | Analysis: Cascade Ensemble Model',
        fontsize=8, color=FT_COLORS['text_light']
    )

    plt.savefig(output_path, dpi=300, bbox_inches='tight',
                facecolor=FT_COLORS['background'], edgecolor='none')
    plt.close()

    print(f"  [OK] Saved: {output_path}")

def generate_enhanced_maps_for_model(model_dir, model_name):
    """Generate enhanced FT-style maps for a model."""

    print(f"\n{'='*80}")
    print(f"Generating ENHANCED FT-STYLE maps for: {model_name}")
    print(f"{'='*80}\n")

    pred_file = model_dir / 'cascade_optimized_predictions.csv'
    if not pred_file.exists():
        print(f"Error: Predictions file not found")
        return

    predictions = pd.read_csv(pred_file)
    print(f"  - Loaded {len(predictions):,} predictions")

    print("  - Loading shapefiles...")
    africa_basemap = load_africa_basemap()
    ipc_gdf = load_ipc_boundaries()

    if ipc_gdf is None:
        print("Error: Could not load IPC boundaries")
        return

    print(f"  - Loaded {len(ipc_gdf):,} IPC districts")

    pred_points = predictions_to_geopoints(predictions)

    periods = predictions[['ipc_period_start', 'ipc_period_end']].drop_duplicates().sort_values('ipc_period_start')
    print(f"\n  - Found {len(periods)} IPC periods")

    output_dir = FIGURES_DIR / model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')
    output_dir.mkdir(parents=True, exist_ok=True)

    print("\n  Generating ENHANCED FT-STYLE period maps...")

    for idx, (_, period) in enumerate(periods.iterrows(), 1):
        period_start = period['ipc_period_start']
        period_end = period['ipc_period_end']

        period_preds = predictions[
            (predictions['ipc_period_start'] == period_start) &
            (predictions['ipc_period_end'] == period_end)
        ]

        n_saves = period_preds['is_key_save'].sum()

        if n_saves > 0:
            period_data = spatial_join_and_aggregate(
                pred_points, ipc_gdf,
                period_filter=(period_start, period_end)
            )

            n_key_save_districts = int(period_data[period_data['is_key_save'] == 1].shape[0])

            period_str = f"{period_start[:7]} to {period_end[:7]}"
            map_filename = f"ft_style_enhanced_{idx:02d}_{period_start[:7].replace('-', '')}.png"
            single_map_filename = f"ft_style_single_key_saves_{idx:02d}_{period_start[:7].replace('-', '')}.png"

            # Three-panel comparison map
            create_enhanced_three_panel_map(
                period_data, africa_basemap, model_name,
                period_str, int(n_saves), n_key_save_districts,
                output_dir / map_filename
            )

            # Single key saves map
            create_single_key_saves_map_ft_style(
                period_data, africa_basemap, model_name,
                period_str, int(n_saves),
                output_dir / single_map_filename
            )

    print(f"\n{'='*80}")
    print(f"Enhanced maps saved to: {output_dir}")
    print(f"{'='*80}\n")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("ENHANCED FT-STYLE CHOROPLETH MAPS FOR CASCADE ENSEMBLE")
    print("="*80)

    model1_dir = RESULTS_DIR / 'cascade_ablation_best'
    generate_enhanced_maps_for_model(model1_dir, "Ablation Model (Ratio + Location Features)")

    model2_dir = RESULTS_DIR / 'cascade_optimized_production'
    generate_enhanced_maps_for_model(model2_dir, "Advanced Model (All Features)")

    print("\n" + "="*80)
    print("ENHANCED FT-STYLE MAP GENERATION COMPLETE")
    print("="*80)
    print("\nFeatures added:")
    print("  - FT-style beige/cream backgrounds")
    print("  - Inset locator maps showing Africa context")
    print("  - Numbered narrative boxes telling the story")
    print("  - Professional typography and styling")
    print("  - Publication-quality 300 DPI resolution")
    print("="*80 + "\n")
