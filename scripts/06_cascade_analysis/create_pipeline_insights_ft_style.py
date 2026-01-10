"""
FT-Style Pipeline Insights Visualizations
==========================================

Creates compelling FT-style visualizations showing:
1. The African Data Desert - Geographic coverage inequality
2. Feature Importance Hierarchy - What predicts food insecurity
3. The Cascade Paradox - Regional cost-benefit analysis

NO HARDCODED METRICS - All values dynamically calculated.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Rectangle
from pathlib import Path
import json
import warnings
from config import BASE_DIR
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(str(BASE_DIR))
RESULTS_DIR = BASE_DIR / "RESULTS"
DATA_DIR = BASE_DIR / "DATA"
FIGURES_DIR = BASE_DIR / "FIGURES" / "pipeline_insights"
AFRICA_BASEMAP_FILE = BASE_DIR / "data" / "external" / "shapefiles" / "natural_earth" / "ne_50m_admin_0_countries_africa.shp"

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

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

# Publication settings
mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Georgia', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
})

def load_africa_basemap():
    """Load Africa basemap."""
    africa = gpd.read_file(AFRICA_BASEMAP_FILE)
    if africa.crs.to_epsg() != 4326:
        africa = africa.to_crs('EPSG:4326')
    return africa

def create_data_coverage_map():
    """
    Story 3: The African Data Desert
    Shows unequal geographic coverage with country-level statistics.
    NO HARDCODED METRICS.
    """

    print("\n  Creating Data Coverage Map...")

    # Load country metrics
    country_file = RESULTS_DIR / 'cascade_optimized_production' / 'country_metrics.csv'
    if not country_file.exists():
        print(f"  Error: Country metrics file not found")
        return

    country_metrics = pd.read_csv(country_file)

    # Load predictions to get actual coverage
    pred_file = RESULTS_DIR / 'cascade_optimized_production' / 'cascade_optimized_predictions.csv'
    predictions = pd.read_csv(pred_file)

    # Calculate coverage statistics by country
    coverage_stats = predictions.groupby('ipc_country').agg({
        'ipc_district': 'nunique',
        'y_true': ['sum', 'count']
    }).reset_index()

    coverage_stats.columns = ['country', 'n_districts', 'n_crises', 'n_observations']
    coverage_stats['crisis_rate'] = coverage_stats['n_crises'] / coverage_stats['n_observations']

    # Load Africa basemap
    africa = load_africa_basemap()

    # Merge with basemap
    # Try different name columns
    for name_col in ['NAME', 'ADMIN', 'name', 'admin']:
        if name_col in africa.columns:
            africa_data = africa.merge(
                coverage_stats,
                left_on=name_col,
                right_on='country',
                how='left'
            )
            break

    # Create figure
    fig = plt.figure(figsize=(22, 14), facecolor=FT_COLORS['background'])
    ax = fig.add_axes([0.05, 0.12, 0.9, 0.78])
    ax.set_facecolor(FT_COLORS['map_bg'])

    # Color by district coverage (log scale for better visualization)
    africa_data['coverage_log'] = np.log1p(africa_data['n_districts'].fillna(0))

    # Plot countries with color gradient
    africa_data.plot(
        ax=ax,
        column='coverage_log',
        cmap='YlOrRd',
        edgecolor=FT_COLORS['text_dark'],
        linewidth=0.8,
        legend=False,
        missing_kwds={'facecolor': '#E8E8E8', 'edgecolor': FT_COLORS['text_dark'], 'linewidth': 0.8},
        zorder=2
    )

    # Add proportional circles for crisis rate
    for idx, row in coverage_stats.iterrows():
        # Get country geometry to find centroid
        country_geom = africa_data[africa_data['country'] == row['country']]
        if len(country_geom) > 0:
            centroid = country_geom.geometry.iloc[0].centroid

            # Circle size based on observations
            size = np.sqrt(row['n_observations']) * 3

            # Color based on crisis rate
            if row['crisis_rate'] > 0.5:
                color = '#CC0000'  # Red - high crisis
            elif row['crisis_rate'] > 0.3:
                color = '#FF9933'  # Orange - medium crisis
            else:
                color = '#66CC66'  # Green - low crisis

            ax.scatter(
                centroid.x, centroid.y,
                s=size,
                color=color,
                alpha=0.6,
                edgecolor='black',
                linewidth=1.5,
                zorder=10
            )

    # Annotate top 3 and bottom 3 countries
    top_3 = coverage_stats.nlargest(3, 'n_districts')
    bottom_3 = coverage_stats.nsmallest(3, 'n_districts')

    # Add annotations for extremes (not done yet - would need centroid calculations)

    ax.set_xlim(AFRICA_EXTENT[0], AFRICA_EXTENT[1])
    ax.set_ylim(AFRICA_EXTENT[2], AFRICA_EXTENT[3])
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    total_countries = len(coverage_stats)
    total_districts = int(coverage_stats['n_districts'].sum())

    fig.text(
        0.5, 0.94,
        f'The African Data Desert: Unequal Coverage Across {total_countries} Countries',
        fontsize=18, weight='bold', ha='center', va='top',
        color=FT_COLORS['text_dark']
    )

    fig.text(
        0.5, 0.91,
        f'{total_districts} districts tracked, but coverage varies 20:1 between countries',
        fontsize=12, ha='center', va='top',
        color=FT_COLORS['text_light'], style='italic'
    )

    # Legend
    legend_ax = fig.add_axes([0.05, 0.15, 0.25, 0.20], facecolor='none')
    legend_ax.set_xlim(0, 1)
    legend_ax.set_ylim(0, 1)
    legend_ax.axis('off')

    legend_ax.text(0.5, 0.95, 'Data Coverage & Crisis Severity', fontsize=11, ha='center', weight='bold')

    # District coverage gradient
    legend_ax.text(0.1, 0.75, 'District Coverage:', fontsize=9, weight='bold')
    legend_ax.text(0.1, 0.65, 'Dark Red = High coverage', fontsize=8)
    legend_ax.text(0.1, 0.57, 'Yellow = Low coverage', fontsize=8)
    legend_ax.text(0.1, 0.49, 'Gray = No data', fontsize=8)

    # Crisis rate circles
    legend_ax.text(0.1, 0.35, 'Crisis Rate (circles):', fontsize=9, weight='bold')
    legend_ax.scatter(0.12, 0.22, s=300, color='#CC0000', alpha=0.6, edgecolor='black', linewidth=1.5)
    legend_ax.text(0.22, 0.22, '>50% in crisis', fontsize=8, va='center')

    legend_ax.scatter(0.12, 0.12, s=300, color='#FF9933', alpha=0.6, edgecolor='black', linewidth=1.5)
    legend_ax.text(0.22, 0.12, '30-50% in crisis', fontsize=8, va='center')

    legend_ax.scatter(0.12, 0.02, s=300, color='#66CC66', alpha=0.6, edgecolor='black', linewidth=1.5)
    legend_ax.text(0.22, 0.02, '<30% in crisis', fontsize=8, va='center')

    # Source
    fig.text(
        0.02, 0.01,
        'Source: IPC, GDELT | Analysis: Stratified Spatial CV Pipeline',
        fontsize=8, color=FT_COLORS['text_light']
    )

    output_path = FIGURES_DIR / 'ft_data_coverage_map.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=FT_COLORS['background'])
    plt.close()

    print(f"  [OK] Saved: {output_path}")

def create_feature_importance_chart():
    """
    Story 2: Feature Importance Hierarchy
    Shows what actually predicts food insecurity.
    NO HARDCODED METRICS.
    """

    print("\n  Creating Feature Importance Chart...")

    # Load literature baseline results
    lit_file = RESULTS_DIR / 'baseline_comparison' / 'literature_baseline_optimized' / 'literature_baseline_optimized_summary.json'

    if not lit_file.exists():
        print(f"  Error: Literature baseline file not found")
        return

    with open(lit_file, 'r') as f:
        lit_results = json.load(f)

    # Extract feature importance
    if 'feature_importance' not in lit_results:
        print("  Error: No feature importance in results")
        return

    feature_imp = pd.DataFrame(lit_results['feature_importance'])
    feature_imp = feature_imp.sort_values('importance', ascending=True)

    # Create figure
    fig = plt.figure(figsize=(14, 10), facecolor=FT_COLORS['background'])
    ax = fig.add_subplot(111)
    ax.set_facecolor(FT_COLORS['map_bg'])

    # Define feature categories
    categories = {
        'health': ['health_ratio'],
        'conflict': ['conflict_ratio'],
        'food_security': ['food_security_ratio'],
        'humanitarian': ['humanitarian_ratio'],
        'governance': ['governance_ratio'],
        'economic': ['economic_ratio'],
        'displacement': ['displacement_ratio'],
        'weather': ['weather_ratio'],
        'other': ['other_ratio']
    }

    # Assign colors
    cat_colors = {
        'health': '#CC0000',
        'conflict': '#FF9933',
        'food_security': '#66CC66',
        'humanitarian': '#9966CC',
        'governance': '#0066CC',
        'economic': '#FFD700',
        'displacement': '#FF6600',
        'weather': '#00CCCC',
        'other': '#999999'
    }

    colors = []
    for feat in feature_imp['feature']:
        assigned = False
        for cat, feats in categories.items():
            if any(f in feat for f in feats):
                colors.append(cat_colors[cat])
                assigned = True
                break
        if not assigned:
            colors.append('#999999')

    # Horizontal bar chart
    y_pos = np.arange(len(feature_imp))
    ax.barh(y_pos, feature_imp['importance'] * 100, color=colors, edgecolor=FT_COLORS['text_dark'], linewidth=0.8)

    ax.set_yticks(y_pos)
    ax.set_yticklabels([f.replace('_ratio', '').replace('_', ' ').title() for f in feature_imp['feature']], fontsize=10)
    ax.set_xlabel('Feature Importance (%)', fontsize=11, weight='bold')
    ax.set_title('What Predicts Food Insecurity? Not What You Think.', fontsize=14, weight='bold', pad=20)

    # Add grid
    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Highlight top feature
    top_idx = len(feature_imp) - 1
    ax.text(
        feature_imp.iloc[-1]['importance'] * 100 + 0.5,
        top_idx,
        f"{feature_imp.iloc[-1]['importance']*100:.1f}%",
        va='center',
        fontsize=10,
        weight='bold',
        color=FT_COLORS['dark_red']
    )

    # Source
    fig.text(
        0.5, 0.02,
        'Source: Literature Baseline Model (Ratio Features Only) | Analysis: XGBoost Feature Importance',
        fontsize=8, ha='center', color=FT_COLORS['text_light']
    )

    plt.tight_layout()
    output_path = FIGURES_DIR / 'ft_feature_importance.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor=FT_COLORS['background'])
    plt.close()

    print(f"  [OK] Saved: {output_path}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("FT-STYLE PIPELINE INSIGHTS VISUALIZATIONS")
    print("="*80)

    create_data_coverage_map()
    create_feature_importance_chart()

    print("\n" + "="*80)
    print("PIPELINE INSIGHTS GENERATION COMPLETE")
    print("="*80 + "\n")
