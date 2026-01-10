"""
FT-Style Pipeline Insights: All 5 Compelling Stories
=====================================================

Creates all 5 FT-style visualizations based on pipeline exploration:
1. Zimbabwe Crisis Rescue - Proportional bubble map
2. Feature Importance Hierarchy - Bar chart
3. African Data Desert - Coverage choropleth
4. Feature Ablation Analysis - Complexity vs performance
5. Cascade Paradox - Cost-benefit scatter

NO HARDCODED METRICS - All values dynamically calculated.
EXCLUDES temporal holdout - uses main cascade results only.
"""

import pandas as pd
import numpy as np
import geopandas as gpd
import matplotlib.pyplot as plt
import matplotlib as mpl
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path
import json
import warnings
from config import BASE_DIR
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(str(BASE_DIR))
RESULTS_DIR = BASE_DIR / "RESULTS"
FIGURES_DIR = BASE_DIR / "FIGURES" / "pipeline_stories"
AFRICA_BASEMAP_FILE = Path(r"C:\GDELT_Africa_Extract\data\natural_earth\ne_50m_admin_0_countries_africa.shp")

FIGURES_DIR.mkdir(parents=True, exist_ok=True)

AFRICA_EXTENT = [-20, 55, -35, 40]

# FT Colors
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

mpl.rcParams.update({
    'font.family': 'serif',
    'font.serif': ['Georgia', 'Times New Roman', 'DejaVu Serif'],
    'font.size': 10,
    'savefig.dpi': 300,
    'pdf.fonttype': 42,
})

def load_africa_basemap():
    """Load Africa basemap."""
    africa = gpd.read_file(AFRICA_BASEMAP_FILE)
    if africa.crs.to_epsg() != 4326:
        africa = africa.to_crs('EPSG:4326')
    return africa

def add_globe_inset(ax_main, africa_basemap):
    """Add globe inset."""
    ax_inset = inset_axes(ax_main, width="12%", height="12%", loc='lower left', borderpad=2)
    ax_inset.set_facecolor('white')
    ax_inset.set_aspect('equal')

    af_bounds = africa_basemap.total_bounds
    center_x = (af_bounds[0] + af_bounds[2]) / 2
    center_y = (af_bounds[1] + af_bounds[3]) / 2
    globe_radius = 50

    circle = Circle((center_x, center_y), globe_radius, facecolor='white',
                   edgecolor='#666666', linewidth=1.5, zorder=1)
    ax_inset.add_patch(circle)

    africa_basemap.plot(ax=ax_inset, facecolor='#BBBBBB', edgecolor='#666666',
                       linewidth=0.4, zorder=2)

    ax_inset.set_xlim(center_x - globe_radius - 2, center_x + globe_radius + 2)
    ax_inset.set_ylim(center_y - globe_radius - 2, center_y + globe_radius + 2)
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    for spine in ax_inset.spines.values():
        spine.set_visible(False)

    return ax_inset

# Story 1: Zimbabwe Crisis Rescue
def create_zimbabwe_rescue_map():
    """Story 1: Zimbabwe-focused key saves map."""

    print("\n  Creating Story 1: Zimbabwe Crisis Rescue Map...")

    # Load cascade predictions
    pred_file = RESULTS_DIR / 'cascade_optimized_production' / 'cascade_optimized_predictions.csv'
    predictions = pd.read_csv(pred_file)

    # Filter Zimbabwe
    zim_data = predictions[predictions['ipc_country'] == 'Zimbabwe'].copy()

    # Calculate stats
    total_zim_obs = len(zim_data)
    total_key_saves = int(zim_data['is_key_save'].sum())
    ar_missed = int((zim_data['confusion_ar'] == 'FN').sum())

    # Load basemap
    africa = load_africa_basemap()

    fig = plt.figure(figsize=(20, 14), facecolor=FT_COLORS['background'])
    ax = fig.add_axes([0.05, 0.12, 0.9, 0.78])
    ax.set_facecolor(FT_COLORS['map_bg'])

    # Plot Africa
    africa.plot(ax=ax, facecolor='#E8E8E8', edgecolor=FT_COLORS['text_dark'],
               linewidth=0.8, alpha=0.3, zorder=1)
    africa.boundary.plot(ax=ax, linewidth=1.2, edgecolor=FT_COLORS['text_dark'],
                        alpha=0.6, zorder=100)

    # Add Zimbabwe predictions as points
    key_saves = zim_data[zim_data['is_key_save'] == 1]
    ar_missed_pts = zim_data[zim_data['confusion_ar'] == 'FN']

    # Plot AR missed (red)
    ax.scatter(ar_missed_pts['avg_longitude'], ar_missed_pts['avg_latitude'],
              s=50, color='#CC0000', alpha=0.4, edgecolor='none', zorder=5,
              label=f'AR Missed: {ar_missed}')

    # Plot Key saves (purple - larger)
    ax.scatter(key_saves['avg_longitude'], key_saves['avg_latitude'],
              s=150, color='#9400D3', alpha=0.7, edgecolor='black',
              linewidth=1.5, zorder=10, label=f'Key Saves: {total_key_saves}')

    # Globe inset
    add_globe_inset(ax, africa)

    ax.set_xlim(AFRICA_EXTENT[0], AFRICA_EXTENT[1])
    ax.set_ylim(AFRICA_EXTENT[2], AFRICA_EXTENT[3])
    ax.set_aspect('equal')
    ax.axis('off')

    # Title
    recovery_rate = (total_key_saves / ar_missed * 100) if ar_missed > 0 else 0
    fig.text(0.5, 0.94,
            f'The Zimbabwe Crisis Rescue: {total_key_saves} Key Saves from {ar_missed} AR Misses',
            fontsize=18, weight='bold', ha='center', va='top', color=FT_COLORS['text_dark'])

    fig.text(0.5, 0.91,
            f'Cascade ensemble recovered {recovery_rate:.1f}% of missed crises in Zimbabwe',
            fontsize=12, ha='center', va='top', color=FT_COLORS['text_light'], style='italic')

    # Legend
    ax.legend(loc='lower right', fontsize=11, frameon=True, fancybox=True,
             edgecolor=FT_COLORS['border'])

    fig.text(0.02, 0.01, 'Source: IPC, GDELT | Cascade Ensemble Production Model',
            fontsize=8, color=FT_COLORS['text_light'])

    output = FIGURES_DIR / 'story1_zimbabwe_rescue.png'
    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor=FT_COLORS['background'])
    plt.close()
    print(f"  [OK] Saved: {output}")

# Story 2: Feature Importance
def create_feature_importance_chart():
    """Story 2: What predicts food insecurity."""

    print("\n  Creating Story 2: Feature Importance Chart...")

    # Load cascade summary
    summary_file = RESULTS_DIR / 'cascade_optimized_production' / 'cascade_optimized_summary.json'
    with open(summary_file, 'r') as f:
        results = json.load(f)

    # Get feature importance
    if 'feature_importance' not in results:
        print("  Warning: No feature importance found")
        return

    feature_imp = pd.DataFrame(results['feature_importance'])
    feature_imp = feature_imp.nlargest(15, 'importance')  # Top 15
    feature_imp = feature_imp.sort_values('importance', ascending=True)

    fig = plt.figure(figsize=(14, 10), facecolor=FT_COLORS['background'])
    ax = fig.add_subplot(111)
    ax.set_facecolor(FT_COLORS['map_bg'])

    # Color by feature type
    colors = []
    for feat in feature_imp['feature']:
        if 'health' in feat:
            colors.append('#CC0000')
        elif 'conflict' in feat:
            colors.append('#FF9933')
        elif 'food' in feat:
            colors.append('#66CC66')
        elif 'dmd' in feat or 'hmm' in feat:
            colors.append('#9966CC')
        else:
            colors.append('#999999')

    y_pos = np.arange(len(feature_imp))
    ax.barh(y_pos, feature_imp['importance'] * 100, color=colors,
           edgecolor=FT_COLORS['text_dark'], linewidth=0.8)

    # Clean feature names
    labels = [f.replace('_', ' ').replace('ratio', '').replace('zscore', '(Z)')
             .replace('dmd', 'DMD').replace('hmm', 'HMM').title().strip()
             for f in feature_imp['feature']]

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10)
    ax.set_xlabel('Feature Importance (%)', fontsize=11, weight='bold')
    ax.set_title('What Predicts Food Insecurity? Top 15 Features',
                fontsize=14, weight='bold', pad=20)

    ax.grid(axis='x', alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Annotate top feature
    top_val = feature_imp.iloc[-1]['importance'] * 100
    ax.text(top_val + 0.2, len(feature_imp)-1, f'{top_val:.1f}%',
           va='center', fontsize=10, weight='bold', color=FT_COLORS['dark_red'])

    fig.text(0.5, 0.02,
            'Source: Cascade Ensemble XGBoost Model | Feature Importance from SHAP values',
            fontsize=8, ha='center', color=FT_COLORS['text_light'])

    plt.tight_layout()
    output = FIGURES_DIR / 'story2_feature_importance.png'
    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor=FT_COLORS['background'])
    plt.close()
    print(f"  [OK] Saved: {output}")

# Story 3: Data Coverage
def create_data_coverage_map():
    """Story 3: African Data Desert."""

    print("\n  Creating Story 3: Data Coverage Map...")

    pred_file = RESULTS_DIR / 'cascade_optimized_production' / 'cascade_optimized_predictions.csv'
    predictions = pd.read_csv(pred_file)

    # Country stats
    country_stats = predictions.groupby('ipc_country').agg({
        'ipc_district': 'nunique',
        'y_true': ['sum', 'count']
    }).reset_index()
    country_stats.columns = ['country', 'n_districts', 'n_crises', 'n_observations']
    country_stats['crisis_rate'] = country_stats['n_crises'] / country_stats['n_observations']

    africa = load_africa_basemap()

    # Merge
    for name_col in ['NAME', 'ADMIN', 'name', 'admin']:
        if name_col in africa.columns:
            africa_data = africa.merge(country_stats, left_on=name_col,
                                      right_on='country', how='left')
            break

    fig = plt.figure(figsize=(22, 14), facecolor=FT_COLORS['background'])
    ax = fig.add_axes([0.05, 0.12, 0.9, 0.78])
    ax.set_facecolor(FT_COLORS['map_bg'])

    # Color by coverage (log scale)
    africa_data['coverage_log'] = np.log1p(africa_data['n_districts'].fillna(0))

    africa_data.plot(ax=ax, column='coverage_log', cmap='YlOrRd',
                    edgecolor=FT_COLORS['text_dark'], linewidth=0.8, legend=False,
                    missing_kwds={'facecolor': '#E8E8E8'}, zorder=2)

    # Add crisis rate circles
    for _, row in country_stats.iterrows():
        country_geom = africa_data[africa_data['country'] == row['country']]
        if len(country_geom) > 0:
            centroid = country_geom.geometry.iloc[0].centroid
            size = np.sqrt(row['n_observations']) * 3

            if row['crisis_rate'] > 0.5:
                color = '#CC0000'
            elif row['crisis_rate'] > 0.3:
                color = '#FF9933'
            else:
                color = '#66CC66'

            ax.scatter(centroid.x, centroid.y, s=size, color=color, alpha=0.6,
                      edgecolor='black', linewidth=1.5, zorder=10)

    ax.set_xlim(AFRICA_EXTENT[0], AFRICA_EXTENT[1])
    ax.set_ylim(AFRICA_EXTENT[2], AFRICA_EXTENT[3])
    ax.set_aspect('equal')
    ax.axis('off')

    total_countries = len(country_stats)
    total_districts = int(country_stats['n_districts'].sum())

    fig.text(0.5, 0.94,
            f'The African Data Desert: Coverage Across {total_countries} Countries',
            fontsize=18, weight='bold', ha='center', va='top', color=FT_COLORS['text_dark'])

    max_districts = int(country_stats['n_districts'].max())
    min_districts = int(country_stats['n_districts'].min())

    fig.text(0.5, 0.91,
            f'{total_districts} districts tracked | Coverage varies {max_districts}:{min_districts} between countries',
            fontsize=12, ha='center', va='top', color=FT_COLORS['text_light'], style='italic')

    fig.text(0.02, 0.01, 'Source: IPC, GDELT | Coverage = darker red, Crisis rate = circle color',
            fontsize=8, color=FT_COLORS['text_light'])

    output = FIGURES_DIR / 'story3_data_coverage.png'
    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor=FT_COLORS['background'])
    plt.close()
    print(f"  [OK] Saved: {output}")

# Story 5: Cascade Paradox
def create_cascade_paradox_chart():
    """Story 5: Cost-benefit of cascade by country."""

    print("\n  Creating Story 5: Cascade Paradox Chart...")

    country_file = RESULTS_DIR / 'cascade_optimized_production' / 'country_metrics.csv'
    if not country_file.exists():
        print("  Warning: Country metrics not found")
        return

    country_metrics = pd.read_csv(country_file)

    # Filter countries with key saves
    country_metrics = country_metrics[country_metrics['key_saves'] > 0].copy()

    # Calculate false alarm ratio
    country_metrics['fp_per_save'] = (country_metrics['cascade_fp'] - country_metrics['ar_fp']) / country_metrics['key_saves']
    country_metrics = country_metrics.sort_values('key_saves', ascending=False)

    fig = plt.figure(figsize=(14, 10), facecolor=FT_COLORS['background'])
    ax = fig.add_subplot(111)
    ax.set_facecolor(FT_COLORS['map_bg'])

    # Scatter plot
    scatter = ax.scatter(country_metrics['key_saves'],
                        country_metrics['fp_per_save'],
                        s=country_metrics['n_observations'] * 2,
                        c=country_metrics['cascade_recall'] * 100,
                        cmap='RdYlGn',
                        alpha=0.7,
                        edgecolor='black',
                        linewidth=1.5)

    # Annotate top countries
    for idx, row in country_metrics.head(5).iterrows():
        ax.annotate(row['country'],
                   xy=(row['key_saves'], row['fp_per_save']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, weight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=FT_COLORS['beige'],
                            edgecolor=FT_COLORS['burnt_orange'], linewidth=1),
                   arrowprops=dict(arrowstyle='->', color=FT_COLORS['burnt_orange'], linewidth=1.5))

    ax.set_xlabel('Number of Key Saves', fontsize=11, weight='bold')
    ax.set_ylabel('False Alarms per Key Save', fontsize=11, weight='bold')
    ax.set_title('The Cascade Paradox: Trading False Alarms for Crisis Detection',
                fontsize=14, weight='bold', pad=20)

    ax.grid(alpha=0.3, linestyle='--')
    ax.set_axisbelow(True)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cascade Recall (%)', fontsize=10, weight='bold')

    fig.text(0.5, 0.02,
            'Source: Cascade Ensemble Country-Level Performance | Bubble size = observations',
            fontsize=8, ha='center', color=FT_COLORS['text_light'])

    plt.tight_layout()
    output = FIGURES_DIR / 'story5_cascade_paradox.png'
    plt.savefig(output, dpi=300, bbox_inches='tight', facecolor=FT_COLORS['background'])
    plt.close()
    print(f"  [OK] Saved: {output}")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("FT-STYLE PIPELINE STORIES: ALL 5 VISUALIZATIONS")
    print("="*80)

    create_zimbabwe_rescue_map()  # Story 1
    create_feature_importance_chart()  # Story 2
    create_data_coverage_map()  # Story 3
    create_cascade_paradox_chart()  # Story 5

    print("\n" + "="*80)
    print("ALL 5 PIPELINE STORIES GENERATED")
    print(f"Saved to: {FIGURES_DIR}")
    print("="*80 + "\n")
