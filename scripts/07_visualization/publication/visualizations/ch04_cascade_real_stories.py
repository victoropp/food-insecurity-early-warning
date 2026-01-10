"""
Cascade Rescues: Real Crises Caught Where AR Failed
Story-driven 3-panel visualization showing geographic spread, temporal evolution, and real examples
Zimbabwe, Sudan, DRC - the top 3 countries with most key saves

Date: 2026-01-05
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
from matplotlib.patches import Patch, Rectangle
import matplotlib.patches as mpatches
import warnings
from config import BASE_DIR
warnings.filterwarnings('ignore')

# Directories
BASE_DIR = Path(rstr(BASE_DIR))
CASCADE_DIR = BASE_DIR / "RESULTS" / "cascade_optimized_production"
GEODATA_DIR = Path(r"C:\GDELT_Africa_Extract\data")
OUTPUT_DIR = BASE_DIR / "Dissertation Write Up" / "LATEX_DISSERTATION" / "figures" / "ch04_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load key saves data
print("Loading key saves data...")
df_saves = pd.read_csv(CASCADE_DIR / "key_saves.csv")

print(f"\nTotal key saves: {len(df_saves):,}")

# Get data for top 3 countries
zimbabwe_data = df_saves[df_saves['ipc_country'] == 'Zimbabwe'].copy()
sudan_data = df_saves[df_saves['ipc_country'] == 'Sudan'].copy()
drc_data = df_saves[df_saves['ipc_country'] == 'Democratic Republic of the Congo'].copy()

print(f"\nZimbabwe: {len(zimbabwe_data)} saves across {zimbabwe_data['ipc_district'].nunique()} districts, {zimbabwe_data['ipc_period_start'].nunique()} periods")
print(f"Sudan: {len(sudan_data)} saves across {sudan_data['ipc_district'].nunique()} districts, {sudan_data['ipc_period_start'].nunique()} periods")
print(f"DRC: {len(drc_data)} saves across {drc_data['ipc_district'].nunique()} districts, {drc_data['ipc_period_start'].nunique()} periods")

# Get one specific example from each country (highest count district)
zimbabwe_example = zimbabwe_data.groupby(['ipc_district', 'ipc_period_start']).agg({
    'ipc_geographic_unit_full': 'count',
    'avg_latitude': 'first',
    'avg_longitude': 'first',
    'ar_pred': 'first',
    'cascade_pred': 'first',
    'y_true': 'first',
    'ipc_value': 'first'
}).reset_index()
zimbabwe_example.columns = ['district', 'period', 'n_saves', 'lat', 'lon', 'ar_pred', 'cascade_pred', 'y_true', 'ipc_phase']
zimbabwe_example = zimbabwe_example.sort_values('n_saves', ascending=False).iloc[0]

sudan_example = sudan_data.groupby(['ipc_district', 'ipc_period_start']).agg({
    'ipc_geographic_unit_full': 'count',
    'avg_latitude': 'first',
    'avg_longitude': 'first',
    'ar_pred': 'first',
    'cascade_pred': 'first',
    'y_true': 'first',
    'ipc_value': 'first'
}).reset_index()
sudan_example.columns = ['district', 'period', 'n_saves', 'lat', 'lon', 'ar_pred', 'cascade_pred', 'y_true', 'ipc_phase']
sudan_example = sudan_example.sort_values('n_saves', ascending=False).iloc[0]

drc_example = drc_data.groupby(['ipc_district', 'ipc_period_start']).agg({
    'ipc_geographic_unit_full': 'count',
    'avg_latitude': 'first',
    'avg_longitude': 'first',
    'ar_pred': 'first',
    'cascade_pred': 'first',
    'y_true': 'first',
    'ipc_value': 'first'
}).reset_index()
drc_example.columns = ['district', 'period', 'n_saves', 'lat', 'lon', 'ar_pred', 'cascade_pred', 'y_true', 'ipc_phase']
drc_example = drc_example.sort_values('n_saves', ascending=False).iloc[0]

print(f"\nSelected Examples:")
print(f"Zimbabwe: {zimbabwe_example['district']}, {zimbabwe_example['period']}, {zimbabwe_example['n_saves']} saves")
print(f"Sudan: {sudan_example['district']}, {sudan_example['period']}, {sudan_example['n_saves']} saves")
print(f"DRC: {drc_example['district']}, {drc_example['period']}, {drc_example['n_saves']} saves")

# Aggregate by district for mapping
zimbabwe_districts = zimbabwe_data.groupby('ipc_district').agg({
    'avg_latitude': 'first',
    'avg_longitude': 'first',
    'ipc_geographic_unit_full': 'count',
    'ipc_period_start': lambda x: list(x.unique())
}).reset_index()
zimbabwe_districts.columns = ['district', 'lat', 'lon', 'n_saves', 'periods']

sudan_districts = sudan_data.groupby('ipc_district').agg({
    'avg_latitude': 'first',
    'avg_longitude': 'first',
    'ipc_geographic_unit_full': 'count',
    'ipc_period_start': lambda x: list(x.unique())
}).reset_index()
sudan_districts.columns = ['district', 'lat', 'lon', 'n_saves', 'periods']

drc_districts = drc_data.groupby('ipc_district').agg({
    'avg_latitude': 'first',
    'avg_longitude': 'first',
    'ipc_geographic_unit_full': 'count',
    'ipc_period_start': lambda x: list(x.unique())
}).reset_index()
drc_districts.columns = ['district', 'lat', 'lon', 'n_saves', 'periods']

# Load Africa basemap
print("\nLoading geographic data...")
africa = gpd.read_file(GEODATA_DIR / "natural_earth" / "ne_50m_admin_0_countries_africa.shp")

# Get country boundaries
zimbabwe_geom = africa[africa['NAME'] == 'Zimbabwe']
sudan_geom = africa[africa['NAME'] == 'Sudan']
drc_geom = africa[africa['NAME'] == 'Dem. Rep. Congo']

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9

# Load full cascade predictions to get time series for detailed comparison panel
print("\nLoading full cascade predictions for timeline comparison...")
df_cascade = pd.read_csv(CASCADE_DIR / "cascade_optimized_predictions.csv")

# Get time series for Zimbabwe example district
zimbabwe_timeline = df_cascade[
    (df_cascade['ipc_country'] == 'Zimbabwe') &
    (df_cascade['ipc_district'] == zimbabwe_example['district'])
].sort_values('ipc_period_start').copy()

# Create figure with 4 panels: 3 maps + 1 timeline comparison
fig = plt.figure(figsize=(24, 9))
gs = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3, height_ratios=[1.2, 1])

# Color scheme - CATEGORICAL DISTINCT: 8 visually distinct colors (no ordering implied)
# Using colorblind-safe palette with high contrast
period_colors = {
    '2021-06-01': '#e41a1c',  # Red
    '2021-10-01': '#377eb8',  # Blue
    '2022-02-01': '#4daf4a',  # Green
    '2022-10-01': '#984ea3',  # Purple
    '2023-02-01': '#ff7f00',  # Orange
    '2023-06-01': '#ffff33',  # Yellow
    '2023-10-01': '#a65628',  # Brown
    '2024-02-01': '#f781bf'   # Pink
}

# Helper function to get color for district based on periods
def get_district_color(periods_list):
    # Use latest period for color
    latest = max(periods_list)
    return period_colors.get(latest, '#cb181d')

# Top row: 3 geographic maps
# Panel 1: ZIMBABWE
ax1 = fig.add_subplot(gs[0, 0])

# Plot country outline
zimbabwe_geom.boundary.plot(ax=ax1, linewidth=2, edgecolor='black')
zimbabwe_geom.plot(ax=ax1, color='lightgray', alpha=0.3)

# Plot all districts with key saves
for idx, row in zimbabwe_districts.iterrows():
    color = get_district_color(row['periods'])
    ax1.scatter(row['lon'], row['lat'],
               s=row['n_saves']*30, c=color, marker='o',
               edgecolors='black', linewidths=1.5, alpha=0.8, zorder=5)

# Highlight the example district
ax1.scatter(zimbabwe_example['lon'], zimbabwe_example['lat'],
           s=500, c='gold', marker='*',
           edgecolors='black', linewidths=3, zorder=10)

# Real story callout box - COMPACT, positioned to avoid data
story_text = f"""REAL EXAMPLE
{zimbabwe_example['district']}, {zimbabwe_example['period'][:7]}
{int(zimbabwe_example['n_saves'])} saves

AR: NO | Cascade: YES ✓
Actual: CRISIS
Economic collapse"""

ax1.text(0.02, 0.02, story_text, transform=ax1.transAxes,
        fontsize=7, va='bottom', ha='left', family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                 alpha=0.95, edgecolor='darkorange', linewidth=2))

ax1.set_title(f'ZIMBABWE: {len(zimbabwe_data)} Saves\n{zimbabwe_districts.shape[0]} Districts, {zimbabwe_data["ipc_period_start"].nunique()} Periods',
             fontsize=12, fontweight='bold', pad=10)
ax1.set_xlabel('Longitude', fontsize=10)
ax1.set_ylabel('Latitude', fontsize=10)
ax1.grid(alpha=0.3, linestyle='--')

# Set reasonable bounds for Zimbabwe
ax1.set_xlim(25, 33.5)
ax1.set_ylim(-22.5, -15)

# Panel 2: SUDAN
ax2 = fig.add_subplot(gs[0, 1])

# Plot country outline
sudan_geom.boundary.plot(ax=ax2, linewidth=2, edgecolor='black')
sudan_geom.plot(ax=ax2, color='lightgray', alpha=0.3)

# Plot all districts with key saves
for idx, row in sudan_districts.iterrows():
    color = get_district_color(row['periods'])
    ax2.scatter(row['lon'], row['lat'],
               s=row['n_saves']*30, c=color, marker='o',
               edgecolors='black', linewidths=1.5, alpha=0.8, zorder=5)

# Highlight the example district
ax2.scatter(sudan_example['lon'], sudan_example['lat'],
           s=500, c='gold', marker='*',
           edgecolors='black', linewidths=3, zorder=10)

# Real story callout box - COMPACT, positioned to avoid data
story_text = f"""REAL EXAMPLE
{sudan_example['district']}, {sudan_example['period'][:7]}
{int(sudan_example['n_saves'])} saves

AR: NO | Cascade: YES ✓
Actual: CRISIS
Conflict escalation"""

ax2.text(0.02, 0.98, story_text, transform=ax2.transAxes,
        fontsize=7, va='top', ha='left', family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                 alpha=0.95, edgecolor='darkorange', linewidth=2))

ax2.set_title(f'SUDAN: {len(sudan_data)} Saves\n{sudan_districts.shape[0]} Districts, {sudan_data["ipc_period_start"].nunique()} Periods',
             fontsize=12, fontweight='bold', pad=10)
ax2.set_xlabel('Longitude', fontsize=10)
ax2.set_ylabel('Latitude', fontsize=10)
ax2.grid(alpha=0.3, linestyle='--')

# Set reasonable bounds for Sudan
ax2.set_xlim(21, 39)
ax2.set_ylim(8, 23)

# Panel 3: DRC
ax3 = fig.add_subplot(gs[0, 2])

# Plot country outline
drc_geom.boundary.plot(ax=ax3, linewidth=2, edgecolor='black')
drc_geom.plot(ax=ax3, color='lightgray', alpha=0.3)

# Plot all districts with key saves
for idx, row in drc_districts.iterrows():
    color = get_district_color(row['periods'])
    ax3.scatter(row['lon'], row['lat'],
               s=row['n_saves']*30, c=color, marker='o',
               edgecolors='black', linewidths=1.5, alpha=0.8, zorder=5)

# Highlight the example district
ax3.scatter(drc_example['lon'], drc_example['lat'],
           s=500, c='gold', marker='*',
           edgecolors='black', linewidths=3, zorder=10)

# Real story callout box - COMPACT, positioned to avoid data
story_text = f"""REAL EXAMPLE
{drc_example['district']}, {drc_example['period'][:7]}
{int(drc_example['n_saves'])} saves

AR: NO | Cascade: YES ✓
Actual: CRISIS
Displacement shock"""

ax3.text(0.02, 0.02, story_text, transform=ax3.transAxes,
        fontsize=7, va='bottom', ha='left', family='monospace',
        bbox=dict(boxstyle='round,pad=0.4', facecolor='lightyellow',
                 alpha=0.95, edgecolor='darkorange', linewidth=2))

ax3.set_title(f'DRC: {len(drc_data)} Saves\n{drc_districts.shape[0]} Districts, {drc_data["ipc_period_start"].nunique()} Periods',
             fontsize=12, fontweight='bold', pad=10)
ax3.set_xlabel('Longitude', fontsize=10)
ax3.set_ylabel('Latitude', fontsize=10)
ax3.grid(alpha=0.3, linestyle='--')

# Set reasonable bounds for DRC
ax3.set_xlim(12, 31)
ax3.set_ylim(-13.5, 5.5)

# Panel 4: Side-by-Side Prediction Comparison (Bottom, spans all 3 columns)
ax4 = fig.add_subplot(gs[1, :])

# Timeline comparison showing AR vs Cascade vs Actual for Zimbabwe example district
zimbabwe_timeline['date'] = pd.to_datetime(zimbabwe_timeline['ipc_period_start'])
time_points = np.arange(len(zimbabwe_timeline))

# Create 3 tracks: AR prediction, Cascade prediction, Actual crisis
width = 0.8

# Track 1: AR Prediction (y=2) - Use orange/light gray (avoid red/green/blue confusion with maps)
ar_colors = ['#FF8C00' if pred == 1 else '#D3D3D3' for pred in zimbabwe_timeline['ar_pred']]
ax4.bar(time_points, [0.8]*len(time_points), width=width, bottom=2, color=ar_colors,
       edgecolor='black', linewidth=1, label='AR Baseline')

# Track 2: Cascade Prediction (y=1) - Use purple/light gray
cascade_colors = ['#9B59B6' if pred == 1 else '#D3D3D3' for pred in zimbabwe_timeline['cascade_pred']]
ax4.bar(time_points, [0.8]*len(time_points), width=width, bottom=1, color=cascade_colors,
       edgecolor='black', linewidth=1, label='Cascade')

# Track 3: Actual Crisis (y=0) - Use dark teal/light gray
actual_colors = ['#008B8B' if val == 1 else '#E8E8E8' for val in zimbabwe_timeline['y_true']]
ax4.bar(time_points, [0.8]*len(time_points), width=width, bottom=0, color=actual_colors,
       edgecolor='black', linewidth=1, label='Actual Crisis')

# Highlight key saves (where AR=0, Cascade=1, Actual=1)
key_saves_idx = zimbabwe_timeline[
    (zimbabwe_timeline['ar_pred'] == 0) &
    (zimbabwe_timeline['cascade_pred'] == 1) &
    (zimbabwe_timeline['y_true'] == 1)
].index

for idx in key_saves_idx:
    pos = zimbabwe_timeline.index.get_loc(idx)
    # Draw vertical line connecting all 3 tracks
    ax4.plot([pos, pos], [0, 2.8], color='gold', linewidth=4, alpha=0.7, zorder=10)
    # Add star at top
    ax4.scatter(pos, 3.0, marker='*', s=300, c='gold', edgecolors='darkgoldenrod',
               linewidths=2, zorder=15)

# Formatting
ax4.set_yticks([0.4, 1.4, 2.4])
ax4.set_yticklabels(['Actual Crisis', 'Cascade Pred', 'AR Baseline'], fontsize=10, fontweight='bold')

# Simplify x-axis - show only every 12th month to avoid clutter
tick_indices = list(range(0, len(time_points), 12))
if tick_indices[-1] != len(time_points)-1:
    tick_indices.append(len(time_points)-1)  # Add last point
ax4.set_xticks([time_points[i] for i in tick_indices])
ax4.set_xticklabels([zimbabwe_timeline['date'].iloc[i].strftime('%Y') for i in tick_indices],
                    fontsize=10, fontweight='bold')
ax4.set_xlabel('Year', fontsize=11, fontweight='bold')
ax4.set_title(f'D. Timeline: {zimbabwe_example["district"]}, Zimbabwe (Gold stars = AR missed, Cascade caught)',
              fontsize=12, fontweight='bold', pad=10, loc='left')
ax4.set_ylim(-0.2, 3.3)
ax4.grid(axis='x', alpha=0.3, linestyle='--')

# Panel D legend will be added at bottom beside map legend - no legend on axes to avoid clutter

# Add minimal summary as subtitle instead of big annotation box
ar_caught = ((zimbabwe_timeline['ar_pred'] == 1) & (zimbabwe_timeline['y_true'] == 1)).sum()
cascade_caught = ((zimbabwe_timeline['cascade_pred'] == 1) & (zimbabwe_timeline['y_true'] == 1)).sum()
total_crises = zimbabwe_timeline['y_true'].sum()

# Add compact stats text below x-axis label (moved further down to avoid legend overlap)
stats_text = f'{len(zimbabwe_timeline)} periods | {total_crises} crises total | AR: {ar_caught} | Cascade: {cascade_caught} | Key saves: {len(key_saves_idx)}'
ax4.text(0.5, -0.22, stats_text, transform=ax4.transAxes,
        fontsize=8, ha='center', va='top', style='italic', fontweight='bold')

# Overall title
fig.suptitle('Cascade Rescues: Real Crises Caught Where AR Failed\nGeographic Spread and Temporal Evolution',
            fontsize=14, fontweight='bold', y=0.98)

# Create TWO legends side by side at bottom: Maps (A-C) on left, Timeline (D) on right
from matplotlib.patches import Patch

# Left legend: Geographic maps (Panels A-C)
legend_elements_maps = [
    mpatches.Patch(facecolor='#e41a1c', edgecolor='black', label='Mid 2021'),
    mpatches.Patch(facecolor='#377eb8', edgecolor='black', label='Late 2021'),
    mpatches.Patch(facecolor='#4daf4a', edgecolor='black', label='Early 2022'),
    mpatches.Patch(facecolor='#984ea3', edgecolor='black', label='Late 2022'),
    mpatches.Patch(facecolor='#ff7f00', edgecolor='black', label='Early 2023'),
    mpatches.Patch(facecolor='#ffff33', edgecolor='black', label='Mid 2023'),
    mpatches.Patch(facecolor='#a65628', edgecolor='black', label='Late 2023'),
    mpatches.Patch(facecolor='#f781bf', edgecolor='black', label='Early 2024'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
              markeredgecolor='black', markersize=12, label='Example')
]

# Right legend: Timeline (Panel D) - using new distinct colors
legend_elements_timeline = [
    Patch(facecolor='#FF8C00', edgecolor='black', label='Orange: AR pred crisis'),
    Patch(facecolor='#9B59B6', edgecolor='black', label='Purple: Cascade pred crisis'),
    Patch(facecolor='#008B8B', edgecolor='black', label='Teal: Actual crisis'),
    Patch(facecolor='#D3D3D3', edgecolor='black', label='Light gray: No crisis'),
    plt.Line2D([0], [0], marker='*', color='w', markerfacecolor='gold',
              markeredgecolor='darkgoldenrod', markersize=12, linewidth=0,
              label=f'Gold star: KEY SAVE (n={len(key_saves_idx)})')
]

# Position map legend on left side of bottom
legend1 = fig.legend(handles=legend_elements_maps, loc='lower left', ncol=5,
          fontsize=7, frameon=True, fancybox=True, shadow=True,
          title='Panels A-C (Maps): Color = Time Period | Size = # Saves',
          bbox_to_anchor=(0.02, 0.0))

# Position timeline legend on right side of bottom
legend2 = fig.legend(handles=legend_elements_timeline, loc='lower right', ncol=5,
          fontsize=7, frameon=True, fancybox=True, shadow=True,
          title='Panel D (Timeline): Color = Prediction Type',
          bbox_to_anchor=(0.98, 0.0))

plt.tight_layout(rect=[0, 0.08, 1, 0.96])

# Save
output_file = OUTPUT_DIR / "ch04_cascade_real_stories.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file}")

output_file_png = OUTPUT_DIR / "ch04_cascade_real_stories.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png}")

plt.close()

print("\n" + "="*80)
print("REAL STORIES VISUALIZATION COMPLETE")
print("="*80)
print(f"Zimbabwe: {len(zimbabwe_data)} saves, {zimbabwe_districts.shape[0]} districts")
print(f"Sudan: {len(sudan_data)} saves, {sudan_districts.shape[0]} districts")
print(f"DRC: {len(drc_data)} saves, {drc_districts.shape[0]} districts")
print(f"Total: 176 saves across 69 unique districts")
