"""
Additional Figure for Chapter 5: Geographic Map of 249 Cascade Rescues
Intuitive real map showing WHERE the cascade breakthrough occurred
Shows district-level key saves across Africa with country highlighting

Date: 2026-01-05
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import geopandas as gpd
from pathlib import Path
import warnings
from config import BASE_DIR
warnings.filterwarnings('ignore')

# Directories
BASE_DIR = Path(rstr(BASE_DIR))
CASCADE_DIR = BASE_DIR / "RESULTS" / "cascade_optimized_production"
GEODATA_DIR = Path(r"C:\GDELT_Africa_Extract\data")
OUTPUT_DIR = BASE_DIR / "Dissertation Write Up" / "LATEX_DISSERTATION" / "figures" / "ch05_discussion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading cascade data...")
key_saves = pd.read_csv(CASCADE_DIR / "key_saves.csv")

print(f"\nTotal key saves: {len(key_saves):,}")

# Aggregate by district
district_saves = key_saves.groupby(['ipc_district', 'ipc_country']).agg({
    'avg_latitude': 'first',
    'avg_longitude': 'first',
    'ipc_geographic_unit_full': 'count'
}).reset_index()
district_saves.columns = ['district', 'country', 'lat', 'lon', 'n_saves']

print(f"Unique districts: {len(district_saves)}")

# Country totals
country_saves = key_saves.groupby('ipc_country').size().sort_values(ascending=False)
print(f"\nTop 5 countries:")
for country, count in country_saves.head(5).items():
    print(f"  {country}: {count}")

# Load Africa basemap
print("\nLoading geographic data...")
africa = gpd.read_file(GEODATA_DIR / "natural_earth" / "ne_50m_admin_0_countries_africa.shp")

# Get top 3 countries with key saves
top3_countries = ['Zimbabwe', 'Sudan', 'Democratic Republic of the Congo']

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# Create figure
fig, ax = plt.subplots(figsize=(16, 14))

# Plot Africa basemap - all countries in light color
africa.plot(ax=ax, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.5)

# Highlight top 3 countries
for country_name in top3_countries:
    if country_name == 'Democratic Republic of the Congo':
        country_geom = africa[africa['NAME'] == 'Dem. Rep. Congo']
    else:
        country_geom = africa[africa['NAME'] == country_name]

    if not country_geom.empty:
        country_geom.plot(ax=ax, color='#FFF9E6', edgecolor='#FFB800', linewidth=2, alpha=0.7)

# Plot district-level key saves as circles sized by count
scatter = ax.scatter(district_saves['lon'], district_saves['lat'],
                     s=district_saves['n_saves']*50,
                     c=district_saves['n_saves'],
                     cmap='YlOrRd',
                     alpha=0.8,
                     edgecolors='darkred',
                     linewidths=1.5,
                     zorder=5)

# Add colorbar
cbar = plt.colorbar(scatter, ax=ax, fraction=0.03, pad=0.02)
cbar.set_label('Number of Key Saves per District', fontsize=11, fontweight='bold')

# Add ALL country labels - small and unobtrusive
for idx, row in africa.iterrows():
    # Get country centroid
    centroid = row.geometry.centroid
    country_name = row['NAME']

    # Use short names for space
    display_name = country_name
    if country_name == 'Dem. Rep. Congo':
        display_name = 'DRC'
    elif country_name == 'Central African Rep.':
        display_name = 'CAR'
    elif country_name == 'Eq. Guinea':
        display_name = 'Eq. Guinea'

    # Add small country label
    ax.text(centroid.x, centroid.y, display_name,
            fontsize=7, ha='center', va='center', color='gray', alpha=0.5)

# Add LARGE labels for top 3 countries - POSITIONED OUTSIDE bubble clusters
# Sudan - position to the right (east) away from bubbles
ax.text(45, 16, 'SUDAN\n59 saves',
        fontsize=13, fontweight='bold', ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='gold',
                 alpha=0.95, edgecolor='darkgoldenrod', linewidth=2.5))

# Arrow pointing to Sudan
ax.annotate('', xy=(30, 15), xytext=(42, 16),
            arrowprops=dict(arrowstyle='->', lw=2, color='darkgoldenrod'))

# DRC - position to the left (west) away from bubbles
ax.text(10, -2, 'DRC\n40 saves',
        fontsize=13, fontweight='bold', ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='gold',
                 alpha=0.95, edgecolor='darkgoldenrod', linewidth=2.5))

# Arrow pointing to DRC
ax.annotate('', xy=(22, -2), xytext=(13, -2),
            arrowprops=dict(arrowstyle='->', lw=2, color='darkgoldenrod'))

# Zimbabwe - position to the right (east) away from bubbles
ax.text(36, -19, 'ZIMBABWE\n77 saves',
        fontsize=13, fontweight='bold', ha='center', va='center',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='gold',
                 alpha=0.95, edgecolor='darkgoldenrod', linewidth=2.5))

# Arrow pointing to Zimbabwe
ax.annotate('', xy=(30, -19), xytext=(33, -19),
            arrowprops=dict(arrowstyle='->', lw=2, color='darkgoldenrod'))

# Add summary statistics box
stats_text = (
    f"TOTAL: 249 Cascade Rescues\n"
    f"Across: {len(district_saves)} districts\n"
    f"In: {key_saves['ipc_country'].nunique()} countries\n\n"
    f"Concentration: 70.7% in top 3\n"
    f"(Zimbabwe, Sudan, DRC)"
)
ax.text(0.02, 0.98, stats_text,
        transform=ax.transAxes,
        fontsize=11, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=0.8', facecolor='lightyellow',
                 alpha=0.95, edgecolor='darkgoldenrod', linewidth=2.5))

# Add title
ax.set_title('Geographic Distribution of 249 Cascade Rescues:\nWhere News Features Caught Crises That AR Baseline Missed',
            fontsize=15, fontweight='bold', pad=20)

# Set bounds to show full Africa
ax.set_xlim(-20, 55)
ax.set_ylim(-37, 40)

# Add grid
ax.grid(True, alpha=0.3, linestyle='--', linewidth=0.5)

# Labels
ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')

# Add legend for size - matching the YlOrRd colormap
import matplotlib.cm as cm
import matplotlib.colors as mcolors

# Create a colormap normalizer
norm = mcolors.Normalize(vmin=district_saves['n_saves'].min(),
                         vmax=district_saves['n_saves'].max())
cmap = cm.get_cmap('YlOrRd')

sizes = [1, 5, 10, 15]
labels = [f'{s} saves' for s in sizes]

# Create legend with colors matching the colormap
legend_points = [plt.scatter([], [], s=s*50,
                            c=[cmap(norm(s))],
                            alpha=0.8,
                            edgecolors='darkred',
                            linewidths=1.5)
                for s in sizes]

legend1 = ax.legend(legend_points, labels,
                   scatterpoints=1, frameon=True, labelspacing=2,
                   title='District-Level\nKey Saves', loc='lower left',
                   fontsize=10, title_fontsize=10)
legend1.get_frame().set_alpha(0.9)
legend1.get_frame().set_edgecolor('black')

# Add note
note_text = (
    "Key Saves = Cases where AR predicted NO crisis, actual was CRISIS, and Cascade predicted CRISIS\n"
    "These are the 249 breakthrough cases: shock-driven crises invisible to temporal/spatial persistence\n"
    "Gold highlighted countries: Top 3 with 70.7% of all rescues (conflict/economic contexts)"
)
ax.text(0.5, 0.01, note_text,
        transform=ax.transAxes, fontsize=9, ha='center', va='bottom',
        style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                 alpha=0.9, edgecolor='gray', linewidth=1.5))

plt.tight_layout()

# Save
output_file = OUTPUT_DIR / "ch05_cascade_geographic_map.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file}")

output_file_png = OUTPUT_DIR / "ch05_cascade_geographic_map.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png}")

plt.close()

print("\n" + "="*80)
print("GEOGRAPHIC MAP COMPLETE: CASCADE RESCUES ACROSS AFRICA")
print("="*80)
print(f"Total key saves: {len(key_saves):,}")
print(f"Unique districts: {len(district_saves)}")
print(f"Countries: {key_saves['ipc_country'].nunique()}")
print(f"Top 3 concentration: {(country_saves.head(3).sum() / len(key_saves) * 100):.1f}%")
