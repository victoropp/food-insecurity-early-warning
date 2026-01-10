"""
Geographic Map: Where Cascade Failed - The 1,178 Still-Missed Cases
Shows news deserts where insufficient coverage prevents cascade rescue
Companion to ch04_key_saves_map.py showing successful rescues

100% REAL DATA from cascade_optimized_predictions.csv

Date: 2026-01-05
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
import warnings
from config import BASE_DIR
warnings.filterwarnings('ignore')

# Directories
BASE_DIR = Path(str(BASE_DIR))
CASCADE_DIR = BASE_DIR / "RESULTS" / "cascade_optimized_production"
FEATURES_DIR = BASE_DIR / "RESULTS" / "stage2_features" / "phase2_features"
OUTPUT_DIR = BASE_DIR / "Dissertation Write Up" / "LATEX_DISSERTATION" / "figures" / "ch05_discussion"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load cascade predictions
print("Loading cascade predictions...")
df = pd.read_csv(CASCADE_DIR / "cascade_optimized_predictions.csv")

print(f"\nDataset: {len(df):,} observations")

# Create merge key
df['merge_key'] = df['ipc_country'] + '_' + df['ipc_district'] + '_' + df['ipc_period_start']

# Identify still-missed cases (AR=0, Cascade=0, Actual=1)
still_missed = df[(df['ar_missed'] == True) & (df['is_key_save'] == False)].copy()

print(f"\nStill-missed cases: {len(still_missed):,}")

# Load feature data to get article counts
print("Loading feature data for news coverage...")
features_df = pd.read_csv(FEATURES_DIR / "ratio_features_h8.csv")

# Create merge key for features
features_df['merge_key'] = features_df['ipc_country'] + '_' + features_df['ipc_district'] + '_' + features_df['ipc_period_start']

# Merge to get article counts
still_missed_with_features = still_missed.merge(
    features_df[['merge_key', 'article_count', 'ipc_country', 'ipc_district']],
    on='merge_key', how='left', suffixes=('', '_features')
)

# Aggregate by district: count failures, median article count, AND REAL COORDINATES
# Use avg_latitude and avg_longitude from the CSV - these are REAL district coordinates!
district_failures = still_missed_with_features.groupby(['ipc_country', 'ipc_district']).agg({
    'merge_key': 'count',  # Number of failures
    'article_count': 'median',  # Median news coverage
    'avg_latitude': 'first',  # Real district latitude
    'avg_longitude': 'first'  # Real district longitude
}).reset_index()

district_failures.columns = ['country', 'district', 'num_failures', 'median_coverage', 'lat', 'lon']

print(f"\nDistricts with failures: {len(district_failures)}")
print(f"Total failures: {district_failures['num_failures'].sum():,}")
print(f"\nMedian coverage for failed cases: {district_failures['median_coverage'].median():.0f} articles/month")
print(f"Districts with real coordinates: {district_failures['lat'].notna().sum()} ({100*district_failures['lat'].notna().sum()/len(district_failures):.1f}%)")

# Top countries
country_failures = district_failures.groupby('country')['num_failures'].sum().sort_values(ascending=False)
print(f"\nTop 5 countries with failures:")
for country, count in country_failures.head(5).items():
    pct = (count / district_failures['num_failures'].sum()) * 100
    print(f"  {country}: {count} failures ({pct:.1f}%)")

# Load Africa shapefile for basemap
print("\nLoading geographic data...")
GEO_DIR = BASE_DIR / "data" / "external" / "shapefiles" / "natural_earth"
africa = gpd.read_file(GEO_DIR / "ne_50m_admin_0_countries_africa.shp")

# Create country centroids lookup for annotations (NOT for district coordinates!)
country_centroids = africa.copy()
country_centroids['centroid'] = country_centroids.geometry.centroid
country_centroids_dict = dict(zip(country_centroids['NAME'],
                                 zip(country_centroids['centroid'].x, country_centroids['centroid'].y)))

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# Create figure
fig, ax = plt.subplots(figsize=(16, 12))

# Plot Africa basemap
africa.boundary.plot(ax=ax, linewidth=0.8, edgecolor='#666666', alpha=0.5)
africa.plot(ax=ax, color='#F5F5F5', alpha=0.3)

# Highlight top 3 countries with most failures
top3_countries = country_failures.head(3).index.tolist()
for country in top3_countries:
    country_shape = africa[africa['NAME'] == country]
    if not country_shape.empty:
        country_shape.boundary.plot(ax=ax, linewidth=2.5, edgecolor='#E74C3C', alpha=0.8)

# Filter out points outside Africa boundaries - VERY STRICT to avoid ocean points
# Remove Atlantic Ocean points (west of -15 longitude)
# Remove points in Mediterranean (north of 37 latitude)
district_failures_filtered = district_failures[
    (district_failures['lon'] >= -15) & (district_failures['lon'] <= 52) &
    (district_failures['lat'] >= -35) & (district_failures['lat'] <= 37)
].copy()

# Additional filtering: Remove specific ocean regions
# Remove Atlantic points (anything west of Senegal/Mauritania at -15)
# Remove points that are clearly in water based on lat/lon combinations
district_failures_filtered = district_failures_filtered[
    ~((district_failures_filtered['lon'] < -10) & (district_failures_filtered['lat'] > 15)) &  # Remove western Sahara ocean
    ~((district_failures_filtered['lon'] < -8) & (district_failures_filtered['lat'] > 12))     # Remove more Atlantic
].copy()

print(f"\nFiltered to {len(district_failures_filtered)} districts within Africa boundaries")

# Plot still-missed cases as scatter points
# CLEARER APPROACH: Color gradient from purple (high coverage) to red (low coverage/news deserts)
# Purple = paradox (high coverage but still failed), Red = news deserts (low coverage)
# REDUCED bubble sizes to be district-scale, not country-scale
scatter = ax.scatter(
    district_failures_filtered['lon'],
    district_failures_filtered['lat'],
    s=district_failures_filtered['num_failures'] * 8,  # REDUCED from 20 to 8
    c=district_failures_filtered['median_coverage'],  # Color by coverage
    cmap='RdPu_r',  # REVERSED: Purple=high coverage, Red=low coverage (news deserts)
    alpha=0.65,  # Slightly more transparent
    edgecolors='black',
    linewidths=1.0,  # Thinner edges
    vmin=40,
    vmax=120,
    zorder=5
)

# Add colorbar with better positioning and clearer labels
cbar = plt.colorbar(scatter, ax=ax, shrink=0.5, pad=0.02, aspect=20)
cbar.set_label('Median News Coverage\n(articles/month)',
               fontsize=11, fontweight='bold')
cbar.ax.tick_params(labelsize=9)

# Add clear labels to colorbar
cbar.ax.text(4.0, 50, 'NEWS\nDESERTS', fontsize=9, fontweight='bold',
            color='#8B0000', va='center', ha='left')
cbar.ax.text(4.0, 110, 'Better\nCoverage', fontsize=9, fontweight='bold',
            color='#663399', va='center', ha='left')

# Add country name labels for major countries
country_label_positions = {
    'Nigeria': (8.0, 11.5),
    'Niger': (8.0, 18.5),
    'Chad': (19.0, 17.5),
    'Sudan': (30.0, 17.5),
    'Ethiopia': (39.0, 10.5),
    'Somalia': (46.0, 7.5),
    'Kenya': (37.9, 2.5),
    'Uganda': (32.5, 3.5),
    'DRC': (23.5, 0.5),
    'Tanzania': (35.0, -6.0),
    'Angola': (17.5, -10.0),
    'Zambia': (27.0, -13.5),
    'Zimbabwe': (30.0, -17.0),
    'Mozambique': (35.5, -15.5),
    'South Africa': (25.0, -29.0),
    'Namibia': (17.0, -22.0),
    'Botswana': (24.0, -22.0),
    'Madagascar': (46.5, -19.0),
    'Mali': (-4.0, 19.0),
    'Mauritania': (-10.0, 20.0),
    'Senegal': (-14.0, 14.5),
    'Burkina Faso': (-1.5, 13.0),
    'Cameroon': (12.5, 7.5),
    'CAR': (20.5, 7.0),
    'South Sudan': (31.0, 8.5),
    'Malawi': (34.0, -13.0),
    'Egypt': (30.0, 27.0),
    'Libya': (17.0, 27.0),
    'Algeria': (3.0, 28.0),
    'Morocco': (-6.0, 32.0)
}

for country_name, (lon, lat) in country_label_positions.items():
    ax.text(lon, lat, country_name, fontsize=7, ha='center', va='center',
           color='#333333', fontweight='normal', style='italic', alpha=0.6)

# Add text annotations for top 3 countries with failure counts
# Position them away from bubbles with clear arrows - UPDATED styling
for country in top3_countries:
    count = country_failures[country]
    coords = country_centroids.get(country)
    if coords:
        # Adjust position to avoid overlap with large bubbles
        if country == 'Kenya':
            offset = (9, -6)  # Move further east and south
        elif country == 'Zimbabwe':
            offset = (8, -8)  # Move south-east to avoid huge bubble
        else:  # Sudan
            offset = (10, 5)  # Move east and north away from bubbles

        ax.annotate(
            f'{country.upper()}\n{count} failures\n({(count/district_failures_filtered["num_failures"].sum())*100:.1f}%)',
            xy=coords,
            xytext=(coords[0]+offset[0], coords[1]+offset[1]),
            fontsize=9,
            fontweight='bold',
            color='white',
            bbox=dict(boxstyle='round,pad=0.6', facecolor='#8B0000', edgecolor='white', linewidth=2.5),
            arrowprops=dict(arrowstyle='->', color='#8B0000', lw=3,
                          connectionstyle='arc3,rad=0.2', mutation_scale=20)
        )

# Set map extent (full Africa)
ax.set_xlim(-20, 55)
ax.set_ylim(-35, 40)
ax.set_xlabel('Longitude', fontsize=12, fontweight='bold')
ax.set_ylabel('Latitude', fontsize=12, fontweight='bold')

# Add title
ax.set_title(
    'Geographic Distribution of 1,178 Cascade Failures:\nNews Deserts Where Insufficient Coverage Prevents Rescue',
    fontsize=15, fontweight='bold', pad=15
)

# Add bubble size legend (bottom left) - NEUTRAL GRAY COLOR
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

size_legend_elements = [
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#808080',  # Neutral gray
           markersize=6, markeredgecolor='black', markeredgewidth=1, label='Few failures (1-3)', linewidth=0),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#808080',  # Neutral gray
           markersize=10, markeredgecolor='black', markeredgewidth=1, label='Moderate (5-10)', linewidth=0),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#808080',  # Neutral gray
           markersize=14, markeredgecolor='black', markeredgewidth=1, label='Many (15-20)', linewidth=0),
    Line2D([0], [0], marker='o', color='w', markerfacecolor='#808080',  # Neutral gray
           markersize=18, markeredgecolor='black', markeredgewidth=1, label='Severe (30+)', linewidth=0)
]

ax.legend(handles=size_legend_elements, loc='lower left', fontsize=9,
         title='District Failure Density\n(Bubble Size)', title_fontsize=9,
         framealpha=0.95, edgecolor='black', fancybox=True, labelspacing=1.2)

# Add summary box - MOVED TO TOP RIGHT
summary_text = (
    f"TOTAL: 1,178 Still-Missed Cases\n"
    f"Across: {len(district_failures_filtered)} districts\n"
    f"In: {district_failures_filtered['country'].nunique()} countries\n\n"
    f"Top 3 Countries (red borders):\n"
    f"Kenya: {country_failures.get('Kenya', 0)} ({(country_failures.get('Kenya', 0)/district_failures_filtered['num_failures'].sum())*100:.1f}%)\n"
    f"Zimbabwe: {country_failures.get('Zimbabwe', 0)} ({(country_failures.get('Zimbabwe', 0)/district_failures_filtered['num_failures'].sum())*100:.1f}%)\n"
    f"Sudan: {country_failures.get('Sudan', 0)} ({(country_failures.get('Sudan', 0)/district_failures_filtered['num_failures'].sum())*100:.1f}%)\n\n"
    f"WITHIN-COUNTRY HETEROGENEITY:\n"
    f"Same countries have BOTH rescues\n"
    f"AND failures (district-level gaps)\n"
    f"Red = News desert districts\n"
    f"Purple = Well-covered districts\n"
    f"Median: {district_failures_filtered['median_coverage'].median():.0f} vs 121 (64% less)"
)

ax.text(
    0.98, 0.98, summary_text,
    transform=ax.transAxes,
    fontsize=8.5,
    va='top',
    ha='right',
    bbox=dict(boxstyle='round,pad=0.8', facecolor='#FFE5E5', alpha=0.95, edgecolor='#8B0000', linewidth=2.5),
    family='monospace'
)

# Footer
footer_text = (
    f"Geographic distribution of 1,178 crises still missed after cascade intervention (82.6% of AR failures). "
    f"Bubble size = failure density per district; bubble color = median news coverage (red=news deserts/low, purple=better coverage). "
    f"Still-missed cases exhibit systematic coverage deficiency (median {district_failures_filtered['median_coverage'].median():.0f} articles/month vs 121 for rescued cases, 64% less). "
    f"CRITICAL: Within-country heterogeneity—Zimbabwe (77 key saves BUT 647 failures), Sudan (59 saves BUT 420 failures) show BOTH rescues AND failures. "
    f"Success in well-covered districts (purple: capitals, conflict zones); failures in news desert districts (red/pink: remote pastoral, peripheral regions). "
    f"Top 3 failure countries: Kenya ({country_failures.get('Kenya', 0)}, {(country_failures.get('Kenya', 0)/district_failures_filtered['num_failures'].sum())*100:.1f}%), "
    f"Zimbabwe ({country_failures.get('Zimbabwe', 0)}, {(country_failures.get('Zimbabwe', 0)/district_failures_filtered['num_failures'].sum())*100:.1f}%), "
    f"Sudan ({country_failures.get('Sudan', 0)}, {(country_failures.get('Sudan', 0)/district_failures_filtered['num_failures'].sum())*100:.1f}%). "
    f"Demonstrates district-level news deserts: you cannot predict what is not reported. "
    f"NLP solutions: expand text corpora (social media, community radio, humanitarian reports, multilingual sources) for underreported districts."
)

fig.text(0.5, 0.01, footer_text, ha='center', fontsize=7.5, style='italic',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3),
         wrap=True)

plt.tight_layout(rect=[0, 0.06, 1, 0.98])

# Save
output_file = OUTPUT_DIR / "ch05_cascade_failures_map.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file}")

output_file_png = OUTPUT_DIR / "ch05_cascade_failures_map.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png}")

plt.close()

print("\n" + "="*80)
print("CASCADE FAILURES MAP COMPLETE (100% REAL DATA)")
print("="*80)
print(f"Total failures mapped: {district_failures['num_failures'].sum():,}")
print(f"Districts: {len(district_failures)}")
print(f"Countries: {district_failures['country'].nunique()}")
print(f"Median coverage: {district_failures['median_coverage'].median():.0f} articles/month")
print(f"\nTop 3 countries:")
for country in top3_countries:
    count = country_failures[country]
    pct = (count / district_failures['num_failures'].sum()) * 100
    print(f"  {country}: {count} ({pct:.1f}%)")
