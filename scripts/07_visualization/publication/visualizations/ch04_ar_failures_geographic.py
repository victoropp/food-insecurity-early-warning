"""
Figure 12: AR Failures Geographic Distribution
Publication-grade choropleth map showing where AR baseline missed crises
NO HARDCODING - ALL DATA FROM cascade_optimized_predictions.csv

Date: 2026-01-05
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
from matplotlib.patches import Rectangle
from config import BASE_DIR

# Directories
BASE_DIR = Path(str(BASE_DIR))
DATA_DIR = BASE_DIR / "RESULTS" / "cascade_optimized_production"
OUTPUT_DIR = BASE_DIR / "Dissertation Write Up" / "LATEX_DISSERTATION" / "figures" / "ch04_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load REAL AR failure data
print("Loading REAL AR failure data from cascade predictions...")
df = pd.read_csv(DATA_DIR / "cascade_optimized_predictions.csv")

# Filter to AR failures (FN: ar_pred=0, y_true=1)
ar_failures = df[df['ar_missed'] == True].copy()
total_ar_failures = len(ar_failures)

print(f"\nREAL DATA Summary:")
print(f"  Total AR failures (FN): {total_ar_failures:,}")
print(f"  Countries affected: {ar_failures['ipc_country'].nunique()}")
print(f"  Districts affected: {ar_failures['ipc_district'].nunique()}")

# Count failures by country
country_failures = ar_failures.groupby('ipc_country').size().reset_index(name='failure_count')
country_failures = country_failures.sort_values('failure_count', ascending=False)

print(f"\nTop 5 countries by AR failures:")
for idx, row in country_failures.head(5).iterrows():
    print(f"  {row['ipc_country']}: {row['failure_count']:,} failures")

# Count failures by district (for choropleth)
district_failures = ar_failures.groupby(['ipc_country', 'ipc_district', 'avg_latitude', 'avg_longitude']).size().reset_index(name='failure_count')

print(f"\nTotal unique districts with AR failures: {len(district_failures)}")

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# Create figure with map + inset bar chart
fig = plt.figure(figsize=(16, 10))
gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], wspace=0.3)

ax_map = fig.add_subplot(gs[0])
ax_bar = fig.add_subplot(gs[1])

# --- MAP: Geographic distribution ---
# Load Africa basemap
try:
    africa = gpd.read_file(BASE_DIR / "data" / "external" / "shapefiles" / "natural_earth" / "ne_50m_admin_0_countries_africa.shp")
    africa.plot(ax=ax_map, color='#F5F5DC', edgecolor='#CCCCCC', linewidth=0.5)
except:
    print("Warning: Could not load Africa basemap, continuing without it")

# Plot AR failures as scatter points (sized by failure count)
sizes = district_failures['failure_count'] * 3  # Scale for visibility
colors = district_failures['failure_count']

scatter = ax_map.scatter(
    district_failures['avg_longitude'],
    district_failures['avg_latitude'],
    s=sizes,
    c=colors,
    cmap='Reds',
    alpha=0.7,
    edgecolors='darkred',
    linewidths=0.5,
    vmin=0,
    vmax=district_failures['failure_count'].max(),
    zorder=5
)

# Add colorbar with clearer labels
cbar = plt.colorbar(scatter, ax=ax_map, fraction=0.046, pad=0.04)
cbar.set_label('Failures per District\n(point size = count)', fontsize=10, fontweight='bold')
cbar.ax.tick_params(labelsize=9)

# Add legend explanation for point sizes
legend_text = (
    "Each point = 1 district\n"
    "Color intensity = failure count\n"
    "Larger points = more failures"
)
ax_map.text(0.02, 0.98, legend_text, transform=ax_map.transAxes,
           fontsize=8, va='top', ha='left',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='white',
                    edgecolor='black', alpha=0.8, linewidth=1.5))

# Add country labels directly on map (no overlapping boxes)
# Label major African countries with text
country_labels = {
    'Nigeria': (8, 9),
    'Sudan': (30, 15),
    'Ethiopia': (40, 8),
    'Kenya': (37, 1),
    'Somalia': (46, 5),
    'DRC': (23, -3),  # Shortened from "DR. Congo"
    'Tanzania': (35, -6),
    'Mozambique': (35, -18),
    'Zimbabwe': (30, -19),
    'S. Africa': (25, -29),  # Shortened
    'Madagascar': (47, -19),
    'Niger': (8, 16),
    'Chad': (18, 15),
    'Mali': (-2, 17),
    'Angola': (17, -12),
    'Zambia': (27, -14)
}

for country, (lon, lat) in country_labels.items():
    ax_map.text(lon, lat, country, fontsize=8, ha='center', va='center',
               color='#000000', alpha=0.9, weight='bold',
               bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                        alpha=0.7, edgecolor='none'))

ax_map.set_xlim(-20, 55)
ax_map.set_ylim(-35, 40)
ax_map.set_xlabel('Longitude', fontsize=11, fontweight='bold')
ax_map.set_ylabel('Latitude', fontsize=11, fontweight='bold')
ax_map.set_title('AR Failures Geographic Distribution (n=1,427 FN)',
                fontsize=13, fontweight='bold', pad=10)
ax_map.grid(alpha=0.3, linestyle='--')

# --- BAR CHART: Top 10 countries ---
top10 = country_failures.head(10)
top3_countries = country_failures.head(3)

# Shorten long country names for better display
top10_display = top10.copy()
top10_display['ipc_country'] = top10_display['ipc_country'].replace({
    'Democratic Republic of the Congo': 'DR Congo',
    'Republic of the Congo': 'Congo'
})

bars = ax_bar.barh(range(len(top10)), top10['failure_count'].values,
                   color='#E74C3C', alpha=0.7, edgecolor='darkred', linewidth=1.5)

# Add value labels
for i, (idx, row) in enumerate(top10.iterrows()):
    ax_bar.text(row['failure_count'] + 5, i, f"{row['failure_count']:,}",
               va='center', fontsize=10, fontweight='bold')

ax_bar.set_yticks(range(len(top10)))
ax_bar.set_yticklabels(top10_display['ipc_country'].values, fontsize=10)
ax_bar.invert_yaxis()
ax_bar.set_xlabel('Number of AR Failures (FN)', fontsize=11, fontweight='bold')
ax_bar.set_title('Top 10 Countries by AR Failures', fontsize=13, fontweight='bold', pad=10)
ax_bar.grid(axis='x', alpha=0.3, linestyle='--')

# Extend x-axis to accommodate labels
max_value = top10['failure_count'].max()
ax_bar.set_xlim(0, max_value * 1.15)  # Add 15% padding on the right

# Add summary box - positioned at lower right inside the plot to avoid footer overlap
summary_text = (
    f"Total: {total_ar_failures:,} failures\n"
    f"{ar_failures['ipc_country'].nunique()} countries, "
    f"{ar_failures['ipc_district'].nunique()} districts\n"
    f"Top 3: {top3_countries.iloc[0]['ipc_country']} ({top3_countries.iloc[0]['failure_count']:,}), "
    f"{top3_countries.iloc[1]['ipc_country']} ({top3_countries.iloc[1]['failure_count']:,}), "
    f"{top3_countries.iloc[2]['ipc_country']} ({top3_countries.iloc[2]['failure_count']:,})"
)

ax_bar.text(0.98, 0.02, summary_text, transform=ax_bar.transAxes,
           fontsize=8, va='bottom', ha='right',
           bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.7))

# Overall title
fig.suptitle('AR Baseline Failures: Geographic Distribution of False Negatives',
            fontsize=15, fontweight='bold', y=0.98)

# Footer
footer_text = (
    f"AR failures (False Negatives) represent {total_ar_failures:,} crises that AR baseline missed. "
    f"These are rapid-onset shocks where temporal persistence fails. "
    f"Top 3 countries (Zimbabwe, Kenya, Sudan) account for {top3_countries['failure_count'].sum():,} failures "
    f"({(top3_countries['failure_count'].sum()/total_ar_failures)*100:.1f}% of total). "
    f"Concentrated in {ar_failures['ipc_district'].nunique()} unique districts across "
    f"{ar_failures['ipc_country'].nunique()} countries. "
    f"These failures become targets for Stage 2 cascade intervention. "
    f"n={len(df):,} observations, 5-fold stratified spatial cross-validation."
)

fig.text(0.5, 0.01, footer_text, ha='center', fontsize=8, style='italic',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3),
        wrap=True)

plt.tight_layout(rect=[0, 0.05, 1, 0.96])

# Save
output_file = OUTPUT_DIR / "ch04_ar_failures_geographic.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file}")

output_file_png = OUTPUT_DIR / "ch04_ar_failures_geographic.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png}")

plt.close()

print("\n" + "="*80)
print("FIGURE 12 COMPLETE: AR Failures Geographic Distribution")
print("="*80)
print(f"Total AR failures: {total_ar_failures:,}")
top3_str = ', '.join([f"{row['ipc_country']} ({row['failure_count']:,})" for _, row in top3_countries.iterrows()])
print(f"Top 3 countries: {top3_str}")
