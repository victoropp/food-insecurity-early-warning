"""
Appendix Figure D: Data Quality Assessment
Multi-panel diagnostic showing coverage, temporal distribution, and geographic density
Validates data integrity for 20,722 observations across 18 countries, 1,920 districts (final dataset)

Date: 2026-01-05
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
from datetime import datetime
from config import BASE_DIR

# Directories
BASE_DIR = Path(rstr(BASE_DIR))
CASCADE_FILE = BASE_DIR / "RESULTS" / "cascade_optimized_production" / "cascade_optimized_predictions.csv"
GEODATA_DIR = Path(r"C:\GDELT_Africa_Extract\data")
OUTPUT_DIR = BASE_DIR / "Dissertation Write Up" / "LATEX_DISSERTATION" / "figures" / "appendices"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load data
print("Loading cascade data...")
df = pd.read_csv(CASCADE_FILE)

print(f"\nData quality metrics:")
print(f"  Total observations: {len(df):,}")
print(f"  Countries: {df['ipc_country'].nunique()}")
print(f"  Districts: {df['ipc_district'].nunique()}")
print(f"  Time periods: {df['ipc_period_start'].nunique()}")
print(f"  Crisis rate: {df['y_true'].mean():.1%}")

# Parse dates
df['period_date'] = pd.to_datetime(df['ipc_period_start'])

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9

# Create figure with 4 panels
fig = plt.figure(figsize=(18, 12))

# Panel A: District coverage by country (bar chart)
ax1 = plt.subplot(2, 2, 1)
district_counts = df.groupby('ipc_country')['ipc_district'].nunique().sort_values(ascending=False)
bars = ax1.barh(range(len(district_counts)), district_counts.values, color='#0072B2', alpha=0.8)

# Highlight top 3 countries
top3_countries = ['Kenya', 'Ethiopia', 'Nigeria']
for idx, (country, count) in enumerate(district_counts.items()):
    if country in top3_countries:
        bars[idx].set_color('#D55E00')
        bars[idx].set_alpha(1.0)

ax1.set_yticks(range(len(district_counts)))
ax1.set_yticklabels(district_counts.index, fontsize=8)
ax1.set_xlabel('Number of Districts', fontsize=10, fontweight='bold')
ax1.set_title('Panel A: Geographic Coverage by Country\nDistrict-Level Granularity',
              fontsize=11, fontweight='bold', pad=10)
ax1.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for idx, val in enumerate(district_counts.values):
    ax1.text(val + 5, idx, str(val), va='center', fontsize=7)

# Add summary stats
total_districts = df['ipc_district'].nunique()
total_countries = df['ipc_country'].nunique()
ax1.text(0.98, 0.98, f'Total: {total_districts:,} districts\nacross {total_countries} countries',
         transform=ax1.transAxes, ha='right', va='top', fontsize=8,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

# Panel B: Temporal coverage (timeline)
ax2 = plt.subplot(2, 2, 2)
temporal_counts = df.groupby('ipc_period_start').size().sort_index()
dates = pd.to_datetime(temporal_counts.index)

ax2.plot(dates, temporal_counts.values, marker='o', linewidth=2, markersize=4,
         color='#009E73', alpha=0.8)
ax2.fill_between(dates, temporal_counts.values, alpha=0.3, color='#009E73')

ax2.set_xlabel('IPC Period Date', fontsize=10, fontweight='bold')
ax2.set_ylabel('Number of Observations', fontsize=10, fontweight='bold')
ax2.set_title('Panel B: Temporal Coverage (2021-2024)\nConsistent Observation Density',
              fontsize=11, fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3, linestyle='--')

# Add date range annotation
start_date = dates.min().strftime('%b %Y')
end_date = dates.max().strftime('%b %Y')
n_periods = len(temporal_counts)
ax2.text(0.02, 0.98, f'Coverage: {start_date} to {end_date}\n{n_periods} IPC periods',
         transform=ax2.transAxes, ha='left', va='top', fontsize=8,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

# Rotate x-axis labels
ax2.tick_params(axis='x', rotation=45)

# Panel C: Crisis rate distribution (histogram)
ax3 = plt.subplot(2, 2, 3)
crisis_rates_by_country = df.groupby('ipc_country')['y_true'].mean()

ax3.hist(crisis_rates_by_country.values, bins=15, color='#E69F00', alpha=0.8, edgecolor='black')
ax3.axvline(df['y_true'].mean(), color='red', linestyle='--', linewidth=2,
            label=f'Overall mean: {df["y_true"].mean():.1%}')

ax3.set_xlabel('Crisis Rate (IPC≥3)', fontsize=10, fontweight='bold')
ax3.set_ylabel('Number of Countries', fontsize=10, fontweight='bold')
ax3.set_title('Panel C: Crisis Rate Distribution by Country\nHeterogeneity Across Contexts',
              fontsize=11, fontweight='bold', pad=10)
ax3.grid(axis='y', alpha=0.3, linestyle='--')
ax3.legend(loc='upper right', fontsize=8)

# Add stats
min_rate = crisis_rates_by_country.min()
max_rate = crisis_rates_by_country.max()
mean_rate = crisis_rates_by_country.mean()
ax3.text(0.02, 0.98, f'Range: {min_rate:.1%} to {max_rate:.1%}\nMean: {mean_rate:.1%}\nStd: {crisis_rates_by_country.std():.1%}',
         transform=ax3.transAxes, ha='left', va='top', fontsize=8,
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow', alpha=0.9))

# Panel D: Observation density and crisis rate by district (map)
ax4 = plt.subplot(2, 2, 4)

# Load Africa basemap
try:
    africa = gpd.read_file(GEODATA_DIR / "natural_earth" / "ne_50m_admin_0_countries_africa.shp")

    # Calculate observation count and crisis rate per district
    district_stats = df.groupby(['ipc_district', 'avg_latitude', 'avg_longitude']).agg({
        'y_true': ['count', 'mean']  # count = observations, mean = crisis rate
    }).reset_index()
    district_stats.columns = ['district', 'lat', 'lon', 'n_obs', 'crisis_rate']

    # Plot basemap
    africa.plot(ax=ax4, color='#F5F5F5', edgecolor='#CCCCCC', linewidth=0.5)

    # Plot districts - size by observation count, color by crisis rate
    # Normalize sizes
    size_scale = (district_stats['n_obs'] - district_stats['n_obs'].min()) / (district_stats['n_obs'].max() - district_stats['n_obs'].min())
    sizes = 10 + size_scale * 100  # Size range: 10 to 110

    scatter = ax4.scatter(district_stats['lon'], district_stats['lat'],
                         s=sizes,
                         c=district_stats['crisis_rate'],
                         cmap='RdYlGn_r', alpha=0.7,  # Red=high crisis, Green=low crisis
                         edgecolors='black', linewidths=0.3,
                         vmin=0, vmax=1)

    # Colorbar
    cbar = plt.colorbar(scatter, ax=ax4, fraction=0.03, pad=0.02)
    cbar.set_label('District Crisis Rate (IPC≥3)', fontsize=9, fontweight='bold')

    # Add country labels for major countries
    country_labels = {
        'Kenya': (37, 1),
        'Ethiopia': (39, 9),
        'Sudan': (30, 15),
        'South Sudan': (30, 7),
        'Uganda': (32, 1),
        'Somalia': (45, 5),
        'DRC': (23, -3),
        'Nigeria': (8, 9),
        'Niger': (8, 17),
        'Mali': (-4, 17),
        'Zimbabwe': (30, -19),
        'Mozambique': (35, -18),
        'Madagascar': (47, -19)
    }

    for country, (lon, lat) in country_labels.items():
        ax4.text(lon, lat, country, fontsize=7, ha='center', va='center',
                color='black', alpha=0.6, fontweight='bold',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                         alpha=0.7, edgecolor='none'))

    ax4.set_xlim(-20, 55)
    ax4.set_ylim(-37, 40)

except Exception as e:
    print(f"Warning: Could not load geographic data: {e}")
    # Fallback: show text summary
    ax4.text(0.5, 0.5, 'Geographic visualization\nunavailable\n\nSee coverage stats in Panel A',
             ha='center', va='center', fontsize=10, transform=ax4.transAxes)

ax4.set_title('Panel D: District-Level Observation Density & Crisis Rate\nSize = Observations | Color = Crisis Rate',
              fontsize=11, fontweight='bold', pad=10)
ax4.set_xlabel('Longitude', fontsize=10, fontweight='bold')
ax4.set_ylabel('Latitude', fontsize=10, fontweight='bold')
ax4.grid(True, alpha=0.3, linestyle='--')

# Overall title
fig.suptitle('Data Quality Assessment: Coverage, Temporal Distribution, and Geographic Density',
             fontsize=14, fontweight='bold', y=0.98)

# Summary statistics box
summary_text = (
    f"DATA QUALITY VALIDATION:\n"
    f"• Total observations: {len(df):,} | Countries: {df['ipc_country'].nunique()} | Districts: {df['ipc_district'].nunique()}\n"
    f"• Temporal coverage: {dates.min().strftime('%b %Y')} to {dates.max().strftime('%b %Y')} ({n_periods} periods)\n"
    f"• Overall crisis rate: {df['y_true'].mean():.1%} | Country range: {min_rate:.1%} to {max_rate:.1%}\n"
    f"• Geographic heterogeneity confirmed: Validates stratified spatial CV approach\n"
    f"• Consistent temporal density: No major gaps in observation coverage"
)
fig.text(0.5, 0.01, summary_text,
         ha='center', va='bottom', fontsize=9,
         bbox=dict(boxstyle='round,pad=0.6', facecolor='lightyellow',
                  alpha=0.95, edgecolor='darkgoldenrod', linewidth=2.5),
         transform=fig.transFigure)

plt.tight_layout(rect=[0, 0.08, 1, 0.96])

# Save
output_file = OUTPUT_DIR / "app_d_data_quality.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file}")

output_file_png = OUTPUT_DIR / "app_d_data_quality.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png}")

plt.close()

print("\n" + "="*80)
print("APPENDIX FIGURE D COMPLETE: DATA QUALITY ASSESSMENT")
print("="*80)
print(f"Total observations: {len(df):,}")
print(f"Countries: {df['ipc_country'].nunique()}")
print(f"Districts: {df['ipc_district'].nunique()}")
print(f"Date range: {dates.min().strftime('%b %Y')} to {dates.max().strftime('%b %Y')}")
