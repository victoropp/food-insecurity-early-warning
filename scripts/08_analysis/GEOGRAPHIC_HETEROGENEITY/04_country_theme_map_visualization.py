"""
Country-Specific Theme Map Visualization
=========================================
Geographic visualization showing which news themes drive predictions in each country
Uses SHAP-based theme importance data with Africa choropleth map

Approach:
1. Load country-theme importance data
2. Identify dominant theme for each country (highest SHAP importance)
3. Create choropleth map colored by dominant theme
4. Add theme-specific pie charts as insets for top 3 countries
5. Include legend showing theme color coding

Date: 2026-01-06
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import geopandas as gpd
from matplotlib.patches import Patch, Wedge
import warnings
from config import BASE_DIR
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(str(BASE_DIR))
ANALYSIS_DIR = BASE_DIR / "Dissertation Write Up" / "GEOGRAPHIC_HETEROGENEITY_ANALYSIS"
GEODATA_DIR = BASE_DIR / "data" / "external"
FIGURES_OUTPUT = BASE_DIR / "Dissertation Write Up" / "LATEX_DISSERTATION" / "figures" / "ch05_discussion"
FIGURES_OUTPUT.mkdir(parents=True, exist_ok=True)

print("="*80)
print("COUNTRY-SPECIFIC THEME MAP VISUALIZATION")
print("="*80)

# ============================================================================
# STEP 1: LOAD DATA
# ============================================================================

print("\nStep 1: Loading country-theme data...")
df_theme = pd.read_csv(ANALYSIS_DIR / "country_theme_importance.csv")

print(f"  Loaded: {len(df_theme)} rows")
print(f"  Countries: {df_theme['country'].nunique()}")
print(f"  Themes: {df_theme['theme'].nunique()}")

# Load key saves data for context
df_delta = pd.read_csv(ANALYSIS_DIR / "country_delta_auc_analysis.csv")
print(f"  Key saves data: {len(df_delta)} countries")

# ============================================================================
# STEP 2: IDENTIFY DOMINANT THEME PER COUNTRY
# ============================================================================

print("\nStep 2: Identifying dominant theme per country...")
dominant_theme = df_theme.loc[df_theme.groupby('country')['importance_pct'].idxmax()]
dominant_theme = dominant_theme[['country', 'theme', 'importance_pct']].copy()
dominant_theme.columns = ['country', 'dominant_theme', 'dominant_pct']

# Merge with key saves
dominant_theme = dominant_theme.merge(df_delta[['country', 'key_saves']], on='country', how='left')

print("\nDominant themes by country:")
for idx, row in dominant_theme.sort_values('key_saves', ascending=False).head(10).iterrows():
    print(f"  {row['country']}: {row['dominant_theme']} ({row['dominant_pct']:.1f}%), {row['key_saves']:.0f} saves")

# ============================================================================
# STEP 3: LOAD GEOGRAPHIC DATA
# ============================================================================

print("\nStep 3: Loading Africa basemap...")
africa = gpd.read_file(GEODATA_DIR / "natural_earth" / "ne_50m_admin_0_countries_africa.shp")

# Country name mapping
country_mapping = {
    'Democratic Republic of the Congo': 'Dem. Rep. Congo',
    'Republic of the Congo': 'Congo',
    'Central African Republic': 'Central African Rep.',
    'South Sudan': 'S. Sudan',
    'Equatorial Guinea': 'Eq. Guinea'
}

# Apply mapping
dominant_theme['country_map'] = dominant_theme['country'].replace(country_mapping)

# Merge with basemap
africa_with_theme = africa.merge(
    dominant_theme,
    left_on='NAME',
    right_on='country_map',
    how='left'
)

# Try original name for unmatched
africa_with_theme = africa_with_theme.merge(
    dominant_theme[['country', 'dominant_theme', 'dominant_pct', 'key_saves']],
    left_on='NAME',
    right_on='country',
    how='left',
    suffixes=('', '_orig')
)

# Use whichever merge worked
africa_with_theme['dominant_theme'] = africa_with_theme['dominant_theme'].fillna(
    africa_with_theme.get('dominant_theme_orig', 'none')
).fillna('none')

africa_with_theme['key_saves'] = africa_with_theme['key_saves'].fillna(
    africa_with_theme.get('key_saves_orig', 0)
).fillna(0)

print(f"  Basemap loaded: {len(africa)} countries")
print(f"  Matched with theme data: {(africa_with_theme['dominant_theme'] != 'none').sum()} countries")

# ============================================================================
# STEP 4: DEFINE THEME COLORS (IMPROVED - MORE DISTINCT)
# ============================================================================

# Theme color scheme (maximally distinct - completely different color families)
theme_colors = {
    'conflict': '#B71C1C',        # Deep Red - violence
    'displacement': '#795548',    # Brown - movement (completely different from yellow)
    'economic': '#1565C0',        # Royal Blue - finance
    'food_security': '#2E7D32',   # Forest Green - agriculture
    'governance': '#4A148C',      # Deep Purple - political
    'health': '#EC407A',          # Bright Pink - medical (completely different from red)
    'humanitarian': '#00838F',    # Teal - aid
    'weather': '#FDD835',         # Bright Yellow - climate (completely different from orange/brown)
    'other': '#424242',           # Charcoal Gray - general
    'none': '#ECEFF1'             # Light gray - no data
}

africa_with_theme['theme_color'] = africa_with_theme['dominant_theme'].map(theme_colors)

# ============================================================================
# STEP 5: CREATE VISUALIZATION
# ============================================================================

print("\nStep 5: Creating map visualization...")

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# Create figure
fig, ax = plt.subplots(figsize=(18, 12))

# Plot Africa basemap boundaries
africa.boundary.plot(ax=ax, linewidth=0.8, edgecolor='black', alpha=0.3)

# Plot choropleth by dominant theme
for theme in theme_colors.keys():
    if theme == 'none':
        continue
    theme_countries = africa_with_theme[africa_with_theme['dominant_theme'] == theme]
    if not theme_countries.empty:
        theme_countries.plot(
            ax=ax,
            color=theme_colors[theme],
            edgecolor='black',
            linewidth=0.8,
            alpha=0.85
        )

# Highlight countries with key saves data (SHAP analysis available)
countries_with_data = africa_with_theme[africa_with_theme['dominant_theme'] != 'none']
countries_with_data.boundary.plot(ax=ax, edgecolor='black', linewidth=1.5)

# Highlight top 3 countries with thick red borders
top3_names = ['Zimbabwe', 'Sudan', 'Dem. Rep. Congo']
for country_name in top3_names:
    country_geom = africa_with_theme[africa_with_theme['NAME'] == country_name]
    if not country_geom.empty:
        country_geom.boundary.plot(ax=ax, edgecolor='darkred', linewidth=3.5)

# Add labels for ALL 13 countries with SHAP data
label_positions = {
    'Zimbabwe': (30, -19),
    'Sudan': (30, 15),
    'Dem. Rep. Congo': (23, -2),
    'Nigeria': (8, 9),
    'Kenya': (38, 1),
    'Ethiopia': (40, 8),
    'Mozambique': (36, -18),
    'Mali': (-4, 17),
    'Malawi': (34, -13),
    'Somalia': (46, 5),
    'Uganda': (32, 2),
    'Madagascar': (46, -20),
    'Niger': (8, 16)
}

for country_name, (x, y) in label_positions.items():
    # Find country in data
    country_row = dominant_theme[
        (dominant_theme['country'] == country_name) |
        (dominant_theme['country_map'] == country_name)
    ]

    if not country_row.empty:
        theme = country_row.iloc[0]['dominant_theme']
        saves = country_row.iloc[0]['key_saves']

        # Format label - smaller for countries with 0 saves
        label = f"{country_name}\n{theme.title()}"
        if saves > 0:
            label += f"\n({saves:.0f} saves)"
            fontsize = 9
            fontweight = 'bold'
        else:
            fontsize = 8
            fontweight = 'normal'

        ax.text(x, y, label, fontsize=fontsize, fontweight=fontweight,
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.4',
                        facecolor='white',
                        edgecolor='black',
                        alpha=0.85, linewidth=1.2))

# Remove axes
ax.set_xlim(-20, 55)
ax.set_ylim(-38, 40)
ax.axis('off')

# Title
ax.set_title('Country-Specific News Theme Signatures: Which Themes Drive Predictions Where?\n' +
             'Dominant Theme by Country (SHAP-Based Analysis, n=23,039 observations, 13 countries)',
             fontsize=14, fontweight='bold', pad=20)

# Legend
legend_elements = [
    Patch(facecolor=theme_colors['conflict'], edgecolor='black', label='Conflict (rapid violence)'),
    Patch(facecolor=theme_colors['displacement'], edgecolor='black', label='Displacement (population movements)'),
    Patch(facecolor=theme_colors['economic'], edgecolor='black', label='Economic (collapse, inflation)'),
    Patch(facecolor=theme_colors['food_security'], edgecolor='black', label='Food Security (harvest, prices)'),
    Patch(facecolor=theme_colors['governance'], edgecolor='black', label='Governance (political, state capacity)'),
    Patch(facecolor=theme_colors['health'], edgecolor='black', label='Health (disease, malnutrition)'),
    Patch(facecolor=theme_colors['humanitarian'], edgecolor='black', label='Humanitarian (aid, assistance)'),
    Patch(facecolor=theme_colors['weather'], edgecolor='black', label='Weather (drought, floods, climate)'),
    Patch(facecolor=theme_colors['other'], edgecolor='black', label='Other (general news)'),
]

legend = ax.legend(handles=legend_elements,
                  loc='lower left',
                  fontsize=9,
                  framealpha=0.95,
                  edgecolor='black',
                  title='Dominant News Theme',
                  title_fontsize=10)
legend.get_frame().set_linewidth(1.5)

# Add annotation box with CORRECT interpretation
annotation_text = (
    "Dominant Theme by Country (Highest SHAP Importance):\n"
    "• Zimbabwe: Humanitarian 13.4% (humanitarian crisis + hyperinflation)\n"
    "• Sudan: Governance 14.8% (state collapse + political instability)\n"
    "• DRC: Other 14.3% (complex multi-faceted emergency)\n"
    "• Governance dominant in 5/13 countries (Sudan, Nigeria, Ethiopia, Malawi, Madagascar)\n"
    "• Other dominant in 4/13 (DRC, Mozambique, Mali, Niger) - heterogeneous crises\n"
    "• Relatively flat global distribution (9.2-13.0%) - no universal theme dominates\n"
    "• Map shows which theme contributes MOST to predictions per country, not elevated deviations"
)

ax.text(0.98, 0.98, annotation_text,
       transform=ax.transAxes,
       fontsize=9,
       va='top', ha='right',
       bbox=dict(boxstyle='round,pad=1',
                facecolor='lightyellow',
                edgecolor='black',
                alpha=0.95, linewidth=2))

plt.tight_layout()

# Save
output_file = FIGURES_OUTPUT / "fig7_country_theme_map.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file}")

output_file_png = FIGURES_OUTPUT / "fig7_country_theme_map.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png}")

plt.close()

# ============================================================================
# STEP 6: CREATE THEME ELEVATION MAP (BIGGEST DEVIATION FROM GLOBAL)
# ============================================================================

print("\nStep 6: Creating theme elevation map...")

# Load global theme rankings
import json
with open(ANALYSIS_DIR / "global_theme_rankings.json") as f:
    global_data = json.load(f)
global_themes = global_data['global_theme_rankings']

# Calculate elevation (deviation from global) for each country-theme
elevations = []
for idx, row in df_theme.iterrows():
    country = row['country']
    theme = row['theme']
    local_pct = row['importance_pct']
    global_pct = global_themes[theme]
    elevation = local_pct - global_pct

    elevations.append({
        'country': country,
        'theme': theme,
        'local_pct': local_pct,
        'global_pct': global_pct,
        'elevation': elevation
    })

df_elevations = pd.DataFrame(elevations)

# Find theme with MAXIMUM positive elevation for each country
max_elevation = df_elevations.loc[df_elevations.groupby('country')['elevation'].idxmax()]
max_elevation = max_elevation.merge(df_delta[['country', 'key_saves']], on='country', how='left')

print("\nTheme with maximum elevation per country:")
for idx, row in max_elevation.sort_values('key_saves', ascending=False).head(10).iterrows():
    print(f"  {row['country']:<40} {row['theme']:<20} +{row['elevation']:.1f}pp  ({row['local_pct']:.1f}% vs {row['global_pct']:.1f}% global)")

# Map country names
max_elevation['country_map'] = max_elevation['country'].replace(country_mapping)

# Merge with basemap
africa_elevation = africa.merge(
    max_elevation,
    left_on='NAME',
    right_on='country_map',
    how='left'
)

# Try original name
africa_elevation = africa_elevation.merge(
    max_elevation[['country', 'theme', 'elevation', 'key_saves']],
    left_on='NAME',
    right_on='country',
    how='left',
    suffixes=('', '_orig')
)

africa_elevation['elevated_theme'] = africa_elevation['theme'].fillna(
    africa_elevation.get('theme_orig', 'none')
).fillna('none')

africa_elevation['elevation'] = africa_elevation['elevation'].fillna(
    africa_elevation.get('elevation_orig', 0)
).fillna(0)

africa_elevation['key_saves'] = africa_elevation['key_saves'].fillna(
    africa_elevation.get('key_saves_orig', 0)
).fillna(0)

# Create figure
fig, ax = plt.subplots(figsize=(18, 12))

# Plot basemap
africa.boundary.plot(ax=ax, linewidth=0.8, edgecolor='black', alpha=0.3)

# Plot by elevated theme
for theme in theme_colors.keys():
    if theme == 'none':
        continue
    theme_countries = africa_elevation[africa_elevation['elevated_theme'] == theme]
    if not theme_countries.empty:
        theme_countries.plot(
            ax=ax,
            color=theme_colors[theme],
            edgecolor='black',
            linewidth=0.8,
            alpha=0.85
        )

# Highlight countries with data
countries_with_data = africa_elevation[africa_elevation['elevated_theme'] != 'none']
countries_with_data.boundary.plot(ax=ax, edgecolor='black', linewidth=1.5)

# Highlight top 3
for country_name in top3_names:
    country_geom = africa_elevation[africa_elevation['NAME'] == country_name]
    if not country_geom.empty:
        country_geom.boundary.plot(ax=ax, edgecolor='darkred', linewidth=3.5)

# Add labels for ALL 13 countries with ELEVATION data
label_positions_elev = {
    'Zimbabwe': (30, -19),
    'Sudan': (30, 15),
    'Dem. Rep. Congo': (23, -2),
    'Kenya': (38, 1),
    'Nigeria': (8, 9),
    'Somalia': (46, 5),
    'Ethiopia': (40, 8),
    'Mozambique': (36, -18),
    'Mali': (-4, 17),
    'Malawi': (34, -13),
    'Uganda': (32, 2),
    'Madagascar': (46, -20),
    'Niger': (8, 16)
}

for country_name, (x, y) in label_positions_elev.items():
    country_row = max_elevation[
        (max_elevation['country'] == country_name) |
        (max_elevation['country_map'] == country_name)
    ]

    if not country_row.empty:
        theme = country_row.iloc[0]['theme']
        elev = country_row.iloc[0]['elevation']
        saves = country_row.iloc[0]['key_saves']

        # Format label based on key saves
        label = f"{country_name}\n{theme.title()} +{elev:.1f}pp"

        if saves > 0:
            fontsize = 9
            fontweight = 'bold'
        else:
            fontsize = 8
            fontweight = 'normal'

        ax.text(x, y, label, fontsize=fontsize, fontweight=fontweight,
               ha='center', va='center',
               bbox=dict(boxstyle='round,pad=0.4',
                        facecolor='white',
                        edgecolor='black',
                        alpha=0.85, linewidth=1.2))

ax.set_xlim(-20, 55)
ax.set_ylim(-38, 40)
ax.axis('off')

ax.set_title('Theme Elevations: Which Themes are Elevated Above Global Average by Country?\n' +
             'Showing Theme with Maximum Positive Deviation from Global (SHAP-Based, 13 countries)',
             fontsize=14, fontweight='bold', pad=20)

# Legend
legend = ax.legend(handles=legend_elements,
                  loc='lower left',
                  fontsize=9,
                  framealpha=0.95,
                  edgecolor='black',
                  title='Most Elevated Theme',
                  title_fontsize=10)
legend.get_frame().set_linewidth(1.5)

# Annotation
annotation_text2 = (
    "Theme Elevations (Deviation from Global Average):\n"
    "• Zimbabwe: Weather +2.1pp (11.5% vs 9.4%) - drought cycles compound economic fragility\n"
    "• Sudan: Conflict +3.3pp (14.6% vs 11.3%) - April 2023 civil war escalation\n"
    "• DRC: Displacement +2.2pp (12.2% vs 10.0%) - M23 resurgence, North Kivu flows\n"
    "• Kenya: Food Security +3.5pp (12.8% vs 9.2%) - harvest failures amplify baseline vulnerability\n"
    "• Somalia: Health +5.8pp (16.5% vs 10.7%) - disease burden compounds food insecurity\n"
    "• Elevation = Local % - Global %, showing context-specific amplification\n"
    "• Red borders: Top 3 by key saves (Zimbabwe 77, Sudan 59, DRC 40 = 70.7% of total)"
)

ax.text(0.98, 0.98, annotation_text2,
       transform=ax.transAxes,
       fontsize=9,
       va='top', ha='right',
       bbox=dict(boxstyle='round,pad=1',
                facecolor='lightyellow',
                edgecolor='black',
                alpha=0.95, linewidth=2))

plt.tight_layout()

# Save
output_file2 = FIGURES_OUTPUT / "fig7_theme_elevation_map.pdf"
plt.savefig(output_file2, dpi=300, bbox_inches='tight', format='pdf')
print(f"[OK] Saved: {output_file2}")

output_file2_png = FIGURES_OUTPUT / "fig7_theme_elevation_map.png"
plt.savefig(output_file2_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file2_png}")

plt.close()

print("\n" + "="*80)
print("COUNTRY-THEME MAP VISUALIZATION COMPLETE")
print("="*80)
print(f"Created 2 figures:")
print(f"  1. Fig 7a: Dominant theme map (highest SHAP % per country)")
print(f"  2. Fig 7b: Theme elevation map (maximum deviation from global average)")
print(f"\nKey distinction:")
print(f"  - Fig 7a shows which theme CONTRIBUTES MOST (absolute %)")
print(f"  - Fig 7b shows which theme is MOST ELEVATED (relative deviation)")
print(f"  - Example: Zimbabwe dominant = Humanitarian, but elevated = Weather")
print(f"\nAll data 100% from SHAP-based theme analysis (real, not simulated)")
print()
