"""
Geographic Heterogeneity: Publication-Grade Visualizations
===========================================================
State-of-the-art visualization suite for RQ5 geographic heterogeneity analysis

Visualizations:
--------------
1. Delta-AUC Country Rankings (horizontal bar chart with categories)
2. Geographic Heterogeneity Map (Africa map with delta-AUC shading)
3. Key Saves Distribution (country-level concentration analysis)
4. Cascade Benefit Matrix (heatmap: countries ? metrics)
5. News Value Scatter (key saves vs delta-AUC with annotations)

Publication Quality:
-------------------
- 300 DPI resolution
- Professional color schemes
- Clear typography (Arial, 10-14pt)
- Comprehensive captions
- Statistical annotations

Date: 2026-01-06
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from matplotlib.patches import Rectangle
import warnings
from config import BASE_DIR
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

BASE_DIR = Path(str(BASE_DIR))
ANALYSIS_DIR = BASE_DIR / "Dissertation Write Up" / "GEOGRAPHIC_HETEROGENEITY_ANALYSIS"
FIGURES_DIR = BASE_DIR / "Dissertation Write Up" / "LATEX_DISSERTATION" / "figures" / "ch05_geographic_heterogeneity"

# Ensure output directory exists
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# Professional color schemes
COLORS = {
    'high_benefit': '#27AE60',      # Green - high news value
    'moderate_benefit': '#F39C12',  # Orange - moderate value
    'minimal_benefit': '#95A5A6',   # Gray - minimal value
    'ar_superior': '#E74C3C',       # Red - AR better
    'zimbabwe': '#C62828',          # Dark red (consistent with dissertation)
    'sudan': '#1565C0',             # Dark blue (consistent with dissertation)
    'drc': '#6A1B9A',               # Dark purple (consistent with dissertation)
    'other': '#455A64'              # Blue-gray
}

print("="*80)
print("GEOGRAPHIC HETEROGENEITY: PUBLICATION VISUALIZATIONS")
print("="*80)
print(f"\nFigures output: {FIGURES_DIR}")
print()

# ============================================================================
# LOAD DATA
# ============================================================================

print("-" * 80)
print("Loading analysis results...")
print("-" * 80)

data_file = ANALYSIS_DIR / "country_delta_auc_analysis.csv"
country_data = pd.read_csv(data_file)
print(f"\n[OK] Loaded: {data_file.name}")
print(f"  Countries: {len(country_data)}")
print(f"  Columns: {list(country_data.columns)}")

# ============================================================================
# FIGURE 1: DELTA-AUC COUNTRY RANKINGS
# ============================================================================

print("\n" + "-" * 80)
print("FIGURE 1: Delta-AUC Country Rankings")
print("-" * 80)

fig, ax = plt.subplots(figsize=(14, 10))

# Sort by delta-AUC
country_data_sorted = country_data.sort_values('delta_auc', ascending=True)

# Assign colors based on news value category
color_map = {
    'High Benefit': COLORS['high_benefit'],
    'Moderate Benefit': COLORS['moderate_benefit'],
    'Minimal Benefit': COLORS['minimal_benefit'],
    'AR Superior': COLORS['ar_superior']
}

colors = [color_map.get(cat, '#95A5A6') for cat in country_data_sorted['news_value_category']]

# Create horizontal bar chart
y_pos = np.arange(len(country_data_sorted))
bars = ax.barh(y_pos, country_data_sorted['delta_auc'], color=colors,
               edgecolor='black', linewidth=1, alpha=0.85)

# Add vertical line at zero
ax.axvline(x=0, color='black', linestyle='--', linewidth=1.5, alpha=0.7)

# Customize axes
ax.set_yticks(y_pos)
ax.set_yticklabels(country_data_sorted['country'], fontsize=10)
ax.set_xlabel('Delta--AUC (Cascade - AR Baseline)', fontsize=12, fontweight='bold')
ax.set_title('Geographic Heterogeneity: News Feature Value by Country\n' +
             'Delta--AUC Measures Marginal Improvement from News-Based Cascade over AR Baseline',
             fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels on bars
for i, (bar, val, country) in enumerate(zip(bars, country_data_sorted['delta_auc'],
                                              country_data_sorted['country'])):
    # Position text based on positive/negative value
    if val >= 0:
        x_pos = val + 0.005
        ha = 'left'
    else:
        x_pos = val - 0.005
        ha = 'right'

    ax.text(x_pos, i, f'{val:+.3f}', va='center', ha=ha,
            fontsize=8, fontweight='bold')

# Add legend
legend_elements = [
    mpatches.Patch(facecolor=COLORS['high_benefit'], edgecolor='black', label='High Benefit (Rescue ?15%)'),
    mpatches.Patch(facecolor=COLORS['moderate_benefit'], edgecolor='black', label='Moderate Benefit (Delta--AUC >0.02)'),
    mpatches.Patch(facecolor=COLORS['minimal_benefit'], edgecolor='black', label='Minimal Benefit (-0.02 < Delta--AUC ?0.02)'),
    mpatches.Patch(facecolor=COLORS['ar_superior'], edgecolor='black', label='AR Superior (Delta--AUC ?-0.02)')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9,
          framealpha=0.95, edgecolor='black', title='News Value Category')

# Add summary annotation
mean_delta = country_data['delta_auc'].mean()
std_delta = country_data['delta_auc'].std()
annotation_text = (f"Mean Delta--AUC: {mean_delta:+.4f}\n"
                   f"Std Dev: {std_delta:.4f}\n"
                   f"Range: {country_data['delta_auc'].min():.3f} to {country_data['delta_auc'].max():.3f}")
ax.text(0.02, 0.98, annotation_text, transform=ax.transAxes,
        fontsize=9, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                 alpha=0.9, edgecolor='black', linewidth=1.5))

plt.tight_layout()

# Save
output_file = FIGURES_DIR / "fig1_delta_auc_country_rankings.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file.name}")

output_file_png = FIGURES_DIR / "fig1_delta_auc_country_rankings.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png.name}")

plt.close()

# ============================================================================
# FIGURE 2: KEY SAVES GEOGRAPHIC CONCENTRATION
# ============================================================================

print("\n" + "-" * 80)
print("FIGURE 2: Key Saves Geographic Concentration")
print("-" * 80)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Panel A: Key Saves by Country (Top 10)
top10_saves = country_data.nlargest(10, 'key_saves').sort_values('key_saves', ascending=True)

# Highlight Zimbabwe, Sudan, DRC
colors_top10 = []
for country in top10_saves['country']:
    if country == 'Zimbabwe':
        colors_top10.append(COLORS['zimbabwe'])
    elif country == 'Sudan':
        colors_top10.append(COLORS['sudan'])
    elif country == 'Democratic Republic of the Congo':
        colors_top10.append(COLORS['drc'])
    else:
        colors_top10.append(COLORS['other'])

y_pos = np.arange(len(top10_saves))
bars = ax1.barh(y_pos, top10_saves['key_saves'], color=colors_top10,
                edgecolor='black', linewidth=1.2, alpha=0.9)

ax1.set_yticks(y_pos)
ax1.set_yticklabels(top10_saves['country'], fontsize=10)
ax1.set_xlabel('Number of Key Saves (Crises Rescued)', fontsize=11, fontweight='bold')
ax1.set_title('A) Top 10 Countries by Key Saves\nGeographic Concentration of News-Based Early Warning Value',
              fontsize=12, fontweight='bold', pad=10)
ax1.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top10_saves['key_saves'])):
    ax1.text(val + 1, i, f'{int(val)}', va='center', ha='left',
            fontsize=9, fontweight='bold')

# Calculate and annotate concentration
top3_saves = country_data.nlargest(3, 'key_saves')['key_saves'].sum()
total_saves = country_data['key_saves'].sum()
concentration = (top3_saves / total_saves) * 100

annotation_text = (f"Top 3 Concentration:\n"
                   f"{top3_saves:.0f} / {total_saves:.0f} saves\n"
                   f"= {concentration:.1f}%")
ax1.text(0.98, 0.98, annotation_text, transform=ax1.transAxes,
        fontsize=9, verticalalignment='top', horizontalalignment='right',
        bbox=dict(boxstyle='round,pad=0.5', facecolor='#FFF9E6',
                 alpha=0.95, edgecolor='darkgoldenrod', linewidth=2))

# Panel B: Rescue Rate (Key Save Rate) by Country (Top 10)
top10_rate = country_data.nlargest(10, 'key_save_rate').sort_values('key_save_rate', ascending=True)

# Highlight Zimbabwe, Sudan, DRC
colors_rate = []
for country in top10_rate['country']:
    if country == 'Zimbabwe':
        colors_rate.append(COLORS['zimbabwe'])
    elif country == 'Sudan':
        colors_rate.append(COLORS['sudan'])
    elif country == 'Democratic Republic of the Congo':
        colors_rate.append(COLORS['drc'])
    else:
        colors_rate.append(COLORS['other'])

y_pos2 = np.arange(len(top10_rate))
bars2 = ax2.barh(y_pos2, top10_rate['key_save_rate'] * 100, color=colors_rate,
                 edgecolor='black', linewidth=1.2, alpha=0.9)

ax2.set_yticks(y_pos2)
ax2.set_yticklabels(top10_rate['country'], fontsize=10)
ax2.set_xlabel('Key Save Rate (% of AR Failures Rescued)', fontsize=11, fontweight='bold')
ax2.set_title('B) Top 10 Countries by Rescue Efficiency\nPercentage of AR Failures Successfully Predicted by Cascade',
              fontsize=12, fontweight='bold', pad=10)
ax2.grid(axis='x', alpha=0.3, linestyle='--')

# Add value labels
for i, (bar, val) in enumerate(zip(bars2, top10_rate['key_save_rate'] * 100)):
    ax2.text(val + 0.5, i, f'{val:.1f}%', va='center', ha='left',
            fontsize=9, fontweight='bold')

# Add legend for country highlighting
legend_elements2 = [
    mpatches.Patch(facecolor=COLORS['zimbabwe'], edgecolor='black', label='Zimbabwe (Top #1)'),
    mpatches.Patch(facecolor=COLORS['sudan'], edgecolor='black', label='Sudan (Top #2)'),
    mpatches.Patch(facecolor=COLORS['drc'], edgecolor='black', label='DRC (Top #3)'),
    mpatches.Patch(facecolor=COLORS['other'], edgecolor='black', label='Other Countries')
]
ax2.legend(handles=legend_elements2, loc='lower right', fontsize=8,
          framealpha=0.95, edgecolor='black')

plt.tight_layout()

# Save
output_file = FIGURES_DIR / "fig2_key_saves_concentration.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file.name}")

output_file_png = FIGURES_DIR / "fig2_key_saves_concentration.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png.name}")

plt.close()

# ============================================================================
# FIGURE 3: NEWS VALUE SCATTER (KEY SAVES VS DELTA-AUC)
# ============================================================================

print("\n" + "-" * 80)
print("FIGURE 3: News Value Scatter Analysis")
print("-" * 80)

fig, ax = plt.subplots(figsize=(14, 10))

# Scatter plot with category-based coloring
for category in country_data['news_value_category'].unique():
    subset = country_data[country_data['news_value_category'] == category]
    ax.scatter(subset['delta_auc'], subset['key_saves'],
              c=color_map.get(category, '#95A5A6'),
              s=subset['n_crisis'] * 0.5,  # Size by crisis count
              alpha=0.75, edgecolors='black', linewidth=1.5,
              label=category)

# Annotate top performers and notable outliers
top_countries_annotate = country_data.nlargest(5, 'key_saves')['country'].tolist()
for _, row in country_data.iterrows():
    if row['country'] in top_countries_annotate:
        ax.annotate(row['country'], (row['delta_auc'], row['key_saves']),
                   xytext=(10, 5), textcoords='offset points',
                   fontsize=9, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor='white',
                            alpha=0.8, edgecolor='black'),
                   arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0.2',
                                 color='black', lw=1.5))

ax.set_xlabel('Delta--AUC (Cascade - AR Baseline)', fontsize=12, fontweight='bold')
ax.set_ylabel('Number of Key Saves (Crises Rescued)', fontsize=12, fontweight='bold')
ax.set_title('News Feature Value: Key Saves vs Model Performance Improvement\n' +
             'Bubble size represents total crisis count in country',
             fontsize=14, fontweight='bold', pad=15)
ax.grid(alpha=0.3, linestyle='--')
ax.axvline(x=0, color='black', linestyle='--', linewidth=1, alpha=0.5)
ax.axhline(y=0, color='black', linestyle='--', linewidth=1, alpha=0.5)

# Legend
ax.legend(loc='upper left', fontsize=10, framealpha=0.95, edgecolor='black',
         title='News Value Category')

# Add correlation annotation
from scipy.stats import spearmanr
corr, p_value = spearmanr(country_data['delta_auc'], country_data['key_saves'])
corr_text = f"Spearman ? = {corr:.3f}\np-value = {p_value:.4f}"
ax.text(0.98, 0.02, corr_text, transform=ax.transAxes,
       fontsize=10, verticalalignment='bottom', horizontalalignment='right',
       bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                alpha=0.95, edgecolor='black', linewidth=1.5))

plt.tight_layout()

# Save
output_file = FIGURES_DIR / "fig3_news_value_scatter.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file.name}")

output_file_png = FIGURES_DIR / "fig3_news_value_scatter.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png.name}")

plt.close()

# ============================================================================
# FIGURE 4: CASCADE BENEFIT MATRIX (HEATMAP)
# ============================================================================

print("\n" + "-" * 80)
print("FIGURE 4: Cascade Benefit Matrix Heatmap")
print("-" * 80)

# Select top 12 countries by key saves for heatmap
top12 = country_data.nlargest(12, 'key_saves').sort_values('key_saves', ascending=False)

# Select key metrics for heatmap
metrics_for_heatmap = ['delta_auc', 'key_save_rate', 'recall_improvement']
metric_labels = ['Delta--AUC\n(Performance)', 'Rescue Rate\n(% AR Failures)', 'Recall Gain\n(Percentage Points)']

# Create matrix
matrix_data = top12[metrics_for_heatmap].values.T

# Normalize each row for better visualization (0-1 scale per metric)
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
matrix_normalized = scaler.fit_transform(matrix_data.T).T

fig, ax = plt.subplots(figsize=(14, 8))

# Create heatmap
im = ax.imshow(matrix_normalized, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)

# Set ticks
ax.set_xticks(np.arange(len(top12)))
ax.set_yticks(np.arange(len(metrics_for_heatmap)))
ax.set_xticklabels(top12['country'], rotation=45, ha='right', fontsize=10)
ax.set_yticklabels(metric_labels, fontsize=11)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
cbar.set_label('Normalized Score (0=Worst, 1=Best)', rotation=270, labelpad=20,
               fontsize=11, fontweight='bold')

# Add value annotations (show actual values, not normalized)
for i in range(len(metrics_for_heatmap)):
    for j in range(len(top12)):
        actual_val = matrix_data[i, j]
        # Format based on metric type
        if i == 0:  # Delta-AUC
            text = f'{actual_val:+.3f}'
        elif i == 1:  # Rescue rate
            text = f'{actual_val*100:.1f}%'
        else:  # Recall improvement
            text = f'{actual_val*100:+.1f}pp'

        color = 'white' if matrix_normalized[i, j] < 0.5 else 'black'
        ax.text(j, i, text, ha='center', va='center',
               color=color, fontsize=9, fontweight='bold')

ax.set_title('Cascade Benefit Matrix: Top 12 Countries by Key Saves\n' +
             'Normalized scores across key performance metrics',
             fontsize=14, fontweight='bold', pad=15)

plt.tight_layout()

# Save
output_file = FIGURES_DIR / "fig4_cascade_benefit_matrix.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file.name}")

output_file_png = FIGURES_DIR / "fig4_cascade_benefit_matrix.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png.name}")

plt.close()

print("\n" + "="*80)
print("PUBLICATION VISUALIZATIONS COMPLETE")
print("="*80)
print(f"\nGenerated 4 publication-grade figures:")
print(f"  1. Delta-AUC Country Rankings")
print(f"  2. Key Saves Geographic Concentration")
print(f"  3. News Value Scatter Analysis")
print(f"  4. Cascade Benefit Matrix Heatmap")
print(f"\nAll figures saved to: {FIGURES_DIR}")
print()

