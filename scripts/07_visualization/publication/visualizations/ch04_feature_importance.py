"""
Figure 15: Feature Importance Rankings
Publication-grade horizontal bar chart showing top features from XGBoost Advanced model

Date: 2026-01-05
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import json
from config import BASE_DIR

# Directories
BASE_DIR = Path(str(BASE_DIR))
DATA_DIR = BASE_DIR / "Dissertation Write Up" / "DATA_INVENTORIES"
OUTPUT_DIR = BASE_DIR / "Dissertation Write Up" / "LATEX_DISSERTATION" / "figures" / "ch04_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load feature importance from JSON
print("Loading feature importance data...")
with open(DATA_DIR / "MASTER_METRICS_ALL_MODELS.json", 'r') as f:
    data = json.load(f)

# Extract XGBoost Advanced feature importance
xgb_advanced = data['xgboost']['advanced']
top_features = xgb_advanced['top_features']

# Convert to DataFrame and sort
# Convert importance to percentage
df_features = pd.DataFrame(list(top_features.items()), columns=['feature', 'importance'])
df_features['importance'] = df_features['importance'] * 100  # Convert to percentage
df_features = df_features.sort_values('importance', ascending=True)  # Ascending for horizontal bars

# Take all available features (JSON has top 10)
top_n = len(df_features)
df_top = df_features

print(f"\nTop {top_n} Features by Importance:")
for idx, row in df_top.iterrows():
    print(f"  {row['feature']}: {row['importance']:.3f}%")

# Clean up feature names for display
def clean_feature_name(name):
    """Make feature names more readable"""
    # Replace underscores with spaces
    name = name.replace('_', ' ')
    # Capitalize first letter of each word
    name = ' '.join(word.capitalize() for word in name.split())
    return name

df_top['display_name'] = df_top['feature'].apply(clean_feature_name)

# Categorize features by type
def categorize_feature(name):
    """Assign feature to category for color coding"""
    name_lower = name.lower()
    if 'country' in name_lower or 'data density' in name_lower or 'baseline' in name_lower:
        return 'Location'
    elif 'ratio' in name_lower and 'hmm' not in name_lower and 'dmd' not in name_lower:
        return 'Ratio'
    elif 'zscore' in name_lower and 'hmm' not in name_lower and 'dmd' not in name_lower:
        return 'Z-score'
    elif 'hmm' in name_lower:
        return 'HMM'
    elif 'dmd' in name_lower:
        return 'DMD'
    else:
        return 'Other'

df_top['category'] = df_top['feature'].apply(categorize_feature)

# Color mapping by category
color_map = {
    'Location': '#E74C3C',  # Red
    'Ratio': '#3498DB',     # Blue
    'Z-score': '#9B59B6',   # Purple
    'HMM': '#27AE60',       # Green
    'DMD': '#F39C12',       # Orange
    'Other': '#95A5A6'      # Gray
}

colors = [color_map[cat] for cat in df_top['category']]

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# Create figure
fig, ax = plt.subplots(figsize=(12, 10))

# Create horizontal bar chart
y_pos = np.arange(len(df_top))
bars = ax.barh(y_pos, df_top['importance'], color=colors,
               alpha=0.8, edgecolor='black', linewidth=1.5)

# Add value labels
for i, (idx, row) in enumerate(df_top.iterrows()):
    ax.text(row['importance'] + 0.3, i, f"{row['importance']:.2f}%",
            va='center', fontsize=9, fontweight='bold')

# Formatting
ax.set_yticks(y_pos)
ax.set_yticklabels(df_top['display_name'], fontsize=9)
ax.set_xlabel('Feature Importance (%)', fontsize=12, fontweight='bold')
ax.set_title(f'XGBoost Advanced: Top {top_n} Features by Importance',
             fontsize=14, fontweight='bold', pad=15)
ax.grid(axis='x', alpha=0.3, linestyle='--')

# Extend x-axis for labels
ax.set_xlim(0, max(df_top['importance']) * 1.25)

# Add legend OUTSIDE plot area
from matplotlib.patches import Patch
legend_elements = []
for cat, color in color_map.items():
    count = (df_top['category'] == cat).sum()
    if count > 0:
        legend_elements.append(
            Patch(facecolor=color, alpha=0.8, edgecolor='black', linewidth=1.5,
                  label=f'{cat} ({count})')
        )

ax.legend(handles=legend_elements, fontsize=9, loc='center left',
          bbox_to_anchor=(1.02, 0.5), framealpha=0.95,
          title='Feature Category', title_fontsize=10)

# Add summary box highlighting news feature contributions
total_importance = df_top['importance'].sum()
hmm_importance = df_top[df_top['category']=='HMM']['importance'].sum()
news_importance = df_top[df_top['category'].isin(['Ratio', 'Z-score'])]['importance'].sum()

summary_text = (
    f"HMM regime features: {hmm_importance:.1f}%\n"
    f"News categories: {news_importance:.1f}%\n"
    f"Complementary signals"
)
ax.text(0.02, 0.98, summary_text, transform=ax.transAxes,
        fontsize=9, va='top', ha='left',
        bbox=dict(boxstyle='round,pad=1.0', facecolor='lightgreen', alpha=0.7, edgecolor='darkgreen', linewidth=1.5))

# Footer - emphasizing unique value of news features
footer_text = (
    f"Feature importance from XGBoost Advanced model (35 features total) trained on WITH_AR_FILTER subset "
    f"(6,553 observations, 534 districts where AR baseline got it wrong). "
    f"Beyond geographic priors, news features provide complementary crisis signals: "
    f"HMM ratio transition risk (#5, 3.2%) captures regime shifts from stable to crisis-prone narratives; "
    f"news category composition (other, health, weather ratios: {df_top[df_top['category']=='Ratio']['importance'].sum():.1f}%) "
    f"identifies media emphasis patterns; "
    f"Z-score anomalies (displacement, food security: {df_top[df_top['category']=='Z-score']['importance'].sum():.1f}%) "
    f"detect temporal deviations signaling rapid-onset shocks. "
    f"HMM crisis probability (#10, 2.5%) confirms regime detection adds orthogonal signal to compositional features. "
    f"These news features target cases where geographic persistence fails—precisely where early warning matters most. "
    f"5-fold stratified spatial cross-validation, h=8 months."
)
fig.text(0.5, 0.01, footer_text, ha='center', fontsize=8, style='italic',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3),
         wrap=True)

plt.tight_layout(rect=[0, 0.06, 1, 0.98])

# Save
output_file = OUTPUT_DIR / "ch04_feature_importance.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file}")

output_file_png = OUTPUT_DIR / "ch04_feature_importance.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png}")

plt.close()

print("\n" + "="*80)
print(f"FIGURE 15 COMPLETE: Feature Importance Rankings (Top {top_n})")
print("="*80)
print(f"HMM features: {hmm_importance:.1f}% importance")
print(f"News category features: {news_importance:.1f}% importance")
print(f"Categories represented: {', '.join(sorted(df_top['category'].unique()))}")
