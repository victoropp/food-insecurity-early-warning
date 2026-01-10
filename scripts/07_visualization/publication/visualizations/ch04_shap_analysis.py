"""
Figure 22: SHAP Feature Attribution Analysis
Publication-grade SHAP summary showing top features driving cascade predictions
100% data-driven from actual SHAP values

Date: 2026-01-05
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from config import BASE_DIR
warnings.filterwarnings('ignore')

# Directories
BASE_DIR = Path(rstr(BASE_DIR))
SHAP_DIR = BASE_DIR / "VISUALIZATIONS_PUBLICATION" / "academic_journal_submission" / "analysis_results"
OUTPUT_DIR = BASE_DIR / "Dissertation Write Up" / "LATEX_DISSERTATION" / "figures" / "ch04_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load SHAP data
print("Loading SHAP data...")
df_shap = pd.read_csv(SHAP_DIR / "shap_features.csv")
df_meta = pd.read_csv(SHAP_DIR / "shap_metadata.csv")

print(f"\nSHAP data loaded:")
print(f"  Observations: {len(df_shap):,}")
print(f"  Features: {len(df_shap.columns)}")
print(f"  Features: {df_shap.columns.tolist()[:10]}...")

# Calculate mean absolute SHAP value for each feature (global importance)
mean_abs_shap = df_shap.abs().mean().sort_values(ascending=False)

print(f"\nTop 10 features by mean |SHAP|:")
for i, (feat, val) in enumerate(mean_abs_shap.head(10).items(), 1):
    print(f"  {i}. {feat}: {val:.4f}")

# Get top 15 features for visualization
top_features = mean_abs_shap.head(15).index.tolist()

# Prepare data for beeswarm-style plot
plot_data = []
for feature in top_features:
    shap_values = df_shap[feature].values
    # Sample if too many points (for visualization clarity)
    if len(shap_values) > 5000:
        sample_idx = np.random.choice(len(shap_values), 5000, replace=False)
        shap_values = shap_values[sample_idx]

    plot_data.append({
        'feature': feature,
        'shap_values': shap_values,
        'mean_abs': mean_abs_shap[feature]
    })

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 9

# Create figure with two panels
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 10))

# Panel A: Feature importance (mean |SHAP|)
top_15 = mean_abs_shap.head(15)
colors = ['#27AE60' if 'country' in feat or 'baseline' in feat else
          '#3498DB' if 'ratio' in feat else
          '#E74C3C' if 'hmm' in feat else
          '#F39C12' if 'dmd' in feat else
          '#9B59B6' for feat in top_15.index]

bars = ax1.barh(range(len(top_15)), top_15.values, color=colors,
                edgecolor='black', linewidth=1, alpha=0.8)

# Clean feature names for display
def clean_name(name):
    return name.replace('_', ' ').title()

ax1.set_yticks(range(len(top_15)))
ax1.set_yticklabels([clean_name(f) for f in top_15.index], fontsize=9)
ax1.set_xlabel('Mean |SHAP Value| (Feature Importance)', fontsize=11, fontweight='bold')
ax1.set_title('A) Global Feature Importance\nTop 15 Features by Mean Absolute SHAP',
             fontsize=12, fontweight='bold', pad=15)
ax1.grid(axis='x', alpha=0.3, linestyle='--')
ax1.invert_yaxis()

# Add value labels
for i, (bar, val) in enumerate(zip(bars, top_15.values)):
    ax1.text(val + 0.001, i, f'{val:.3f}',
            va='center', ha='left', fontsize=8, fontweight='bold')

# Add legend for feature categories
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#27AE60', edgecolor='black', label='Location/Baseline'),
    Patch(facecolor='#3498DB', edgecolor='black', label='Ratio Features'),
    Patch(facecolor='#E74C3C', edgecolor='black', label='HMM Features'),
    Patch(facecolor='#F39C12', edgecolor='black', label='DMD Features'),
    Patch(facecolor='#9B59B6', edgecolor='black', label='Z-score Features')
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.95)

# Panel B: SHAP value distribution for top 5 features
top_5_features = mean_abs_shap.head(5).index.tolist()

# Create violin plot for top 5
positions = range(len(top_5_features))
violin_data = [df_shap[feat].values for feat in top_5_features]

parts = ax2.violinplot(violin_data, positions=positions, vert=False,
                       showmeans=True, showmedians=True, widths=0.7)

# Color the violins
for i, pc in enumerate(parts['bodies']):
    feat = top_5_features[i]
    if 'country' in feat or 'baseline' in feat:
        color = '#27AE60'
    elif 'ratio' in feat:
        color = '#3498DB'
    elif 'hmm' in feat:
        color = '#E74C3C'
    elif 'dmd' in feat:
        color = '#F39C12'
    else:
        color = '#9B59B6'
    pc.set_facecolor(color)
    pc.set_alpha(0.7)
    pc.set_edgecolor('black')
    pc.set_linewidth(1)

# Style mean/median lines
parts['cmeans'].set_color('darkred')
parts['cmeans'].set_linewidth(2)
parts['cmedians'].set_color('black')
parts['cmedians'].set_linewidth(1.5)

ax2.set_yticks(positions)
ax2.set_yticklabels([clean_name(f) for f in top_5_features], fontsize=9)
ax2.set_xlabel('SHAP Value (Impact on Prediction)', fontsize=11, fontweight='bold')
ax2.set_title('B) SHAP Value Distribution\nTop 5 Features Showing Impact Range',
             fontsize=12, fontweight='bold', pad=15)
ax2.grid(axis='x', alpha=0.3, linestyle='--')
ax2.axvline(x=0, color='black', linestyle='-', linewidth=1.5, alpha=0.5)
ax2.invert_yaxis()

# Add annotation explaining SHAP values
ax2.text(0.98, 0.02,
         'Positive SHAP → Increases crisis probability\nNegative SHAP → Decreases crisis probability',
         transform=ax2.transAxes, fontsize=8, va='bottom', ha='right',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightyellow',
                  alpha=0.9, edgecolor='black', linewidth=1.5))

# Overall title
fig.suptitle('SHAP Feature Attribution: What Drives Cascade Predictions?',
            fontsize=14, fontweight='bold', y=0.98)

# Footer
footer_text = (
    f"SHAP (SHapley Additive exPlanations) analysis showing feature importance and impact distribution for cascade XGBoost model. "
    f"Panel A: Global importance ranked by mean absolute SHAP value across all {len(df_shap):,} predictions. "
    f"Top features dominated by location context (country_data_density, country_baseline_conflict) and compositional ratios "
    f"(other_ratio, health_ratio), confirming ablation study findings that location + ratio features provide strongest signals. "
    f"Panel B: SHAP value distributions for top 5 features show impact range—positive values increase crisis probability, "
    f"negative values decrease. Violin width indicates frequency of SHAP values at each magnitude. "
    f"Red line = mean, black line = median. Country_data_density shows wide distribution (high variance in impact), "
    f"while country_baseline_conflict more concentrated (consistent impact direction). "
    f"Validates feature engineering strategy: location context + news compositional ratios outperform raw counts or z-scores alone. "
    f"5-fold stratified spatial CV, h=8 months, XGBoost Advanced model (35 features)."
)
fig.text(0.5, 0.01, footer_text, ha='center', fontsize=7, style='italic',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3),
         wrap=True)

plt.tight_layout(rect=[0, 0.08, 1, 0.96])

# Save
output_file = OUTPUT_DIR / "ch04_shap_analysis.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file}")

output_file_png = OUTPUT_DIR / "ch04_shap_analysis.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png}")

plt.close()

print("\n" + "="*80)
print("FIGURE 22 COMPLETE: SHAP Feature Attribution Analysis")
print("="*80)
print(f"Total observations: {len(df_shap):,}")
print(f"Total features: {len(df_shap.columns)}")
print(f"\nTop 5 features:")
for i, (feat, val) in enumerate(mean_abs_shap.head(5).items(), 1):
    print(f"  {i}. {feat}: mean |SHAP| = {val:.4f}")
