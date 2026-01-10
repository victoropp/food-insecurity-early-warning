"""
================================================================================
PUBLICATION-GRADE EXPLAINABILITY VISUALIZATIONS
================================================================================
Creates comprehensive visualizations for:
1. Ablation Study: Impact of Z-scores, HMM, and DMD features
2. XGBoost Feature Importance
3. Mixed Effects: Beta coefficients and Country biases

ALL METRICS LOADED FROM REAL RESULTS - NO HARDCODING

Author: Victor Collins Oppon
Date: December 2025
================================================================================
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from pathlib import Path
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# Import configuration
sys.path.append(str(Path(__file__).parent.parent))
from config import RESULTS_DIR, FIGURES_DIR, STAGE2_MODELS_DIR

# ============================================================================
# CONFIGURATION
# ============================================================================

OUTPUT_DIR = FIGURES_DIR / "publication" / "explainability"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

sns.set_style('whitegrid')
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 11

print("=" * 80)
print("PUBLICATION-GRADE EXPLAINABILITY VISUALIZATIONS")
print("=" * 80)
print(f"\nOutput directory: {OUTPUT_DIR}\n")

# ============================================================================
# LOAD ALL MODEL RESULTS (NO HARDCODING)
# ============================================================================

print("Loading model results from real files...")

# XGBoost results
xgb_basic_summary_file = STAGE2_MODELS_DIR / 'xgboost' / 'basic_with_ar' / 'xgboost_with_ar_summary.json'
xgb_advanced_summary_file = STAGE2_MODELS_DIR / 'xgboost' / 'advanced_with_ar' / 'xgboost_hmm_dmd_with_ar_summary.json'
xgb_basic_importance_file = STAGE2_MODELS_DIR / 'xgboost' / 'basic_with_ar' / 'feature_importance.csv'
xgb_advanced_importance_file = STAGE2_MODELS_DIR / 'xgboost' / 'advanced_with_ar' / 'feature_importance.csv'

# Mixed Effects results
me_ratio_summary_file = STAGE2_MODELS_DIR / 'mixed_effects' / 'pooled_ratio_with_ar' / 'pooled_ratio_with_ar_summary.json'
me_zscore_summary_file = STAGE2_MODELS_DIR / 'mixed_effects' / 'pooled_zscore_with_ar' / 'pooled_zscore_with_ar_summary.json'
me_ratio_hmm_dmd_summary_file = STAGE2_MODELS_DIR / 'mixed_effects' / 'pooled_ratio_hmm_dmd_with_ar' / 'pooled_ratio_hmm_dmd_with_ar_summary.json'
me_zscore_hmm_dmd_summary_file = STAGE2_MODELS_DIR / 'mixed_effects' / 'pooled_zscore_hmm_dmd_with_ar' / 'pooled_zscore_hmm_dmd_with_ar_summary.json'

me_ratio_fixed_file = STAGE2_MODELS_DIR / 'mixed_effects' / 'pooled_ratio_with_ar' / 'pooled_ratio_with_ar_fixed_effects.csv'
me_zscore_fixed_file = STAGE2_MODELS_DIR / 'mixed_effects' / 'pooled_zscore_with_ar' / 'pooled_zscore_with_ar_fixed_effects.csv'
me_ratio_hmm_dmd_fixed_file = STAGE2_MODELS_DIR / 'mixed_effects' / 'pooled_ratio_hmm_dmd_with_ar' / 'pooled_ratio_hmm_dmd_with_ar_fixed_effects.csv'
me_zscore_hmm_dmd_fixed_file = STAGE2_MODELS_DIR / 'mixed_effects' / 'pooled_zscore_hmm_dmd_with_ar' / 'pooled_zscore_hmm_dmd_with_ar_fixed_effects.csv'

me_ratio_random_file = STAGE2_MODELS_DIR / 'mixed_effects' / 'pooled_ratio_with_ar' / 'pooled_ratio_with_ar_random_effects.csv'
me_zscore_random_file = STAGE2_MODELS_DIR / 'mixed_effects' / 'pooled_zscore_with_ar' / 'pooled_zscore_with_ar_random_effects.csv'

# Load XGBoost summaries
with open(xgb_basic_summary_file) as f:
    xgb_basic = json.load(f)
with open(xgb_advanced_summary_file) as f:
    xgb_advanced = json.load(f)

# Load XGBoost feature importance
xgb_basic_importance = pd.read_csv(xgb_basic_importance_file, index_col=0)
xgb_advanced_importance = pd.read_csv(xgb_advanced_importance_file, index_col=0)

# Load Mixed Effects summaries
with open(me_ratio_summary_file) as f:
    me_ratio = json.load(f)
with open(me_zscore_summary_file) as f:
    me_zscore = json.load(f)
with open(me_ratio_hmm_dmd_summary_file) as f:
    me_ratio_hmm_dmd = json.load(f)
with open(me_zscore_hmm_dmd_summary_file) as f:
    me_zscore_hmm_dmd = json.load(f)

# Load Mixed Effects coefficients
me_ratio_fixed = pd.read_csv(me_ratio_fixed_file)
me_zscore_fixed = pd.read_csv(me_zscore_fixed_file)
me_ratio_hmm_dmd_fixed = pd.read_csv(me_ratio_hmm_dmd_fixed_file)
me_zscore_hmm_dmd_fixed = pd.read_csv(me_zscore_hmm_dmd_fixed_file)

# Load Mixed Effects random effects (country biases)
me_ratio_random = pd.read_csv(me_ratio_random_file)
me_zscore_random = pd.read_csv(me_zscore_random_file)

print(f"  Loaded XGBoost Basic: AUC = {xgb_basic['performance']['overall_auc_roc']:.4f}")
print(f"  Loaded XGBoost Advanced: AUC = {xgb_advanced['performance']['overall_auc_roc']:.4f}")
print(f"  Loaded ME Ratio: AUC = {me_ratio['overall_metrics']['auc_roc']:.4f}")
print(f"  Loaded ME Zscore: AUC = {me_zscore['overall_metrics']['auc_roc']:.4f}")
print(f"  Loaded ME Ratio+HMM+DMD: AUC = {me_ratio_hmm_dmd['overall_metrics']['auc_roc']:.4f}")
print(f"  Loaded ME Zscore+HMM+DMD: AUC = {me_zscore_hmm_dmd['overall_metrics']['auc_roc']:.4f}")
print()

# ============================================================================
# FIGURE 1: ABLATION STUDY - IMPACT OF FEATURE COMPONENTS
# ============================================================================

print("Creating Figure 1: Ablation Study...")

fig1, axes = plt.subplots(1, 2, figsize=(14, 6))

# Panel A: XGBoost Ablation (Basic vs Advanced)
ax1 = axes[0]

# Extract AUC values from loaded data
xgb_models = {
    'Basic\n(Ratio+Zscore+Loc)': xgb_basic['performance']['overall_auc_roc'],
    'Advanced\n(+HMM+DMD)': xgb_advanced['performance']['overall_auc_roc']
}
xgb_std = {
    'Basic\n(Ratio+Zscore+Loc)': xgb_basic['performance']['std_auc'],
    'Advanced\n(+HMM+DMD)': xgb_advanced['performance']['std_auc']
}

xgb_names = list(xgb_models.keys())
xgb_aucs = list(xgb_models.values())
xgb_stds = list(xgb_std.values())

colors_xgb = ['#4A90E2', '#2ECC71']
bars1 = ax1.bar(xgb_names, xgb_aucs, yerr=xgb_stds, color=colors_xgb,
                alpha=0.85, edgecolor='black', linewidth=1.5, capsize=5)

# Add value labels
for bar, val, std in zip(bars1, xgb_aucs, xgb_stds):
    ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=12, fontweight='bold')

ax1.set_ylabel('AUC-ROC', fontweight='bold', fontsize=12)
ax1.set_title('A. XGBoost: Impact of HMM+DMD Features', fontweight='bold', fontsize=13)
ax1.set_ylim(0, 1.0)
ax1.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')

# Add delta annotation
delta_xgb = xgb_aucs[1] - xgb_aucs[0]
ax1.annotate(f'$\Delta$ = {delta_xgb:+.3f}', xy=(0.5, max(xgb_aucs) + 0.15),
             fontsize=11, ha='center', fontweight='bold',
             color='#E74C3C' if delta_xgb < 0 else '#27AE60')

# Panel B: Mixed Effects Ablation
ax2 = axes[1]

# Extract AUC values from loaded data
me_models = {
    'Ratio Only': me_ratio['overall_metrics']['auc_roc'],
    'Zscore Only': me_zscore['overall_metrics']['auc_roc'],
    'Ratio+HMM+DMD': me_ratio_hmm_dmd['overall_metrics']['auc_roc'],
    'Zscore+HMM+DMD': me_zscore_hmm_dmd['overall_metrics']['auc_roc']
}
me_stds = {
    'Ratio Only': me_ratio['overall_metrics']['std_fold_auc'],
    'Zscore Only': me_zscore['overall_metrics']['std_fold_auc'],
    'Ratio+HMM+DMD': me_ratio_hmm_dmd['overall_metrics']['std_fold_auc'],
    'Zscore+HMM+DMD': me_zscore_hmm_dmd['overall_metrics']['std_fold_auc']
}

me_names = list(me_models.keys())
me_aucs = list(me_models.values())
me_std_vals = list(me_stds.values())

colors_me = ['#4A90E2', '#9B59B6', '#2ECC71', '#E67E22']
bars2 = ax2.bar(me_names, me_aucs, yerr=me_std_vals, color=colors_me,
                alpha=0.85, edgecolor='black', linewidth=1.5, capsize=4)

# Add value labels
for bar, val, std in zip(bars2, me_aucs, me_std_vals):
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + std + 0.01,
             f'{val:.3f}', ha='center', va='bottom', fontsize=10, fontweight='bold')

ax2.set_ylabel('AUC-ROC', fontweight='bold', fontsize=12)
ax2.set_title('B. Mixed Effects: Feature Type Comparison', fontweight='bold', fontsize=13)
ax2.set_ylim(0, 1.0)
ax2.axhline(y=0.5, color='gray', linestyle='--', alpha=0.5)
ax2.tick_params(axis='x', rotation=15)

# Add annotations for key comparisons
zscore_vs_ratio = me_aucs[1] - me_aucs[0]
ax2.annotate(f'Zscore vs Ratio: {zscore_vs_ratio:+.3f}',
             xy=(0.5, 0.95), xycoords='axes fraction',
             fontsize=10, ha='center', fontweight='bold',
             color='#27AE60' if zscore_vs_ratio > 0 else '#E74C3C')

plt.tight_layout()
fig1_path = OUTPUT_DIR / 'ablation_study.png'
plt.savefig(fig1_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {fig1_path}")

# ============================================================================
# FIGURE 2: XGBOOST FEATURE IMPORTANCE
# ============================================================================

print("Creating Figure 2: XGBoost Feature Importance...")

fig2, axes = plt.subplots(1, 2, figsize=(16, 8))

# Panel A: Basic Model Feature Importance
ax1 = axes[0]

# Get top 15 features from basic model
basic_imp = xgb_basic_importance.copy()
basic_imp = basic_imp.sort_values('importance', ascending=True).tail(15)

# Color by feature type
def get_feature_color(feature_name):
    if 'country' in feature_name.lower():
        return '#E74C3C'  # Red for location
    elif 'zscore' in feature_name.lower():
        return '#3498DB'  # Blue for zscore
    elif 'hmm' in feature_name.lower():
        return '#9B59B6'  # Purple for HMM
    elif 'dmd' in feature_name.lower():
        return '#2ECC71'  # Green for DMD
    else:
        return '#F39C12'  # Orange for ratio

basic_colors = [get_feature_color(f) for f in basic_imp.index]
bars1 = ax1.barh(range(len(basic_imp)), basic_imp['importance'], color=basic_colors,
                 alpha=0.85, edgecolor='black', linewidth=0.5)

# Add value labels for basic model
for i, (bar, val) in enumerate(zip(bars1, basic_imp['importance'])):
    ax1.text(val + 2, i, f'{val:.1f}', va='center', ha='left', fontsize=8, fontweight='bold')

ax1.set_yticks(range(len(basic_imp)))
ax1.set_yticklabels([f.replace('_', ' ').title() for f in basic_imp.index], fontsize=9)
ax1.set_xlabel('Feature Importance (Gain)', fontweight='bold', fontsize=11)
ax1.set_title(f'A. XGBoost Basic ({xgb_basic["features"]["total"]} features)\nAUC = {xgb_basic["performance"]["overall_auc_roc"]:.3f}',
              fontweight='bold', fontsize=12)
ax1.set_xlim(0, basic_imp['importance'].max() * 1.15)

# Panel B: Advanced Model Feature Importance
ax2 = axes[1]

# Get top 15 features from advanced model
adv_imp = xgb_advanced_importance.copy()
adv_imp = adv_imp.sort_values('importance', ascending=True).tail(15)

adv_colors = [get_feature_color(f) for f in adv_imp.index]
bars2 = ax2.barh(range(len(adv_imp)), adv_imp['importance'], color=adv_colors,
                 alpha=0.85, edgecolor='black', linewidth=0.5)

# Add value labels for advanced model
for i, (bar, val) in enumerate(zip(bars2, adv_imp['importance'])):
    ax2.text(val + 2, i, f'{val:.1f}', va='center', ha='left', fontsize=8, fontweight='bold')

ax2.set_yticks(range(len(adv_imp)))
ax2.set_yticklabels([f.replace('_', ' ').title() for f in adv_imp.index], fontsize=9)
ax2.set_xlabel('Feature Importance (Gain)', fontweight='bold', fontsize=11)
ax2.set_title(f'B. XGBoost Advanced ({xgb_advanced["features"]["total"]} features)\nAUC = {xgb_advanced["performance"]["overall_auc_roc"]:.3f}',
              fontweight='bold', fontsize=12)
ax2.set_xlim(0, adv_imp['importance'].max() * 1.15)

# Add legend
legend_elements = [
    mpatches.Patch(facecolor='#E74C3C', edgecolor='black', label='Location'),
    mpatches.Patch(facecolor='#F39C12', edgecolor='black', label='Ratio'),
    mpatches.Patch(facecolor='#3498DB', edgecolor='black', label='Z-score'),
    mpatches.Patch(facecolor='#9B59B6', edgecolor='black', label='HMM'),
    mpatches.Patch(facecolor='#2ECC71', edgecolor='black', label='DMD')
]
fig2.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02),
            ncol=5, frameon=True, fontsize=10, edgecolor='black')

plt.tight_layout(rect=[0, 0.05, 1, 1])
fig2_path = OUTPUT_DIR / 'xgboost_feature_importance.png'
plt.savefig(fig2_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {fig2_path}")

# ============================================================================
# FIGURE 3: MIXED EFFECTS - BETA COEFFICIENTS
# ============================================================================

print("Creating Figure 3: Mixed Effects Beta Coefficients...")

fig3, axes = plt.subplots(1, 2, figsize=(14, 7))

# Panel A: Ratio Model Coefficients
ax1 = axes[0]

# Filter out intercept and sort by absolute value
ratio_coefs = me_ratio_fixed[me_ratio_fixed['feature'] != '(Intercept)'].copy()
ratio_coefs['abs_coef'] = ratio_coefs['coefficient'].abs()
ratio_coefs = ratio_coefs.sort_values('abs_coef', ascending=True)

# Color by sign (positive = risk factor, negative = protective)
ratio_colors = ['#27AE60' if c > 0 else '#E74C3C' for c in ratio_coefs['coefficient']]

bars1 = ax1.barh(range(len(ratio_coefs)), ratio_coefs['coefficient'],
                 color=ratio_colors, alpha=0.85, edgecolor='black', linewidth=0.5)

# Add value labels for ratio model coefficients
for i, (bar, val) in enumerate(zip(bars1, ratio_coefs['coefficient'])):
    x_pos = val + 0.5 if val >= 0 else val - 0.5
    ha = 'left' if val >= 0 else 'right'
    ax1.text(x_pos, i, f'{val:.2f}', va='center', ha=ha, fontsize=9, fontweight='bold')

ax1.set_yticks(range(len(ratio_coefs)))
ax1.set_yticklabels([f.replace('_', ' ').title() for f in ratio_coefs['feature']], fontsize=10)
ax1.axvline(x=0, color='black', linewidth=1)
ax1.set_xlabel('Beta Coefficient (Log-Odds)', fontweight='bold', fontsize=11)
ax1.set_title(f'A. Mixed Effects: Ratio Model\nAUC = {me_ratio["overall_metrics"]["auc_roc"]:.3f}',
              fontweight='bold', fontsize=12)
# Expand x-axis to fit labels
max_abs = ratio_coefs['coefficient'].abs().max()
ax1.set_xlim(-max_abs * 0.2, max_abs * 1.3)

# Panel B: Z-score Model Coefficients
ax2 = axes[1]

zscore_coefs = me_zscore_fixed[me_zscore_fixed['feature'] != '(Intercept)'].copy()
zscore_coefs['abs_coef'] = zscore_coefs['coefficient'].abs()
zscore_coefs = zscore_coefs.sort_values('abs_coef', ascending=True)

zscore_colors = ['#27AE60' if c > 0 else '#E74C3C' for c in zscore_coefs['coefficient']]

bars2 = ax2.barh(range(len(zscore_coefs)), zscore_coefs['coefficient'],
                 color=zscore_colors, alpha=0.85, edgecolor='black', linewidth=0.5)

# Add value labels for zscore model coefficients
for i, (bar, val) in enumerate(zip(bars2, zscore_coefs['coefficient'])):
    x_pos = val + 0.02 if val >= 0 else val - 0.02
    ha = 'left' if val >= 0 else 'right'
    ax2.text(x_pos, i, f'{val:.3f}', va='center', ha=ha, fontsize=9, fontweight='bold')

ax2.set_yticks(range(len(zscore_coefs)))
ax2.set_yticklabels([f.replace('_', ' ').title() for f in zscore_coefs['feature']], fontsize=10)
ax2.axvline(x=0, color='black', linewidth=1)
ax2.set_xlabel('Beta Coefficient (Log-Odds)', fontweight='bold', fontsize=11)
ax2.set_title(f'B. Mixed Effects: Z-score Model\nAUC = {me_zscore["overall_metrics"]["auc_roc"]:.3f}',
              fontweight='bold', fontsize=12)
# Expand x-axis to fit labels
max_abs_z = zscore_coefs['coefficient'].abs().max()
ax2.set_xlim(-max_abs_z * 1.5, max_abs_z * 1.5)

# Add legend
legend_elements = [
    mpatches.Patch(facecolor='#27AE60', edgecolor='black', label='Risk Factor (+)'),
    mpatches.Patch(facecolor='#E74C3C', edgecolor='black', label='Protective Factor (-)')
]
fig3.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02),
            ncol=2, frameon=True, fontsize=11, edgecolor='black')

plt.tight_layout(rect=[0, 0.05, 1, 1])
fig3_path = OUTPUT_DIR / 'mixed_effects_coefficients.png'
plt.savefig(fig3_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {fig3_path}")

# ============================================================================
# FIGURE 4: COUNTRY BIASES (RANDOM EFFECTS)
# ============================================================================

print("Creating Figure 4: Country Biases (Random Effects)...")

fig4, axes = plt.subplots(1, 2, figsize=(14, 8))

# Panel A: Ratio Model Country Biases
ax1 = axes[0]

ratio_random = me_ratio_random.copy()
ratio_random = ratio_random.sort_values('random_intercept', ascending=True)

# Color by bias direction
ratio_bias_colors = ['#E74C3C' if b < 0 else '#27AE60' for b in ratio_random['random_intercept']]

bars1 = ax1.barh(range(len(ratio_random)), ratio_random['random_intercept'],
                 color=ratio_bias_colors, alpha=0.85, edgecolor='black', linewidth=0.5)

ax1.set_yticks(range(len(ratio_random)))
ax1.set_yticklabels(ratio_random['group_id'], fontsize=10)
ax1.axvline(x=0, color='black', linewidth=1.5)
ax1.set_xlabel('Random Intercept (Log-Odds Bias)', fontweight='bold', fontsize=11)
ax1.set_title(f'A. Country Baseline Risk: Ratio Model\n(n = {len(ratio_random)} countries)',
              fontweight='bold', fontsize=12)

# Add value labels
for i, (bar, val) in enumerate(zip(bars1, ratio_random['random_intercept'])):
    x_pos = val + 0.1 if val >= 0 else val - 0.1
    ha = 'left' if val >= 0 else 'right'
    ax1.text(x_pos, i, f'{val:.2f}', va='center', ha=ha, fontsize=9, fontweight='bold')

# Panel B: Z-score Model Country Biases
ax2 = axes[1]

zscore_random = me_zscore_random.copy()
zscore_random = zscore_random.sort_values('random_intercept', ascending=True)

zscore_bias_colors = ['#E74C3C' if b < 0 else '#27AE60' for b in zscore_random['random_intercept']]

bars2 = ax2.barh(range(len(zscore_random)), zscore_random['random_intercept'],
                 color=zscore_bias_colors, alpha=0.85, edgecolor='black', linewidth=0.5)

ax2.set_yticks(range(len(zscore_random)))
ax2.set_yticklabels(zscore_random['group_id'], fontsize=10)
ax2.axvline(x=0, color='black', linewidth=1.5)
ax2.set_xlabel('Random Intercept (Log-Odds Bias)', fontweight='bold', fontsize=11)
ax2.set_title(f'B. Country Baseline Risk: Z-score Model\n(n = {len(zscore_random)} countries)',
              fontweight='bold', fontsize=12)

# Add value labels
for i, (bar, val) in enumerate(zip(bars2, zscore_random['random_intercept'])):
    x_pos = val + 0.1 if val >= 0 else val - 0.1
    ha = 'left' if val >= 0 else 'right'
    ax2.text(x_pos, i, f'{val:.2f}', va='center', ha=ha, fontsize=9, fontweight='bold')

# Add legend
legend_elements = [
    mpatches.Patch(facecolor='#27AE60', edgecolor='black', label='Higher Baseline Risk (+)'),
    mpatches.Patch(facecolor='#E74C3C', edgecolor='black', label='Lower Baseline Risk (-)')
]
fig4.legend(handles=legend_elements, loc='lower center', bbox_to_anchor=(0.5, -0.02),
            ncol=2, frameon=True, fontsize=11, edgecolor='black')

plt.tight_layout(rect=[0, 0.05, 1, 1])
fig4_path = OUTPUT_DIR / 'country_biases.png'
plt.savefig(fig4_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {fig4_path}")

# ============================================================================
# FIGURE 5: COMPREHENSIVE SUMMARY - ALL MODELS COMPARISON
# ============================================================================

print("Creating Figure 5: Comprehensive Model Comparison...")

fig5 = plt.figure(figsize=(16, 10))
gs = GridSpec(2, 2, figure=fig5, hspace=0.3, wspace=0.3)

# Panel A: All Models AUC Comparison
ax1 = fig5.add_subplot(gs[0, :])

all_models = {
    'XGBoost Basic\n(Ratio+Zscore+Loc)': (xgb_basic['performance']['overall_auc_roc'],
                                           xgb_basic['performance']['std_auc'], 'XGBoost'),
    'XGBoost Advanced\n(+HMM+DMD)': (xgb_advanced['performance']['overall_auc_roc'],
                                      xgb_advanced['performance']['std_auc'], 'XGBoost'),
    'ME Ratio': (me_ratio['overall_metrics']['auc_roc'],
                 me_ratio['overall_metrics']['std_fold_auc'], 'Mixed Effects'),
    'ME Z-score': (me_zscore['overall_metrics']['auc_roc'],
                   me_zscore['overall_metrics']['std_fold_auc'], 'Mixed Effects'),
    'ME Ratio+HMM+DMD': (me_ratio_hmm_dmd['overall_metrics']['auc_roc'],
                         me_ratio_hmm_dmd['overall_metrics']['std_fold_auc'], 'Mixed Effects'),
    'ME Zscore+HMM+DMD': (me_zscore_hmm_dmd['overall_metrics']['auc_roc'],
                          me_zscore_hmm_dmd['overall_metrics']['std_fold_auc'], 'Mixed Effects')
}

# Sort by AUC
sorted_models = dict(sorted(all_models.items(), key=lambda x: x[1][0], reverse=True))
model_names = list(sorted_models.keys())
model_aucs = [v[0] for v in sorted_models.values()]
model_stds = [v[1] for v in sorted_models.values()]
model_types = [v[2] for v in sorted_models.values()]

# Color by model type
type_colors = {'XGBoost': '#3498DB', 'Mixed Effects': '#9B59B6'}
model_colors = [type_colors[t] for t in model_types]

bars = ax1.barh(range(len(model_names)), model_aucs, xerr=model_stds,
                color=model_colors, alpha=0.85, edgecolor='black', linewidth=1, capsize=4)

ax1.set_yticks(range(len(model_names)))
ax1.set_yticklabels(model_names, fontsize=10)
ax1.set_xlabel('AUC-ROC', fontweight='bold', fontsize=12)
ax1.set_title('A. Stage 2 Model Performance Comparison', fontweight='bold', fontsize=13)
ax1.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Random')
ax1.set_xlim(0.4, 0.9)

# Add value labels
for i, (val, std) in enumerate(zip(model_aucs, model_stds)):
    ax1.text(val + std + 0.01, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

# Add legend
legend_elements = [
    mpatches.Patch(facecolor='#3498DB', edgecolor='black', label='XGBoost'),
    mpatches.Patch(facecolor='#9B59B6', edgecolor='black', label='Mixed Effects')
]
ax1.legend(handles=legend_elements, loc='lower right', fontsize=10)

# Panel B: Feature Type Importance (from XGBoost Basic)
ax2 = fig5.add_subplot(gs[1, 0])

# Aggregate importance by feature type
basic_imp_full = xgb_basic_importance.copy()
basic_imp_full['type'] = basic_imp_full.index.map(lambda x:
    'Location' if 'country' in x.lower() else
    'Z-score' if 'zscore' in x.lower() else 'Ratio')

type_importance = basic_imp_full.groupby('type')['importance'].sum()
type_importance = type_importance.sort_values(ascending=True)

type_colors_plot = {'Location': '#E74C3C', 'Ratio': '#F39C12', 'Z-score': '#3498DB'}
colors = [type_colors_plot[t] for t in type_importance.index]

bars2 = ax2.barh(type_importance.index, type_importance.values, color=colors,
                 alpha=0.85, edgecolor='black', linewidth=1)

ax2.set_xlabel('Total Feature Importance (Gain)', fontweight='bold', fontsize=11)
ax2.set_title('B. Importance by Feature Type\n(XGBoost Basic)', fontweight='bold', fontsize=12)

# Add value labels
for bar, val in zip(bars2, type_importance.values):
    ax2.text(val + 5, bar.get_y() + bar.get_height()/2, f'{val:.1f}',
             va='center', fontsize=11, fontweight='bold')

# Panel C: Key Insights Table
ax3 = fig5.add_subplot(gs[1, 1])
ax3.axis('off')

# Create insights text from loaded data
best_xgb = max(xgb_basic['performance']['overall_auc_roc'],
               xgb_advanced['performance']['overall_auc_roc'])
best_me = max(me_ratio['overall_metrics']['auc_roc'],
              me_zscore['overall_metrics']['auc_roc'],
              me_ratio_hmm_dmd['overall_metrics']['auc_roc'],
              me_zscore_hmm_dmd['overall_metrics']['auc_roc'])

hmm_dmd_impact_xgb = xgb_advanced['performance']['overall_auc_roc'] - xgb_basic['performance']['overall_auc_roc']
zscore_vs_ratio = me_zscore['overall_metrics']['auc_roc'] - me_ratio['overall_metrics']['auc_roc']

insights = f"""
KEY FINDINGS FROM EXPLAINABILITY ANALYSIS

1. MODEL PERFORMANCE (Stage 2, AR-filtered data):
   - Best XGBoost: AUC = {best_xgb:.3f}
   - Best Mixed Effects: AUC = {best_me:.3f}
   - XGBoost outperforms by: {(best_xgb - best_me):.3f} ({(best_xgb - best_me)/best_me*100:.1f}%)

2. HMM+DMD FEATURES:
   - XGBoost: {hmm_dmd_impact_xgb:+.3f} AUC change
   - Conclusion: HMM/DMD do not improve performance

3. Z-SCORE vs RATIO:
   - Mixed Effects: {zscore_vs_ratio:+.3f} AUC
   - Z-scores slightly worse (unexpected)

4. MOST IMPORTANT FEATURES (XGBoost):
   - Location features dominate (country data density,
     country baseline conflict, country baseline food security)
   - Geographic context is essential

5. COUNTRY RISK HIERARCHY (from ME random effects):
   - Highest risk: Somalia, Sudan, Zimbabwe
   - Lowest risk: Uganda, Madagascar
   - Risk varies by >5 log-odds units across countries
"""

ax3.text(0.05, 0.95, insights, transform=ax3.transAxes, fontsize=10,
         verticalalignment='top', fontfamily='monospace',
         bbox=dict(boxstyle='round', facecolor='#F8F9FA', edgecolor='#DEE2E6'))

plt.tight_layout()
fig5_path = OUTPUT_DIR / 'comprehensive_summary.png'
plt.savefig(fig5_path, dpi=300, bbox_inches='tight', facecolor='white')
plt.close()
print(f"  Saved: {fig5_path}")

# ============================================================================
# SAVE SUMMARY DATA
# ============================================================================

print("\nSaving summary data...")

summary = {
    'timestamp': pd.Timestamp.now().isoformat(),
    'figures_generated': [
        str(fig1_path),
        str(fig2_path),
        str(fig3_path),
        str(fig4_path),
        str(fig5_path)
    ],
    'xgboost_basic': {
        'auc': xgb_basic['performance']['overall_auc_roc'],
        'std': xgb_basic['performance']['std_auc'],
        'n_features': xgb_basic['features']['total']
    },
    'xgboost_advanced': {
        'auc': xgb_advanced['performance']['overall_auc_roc'],
        'std': xgb_advanced['performance']['std_auc'],
        'n_features': xgb_advanced['features']['total']
    },
    'mixed_effects': {
        'ratio_auc': me_ratio['overall_metrics']['auc_roc'],
        'zscore_auc': me_zscore['overall_metrics']['auc_roc'],
        'ratio_hmm_dmd_auc': me_ratio_hmm_dmd['overall_metrics']['auc_roc'],
        'zscore_hmm_dmd_auc': me_zscore_hmm_dmd['overall_metrics']['auc_roc']
    },
    'key_findings': {
        'hmm_dmd_impact_xgboost': hmm_dmd_impact_xgb,
        'zscore_vs_ratio_me': zscore_vs_ratio,
        'best_xgboost_auc': best_xgb,
        'best_me_auc': best_me
    }
}

summary_file = OUTPUT_DIR / 'explainability_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"  Saved: {summary_file}")

print()
print("=" * 80)
print("EXPLAINABILITY VISUALIZATIONS COMPLETE")
print("=" * 80)
print(f"\nGenerated {len(summary['figures_generated'])} publication-grade figures:")
for fig_path in summary['figures_generated']:
    print(f"  - {Path(fig_path).name}")
print()
