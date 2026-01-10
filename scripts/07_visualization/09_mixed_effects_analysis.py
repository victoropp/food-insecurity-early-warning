#!/usr/bin/env python3
"""
Author: Victor Collins Oppon
MSc Data Science Dissertation
Middlesex University, 2025
"""

"""
Mixed Effects Model Analysis - Dedicated Visualization Suite
=============================================================
Comprehensive analysis of 4 mixed effects models with country-level random effects.

Note: pooled_ratio_hmm_dmd_with_ar was recently corrected on Dec 24, 2025.
The model now properly handles observations with missing HMM features (100%
missing after AR filter) and extreme DMD amplitude values by excluding
problematic features and using only summary DMD metrics (6 features instead
of 21). This improved convergence from 2/5 folds to 5/5 folds, increasing
coverage from 2,545 to 6,553 observations.

Creates 6 visualizations:
1. Performance Ranking - Horizontal bar chart of all 4 models by AUC
2. Country Variance - Box plots showing country-level AUC distribution
3. Feature Importance Comparison - Grouped bars comparing fixed effects
4. ME vs XGBoost Positioning - Complexity-performance trade-off scatter
5. Precision-Recall Curves - All 4 models overlaid
6. Variance Components - Within vs between-country variance decomposition

Date: December 24, 2025
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # MUST be before importing pyplot (non-interactive backend)
import matplotlib.pyplot as plt
# NOTE: seaborn removed - causes hanging on Windows
from datetime import datetime
import json

from config import RESULTS_DIR, FIGURES_DIR, VISUALIZATION_CONFIG

# Set publication style
plt.rcParams['axes.grid'] = True
plt.rcParams['grid.alpha'] = 0.3
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.size'] = 10

print("=" * 80)
print("VISUALIZATION: MIXED EFFECTS MODEL ANALYSIS")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Output directory
OUTPUT_DIR = FIGURES_DIR / 'mixed_effects'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# =============================================================================
# DATA LOADING
# =============================================================================

print("Loading mixed effects model data...")

ME_DIR = RESULTS_DIR / 'stage2_models' / 'mixed_effects'

# List of 4 mixed effects models
ME_MODELS = [
    'pooled_ratio_with_ar',
    'pooled_zscore_with_ar',
    'pooled_ratio_hmm_dmd_with_ar',
    'pooled_zscore_hmm_dmd_with_ar'
]

# Load data for each model
model_data = {}

for model_name in ME_MODELS:
    model_dir = ME_DIR / model_name
    print(f"\n  Loading {model_name}...")

    data = {}

    # Load summary JSON
    summary_file = model_dir / f"{model_name}_summary.json"
    if summary_file.exists():
        with open(summary_file, 'r') as f:
            data['summary'] = json.load(f)
        print(f"    [OK] Summary loaded")
    else:
        print(f"    [SKIP] Summary not found: {summary_file}")
        continue

    # Load predictions CSV
    pred_file = model_dir / f"{model_name}_predictions.csv"
    if pred_file.exists():
        data['predictions'] = pd.read_csv(pred_file)
        print(f"    [OK] Predictions loaded ({len(data['predictions'])} obs)")
    else:
        print(f"    [SKIP] Predictions not found: {pred_file}")

    # Load CV results
    cv_file = model_dir / f"{model_name}_cv_results.csv"
    if cv_file.exists():
        data['cv_results'] = pd.read_csv(cv_file)
        print(f"    [OK] CV results loaded ({len(data['cv_results'])} folds)")
    else:
        print(f"    [SKIP] CV results not found: {cv_file}")

    # Load country metrics
    country_file = model_dir / f"{model_name}_country_metrics.csv"
    if country_file.exists():
        data['country_metrics'] = pd.read_csv(country_file)
        print(f"    [OK] Country metrics loaded ({len(data['country_metrics'])} countries)")
    else:
        print(f"    [SKIP] Country metrics not found: {country_file}")

    # Load threshold analysis
    threshold_file = model_dir / f"{model_name}_threshold_analysis.csv"
    if threshold_file.exists():
        data['threshold_analysis'] = pd.read_csv(threshold_file)
        print(f"    [OK] Threshold analysis loaded")
    else:
        print(f"    [SKIP] Threshold analysis not found: {threshold_file}")

    model_data[model_name] = data

# Load full model ranking for context
ranking_file = RESULTS_DIR / 'analysis' / 'model_ranking_table.csv'
if ranking_file.exists():
    full_ranking = pd.read_csv(ranking_file)
    print(f"\n[OK] Full model ranking loaded ({len(full_ranking)} models)")
else:
    full_ranking = None
    print("\n[SKIP] Full model ranking not found")

# =============================================================================
# FIGURE 1: PERFORMANCE RANKING (Horizontal Bar Chart)
# =============================================================================

print("\n" + "=" * 80)
print("FIGURE 1: Performance Ranking")
print("=" * 80)

fig, ax = plt.subplots(figsize=(12, 6))

# Prepare data
perf_data = []
for model_name in ME_MODELS:
    if model_name in model_data and 'summary' in model_data[model_name]:
        summary = model_data[model_name]['summary']
        auc = summary['overall_metrics']['mean_fold_auc']
        perf_data.append({
            'model': model_name,
            'auc': auc,
            'feature_type': summary.get('feature_type', ''),
            'n_obs': summary.get('n_observations', 0)
        })

perf_df = pd.DataFrame(perf_data).sort_values('auc', ascending=True)

# Color by feature type
colors = []
for ft in perf_df['feature_type']:
    if 'hmm' in str(ft).lower():
        colors.append('#2196F3' if 'ratio' in str(ft).lower() else '#4CAF50')
    else:
        colors.append('#FF9800' if 'ratio' in str(ft).lower() else '#9C27B0')

# Create horizontal bar chart
bars = ax.barh(range(len(perf_df)), perf_df['auc'], color=colors,
               edgecolor='black', linewidth=0.5, alpha=0.8)

# Add value labels
for i, (bar, val) in enumerate(zip(bars, perf_df['auc'])):
    ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.4f}',
            va='center', fontsize=10, fontweight='bold')

# Format
ax.set_yticks(range(len(perf_df)))
ax.set_yticklabels(perf_df['model'].str.replace('pooled_', '').str.replace('_', ' '),
                   fontsize=11)
ax.set_xlabel('AUC-ROC', fontsize=12, fontweight='bold')
ax.set_title('Mixed Effects Model Performance Ranking\n(Country-Level Random Effects)',
             fontsize=14, fontweight='bold')
ax.set_xlim(0.75, max(perf_df['auc']) * 1.05)
ax.grid(axis='x', alpha=0.3)

# Legend
from matplotlib.patches import Patch
legend_elements = [
    Patch(facecolor='#FF9800', label='Ratio'),
    Patch(facecolor='#9C27B0', label='Z-score'),
    Patch(facecolor='#2196F3', label='Ratio+HMM+DMD'),
    Patch(facecolor='#4CAF50', label='Z-score+HMM+DMD')
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10)

plt.tight_layout()
fig_file = OUTPUT_DIR / '01_performance_ranking.png'
plt.savefig(fig_file, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
plt.close()
print(f"[OK] Saved: {fig_file}")

# =============================================================================
# FIGURE 2: COUNTRY VARIANCE (Box Plots)
# =============================================================================

print("\n" + "=" * 80)
print("FIGURE 2: Country Variance Analysis")
print("=" * 80)

fig, ax = plt.subplots(figsize=(14, 7))

# Prepare data for box plots
country_data_list = []
for model_name in ME_MODELS:
    if model_name in model_data and 'country_metrics' in model_data[model_name]:
        cm = model_data[model_name]['country_metrics']
        for _, row in cm.iterrows():
            country_data_list.append({
                'model': model_name.replace('pooled_', '').replace('_', '\n'),
                'country': row.get('ipc_country', row.get('country', 'Unknown')),
                'auc': row.get('auc_roc', row.get('auc', np.nan))
            })

if country_data_list:
    country_df = pd.DataFrame(country_data_list)

    # Box plot using matplotlib (avoid seaborn boxplot issue)
    models = country_df['model'].unique()
    positions = range(len(models))
    boxplot_data = [country_df[country_df['model'] == m]['auc'].dropna() for m in models]

    bp = ax.boxplot(boxplot_data, positions=positions, patch_artist=True,
                    widths=0.6, showfliers=True)

    # Color the boxes
    colors = plt.cm.Set2(range(len(models)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    # Overlay strip plot (scatter points)
    for i, model in enumerate(models):
        model_subset = country_df[country_df['model'] == model]
        x = np.random.normal(i, 0.04, size=len(model_subset))  # Add jitter
        ax.scatter(x, model_subset['auc'], alpha=0.4, s=30, color='black', zorder=3)

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax.set_title('Country-Level Performance Variability\n(Random Effects Heterogeneity)',
                 fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0.5, 1.0)

    ax.set_xticks(positions)
    ax.set_xticklabels(models, rotation=0)
    plt.tight_layout()
    fig_file = OUTPUT_DIR / '02_country_variance.png'
    plt.savefig(fig_file, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {fig_file}")
else:
    plt.close()
    print("[SKIP] No country metrics data available")

# =============================================================================
# FIGURE 3: FEATURE IMPORTANCE COMPARISON (Grouped Bars)
# =============================================================================

print("\n" + "=" * 80)
print("FIGURE 3: Feature Importance Comparison")
print("=" * 80)

# For this figure, we'd ideally need coefficient information from the models
# Since we don't have coefficient data in the current outputs, we'll create
# a feature count comparison instead

fig, ax = plt.subplots(figsize=(12, 6))

# Prepare feature breakdown data by analyzing feature_cols
feature_breakdown = []
for model_name in ME_MODELS:
    if model_name in model_data and 'summary' in model_data[model_name]:
        summary = model_data[model_name]['summary']
        feature_cols = summary.get('feature_cols', [])

        # Count feature types based on naming patterns
        ratio_count = sum(1 for f in feature_cols if 'ratio' in f.lower() and 'hmm' not in f.lower() and 'dmd' not in f.lower())
        zscore_count = sum(1 for f in feature_cols if 'zscore' in f.lower() and 'hmm' not in f.lower() and 'dmd' not in f.lower())
        hmm_count = sum(1 for f in feature_cols if 'hmm' in f.lower())
        dmd_count = sum(1 for f in feature_cols if 'dmd' in f.lower())

        feature_breakdown.append({
            'model': model_name.replace('pooled_', '').replace('_', '\n'),
            'base_features': ratio_count + zscore_count,
            'hmm_features': hmm_count,
            'dmd_features': dmd_count
        })

print(f"Feature breakdown collected: {len(feature_breakdown)} models")

if feature_breakdown:
    feat_df = pd.DataFrame(feature_breakdown)

    # Set up bar positions
    x = np.arange(len(feat_df))
    width = 0.25

    # Create grouped bars
    bars1 = ax.bar(x - width, feat_df['base_features'], width, label='Base Features',
                   color='#FF9800', edgecolor='black', linewidth=0.5)
    bars2 = ax.bar(x, feat_df['hmm_features'], width, label='HMM',
                   color='#2196F3', edgecolor='black', linewidth=0.5)
    bars3 = ax.bar(x + width, feat_df['dmd_features'], width, label='DMD',
                   color='#4CAF50', edgecolor='black', linewidth=0.5)

    # Add value labels
    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            if height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{int(height)}', ha='center', va='bottom', fontsize=9)

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Number of Features', fontsize=12, fontweight='bold')
    ax.set_title('Feature Type Distribution Across Mixed Effects Models',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(feat_df['model'])
    ax.legend(fontsize=10)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()
    fig_file = OUTPUT_DIR / '03_feature_type_comparison.png'
    plt.savefig(fig_file, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {fig_file}")
else:
    plt.close()
    print("[SKIP] No feature breakdown data available")

# =============================================================================
# FIGURE 4: ME vs XGBoost Positioning (Scatter Plot)
# =============================================================================

print("\n" + "=" * 80)
print("FIGURE 4: Mixed Effects vs XGBoost Positioning")
print("=" * 80)

if full_ranking is not None:
    fig, ax = plt.subplots(figsize=(12, 8))

    # Prepare data - load n_features from summary files
    scatter_data = []
    for _, row in full_ranking.iterrows():
        model_path = row['model']
        category = 'Mixed Effects' if 'mixed_effects' in model_path else 'XGBoost'

        # Try to get n_features from loaded model_data first
        n_features = 0
        model_name = model_path.split('/')[-1] if '/' in model_path else model_path
        if model_name in model_data and 'summary' in model_data[model_name]:
            n_features = model_data[model_name]['summary'].get('n_features', 0)

        # Fallback to row data if available
        if n_features == 0:
            n_features = row.get('n_features', 0)

        scatter_data.append({
            'model': model_name,
            'category': category,
            'n_features': n_features,
            'auc': row.get('mean_fold_auc', 0),
            'feature_type': row.get('feature_type', '')
        })

    scatter_df = pd.DataFrame(scatter_data)

    print(f"   Scatter plot data prepared: {len(scatter_df)} models")
    try:
        print(f"   Feature counts: {scatter_df['n_features'].tolist()}")
    except:
        print(f"   Feature counts: (varying)")

    # Plot
    for category, color, marker in [('XGBoost', 'blue', 'o'), ('Mixed Effects', 'green', 's')]:
        subset = scatter_df[scatter_df['category'] == category]
        ax.scatter(subset['n_features'], subset['auc'], c=color, marker=marker,
                   s=150, alpha=0.7, edgecolors='black', linewidth=1, label=category)

    # Annotate models
    for _, row in scatter_df.iterrows():
        ax.annotate(row['model'], (row['n_features'], row['auc']),
                    fontsize=7, alpha=0.7, xytext=(3, 3), textcoords='offset points')

    ax.set_xlabel('Number of Features (Model Complexity)', fontsize=12, fontweight='bold')
    ax.set_ylabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax.set_title('Model Complexity vs Performance\n(Mixed Effects vs XGBoost)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=11, loc='lower right')
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig_file = OUTPUT_DIR / '04_me_vs_xgboost.png'
    plt.savefig(fig_file, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {fig_file}")
else:
    print("[SKIP] Full ranking data not available")

# =============================================================================
# FIGURE 5: PRECISION-RECALL CURVES (Line Plot)
# =============================================================================

print("\n" + "=" * 80)
print("FIGURE 5: Precision-Recall Curves")
print("=" * 80)

from sklearn.metrics import precision_recall_curve, auc as compute_auc

fig, ax = plt.subplots(figsize=(10, 8))

colors = ['#FF9800', '#9C27B0', '#2196F3', '#4CAF50']
model_curves = []

for idx, model_name in enumerate(ME_MODELS):
    if model_name in model_data and 'predictions' in model_data[model_name]:
        preds = model_data[model_name]['predictions']

        # Check for required columns (use 'future_crisis' not 'ipc_future_crisis')
        if 'future_crisis' in preds.columns and 'pred_prob' in preds.columns:
            y_true = preds['future_crisis'].values
            y_pred = preds['pred_prob'].values

            # Compute precision-recall curve
            precision, recall, _ = precision_recall_curve(y_true, y_pred)
            pr_auc = compute_auc(recall, precision)

            # Plot
            label = model_name.replace('pooled_', '').replace('_', ' ')
            ax.plot(recall, precision, color=colors[idx], linewidth=2.5,
                    label=f'{label} (AUC={pr_auc:.3f})', alpha=0.8)

            model_curves.append(model_name)
        else:
            print(f"  [SKIP] {model_name}: Missing required columns (available: {list(preds.columns[:5])}...)")

if model_curves:
    ax.set_xlabel('Recall', fontsize=12, fontweight='bold')
    ax.set_ylabel('Precision', fontsize=12, fontweight='bold')
    ax.set_title('Precision-Recall Curves - Mixed Effects Models',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10, loc='upper right')
    ax.grid(alpha=0.3)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)

    plt.tight_layout()
    fig_file = OUTPUT_DIR / '05_pr_curves.png'
    plt.savefig(fig_file, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {fig_file} ({len(model_curves)} models)")
else:
    plt.close()
    print("[SKIP] No precision-recall data available")

# =============================================================================
# FIGURE 6: VARIANCE COMPONENTS (Stacked Bars)
# =============================================================================

print("\n" + "=" * 80)
print("FIGURE 6: Variance Components Decomposition")
print("=" * 80)

# For this figure, we need ICC (Intraclass Correlation Coefficient) or variance components
# Since we don't have ICC in the current outputs, we'll create a comparison using
# country-level variance statistics

fig, ax = plt.subplots(figsize=(12, 7))

variance_data = []
for model_name in ME_MODELS:
    if model_name in model_data and 'country_metrics' in model_data[model_name]:
        cm = model_data[model_name]['country_metrics']

        # Try different column names for AUC
        auc_col = None
        for col_name in ['auc_roc', 'auc', 'mean_auc', 'country_auc']:
            if col_name in cm.columns:
                auc_col = col_name
                break

        if auc_col is not None:
            auc_values = cm[auc_col].dropna()

            if len(auc_values) > 1:
                # ONLY use actual between-country variance (no proxies or estimates)
                between_country_var = auc_values.var()
                mean_auc = auc_values.mean()
                std_country = auc_values.std()

                variance_data.append({
                    'model': model_name.replace('pooled_', '').replace('_', '\n'),
                    'between_country_var': between_country_var,
                    'between_country_std': std_country,
                    'mean_auc': mean_auc,
                    'n_countries': len(auc_values),
                    'min_auc': auc_values.min(),
                    'max_auc': auc_values.max()
                })

if variance_data:
    var_df = pd.DataFrame(variance_data)

    # Create simple bar chart showing ONLY between-country variance
    x = np.arange(len(var_df))
    width = 0.6

    bars = ax.bar(x, var_df['between_country_var'], width,
                  color='#FF6B6B', edgecolor='black', linewidth=0.5)

    # Add variance and std labels
    for i, row in var_df.iterrows():
        height = row['between_country_var']
        ax.text(i, height + 0.001, f"Var={height:.4f}\nSD={row['between_country_std']:.4f}",
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Between-Country Variance (AUC)', fontsize=12, fontweight='bold')
    ax.set_title('Country-Level Performance Heterogeneity\n(Variance of Country-Specific AUC Values)',
                 fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(var_df['model'])
    ax.grid(axis='y', alpha=0.3)

    # Add text annotation explaining what this shows
    textstr = f'Range across {var_df["n_countries"].iloc[0]} countries'
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()
    fig_file = OUTPUT_DIR / '06_variance_components.png'
    plt.savefig(fig_file, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
    plt.close()
    print(f"[OK] Saved: {fig_file}")
else:
    plt.close()
    print("[SKIP] No variance components data available")

# =============================================================================
# SUMMARY
# =============================================================================

print("\n" + "=" * 80)
print("MIXED EFFECTS ANALYSIS COMPLETE")
print("=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nOutput directory: {OUTPUT_DIR}")
print("\nFigures created:")
print("  1. 01_performance_ranking.png - Model performance comparison")
print("  2. 02_country_variance.png - Country-level variability")
print("  3. 03_feature_type_comparison.png - Feature distribution")
print("  4. 04_me_vs_xgboost.png - Complexity-performance positioning")
print("  5. 05_pr_curves.png - Precision-recall curves")
print("  6. 06_variance_components.png - Variance decomposition")
print("\n" + "=" * 80)
