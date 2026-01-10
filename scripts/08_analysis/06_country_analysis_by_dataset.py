#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Country-Level Analysis: Separated by Dataset Type
==================================================
Creates separate visualizations for models grouped by dataset coverage:

1. AR Baseline (18 countries, ~20k obs): Full dataset
2. Literature Baseline (14 countries, ~62k obs): Literature-filtered dataset
3. AR-Filtered Dataset Models (13 countries, ~6.5k obs): Stage 2 models
4. Common/Overlap Dataset (1,383 obs): Weighted Ensemble

This separation is critical because:
- Different models operate on different subsets of the data
- Comparing country performance across different datasets is misleading
- Each group requires its own context and interpretation

Author: Victor Collins Oppon
Date: December 2025
"""

import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend

print("[DEBUG] Matplotlib configured")

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# import seaborn as sns  # TEMPORARILY DISABLED TO DEBUG
from pathlib import Path
import sys
import warnings
warnings.filterwarnings('ignore')

print("[DEBUG] Basic imports complete")

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("[DEBUG] Path updated, importing config...")

from config import RESULTS_DIR, FIGURES_DIR

print("[DEBUG] Config imported successfully")

# Input files
COUNTRY_METRICS_FILE = RESULTS_DIR / 'analysis' / 'country_level_metrics_all_models.csv'
MODEL_RANKING_FILE = RESULTS_DIR / 'analysis' / 'model_ranking_table.csv'

# Output directory
OUTPUT_DIR = FIGURES_DIR / 'analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("COUNTRY-LEVEL ANALYSIS BY DATASET TYPE")
print("=" * 80)
print()

# =============================================================================
# LOAD DATA
# =============================================================================

print("Loading data...")
country_df = pd.read_csv(COUNTRY_METRICS_FILE)
models_df = pd.read_csv(MODEL_RANKING_FILE)

print(f"  Country metrics: {len(country_df)} rows")
print(f"  Model rankings: {len(models_df)} models")
print()

# =============================================================================
# CATEGORIZE MODELS BY DATASET
# =============================================================================

# Model name standardization mapping
model_name_mapping = {
    'mixed_effects/pooled_zscore_with_ar': 'Stage 2 Mixed-Effects Z-Score',
    'mixed_effects/pooled_ratio_with_ar': 'Stage 2 Mixed-Effects Ratio',
    'mixed_effects/pooled_zscore_hmm_dmd_with_ar': 'Stage 2 Mixed-Effects Z-Score + HMM + DMD',
    'mixed_effects/pooled_ratio_hmm_dmd_with_ar': 'Stage 2 Mixed-Effects Ratio + HMM + DMD',
    'xgboost_hmm_dmd_with_ar': 'Stage 2 XGBoost Advanced',
    'xgboost_with_ar': 'Stage 2 XGBoost Basic',
    'Stage 1 AR Baseline': 'Stage 1 AR Baseline',
    'Literature Baseline': 'Literature Baseline',
}

# Apply mapping to country data
country_df['model_name'] = country_df['model'].map(model_name_mapping).fillna(country_df['model'])

# Define dataset groups
AR_BASELINE_MODEL = ['Stage 1 AR Baseline']
LITERATURE_MODEL = ['Literature Baseline']
AR_FILTERED_MODELS = [
    'Stage 2 XGBoost Advanced',
    'Stage 2 XGBoost Basic',
    'Stage 2 Mixed-Effects Z-Score',
    'Stage 2 Mixed-Effects Ratio',
    'Stage 2 Mixed-Effects Z-Score + HMM + DMD',
    'Stage 2 Mixed-Effects Ratio + HMM + DMD'
]

# =============================================================================
# PROFESSIONAL COLOR SCHEMES
# =============================================================================

# Colors for models
MODEL_COLORS = {
    'Stage 1 AR Baseline': '#2C5F8D',          # Navy Blue
    'Literature Baseline': '#5B7C99',          # Steel Blue
    'Stage 2 XGBoost Advanced': '#2E7D5F',     # Forest Green (darker)
    'Stage 2 XGBoost Basic': '#50C878',        # Emerald Green (lighter)
    'Stage 2 Mixed-Effects Z-Score': '#3A6B5F',      # Teal
    'Stage 2 Mixed-Effects Ratio': '#5C8A7F',        # Sage
    'Stage 2 Mixed-Effects Z-Score + HMM + DMD': '#2B5F4F',  # Dark Teal
    'Stage 2 Mixed-Effects Ratio + HMM + DMD': '#4A7A6A',    # Medium Teal
}

# =============================================================================
# VISUALIZATION 1: AR BASELINE (18 COUNTRIES)
# =============================================================================

print("Creating visualization for AR Baseline...")

ar_data = country_df[country_df['model_name'].isin(AR_BASELINE_MODEL)].copy()

if len(ar_data) > 0:
    fig = plt.figure(figsize=(14, 6))

    # Single panel: Country performance
    pivot_ar_base = ar_data.pivot_table(
        index='ipc_country',
        columns='model_name',
        values='auc_roc',
        aggfunc='first'
    )

    # Sort by performance
    pivot_ar_base = pivot_ar_base.sort_values('Stage 1 AR Baseline', ascending=False)

    # Create bar chart
    ax = plt.gca()
    y_pos = np.arange(len(pivot_ar_base))
    bars = ax.barh(y_pos, pivot_ar_base['Stage 1 AR Baseline'].values,
                   color='#2C5F8D', alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pivot_ar_base.index, fontsize=10)
    ax.set_xlabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax.set_title('AR Baseline: Country-Level Performance\n(18 countries, 20,722 observations)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, pivot_ar_base['Stage 1 AR Baseline'].values)):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

    # Add overall AUC
    ar_overall = models_df[models_df['model'] == 'Stage 1 AR Baseline']['auc_roc'].values[0]
    ax.text(0.02, 0.98, f'Overall AUC: {ar_overall:.4f}',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'country_analysis_ar_baseline.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file.name}")
    print(f"  Countries: {len(pivot_ar_base)}")
else:
    print("  WARNING: AR Baseline not found in country metrics")

print()

# =============================================================================
# VISUALIZATION 2: LITERATURE BASELINE (14 COUNTRIES)
# =============================================================================

print("Creating visualization for Literature Baseline...")

lit_data = country_df[country_df['model_name'].isin(LITERATURE_MODEL)].copy()

if len(lit_data) > 0:
    fig = plt.figure(figsize=(14, 6))

    # Single panel: Country performance
    pivot_lit = lit_data.pivot_table(
        index='ipc_country',
        columns='model_name',
        values='auc_roc',
        aggfunc='first'
    )

    # Sort by performance and drop NaN values
    pivot_lit = pivot_lit.sort_values('Literature Baseline', ascending=False)
    pivot_lit = pivot_lit.dropna()  # Remove countries with NaN AUC

    n_valid_countries = len(pivot_lit)

    # Create bar chart
    ax = plt.gca()
    y_pos = np.arange(len(pivot_lit))
    bars = ax.barh(y_pos, pivot_lit['Literature Baseline'].values,
                   color='#5B7C99', alpha=0.85)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(pivot_lit.index, fontsize=10)
    ax.set_xlabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax.set_title(f'Literature Baseline: Country-Level Performance\n({n_valid_countries} countries with valid metrics, 62,564 observations)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, pivot_lit['Literature Baseline'].values)):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

    # Add overall AUC
    lit_overall = models_df[models_df['model'] == 'Literature Baseline']['auc_roc'].values[0]
    ax.text(0.02, 0.98, f'Overall AUC: {lit_overall:.4f}',
            transform=ax.transAxes, fontsize=11, fontweight='bold',
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightsteelblue', alpha=0.7))

    # Add note about missing countries
    note_text = 'Note: Missing 6 countries (Burkina Faso, Burundi, Cameroon, Chad + 2 with invalid metrics)'
    ax.text(0.98, 0.02, note_text, transform=ax.transAxes, fontsize=8.5,
            ha='right', va='bottom', style='italic', color='red')

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'country_analysis_literature.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file.name}")
    print(f"  Countries: {len(pivot_lit)}")
else:
    print("  WARNING: Literature Baseline not found in country metrics")

print()

# =============================================================================
# VISUALIZATION 3: AR-FILTERED MODELS (STAGE 2)
# =============================================================================

print("Creating visualization for AR-Filtered Dataset Models...")

ar_filtered_data = country_df[country_df['model_name'].isin(AR_FILTERED_MODELS)].copy()

if len(ar_filtered_data) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(18, 7))

    # Panel A: Overall AUC comparison
    ax1 = axes[0]
    ar_models = models_df[models_df['model'].isin(AR_FILTERED_MODELS)].copy()
    ar_models = ar_models.sort_values('auc_roc', ascending=True)

    y_pos = np.arange(len(ar_models))
    # Use single green color for all AR-filtered models
    single_color = '#2E7D5F'  # Forest Green

    bars = ax1.barh(y_pos, ar_models['auc_roc'], color=single_color, alpha=0.85)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([m.replace('Stage 2 ', '') for m in ar_models['model']], fontsize=10)
    ax1.set_xlabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax1.set_title('A. AR-Filtered Dataset Models\n(13 countries, ~6.5k observations, IPC≤2 & AR=0)',
                  fontsize=13, fontweight='bold', pad=15)
    ax1.set_xlim(0, 1.0)
    ax1.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, ar_models['auc_roc'])):
        ax1.text(val + 0.02, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9, fontweight='bold')

    # Panel B: Country × Model heatmap for AR-filtered
    ax2 = axes[1]

    # Pivot for heatmap
    pivot_ar = ar_filtered_data.pivot_table(
        index='ipc_country',
        columns='model_name',
        values='auc_roc',
        aggfunc='first'
    )

    # Sort by average performance
    pivot_ar['avg'] = pivot_ar.mean(axis=1)
    pivot_ar = pivot_ar.sort_values('avg', ascending=False).drop('avg', axis=1)

    # Reorder columns by model performance
    col_order = [m for m in ar_models.sort_values('auc_roc', ascending=False)['model'].tolist()
                 if m in pivot_ar.columns]
    pivot_ar = pivot_ar[col_order]

    # Create heatmap
    im = ax2.imshow(pivot_ar.values, cmap='Greens', aspect='auto', vmin=0.5, vmax=1.0)

    # Add colorbar
    cbar = plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)
    cbar.set_label('AUC-ROC', rotation=270, labelpad=20, fontsize=11, fontweight='bold')

    # Set ticks and labels
    ax2.set_xticks(np.arange(len(pivot_ar.columns)))
    ax2.set_yticks(np.arange(len(pivot_ar.index)))

    # Shorten model names for display
    short_names = {
        'Stage 2 XGBoost Advanced': 'XGB\nAdv',
        'Stage 2 XGBoost Basic': 'XGB\nBasic',
        'Stage 2 Mixed-Effects Z-Score': 'ME\nZ',
        'Stage 2 Mixed-Effects Ratio': 'ME\nR',
        'Stage 2 Mixed-Effects Z-Score + HMM + DMD': 'ME\nZ+',
        'Stage 2 Mixed-Effects Ratio + HMM + DMD': 'ME\nR+',
    }
    x_labels = [short_names.get(c, c) for c in pivot_ar.columns]
    ax2.set_xticklabels(x_labels, rotation=0, ha='center', fontsize=9)
    ax2.set_yticklabels(pivot_ar.index, fontsize=10)

    ax2.set_title('B. Performance by Country\n(AR-Filtered Models Only)',
                  fontsize=13, fontweight='bold', pad=15)

    # Add value annotations
    for i in range(len(pivot_ar.index)):
        for j in range(len(pivot_ar.columns)):
            val = pivot_ar.values[i, j]
            if not np.isnan(val):
                text_color = 'white' if val > 0.75 else 'black'
                ax2.text(j, i, f'{val:.2f}', ha='center', va='center',
                        color=text_color, fontsize=8, fontweight='bold')

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'country_analysis_ar_filtered.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file.name}")
    print(f"  Models: {len(ar_models)}")
    print(f"  Countries: {len(pivot_ar)}")
else:
    print("  WARNING: No AR-filtered models found in country metrics")

print()

# =============================================================================
# VISUALIZATION 4: ENSEMBLE (COMMON DATASET)
# =============================================================================

print("Creating visualization for Weighted Ensemble...")

# Get ensemble from model rankings
ensemble = models_df[models_df['model'] == 'Two-Stage Ensemble (Stage 1 + 2)']

if len(ensemble) > 0:
    fig, ax = plt.subplots(1, 1, figsize=(10, 6))

    # Show all 9 models with ensemble highlighted
    all_models = models_df.sort_values('auc_roc', ascending=True).copy()

    y_pos = np.arange(len(all_models))

    # Color ensemble in purple, others in their respective colors
    colors = []
    for model in all_models['model']:
        if model == 'Two-Stage Ensemble (Stage 1 + 2)':
            colors.append('#7B68BE')  # Purple
        elif model in AR_BASELINE_MODEL or model in LITERATURE_MODEL:
            colors.append('#4A90E2')  # Blue (baselines)
        else:
            colors.append('#50C878')  # Green (Stage 2)

    bars = ax.barh(y_pos, all_models['auc_roc'], color=colors, alpha=0.85)
    ax.set_yticks(y_pos)

    # Shorten labels
    labels = []
    for m in all_models['model']:
        if m == 'Two-Stage Ensemble (Stage 1 + 2)':
            labels.append('Ensemble')
        elif m == 'Stage 1 AR Baseline':
            labels.append('AR Baseline')
        elif m == 'Literature Baseline':
            labels.append('Literature')
        else:
            labels.append(m.replace('Stage 2 ', ''))

    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('AUC-ROC', fontsize=12, fontweight='bold')
    ax.set_title('Weighted Ensemble vs All Models\n(Ensemble operates on 6,553 AR-filtered observations)',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlim(0, 1.0)
    ax.grid(axis='x', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, all_models['auc_roc'])):
        ax.text(val + 0.02, bar.get_y() + bar.get_height()/2,
               f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

    # Add legend (smaller)
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='#7B68BE', alpha=0.85, label='Ensemble'),
        Patch(facecolor='#4A90E2', alpha=0.85, label='Baselines'),
        Patch(facecolor='#50C878', alpha=0.85, label='Stage 2')
    ]
    ax.legend(handles=legend_elements, loc='lower right', fontsize=8, framealpha=0.9)

    # Add annotation on the ensemble bar
    ensemble_idx = all_models[all_models['model'] == 'Two-Stage Ensemble (Stage 1 + 2)'].index[0]
    ensemble_y_pos = list(all_models.index).index(ensemble_idx)
    ensemble_auc = ensemble.iloc[0]["auc_roc"]

    textstr = f'0.55 × Stage 1\n+ 0.45 × Stage 2\nn=6,553'
    # Position at middle of ensemble bar
    ax.text(ensemble_auc / 2, ensemble_y_pos, textstr,
            ha='center', va='center', fontsize=7.5,
            color='white', fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='#5B4E8B', alpha=0.8, edgecolor='white', linewidth=1.5))

    plt.tight_layout()
    output_file = OUTPUT_DIR / 'ensemble_context.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()

    print(f"  Saved: {output_file.name}")
    print(f"  Ensemble AUC: {ensemble.iloc[0]['auc_roc']:.4f}")
    print(f"  Dataset: {int(ensemble.iloc[0]['n_obs'])} observations (common)")
else:
    print("  WARNING: Ensemble not found in model rankings")

print()
print("=" * 80)
print("COUNTRY ANALYSIS BY DATASET TYPE COMPLETE")
print("=" * 80)
print()
print("Four separate visualizations created:")
print("  1. country_analysis_ar_baseline.png - AR Baseline (18 countries)")
print("  2. country_analysis_literature.png - Literature Baseline (14 countries)")
print("  3. country_analysis_ar_filtered.png - Stage 2 models (13 countries)")
print("  4. ensemble_context.png - Weighted ensemble in context (1,383 common obs)")
print()
