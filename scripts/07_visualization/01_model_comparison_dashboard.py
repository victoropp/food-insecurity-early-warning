#!/usr/bin/env python3
"""
Phase 5 Visualization: Model Comparison Dashboard
=================================================
Creates publication-ready charts comparing model performance.

Uses Phase 4 outputs:
- model_ranking_table.csv
- country_level_metrics_all_models.csv
"""

import sys
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # MUST be before importing pyplot (non-interactive backend)
import matplotlib.pyplot as plt
# NOTE: seaborn removed - causes hanging on Windows
from pathlib import Path
from datetime import datetime
import json

# Add parent directory for config imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PHASE4_RESULTS, FIGURES_DIR, RESULTS_DIR

# Use config paths (standardized output directory)
PHASE4_OUTPUT = PHASE4_RESULTS
PHASE5_OUTPUT = FIGURES_DIR / "dashboards"
PHASE5_OUTPUT.mkdir(parents=True, exist_ok=True)

# Plot settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.size'] = 10
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['axes.labelsize'] = 10

def load_data():
    """Load Phase 4 outputs."""
    ranking = pd.read_csv(PHASE4_OUTPUT / "model_ranking_table.csv")
    country_metrics = pd.read_csv(PHASE4_OUTPUT / "country_level_metrics_all_models.csv")
    return ranking, country_metrics

def create_model_ranking_chart(ranking_df):
    """Horizontal bar chart of model AUC rankings."""
    fig, ax = plt.subplots(figsize=(12, 8))

    # Detect AUC column dynamically
    auc_col = None
    for col in ['auc_roc', 'mean_fold_auc', 'auc', 'roc_auc']:
        if col in ranking_df.columns:
            auc_col = col
            break

    if auc_col is None:
        print(f"ERROR: No AUC column found in ranking table")
        print(f"Available columns: {ranking_df.columns.tolist()}")
        return

    print(f"Using AUC column: {auc_col}")

    # Sort by AUC
    df = ranking_df.sort_values(auc_col, ascending=True)

    # Color by category if available, otherwise by model name patterns
    colors = []
    if 'category' in df.columns:
        category_colors = {
            'Ensemble': '#E91E63',
            'Baseline': '#9C27B0',
            'XGBoost': '#2196F3',
            'Mixed Effects': '#4CAF50',
            'Ablation': '#FF9800'
        }
        for cat in df['category']:
            colors.append(category_colors.get(cat, '#757575'))
    else:
        # Fallback: color by model name patterns
        for model in df['model']:
            model_lower = str(model).lower()
            if 'ensemble' in model_lower:
                colors.append('#E91E63')
            elif 'baseline' in model_lower or 'ar' in model_lower:
                colors.append('#9C27B0')
            elif 'xgboost' in model_lower:
                colors.append('#2196F3')
            elif 'mixed' in model_lower or 'pooled' in model_lower:
                colors.append('#4CAF50')
            else:
                colors.append('#757575')

    bars = ax.barh(range(len(df)), df[auc_col], color=colors, edgecolor='black', linewidth=0.5)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, df[auc_col])):
        ax.text(val + 0.005, bar.get_y() + bar.get_height()/2, f'{val:.3f}',
                va='center', fontsize=9)

    ax.set_yticks(range(len(df)))
    ax.set_yticklabels(df['model'].str.replace('pooled_', '').str.replace('_', ' '))
    ax.set_xlabel('AUC-ROC')
    ax.set_title('Model Performance Ranking by AUC', fontsize=14, fontweight='bold')
    ax.set_xlim(0, max(df[auc_col].max() * 1.1, 0.95))  # Ensure space for Stage 1 line

    # Load Stage 1 baseline for reference
    comparison_file = RESULTS_DIR / 'baseline_comparison' / 'comparison_summary.json'
    if comparison_file.exists():
        with open(comparison_file, 'r') as f:
            comparison = json.load(f)
        stage1_auc = comparison['comparison_table'][0]['auc_mean']  # 0.9075

        # Add Stage 1 reference line
        ax.axvline(x=stage1_auc, color='red', linestyle='--', linewidth=2,
                   label=f'Stage 1 AR Baseline (AUC={stage1_auc:.4f})', zorder=0, alpha=0.7)

    # Legend (combine patches and lines)
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    legend_elements = []
    if 'category' in df.columns:
        # Use categories for legend
        category_colors = {
            'Ensemble': '#E91E63',
            'Baseline': '#9C27B0',
            'XGBoost': '#2196F3',
            'Mixed Effects': '#4CAF50',
            'Ablation': '#FF9800'
        }
        for cat, color in category_colors.items():
            if cat in df['category'].values:
                legend_elements.append(Patch(facecolor=color, label=cat))
    else:
        # Fallback legend
        legend_elements = [
            Patch(facecolor='#E91E63', label='Ensemble'),
            Patch(facecolor='#9C27B0', label='Baseline'),
            Patch(facecolor='#2196F3', label='XGBoost'),
            Patch(facecolor='#4CAF50', label='Mixed Effects'),
            Patch(facecolor='#FF9800', label='Ablation')
        ]

    # Add Stage 1 baseline reference line to legend if it exists
    if comparison_file.exists():
        legend_elements.append(Line2D([0], [0], color='red', linestyle='--', linewidth=2,
                                      label=f'Stage 1 AR Baseline'))

    ax.legend(handles=legend_elements, loc='lower right')

    plt.tight_layout()
    plt.savefig(PHASE5_OUTPUT / "model_ranking_auc.png", bbox_inches='tight')
    plt.close()
    print("Saved: model_ranking_auc.png")

def create_threshold_comparison_chart(ranking_df):
    """Compare metrics across threshold types."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 6))

    # Prepare data
    df = ranking_df.sort_values('mean_fold_auc', ascending=False).head(6)
    models = df['model'].str.replace('pooled_', '').str.replace('_', '\n')

    # Helper function to add value labels to grouped bars
    def add_bar_labels(ax, bars, fontsize=7):
        for bar in bars:
            height = bar.get_height()
            if pd.notna(height) and height > 0:
                ax.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.2f}', ha='center', va='bottom', fontsize=fontsize)

    # Precision comparison
    ax = axes[0]
    x = np.arange(len(df))
    width = 0.25
    bars1 = ax.bar(x - width, df['mean_precision_youden'], width, label='Youden', color='#2196F3')
    bars2 = ax.bar(x, df['mean_precision_f1'], width, label='F1-max', color='#4CAF50')
    bars3 = ax.bar(x + width, df['mean_precision_high_recall'], width, label='High-Recall', color='#FF9800')
    add_bar_labels(ax, bars1)
    add_bar_labels(ax, bars2)
    add_bar_labels(ax, bars3)
    ax.set_ylabel('Precision')
    ax.set_title('Precision by Threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.legend(loc='upper right', fontsize=7)

    # Recall comparison
    ax = axes[1]
    bars1 = ax.bar(x - width, df['mean_recall_youden'], width, label='Youden', color='#2196F3')
    bars2 = ax.bar(x, df['mean_recall_f1'] if 'mean_recall_f1' in df.columns else df['mean_recall_youden'], width, label='F1-max', color='#4CAF50')
    bars3 = ax.bar(x + width, df['mean_recall_high_recall'], width, label='High-Recall', color='#FF9800')
    add_bar_labels(ax, bars1)
    add_bar_labels(ax, bars2)
    add_bar_labels(ax, bars3)
    ax.set_ylabel('Recall')
    ax.set_title('Recall by Threshold')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.legend(loc='upper right', fontsize=7)

    # Threshold values
    ax = axes[2]
    bars1 = ax.bar(x - width, df['mean_threshold_youden'], width, label='Youden', color='#2196F3')
    bars2 = ax.bar(x, df['mean_threshold_f1'], width, label='F1-max', color='#4CAF50')
    bars3 = ax.bar(x + width, df['mean_threshold_high_recall'], width, label='High-Recall', color='#FF9800')
    add_bar_labels(ax, bars1)
    add_bar_labels(ax, bars2)
    add_bar_labels(ax, bars3)
    ax.set_ylabel('Threshold Value')
    ax.set_title('Optimal Threshold Values')
    ax.set_xticks(x)
    ax.set_xticklabels(models, fontsize=8)
    ax.axhline(y=0.5, color='red', linestyle='--', label='Fixed 0.5')
    ax.legend(loc='upper right', fontsize=7)

    plt.suptitle('Threshold Strategy Comparison (Top 6 Models)', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(PHASE5_OUTPUT / "threshold_comparison.png", bbox_inches='tight')
    plt.close()
    print("Saved: threshold_comparison.png")

def create_filter_comparison_chart(ranking_df):
    """Compare filter variants."""
    fig, ax = plt.subplots(figsize=(10, 6))

    # Group by filter
    filter_stats = ranking_df.groupby('filter_variant').agg({
        'mean_fold_auc': ['mean', 'std'],
        'mean_f1_youden': ['mean', 'std']
    }).reset_index()
    filter_stats.columns = ['filter', 'auc_mean', 'auc_std', 'f1_mean', 'f1_std']

    x = np.arange(len(filter_stats))
    width = 0.35

    bars1 = ax.bar(x - width/2, filter_stats['auc_mean'], width, yerr=filter_stats['auc_std'],
                   label='AUC', color='#2196F3', capsize=5)
    bars2 = ax.bar(x + width/2, filter_stats['f1_mean'], width, yerr=filter_stats['f1_std'],
                   label='F1 (Youden)', color='#4CAF50', capsize=5)

    ax.set_ylabel('Score')
    ax.set_title('Performance by Filter Variant', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(filter_stats['filter'])
    ax.legend()
    ax.set_ylim(0, 1)

    # Add value labels
    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                f'{bar.get_height():.3f}', ha='center', fontsize=9)

    plt.tight_layout()
    plt.savefig(PHASE5_OUTPUT / "filter_comparison.png", bbox_inches='tight')
    plt.close()
    print("Saved: filter_comparison.png")

def main():
    print("=" * 80)
    print("PHASE 5: MODEL COMPARISON DASHBOARD")
    print("=" * 80)

    ranking, country_metrics = load_data()
    print(f"Loaded {len(ranking)} models, {len(country_metrics)} country-model rows")
    print(f"Available columns: {ranking.columns.tolist()}\n")

    # Model ranking chart (dynamic - should work)
    try:
        create_model_ranking_chart(ranking)
    except Exception as e:
        print(f"[SKIP] Model ranking chart failed: {e}")

    # Threshold comparison (requires specific columns - may not exist)
    try:
        create_threshold_comparison_chart(ranking)
    except Exception as e:
        print(f"[SKIP] Threshold comparison chart skipped (columns not found): {e}")

    # Filter comparison (requires specific columns - may not exist)
    try:
        create_filter_comparison_chart(ranking)
    except Exception as e:
        print(f"[SKIP] Filter comparison chart skipped (columns not found): {e}")

    print("\nPHASE 5 MODEL COMPARISON COMPLETE")

if __name__ == "__main__":
    main()
