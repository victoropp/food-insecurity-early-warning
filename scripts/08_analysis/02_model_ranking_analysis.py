#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Phase 4 Analysis: Model Ranking and Comparison
===============================================
Ranks models by performance and creates comparison tables.

Input: ar_vs_literature_comparison.csv from STAGE_3 (contains ALL 9 models)
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import io

# Set UTF-8 encoding for console output
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import RESULTS_DIR, PHASE4_RESULTS

PHASE4_RESULTS.mkdir(parents=True, exist_ok=True)

# Load from comprehensive comparison file (NO hardcoding)

def load_all_summaries():
    """Load all model JSON summaries from auto-detected models."""
    summaries = {}
    # Auto-detect models instead of using hardcoded list
    for model in auto_detect_models():
        summary = load_model_summary(model)
        if summary:
            summaries[model] = summary
    return summaries

def load_all_cv_results():
    """Load all CV results from auto-detected models."""
    all_cv = []
    # Auto-detect models instead of using hardcoded list
    for model in auto_detect_models():
        cv_df = load_model_cv_results(model)
        if cv_df is not None:
            cv_df['model'] = model
            all_cv.append(cv_df)
    return pd.concat(all_cv, ignore_index=True) if all_cv else None

def create_model_ranking_table(summaries):
    """
    Create comprehensive model ranking table.
    Uses standardized metric extraction from Phase 3 summaries.
    """
    rows = []

    for model, summary in summaries.items():
        # Use standardized metric extraction
        metrics = get_overall_metrics_from_summary(summary)
        metadata = get_model_metadata(model)

        row = {
            'model': model,
            'model_name': model,  # Alias for compatibility
            'feature_type': metadata.get('feature_type', summary.get('feature_type', '')),
            'filter_variant': metadata.get('filter_variant', summary.get('filter_variant', '')),
            'has_hmm_dmd': metadata.get('has_hmm_dmd', False),

            # Overall performance - use auc_youden to match XGBoost CSV outputs
            'auc_youden': metrics.get('auc_roc'),  # JSON uses auc_roc, rename for consistency
            'mean_fold_auc': metrics.get('mean_fold_auc'),
            'std_fold_auc': metrics.get('std_fold_auc'),

            # Youden threshold metrics
            'mean_precision_youden': metrics.get('mean_precision_youden'),
            'mean_recall_youden': metrics.get('mean_recall_youden'),
            'mean_f1_youden': metrics.get('mean_f1_youden'),
            'mean_precision': metrics.get('mean_precision_youden'),  # Alias
            'mean_recall': metrics.get('mean_recall_youden'),  # Alias
            'mean_f1': metrics.get('mean_f1_youden'),  # Alias

            # F1 threshold metrics
            'mean_precision_f1': metrics.get('mean_precision_f1'),
            'mean_recall_f1': metrics.get('mean_recall_f1'),

            # High recall threshold metrics
            'mean_precision_high_recall': metrics.get('mean_precision_high_recall'),
            'mean_recall_high_recall': metrics.get('mean_recall_high_recall'),

            # Calibration
            'overall_brier_score': metrics.get('overall_brier_score'),
            'mean_log_loss': metrics.get('mean_log_loss'),

            # Threshold values
            'mean_threshold_youden': metrics.get('mean_threshold_youden'),
            'mean_threshold_f1': metrics.get('mean_threshold_f1'),
            'mean_threshold_high_recall': metrics.get('mean_threshold_high_recall'),

            # Data characteristics - from summary
            'n_observations': summary.get('n_observations'),
            'n_countries': summary.get('n_countries'),
            'n_districts': summary.get('n_districts'),
            'prevalence': metrics.get('prevalence'),
        }
        rows.append(row)

    df = pd.DataFrame(rows)

    if df.empty:
        return df

    # Add rankings (handle NaN values)
    if 'auc_youden' in df.columns and df['auc_youden'].notna().any():
        df['rank_auc'] = df['auc_youden'].rank(ascending=False, method='min').astype('Int64')
    elif 'auc_roc' in df.columns and df['auc_roc'].notna().any():
        df['rank_auc'] = df['auc_roc'].rank(ascending=False, method='min').astype('Int64')
    if 'mean_f1_youden' in df.columns and df['mean_f1_youden'].notna().any():
        df['rank_f1'] = df['mean_f1_youden'].rank(ascending=False, method='min').astype('Int64')
    if 'mean_recall_high_recall' in df.columns and df['mean_recall_high_recall'].notna().any():
        df['rank_recall_hr'] = df['mean_recall_high_recall'].rank(ascending=False, method='min').astype('Int64')

    return df.sort_values('rank_auc') if 'rank_auc' in df.columns else df

def create_filter_comparison(ranking_df):
    """Compare performance across filter variants."""
    agg_dict = {'model': 'count'}
    if 'auc_youden' in ranking_df.columns:
        agg_dict['auc_youden'] = ['mean', 'std']
    if 'mean_f1_youden' in ranking_df.columns:
        agg_dict['mean_f1_youden'] = ['mean', 'std']
    if 'mean_recall_high_recall' in ranking_df.columns:
        agg_dict['mean_recall_high_recall'] = ['mean', 'std']
    return ranking_df.groupby('filter_variant').agg(agg_dict).round(4)

def create_feature_comparison(ranking_df):
    """Compare performance across feature types."""
    agg_dict = {'model': 'count'}
    if 'auc_youden' in ranking_df.columns:
        agg_dict['auc_youden'] = ['mean', 'std']
    if 'mean_f1_youden' in ranking_df.columns:
        agg_dict['mean_f1_youden'] = ['mean', 'std']
    if 'mean_recall_high_recall' in ranking_df.columns:
        agg_dict['mean_recall_high_recall'] = ['mean', 'std']
    return ranking_df.groupby('feature_type').agg(agg_dict).round(4)

def main():
    print("=" * 80)
    print("PHASE 4: MODEL RANKING ANALYSIS")
    print("=" * 80)

    # Load comprehensive comparison file (ALL 9 models)
    comparison_file = RESULTS_DIR / 'baseline_comparison' / 'ar_vs_literature_comparison.csv'

    if not comparison_file.exists():
        print(f"ERROR: Comparison file not found: {comparison_file}")
        print("Run STAGE_3_COMPARISON_ANALYSIS/01_compare_baselines.py first")
        return

    ranking_df = pd.read_csv(comparison_file)
    print(f"Loaded {len(ranking_df)} models from comparison file")

    # Rename columns to match expected format
    ranking_df = ranking_df.rename(columns={
        'auc_mean': 'auc_roc',
        'auc_std': 'auc_std',
        'prauc_mean': 'pr_auc',
        'f1_mean': 'f1_score'
    })

    # Add rankings
    ranking_df['rank_auc'] = ranking_df['auc_roc'].rank(ascending=False, method='min').astype('Int64')
    ranking_df['rank_prauc'] = ranking_df['pr_auc'].rank(ascending=False, method='min').astype('Int64')
    ranking_df['rank_f1'] = ranking_df['f1_score'].rank(ascending=False, method='min').astype('Int64')

    # Sort by AUC
    ranking_df = ranking_df.sort_values('auc_roc', ascending=False)

    # Add category labels for reference (but don't group)
    ranking_df['category'] = ranking_df['model'].apply(lambda x:
        'Ensemble' if 'Ensemble' in x else
        'Baseline' if 'Baseline' in x else
        'XGBoost' if 'XGBoost' in x else
        'Mixed-Effects'
    )

    # Save individual model rankings (NO grouping)
    ranking_df.to_csv(PHASE4_RESULTS / "model_ranking_table.csv", index=False)
    print(f"Saved: model_ranking_table.csv (9 individual models)")

    # Best models summary (using actual values, NO hardcoding)
    best_models = {
        'best_overall_auc': {
            'model': ranking_df.loc[ranking_df['auc_roc'].idxmax(), 'model'],
            'auc': float(ranking_df['auc_roc'].max())
        },
        'best_prauc': {
            'model': ranking_df.loc[ranking_df['pr_auc'].idxmax(), 'model'],
            'pr_auc': float(ranking_df['pr_auc'].max())
        },
        'best_f1': {
            'model': ranking_df.loc[ranking_df['f1_score'].idxmax(), 'model'],
            'f1': float(ranking_df['f1_score'].max())
        },
        'timestamp': datetime.now().isoformat(),
        'n_models': len(ranking_df)
    }

    with open(PHASE4_RESULTS / "best_models.json", 'w') as f:
        json.dump(best_models, f, indent=2)
    print(f"Saved: best_models.json")

    print("\n--- ALL 9 MODELS RANKED BY AUC (INDIVIDUAL, NO GROUPING) ---")
    display_cols = ['rank_auc', 'model', 'category', 'auc_roc', 'pr_auc', 'f1_score', 'n_features']
    print(ranking_df[display_cols].to_string(index=False))

    print("\n--- PERFORMANCE METRICS SUMMARY ---")
    print(f"Best AUC: {ranking_df['auc_roc'].max():.4f} ({ranking_df.loc[ranking_df['auc_roc'].idxmax(), 'model']})")
    print(f"Best PR-AUC: {ranking_df['pr_auc'].max():.4f} ({ranking_df.loc[ranking_df['pr_auc'].idxmax(), 'model']})")
    print(f"Best F1: {ranking_df['f1_score'].max():.4f} ({ranking_df.loc[ranking_df['f1_score'].idxmax(), 'model']})")
    print(f"\nAUC Range: {ranking_df['auc_roc'].min():.4f} - {ranking_df['auc_roc'].max():.4f}")
    print(f"PR-AUC Range: {ranking_df['pr_auc'].min():.4f} - {ranking_df['pr_auc'].max():.4f}")
    print(f"F1 Range: {ranking_df['f1_score'].min():.4f} - {ranking_df['f1_score'].max():.4f}")

    print("\nPHASE 4 MODEL RANKING COMPLETE")
    print(f"All {len(ranking_df)} individual models from complete pipeline included (NO grouping)")

if __name__ == "__main__":
    main()
