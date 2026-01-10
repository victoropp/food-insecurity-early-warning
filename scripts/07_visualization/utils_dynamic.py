"""
Dynamic Utilities for Visualization Scripts
============================================
Helper functions to detect columns and load data dynamically without hardcoding.

Date: December 26, 2025
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import pandas as pd
import numpy as np
from pathlib import Path
import json
from typing import Dict, List, Optional, Tuple


def detect_auc_column(df: pd.DataFrame) -> Optional[str]:
    """Detect AUC column dynamically."""
    possible_names = ['auc_roc', 'mean_fold_auc', 'auc', 'roc_auc', 'overall_auc_roc']
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def detect_pr_auc_column(df: pd.DataFrame) -> Optional[str]:
    """Detect PR-AUC column dynamically."""
    possible_names = ['pr_auc', 'auc_pr', 'prauc', 'mean_pr_auc', 'overall_pr_auc']
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def detect_f1_column(df: pd.DataFrame) -> Optional[str]:
    """Detect F1 score column dynamically."""
    possible_names = ['f1_score', 'f1', 'mean_f1', 'f1_youden', 'mean_f1_youden']
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def detect_precision_column(df: pd.DataFrame) -> Optional[str]:
    """Detect precision column dynamically."""
    possible_names = ['precision', 'mean_precision', 'precision_youden', 'mean_precision_youden']
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def detect_recall_column(df: pd.DataFrame) -> Optional[str]:
    """Detect recall column dynamically."""
    possible_names = ['recall', 'mean_recall', 'recall_youden', 'mean_recall_youden']
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def detect_model_column(df: pd.DataFrame) -> Optional[str]:
    """Detect model name column dynamically."""
    possible_names = ['model', 'model_name', 'model_id']
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def detect_category_column(df: pd.DataFrame) -> Optional[str]:
    """Detect category column dynamically."""
    possible_names = ['category', 'model_category', 'type', 'model_type']
    for name in possible_names:
        if name in df.columns:
            return name
    return None


def get_metric_columns(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Get all available metric columns."""
    return {
        'auc_roc': detect_auc_column(df),
        'pr_auc': detect_pr_auc_column(df),
        'f1': detect_f1_column(df),
        'precision': detect_precision_column(df),
        'recall': detect_recall_column(df),
        'model': detect_model_column(df),
        'category': detect_category_column(df)
    }


def sort_by_best_metric(df: pd.DataFrame, ascending: bool = False) -> pd.DataFrame:
    """Sort DataFrame by best available metric (AUC > PR-AUC > F1)."""
    metrics = get_metric_columns(df)

    if metrics['auc_roc']:
        return df.sort_values(metrics['auc_roc'], ascending=ascending)
    elif metrics['pr_auc']:
        return df.sort_values(metrics['pr_auc'], ascending=ascending)
    elif metrics['f1']:
        return df.sort_values(metrics['f1'], ascending=ascending)
    else:
        print("WARNING: No sortable metric found")
        return df


def get_category_colors() -> Dict[str, str]:
    """Get standard category color mapping."""
    return {
        'Ensemble': '#E91E63',
        'Baseline': '#9C27B0',
        'XGBoost': '#2196F3',
        'Mixed Effects': '#4CAF50',
        'Ablation': '#FF9800',
        'Other': '#757575'
    }


def assign_colors_by_category(df: pd.DataFrame, category_col: str) -> List[str]:
    """Assign colors based on category column."""
    colors = get_category_colors()
    return [colors.get(cat, colors['Other']) for cat in df[category_col]]


def assign_colors_by_model_name(df: pd.DataFrame, model_col: str) -> List[str]:
    """Assign colors based on model name patterns (fallback)."""
    colors = []
    for model in df[model_col]:
        model_lower = str(model).lower()
        if 'ensemble' in model_lower:
            colors.append('#E91E63')
        elif 'baseline' in model_lower or 'ar' in model_lower:
            colors.append('#9C27B0')
        elif 'xgboost' in model_lower:
            colors.append('#2196F3')
        elif 'mixed' in model_lower or 'pooled' in model_lower:
            colors.append('#4CAF50')
        elif 'ablation' in model_lower:
            colors.append('#FF9800')
        else:
            colors.append('#757575')
    return colors


def load_model_ranking(results_dir: Path) -> pd.DataFrame:
    """Load model ranking table dynamically."""
    ranking_file = results_dir / 'analysis' / 'model_ranking_table.csv'
    if not ranking_file.exists():
        raise FileNotFoundError(f"Model ranking table not found: {ranking_file}")
    return pd.read_csv(ranking_file)


def load_country_metrics(results_dir: Path) -> pd.DataFrame:
    """Load country-level metrics dynamically."""
    metrics_file = results_dir / 'analysis' / 'country_level_metrics_all_models.csv'
    if not metrics_file.exists():
        print(f"WARNING: Country metrics not found: {metrics_file}")
        return pd.DataFrame()
    return pd.read_csv(metrics_file)


def get_model_path_mapping(results_dir: Path) -> Dict[str, Dict[str, Path]]:
    """
    Create mapping from human-readable model names to actual file paths.
    Returns dict with keys: 'model_dir', 'prediction_file', 'summary_file'
    """
    mapping = {}

    # Stage 1 AR Baseline
    stage1_dir = results_dir / 'stage1_ar_baseline'
    if (stage1_dir / 'predictions_h8_averaged.csv').exists():
        mapping['Stage 1 AR Baseline'] = {
            'model_dir': stage1_dir,
            'prediction_file': stage1_dir / 'predictions_h8_averaged.csv',
            'summary_file': stage1_dir / 'cv_results_h8.json' if (stage1_dir / 'cv_results_h8.json').exists() else None
        }

    # Stage 2 XGBoost Advanced
    xgb_adv_dir = results_dir / 'stage2_models' / 'xgboost' / 'advanced_with_ar'
    if (xgb_adv_dir / 'xgboost_hmm_dmd_with_ar_predictions.csv').exists():
        mapping['Stage 2 XGBoost Advanced'] = {
            'model_dir': xgb_adv_dir,
            'prediction_file': xgb_adv_dir / 'xgboost_hmm_dmd_with_ar_predictions.csv',
            'summary_file': xgb_adv_dir / 'xgboost_hmm_dmd_with_ar_summary.json'
        }

    # Stage 2 XGBoost Basic
    xgb_basic_dir = results_dir / 'stage2_models' / 'xgboost' / 'basic_with_ar'
    if (xgb_basic_dir / 'xgboost_basic_predictions.csv').exists():
        mapping['Stage 2 XGBoost Basic'] = {
            'model_dir': xgb_basic_dir,
            'prediction_file': xgb_basic_dir / 'xgboost_basic_predictions.csv',
            'summary_file': xgb_basic_dir / 'xgboost_basic_summary.json'
        }

    # Stage 2 Mixed-Effects Z-Score
    me_zscore_dir = results_dir / 'stage2_models' / 'mixed_effects' / 'pooled_zscore_with_ar'
    if (me_zscore_dir / 'pooled_zscore_with_ar_predictions.csv').exists():
        mapping['Stage 2 Mixed-Effects Z-Score'] = {
            'model_dir': me_zscore_dir,
            'prediction_file': me_zscore_dir / 'pooled_zscore_with_ar_predictions.csv',
            'summary_file': me_zscore_dir / 'pooled_zscore_with_ar_summary.json'
        }

    # Stage 2 Mixed-Effects Ratio
    me_ratio_dir = results_dir / 'stage2_models' / 'mixed_effects' / 'pooled_ratio_with_ar'
    if (me_ratio_dir / 'pooled_ratio_with_ar_predictions.csv').exists():
        mapping['Stage 2 Mixed-Effects Ratio'] = {
            'model_dir': me_ratio_dir,
            'prediction_file': me_ratio_dir / 'pooled_ratio_with_ar_predictions.csv',
            'summary_file': me_ratio_dir / 'pooled_ratio_with_ar_summary.json'
        }

    # Stage 2 Mixed-Effects Z-Score + HMM + DMD
    me_zscore_hmm_dir = results_dir / 'stage2_models' / 'mixed_effects' / 'pooled_zscore_hmm_dmd_with_ar'
    if (me_zscore_hmm_dir / 'pooled_zscore_hmm_dmd_with_ar_predictions.csv').exists():
        mapping['Stage 2 Mixed-Effects Z-Score + HMM + DMD'] = {
            'model_dir': me_zscore_hmm_dir,
            'prediction_file': me_zscore_hmm_dir / 'pooled_zscore_hmm_dmd_with_ar_predictions.csv',
            'summary_file': me_zscore_hmm_dir / 'pooled_zscore_hmm_dmd_with_ar_summary.json'
        }

    # Stage 2 Mixed-Effects Ratio + HMM + DMD
    me_ratio_hmm_dir = results_dir / 'stage2_models' / 'mixed_effects' / 'pooled_ratio_hmm_dmd_with_ar'
    if (me_ratio_hmm_dir / 'pooled_ratio_hmm_dmd_with_ar_predictions.csv').exists():
        mapping['Stage 2 Mixed-Effects Ratio + HMM + DMD'] = {
            'model_dir': me_ratio_hmm_dir,
            'prediction_file': me_ratio_hmm_dir / 'pooled_ratio_hmm_dmd_with_ar_predictions.csv',
            'summary_file': me_ratio_hmm_dir / 'pooled_ratio_hmm_dmd_with_ar_summary.json'
        }

    # Literature Baseline
    lit_baseline_dir = results_dir / 'baseline_comparison' / 'literature_baseline'
    if (lit_baseline_dir / 'literature_baseline_predictions.csv').exists():
        mapping['Literature Baseline'] = {
            'model_dir': lit_baseline_dir,
            'prediction_file': lit_baseline_dir / 'literature_baseline_predictions.csv',
            'summary_file': lit_baseline_dir / 'literature_baseline_summary.json'
        }

    # Two-Stage Ensemble
    ensemble_dir = results_dir / 'ensemble_stage1_stage2'
    if (ensemble_dir / 'ensemble_predictions.csv').exists():
        mapping['Two-Stage Ensemble (Stage 1 + 2)'] = {
            'model_dir': ensemble_dir,
            'prediction_file': ensemble_dir / 'ensemble_predictions.csv',
            'summary_file': ensemble_dir / 'ensemble_summary.json'
        }

    return mapping


def find_model_predictions(model_name: str, results_dir: Path) -> Optional[Path]:
    """Find prediction file for a model by human-readable name."""
    mapping = get_model_path_mapping(results_dir)
    if model_name in mapping:
        return mapping[model_name]['prediction_file']
    return None


def check_required_columns(df: pd.DataFrame, required: List[str], df_name: str = "DataFrame") -> bool:
    """Check if DataFrame has required columns and print helpful error if not."""
    missing = [col for col in required if col not in df.columns]
    if missing:
        print(f"ERROR: {df_name} missing required columns: {missing}")
        print(f"Available columns: {df.columns.tolist()}")
        return False
    return True
