"""
Phase 4 Shared Utilities
========================
Data-driven utility functions for Phase 4 analysis scripts.
NO hardcoded values - everything auto-detected from actual Phase 3 results.

Provides:
- Auto-detection of available models from Phase 3 directory
- Auto-selection of best model by AUC
- Auto-detection of threshold types from prediction columns
- Loading functions that read actual metrics (not recompute)
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PHASE3_RESULTS, PHASE4_RESULTS


def auto_detect_models() -> List[str]:
    """
    Auto-detect available models from Phase 3 results directory.
    NO hardcoded model names - recursively scans directory for actual outputs.

    Handles nested directory structures like:
    - stage2_models/xgboost/basic_with_ar/xgboost_with_ar_summary.json
    - stage2_models/mixed_effects/pooled_ratio_with_ar/pooled_ratio_with_ar_summary.json
    - stage2_models/ablation/ratio_location/ablation_ratio_location_summary.json

    Returns:
        List of relative paths to model directories that have summary.json files
    """
    models = []

    if not PHASE3_RESULTS.exists():
        print(f"   WARNING: Phase 3 results directory not found: {PHASE3_RESULTS}")
        return models

    # Recursively find all *_summary.json files
    for summary_file in PHASE3_RESULTS.rglob('*_summary.json'):
        # Get the directory containing the summary file
        model_dir = summary_file.parent

        # Get the relative path from PHASE3_RESULTS
        try:
            relative_path = model_dir.relative_to(PHASE3_RESULTS)
            model_path_str = str(relative_path).replace('\\', '/')
            models.append(model_path_str)
        except ValueError:
            # Skip if not relative to PHASE3_RESULTS
            continue

    return sorted(models)


def auto_detect_threshold_types(sample_predictions_path: Optional[Path] = None) -> List[str]:
    """
    Auto-detect available threshold types from prediction columns.
    Looks for columns matching pattern 'y_pred_{threshold_type}'.

    Returns:
        List of threshold types (e.g., ['youden', 'f1', 'high_recall'])
    """
    threshold_types = []

    # Find a sample predictions file
    if sample_predictions_path is None:
        models = auto_detect_models()
        if not models:
            return ['youden']  # fallback

        for model in models:
            model_dir = PHASE3_RESULTS / model
            pred_files = list(model_dir.glob('*_predictions.csv'))
            if pred_files:
                sample_predictions_path = pred_files[0]
                break

    if sample_predictions_path is None or not sample_predictions_path.exists():
        return ['youden']  # fallback

    # Read just column names
    df = pd.read_csv(sample_predictions_path, nrows=0)

    # Find y_pred_* columns
    for col in df.columns:
        if col.startswith('y_pred_'):
            threshold_type = col.replace('y_pred_', '')
            threshold_types.append(threshold_type)

    return sorted(threshold_types) if threshold_types else ['youden']


def auto_select_best_model(metric: str = 'auc_roc') -> Tuple[str, float]:
    """
    Auto-select best model based on specified metric from Phase 3 summaries.

    Scientific ranking approach (standard practice):
    1. Primary: AUC-ROC (discrimination ability)
    2. Secondary: F1 score (precision-recall balance) - tiebreaker when AUC tied
    3. Tertiary: Brier score (calibration quality, lower=better) - tiebreaker when AUC & F1 tied
    4. Quaternary: Model simplicity (prefer models without HMM/DMD for parsimony)

    NO hardcoded model selection.

    Args:
        metric: Primary metric to use for selection (default: 'auc_roc')

    Returns:
        Tuple of (best_model_name, metric_value)
    """
    models = auto_detect_models()

    if not models:
        return ('', 0.0)

    # Collect all models with their metrics for multi-level tiebreaking
    model_scores = []

    for model in models:
        summary = load_model_summary(model)
        if summary:
            overall_metrics = summary.get('overall_metrics', {})

            # Primary: AUC
            auc = overall_metrics.get(metric,
                    overall_metrics.get('mean_fold_auc',
                    overall_metrics.get('auc', 0.0)))

            # Secondary: F1 (higher is better)
            f1 = overall_metrics.get('mean_f1_youden',
                    overall_metrics.get('f1', 0.0)) or 0.0

            # Tertiary: Brier score (lower is better, so negate for sorting)
            brier = overall_metrics.get('overall_brier_score', 1.0) or 1.0
            brier_neg = -brier  # Negate so higher (less negative) = better calibration

            # Quaternary: Model simplicity (prefer simpler models)
            # Models without hmm_dmd are simpler
            is_simple = 0 if 'hmm_dmd' in model else 1

            if auc:
                model_scores.append((model, auc, f1, brier_neg, is_simple))

    if not model_scores:
        return ('', 0.0)

    # Sort by: AUC (desc), F1 (desc), Brier (desc=less negative=lower), Simplicity (desc)
    model_scores.sort(key=lambda x: (x[1], x[2], x[3], x[4]), reverse=True)
    best = model_scores[0]

    return (best[0], best[1])


def load_model_summary(model_name: str) -> Optional[Dict]:
    """
    Load JSON summary for a model from Phase 3 results.

    Args:
        model_name: Relative path to model directory (e.g., 'xgboost/basic_with_ar')

    Returns:
        Dictionary with summary data, or None if not found
    """
    model_dir = PHASE3_RESULTS / model_name

    # Find any *_summary.json file in this directory
    summary_files = list(model_dir.glob('*_summary.json'))

    if summary_files:
        with open(summary_files[0], 'r') as f:
            return json.load(f)

    return None


def load_model_predictions(model_name: str) -> Optional[pd.DataFrame]:
    """
    Load predictions CSV for a model from Phase 3 results.

    Args:
        model_name: Relative path to model directory (e.g., 'xgboost/basic_with_ar')

    Returns:
        DataFrame with predictions, or None if not found
    """
    model_dir = PHASE3_RESULTS / model_name

    # Find any *_predictions.csv file in this directory
    pred_files = list(model_dir.glob('*_predictions.csv'))

    if pred_files:
        return pd.read_csv(pred_files[0])

    return None


def load_model_cv_results(model_name: str) -> Optional[pd.DataFrame]:
    """
    Load CV results CSV for a model from Phase 3 results.

    Args:
        model_name: Relative path to model directory (e.g., 'xgboost/basic_with_ar')

    Returns:
        DataFrame with CV results, or None if not found
    """
    model_dir = PHASE3_RESULTS / model_name

    # Find any *_cv_results.csv file in this directory
    cv_files = list(model_dir.glob('*_cv_results.csv'))

    if cv_files:
        return pd.read_csv(cv_files[0])

    return None


def load_model_country_metrics(model_name: str) -> Optional[pd.DataFrame]:
    """
    Load country metrics CSV for a model from Phase 3 results.
    THIS IS THE CORRECT SOURCE - metrics already computed in Phase 3.

    Args:
        model_name: Relative path to model directory (e.g., 'xgboost/basic_with_ar')

    Returns:
        DataFrame with country-level metrics, or None if not found
    """
    model_dir = PHASE3_RESULTS / model_name

    # Find any *_country_metrics.csv file in this directory
    country_files = list(model_dir.glob('*_country_metrics.csv'))

    if country_files:
        return pd.read_csv(country_files[0])

    return None


def load_model_threshold_analysis(model_name: str) -> Optional[pd.DataFrame]:
    """
    Load threshold analysis CSV for a model from Phase 3 results.

    Args:
        model_name: Relative path to model directory (e.g., 'xgboost/basic_with_ar')

    Returns:
        DataFrame with threshold analysis, or None if not found
    """
    model_dir = PHASE3_RESULTS / model_name

    # Find any *_threshold_analysis.csv file in this directory
    thresh_files = list(model_dir.glob('*_threshold_analysis.csv'))

    if thresh_files:
        return pd.read_csv(thresh_files[0])

    return None


def get_model_metadata(model_name: str) -> Dict:
    """
    Extract metadata from model name and summary.
    Auto-detects feature_type, filter_variant from actual data.

    Args:
        model_name: Name of the model

    Returns:
        Dictionary with model metadata
    """
    summary = load_model_summary(model_name)

    metadata = {
        'model_name': model_name,
        'feature_type': '',
        'filter_variant': '',
        'has_hmm_dmd': False,
    }

    if summary:
        # Read directly from summary JSON (authoritative source)
        metadata['feature_type'] = summary.get('feature_type', '')
        metadata['filter_variant'] = summary.get('filter_variant', '')
        metadata['n_observations'] = summary.get('n_observations', 0)
        metadata['n_countries'] = summary.get('n_countries', 0)
        metadata['n_districts'] = summary.get('n_districts', 0)

        # Check for HMM+DMD
        feature_breakdown = summary.get('feature_breakdown', {})
        if feature_breakdown.get('n_hmm_features', 0) > 0 or feature_breakdown.get('n_dmd_features', 0) > 0:
            metadata['has_hmm_dmd'] = True
    else:
        # Fallback: infer from model name (not preferred)
        if 'ratio' in model_name and 'zscore' not in model_name:
            metadata['feature_type'] = 'ratio'
        elif 'zscore' in model_name:
            metadata['feature_type'] = 'zscore'

        if 'with_ar' in model_name:
            metadata['filter_variant'] = 'WITH_AR_FILTER'
        elif 'no_ar' in model_name and 'no_filter' not in model_name:
            metadata['filter_variant'] = 'NO_AR_FILTER'
        elif 'no_filter' in model_name:
            metadata['filter_variant'] = 'NO_FILTER'

        metadata['has_hmm_dmd'] = 'hmm_dmd' in model_name

    return metadata


def get_all_model_summaries() -> Dict[str, Dict]:
    """
    Load all model summaries from Phase 3 results.

    Returns:
        Dictionary mapping model_name -> summary_dict
    """
    summaries = {}

    for model in auto_detect_models():
        summary = load_model_summary(model)
        if summary:
            summaries[model] = summary

    return summaries


def get_overall_metrics_from_summary(summary: Dict) -> Dict:
    """
    Extract overall metrics from a model summary.
    Handles multiple possible field names.

    Args:
        summary: Model summary dictionary

    Returns:
        Dictionary with standardized metric names
    """
    overall_metrics = summary.get('overall_metrics', {})
    threshold_opt = summary.get('threshold_optimization', {})
    class_bal = summary.get('class_balance', {})

    return {
        # AUC metrics
        'auc_roc': overall_metrics.get('auc_roc', overall_metrics.get('mean_fold_auc')),
        'mean_fold_auc': overall_metrics.get('mean_fold_auc'),
        'std_fold_auc': overall_metrics.get('std_fold_auc'),

        # Youden threshold metrics
        'mean_precision_youden': overall_metrics.get('mean_precision_youden', overall_metrics.get('mean_precision')),
        'mean_recall_youden': overall_metrics.get('mean_recall_youden', overall_metrics.get('mean_recall')),
        'mean_f1_youden': overall_metrics.get('mean_f1_youden', overall_metrics.get('mean_f1')),
        'mean_specificity_youden': overall_metrics.get('mean_specificity_youden', overall_metrics.get('mean_specificity')),

        # F1 threshold metrics
        'mean_precision_f1': overall_metrics.get('mean_precision_f1'),
        'mean_recall_f1': overall_metrics.get('mean_recall_f1'),

        # High recall threshold metrics
        'mean_precision_high_recall': overall_metrics.get('mean_precision_high_recall', overall_metrics.get('mean_precision_hr')),
        'mean_recall_high_recall': overall_metrics.get('mean_recall_high_recall', overall_metrics.get('mean_recall_hr')),

        # Calibration
        'overall_brier_score': overall_metrics.get('overall_brier_score'),
        'mean_log_loss': overall_metrics.get('mean_log_loss'),

        # Thresholds
        'mean_threshold_youden': threshold_opt.get('mean_threshold_youden', threshold_opt.get('mean_youden_threshold')),
        'mean_threshold_f1': threshold_opt.get('mean_threshold_f1', threshold_opt.get('mean_f1_threshold')),
        'mean_threshold_high_recall': threshold_opt.get('mean_threshold_high_recall', threshold_opt.get('mean_high_recall_threshold')),

        # Class balance
        'prevalence': class_bal.get('prevalence'),
        'n_crisis': class_bal.get('n_crisis'),
        'n_no_crisis': class_bal.get('n_no_crisis'),
    }


def combine_all_country_metrics() -> pd.DataFrame:
    """
    Combine country metrics from all Phase 3 models into single DataFrame.
    READS actual metrics from Phase 3 - does NOT recompute.

    Returns:
        DataFrame with country metrics from all models
    """
    all_metrics = []

    for model in auto_detect_models():
        country_df = load_model_country_metrics(model)

        if country_df is not None:
            # Ensure model column exists
            if 'model' not in country_df.columns:
                country_df['model'] = model

            # Add metadata
            metadata = get_model_metadata(model)
            country_df['feature_type'] = metadata['feature_type']
            country_df['filter_variant'] = metadata['filter_variant']
            country_df['has_hmm_dmd'] = metadata['has_hmm_dmd']

            all_metrics.append(country_df)

    if all_metrics:
        return pd.concat(all_metrics, ignore_index=True)

    return pd.DataFrame()


def print_auto_detection_summary():
    """Print summary of auto-detected configuration."""
    print("\n" + "=" * 60)
    print("AUTO-DETECTION SUMMARY")
    print("=" * 60)

    models = auto_detect_models()
    print(f"\nModels detected: {len(models)}")
    for m in models:
        print(f"   - {m}")

    threshold_types = auto_detect_threshold_types()
    print(f"\nThreshold types detected: {threshold_types}")

    best_model, best_auc = auto_select_best_model()
    print(f"\nBest model (by AUC): {best_model} (AUC = {best_auc:.4f})")

    print("=" * 60 + "\n")


# Run detection on import
if __name__ == "__main__":
    print_auto_detection_summary()
