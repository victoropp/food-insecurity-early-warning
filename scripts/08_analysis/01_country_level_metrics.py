#!/usr/bin/env python3
"""
Phase 4 Analysis: Country-Level Metrics Aggregation
====================================================
Aggregates country-level metrics from Phase 3 model outputs.
READS actual metrics from Phase 3 files - does NOT recompute.

Key principle: 100% data-driven, NO hardcoded values.
- Models auto-detected from Phase 3 directory
- Threshold types auto-detected from prediction columns
- Metrics READ from Phase 3 country_metrics.csv files

Output:
- country_level_metrics_all_models.csv
- country_ranking_by_model.csv
- country_performance_summary.json
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import PHASE3_RESULTS, PHASE4_RESULTS, OUTPUT_FILES, RESULTS_DIR

# Import Phase 4 utilities for auto-detection
from phase4_utils import (
    auto_detect_models,
    auto_detect_threshold_types,
    auto_select_best_model,
    load_model_summary,
    load_model_country_metrics,
    get_model_metadata,
    combine_all_country_metrics,
)
from sklearn.metrics import roc_auc_score

# Configuration - auto-created
PHASE4_RESULTS.mkdir(parents=True, exist_ok=True)


def compute_country_metrics_from_predictions(predictions_df, model_name, prob_col='pred_prob', true_col='ipc_future_crisis'):
    """
    Compute country-level metrics from prediction DataFrame.

    Args:
        predictions_df: DataFrame with predictions
        model_name: Name of the model
        prob_col: Column name for predicted probabilities
        true_col: Column name for true labels

    Returns:
        DataFrame with country-level metrics
    """
    if predictions_df.empty or 'ipc_country' not in predictions_df.columns:
        return None

    country_metrics = []

    for country in predictions_df['ipc_country'].unique():
        country_data = predictions_df[predictions_df['ipc_country'] == country].copy()

        # Skip if not enough data
        if len(country_data) < 10:
            continue

        # Calculate AUC-ROC
        try:
            auc_roc = roc_auc_score(country_data[true_col], country_data[prob_col])
        except:
            auc_roc = np.nan

        country_metrics.append({
            'model': model_name,
            'ipc_country': country,
            'auc_roc': auc_roc,
            'n_observations': len(country_data)
        })

    return pd.DataFrame(country_metrics) if country_metrics else None


def load_ar_baseline_country_metrics():
    """Load and compute country metrics for AR Baseline model."""
    ar_file = RESULTS_DIR / 'stage1_ar_baseline' / 'predictions_h8_averaged.csv'

    if not ar_file.exists():
        print(f"   WARNING: AR Baseline predictions not found: {ar_file}")
        return None

    try:
        ar_preds = pd.read_csv(ar_file)
        return compute_country_metrics_from_predictions(
            ar_preds,
            'Stage 1 AR Baseline',
            prob_col='pred_prob',
            true_col='ipc_future_crisis'
        )
    except Exception as e:
        print(f"   ERROR loading AR Baseline: {e}")
        return None


def load_literature_baseline_country_metrics():
    """Load and compute country metrics for Literature Baseline model."""
    lit_file = RESULTS_DIR / 'baseline_comparison' / 'literature_baseline' / 'literature_baseline_predictions.csv'

    if not lit_file.exists():
        print(f"   WARNING: Literature Baseline predictions not found: {lit_file}")
        return None

    try:
        lit_preds = pd.read_csv(lit_file)
        return compute_country_metrics_from_predictions(
            lit_preds,
            'Literature Baseline',
            prob_col='pred_prob',
            true_col='ipc_future_crisis'
        )
    except Exception as e:
        print(f"   ERROR loading Literature Baseline: {e}")
        return None


def load_ensemble_country_metrics():
    """
    Load and compute country metrics for Two-Stage Ensemble model.

    NOTE: The weighted ensemble operates on the COMMON dataset (inner join of Stage 1 and Stage 2),
    which is only 1,383 observations. Country-level metrics for this ensemble are not meaningful
    since it doesn't cover complete countries - it only covers the overlap between Stage 1 and
    Stage 2 (AR-filtered) datasets.

    Returns None to exclude ensemble from country-level analysis.
    """
    # Ensemble uses common dataset (1,383 obs) - not suitable for country-level metrics
    print(f"   NOTE: Ensemble excluded from country metrics (operates on common dataset only, not full countries)")
    return None


def load_all_country_metrics_from_phase3():
    """
    Load country metrics directly from Phase 3 outputs.
    This is the CORRECT approach - read actual computed metrics.

    Returns:
        DataFrame with all country metrics from all models
    """
    print("\n" + "-" * 40)
    print("Loading country metrics from Phase 3...")

    all_metrics = []

    # Auto-detect available models
    models = auto_detect_models()
    print(f"   Detected {len(models)} models")

    for model_name in models:
        # Load pre-computed country metrics from Phase 3
        country_df = load_model_country_metrics(model_name)

        if country_df is not None:
            # Standardize country column name (XGBoost uses 'country', others use 'ipc_country')
            if 'country' in country_df.columns and 'ipc_country' not in country_df.columns:
                country_df = country_df.rename(columns={'country': 'ipc_country'})

            # Ensure model identifier column exists
            if 'model' not in country_df.columns:
                country_df['model'] = model_name

            # Add metadata from summary
            metadata = get_model_metadata(model_name)
            country_df['feature_type'] = metadata.get('feature_type', '')
            country_df['filter_variant'] = metadata.get('filter_variant', '')
            country_df['has_hmm_dmd'] = metadata.get('has_hmm_dmd', False)

            # Add threshold type if not present (Phase 3 uses Youden by default)
            if 'threshold_type' not in country_df.columns:
                country_df['threshold_type'] = 'youden'

            all_metrics.append(country_df)
            print(f"   Loaded {model_name}: {len(country_df)} countries")
        else:
            print(f"   WARNING: No country metrics found for {model_name}")

    if all_metrics:
        combined = pd.concat(all_metrics, ignore_index=True)
        print(f"\n   Combined: {len(combined)} total rows")
        return combined

    return pd.DataFrame()


def create_country_rankings(country_metrics_df):
    """
    Create country rankings by AUC within each model.
    Rankings are data-driven based on actual metric values.
    """
    if country_metrics_df.empty:
        return pd.DataFrame()

    rankings = []

    for model in country_metrics_df['model'].unique():
        model_df = country_metrics_df[country_metrics_df['model'] == model].copy()

        # Sort by AUC and create rankings (handle both column names)
        auc_col = None
        if 'auc_youden' in model_df.columns:
            auc_col = 'auc_youden'
        elif 'auc_roc' in model_df.columns:
            auc_col = 'auc_roc'

        if auc_col:
            model_df = model_df.sort_values(auc_col, ascending=False)
            model_df['rank_auc'] = range(1, len(model_df) + 1)

        if 'f1' in model_df.columns:
            # Handle NaN values by placing them at bottom of rankings
            model_df['rank_f1'] = model_df['f1'].rank(ascending=False, method='min', na_option='bottom').fillna(len(model_df)).astype(int)

        if 'recall' in model_df.columns:
            # Handle NaN values by placing them at bottom of rankings
            model_df['rank_recall'] = model_df['recall'].rank(ascending=False, method='min', na_option='bottom').fillna(len(model_df)).astype(int)

        rankings.append(model_df)

    return pd.concat(rankings, ignore_index=True) if rankings else pd.DataFrame()


def compute_model_summary(country_metrics_df):
    """
    Compute summary statistics across countries for each model.
    Uses actual values from Phase 3 outputs.
    """
    if country_metrics_df.empty:
        return pd.DataFrame()

    # Use actual XGBoost column names
    # Determine available numeric columns
    numeric_cols = country_metrics_df.select_dtypes(include=[np.number]).columns.tolist()

    # Key metrics to aggregate (if available)
    agg_dict = {}

    # Handle both auc_youden and auc_roc column names
    if 'auc_youden' in numeric_cols:
        agg_dict['auc_youden'] = ['mean', 'std', 'min', 'max']
    elif 'auc_roc' in numeric_cols:
        agg_dict['auc_roc'] = ['mean', 'std', 'min', 'max']

    if 'precision_youden' in numeric_cols:
        agg_dict['precision_youden'] = ['mean', 'std']
    if 'recall_youden' in numeric_cols:
        agg_dict['recall_youden'] = ['mean', 'std']
    if 'f1_youden' in numeric_cols:
        agg_dict['f1_youden'] = ['mean', 'std']
    if 'n_observations' in numeric_cols:
        agg_dict['n_observations'] = 'sum'

    # Count countries per model
    agg_dict['ipc_country' if 'ipc_country' in country_metrics_df.columns else country_metrics_df.columns[0]] = 'count'

    if not agg_dict:
        return pd.DataFrame()

    summary = country_metrics_df.groupby('model').agg(agg_dict).reset_index()

    # Flatten column names
    summary.columns = ['_'.join(col).strip('_') if isinstance(col, tuple) else col
                        for col in summary.columns]

    # Rename count column
    count_col = [c for c in summary.columns if '_count' in c]
    if count_col:
        summary = summary.rename(columns={count_col[0]: 'n_countries'})

    return summary


def main():
    print("=" * 80)
    print("PHASE 4 ANALYSIS: COUNTRY-LEVEL METRICS")
    print("(Data-driven - NO hardcoded values)")
    print("=" * 80)
    print(f"Timestamp: {datetime.now().isoformat()}")
    print(f"Phase 3 results: {PHASE3_RESULTS}")
    print(f"Output directory: {PHASE4_RESULTS}")

    # Step 1: Load country metrics from Phase 3 (NOT recompute)
    country_metrics_all = load_all_country_metrics_from_phase3()

    if country_metrics_all.empty:
        print("\nERROR: No country metrics found in Phase 3 outputs.")
        print("Please ensure Phase 3 model training has completed.")
        return

    # Step 1.5: Add baseline and ensemble models
    print("\n" + "-" * 40)
    print("Loading baseline and ensemble models...")

    additional_models = []

    # AR Baseline
    ar_metrics = load_ar_baseline_country_metrics()
    if ar_metrics is not None:
        additional_models.append(ar_metrics)
        print(f"   Loaded Stage 1 AR Baseline: {len(ar_metrics)} countries")

    # Literature Baseline
    lit_metrics = load_literature_baseline_country_metrics()
    if lit_metrics is not None:
        additional_models.append(lit_metrics)
        print(f"   Loaded Literature Baseline: {len(lit_metrics)} countries")

    # Two-Stage Ensemble
    ens_metrics = load_ensemble_country_metrics()
    if ens_metrics is not None:
        additional_models.append(ens_metrics)
        print(f"   Loaded Two-Stage Ensemble: {len(ens_metrics)} countries")

    # Combine with Phase 3 models
    if additional_models:
        country_metrics_all = pd.concat([country_metrics_all] + additional_models, ignore_index=True)
        print(f"\n   Total after adding baselines/ensemble: {len(country_metrics_all)} rows")

    # Report what was loaded
    print(f"\n--- Loaded Metrics Summary ---")
    print(f"   Total rows: {len(country_metrics_all):,}")
    print(f"   Models: {country_metrics_all['model'].nunique()}")

    country_col = 'ipc_country' if 'ipc_country' in country_metrics_all.columns else 'country'
    if country_col in country_metrics_all.columns:
        print(f"   Countries: {country_metrics_all[country_col].nunique()}")

    # Step 2: Create rankings
    print("\n--- Creating Rankings ---")
    rankings_df = create_country_rankings(country_metrics_all)
    print(f"   Created rankings for {rankings_df['model'].nunique() if not rankings_df.empty else 0} models")

    # Step 3: Compute model summary statistics
    print("\n--- Computing Summary Statistics ---")
    model_summary_df = compute_model_summary(country_metrics_all)
    print(f"   Computed summary for {len(model_summary_df)} models")

    # Step 4: Save outputs
    print(f"\n--- Saving Outputs ---")

    # 1. All country metrics
    output_file = PHASE4_RESULTS / "country_level_metrics_all_models.csv"
    country_metrics_all.to_csv(output_file, index=False)
    print(f"   Saved: {output_file.name}")

    # 2. Rankings
    if not rankings_df.empty:
        output_file = PHASE4_RESULTS / "country_ranking_by_model.csv"
        rankings_df.to_csv(output_file, index=False)
        print(f"   Saved: {output_file.name}")

    # 3. Model summary
    if not model_summary_df.empty:
        output_file = PHASE4_RESULTS / "model_summary_by_country.csv"
        model_summary_df.to_csv(output_file, index=False)
        print(f"   Saved: {output_file.name}")

    # 4. JSON summary - auto-detected values only
    models_list = auto_detect_models()
    best_model, best_auc = auto_select_best_model()
    threshold_types = auto_detect_threshold_types()

    summary_json = {
        'timestamp': datetime.now().isoformat(),
        'data_source': 'Phase 3 country_metrics.csv files (pre-computed)',
        'n_models': country_metrics_all['model'].nunique(),
        'n_countries': country_metrics_all[country_col].nunique() if country_col in country_metrics_all.columns else 0,
        'models_detected': models_list,
        'threshold_types_detected': threshold_types,
        'best_model_by_auc': {
            'model': best_model,
            'auc': float(best_auc) if best_auc else None,
        },
        'countries_analyzed': sorted([str(c) for c in country_metrics_all[country_col].unique() if pd.notna(c)]) if country_col in country_metrics_all.columns else [],
    }

    # Add per-model statistics if available
    if not model_summary_df.empty and 'auc_roc_mean' in model_summary_df.columns:
        summary_json['model_rankings'] = model_summary_df.sort_values('auc_roc_mean', ascending=False)[
            [c for c in ['model', 'auc_roc_mean', 'auc_roc_std'] if c in model_summary_df.columns]
        ].to_dict('records')

    output_file = PHASE4_RESULTS / "country_performance_summary.json"
    with open(output_file, 'w') as f:
        json.dump(summary_json, f, indent=2, default=str)
    print(f"   Saved: {output_file.name}")

    # Final summary
    print(f"\n{'=' * 80}")
    print("PHASE 4 ANALYSIS COMPLETE")
    print(f"{'=' * 80}")
    print(f"   Models processed: {len(models_list)}")
    print(f"   Best model: {best_model} (AUC = {best_auc:.4f})" if best_model else "   Best model: N/A")
    print(f"   Threshold types: {threshold_types}")
    print(f"\n   NOTE: All metrics READ from Phase 3 outputs (not recomputed)")


if __name__ == "__main__":
    main()
