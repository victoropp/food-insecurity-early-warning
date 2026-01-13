"""
Stage 3: Optimized Cascade Ensemble - PRODUCTION VERSION
=========================================================
Production-ready cascade ensemble using optimized Stage 2 XGBoost model.

This script creates the final cascade ensemble with:
- Optimized Stage 2 XGBoost (hyperparameter-tuned)
- Simple binary override logic
- Full metadata for district and country level analysis
- Comprehensive metrics for dissertation

CASCADE STRATEGY (Simple Binary Logic):
1. Use AR baseline predictions as primary
2. If AR = 1: Keep as 1 (trust AR's crisis prediction)
3. If AR = 0: Use Stage 2's binary prediction
   - If Stage 2 = 1: Override to 1 (Stage 2 detected crisis)
   - If Stage 2 = 0: Keep as 0 (both agree: no crisis)

Author: Victor Collins Oppon
Date: December 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    f1_score, precision_score, recall_score, accuracy_score,
    confusion_matrix, classification_report, roc_auc_score,
    average_precision_score, brier_score_loss, log_loss,
    precision_recall_curve, roc_curve
)
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import BASE_DIR, RESULTS_DIR, STAGE1_RESULTS_DIR, STAGE2_MODELS_DIR

# =============================================================================
# OUTPUT DIRECTORIES
# =============================================================================

OUTPUT_DIR = RESULTS_DIR / 'cascade_optimized_production'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FIGURES_DIR = BASE_DIR / 'FIGURES' / 'cascade_optimized'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("OPTIMIZED CASCADE ENSEMBLE - PRODUCTION VERSION")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()
print("Strategy: AR Baseline + Optimized Stage 2 XGBoost Override")
print("Logic: Simple binary override (AR=0 cases use Stage 2 prediction)")
print()

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def cost_sensitive_threshold_search(y_true, y_pred_prob, cost_fn=10, cost_fp=1,
                                    thresholds=None):
    """
    Find optimal threshold that minimizes weighted cost.

    Cost function: 10*FN + 1*FP (missing a crisis is 10x worse than false alarm)
    """
    if thresholds is None:
        thresholds = np.linspace(0.05, 0.95, 91)

    results = []
    for thresh in thresholds:
        y_pred = (y_pred_prob >= thresh).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0

        total_cost = cost_fn * fn + cost_fp * fp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

        results.append({
            'threshold': thresh,
            'cost': total_cost,
            'fn': fn, 'fp': fp, 'tp': tp, 'tn': tn,
            'precision': precision, 'recall': recall, 'f1': f1
        })

    results_df = pd.DataFrame(results)
    optimal_idx = results_df['cost'].idxmin()
    optimal = results_df.iloc[optimal_idx]

    return {
        'optimal_threshold': optimal['threshold'],
        'optimal_cost': optimal['cost'],
        'optimal_metrics': optimal.to_dict(),
        'all_results': results_df
    }


def assign_confusion_class(y_true, y_pred):
    """Assign confusion class labels for cartographic mapping."""
    classes = np.empty(len(y_true), dtype=object)
    classes[(y_true == 0) & (y_pred == 0)] = 'TN'
    classes[(y_true == 1) & (y_pred == 1)] = 'TP'
    classes[(y_true == 0) & (y_pred == 1)] = 'FP'
    classes[(y_true == 1) & (y_pred == 0)] = 'FN'
    return classes


def compute_comprehensive_metrics(y_true, y_pred, y_pred_proba=None):
    """Compute comprehensive metrics for a prediction."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    balanced_accuracy = (recall + specificity) / 2

    # Positive Predictive Value (same as precision)
    ppv = precision
    # Negative Predictive Value
    npv = tn / (tn + fn) if (tn + fn) > 0 else 0
    # False Positive Rate
    fpr = fp / (fp + tn) if (fp + tn) > 0 else 0
    # False Negative Rate
    fnr = fn / (fn + tp) if (fn + tp) > 0 else 0

    # Cost (10:1 ratio for FN:FP)
    cost_10_1 = 10 * fn + 1 * fp

    metrics = {
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'ppv': ppv,
        'npv': npv,
        'fpr': fpr,
        'fnr': fnr,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn),
        'total': int(tp + tn + fp + fn),
        'cost_10_1': int(cost_10_1)
    }

    # Add AUC metrics if probabilities provided
    if y_pred_proba is not None:
        try:
            if len(np.unique(y_true)) > 1:
                metrics['auc_roc'] = roc_auc_score(y_true, y_pred_proba)
                metrics['pr_auc'] = average_precision_score(y_true, y_pred_proba)
                metrics['brier_score'] = brier_score_loss(y_true, y_pred_proba)
            else:
                metrics['auc_roc'] = np.nan
                metrics['pr_auc'] = np.nan
                metrics['brier_score'] = np.nan
        except:
            metrics['auc_roc'] = np.nan
            metrics['pr_auc'] = np.nan
            metrics['brier_score'] = np.nan

    return metrics


# =============================================================================
# STEP 1: LOAD DATA
# =============================================================================

print("-" * 80)
print("STEP 1: Loading Data")
print("-" * 80)

# Stage 1: AR Baseline predictions
stage1_file = STAGE1_RESULTS_DIR / 'predictions_h8_averaged.csv'
print(f"Loading Stage 1 (AR Baseline): {stage1_file.name}")
stage1_df = pd.read_csv(stage1_file)
print(f"  Total observations: {len(stage1_df):,}")

# Extract relevant columns
stage1_cols = [
    # Geographic identifiers
    'ipc_geographic_unit_full', 'ipc_district', 'ipc_region',
    'ipc_country', 'ipc_country_code',
    # Coordinates
    'avg_latitude', 'avg_longitude',
    # Temporal identifiers
    'year_month', 'ipc_period_start', 'ipc_period_end',
    # IPC data
    'ipc_value', 'ipc_future_crisis',
    # AR predictions
    'pred_prob', 'y_pred_optimal',
    # Fold
    'fold'
]

available_cols = [c for c in stage1_cols if c in stage1_df.columns]
stage1_data = stage1_df[available_cols].copy()

# Rename for clarity
stage1_data = stage1_data.rename(columns={
    'pred_prob': 'ar_prob',
    'y_pred_optimal': 'ar_pred',
    'ipc_future_crisis': 'y_true'
})

stage1_data['date'] = pd.to_datetime(stage1_data['ipc_period_start'])
stage1_data = stage1_data[stage1_data['y_true'].notna()].copy()

n_total = len(stage1_data)
n_crisis = int(stage1_data['y_true'].sum())
crisis_rate = 100 * n_crisis / n_total

print(f"  Valid observations: {n_total:,}")
print(f"  Crisis events: {n_crisis:,} ({crisis_rate:.1f}%)")
print(f"  Non-crisis events: {n_total - n_crisis:,} ({100-crisis_rate:.1f}%)")
print(f"  Countries: {stage1_data['ipc_country'].nunique()}")
print(f"  Districts: {stage1_data['ipc_geographic_unit_full'].nunique()}")

# Stage 2: Optimized XGBoost predictions
stage2_file = STAGE2_MODELS_DIR / 'xgboost' / 'advanced_with_ar_optimized' / 'xgboost_optimized_predictions.csv'
print(f"\nLoading Stage 2 (Optimized XGBoost): {stage2_file.name}")
stage2_df = pd.read_csv(stage2_file)
print(f"  Total observations: {len(stage2_df):,}")

# Extract Stage 2 predictions (using Youden optimal threshold)
stage2_cols = ['ipc_geographic_unit_full', 'ipc_period_start', 'y_pred_youden', 'ipc_future_crisis']
stage2_data = stage2_df[stage2_cols].copy()
stage2_data.columns = ['geographic_unit', 'date', 'stage2_pred', 'y_true_s2']
stage2_data['date'] = pd.to_datetime(stage2_data['date'])

# =============================================================================
# STEP 2: MERGE DATASETS
# =============================================================================

print()
print("-" * 80)
print("STEP 2: Merging Datasets")
print("-" * 80)

# Merge Stage 1 and Stage 2
full_df = stage1_data.merge(
    stage2_data[['geographic_unit', 'date', 'stage2_pred']],
    left_on=['ipc_geographic_unit_full', 'date'],
    right_on=['geographic_unit', 'date'],
    how='left'
)
full_df = full_df.drop(columns=['geographic_unit'], errors='ignore')

n_with_s2 = full_df['stage2_pred'].notna().sum()
n_without_s2 = full_df['stage2_pred'].isna().sum()

print(f"Full dataset: {len(full_df):,} observations")
print(f"  With Stage 2 predictions: {n_with_s2:,} ({100*n_with_s2/len(full_df):.1f}%)")
print(f"  Without Stage 2: {n_without_s2:,} ({100*n_without_s2/len(full_df):.1f}%)")
print()
print("Stage 2 coverage explanation:")
print("  Stage 2 only covers IPC <= 2 AND AR == 0 (cases where AR predicts no crisis)")
print("  These are the cases where we want to detect crises AR might miss")

# =============================================================================
# STEP 3: SIMPLE CASCADE LOGIC (BINARY PREDICTIONS ONLY)
# =============================================================================

print()
print("-" * 80)
print("STEP 3: Applying Simple Cascade Logic")
print("-" * 80)
print()
print("CASCADE RULE:")
print("  - If AR prediction = 1: Keep as 1 (trust AR's crisis prediction)")
print("  - If AR prediction = 0: Use Stage 2's binary prediction")
print("    * If Stage 2 prediction = 1: Override to 1 (Stage 2 detected crisis)")
print("    * If Stage 2 prediction = 0: Keep as 0 (both agree: no crisis)")

# =============================================================================
# STEP 4: BUILD SIMPLE CASCADE ENSEMBLE
# =============================================================================

print()
print("-" * 80)
print("STEP 4: Building Simple Cascade Ensemble")
print("-" * 80)

# Get arrays
y_true = full_df['y_true'].values
ar_pred = full_df['ar_pred'].astype(int).values
ar_prob = full_df['ar_prob'].values  # Keep for AUC calculations

# Identify where Stage 2 can contribute
has_s2 = full_df['stage2_pred'].notna()
ar_predicts_no_crisis = ar_pred == 0
can_use_s2 = ar_predicts_no_crisis & has_s2

print(f"\nCases where AR predicted 0: {ar_predicts_no_crisis.sum():,}")
print(f"Cases with Stage 2 prediction: {can_use_s2.sum():,}")

# Simple cascade logic
cascade_pred = ar_pred.copy()  # Start with AR predictions

# Where AR=0 and Stage 2 available, use Stage 2's prediction
override_mask = np.zeros(len(full_df), dtype=bool)
for idx in full_df[can_use_s2].index:
    row_loc = full_df.index.get_loc(idx)
    s2_pred = int(full_df.loc[idx, 'stage2_pred'])

    if s2_pred == 1:
        cascade_pred[row_loc] = 1  # Stage 2 says crisis
        override_mask[row_loc] = True

full_df['cascade_pred'] = cascade_pred
full_df['was_overridden'] = override_mask.astype(int)

n_overrides = override_mask.sum()
print(f"\nStage 2 overrides (AR=0 -> 1): {n_overrides:,}")
print(f"Stage 2 confirms AR=0: {(can_use_s2 & ~override_mask).sum():,}")
print(f"Override rate: {100*n_overrides/can_use_s2.sum():.1f}% of AR=0 cases with Stage 2")

# =============================================================================
# STEP 5: COMPUTE KEY SAVES
# =============================================================================

print()
print("-" * 80)
print("STEP 5: Computing Key Saves")
print("-" * 80)

# Key saves: crises that AR missed but cascade caught
ar_missed_crisis = (ar_pred == 0) & (y_true == 1)
cascade_caught = cascade_pred == 1
key_saves_mask = ar_missed_crisis & cascade_caught

full_df['ar_missed'] = ar_missed_crisis.astype(int)
full_df['is_key_save'] = key_saves_mask.astype(int)

n_key_saves = key_saves_mask.sum()
n_ar_missed = ar_missed_crisis.sum()

print(f"AR baseline missed: {n_ar_missed:,} crises")
print(f"Cascade caught (key saves): {n_key_saves:,} crises")
print(f"Key save rate: {100*n_key_saves/n_ar_missed:.1f}% of AR misses")

# Key saves by country
print("\nKey saves by country:")
key_saves_by_country = full_df[full_df['is_key_save'] == 1].groupby('ipc_country').size().sort_values(ascending=False)
for country, count in key_saves_by_country.head(10).items():
    print(f"  {country:<30}: {count:>4}")

# =============================================================================
# STEP 6: COMPUTE COMPREHENSIVE METRICS
# =============================================================================

print()
print("-" * 80)
print("STEP 6: Computing Comprehensive Metrics")
print("-" * 80)

# AR Baseline metrics
ar_metrics = compute_comprehensive_metrics(y_true, ar_pred, ar_prob)
print("\nAR BASELINE METRICS:")
print(f"  Precision: {ar_metrics['precision']:.4f}")
print(f"  Recall: {ar_metrics['recall']:.4f}")
print(f"  F1: {ar_metrics['f1']:.4f}")
print(f"  Specificity: {ar_metrics['specificity']:.4f}")
print(f"  Balanced Accuracy: {ar_metrics['balanced_accuracy']:.4f}")
print(f"  AUC-ROC: {ar_metrics.get('auc_roc', 'N/A')}")
print(f"  FN (missed crises): {ar_metrics['fn']:,}")

# Cascade metrics
cascade_metrics = compute_comprehensive_metrics(y_true, cascade_pred)
print("\nCASCADE ENSEMBLE METRICS:")
print(f"  Precision: {cascade_metrics['precision']:.4f}")
print(f"  Recall: {cascade_metrics['recall']:.4f}")
print(f"  F1: {cascade_metrics['f1']:.4f}")
print(f"  Specificity: {cascade_metrics['specificity']:.4f}")
print(f"  Balanced Accuracy: {cascade_metrics['balanced_accuracy']:.4f}")
print(f"  FN (missed crises): {cascade_metrics['fn']:,}")

# Improvement
print("\nIMPROVEMENT OVER AR BASELINE:")
print(f"  Recall change: {cascade_metrics['recall'] - ar_metrics['recall']:+.4f}")
print(f"  Precision change: {cascade_metrics['precision'] - ar_metrics['precision']:+.4f}")
print(f"  F1 change: {cascade_metrics['f1'] - ar_metrics['f1']:+.4f}")
print(f"  FN reduction: {ar_metrics['fn'] - cascade_metrics['fn']:,} fewer missed crises")
print(f"  Key saves: {n_key_saves:,} crises saved")

# =============================================================================
# STEP 7: ASSIGN CONFUSION CLASSES FOR MAPPING
# =============================================================================

print()
print("-" * 80)
print("STEP 7: Assigning Confusion Classes for Mapping")
print("-" * 80)

full_df['confusion_ar'] = assign_confusion_class(y_true, ar_pred)
full_df['confusion_cascade'] = assign_confusion_class(y_true, cascade_pred)

# Summary
for model_name, col in [('AR Baseline', 'confusion_ar'), ('Cascade', 'confusion_cascade')]:
    print(f"\n{model_name} Confusion Distribution:")
    for cls in ['TP', 'TN', 'FP', 'FN']:
        count = (full_df[col] == cls).sum()
        pct = 100 * count / len(full_df)
        print(f"  {cls}: {count:,} ({pct:.1f}%)")

# =============================================================================
# STEP 8: COMPUTE COUNTRY-LEVEL METRICS
# =============================================================================

print()
print("-" * 80)
print("STEP 8: Computing Country-Level Metrics")
print("-" * 80)

country_metrics = []

for country in full_df['ipc_country'].unique():
    country_df = full_df[full_df['ipc_country'] == country].copy()

    y_true_c = country_df['y_true'].values
    ar_pred_c = country_df['ar_pred'].values
    cascade_pred_c = country_df['cascade_pred'].values
    ar_prob_c = country_df['ar_prob'].values

    # AR metrics
    ar_m = compute_comprehensive_metrics(y_true_c, ar_pred_c, ar_prob_c)

    # Cascade metrics
    cas_m = compute_comprehensive_metrics(y_true_c, cascade_pred_c)

    # Key saves for this country
    country_key_saves = country_df['is_key_save'].sum()
    country_ar_missed = country_df['ar_missed'].sum()

    country_row = {
        'country': country,
        'n_observations': len(country_df),
        'n_crisis': int(y_true_c.sum()),
        'crisis_rate': float(y_true_c.mean()),
        'n_districts': country_df['ipc_geographic_unit_full'].nunique(),

        # AR Baseline
        'ar_precision': ar_m['precision'],
        'ar_recall': ar_m['recall'],
        'ar_f1': ar_m['f1'],
        'ar_specificity': ar_m['specificity'],
        'ar_balanced_accuracy': ar_m['balanced_accuracy'],
        'ar_auc_roc': ar_m.get('auc_roc', np.nan),
        'ar_tp': ar_m['tp'],
        'ar_tn': ar_m['tn'],
        'ar_fp': ar_m['fp'],
        'ar_fn': ar_m['fn'],

        # Cascade
        'cascade_precision': cas_m['precision'],
        'cascade_recall': cas_m['recall'],
        'cascade_f1': cas_m['f1'],
        'cascade_specificity': cas_m['specificity'],
        'cascade_balanced_accuracy': cas_m['balanced_accuracy'],
        'cascade_tp': cas_m['tp'],
        'cascade_tn': cas_m['tn'],
        'cascade_fp': cas_m['fp'],
        'cascade_fn': cas_m['fn'],

        # Key saves
        'key_saves': int(country_key_saves),
        'ar_missed_crises': int(country_ar_missed),
        'key_save_rate': float(country_key_saves / country_ar_missed) if country_ar_missed > 0 else 0,

        # Improvement
        'recall_improvement': cas_m['recall'] - ar_m['recall'],
        'precision_change': cas_m['precision'] - ar_m['precision'],
        'f1_change': cas_m['f1'] - ar_m['f1'],
        'fn_reduction': ar_m['fn'] - cas_m['fn']
    }

    country_metrics.append(country_row)

country_metrics_df = pd.DataFrame(country_metrics)
country_metrics_df = country_metrics_df.sort_values('key_saves', ascending=False)

print(f"\nCountry-level metrics computed for {len(country_metrics_df)} countries")
print("\nTop countries by key saves:")
for _, row in country_metrics_df.head(5).iterrows():
    print(f"  {row['country']:<25}: {row['key_saves']:>3} key saves, "
          f"recall +{row['recall_improvement']:.3f}")

# =============================================================================
# STEP 9: COMPUTE DISTRICT-LEVEL METRICS
# =============================================================================

print()
print("-" * 80)
print("STEP 9: Computing District-Level Metrics")
print("-" * 80)

district_metrics = []

for district in full_df['ipc_geographic_unit_full'].unique():
    district_df = full_df[full_df['ipc_geographic_unit_full'] == district].copy()

    if len(district_df) < 2:
        continue

    y_true_d = district_df['y_true'].values
    ar_pred_d = district_df['ar_pred'].values
    cascade_pred_d = district_df['cascade_pred'].values

    # Simple metrics (may not have enough data for all metrics)
    ar_m = compute_comprehensive_metrics(y_true_d, ar_pred_d)
    cas_m = compute_comprehensive_metrics(y_true_d, cascade_pred_d)

    district_row = {
        'district': district,
        'country': district_df['ipc_country'].iloc[0],
        'avg_latitude': district_df['avg_latitude'].iloc[0] if 'avg_latitude' in district_df.columns else np.nan,
        'avg_longitude': district_df['avg_longitude'].iloc[0] if 'avg_longitude' in district_df.columns else np.nan,
        'n_observations': len(district_df),
        'n_crisis': int(y_true_d.sum()),
        'crisis_rate': float(y_true_d.mean()),

        # AR Baseline
        'ar_recall': ar_m['recall'],
        'ar_precision': ar_m['precision'],
        'ar_f1': ar_m['f1'],
        'ar_fn': ar_m['fn'],

        # Cascade
        'cascade_recall': cas_m['recall'],
        'cascade_precision': cas_m['precision'],
        'cascade_f1': cas_m['f1'],
        'cascade_fn': cas_m['fn'],

        # Key saves
        'key_saves': int(district_df['is_key_save'].sum()),
        'ar_missed_crises': int(district_df['ar_missed'].sum()),

        # Improvement
        'recall_improvement': cas_m['recall'] - ar_m['recall'],
        'fn_reduction': ar_m['fn'] - cas_m['fn']
    }

    district_metrics.append(district_row)

district_metrics_df = pd.DataFrame(district_metrics)
print(f"District-level metrics computed for {len(district_metrics_df)} districts")

# =============================================================================
# STEP 10: SAVE ALL OUTPUTS
# =============================================================================

print()
print("-" * 80)
print("STEP 10: Saving All Outputs")
print("-" * 80)

# 1. Full predictions with all metadata
predictions_file = OUTPUT_DIR / 'cascade_optimized_predictions.csv'
full_df.to_csv(predictions_file, index=False)
print(f"[OK] Predictions: {predictions_file.name} ({len(full_df):,} rows)")

# 2. Key saves dataset
key_saves_df = full_df[full_df['is_key_save'] == 1].copy()
key_saves_file = OUTPUT_DIR / 'key_saves.csv'
key_saves_df.to_csv(key_saves_file, index=False)
print(f"[OK] Key saves: {key_saves_file.name} ({len(key_saves_df)} rows)")

# 3. Country-level metrics
country_file = OUTPUT_DIR / 'country_metrics.csv'
country_metrics_df.to_csv(country_file, index=False)
print(f"[OK] Country metrics: {country_file.name} ({len(country_metrics_df)} countries)")

# 4. District-level metrics
district_file = OUTPUT_DIR / 'district_metrics.csv'
district_metrics_df.to_csv(district_file, index=False)
print(f"[OK] District metrics: {district_file.name} ({len(district_metrics_df)} districts)")

# 5. Simple summary (no threshold analysis needed for binary logic)
print("[OK] Simple binary cascade - no threshold tuning required")

# 6. Comprehensive summary JSON
summary = {
    'model': 'Optimized Cascade Ensemble',
    'strategy': 'AR Baseline + Optimized Stage 2 XGBoost with Simple Binary Override',
    'timestamp': datetime.now().isoformat(),

    'data': {
        'total_observations': int(len(full_df)),
        'total_crises': int(n_crisis),
        'crisis_rate': float(n_crisis / len(full_df)),
        'countries': int(full_df['ipc_country'].nunique()),
        'districts': int(full_df['ipc_geographic_unit_full'].nunique()),
        'with_stage2_predictions': int(n_with_s2),
        'override_candidates': int(can_use_s2.sum())
    },

    'cascade_strategy': {
        'method': 'Simple Binary Logic',
        'rule': 'If AR=1 keep, if AR=0 use Stage2 prediction',
        'total_overrides': int(n_overrides),
        'override_rate': float(n_overrides / can_use_s2.sum()) if can_use_s2.sum() > 0 else 0
    },

    'ar_baseline_performance': {
        'precision': float(ar_metrics['precision']),
        'recall': float(ar_metrics['recall']),
        'f1': float(ar_metrics['f1']),
        'specificity': float(ar_metrics['specificity']),
        'balanced_accuracy': float(ar_metrics['balanced_accuracy']),
        'auc_roc': float(ar_metrics.get('auc_roc', 0)),
        'confusion_matrix': {
            'tp': ar_metrics['tp'],
            'tn': ar_metrics['tn'],
            'fp': ar_metrics['fp'],
            'fn': ar_metrics['fn']
        }
    },

    'cascade_performance': {
        'precision': float(cascade_metrics['precision']),
        'recall': float(cascade_metrics['recall']),
        'f1': float(cascade_metrics['f1']),
        'specificity': float(cascade_metrics['specificity']),
        'balanced_accuracy': float(cascade_metrics['balanced_accuracy']),
        'confusion_matrix': {
            'tp': cascade_metrics['tp'],
            'tn': cascade_metrics['tn'],
            'fp': cascade_metrics['fp'],
            'fn': cascade_metrics['fn']
        }
    },

    'improvement': {
        'recall_change': float(cascade_metrics['recall'] - ar_metrics['recall']),
        'precision_change': float(cascade_metrics['precision'] - ar_metrics['precision']),
        'f1_change': float(cascade_metrics['f1'] - ar_metrics['f1']),
        'fn_reduction': ar_metrics['fn'] - cascade_metrics['fn'],
        'key_saves': int(n_key_saves),
        'ar_missed_crises': int(n_ar_missed),
        'key_save_rate': float(n_key_saves / n_ar_missed) if n_ar_missed > 0 else 0
    },

    'key_saves_by_country': key_saves_by_country.to_dict(),

    'output_files': {
        'predictions': str(predictions_file.name),
        'key_saves': str(key_saves_file.name),
        'country_metrics': str(country_file.name),
        'district_metrics': str(district_file.name),
    }
}

summary_file = OUTPUT_DIR / 'cascade_optimized_summary.json'
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"[OK] Summary: {summary_file.name}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print("=" * 80)
print("OPTIMIZED CASCADE ENSEMBLE - COMPLETE")
print("=" * 80)
print()
print("STRATEGY:")
print("  - AR Baseline as primary predictor (using optimal threshold)")
print("  - Stage 2 (Optimized XGBoost Advanced) refines AR=0 cases")
print("  - Simple binary logic:")
print("    * If AR = 1: Keep as 1 (trust AR)")
print("    * If AR = 0 and Stage 2 = 1: Override to 1 (Stage 2 detects crisis)")
print("    * If AR = 0 and Stage 2 = 0: Keep as 0 (both agree)")
print()
print("=" * 80)
print("PERFORMANCE: AR BASELINE vs ENSEMBLE")
print("=" * 80)
print()
print("RECALL (Most Important - Catching Crises):")
print(f"  AR Baseline:  {ar_metrics['recall']:.4f} ({ar_metrics['recall']*100:.2f}%)")
print(f"  Ensemble:     {cascade_metrics['recall']:.4f} ({cascade_metrics['recall']*100:.2f}%)")
print(f"  Improvement:  {cascade_metrics['recall']-ar_metrics['recall']:+.4f} ({(cascade_metrics['recall']-ar_metrics['recall'])*100:+.2f} percentage points)")
print(f"  --> THE ENSEMBLE CATCHES {n_key_saves} MORE CRISES THAT AR MISSED")
print()
print("PRECISION:")
print(f"  AR Baseline:  {ar_metrics['precision']:.4f}")
print(f"  Ensemble:     {cascade_metrics['precision']:.4f}")
print(f"  Change:       {cascade_metrics['precision']-ar_metrics['precision']:+.4f}")
print()
print("F1 SCORE:")
print(f"  AR Baseline:  {ar_metrics['f1']:.4f}")
print(f"  Ensemble:     {cascade_metrics['f1']:.4f}")
print(f"  Change:       {cascade_metrics['f1']-ar_metrics['f1']:+.4f}")
print()
print("MISSED CRISES (False Negatives - The Critical Metric):")
print(f"  AR Baseline:  {ar_metrics['fn']:,} crises missed")
print(f"  Ensemble:     {cascade_metrics['fn']:,} crises missed")
print(f"  Reduction:    {ar_metrics['fn']-cascade_metrics['fn']:,} fewer missed crises ({100*(ar_metrics['fn']-cascade_metrics['fn'])/ar_metrics['fn']:.1f}% reduction)")
print()
print("=" * 80)
print("STAGE 2 INCREMENTAL CONTRIBUTION (On AR=0 Cases Only)")
print("=" * 80)
print(f"  Cases where AR predicted 'no crisis': {can_use_s2.sum():,}")
print(f"  Stage 2 overrides (predicted crisis):  {n_overrides:,} ({100*n_overrides/can_use_s2.sum():.1f}%)")
print(f"  Stage 2 confirms AR (no crisis):       {(can_use_s2 & ~override_mask).sum():,}")
print()
print(f"  Stage 2 Performance on Overrides:")
print(f"    - True crises caught: {n_key_saves:,}")
print(f"    - False alarms: {n_overrides - n_key_saves:,}")
print(f"    - Precision on overrides: {n_key_saves/n_overrides:.4f}")
print(f"    - Key save rate: {100*n_key_saves/n_ar_missed:.1f}% of AR's misses")
print()
print("=" * 80)
print("KEY INSIGHT")
print("=" * 80)
print(f"The ensemble improves recall from {ar_metrics['recall']:.4f} to {cascade_metrics['recall']:.4f}.")
print(f"This is not just an improved measurement - it means {n_key_saves} REAL CRISES")
print(f"that would have been MISSED are now PREDICTED and can trigger early warning.")
print(f"These are the cases that matter most: vulnerable populations facing food insecurity.")
print()
print("OUTPUT FILES:")
for name, path in [
    ('Predictions', predictions_file),
    ('Key Saves', key_saves_file),
    ('Country Metrics', country_file),
    ('District Metrics', district_file),
    ('Summary', summary_file)
]:
    print(f"  - {name}: {path.name}")
print()
print(f"Output Directory: {OUTPUT_DIR}")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
