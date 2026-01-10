"""
Stage 2 Ablation Model 3: XGBoost with Ratio + Zscore + HMM + Location (NO DMD) - OPTIMIZED
============================================================================================
Train XGBoost using ratio + zscore + HMM + location features (NO DMD)
WITH_AR_FILTER applied (IPC <= 2 AND AR == 0)

OPTIMIZATION:
- GridSearchCV with Stratified Spatial CV for hyperparameter tuning
- Same methodology as basic/advanced optimized models for fair comparison
- NO feature selection (to cleanly isolate ablation effect)

This tests the contribution of HMM features WITHOUT DMD features.

Author: Victor Collins Oppon
Date: December 2025
"""

import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                            roc_curve, confusion_matrix, brier_score_loss,
                            log_loss, accuracy_score)
from sklearn.model_selection import GridSearchCV
from pathlib import Path
from datetime import datetime
import json
import sys
import joblib
import warnings
warnings.filterwarnings('ignore')

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

# Add parent directories to path for config import
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import from config
from config import (
    BASE_DIR,
    STAGE2_FEATURES_DIR,
    STAGE2_MODELS_DIR,
    RANDOM_STATE
)

# Import stratified spatial CV utility
sys.path.append(str(Path(__file__).parent.parent.parent / 'utils'))
from stratified_spatial_cv import create_stratified_spatial_folds, create_safe_location_features

# Constants
N_FOLDS = 5
N_SPATIAL_CLUSTERS = 20
HIGH_RECALL_TARGET = 0.90

# =============================================================================
# SPATIAL CV ITERATOR CLASS
# =============================================================================

class SpatialFoldCV:
    """Custom CV iterator that uses pre-computed spatial folds."""
    def __init__(self, fold_array, n_folds=5):
        self.fold_array = fold_array
        self.n_folds = n_folds

    def split(self, X, y=None, groups=None):
        for fold in range(self.n_folds):
            train_idx = np.where(self.fold_array != fold)[0]
            test_idx = np.where(self.fold_array == fold)[0]
            yield train_idx, test_idx

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_folds

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_optimal_thresholds(y_true, y_pred_proba):
    """Compute optimal thresholds using multiple strategies."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    youden_j = tpr - fpr
    youden_idx = np.argmax(youden_j)
    youden_threshold = thresholds[youden_idx]
    youden_index = youden_j[youden_idx]

    f1_scores = []
    for threshold in thresholds:
        y_pred = (y_pred_proba >= threshold).astype(int)
        cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
        if cm.size == 4:
            tn, fp, fn, tp = cm.ravel()
        else:
            tn, fp, fn, tp = 0, 0, 0, 0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
        f1_scores.append(f1)

    f1_idx = np.argmax(f1_scores)
    f1_threshold = thresholds[f1_idx]

    high_recall_idx = np.where(tpr >= HIGH_RECALL_TARGET)[0]
    if len(high_recall_idx) > 0:
        high_recall_threshold = thresholds[high_recall_idx[-1]]
    else:
        high_recall_threshold = thresholds[np.argmax(tpr)]

    return {
        'youden_threshold': float(youden_threshold),
        'youden_index': float(youden_index),
        'f1_threshold': float(f1_threshold),
        'high_recall_threshold': float(high_recall_threshold)
    }

def compute_metrics_at_threshold(y_true, y_pred_proba, threshold):
    """Compute comprehensive metrics at a specific threshold."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    if cm.size == 4:
        tn, fp, fn, tp = cm.ravel()
    else:
        tn, fp, fn, tp = 0, 0, 0, 0

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0
    balanced_accuracy = (recall + specificity) / 2

    return {
        'precision': precision, 'recall': recall, 'specificity': specificity,
        'f1': f1, 'accuracy': accuracy, 'balanced_accuracy': balanced_accuracy,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }

def assign_confusion_class(y_true, y_pred):
    """Assign confusion class labels for cartographic mapping."""
    classes = np.empty(len(y_true), dtype=object)
    classes[(y_true == 0) & (y_pred == 0)] = 'TN'
    classes[(y_true == 1) & (y_pred == 1)] = 'TP'
    classes[(y_true == 0) & (y_pred == 1)] = 'FP'
    classes[(y_true == 1) & (y_pred == 0)] = 'FN'
    return classes

# =============================================================================
# PATHS
# =============================================================================

INPUT_FILE = STAGE2_FEATURES_DIR / "phase3_combined" / "combined_advanced_features_h8.csv"
OUTPUT_DIR = STAGE2_MODELS_DIR / "ablation" / "ratio_zscore_hmm_location_optimized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ABLATION MODEL 3 - OPTIMIZED: RATIO + ZSCORE + HMM + LOCATION (NO DMD)")
print("=" * 80)
print(f"\nModel: ablation_ratio_zscore_hmm_location_optimized")
print(f"Filter: WITH_AR_FILTER (IPC <= 2 AND AR == 0)")
print(f"Enhancement: GridSearchCV with Stratified Spatial CV")
print(f"Start time: {datetime.now()}\n")

# =============================================================================
# LOAD DATA
# =============================================================================

print("-" * 80)
print("STEP 1: Loading Data")
print("-" * 80)

df = pd.read_csv(INPUT_FILE)
print(f"Loaded: {len(df):,} observations")

# =============================================================================
# APPLY FILTER
# =============================================================================

print()
print("-" * 80)
print("STEP 2: Applying WITH_AR_FILTER")
print("-" * 80)

initial_rows = len(df)
df_filtered = df[(df['ipc_value_filled'] <= 2) & (df['ar_pred_optimal_filled'] == 0)].copy()
filtered_out = initial_rows - len(df_filtered)

print(f"After IPC <= 2 filter: {df[df['ipc_value_filled'] <= 2].shape[0]:,} rows")
print(f"After AR == 0 filter: {len(df_filtered):,} rows")
print(f"Total filtered out: {filtered_out:,} rows")

# =============================================================================
# CREATE SAFE LOCATION FEATURES
# =============================================================================

print()
print("-" * 80)
print("STEP 3: Creating Safe Location Features")
print("-" * 80)

df_filtered = create_safe_location_features(
    df=df_filtered,
    district_col='ipc_geographic_unit_full',
    country_col='ipc_country',
    verbose=True
)

# =============================================================================
# PREPARE FEATURES - RATIO + ZSCORE + HMM + LOCATION (NO DMD)
# =============================================================================

print()
print("-" * 80)
print("STEP 4: Preparing Features (ABLATION: NO DMD)")
print("-" * 80)

macro_categories = ['conflict', 'displacement', 'economic', 'food_security', 'governance',
                    'health', 'humanitarian', 'other', 'weather']

ratio_cols = [f'{cat}_ratio' for cat in macro_categories]
zscore_cols = [f'{cat}_zscore' for cat in macro_categories]

# HMM features (NO DMD)
hmm_cols = [
    'hmm_ratio_crisis_prob', 'hmm_ratio_transition_risk', 'hmm_ratio_entropy',
    'hmm_zscore_crisis_prob', 'hmm_zscore_transition_risk', 'hmm_zscore_entropy'
]

# 3 derived location features (same as optimized XGBoost models)
safe_location_cols = [
    'country_baseline_conflict',
    'country_baseline_food_security',
    'country_data_density'
]
location_cols = [col for col in safe_location_cols if col in df_filtered.columns]

# ABLATION 3: Ratio + Zscore + HMM + Location (NO DMD)
feature_cols = ratio_cols + zscore_cols + hmm_cols + location_cols
feature_cols = [f for f in feature_cols if f in df_filtered.columns]

print(f"Ratio features: {len([f for f in feature_cols if f in ratio_cols])}")
print(f"Zscore features: {len([f for f in feature_cols if f in zscore_cols])}")
print(f"HMM features: {len([f for f in feature_cols if f in hmm_cols])}")
print(f"Location features: {len(location_cols)}")
print(f"Total features: {len(feature_cols)}")
print(f"EXCLUDED: DMD features")

df_filtered = df_filtered[df_filtered['ipc_future_crisis'].notna()].copy()
print(f"After removing missing target: {len(df_filtered):,} rows")

n_positive = int(df_filtered['ipc_future_crisis'].sum())
n_negative = len(df_filtered) - n_positive
crisis_rate = 100 * n_positive / len(df_filtered)

print(f"Crisis events: {n_positive:,} ({crisis_rate:.1f}%)")
print(f"Non-crisis events: {n_negative:,} ({100-crisis_rate:.1f}%)")
print(f"Imbalance ratio: {n_negative/n_positive:.2f}:1")

# =============================================================================
# CREATE STRATIFIED SPATIAL CV FOLDS
# =============================================================================

print()
print("-" * 80)
print("STEP 5: Creating Stratified Spatial CV Folds")
print("-" * 80)

df_filtered['fold'] = create_stratified_spatial_folds(
    df_filtered,
    n_folds=N_FOLDS,
    n_spatial_clusters=N_SPATIAL_CLUSTERS,
    random_state=RANDOM_STATE
)

# =============================================================================
# HYPERPARAMETER OPTIMIZATION
# =============================================================================

print()
print("-" * 80)
print("STEP 5: Hyperparameter Optimization with GridSearchCV")
print("-" * 80)

X = df_filtered[feature_cols].fillna(0)
y = df_filtered['ipc_future_crisis'].values
fold_array = df_filtered['fold'].values

spatial_cv = SpatialFoldCV(fold_array, n_folds=N_FOLDS)

# Define hyperparameter grid (expanded for fair comparison across feature set sizes)
# Key changes from original:
# - max_depth expanded upward (5-10) to allow complex models to utilize more features
# - min_child_weight expanded downward (1-5) for finer splits with more features
# - Total combinations: 2×3×3×3×2×2×3×2×2 = 2,592 (same order as original ~2k)
param_grid = {
    'n_estimators': [100, 200, 300],                # 2 options
    'max_depth': [5, 7, 10],                   # 3 options (expanded upward from 3,5,7)
    'learning_rate': [0.01, 0.05, 0.1],        # 3 options
    'min_child_weight': [1, 3, 5],             # 3 options (expanded downward from 3,5,10)
    'subsample': [0.7, 0.8],                   # 2 options
    'colsample_bytree': [0.6, 0.8],            # 2 options
    'gamma': [0, 0.5, 1],                      # 3 options
    'reg_alpha': [0, 0.1],                     # 2 options
    'reg_lambda': [1, 2]                       # 2 options
}

total_combinations = 3*3*3*3*2*2*3*2*2
print(f"Hyperparameter grid: {total_combinations} combinations")
print("Running GridSearchCV with Stratified Spatial CV...")

scale_pos_weight = n_negative / n_positive
base_xgb = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    verbosity=0,
    use_label_encoder=False,
    eval_metric='logloss',
    tree_method='hist'  # Fast histogram-based method
)

grid_search = GridSearchCV(
    estimator=base_xgb,
    param_grid=param_grid,
    cv=spatial_cv,
    scoring='roc_auc',
    n_jobs=8,
    verbose=1,
    return_train_score=True
)

grid_search.fit(X, y)

print(f"\nBest parameters: {grid_search.best_params_}")
print(f"Best CV recall: {grid_search.best_score_:.4f}")

best_xgb = grid_search.best_estimator_
best_params = grid_search.best_params_

# =============================================================================
# FINAL CROSS-VALIDATION
# =============================================================================

print()
print("-" * 80)
print("STEP 6: Final Cross-Validation with Best Parameters")
print("-" * 80)

cv_results = []
all_predictions = []
feature_importance_list = []

for fold in range(N_FOLDS):
    print(f"\nProcessing fold {fold}...")

    train_idx = df_filtered['fold'] != fold
    test_idx = df_filtered['fold'] == fold

    X_train = df_filtered.loc[train_idx, feature_cols].fillna(0)
    y_train = df_filtered.loc[train_idx, 'ipc_future_crisis'].values
    X_test = df_filtered.loc[test_idx, feature_cols].fillna(0)
    y_test = df_filtered.loc[test_idx, 'ipc_future_crisis'].values

    train_crisis = int(y_train.sum())
    train_no_crisis = len(y_train) - train_crisis
    test_crisis = int(y_test.sum())

    print(f"  Train: {len(X_train):,} ({train_crisis:,} crisis)")
    print(f"  Test: {len(X_test):,} ({test_crisis:,} crisis)")

    fold_scale_pos_weight = train_no_crisis / train_crisis if train_crisis > 0 else 1.0

    fold_model = XGBClassifier(
        **best_params,
        scale_pos_weight=fold_scale_pos_weight,
        random_state=RANDOM_STATE,
        verbosity=0,
        use_label_encoder=False,
        eval_metric='logloss'
    )

    fold_model.fit(X_train, y_train)

    model_file = OUTPUT_DIR / f"ablation_hmm_fold_{fold}.pkl"
    joblib.dump(fold_model, model_file)

    importance = dict(zip(feature_cols, fold_model.feature_importances_))
    feature_importance_list.append({'fold': fold, **importance})

    y_pred_proba = fold_model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    threshold_results = compute_optimal_thresholds(y_test, y_pred_proba)
    youden_threshold = threshold_results['youden_threshold']
    f1_threshold = threshold_results['f1_threshold']
    high_recall_threshold = threshold_results['high_recall_threshold']

    youden_metrics = compute_metrics_at_threshold(y_test, y_pred_proba, youden_threshold)
    f1_metrics = compute_metrics_at_threshold(y_test, y_pred_proba, f1_threshold)
    high_recall_metrics = compute_metrics_at_threshold(y_test, y_pred_proba, high_recall_threshold)

    brier_score = brier_score_loss(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)

    print(f"  AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print(f"  Youden - P: {youden_metrics['precision']:.3f} | R: {youden_metrics['recall']:.3f} | F1: {youden_metrics['f1']:.3f}")

    # Store CV results (with all metrics matching XGBoost models)
    cv_results.append({
        'fold': fold,
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_crisis_test': test_crisis,
        'auc_roc': auc,
        'pr_auc': pr_auc,
        'brier_score': brier_score,
        'log_loss': logloss,
        'threshold_youden': youden_threshold,
        'threshold_f1': f1_threshold,
        'threshold_high_recall': high_recall_threshold,
        'precision_youden': youden_metrics['precision'],
        'recall_youden': youden_metrics['recall'],
        'f1_youden': youden_metrics['f1'],
        'specificity_youden': youden_metrics['specificity'],
        'precision_f1': f1_metrics['precision'],
        'recall_f1': f1_metrics['recall'],
        'f1_f1': f1_metrics['f1'],
        'precision_high_recall': high_recall_metrics['precision'],
        'recall_high_recall': high_recall_metrics['recall'],
        'f1_high_recall': high_recall_metrics['f1'],
        'tp_high_recall': high_recall_metrics['tp'],
        'fn_high_recall': high_recall_metrics['fn'],
        'fp_high_recall': high_recall_metrics['fp'],
        'tn_high_recall': high_recall_metrics['tn']
    })

    # Store predictions with metadata (matching XGBoost models)
    metadata_cols = [
        'ipc_geographic_unit_full', 'ipc_district', 'ipc_region',
        'ipc_country', 'ipc_country_code',
        'avg_latitude', 'avg_longitude',
        'year_month', 'ipc_period_start', 'ipc_period_end',
        'ipc_value', 'ipc_value_filled', 'ipc_binary_crisis',
        'ar_pred_optimal_filled', 'ar_prob_filled'
    ]

    available_metadata = [col for col in metadata_cols if col in df_filtered.columns]
    fold_predictions = df_filtered.loc[test_idx, available_metadata + ['ipc_future_crisis']].copy()

    fold_predictions['pred_prob'] = y_pred_proba
    fold_predictions['y_pred_youden'] = (y_pred_proba >= youden_threshold).astype(int)
    fold_predictions['y_pred_f1'] = (y_pred_proba >= f1_threshold).astype(int)
    fold_predictions['y_pred_high_recall'] = (y_pred_proba >= high_recall_threshold).astype(int)
    fold_predictions['threshold_youden'] = youden_threshold
    fold_predictions['threshold_f1'] = f1_threshold
    fold_predictions['threshold_high_recall'] = high_recall_threshold

    fold_predictions['confusion_youden'] = assign_confusion_class(
        fold_predictions['ipc_future_crisis'].values,
        fold_predictions['y_pred_youden'].values
    )
    fold_predictions['confusion_high_recall'] = assign_confusion_class(
        fold_predictions['ipc_future_crisis'].values,
        fold_predictions['y_pred_high_recall'].values
    )

    fold_predictions['model'] = 'ablation_ratio_zscore_hmm_location_optimized'
    fold_predictions['filter_variant'] = 'WITH_AR_FILTER'
    fold_predictions['fold'] = fold

    all_predictions.append(fold_predictions)

# =============================================================================
# AGGREGATE RESULTS
# =============================================================================

print()
print("-" * 80)
print("STEP 7: Computing Overall Metrics")
print("-" * 80)

cv_df = pd.DataFrame(cv_results)
overall_auc = cv_df['auc_roc'].mean()
std_auc = cv_df['auc_roc'].std()
overall_pr_auc = cv_df['pr_auc'].mean()
std_pr_auc = cv_df['pr_auc'].std()

print(f"\nOverall AUC-ROC: {overall_auc:.4f} +/- {std_auc:.4f}")
print(f"Overall PR-AUC: {overall_pr_auc:.4f} +/- {std_pr_auc:.4f}")
print(f"\nYouden Threshold Metrics:")
print(f"  Precision: {cv_df['precision_youden'].mean():.4f}")
print(f"  Recall: {cv_df['recall_youden'].mean():.4f}")
print(f"  F1: {cv_df['f1_youden'].mean():.4f}")

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

print()
print("-" * 80)
print("STEP 8: Feature Importance")
print("-" * 80)

importance_df = pd.DataFrame(feature_importance_list).fillna(0)
avg_importance = importance_df.drop('fold', axis=1).mean().sort_values(ascending=False)

print("\nTop features:")
for i, (feat, score) in enumerate(avg_importance.head(10).items(), 1):
    print(f"  {i}. {feat}: {score:.4f}")

importance_file = OUTPUT_DIR / "feature_importance.csv"
avg_importance.to_csv(importance_file, header=['importance'])

# =============================================================================
# SAVE RESULTS
# =============================================================================

print()
print("-" * 80)
print("STEP 9: Saving Results")
print("-" * 80)

predictions_df = pd.concat(all_predictions, ignore_index=True)
predictions_file = OUTPUT_DIR / "ablation_hmm_optimized_predictions.csv"
predictions_df.to_csv(predictions_file, index=False)
print(f"[OK] Predictions: {predictions_file.name}")

cv_file = OUTPUT_DIR / "ablation_hmm_optimized_cv_results.csv"
cv_df.to_csv(cv_file, index=False)
print(f"[OK] CV results: {cv_file.name}")

# =============================================================================
# COMPUTE COUNTRY-LEVEL METRICS
# =============================================================================

print()
print("-" * 80)
print("STEP 10: Computing Country-Level Metrics")
print("-" * 80)

country_metrics = []

for country in predictions_df['ipc_country'].unique():
    country_df = predictions_df[predictions_df['ipc_country'] == country].copy()

    y_true = country_df['ipc_future_crisis'].values
    y_pred_proba = country_df['pred_prob'].values

    country_row = {
        'country': country,
        'n_observations': len(country_df),
        'n_crisis': int(y_true.sum()),
        'crisis_rate': float(y_true.mean()),
        'model': 'ablation_ratio_zscore_hmm_location_optimized',
        'filter_variant': 'WITH_AR_FILTER'
    }

    for threshold_type in ['youden', 'f1', 'high_recall']:
        pred_col = f'y_pred_{threshold_type}'

        if pred_col in country_df.columns:
            y_pred = country_df[pred_col].values

            cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
            if cm.size == 4:
                tn, fp, fn, tp = cm.ravel()
            else:
                tn, fp, fn, tp = 0, 0, 0, 0

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            try:
                auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else np.nan
            except:
                auc = np.nan

            country_row[f'auc_{threshold_type}'] = auc
            country_row[f'precision_{threshold_type}'] = precision
            country_row[f'recall_{threshold_type}'] = recall
            country_row[f'f1_{threshold_type}'] = f1
            country_row[f'tp_{threshold_type}'] = int(tp)
            country_row[f'fn_{threshold_type}'] = int(fn)
            country_row[f'fp_{threshold_type}'] = int(fp)
            country_row[f'tn_{threshold_type}'] = int(tn)

    country_metrics.append(country_row)

country_metrics_df = pd.DataFrame(country_metrics)
country_metrics_file = OUTPUT_DIR / "country_metrics.csv"
country_metrics_df.to_csv(country_metrics_file, index=False)
print(f"[OK] Country metrics: {country_metrics_file.name} ({len(country_metrics_df)} countries)")

# Summary JSON (matching XGBoost model format)
summary = {
    'model': 'ablation_ratio_zscore_hmm_location_optimized',
    'model_type': 'ABLATION - Ratio + Zscore + HMM + Location (NO DMD)',
    'description': 'Ablation Model 3: Ratio + Zscore + HMM + Location (NO DMD) with GridSearchCV',
    'filter': 'WITH_AR_FILTER',
    'enhancement': 'GridSearchCV with Stratified Spatial CV',
    'timestamp': datetime.now().isoformat(),
    'data': {
        'total_observations': int(len(df_filtered)),
        'crisis_events': int(n_positive),
        'non_crisis_events': int(n_negative),
        'crisis_rate': float(crisis_rate / 100),
        'imbalance_ratio': float(n_negative / n_positive),
        'countries': int(df_filtered['ipc_country'].nunique()),
        'districts': int(df_filtered['ipc_geographic_unit_full'].nunique())
    },
    'features': {
        'total': len(feature_cols),
        'ratio': len([f for f in feature_cols if f in ratio_cols]),
        'zscore': len([f for f in feature_cols if f in zscore_cols]),
        'hmm': len([f for f in feature_cols if f in hmm_cols]),
        'dmd': 0,
        'location': len([f for f in feature_cols if f in location_cols]),
        'excluded': 'DMD features'
    },
    'hyperparameter_tuning': {
        'method': 'GridSearchCV with Stratified Spatial CV',
        'scoring': 'recall',
        'best_params': best_params,
        'best_cv_score': float(grid_search.best_score_)
    },
    'cv_performance': {
        'auc_roc_mean': float(overall_auc),
        'auc_roc_std': float(std_auc),
        'pr_auc_mean': float(overall_pr_auc),
        'pr_auc_std': float(std_pr_auc),
        'brier_score_mean': float(cv_df['brier_score'].mean()),
        'log_loss_mean': float(cv_df['log_loss'].mean()),
        'youden': {
            'precision': float(cv_df['precision_youden'].mean()),
            'recall': float(cv_df['recall_youden'].mean()),
            'f1': float(cv_df['f1_youden'].mean()),
            'specificity': float(cv_df['specificity_youden'].mean())
        },
        'high_recall': {
            'precision': float(cv_df['precision_high_recall'].mean()),
            'recall': float(cv_df['recall_high_recall'].mean()),
            'f1': float(cv_df['f1_high_recall'].mean())
        }
    },
    'top_features': {k: float(v) for k, v in avg_importance.head(10).to_dict().items()}
}

summary_file = OUTPUT_DIR / "ablation_hmm_optimized_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"[OK] Summary: {summary_file.name}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print("=" * 80)
print("ABLATION MODEL 3 - OPTIMIZED COMPLETE (NO DMD)")
print("=" * 80)
print(f"\nFeatures: {len(feature_cols)} (Ratio + Zscore + HMM + Location)")
print(f"Excluded: DMD features")
print(f"\nBest Parameters: {best_params}")
print(f"Best CV Recall: {grid_search.best_score_:.4f}")
print(f"\nCV PERFORMANCE:")
print(f"  AUC-ROC: {overall_auc:.4f} +/- {std_auc:.4f}")
print(f"  PR-AUC: {overall_pr_auc:.4f} +/- {std_pr_auc:.4f}")
print(f"  Youden - P: {cv_df['precision_youden'].mean():.3f}, R: {cv_df['recall_youden'].mean():.3f}, F1: {cv_df['f1_youden'].mean():.3f}")
print(f"  High-Recall - P: {cv_df['precision_high_recall'].mean():.3f}, R: {cv_df['recall_high_recall'].mean():.3f}, F1: {cv_df['f1_high_recall'].mean():.3f}")
print(f"\nOUTPUT FILES:")
print(f"  - ablation_hmm_optimized_predictions.csv")
print(f"  - ablation_hmm_optimized_cv_results.csv")
print(f"  - country_metrics.csv")
print(f"  - feature_importance.csv")
print(f"  - ablation_hmm_optimized_summary.json")
print(f"\nOutput Directory: {OUTPUT_DIR}")
print(f"End time: {datetime.now()}")
print("=" * 80)
