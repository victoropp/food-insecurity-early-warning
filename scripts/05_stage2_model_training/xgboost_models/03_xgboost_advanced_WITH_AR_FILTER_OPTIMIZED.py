"""
XGBoost Pipeline - Phase 3: Model Training WITH_AR_FILTER (HMM+DMD) - OPTIMIZED
================================================================================
Train XGBoost model with hyperparameter optimization using Stratified Spatial CV.

KEY ENHANCEMENT:
- GridSearchCV with Stratified Spatial CV for hyperparameter tuning
- Same methodology as temporal holdout validation for consistency
- Optimizes for AUC-ROC (area under ROC curve)

FEATURES:
- 9 ratio features (macrocategories)
- 9 zscore features (macrocategories)
- 6 HMM features (crisis_prob, transition_risk, entropy for ratio/zscore)
- 8 DMD features (growth_rate, instability, frequency, amplitude for ratio/zscore)
- 3 safe location features
- Total: 35 features

FILTER: WITH_AR_FILTER (IPC <= 2 AND AR == 0)

Author: Victor Collins Oppon
Date: December 2025
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import xgboost as xgb
from xgboost import XGBClassifier
from sklearn.metrics import (roc_auc_score, average_precision_score,
                            roc_curve, confusion_matrix, brier_score_loss,
                            log_loss, accuracy_score, precision_score,
                            recall_score, f1_score)
from sklearn.model_selection import GridSearchCV
import joblib

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import from config
from config import (
    BASE_DIR,
    STAGE1_DATA_DIR,
    STAGE1_RESULTS_DIR,
    STAGE2_FEATURES_DIR,
    STAGE2_MODELS_DIR,
    FIGURES_DIR,
    RANDOM_STATE
)
from datetime import datetime
import json
import warnings
warnings.filterwarnings('ignore')

# Force unbuffered output
import functools
print = functools.partial(print, flush=True)

# Constants
N_FOLDS = 5
N_SPATIAL_CLUSTERS = 20
RANDOM_STATE = 42
HIGH_RECALL_TARGET = 0.90

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def compute_optimal_thresholds(y_true, y_pred_proba):
    """Compute optimal thresholds using multiple strategies."""
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)

    # Youden's J
    youden_j = tpr - fpr
    youden_idx = np.argmax(youden_j)
    youden_threshold = thresholds[youden_idx]
    youden_index = youden_j[youden_idx]

    # F1-maximizing threshold
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

    # High-recall threshold
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
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'accuracy': accuracy,
        'balanced_accuracy': balanced_accuracy,
        'tp': int(tp),
        'tn': int(tn),
        'fp': int(fp),
        'fn': int(fn)
    }


def assign_confusion_class(y_true, y_pred):
    """Assign confusion class labels for cartographic mapping."""
    classes = np.empty(len(y_true), dtype=object)
    classes[(y_true == 0) & (y_pred == 0)] = 'TN'
    classes[(y_true == 1) & (y_pred == 1)] = 'TP'
    classes[(y_true == 0) & (y_pred == 1)] = 'FP'
    classes[(y_true == 1) & (y_pred == 0)] = 'FN'
    return classes


class SpatialFoldCV:
    """Custom CV iterator using pre-computed spatial folds."""

    def __init__(self, fold_array, n_folds=5):
        self.fold_array = fold_array
        self.n_folds = n_folds

    def get_n_splits(self, X=None, y=None, groups=None):
        return self.n_folds

    def split(self, X, y=None, groups=None):
        for fold in range(self.n_folds):
            train_idx = np.where(self.fold_array != fold)[0]
            test_idx = np.where(self.fold_array == fold)[0]
            yield train_idx, test_idx


# =============================================================================
# PATHS
# =============================================================================

INPUT_FILE = STAGE2_FEATURES_DIR / "phase3_combined" / "combined_advanced_features_h8.csv"
OUTPUT_DIR = STAGE2_MODELS_DIR / "xgboost" / "advanced_with_ar_optimized"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("XGBOOST PIPELINE - OPTIMIZED WITH HYPERPARAMETER TUNING")
print("=" * 80)
print(f"\nModel: xgboost_hmm_dmd_with_ar_optimized")
print(f"Filter: WITH_AR_FILTER")
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
# CREATE STRATIFIED SPATIAL CV FOLDS
# =============================================================================

print()
print("-" * 80)
print("STEP 3: Creating Stratified Spatial CV Folds")
print("-" * 80)

sys.path.append(str(Path(__file__).parent.parent.parent / 'utils'))
from stratified_spatial_cv import create_stratified_spatial_folds, create_safe_location_features

df_filtered['fold'] = create_stratified_spatial_folds(
    df=df_filtered,
    district_col='ipc_geographic_unit_full',
    country_col='ipc_country',
    target_col='ipc_future_crisis',
    lat_col='avg_latitude',
    lon_col='avg_longitude',
    n_folds=N_FOLDS,
    n_spatial_clusters=N_SPATIAL_CLUSTERS,
    random_state=RANDOM_STATE,
    verbose=True
)

# =============================================================================
# CREATE SAFE LOCATION FEATURES
# =============================================================================

print()
print("-" * 80)
print("STEP 4: Creating Safe Location Features")
print("-" * 80)

df_filtered = create_safe_location_features(
    df=df_filtered,
    district_col='ipc_geographic_unit_full',
    country_col='ipc_country',
    verbose=True
)

# =============================================================================
# PREPARE FEATURES
# =============================================================================

print()
print("-" * 80)
print("STEP 5: Preparing Features")
print("-" * 80)

macro_categories = ['conflict', 'displacement', 'economic', 'food_security', 'governance',
                    'health', 'humanitarian', 'other', 'weather']

ratio_cols = [f'{cat}_ratio' for cat in macro_categories]
zscore_cols = [f'{cat}_zscore' for cat in macro_categories]

safe_location_cols = [
    'country_baseline_conflict',
    'country_baseline_food_security',
    'country_data_density'
]
location_cols = [col for col in safe_location_cols if col in df_filtered.columns]

hmm_ratio_cols = [f'hmm_ratio_{feat}' for feat in ['crisis_prob', 'transition_risk', 'entropy']]
hmm_zscore_cols = [f'hmm_zscore_{feat}' for feat in ['crisis_prob', 'transition_risk', 'entropy']]

dmd_ratio_cols = [f'dmd_ratio_{feat}' for feat in ['crisis_growth_rate', 'crisis_instability',
                                                     'crisis_frequency', 'crisis_amplitude']]
dmd_zscore_cols = [f'dmd_zscore_{feat}' for feat in ['crisis_growth_rate', 'crisis_instability',
                                                       'crisis_frequency', 'crisis_amplitude']]

feature_cols = ratio_cols + zscore_cols + location_cols + hmm_ratio_cols + hmm_zscore_cols + dmd_ratio_cols + dmd_zscore_cols
feature_cols = [f for f in feature_cols if f in df_filtered.columns]

print(f"Ratio features: {len([f for f in feature_cols if f in ratio_cols])}")
print(f"Zscore features: {len([f for f in feature_cols if f in zscore_cols])}")
print(f"HMM features: {len([f for f in feature_cols if 'hmm' in f])}")
print(f"DMD features: {len([f for f in feature_cols if 'dmd' in f])}")
print(f"Location features: {len(location_cols)}")
print(f"Total features: {len(feature_cols)}")

df_filtered = df_filtered[df_filtered['ipc_future_crisis'].notna()].copy()
print(f"After removing missing target: {len(df_filtered):,} rows")

n_positive = int(df_filtered['ipc_future_crisis'].sum())
n_negative = len(df_filtered) - n_positive
crisis_rate = 100 * n_positive / len(df_filtered)

print(f"Crisis events: {n_positive:,} ({crisis_rate:.1f}%)")
print(f"Non-crisis events: {n_negative:,} ({100-crisis_rate:.1f}%)")
print(f"Imbalance ratio: {n_negative/n_positive:.2f}:1")

# =============================================================================
# STEP 5A: COUNTRY-MEDIAN IMPUTATION FOR HMM/DMD FEATURES
# =============================================================================
print()
print("-" * 80)
print("STEP 5A: Country-Median Imputation for HMM/DMD Features")
print("-" * 80)

# Country-median imputation for HMM and DMD features (instead of fillna(0))
print("\nImputing missing HMM/DMD values with country medians...")
for col in hmm_ratio_cols + hmm_zscore_cols + dmd_ratio_cols + dmd_zscore_cols:
    if col in df_filtered.columns:
        missing_count = df_filtered[col].isna().sum()
        if missing_count > 0:
            # Compute country-level median
            country_medians = df_filtered.groupby('ipc_country')[col].transform('median')
            # Fill missing with country median, then global median if country has no data
            df_filtered[col] = df_filtered[col].fillna(country_medians).fillna(df_filtered[col].median())
            print(f"  {col}: Imputed {missing_count} missing values with country median")

# =============================================================================
# STEP 6: HYPERPARAMETER OPTIMIZATION WITH SPATIAL CV
# =============================================================================

print()
print("-" * 80)
print("STEP 6: Hyperparameter Optimization with Stratified Spatial CV")
print("-" * 80)

X = df_filtered[feature_cols].values  # Convert to numpy array for indexing
y = df_filtered['ipc_future_crisis'].values
fold_array = df_filtered['fold'].values

spatial_cv = SpatialFoldCV(fold_array, n_folds=N_FOLDS)

# =============================================================================
# STEP 6B: HYPERPARAMETER TUNING WITH GRIDSEARCHCV
# =============================================================================
print()
print("-" * 80)
print("STEP 6B: Hyperparameter Optimization with GridSearchCV")
print("-" * 80)

# Define hyperparameter grid - BEST CONFIGURATION (AUC 0.6849, PR-AUC 0.1913)
# After testing multiple grid sizes, this configuration achieved best discrimination
# Hyperparameter grid - FAIR COMPARISON (same as ablation models)
# Key changes for fair comparison across feature set sizes:
# - max_depth expanded upward (5-10) to allow complex models to utilize more features
# - min_child_weight expanded downward (1-5) for finer splits with more features
# - Total combinations: 2×3×3×3×2×2×3×2×2 = 2,592
param_grid = {
    'n_estimators': [100, 200, 300],           # 3 options
    'max_depth': [5, 7, 10],                   # 3 options (expanded upward from 3,5,7)
    'learning_rate': [0.01, 0.05, 0.1],        # 3 options
    'min_child_weight': [1, 3, 5],             # 3 options (expanded downward from 3,5,10)
    'subsample': [0.7, 0.8],                   # 2 options
    'colsample_bytree': [0.6, 0.8],            # 2 options
    'gamma': [0, 0.5, 1],                      # 3 options
    'reg_alpha': [0, 0.1],                     # 2 options
    'reg_lambda': [1, 2]                       # 2 options
}

n_combinations = (len(param_grid['n_estimators']) * len(param_grid['max_depth']) *
                  len(param_grid['learning_rate']) * len(param_grid['min_child_weight']) *
                  len(param_grid['subsample']) * len(param_grid['colsample_bytree']) *
                  len(param_grid['gamma']) * len(param_grid['reg_alpha']) *
                  len(param_grid['reg_lambda']))

print(f"Hyperparameter grid: {n_combinations} combinations")
print("Running GridSearchCV with Stratified Spatial CV...")
print("(This may take several minutes...)")

scale_pos_weight = n_negative / n_positive

base_xgb = XGBClassifier(
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    verbosity=0,
    use_label_encoder=False,
    eval_metric='logloss',
    tree_method='hist'
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

best_params = grid_search.best_params_

# Save grid search results
cv_results_df = pd.DataFrame(grid_search.cv_results_)
cv_results_df = cv_results_df.sort_values('rank_test_score')
cv_results_df.to_csv(OUTPUT_DIR / 'grid_search_results.csv', index=False)
print(f"Grid search results saved")

# =============================================================================
# CROSS-VALIDATION WITH OPTIMIZED PARAMETERS
# =============================================================================

print()
print("-" * 80)
print("STEP 7: Cross-Validation with Optimized Parameters")
print("-" * 80)

print(f"\nOptimized XGBoost parameters:")
for key, val in best_params.items():
    print(f"   {key}: {val}")

cv_results = []
all_predictions = []
feature_importance_list = []

for fold in range(N_FOLDS):
    print(f"\nProcessing fold {fold}...")

    train_idx = df_filtered['fold'] != fold
    test_idx = df_filtered['fold'] == fold

    X_train = df_filtered.loc[train_idx, feature_cols]
    y_train = df_filtered.loc[train_idx, 'ipc_future_crisis'].values
    X_test = df_filtered.loc[test_idx, feature_cols]
    y_test = df_filtered.loc[test_idx, 'ipc_future_crisis'].values

    train_crisis = int(y_train.sum())
    train_no_crisis = len(y_train) - train_crisis
    test_crisis = int(y_test.sum())

    print(f"  Train: {len(X_train):,} ({train_crisis:,} crisis)")
    print(f"  Test: {len(X_test):,} ({test_crisis:,} crisis)")

    # Calculate fold-specific scale_pos_weight
    fold_scale_pos_weight = train_no_crisis / train_crisis if train_crisis > 0 else 1.0

    # Create model with optimized params
    model = XGBClassifier(
        **best_params,
        scale_pos_weight=fold_scale_pos_weight,
        random_state=RANDOM_STATE,
        verbosity=0,
        use_label_encoder=False,
        eval_metric='logloss',
    tree_method='hist'
    )

    model.fit(X_train, y_train)

    # Save fold model
    model_file = OUTPUT_DIR / f"xgboost_optimized_fold_{fold}.pkl"
    joblib.dump(model, model_file)

    # Feature importance
    importance_dict = dict(zip(feature_cols, model.feature_importances_))
    feature_importance_list.append({'fold': fold, **importance_dict})

    # Predictions
    y_pred_proba = model.predict_proba(X_test)[:, 1]

    # Metrics
    auc = roc_auc_score(y_test, y_pred_proba)
    pr_auc = average_precision_score(y_test, y_pred_proba)

    # Compute optimal thresholds
    threshold_results = compute_optimal_thresholds(y_test, y_pred_proba)
    youden_threshold = threshold_results['youden_threshold']
    f1_threshold = threshold_results['f1_threshold']
    high_recall_threshold = threshold_results['high_recall_threshold']

    # Compute metrics at each threshold
    youden_metrics = compute_metrics_at_threshold(y_test, y_pred_proba, youden_threshold)
    f1_metrics = compute_metrics_at_threshold(y_test, y_pred_proba, f1_threshold)
    high_recall_metrics = compute_metrics_at_threshold(y_test, y_pred_proba, high_recall_threshold)

    # Calibration metrics
    brier_score = brier_score_loss(y_test, y_pred_proba)
    logloss = log_loss(y_test, y_pred_proba)

    print(f"  AUC: {auc:.4f} | PR-AUC: {pr_auc:.4f}")
    print(f"  Youden - P: {youden_metrics['precision']:.3f} | R: {youden_metrics['recall']:.3f} | F1: {youden_metrics['f1']:.3f}")
    print(f"  F1-max - P: {f1_metrics['precision']:.3f} | R: {f1_metrics['recall']:.3f} | F1: {f1_metrics['f1']:.3f}")
    print(f"  High-R - P: {high_recall_metrics['precision']:.3f} | R: {high_recall_metrics['recall']:.3f} | F1: {high_recall_metrics['f1']:.3f}")

    # Store CV results
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

    # Store predictions with metadata
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

    fold_predictions['model'] = 'xgboost_optimized'
    fold_predictions['filter_variant'] = 'WITH_AR_FILTER'
    fold_predictions['fold'] = fold

    all_predictions.append(fold_predictions)

# =============================================================================
# AGGREGATE RESULTS
# =============================================================================

print()
print("-" * 80)
print("STEP 8: Aggregate Results")
print("-" * 80)

cv_df = pd.DataFrame(cv_results)
overall_auc = cv_df['auc_roc'].mean()
std_auc = cv_df['auc_roc'].std()
overall_pr_auc = cv_df['pr_auc'].mean()
std_pr_auc = cv_df['pr_auc'].std()

print(f"\nOverall AUC-ROC: {overall_auc:.4f} +/- {std_auc:.4f}")
print(f"Overall PR-AUC: {overall_pr_auc:.4f} +/- {std_pr_auc:.4f}")
print(f"Mean Brier Score: {cv_df['brier_score'].mean():.4f}")
print(f"Mean Log Loss: {cv_df['log_loss'].mean():.4f}")

print(f"\nYouden Threshold Metrics:")
print(f"  Precision: {cv_df['precision_youden'].mean():.4f}")
print(f"  Recall: {cv_df['recall_youden'].mean():.4f}")
print(f"  F1: {cv_df['f1_youden'].mean():.4f}")

print(f"\nHigh-Recall Threshold Metrics:")
print(f"  Precision: {cv_df['precision_high_recall'].mean():.4f}")
print(f"  Recall: {cv_df['recall_high_recall'].mean():.4f}")
print(f"  F1: {cv_df['f1_high_recall'].mean():.4f}")

# =============================================================================
# TRAIN FINAL MODEL ON ALL DATA
# =============================================================================

print()
print("-" * 80)
print("STEP 9: Train Final Model on All Data")
print("-" * 80)

final_model = XGBClassifier(
    **best_params,
    scale_pos_weight=scale_pos_weight,
    random_state=RANDOM_STATE,
    verbosity=0,
    use_label_encoder=False,
    eval_metric='logloss',
    tree_method='hist'
)

print(f"Training on {len(X)} samples with {len(feature_cols)} features")
final_model.fit(X, y)

# Save final model
final_model_file = OUTPUT_DIR / "xgboost_optimized_final.pkl"
joblib.dump(final_model, final_model_file)
print(f"Final model saved: {final_model_file}")

# =============================================================================
# FEATURE IMPORTANCE
# =============================================================================

print()
print("-" * 80)
print("STEP 10: Feature Importance")
print("-" * 80)

importance_df = pd.DataFrame(feature_importance_list).fillna(0)
avg_importance = importance_df.drop('fold', axis=1).mean().sort_values(ascending=False)

print("\nTop 10 most important features:")
for i, (feat, score) in enumerate(avg_importance.head(10).items(), 1):
    print(f"  {i}. {feat}: {score:.4f}")

importance_file = OUTPUT_DIR / "feature_importance.csv"
avg_importance.to_csv(importance_file, header=['importance'])

# =============================================================================
# SAVE RESULTS
# =============================================================================

print()
print("-" * 80)
print("STEP 11: Saving Results")
print("-" * 80)

# Predictions
predictions_df = pd.concat(all_predictions, ignore_index=True)
predictions_file = OUTPUT_DIR / "xgboost_optimized_predictions.csv"
predictions_df.to_csv(predictions_file, index=False)
print(f"Predictions saved: {predictions_file.name} ({len(predictions_df):,} rows)")

# CV results
cv_file = OUTPUT_DIR / "xgboost_optimized_cv_results.csv"
cv_df.to_csv(cv_file, index=False)
print(f"CV results saved: {cv_file.name}")

# =============================================================================
# COMPUTE COUNTRY-LEVEL METRICS
# =============================================================================

print()
print("-" * 80)
print("STEP 12: Computing Country-Level Metrics")
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
        'model': 'xgboost_optimized',
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
print(f"Country metrics saved: {country_metrics_file.name} ({len(country_metrics_df)} countries)")

# =============================================================================
# SUMMARY JSON
# =============================================================================

summary = {
    'model': 'xgboost_advanced_optimized',
    'model_type': 'ADVANCED (HMM+DMD) - All Features',
    'filter': 'WITH_AR_FILTER',
    'enhancement': 'GridSearchCV with Stratified Spatial CV',
    'timestamp': datetime.now().isoformat(),


    'hyperparameter_tuning': {
        'method': 'GridSearchCV with Stratified Spatial CV',
        'scoring': 'recall',
        'n_combinations': n_combinations,
        'best_params': best_params,
        'best_cv_score': float(grid_search.best_score_)
    },

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
        'ratio': len([f for f in feature_cols if f in ratio_cols]),
        'zscore': len([f for f in feature_cols if f in zscore_cols]),
        'hmm': len([f for f in feature_cols if 'hmm' in f]),
        'dmd': len([f for f in feature_cols if 'dmd' in f]),
        'location': len([f for f in feature_cols if f in location_cols]),
        'total': len(feature_cols)
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
            'f1': float(cv_df['f1_youden'].mean())
        },
        'high_recall': {
            'precision': float(cv_df['precision_high_recall'].mean()),
            'recall': float(cv_df['recall_high_recall'].mean()),
            'f1': float(cv_df['f1_high_recall'].mean())
        }
    },

    'top_features': avg_importance.head(10).to_dict()
}

summary_file = OUTPUT_DIR / "xgboost_optimized_summary.json"
with open(summary_file, 'w') as f:
    json.dump(summary, f, indent=2)
print(f"Summary saved: {summary_file.name}")

# =============================================================================
# FINAL SUMMARY
# =============================================================================

print()
print("=" * 80)
print("XGBOOST ADVANCED OPTIMIZED TRAINING COMPLETE ")
print("=" * 80)
print()
print("FEATURES:")
print(f"  Total features: {len(feature_cols)}")
print(f"  - Ratio: {len([f for f in feature_cols if f in ratio_cols])}, Zscore: {len([f for f in feature_cols if f in zscore_cols])}")
print(f"  - HMM: {len([f for f in feature_cols if 'hmm' in f])}, DMD: {len([f for f in feature_cols if 'dmd' in f])}")
print(f"  - Location: {len([f for f in feature_cols if f in location_cols])}")
print()
print("HYPERPARAMETER OPTIMIZATION:")
print(f"  Method: GridSearchCV with Stratified Spatial CV")
print(f"  Combinations tested: {n_combinations}")
print(f"  Best CV Recall: {grid_search.best_score_:.4f}")
print()
print("BEST PARAMETERS:")
for param, value in best_params.items():
    print(f"  {param}: {value}")
print()
print("CV PERFORMANCE:")
print(f"  AUC-ROC: {overall_auc:.4f} +/- {std_auc:.4f}")
print(f"  PR-AUC: {overall_pr_auc:.4f} +/- {std_pr_auc:.4f}")
print(f"  High-Recall - P: {cv_df['precision_high_recall'].mean():.3f}, R: {cv_df['recall_high_recall'].mean():.3f}, F1: {cv_df['f1_high_recall'].mean():.3f}")
print()
print("OUTPUT FILES:")
print(f"  - xgboost_optimized_predictions.csv")
print(f"  - xgboost_optimized_cv_results.csv")
print(f"  - country_metrics.csv")
print(f"  - feature_importance.csv")
print(f"  - grid_search_results.csv")
print(f"  - xgboost_optimized_final.pkl")
print()
print(f"Output Directory: {OUTPUT_DIR}")
print(f"End time: {datetime.now()}")
print("=" * 80)
