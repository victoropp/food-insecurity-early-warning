"""
Stage 1: Logistic Regression Baseline with Spatial CV - DISTRICT LEVEL
Predicts binary crisis onset using spatio-temporal autoregressive features (Lt, Ls)
Saves district-level predictions for cartographic analysis.

KEY FEATURES:
1. Uses ipc_geographic_unit_full as unique district identifier
2. Spatial CV clusters based on district coordinates
3. Predictions saved at district level
4. THRESHOLD TUNING for balanced precision/recall

Author: Updated with threshold optimization
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
from config import BASE_DIR
    confusion_matrix, roc_auc_score,
    average_precision_score, accuracy_score,
    precision_recall_curve, roc_curve
)
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(str(BASE_DIR.parent.parent.parent))

# District pipeline I/O (district_level subfolders)
DISTRICT_DATA_DIR = BASE_DIR / 'data' / 'district_level'
DISTRICT_RESULTS_DIR = BASE_DIR / 'results' / 'district_level'
DISTRICT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

INPUT_FILE = DISTRICT_DATA_DIR / 'stage1_features.parquet'
OUTPUT_DIR = DISTRICT_RESULTS_DIR / 'stage1_baseline'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Parameters
HORIZONS = [4, 8, 12]  # Prediction horizons in months
N_FOLDS = 5  # Number of spatial CV folds
RANDOM_STATE = 42
THRESHOLD_RANGE = np.arange(0.1, 0.95, 0.05)  # Thresholds to evaluate

# Figures directory
FIGURES_DIR = BASE_DIR / 'figures' / 'district_level' / 'stage1_threshold_tuning'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)


def find_optimal_thresholds(y_true, y_proba, min_metric=0.6):
    """
    Find optimal classification thresholds using multiple strategies.

    Returns dict with:
    - f1_optimal: threshold maximizing F1 score
    - gmean_optimal: threshold maximizing geometric mean of precision & recall
    - youden_optimal: threshold maximizing Youden's J (sensitivity + specificity - 1)
    - balanced: threshold where precision = recall
    - balanced_constrained: threshold where P=R with minimum constraint (default 0.6)
    """
    precision, recall, pr_thresholds = precision_recall_curve(y_true, y_proba)
    fpr, tpr, roc_thresholds = roc_curve(y_true, y_proba)

    # F1-optimal threshold
    f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
    best_f1_idx = np.argmax(f1_scores)
    f1_threshold = pr_thresholds[best_f1_idx]

    # Geometric mean optimal
    gmean = np.sqrt(precision[:-1] * recall[:-1])
    best_gmean_idx = np.argmax(gmean)
    gmean_threshold = pr_thresholds[best_gmean_idx]

    # Youden's J statistic (from ROC)
    youdens_j = tpr - fpr
    best_youden_idx = np.argmax(youdens_j)
    youden_threshold = roc_thresholds[best_youden_idx]

    # Balanced threshold (where precision = recall, unconstrained)
    diff = np.abs(precision[:-1] - recall[:-1])
    balanced_idx = np.argmin(diff)
    balanced_threshold = pr_thresholds[balanced_idx]

    # Balanced threshold with minimum constraint (P=R >= min_metric)
    # Find indices where both precision and recall >= min_metric
    valid_mask = (precision[:-1] >= min_metric) & (recall[:-1] >= min_metric)

    if valid_mask.any():
        # Among valid points, find where P and R are closest (P = R)
        valid_diff = np.where(valid_mask, diff, np.inf)
        constrained_idx = np.argmin(valid_diff)
        constrained_threshold = pr_thresholds[constrained_idx]
        constrained_precision = precision[constrained_idx]
        constrained_recall = recall[constrained_idx]
        constraint_met = True
    else:
        # If no threshold meets the constraint, find the best we can do
        # Find the threshold that maximizes min(precision, recall)
        min_pr = np.minimum(precision[:-1], recall[:-1])
        best_min_idx = np.argmax(min_pr)
        constrained_threshold = pr_thresholds[best_min_idx]
        constrained_precision = precision[best_min_idx]
        constrained_recall = recall[best_min_idx]
        constraint_met = False

    return {
        'f1_optimal': {
            'threshold': float(f1_threshold),
            'f1': float(f1_scores[best_f1_idx]),
            'precision': float(precision[best_f1_idx]),
            'recall': float(recall[best_f1_idx])
        },
        'gmean_optimal': {
            'threshold': float(gmean_threshold),
            'gmean': float(gmean[best_gmean_idx]),
            'precision': float(precision[best_gmean_idx]),
            'recall': float(recall[best_gmean_idx])
        },
        'youden_optimal': {
            'threshold': float(youden_threshold),
            'youden_j': float(youdens_j[best_youden_idx]),
            'tpr': float(tpr[best_youden_idx]),
            'fpr': float(fpr[best_youden_idx])
        },
        'balanced': {
            'threshold': float(balanced_threshold),
            'precision': float(precision[balanced_idx]),
            'recall': float(recall[balanced_idx])
        },
        'balanced_constrained': {
            'threshold': float(constrained_threshold),
            'precision': float(constrained_precision),
            'recall': float(constrained_recall),
            'min_constraint': min_metric,
            'constraint_met': constraint_met
        },
        'curves': {
            'precision': precision,
            'recall': recall,
            'pr_thresholds': pr_thresholds,
            'fpr': fpr,
            'tpr': tpr,
            'roc_thresholds': roc_thresholds
        }
    }


def evaluate_at_threshold(y_true, y_proba, threshold):
    """Evaluate predictions at a specific threshold."""
    y_pred = (y_proba >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    gmean = np.sqrt(precision * recall)

    return {
        'threshold': threshold,
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision,
        'recall': recall,
        'specificity': specificity,
        'f1': f1,
        'gmean': gmean,
        'true_positives': int(tp),
        'false_positives': int(fp),
        'true_negatives': int(tn),
        'false_negatives': int(fn),
        'n_predicted_positive': int(tp + fp),
        'n_predicted_negative': int(tn + fn)
    }


def plot_threshold_analysis(y_true, y_proba, optimal_thresholds, horizon, save_path):
    """Create comprehensive threshold analysis visualization."""
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    precision = optimal_thresholds['curves']['precision']
    recall = optimal_thresholds['curves']['recall']
    pr_thresholds = optimal_thresholds['curves']['pr_thresholds']
    fpr = optimal_thresholds['curves']['fpr']
    tpr = optimal_thresholds['curves']['tpr']

    # 1. Precision-Recall Curve
    ax1 = axes[0, 0]
    ax1.plot(recall, precision, 'b-', linewidth=2, label='PR Curve')

    # Mark optimal thresholds
    f1_opt = optimal_thresholds['f1_optimal']
    ax1.scatter([f1_opt['recall']], [f1_opt['precision']],
                c='orange', s=100, marker='o', zorder=5,
                label=f"F1 Optimal (t={f1_opt['threshold']:.3f})")

    balanced = optimal_thresholds['balanced']
    ax1.scatter([balanced['recall']], [balanced['precision']],
                c='blue', s=100, marker='s', zorder=5,
                label=f"Balanced (t={balanced['threshold']:.3f})")

    # Balanced constrained (P=R >= 0.6) - PRIMARY
    bal_const = optimal_thresholds['balanced_constrained']
    marker_label = f"P=R>={bal_const['min_constraint']:.1f} (t={bal_const['threshold']:.3f})"
    if not bal_const['constraint_met']:
        marker_label += " [!]"
    ax1.scatter([bal_const['recall']], [bal_const['precision']],
                c='red', s=200, marker='*', zorder=6,
                label=marker_label)

    # Draw horizontal/vertical lines at 0.6 constraint
    ax1.axhline(y=0.6, color='gray', linestyle='--', alpha=0.5, linewidth=1)
    ax1.axvline(x=0.6, color='gray', linestyle='--', alpha=0.5, linewidth=1)

    ax1.set_xlabel('Recall', fontsize=12)
    ax1.set_ylabel('Precision', fontsize=12)
    ax1.set_title(f'Precision-Recall Curve (h={horizon})', fontsize=14)
    ax1.legend(loc='lower left')
    ax1.grid(True, alpha=0.3)
    ax1.set_xlim([0, 1.02])
    ax1.set_ylim([0, 1.02])

    # 2. ROC Curve
    ax2 = axes[0, 1]
    ax2.plot(fpr, tpr, 'b-', linewidth=2, label='ROC Curve')
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5, label='Random')

    youden = optimal_thresholds['youden_optimal']
    ax2.scatter([youden['fpr']], [youden['tpr']],
                c='purple', s=150, marker='*', zorder=5,
                label=f"Youden (t={youden['threshold']:.3f})")

    auc_roc = roc_auc_score(y_true, y_proba)
    ax2.set_xlabel('False Positive Rate', fontsize=12)
    ax2.set_ylabel('True Positive Rate', fontsize=12)
    ax2.set_title(f'ROC Curve (AUC={auc_roc:.3f})', fontsize=14)
    ax2.legend(loc='lower right')
    ax2.grid(True, alpha=0.3)

    # 3. Metrics vs Threshold
    ax3 = axes[1, 0]
    thresholds = np.arange(0.05, 0.95, 0.01)
    metrics_at_thresh = [evaluate_at_threshold(y_true, y_proba, t) for t in thresholds]

    prec_vals = [m['precision'] for m in metrics_at_thresh]
    rec_vals = [m['recall'] for m in metrics_at_thresh]
    f1_vals = [m['f1'] for m in metrics_at_thresh]
    spec_vals = [m['specificity'] for m in metrics_at_thresh]

    ax3.plot(thresholds, prec_vals, 'b-', linewidth=2, label='Precision')
    ax3.plot(thresholds, rec_vals, 'r-', linewidth=2, label='Recall')
    ax3.plot(thresholds, f1_vals, 'g-', linewidth=2, label='F1 Score')
    ax3.plot(thresholds, spec_vals, 'm--', linewidth=1.5, label='Specificity')

    # Mark optimal threshold (balanced constrained)
    ax3.axvline(x=bal_const['threshold'], color='red', linestyle=':', linewidth=2, alpha=0.8,
                label=f"Balanced P=R (t={bal_const['threshold']:.3f})")
    ax3.axvline(x=0.5, color='gray', linestyle='--', alpha=0.5, label='Default (0.5)')
    ax3.axhline(y=0.6, color='gray', linestyle='--', alpha=0.3)  # Min constraint line

    ax3.set_xlabel('Threshold', fontsize=12)
    ax3.set_ylabel('Score', fontsize=12)
    ax3.set_title('Metrics vs Classification Threshold', fontsize=14)
    ax3.legend(loc='lower left')
    ax3.grid(True, alpha=0.3)
    ax3.set_xlim([0, 1])
    ax3.set_ylim([0, 1.02])

    # 4. Confusion Matrix Comparison
    ax4 = axes[1, 1]

    # Compare default (0.5) vs balanced constrained threshold
    metrics_default = evaluate_at_threshold(y_true, y_proba, 0.5)
    metrics_optimal = evaluate_at_threshold(y_true, y_proba, bal_const['threshold'])

    labels = ['Threshold', 'Precision', 'Recall', 'F1', 'FP (x1000)', 'FN (x1000)']
    default_vals = [0.5, metrics_default['precision'], metrics_default['recall'],
                    metrics_default['f1'], metrics_default['false_positives']/1000,
                    metrics_default['false_negatives']/1000]
    optimal_vals = [bal_const['threshold'], metrics_optimal['precision'], metrics_optimal['recall'],
                    metrics_optimal['f1'], metrics_optimal['false_positives']/1000,
                    metrics_optimal['false_negatives']/1000]

    x = np.arange(len(labels))
    width = 0.35

    bars1 = ax4.bar(x - width/2, default_vals, width, label='Default (0.5)', color='lightcoral')
    bars2 = ax4.bar(x + width/2, optimal_vals, width, label='Balanced P=R', color='lightgreen')

    ax4.set_ylabel('Value', fontsize=12)
    ax4.set_title('Default vs Optimal Threshold Comparison', fontsize=14)
    ax4.set_xticks(x)
    ax4.set_xticklabels(labels, rotation=15, ha='right')
    ax4.legend()
    ax4.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax4.annotate(f'{height:.2f}',
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha='center', va='bottom', fontsize=8)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()

    return fig


def create_spatial_folds(df, n_folds=5, random_state=42):
    """
    Create spatially separated folds using K-means clustering on DISTRICT coordinates.
    Prevents spatial autocorrelation in train/test splits.

    KEY CHANGE: Uses ipc_geographic_unit_full as district identifier
    """
    print(f"\nCreating {n_folds} spatial folds at DISTRICT level...")

    # Get unique districts with their coordinates
    districts = df[['ipc_geographic_unit_full', 'ipc_district', 'ipc_country',
                    'avg_latitude', 'avg_longitude']].drop_duplicates()
    districts = districts.dropna(subset=['avg_latitude', 'avg_longitude'])

    print(f"   Unique districts with coordinates: {len(districts)}")

    # Cluster districts into spatially separated groups
    coords = districts[['avg_latitude', 'avg_longitude']].values
    kmeans = KMeans(n_clusters=n_folds, random_state=random_state, n_init=10)
    districts['fold'] = kmeans.fit_predict(coords)

    # Map folds back to full dataset
    fold_map = dict(zip(districts['ipc_geographic_unit_full'], districts['fold']))
    df['fold'] = df['ipc_geographic_unit_full'].map(fold_map)

    # Print fold statistics
    print(f"   Fold distribution:")
    for fold in range(n_folds):
        n_districts = (districts['fold'] == fold).sum()
        n_obs = (df['fold'] == fold).sum()
        print(f"      Fold {fold}: {n_districts} districts, {n_obs:,} observations")

    return df


def train_and_predict_fold(X_train, y_train, X_test, fold_idx, horizon):
    """Train logistic regression and predict on test fold"""
    model = LogisticRegression(
        penalty='l2',
        C=1.0,
        solver='lbfgs',
        max_iter=1000,
        class_weight='balanced',
        random_state=RANDOM_STATE
    )

    model.fit(X_train, y_train)

    y_pred_proba = model.predict_proba(X_test)[:, 1]
    y_pred = (y_pred_proba >= 0.5).astype(int)

    coef_dict = {
        'intercept': model.intercept_[0],
        'coef_Lt': model.coef_[0][0],
        'coef_Ls': model.coef_[0][1]
    }

    return y_pred_proba, y_pred, coef_dict, model


def evaluate_predictions(y_true, y_pred, y_pred_proba, horizon, fold_idx=None):
    """Compute comprehensive evaluation metrics"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    auc_roc = roc_auc_score(y_true, y_pred_proba)
    auc_pr = average_precision_score(y_true, y_pred_proba)

    metrics = {
        'horizon': horizon,
        'fold': fold_idx,
        'n_samples': len(y_true),
        'n_crisis': int(y_true.sum()),
        'n_no_crisis': int((1 - y_true).sum()),
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'auc_roc': auc_roc,
        'auc_pr': auc_pr,
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn),
        'true_positives': int(tp)
    }

    return metrics


def process_horizon(df, horizon):
    """Process one prediction horizon with spatial CV"""
    print(f"\n{'='*80}")
    print(f"Processing Horizon h={horizon} months - DISTRICT LEVEL")
    print(f"{'='*80}")

    target_col = f'y_h{horizon}'

    # Filter to complete cases
    required_cols = ['ipc_id', 'ipc_country', 'ipc_district', 'ipc_geographic_unit_full',
                     'ipc_period_start', 'ipc_period_end', 'ipc_value',
                     'avg_latitude', 'avg_longitude',
                     'Lt', 'Ls', target_col, 'fold']

    df_complete = df[required_cols].copy()
    df_complete = df_complete.dropna(subset=['Lt', 'Ls', target_col])

    print(f"\nDataset info:")
    print(f"   Total observations: {len(df_complete):,}")
    print(f"   Unique districts: {df_complete['ipc_geographic_unit_full'].nunique():,}")
    print(f"   Crisis cases (y=1): {df_complete[target_col].sum():,} ({df_complete[target_col].mean()*100:.1f}%)")

    # Prepare features
    feature_cols = ['Lt', 'Ls']
    X = df_complete[feature_cols].values
    y = df_complete[target_col].values

    predictions_list = []
    metrics_list = []
    coefficients_list = []

    # Spatial Cross-Validation
    print(f"\nRunning {N_FOLDS}-fold Spatial Cross-Validation...")

    for fold in range(N_FOLDS):
        print(f"\n--- Fold {fold+1}/{N_FOLDS} ---")

        train_mask = df_complete['fold'] != fold
        test_mask = df_complete['fold'] == fold

        X_train, X_test = X[train_mask], X[test_mask]
        y_train, y_test = y[train_mask], y[test_mask]

        print(f"   Training: {len(X_train):,} samples (crisis: {y_train.sum():,})")
        print(f"   Test: {len(X_test):,} samples (crisis: {y_test.sum():,})")

        # Train and predict
        y_pred_proba, y_pred, coef_dict, model = train_and_predict_fold(
            X_train, y_train, X_test, fold, horizon
        )

        # Store coefficients
        coef_dict['fold'] = fold
        coef_dict['horizon'] = horizon
        coefficients_list.append(coef_dict)

        # Evaluate
        fold_metrics = evaluate_predictions(y_test, y_pred, y_pred_proba, horizon, fold)
        metrics_list.append(fold_metrics)

        print(f"   Accuracy: {fold_metrics['accuracy']:.3f}")
        print(f"   Precision: {fold_metrics['precision']:.3f}")
        print(f"   Recall: {fold_metrics['recall']:.3f}")
        print(f"   AUC-ROC: {fold_metrics['auc_roc']:.3f}")

        # Store predictions
        test_df = df_complete[test_mask].copy()
        test_df['y_true'] = y_test
        test_df['y_pred_proba'] = y_pred_proba
        test_df['y_pred'] = y_pred
        test_df['fold'] = fold
        test_df['horizon'] = horizon
        test_df['correct'] = (y_test == y_pred).astype(int)
        test_df['ar_failure'] = ((y_pred == 0) & (y_test == 1)).astype(int)
        test_df['false_alarm'] = ((y_pred == 1) & (y_test == 0)).astype(int)

        predictions_list.append(test_df)

    # Combine predictions
    all_predictions = pd.concat(predictions_list, ignore_index=True)

    # Average predictions per district-period
    avg_predictions = all_predictions.groupby('ipc_id').agg({
        'ipc_country': 'first',
        'ipc_district': 'first',
        'ipc_geographic_unit_full': 'first',
        'ipc_period_start': 'first',
        'ipc_period_end': 'first',
        'ipc_value': 'first',
        'avg_latitude': 'first',
        'avg_longitude': 'first',
        'Lt': 'first',
        'Ls': 'first',
        'y_true': 'first',
        'y_pred_proba': 'mean',
        'y_pred': lambda x: (x.mean() >= 0.5).astype(int),
        'fold': 'first',
        'horizon': 'first',
        'correct': lambda x: (x.mean() >= 0.5).astype(int),
        'ar_failure': lambda x: (x.mean() >= 0.5).astype(int),
        'false_alarm': lambda x: (x.mean() >= 0.5).astype(int)
    }).reset_index()

    # =========================================================================
    # THRESHOLD TUNING
    # =========================================================================
    print(f"\n{'='*80}")
    print(f"THRESHOLD TUNING (h={horizon} months)")
    print(f"{'='*80}")

    y_true = avg_predictions['y_true'].values
    y_proba = avg_predictions['y_pred_proba'].values

    # Find optimal thresholds
    optimal_thresholds = find_optimal_thresholds(y_true, y_proba)

    print(f"\n--- Optimal Threshold Strategies ---")
    print(f"\n1. F1-Optimal Threshold: {optimal_thresholds['f1_optimal']['threshold']:.3f}")
    print(f"   Precision: {optimal_thresholds['f1_optimal']['precision']:.3f}")
    print(f"   Recall: {optimal_thresholds['f1_optimal']['recall']:.3f}")
    print(f"   F1: {optimal_thresholds['f1_optimal']['f1']:.3f}")

    print(f"\n2. Geometric Mean Threshold: {optimal_thresholds['gmean_optimal']['threshold']:.3f}")
    print(f"   Precision: {optimal_thresholds['gmean_optimal']['precision']:.3f}")
    print(f"   Recall: {optimal_thresholds['gmean_optimal']['recall']:.3f}")

    print(f"\n3. Youden's J Threshold: {optimal_thresholds['youden_optimal']['threshold']:.3f}")
    print(f"   TPR: {optimal_thresholds['youden_optimal']['tpr']:.3f}")
    print(f"   FPR: {optimal_thresholds['youden_optimal']['fpr']:.3f}")

    print(f"\n4. Balanced (P=R) Threshold: {optimal_thresholds['balanced']['threshold']:.3f}")
    print(f"   Precision: {optimal_thresholds['balanced']['precision']:.3f}")
    print(f"   Recall: {optimal_thresholds['balanced']['recall']:.3f}")

    bal_const = optimal_thresholds['balanced_constrained']
    constraint_status = "MET" if bal_const['constraint_met'] else "NOT MET"
    print(f"\n5. BALANCED CONSTRAINED (P=R >= {bal_const['min_constraint']:.1f}) - [{constraint_status}]")
    print(f"   Threshold: {bal_const['threshold']:.3f}")
    print(f"   Precision: {bal_const['precision']:.3f}")
    print(f"   Recall: {bal_const['recall']:.3f}")

    # Evaluate at multiple thresholds
    print(f"\n--- Performance at Various Thresholds ---")
    threshold_results = []
    for thresh in THRESHOLD_RANGE:
        metrics = evaluate_at_threshold(y_true, y_proba, thresh)
        metrics['horizon'] = horizon
        threshold_results.append(metrics)
        if thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            print(f"   t={thresh:.1f}: Prec={metrics['precision']:.3f}, "
                  f"Rec={metrics['recall']:.3f}, F1={metrics['f1']:.3f}, "
                  f"FP={metrics['false_positives']}, FN={metrics['false_negatives']}")

    # Save threshold analysis
    thresh_df = pd.DataFrame(threshold_results)
    thresh_file = OUTPUT_DIR / f'threshold_analysis_h{horizon}.csv'
    thresh_df.to_csv(thresh_file, index=False)
    print(f"\n   Saved: {thresh_file}")

    # Generate threshold analysis visualization
    fig_path = FIGURES_DIR / f'threshold_analysis_h{horizon}.png'
    plot_threshold_analysis(y_true, y_proba, optimal_thresholds, horizon, fig_path)
    print(f"   Saved: {fig_path}")

    # Apply optimal threshold (Balanced Constrained: P=R >= 0.6)
    optimal_thresh = optimal_thresholds['balanced_constrained']['threshold']
    avg_predictions['y_pred_optimal'] = (avg_predictions['y_pred_proba'] >= optimal_thresh).astype(int)
    avg_predictions['ar_failure_optimal'] = (
        (avg_predictions['y_pred_optimal'] == 0) & (avg_predictions['y_true'] == 1)
    ).astype(int)
    avg_predictions['false_alarm_optimal'] = (
        (avg_predictions['y_pred_optimal'] == 1) & (avg_predictions['y_true'] == 0)
    ).astype(int)
    avg_predictions['optimal_threshold'] = optimal_thresh
    avg_predictions['constraint_met'] = optimal_thresholds['balanced_constrained']['constraint_met']

    # Overall metrics at default threshold (0.5) - for reference only
    print(f"\n{'='*80}")
    print(f"Overall Performance at DEFAULT Threshold (0.5) - Reference")
    print(f"{'='*80}")

    default_metrics = evaluate_predictions(
        avg_predictions['y_true'].values,
        avg_predictions['y_pred'].values,
        avg_predictions['y_pred_proba'].values,
        horizon,
        fold_idx=None
    )

    print(f"   Total districts: {avg_predictions['ipc_geographic_unit_full'].nunique():,}")
    print(f"   Accuracy: {default_metrics['accuracy']:.3f}")
    print(f"   Precision: {default_metrics['precision']:.3f}")
    print(f"   Recall: {default_metrics['recall']:.3f}")
    print(f"   F1 Score: {default_metrics['f1']:.3f}")
    print(f"   AUC-ROC: {default_metrics['auc_roc']:.3f}")
    print(f"   AUC-PR: {default_metrics['auc_pr']:.3f}")

    # AR Failures at default
    n_ar_failures_default = avg_predictions['ar_failure'].sum()
    n_crisis = int(avg_predictions['y_true'].sum())
    ar_pct_default = (n_ar_failures_default/n_crisis*100) if n_crisis > 0 else 0
    print(f"\n   AR Failures (missed crises): {n_ar_failures_default}/{n_crisis} ({ar_pct_default:.1f}%)")
    print(f"   False Alarms: {avg_predictions['false_alarm'].sum()}")

    # Overall metrics at OPTIMAL threshold (Balanced Constrained)
    constraint_status = "MET" if optimal_thresholds['balanced_constrained']['constraint_met'] else "NOT MET"
    print(f"\n{'='*80}")
    print(f"Overall Performance at BALANCED P=R Threshold ({optimal_thresh:.3f}) [Constraint {constraint_status}]")
    print(f"{'='*80}")

    optimal_overall = evaluate_at_threshold(y_true, y_proba, optimal_thresh)
    print(f"   Accuracy: {optimal_overall['accuracy']:.3f}")
    print(f"   Precision: {optimal_overall['precision']:.3f}")
    print(f"   Recall: {optimal_overall['recall']:.3f}")
    print(f"   F1 Score: {optimal_overall['f1']:.3f}")
    print(f"   Specificity: {optimal_overall['specificity']:.3f}")

    n_ar_failures_opt = avg_predictions['ar_failure_optimal'].sum()
    ar_pct_opt = (n_ar_failures_opt/n_crisis*100) if n_crisis > 0 else 0
    print(f"\n   AR Failures (missed crises): {n_ar_failures_opt}/{n_crisis} ({ar_pct_opt:.1f}%)")
    print(f"   False Alarms: {avg_predictions['false_alarm_optimal'].sum()}")

    # Improvement summary
    print(f"\n--- Improvement Summary (Default -> Optimal) ---")
    print(f"   Precision: {default_metrics['precision']:.3f} -> {optimal_overall['precision']:.3f} "
          f"({(optimal_overall['precision']-default_metrics['precision'])*100:+.1f}%)")
    print(f"   Recall: {default_metrics['recall']:.3f} -> {optimal_overall['recall']:.3f} "
          f"({(optimal_overall['recall']-default_metrics['recall'])*100:+.1f}%)")
    print(f"   F1: {default_metrics['f1']:.3f} -> {optimal_overall['f1']:.3f} "
          f"({(optimal_overall['f1']-default_metrics['f1'])*100:+.1f}%)")
    print(f"   False Alarms: {default_metrics['false_positives']} -> {optimal_overall['false_positives']} "
          f"({optimal_overall['false_positives']-default_metrics['false_positives']:+d})")
    print(f"   Missed Crises: {default_metrics['false_negatives']} -> {optimal_overall['false_negatives']} "
          f"({optimal_overall['false_negatives']-default_metrics['false_negatives']:+d})")

    # Create overall_metrics using OPTIMAL threshold for saving to CSV
    # This ensures all downstream scripts use optimal threshold metrics
    overall_metrics = evaluate_predictions(
        avg_predictions['y_true'].values,
        avg_predictions['y_pred_optimal'].values,
        avg_predictions['y_pred_proba'].values,
        horizon,
        fold_idx=None
    )

    # Save results
    print(f"\nSaving results...")

    # Detailed predictions
    pred_file = OUTPUT_DIR / f'predictions_h{horizon}_district_folds.csv'
    all_predictions.to_csv(pred_file, index=False)
    print(f"   Saved: {pred_file}")

    # Averaged predictions (includes both default and optimal threshold predictions)
    avg_file = OUTPUT_DIR / f'predictions_h{horizon}_district_averaged.csv'
    avg_predictions.to_csv(avg_file, index=False)
    print(f"   Saved: {avg_file}")

    # Parquet
    parquet_file = OUTPUT_DIR / f'predictions_h{horizon}_district_averaged.parquet'
    avg_predictions.to_parquet(parquet_file, index=False)
    print(f"   Saved: {parquet_file}")

    # AR failures at default threshold (0.5)
    ar_failures = avg_predictions[avg_predictions['ar_failure'] == 1].copy()
    ar_file = OUTPUT_DIR / f'ar_failures_h{horizon}_district.csv'
    ar_failures.to_csv(ar_file, index=False)
    print(f"   Saved: {ar_file} ({len(ar_failures)} failures at t=0.5)")

    # AR failures at optimal threshold
    ar_failures_opt = avg_predictions[avg_predictions['ar_failure_optimal'] == 1].copy()
    ar_opt_file = OUTPUT_DIR / f'ar_failures_h{horizon}_district_optimal.csv'
    ar_failures_opt.to_csv(ar_opt_file, index=False)
    print(f"   Saved: {ar_opt_file} ({len(ar_failures_opt)} failures at t={optimal_thresh:.3f})")

    # Save optimal threshold summary
    threshold_summary = {
        'horizon': horizon,
        'default_threshold': 0.5,
        'optimal_threshold': optimal_thresh,
        'optimal_strategy': 'balanced_constrained',
        'constraint_met': optimal_thresholds['balanced_constrained']['constraint_met'],
        'min_constraint': optimal_thresholds['balanced_constrained']['min_constraint'],
        'f1_optimal': optimal_thresholds['f1_optimal'],
        'gmean_optimal': optimal_thresholds['gmean_optimal'],
        'youden_optimal': optimal_thresholds['youden_optimal'],
        'balanced': optimal_thresholds['balanced'],
        'balanced_constrained': optimal_thresholds['balanced_constrained'],
        'default_metrics': {
            'precision': overall_metrics['precision'],
            'recall': overall_metrics['recall'],
            'f1': overall_metrics['f1'],
            'false_positives': overall_metrics['false_positives'],
            'false_negatives': overall_metrics['false_negatives']
        },
        'optimal_metrics': {
            'precision': optimal_overall['precision'],
            'recall': optimal_overall['recall'],
            'f1': optimal_overall['f1'],
            'false_positives': optimal_overall['false_positives'],
            'false_negatives': optimal_overall['false_negatives']
        }
    }

    return avg_predictions, metrics_list, coefficients_list, overall_metrics, threshold_summary


def main():
    print("=" * 80)
    print("Stage 1: Logistic Regression Baseline - DISTRICT LEVEL")
    print("=" * 80)
    print(f"Start time: {datetime.now()}\n")

    # Load data
    print("1. Loading feature-engineered district dataset...")
    df = pd.read_parquet(INPUT_FILE)
    print(f"   Loaded {len(df):,} rows, {len(df.columns)} columns")
    print(f"   Unique districts: {df['ipc_geographic_unit_full'].nunique():,}")

    # Convert dates
    df['ipc_period_start'] = pd.to_datetime(df['ipc_period_start'])
    df['ipc_period_end'] = pd.to_datetime(df['ipc_period_end'])

    # Create spatial folds
    df = create_spatial_folds(df, n_folds=N_FOLDS, random_state=RANDOM_STATE)

    # Process each horizon
    all_metrics = []
    all_coefficients = []
    all_threshold_summaries = []

    for horizon in HORIZONS:
        predictions, metrics, coefficients, overall, thresh_summary = process_horizon(df, horizon)
        all_metrics.extend(metrics)
        all_coefficients.extend(coefficients)
        all_metrics.append(overall)
        all_threshold_summaries.append(thresh_summary)

    # Save summary metrics
    print(f"\n{'='*80}")
    print("Saving Summary Statistics")
    print(f"{'='*80}")

    metrics_df = pd.DataFrame(all_metrics)
    metrics_file = OUTPUT_DIR / 'performance_metrics_district.csv'
    metrics_df.to_csv(metrics_file, index=False)
    print(f"   Saved: {metrics_file}")

    coefficients_df = pd.DataFrame(all_coefficients)
    coef_file = OUTPUT_DIR / 'model_coefficients_district.csv'
    coefficients_df.to_csv(coef_file, index=False)
    print(f"   Saved: {coef_file}")

    # Save threshold summaries
    import json
    thresh_summary_file = OUTPUT_DIR / 'threshold_tuning_summary.json'
    with open(thresh_summary_file, 'w') as f:
        # Convert to serializable format
        serializable_summaries = []
        for s in all_threshold_summaries:
            summary = {k: v for k, v in s.items() if k not in ['f1_optimal', 'gmean_optimal', 'youden_optimal', 'balanced']}
            summary['f1_optimal'] = s['f1_optimal']
            summary['gmean_optimal'] = s['gmean_optimal']
            summary['youden_optimal'] = s['youden_optimal']
            summary['balanced'] = s['balanced']
            summary['default_metrics'] = s['default_metrics']
            summary['optimal_metrics'] = s['optimal_metrics']
            serializable_summaries.append(summary)
        json.dump(serializable_summaries, f, indent=2)
    print(f"   Saved: {thresh_summary_file}")

    # Create summary DataFrame for threshold results
    thresh_rows = []
    for s in all_threshold_summaries:
        thresh_rows.append({
            'horizon': s['horizon'],
            'default_threshold': 0.5,
            'optimal_threshold': s['optimal_threshold'],
            'default_precision': s['default_metrics']['precision'],
            'default_recall': s['default_metrics']['recall'],
            'default_f1': s['default_metrics']['f1'],
            'default_fp': s['default_metrics']['false_positives'],
            'default_fn': s['default_metrics']['false_negatives'],
            'optimal_precision': s['optimal_metrics']['precision'],
            'optimal_recall': s['optimal_metrics']['recall'],
            'optimal_f1': s['optimal_metrics']['f1'],
            'optimal_fp': s['optimal_metrics']['false_positives'],
            'optimal_fn': s['optimal_metrics']['false_negatives'],
            'precision_gain': s['optimal_metrics']['precision'] - s['default_metrics']['precision'],
            'recall_change': s['optimal_metrics']['recall'] - s['default_metrics']['recall'],
            'f1_gain': s['optimal_metrics']['f1'] - s['default_metrics']['f1'],
            'fp_reduction': s['default_metrics']['false_positives'] - s['optimal_metrics']['false_positives'],
        })
    thresh_summary_df = pd.DataFrame(thresh_rows)
    thresh_csv = OUTPUT_DIR / 'threshold_tuning_comparison.csv'
    thresh_summary_df.to_csv(thresh_csv, index=False)
    print(f"   Saved: {thresh_csv}")

    # Summary report
    print(f"\n{'='*80}")
    print("Summary Report - DISTRICT LEVEL WITH THRESHOLD TUNING")
    print(f"{'='*80}")

    for i, horizon in enumerate(HORIZONS):
        horizon_metrics = metrics_df[
            (metrics_df['horizon'] == horizon) &
            (metrics_df['fold'].isna())
        ].iloc[0]

        thresh = all_threshold_summaries[i]

        print(f"\n{'-'*60}")
        print(f"Horizon h={horizon} months")
        print(f"{'-'*60}")
        print(f"   Samples: {horizon_metrics['n_samples']:,}")
        print(f"   AUC-ROC: {horizon_metrics['auc_roc']:.3f}")
        print(f"   AUC-PR: {horizon_metrics['auc_pr']:.3f}")

        print(f"\n   --- Default (t=0.5) ---")
        print(f"   Precision: {horizon_metrics['precision']:.3f}")
        print(f"   Recall: {horizon_metrics['recall']:.3f}")
        print(f"   F1 Score: {horizon_metrics['f1']:.3f}")
        print(f"   False Positives: {thresh['default_metrics']['false_positives']}")
        print(f"   False Negatives: {thresh['default_metrics']['false_negatives']}")

        constraint_status = "MET" if thresh['constraint_met'] else "NOT MET"
        print(f"\n   --- Balanced P=R (t={thresh['optimal_threshold']:.3f}) [Constraint {constraint_status}] ---")
        print(f"   Precision: {thresh['optimal_metrics']['precision']:.3f} ({(thresh['optimal_metrics']['precision']-horizon_metrics['precision'])*100:+.1f}%)")
        print(f"   Recall: {thresh['optimal_metrics']['recall']:.3f} ({(thresh['optimal_metrics']['recall']-horizon_metrics['recall'])*100:+.1f}%)")
        print(f"   F1 Score: {thresh['optimal_metrics']['f1']:.3f} ({(thresh['optimal_metrics']['f1']-horizon_metrics['f1'])*100:+.1f}%)")
        print(f"   False Positives: {thresh['optimal_metrics']['false_positives']} ({thresh['optimal_metrics']['false_positives']-thresh['default_metrics']['false_positives']:+d})")
        print(f"   False Negatives: {thresh['optimal_metrics']['false_negatives']} ({thresh['optimal_metrics']['false_negatives']-thresh['default_metrics']['false_negatives']:+d})")

        avg_coefs = coefficients_df[coefficients_df['horizon'] == horizon].mean(numeric_only=True)
        print(f"\n   Average Coefficients:")
        print(f"      Intercept: {avg_coefs['intercept']:.3f}")
        print(f"      Lt (temporal): {avg_coefs['coef_Lt']:.3f}")
        print(f"      Ls (spatial): {avg_coefs['coef_Ls']:.3f}")

    print(f"\n{'='*80}")
    print("THRESHOLD TUNING SUMMARY (Balanced P=R >= 0.6)")
    print(f"{'='*80}")
    print(f"\n{'Horizon':<8} {'Thresh':<8} {'Status':<10} {'P=R':<8} {'F1':<8} {'FP Chg':<10} {'FN Chg':<10}")
    print(f"{'-'*70}")
    for s in all_threshold_summaries:
        status = "OK" if s['constraint_met'] else "BELOW"
        prec = s['optimal_metrics']['precision']
        f1 = s['optimal_metrics']['f1']
        fp_chg = s['optimal_metrics']['false_positives'] - s['default_metrics']['false_positives']
        fn_chg = s['optimal_metrics']['false_negatives'] - s['default_metrics']['false_negatives']
        print(f"h={s['horizon']:<5} {s['optimal_threshold']:.3f}{'':>3} {status:<10} {prec:.3f}{'':>3} {f1:.3f}{'':>3} {fp_chg:+d}{'':>5} {fn_chg:+d}")

    print(f"\n{'='*80}")
    print("Stage 1 Baseline Complete - WITH THRESHOLD TUNING")
    print(f"{'='*80}")
    print(f"\nEnd time: {datetime.now()}")
    print(f"Results saved to: {OUTPUT_DIR}")
    print(f"Figures saved to: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
