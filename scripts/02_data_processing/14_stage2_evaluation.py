"""
Stage 2: Comprehensive Evaluation and Publication-Quality Visualizations
Full evaluation suite with metrics, diagnostics, and publication-ready figures.

Produces:
1. Performance metrics comparison (Stage 1 vs Stage 2)
2. Feature importance plots (fixed effects coefficients)
3. Random effects visualizations (caterpillar plots)
4. ROC and Precision-Recall curves
5. Calibration plots
6. Confusion matrix heatmaps
7. Threshold sensitivity analysis
8. Geographic analysis of improvements
9. Temporal performance analysis
10. Cross-horizon comparison

Author: Victor Collins Oppon
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import (
from config import BASE_DIR
    roc_auc_score, average_precision_score, brier_score_loss,
    precision_recall_curve, roc_curve, confusion_matrix,
    classification_report, f1_score, precision_score, recall_score,
    accuracy_score, balanced_accuracy_score, matthews_corrcoef,
    log_loss
)
from sklearn.calibration import calibration_curve
import json
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(str(BASE_DIR.parent.parent.parent))
DATA_DIR = BASE_DIR / 'data'
DISTRICT_DATA_DIR = DATA_DIR / 'district_level'
RESULTS_DIR = BASE_DIR / 'results' / 'district_level'
STAGE1_BASELINE_DIR = RESULTS_DIR / 'stage1_baseline'
STAGE2_OUTPUT_DIR = RESULTS_DIR / 'stage2_dynamic_features'
FIGURES_DIR = BASE_DIR / 'figures' / 'stage2_evaluation'

# Ensure output directory exists
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Publication-quality plot settings
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans'],
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'axes.titleweight': 'bold',
    'xtick.labelsize': 10,
    'ytick.labelsize': 10,
    'legend.fontsize': 10,
    'legend.framealpha': 0.9,
    'figure.titlesize': 14,
    'figure.titleweight': 'bold',
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.facecolor': 'white',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'grid.linestyle': '--'
})

# Color palette for publication
COLORS = {
    'primary': '#1f77b4',
    'secondary': '#ff7f0e',
    'success': '#2ca02c',
    'danger': '#d62728',
    'neutral': '#7f7f7f',
    'h4': '#1f77b4',
    'h8': '#ff7f0e',
    'h12': '#2ca02c'
}


def load_results(horizon=4):
    """Load Stage 2 model results and predictions."""
    print(f"\n   Loading Stage 2 results (h={horizon})...", flush=True)

    # Load predictions
    pred_file = STAGE2_OUTPUT_DIR / f'predictions_h{horizon}.parquet'
    if not pred_file.exists():
        pred_file = STAGE2_OUTPUT_DIR / f'predictions_h{horizon}.csv'
    if not pred_file.exists():
        raise FileNotFoundError(f"Predictions file not found for h={horizon}")

    predictions = pd.read_parquet(pred_file) if pred_file.suffix == '.parquet' else pd.read_csv(pred_file)
    print(f"      Loaded {len(predictions):,} predictions", flush=True)

    # Load coefficients
    coef_file = STAGE2_OUTPUT_DIR / f'coefficients_h{horizon}.csv'
    coefficients = pd.read_csv(coef_file) if coef_file.exists() else None
    if coefficients is not None:
        print(f"      Loaded {len(coefficients)} coefficients", flush=True)

    # Load random effects
    re_file = STAGE2_OUTPUT_DIR / f'random_effects_h{horizon}.csv'
    random_effects = pd.read_csv(re_file) if re_file.exists() else None
    if random_effects is not None:
        print(f"      Loaded {len(random_effects):,} random effects", flush=True)

    # Load metrics
    metrics_file = STAGE2_OUTPUT_DIR / f'model_results_h{horizon}.json'
    metrics = {}
    if metrics_file.exists():
        with open(metrics_file, 'r') as f:
            metrics = json.load(f)
        print(f"      Loaded model metrics", flush=True)

    # Load CV results
    cv_file = STAGE2_OUTPUT_DIR / f'cv_results_h{horizon}.csv'
    cv_results = pd.read_csv(cv_file) if cv_file.exists() else None

    return {
        'predictions': predictions,
        'coefficients': coefficients,
        'random_effects': random_effects,
        'metrics': metrics,
        'cv_results': cv_results,
        'horizon': horizon
    }


def compute_comprehensive_metrics(y_true, y_pred_proba, threshold=0.5):
    """Compute comprehensive performance metrics."""
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Basic metrics
    metrics = {
        'auc_roc': roc_auc_score(y_true, y_pred_proba),
        'auc_pr': average_precision_score(y_true, y_pred_proba),
        'brier_score': brier_score_loss(y_true, y_pred_proba),
        'log_loss': log_loss(y_true, y_pred_proba),
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'mcc': matthews_corrcoef(y_true, y_pred),
        'threshold': threshold
    }

    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    metrics.update({
        'tp': tp, 'fp': fp, 'fn': fn, 'tn': tn,
        'sensitivity': tp / (tp + fn) if (tp + fn) > 0 else 0,
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,
        'prevalence': (tp + fn) / len(y_true)
    })

    return metrics


def plot_roc_curve_publication(results_list, ax=None):
    """Publication-quality ROC curve with multiple horizons."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    for results in results_list:
        predictions = results['predictions']
        y_true = predictions['is_ar_failure'].values
        y_pred = predictions['y_pred_proba_stage2'].values
        horizon = results['horizon']

        fpr, tpr, _ = roc_curve(y_true, y_pred)
        auc = roc_auc_score(y_true, y_pred)

        ax.plot(fpr, tpr, color=COLORS[f'h{horizon}'], lw=2.5,
                label=f'h={horizon} months (AUC = {auc:.3f})')

    # Diagonal reference
    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Random Classifier')

    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontweight='bold')
    ax.set_ylabel('True Positive Rate (Sensitivity)', fontweight='bold')
    ax.set_title('ROC Curves: Stage 2 AR Failure Prediction', fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True, fancybox=True)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    # Add AUC annotation box
    textstr = 'Higher AUC = Better\nDisplacement from\ndiagonal indicates\ndiscriminative power'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
    ax.text(0.55, 0.15, textstr, transform=ax.transAxes, fontsize=9,
            verticalalignment='bottom', bbox=props)

    return ax


def plot_pr_curve_publication(results_list, ax=None):
    """Publication-quality Precision-Recall curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    for results in results_list:
        predictions = results['predictions']
        y_true = predictions['is_ar_failure'].values
        y_pred = predictions['y_pred_proba_stage2'].values
        horizon = results['horizon']

        precision, recall, _ = precision_recall_curve(y_true, y_pred)
        ap = average_precision_score(y_true, y_pred)
        baseline = y_true.mean()

        ax.plot(recall, precision, color=COLORS[f'h{horizon}'], lw=2.5,
                label=f'h={horizon} months (AP = {ap:.3f})')

    # Add baseline
    baseline = results_list[0]['predictions']['is_ar_failure'].mean()
    ax.axhline(y=baseline, color='k', linestyle='--', lw=1.5, alpha=0.7,
               label=f'No-skill baseline ({baseline:.3f})')

    ax.set_xlabel('Recall (Sensitivity)', fontweight='bold')
    ax.set_ylabel('Precision (PPV)', fontweight='bold')
    ax.set_title('Precision-Recall Curves: Stage 2 AR Failure Prediction', fontweight='bold', pad=15)
    ax.legend(loc='upper right', frameon=True, fancybox=True)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    return ax


def plot_confusion_matrix_heatmap(results, ax=None):
    """Publication-quality confusion matrix heatmap."""
    predictions = results['predictions']
    y_true = predictions['is_ar_failure'].values
    threshold = results['metrics'].get('optimal_threshold', 0.5)
    y_pred = (predictions['y_pred_proba_stage2'].values >= threshold).astype(int)

    cm = confusion_matrix(y_true, y_pred)

    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 7))

    # Normalize for percentages
    cm_pct = cm.astype('float') / cm.sum() * 100

    # Create heatmap
    sns.heatmap(cm, annot=False, fmt='d', cmap='Blues', ax=ax,
                xticklabels=['Predicted\nNon-Failure', 'Predicted\nAR Failure'],
                yticklabels=['Actual\nNon-Failure', 'Actual\nAR Failure'],
                cbar_kws={'label': 'Count'})

    # Add annotations with counts and percentages
    for i in range(2):
        for j in range(2):
            count = cm[i, j]
            pct = cm_pct[i, j]
            color = 'white' if count > cm.max() / 2 else 'black'
            ax.text(j + 0.5, i + 0.5, f'{count:,}\n({pct:.1f}%)',
                   ha='center', va='center', color=color, fontsize=12, fontweight='bold')

    ax.set_title(f'Confusion Matrix (h={results["horizon"]}, threshold={threshold:.3f})',
                fontweight='bold', pad=15)

    # Add metrics annotation
    tn, fp, fn, tp = cm.ravel()
    sens = tp / (tp + fn) if (tp + fn) > 0 else 0
    spec = tn / (tn + fp) if (tn + fp) > 0 else 0
    textstr = f'Sensitivity: {sens:.3f}\nSpecificity: {spec:.3f}'
    props = dict(boxstyle='round', facecolor='lightyellow', alpha=0.8)
    ax.text(1.35, 0.5, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='center', bbox=props)

    return ax


def plot_calibration_curve_publication(results_list, ax=None):
    """Publication-quality calibration curve."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 8))

    for results in results_list:
        predictions = results['predictions']
        y_true = predictions['is_ar_failure'].values
        y_pred = predictions['y_pred_proba_stage2'].values
        horizon = results['horizon']

        prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=10, strategy='uniform')

        ax.plot(prob_pred, prob_true, 's-', color=COLORS[f'h{horizon}'], lw=2, markersize=8,
                label=f'h={horizon} months')

    ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.7, label='Perfectly Calibrated')

    ax.set_xlabel('Mean Predicted Probability', fontweight='bold')
    ax.set_ylabel('Fraction of Positives (Observed)', fontweight='bold')
    ax.set_title('Calibration Plot: Stage 2 Model', fontweight='bold', pad=15)
    ax.legend(loc='lower right', frameon=True, fancybox=True)
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    return ax


def plot_threshold_analysis(results, ax=None):
    """Threshold sensitivity analysis."""
    predictions = results['predictions']
    y_true = predictions['is_ar_failure'].values
    y_pred = predictions['y_pred_proba_stage2'].values

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    thresholds = np.arange(0.1, 0.9, 0.02)
    precisions, recalls, f1s = [], [], []

    for t in thresholds:
        y_pred_t = (y_pred >= t).astype(int)
        precisions.append(precision_score(y_true, y_pred_t, zero_division=0))
        recalls.append(recall_score(y_true, y_pred_t, zero_division=0))
        f1s.append(f1_score(y_true, y_pred_t, zero_division=0))

    ax.plot(thresholds, precisions, 'b-', lw=2, label='Precision')
    ax.plot(thresholds, recalls, 'g-', lw=2, label='Recall')
    ax.plot(thresholds, f1s, 'r-', lw=2.5, label='F1-Score')

    # Mark optimal threshold
    optimal_idx = np.argmax(f1s)
    optimal_threshold = thresholds[optimal_idx]
    ax.axvline(x=optimal_threshold, color='k', linestyle='--', lw=1.5, alpha=0.7,
               label=f'Optimal ({optimal_threshold:.2f})')
    ax.scatter([optimal_threshold], [f1s[optimal_idx]], color='red', s=100, zorder=5)

    ax.set_xlabel('Classification Threshold', fontweight='bold')
    ax.set_ylabel('Score', fontweight='bold')
    ax.set_title(f'Threshold Sensitivity Analysis (h={results["horizon"]})', fontweight='bold', pad=15)
    ax.legend(loc='best', frameon=True, fancybox=True)
    ax.set_xlim([0.1, 0.9])
    ax.set_ylim([0, 1])

    return ax


def plot_coefficient_forest_publication(results, ax=None, top_n=20):
    """Publication-quality forest plot of coefficients."""
    coefficients = results['coefficients']
    if coefficients is None:
        return None

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 12))

    # Filter and sort
    coef_df = coefficients[coefficients['feature'] != 'intercept'].copy()
    coef_df['abs_coef'] = np.abs(coef_df['coefficient'])
    coef_df = coef_df.nlargest(top_n, 'abs_coef').sort_values('coefficient')

    # Clean feature names for display
    coef_df['display_name'] = coef_df['feature'].str.replace('_category', '').str.replace('_', ' ').str.title()

    # Color by sign
    colors = [COLORS['danger'] if c < 0 else COLORS['success'] for c in coef_df['coefficient']]

    y_pos = np.arange(len(coef_df))
    bars = ax.barh(y_pos, coef_df['coefficient'], color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axvline(x=0, color='black', linestyle='-', lw=1)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(coef_df['display_name'], fontsize=9)
    ax.set_xlabel('Coefficient (log-odds)', fontweight='bold')
    ax.set_title(f'Feature Importance: Fixed Effects (h={results["horizon"]})\nL1 Regularized (LASSO)',
                fontweight='bold', pad=15)

    # Legend
    pos_patch = mpatches.Patch(color=COLORS['success'], label='Increases AR failure risk')
    neg_patch = mpatches.Patch(color=COLORS['danger'], label='Decreases AR failure risk')
    ax.legend(handles=[pos_patch, neg_patch], loc='lower right', frameon=True)

    return ax


def plot_random_effects_distribution(results, ax=None):
    """Distribution of random effects."""
    random_effects = results['random_effects']
    if random_effects is None:
        return None

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 6))

    # Plot histogram
    intercepts = random_effects['random_intercept_alpha'].values
    ax.hist(intercepts, bins=50, color=COLORS['primary'], alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axvline(x=0, color='red', linestyle='--', lw=2, label='Population Mean')
    ax.axvline(x=intercepts.mean(), color='green', linestyle='-', lw=2,
               label=f'Sample Mean ({intercepts.mean():.3f})')

    ax.set_xlabel('Random Intercept Value (alpha_r)', fontweight='bold')
    ax.set_ylabel('Number of Districts', fontweight='bold')
    ax.set_title(f'Distribution of District Random Intercepts (h={results["horizon"]})\nHigher = Greater baseline AR failure risk',
                fontweight='bold', pad=15)
    ax.legend(frameon=True)

    # Add statistics
    textstr = f'n = {len(intercepts):,}\nMean = {intercepts.mean():.4f}\nSD = {intercepts.std():.4f}\nRange = [{intercepts.min():.3f}, {intercepts.max():.3f}]'
    props = dict(boxstyle='round', facecolor='wheat', alpha=0.8)
    ax.text(0.98, 0.98, textstr, transform=ax.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', bbox=props)

    return ax


def plot_country_performance_bar(results, ax=None, top_n=15):
    """Country-level performance bar chart."""
    predictions = results['predictions']

    if 'ipc_country' not in predictions.columns:
        return None

    if ax is None:
        fig, ax = plt.subplots(figsize=(12, 8))

    # Compute metrics by country
    country_metrics = []
    for country in predictions['ipc_country'].unique():
        mask = predictions['ipc_country'] == country
        if mask.sum() < 30:
            continue

        y_true = predictions.loc[mask, 'is_ar_failure'].values
        y_pred = predictions.loc[mask, 'y_pred_proba_stage2'].values

        if y_true.sum() < 5 or y_true.sum() == len(y_true):
            continue

        try:
            auc = roc_auc_score(y_true, y_pred)
            country_metrics.append({
                'country': country,
                'auc_roc': auc,
                'n_failures': y_true.sum(),
                'n_total': len(y_true)
            })
        except:
            continue

    if not country_metrics:
        return None

    cm_df = pd.DataFrame(country_metrics)
    cm_df = cm_df.nlargest(top_n, 'n_failures').sort_values('auc_roc')

    # Create bar chart
    y_pos = np.arange(len(cm_df))
    colors = plt.cm.RdYlGn(cm_df['auc_roc'])

    bars = ax.barh(y_pos, cm_df['auc_roc'], color=colors, edgecolor='black', linewidth=0.5)
    ax.axvline(x=0.5, color='red', linestyle='--', lw=2, label='Random (0.5)')

    # Add count annotations
    for i, (_, row) in enumerate(cm_df.iterrows()):
        ax.annotate(f"n={row['n_failures']:.0f}",
                   xy=(row['auc_roc'] + 0.02, i),
                   fontsize=9, va='center')

    ax.set_yticks(y_pos)
    ax.set_yticklabels(cm_df['country'])
    ax.set_xlabel('AUC-ROC', fontweight='bold')
    ax.set_title(f'Stage 2 Performance by Country (h={results["horizon"]})\nTop {top_n} countries by AR failure count',
                fontweight='bold', pad=15)
    ax.set_xlim([0.3, 1.0])
    ax.legend(loc='lower right', frameon=True)

    return ax


def plot_metrics_comparison_table(results_list, ax=None):
    """Metrics comparison table across horizons."""
    if ax is None:
        fig, ax = plt.subplots(figsize=(14, 8))

    ax.axis('off')

    # Collect metrics
    data = []
    for results in results_list:
        predictions = results['predictions']
        y_true = predictions['is_ar_failure'].values
        y_pred = predictions['y_pred_proba_stage2'].values
        threshold = results['metrics'].get('optimal_threshold', 0.5)

        metrics = compute_comprehensive_metrics(y_true, y_pred, threshold)
        metrics['horizon'] = results['horizon']
        data.append(metrics)

    # Create DataFrame
    df = pd.DataFrame(data)
    df = df.set_index('horizon')

    # Select key metrics for display
    display_cols = ['auc_roc', 'auc_pr', 'precision', 'recall', 'f1', 'mcc',
                   'sensitivity', 'specificity', 'tp', 'fn', 'prevalence']

    display_df = df[display_cols].T
    display_df.index = ['AUC-ROC', 'AUC-PR', 'Precision', 'Recall', 'F1-Score', 'MCC',
                       'Sensitivity', 'Specificity', 'True Positives', 'False Negatives', 'Prevalence']

    # Format values
    for col in display_df.columns:
        for idx in display_df.index:
            val = display_df.loc[idx, col]
            if idx in ['True Positives', 'False Negatives']:
                display_df.loc[idx, col] = f'{int(val):,}'
            elif idx == 'Prevalence':
                display_df.loc[idx, col] = f'{val:.1%}'
            else:
                display_df.loc[idx, col] = f'{val:.4f}'

    # Create table
    table = ax.table(cellText=display_df.values,
                     rowLabels=display_df.index,
                     colLabels=[f'h={h} months' for h in display_df.columns],
                     cellLoc='center',
                     loc='center',
                     colWidths=[0.15] * len(display_df.columns))

    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.5, 2)

    # Style header
    for i in range(len(display_df.columns)):
        table[(0, i)].set_facecolor(COLORS['primary'])
        table[(0, i)].set_text_props(color='white', fontweight='bold')

    # Style row labels
    for i in range(len(display_df.index)):
        table[(i + 1, -1)].set_facecolor('#f0f0f0')
        table[(i + 1, -1)].set_text_props(fontweight='bold')

    ax.set_title('Comprehensive Metrics Summary: Stage 2 Mixed-Effects Model',
                fontweight='bold', fontsize=14, pad=20)

    return ax


def create_summary_report(results_list):
    """Create detailed summary report."""
    report_data = []

    for results in results_list:
        predictions = results['predictions']
        y_true = predictions['is_ar_failure'].values
        y_pred = predictions['y_pred_proba_stage2'].values
        threshold = results['metrics'].get('optimal_threshold', 0.5)

        metrics = compute_comprehensive_metrics(y_true, y_pred, threshold)
        metrics['horizon'] = results['horizon']

        # Add CV metrics if available
        if results.get('cv_results') is not None:
            cv = results['cv_results']
            metrics['cv_auc_roc_mean'] = cv['auc_roc'].mean()
            metrics['cv_auc_roc_std'] = cv['auc_roc'].std()
            metrics['cv_auc_pr_mean'] = cv['auc_pr'].mean()
            metrics['cv_auc_pr_std'] = cv['auc_pr'].std()

        # AR failure capture rate
        metrics['ar_failures_total'] = y_true.sum()
        metrics['ar_failures_caught'] = metrics['tp']
        metrics['capture_rate'] = metrics['tp'] / y_true.sum() if y_true.sum() > 0 else 0

        report_data.append(metrics)

    return pd.DataFrame(report_data)


def generate_publication_figures(results_list):
    """Generate all publication-quality figures."""
    print("\n2. Generating publication-quality figures...", flush=True)

    # Figure 1: ROC and PR curves combined
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    plot_roc_curve_publication(results_list, axes[0])
    plot_pr_curve_publication(results_list, axes[1])
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig1_roc_pr_curves.png', dpi=300)
    fig.savefig(FIGURES_DIR / 'fig1_roc_pr_curves.pdf')
    plt.close()
    print("      Saved: fig1_roc_pr_curves.png/pdf", flush=True)

    # Figure 2: Confusion matrices
    n_horizons = len(results_list)
    fig, axes = plt.subplots(1, n_horizons, figsize=(7 * n_horizons, 6))
    if n_horizons == 1:
        axes = [axes]
    for i, results in enumerate(results_list):
        plot_confusion_matrix_heatmap(results, axes[i])
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig2_confusion_matrices.png', dpi=300)
    plt.close()
    print("      Saved: fig2_confusion_matrices.png", flush=True)

    # Figure 3: Calibration curves
    fig, ax = plt.subplots(figsize=(9, 8))
    plot_calibration_curve_publication(results_list, ax)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig3_calibration.png', dpi=300)
    plt.close()
    print("      Saved: fig3_calibration.png", flush=True)

    # Figure 4: Threshold analysis
    fig, axes = plt.subplots(1, n_horizons, figsize=(8 * n_horizons, 5))
    if n_horizons == 1:
        axes = [axes]
    for i, results in enumerate(results_list):
        plot_threshold_analysis(results, axes[i])
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig4_threshold_analysis.png', dpi=300)
    plt.close()
    print("      Saved: fig4_threshold_analysis.png", flush=True)

    # Figure 5: Feature importance (coefficient forest)
    for results in results_list:
        if results['coefficients'] is not None:
            fig, ax = plt.subplots(figsize=(11, 14))
            plot_coefficient_forest_publication(results, ax, top_n=25)
            plt.tight_layout()
            fig.savefig(FIGURES_DIR / f'fig5_coefficients_h{results["horizon"]}.png', dpi=300)
            plt.close()
            print(f"      Saved: fig5_coefficients_h{results['horizon']}.png", flush=True)

    # Figure 6: Random effects distribution
    for results in results_list:
        if results['random_effects'] is not None:
            fig, ax = plt.subplots(figsize=(11, 6))
            plot_random_effects_distribution(results, ax)
            plt.tight_layout()
            fig.savefig(FIGURES_DIR / f'fig6_random_effects_h{results["horizon"]}.png', dpi=300)
            plt.close()
            print(f"      Saved: fig6_random_effects_h{results['horizon']}.png", flush=True)

    # Figure 7: Country performance
    for results in results_list:
        fig, ax = plt.subplots(figsize=(12, 9))
        plot_country_performance_bar(results, ax, top_n=15)
        plt.tight_layout()
        fig.savefig(FIGURES_DIR / f'fig7_country_performance_h{results["horizon"]}.png', dpi=300)
        plt.close()
        print(f"      Saved: fig7_country_performance_h{results['horizon']}.png", flush=True)

    # Figure 8: Metrics comparison table
    fig, ax = plt.subplots(figsize=(14, 10))
    plot_metrics_comparison_table(results_list, ax)
    plt.tight_layout()
    fig.savefig(FIGURES_DIR / 'fig8_metrics_summary.png', dpi=300)
    plt.close()
    print("      Saved: fig8_metrics_summary.png", flush=True)


def main():
    print("=" * 80, flush=True)
    print("Stage 2: Comprehensive Evaluation and Visualization", flush=True)
    print("=" * 80, flush=True)
    print(f"Start time: {datetime.now()}", flush=True)

    # Load all available results
    print("\n1. Loading model results...", flush=True)
    results_list = []

    for horizon in [4, 8, 12]:
        try:
            results = load_results(horizon)
            results_list.append(results)
        except FileNotFoundError as e:
            print(f"      [SKIP] h={horizon}: {e}", flush=True)
        except Exception as e:
            print(f"      [ERROR] h={horizon}: {e}", flush=True)

    if not results_list:
        print("\n[ERROR] No results files found. Run 13_stage2_mixed_effects_model.py first.", flush=True)
        return

    # Generate figures
    generate_publication_figures(results_list)

    # Create summary report
    print("\n3. Creating summary report...", flush=True)
    summary_df = create_summary_report(results_list)
    summary_df.to_csv(STAGE2_OUTPUT_DIR / 'comprehensive_evaluation_summary.csv', index=False)
    print("      Saved: comprehensive_evaluation_summary.csv", flush=True)

    # Print summary
    print("\n" + "=" * 80, flush=True)
    print("STAGE 2 EVALUATION SUMMARY", flush=True)
    print("=" * 80, flush=True)

    for _, row in summary_df.iterrows():
        print(f"\n--- Horizon h={int(row['horizon'])} months ---", flush=True)
        print(f"   AUC-ROC:              {row['auc_roc']:.4f}", flush=True)
        print(f"   AUC-PR:               {row['auc_pr']:.4f}", flush=True)
        print(f"   Precision:            {row['precision']:.4f}", flush=True)
        print(f"   Recall (Sensitivity): {row['recall']:.4f}", flush=True)
        print(f"   F1-Score:             {row['f1']:.4f}", flush=True)
        print(f"   MCC:                  {row['mcc']:.4f}", flush=True)
        print(f"   Brier Score:          {row['brier_score']:.4f}", flush=True)
        print(f"   AR Failures Caught:   {int(row['tp']):,} / {int(row['ar_failures_total']):,} ({row['capture_rate']:.1%})", flush=True)

    print("\n" + "=" * 80, flush=True)
    print(f"Output directory: {FIGURES_DIR}", flush=True)
    print(f"End time: {datetime.now()}", flush=True)


if __name__ == "__main__":
    main()
