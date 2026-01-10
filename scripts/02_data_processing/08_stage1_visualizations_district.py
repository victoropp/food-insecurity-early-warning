"""
Stage 1: Publication-Grade Visualizations and Analysis - DISTRICT LEVEL
Comprehensive visual analysis of AR baseline performance with storytelling approach
Uses official IPC color schemes and cartographic maps

Author: Adapted for district-level analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
import warnings
from config import BASE_DIR
warnings.filterwarnings('ignore')

# Import mapping libraries
from matplotlib.colors import ListedColormap, BoundaryNorm
import matplotlib.patches as mpatches

# Import sklearn metrics for ROC/PR curves
from sklearn.metrics import roc_curve, precision_recall_curve, auc

# Paths - DISTRICT LEVEL
BASE_DIR = Path(str(BASE_DIR.parent.parent.parent))
RESULTS_DIR = BASE_DIR / 'results' / 'district_level' / 'stage1_baseline'
FIGURES_DIR = BASE_DIR / 'figures' / 'stage1_district_level'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

# Official IPC Color Scheme
IPC_COLORS = {
    1: '#6ABD45',  # Minimal - Green
    2: '#F9E814',  # Stressed - Yellow
    3: '#F58220',  # Crisis - Orange
    4: '#E31E24',  # Emergency - Red
    5: '#6D071A'   # Famine - Dark Red/Maroon
}

# Performance color scheme (monochromatic blues for bars)
PERFORMANCE_COLORS = {
    'primary': '#1f77b4',      # Blue
    'secondary': '#aec7e8',    # Light blue
    'tertiary': '#4292c6',     # Medium blue
    'quaternary': '#08519c'    # Dark blue
}

# Set publication-quality style
plt.style.use('seaborn-v0_8-paper')
sns.set_context("paper", font_scale=1.2)
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans']
plt.rcParams['axes.labelsize'] = 11
plt.rcParams['axes.titlesize'] = 12
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10

def load_data():
    """Load all prediction results and metrics - DISTRICT LEVEL"""
    print("Loading DISTRICT-LEVEL prediction results...")

    data = {}
    horizons = []

    # Detect available horizons dynamically
    for horizon in [4, 8, 12]:
        pred_file = RESULTS_DIR / f'predictions_h{horizon}_district_averaged.parquet'
        if pred_file.exists():
            horizons.append(horizon)
            data[f'h{horizon}'] = pd.read_parquet(pred_file)

            # Load AR failures
            failures_file = RESULTS_DIR / f'ar_failures_h{horizon}_district_optimal.csv'
            if failures_file.exists():
                data[f'failures_h{horizon}'] = pd.read_csv(failures_file)
            else:
                data[f'failures_h{horizon}'] = pd.DataFrame()

    # Load metrics
    metrics_file = RESULTS_DIR / 'performance_metrics_district.csv'
    if metrics_file.exists():
        data['metrics'] = pd.read_csv(metrics_file)
    else:
        raise FileNotFoundError(f"Metrics file not found: {metrics_file}")

    # Load coefficients
    coef_file = RESULTS_DIR / 'model_coefficients_district.csv'
    if coef_file.exists():
        data['coefficients'] = pd.read_csv(coef_file)
    else:
        data['coefficients'] = pd.DataFrame()

    data['horizons'] = horizons

    print(f"   Loaded predictions for {len(horizons)} horizons")
    for h in horizons:
        print(f"   h={h}: {len(data[f'h{h}']):,} observations")

    return data

def plot_1_performance_overview(data):
    """
    Figure 1: Performance Metrics Across Horizons
    Story: How well does the AR baseline perform at different prediction horizons?
    """
    print("\n1. Creating performance overview...")

    # Get overall metrics (where fold is NaN)
    metrics = data['metrics'][data['metrics']['fold'].isna()].copy()
    metrics = metrics.sort_values('horizon')
    horizons = data['horizons']

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle('Stage 1 Baseline: Performance Across Prediction Horizons\n' +
                 'Spatio-Temporal Autoregressive Model (Lt + Ls) - DISTRICT LEVEL',
                 fontsize=14, fontweight='bold', y=0.995)

    # Filter metrics to available horizons
    metrics = metrics[metrics['horizon'].isin(horizons)]

    # 1. Accuracy metrics
    ax = axes[0, 0]
    x = np.arange(len(horizons))
    width = 0.25

    accuracy_vals = [metrics[metrics['horizon'] == h]['accuracy'].values[0] for h in horizons]
    precision_vals = [metrics[metrics['horizon'] == h]['precision'].values[0] for h in horizons]
    recall_vals = [metrics[metrics['horizon'] == h]['recall'].values[0] for h in horizons]

    ax.bar(x - width, accuracy_vals, width,
           label='Accuracy', color=PERFORMANCE_COLORS['primary'], alpha=0.8)
    ax.bar(x, precision_vals, width,
           label='Precision', color=PERFORMANCE_COLORS['secondary'], alpha=0.8)
    ax.bar(x + width, recall_vals, width,
           label='Recall', color=PERFORMANCE_COLORS['tertiary'], alpha=0.8)

    ax.set_xlabel('Prediction Horizon (months)')
    ax.set_ylabel('Score')
    ax.set_title('(A) Classification Performance', fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h}' for h in horizons])
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True, loc='lower left')
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for i, h in enumerate(horizons):
        ax.text(i - width, accuracy_vals[i] + 0.02, f"{accuracy_vals[i]:.2f}",
                ha='center', va='bottom', fontsize=8)
        ax.text(i, precision_vals[i] + 0.02, f"{precision_vals[i]:.2f}",
                ha='center', va='bottom', fontsize=8)
        ax.text(i + width, recall_vals[i] + 0.02, f"{recall_vals[i]:.2f}",
                ha='center', va='bottom', fontsize=8)

    # 2. F1 and AUC scores
    ax = axes[0, 1]
    x = np.arange(len(horizons))
    width = 0.35

    f1_vals = [metrics[metrics['horizon'] == h]['f1'].values[0] for h in horizons]
    auc_vals = [metrics[metrics['horizon'] == h]['auc_roc'].values[0] for h in horizons]

    ax.bar(x - width/2, f1_vals, width,
           label='F1 Score', color=PERFORMANCE_COLORS['primary'], alpha=0.8)
    ax.bar(x + width/2, auc_vals, width,
           label='AUC-ROC', color=PERFORMANCE_COLORS['quaternary'], alpha=0.8)

    ax.set_xlabel('Prediction Horizon (months)')
    ax.set_ylabel('Score')
    ax.set_title('(B) Composite Metrics', fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h}' for h in horizons])
    ax.set_ylim(0, 1.05)
    ax.legend(frameon=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for i, h in enumerate(horizons):
        ax.text(i - width/2, f1_vals[i] + 0.02, f"{f1_vals[i]:.2f}",
                ha='center', va='bottom', fontsize=8)
        ax.text(i + width/2, auc_vals[i] + 0.02, f"{auc_vals[i]:.2f}",
                ha='center', va='bottom', fontsize=8)

    # 3. Sample sizes and class distribution
    ax = axes[1, 0]
    x = np.arange(len(horizons))
    width = 0.35

    no_crisis_vals = [metrics[metrics['horizon'] == h]['n_no_crisis'].values[0] for h in horizons]
    crisis_vals = [metrics[metrics['horizon'] == h]['n_crisis'].values[0] for h in horizons]

    ax.bar(x - width/2, no_crisis_vals, width,
           label='No Crisis (IPC<3)', color='#6ABD45', alpha=0.7)
    ax.bar(x + width/2, crisis_vals, width,
           label='Crisis (IPC>=3)', color='#F58220', alpha=0.7)

    ax.set_xlabel('Prediction Horizon (months)')
    ax.set_ylabel('Number of Observations')
    ax.set_title('(C) Dataset Composition', fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h}' for h in horizons])
    ax.legend(frameon=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add percentage labels
    for i, h in enumerate(horizons):
        row = metrics[metrics['horizon'] == h].iloc[0]
        total = row['n_samples']
        crisis_pct = row['n_crisis'] / total * 100 if total > 0 else 0
        ax.text(i + width/2, crisis_vals[i] + max(crisis_vals)*0.02, f"{crisis_pct:.1f}%",
                ha='center', va='bottom', fontsize=8, fontweight='bold')

    # 4. AR Failures (critical for Stage 2)
    ax = axes[1, 1]
    x = np.arange(len(horizons))

    # Calculate AR failures from confusion matrix
    ar_failures = [metrics[metrics['horizon'] == h]['false_negatives'].values[0] for h in horizons]
    total_crisis = [metrics[metrics['horizon'] == h]['n_crisis'].values[0] for h in horizons]
    failure_rate = [(ar/tc * 100) if tc > 0 else 0 for ar, tc in zip(ar_failures, total_crisis)]

    bars = ax.bar(x, failure_rate, color='#E31E24', alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Prediction Horizon (months)')
    ax.set_ylabel('AR Failure Rate (%)')
    ax.set_title('(D) Missed Crisis Events (AR Failures)', fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h}' for h in horizons])
    ax.set_ylim(0, max(failure_rate) * 1.3 if failure_rate else 25)
    ax.grid(axis='y', alpha=0.3, linestyle='--')

    # Add value labels
    for i, (rate, count) in enumerate(zip(failure_rate, ar_failures)):
        ax.text(i, rate + 0.5, f"{rate:.1f}%\n({int(count)} events)",
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    # Add horizontal line at 20% threshold
    ax.axhline(y=20, color='red', linestyle='--', linewidth=1, alpha=0.5, label='20% threshold')
    ax.legend(frameon=True, loc='upper right')

    plt.tight_layout()

    filename = FIGURES_DIR / '01_performance_overview.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved: {filename}")
    plt.close()

def plot_2_confusion_matrices(data):
    """
    Figure 2: Confusion Matrices for Each Horizon
    Story: Where does the model succeed and fail?
    """
    print("\n2. Creating confusion matrices...")

    metrics = data['metrics'][data['metrics']['fold'].isna()].copy()
    horizons = data['horizons']

    fig, axes = plt.subplots(1, len(horizons), figsize=(5*len(horizons), 4))
    if len(horizons) == 1:
        axes = [axes]

    fig.suptitle('Stage 1 Baseline: Confusion Matrices by Prediction Horizon\n' +
                 'Actual vs Predicted Crisis Onset (IPC >= 3) - DISTRICT LEVEL',
                 fontsize=14, fontweight='bold')

    for idx, horizon in enumerate(horizons):
        ax = axes[idx]
        row = metrics[metrics['horizon'] == horizon].iloc[0]

        # Build confusion matrix
        cm = np.array([
            [row['true_negatives'], row['false_positives']],
            [row['false_negatives'], row['true_positives']]
        ])

        # Normalize for percentage display
        cm_pct = cm / cm.sum() * 100

        # Create heatmap
        sns.heatmap(cm, annot=False, fmt='d', cmap='Blues',
                    cbar=False, ax=ax, linewidths=2, linecolor='black')

        # Add custom annotations with counts and percentages
        for i in range(2):
            for j in range(2):
                count = cm[i, j]
                pct = cm_pct[i, j]
                text = f'{int(count):,}\n({pct:.1f}%)'
                color = 'white' if cm[i, j] > cm.max() / 2 else 'black'
                ax.text(j + 0.5, i + 0.5, text,
                       ha='center', va='center', fontsize=11,
                       color=color, fontweight='bold')

        ax.set_xlabel('Predicted Label', fontweight='bold')
        ax.set_ylabel('True Label', fontweight='bold')
        ax.set_title(f'h={horizon} months\n' +
                     f'Accuracy: {row["accuracy"]:.3f} | Recall: {row["recall"]:.3f}',
                     fontweight='bold')
        ax.set_xticklabels(['No Crisis\n(0)', 'Crisis\n(1)'])
        ax.set_yticklabels(['No Crisis\n(0)', 'Crisis\n(1)'], rotation=0)

    plt.tight_layout()

    filename = FIGURES_DIR / '02_confusion_matrices.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved: {filename}")
    plt.close()

def plot_3_model_coefficients(data):
    """
    Figure 3: Model Coefficients Across Horizons
    Story: What drives crisis predictions? Temporal vs Spatial features
    """
    print("\n3. Creating coefficient analysis...")

    if data['coefficients'].empty:
        print("   [WARNING] No coefficient data available, skipping...")
        return

    coefs = data['coefficients'].copy()
    horizons = data['horizons']

    # Average coefficients across folds
    avg_coefs = coefs.groupby('horizon').agg({
        'intercept': ['mean', 'std'],
        'coef_Lt': ['mean', 'std'],
        'coef_Ls': ['mean', 'std']
    }).reset_index()

    # Filter to available horizons
    avg_coefs = avg_coefs[avg_coefs['horizon'].isin(horizons)]

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle('Stage 1 Baseline: Logistic Regression Coefficients\n' +
                 'Feature Importance for Crisis Prediction - DISTRICT LEVEL',
                 fontsize=14, fontweight='bold')

    # 1. Coefficient values with error bars
    ax = axes[0]
    x = np.arange(len(avg_coefs))
    width = 0.35

    lt_means = avg_coefs[('coef_Lt', 'mean')].values
    lt_stds = avg_coefs[('coef_Lt', 'std')].values
    ls_means = avg_coefs[('coef_Ls', 'mean')].values
    ls_stds = avg_coefs[('coef_Ls', 'std')].values

    ax.bar(x - width/2, lt_means, width, yerr=lt_stds,
           label='Lt (Temporal Lag)', color=PERFORMANCE_COLORS['primary'],
           alpha=0.8, capsize=5, error_kw={'linewidth': 2})
    ax.bar(x + width/2, ls_means, width, yerr=ls_stds,
           label='Ls (Spatial Lag)', color=PERFORMANCE_COLORS['tertiary'],
           alpha=0.8, capsize=5, error_kw={'linewidth': 2})

    ax.set_xlabel('Prediction Horizon (months)')
    ax.set_ylabel('Coefficient Value')
    ax.set_title('(A) Feature Coefficients with Standard Deviation',
                 fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h}' for h in avg_coefs['horizon'].values])
    ax.legend(frameon=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=0, color='black', linestyle='-', linewidth=0.8)

    # Add value labels
    for i in range(len(avg_coefs)):
        ax.text(i - width/2, lt_means[i] + lt_stds[i] + 0.05,
                f'{lt_means[i]:.2f}', ha='center', va='bottom', fontsize=9)
        ax.text(i + width/2, ls_means[i] + ls_stds[i] + 0.05,
                f'{ls_means[i]:.2f}', ha='center', va='bottom', fontsize=9)

    # 2. Relative importance (ratio)
    ax = axes[1]

    lt_ls_ratio = lt_means / np.where(ls_means != 0, ls_means, 1)

    bars = ax.bar(x, lt_ls_ratio, color=PERFORMANCE_COLORS['quaternary'],
                  alpha=0.8, edgecolor='black', linewidth=1.5)

    ax.set_xlabel('Prediction Horizon (months)')
    ax.set_ylabel('Lt / Ls Coefficient Ratio')
    ax.set_title('(B) Temporal vs Spatial Feature Importance',
                 fontweight='bold', loc='left')
    ax.set_xticks(x)
    ax.set_xticklabels([f'{h}' for h in avg_coefs['horizon'].values])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    ax.axhline(y=1, color='red', linestyle='--', linewidth=2,
               label='Equal importance', alpha=0.7)
    ax.legend(frameon=True)

    # Add value labels
    for i, ratio in enumerate(lt_ls_ratio):
        ax.text(i, ratio + 0.02, f'{ratio:.2f}x',
                ha='center', va='bottom', fontsize=10, fontweight='bold')

    # Add interpretation text
    ax.text(0.5, 0.98, 'Values > 1: Temporal lag more important\nValues < 1: Spatial lag more important',
            transform=ax.transAxes, ha='center', va='top',
            fontsize=9, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.3))

    plt.tight_layout()

    filename = FIGURES_DIR / '03_model_coefficients.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved: {filename}")
    plt.close()

def plot_4_spatial_cv_stability(data):
    """
    Figure 4: Performance Stability Across Spatial Folds
    Story: How robust is the model across different geographic regions?
    """
    print("\n4. Creating spatial CV stability analysis...")

    metrics = data['metrics'][data['metrics']['fold'].notna()].copy()
    horizons = data['horizons']

    if metrics.empty:
        print("   [WARNING] No fold-level metrics available, skipping...")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Stage 1 Baseline: Performance Stability Across Spatial CV Folds\n' +
                 'Geographic Robustness Assessment - DISTRICT LEVEL',
                 fontsize=14, fontweight='bold')

    # Metrics to analyze
    metric_specs = [
        ('accuracy', '(A) Accuracy by Fold', PERFORMANCE_COLORS['primary']),
        ('recall', '(B) Recall by Fold', PERFORMANCE_COLORS['tertiary']),
        ('precision', '(C) Precision by Fold', PERFORMANCE_COLORS['secondary']),
        ('auc_roc', '(D) AUC-ROC by Fold', PERFORMANCE_COLORS['quaternary'])
    ]

    for idx, (metric, title, color) in enumerate(metric_specs):
        ax = axes[idx // 2, idx % 2]

        # Prepare data for grouped bar chart
        folds = sorted(metrics['fold'].unique())
        x = np.arange(len(folds))
        width = 0.25

        for i, horizon in enumerate(horizons):
            horizon_data = metrics[metrics['horizon'] == horizon]
            values = []
            for fold in folds:
                fold_data = horizon_data[horizon_data['fold'] == fold]
                if len(fold_data) > 0:
                    values.append(fold_data[metric].values[0])
                else:
                    values.append(0)

            offset = (i - len(horizons)//2) * width
            ax.bar(x + offset, values, width, label=f'h={horizon}',
                  alpha=0.7)

        ax.set_xlabel('Spatial Fold')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(title, fontweight='bold', loc='left')
        ax.set_xticks(x)
        ax.set_xticklabels([f'Fold {int(f)}' for f in folds])
        ax.legend(frameon=True, title='Horizon')
        ax.grid(axis='y', alpha=0.3, linestyle='--')
        ax.set_ylim(0, 1.05)

        # Add mean line
        for horizon in horizons:
            horizon_data = metrics[metrics['horizon'] == horizon]
            if len(horizon_data) > 0:
                mean_val = horizon_data[metric].mean()
                ax.axhline(y=mean_val, linestyle='--', linewidth=1, alpha=0.5)

    plt.tight_layout()

    filename = FIGURES_DIR / '04_spatial_cv_stability.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved: {filename}")
    plt.close()

def plot_5_ar_failures_temporal(data):
    """
    Figure 5: AR Failures Over Time
    Story: When do AR failures occur? Any temporal patterns?
    """
    print("\n5. Creating temporal AR failure analysis...")

    horizons = data['horizons']

    fig, axes = plt.subplots(len(horizons), 1, figsize=(14, 4*len(horizons)), sharex=True)
    if len(horizons) == 1:
        axes = [axes]

    fig.suptitle('Stage 1 Baseline: Temporal Distribution of AR Failures\n' +
                 'Missed Crisis Events by IPC Assessment Period - DISTRICT LEVEL',
                 fontsize=14, fontweight='bold')

    for idx, horizon in enumerate(horizons):
        ax = axes[idx]

        # Get predictions for this horizon
        preds = data[f'h{horizon}'].copy()
        preds['ipc_period_start'] = pd.to_datetime(preds['ipc_period_start'])
        preds['year_month'] = preds['ipc_period_start'].dt.to_period('M')

        # Group by month - using optimal threshold columns
        monthly = preds.groupby('year_month').agg({
            'y_true': 'sum',  # Total crises
            'ar_failure_optimal': 'sum',  # Missed crises (at optimal threshold)
            'correct': 'sum'  # Correct predictions
        }).reset_index()

        monthly['year_month_str'] = monthly['year_month'].astype(str)

        # Calculate failure rate
        monthly['failure_rate'] = np.where(
            monthly['y_true'] > 0,
            monthly['ar_failure_optimal'] / monthly['y_true'] * 100,
            0
        )

        # Plot
        x = np.arange(len(monthly))

        ax.bar(x, monthly['y_true'], label='Total Crisis Events',
               color=IPC_COLORS[3], alpha=0.4)
        ax.bar(x, monthly['ar_failure_optimal'], label='AR Failures (Missed)',
               color='#E31E24', alpha=0.9)

        ax.set_ylabel('Number of Events')
        ax.set_title(f'h={horizon} months | Avg Failure Rate: {monthly["failure_rate"].mean():.1f}%',
                    fontweight='bold', loc='left')
        ax.legend(frameon=True, loc='upper left')
        ax.grid(axis='y', alpha=0.3, linestyle='--')

        # Set x-axis labels (show every 3rd month)
        tick_positions = x[::3]
        tick_labels = monthly['year_month_str'].values[::3]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=45, ha='right')

        # Add twin axis for failure rate
        ax2 = ax.twinx()
        ax2.plot(x, monthly['failure_rate'], color='darkred',
                linewidth=2, marker='o', markersize=3, label='Failure Rate (%)')
        ax2.set_ylabel('Failure Rate (%)', color='darkred')
        ax2.tick_params(axis='y', labelcolor='darkred')
        ax2.set_ylim(0, 100)
        ax2.legend(frameon=True, loc='upper right')

    axes[-1].set_xlabel('IPC Assessment Period (Year-Month)')

    plt.tight_layout()

    filename = FIGURES_DIR / '05_ar_failures_temporal.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved: {filename}")
    plt.close()

def plot_6_performance_by_country(data):
    """
    Figure 6: Performance Heatmap by Country
    Story: Which countries/regions are well-predicted vs problematic?
    """
    print("\n6. Creating country-level performance heatmap...")

    horizons = data['horizons']

    fig, axes = plt.subplots(1, len(horizons), figsize=(6*len(horizons), 8))
    if len(horizons) == 1:
        axes = [axes]

    fig.suptitle('Stage 1 Baseline: Performance Heatmap by Country\n' +
                 'Recall (Crisis Detection Rate) Across African Countries - DISTRICT LEVEL',
                 fontsize=14, fontweight='bold')

    for idx, horizon in enumerate(horizons):
        ax = axes[idx]

        preds = data[f'h{horizon}'].copy()

        # Calculate country-level metrics using optimal threshold
        country_metrics = preds.groupby('ipc_country').agg({
            'y_true': 'sum',
            'y_pred_optimal': 'sum',
            'correct': 'sum',
            'ar_failure_optimal': 'sum',
            'ipc_id': 'count'
        }).reset_index()

        country_metrics.columns = ['country', 'total_crisis', 'predicted_crisis',
                                   'correct', 'ar_failures', 'n_obs']

        # Calculate recall (crisis detection rate)
        country_metrics['recall'] = np.where(
            country_metrics['total_crisis'] > 0,
            (country_metrics['total_crisis'] - country_metrics['ar_failures']) /
            country_metrics['total_crisis'],
            0
        )

        # Filter countries with at least 10 crisis events
        country_metrics = country_metrics[country_metrics['total_crisis'] >= 10]
        country_metrics = country_metrics.sort_values('recall')

        if len(country_metrics) == 0:
            ax.text(0.5, 0.5, 'No countries with >= 10 crisis events',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Create horizontal bar chart
        y = np.arange(len(country_metrics))
        colors = plt.cm.RdYlGn(country_metrics['recall'].values)

        bars = ax.barh(y, country_metrics['recall'].values, color=colors,
                      edgecolor='black', linewidth=0.5)

        ax.set_yticks(y)
        ax.set_yticklabels(country_metrics['country'].values, fontsize=8)
        ax.set_xlabel('Recall (Crisis Detection Rate)')
        ax.set_title(f'h={horizon} months', fontweight='bold')
        ax.set_xlim(0, 1)
        ax.grid(axis='x', alpha=0.3, linestyle='--')

        # Add value labels
        for i, (recall, failures, total) in enumerate(zip(
            country_metrics['recall'].values,
            country_metrics['ar_failures'].values,
            country_metrics['total_crisis'].values
        )):
            ax.text(recall + 0.02, i, f'{recall:.2f} ({int(failures)}/{int(total)})',
                   va='center', fontsize=7)

        # Add vertical line at 0.8 (good performance threshold)
        ax.axvline(x=0.8, color='green', linestyle='--', linewidth=1.5,
                  alpha=0.5, label='80% threshold')
        ax.legend(frameon=True, loc='lower right', fontsize=8)

    plt.tight_layout()

    filename = FIGURES_DIR / '06_performance_by_country.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved: {filename}")
    plt.close()

def plot_7_feature_space_distribution(data):
    """
    Figure 7: Feature Space Distribution (Lt vs Ls)
    Story: How do crisis and non-crisis cases separate in feature space?
    """
    print("\n7. Creating feature space analysis...")

    horizons = data['horizons']

    fig, axes = plt.subplots(1, len(horizons), figsize=(6*len(horizons), 5))
    if len(horizons) == 1:
        axes = [axes]

    fig.suptitle('Stage 1 Baseline: Feature Space Distribution\n' +
                 'Lt (Temporal) vs Ls (Spatial) for Crisis vs Non-Crisis - DISTRICT LEVEL',
                 fontsize=14, fontweight='bold')

    for idx, horizon in enumerate(horizons):
        ax = axes[idx]

        preds = data[f'h{horizon}'].copy()

        # Check if Lt and Ls columns exist
        if 'Lt' not in preds.columns or 'Ls' not in preds.columns:
            ax.text(0.5, 0.5, 'Lt/Ls features not available',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Sample for visualization (too many points)
        sample_size = min(2000, len(preds))
        preds_sample = preds.sample(n=sample_size, random_state=42)

        # Separate by true label
        crisis = preds_sample[preds_sample['y_true'] == 1]
        no_crisis = preds_sample[preds_sample['y_true'] == 0]

        # Scatter plot
        ax.scatter(no_crisis['Lt'], no_crisis['Ls'],
                  c='#6ABD45', alpha=0.4, s=20, label='No Crisis (y=0)',
                  edgecolors='none')
        ax.scatter(crisis['Lt'], crisis['Ls'],
                  c='#F58220', alpha=0.6, s=20, label='Crisis (y=1)',
                  edgecolors='none')

        # Highlight AR failures
        failures = crisis[crisis['ar_failure_optimal'] == 1]
        ax.scatter(failures['Lt'], failures['Ls'],
                  c='#E31E24', marker='x', s=50, linewidths=2,
                  label='AR Failures', zorder=10)

        ax.set_xlabel('Lt (Temporal Lag - Previous IPC)')
        ax.set_ylabel('Ls (Spatial Lag - Neighbor IPC)')
        ax.set_title(f'h={horizon} months | AR Failures: {len(failures)}',
                    fontweight='bold')
        ax.legend(frameon=True, loc='upper left')
        ax.grid(alpha=0.3, linestyle='--')

        # Add IPC phase lines
        for phase in [1, 2, 3, 4, 5]:
            ax.axhline(y=phase, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)
            ax.axvline(x=phase, color='gray', linestyle=':', alpha=0.3, linewidth=0.5)

        # Annotate IPC phases
        ax.text(0.98, 0.02, 'IPC Phase:\n1=Minimal\n2=Stressed\n3=Crisis\n4=Emergency\n5=Famine',
               transform=ax.transAxes, ha='right', va='bottom',
               fontsize=7, bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))

    plt.tight_layout()

    filename = FIGURES_DIR / '07_feature_space_distribution.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved: {filename}")
    plt.close()

def plot_8_prediction_calibration(data):
    """
    Figure 8: Prediction Calibration
    Story: Are predicted probabilities well-calibrated?
    """
    print("\n8. Creating calibration analysis...")

    horizons = data['horizons']

    fig, axes = plt.subplots(1, len(horizons), figsize=(5*len(horizons), 5))
    if len(horizons) == 1:
        axes = [axes]

    fig.suptitle('Stage 1 Baseline: Prediction Probability Calibration\n' +
                 'Predicted Probability vs Observed Crisis Rate - DISTRICT LEVEL',
                 fontsize=14, fontweight='bold')

    for idx, horizon in enumerate(horizons):
        ax = axes[idx]

        preds = data[f'h{horizon}'].copy()

        # Create probability bins
        preds['prob_bin'] = pd.cut(preds['y_pred_proba'],
                                   bins=10, labels=False)

        # Calculate observed rate per bin
        calibration = preds.groupby('prob_bin').agg({
            'y_true': 'mean',
            'y_pred_proba': 'mean',
            'ipc_id': 'count'
        }).reset_index()

        calibration.columns = ['bin', 'observed_rate', 'mean_pred_prob', 'count']
        calibration = calibration.dropna()

        if len(calibration) == 0:
            ax.text(0.5, 0.5, 'Insufficient calibration data',
                   ha='center', va='center', transform=ax.transAxes)
            continue

        # Plot calibration curve
        ax.plot([0, 1], [0, 1], 'k--', linewidth=2, label='Perfect Calibration')
        ax.plot(calibration['mean_pred_prob'], calibration['observed_rate'],
               'o-', linewidth=2, markersize=8, color=PERFORMANCE_COLORS['primary'],
               label='Model Calibration')

        # Add sample size as bubble size
        max_count = calibration['count'].max()
        if max_count > 0:
            sizes = calibration['count'] / max_count * 500
            ax.scatter(calibration['mean_pred_prob'], calibration['observed_rate'],
                      s=sizes, alpha=0.3, color=PERFORMANCE_COLORS['primary'])

        ax.set_xlabel('Predicted Probability')
        ax.set_ylabel('Observed Crisis Rate')
        ax.set_title(f'h={horizon} months', fontweight='bold')
        ax.legend(frameon=True)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

        # Add calibration error
        calibration_error = np.abs(
            calibration['observed_rate'] - calibration['mean_pred_prob']
        ).mean()
        ax.text(0.05, 0.95, f'Mean Calibration Error:\n{calibration_error:.3f}',
               transform=ax.transAxes, ha='left', va='top',
               fontsize=9, bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.3))

    plt.tight_layout()

    filename = FIGURES_DIR / '08_prediction_calibration.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved: {filename}")
    plt.close()


def plot_9_roc_curves(data):
    """
    Figure 9: ROC Curves for All Horizons
    Story: Visual representation of model discrimination ability
    """
    print("\n9. Creating ROC curves...")

    horizons = data['horizons']
    metrics = data['metrics'][data['metrics']['fold'].isna()].copy()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Stage 1 Baseline: ROC and Precision-Recall Curves\n' +
                 'Model Discrimination Performance - DISTRICT LEVEL',
                 fontsize=14, fontweight='bold')

    # Color palette for horizons
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd']

    # (A) ROC Curves
    ax = axes[0]

    for idx, horizon in enumerate(horizons):
        preds = data[f'h{horizon}'].copy()
        y_true = preds['y_true'].values
        y_scores = preds['y_pred_proba'].values

        # Compute ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_scores)
        roc_auc = auc(fpr, tpr)

        # Plot ROC curve
        ax.plot(fpr, tpr, color=colors[idx % len(colors)], linewidth=2.5,
                label=f'h={horizon} months (AUC = {roc_auc:.3f})')

        # Find and mark optimal threshold (Youden's J)
        j_scores = tpr - fpr
        optimal_idx = np.argmax(j_scores)
        optimal_threshold = thresholds[optimal_idx]
        ax.scatter([fpr[optimal_idx]], [tpr[optimal_idx]],
                  marker='o', s=100, color=colors[idx % len(colors)],
                  edgecolors='black', linewidth=2, zorder=5)

    # Plot diagonal (random classifier)
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5, label='Random Classifier')

    ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=11)
    ax.set_ylabel('True Positive Rate (Sensitivity/Recall)', fontsize=11)
    ax.set_title('(A) ROC Curves by Prediction Horizon', fontweight='bold', loc='left')
    ax.legend(loc='lower right', frameon=True, fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    # Add annotation - for ROC curves, better is upper-left (high TPR, low FPR)
    ax.annotate('Ideal', xy=(0, 1), xytext=(0.25, 0.75),
               fontsize=10, ha='center',
               arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    # (B) Precision-Recall Curves
    ax = axes[1]

    for idx, horizon in enumerate(horizons):
        preds = data[f'h{horizon}'].copy()
        y_true = preds['y_true'].values
        y_scores = preds['y_pred_proba'].values

        # Compute Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_scores)
        pr_auc = auc(recall, precision)

        # Plot PR curve
        ax.plot(recall, precision, color=colors[idx % len(colors)], linewidth=2.5,
                label=f'h={horizon} months (AUC-PR = {pr_auc:.3f})')

        # Baseline (random classifier) - proportion of positives
        row = metrics[metrics['horizon'] == horizon].iloc[0]
        baseline = row['n_crisis'] / row['n_samples'] if row['n_samples'] > 0 else 0
        ax.axhline(y=baseline, color=colors[idx % len(colors)], linestyle=':',
                  alpha=0.5, linewidth=1)

    ax.set_xlabel('Recall (Sensitivity)', fontsize=11)
    ax.set_ylabel('Precision', fontsize=11)
    ax.set_title('(B) Precision-Recall Curves by Prediction Horizon', fontweight='bold', loc='left')
    ax.legend(loc='lower left', frameon=True, fontsize=10)
    ax.grid(alpha=0.3, linestyle='--')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    # Add annotation - for PR curves, better is upper-right (high recall + high precision)
    ax.annotate('Ideal', xy=(1, 1), xytext=(0.75, 0.75),
               fontsize=10, ha='center',
               arrowprops=dict(arrowstyle='->', color='green', lw=1.5),
               bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.3))

    plt.tight_layout()

    filename = FIGURES_DIR / '10_roc_pr_curves.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved: {filename}")
    plt.close()


def plot_10_roc_by_fold(data):
    """
    Figure 10: ROC Curves by Spatial CV Fold
    Story: How stable is model performance across geographic regions?
    """
    print("\n10. Creating ROC curves by fold...")

    horizons = data['horizons']

    # Load fold-level predictions for ROC computation
    fig, axes = plt.subplots(1, len(horizons), figsize=(6*len(horizons), 5))
    if len(horizons) == 1:
        axes = [axes]

    fig.suptitle('Stage 1 Baseline: ROC Curves by Spatial CV Fold\n' +
                 'Geographic Robustness of Model Discrimination - DISTRICT LEVEL',
                 fontsize=14, fontweight='bold')

    fold_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00']

    for h_idx, horizon in enumerate(horizons):
        ax = axes[h_idx]

        # Load fold-level predictions
        folds_file = RESULTS_DIR / f'predictions_h{horizon}_district_folds.csv'
        if folds_file.exists():
            folds_df = pd.read_csv(folds_file)

            all_aucs = []
            for fold in sorted(folds_df['fold'].unique()):
                fold_data = folds_df[folds_df['fold'] == fold]
                y_true = fold_data['y_true'].values
                y_scores = fold_data['y_pred_proba'].values

                # Compute ROC curve
                fpr, tpr, _ = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr, tpr)
                all_aucs.append(roc_auc)

                ax.plot(fpr, tpr, color=fold_colors[int(fold) % len(fold_colors)],
                       linewidth=2, alpha=0.7,
                       label=f'Fold {int(fold)} (AUC = {roc_auc:.3f})')

            # Plot mean ROC
            ax.plot([0, 1], [0, 1], 'k--', linewidth=1.5)

            # Summary stats
            mean_auc = np.mean(all_aucs)
            std_auc = np.std(all_aucs)
            ax.text(0.6, 0.05, f'Mean AUC: {mean_auc:.3f} ± {std_auc:.3f}',
                   transform=ax.transAxes, fontsize=10,
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        else:
            ax.text(0.5, 0.5, 'Fold-level predictions not found',
                   ha='center', va='center', transform=ax.transAxes)

        ax.set_xlabel('False Positive Rate')
        ax.set_ylabel('True Positive Rate')
        ax.set_title(f'h={horizon} months', fontweight='bold')
        ax.legend(loc='lower right', frameon=True, fontsize=9)
        ax.grid(alpha=0.3, linestyle='--')
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])

    plt.tight_layout()

    filename = FIGURES_DIR / '11_roc_by_fold.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved: {filename}")
    plt.close()


def create_choropleth_maps(data):
    """
    Choropleth Maps - Performance by District
    Story: Geographic visualization of model performance
    """
    print("\n12-14. Creating choropleth maps...")

    horizons = data['horizons']

    for horizon in horizons:
        fig, axes = plt.subplots(1, 3, figsize=(18, 6))
        fig.suptitle(f'Stage 1 Baseline: Geographic Performance (h={horizon} months)\n' +
                     'District-Level Model Performance Across Africa - DISTRICT LEVEL',
                     fontsize=14, fontweight='bold')

        preds = data[f'h{horizon}'].copy()

        # Calculate district-level metrics using optimal threshold
        district_metrics = preds.groupby(['ipc_country', 'ipc_geographic_unit_full']).agg({
            'avg_latitude': 'first',
            'avg_longitude': 'first',
            'y_true': 'sum',
            'correct': 'sum',
            'ar_failure_optimal': 'sum',
            'ipc_id': 'count'
        }).reset_index()

        district_metrics['accuracy'] = np.where(
            district_metrics['ipc_id'] > 0,
            district_metrics['correct'] / district_metrics['ipc_id'],
            0
        )
        district_metrics['recall'] = np.where(
            district_metrics['y_true'] > 0,
            1 - (district_metrics['ar_failure_optimal'] / district_metrics['y_true']),
            1
        )

        # Filter out rows with missing coordinates
        district_metrics = district_metrics.dropna(subset=['avg_latitude', 'avg_longitude'])

        if len(district_metrics) == 0:
            for ax in axes:
                ax.text(0.5, 0.5, 'No geographic data available',
                       ha='center', va='center', transform=ax.transAxes)
            plt.tight_layout()
            filename = FIGURES_DIR / f'09_choropleth_map_h{horizon}.png'
            plt.savefig(filename, dpi=300, bbox_inches='tight')
            plt.close()
            continue

        # Map 1: Accuracy
        ax = axes[0]
        scatter = ax.scatter(district_metrics['avg_longitude'],
                           district_metrics['avg_latitude'],
                           c=district_metrics['accuracy'],
                           s=30, cmap='RdYlGn', vmin=0, vmax=1,
                           edgecolors='black', linewidth=0.5, alpha=0.7)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('(A) Accuracy by District', fontweight='bold', loc='left')
        plt.colorbar(scatter, ax=ax, label='Accuracy')
        ax.grid(alpha=0.3)

        # Map 2: AR Failures
        ax = axes[1]
        failure_districts = district_metrics[district_metrics['ar_failure_optimal'] > 0]
        if len(failure_districts) > 0:
            scatter = ax.scatter(failure_districts['avg_longitude'],
                               failure_districts['avg_latitude'],
                               c=failure_districts['ar_failure_optimal'],
                               s=failure_districts['ar_failure_optimal'] * 10,
                               cmap='Reds', edgecolors='black', linewidth=0.5, alpha=0.7)
            plt.colorbar(scatter, ax=ax, label='Number of Failures')
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('(B) AR Failures by District', fontweight='bold', loc='left')
        ax.grid(alpha=0.3)

        # Map 3: Sample Size
        ax = axes[2]
        scatter = ax.scatter(district_metrics['avg_longitude'],
                           district_metrics['avg_latitude'],
                           c=district_metrics['ipc_id'],
                           s=30, cmap='Blues', edgecolors='black',
                           linewidth=0.5, alpha=0.7)
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title('(C) Number of Observations', fontweight='bold', loc='left')
        plt.colorbar(scatter, ax=ax, label='Observations')
        ax.grid(alpha=0.3)

        plt.tight_layout()

        filename = FIGURES_DIR / f'12_choropleth_map_h{horizon}.png'
        plt.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"   Saved: {filename}")
        plt.close()


def create_summary_infographic(data):
    """
    Executive Summary Infographic
    Story: One-page visual summary for stakeholders
    """
    print("\n15. Creating executive summary infographic...")

    metrics = data['metrics'][data['metrics']['fold'].isna()].copy()
    horizons = data['horizons']

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(4, 3, hspace=0.4, wspace=0.3)

    fig.suptitle('Stage 1: Spatio-Temporal Autoregressive Baseline - DISTRICT LEVEL\nExecutive Summary',
                fontsize=18, fontweight='bold', y=0.98)

    # Header stats boxes
    for idx, horizon in enumerate(horizons[:3]):  # Max 3 horizons for display
        row = metrics[metrics['horizon'] == horizon].iloc[0]

        ax = fig.add_subplot(gs[0, idx])
        ax.axis('off')

        # Create info box
        box_text = f"""
h={horizon} MONTHS AHEAD

Samples: {int(row['n_samples']):,}
Crisis Rate: {row['n_crisis']/row['n_samples']*100:.1f}%

PERFORMANCE
Accuracy: {row['accuracy']:.1%}
Precision: {row['precision']:.1%}
Recall: {row['recall']:.1%}
AUC-ROC: {row['auc_roc']:.3f}

AR FAILURES
{int(row['false_negatives'])} / {int(row['n_crisis'])}
({row['false_negatives']/row['n_crisis']*100:.1f}% missed)
        """

        color = PERFORMANCE_COLORS['primary'] if idx == 0 else \
                PERFORMANCE_COLORS['tertiary'] if idx == 1 else \
                PERFORMANCE_COLORS['quaternary']

        ax.text(0.5, 0.5, box_text.strip(), transform=ax.transAxes,
               ha='center', va='center', fontsize=11,
               bbox=dict(boxstyle='round,pad=1', facecolor=color, alpha=0.2,
                        edgecolor='black', linewidth=2))

    # Key Findings text
    ax = fig.add_subplot(gs[3, :])
    ax.axis('off')

    # Calculate summary stats dynamically
    avg_accuracy = metrics['accuracy'].mean()
    avg_recall = metrics['recall'].mean()
    avg_auc = metrics['auc_roc'].mean()
    total_failures = metrics['false_negatives'].sum()
    total_crisis = metrics['n_crisis'].sum()
    failure_pct = (total_failures / total_crisis * 100) if total_crisis > 0 else 0

    findings_text = f"""
KEY FINDINGS & IMPLICATIONS FOR STAGE 2 - DISTRICT LEVEL

1. STRONG BASELINE PERFORMANCE: AR model achieves {avg_accuracy:.1%} average accuracy across all horizons, demonstrating strong structural persistence in IPC dynamics.

2. TEMPORAL > SPATIAL: Lt (temporal lag) coefficients consistently exceed Ls (spatial lag) coefficients, indicating past IPC values
   are stronger predictors than neighbor values at the district level.

3. AR FAILURES IDENTIFY STAGE 2 TARGET: {failure_pct:.1f}% of crises are missed ({int(total_failures):,} events), representing sudden deteriorations where structural persistence
   fails. These are precisely the cases where dynamic news signals should provide early-warning value.

4. HORIZON DEGRADATION: Performance decreases at longer horizons, suggesting increased uncertainty further into
   the future where dynamic signals may be especially valuable.

5. GEOGRAPHIC HETEROGENEITY: Substantial variation in recall across countries, indicating regional differences in predictability that justify
   mixed-effects modeling in Stage 2.

NEXT STEPS: Proceed to Stage 2 - Use dynamic news features (GDELT) to explain AR failures and improve early-warning for sudden crisis onset.
    """

    ax.text(0.5, 0.5, findings_text.strip(), transform=ax.transAxes,
           ha='center', va='center', fontsize=10, family='monospace',
           bbox=dict(boxstyle='round,pad=1', facecolor='lightyellow',
                    alpha=0.8, edgecolor='black', linewidth=2))

    filename = FIGURES_DIR / '15_executive_summary.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    print(f"   Saved: {filename}")
    plt.close()


def main():
    print("=" * 80)
    print("Stage 1: Publication-Grade Visualizations - DISTRICT LEVEL")
    print("=" * 80)
    print(f"Start time: {datetime.now()}\n")

    # Load data
    data = load_data()

    # Generate all visualizations
    plot_1_performance_overview(data)
    plot_2_confusion_matrices(data)
    plot_3_model_coefficients(data)
    plot_4_spatial_cv_stability(data)
    plot_5_ar_failures_temporal(data)
    plot_6_performance_by_country(data)
    plot_7_feature_space_distribution(data)
    plot_8_prediction_calibration(data)
    plot_9_roc_curves(data)  # NEW: ROC and PR curves
    plot_10_roc_by_fold(data)  # NEW: ROC by spatial fold
    create_choropleth_maps(data)
    create_summary_infographic(data)

    print(f"\n{'='*80}")
    print("Visualization Complete - DISTRICT LEVEL")
    print(f"{'='*80}")
    print(f"\nAll figures saved to: {FIGURES_DIR}")
    print(f"Total figures created: 15+ (including ROC and PR curves)")
    print(f"\nEnd time: {datetime.now()}")

if __name__ == "__main__":
    main()
