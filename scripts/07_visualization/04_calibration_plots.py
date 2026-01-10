"""
Visualization: Calibration Plots
=================================
Create calibration curves and reliability diagrams for ALL models (XGBoost + Mixed Effects).

This script automatically detects all models from model_ranking_table.csv and creates
a multi-panel figure showing calibration curves for each model.

Date: December 24, 2025
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))
from config import (BASE_DIR, RESULTS_DIR, STAGE2_MODELS_DIR, FIGURES_DIR,
                    VISUALIZATION_CONFIG)

sys.path.append(str(Path(__file__).parent))
from utils_dynamic import get_model_path_mapping

OUTPUT_DIR = FIGURES_DIR / 'calibration'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("VISUALIZATION: CALIBRATION PLOTS (ALL MODELS)")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# ============================================================================
# LOAD ALL MODELS FROM MODEL RANKING TABLE
# ============================================================================

print("1. Loading model ranking table...")
ranking_file = RESULTS_DIR / 'analysis' / 'model_ranking_table.csv'

if not ranking_file.exists():
    print(f"ERROR: Model ranking table not found: {ranking_file}")
    sys.exit(1)

ranking_df = pd.read_csv(ranking_file)
all_models = ranking_df['model'].tolist()
print(f"   Found {len(all_models)} models:\n")
for idx, model in enumerate(all_models, 1):
    print(f"   {idx}. {model}")

# Get model path mapping
print("\n   Loading model path mapping...")
model_mapping = get_model_path_mapping(RESULTS_DIR)
print(f"   Found prediction files for {len(model_mapping)} models")

# ============================================================================
# CREATE MULTI-PANEL CALIBRATION FIGURE
# ============================================================================

print("\n2. Creating calibration plots...")

# Calculate grid layout
n_models = len(all_models)
n_cols = 3
n_rows = (n_models + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
axes = axes.flatten()  # Flatten to 1D for easy indexing

models_plotted = 0

for idx, model_name in enumerate(all_models):
    ax = axes[idx]

    # Get prediction file from mapping
    if model_name not in model_mapping:
        print(f"\n   Processing: {model_name}")
        print(f"   [SKIP] No prediction file mapping found")
        ax.text(0.5, 0.5, f'Model: {model_name}\n\nPredictions not found',
                ha='center', va='center', fontsize=10, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'{model_name}\n(No Data)', fontsize=9)
        continue

    pred_file = model_mapping[model_name]['prediction_file']
    print(f"\n   Processing: {model_name}")
    print(f"   Predictions: {pred_file}")

    if not pred_file.exists():
        print(f"   [WARN]  Predictions file not found")
        ax.text(0.5, 0.5, f'Model: {model_name}\n\nPredictions not found',
                ha='center', va='center', fontsize=10, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'{model_name}\n(No Data)', fontsize=9)
        continue

    # Load predictions
    try:
        predictions = pd.read_csv(pred_file)
    except Exception as e:
        print(f"   [WARN]  Error loading predictions: {e}")
        ax.text(0.5, 0.5, f'Model: {model_name}\n\nError loading data',
                ha='center', va='center', fontsize=10, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'{model_path}\n(Load Error)', fontsize=9)
        continue

    # Determine column names dynamically - try multiple possibilities
    y_true_col = None
    for col in ['y_true', 'future_crisis', 'ipc_future_crisis', 'target', 'label']:
        if col in predictions.columns:
            y_true_col = col
            break

    y_pred_col = None
    for col in ['ensemble_prob', 'pred_prob', 'y_pred_prob', 'probability', 'prob']:
        if col in predictions.columns:
            y_pred_col = col
            break

    # Check if required columns exist
    if y_true_col is None or y_pred_col is None:
        print(f"   [WARN]  Required columns not found")
        print(f"   Available columns: {list(predictions.columns)}")
        ax.text(0.5, 0.5, f'Model: {model_name}\n\nMissing columns',
                ha='center', va='center', fontsize=10, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'{model_name}\n(Column Error)', fontsize=9)
        continue

    y_true = predictions[y_true_col].values
    y_pred_proba = predictions[y_pred_col].values

    # Compute calibration curve
    try:
        prob_true, prob_pred = calibration_curve(y_true, y_pred_proba, n_bins=10, strategy='uniform')
        brier = brier_score_loss(y_true, y_pred_proba)

        # Plot perfect calibration line
        ax.plot([0, 1], [0, 1], 'k--', label='Perfect calibration', linewidth=1.5, alpha=0.7)

        # Plot model calibration
        model_lower = model_name.lower()
        if 'xgboost' in model_lower:
            color = 'blue'
            category_display = 'XGBoost'
        elif 'mixed' in model_lower or 'pooled' in model_lower:
            color = 'green'
            category_display = 'Mixed Effects'
        elif 'ensemble' in model_lower:
            color = 'purple'
            category_display = 'Ensemble'
        elif 'baseline' in model_lower or 'ar' in model_lower:
            color = 'orange'
            category_display = 'Baseline'
        else:
            color = 'gray'
            category_display = 'Other'

        # Shortened model name for display
        short_name = model_name.replace('Stage 2 ', '').replace('Stage 1 ', '')
        ax.plot(prob_pred, prob_true, 's-', label=short_name, linewidth=2, markersize=6,
                color=color, alpha=0.8)

        ax.set_xlabel('Mean Predicted Probability', fontsize=9)
        ax.set_ylabel('Fraction of Positives', fontsize=9)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)

        # Title with model name and Brier score
        ax.set_title(f'{category_display}: {short_name}\nBrier Score: {brier:.4f}', fontsize=9)

        print(f"   [OK] Calibration curve created (Brier: {brier:.4f})")
        models_plotted += 1

    except Exception as e:
        print(f"   [WARN]  Error creating calibration curve: {e}")
        ax.text(0.5, 0.5, f'Model: {model_name}\n\nCalibration Error',
                ha='center', va='center', fontsize=10, color='red')
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'{model_path}\n(Calibration Error)', fontsize=9)

# Hide unused subplots
for idx in range(len(all_models), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Model Calibration Curves - All Models', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

fig_file = OUTPUT_DIR / 'calibration_plots_all_models.png'
plt.savefig(fig_file, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
plt.close()

print(f"\n[OK] Saved: {fig_file}")
print(f"   Models plotted: {models_plotted} / {len(all_models)}")

# ============================================================================
# CREATE RELIABILITY DIAGRAM COMPARISON
# ============================================================================

print("\n3. Creating reliability diagrams...")

fig, axes = plt.subplots(n_rows, n_cols, figsize=(18, 5*n_rows))
axes = axes.flatten()

models_plotted = 0

for idx, model_name in enumerate(all_models):
    ax = axes[idx]

    # Get prediction file from mapping
    if model_name not in model_mapping:
        ax.text(0.5, 0.5, f'Model: {model_name}\n\nPredictions not found',
                ha='center', va='center', fontsize=10, color='red')
        ax.set_title(f'{model_name}\n(No Data)', fontsize=9)
        continue

    pred_file = model_mapping[model_name]['prediction_file']

    if not pred_file.exists():
        ax.text(0.5, 0.5, f'Model: {model_name}\n\nPredictions not found',
                ha='center', va='center', fontsize=10, color='red')
        ax.set_title(f'{model_name}\n(No Data)', fontsize=9)
        continue

    try:
        predictions = pd.read_csv(pred_file)

        # Determine column names dynamically
        y_true_col = None
        for col in ['y_true', 'future_crisis', 'ipc_future_crisis', 'target', 'label']:
            if col in predictions.columns:
                y_true_col = col
                break

        y_pred_col = None
        for col in ['ensemble_prob', 'pred_prob', 'y_pred_prob', 'probability', 'prob']:
            if col in predictions.columns:
                y_pred_col = col
                break

        if y_true_col is None or y_pred_col is None:
            ax.text(0.5, 0.5, f'Model: {model_name}\n\nMissing columns',
                    ha='center', va='center', fontsize=10, color='red')
            ax.set_title(f'{model_path}\n(Column Error)', fontsize=9)
            continue

        y_true = predictions[y_true_col].values
        y_pred_proba = predictions[y_pred_col].values

        brier = brier_score_loss(y_true, y_pred_proba)

        # Reliability histogram
        ax.hist(y_pred_proba[y_true == 0], bins=20, alpha=0.5, label='No Crisis', density=True, color='green')
        ax.hist(y_pred_proba[y_true == 1], bins=20, alpha=0.5, label='Crisis', density=True, color='red')
        ax.set_xlabel('Predicted Probability', fontsize=9)
        ax.set_ylabel('Density', fontsize=9)
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

        # Determine category
        model_lower = model_name.lower()
        if 'xgboost' in model_lower:
            category_display = 'XGBoost'
        elif 'mixed' in model_lower or 'pooled' in model_lower:
            category_display = 'Mixed Effects'
        elif 'ensemble' in model_lower:
            category_display = 'Ensemble'
        elif 'baseline' in model_lower or 'ar' in model_lower:
            category_display = 'Baseline'
        else:
            category_display = 'Other'

        short_name = model_name.replace('Stage 2 ', '').replace('Stage 1 ', '')
        ax.set_title(f'{category_display}: {short_name}\nBrier Score: {brier:.4f}', fontsize=9)

        models_plotted += 1

    except Exception as e:
        ax.text(0.5, 0.5, f'Model: {model_name}\n\nError: {str(e)[:50]}',
                ha='center', va='center', fontsize=10, color='red')
        ax.set_title(f'{model_path}\n(Error)', fontsize=9)

# Hide unused subplots
for idx in range(len(all_models), len(axes)):
    axes[idx].axis('off')

plt.suptitle('Reliability Diagrams - All Models', fontsize=14, fontweight='bold', y=0.995)
plt.tight_layout()

fig_file = OUTPUT_DIR / 'reliability_diagrams_all_models.png'
plt.savefig(fig_file, dpi=VISUALIZATION_CONFIG['dpi'], bbox_inches='tight')
plt.close()

print(f"[OK] Saved: {fig_file}")
print(f"   Models plotted: {models_plotted} / {len(all_models)}")

print("\n" + "=" * 80)
print("CALIBRATION PLOTS COMPLETE")
print("=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\nOutput directory: {OUTPUT_DIR}")
print("Files created:")
print("  - calibration_plots_all_models.png")
print("  - reliability_diagrams_all_models.png")
