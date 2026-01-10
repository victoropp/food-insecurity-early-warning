#!/usr/bin/env python3
"""
Ensemble Comparative Analysis
==============================
Deep dive into where AR excels vs where Stage 2 improves, and how ensemble combines them.

Key Questions:
1. Is the ensemble scientifically valid?
2. On the SAME dataset (6,553 obs), how do AR, Stage 2, and Ensemble compare?
3. Where does AR succeed? Where does Stage 2 improve?
4. Does the ensemble effectively combine both strengths?

Outputs:
- Comparative performance metrics on same dataset
- Error analysis: AR wins vs Stage 2 wins vs Ensemble wins
- Spatial maps showing complementary strengths
- District-level analysis
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, f1_score, precision_score, recall_score,
    roc_curve
)
import json
import sys

sys.path.insert(0, str(Path(__file__).parent.parent))
from config import RESULTS_DIR, PHASE4_RESULTS

# Output directory
OUTPUT_DIR = PHASE4_RESULTS / 'ensemble_analysis'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ENSEMBLE COMPARATIVE ANALYSIS")
print("=" * 80)
print(f"Timestamp: {datetime.now().isoformat()}")
print()

# =============================================================================
# LOAD DATA - All three models on SAME observations
# =============================================================================

print("Loading predictions from all three models...")

# Load ensemble predictions (contains all three)
ensemble_file = RESULTS_DIR / 'ensemble_stage1_stage2' / 'ensemble_predictions.csv'
ensemble_df = pd.read_csv(ensemble_file)

print(f"  Ensemble dataset: {len(ensemble_df):,} observations")
print(f"  Crisis events: {int(ensemble_df['y_true'].sum()):,} ({100*ensemble_df['y_true'].mean():.1f}%)")
print()

# =============================================================================
# PERFORMANCE COMPARISON ON SAME DATASET
# =============================================================================

print("-" * 80)
print("PERFORMANCE ON SAME 6,553 OBSERVATIONS (AR-filtered dataset)")
print("-" * 80)

y_true = ensemble_df['y_true'].values

# Stage 1 (AR Baseline)
ar_auc = roc_auc_score(y_true, ensemble_df['stage1_prob'])
ar_prauc = average_precision_score(y_true, ensemble_df['stage1_prob'])

# Find optimal threshold for AR using Youden
fpr, tpr, thresholds = roc_curve(y_true, ensemble_df['stage1_prob'])
youden_j = tpr - fpr
ar_threshold = thresholds[np.argmax(youden_j)]
ar_pred = (ensemble_df['stage1_prob'] >= ar_threshold).astype(int)
ar_f1 = f1_score(y_true, ar_pred)
ar_precision = precision_score(y_true, ar_pred, zero_division=0)
ar_recall = recall_score(y_true, ar_pred, zero_division=0)

# Stage 2 (XGBoost Advanced)
s2_auc = roc_auc_score(y_true, ensemble_df['stage2_prob'])
s2_prauc = average_precision_score(y_true, ensemble_df['stage2_prob'])

fpr, tpr, thresholds = roc_curve(y_true, ensemble_df['stage2_prob'])
youden_j = tpr - fpr
s2_threshold = thresholds[np.argmax(youden_j)]
s2_pred = (ensemble_df['stage2_prob'] >= s2_threshold).astype(int)
s2_f1 = f1_score(y_true, s2_pred)
s2_precision = precision_score(y_true, s2_pred, zero_division=0)
s2_recall = recall_score(y_true, s2_pred, zero_division=0)

# Ensemble
ens_auc = roc_auc_score(y_true, ensemble_df['ensemble_prob'])
ens_prauc = average_precision_score(y_true, ensemble_df['ensemble_prob'])
ens_pred = ensemble_df['ensemble_pred'].values
ens_f1 = f1_score(y_true, ens_pred)
ens_precision = precision_score(y_true, ens_pred, zero_division=0)
ens_recall = recall_score(y_true, ens_pred, zero_division=0)

# Print comparison table
print(f"\n{'Model':<25} {'AUC':<8} {'PR-AUC':<8} {'F1':<8} {'Precision':<10} {'Recall':<8}")
print("-" * 80)
print(f"{'Stage 1 (AR Baseline)':<25} {ar_auc:<8.4f} {ar_prauc:<8.4f} {ar_f1:<8.4f} {ar_precision:<10.4f} {ar_recall:<8.4f}")
print(f"{'Stage 2 (XGBoost)':<25} {s2_auc:<8.4f} {s2_prauc:<8.4f} {s2_f1:<8.4f} {s2_precision:<10.4f} {s2_recall:<8.4f}")
print(f"{'Ensemble':<25} {ens_auc:<8.4f} {ens_prauc:<8.4f} {ens_f1:<8.4f} {ens_precision:<10.4f} {ens_recall:<8.4f}")
print()

print("IMPROVEMENTS:")
print(f"  Stage 2 vs AR:     AUC {s2_auc - ar_auc:+.4f} ({100*(s2_auc - ar_auc)/ar_auc:+.2f}%)")
print(f"  Ensemble vs AR:    AUC {ens_auc - ar_auc:+.4f} ({100*(ens_auc - ar_auc)/ar_auc:+.2f}%)")
print(f"  Ensemble vs Stage 2: AUC {ens_auc - s2_auc:+.4f} ({100*(ens_auc - s2_auc)/s2_auc:+.2f}%)")
print()

# =============================================================================
# ERROR ANALYSIS: Who predicts what correctly?
# =============================================================================

print("-" * 80)
print("ERROR ANALYSIS: Complementary Strengths")
print("-" * 80)

# Calculate prediction errors (absolute probability errors)
ensemble_df['ar_error'] = np.abs(ensemble_df['stage1_prob'] - ensemble_df['y_true'])
ensemble_df['s2_error'] = np.abs(ensemble_df['stage2_prob'] - ensemble_df['y_true'])
ensemble_df['ens_error'] = np.abs(ensemble_df['ensemble_prob'] - ensemble_df['y_true'])

# Who performs best on each observation?
ensemble_df['best_model'] = ensemble_df[['ar_error', 's2_error', 'ens_error']].idxmin(axis=1)
ensemble_df['best_model'] = ensemble_df['best_model'].map({
    'ar_error': 'AR Baseline',
    's2_error': 'Stage 2',
    'ens_error': 'Ensemble'
})

# Count wins
best_counts = ensemble_df['best_model'].value_counts()
print("\nObservations where each model is most accurate:")
for model, count in best_counts.items():
    print(f"  {model:<20}: {count:>5,} ({100*count/len(ensemble_df):>5.1f}%)")
print()

# AR vs Stage 2 head-to-head (ignoring ensemble for now)
ensemble_df['ar_vs_s2_winner'] = np.where(
    ensemble_df['ar_error'] < ensemble_df['s2_error'],
    'AR Better',
    np.where(
        ensemble_df['s2_error'] < ensemble_df['ar_error'],
        'Stage 2 Better',
        'Tie'
    )
)

winner_counts = ensemble_df['ar_vs_s2_winner'].value_counts()
print("AR vs Stage 2 head-to-head:")
for winner, count in winner_counts.items():
    print(f"  {winner:<20}: {count:>5,} ({100*count/len(ensemble_df):>5.1f}%)")
print()

# When does ensemble beat both?
ensemble_df['ensemble_best'] = (
    (ensemble_df['ens_error'] < ensemble_df['ar_error']) &
    (ensemble_df['ens_error'] < ensemble_df['s2_error'])
)
n_ensemble_best = ensemble_df['ensemble_best'].sum()
print(f"Ensemble outperforms BOTH: {n_ensemble_best:,} ({100*n_ensemble_best/len(ensemble_df):.1f}%)")
print()

# =============================================================================
# SPATIAL ANALYSIS: Geographic patterns
# =============================================================================

print("-" * 80)
print("SPATIAL ANALYSIS: Where does each model excel?")
print("-" * 80)

# Group by country
country_analysis = ensemble_df.groupby('geographic_unit').agg({
    'y_true': 'sum',  # Total crises
    'ar_error': 'mean',
    's2_error': 'mean',
    'ens_error': 'mean',
    'geographic_unit': 'count'  # Sample size
}).rename(columns={'geographic_unit': 'n_obs'})

country_analysis['ar_auc'] = ensemble_df.groupby('geographic_unit').apply(
    lambda x: roc_auc_score(x['y_true'], x['stage1_prob']) if x['y_true'].sum() > 0 and x['y_true'].sum() < len(x) else np.nan
)
country_analysis['s2_auc'] = ensemble_df.groupby('geographic_unit').apply(
    lambda x: roc_auc_score(x['y_true'], x['stage2_prob']) if x['y_true'].sum() > 0 and x['y_true'].sum() < len(x) else np.nan
)
country_analysis['ens_auc'] = ensemble_df.groupby('geographic_unit').apply(
    lambda x: roc_auc_score(x['y_true'], x['ensemble_prob']) if x['y_true'].sum() > 0 and x['y_true'].sum() < len(x) else np.nan
)

# Determine winner per country
country_analysis['best_model'] = country_analysis[['ar_auc', 's2_auc', 'ens_auc']].idxmax(axis=1)
country_analysis['best_model'] = country_analysis['best_model'].map({
    'ar_auc': 'AR',
    's2_auc': 'Stage 2',
    'ens_auc': 'Ensemble'
})

print(f"\nCountries analyzed: {len(country_analysis)}")
print("\nBest model by country (by AUC):")
print(country_analysis['best_model'].value_counts())
print()

# Save spatial analysis
country_analysis.to_csv(OUTPUT_DIR / 'country_comparative_performance.csv')
print(f"[OK] Saved: country_comparative_performance.csv")

# =============================================================================
# CRISIS TYPE ANALYSIS: When does AR fail vs succeed?
# =============================================================================

print("-" * 80)
print("CRISIS ANALYSIS: AR performance by crisis severity")
print("-" * 80)

# Separate by actual outcome
crisis_obs = ensemble_df[ensemble_df['y_true'] == 1].copy()
non_crisis_obs = ensemble_df[ensemble_df['y_true'] == 0].copy()

print(f"\nCrisis observations (y=1): {len(crisis_obs):,}")
print(f"  AR mean prob:       {crisis_obs['stage1_prob'].mean():.3f}")
print(f"  Stage 2 mean prob:  {crisis_obs['stage2_prob'].mean():.3f}")
print(f"  Ensemble mean prob: {crisis_obs['ensemble_prob'].mean():.3f}")

print(f"\nNon-crisis observations (y=0): {len(non_crisis_obs):,}")
print(f"  AR mean prob:       {non_crisis_obs['stage1_prob'].mean():.3f}")
print(f"  Stage 2 mean prob:  {non_crisis_obs['stage2_prob'].mean():.3f}")
print(f"  Ensemble mean prob: {non_crisis_obs['ensemble_prob'].mean():.3f}")
print()

# =============================================================================
# SAVE COMPREHENSIVE RESULTS
# =============================================================================

print("-" * 80)
print("SAVING COMPREHENSIVE RESULTS")
print("-" * 80)

# Save detailed comparison
comparison_summary = {
    'timestamp': datetime.now().isoformat(),
    'dataset': {
        'description': 'AR-filtered dataset (IPC<=2 & AR=0)',
        'n_observations': int(len(ensemble_df)),
        'n_crises': int(y_true.sum()),
        'crisis_rate': float(y_true.mean())
    },
    'performance_same_dataset': {
        'stage1_ar': {
            'auc_roc': float(ar_auc),
            'pr_auc': float(ar_prauc),
            'f1': float(ar_f1),
            'precision': float(ar_precision),
            'recall': float(ar_recall)
        },
        'stage2_xgboost': {
            'auc_roc': float(s2_auc),
            'pr_auc': float(s2_prauc),
            'f1': float(s2_f1),
            'precision': float(s2_precision),
            'recall': float(s2_recall)
        },
        'ensemble': {
            'auc_roc': float(ens_auc),
            'pr_auc': float(ens_prauc),
            'f1': float(ens_f1),
            'precision': float(ens_precision),
            'recall': float(ens_recall)
        }
    },
    'improvements': {
        'stage2_vs_ar_auc': float(s2_auc - ar_auc),
        'stage2_vs_ar_pct': float(100 * (s2_auc - ar_auc) / ar_auc),
        'ensemble_vs_ar_auc': float(ens_auc - ar_auc),
        'ensemble_vs_ar_pct': float(100 * (ens_auc - ar_auc) / ar_auc),
        'ensemble_vs_stage2_auc': float(ens_auc - s2_auc),
        'ensemble_vs_stage2_pct': float(100 * (ens_auc - s2_auc) / s2_auc)
    },
    'complementary_strengths': {
        'observations_where_ar_better': int((ensemble_df['ar_vs_s2_winner'] == 'AR Better').sum()),
        'observations_where_s2_better': int((ensemble_df['ar_vs_s2_winner'] == 'Stage 2 Better').sum()),
        'observations_where_ensemble_best': int(n_ensemble_best),
        'ensemble_best_pct': float(100 * n_ensemble_best / len(ensemble_df))
    },
    'best_model_by_observation': {
        model: int(count) for model, count in best_counts.items()
    },
    'spatial_summary': {
        'n_countries': int(len(country_analysis)),
        'countries_where_ar_best': int((country_analysis['best_model'] == 'AR').sum()),
        'countries_where_s2_best': int((country_analysis['best_model'] == 'Stage 2').sum()),
        'countries_where_ens_best': int((country_analysis['best_model'] == 'Ensemble').sum())
    }
}

with open(OUTPUT_DIR / 'ensemble_comparative_summary.json', 'w') as f:
    json.dump(comparison_summary, f, indent=2)
print(f"[OK] Saved: ensemble_comparative_summary.json")

# Save observation-level analysis
obs_analysis = ensemble_df[[
    'geographic_unit', 'date', 'y_true',
    'stage1_prob', 'stage2_prob', 'ensemble_prob',
    'ar_error', 's2_error', 'ens_error',
    'best_model', 'ar_vs_s2_winner', 'ensemble_best'
]].copy()
obs_analysis.to_csv(OUTPUT_DIR / 'observation_level_comparison.csv', index=False)
print(f"[OK] Saved: observation_level_comparison.csv")

print()
print("=" * 80)
print("ENSEMBLE COMPARATIVE ANALYSIS COMPLETE")
print("=" * 80)
print()
print("KEY FINDINGS:")
print(f"  1. On the same 6,553 observations:")
print(f"     - AR Baseline:  AUC = {ar_auc:.4f}")
print(f"     - Stage 2:      AUC = {s2_auc:.4f} ({100*(s2_auc-ar_auc)/ar_auc:+.1f}% vs AR)")
print(f"     - Ensemble:     AUC = {ens_auc:.4f} ({100*(ens_auc-s2_auc)/s2_auc:+.1f}% vs Stage 2)")
print()
print(f"  2. Ensemble outperforms BOTH individual models on {100*n_ensemble_best/len(ensemble_df):.1f}% of observations")
print()
print(f"  3. Stage 2 outperforms AR on {100*(ensemble_df['ar_vs_s2_winner'] == 'Stage 2 Better').sum()/len(ensemble_df):.1f}% of observations")
print()
print("=" * 80)
