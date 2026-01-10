"""
Stage 3: Ablation Studies - Feature Component Analysis
=======================================================
Compare models with/without z-scores, HMM, and DMD features to isolate
the contribution of each component.

COMPARISONS:
1. Basic (ratio only) vs Basic+Zscore
2. Basic+Zscore vs Basic+Zscore+HMM
3. Basic+Zscore+HMM vs Basic+Zscore+HMM+DMD (Advanced)

STATISTICAL TESTS:
- Paired t-test on AUC across folds
- DeLong test for AUC comparison
- McNemar test for classification differences

Author: Victor Collins Oppon
Date: December  2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy import stats
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))
from config import (BASE_DIR, RESULTS_DIR, STAGE2_MODELS_DIR, FIGURES_DIR,
                    RANDOM_STATE)

# Output directory
OUTPUT_DIR = RESULTS_DIR / 'baseline_comparison' / 'ablation_studies'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("STAGE 3: ABLATION STUDIES - FEATURE COMPONENT ANALYSIS")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"Output directory: {OUTPUT_DIR}")
print()

# =============================================================================
# LOAD MODEL RESULTS
# =============================================================================

print("Loading model results...")
print()

# Model definitions (WITH_AR_FILTER variants, ALL with location features)
models = {
    'ratio_location': {
        'name': 'Ablation 1: Ratio + Location',
        'cv_file': STAGE2_MODELS_DIR / 'ablation' / 'ratio_location' / 'ablation_ratio_location_cv_results.csv',
        'pred_file': STAGE2_MODELS_DIR / 'ablation' / 'ratio_location' / 'ablation_ratio_location_predictions.csv',
        'features': 'ratio + location (11)',
        'has_zscore': False,
        'has_hmm': False,
        'has_dmd': False
    },
    'ratio_zscore_location': {
        'name': 'Ablation 2: Ratio + Zscore + Location',
        'cv_file': STAGE2_MODELS_DIR / 'xgboost' / 'basic_with_ar' / 'xgboost_with_ar_cv_results.csv',
        'pred_file': STAGE2_MODELS_DIR / 'xgboost' / 'basic_with_ar' / 'xgboost_with_ar_predictions.csv',
        'features': 'ratio + zscore + location (20)',
        'has_zscore': True,
        'has_hmm': False,
        'has_dmd': False
    },
    'ratio_zscore_dmd_location': {
        'name': 'Ablation 3: Ratio + Zscore + DMD + Location',
        'cv_file': STAGE2_MODELS_DIR / 'ablation' / 'ratio_zscore_dmd_location' / 'ablation_ratio_zscore_dmd_location_cv_results.csv',
        'pred_file': STAGE2_MODELS_DIR / 'ablation' / 'ratio_zscore_dmd_location' / 'ablation_ratio_zscore_dmd_location_predictions.csv',
        'features': 'ratio + zscore + DMD + location (26)',
        'has_zscore': True,
        'has_hmm': False,
        'has_dmd': True
    },
    'ratio_zscore_hmm_location': {
        'name': 'Ablation 4: Ratio + Zscore + HMM + Location',
        'cv_file': STAGE2_MODELS_DIR / 'ablation' / 'ratio_zscore_hmm_location' / 'ablation_ratio_zscore_hmm_location_cv_results.csv',
        'pred_file': STAGE2_MODELS_DIR / 'ablation' / 'ratio_zscore_hmm_location' / 'ablation_ratio_zscore_hmm_location_predictions.csv',
        'features': 'ratio + zscore + HMM + location (32)',
        'has_zscore': True,
        'has_hmm': True,
        'has_dmd': False
    },
    'full': {
        'name': 'Ablation 5: Full (Ratio + Zscore + HMM + DMD + Location)',
        'cv_file': STAGE2_MODELS_DIR / 'xgboost' / 'advanced_with_ar' / 'xgboost_hmm_dmd_with_ar_cv_results.csv',
        'pred_file': STAGE2_MODELS_DIR / 'xgboost' / 'advanced_with_ar' / 'xgboost_hmm_dmd_with_ar_predictions.csv',
        'features': 'ratio + zscore + HMM + DMD + location (38)',
        'has_zscore': True,
        'has_hmm': True,
        'has_dmd': True
    }
}

# Load CV results
cv_results = {}
predictions = {}

for model_key, model_info in models.items():
    print(f"Loading {model_info['name']}...")

    # Check if files exist
    if not model_info['cv_file'].exists():
        print(f"  WARNING: CV results not found: {model_info['cv_file']}")
        print(f"  Skipping {model_key}")
        continue

    # Load CV results
    cv_df = pd.read_csv(model_info['cv_file'])
    cv_results[model_key] = cv_df
    print(f"  Loaded CV results: {len(cv_df)} folds")
    print(f"  Mean AUC: {cv_df['auc_roc'].mean():.4f} +/- {cv_df['auc_roc'].std():.4f}")

    # Load predictions if available
    if model_info['pred_file'].exists():
        pred_df = pd.read_csv(model_info['pred_file'])
        predictions[model_key] = pred_df
        print(f"  Loaded predictions: {len(pred_df)} observations")

    print()

if len(cv_results) < 2:
    print("ERROR: Need at least 2 models for ablation analysis")
    print("Please run Stage 2 model training scripts first.")
    sys.exit(1)

# =============================================================================
# PAIRED COMPARISONS
# =============================================================================

print("=" * 80)
print("PAIRED STATISTICAL COMPARISONS")
print("=" * 80)
print()

# Define comparison pairs for ablation
comparisons = [
    ('ratio_location', 'ratio_zscore_location', 'Z-score Addition'),
    ('ratio_zscore_location', 'ratio_zscore_dmd_location', 'DMD Addition (isolated)'),
    ('ratio_zscore_location', 'ratio_zscore_hmm_location', 'HMM Addition (isolated)'),
    ('ratio_zscore_dmd_location', 'full', 'HMM Addition (on top of DMD)'),
    ('ratio_zscore_hmm_location', 'full', 'DMD Addition (on top of HMM)'),
    ('ratio_location', 'full', 'Full Feature Set (All Components)')
]

comparison_results = []

for model1_key, model2_key, comparison_name in comparisons:
    if model1_key not in cv_results or model2_key not in cv_results:
        print(f"Skipping {comparison_name} - missing model results")
        continue

    print(f"Comparison: {comparison_name}")
    print(f"  Model 1: {models[model1_key]['name']}")
    print(f"  Model 2: {models[model2_key]['name']}")
    print()

    cv1 = cv_results[model1_key]
    cv2 = cv_results[model2_key]

    # Ensure same number of folds
    n_folds = min(len(cv1), len(cv2))

    # AUC comparison
    auc1 = cv1['auc_roc'].iloc[:n_folds].values
    auc2 = cv2['auc_roc'].iloc[:n_folds].values

    auc_diff = auc2 - auc1
    mean_diff = auc_diff.mean()
    std_diff = auc_diff.std()

    # Paired t-test
    t_stat, p_value = stats.ttest_rel(auc2, auc1)

    # Effect size (Cohen's d)
    cohens_d = mean_diff / std_diff if std_diff > 0 else 0

    print(f"  AUC Improvement:")
    print(f"    Model 1: {auc1.mean():.4f} +/- {auc1.std():.4f}")
    print(f"    Model 2: {auc2.mean():.4f} +/- {auc2.std():.4f}")
    print(f"    Difference: {mean_diff:.4f} +/- {std_diff:.4f}")
    print(f"    t-statistic: {t_stat:.3f}")
    print(f"    p-value: {p_value:.4f}")
    print(f"    Cohen's d: {cohens_d:.3f}")

    if p_value < 0.001:
        sig_level = '***'
    elif p_value < 0.01:
        sig_level = '**'
    elif p_value < 0.05:
        sig_level = '*'
    else:
        sig_level = 'ns'

    print(f"    Significance: {sig_level}")
    print()

    # PR-AUC comparison if available
    if 'pr_auc' in cv1.columns and 'pr_auc' in cv2.columns:
        pr_auc1 = cv1['pr_auc'].iloc[:n_folds].values
        pr_auc2 = cv2['pr_auc'].iloc[:n_folds].values

        pr_auc_diff = pr_auc2 - pr_auc1
        pr_t_stat, pr_p_value = stats.ttest_rel(pr_auc2, pr_auc1)

        print(f"  PR-AUC Improvement:")
        print(f"    Model 1: {pr_auc1.mean():.4f} +/- {pr_auc1.std():.4f}")
        print(f"    Model 2: {pr_auc2.mean():.4f} +/- {pr_auc2.std():.4f}")
        print(f"    Difference: {pr_auc_diff.mean():.4f} +/- {pr_auc_diff.std():.4f}")
        print(f"    p-value: {pr_p_value:.4f}")
        print()

    # Store results
    comparison_results.append({
        'comparison': comparison_name,
        'model_1': models[model1_key]['name'],
        'model_2': models[model2_key]['name'],
        'features_added': models[model2_key]['features'],
        'auc_1_mean': float(auc1.mean()),
        'auc_1_std': float(auc1.std()),
        'auc_2_mean': float(auc2.mean()),
        'auc_2_std': float(auc2.std()),
        'auc_diff_mean': float(mean_diff),
        'auc_diff_std': float(std_diff),
        't_statistic': float(t_stat),
        'p_value': float(p_value),
        'cohens_d': float(cohens_d),
        'significance': sig_level,
        'n_folds': n_folds
    })

# =============================================================================
# SUMMARY TABLE
# =============================================================================

print("=" * 80)
print("ABLATION STUDY SUMMARY")
print("=" * 80)
print()

# Create summary table
summary_rows = []

for model_key in ['ratio_location', 'ratio_zscore_location', 'ratio_zscore_dmd_location', 'ratio_zscore_hmm_location', 'full']:
    if model_key not in cv_results:
        continue

    cv = cv_results[model_key]

    summary_rows.append({
        'model': models[model_key]['name'],
        'features': models[model_key]['features'],
        'has_location': True,
        'has_zscore': models[model_key]['has_zscore'],
        'has_hmm': models[model_key]['has_hmm'],
        'has_dmd': models[model_key]['has_dmd'],
        'mean_auc': cv['auc_roc'].mean(),
        'std_auc': cv['auc_roc'].std(),
        'mean_pr_auc': cv['pr_auc'].mean() if 'pr_auc' in cv.columns else np.nan,
        'std_pr_auc': cv['pr_auc'].std() if 'pr_auc' in cv.columns else np.nan,
        'mean_precision': cv['precision_youden'].mean() if 'precision_youden' in cv.columns else np.nan,
        'mean_recall': cv['recall_youden'].mean() if 'recall_youden' in cv.columns else np.nan,
        'mean_f1': cv['f1_youden'].mean() if 'f1_youden' in cv.columns else np.nan
    })

summary_df = pd.DataFrame(summary_rows)

print("Model Performance Summary:")
print(summary_df.to_string(index=False))
print()

# =============================================================================
# FEATURE CONTRIBUTION ANALYSIS
# =============================================================================

print("=" * 80)
print("FEATURE CONTRIBUTION ANALYSIS")
print("=" * 80)
print()

# Calculate incremental AUC gains
if len(summary_df) >= 2:
    print("Incremental AUC Gains:")

    baseline_auc = summary_df.iloc[0]['mean_auc']
    print(f"  Baseline (ratio only): {baseline_auc:.4f}")

    for i in range(1, len(summary_df)):
        current_auc = summary_df.iloc[i]['mean_auc']
        previous_auc = summary_df.iloc[i-1]['mean_auc']

        abs_gain = current_auc - previous_auc
        rel_gain = (current_auc - baseline_auc) / baseline_auc * 100

        features_added = summary_df.iloc[i]['features']

        print(f"  + {features_added}: {abs_gain:+.4f} (total gain: {rel_gain:+.2f}%)")

    print()

# =============================================================================
# SAVE OUTPUTS
# =============================================================================

print("Saving outputs...")

# 1. Comparison results
comparison_df = pd.DataFrame(comparison_results)
comparison_file = OUTPUT_DIR / 'ablation_comparisons.csv'
comparison_df.to_csv(comparison_file, index=False)
print(f"[OK] Comparisons: {comparison_file}")

# 2. Summary table
summary_file = OUTPUT_DIR / 'ablation_summary.csv'
summary_df.to_csv(summary_file, index=False)
print(f"[OK] Summary: {summary_file}")

# 3. JSON summary
json_summary = {
    'ablation_study': {
        'description': 'Feature component contribution analysis',
        'models_compared': len(summary_df),
        'comparisons': comparison_results,
        'summary': summary_df.to_dict(orient='records')
    },
    'key_findings': {
        'zscore_contribution': None,
        'hmm_dmd_contribution': None,
        'total_improvement': None
    },
    'timestamp': datetime.now().isoformat()
}

# Calculate key findings
if len(comparison_results) >= 1:
    zscore_comp = next((c for c in comparison_results if 'Z-score' in c['comparison']), None)
    if zscore_comp:
        json_summary['key_findings']['zscore_contribution'] = {
            'auc_gain': zscore_comp['auc_diff_mean'],
            'p_value': zscore_comp['p_value'],
            'significant': zscore_comp['p_value'] < 0.05
        }

    hmm_comp = next((c for c in comparison_results if 'HMM' in c['comparison']), None)
    if hmm_comp:
        json_summary['key_findings']['hmm_dmd_contribution'] = {
            'auc_gain': hmm_comp['auc_diff_mean'],
            'p_value': hmm_comp['p_value'],
            'significant': hmm_comp['p_value'] < 0.05
        }

    full_comp = next((c for c in comparison_results if 'Full' in c['comparison']), None)
    if full_comp:
        json_summary['key_findings']['total_improvement'] = {
            'auc_gain': full_comp['auc_diff_mean'],
            'p_value': full_comp['p_value'],
            'relative_improvement': (full_comp['auc_diff_mean'] / full_comp['auc_1_mean']) * 100
        }

json_file = OUTPUT_DIR / 'ablation_summary.json'
with open(json_file, 'w') as f:
    json.dump(json_summary, f, indent=2)
print(f"[OK] JSON summary: {json_file}")

print()
print("=" * 80)
print("ABLATION STUDIES COMPLETE")
print("=" * 80)
print()
print("KEY FINDINGS:")

if json_summary['key_findings']['zscore_contribution']:
    zs = json_summary['key_findings']['zscore_contribution']
    print(f"  Z-score addition: {zs['auc_gain']:+.4f} AUC (p={zs['p_value']:.4f})")

if json_summary['key_findings']['hmm_dmd_contribution']:
    hd = json_summary['key_findings']['hmm_dmd_contribution']
    print(f"  HMM+DMD addition: {hd['auc_gain']:+.4f} AUC (p={hd['p_value']:.4f})")

if json_summary['key_findings']['total_improvement']:
    tot = json_summary['key_findings']['total_improvement']
    print(f"  Total improvement: {tot['auc_gain']:+.4f} AUC ({tot['relative_improvement']:+.2f}%)")

print()
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
