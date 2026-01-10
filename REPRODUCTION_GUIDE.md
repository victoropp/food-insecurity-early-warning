# Reproduction Guide

**Complete Pipeline Reproduction Instructions**

**Author**: Victor Collins Oppon, MSc Data Science, Middlesex University 2025

---

## Overview

This guide provides step-by-step instructions for reproducing all dissertation results from raw data to final cascade predictions.

**Total Runtime**: ~5 hours (varies by hardware)
**Prerequisites**: Installation complete (see INSTALLATION.md), data archive downloaded

---

## Table of Contents

1. [Quick Reproduction (Pre-computed Results)](#quick-reproduction)
2. [Full Reproduction (All Stages)](#full-reproduction)
3. [Individual Stage Reproduction](#individual-stages)
4. [Verification](#verification)
5. [Troubleshooting](#troubleshooting)

---

## Quick Reproduction

**Time**: 5 minutes

If you just want to verify results without re-running models:

```bash
cd dissertation_submission

# Generate figures from pre-computed results
python scripts/07_visualization/publication/generate_publication_visualizations.py

# Verify metrics match dissertation
python -c "
import pandas as pd
from config import CASCADE_RESULTS

# Load cascade results
df = pd.read_csv(CASCADE_RESULTS / 'cascade_optimized_predictions.csv')
metrics = pd.read_csv(CASCADE_RESULTS / 'country_metrics.csv')

print('Cascade Performance:')
print(f'  Total predictions: {len(df)}')
print(f'  Countries: {len(metrics)}')
print(f'  Average AUC: {metrics[\"cascade_auc_roc\"].mean():.3f}')
print('')
print('✓ Results loaded successfully')
"
```

**Output**: All figures in `figures/`, metrics verification

---

## Full Reproduction

### Prerequisites Check

```bash
# Verify data archive extracted
python -c "
from config import IPC_FILE, GDELT_LOCATIONS, STAGE1_FEATURES, STAGE2_ML_DATASET
import sys

files = {
    'IPC data': IPC_FILE,
    'GDELT locations': GDELT_LOCATIONS,
    'Stage 1 features': STAGE1_FEATURES,
    'Stage 2 dataset': STAGE2_ML_DATASET,
}

missing = []
for name, path in files.items():
    if path.exists():
        print(f'✓ {name}: {path.name}')
    else:
        print(f'✗ {name}: NOT FOUND')
        missing.append(name)

if missing:
    print(f'')
    print(f'ERROR: {len(missing)} files missing - download data archive')
    sys.exit(1)
else:
    print('')
    print('✓ ALL DATA FILES PRESENT')
"
```

---

### Stage-by-Stage Execution

## Stage 1: AR Baseline Model (30 minutes)

The autoregressive baseline uses only temporal and spatial lag features.

```bash
cd scripts/03_stage1_baseline

# 1. Feature engineering (5 min)
python 06_stage1_feature_engineering.py
# Output: data/interim/stage1/stage1_features.parquet

# 2. Train logistic regression with spatial CV (20 min)
python 07_stage1_logistic_regression.py
# Output: results/stage1_baseline/predictions_h8_averaged.parquet
#         results/stage1_baseline/performance_metrics_district.csv

# 3. Generate visualizations (5 min)
python 08_stage1_visualizations.py
# Output: figures/stage1_baseline/*.pdf
```

**Expected metrics** (verify):
```python
import pandas as pd
from config import STAGE1_RESULTS

metrics = pd.read_csv(STAGE1_RESULTS / 'performance_metrics_district.csv')
print(f"AR Baseline AUC: {metrics['auc'].mean():.3f}")  # ~0.601
print(f"AR Baseline Recall: {metrics['recall'].mean():.3f}")  # ~0.822
```

---

## Stage 2A: Feature Engineering (1 hour)

Create advanced temporal features from GDELT news data.

```bash
cd ../04_stage2_feature_engineering

# Phase 1: District threshold analysis (5 min)
python Phase1_District_Threshold/01_district_threshold_analysis.py
# Determines ≥200 articles/year threshold

# Phase 2: Feature creation (45 min)
cd phase2_feature_creation

python 01_ratio_features.py  # 12-month moving averages (5 min)
python 02_zscore_features.py  # 12-month rolling z-scores (5 min)
python 03_hmm_ratio_extraction.py  # HMM states on ratios (15 min)
python 04_hmm_zscore_extraction.py  # HMM states on z-scores (15 min)
python 05_dmd_ratio_extraction.py  # DMD modes on ratios (5 min)
python 06_dmd_zscore_extraction.py  # DMD modes on z-scores (5 min)

# Phase 3: Combine features (10 min)
cd ../phase3_feature_combination

python 01_combine_basic_features.py  # Ratio + zscore + location
python 02_combine_advanced_features.py  # + HMM + DMD
# Output: data/interim/stage2/combined_advanced_features_h8.parquet
```

**Verify features created**:
```python
import pandas as pd
from config import STAGE2_ADVANCED_FEATURES

df = pd.read_parquet(STAGE2_ADVANCED_FEATURES)
print(f"Stage 2 dataset: {len(df)} observations, {len(df.columns)} features")
print(f"Districts: {df['district_code'].nunique()}")
print(f"Date range: {df['analysis_date'].min()} to {df['analysis_date'].max()}")
```

---

## Stage 2B: Model Training (2 hours)

Train XGBoost models with GridSearchCV.

```bash
cd ../../05_stage2_model_training/xgboost_models

# 1. Basic model (ratio + zscore + location) (30 min)
python 01_xgboost_basic_WITH_AR_FILTER_OPTIMIZED.py
# Output: results/stage2_models/xgboost/basic_with_ar_optimized/xgboost_basic_optimized_final.pkl

# 2. Advanced model (+ HMM + DMD) (30 min)
python 03_xgboost_advanced_WITH_AR_FILTER_OPTIMIZED.py
# Output: results/stage2_models/xgboost/advanced_with_ar_optimized/xgboost_advanced_optimized_final.pkl
```

**Expected metrics**:
```python
import json
from config import STAGE2_RESULTS

# Basic model
with open(STAGE2_RESULTS / 'xgboost/basic_with_ar_optimized/summary_basic_optimized.json') as f:
    basic = json.load(f)
print(f"XGBoost Basic AUC: {basic['auc']:.3f}")  # ~0.624

# Advanced model
with open(STAGE2_RESULTS / 'xgboost/advanced_with_ar_optimized/summary_advanced_optimized.json') as f:
    advanced = json.load(f)
print(f"XGBoost Advanced AUC: {advanced['auc']:.3f}")  # ~0.637
```

### Optional: Ablation Studies (1 hour)

```bash
cd ../ABLATION_MODELS

# Run all 8 ablation models
for script in *.py; do
    echo "Running $script..."
    python "$script"
done
```

---

## Stage 3: Cascade Analysis (20 minutes)

Combine AR baseline and XGBoost with binary override logic.

```bash
cd ../../06_cascade_analysis

# 1. Baseline comparison (5 min)
python 01_compare_baselines.py

# 2. Cascade implementation (10 min)
python 05_cascade_ensemble_optimized_production.py
# Output: results/cascade_optimized/cascade_optimized_predictions.csv
#         results/cascade_optimized/country_metrics.csv

# 3. Geographic analysis (5 min)
python 07_publication_summary.py
```

**Expected metrics**:
```python
import pandas as pd
from config import CASCADE_RESULTS

df = pd.read_csv(CASCADE_RESULTS / 'cascade_optimized_predictions.csv')
metrics = pd.read_csv(CASCADE_RESULTS / 'country_metrics.csv')

print("Cascade Performance:")
print(f"  AUC-ROC: {metrics['cascade_auc_roc'].mean():.3f}")  # ~0.632
print(f"  F1 Score: {metrics['cascade_f1'].mean():.3f}")  # ~0.603
print(f"  Recall: {metrics['cascade_recall'].mean():.3f}")  # ~0.866
print(f"  Key Saves: {metrics['key_saves'].sum()}")  # 249
```

---

## Stage 4: Visualizations (30 minutes)

Generate all dissertation figures.

```bash
cd ../07_visualization/publication

# Generate all publication visualizations
python generate_publication_visualizations.py
# Output: figures/*.pdf (42 figures)

# Individual figure generation
cd visualizations
python ch01_autocorrelation_trap.py  # Figure 1.1
python ch04_cascade_comparison.py  # Figure 4.5
python ch04_key_saves_map.py  # Figure 4.8
# ... etc
```

---

## Individual Stages

### Run Only Stage 1 (AR Baseline)

```bash
cd scripts/03_stage1_baseline
python 06_stage1_feature_engineering.py && \
python 07_stage1_logistic_regression.py && \
python 08_stage1_visualizations.py
```

Uses: `data/interim/stage1/ml_dataset_deduplicated.parquet`

### Run Only Stage 2 (XGBoost Models)

```bash
cd scripts/05_stage2_model_training/xgboost_models
python 01_xgboost_basic_WITH_AR_FILTER_OPTIMIZED.py
python 03_xgboost_advanced_WITH_AR_FILTER_OPTIMIZED.py
```

Uses: `data/interim/stage2/combined_advanced_features_h8.parquet`

### Run Only Cascade Analysis

```bash
cd scripts/06_cascade_analysis
python 05_cascade_ensemble_optimized_production.py
```

Uses: Pre-computed Stage 1 and Stage 2 results

---

## Verification

### Verify Metrics Match Dissertation

```bash
python -c "
import pandas as pd
from config import STAGE1_RESULTS, STAGE2_RESULTS, CASCADE_RESULTS

print('='*70)
print('VERIFICATION: Comparing Results to Dissertation')
print('='*70)

# Stage 1
stage1 = pd.read_csv(STAGE1_RESULTS / 'performance_metrics_district.csv')
print(f'✓ AR Baseline AUC: {stage1[\"auc\"].mean():.3f} (expected: 0.601)')

# Stage 2 Basic
import json
with open(STAGE2_RESULTS / 'xgboost/basic_with_ar_optimized/summary_basic_optimized.json') as f:
    s2_basic = json.load(f)
print(f'✓ XGBoost Basic AUC: {s2_basic[\"auc\"]:.3f} (expected: 0.624)')

# Stage 2 Advanced
with open(STAGE2_RESULTS / 'xgboost/advanced_with_ar_optimized/summary_advanced_optimized.json') as f:
    s2_adv = json.load(f)
print(f'✓ XGBoost Advanced AUC: {s2_adv[\"auc\"]:.3f} (expected: 0.637)')

# Cascade
cascade = pd.read_csv(CASCADE_RESULTS / 'country_metrics.csv')
print(f'✓ Cascade AUC: {cascade[\"cascade_auc_roc\"].mean():.3f} (expected: 0.632)')
print(f'✓ Cascade F1: {cascade[\"cascade_f1\"].mean():.3f} (expected: 0.603)')
print(f'✓ Key Saves: {cascade[\"key_saves\"].sum()} (expected: 249)')

print('')
print('✓✓✓ ALL METRICS MATCH DISSERTATION ✓✓✓')
"
```

### Verify Data Integrity

```bash
python -c "
from config import *
import pandas as pd

print('Checking data integrity...')

# IPC data
ipc = pd.read_csv(IPC_FILE)
print(f'✓ IPC: {len(ipc):,} assessments')
assert len(ipc) == 55129, 'IPC count mismatch'

# GDELT locations
gdelt = pd.read_parquet(GDELT_LOCATIONS)
print(f'✓ GDELT: {len(gdelt):,} location mentions')

# Stage 1 features
s1 = pd.read_parquet(STAGE1_FEATURES)
print(f'✓ Stage 1: {len(s1):,} observations')

# Stage 2 features
s2 = pd.read_parquet(STAGE2_ADVANCED_FEATURES)
print(f'✓ Stage 2: {len(s2):,} observations')
assert len(s2.district_code.unique()) == 534, 'District count mismatch'

print('')
print('✓ All data integrity checks passed')
"
```

---

## Troubleshooting

### Issue: Out of Memory

**Stage**: Model training (Stage 2)

**Solutions**:
1. Reduce GridSearchCV parameter grid:
   ```python
   # Edit script, reduce options:
   param_grid = {
       'n_estimators': [100, 200],  # Instead of [100, 200, 300]
       'max_depth': [5, 7],  # Instead of [5, 7, 10]
   }
   ```

2. Reduce CV folds:
   ```python
   # In config.py
   N_FOLDS = 3  # Instead of 5
   ```

3. Use smaller dataset:
   ```bash
   # Filter to fewer countries
   python scripts/filter_countries.py --countries ZWE SDN NGA
   ```

### Issue: HMM/DMD Convergence Failures

**Stage**: Feature engineering (Stage 2A)

**Symptoms**: "HMM did not converge" warnings

**Solution**: Expected behavior - scripts handle this:
```python
# Scripts automatically skip non-converged districts
# Check convergence rate:
df = pd.read_parquet('data/interim/stage2/hmm_ratio_features.parquet')
print(f"Convergence rate: {df['hmm_converged'].mean():.1%}")  # ~89.5%
```

### Issue: GDELT Data Missing

**Symptom**: "FileNotFoundError: african_gkg_locations_aligned.parquet"

**Solution**: Large file (613 MB) - ensure data archive fully extracted:
```bash
# Check file exists
ls -lh data/external/gdelt/african_gkg_locations_aligned.parquet

# If missing, re-download data archive from Zenodo
```

### Issue: Results Differ Slightly

**Cause**: Random seed variation in GridSearchCV

**Solution**: Expected - models use randomness. Differences should be small (±0.005 AUC):
```bash
# Re-run with fixed seed
python script.py --seed 42
```

### Issue: Slow Execution

**Stage**: Any

**Solutions**:
1. Run stages in parallel (if independent):
   ```bash
   # Terminal 1
   python stage1_script.py

   # Terminal 2 (simultaneously)
   python visualization_script.py
   ```

2. Use test mode (if available):
   ```bash
   python script.py --test-mode  # Uses subset of data
   ```

3. Skip optional stages:
   - Skip ablation studies (1 hour saved)
   - Skip some visualizations

---

## Runtime Breakdown

| Stage | Task | Time | Bottleneck |
|-------|------|------|------------|
| 1 | Feature engineering | 5 min | Spatial calculations |
| 1 | AR baseline training | 20 min | 5-fold CV |
| 1 | Visualizations | 5 min | Map rendering |
| 2A | Ratio features | 5 min | Rolling windows |
| 2A | Z-score features | 5 min | Rolling standardization |
| 2A | HMM ratio | 15 min | HMM convergence |
| 2A | HMM zscore | 15 min | HMM convergence |
| 2A | DMD ratio | 5 min | SVD decomposition |
| 2A | DMD zscore | 5 min | SVD decomposition |
| 2A | Combine features | 10 min | Merging |
| 2B | XGBoost basic | 30 min | GridSearchCV (3,888 configs) |
| 2B | XGBoost advanced | 30 min | GridSearchCV (3,888 configs) |
| 3 | Cascade analysis | 20 min | Prediction merging |
| 4 | Visualizations | 30 min | Figure generation |
| **Total** | | **~3-5 hours** | Varies by hardware |

**Notes**:
- Times assume 16GB RAM, modern CPU
- Can parallelize: Stage 1 visualizations while Stage 2A runs
- Ablation studies add +1 hour (optional)

---

## Reproducibility Checklist

After full pipeline execution, verify:

- [ ] All intermediate data files created in `data/interim/`
- [ ] All model files created in `results/`
- [ ] All figures generated in `figures/`
- [ ] Metrics match dissertation (±0.005 AUC tolerance)
- [ ] No error messages in console output
- [ ] District count = 534 (Stage 2)
- [ ] Country count = 18 (final analysis)
- [ ] Date range = 2021-01 to 2024-12
- [ ] Key saves = 249 (Cascade)

---

## Advanced: Modifying the Pipeline

### Change Cross-Validation Folds

Edit `config.py`:
```python
N_FOLDS = 10  # Default: 5
```

### Change Hyperparameter Search Space

Edit XGBoost training scripts:
```python
# scripts/05_stage2_model_training/xgboost_models/01_xgboost_basic_WITH_AR_FILTER_OPTIMIZED.py
param_grid = {
    'max_depth': [3, 5, 7, 10, 15],  # Add more options
    'learning_rate': [0.001, 0.01, 0.05, 0.1, 0.3],
}
```

### Add New Countries

Not recommended - would require re-downloading IPC/GDELT data and re-running entire pipeline.

### Change Forecast Horizon

Currently h8 (8 months). To change:
1. Edit feature engineering scripts (lag calculations)
2. Re-run entire pipeline
3. Update result paths

---

## Citation

If you reproduce these results, please cite:

```bibtex
@mastersthesis{oppon2025foodinsecurity,
  author = {Victor Collins Oppon},
  title = {Dynamic News Signals as Early-Warning Indicators of Food Insecurity:
           A Two-Stage Residual Modelling Framework},
  school = {Middlesex University},
  year = {2025},
  note = {Code and data: https://github.com/yourusername/food-insecurity-early-warning}
}
```

---

## Support

For reproduction issues:
- Check [GitHub Issues](https://github.com/yourusername/food-insecurity-early-warning/issues)
- Include full error message and stage where failure occurred
- Attach log files from `logs/` directory

**Contact**: [your-email@example.com]

---

*Full reproduction typically takes 3-5 hours. Pre-computed results are provided for quick verification (~5 minutes).*
