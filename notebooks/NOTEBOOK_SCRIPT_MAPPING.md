# Notebook to Script Mapping

**Author**: Victor Collins Oppon, MSc Data Science, Middlesex University 2025

This document maps each Jupyter notebook to the exact Python scripts it should replicate.

---

## 01_Data_Acquisition.ipynb ✅ DONE

**Scripts**:
- `scripts/01_data_acquisition/download_gadm_africa.py`
- `scripts/01_data_acquisition/download_livelihood_zones.py` (optional)
- IPC data download (manual - documented in notebook)
- GDELT data (reference to data archive)

**Status**: Complete - functional notebook created

---

## 02_Data_Processing.ipynb

**Scripts** (in order):
1. `scripts/02_data_processing/01_prepare_ipc_reference.py` - Prepare IPC reference data
2. `scripts/02_data_processing/02_aggregate_articles.py` - Aggregate GDELT articles to districts
3. `scripts/02_data_processing/03_aggregate_locations.py` - Aggregate GDELT locations to districts
4. `scripts/02_data_processing/04_create_ml_dataset.py` - Merge articles + locations + IPC
5. `scripts/02_data_processing/05_deduplicate.py` - Deduplicate district-period observations

**Output**: `data/interim/stage1/ml_dataset_deduplicated.parquet`

**Note**: These scripts are also in `03_stage1_baseline/` directory (duplicates). Use the ones from `02_data_processing/` as they are the canonical versions.

---

## 03_Stage1_Baseline.ipynb

**Scripts** (in order):
1. `scripts/03_stage1_baseline/06_stage1_feature_engineering.py` - Create Lt and Ls features
2. `scripts/03_stage1_baseline/07_stage1_logistic_regression.py` - Train AR baseline with spatial CV
3. `scripts/03_stage1_baseline/08_stage1_visualizations.py` - Visualize AR baseline results

**Input**: `data/interim/stage1/ml_dataset_deduplicated.parquet`
**Output**:
- `data/interim/stage1/stage1_features.parquet`
- `results/stage1_baseline/predictions_h8_averaged.parquet`
- `results/stage1_baseline/performance_metrics_district.csv`

---

## 04_Stage2_Features.ipynb

**Scripts** (in order):

### Phase 1: District Threshold
1. `scripts/04_stage2_feature_engineering/Phase1_District_Threshold/01_district_threshold_analysis.py`
2. `scripts/04_stage2_feature_engineering/Phase1_District_Threshold/02_country_inclusion_by_districts.py`

### Phase 2: Feature Creation
3. `scripts/04_stage2_feature_engineering/phase2_feature_creation/01_ratio_features.py`
4. `scripts/04_stage2_feature_engineering/phase2_feature_creation/02_zscore_features.py`
5. `scripts/04_stage2_feature_engineering/phase2_feature_creation/03_hmm_ratio_extraction.py`
6. `scripts/04_stage2_feature_engineering/phase2_feature_creation/04_hmm_zscore_extraction.py`
7. `scripts/04_stage2_feature_engineering/phase2_feature_creation/05_dmd_ratio_extraction.py`
8. `scripts/04_stage2_feature_engineering/phase2_feature_creation/06_dmd_zscore_extraction.py`

### Phase 3: Feature Combination
9. `scripts/04_stage2_feature_engineering/phase3_feature_combination/01_combine_basic_features.py`
10. `scripts/04_stage2_feature_engineering/phase3_feature_combination/02_combine_advanced_features.py`

**Input**: `data/interim/stage1/stage1_features.parquet` + Stage 1 AR predictions
**Output**:
- `data/interim/stage2/combined_basic_features_h8.parquet`
- `data/interim/stage2/combined_advanced_features_h8.parquet`

---

## 05_Stage2_Models.ipynb

**Scripts** (in order):
1. `scripts/05_stage2_model_training/xgboost_models/01_xgboost_basic_WITH_AR_FILTER_OPTIMIZED.py`
2. `scripts/05_stage2_model_training/xgboost_models/03_xgboost_advanced_WITH_AR_FILTER_OPTIMIZED.py`

**Optional - Ablation Studies** (show 1-2 as examples):
3. `scripts/05_stage2_model_training/ABLATION_MODELS/01_ablation_ratio_location_OPTIMIZED.py`
4. `scripts/05_stage2_model_training/ABLATION_MODELS/05_ablation_ratio_zscore_location_OPTIMIZED.py`

**Input**: `data/interim/stage2/combined_advanced_features_h8.parquet`
**Output**:
- `results/stage2_models/xgboost/basic_with_ar_optimized/xgboost_basic_optimized_final.pkl`
- `results/stage2_models/xgboost/advanced_with_ar_optimized/xgboost_advanced_optimized_final.pkl`

---

## 06_Cascade_Analysis.ipynb

**Scripts** (in order):
1. `scripts/06_cascade_analysis/01_compare_baselines.py` - Compare AR vs Stage 2
2. `scripts/06_cascade_analysis/05_cascade_ensemble_optimized_production.py` - Implement cascade
3. `scripts/06_cascade_analysis/07_publication_summary.py` - Analyze results

**Input**:
- `results/stage1_baseline/predictions_h8_averaged.parquet`
- `results/stage2_models/xgboost/advanced_with_ar_optimized/predictions_advanced_optimized.csv`

**Output**:
- `results/cascade_optimized/cascade_optimized_predictions.csv`
- `results/cascade_optimized/country_metrics.csv`
- `results/cascade_optimized/key_saves.csv`

---

## 07_Visualizations.ipynb

**Scripts** (select key ones):
1. `scripts/07_visualization/publication/generate_publication_visualizations.py` - Master script
2. `scripts/07_visualization/publication/visualizations/ch01_autocorrelation_trap.py` - Figure 1.1
3. `scripts/07_visualization/publication/visualizations/ch04_cascade_comparison.py` - Figure 4.5
4. `scripts/07_visualization/publication/visualizations/ch04_key_saves_map.py` - Figure 4.8
5. `scripts/07_visualization/publication/visualizations/ch04_shap_analysis.py` - SHAP analysis

**Input**: All results from previous stages
**Output**: `figures/*.pdf` (42 figures)

**Note**: This notebook demonstrates key visualizations, not all 64 scripts. User can run individual scripts for specific figures.

---

## 08_Complete_Pipeline.ipynb

**Combines**: All of the above in sequence
- Imports functions from notebooks 01-07
- Runs complete pipeline end-to-end
- Shows progress and intermediate results
- Generates final summary

**Runtime**: 5-8 hours (full execution)
**Alternative**: Load pre-computed results for quick verification

---

## Implementation Strategy

### For Each Notebook:

1. **Read source script(s)** completely
2. **Convert to notebook cells**:
   - Docstring → Markdown cell
   - Import statements → Code cell
   - Each major function → Separate code cell with markdown explanation
   - Main execution → Sequential code cells
3. **Add narrative**:
   - Markdown before each section explaining what happens
   - Show intermediate results (df.head(), df.shape, etc.)
   - Add progress indicators
4. **Test execution**:
   - Run notebook cell-by-cell
   - Verify outputs match script outputs
   - Save with outputs for viewing

### Code Cell Organization:

```python
# Each code cell should be ~10-50 lines
# Break long scripts into logical sections:
# - Imports
# - Configuration
# - Function definitions (one per cell or grouped logically)
# - Data loading
# - Processing (with progress prints)
# - Results
# - Saving outputs
```

### Markdown Cell Organization:

```markdown
# Section headers use ## or ###
Brief explanation of what the next code cells do
Why it's necessary
What to expect in outputs
```

---

## Verification Checklist

For each notebook:
- [ ] All source scripts identified
- [ ] Scripts read completely
- [ ] Converted to notebook format
- [ ] Markdown narrative added
- [ ] Tested execution
- [ ] Outputs verified against script outputs
- [ ] Saved with outputs
- [ ] Added to git

---

**Status**:
- ✅ 01_Data_Acquisition.ipynb
- ⏳ 02-08 in progress
