# Dynamic News Signals as Early-Warning Indicators of Food Insecurity

**A Two-Stage Residual Modelling Framework**

**Author**: Victor Collins Oppon
**Institution**: Middlesex University
**Program**: MSc Data Science
**Year**: 2025

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This repository contains the complete implementation of a two-stage cascade early warning system for food insecurity in Sub-Saharan Africa. The system combines autoregressive (AR) baselines with advanced machine learning models using GDELT news event data and IPC (Integrated Food Security Phase Classification) assessments.

**Key Innovation**: The framework strategically deploys lightweight AR models for persistence-dominated cases and sophisticated news-based models for shock-driven crises where early warning saves lives.

### Key Results

- **Geographic Coverage**: 534 districts across 18 African countries (2021-2024)
- **Cascade Performance**: 63.2% AUC-ROC, 60.3% F1 score, 86.6% recall
- **Crisis Detection**: Successfully predicts 86.6% of crisis onsets with 8-month lead time
- **Key Saves**: 249 crises rescued from AR baseline failures (17.4% improvement)
- **Data Scale**: 55,129 IPC assessments, 7.6 million GDELT articles, 5.2 million location mentions

---

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/yourusername/food-insecurity-early-warning.git
cd food-insecurity-early-warning
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Data Archive
Data archive (~900 MB) available on Zenodo: [DOI: 10.5281/zenodo.XXXXXXX]

```bash
# Download and extract
# Place in data/ directory
```

### 4. Run Pipeline

**Option A: Python Scripts**
```bash
python scripts/03_stage1_baseline/07_stage1_logistic_regression.py
python scripts/05_stage2_model_training/xgboost_models/01_xgboost_basic_WITH_AR_FILTER_OPTIMIZED.py
python scripts/06_cascade_analysis/05_cascade_ensemble_optimized_production.py
```

**Option B: Jupyter Notebooks**
```bash
jupyter lab
# Open notebooks/08_Complete_Pipeline.ipynb
```

---

## Repository Structure

```
dissertation_submission/
├── README.md                          # This file
├── INSTALLATION.md                    # Detailed installation guide
├── REPRODUCTION_GUIDE.md              # Step-by-step reproduction instructions
├── config.py                         # Centralized configuration
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git exclusions
│
├── data/                             # Data directory (see Zenodo archive)
│   ├── external/                     # External source data
│   │   ├── ipc/                      # IPC assessments (55,129 records)
│   │   ├── gdelt/                    # GDELT news data (7.6M articles)
│   │   └── shapefiles/               # Geographic boundaries
│   ├── interim/                      # Intermediate processed data
│   │   ├── stage1/                   # Stage 1 features
│   │   └── stage2/                   # Stage 2 features (HMM, DMD)
│   └── processed/                    # Final model-ready data
│
├── results/                          # Model outputs and predictions
│   ├── stage1_baseline/              # AR baseline results
│   ├── stage2_models/                # XGBoost and ablation models
│   └── cascade_optimized/            # Final cascade results
│
├── scripts/                          # Organized pipeline scripts (150 files)
│   ├── 01_data_acquisition/          # Download and preparation (3 scripts)
│   ├── 02_data_processing/           # Aggregation and alignment (22 scripts)
│   ├── 03_stage1_baseline/           # AR baseline model (8 scripts)
│   ├── 04_stage2_feature_engineering/# Advanced features (10 scripts)
│   ├── 05_stage2_model_training/     # XGBoost and ablations (10 scripts)
│   ├── 06_cascade_analysis/          # Cascade system (15 scripts)
│   ├── 07_visualization/             # Publication figures (64 scripts)
│   └── 08_analysis/                  # Geographic heterogeneity (18 scripts)
│
├── src/                              # Utility modules
│   ├── stratified_spatial_cv.py      # Custom cross-validation
│   ├── phase4_utils.py               # Analysis utilities
│   └── utils_dynamic.py              # Visualization helpers
│
├── notebooks/                        # Jupyter notebook versions (8 notebooks)
│   ├── 01_Data_Acquisition.ipynb
│   ├── 02_Data_Processing.ipynb
│   ├── 03_Stage1_Baseline.ipynb
│   ├── 04_Stage2_Features.ipynb
│   ├── 05_Stage2_Models.ipynb
│   ├── 06_Cascade_Analysis.ipynb
│   ├── 07_Visualizations.ipynb
│   └── 08_Complete_Pipeline.ipynb    # End-to-end walkthrough
│
├── dissertation/                     # LaTeX dissertation materials
│   ├── main.pdf                      # Final dissertation (356 pages)
│   ├── main.tex                      # LaTeX source
│   ├── chapters/                     # Chapter files
│   ├── appendices/                   # Appendix files
│   └── figures/                      # All dissertation figures
│
└── docs/                             # Additional documentation
    └── verification_logs/            # Verification documentation
```

---

## Methodology Overview

### Two-Stage Cascade Framework

1. **Stage 1: Autoregressive Baseline**
   - Features: Temporal lag (L_t) + Spatial lag (L_s)
   - Model: L2-regularized logistic regression
   - Cross-validation: 5-fold stratified spatial CV
   - Performance: 60.1% AUC-ROC, 82.2% recall
   - Purpose: Identify structurally persistent crises

2. **Stage 2: News-Based Models**
   - Features: Thematic ratios, z-scores, HMM regimes, DMD modes, location
   - Model: XGBoost with GridSearchCV (3,888 configurations)
   - Applied to: 26.8% of cases (AR failures, IPC ≤ 2)
   - Performance: 63.7% AUC-ROC, 85.2% recall
   - Purpose: Detect shock-driven crises invisible to AR

3. **Cascade Decision Logic**
   ```python
   # Binary override (not threshold-based)
   if AR_prediction == 1:
       final_prediction = 1  # Trust AR crisis detection
   elif AR_prediction == 0:
       final_prediction = Stage2_prediction  # Use news-based model
   ```

### Advanced Features

- **Ratio Features**: 12-month moving averages capture sustained compositional shifts in news coverage
- **Z-Score Features**: 12-month rolling standardization detects rapid temporal anomalies
- **HMM Features**: 2-state binary regimes (Pre-Crisis vs Crisis-Prone) capture narrative transitions
- **DMD Features**: Rank-5 SVD decomposes temporal evolution patterns, isolates crisis modes
- **Location Features**: Country, district_area_km2, centroid coordinates for stratification
- **Mixed-Effects**: Country-level random effects quantify geographic heterogeneity

---

## Data Sources

### 1. IPC Data (FEWSNET)
- **Records**: 55,129 district-level assessments
- **Coverage**: 24 African countries (raw data), 18 countries (analysis after filtering)
- **Period**: 2021-01 to 2024-12
- **Source**: https://fews.net/fews-data/335
- **Citation**: Famine Early Warning Systems Network (FEWSNET). (2024). Integrated Food Security Phase Classification (IPC) data.

### 2. GDELT News Events
- **Articles**: 7.6 million African news articles
- **Location Mentions**: 5.2 million geographic mentions
- **Themes**: 9 categories (conflict, displacement, economic, weather, food_security, health, humanitarian, governance, other)
- **Source**: https://www.gdeltproject.org/
- **Citation**: Leetaru, K., & Schrodt, P. A. (2013). GDELT: Global data on events, location and tone. ISA Annual Convention.

### 3. Geographic Boundaries
- **GADM 4.1**: Administrative boundaries (district-level)
- **Natural Earth 1:50m**: Country boundaries
- **IPC Custom Boundaries**: From FEWSNET portal

**Full Data Archive**: https://doi.org/10.5281/zenodo.XXXXXXX (900 MB)

---

## Key Findings

### RQ1: The Autocorrelation Trap
> AR baselines achieve 93.8% of published news-based model performance using **zero text features** (AR PR-AUC: 0.765 vs Balashankar et al. PR-AUC: 0.816). Most published results (AUC 0.75-0.85) lacking AR comparisons may primarily reflect temporal and spatial autocorrelation rather than genuine text feature value.

### RQ2: When News Matters
- **Split Frequency ≠ Predictive Power**: Location features dominate tree splits (29.3%) but contribute only 2.6% marginal attribution (15.5× overstatement)
- **Z-score features drive 74.7% of predictions** despite lower tree rankings
- **Country-specific signals**: Zimbabwe weather +2.1pp, Sudan conflict +3.3pp, DRC displacement +2.2pp, Somalia health +5.8pp above global average

### RQ3: Hidden Variables (HMM/DMD)
- **HMM**: 3.2% feature importance, detects regime transitions invisible to compositional features
- **DMD**: Largest mixed-effects coefficient (+352.38) for rare complex emergencies where multiple crisis drivers converge

### RQ4: Two-Stage Framework Performance
- **249 crises rescued** (17.4% of AR failures)
- **Precision-recall trade-off**: 0.732 → 0.585 precision (-14.7pp), but recall improvement 0.732 → 0.779 (+4.7pp)
- **Humanitarian cost-benefit**: 10:1 FN:FP weighting yields 6.2% total cost reduction

### RQ5: Geographic Heterogeneity
- **70.7% of saves in 3 countries**: Zimbabwe (77), Sudan (59), DRC (40)
- **Country-level AUC range**: 0.068 (Niger) to 0.682 (Sudan) - 10-fold difference
- **News deserts**: 1,178 crises (82.6%) still missed have 64% less coverage (74 vs 121 articles/month)

---

## 18 Countries in Analysis

After district threshold filtering (≥200 articles/year per district), 18 countries remain:

| Code | Country | Districts | Crisis Rate | Key Dynamics |
|------|---------|-----------|-------------|--------------|
| ZWE | Zimbabwe | 203 | 45.0% | Economic collapse + droughts |
| SDN | Sudan | 226 | 63.9% | Civil war escalation |
| COD | Dem. Rep. Congo | 221 | 15.9% | M23 resurgence, displacement |
| NGA | Nigeria | 281 | 28.4% | Boko Haram, farmer-herder conflict |
| MOZ | Mozambique | 130 | 12.8% | Climate shocks |
| MLI | Mali | 113 | 6.7% | Sahel instability |
| KEN | Kenya | 578 | 34.0% | Drought cycles |
| ETH | Ethiopia | 712 | 24.8% | Tigray conflict aftermath |
| MWI | Malawi | 61 | 33.9% | Climate vulnerability |
| SOM | Somalia | 105 | 44.7% | Al-Shabaab, recurrent droughts |
| TCD | Chad | 21 | 23.4% | Sahel crisis |
| NER | Niger | 148 | 16.1% | Food security structural issues |
| CMR | Cameroon | 82 | 20.2% | Boko Haram spillover |
| UGA | Uganda | 308 | 1.4% | Refugee hosting |
| BDI | Burundi | 36 | 0.5% | Low crisis prevalence |
| MDG | Madagascar | 124 | 3.7% | Cyclone impacts |
| SSD | South Sudan | 53 | 95.9% | Chronic crisis |
| BFA | Burkina Faso | 36 | 20.9% | Sahel instability |

**Note**: Dissertation mentions "24 African countries" referring to raw IPC data coverage. Analysis uses 18 countries after applying district threshold (≥5 valid districts per country).

---

## Performance Summary

| Model | AUC-ROC | F1 | Precision | Recall | Specificity | Brier |
|-------|---------|-----|-----------|--------|-------------|-------|
| AR Baseline | 0.601 | 0.580 | 0.631 | 0.822 | 0.420 | 0.192 |
| XGBoost Basic | 0.624 | 0.593 | 0.642 | 0.844 | 0.431 | 0.187 |
| XGBoost Advanced | 0.637 | 0.601 | 0.648 | 0.852 | 0.444 | 0.184 |
| **Cascade (Optimized)** | **0.632** | **0.603** | **0.653** | **0.866** | **0.426** | **0.185** |

**Humanitarian Cost (10:1 FN:FP)**: AR: 0.250 → Cascade: 0.234 (-6.2%)

---

## Installation

See [INSTALLATION.md](INSTALLATION.md) for detailed setup instructions.

**Quick install**:
```bash
pip install -r requirements.txt
```

**System requirements**:
- Python 3.9-3.11
- 16GB RAM recommended
- ~5GB free space

---

## Reproduction

See [REPRODUCTION_GUIDE.md](REPRODUCTION_GUIDE.md) for step-by-step instructions to reproduce all results.

**Estimated runtime**: ~5 hours for complete pipeline

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@mastersthesis{oppon2025foodinsecurity,
  author = {Victor Collins Oppon},
  title = {Dynamic News Signals as Early-Warning Indicators of Food Insecurity: A Two-Stage Residual Modelling Framework},
  school = {Middlesex University},
  year = {2025},
  type = {MSc Dissertation},
  note = {Available at: https://github.com/yourusername/food-insecurity-early-warning}
}
```

**Data archive citation**:
```bibtex
@dataset{oppon2025data,
  author = {Victor Collins Oppon},
  title = {Data Archive - Food Insecurity Early Warning System Using GDELT and IPC},
  year = {2025},
  publisher = {Zenodo},
  doi = {10.5281/zenodo.XXXXXXX},
  url = {https://doi.org/10.5281/zenodo.XXXXXXX}
}
```

---

## License

This project is licensed under the MIT License - see [LICENSE](LICENSE) file for details.

**Data licenses**:
- IPC data: FEWSNET terms of use
- GDELT: Creative Commons Attribution 4.0
- GADM: Free for academic use

---

## Contact

**Victor Collins Oppon**
MSc Data Science, Middlesex University
Email: [your-email@example.com]
LinkedIn: [your-profile]
GitHub: [yourusername]

---

## Acknowledgments

- Dr. Giovanni Quattrone (Supervisor), Middlesex University
- Famine Early Warning Systems Network (FEWSNET) for IPC data
- GDELT Project for news event data
- GADM for administrative boundaries
- Middlesex University Department of Computer Science

---

## Dissertation

The full dissertation (356 pages) is available in `dissertation/main.pdf`.

**Abstract**: This dissertation develops a two-stage cascade early warning system for food insecurity in Sub-Saharan Africa, combining autoregressive baselines with advanced machine learning. Using 5.2 million GDELT news location mentions matched to IPC assessments across 534 districts in 18 countries (2021-2024), we demonstrate that a strategic cascade—using simple autoregressive models for confident predictions and XGBoost with advanced temporal features (HMM, DMD) for uncertain cases—achieves 63.2% AUC while maintaining interpretability and computational efficiency. The system successfully predicts 86.6% of crisis onsets with 8-month lead time, with particular effectiveness in conflict-driven crises. However, performance varies significantly by country (Nigeria 74.0% AUC vs DRC 53.8%) due to news coverage patterns and data quality issues. This work contributes a practical early warning system, methodological innovations in spatial cross-validation, and insights into the geographic heterogeneity of news-based prediction models.

---

*For questions about running the code or reproducing results, please open an issue on GitHub or contact the author directly.*
