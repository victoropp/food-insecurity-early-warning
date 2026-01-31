# Dynamic News Signals as Early-Warning Indicators of Food Insecurity

**A Two-Stage Residual Modelling Framework**

**Author**: Victor Collins Oppon
**Student ID**: M01040265
**Institution**: Middlesex University London
**Program**: MSc Data Science
**Supervisor**: Dr. Giovanni Quattrone
**Year**: 2026

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## Overview

This repository contains the complete implementation of a two-stage cascade early warning system for food insecurity in Sub-Saharan Africa. The system combines autoregressive (AR) baselines with advanced machine learning models using GDELT news event data and IPC (Integrated Food Security Phase Classification) assessments.

**Key Innovation**: The framework strategically deploys lightweight AR models for persistence-dominated cases and sophisticated news-based models for shock-driven crises where early warning saves lives.

### Key Results

| Metric | AR Baseline | Cascade | Change |
|--------|-------------|---------|--------|
| **AUC-ROC** | 0.907 | - | - |
| **Precision** | 73.2% | 58.5% | -14.7pp |
| **Recall** | 73.2% | 77.9% | +4.7pp |
| **F1 Score** | 0.732 | 0.668 | -0.064 |
| **Key Saves** | - | **249** | 17.4% rescue rate |

### Data Scale

- **20,722** district-period observations
- **5,322** food crisis events (25.7% crisis rate)
- **18** African countries
- **1,920** unique districts
- **h=8 months** forecast horizon
- **5-fold** stratified spatial cross-validation

---

## Quick Start

### 1. Clone Repository
```bash
git clone https://github.com/victoropp/food-insecurity-early-warning.git
cd food-insecurity-early-warning
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Download Data Archive
Data archive available on Zenodo (DOI pending post-submission).

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
├── data/                             # Data directory
│   ├── external/                     # External source data
│   │   ├── ipc/                      # IPC assessments
│   │   ├── gdelt/                    # GDELT news data
│   │   └── shapefiles/               # Geographic boundaries
│   ├── interim/                      # Intermediate processed data
│   │   ├── stage1/                   # Stage 1 features
│   │   └── stage2/                   # Stage 2 features (HMM, DMD)
│   └── processed/                    # Final model-ready data
│
├── results/                          # Model outputs and predictions
│   ├── stage1_ar_baseline/           # AR baseline results
│   ├── stage2_models/                # XGBoost and ablation models
│   └── cascade_optimized_production/ # Final cascade results
│
├── scripts/                          # Organized pipeline scripts
│   ├── 01_data_acquisition/          # Download and preparation
│   ├── 02_data_processing/           # Aggregation and alignment
│   ├── 03_stage1_baseline/           # AR baseline model
│   ├── 04_stage2_feature_engineering/# Advanced features
│   ├── 05_stage2_model_training/     # XGBoost and ablations
│   ├── 06_cascade_analysis/          # Cascade system
│   ├── 07_visualization/             # Publication figures
│   └── 08_analysis/                  # Geographic heterogeneity
│
├── src/                              # Utility modules
│   ├── stratified_spatial_cv.py      # Custom cross-validation
│   ├── phase4_utils.py               # Analysis utilities
│   └── utils_dynamic.py              # Visualization helpers
│
├── notebooks/                        # Jupyter notebook versions
│   ├── 01-04: Data processing notebooks
│   ├── 05a-05j: Feature engineering notebooks
│   ├── 06-07: Model training notebooks
│   └── 08_Complete_Pipeline.ipynb    # End-to-end walkthrough
│
├── dissertation writeup/             # LaTeX dissertation materials
│   └── main.pdf                      # Final dissertation
│
└── poster/                           # Viva poster materials
    └── Oppon_MSc_Viva_Poster_A1_FINAL.pdf
```

---

## Methodology Overview

### Two-Stage Cascade Framework

1. **Stage 1: Autoregressive Baseline**
   - Features: Temporal autoregressive (L_t = IPC_{t-1}) + Spatial autoregressive (L_s)
   - Model: L2-regularized logistic regression
   - Cross-validation: 5-fold stratified spatial CV
   - Performance: AUC 0.907, Precision/Recall 73.2%
   - Purpose: Identify structurally persistent crises

2. **Stage 2: News-Based Models**
   - Features: Thematic ratios, z-scores, HMM regimes, DMD modes, location
   - Model: XGBoost with GridSearchCV (35 features)
   - Applied to: 6,553 AR failures and uncertain cases
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

- **Ratio Features**: Thematic proportions (category articles / total articles) capture news composition
- **Z-Score Features**: 12-month rolling standardization detects rapid temporal anomalies
- **HMM Features**: 2-state Hidden Markov Models (12-month window) capture narrative regime transitions
- **DMD Features**: Dynamic Mode Decomposition (12-month window) isolates crisis growth patterns
- **Location Features**: Country baseline rates, data density for stratification

### Key Saves by Country

The cascade rescued **249 crises** missed by AR baseline:

| Country | Key Saves |
|---------|-----------|
| Zimbabwe | 77 |
| Sudan | 59 |
| DR Congo | 40 |
| Nigeria | 27 |
| Mozambique | 15 |
| Mali | 12 |
| Kenya | 8 |
| Ethiopia | 6 |
| Malawi | 3 |
| Somalia | 2 |

---

## Data Sources

### 1. IPC Data (FEWSNET)
- **Source**: https://fews.net/fews-data/335
- **Coverage**: 18 African countries (2021-2024)
- **Granularity**: District-level (Admin Level 2)

### 2. GDELT News Events
- **Source**: https://www.gdeltproject.org/
- **Coverage**: African news articles and location mentions
- **Themes**: Conflict, displacement, economic, weather, food security, health, humanitarian, governance

### 3. Geographic Boundaries
- **GADM 4.1**: Administrative boundaries (district-level)
- **IPC Custom Boundaries**: From FEWSNET portal

---

## Key Research Findings

### RQ1: The Autocorrelation Trap
AR baseline achieves **AUC 0.907** using only two features (temporal and spatial lags), demonstrating that most food insecurity is structurally persistent. This exposes a "trap" where text-based models may primarily reflect autocorrelation rather than genuine predictive signal.

### RQ2: When News Matters
News features provide value for **shock-driven crises** that AR misses. The cascade rescues **249 key saves** (17.4% of AR failures) concentrated in conflict zones: Zimbabwe (77), Sudan (59), DR Congo (40).

### RQ3: Hidden Variables (HMM/DMD)
- HMM detects regime transitions with **transition_risk** as top feature
- DMD shows large coefficients for rare complex emergencies
- Combined, they capture dynamics invisible to compositional features

### RQ4: Geographic Heterogeneity
Performance varies significantly by country due to news coverage patterns. High-value countries (DRC, Sudan, Zimbabwe) show news features provide +2-8% AUC improvement over AR alone.

---

## Ablation Study Results

Best performing ablation: **Ratio + Location** (AUC: 0.727)

| Model Variant | Features | AUC-ROC |
|--------------|----------|---------|
| Ratio + Location | 12 | **0.727** |
| Ratio + HMM + DMD | 19 | 0.723 |
| Ratio + HMM | 15 | 0.718 |
| XGBoost Advanced | 35 | 0.697 |
| Z-score + Location | 12 | 0.699 |

---

## Installation

**System requirements**:
- Python 3.9-3.11
- 16GB RAM recommended
- ~5GB free space

**Quick install**:
```bash
pip install -r requirements.txt
```

**Core dependencies**:
- pandas, numpy, scikit-learn
- xgboost, statsmodels
- hmmlearn, pydmd
- geopandas, matplotlib, seaborn
- shap (for interpretability)

---

## Citation

If you use this code or data in your research, please cite:

```bibtex
@mastersthesis{oppon2026foodinsecurity,
  author = {Victor Collins Oppon},
  title = {Dynamic News Signals as Early-Warning Indicators of Food Insecurity: A Two-Stage Residual Modelling Framework},
  school = {Middlesex University London},
  year = {2026},
  type = {MSc Dissertation},
  note = {Available at: https://github.com/victoropp/food-insecurity-early-warning}
}
```

---

## License

This project is licensed under the MIT License.

**Data licenses**:
- IPC data: FEWSNET terms of use
- GDELT: Creative Commons Attribution 4.0
- GADM: Free for academic use

---

## Contact

**Victor Collins Oppon**
MSc Data Science, Middlesex University London
Email: victoropp@gmail.com

---

## Acknowledgments

- Dr. Giovanni Quattrone (Supervisor), Middlesex University London
- Famine Early Warning Systems Network (FEWSNET) for IPC data
- GDELT Project for news event data
- GADM for administrative boundaries

---

*For questions about running the code or reproducing results, please open an issue on GitHub or contact the author directly.*

