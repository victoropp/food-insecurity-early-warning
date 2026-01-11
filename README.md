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

### Key Components

- **Geographic Coverage**: Districts across African countries (2021-2024)
- **Two-Stage Cascade**: AR baseline → XGBoost advanced (binary override logic)
- **Advanced Features**: Ratio, z-score, HMM regimes, DMD modes
- **Data Scale**: IPC assessments, GDELT articles, location mentions
- **Custom Cross-Validation**: Stratified spatial CV to prevent information leakage

Run the scripts to discover the performance metrics.

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
Data archive available on Zenodo: [DOI: 10.5281/zenodo.XXXXXXX]

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
├── DATA_DICTIONARY.md                 # Variable descriptions
├── config.py                         # Centralized configuration
├── requirements.txt                   # Python dependencies
├── .gitignore                        # Git exclusions
│
├── data/                             # Data directory (see Zenodo archive)
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
│   ├── stage1_baseline/              # AR baseline results
│   ├── stage2_models/                # XGBoost and ablation models
│   └── cascade_optimized/            # Final cascade results
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
├── notebooks/                        # Jupyter notebook versions (27 notebooks)
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
│   ├── main.pdf                      # Final dissertation
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
   - Purpose: Identify structurally persistent crises

2. **Stage 2: News-Based Models**
   - Features: Thematic ratios, z-scores, HMM regimes, DMD modes, location
   - Model: XGBoost with GridSearchCV
   - Applied to: AR failures and uncertain cases
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

---

## Data Sources

### 1. IPC Data (FEWSNET)
- **Source**: https://fews.net/fews-data/335
- **Citation**: Famine Early Warning Systems Network (FEWSNET). (2024). Integrated Food Security Phase Classification (IPC) data.
- **Coverage**: African countries (2021-2024)
- **Granularity**: District-level (Admin Level 2)

### 2. GDELT News Events
- **Source**: https://www.gdeltproject.org/
- **Citation**: Leetaru, K., & Schrodt, P. A. (2013). GDELT: Global data on events, location and tone. ISA Annual Convention.
- **Coverage**: African news articles and location mentions
- **Themes**: Conflict, displacement, economic, weather, food security, health, humanitarian, governance

### 3. Geographic Boundaries
- **GADM 4.1**: Administrative boundaries (district-level)
- **Natural Earth 1:50m**: Country boundaries
- **IPC Custom Boundaries**: From FEWSNET portal

**Full Data Archive**: https://doi.org/10.5281/zenodo.XXXXXXX

---

## Key Research Questions

### RQ1: The Autocorrelation Trap
AR baselines achieve high performance using zero text features, suggesting most published results lacking AR comparisons may primarily reflect temporal and spatial autocorrelation rather than genuine text feature value.

### RQ2: When News Matters
Location features dominate tree splits but contribute minimal marginal attribution. Z-score features drive the majority of predictions despite lower tree rankings. Country-specific signals vary significantly.

### RQ3: Hidden Variables (HMM/DMD)
HMM detects regime transitions invisible to compositional features. DMD shows large mixed-effects coefficients for rare complex emergencies where multiple crisis drivers converge.

### RQ4: Two-Stage Framework Performance
The cascade rescues crises missed by AR baseline through strategic deployment of advanced features. Precision-recall trade-offs balanced for humanitarian cost minimization.

### RQ5: Geographic Heterogeneity
Performance varies significantly by country due to news coverage patterns. News deserts identified where coverage-based models fail.

**Run the scripts to discover the specific metrics for each research question.**

---

## Performance Summary

Run the pipeline to generate performance metrics:

```bash
# After running the pipeline
python -c "
import pandas as pd
from config import STAGE1_RESULTS, STAGE2_RESULTS, CASCADE_RESULTS

# View AR Baseline
stage1 = pd.read_csv(STAGE1_RESULTS / 'performance_metrics_district.csv')
print(f'AR Baseline AUC: {stage1[\"auc\"].mean():.3f}')

# View XGBoost
import json
with open(STAGE2_RESULTS / 'xgboost/basic_with_ar_optimized/summary_basic_optimized.json') as f:
    basic = json.load(f)
print(f'XGBoost Basic AUC: {basic[\"auc\"]:.3f}')

# View Cascade
cascade = pd.read_csv(CASCADE_RESULTS / 'country_metrics.csv')
print(f'Cascade AUC: {cascade[\"cascade_auc_roc\"].mean():.3f}')
print(f'Key Saves: {cascade[\"key_saves\"].sum()}')
"
```

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

**Estimated runtime**: Several hours for complete pipeline

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

**Abstract**: This dissertation develops a two-stage cascade early warning system for food insecurity in Sub-Saharan Africa, combining autoregressive baselines with advanced machine learning. Using GDELT news location mentions matched to IPC assessments, we demonstrate that a strategic cascade—using simple autoregressive models for confident predictions and XGBoost with advanced temporal features (HMM, DMD) for uncertain cases—achieves strong performance while maintaining interpretability and computational efficiency. The system successfully predicts crisis onsets with multi-month lead time, with particular effectiveness in conflict-driven crises. However, performance varies significantly by country due to news coverage patterns and data quality issues. This work contributes a practical early warning system, methodological innovations in spatial cross-validation, and insights into the geographic heterogeneity of news-based prediction models.

---

*For questions about running the code or reproducing results, please open an issue on GitHub or contact the author directly.*
