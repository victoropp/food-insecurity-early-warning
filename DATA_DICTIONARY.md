# Data Dictionary - Food Insecurity Early Warning System

**Author**: Victor Collins Oppon
**Institution**: Middlesex University
**Program**: MSc Data Science
**Year**: 2025

---

## Overview

This data dictionary provides comprehensive descriptions of all variables used in the dissertation "Dynamic News Signals as Early-Warning Indicators of Food Insecurity: A Two-Stage Residual Modelling Framework".

**Data Coverage**:
- **Geographic**: 534 districts across 18 African countries
- **Temporal**: 2021-01 to 2024-12 (48 months)
- **Observations**: ~25,632 district-month combinations

---

## Table of Contents

1. [Core Identifiers](#core-identifiers)
2. [Target Variables](#target-variables)
3. [IPC Assessment Variables](#ipc-assessment-variables)
4. [GDELT Article Variables](#gdelt-article-variables)
5. [GDELT Location Variables](#gdelt-location-variables)
6. [Temporal Lag Features (L_t)](#temporal-lag-features-lt)
7. [Spatial Lag Features (L_s)](#spatial-lag-features-ls)
8. [Ratio Features](#ratio-features)
9. [Z-Score Features](#z-score-features)
10. [HMM Features](#hmm-features)
11. [DMD Features](#dmd-features)
12. [Location Features](#location-features)
13. [Derived Variables](#derived-variables)

---

## Core Identifiers

### country
- **Type**: String (categorical)
- **Description**: ISO 3166-1 alpha-3 country code
- **Values**: 18 unique countries (AGO, BDI, BFA, BWA, CAF, CIV, CMR, COD, COG, DJI, ERI, ETH, GHA, GIN, GMB, KEN, LBR, LSO, MDG, MLI, MOZ, MRT, MWI, NAM, NER, NGA, RWA, SDN, SEN, SLE, SOM, SSD, SWZ, TCD, TGO, TZA, UGA, ZAF, ZMB, ZWE)
- **Example**: "ZWE" (Zimbabwe)
- **Notes**: Analysis uses 18 countries after district threshold filtering (≥8 months data)

### district_name
- **Type**: String (categorical)
- **Description**: IPC Admin Level 2 district name (standardized)
- **Values**: 534 unique districts
- **Example**: "Harare Metropolitan"
- **Notes**: Names standardized from IPC reference data; may differ from GADM names

### period
- **Type**: String (temporal)
- **Format**: "YYYY-MM" for monthly data, "YYYY-QX" for quarterly data
- **Range**: 2021-01 to 2024-12
- **Example**: "2023-06" (June 2023)
- **Notes**: Stage 1 uses quarterly periods, Stage 2 uses monthly periods

### period_start
- **Type**: Date
- **Description**: First day of the period
- **Format**: YYYY-MM-DD
- **Example**: 2023-06-01
- **Notes**: Used for temporal ordering and joins

### period_end
- **Type**: Date
- **Description**: Last day of the period
- **Format**: YYYY-MM-DD
- **Example**: 2023-06-30
- **Notes**: Used for temporal range filtering

---

## Target Variables

### ipc_phase
- **Type**: Integer (ordinal)
- **Description**: IPC food security phase classification
- **Range**: 1-5
- **Values**:
  - 1: Minimal/None
  - 2: Stressed
  - 3: Crisis (WARNING THRESHOLD ⚠️)
  - 4: Emergency
  - 5: Catastrophe/Famine
- **Example**: 3
- **Distribution**: Phase 1-2: 73.2%, Phase 3-5: 26.8%
- **Source**: FEWSNET IPC assessments
- **Notes**: Binary target is derived from this (phases 3-5 = 1)

### ipc_phase_binary
- **Type**: Integer (binary)
- **Description**: Binary target variable for classification
- **Values**:
  - 0: Non-crisis (IPC phases 1-2)
  - 1: Crisis (IPC phases 3-5)
- **Example**: 1
- **Distribution**: 0: 73.2%, 1: 26.8% (class imbalance: 2.73:1)
- **Notes**: Primary target for all models; 4-month lead time in Stage 2

### ipc_phase_binary_h4
- **Type**: Integer (binary)
- **Description**: Binary target with 4-month lead time
- **Formula**: `ipc_phase_binary[t+4]` (4 months ahead)
- **Example**: 1
- **Notes**: Stage 2 uses this for 4-month early warning horizon

### ipc_phase_binary_h8
- **Type**: Integer (binary)
- **Description**: Binary target with 8-month lead time
- **Formula**: `ipc_phase_binary[t+8]` (8 months ahead)
- **Example**: 1
- **Notes**: Experimental longer-horizon prediction (not primary analysis)

---

## IPC Assessment Variables

### ipc_date
- **Type**: Date
- **Description**: Date of IPC assessment publication
- **Format**: YYYY-MM-DD
- **Example**: 2023-05-15
- **Notes**: Assessments typically quarterly; dates may lag actual conditions

### ipc_population
- **Type**: Integer
- **Description**: Total population in assessed area
- **Range**: 1,000 - 5,000,000
- **Example**: 1,485,000
- **Units**: Persons
- **Notes**: Used to weight spatial lag calculations

### ipc_population_phase3plus
- **Type**: Integer
- **Description**: Population in crisis (IPC phases 3-5)
- **Range**: 0 - 5,000,000
- **Example**: 450,000
- **Units**: Persons
- **Notes**: Not directly used in models; for humanitarian context

### crisis_severity
- **Type**: Float
- **Description**: Proportion of population in crisis
- **Formula**: `ipc_population_phase3plus / ipc_population`
- **Range**: 0.0 - 1.0
- **Example**: 0.303 (30.3%)
- **Notes**: Continuous severity measure (not used as target)

---

## GDELT Article Variables

### article_count
- **Type**: Integer
- **Description**: Number of GDELT articles mentioning the district in the period
- **Range**: 0 - 5,000+
- **Median**: 34
- **Mean**: 121
- **Example**: 145
- **Notes**: Core feature for news-based models; high variance across districts

### article_count_ma_3
- **Type**: Float
- **Description**: 3-month moving average of article count
- **Formula**: `mean(article_count[t-2:t])`
- **Example**: 128.3
- **Notes**: Smooths short-term volatility

### article_count_ma_6
- **Type**: Float
- **Description**: 6-month moving average of article count
- **Formula**: `mean(article_count[t-5:t])`
- **Example**: 115.7
- **Notes**: Captures medium-term trends

### article_count_ma_12
- **Type**: Float
- **Description**: 12-month moving average of article count
- **Formula**: `mean(article_count[t-11:t])`
- **Example**: 103.2
- **Notes**: Captures long-term baseline

### unique_sources
- **Type**: Integer
- **Description**: Number of unique news sources (domains) covering district
- **Range**: 0 - 500+
- **Median**: 12
- **Example**: 28
- **Notes**: Proxy for news diversity; correlated with article_count (r=0.87)

---

## GDELT Location Variables

### location_count
- **Type**: Integer
- **Description**: Number of GDELT location mentions for the district
- **Range**: 0 - 10,000+
- **Median**: 87
- **Mean**: 312
- **Example**: 423
- **Notes**: One article can have multiple location mentions; higher than article_count

### location_count_ma_3
- **Type**: Float
- **Description**: 3-month moving average of location count
- **Formula**: `mean(location_count[t-2:t])`
- **Example**: 398.7

### location_count_ma_6
- **Type**: Float
- **Description**: 6-month moving average of location count
- **Formula**: `mean(location_count[t-5:t])`
- **Example**: 356.2

### location_count_ma_12
- **Type**: Float
- **Description**: 12-month moving average of location count
- **Formula**: `mean(location_count[t-11:t])`
- **Example**: 301.5

---

## GDELT Tone Variables

### avg_tone
- **Type**: Float
- **Description**: Average GDELT tone score across all articles
- **Range**: -100 to +100 (theoretical), typically -10 to +5
- **Median**: -1.8
- **Example**: -2.3
- **Interpretation**: Negative = more negative sentiment
- **Notes**: Tone rarely used in final models due to limited predictive power

### positive_tone
- **Type**: Float
- **Description**: Average positive tone score
- **Range**: 0 to +100
- **Example**: 3.2

### negative_tone
- **Type**: Float
- **Description**: Average negative tone score (absolute value)
- **Range**: 0 to +100
- **Example**: 5.5

---

## GDELT Theme Variables

### Theme Counts
**Format**: `theme_{category}_count`
**Type**: Integer
**Description**: Number of articles with specific GDELT theme

**Categories**:
- `theme_conflict_count`: Conflict, violence, protests
- `theme_displacement_count`: Refugees, IDPs, migration
- `theme_economic_count`: Markets, trade, livelihoods
- `theme_weather_count`: Drought, floods, climate
- `theme_food_security_count`: Famine, hunger, food prices
- `theme_health_count`: Disease, malnutrition, epidemics
- `theme_humanitarian_count`: Aid, relief operations
- `theme_governance_count`: Government, policy, institutions
- `theme_other_count`: Unclassified themes

**Range**: 0 - 1,000+
**Example**: `theme_conflict_count = 45`
**Notes**: Themes extracted from GDELT GKG theme codes; one article can have multiple themes

---

## Temporal Lag Features (L_t)

### Lt_1 to Lt_12
- **Type**: Integer (binary)
- **Description**: Lagged binary IPC phase from previous periods
- **Values**: 0 (non-crisis) or 1 (crisis)
- **Example**: `Lt_1 = 1` (crisis in previous period)
- **Indexing**:
  - `Lt_1`: IPC phase 1 period ago (previous quarter/month)
  - `Lt_2`: IPC phase 2 periods ago
  - ...
  - `Lt_12`: IPC phase 12 periods ago
- **Notes**: Core AR baseline features; capture temporal persistence

### Lt_sum_12
- **Type**: Integer
- **Description**: Sum of crisis periods in last 12 periods
- **Formula**: `sum(Lt_1, Lt_2, ..., Lt_12)`
- **Range**: 0 - 12
- **Example**: 7 (crisis in 7 of last 12 periods)
- **Notes**: Proxy for chronic vs acute crisis

---

## Spatial Lag Features (L_s)

### Ls_mean
- **Type**: Float
- **Description**: Population-weighted mean IPC phase of neighboring districts
- **Formula**: `sum(neighbor_ipc * neighbor_population) / sum(neighbor_population)`
- **Range**: 0.0 - 1.0
- **Example**: 0.42
- **Notes**: Captures spatial autocorrelation; neighbors defined by shared border

### Ls_max
- **Type**: Integer (binary)
- **Description**: Maximum IPC phase among neighboring districts
- **Formula**: `max(neighbor_ipc_phase_binary)`
- **Values**: 0 or 1
- **Example**: 1 (at least one neighbor in crisis)
- **Notes**: Binary indicator of nearby crisis

### neighbor_count
- **Type**: Integer
- **Description**: Number of neighboring districts (shared border)
- **Range**: 0 - 15
- **Median**: 5
- **Example**: 7
- **Notes**: Used to normalize spatial lag; border districts may have fewer neighbors

---

## Ratio Features

**Format**: `{theme}_ratio_ma_{window}`
**Type**: Float
**Description**: Moving average of theme proportion

### Formula
```
ratio = theme_count / article_count
ratio_ma_3 = mean(ratio[t-2:t])
```

### Examples
- `conflict_ratio_ma_3`: 3-month MA of conflict article proportion
- `weather_ratio_ma_6`: 6-month MA of weather article proportion
- `displacement_ratio_ma_12`: 12-month MA of displacement article proportion

### Windows
- `ma_3`: 3-month moving average
- `ma_6`: 6-month moving average
- `ma_12`: 12-month moving average

### Range
- 0.0 - 1.0 (proportion of articles)
- Typical: 0.05 - 0.30

### Example
- `conflict_ratio_ma_3 = 0.28` (28% of articles about conflict in last 3 months)

### Notes
- **Capture compositional shifts** in news coverage
- **Window choice**: Longer windows (12m) capture sustained trends
- **Top predictive ratios**: weather (Zimbabwe), conflict (Sudan), displacement (DRC)

---

## Z-Score Features

**Format**: `{theme}_zscore_ma_{window}`
**Type**: Float
**Description**: Rolling standardized theme count

### Formula
```
zscore = (theme_count - rolling_mean) / rolling_std
zscore_ma_3 = mean(zscore[t-2:t])
```

### Rolling Statistics
- **Window**: 12 months
- **Mean**: Historical mean theme count
- **Std**: Historical standard deviation

### Range
- Typically -3.0 to +3.0 (standardized units)
- Extreme values: ±5.0+ (rare events)

### Example
- `conflict_zscore_ma_3 = 2.3` (conflict coverage 2.3 std devs above historical mean)

### Interpretation
- **Positive**: Above-average coverage (potential crisis signal)
- **Negative**: Below-average coverage
- **|z| > 2**: Unusual event (95th percentile)
- **|z| > 3**: Extreme event (99.7th percentile)

### Notes
- **74.7% of model predictions** driven by z-score features (SHAP analysis)
- **Detect rapid anomalies** vs gradual trends
- **Most predictive**: weather_zscore (droughts), conflict_zscore (escalations)

---

## HMM Features

**Hidden Markov Model Features** (2-state binary regimes)

### hmm_state_0_prob
- **Type**: Float
- **Description**: Probability of being in "Pre-Crisis" state
- **Range**: 0.0 - 1.0
- **Example**: 0.65
- **Model**: Gaussian HMM trained per district on ratio features
- **Interpretation**: High value = stable, low-coverage regime

### hmm_state_1_prob
- **Type**: Float
- **Description**: Probability of being in "Crisis-Prone" state
- **Range**: 0.0 - 1.0
- **Example**: 0.35
- **Model**: Gaussian HMM trained per district on ratio features
- **Interpretation**: High value = elevated coverage, potential crisis

### hmm_predicted_state
- **Type**: Integer (binary)
- **Description**: Most likely HMM state (argmax of probabilities)
- **Values**: 0 (Pre-Crisis) or 1 (Crisis-Prone)
- **Example**: 0
- **Notes**: Discrete version of probabilistic states

### hmm_transition_from_0_to_1
- **Type**: Float
- **Description**: Transition probability from Pre-Crisis to Crisis-Prone
- **Range**: 0.0 - 1.0
- **Example**: 0.12
- **Notes**: Derived from HMM transition matrix; measures regime persistence

### hmm_emission_mean_state_0
- **Type**: Float (vector)
- **Description**: Mean emission parameters for Pre-Crisis state
- **Dimension**: 9 themes
- **Example**: `[0.08, 0.05, 0.12, ...]` (conflict, displacement, economic, ...)
- **Notes**: Characterizes "normal" news coverage pattern

### hmm_emission_mean_state_1
- **Type**: Float (vector)
- **Description**: Mean emission parameters for Crisis-Prone state
- **Dimension**: 9 themes
- **Example**: `[0.25, 0.18, 0.08, ...]`
- **Notes**: Characterizes "crisis" news coverage pattern

### Notes
- **Feature Importance**: 3.2% (modest but interpretable)
- **Use Case**: Detects narrative transitions invisible to compositional features
- **Training**: District-specific HMMs (534 separate models)
- **Failure Mode**: Districts with <8 months data excluded (insufficient for HMM convergence)

---

## DMD Features

**Dynamic Mode Decomposition Features** (Rank-5 SVD)

### dmd_amplitude_mode_0 to dmd_amplitude_mode_4
- **Type**: Float
- **Description**: Amplitude of DMD mode (temporal pattern strength)
- **Range**: 0.0 - 1000+ (unbounded)
- **Typical**: 0.1 - 50
- **Example**: `dmd_amplitude_mode_0 = 12.5`
- **Interpretation**: Higher = stronger contribution of this mode to temporal dynamics

### dmd_frequency_mode_0 to dmd_frequency_mode_4
- **Type**: Float (complex)
- **Description**: Frequency of DMD mode (oscillation rate)
- **Range**: -π to +π (radians)
- **Example**: `dmd_frequency_mode_0 = 0.52` (slow oscillation)
- **Interpretation**: Low freq = slow trend, high freq = rapid oscillation

### dmd_growth_rate_mode_0 to dmd_growth_rate_4
- **Type**: Float
- **Description**: Growth rate of DMD mode
- **Range**: -∞ to +∞ (theoretical), typically -1.0 to +1.0
- **Example**: `dmd_growth_rate_mode_2 = 0.15` (exponential growth)
- **Interpretation**: Positive = growing mode, negative = decaying mode

### dmd_reconstruction_error
- **Type**: Float
- **Description**: Reconstruction error of DMD approximation
- **Formula**: `||X - X_dmd|| / ||X||` (normalized Frobenius norm)
- **Range**: 0.0 - 1.0
- **Example**: 0.08
- **Interpretation**: Lower = better DMD fit to data

### Notes
- **Feature Importance**: Very low (<1%) in tree models BUT...
- **Mixed-Effects Coefficient**: +352.38 (largest coefficient)
- **Use Case**: Detects complex emergencies (multiple crisis drivers converging)
- **Failure Mode**: High reconstruction error for erratic coverage patterns
- **Computational Cost**: Rank-5 SVD per district-period (~5ms each)

---

## Location Features

### district_area_km2
- **Type**: Float
- **Description**: District area in square kilometers
- **Range**: 10 - 100,000+ km²
- **Median**: 5,200 km²
- **Example**: 8,450.2
- **Source**: GADM 4.1 administrative boundaries
- **Notes**: Used for density calculations and geographic stratification

### centroid_latitude
- **Type**: Float
- **Description**: Latitude of district centroid
- **Range**: -35° to +20° (Sub-Saharan Africa)
- **Example**: -17.8296
- **Units**: Decimal degrees
- **Notes**: Used for spatial clustering (cross-validation folds)

### centroid_longitude
- **Type**: Float
- **Description**: Longitude of district centroid
- **Range**: -20° to +50° (Sub-Saharan Africa)
- **Example**: 31.0522
- **Units**: Decimal degrees
- **Notes**: Used for spatial clustering (cross-validation folds)

### is_coastal
- **Type**: Integer (binary)
- **Description**: Whether district has coastline
- **Values**: 0 (inland) or 1 (coastal)
- **Example**: 0
- **Source**: Intersection with Natural Earth water bodies
- **Notes**: Coastal districts may have different news coverage patterns

### population_density
- **Type**: Float
- **Description**: Population per square kilometer
- **Formula**: `ipc_population / district_area_km2`
- **Range**: 1 - 5,000+ persons/km²
- **Example**: 175.3
- **Units**: Persons per km²
- **Notes**: Urban vs rural proxy

---

## Derived Variables

### crisis_duration_months
- **Type**: Integer
- **Description**: Number of consecutive months in crisis (IPC 3+)
- **Range**: 0 - 48
- **Example**: 7
- **Notes**: Chronic (>12 months) vs acute (<3 months) crisis classification

### crisis_onset
- **Type**: Integer (binary)
- **Description**: Transition from non-crisis to crisis
- **Formula**: `(ipc_phase_binary[t] == 1) & (ipc_phase_binary[t-1] == 0)`
- **Example**: 1
- **Distribution**: 8.7% of observations
- **Notes**: Key evaluation metric (early warning effectiveness)

### crisis_persistence
- **Type**: Float
- **Description**: Proportion of last 12 months in crisis
- **Formula**: `Lt_sum_12 / 12`
- **Range**: 0.0 - 1.0
- **Example**: 0.58 (crisis 7 of last 12 months)

### news_coverage_density
- **Type**: Float
- **Description**: Article count per 1000 km²
- **Formula**: `article_count / (district_area_km2 / 1000)`
- **Range**: 0 - 500+
- **Example**: 17.2
- **Notes**: Normalizes coverage by district size

### news_desert_indicator
- **Type**: Integer (binary)
- **Description**: District with persistently low news coverage
- **Threshold**: <10 articles/month for ≥6 consecutive months
- **Values**: 0 (normal coverage) or 1 (news desert)
- **Example**: 0
- **Distribution**: 23.4% of districts
- **Notes**: Identifies areas where news-based models likely to fail

---

## Cross-Validation Variables

### spatial_fold
- **Type**: Integer
- **Description**: Spatial fold assignment for cross-validation
- **Range**: 0 - 4 (5-fold CV)
- **Example**: 2
- **Method**: K-means clustering on (latitude, longitude)
- **Notes**: Ensures held-out folds are geographically distinct

### temporal_fold
- **Type**: Integer
- **Description**: Temporal fold assignment (if using temporal CV)
- **Range**: 0 - 4
- **Example**: 1
- **Method**: Sequential time blocks
- **Notes**: Not used in primary analysis (spatial CV preferred)

---

## Prediction Variables

### y_pred_ar
- **Type**: Integer (binary)
- **Description**: Prediction from AR baseline model
- **Values**: 0 or 1
- **Example**: 0

### y_pred_stage2
- **Type**: Integer (binary)
- **Description**: Prediction from Stage 2 XGBoost model
- **Values**: 0 or 1
- **Example**: 1

### y_pred_cascade
- **Type**: Integer (binary)
- **Description**: Final cascade prediction
- **Logic**: `if y_pred_ar == 1: return 1, else: return y_pred_stage2`
- **Example**: 1

### y_proba_ar
- **Type**: Float
- **Description**: Probability from AR baseline (logistic regression)
- **Range**: 0.0 - 1.0
- **Example**: 0.35

### y_proba_stage2
- **Type**: Float
- **Description**: Probability from Stage 2 XGBoost
- **Range**: 0.0 - 1.0
- **Example**: 0.72

### y_proba_cascade
- **Type**: Float
- **Description**: Final cascade probability
- **Formula**: `if y_pred_ar == 1: y_proba_ar, else: y_proba_stage2`
- **Example**: 0.72

---

## Performance Metrics

### auc_roc
- **Type**: Float
- **Description**: Area Under Receiver Operating Characteristic Curve
- **Range**: 0.0 - 1.0 (0.5 = random, 1.0 = perfect)
- **Example**: 0.632
- **Interpretation**: Ability to rank crisis vs non-crisis

### f1_score
- **Type**: Float
- **Description**: Harmonic mean of precision and recall
- **Formula**: `2 * (precision * recall) / (precision + recall)`
- **Range**: 0.0 - 1.0
- **Example**: 0.603

### precision
- **Type**: Float
- **Description**: Proportion of predicted crises that are true crises
- **Formula**: `TP / (TP + FP)`
- **Range**: 0.0 - 1.0
- **Example**: 0.653

### recall (sensitivity)
- **Type**: Float
- **Description**: Proportion of true crises successfully predicted
- **Formula**: `TP / (TP + FN)`
- **Range**: 0.0 - 1.0
- **Example**: 0.866
- **Interpretation**: High recall = few missed crises (low FN)

### specificity
- **Type**: Float
- **Description**: Proportion of true non-crises correctly classified
- **Formula**: `TN / (TN + FP)`
- **Range**: 0.0 - 1.0
- **Example**: 0.426

### brier_score
- **Type**: Float
- **Description**: Mean squared error of probability predictions
- **Formula**: `mean((y_proba - y_true)²)`
- **Range**: 0.0 - 1.0 (lower is better)
- **Example**: 0.185

---

## Data Quality Indicators

### missing_ipc_assessment
- **Type**: Integer (binary)
- **Description**: Whether IPC assessment is missing for period
- **Values**: 0 (assessed) or 1 (missing)
- **Distribution**: 12.3% missing

### imputed_ipc_phase
- **Type**: Integer (binary)
- **Description**: Whether IPC phase was imputed (forward-filled)
- **Values**: 0 (observed) or 1 (imputed)
- **Distribution**: 3.1% imputed
- **Notes**: Max 2-month forward fill; longer gaps left missing

### low_article_count_flag
- **Type**: Integer (binary)
- **Description**: District-period with <5 articles
- **Values**: 0 (normal) or 1 (low coverage)
- **Distribution**: 18.7% flagged
- **Notes**: Low-coverage periods have higher uncertainty

---

## Variable Naming Conventions

### Prefixes
- `ipc_`: IPC assessment variables
- `theme_`: GDELT theme counts
- `Lt_`: Temporal lag features
- `Ls_`: Spatial lag features
- `ratio_`: Ratio features
- `zscore_`: Z-score features
- `hmm_`: Hidden Markov Model features
- `dmd_`: Dynamic Mode Decomposition features
- `y_pred_`: Prediction variables
- `y_proba_`: Probability predictions

### Suffixes
- `_count`: Count variables
- `_ma_3/6/12`: Moving average with window size
- `_binary`: Binary (0/1) variables
- `_h4/h8`: Lead time horizon (4 or 8 months)
- `_km2`: Area in square kilometers
- `_prob`: Probability (0-1)
- `_flag`: Binary indicator

---

## Missing Data

### Patterns
- **IPC assessments**: 12.3% missing (sporadic coverage)
- **GDELT articles**: 0% missing (zeros are valid - no coverage)
- **Spatial neighbors**: 0.8% missing (border districts)
- **HMM features**: 27.2% missing (districts with <8 months data)
- **DMD features**: 15.4% missing (early periods with insufficient history)

### Handling
- **IPC**: Forward-fill up to 2 months, then mark as missing
- **GDELT**: Zeros treated as true absence (not missing)
- **HMM/DMD**: Districts without sufficient data excluded from Stage 2
- **Spatial lags**: Use available neighbors, normalize by count

---

## Units and Scales

| Variable Type | Unit | Typical Scale |
|---------------|------|---------------|
| Counts | Integer | 0 - 5,000 |
| Ratios | Proportion | 0.0 - 1.0 |
| Z-scores | Std devs | -3.0 to +3.0 |
| Probabilities | Proportion | 0.0 - 1.0 |
| Areas | km² | 10 - 100,000 |
| Coordinates | Decimal degrees | -35° to +50° |
| Dates | YYYY-MM-DD | 2021-01-01 to 2024-12-31 |

---

## Data Transformations

### Log Transformations
- `log_article_count = log(article_count + 1)` - Used in exploratory analysis
- `log_location_count = log(location_count + 1)` - Used in exploratory analysis

### Standardization
- Z-score features: 12-month rolling mean and std
- Not applied to binary variables (Lt, Ls, ipc_phase_binary)

### Encoding
- Country: One-hot encoding (18 dummy variables)
- Period: Converted to months since 2021-01 (0-47)
- Binary variables: No encoding needed (0/1)

---

## Data Sources and Updates

### IPC Data
- **Last Updated**: 2024-12-15 (most recent assessment)
- **Update Frequency**: Quarterly to biannual (varies by country)
- **Lag Time**: 1-3 months from reference period to publication

### GDELT Data
- **Last Updated**: 2024-12-31
- **Update Frequency**: Real-time (15-minute updates)
- **Processing Lag**: Monthly aggregation performed Jan 2025

### Geographic Boundaries
- **GADM Version**: 4.1 (released 2022)
- **IPC Boundaries**: Updated 2023-06 (from FEWSNET)
- **Natural Earth**: Version 5.1.1 (2022)

---

## Version History

- **v1.0** (2025-01-10): Initial release for dissertation submission
  - 534 districts, 18 countries
  - 48 months (2021-01 to 2024-12)
  - 25,632 district-month observations

---

## Contact

For questions about variable definitions or data quality:
- **Author**: Victor Collins Oppon
- 
---

**Last Updated**: January 10, 2025
**Author**: Victor Collins Oppon
**MSc Data Science Dissertation, Middlesex University 2025**
