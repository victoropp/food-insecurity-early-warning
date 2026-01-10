# Data Directory - Food Insecurity Early Warning System

**Author**: Victor Collins Oppon
**Institution**: Middlesex University
**Program**: MSc Data Science
**Year**: 2025

---

## Overview

This directory contains all data used in the dissertation "Food Insecurity Early Warning System Using GDELT News Events and Machine Learning". The data is organized into three subdirectories:

- **external/** - Source data from external providers (2.2 GB)
- **interim/** - Intermediate processed data (715 MB)
- **processed/** - Final model-ready datasets (referenced in interim/)

**Total Size**: ~2.9 GB

---

## Data Structure

```
data/
├── README.md (this file)
├── external/
│   ├── ipc/                          # IPC food security assessments
│   ├── gdelt/                        # GDELT news event data
│   └── shapefiles/                   # Geographic boundaries
├── interim/
│   ├── stage1/                       # Stage 1 processed data
│   └── stage2/                       # Stage 2 processed data
└── processed/
    └── (symbolic links to interim/stage2/)
```

---

## External Data Sources

### 1. IPC Food Security Assessments

**Source**: Famine Early Warning Systems Network (FEWSNET)
**URL**: https://fews.net/fews-data/335
**File**: `external/ipc/ipcFic_Africa_Current_Only.csv` (24 MB)

**Description**:
- Integrated Food Security Phase Classification (IPC) assessments for African countries
- Coverage: 2021-01 to 2024-12
- Granularity: District-level (Admin Level 2)
- Records: 55,129 district-period observations
- Variables: Country, district name, IPC phase (1-5), population, dates

**Citation**:
```
Famine Early Warning Systems Network (FEWSNET). (2024).
Integrated Food Security Phase Classification (IPC) data.
Retrieved from https://fews.net/fews-data/335
```

**License**: Public domain - U.S. Government work

**IPC Phase Classification**:
- Phase 1: Minimal/None
- Phase 2: Stressed
- Phase 3: Crisis ⚠️ (WARNING THRESHOLD)
- Phase 4: Emergency
- Phase 5: Catastrophe/Famine

**Countries Covered** (41 total):
AGO, BDI, BEN, BFA, BWA, CAF, CIV, CMR, COD, COG, DJI, ERI, ETH, GHA, GIN, GMB, KEN, LBR, LSO, MDG, MLI, MOZ, MRT, MWI, NAM, NER, NGA, RWA, SDN, SEN, SLE, SOM, SSD, SWZ, TCD, TGO, TZA, UGA, ZAF, ZMB, ZWE

**Quality Notes**:
- Some districts have missing data for certain periods
- Geographic coverage varies by country
- Assessments are typically quarterly or biannual

---

### 2. GDELT News Event Data

**Source**: GDELT Project (Global Database of Events, Language, and Tone)
**URL**: https://www.gdeltproject.org/
**Files**:
- `external/gdelt/african_gkg_locations_aligned.parquet` (613 MB)
- `external/gdelt/LARGE_FILE_LOCATION.txt` (points to original 47 GB CSV)

**Description**:
- GDELT 2.0 Global Knowledge Graph (GKG) news articles
- Coverage: 2021-01 to 2024-12
- Location mentions: 5.2 million mentions across Africa
- Aligned to IPC district boundaries using spatial joins

**Citation**:
```
Leetaru, K., & Schrodt, P. A. (2013).
GDELT: Global data on events, location and tone.
ISA Annual Convention, 2, 1-49.
```

**License**: Creative Commons Attribution 4.0 International (CC BY 4.0)

**Variables**:
- Article URL, publication date
- Location coordinates (latitude, longitude)
- Tone (average, positive, negative)
- Matched IPC district
- Themes (comma-separated GDELT theme codes)

**Processing Notes**:
- Original data downloaded from GDELT BigQuery
- Filtered to African continent bounding box
- Spatially joined to IPC district boundaries
- Location mentions aggregated to district-period level

**Large File Warning**:
The original CSV file (`african_gkg_full_2021_2024.csv`) is **47 GB** and is NOT included in this repository. The processed parquet file (613 MB) contains all necessary location mentions pre-aggregated. To reproduce from scratch, see `external/gdelt/LARGE_FILE_LOCATION.txt` for download instructions.

---

### 3. Geographic Boundaries

**Source**: Multiple providers
**Directory**: `external/shapefiles/`

#### 3.1 IPC Administrative Boundaries

**Source**: FEWSNET/IPC Technical Working Group
**Directory**: `external/shapefiles/ipc_boundaries/`
**Size**: ~158 MB
**Format**: Shapefiles (.shp, .shx, .dbf, .prj)

**Description**:
- Official IPC administrative boundaries (Admin Level 2)
- Used by FEWSNET for IPC assessments
- Matches district names in IPC CSV data
- Includes country, admin1, admin2 names

**Files**:
- `ipc_admin2_*.shp` - District-level boundaries for each country

#### 3.2 GADM Administrative Boundaries

**Source**: GADM (Database of Global Administrative Areas)
**URL**: https://gadm.org/
**Version**: 4.1
**Directory**: `external/shapefiles/gadm/`
**Size**: ~1.4 GB
**Format**: GeoPackage (.gpkg)

**Description**:
- Global administrative boundaries
- Used for supplementary geographic analysis
- Coverage: All African countries at Admin Level 0, 1, 2

**Citation**:
```
GADM. (2022). GADM database of Global Administrative Areas, version 4.1.
Retrieved from https://gadm.org/
```

**License**: Free for academic use

**Download Script**: `external/shapefiles/download_gadm_africa.py`

#### 3.3 Natural Earth Boundaries

**Source**: Natural Earth
**URL**: https://www.naturalearthdata.com/
**Directory**: `external/shapefiles/natural_earth/`
**Size**: ~4 MB
**Format**: Shapefiles

**Description**:
- Country-level boundaries (Admin Level 0)
- Used for map visualizations
- Scale: 1:50m (medium detail)

**Citation**:
```
Natural Earth. (2022). Natural Earth, Free vector and raster map data.
Retrieved from https://www.naturalearthdata.com/
```

**License**: Public domain

**Files**:
- `ne_50m_admin_0_countries_africa.shp` - African country boundaries
- `ne_50m_lakes.shp` - Major lakes
- `ne_50m_rivers_lake_centerlines.shp` - Rivers

---

## Interim Data

**Directory**: `interim/`
**Size**: 715 MB

Intermediate processed data generated by pipeline scripts.

### Stage 1: Quarterly Aggregation

**Directory**: `interim/stage1/`
**Size**: 246 MB

**Files**:
- `ipc_reference.parquet` (1.4 MB) - Processed IPC reference data
- `articles_aggregated.parquet` (4.7 MB) - GDELT articles aggregated to district-quarter
- `locations_aggregated.parquet` (3.4 MB) - GDELT locations aggregated to district-quarter
- `ml_dataset_complete.parquet` (8.2 MB) - Merged IPC + GDELT dataset
- `ml_dataset_deduplicated.parquet` (5.5 MB) - Unique district-quarter observations
- `stage1_features.parquet` (5.7 MB) ⭐ **CRITICAL** - AR baseline features (Lt, Ls)
- `spatial_weights.parquet` (12 MB) - Spatial adjacency matrix
- `district_summaries.parquet` (205 MB) - Detailed district-level summaries

**Variables** (stage1_features.parquet):
- District identifiers: country, district_name, period
- Target: ipc_phase_binary (0=phases 1-2, 1=phases 3-5)
- Temporal lags: Lt_1 to Lt_12 (previous 12 quarters)
- Spatial lags: Ls_mean, Ls_max (neighboring districts)
- GDELT counts: article_count, location_count, unique_sources
- GDELT tone: avg_tone, positive_tone, negative_tone

**Coverage**:
- Districts: 1,920
- Quarters: 2021-Q1 to 2024-Q4
- Observations: ~30,720 (1,920 × 16 quarters)
- After deduplication: ~15,000 (districts with IPC assessments)

### Stage 2: Monthly Aggregation

**Directory**: `interim/stage2/`
**Size**: 469 MB

**Files**:
- `articles_aggregated_monthly.parquet` (11 MB) - GDELT articles by district-month
- `locations_aggregated_monthly.parquet` (2.9 MB) - GDELT locations by district-month
- `ml_dataset_monthly.parquet` (16 MB) ⭐ **CRITICAL** - Monthly IPC + GDELT dataset
- `combined_basic_features_h8.parquet` (5.6 MB) - Ratio + Z-score features (8-month horizon)
- `combined_advanced_features_h8.parquet` (7.4 MB) ⭐ **CRITICAL** - All features including HMM, DMD
- `phase1_analysis/` (200 MB) - District threshold analysis outputs
- `phase2_features/` (225 MB) - Individual feature engineering outputs

**Variables** (combined_advanced_features_h8.parquet):
- All stage1 variables (Lt, Ls)
- Basic temporal features:
  - Ratio features: moving averages (3, 6, 12 months)
  - Z-score features: standardized article/location counts
- Advanced temporal features:
  - HMM features: hidden state probabilities (3 states)
  - DMD features: dynamic mode amplitudes, frequencies
- District-specific features: historical means, standard deviations
- Target: ipc_phase_binary (4-month lead time)

**Coverage**:
- Districts: 534 (after threshold filtering ≥8 months of data)
- Months: 2021-01 to 2024-12
- Observations: ~25,632 (534 × 48 months)
- Training set: ~20,000 (after 4-month lead time)

**Threshold Filtering**:
Stage 2 requires districts with ≥8 months of consecutive GDELT data to enable HMM/DMD feature extraction. This reduces district count from 1,920 (Stage 1) to 534 (Stage 2).

---

## Processed Data

**Directory**: `processed/`
**Size**: 0 (symbolic links to interim/stage2/)

This directory contains symbolic links to final model-ready datasets in `interim/stage2/`. The separation is conceptual (following cookiecutter-data-science conventions) but physically both are stored in `interim/`.

**Key Datasets**:
- `combined_advanced_features_h8.parquet` → Final dataset for XGBoost advanced model
- `combined_basic_features_h8.parquet` → Final dataset for XGBoost basic model
- `ml_dataset_monthly.parquet` → Final dataset for AR baseline monthly model

---

## Data Quality Notes

### Missing Data

1. **IPC Assessments**: Not all districts have assessments for all periods
   - Some countries have only quarterly assessments
   - Conflict zones may have missing data
   - Assessments are released with delays (1-3 months)

2. **GDELT Coverage**: News coverage varies significantly by country
   - High coverage: Nigeria, Kenya, Ethiopia, South Africa
   - Low coverage: Small countries, rural districts
   - Bias toward English-language sources

3. **Geographic Matching**: ~5% of GDELT locations could not be matched to IPC districts
   - Locations outside IPC boundary polygons
   - Coordinate precision issues
   - Unrecognized district names

### Data Cleaning Applied

1. **IPC Data**:
   - Removed duplicate district-period combinations
   - Standardized district names (capitalization, special characters)
   - Filtered to countries with ≥8 quarters of data

2. **GDELT Data**:
   - Removed articles with missing coordinates
   - Filtered to African continent bounding box (±40° latitude/longitude)
   - Removed duplicate articles (same URL, same date)
   - Capped outliers (tone values beyond ±100)

3. **Spatial Matching**:
   - Used 0.01° coordinate precision (~1 km)
   - Spatial joins with st_intersects (PostGIS)
   - Manual verification for major cities

---

## Reproducibility

### Data Provenance

All data files include metadata headers documenting:
- Creation date
- Source script
- Software versions (pandas, geopandas, Python)
- Random seeds (where applicable)

### Checksums

SHA256 checksums for key files:

```
# IPC source data
ipcFic_Africa_Current_Only.csv: [checksum to be added]

# GDELT source data
african_gkg_locations_aligned.parquet: [checksum to be added]

# Critical intermediate files
stage1_features.parquet: [checksum to be added]
ml_dataset_monthly.parquet: [checksum to be added]
combined_advanced_features_h8.parquet: [checksum to be added]
```

### Regenerating Interim Data

To regenerate all interim data from source:

```bash
# Stage 1: Quarterly aggregation
python scripts/02_data_processing/aggregate_articles.py
python scripts/02_data_processing/aggregate_locations.py
python scripts/02_data_processing/create_ml_dataset.py
python scripts/02_data_processing/deduplicate.py

# Stage 1: Feature engineering
python scripts/03_stage1_baseline/feature_engineering.py

# Stage 2: Monthly aggregation
python scripts/04_stage2_data_processing/aggregate_articles_monthly.py
python scripts/04_stage2_data_processing/aggregate_locations_monthly.py
python scripts/04_stage2_data_processing/create_ml_dataset.py

# Stage 2: Feature engineering (3 phases)
python scripts/05_stage2_feature_engineering/phase1_*/analyze_thresholds.py
python scripts/05_stage2_feature_engineering/phase2_*/create_*_features.py
python scripts/05_stage2_feature_engineering/phase3_*/combine_features.py
```

**Runtime**: ~4 hours total

---

## Data Archive for Download

**Large data files (>100 MB) are hosted separately on Zenodo.**

### Archive Contents

The data archive includes:
- All external/ data sources (2.2 GB)
- All interim/ processed data (715 MB)
- Checksums and verification scripts

**Download**: [Zenodo DOI link to be added]

**Mirror**: [Alternative hosting to be added if needed]

### Installation

```bash
# Download data archive
wget https://zenodo.org/record/XXXXXXX/files/data_archive.zip

# Extract to data/ directory
unzip data_archive.zip -d data/

# Verify checksums
python scripts/verify_data_checksums.py
```

---

## Data Dictionary

For detailed variable descriptions, see `DATA_DICTIONARY.md` (to be created).

**Quick reference**:

| Variable | Description | Type | Range |
|----------|-------------|------|-------|
| country | ISO 3-letter country code | str | AGO-ZWE |
| district_name | IPC Admin Level 2 district name | str | - |
| period | Quarterly period (YYYY-QX) or monthly (YYYY-MM) | str | 2021-Q1 to 2024-Q4 |
| ipc_phase | IPC phase classification | int | 1-5 |
| ipc_phase_binary | Binary target (0=phases 1-2, 1=phases 3-5) | int | 0-1 |
| article_count | Number of GDELT articles mentioning district | int | 0-5000+ |
| location_count | Number of GDELT location mentions | int | 0-10000+ |
| avg_tone | Average GDELT tone score | float | -100 to +100 |
| Lt_1 to Lt_12 | Temporal lag features (previous periods) | int | 0-1 |
| Ls_mean | Spatial lag (mean of neighbors) | float | 0-1 |
| ratio_ma_3/6/12 | Moving average ratios | float | 0-10+ |
| zscore_* | Z-score normalized features | float | -5 to +5 |
| hmm_state_* | HMM state probabilities | float | 0-1 |
| dmd_amplitude_* | DMD mode amplitudes | float | 0-1000+ |

---

## Ethical Considerations

### Data Privacy

- No personally identifiable information (PII) in any dataset
- News articles are public domain (URLs only, not article text)
- IPC assessments are aggregated population-level data
- No individual-level food security information

### Bias and Limitations

1. **News Coverage Bias**:
   - Urban areas over-represented vs rural areas
   - Conflict/crisis events over-represented vs slow-onset emergencies
   - English-language sources dominate

2. **IPC Assessment Bias**:
   - Assessments prioritize crisis zones
   - Data-poor countries have fewer assessments
   - Political factors may influence reporting

3. **Geographic Bias**:
   - East Africa well-covered (Ethiopia, Kenya, Somalia)
   - West Africa moderately covered (Nigeria, Sahel)
   - Central Africa under-covered (CAR, DRC)

These biases are discussed in detail in the dissertation (Chapter 6, Section 6.2).

---

## License and Usage

### Data Licenses

- **IPC Data**: Public domain (U.S. Government work)
- **GDELT Data**: CC BY 4.0 (attribution required)
- **GADM Shapefiles**: Free for academic use
- **Natural Earth**: Public domain

### Citation Requirements

When using this data, please cite:

**For the dataset**:
```
Oppon, V. C. (2025). Data Archive - Food Insecurity Early Warning System
Using GDELT and IPC [Data set]. Zenodo. https://doi.org/10.5281/zenodo.XXXXXXX
```

**For the dissertation**:
```
Oppon, V. C. (2025). Food Insecurity Early Warning System Using GDELT
News Events and Machine Learning [Master's thesis, Middlesex University].
```

**For original data sources**: See citations under each data source above.

---

## Support

For questions about the data:
- **GitHub Issues**: [Repository link to be added]
- **Email**: [Contact email to be added]
- **Documentation**: See dissertation Appendix E (Data Documentation)

---

## Version History

- **v1.0.0** (2025-01-10): Initial release for dissertation submission
  - IPC data: 2021-01 to 2024-12 (55,129 records)
  - GDELT data: 2021-01 to 2024-12 (5.2M location mentions)
  - Stage 1: 1,920 districts, quarterly aggregation
  - Stage 2: 534 districts, monthly aggregation
  - All processing scripts validated and tested

---

**Last Updated**: January 10, 2025
**Author**: Victor Collins Oppon
**MSc Data Science Dissertation, Middlesex University 2025**
