"""
Central configuration for dissertation pipeline.
Auto-detects paths based on script location for full portability.

NO HARDCODED PERFORMANCE METRICS - All metrics loaded from results files.

Author: Victor Collins Oppon
MSc Data Science Dissertation
Middlesex University, 2025
"""

from pathlib import Path
import os
import sys

# Auto-detect base directory (works from any location)
BASE_DIR = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = BASE_DIR / "data"
EXTERNAL_DATA_DIR = DATA_DIR / "external"
INTERIM_DATA_DIR = DATA_DIR / "interim"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

# Results directories
RESULTS_DIR = BASE_DIR / "results"
STAGE1_RESULTS = RESULTS_DIR / "stage1_baseline"
STAGE2_RESULTS = RESULTS_DIR / "stage2_models"
CASCADE_RESULTS = RESULTS_DIR / "cascade_optimized"

# Aliases for backward compatibility with original scripts
STAGE1_RESULTS_DIR = STAGE1_RESULTS
STAGE2_RESULTS_DIR = STAGE2_RESULTS
STAGE2_MODELS_DIR = STAGE2_RESULTS / "xgboost"
CASCADE_RESULTS_DIR = CASCADE_RESULTS

# Phase naming (legacy - maps to stages)
PHASE1_RESULTS = STAGE1_RESULTS
PHASE2_RESULTS = STAGE2_RESULTS
PHASE3_RESULTS = STAGE2_RESULTS  # Comparison analysis
PHASE4_RESULTS = CASCADE_RESULTS  # Final cascade results

# Output directories
FIGURES_DIR = BASE_DIR / "figures"
LOGS_DIR = BASE_DIR / "logs"

# Script directories
SCRIPTS_DIR = BASE_DIR / "scripts"

# Source code directory
SRC_DIR = BASE_DIR / "src"

# IPC data paths
IPC_DIR = EXTERNAL_DATA_DIR / "ipc"
IPC_FILE = IPC_DIR / "ipcFic_Africa_Current_Only.csv"
IPC_SOURCE_FILE = IPC_FILE  # Alias

# GDELT data paths
GDELT_DIR = EXTERNAL_DATA_DIR / "gdelt"
GDELT_LOCATIONS = GDELT_DIR / "african_gkg_locations_aligned.parquet"
GDELT_LOCATIONS_FILE = GDELT_LOCATIONS  # Alias
GDELT_ARTICLES_INFO = GDELT_DIR / "LARGE_FILE_LOCATION.txt"

# Shapefiles directories
SHAPEFILES_DIR = EXTERNAL_DATA_DIR / "shapefiles"
IPC_BOUNDARIES = SHAPEFILES_DIR / "ipc_boundaries"
IPC_BOUNDARIES_FILE = IPC_BOUNDARIES  # Alias
NATURAL_EARTH = SHAPEFILES_DIR / "natural_earth"
AFRICA_BASEMAP_FILE = NATURAL_EARTH / "ne_50m_admin_0_countries.shp"  # Common basemap
GADM_DIR = SHAPEFILES_DIR / "gadm"

# Stage 1 intermediate data paths
STAGE1_INTERIM = INTERIM_DATA_DIR / "stage1"
STAGE1_FEATURES = STAGE1_INTERIM / "stage1_features.parquet"
STAGE1_ML_DATASET = STAGE1_INTERIM / "ml_dataset_deduplicated.parquet"
STAGE1_SPATIAL_WEIGHTS = STAGE1_INTERIM / "spatial_weights.parquet"
STAGE1_ARTICLES = STAGE1_INTERIM / "articles_aggregated.parquet"
STAGE1_LOCATIONS = STAGE1_INTERIM / "locations_aggregated.parquet"
IPC_REFERENCE = STAGE1_INTERIM / "ipc_reference.parquet"

# Aliases for backward compatibility
STAGE1_DATA_DIR = STAGE1_INTERIM

# Stage 2 intermediate data paths
STAGE2_INTERIM = INTERIM_DATA_DIR / "stage2"
STAGE2_ML_DATASET = STAGE2_INTERIM / "ml_dataset_monthly.parquet"
STAGE2_ADVANCED_FEATURES = STAGE2_INTERIM / "combined_advanced_features_h8.parquet"
STAGE2_BASIC_FEATURES = STAGE2_INTERIM / "combined_basic_features_h8.parquet"
STAGE2_ARTICLES = STAGE2_INTERIM / "articles_aggregated_monthly.parquet"
STAGE2_LOCATIONS = STAGE2_INTERIM / "locations_aggregated_monthly.parquet"

# Aliases for backward compatibility
STAGE2_DATA_DIR = STAGE2_INTERIM
STAGE2_FEATURES_DIR = STAGE2_INTERIM

# Model parameters (NO hardcoded performance metrics - all loaded from results)
N_FOLDS = 5  # Cross-validation folds
RANDOM_SEED = 42  # For reproducibility
RANDOM_STATE = RANDOM_SEED  # Alias for sklearn compatibility

# Countries included in final analysis (18 countries with sufficient data)
# Note: Dissertation mentions "24 African countries" referring to raw IPC data coverage,
# but actual cascade analysis uses 18 countries after district threshold filtering
COUNTRIES = [
    'BDI',  # Burundi
    'BFA',  # Burkina Faso
    'CMR',  # Cameroon
    'COD',  # Democratic Republic of the Congo
    'ETH',  # Ethiopia
    'KEN',  # Kenya
    'MDG',  # Madagascar
    'MLI',  # Mali
    'MOZ',  # Mozambique
    'MWI',  # Malawi
    'NER',  # Niger
    'NGA',  # Nigeria
    'SDN',  # Sudan
    'SOM',  # Somalia
    'SSD',  # South Sudan
    'TCD',  # Chad
    'UGA',  # Uganda
    'ZWE',  # Zimbabwe
]

# ISO-3 to full country name mapping
COUNTRY_NAMES = {
    'BDI': 'Burundi',
    'BFA': 'Burkina Faso',
    'CMR': 'Cameroon',
    'COD': 'Democratic Republic of the Congo',
    'ETH': 'Ethiopia',
    'KEN': 'Kenya',
    'MDG': 'Madagascar',
    'MLI': 'Mali',
    'MOZ': 'Mozambique',
    'MWI': 'Malawi',
    'NER': 'Niger',
    'NGA': 'Nigeria',
    'SDN': 'Sudan',
    'SOM': 'Somalia',
    'SSD': 'South Sudan',
    'TCD': 'Chad',
    'UGA': 'Uganda',
    'ZWE': 'Zimbabwe',
}

# Temporal parameters
DATE_RANGE_START = "2021-01-01"
DATE_RANGE_END = "2024-12-31"
IPC_ASSESSMENT_FREQUENCY_MONTHS = 4  # IPC assessments every 4 months

# Spatial parameters (from dissertation config)
SPATIAL_DISTANCE_KM = 300  # Distance for spatial lag calculation
ADMIN_LEVEL = 2  # GADM administrative level (ADM2 = districts)

# Feature engineering parameters (from actual config.py)
MOVING_WINDOW_SIZES = [3, 6, 9, 12]  # Months for moving averages
ZSCORE_WINDOW = 12  # 12-month rolling z-scores
ZSCORE_MIN_PERIODS = 3  # Minimum months for z-score

# HMM parameters (from config.py - REDESIGNED Dec 24, 2025)
HMM_N_STATES = 2  # Binary regime: Pre-Crisis vs Crisis-Prone
HMM_MIN_SEQUENCE_LENGTH = 6  # Minimum months of data required
HMM_ROLLING_WINDOW = 12  # Rolling window size
HMM_CRISIS_PERSISTENCE_MIN = 0.85  # Minimum P(Crisis→Crisis)
HMM_N_ITER = 100
HMM_INPUT_FEATURES = [
    'food_security_ratio',
    'conflict_ratio',
    'economic_ratio',
    'weather_ratio'
]
HMM_OUTPUT_FEATURES = [
    'crisis_prob',
    'transition_risk',
    'entropy'
]

# DMD parameters (from config.py - REDESIGNED Dec 24, 2025)
DMD_SVD_RANK = 5  # Number of modes to extract (NOT 10!)
DMD_TLSQ_RANK = 3  # Total least squares rank
DMD_MIN_SEQUENCE_LENGTH = 8  # Minimum months of data required
DMD_ROLLING_WINDOW = 12  # Rolling window size
DMD_GROWTH_THRESHOLD = 0.01  # 1% monthly growth minimum
DMD_FREQUENCY_MIN = 1/6  # 6-month max period
DMD_FREQUENCY_MAX = 1/2  # 2-month min period
DMD_CRISIS_WEIGHT = 1.0
DMD_CONTEXTUAL_WEIGHT = 0.5
DMD_CRISIS_CATEGORIES = ['conflict', 'food_security', 'displacement', 'humanitarian']
DMD_CONTEXTUAL_CATEGORIES = ['economic']
DMD_OUTPUT_FEATURES = [
    'crisis_growth_rate',
    'crisis_instability',
    'crisis_frequency',
    'crisis_amplitude'
]

# XGBoost hyperparameters (from actual training script)
XGBOOST_PARAMS_DEFAULT = {
    'objective': 'binary:logistic',
    'random_state': RANDOM_SEED,
    'verbosity': 0,
    'use_label_encoder': False,
    'eval_metric': 'logloss',
    'tree_method': 'hist',
    'n_jobs': -1,
}

# GridSearchCV parameter grid (from 01_xgboost_basic_WITH_AR_FILTER_OPTIMIZED.py)
XGBOOST_PARAM_GRID = {
    'n_estimators': [100, 200, 300],        # 3 options
    'max_depth': [5, 7, 10],                # 3 options
    'learning_rate': [0.01, 0.05, 0.1],     # 3 options
    'min_child_weight': [1, 3, 5],          # 3 options
    'subsample': [0.7, 0.8],                # 2 options
    'colsample_bytree': [0.6, 0.8],         # 2 options
    'gamma': [0, 0.5, 1],                   # 3 options
    'reg_alpha': [0, 0.1],                  # 2 options
    'reg_lambda': [1, 2]                    # 2 options
}
# Total combinations: 3*3*3*3*2*2*3*2*2 = 3,888 configurations

# Cascade decision logic (from 05_cascade_ensemble_optimized_production.py)
# CASCADE STRATEGY (Simple Binary Logic):
# 1. Use AR baseline predictions as primary
# 2. If AR = 1: Keep as 1 (trust AR's crisis prediction)
# 3. If AR = 0: Use Stage 2's binary prediction
#    - If Stage 2 = 1: Override to 1 (Stage 2 detected crisis)
#    - If Stage 2 = 0: Keep as 0 (both agree: no crisis)
CASCADE_STRATEGY = "binary_override"  # Not a threshold-based approach

# Cost-sensitive evaluation (from cascade script)
COST_FN = 10  # Missing a crisis is 10x worse than false alarm
COST_FP = 1   # False positive cost

# Stage 2 filter (WITH_AR_FILTER)
WITH_AR_FILTER = {
    'description': 'IPC <= 2 AND AR prediction = 0',
    'ipc_max': 2,
    'ar_pred_required': 0,
}

# District threshold configuration
DISTRICT_THRESHOLD_MIN_ARTICLES = 200  # Minimum articles per year
DISTRICT_THRESHOLD_MIN_DISTRICTS = 5   # Minimum valid districts per country

# Visualization parameters
FIGURE_DPI = 300  # DPI for publication-quality figures
FIGURE_FORMAT = 'pdf'  # Primary format for dissertation figures
COLOR_PALETTE = {
    'crisis': '#d62728',      # Red
    'non_crisis': '#2ca02c',  # Green
    'warning': '#ff7f0e',     # Orange
    'uncertain': '#9467bd',   # Purple
    'ar_baseline': '#1f77b4', # Blue
    'xgboost': '#e377c2',     # Pink
    'cascade': '#8c564b',     # Brown
}

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'

# Create directories if they don't exist
def ensure_directories():
    """Create all required directories if they don't exist."""
    directories = [
        DATA_DIR, EXTERNAL_DATA_DIR, INTERIM_DATA_DIR, PROCESSED_DATA_DIR,
        IPC_DIR, GDELT_DIR, SHAPEFILES_DIR, IPC_BOUNDARIES, NATURAL_EARTH, GADM_DIR,
        STAGE1_INTERIM, STAGE2_INTERIM,
        RESULTS_DIR, STAGE1_RESULTS, STAGE2_RESULTS, CASCADE_RESULTS,
        FIGURES_DIR, LOGS_DIR,
    ]

    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    return True

def ensure_path_exists(path):
    """
    Ensure a path exists, create parent directories if needed.

    Args:
        path: Path object or string to ensure exists

    Returns:
        Path object
    """
    path = Path(path)
    if not path.parent.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
    return path

def get_project_root():
    """
    Get the project root directory.

    Returns:
        Path object pointing to project root
    """
    return BASE_DIR

def add_src_to_path():
    """Add src directory to Python path for imports."""
    src_path = str(SRC_DIR)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

# Auto-add src to path when config is imported
add_src_to_path()

def get_stage1_results_path(horizon='h8'):
    """
    Get path to Stage 1 results for a specific horizon.

    Args:
        horizon: Forecast horizon (h4, h8, or h12)

    Returns:
        Path object
    """
    return STAGE1_RESULTS / f"predictions_{horizon}_averaged.parquet"

def get_stage2_model_path(model_type='advanced', optimized=True):
    """
    Get path to Stage 2 model.

    Args:
        model_type: 'basic' or 'advanced'
        optimized: Whether to get optimized version

    Returns:
        Path object
    """
    suffix = '_optimized' if optimized else ''
    model_dir = model_type + '_with_ar' + suffix
    return STAGE2_RESULTS / "xgboost" / model_dir / f"xgboost_{model_type}{suffix}_final.pkl"

def get_cascade_results_path():
    """
    Get path to cascade results.

    Returns:
        Path object
    """
    return CASCADE_RESULTS / "cascade_optimized_predictions.csv"

# Version information
VERSION = "1.0.0"
DISSERTATION_TITLE = "Dynamic News Signals as Early-Warning Indicators of Food Insecurity: A Two-Stage Residual Modelling Framework"
AUTHOR = "Victor Collins Oppon"
INSTITUTION = "Middlesex University"
PROGRAM = "MSc Data Science"
YEAR = 2025

# Legacy CONFIG dictionaries for backward compatibility
STAGE1_CONFIG = {
    'data_dir': STAGE1_DATA_DIR,
    'results_dir': STAGE1_RESULTS_DIR,
    'random_state': RANDOM_STATE,
    'n_folds': N_FOLDS,
}

DISTRICT_THRESHOLD_CONFIG = {
    'min_articles_per_year': DISTRICT_THRESHOLD_MIN_ARTICLES,
    'min_districts_per_country': DISTRICT_THRESHOLD_MIN_DISTRICTS,
}

FEATURE_CONFIG = {
    'hmm_n_states': HMM_N_STATES,
    'hmm_min_sequence': HMM_MIN_SEQUENCE_LENGTH,
    'dmd_svd_rank': DMD_SVD_RANK,
    'dmd_min_sequence': DMD_MIN_SEQUENCE_LENGTH,
    'moving_window_sizes': MOVING_WINDOW_SIZES,
    'zscore_window': ZSCORE_WINDOW,
}

VISUALIZATION_CONFIG = {
    'figures_dir': FIGURES_DIR,
    'dpi': FIGURE_DPI,
    'format': FIGURE_FORMAT,
    'colors': COLOR_PALETTE,
}

OUTPUT_FILES = {
    'stage1_predictions': STAGE1_RESULTS / 'predictions_h8_averaged.parquet',
    'stage2_predictions': STAGE2_RESULTS / 'xgboost' / 'predictions_advanced_optimized.csv',
    'cascade_predictions': CASCADE_RESULTS / 'cascade_optimized_predictions.csv',
}

# Initialize directories on import
ensure_directories()

# Print configuration info (can be disabled in production)
if __name__ == "__main__":
    print(f"\n{'='*70}")
    print(f"  {DISSERTATION_TITLE}")
    print(f"  {AUTHOR} - {PROGRAM}, {INSTITUTION} {YEAR}")
    print(f"  Version: {VERSION}")
    print(f"{'='*70}\n")
    print(f"Project Root: {BASE_DIR}")
    print(f"Data Directory: {DATA_DIR}")
    print(f"Results Directory: {RESULTS_DIR}")
    print(f"Scripts Directory: {SCRIPTS_DIR}")
    print(f"Source Directory: {SRC_DIR}")
    print(f"\nCountries: {len(COUNTRIES)} African countries")
    print(f"Cross-validation folds: {N_FOLDS}")
    print(f"Date range: {DATE_RANGE_START} to {DATE_RANGE_END}")
    print(f"Spatial distance: {SPATIAL_DISTANCE_KM} km")
    print(f"\nHMM: {HMM_N_STATES} states, {HMM_MIN_SEQUENCE_LENGTH} min months")
    print(f"DMD: Rank {DMD_SVD_RANK}, {DMD_MIN_SEQUENCE_LENGTH} min months")
    print(f"XGBoost GridSearch: {3*3*3*3*2*2*3*2*2:,} configurations")
    print(f"\nCascade strategy: {CASCADE_STRATEGY}")
    print(f"  - If AR = 1: Keep as 1 (trust AR)")
    print(f"  - If AR = 0 and Stage 2 = 1: Override to 1")
    print(f"  - If AR = 0 and Stage 2 = 0: Keep as 0")
    print(f"\nAll directories initialized successfully!")
    print(f"{'='*70}\n")
