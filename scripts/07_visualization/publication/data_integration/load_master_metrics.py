"""
Master Metrics Data Loader
Loads all data sources for publication-grade dissertation visualizations

NO HARDCODED METRICS - All data pulled from source files:
- COMPLETE_MASTER_METRICS_WORKBOOK.xlsx
- MASTER_METRICS_ALL_MODELS.json

Date: 2026-01-04
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import pandas as pd
import numpy as np
import json
from pathlib import Path
import warnings
from config import BASE_DIR
warnings.filterwarnings('ignore')

# Directory paths
BASE_DIR = Path(rstr(BASE_DIR))
DATA_DIR = BASE_DIR / "Dissertation Write Up" / "DATA_INVENTORIES"
RESULTS_DIR = BASE_DIR / "RESULTS"

# Data source files
EXCEL_FILE = DATA_DIR / "COMPLETE_MASTER_METRICS_WORKBOOK.xlsx"
JSON_FILE = DATA_DIR / "MASTER_METRICS_ALL_MODELS.json"
CASCADE_DIR = RESULTS_DIR / "cascade_optimized_production"

print("="*80)
print("MASTER METRICS DATA LOADER")
print("Loading all data sources for publication visualizations")
print("="*80)

#%%============================================================================
# 1. LOAD EXCEL WORKBOOK (ALL SHEETS)
#==============================================================================

def load_excel_data():
    """Load complete master metrics workbook with all sheets."""
    print("\n[1/4] Loading COMPLETE_MASTER_METRICS_WORKBOOK.xlsx...")

    if not EXCEL_FILE.exists():
        raise FileNotFoundError(f"Excel file not found: {EXCEL_FILE}")

    # Load all sheets
    excel_data = pd.read_excel(EXCEL_FILE, sheet_name=None, engine='openpyxl')

    print(f"   [OK] Loaded {len(excel_data)} sheets:")
    for sheet_name, df in excel_data.items():
        print(f"     - {sheet_name}: {len(df)} rows × {len(df.columns)} cols")

    return excel_data

#%%============================================================================
# 2. LOAD JSON METRICS (ALL MODELS)
#==============================================================================

def load_json_metrics():
    """Load MASTER_METRICS_ALL_MODELS.json with all model variants."""
    print("\n[2/4] Loading MASTER_METRICS_ALL_MODELS.json...")

    if not JSON_FILE.exists():
        raise FileNotFoundError(f"JSON file not found: {JSON_FILE}")

    with open(JSON_FILE, 'r') as f:
        json_data = json.load(f)

    # Extract model names and metrics
    models = list(json_data.keys())
    print(f"   [OK] Loaded {len(models)} model variants:")
    for model_name in models:
        if 'auc_roc_mean' in json_data[model_name]:
            auc = json_data[model_name]['auc_roc_mean']
            print(f"     - {model_name}: AUC-ROC = {auc:.4f}")

    return json_data

#%%============================================================================
# 3. LOAD CASCADE PRODUCTION RESULTS
#==============================================================================

def load_cascade_production():
    """Load cascade optimized production results (predictions, metrics, key saves)."""
    print("\n[3/4] Loading CASCADE production results...")

    cascade_data = {}

    # Predictions
    pred_file = CASCADE_DIR / "predictions_with_metadata.csv"
    if pred_file.exists():
        cascade_data['predictions'] = pd.read_csv(pred_file)
        print(f"   [OK] Predictions: {len(cascade_data['predictions'])} observations")

    # Country metrics
    country_file = CASCADE_DIR / "country_metrics.csv"
    if country_file.exists():
        cascade_data['country_metrics'] = pd.read_csv(country_file)
        print(f"   [OK] Country metrics: {len(cascade_data['country_metrics'])} countries")

    # District metrics
    district_file = CASCADE_DIR / "district_metrics.csv"
    if district_file.exists():
        cascade_data['district_metrics'] = pd.read_csv(district_file)
        print(f"   [OK] District metrics: {len(cascade_data['district_metrics'])} districts")

    # Key saves
    saves_file = CASCADE_DIR / "key_saves_detailed.csv"
    if saves_file.exists():
        cascade_data['key_saves'] = pd.read_csv(saves_file)
        print(f"   [OK] Key saves: {len(cascade_data['key_saves'])} rescued crises")

    return cascade_data

#%%============================================================================
# 4. COMPUTE DERIVED METRICS
#==============================================================================

def compute_derived_metrics(cascade_data, json_data):
    """Compute derived metrics needed for visualizations."""
    print("\n[4/4] Computing derived metrics...")

    derived = {}

    # PR-AUC for AR baseline (for Balashankar comparison)
    if 'predictions' in cascade_data:
        from sklearn.metrics import precision_recall_curve, auc

        preds = cascade_data['predictions']
        y_true = preds['y_true'].values
        y_prob_ar = preds['ar_prob'].values

        precision, recall, _ = precision_recall_curve(y_true, y_prob_ar)
        pr_auc_ar = auc(recall, precision)

        derived['ar_pr_auc'] = pr_auc_ar
        print(f"   [OK] AR baseline PR-AUC: {pr_auc_ar:.4f}")

    # Crisis rate
    if 'predictions' in cascade_data:
        crisis_rate = cascade_data['predictions']['y_true'].mean()
        derived['crisis_rate'] = crisis_rate
        print(f"   [OK] Overall crisis rate: {crisis_rate:.3f} ({crisis_rate*100:.1f}%)")

    # Key saves by country
    if 'key_saves' in cascade_data:
        saves_by_country = cascade_data['key_saves'].groupby('ipc_country').size().sort_values(ascending=False)
        derived['saves_by_country'] = saves_by_country
        print(f"   [OK] Top 3 countries for key saves:")
        for country, count in saves_by_country.head(3).items():
            print(f"     - {country}: {count}")

    # Total observations count
    if 'predictions' in cascade_data:
        derived['n_total'] = len(cascade_data['predictions'])
        derived['n_crisis'] = cascade_data['predictions']['y_true'].sum()
        derived['n_non_crisis'] = len(cascade_data['predictions']) - derived['n_crisis']
        print(f"   [OK] Total observations: {derived['n_total']:,}")
        print(f"   [OK] Crisis cases: {derived['n_crisis']:,}")
        print(f"   [OK] Non-crisis cases: {derived['n_non_crisis']:,}")

    return derived

#%%============================================================================
# MAIN DATA LOADING FUNCTION
#==============================================================================

def load_all_data():
    """
    Load all data sources for dissertation visualizations.

    Returns:
        dict: Dictionary containing:
            - excel_data: All Excel sheets as DataFrames
            - json_data: All JSON model metrics
            - cascade_data: CASCADE production results
            - derived: Computed derived metrics
    """
    try:
        # Load all sources
        excel_data = load_excel_data()
        json_data = load_json_metrics()
        cascade_data = load_cascade_production()
        derived = compute_derived_metrics(cascade_data, json_data)

        print("\n" + "="*80)
        print("[OK] ALL DATA LOADED SUCCESSFULLY")
        print("="*80)

        return {
            'excel': excel_data,
            'json': json_data,
            'cascade': cascade_data,
            'derived': derived
        }

    except Exception as e:
        print(f"\n[ERROR] loading data: {str(e)}")
        raise

#%%============================================================================
# UTILITY FUNCTIONS FOR SPECIFIC VISUALIZATIONS
#==============================================================================

def get_ar_baseline_metrics(data, horizon='h8'):
    """
    Extract AR baseline performance metrics for specified horizon.

    Args:
        data: Master data dictionary
        horizon: 'h4', 'h8', or 'h12' (default: 'h8')

    Returns:
        dict: AR baseline metrics including tp, tn, fp, fn, precision, recall, f1, auc_roc
    """
    json_data = data['json']

    # Check if ar_baseline exists in JSON
    if 'ar_baseline' in json_data and horizon in json_data['ar_baseline']:
        ar_h = json_data['ar_baseline'][horizon]

        # JSON has precision, recall, f1, fp, fn
        # Need to compute tp, tn from these
        fp = ar_h.get('fp', None)
        fn = ar_h.get('fn', None)
        precision = ar_h.get('precision', None)
        recall = ar_h.get('recall', None)

        # Compute tp from precision: precision = tp / (tp + fp)
        # So: tp = precision * (tp + fp), tp = precision * tp + precision * fp
        # tp - precision * tp = precision * fp, tp(1 - precision) = precision * fp
        # tp = (precision * fp) / (1 - precision)
        if precision is not None and fp is not None and precision < 1.0:
            tp = int(round((precision * fp) / (1 - precision)))
        else:
            # Alternative: use recall: recall = tp / (tp + fn)
            # tp = recall * (tp + fn), tp = recall * tp + recall * fn
            # tp(1 - recall) = recall * fn, tp = (recall * fn) / (1 - recall)
            if recall is not None and fn is not None and recall < 1.0:
                tp = int(round((recall * fn) / (1 - recall)))
            else:
                tp = None

        # If we have predictions data, compute directly
        if 'predictions' in data['cascade'] and tp is None:
            from sklearn.metrics import confusion_matrix
            preds = data['cascade']['predictions']
            y_true = preds['y_true'].values
            y_pred_ar = preds['ar_pred'].values
            cm = confusion_matrix(y_true, y_pred_ar)
            tn, fp_calc, fn_calc, tp = cm.ravel()
        else:
            # Compute tn from total observations
            if 'predictions' in data['cascade']:
                n_total = len(data['cascade']['predictions'])
                tn = n_total - tp - fp - fn
            else:
                tn = None

        # Get AUC-ROC from predictions
        auc_roc = None
        if 'predictions' in data['cascade']:
            from sklearn.metrics import roc_auc_score
            preds = data['cascade']['predictions']
            y_true = preds['y_true'].values
            y_prob_ar = preds['ar_prob'].values
            auc_roc = roc_auc_score(y_true, y_prob_ar)

        return {
            'tp': tp,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'precision': precision,
            'recall': recall,
            'f1': ar_h.get('f1', None),
            'auc_roc': auc_roc,
            'optimal_threshold': ar_h.get('optimal_threshold', None),
            'optimal_strategy': ar_h.get('optimal_strategy', None)
        }

    raise ValueError(f"Could not find AR baseline metrics for horizon {horizon} in data")

def get_cascade_metrics(data):
    """Extract cascade ensemble performance metrics."""
    if 'predictions' in data['cascade']:
        preds = data['cascade']['predictions']

        from sklearn.metrics import confusion_matrix, roc_auc_score, precision_recall_curve, auc

        y_true = preds['y_true'].values
        y_pred_cascade = preds['cascade_pred'].values

        cm = confusion_matrix(y_true, y_pred_cascade)
        tn, fp, fn, tp = cm.ravel()

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        return {
            'tp': int(tp),
            'tn': int(tn),
            'fp': int(fp),
            'fn': int(fn),
            'precision': precision,
            'recall': recall,
            'f1': f1
        }

    raise ValueError("Could not find cascade predictions in data")

def get_key_saves_summary(data):
    """Extract key saves summary statistics."""
    if 'key_saves' in data['cascade']:
        saves = data['cascade']['key_saves']

        total_saves = len(saves)
        saves_by_country = saves.groupby('ipc_country').size().sort_values(ascending=False)
        top_3_countries = saves_by_country.head(3)
        top_3_total = top_3_countries.sum()
        top_3_pct = (top_3_total / total_saves * 100) if total_saves > 0 else 0

        return {
            'total': total_saves,
            'by_country': saves_by_country,
            'top_3_countries': top_3_countries,
            'top_3_total': top_3_total,
            'top_3_pct': top_3_pct
        }

    return None

#%%============================================================================
# TEST THE LOADER
#==============================================================================

if __name__ == "__main__":
    print("\n" + "="*80)
    print("TESTING DATA LOADER")
    print("="*80)

    # Load all data
    data = load_all_data()

    # Test AR baseline metrics
    print("\n--- AR Baseline Metrics Test ---")
    ar_metrics = get_ar_baseline_metrics(data)
    print(f"AR Precision: {ar_metrics['precision']:.4f}")
    print(f"AR Recall: {ar_metrics['recall']:.4f}")
    print(f"AR F1: {ar_metrics['f1']:.4f}")
    print(f"AR AUC-ROC: {ar_metrics['auc_roc']:.4f}")
    print(f"AR TP/TN/FP/FN: {ar_metrics['tp']}/{ar_metrics['tn']}/{ar_metrics['fp']}/{ar_metrics['fn']}")

    # Test PR-AUC
    print("\n--- PR-AUC Test ---")
    print(f"AR PR-AUC: {data['derived']['ar_pr_auc']:.4f}")
    print(f"(For comparison with Balashankar 2023 News model: 0.8158)")

    # Test key saves
    print("\n--- Key Saves Test ---")
    saves_summary = get_key_saves_summary(data)
    if saves_summary:
        print(f"Total key saves: {saves_summary['total']}")
        print(f"Top 3 countries: {saves_summary['top_3_pct']:.1f}% of total")
        for country, count in saves_summary['top_3_countries'].items():
            print(f"  - {country}: {count}")

    print("\n" + "="*80)
    print("[OK] DATA LOADER TEST COMPLETE")
    print("="*80)
