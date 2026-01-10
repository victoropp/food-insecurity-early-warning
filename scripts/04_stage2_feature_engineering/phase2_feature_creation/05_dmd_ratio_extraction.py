"""
DMD (Dynamic Mode Decomposition) Ratio Mode Extraction (REDESIGNED)
=====================================================================
Phase 2, Step 4b: Extract crisis-focused dynamic modes from ratio features.

REDESIGN (December 24, 2025):
- Crisis-focused mode selection (growth>0.01, frequency in [1/6,1/2], category weighting)
- Input: 15 features (5 crisis categories × 3 derivatives: ratio, delta, trend)
- Output: 4 crisis features (growth_rate, instability, frequency, amplitude)
- Category weighting: crisis=1.0 (conflict, food_security, displacement, humanitarian), economic=0.5

RESEARCH PROPOSAL ALIGNMENT:
"Apply DMD to extract crisis-predictive dynamic modes, filtering seasonal noise
and market volatility to focus on escalating multi-sector crisis patterns."

Features Created:
- dmd_ratio_crisis_growth_rate: dominant crisis mode exponential growth
- dmd_ratio_crisis_instability: crisis-weighted sum of growing modes
- dmd_ratio_crisis_frequency: dominant crisis oscillation period (months)
- dmd_ratio_crisis_amplitude: dominant crisis mode strength

Author: Victor Collins Oppon, Claude Code
Date: December 2025 (Redesigned: Dec 24, 2025)
"""

import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent.parent))

# Import from config
from config import (
    BASE_DIR,
    STAGE1_DATA_DIR,
    STAGE1_RESULTS_DIR,
    STAGE2_DATA_DIR,
    STAGE2_FEATURES_DIR,
    STAGE2_MODELS_DIR,
    FIGURES_DIR,
    RANDOM_STATE,
    FEATURE_CONFIG
)

import pandas as pd
import numpy as np
from datetime import datetime
import warnings
from scipy import linalg

warnings.filterwarnings('ignore')

# Define Phase 2 output directory
PHASE2_RESULTS = STAGE2_FEATURES_DIR / 'phase2_features'
PHASE2_RESULTS.mkdir(parents=True, exist_ok=True)

def ensure_directories():
    """Ensure output directories exist."""
    PHASE2_RESULTS.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("XGBOOST PIPELINE - PHASE 2: DMD RATIO FEATURES")
print("=" * 80)

# DMD Parameters (from config - REDESIGNED)
N_MODES = FEATURE_CONFIG['dmd_svd_rank']
MIN_SEQUENCE_LENGTH = FEATURE_CONFIG['dmd_min_sequence_length']
ROLLING_WINDOW = FEATURE_CONFIG['dmd_rolling_window']
REGULARIZATION = 1e-6  # SVD regularization

# REDESIGNED: Crisis-focused parameters
CRISIS_CATEGORIES = FEATURE_CONFIG['dmd_crisis_categories']  # 4 core
CONTEXTUAL_CATEGORIES = FEATURE_CONFIG['dmd_contextual_categories']  # 1 (economic)
CRISIS_WEIGHT = FEATURE_CONFIG['dmd_crisis_weight']  # 1.0
CONTEXTUAL_WEIGHT = FEATURE_CONFIG['dmd_contextual_weight']  # 0.5
GROWTH_THRESHOLD = FEATURE_CONFIG['dmd_growth_threshold']  # 0.01
FREQ_MIN = FEATURE_CONFIG['dmd_frequency_min']  # 1/6
FREQ_MAX = FEATURE_CONFIG['dmd_frequency_max']  # 1/2

print(f"DMD Configuration (REDESIGNED):")
print(f"  N_MODES (SVD rank): {N_MODES}")
print(f"  MIN_SEQUENCE_LENGTH: {MIN_SEQUENCE_LENGTH} months")
print(f"  ROLLING_WINDOW: {ROLLING_WINDOW} months")
print(f"  REGULARIZATION: {REGULARIZATION}")
print(f"  CRISIS_CATEGORIES: {CRISIS_CATEGORIES} (weight={CRISIS_WEIGHT})")
print(f"  CONTEXTUAL_CATEGORIES: {CONTEXTUAL_CATEGORIES} (weight={CONTEXTUAL_WEIGHT})")
print(f"  GROWTH_THRESHOLD: {GROWTH_THRESHOLD} (crisis escalation)")
print(f"  FREQUENCY_RANGE: [{FREQ_MIN:.3f}, {FREQ_MAX:.3f}] (periods: 2-6 months)")

# REDESIGNED: Crisis-focused features (5 categories × 3 derivatives = 15 features)
# Core crisis categories: conflict, food_security, displacement, humanitarian
# Contextual category: economic (filtered by weight=0.5)
ALL_CRISIS_CATS = CRISIS_CATEGORIES + CONTEXTUAL_CATEGORIES

DMD_RATIO_FEATURES = []
for cat in ALL_CRISIS_CATS:
    DMD_RATIO_FEATURES.extend([
        f'{cat}_ratio',
        f'{cat}_delta',
        f'{cat}_trend_3m'
    ])


def load_hmm_ratio_features():
    """Load HMM ratio features (which include ratios)."""
    print("\n   Loading HMM ratio features...")

    hmm_file = PHASE2_RESULTS / 'hmm_ratio_features_h8.parquet'
    if hmm_file.exists():
        df = pd.read_parquet(hmm_file)
    else:
        csv_file = PHASE2_RESULTS / 'hmm_ratio_features_h8.csv'
        if csv_file.exists():
            df = pd.read_csv(csv_file)
        else:
            raise FileNotFoundError("HMM ratio features not found. Run 03_hmm_ratio_regime_extraction.py first.")

    print(f"   Loaded {len(df):,} observations")

    # FORCE use of canonical district identifier
    district_col = 'ipc_geographic_unit_full'
    if district_col not in df.columns:
        raise ValueError(
            f"Required column '{district_col}' not found in data. "
            f"Available columns: {df.columns.tolist()}"
        )

    # Sort by district and time
    df = df.sort_values([district_col, 'year_month']).reset_index(drop=True)

    return df, district_col


def dmd_analysis(X, r=None, regularization=1e-6):
    """Perform Dynamic Mode Decomposition on data matrix X."""
    n_features, n_timesteps = X.shape

    if n_timesteps < 3:
        return None

    X1 = X[:, :-1]
    X2 = X[:, 1:]

    col_std = np.std(X1, axis=1)
    valid_rows = col_std > 1e-10
    if valid_rows.sum() < 2:
        return None

    X1_valid = X1[valid_rows, :]
    X2_valid = X2[valid_rows, :]

    try:
        U, S, Vh = linalg.svd(X1_valid, full_matrices=False)
    except:
        return None

    S_regularized = np.where(S > regularization * S[0], S, regularization * S[0])

    if r is None:
        energy_ratio = np.cumsum(S ** 2) / np.sum(S ** 2)
        r = max(2, np.searchsorted(energy_ratio, 0.99) + 1)
        r = min(r, len(S), n_timesteps - 1)

    r = min(r, len(S), U.shape[1], Vh.shape[0])

    if r < 1:
        return None

    U_r = U[:, :r]
    S_r = S_regularized[:r]
    Vh_r = Vh[:r, :]

    try:
        A_tilde = U_r.T @ X2_valid @ Vh_r.T @ np.diag(1 / S_r)
    except:
        return None

    try:
        eigenvalues, W = linalg.eig(A_tilde)
    except:
        return None

    Phi = X2_valid @ Vh_r.T @ np.diag(1 / S_r) @ W

    x0 = X1_valid[:, 0]
    try:
        amplitudes = linalg.lstsq(Phi, x0)[0]
    except:
        amplitudes = np.ones(len(eigenvalues))

    omega = np.log(eigenvalues + 1e-10)
    growth_rates = np.real(omega)
    frequencies = np.abs(np.imag(omega)) / (2 * np.pi)

    return {
        'modes': Phi,
        'eigenvalues': eigenvalues,
        'amplitudes': amplitudes,
        'growth_rates': growth_rates,
        'frequencies': frequencies,
        'rank': r,
        'valid_rows': valid_rows  # FIX: Track which features are valid
    }


def compute_crisis_mode_weights(modes, feature_names):
    """Compute crisis-relevance weight for each DMD mode.

    REDESIGN: Weight modes by contribution from crisis-relevant categories.
    - Crisis categories (conflict, food_security, displacement, humanitarian): weight = 1.0
    - Contextual categories (economic): weight = 0.5
    - Excluded categories: weight = 0.0

    Returns:
        crisis_weights: array of shape (n_modes,) with crisis-relevance scores [0, 1]
    """
    n_modes = modes.shape[1]
    crisis_weights = np.zeros(n_modes)

    for mode_idx in range(n_modes):
        mode_vector = modes[:, mode_idx]
        mode_abs = np.abs(mode_vector)
        total_contribution = np.sum(mode_abs)

        if total_contribution < 1e-10:
            crisis_weights[mode_idx] = 0.0
            continue

        crisis_contribution = 0.0
        for feat_idx, feat_name in enumerate(feature_names):
            # Check if feature belongs to crisis or contextual categories
            for crisis_cat in CRISIS_CATEGORIES:
                if crisis_cat in feat_name:
                    crisis_contribution += mode_abs[feat_idx] * CRISIS_WEIGHT
                    break
            else:  # No break - not a crisis category
                for context_cat in CONTEXTUAL_CATEGORIES:
                    if context_cat in feat_name:
                        crisis_contribution += mode_abs[feat_idx] * CONTEXTUAL_WEIGHT
                        break

        # Normalize by total contribution
        crisis_weights[mode_idx] = crisis_contribution / total_contribution

    return crisis_weights


def filter_crisis_modes(dmd_result, feature_names):
    """Filter DMD modes to select only crisis-predictive dynamics.

    REDESIGN: 3-step crisis mode filter
    1. Growth threshold: λ > GROWTH_THRESHOLD (escalation, not decay)
    2. Frequency filter: period in [2, 6] months (crisis timescale, not seasonal)
    3. Category weighting: High contribution from crisis categories

    Returns:
        crisis_mode_indices: indices of crisis-predictive modes (sorted by crisis score)
    """
    growth_rates = dmd_result['growth_rates']
    frequencies = dmd_result['frequencies']
    amplitudes = np.abs(dmd_result['amplitudes'])
    modes = dmd_result['modes']
    valid_rows = dmd_result['valid_rows']

    # FIX: Only use valid feature names (those that passed variance check)
    valid_feature_names = [fname for i, fname in enumerate(feature_names) if valid_rows[i]]

    # Compute crisis weights
    crisis_weights = compute_crisis_mode_weights(modes, valid_feature_names)

    # Step 1: Growth threshold (escalation)
    growth_mask = growth_rates > GROWTH_THRESHOLD

    # Step 2: Frequency filter (crisis timescale, 2-6 month periods)
    freq_mask = (frequencies >= FREQ_MIN) & (frequencies <= FREQ_MAX)

    # Step 3: Crisis category weighting (threshold = 0.3, must have significant crisis contribution)
    crisis_mask = crisis_weights > 0.3

    # Combine all filters
    valid_mask = growth_mask & freq_mask & crisis_mask

    if not valid_mask.any():
        return np.array([])

    # Score by crisis_weight × growth_rate × amplitude (multi-sector escalating crises)
    crisis_scores = crisis_weights[valid_mask] * growth_rates[valid_mask] * amplitudes[valid_mask]
    valid_indices = np.where(valid_mask)[0]

    # Sort by crisis score (descending)
    sorted_order = np.argsort(crisis_scores)[::-1]

    return valid_indices[sorted_order]


def extract_dmd_features_for_district(X, available_features, n_modes=5):
    """Extract crisis-focused DMD features for a single district's time series.

    REDESIGN: Only 4 crisis features extracted from filtered crisis modes.
    """
    n_timesteps = X.shape[0]

    # REDESIGNED: Only 4 output features
    features = {
        'dmd_ratio_crisis_growth_rate': np.nan,
        'dmd_ratio_crisis_instability': np.nan,
        'dmd_ratio_crisis_frequency': np.nan,
        'dmd_ratio_crisis_amplitude': np.nan
    }

    if n_timesteps < MIN_SEQUENCE_LENGTH:
        return features

    X_t = X.T

    col_means = np.nanmean(X_t, axis=1, keepdims=True)
    col_means = np.where(np.isnan(col_means), 0, col_means)
    X_t = np.where(np.isnan(X_t), col_means, X_t)

    dmd_result = dmd_analysis(X_t, regularization=REGULARIZATION)

    if dmd_result is None:
        return features

    # REDESIGN: Apply crisis mode filter
    crisis_mode_indices = filter_crisis_modes(dmd_result, available_features)

    if len(crisis_mode_indices) == 0:
        # No crisis modes found - return zeros (stable dynamics)
        features['dmd_ratio_crisis_growth_rate'] = 0.0
        features['dmd_ratio_crisis_instability'] = 0.0
        features['dmd_ratio_crisis_frequency'] = 0.0
        features['dmd_ratio_crisis_amplitude'] = 0.0
        return features

    # Extract crisis mode properties
    growth_rates = dmd_result['growth_rates'][crisis_mode_indices]
    amplitudes = np.abs(dmd_result['amplitudes'][crisis_mode_indices])
    frequencies = dmd_result['frequencies'][crisis_mode_indices]
    modes = dmd_result['modes'][:, crisis_mode_indices]
    valid_rows = dmd_result['valid_rows']

    # FIX: Only use valid feature names (those that passed variance check)
    valid_feature_names = [fname for i, fname in enumerate(available_features) if valid_rows[i]]

    # Compute crisis weights for filtered modes
    crisis_weights = compute_crisis_mode_weights(modes, valid_feature_names)

    # REDESIGN: 4 crisis features
    # 1. Dominant crisis mode growth rate (exponential escalation)
    features['dmd_ratio_crisis_growth_rate'] = growth_rates[0]

    # 2. Crisis instability score (crisis-weighted sum of growing modes)
    features['dmd_ratio_crisis_instability'] = np.sum(
        growth_rates * amplitudes * crisis_weights
    )

    # 3. Dominant crisis frequency (oscillation period in months)
    features['dmd_ratio_crisis_frequency'] = frequencies[0]

    # 4. Dominant crisis amplitude (strength of crisis mode)
    features['dmd_ratio_crisis_amplitude'] = amplitudes[0]

    return features


def extract_dmd_features_rolling(df, district_col):
    """Extract crisis-focused DMD features using rolling windows.

    REDESIGN: Only 4 crisis output features.
    """
    print("\n" + "-" * 40)
    print("Extracting DMD ratio features (rolling window, REDESIGNED)...")
    print(f"   Window size: {ROLLING_WINDOW} months")

    available_features = [f for f in DMD_RATIO_FEATURES if f in df.columns]
    print(f"   Using {len(available_features)} crisis-focused features for DMD (REDESIGNED)")
    print(f"   Features: {available_features}")

    if len(available_features) == 0:
        print("   ERROR: No DMD ratio features found!")
        return df

    # REDESIGN: Initialize only 4 output columns
    dmd_cols = [
        'dmd_ratio_crisis_growth_rate',
        'dmd_ratio_crisis_instability',
        'dmd_ratio_crisis_frequency',
        'dmd_ratio_crisis_amplitude'
    ]

    for col in dmd_cols:
        df[col] = np.nan

    districts = df[district_col].unique()
    n_districts = len(districts)

    success_count = 0
    fail_count = 0

    for i, district in enumerate(districts):
        if (i + 1) % 200 == 0 or i == 0:
            print(f"   Processing district {i+1}/{n_districts}...", flush=True)

        district_mask = df[district_col] == district
        district_data = df.loc[district_mask].copy()

        if len(district_data) < MIN_SEQUENCE_LENGTH:
            fail_count += 1
            continue

        district_data = district_data.sort_values('year_month')
        district_idx = district_data.index.tolist()

        X_full = district_data[available_features].values

        district_success = False

        for t_idx in range(MIN_SEQUENCE_LENGTH - 1, len(district_data)):
            # FIX ISSUE #4: DMD MODE LEAKAGE
            # CRITICAL: DMD window must end at t-1 (exclude current month)
            # Original (WRONG): window = [t-11, ..., t] includes current month
            # Fixed (CORRECT): window = [t-12, ..., t-1] uses only historical data
            #
            # We predict at time t using data from [t-12, ..., t-1]
            # This ensures no current-month information leaks into predictions
            start_idx = max(0, t_idx - ROLLING_WINDOW)
            X_window = X_full[start_idx:t_idx, :]  # FIXED: Exclude current time point (t_idx)

            if len(X_window) >= MIN_SEQUENCE_LENGTH:
                # REDESIGN: Pass available_features for crisis mode weighting
                features = extract_dmd_features_for_district(X_window, available_features, n_modes=N_MODES)

                current_idx = district_idx[t_idx]
                for col, val in features.items():
                    df.loc[current_idx, col] = val

                # REDESIGN: Check new feature name
                if not np.isnan(features['dmd_ratio_crisis_instability']):
                    district_success = True

        if district_success:
            success_count += 1
        else:
            fail_count += 1

    print(f"\n   DMD ratio extraction complete:")
    print(f"   Successful: {success_count:,} districts")
    print(f"   Failed: {fail_count:,} districts")
    print(f"   Coverage: {100 * success_count / n_districts:.1f}%")

    return df


def main():
    """Main execution function."""
    ensure_directories()

    print(f"\nStart time: {datetime.now()}")

    # Load HMM ratio features (which include ratios)
    df, district_col = load_hmm_ratio_features()

    # Extract DMD features
    df = extract_dmd_features_rolling(df, district_col)

    # Save features
    print("\n" + "-" * 40)
    print("Saving DMD ratio features...")

    output_path = PHASE2_RESULTS / 'hmm_dmd_ratio_features_h8.parquet'
    df.to_parquet(output_path, index=False)
    print(f"   Saved: {output_path}")

    csv_path = PHASE2_RESULTS / 'hmm_dmd_ratio_features_h8.csv'
    df.to_csv(csv_path, index=False)
    print(f"   Saved: {csv_path}")

    # Summary
    print("\n" + "=" * 80)
    print("DMD RATIO FEATURE SUMMARY (REDESIGNED)")
    print("=" * 80)

    # REDESIGN: Only 4 crisis output features
    dmd_cols = [
        'dmd_ratio_crisis_growth_rate',
        'dmd_ratio_crisis_instability',
        'dmd_ratio_crisis_frequency',
        'dmd_ratio_crisis_amplitude'
    ]
    for col in dmd_cols:
        if col in df.columns:
            valid = df[col].notna().sum()
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f"   {col}: {valid:,} valid, mean={mean_val:.4f}, std={std_val:.4f}")

    # Report crisis mode detection rate
    crisis_detected = df['dmd_ratio_crisis_instability'].notna().sum()
    total_obs = len(df)
    print(f"\n   Crisis modes detected: {crisis_detected:,} / {total_obs:,} ({100*crisis_detected/total_obs:.1f}%)")

    print("\n" + "=" * 80)
    print("PHASE 2 STEP 4b COMPLETE: DMD Ratio Features (REDESIGNED)")
    print("=" * 80)
    print(f"End time: {datetime.now()}")
    print(f"\nREDESIGN SUMMARY:")
    print(f"  - Input features: {len(DMD_RATIO_FEATURES)} crisis-focused (5 categories × 3 derivatives)")
    print(f"  - Crisis categories: {CRISIS_CATEGORIES} (weight={CRISIS_WEIGHT})")
    print(f"  - Contextual categories: {CONTEXTUAL_CATEGORIES} (weight={CONTEXTUAL_WEIGHT})")
    print(f"  - Output features: 4 (81% reduction from 21)")
    print(f"  - Mode selection: 3-step crisis filter (growth>0.01, freq in [1/6,1/2], category weighting)")


if __name__ == '__main__':
    main()
