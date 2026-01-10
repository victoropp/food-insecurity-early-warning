"""
HMM Z-Score Regime Extraction (REDESIGNED)
===========================================
Phase 2, Step 3a: Apply Hidden Markov Models to z-score features to identify latent narrative regimes.

REDESIGN (December 24, 2025):
- Binary regime: Pre-Crisis (0) vs Crisis-Prone (1)
- 4 core input features: food_security, conflict, economic, weather (z-score variants)
- 3 output features: crisis_prob, transition_risk, entropy
- Asymmetric transitions: Crisis persistence constraint (P(Crisis→Crisis) > 0.85)
- District-level pooling: 1,322 separate HMMs (reduced parameters per model)

RESEARCH PROPOSAL ALIGNMENT:
"Apply Hidden Markov Models (HMM) to the sequence of macro-category z-scores
to identify latent regimes aligned with IPC Phase 3 crisis threshold."

Features Created:
- hmm_zscore_crisis_prob: P(state=Crisis-Prone)
- hmm_zscore_transition_risk: P(next_state=Crisis | current_state)
- hmm_zscore_entropy: state uncertainty

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
from hmmlearn import hmm

warnings.filterwarnings('ignore')

# Define Phase 2 output directory
PHASE2_RESULTS = STAGE2_FEATURES_DIR / 'phase2_features'
PHASE2_RESULTS.mkdir(parents=True, exist_ok=True)

def ensure_directories():
    """Ensure output directories exist."""
    PHASE2_RESULTS.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PHASE 2: HMM Z-SCORE REGIME EXTRACTION")
print("=" * 80)

# HMM Parameters (from config)
N_STATES = FEATURE_CONFIG['hmm_n_states']  # 2 states (redesigned)
MIN_SEQUENCE_LENGTH = FEATURE_CONFIG['hmm_min_sequence_length']
ROLLING_WINDOW = FEATURE_CONFIG['hmm_rolling_window']
HMM_INPUT_FEATURES = FEATURE_CONFIG['hmm_input_features']  # 4 core features
HMM_OUTPUT_FEATURES = FEATURE_CONFIG['hmm_output_features']  # 3 outputs
CRISIS_PERSISTENCE_MIN = FEATURE_CONFIG['hmm_crisis_persistence_min']  # 0.85

print(f"HMM Configuration (REDESIGNED):")
print(f"  N_STATES: {N_STATES} (Binary regime: Pre-Crisis vs Crisis-Prone)")
print(f"  INPUT_FEATURES (z-score): {[f.replace('_ratio', '_zscore') for f in HMM_INPUT_FEATURES]}")
print(f"  OUTPUT_FEATURES: {HMM_OUTPUT_FEATURES}")
print(f"  MIN_SEQUENCE_LENGTH: {MIN_SEQUENCE_LENGTH} months")
print(f"  ROLLING_WINDOW: {ROLLING_WINDOW} months")
print(f"  CRISIS_PERSISTENCE: >={CRISIS_PERSISTENCE_MIN} (asymmetric transitions)")

# REDESIGNED: Only 4 core z-score features (no deltas - HMM models dynamics natively)
# Convert ratio feature names to zscore variants
HMM_ZSCORE_FEATURES = [f.replace('_ratio', '_zscore') for f in HMM_INPUT_FEATURES]


def load_zscore_features():
    """Load z-score features from Phase 2."""
    print("\n   Loading z-score features...")

    zscore_file = PHASE2_RESULTS / 'zscore_features_h8.parquet'
    if zscore_file.exists():
        df = pd.read_parquet(zscore_file)
    else:
        csv_file = PHASE2_RESULTS / 'zscore_features_h8.csv'
        if csv_file.exists():
            df = pd.read_csv(csv_file)
        else:
            raise FileNotFoundError("Z-score features not found. Run 02_zscore_feature_engineering.py first.")

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


def fit_hmm(X, n_states=2, random_state=42):
    """Fit Gaussian HMM to observation sequence with asymmetric transition constraint.

    REDESIGN: Binary regime (Pre-Crisis=0, Crisis-Prone=1) with crisis persistence.
    """
    if len(X) < MIN_SEQUENCE_LENGTH:
        return None

    # Handle missing values
    col_means = np.nanmean(X, axis=0)
    for j in range(X.shape[1]):
        mask = np.isnan(X[:, j])
        X[mask, j] = col_means[j] if not np.isnan(col_means[j]) else 0

    # Handle constant features
    feature_stds = np.std(X, axis=0)
    for j in range(X.shape[1]):
        if feature_stds[j] == 0:
            X[:, j] = X[:, j] + np.random.randn(len(X)) * 0.01

    try:
        model = hmm.GaussianHMM(
            n_components=n_states,
            covariance_type='diag',
            n_iter=300,  # IMPROVED: Increased from 100 to 300 for better convergence
            tol=1e-3,    # IMPROVED: Explicit tolerance threshold
            random_state=random_state,
            init_params='stmc',  # IMPROVED: Initialize starts/transitions/means/covariances
            verbose=False
        )

        # REDESIGN: Initialize asymmetric transition matrix
        # State 0 (Pre-Crisis): Can transition to Crisis
        # State 1 (Crisis-Prone): High persistence (asset depletion irreversibility)
        if n_states == 2:
            model.transmat_ = np.array([
                [0.80, 0.20],  # Pre-Crisis: 80% stay, 20% worsen
                [0.10, 0.90]   # Crisis: 10% recover, 90% persist
            ])

        model.fit(X)

        # Enforce crisis persistence constraint after fitting
        if n_states == 2 and model.transmat_[1, 1] < CRISIS_PERSISTENCE_MIN:
            # Adjust transition matrix to meet constraint
            recovery_prob = 1 - CRISIS_PERSISTENCE_MIN
            model.transmat_[1, :] = [recovery_prob, CRISIS_PERSISTENCE_MIN]

        hidden_states = model.predict(X)
        state_probs = model.predict_proba(X)
        transition_matrix = model.transmat_

        return {
            'hidden_states': hidden_states,
            'state_probs': state_probs,
            'transition_matrix': transition_matrix,
            'converged': model.monitor_.converged,
            'n_iter': model.monitor_.iter
        }

    except Exception as e:
        return None


def order_states_by_ipc(hmm_result, ipc_values):
    """Order HMM states based on IPC outcomes (supervised labeling).

    CORRECTED (Dec 24, 2025): ipc_values should be lag-1 only (length = len(hidden_states) - 1)
    to prevent data leakage. We use historical IPC to label states, excluding current time.
    """
    hidden_states = hmm_result['hidden_states']
    state_probs = hmm_result['state_probs']

    # CRITICAL FIX: Use lag-1 IPC only (exclude current time to prevent leakage)
    # ipc_values has length = len(hidden_states) - 1
    # We use historical states (excluding last) for state ordering
    historical_states = hidden_states[:-1]  # Exclude current time state

    state_ipc_means = []
    for s in range(N_STATES):
        state_mask = historical_states == s
        valid_ipc = ipc_values[state_mask]
        valid_ipc = valid_ipc[~np.isnan(valid_ipc)]

        if len(valid_ipc) > 0:
            mean_ipc = np.mean(valid_ipc)
        else:
            mean_ipc = 0
        state_ipc_means.append((s, mean_ipc))

    state_order = sorted(state_ipc_means, key=lambda x: x[1])
    old_to_new = {old: new for new, (old, _) in enumerate(state_order)}

    new_hidden_states = np.array([old_to_new[s] for s in hidden_states])

    new_state_probs = np.zeros_like(state_probs)
    for old, new in old_to_new.items():
        new_state_probs[:, new] = state_probs[:, old]

    trans = hmm_result['transition_matrix']
    new_trans = np.zeros_like(trans)
    for old_from, new_from in old_to_new.items():
        for old_to, new_to in old_to_new.items():
            new_trans[new_from, new_to] = trans[old_from, old_to]

    return new_hidden_states, new_state_probs, new_trans


def compute_transition_risk(state_probs, transition_matrix, crisis_state=1):
    """Compute probability of transitioning to crisis state.

    REDESIGN: Binary states, so crisis_state=1 (Crisis-Prone).
    """
    transition_risk = np.zeros(len(state_probs))

    for t in range(len(state_probs)):
        for s in range(N_STATES):
            transition_risk[t] += state_probs[t, s] * transition_matrix[s, crisis_state]

    return transition_risk


def compute_entropy(state_probs):
    """Compute entropy of state distribution."""
    eps = 1e-10
    entropy = -np.sum(state_probs * np.log(state_probs + eps), axis=1)
    return entropy


def extract_hmm_features_for_window(X_window, ipc_window):
    """Extract HMM features for a single rolling window.

    REDESIGN: Only 3 outputs (crisis_prob, transition_risk, entropy).
    """
    features = {
        'hmm_zscore_crisis_prob': np.nan,
        'hmm_zscore_transition_risk': np.nan,
        'hmm_zscore_entropy': np.nan,
        'hmm_zscore_converged': 0
    }

    if len(X_window) < MIN_SEQUENCE_LENGTH:
        return features

    hmm_result = fit_hmm(X_window.copy(), n_states=N_STATES, random_state=RANDOM_STATE)

    if hmm_result is None:
        return features

    try:
        hidden_states, state_probs, trans_matrix = order_states_by_ipc(
            hmm_result, ipc_window
        )
    except:
        return features

    transition_risk = compute_transition_risk(state_probs, trans_matrix, crisis_state=1)  # Binary: crisis=1
    entropy = compute_entropy(state_probs)

    last_idx = len(hidden_states) - 1
    # REDESIGN: Only 3 features
    # crisis_prob = P(Crisis-Prone state) = state_probs[:, 1]
    features['hmm_zscore_crisis_prob'] = state_probs[last_idx, 1]  # State 1 = Crisis-Prone
    features['hmm_zscore_transition_risk'] = transition_risk[last_idx]
    features['hmm_zscore_entropy'] = entropy[last_idx]
    features['hmm_zscore_converged'] = int(hmm_result['converged'])

    return features


def extract_hmm_features(df, district_col):
    """Extract HMM features for all districts using rolling windows.

    REDESIGN: Only 3 output features (crisis_prob, transition_risk, entropy).
    """
    print("\n" + "-" * 40)
    print("Extracting HMM z-score features (rolling window, REDESIGNED)...")
    print(f"   Window size: {ROLLING_WINDOW} months")

    available_features = [f for f in HMM_ZSCORE_FEATURES if f in df.columns]
    print(f"   Using {len(available_features)} core features for HMM (REDESIGNED)")
    print(f"   Features: {available_features}")

    if len(available_features) == 0:
        print("   ERROR: No HMM z-score features found!")
        return df

    # REDESIGN: Initialize only 3 output columns
    df['hmm_zscore_crisis_prob'] = np.nan
    df['hmm_zscore_transition_risk'] = np.nan
    df['hmm_zscore_entropy'] = np.nan
    df['hmm_zscore_converged'] = 0

    districts = df[district_col].unique()
    n_districts = len(districts)

    success_count = 0
    fail_count = 0
    convergence_fail_count = 0

    for i, district in enumerate(districts):
        if (i + 1) % 200 == 0 or i == 0:
            print(f"   Processing district {i+1}/{n_districts}... (convergence failures: {convergence_fail_count})", flush=True)

        district_mask = df[district_col] == district
        district_data = df.loc[district_mask].copy()

        if len(district_data) < MIN_SEQUENCE_LENGTH:
            fail_count += 1
            continue

        district_data = district_data.sort_values('year_month')
        district_idx = district_data.index.tolist()

        X_full = district_data[available_features].values

        ipc_col = 'ipc_value_filled' if 'ipc_value_filled' in district_data.columns else 'ipc_value'
        if ipc_col in district_data.columns:
            ipc_full = district_data[ipc_col].values
        else:
            ipc_full = np.zeros(len(X_full))

        district_success = False

        for t_idx in range(MIN_SEQUENCE_LENGTH - 1, len(district_data)):
            start_idx = max(0, t_idx - ROLLING_WINDOW + 1)

            # DATA LEAKAGE FIX (December 24, 2025):
            # Use rolling window including current time for HMM fitting (OK - uses only GDELT features)
            X_window = X_full[start_idx:t_idx + 1, :]

            # CRITICAL FIX: Exclude current IPC from state ordering to prevent leakage
            # IPC window must use LAG-1 ONLY (exclude current time t)
            # Otherwise, current IPC value influences how we label states → circular dependency
            # At prediction time, we won't have current IPC, so states would be mislabeled
            ipc_window = ipc_full[start_idx:t_idx]  # CORRECTED: Exclude current time (lag-1 only)

            if len(X_window) >= MIN_SEQUENCE_LENGTH:
                features = extract_hmm_features_for_window(X_window, ipc_window)

                current_idx = district_idx[t_idx]
                # REDESIGN: Only 3 features
                df.loc[current_idx, 'hmm_zscore_crisis_prob'] = features['hmm_zscore_crisis_prob']
                df.loc[current_idx, 'hmm_zscore_transition_risk'] = features['hmm_zscore_transition_risk']
                df.loc[current_idx, 'hmm_zscore_entropy'] = features['hmm_zscore_entropy']
                df.loc[current_idx, 'hmm_zscore_converged'] = features['hmm_zscore_converged']

                if not np.isnan(features['hmm_zscore_crisis_prob']):
                    district_success = True

                # Track convergence failures
                if features['hmm_zscore_converged'] == 0:
                    convergence_fail_count += 1

        if district_success:
            success_count += 1
        else:
            fail_count += 1

    print(f"\n   HMM z-score extraction complete:")
    print(f"   Successful: {success_count:,} districts")
    print(f"   Failed: {fail_count:,} districts")
    print(f"   Coverage: {100 * success_count / n_districts:.1f}%")
    print(f"   Convergence failures: {convergence_fail_count:,} observations")

    return df


def main():
    """Main execution function."""
    ensure_directories()

    print(f"\nStart time: {datetime.now()}")

    # Load z-score features
    df, district_col = load_zscore_features()

    # Extract HMM features
    df = extract_hmm_features(df, district_col)

    # Save features
    print("\n" + "-" * 40)
    print("Saving HMM z-score features...")

    output_path = PHASE2_RESULTS / 'hmm_zscore_features_h8.parquet'
    df.to_parquet(output_path, index=False)
    print(f"   Saved: {output_path}")

    csv_path = PHASE2_RESULTS / 'hmm_zscore_features_h8.csv'
    df.to_csv(csv_path, index=False)
    print(f"   Saved: {csv_path}")

    # Summary
    print("\n" + "=" * 80)
    print("HMM Z-SCORE FEATURE SUMMARY (REDESIGNED)")
    print("=" * 80)

    # REDESIGN: Only 3 output features
    hmm_cols = ['hmm_zscore_crisis_prob', 'hmm_zscore_transition_risk', 'hmm_zscore_entropy']
    for col in hmm_cols:
        if col in df.columns:
            valid = df[col].notna().sum()
            mean_val = df[col].mean()
            std_val = df[col].std()
            print(f"   {col}: {valid:,} valid, mean={mean_val:.4f}, std={std_val:.4f}")

    converged_count = df['hmm_zscore_converged'].sum()
    total_obs = len(df)
    print(f"\n   Convergence: {converged_count:,} / {total_obs:,} ({100*converged_count/total_obs:.1f}%)")

    print("\n" + "=" * 80)
    print("PHASE 2 STEP 3a COMPLETE: HMM Z-Score Features (REDESIGNED)")
    print("=" * 80)
    print(f"End time: {datetime.now()}")
    print(f"\nREDESIGN SUMMARY:")
    print(f"  - States: 2 (Pre-Crisis vs Crisis-Prone)")
    print(f"  - Input features: {len(HMM_ZSCORE_FEATURES)} core categories (z-score)")
    print(f"  - Output features: {len(HMM_OUTPUT_FEATURES)} (75% reduction from 12)")
    print(f"  - Asymmetric transitions: Crisis persistence >= {CRISIS_PERSISTENCE_MIN}")


if __name__ == '__main__':
    main()
