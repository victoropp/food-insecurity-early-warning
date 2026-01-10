"""
Analysis: Feature Importance Comparison Across Models
======================================================
Compare and rank features by consensus importance across all models.

Date: December 21, 2025
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import sys
import warnings
warnings.filterwarnings('ignore')

sys.path.append(str(Path(__file__).parent.parent))
from config import (BASE_DIR, RESULTS_DIR)

OUTPUT_DIR = RESULTS_DIR / 'analysis' / 'feature_comparison'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("ANALYSIS: FEATURE IMPORTANCE COMPARISON")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Load consensus features
consensus_file = RESULTS_DIR / 'explainability' / 'integrated' / 'consensus_features.csv'

if not consensus_file.exists():
    print(f"ERROR: Consensus features not found: {consensus_file}")
    sys.exit(1)

consensus_df = pd.read_csv(consensus_file)

# Rank features
consensus_df['rank'] = range(1, len(consensus_df) + 1)

# Save ranking
ranking_file = OUTPUT_DIR / 'feature_ranking.csv'
consensus_df.to_csv(ranking_file, index=False)
print(f"Saved feature ranking: {ranking_file}")

print("\nTop 20 features by consensus:")
print(consensus_df[['rank', 'feature', 'mean_importance', 'n_sources']].head(20).to_string(index=False))

print("\n" + "=" * 80)
print("FEATURE IMPORTANCE COMPARISON COMPLETE")
print("=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
