"""
Stage 3: Cascade Story Analysis - Case Studies for Publication
===============================================================
Generates compelling humanitarian case studies showing crises
that were missed by AR but caught by the cascade ensemble.

Outputs:
- Top 10 case studies with full context
- Country-level analysis
- Severity breakdown (IPC 3/4/5)
- Narrative templates for publication

Author: Victor Collins Oppon
Date: December 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import json
import sys
import warnings
warnings.filterwarnings('ignore')

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent))
from config import BASE_DIR, RESULTS_DIR, STAGE1_RESULTS_DIR

# Directories
INPUT_DIR = RESULTS_DIR / 'ensemble_improved'
OUTPUT_DIR = INPUT_DIR  # Save to same directory
FIGURES_DIR = RESULTS_DIR.parent / 'FIGURES' / 'cascade_stories'
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("CASCADE STORY ANALYSIS - CASE STUDIES FOR PUBLICATION")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# =============================================================================
# LOAD DATA
# =============================================================================

print("-" * 80)
print("STEP 1: Loading Data")
print("-" * 80)

# Load predictions
predictions_file = INPUT_DIR / 'improved_cascade_predictions.csv'
df = pd.read_csv(predictions_file)
df['date'] = pd.to_datetime(df['date'])
print(f"Loaded predictions: {len(df):,} observations")

# Load key saves
key_saves_file = INPUT_DIR / 'key_saves_for_stories.csv'
key_saves = pd.read_csv(key_saves_file)
key_saves['date'] = pd.to_datetime(key_saves['date'])
print(f"Loaded key saves: {len(key_saves):,} crises caught by ensemble")

# Load summary
with open(INPUT_DIR / 'improved_cascade_summary.json', 'r') as f:
    summary = json.load(f)

print()

# =============================================================================
# STEP 2: ANALYZE KEY SAVES BY SEVERITY
# =============================================================================

print("-" * 80)
print("STEP 2: Analysis by IPC Severity")
print("-" * 80)

if 'ipc_value' in key_saves.columns:
    severity_analysis = key_saves.groupby('ipc_value').agg({
        'geographic_unit': 'count',
        'ar_prob': 'mean',
        'stage2_prob': 'mean'
    }).rename(columns={'geographic_unit': 'count'})

    severity_labels = {
        3: 'IPC 3 - Crisis',
        4: 'IPC 4 - Emergency',
        5: 'IPC 5 - Catastrophe'
    }

    print("CRISES CAUGHT BY ENSEMBLE (that AR missed):")
    print(f"{'IPC Level':<25} {'Count':>10} {'Avg AR Prob':>15} {'Avg S2 Prob':>15}")
    print("-" * 70)

    for ipc, row in severity_analysis.iterrows():
        label = severity_labels.get(int(ipc), f'IPC {int(ipc)}')
        print(f"{label:<25} {int(row['count']):>10} {row['ar_prob']:>15.3f} {row['stage2_prob']:>15.3f}")

    print()

    # Most critical: IPC 4 and 5 (Emergency and Catastrophe)
    critical_saves = key_saves[key_saves['ipc_value'] >= 4]
    print(f"CRITICAL SAVES (IPC 4-5): {len(critical_saves):,} cases")
    if len(critical_saves) > 0:
        print("  These are the most severe food insecurity events that AR missed.")
        print("  Catching these early enables humanitarian response before famine conditions.")

print()

# =============================================================================
# STEP 3: COUNTRY-LEVEL ANALYSIS
# =============================================================================

print("-" * 80)
print("STEP 3: Country-Level Analysis")
print("-" * 80)

country_analysis = key_saves.groupby('country').agg({
    'geographic_unit': 'count',
    'ipc_value': 'mean',
    'ar_prob': 'mean',
    'stage2_prob': 'mean'
}).rename(columns={'geographic_unit': 'count', 'ipc_value': 'avg_ipc'})

country_analysis = country_analysis.sort_values('count', ascending=False)

print("KEY SAVES BY COUNTRY:")
print(f"{'Country':<30} {'Saves':>8} {'Avg IPC':>10} {'Avg AR':>10} {'Avg S2':>10}")
print("-" * 75)

for country, row in country_analysis.iterrows():
    print(f"{country:<30} {int(row['count']):>8} {row['avg_ipc']:>10.2f} {row['ar_prob']:>10.3f} {row['stage2_prob']:>10.3f}")

print()

# =============================================================================
# STEP 4: GENERATE TOP 10 CASE STUDIES
# =============================================================================

print("-" * 80)
print("STEP 4: Top 10 Case Studies")
print("-" * 80)
print()

# Sort by Stage 2 confidence (higher confidence = more compelling)
key_saves_sorted = key_saves.sort_values('stage2_prob', ascending=False)

# Prioritize critical severity cases
if 'ipc_value' in key_saves_sorted.columns:
    key_saves_sorted['priority_score'] = (
        key_saves_sorted['stage2_prob'] * 0.5 +
        (key_saves_sorted['ipc_value'] / 5) * 0.3 +
        (1 - key_saves_sorted['ar_prob']) * 0.2  # Lower AR prob = more surprising
    )
    key_saves_sorted = key_saves_sorted.sort_values('priority_score', ascending=False)

case_studies = []

print("TOP 10 HUMANITARIAN CASES (Crises Caught by Ensemble):")
print("=" * 80)

for idx, (_, row) in enumerate(key_saves_sorted.head(10).iterrows(), 1):
    case = {
        'rank': idx,
        'location': row['geographic_unit'],
        'country': row['country'],
        'date': row['date'].strftime('%Y-%m'),
        'ipc_severity': int(row.get('ipc_value', 3)),
        'ar_probability': float(row['ar_prob']),
        'ensemble_probability': float(row['stage2_prob']),
        'ar_prediction': 'No Crisis',
        'ensemble_prediction': 'CRISIS',
        'actual_outcome': 'CRISIS OCCURRED'
    }

    severity_text = {3: 'Crisis', 4: 'Emergency', 5: 'Catastrophe'}.get(
        case['ipc_severity'], f"IPC {case['ipc_severity']}"
    )

    print(f"\nCASE #{idx}: {case['country']} - {case['date']}")
    print(f"Location: {case['location'][:70]}...")
    print(f"Severity: IPC {case['ipc_severity']} ({severity_text})")
    print(f"AR Model: Predicted NO CRISIS (prob: {case['ar_probability']:.1%})")
    print(f"Ensemble: Predicted CRISIS (prob: {case['ensemble_probability']:.1%})")
    print(f"Outcome: CRISIS OCCURRED - Ensemble was CORRECT")
    print("-" * 60)

    # Generate narrative
    if case['ar_probability'] < 0.3:
        ar_confidence = "highly confident there was no crisis"
    elif case['ar_probability'] < 0.5:
        ar_confidence = "somewhat confident there was no crisis"
    else:
        ar_confidence = "uncertain, leaning toward no crisis"

    if case['ensemble_probability'] > 0.7:
        s2_confidence = "strong indicators from news signals"
    elif case['ensemble_probability'] > 0.5:
        s2_confidence = "moderate indicators from news signals"
    else:
        s2_confidence = "subtle but detectable news signals"

    narrative = (
        f"In {case['date']}, the AR baseline model was {ar_confidence} "
        f"(probability: {case['ar_probability']:.1%}) for {case['location'][:50]}. "
        f"However, the Stage 2 model detected {s2_confidence} "
        f"(probability: {case['ensemble_probability']:.1%}) that indicated an emerging crisis. "
        f"The cascade ensemble correctly overrode AR's prediction, and an IPC {case['ipc_severity']} "
        f"({severity_text}) food insecurity event was confirmed. "
        f"This demonstrates how combining spatial-temporal patterns with news-based signals "
        f"can catch crises that purely historical data misses."
    )

    case['narrative'] = narrative
    case_studies.append(case)

    print(f"NARRATIVE: {narrative[:200]}...")

print()

# =============================================================================
# STEP 5: TEMPORAL ANALYSIS
# =============================================================================

print("-" * 80)
print("STEP 5: Temporal Analysis")
print("-" * 80)

key_saves['year_month'] = key_saves['date'].dt.to_period('M')
temporal_analysis = key_saves.groupby('year_month').size()

print("KEY SAVES BY TIME PERIOD:")
print(f"{'Period':<15} {'Saves':>10}")
print("-" * 30)

for period, count in temporal_analysis.items():
    print(f"{str(period):<15} {count:>10}")

print()

# Identify any seasonal patterns
key_saves['month'] = key_saves['date'].dt.month
seasonal_analysis = key_saves.groupby('month').size()

print("SEASONAL PATTERN (by month):")
month_names = {1: 'Jan', 2: 'Feb', 3: 'Mar', 4: 'Apr', 5: 'May', 6: 'Jun',
               7: 'Jul', 8: 'Aug', 9: 'Sep', 10: 'Oct', 11: 'Nov', 12: 'Dec'}

for month, count in seasonal_analysis.items():
    print(f"  {month_names[month]}: {count}")

print()

# =============================================================================
# STEP 6: WHY AR FAILED ANALYSIS
# =============================================================================

print("-" * 80)
print("STEP 6: Why AR Failed - Pattern Analysis")
print("-" * 80)

# Analyze characteristics of AR failures
print("CHARACTERISTICS OF AR FAILURES CAUGHT BY ENSEMBLE:")
print()

# 1. AR probability distribution
ar_prob_bins = pd.cut(key_saves['ar_prob'], bins=[0, 0.2, 0.4, 0.5, 0.63, 1.0])
ar_prob_dist = key_saves.groupby(ar_prob_bins, observed=True).size()

print("1. AR Probability Distribution (for missed crises):")
for bin_range, count in ar_prob_dist.items():
    print(f"   {bin_range}: {count} cases")

print()

# 2. Stage 2 confidence distribution
s2_prob_bins = pd.cut(key_saves['stage2_prob'], bins=[0, 0.3, 0.5, 0.7, 0.9, 1.0])
s2_prob_dist = key_saves.groupby(s2_prob_bins, observed=True).size()

print("2. Stage 2 Confidence Distribution:")
for bin_range, count in s2_prob_dist.items():
    print(f"   {bin_range}: {count} cases")

print()

# 3. High-confidence catches (S2 > 0.7, AR < 0.4)
high_confidence_catches = key_saves[
    (key_saves['stage2_prob'] > 0.7) & (key_saves['ar_prob'] < 0.4)
]
print(f"3. High-Confidence Catches (S2 > 0.7 AND AR < 0.4): {len(high_confidence_catches)} cases")
print("   These represent cases where AR was confidently wrong but news signals were strong.")

print()

# =============================================================================
# STEP 7: HUMANITARIAN IMPACT NARRATIVE
# =============================================================================

print("-" * 80)
print("STEP 7: Humanitarian Impact Summary")
print("-" * 80)

total_saves = len(key_saves)
critical_saves_count = len(key_saves[key_saves['ipc_value'] >= 4]) if 'ipc_value' in key_saves.columns else 0
countries_benefited = key_saves['country'].nunique()

print()
print("HUMANITARIAN IMPACT NARRATIVE FOR PUBLICATION:")
print("=" * 60)
print()

impact_narrative = f"""
The improved cascade ensemble successfully identified {total_saves:,} food insecurity
events that the AR baseline model missed. Among these, {critical_saves_count:,} were
IPC 4 (Emergency) or IPC 5 (Catastrophe) severity - representing the most critical
humanitarian situations.

These improvements span {countries_benefited} countries, with the largest gains in:
"""

for i, (country, row) in enumerate(country_analysis.head(5).iterrows()):
    impact_narrative += f"\n- {country}: {int(row['count'])} additional crises detected"

impact_narrative += f"""

The cascade logic - trusting AR when it predicts crisis, but checking Stage 2
when AR predicts no crisis - enables the system to catch emerging crises that
historical spatial-temporal patterns alone would miss.

For humanitarian agencies, this means:
1. EARLIER WARNING: News-based signals can detect crisis buildup before it
   manifests in neighboring districts (the basis of AR's spatial component)
2. BETTER COVERAGE: Isolated crises without regional spread are now detectable
3. REDUCED MISSED EVENTS: {total_saves:,} fewer communities left without early warning

Each additional detection represents lives that can be saved through timely
humanitarian intervention - pre-positioning food supplies, activating response
mechanisms, and advocating for resources before crisis peaks.
"""

print(impact_narrative)

# =============================================================================
# STEP 8: SAVE RESULTS
# =============================================================================

print()
print("-" * 80)
print("STEP 8: Saving Results")
print("-" * 80)

# 1. Case studies JSON
case_studies_output = {
    'generated': datetime.now().isoformat(),
    'total_key_saves': total_saves,
    'critical_saves_ipc4_5': critical_saves_count,
    'countries_benefited': countries_benefited,
    'case_studies': case_studies,
    'country_analysis': country_analysis.to_dict('index'),
    'severity_analysis': severity_analysis.to_dict('index') if 'ipc_value' in key_saves.columns else {},
    'impact_narrative': impact_narrative
}

with open(OUTPUT_DIR / 'cascade_story_analysis.json', 'w') as f:
    json.dump(case_studies_output, f, indent=2, default=str)
print(f"[OK] Case studies: {OUTPUT_DIR / 'cascade_story_analysis.json'}")

# 2. Detailed key saves with narratives
key_saves_detailed = key_saves.copy()
key_saves_detailed['severity_label'] = key_saves_detailed['ipc_value'].map(
    {3: 'Crisis', 4: 'Emergency', 5: 'Catastrophe'}
)
key_saves_detailed.to_csv(OUTPUT_DIR / 'key_saves_detailed.csv', index=False)
print(f"[OK] Detailed saves: {OUTPUT_DIR / 'key_saves_detailed.csv'}")

# 3. Country summary
country_analysis.to_csv(OUTPUT_DIR / 'country_analysis.csv')
print(f"[OK] Country analysis: {OUTPUT_DIR / 'country_analysis.csv'}")

# 4. Publication-ready summary table
pub_summary = pd.DataFrame([
    {'Metric': 'Total Crises Caught', 'Value': total_saves},
    {'Metric': 'Critical Severity (IPC 4-5)', 'Value': critical_saves_count},
    {'Metric': 'Countries Benefited', 'Value': countries_benefited},
    {'Metric': 'Avg AR Probability (missed)', 'Value': f"{key_saves['ar_prob'].mean():.1%}"},
    {'Metric': 'Avg S2 Probability (caught)', 'Value': f"{key_saves['stage2_prob'].mean():.1%}"},
])
pub_summary.to_csv(OUTPUT_DIR / 'publication_summary_table.csv', index=False)
print(f"[OK] Publication summary: {OUTPUT_DIR / 'publication_summary_table.csv'}")

print()
print("=" * 80)
print("STORY ANALYSIS COMPLETE")
print("=" * 80)
print()
print(f"Key outputs:")
print(f"  - {total_saves:,} case studies generated")
print(f"  - Top 10 detailed narratives ready for publication")
print(f"  - Country and severity breakdowns available")
print()
print(f"Output directory: {OUTPUT_DIR}")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
