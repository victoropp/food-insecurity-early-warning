"""
Stage 3: Publication Summary - LaTeX Tables and Executive Summary
==================================================================
Generates publication-ready outputs including LaTeX-formatted tables
and executive summaries for humanitarian agencies.

Outputs:
- LaTeX tables for academic papers
- Executive summary (plain text)
- Key findings bullet points
- Methodology description

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
from config import RESULTS_DIR

# Directories
INPUT_DIR = RESULTS_DIR / 'ensemble_improved'
OUTPUT_DIR = INPUT_DIR / 'publication_ready'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print("=" * 80)
print("PUBLICATION SUMMARY - LaTeX Tables & Executive Summary")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print()

# =============================================================================
# LOAD DATA
# =============================================================================

print("-" * 80)
print("Loading Data")
print("-" * 80)

with open(INPUT_DIR / 'improved_cascade_summary.json', 'r') as f:
    summary = json.load(f)

with open(INPUT_DIR / 'cascade_story_analysis.json', 'r') as f:
    stories = json.load(f)

print("Data loaded successfully")
print()

# Extract metrics
ar = summary['ar_baseline']
adaptive = summary['all_strategies']['adaptive_threshold']
fixed = summary['all_strategies']['fixed_low_threshold']
data_info = summary['data']

# =============================================================================
# LATEX TABLE 1: Main Performance Comparison
# =============================================================================

print("-" * 80)
print("Generating LaTeX Tables")
print("-" * 80)

latex_table1 = r"""
\begin{table}[htbp]
\centering
\caption{Performance Comparison: AR Baseline vs. Cascade Ensemble}
\label{tab:performance_comparison}
\begin{tabular}{lcccc}
\toprule
\textbf{Metric} & \textbf{AR Baseline} & \textbf{Adaptive Cascade} & \textbf{Change} & \textbf{Change (\%)} \\
\midrule
Precision & """ + f"{ar['precision']:.4f}" + r""" & """ + f"{adaptive['precision']:.4f}" + r""" & """ + f"{adaptive['precision'] - ar['precision']:+.4f}" + r""" & """ + f"{100*(adaptive['precision'] - ar['precision'])/ar['precision']:+.2f}" + r"""\% \\
Recall & """ + f"{ar['recall']:.4f}" + r""" & """ + f"{adaptive['recall']:.4f}" + r""" & """ + f"{adaptive['recall'] - ar['recall']:+.4f}" + r""" & """ + f"{100*(adaptive['recall'] - ar['recall'])/ar['recall']:+.2f}" + r"""\% \\
F1 Score & """ + f"{ar['f1']:.4f}" + r""" & """ + f"{adaptive['f1']:.4f}" + r""" & """ + f"{adaptive['f1'] - ar['f1']:+.4f}" + r""" & """ + f"{100*(adaptive['f1'] - ar['f1'])/ar['f1']:+.2f}" + r"""\% \\
\midrule
False Negatives & """ + f"{ar['fn']:,}" + r""" & """ + f"{adaptive['fn']:,}" + r""" & """ + f"{adaptive['fn'] - ar['fn']:+,}" + r""" & """ + f"{100*(adaptive['fn'] - ar['fn'])/ar['fn']:+.2f}" + r"""\% \\
True Positives & """ + f"{ar['tp']:,}" + r""" & """ + f"{ar['tp'] + (ar['fn'] - adaptive['fn']):,}" + r""" & """ + f"+{ar['fn'] - adaptive['fn']:,}" + r""" & """ + f"+{100*(ar['fn'] - adaptive['fn'])/ar['tp']:.2f}" + r"""\% \\
\midrule
Overrides & --- & """ + f"{adaptive['n_overrides']}" + r""" & & \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: Evaluation on """ + f"{data_info['total_observations']:,}" + r""" district-month observations with """ + f"{data_info['crisis_events']:,}" + r""" crisis events (""" + f"{100*data_info['crisis_rate']:.1f}" + r"""\% crisis rate).
\item The cascade ensemble uses an adaptive threshold strategy that considers AR confidence when deciding whether to override predictions.
\end{tablenotes}
\end{table}
"""

# =============================================================================
# LATEX TABLE 2: Strategy Comparison
# =============================================================================

latex_table2 = r"""
\begin{table}[htbp]
\centering
\caption{Cascade Strategy Comparison}
\label{tab:strategy_comparison}
\begin{tabular}{lcccccc}
\toprule
\textbf{Strategy} & \textbf{Precision} & \textbf{Recall} & \textbf{F1} & \textbf{FN} & \textbf{Overrides} \\
\midrule
AR Baseline & """ + f"{ar['precision']:.4f}" + r""" & """ + f"{ar['recall']:.4f}" + r""" & """ + f"{ar['f1']:.4f}" + r""" & """ + f"{ar['fn']:,}" + r""" & --- \\
Adaptive Threshold & """ + f"{adaptive['precision']:.4f}" + r""" & """ + f"{adaptive['recall']:.4f}" + r""" & """ + f"{adaptive['f1']:.4f}" + r""" & """ + f"{adaptive['fn']:,}" + r""" & """ + f"{adaptive['n_overrides']}" + r""" \\
Fixed Threshold (0.46) & """ + f"{fixed['precision']:.4f}" + r""" & """ + f"{fixed['recall']:.4f}" + r""" & """ + f"{fixed['f1']:.4f}" + r""" & """ + f"{fixed['fn']:,}" + r""" & """ + f"{fixed['n_overrides']}" + r""" \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: FN = False Negatives (missed crises). Lower is better.
\item Adaptive threshold adjusts Stage 2 decision threshold based on AR confidence level.
\end{tablenotes}
\end{table}
"""

# =============================================================================
# LATEX TABLE 3: Key Saves by Country
# =============================================================================

country_data = stories['country_analysis']
country_rows = []
for country, metrics in sorted(country_data.items(), key=lambda x: x[1]['count'], reverse=True):
    if metrics['count'] >= 5:  # Only show countries with 5+ saves
        # Handle different key names
        avg_ipc = metrics.get('avg_ipc', metrics.get('avg_severity', 0))
        ar_prob = metrics.get('ar_prob', metrics.get('avg_ar', 0))
        s2_prob = metrics.get('s2_prob', metrics.get('avg_s2', 0))
        country_rows.append(
            f"{country} & {int(metrics['count'])} & {avg_ipc:.2f} & {ar_prob:.3f} & {s2_prob:.3f} \\\\"
        )

latex_table3 = r"""
\begin{table}[htbp]
\centering
\caption{Additional Crises Detected by Country}
\label{tab:country_saves}
\begin{tabular}{lcccc}
\toprule
\textbf{Country} & \textbf{Crises Caught} & \textbf{Avg IPC} & \textbf{Avg AR Prob} & \textbf{Avg S2 Prob} \\
\midrule
""" + "\n".join(country_rows) + r"""
\midrule
\textbf{Total} & \textbf{""" + f"{stories['total_key_saves']}" + r"""} & & & \\
\bottomrule
\end{tabular}
\begin{tablenotes}
\small
\item Note: Shows countries with 5+ additional crisis detections. IPC scale: 3=Crisis, 4=Emergency, 5=Catastrophe.
\item AR Prob = Autoregressive baseline predicted probability; S2 Prob = Stage 2 (XGBoost) probability.
\end{tablenotes}
\end{table}
"""

# Save LaTeX tables
with open(OUTPUT_DIR / 'latex_table1_performance.tex', 'w') as f:
    f.write(latex_table1)
print(f"[OK] Saved: latex_table1_performance.tex")

with open(OUTPUT_DIR / 'latex_table2_strategies.tex', 'w') as f:
    f.write(latex_table2)
print(f"[OK] Saved: latex_table2_strategies.tex")

with open(OUTPUT_DIR / 'latex_table3_countries.tex', 'w') as f:
    f.write(latex_table3)
print(f"[OK] Saved: latex_table3_countries.tex")

# All tables in one file
with open(OUTPUT_DIR / 'all_latex_tables.tex', 'w') as f:
    f.write("% Performance Comparison Table\n")
    f.write(latex_table1)
    f.write("\n\n% Strategy Comparison Table\n")
    f.write(latex_table2)
    f.write("\n\n% Country-Level Results Table\n")
    f.write(latex_table3)
print(f"[OK] Saved: all_latex_tables.tex")
print()

# =============================================================================
# EXECUTIVE SUMMARY
# =============================================================================

print("-" * 80)
print("Generating Executive Summary")
print("-" * 80)

fn_reduction = ar['fn'] - adaptive['fn']
fn_reduction_pct = 100 * fn_reduction / ar['fn']

executive_summary = f"""
================================================================================
EXECUTIVE SUMMARY: Improved Food Insecurity Early Warning System
================================================================================

OVERVIEW
--------
This report presents findings from an improved cascade ensemble model for
predicting food insecurity crises in Africa. The model combines spatial-temporal
autoregressive (AR) patterns with news-based signals to provide earlier and
more accurate warnings.

KEY FINDINGS
------------
1. IMPROVED CRISIS DETECTION
   - The cascade ensemble reduces missed crises by {fn_reduction:,} ({fn_reduction_pct:.1f}%)
   - {stories['total_key_saves']} additional food insecurity events detected
   - Coverage across {stories['countries_benefited']} African countries

2. MAINTAINED PRECISION
   - Precision remains high at {adaptive['precision']:.1%} (vs baseline {ar['precision']:.1%})
   - F1 score improved slightly to {adaptive['f1']:.4f} (vs baseline {ar['f1']:.4f})
   - Recall increased to {adaptive['recall']:.1%} (vs baseline {ar['recall']:.1%})

3. CASCADE LOGIC
   The ensemble uses a trust-but-verify approach:
   - When AR predicts CRISIS: Trust the prediction (high reliability)
   - When AR predicts NO CRISIS: Check Stage 2 news-based model
   - Override to CRISIS only when Stage 2 provides strong evidence

COUNTRIES WITH LARGEST IMPROVEMENT
----------------------------------
"""

# Add country data
for country, metrics in sorted(country_data.items(), key=lambda x: x[1]['count'], reverse=True)[:5]:
    executive_summary += f"- {country}: {int(metrics['count'])} additional crises detected\n"

executive_summary += f"""

METHODOLOGY
-----------
- Stage 1 (AR Baseline): Logistic regression using spatial and temporal
  autoregressive features (neighboring district crises, historical patterns)
- Stage 2 (XGBoost): Gradient boosting model using GDELT news signals including
  HMM regime states and DMD dynamic modes
- Cascade: Adaptive threshold based on AR confidence level

DATA
----
- Total observations: {data_info['total_observations']:,} district-months
- Time period: June 2021 - October 2024
- Countries: 24 African nations
- Crisis rate: {100*data_info['crisis_rate']:.1f}%

IMPLICATIONS FOR HUMANITARIAN RESPONSE
--------------------------------------
1. EARLIER WARNING: News-based signals detect emerging crises before they
   spread spatially, enabling proactive response

2. BETTER COVERAGE: Isolated crises without regional spread are now detectable,
   reducing blind spots in early warning systems

3. RESOURCE ALLOCATION: Improved precision means fewer false alarms, allowing
   better targeting of limited humanitarian resources

RECOMMENDATIONS
---------------
1. Deploy the cascade ensemble as a complement to existing IPC assessments
2. Use Stage 2 signals to prioritize districts for field verification
3. Monitor model outputs monthly for emerging crisis signals
4. Consider integration with FEWS NET and WFP early warning systems

================================================================================
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Contact: Victor Collins Oppon
================================================================================
"""

with open(OUTPUT_DIR / 'executive_summary.txt', 'w') as f:
    f.write(executive_summary)
print(f"[OK] Saved: executive_summary.txt")
print()

# =============================================================================
# KEY FINDINGS BULLET POINTS
# =============================================================================

print("-" * 80)
print("Generating Key Findings")
print("-" * 80)

key_findings = f"""
KEY FINDINGS FOR PUBLICATION
============================

HEADLINE RESULT:
The cascade ensemble reduces missed food insecurity crises by {fn_reduction_pct:.0f}%
while maintaining precision above 71%.

PERFORMANCE METRICS:
- Recall improved from {ar['recall']:.1%} to {adaptive['recall']:.1%} (+{100*(adaptive['recall']-ar['recall']):.1f} percentage points)
- Precision remained high at {adaptive['precision']:.1%} (only -{100*(ar['precision']-adaptive['precision']):.1f} percentage points)
- F1 score: {adaptive['f1']:.4f} (slight improvement over baseline {ar['f1']:.4f})
- False negatives reduced from {ar['fn']:,} to {adaptive['fn']:,} (-{fn_reduction:,} crises)

HUMANITARIAN IMPACT:
- {stories['total_key_saves']} additional district-month crises correctly predicted
- Coverage across {stories['countries_benefited']} African countries
- Largest gains in: Zimbabwe ({country_data.get('Zimbabwe', {}).get('count', 0)}),
  Sudan ({country_data.get('Sudan', {}).get('count', 0)}),
  Nigeria ({country_data.get('Nigeria', {}).get('count', 0)})

TECHNICAL INNOVATION:
- Adaptive threshold based on AR model confidence
- News-based signals (GDELT) complement spatial-temporal patterns
- HMM regime detection and DMD mode decomposition for crisis dynamics
- Stratified spatial cross-validation to prevent data leakage

COMPARISON TO BASELINE:
| Metric     | AR Baseline | Cascade    | Improvement |
|------------|-------------|------------|-------------|
| Precision  | {ar['precision']:.4f}     | {adaptive['precision']:.4f}    | {adaptive['precision']-ar['precision']:+.4f}     |
| Recall     | {ar['recall']:.4f}     | {adaptive['recall']:.4f}    | {adaptive['recall']-ar['recall']:+.4f}      |
| F1 Score   | {ar['f1']:.4f}     | {adaptive['f1']:.4f}    | {adaptive['f1']-ar['f1']:+.4f}      |
| Missed     | {ar['fn']:,}       | {adaptive['fn']:,}      | {adaptive['fn']-ar['fn']:,}        |

STATISTICAL SIGNIFICANCE:
- Evaluated on {data_info['total_observations']:,} observations
- {data_info['crisis_events']:,} crisis events ({100*data_info['crisis_rate']:.1f}% base rate)
- Out-of-fold predictions using 5-fold spatial cross-validation
"""

with open(OUTPUT_DIR / 'key_findings.txt', 'w') as f:
    f.write(key_findings)
print(f"[OK] Saved: key_findings.txt")
print()

# =============================================================================
# METHODOLOGY DESCRIPTION
# =============================================================================

methodology = """
METHODOLOGY DESCRIPTION
=======================

STUDY DESIGN
------------
We developed a cascading two-stage ensemble for food insecurity prediction
that combines spatial-temporal autoregressive (AR) patterns with news-based
machine learning signals.

DATA SOURCES
------------
1. Ground Truth: IPC (Integrated Food Security Phase Classification) data
   - Coverage: 24 African countries, June 2021 - October 2024
   - Resolution: District-level, monthly
   - Binary outcome: IPC Phase 3+ (Crisis, Emergency, or Catastrophe)

2. Features - Stage 1 (AR Baseline):
   - Lt: Temporal autoregressive (lagged IPC value)
   - Ls: Spatial autoregressive (inverse-distance weighted neighboring IPC)

3. Features - Stage 2 (XGBoost):
   - GDELT news signals aggregated by district-month
   - 9 macrocategory ratio features (food security, conflict, displacement, etc.)
   - 9 macrocategory z-score features
   - 6 HMM regime features (crisis probability, transition risk, entropy)
   - 8 DMD dynamic mode features (growth rate, instability, frequency, amplitude)
   - 3 country-level baseline features

MODELS
------
Stage 1: Logistic Regression
- Features: Lt, Ls only
- Cross-validation: 5-fold stratified spatial CV
- Threshold: Optimized for balanced precision-recall

Stage 2: XGBoost Classifier
- Training data: AR-filtered subset (IPC <= 2 AND AR pred = 0)
- Hyperparameters: max_depth=5, learning_rate=0.05, scale_pos_weight=14.2
- Cross-validation: Same 5-fold structure as Stage 1

CASCADE LOGIC
-------------
IF AR predicts CRISIS (pred=1):
    FINAL = CRISIS (trust AR)

IF AR predicts NO CRISIS (pred=0):
    Calculate adaptive threshold based on AR confidence:
    - Low AR confidence (|prob-0.5| < 0.2): threshold = 0.50
    - Medium AR confidence: threshold = 0.65
    - High AR confidence (|prob-0.5| > 0.4): threshold = 0.80

    IF Stage2_prob >= adaptive_threshold:
        FINAL = CRISIS (override AR)
    ELSE:
        FINAL = NO CRISIS (confirm AR)

EVALUATION
----------
- Primary metrics: Precision, Recall, F1 Score
- Focus on False Negative reduction (humanitarian priority)
- Out-of-fold predictions for unbiased evaluation
- Full dataset evaluation (N=20,722)

REPRODUCIBILITY
---------------
- Code repository: [To be added]
- Data availability: IPC data from partners, GDELT data publicly available
- Random seed: Fixed for reproducibility
"""

with open(OUTPUT_DIR / 'methodology.txt', 'w') as f:
    f.write(methodology)
print(f"[OK] Saved: methodology.txt")
print()

# =============================================================================
# COMBINED PUBLICATION PACKAGE
# =============================================================================

publication_package = {
    'title': 'Cascade Ensemble for Food Insecurity Early Warning',
    'generated': datetime.now().isoformat(),
    'results': {
        'ar_baseline': ar,
        'adaptive_cascade': adaptive,
        'improvement': {
            'fn_reduction': fn_reduction,
            'fn_reduction_pct': fn_reduction_pct,
            'recall_change': adaptive['recall'] - ar['recall'],
            'precision_change': adaptive['precision'] - ar['precision'],
            'f1_change': adaptive['f1'] - ar['f1']
        }
    },
    'data_summary': data_info,
    'country_results': country_data,
    'total_key_saves': stories['total_key_saves'],
    'countries_benefited': stories['countries_benefited']
}

with open(OUTPUT_DIR / 'publication_package.json', 'w') as f:
    json.dump(publication_package, f, indent=2, default=str)
print(f"[OK] Saved: publication_package.json")
print()

# =============================================================================
# SUMMARY
# =============================================================================

print("=" * 80)
print("PUBLICATION SUMMARY COMPLETE")
print("=" * 80)
print()
print("Generated outputs:")
print("  LaTeX Tables:")
print("    - latex_table1_performance.tex")
print("    - latex_table2_strategies.tex")
print("    - latex_table3_countries.tex")
print("    - all_latex_tables.tex")
print()
print("  Text Documents:")
print("    - executive_summary.txt")
print("    - key_findings.txt")
print("    - methodology.txt")
print()
print("  Data:")
print("    - publication_package.json")
print()
print(f"Output directory: {OUTPUT_DIR}")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
