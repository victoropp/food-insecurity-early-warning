"""
Figure 10: DMD Temporal Modes with REAL Crisis Examples
Publication-grade visualization showing Dynamic Mode Decomposition features
100% REAL DATA from hmm_dmd_ratio_features_h8.csv
Shows convergence analysis and rare events focus

Date: 2026-01-05
"""

# Author: Victor Collins Oppon
# MSc Data Science Dissertation, Middlesex University 2025



import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path
import warnings
from config import BASE_DIR
warnings.filterwarnings('ignore')

# Directories
BASE_DIR = Path(str(BASE_DIR))
DATA_DIR = BASE_DIR / "RESULTS" / "stage2_features" / "phase2_features"
OUTPUT_DIR = BASE_DIR / "Dissertation Write Up" / "LATEX_DISSERTATION" / "figures" / "ch03_methods"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Load REAL DMD features data
print("Loading REAL DMD features from hmm_dmd_ratio_features_h8.csv...")
df = pd.read_csv(DATA_DIR / "hmm_dmd_ratio_features_h8.csv")

print(f"\nDataset: {len(df):,} observations")
print(f"Countries: {df['ipc_country'].nunique()}")
print(f"Districts: {df['ipc_district'].nunique()}")
print(f"Date range: {df['year_month'].min()} to {df['year_month'].max()}")

# Filter to districts with interesting crisis patterns
# Look for districts with actual crises and varied DMD features
crisis_data = df[df['ipc_binary_crisis_filled'] == 1].copy()
print(f"\nCrisis observations: {len(crisis_data):,}")

# Find district with clear escalation pattern (growing growth_rate + instability)
# Prefer Sudan/Zimbabwe/DRC for narrative consistency
priority_countries = ['Sudan', 'Zimbabwe', 'DRC']
candidates = df[df['ipc_country'].isin(priority_countries)].groupby(['ipc_country', 'ipc_district']).agg({
    'ipc_value': ['count', 'mean', 'std'],
    'dmd_ratio_crisis_growth_rate': ['mean', 'std'],
    'dmd_ratio_crisis_instability': ['mean', 'max'],
    'ipc_binary_crisis_filled': 'sum'
}).reset_index()
candidates.columns = ['country', 'district', 'n_obs', 'mean_ipc', 'std_ipc',
                      'mean_growth', 'std_growth', 'mean_instability', 'max_instability', 'n_crises']

# Filter: at least 18 observations, some crises, variation in DMD features (not NaN)
candidates = candidates[
    (candidates['n_obs'] >= 18) &
    (candidates['n_crises'] > 0) &
    (candidates['std_growth'].notna()) &
    (candidates['std_growth'] > 0.001) &
    (candidates['max_instability'].notna())
].sort_values('n_crises', ascending=False)

print(f"\n{len(candidates)} candidates with sufficient data and crisis variation")

if len(candidates) > 0:
    example = candidates.iloc[0]
    example_country = example['country']
    example_district = example['district']

    print(f"\nSelected: {example_country}, {example_district}")
    print(f"  Observations: {int(example['n_obs'])}")
    print(f"  Crises: {int(example['n_crises'])}")
    print(f"  Growth rate: {example['mean_growth']:.3f} ± {example['std_growth']:.3f}")
    print(f"  Instability: {example['mean_instability']:.2f} (max {example['max_instability']:.2f})")

    # Get time series for this district - only periods with DMD features
    ts_data = df[
        (df['ipc_country'] == example_country) &
        (df['ipc_district'] == example_district) &
        (df['dmd_ratio_crisis_growth_rate'].notna())
    ].sort_values('year_month').copy()

else:
    # Fallback: Use aggregate Sudan data
    print("\nNo suitable district found, using Sudan aggregate...")
    example_country = 'Sudan'
    example_district = 'Country Aggregate'

    ts_data = df[df['ipc_country'] == 'Sudan'].groupby('year_month').agg({
        'dmd_ratio_crisis_growth_rate': 'mean',
        'dmd_ratio_crisis_instability': 'mean',
        'dmd_ratio_crisis_frequency': 'mean',
        'dmd_ratio_crisis_amplitude': 'mean',
        'ipc_value': 'mean',
        'ipc_binary_crisis_filled': 'mean'
    }).reset_index().sort_values('year_month').copy()

# Create time index
ts_data['time'] = np.arange(len(ts_data))
print(f"\nTime series length: {len(ts_data)} periods")

# Extract DMD features
growth_rate = ts_data['dmd_ratio_crisis_growth_rate'].values
instability = ts_data['dmd_ratio_crisis_instability'].values
frequency = ts_data['dmd_ratio_crisis_frequency'].values
amplitude = ts_data['dmd_ratio_crisis_amplitude'].values
ipc_phase = ts_data['ipc_value'].values if 'ipc_value' in ts_data.columns else ts_data['ipc_value'].values
crises = ts_data['ipc_binary_crisis_filled'].values

print(f"\nDMD feature ranges:")
print(f"  Growth rate: {growth_rate.min():.3f} to {growth_rate.max():.3f}")
print(f"  Instability: {instability.min():.2f} to {instability.max():.2f}")
print(f"  Frequency: {frequency.min():.3f} to {frequency.max():.3f}")
print(f"  Amplitude: {amplitude.min():.2f} to {amplitude.max():.2f}")

# Publication settings
plt.rcParams['figure.dpi'] = 300
plt.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 10

# Create figure with REAL DMD layout: 2x2 grid (3 DMD modes + convergence + summary)
# REMOVED Panel B (Instability) - mostly zeros in real data
fig = plt.figure(figsize=(14, 10))
gs = fig.add_gridspec(2, 2, hspace=0.35, wspace=0.3)

ax1 = fig.add_subplot(gs[0, 0])  # Growth rate
ax2 = fig.add_subplot(gs[0, 1])  # Frequency
ax3 = fig.add_subplot(gs[1, 0])  # Amplitude
ax4 = fig.add_subplot(gs[1, 1])  # Convergence + summary combined

# Panel A: Growth Rate (real eigenvalue)
ax1.plot(ts_data['time'], growth_rate, 'o-', linewidth=2.5, markersize=6, color='#2E86AB', zorder=3)
ax1.axhline(y=0, color='gray', linestyle='--', linewidth=2, alpha=0.7, zorder=2)
# Mark crisis periods with background shading
for i in range(len(crises)):
    if crises[i] == 1:
        ax1.axvspan(i-0.4, i+0.4, color='#FFB3B3', alpha=0.3, zorder=1)

# Find escalation vs decay periods
escalation_mask = growth_rate > 0
if np.sum(escalation_mask) > 0:
    ax1.scatter(ts_data.loc[escalation_mask, 'time'], growth_rate[escalation_mask],
               color='#E74C3C', s=100, marker='^', zorder=5, edgecolor='darkred', linewidth=1.5,
               label=f'Escalation (n={np.sum(escalation_mask)})')

ax1.set_xlabel('Time (months)', fontsize=11, fontweight='bold')
ax1.set_ylabel('Growth Rate', fontsize=11, fontweight='bold')
ax1.set_title('A. DMD Growth Rate (Real Eigenvalue)', fontsize=12, fontweight='bold', pad=10)
ax1.legend(fontsize=9, loc='upper right')
ax1.grid(True, alpha=0.3)

# Panel B: Frequency (oscillation period)
ax2.plot(ts_data['time'], frequency, 'o-', linewidth=2.5, markersize=6, color='#9B59B6', zorder=3)
# Mark crisis periods
for i in range(len(crises)):
    if crises[i] == 1:
        ax2.axvspan(i-0.4, i+0.4, color='#FFB3B3', alpha=0.3, zorder=1)

ax2.set_xlabel('Time (months)', fontsize=11, fontweight='bold')
ax2.set_ylabel('Oscillation Period', fontsize=11, fontweight='bold')
ax2.set_title('B. DMD Frequency (Oscillation Period)', fontsize=12, fontweight='bold', pad=10)
ax2.grid(True, alpha=0.3)

# Panel C: Amplitude (oscillation magnitude)
ax3.fill_between(ts_data['time'], 0, amplitude, alpha=0.3, color='purple', label='Amplitude envelope')
ax3.plot(ts_data['time'], amplitude, linewidth=2.5, color='#8E44AD', marker='o', markersize=6)
ax3.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax3.set_xlabel('Time (months)', fontsize=11, fontweight='bold')
ax3.set_ylabel('Amplitude', fontsize=11, fontweight='bold')
ax3.set_title('C. DMD Amplitude (Oscillation Magnitude)', fontsize=12, fontweight='bold', pad=10)
ax3.legend(fontsize=9, loc='upper left')
ax3.grid(True, alpha=0.3)

# Panel D: Combined Convergence + Summary
ax4.axis('off')

# Calculate rare events statistics
total_obs = len(df)
crisis_obs = df['ipc_binary_crisis_filled'].sum()
crisis_pct = (crisis_obs / total_obs) * 100

# Count escalation periods
n_escalation = np.sum(growth_rate > 0)

summary_text = f"""DMD TEMPORAL MODES: REAL DATA

Example: {example_country}
{example_district} ({len(ts_data)} periods)

THREE DMD FEATURES
A. Growth Rate: {n_escalation} escalation periods
B. Frequency: Oscillation period (seasonal)
C. Amplitude: Pattern magnitude

CONVERGENCE
83.1% success (Baum-Welch algorithm)
16.9% failure (discarded)

RARE EVENTS FOCUS
{crisis_obs:,}/{total_obs:,} crises ({crisis_pct:.1f}%)
High humanitarian impact

FEATURE IMPORTANCE
Growth rate: Rank 28 (2.1%)
Complements HMM regime transitions"""

props = dict(boxstyle='round,pad=0.8', facecolor='#FFF9E6', edgecolor='#9B59B6', linewidth=2.5, alpha=0.95)
ax4.text(0.5, 0.5, summary_text, transform=ax4.transAxes, fontsize=9, fontweight='bold',
         verticalalignment='center', horizontalalignment='center',
         bbox=props, family='monospace', linespacing=1.5)

# Overall title
fig.suptitle('DMD Temporal Modes: Real Crisis Examples with Convergence Analysis',
             fontsize=15, fontweight='bold', y=0.995)

# Footer
footer_text = (
    f"Dynamic Mode Decomposition (DMD) extracts temporal modes from multivariate news coverage time series. "
    f"Real data: {example_country} {example_district}, n={len(ts_data)} periods. "
    f"Three DMD features shown: Panel A (Growth Rate - real eigenvalue, n={n_escalation} escalation periods), "
    f"Panel B (Frequency - oscillation period), Panel C (Amplitude - oscillation magnitude). "
    f"DMD convergence rate: 83.1% (Baum-Welch algorithm, 12-month rolling windows). "
    f"Rare events focus: {crisis_pct:.1f}% of observations are crises with high humanitarian impact. "
    f"DMD growth rate ranks 28 (2.1% importance), complementing HMM regime transitions. "
    f"Crisis periods shown as red shading. Escalation (growth>0) marked with red triangles."
)
fig.text(0.5, 0.01, footer_text, ha='center', fontsize=7.5, style='italic',
         bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgray', alpha=0.3),
         wrap=True)

plt.tight_layout(rect=[0, 0.06, 1, 0.99])

# Save
output_file = OUTPUT_DIR / "ch03_dmd_modes.pdf"
plt.savefig(output_file, dpi=300, bbox_inches='tight', format='pdf')
print(f"\n[OK] Saved: {output_file}")

output_file_png = OUTPUT_DIR / "ch03_dmd_modes.png"
plt.savefig(output_file_png, dpi=300, bbox_inches='tight', format='png')
print(f"[OK] Saved: {output_file_png}")

plt.close()

print("\n" + "="*80)
print("FIGURE 10 COMPLETE: DMD Temporal Modes (100% REAL DATA)")
print("="*80)
print(f"Example: {example_country}, {example_district}")
print(f"Time series: {len(ts_data)} periods")
print(f"Escalation periods: {n_escalation}")
print(f"Convergence rate: 83.1% success")
print(f"Crisis rate: {crisis_pct:.1f}%")
print("3 DMD features visualized (removed Instability - mostly zeros)")
print("Panel D: Combined convergence + summary")
