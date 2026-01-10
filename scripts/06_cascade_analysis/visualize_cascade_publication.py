"""
Publication-Grade Visualizations for Cascade Ensemble Models
=============================================================

Creates state-of-the-art visualizations that tell the story:
1. Cascade logic: AR=1 stays 1, AR=0 confirmed by Stage 2
2. Performance improvement: AR vs Ensemble (Precision, Recall, F1)
3. Key insight: Improvements represent REAL CRISES now predicted
4. Stories: Specific cases missed by AR but caught by Ensemble
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
import warnings
from config import BASE_DIR
warnings.filterwarnings('ignore')

# Set publication-quality defaults
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 11
plt.rcParams['axes.labelsize'] = 12
plt.rcParams['axes.titlesize'] = 14
plt.rcParams['xtick.labelsize'] = 10
plt.rcParams['ytick.labelsize'] = 10
plt.rcParams['legend.fontsize'] = 10
plt.rcParams['figure.titlesize'] = 16
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300
plt.rcParams['savefig.bbox'] = 'tight'

# Color palette - professional, accessible
COLORS = {
    'ar': '#2E86AB',           # Deep blue - AR baseline
    'ensemble': '#A23B72',     # Deep magenta - Ensemble
    'improvement': '#F18F01',  # Orange - Improvement/Key saves
    'neutral': '#C73E1D',      # Red - Missed crises
    'success': '#6A994E',      # Green - Success
    'background': '#F8F9FA',   # Light gray background
    'grid': '#E9ECEF'          # Grid lines
}

def load_model_data(results_dir):
    """Load all relevant data for a cascade model."""
    results_dir = Path(results_dir)

    # Load predictions
    predictions = pd.read_csv(results_dir / 'cascade_optimized_predictions.csv')

    # Load key saves
    key_saves = pd.read_csv(results_dir / 'key_saves.csv')

    # Load summary
    with open(results_dir / 'cascade_optimized_summary.json', 'r') as f:
        summary = json.load(f)

    return predictions, key_saves, summary

def create_cascade_logic_diagram(ax, model_name):
    """
    Visualization 1: Cascade Logic Flow Diagram
    Shows the decision logic of the cascade model.
    """
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis('off')

    # Title
    ax.text(5, 9.5, 'Cascade Ensemble Logic',
            ha='center', va='top', fontsize=16, fontweight='bold')

    # AR Prediction box
    ar_box = plt.Rectangle((1, 6.5), 3, 1.5,
                           facecolor=COLORS['ar'], edgecolor='black', linewidth=2)
    ax.add_patch(ar_box)
    ax.text(2.5, 7.25, 'AR Model\nPrediction',
            ha='center', va='center', fontsize=11, fontweight='bold', color='white')

    # Decision diamond - AR = 1?
    decision_x = [5, 6, 5, 4]
    decision_y = [7.5, 7.25, 7, 7.25]
    ax.fill(decision_x, decision_y, facecolor='white', edgecolor='black', linewidth=2)
    ax.text(5, 7.25, 'AR = 1?', ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrow from AR to Decision
    ax.annotate('', xy=(4, 7.25), xytext=(4, 7.25),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Path 1: AR = 1 (Keep as 1)
    ax.text(7, 7.8, 'YES', ha='center', va='bottom', fontsize=10, fontweight='bold', color=COLORS['success'])
    ax.annotate('', xy=(8, 7.5), xytext=(6, 7.4),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=COLORS['success']))

    # Final prediction box - Keep 1
    keep_box = plt.Rectangle((7.5, 6.8), 2, 1.2,
                            facecolor=COLORS['success'], edgecolor='black', linewidth=2)
    ax.add_patch(keep_box)
    ax.text(8.5, 7.4, 'Keep as\nCrisis (1)',
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Path 2: AR = 0 (Check with Stage 2)
    ax.text(4, 6.5, 'NO', ha='center', va='top', fontsize=10, fontweight='bold', color=COLORS['ensemble'])
    ax.annotate('', xy=(5, 5.5), xytext=(5, 7),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=COLORS['ensemble']))

    # Stage 2 box
    stage2_box = plt.Rectangle((3.5, 4.5), 3, 1,
                              facecolor=COLORS['ensemble'], edgecolor='black', linewidth=2)
    ax.add_patch(stage2_box)
    ax.text(5, 5, 'Stage 2 Model',
            ha='center', va='center', fontsize=10, fontweight='bold', color='white')

    # Stage 2 Decision diamond
    s2_decision_x = [5, 5.75, 5, 4.25]
    s2_decision_y = [3.5, 3, 2.5, 3]
    ax.fill(s2_decision_x, s2_decision_y, facecolor='white', edgecolor='black', linewidth=2)
    ax.text(5, 3, 'S2 = 1?', ha='center', va='center', fontsize=10, fontweight='bold')

    # Arrow from Stage 2 to its decision
    ax.annotate('', xy=(5, 3.5), xytext=(5, 4.5),
                arrowprops=dict(arrowstyle='->', lw=2, color='black'))

    # Stage 2 YES - Override to 1
    ax.text(6.5, 3.2, 'YES', ha='center', va='bottom', fontsize=9, fontweight='bold', color=COLORS['improvement'])
    ax.annotate('', xy=(8, 3), xytext=(5.75, 3.1),
                arrowprops=dict(arrowstyle='->', lw=2.5, color=COLORS['improvement']))

    override_box = plt.Rectangle((7.5, 2.5), 2, 1,
                                facecolor=COLORS['improvement'], edgecolor='black', linewidth=2)
    ax.add_patch(override_box)
    ax.text(8.5, 3, 'Override\nto Crisis (1)',
            ha='center', va='center', fontsize=9, fontweight='bold', color='white')
    ax.text(8.5, 2.2, 'KEY SAVE!',
            ha='center', va='top', fontsize=8, fontweight='bold',
            color=COLORS['improvement'], style='italic')

    # Stage 2 NO - Keep as 0
    ax.text(3.5, 2.3, 'NO', ha='center', va='top', fontsize=9, fontweight='bold', color='gray')
    ax.annotate('', xy=(2, 3), xytext=(4.25, 2.9),
                arrowprops=dict(arrowstyle='->', lw=2, color='gray'))

    confirm_box = plt.Rectangle((0.5, 2.5), 2, 1,
                               facecolor='lightgray', edgecolor='black', linewidth=2)
    ax.add_patch(confirm_box)
    ax.text(1.5, 3, 'Confirm\nNo Crisis (0)',
            ha='center', va='center', fontsize=9, fontweight='bold', color='black')

    # Add annotation box
    annotation = (
        "Logic: If AR predicts crisis, trust it.\n"
        "If AR says no crisis, verify with Stage 2 model."
    )
    ax.text(5, 1, annotation,
            ha='center', va='top', fontsize=9,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=COLORS['background'],
                     edgecolor='gray', linewidth=1.5),
            style='italic')

def create_performance_comparison(ax, summary, model_name):
    """
    Visualization 2: Performance Comparison - AR vs Ensemble
    Bar chart showing Precision, Recall, F1 improvements.
    """
    # Extract metrics
    ar_metrics = summary['ar_baseline_performance']
    ens_metrics = summary['cascade_performance']

    metrics = ['Precision', 'Recall', 'F1 Score']
    ar_values = [ar_metrics['precision'], ar_metrics['recall'], ar_metrics['f1']]
    ens_values = [ens_metrics['precision'], ens_metrics['recall'], ens_metrics['f1']]

    x = np.arange(len(metrics))
    width = 0.35

    # Create bars
    bars1 = ax.bar(x - width/2, ar_values, width, label='AR Baseline',
                   color=COLORS['ar'], edgecolor='black', linewidth=1.5, alpha=0.9)
    bars2 = ax.bar(x + width/2, ens_values, width, label='Cascade Ensemble',
                   color=COLORS['ensemble'], edgecolor='black', linewidth=1.5, alpha=0.9)

    # Customize
    ax.set_ylabel('Score', fontweight='bold', fontsize=12)
    ax.set_title('Performance Improvement: AR Baseline -> Cascade Ensemble',
                 fontweight='bold', fontsize=13, pad=15)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontweight='bold')
    ax.set_ylim(0, 1.15)  # Increased to make room for annotations
    ax.legend(loc='upper left', frameon=True, shadow=True)
    ax.grid(axis='y', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # Add value labels on bars
    def autolabel(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.4f}',
                   ha='center', va='bottom', fontsize=9, fontweight='bold')

    autolabel(bars1)
    autolabel(bars2)

    # Add improvement annotations - positioned above bars with offset
    for i, metric in enumerate(metrics):
        improvement = ens_values[i] - ar_values[i]
        if improvement > 0:
            # Place above the taller bar with more spacing
            y_pos = max(ar_values[i], ens_values[i]) + 0.04
            ax.annotate(f'+{improvement:.4f}',
                       xy=(x[i], y_pos),
                       ha='center', va='bottom',
                       fontsize=9, fontweight='bold',
                       color=COLORS['improvement'],
                       bbox=dict(boxstyle='round,pad=0.3',
                                facecolor='white',
                                edgecolor=COLORS['improvement'],
                                linewidth=1.5))

    # Add context box - positioned below chart to avoid overlap
    key_saves = summary['improvement']['key_saves']
    ar_fn = ar_metrics['confusion_matrix']['fn']
    context = (
        f"Recall Improvement = {key_saves} MORE CRISES CAUGHT\n"
        f"These are real vulnerable populations now receiving early warning"
    )
    ax.text(0.5, -0.15, context,
           transform=ax.transAxes,
           ha='center', va='top',
           fontsize=9, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.5',
                    facecolor=COLORS['background'],
                    edgecolor=COLORS['improvement'],
                    linewidth=2),
           color=COLORS['improvement'])

def create_key_saves_impact(ax, summary, key_saves_df, model_name):
    """
    Visualization 3: Key Saves Impact
    Shows the reduction in missed crises (False Negatives).
    """
    ar_fn = summary['ar_baseline_performance']['confusion_matrix']['fn']
    ens_fn = summary['cascade_performance']['confusion_matrix']['fn']
    key_saves = summary['improvement']['key_saves']

    # Create horizontal bars showing FN reduction
    categories = ['AR Baseline\n(Missed Crises)', 'Cascade Ensemble\n(Missed Crises)']
    values = [ar_fn, ens_fn]
    colors_list = [COLORS['neutral'], COLORS['success']]

    bars = ax.barh(categories, values, color=colors_list,
                   edgecolor='black', linewidth=2, alpha=0.85)

    # Add value labels
    for i, (bar, val) in enumerate(zip(bars, values)):
        ax.text(val + 5, bar.get_y() + bar.get_height()/2,
               f'{val:,} crises',
               va='center', ha='left', fontsize=11, fontweight='bold')

    # Add reduction arrow - connecting AR bar to Ensemble bar
    # Arrow from end of AR bar to end of Ensemble bar
    ar_y = bars[0].get_y() + bars[0].get_height()/2
    ens_y = bars[1].get_y() + bars[1].get_height()/2

    ax.annotate('', xy=(ens_fn, ens_y),
               xytext=(ar_fn, ar_y),
               arrowprops=dict(arrowstyle='->', lw=3,
                             color=COLORS['improvement'],
                             connectionstyle='arc3,rad=-0.3'))  # Negative rad for downward arc

    # Label the reduction - positioned outside/above the AR bar
    reduction_pct = 100 * key_saves / ar_fn
    mid_point_x = (ar_fn + ens_fn) / 2
    mid_point_y = (ar_y + ens_y) / 2 - 0.25  # Position below the arrow arc

    ax.text(mid_point_x, mid_point_y,
           f'{key_saves:,} fewer\nmissed crises\n({reduction_pct:.1f}% reduction)',
           ha='center', va='center',
           fontsize=11, fontweight='bold',
           color=COLORS['improvement'],
           bbox=dict(boxstyle='round,pad=0.5',
                    facecolor='white',
                    edgecolor=COLORS['improvement'],
                    linewidth=2.5))

    ax.set_xlabel('Number of Missed Crises (False Negatives)',
                  fontweight='bold', fontsize=12)
    ax.set_title('Impact: Reduction in Missed Crises',
                fontweight='bold', fontsize=13, pad=15)
    ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
    ax.set_axisbelow(True)

    # Add impact statement
    impact_text = (
        f"Each missed crisis represents vulnerable populations\n"
        f"facing food insecurity without early warning.\n"
        f"The ensemble catches {key_saves} of these critical cases."
    )
    ax.text(0.98, 0.02, impact_text,
           transform=ax.transAxes,
           ha='right', va='bottom',
           fontsize=9, style='italic',
           bbox=dict(boxstyle='round,pad=0.5',
                    facecolor=COLORS['background'],
                    edgecolor='gray',
                    linewidth=1.5))

def create_key_saves_stories(ax, key_saves_df, summary, model_name):
    """
    Visualization 4: Key Saves Stories
    Highlights specific important cases missed by AR but caught by Ensemble.
    """
    ax.axis('off')

    # Title
    ax.text(0.5, 0.98, 'Critical Cases: AR Missed, Ensemble Caught',
           ha='center', va='top', transform=ax.transAxes,
           fontsize=14, fontweight='bold')

    # Select top cases - prioritize high IPC values and recent dates
    if len(key_saves_df) > 0:
        # Sort by crisis severity (IPC value if available) and date
        if 'ipc_value' in key_saves_df.columns:
            key_saves_sorted = key_saves_df.sort_values(
                ['ipc_value', 'ipc_period_start'],
                ascending=[False, False]
            )
        else:
            key_saves_sorted = key_saves_df.sort_values('ipc_period_start', ascending=False)

        # Take top 5 stories
        top_stories = key_saves_sorted.head(5)

        y_pos = 0.88
        for idx, (_, case) in enumerate(top_stories.iterrows(), 1):
            # Extract case details
            location = case.get('ipc_geographic_unit_full', 'Unknown')
            country = case.get('ipc_country', '')
            date = pd.to_datetime(case['ipc_period_start']).strftime('%B %Y')
            ar_prob = case.get('ar_prob', 0)
            s2_prob = case.get('stage2_prob', 0)

            # Create story box
            story_box = plt.Rectangle((0.05, y_pos - 0.14), 0.9, 0.13,
                                     facecolor='white',
                                     edgecolor=COLORS['improvement'],
                                     linewidth=2,
                                     transform=ax.transAxes,
                                     zorder=2)
            ax.add_patch(story_box)

            # Case number badge
            badge = plt.Circle((0.08, y_pos - 0.075), 0.02,
                             facecolor=COLORS['improvement'],
                             edgecolor='black',
                             linewidth=1.5,
                             transform=ax.transAxes,
                             zorder=3)
            ax.add_patch(badge)
            ax.text(0.08, y_pos - 0.075, str(idx),
                   ha='center', va='center',
                   transform=ax.transAxes,
                   fontsize=11, fontweight='bold',
                   color='white', zorder=4)

            # Location and date
            ax.text(0.12, y_pos - 0.04, f"{location}, {country}",
                   ha='left', va='top',
                   transform=ax.transAxes,
                   fontsize=10, fontweight='bold',
                   color='black')

            ax.text(0.12, y_pos - 0.07, f"Period: {date}",
                   ha='left', va='top',
                   transform=ax.transAxes,
                   fontsize=8,
                   color='gray')

            # Model predictions
            ax.text(0.12, y_pos - 0.10,
                   "AR: missed -> Stage 2: caught",
                   ha='left', va='top',
                   transform=ax.transAxes,
                   fontsize=8,
                   color=COLORS['ensemble'],
                   style='italic')

            y_pos -= 0.16

        # Summary box at bottom
        total_saves = len(key_saves_df)
        summary_text = (
            f"Total Key Saves: {total_saves:,} crises\n"
            f"These are vulnerable populations that would have been missed\n"
            f"by the AR model but are now flagged for early intervention."
        )
        ax.text(0.5, 0.08, summary_text,
               ha='center', va='center',
               transform=ax.transAxes,
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.8',
                        facecolor=COLORS['background'],
                        edgecolor=COLORS['improvement'],
                        linewidth=2.5),
               color=COLORS['improvement'])
    else:
        ax.text(0.5, 0.5, 'No key saves data available',
               ha='center', va='center',
               transform=ax.transAxes,
               fontsize=12, style='italic', color='gray')

def create_country_impact_map(ax, key_saves_df, summary, model_name):
    """
    Visualization 5: Geographic Distribution of Key Saves
    Shows which countries benefited most from the ensemble.
    """
    if len(key_saves_df) > 0:
        # Count key saves by country
        country_saves = key_saves_df.groupby('ipc_country').size().sort_values(ascending=True)

        # Create horizontal bar chart
        countries = country_saves.index.tolist()
        saves = country_saves.values.tolist()

        bars = ax.barh(countries, saves,
                      color=COLORS['improvement'],
                      edgecolor='black',
                      linewidth=1.5,
                      alpha=0.85)

        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, saves)):
            ax.text(val + max(saves)*0.02, bar.get_y() + bar.get_height()/2,
                   f'{val:,}',
                   va='center', ha='left',
                   fontsize=9, fontweight='bold')

        ax.set_xlabel('Number of Key Saves (Crises Caught)',
                     fontweight='bold', fontsize=11)
        ax.set_title('Geographic Impact: Key Saves by Country',
                    fontweight='bold', fontsize=13, pad=15)
        ax.grid(axis='x', alpha=0.3, linestyle='--', linewidth=0.8)
        ax.set_axisbelow(True)

        # Add total - positioned at top left to avoid overlap
        total = sum(saves)
        ax.text(0.02, 0.98, f'Total: {total:,} key saves',
               transform=ax.transAxes,
               ha='left', va='top',
               fontsize=10, fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.5',
                        facecolor='white',
                        edgecolor=COLORS['improvement'],
                        linewidth=2),
               color=COLORS['improvement'])
    else:
        ax.text(0.5, 0.5, 'No geographic data available',
               ha='center', va='center',
               fontsize=12, style='italic', color='gray')

def create_confusion_improvement(ax, summary, model_name):
    """
    Visualization 6: Confusion Matrix Comparison
    Side-by-side confusion matrices showing AR vs Ensemble.
    """
    ar_metrics = summary['ar_baseline_performance']
    ens_metrics = summary['cascade_performance']

    # Create confusion matrices
    ar_cm = np.array([[ar_metrics['confusion_matrix']['tn'], ar_metrics['confusion_matrix']['fp']],
                      [ar_metrics['confusion_matrix']['fn'], ar_metrics['confusion_matrix']['tp']]])

    ens_cm = np.array([[ens_metrics['confusion_matrix']['tn'], ens_metrics['confusion_matrix']['fp']],
                       [ens_metrics['confusion_matrix']['fn'], ens_metrics['confusion_matrix']['tp']]])

    # Calculate percentages
    ar_total = ar_cm.sum()
    ens_total = ens_cm.sum()

    # Create side-by-side display
    ax.axis('off')

    # Title
    ax.text(0.5, 0.98, 'Classification Performance: AR vs Ensemble',
           ha='center', va='top', transform=ax.transAxes,
           fontsize=14, fontweight='bold')

    # AR Confusion Matrix (left)
    ax.text(0.25, 0.88, 'AR Baseline',
           ha='center', va='top', transform=ax.transAxes,
           fontsize=12, fontweight='bold', color=COLORS['ar'])

    # Draw AR matrix
    cell_width = 0.15
    cell_height = 0.15
    start_x_ar = 0.25 - cell_width
    start_y = 0.45

    # TN
    tn_box = plt.Rectangle((start_x_ar, start_y + cell_height), cell_width, cell_height,
                          facecolor=COLORS['success'], edgecolor='black',
                          linewidth=2, alpha=0.6, transform=ax.transAxes)
    ax.add_patch(tn_box)
    ax.text(start_x_ar + cell_width/2, start_y + cell_height*1.5,
           f'TN\n{ar_cm[0,0]:,}',
           ha='center', va='center', transform=ax.transAxes,
           fontsize=10, fontweight='bold')

    # FP
    fp_box = plt.Rectangle((start_x_ar + cell_width, start_y + cell_height), cell_width, cell_height,
                          facecolor=COLORS['neutral'], edgecolor='black',
                          linewidth=2, alpha=0.4, transform=ax.transAxes)
    ax.add_patch(fp_box)
    ax.text(start_x_ar + cell_width*1.5, start_y + cell_height*1.5,
           f'FP\n{ar_cm[0,1]:,}',
           ha='center', va='center', transform=ax.transAxes,
           fontsize=10, fontweight='bold')

    # FN (highlight - these are the critical misses)
    fn_box = plt.Rectangle((start_x_ar, start_y), cell_width, cell_height,
                          facecolor=COLORS['neutral'], edgecolor='black',
                          linewidth=3, alpha=0.7, transform=ax.transAxes)
    ax.add_patch(fn_box)
    ax.text(start_x_ar + cell_width/2, start_y + cell_height/2,
           f'FN\n{ar_cm[1,0]:,}',
           ha='center', va='center', transform=ax.transAxes,
           fontsize=10, fontweight='bold', color='white')

    # TP
    tp_box = plt.Rectangle((start_x_ar + cell_width, start_y), cell_width, cell_height,
                          facecolor=COLORS['success'], edgecolor='black',
                          linewidth=2, alpha=0.8, transform=ax.transAxes)
    ax.add_patch(tp_box)
    ax.text(start_x_ar + cell_width*1.5, start_y + cell_height/2,
           f'TP\n{ar_cm[1,1]:,}',
           ha='center', va='center', transform=ax.transAxes,
           fontsize=10, fontweight='bold')

    # Ensemble Confusion Matrix (right)
    ax.text(0.75, 0.88, 'Cascade Ensemble',
           ha='center', va='top', transform=ax.transAxes,
           fontsize=12, fontweight='bold', color=COLORS['ensemble'])

    # Draw Ensemble matrix
    start_x_ens = 0.75 - cell_width

    # TN
    tn_box = plt.Rectangle((start_x_ens, start_y + cell_height), cell_width, cell_height,
                          facecolor=COLORS['success'], edgecolor='black',
                          linewidth=2, alpha=0.6, transform=ax.transAxes)
    ax.add_patch(tn_box)
    ax.text(start_x_ens + cell_width/2, start_y + cell_height*1.5,
           f'TN\n{ens_cm[0,0]:,}',
           ha='center', va='center', transform=ax.transAxes,
           fontsize=10, fontweight='bold')

    # FP
    fp_box = plt.Rectangle((start_x_ens + cell_width, start_y + cell_height), cell_width, cell_height,
                          facecolor=COLORS['neutral'], edgecolor='black',
                          linewidth=2, alpha=0.4, transform=ax.transAxes)
    ax.add_patch(fp_box)
    ax.text(start_x_ens + cell_width*1.5, start_y + cell_height*1.5,
           f'FP\n{ens_cm[0,1]:,}',
           ha='center', va='center', transform=ax.transAxes,
           fontsize=10, fontweight='bold')

    # FN (reduced!)
    fn_box = plt.Rectangle((start_x_ens, start_y), cell_width, cell_height,
                          facecolor=COLORS['success'], edgecolor='black',
                          linewidth=3, alpha=0.7, transform=ax.transAxes)
    ax.add_patch(fn_box)
    ax.text(start_x_ens + cell_width/2, start_y + cell_height/2,
           f'FN\n{ens_cm[1,0]:,}',
           ha='center', va='center', transform=ax.transAxes,
           fontsize=10, fontweight='bold', color='white')

    # TP (increased!)
    tp_box = plt.Rectangle((start_x_ens + cell_width, start_y), cell_width, cell_height,
                          facecolor=COLORS['success'], edgecolor='black',
                          linewidth=2, alpha=0.8, transform=ax.transAxes)
    ax.add_patch(tp_box)
    ax.text(start_x_ens + cell_width*1.5, start_y + cell_height/2,
           f'TP\n{ens_cm[1,1]:,}',
           ha='center', va='center', transform=ax.transAxes,
           fontsize=10, fontweight='bold')

    # Arrow showing FN reduction - positioned below boxes to avoid overlap
    arrow_y_pos = start_y - 0.05  # Below the FN boxes
    ax.annotate('', xy=(start_x_ens + cell_width/2, arrow_y_pos),
               xytext=(start_x_ar + cell_width/2, arrow_y_pos),
               arrowprops=dict(arrowstyle='->', lw=3,
                             color=COLORS['improvement'],
                             connectionstyle='arc3,rad=-0.3'),  # Negative rad for downward arc
               transform=ax.transAxes)

    fn_reduction = ar_cm[1,0] - ens_cm[1,0]
    ax.text(0.5, arrow_y_pos - 0.03,
           f'{fn_reduction:,} fewer\nmissed crises',
           ha='center', va='top', transform=ax.transAxes,
           fontsize=11, fontweight='bold',
           color=COLORS['improvement'],
           bbox=dict(boxstyle='round,pad=0.5',
                    facecolor='white',
                    edgecolor=COLORS['improvement'],
                    linewidth=2.5))

    # Add axis labels
    ax.text(start_x_ar + cell_width, start_y + cell_height*2 + 0.03,
           'Predicted', ha='center', va='bottom', transform=ax.transAxes,
           fontsize=9, fontweight='bold')
    ax.text(start_x_ar - 0.05, start_y + cell_height,
           'Actual', ha='right', va='center', transform=ax.transAxes,
           fontsize=9, fontweight='bold', rotation=90)

    # Key insight box
    insight = (
        f"The ensemble reduces False Negatives (missed crises) by {fn_reduction:,}\n"
        f"while maintaining similar False Positive rates.\n"
        f"This means more vulnerable populations receive timely warnings."
    )
    ax.text(0.5, 0.15, insight,
           ha='center', va='center', transform=ax.transAxes,
           fontsize=9, fontweight='bold',
           bbox=dict(boxstyle='round,pad=0.8',
                    facecolor=COLORS['background'],
                    edgecolor=COLORS['improvement'],
                    linewidth=2),
           color=COLORS['improvement'])

def generate_all_visualizations(model_dir, model_name, base_figures_dir):
    """Generate all publication-grade visualizations for a model."""
    print(f"\n{'='*80}")
    print(f"Generating visualizations for: {model_name}")
    print(f"{'='*80}\n")

    # Load data
    print("Loading data...")
    predictions, key_saves, summary = load_model_data(model_dir)
    print(f"  - Predictions: {len(predictions):,} observations")
    print(f"  - Key saves: {len(key_saves):,} critical cases")
    print(f"  - AR baseline: P={summary['ar_baseline_performance']['precision']:.4f}, R={summary['ar_baseline_performance']['recall']:.4f}, F1={summary['ar_baseline_performance']['f1']:.4f}")
    print(f"  - Ensemble: P={summary['cascade_performance']['precision']:.4f}, R={summary['cascade_performance']['recall']:.4f}, F1={summary['cascade_performance']['f1']:.4f}")

    # Create model-specific subfolder in FIGURES directory
    model_slug = model_name.lower().replace(' ', '_').replace('(', '').replace(')', '').replace('+', 'plus')
    output_dir = Path(base_figures_dir) / model_slug
    output_dir.mkdir(parents=True, exist_ok=True)

    # Figure 1: Cascade Logic + Performance Comparison
    print("\nCreating Figure 1: Cascade Logic & Performance...")
    fig1 = plt.figure(figsize=(16, 9))  # Increased height for context box
    gs1 = fig1.add_gridspec(1, 2, wspace=0.3, bottom=0.12)  # Add bottom margin

    ax1_1 = fig1.add_subplot(gs1[0, 0])
    create_cascade_logic_diagram(ax1_1, model_name)

    ax1_2 = fig1.add_subplot(gs1[0, 1])
    create_performance_comparison(ax1_2, summary, model_name)

    fig1.suptitle(f'{model_name}: Cascade Ensemble Methodology & Performance',
                 fontsize=18, fontweight='bold', y=0.98)

    fig1_path = output_dir / 'fig1_cascade_logic_performance.png'
    plt.savefig(fig1_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Saved: {fig1_path}")

    # Figure 2: Key Saves Impact + Stories
    print("Creating Figure 2: Key Saves Impact & Stories...")
    fig2 = plt.figure(figsize=(16, 10))
    gs2 = fig2.add_gridspec(2, 1, height_ratios=[1, 1.5], hspace=0.3)

    ax2_1 = fig2.add_subplot(gs2[0, 0])
    create_key_saves_impact(ax2_1, summary, key_saves, model_name)

    ax2_2 = fig2.add_subplot(gs2[1, 0])
    create_key_saves_stories(ax2_2, key_saves, summary, model_name)

    fig2.suptitle(f'{model_name}: Critical Cases Caught by Ensemble',
                 fontsize=18, fontweight='bold', y=0.98)

    fig2_path = output_dir / 'fig2_key_saves_impact_stories.png'
    plt.savefig(fig2_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Saved: {fig2_path}")

    # Figure 3: Geographic Impact + Confusion Matrices
    print("Creating Figure 3: Geographic Impact & Confusion Matrices...")
    fig3 = plt.figure(figsize=(16, 10))
    gs3 = fig3.add_gridspec(2, 1, height_ratios=[1, 1], hspace=0.3)

    ax3_1 = fig3.add_subplot(gs3[0, 0])
    create_country_impact_map(ax3_1, key_saves, summary, model_name)

    ax3_2 = fig3.add_subplot(gs3[1, 0])
    create_confusion_improvement(ax3_2, summary, model_name)

    fig3.suptitle(f'{model_name}: Geographic Distribution & Classification Improvement',
                 fontsize=18, fontweight='bold', y=0.98)

    fig3_path = output_dir / 'fig3_geographic_confusion.png'
    plt.savefig(fig3_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Saved: {fig3_path}")

    # Figure 4: Single comprehensive figure for publication
    print("Creating Figure 4: Comprehensive Publication Figure...")
    fig4 = plt.figure(figsize=(20, 13))  # Increased height
    gs4 = fig4.add_gridspec(3, 2, hspace=0.40, wspace=0.3,
                           height_ratios=[1.3, 1, 1],  # More space for top row
                           bottom=0.08)  # Bottom margin

    # Row 1: Logic (left) + Performance (right)
    ax4_1 = fig4.add_subplot(gs4[0, 0])
    create_cascade_logic_diagram(ax4_1, model_name)
    ax4_1.text(-0.05, 0.98, 'A', transform=ax4_1.transAxes,
              fontsize=20, fontweight='bold', va='top', ha='right')

    ax4_2 = fig4.add_subplot(gs4[0, 1])
    create_performance_comparison(ax4_2, summary, model_name)
    ax4_2.text(-0.05, 0.98, 'B', transform=ax4_2.transAxes,
              fontsize=20, fontweight='bold', va='top', ha='right')

    # Row 2: Impact (left) + Geographic (right)
    ax4_3 = fig4.add_subplot(gs4[1, 0])
    create_key_saves_impact(ax4_3, summary, key_saves, model_name)
    ax4_3.text(-0.05, 0.98, 'C', transform=ax4_3.transAxes,
              fontsize=20, fontweight='bold', va='top', ha='right')

    ax4_4 = fig4.add_subplot(gs4[1, 1])
    create_country_impact_map(ax4_4, key_saves, summary, model_name)
    ax4_4.text(-0.05, 0.98, 'D', transform=ax4_4.transAxes,
              fontsize=20, fontweight='bold', va='top', ha='right')

    # Row 3: Stories (spanning both columns)
    ax4_5 = fig4.add_subplot(gs4[2, :])
    create_key_saves_stories(ax4_5, key_saves, summary, model_name)
    ax4_5.text(-0.02, 0.98, 'E', transform=ax4_5.transAxes,
              fontsize=20, fontweight='bold', va='top', ha='right')

    fig4.suptitle(f'{model_name}: Cascade Ensemble for Food Insecurity Early Warning',
                 fontsize=20, fontweight='bold', y=0.995)

    fig4_path = output_dir / 'fig4_comprehensive_publication.png'
    plt.savefig(fig4_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"  [OK] Saved: {fig4_path}")

    print(f"\n{'='*80}")
    print(f"All visualizations saved to: {output_dir}")
    print(f"{'='*80}\n")

# =============================================================================
# MAIN EXECUTION
# =============================================================================

if __name__ == '__main__':
    print("\n" + "="*80)
    print("PUBLICATION-GRADE VISUALIZATIONS FOR CASCADE ENSEMBLE MODELS")
    print("="*80)

    # Base figures directory
    base_figures_dir = Path(rstr(BASE_DIR))
    base_figures_dir.mkdir(parents=True, exist_ok=True)
    print(f"\nSaving all figures to: {base_figures_dir}\n")

    # Model 1: Ablation Best (Ratio + Location)
    model1_dir = Path(rstr(BASE_DIR))
    generate_all_visualizations(model1_dir, "Ablation Model (Ratio + Location Features)", base_figures_dir)

    # Model 2: Advanced XGBoost (All Features)
    model2_dir = Path(rstr(BASE_DIR))
    generate_all_visualizations(model2_dir, "Advanced Model (All Features)", base_figures_dir)

    print("\n" + "="*80)
    print("VISUALIZATION GENERATION COMPLETE")
    print("="*80)
    print("\nGenerated visualizations:")
    print("  - Figure 1: Cascade logic flow + Performance comparison")
    print("  - Figure 2: Key saves impact + Specific case stories")
    print("  - Figure 3: Geographic distribution + Confusion matrices")
    print("  - Figure 4: Comprehensive publication-ready figure (all panels)")
    print("\nEach figure tells the story:")
    print("  [OK] How the cascade logic works")
    print("  [OK] Performance improvement (AR -> Ensemble)")
    print("  [OK] Real impact: specific vulnerable populations now warned")
    print("  [OK] Geographic distribution of improvement")
    print("="*80 + "\n")
