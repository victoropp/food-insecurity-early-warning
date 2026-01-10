#!/usr/bin/env python3
"""
Author: Victor Collins Oppon
MSc Data Science Dissertation
Middlesex University, 2025
"""

"""
Generate Comprehensive Figure Index
====================================
Creates a detailed catalog of all visualization outputs with descriptions.

Date: December 24, 2025
"""

import sys
from pathlib import Path
from datetime import datetime
import json

sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FIGURES_DIR

print("=" * 80)
print("GENERATING FIGURE INDEX")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

# Scan all figures
all_figures = list(FIGURES_DIR.rglob("*.png"))
print(f"Found {len(all_figures)} PNG figures\n")

# Organize by category
figure_catalog = {}

for fig_path in sorted(all_figures):
    # Get relative path from FIGURES_DIR
    rel_path = fig_path.relative_to(FIGURES_DIR)
    category = str(rel_path.parent) if rel_path.parent != Path('.') else 'root'

    if category not in figure_catalog:
        figure_catalog[category] = []

    figure_catalog[category].append({
        'filename': fig_path.name,
        'path': str(rel_path),
        'size_kb': fig_path.stat().st_size / 1024,
        'modified': datetime.fromtimestamp(fig_path.stat().st_mtime).strftime('%Y-%m-%d %H:%M:%S')
    })

# Generate markdown index
index_md = []
index_md.append("# FINAL PIPELINE - Figure Index")
index_md.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
index_md.append(f"\nTotal figures: {len(all_figures)}")
index_md.append(f"\nBase directory: {FIGURES_DIR}")
index_md.append("\n---\n")

# Figure descriptions by category
descriptions = {
    'ensemble': {
        'title': '## Ensemble Analysis Visualizations',
        'description': 'Two-stage ensemble model combining AR Baseline (Stage 1) with XGBoost Advanced (Stage 2)',
        'figures': {
            '01_weight_optimization_curve.png': 'Weight optimization showing AUC vs alpha (Stage 1 weight). Optimal weight: 58% Stage 1, 42% Stage 2, achieving AUC of 0.9394.',
            '02_performance_comparison.png': 'Performance comparison across Stage 1, Stage 2, and Ensemble for AUC-ROC, PR-AUC, Brier Score, and F1 Score.',
            '03_weight_breakdown_and_metrics.png': 'Pie chart of ensemble weights and precision-recall comparison across all three approaches.',
            '04_data_coverage.png': 'Data coverage comparison showing Stage 1-only, common observations (1,383), and Stage 2-only observations.'
        }
    },
    'pipeline': {
        'title': '## Pipeline Architecture',
        'description': 'End-to-end pipeline flow from raw data to final predictions',
        'figures': {
            'pipeline_architecture.png': 'Complete pipeline architecture showing Stage 0 (Literature Baseline, AUC 0.70), Stage 1 (AR Baseline, AUC 0.845), Stage 2 (XGBoost Advanced WITH_AR, AUC 0.918), and Ensemble (AUC 0.939). Includes data flow, filtering logic (WITH_AR_FILTER), and performance metrics at each stage.'
        }
    },
    'dashboards': {
        'title': '## Model Comparison Dashboard',
        'description': 'Comprehensive comparison of all 9 trained models',
        'figures': {
            'model_ranking_auc.png': 'Horizontal bar chart ranking all models by AUC-ROC. Color-coded by feature type (Ratio, Z-score, +HMM+DMD variants).',
            'threshold_comparison.png': 'Three-panel comparison of precision, recall, and threshold values across Youden, F1-max, and High-Recall strategies for top 6 models.',
            'filter_comparison.png': 'Performance by filter variant (WITH_AR_FILTER vs NO_FILTER vs NO_AR_FILTER), showing mean AUC and F1 scores with error bars.'
        }
    },
    'feature_importance': {
        'title': '## Feature Importance Analysis',
        'description': 'Consensus feature importance across multiple models and methods',
        'figures': {
            'feature_importance_comparison.png': 'Top 20 features ranked by consensus importance score. Color intensity indicates number of independent sources supporting each feature.'
        }
    },
    'calibration': {
        'title': '## Model Calibration',
        'description': 'Calibration and reliability analysis for XGBoost models',
        'figures': {
            'calibration_plots.png': 'Two-panel calibration analysis: (1) Calibration curve comparing predicted vs actual probabilities, (2) Reliability diagram showing probability distributions for positive and negative classes with Brier score.'
        }
    },
    'publication_figures': {
        'title': '## Publication-Ready Figures',
        'description': 'Curated set of figures formatted for publication',
        'figures': {
            'fig1_model_comparison.png': 'Model performance comparison bar chart with error bars, showing mean AUC-ROC across all baseline and advanced models.'
        }
    },
    'mixed_effects': {
        'title': '## Mixed Effects Model Analysis',
        'description': 'Dedicated analysis of 4 mixed effects models with country-level random effects.',
        'figures': {
            '01_performance_ranking.png': 'Horizontal bar chart ranking all 4 mixed effects models by AUC. Color-coded by feature type (ratio vs zscore, ±HMM+DMD).',
            '02_country_variance.png': 'Box plots showing country-level AUC distribution for each model. Reveals random effects heterogeneity across 13 countries.',
            '03_feature_type_comparison.png': 'Grouped bar chart comparing feature type distribution (Ratio/Z-score, HMM, DMD) across all mixed effects models.',
            '04_me_vs_xgboost.png': 'Scatter plot showing complexity-performance trade-off. X-axis: number of features, Y-axis: AUC. Compares mixed effects vs XGBoost models.',
            '05_pr_curves.png': 'Precision-Recall curves for all 4 mixed effects models, showing calibration differences and PR-AUC values.',
            '06_variance_components.png': 'Bar chart showing between-country variance in AUC values across 13 countries, demonstrating country-level performance heterogeneity.'
        }
    }
}

# Write detailed catalog
for category in sorted(figure_catalog.keys()):
    if category in descriptions:
        index_md.append(descriptions[category]['title'])
        index_md.append(f"\n**{descriptions[category]['description']}**\n")
    else:
        index_md.append(f"## {category}")
        index_md.append("")

    figures = figure_catalog[category]
    index_md.append(f"**Total figures:** {len(figures)}\n")

    for fig in figures:
        fig_name = fig['filename']

        # Add description if available
        if category in descriptions and fig_name in descriptions[category]['figures']:
            desc = descriptions[category]['figures'][fig_name]
            index_md.append(f"### {fig_name}")
            index_md.append(f"\n**Description:** {desc}")
        else:
            index_md.append(f"### {fig_name}")

        index_md.append(f"- **Path:** `{fig['path']}`")
        index_md.append(f"- **Size:** {fig['size_kb']:.1f} KB")
        index_md.append(f"- **Modified:** {fig['modified']}")
        index_md.append("")

    index_md.append("---\n")

# Summary statistics
index_md.append("## Summary Statistics\n")
total_size = sum(fig['size_kb'] for figs in figure_catalog.values() for fig in figs)
index_md.append(f"- **Total figures:** {len(all_figures)}")
index_md.append(f"- **Total size:** {total_size / 1024:.2f} MB")
index_md.append(f"- **Categories:** {len(figure_catalog)}")
index_md.append(f"- **Resolution:** 300 DPI (publication quality)")

# Category breakdown
index_md.append("\n### Figures by Category\n")
for category, figs in sorted(figure_catalog.items(), key=lambda x: len(x[1]), reverse=True):
    cat_size = sum(fig['size_kb'] for fig in figs)
    index_md.append(f"- **{category}:** {len(figs)} figures ({cat_size/1024:.2f} MB)")

# Write index file
index_file = FIGURES_DIR / 'FIGURE_INDEX.md'
with open(index_file, 'w', encoding='utf-8') as f:
    f.write('\n'.join(index_md))

print(f"Figure index written to: {index_file}")
print(f"\nCategories indexed: {len(figure_catalog)}")
print(f"Total figures documented: {len(all_figures)}")
print(f"Total size: {total_size / 1024:.2f} MB")

# Also create JSON catalog
json_catalog = {
    'generated': datetime.now().isoformat(),
    'total_figures': len(all_figures),
    'total_size_mb': total_size / 1024,
    'categories': {}
}

for category, figs in figure_catalog.items():
    json_catalog['categories'][category] = {
        'count': len(figs),
        'size_mb': sum(fig['size_kb'] for fig in figs) / 1024,
        'figures': figs
    }

json_file = FIGURES_DIR / 'figure_catalog.json'
with open(json_file, 'w', encoding='utf-8') as f:
    json.dump(json_catalog, f, indent=2)

print(f"JSON catalog written to: {json_file}")

print("\n" + "=" * 80)
print("FIGURE INDEX GENERATION COMPLETE")
print("=" * 80)
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
