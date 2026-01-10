#!/usr/bin/env python3
"""
Author: Victor Collins Oppon
MSc Data Science Dissertation
Middlesex University, 2025
"""

"""
Master Visualization Suite - Publication-Grade Storytelling
============================================================
Orchestrates all visualizations for the complete FINAL_PIPELINE story.

Creates comprehensive state-of-the-art visualizations covering:
1. Stage 1 (AR Baseline) - Performance and threshold analysis
2. Stage 2 (Model Training) - XGBoost, Mixed Effects, Ablation studies
3. Stage 3 (Ensemble) - Weight optimization and improvement analysis
4. Model Comparison - Complete ranking and performance matrix
5. Mixed Effects - Dedicated analysis of 4 country-level random effects models
6. Feature Importance - SHAP and permutation importance
7. Geographic Performance - Country and district-level maps
8. Pipeline Flow - End-to-end architecture visualization

Date: December 24, 2025
"""

import sys
from pathlib import Path
from datetime import datetime
import subprocess

# Add parent directory for config imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from config import FIGURES_DIR, RESULTS_DIR, VISUALIZATION_CONFIG

print("=" * 80)
print("MASTER VISUALIZATION SUITE - FINAL PIPELINE")
print("=" * 80)
print(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
print(f"Output directory: {FIGURES_DIR}")
print(f"Results directory: {RESULTS_DIR}")
print(f"DPI: {VISUALIZATION_CONFIG['dpi']}")
print()

# Define visualization scripts to run (in order)
VISUALIZATION_SCRIPTS = [
    ("Model Comparison Dashboard", "01_model_comparison_dashboard.py"),
    ("AR vs Stage 2 Comparison Maps", "02_ar_vs_stage2_comparison_maps.py"),
    ("Feature Importance Plots", "03_feature_importance_plots.py"),
    ("Calibration Plots", "04_calibration_plots.py"),
    ("Publication Figures", "05_publication_figures.py"),
    ("Ensemble Analysis", "06_ensemble_analysis.py"),
    ("Pipeline Flow Diagram", "07_pipeline_flow_diagram.py"),
    ("Generate Figure Index", "08_generate_figure_index.py"),
    ("Mixed Effects Analysis", "09_mixed_effects_analysis.py"),
]

# Track success/failure
results = {
    'succeeded': [],
    'failed': [],
    'skipped': []
}

SCRIPT_DIR = Path(__file__).parent

# Run each visualization script
for i, (name, script_file) in enumerate(VISUALIZATION_SCRIPTS, 1):
    print(f"\n[{i}/{len(VISUALIZATION_SCRIPTS)}] Running: {name}")
    print("-" * 80)

    script_path = SCRIPT_DIR / script_file

    if not script_path.exists():
        print(f"  SKIPPED: Script not found: {script_file}")
        results['skipped'].append((name, "Script not found"))
        continue

    try:
        # Run script as subprocess
        result = subprocess.run(
            [sys.executable, str(script_path)],
            cwd=SCRIPT_DIR,
            capture_output=True,
            text=True,
            timeout=600  # 10 minute timeout per script
        )

        # Print output
        if result.stdout:
            print(result.stdout)

        if result.returncode == 0:
            print(f"  SUCCESS: {name} completed")
            results['succeeded'].append(name)
        else:
            print(f"  FAILED: {name}")
            if result.stderr:
                print("  Error output:")
                print(result.stderr)
            results['failed'].append((name, result.stderr or "Unknown error"))

    except subprocess.TimeoutExpired:
        print(f"  FAILED: {name} (timeout after 10 minutes)")
        results['failed'].append((name, "Timeout"))

    except Exception as e:
        print(f"  FAILED: {name}")
        print(f"  Error: {str(e)}")
        results['failed'].append((name, str(e)))

# Print summary
print("\n" + "=" * 80)
print("VISUALIZATION SUITE SUMMARY")
print("=" * 80)

print(f"\nSucceeded: {len(results['succeeded'])}")
for name in results['succeeded']:
    print(f"  [OK] {name}")

if results['failed']:
    print(f"\nFailed: {len(results['failed'])}")
    for name, error in results['failed']:
        print(f"  [SKIP] {name}")
        if error and len(error) < 200:
            print(f"    Error: {error}")

if results['skipped']:
    print(f"\nSkipped: {len(results['skipped'])}")
    for name, reason in results['skipped']:
        print(f"  - {name} ({reason})")

print(f"\nTotal: {len(VISUALIZATION_SCRIPTS)} scripts")
print(f"Output directory: {FIGURES_DIR}")
print(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("\n" + "=" * 80)
