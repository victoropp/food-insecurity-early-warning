"""
DISTRICT-LEVEL PIPELINE RUNNER
Executes the complete corrected pipeline for district-level IPC analysis.

This script runs all pipeline stages in sequence:
1. IPC Reference Preparation (district extraction)
2. Articles Aggregation (GADM3/GADM2 matching)
3. Locations Aggregation (GADM3/GADM2 matching)
4. ML Dataset Creation (merge articles + locations)
5. Deduplication (unique district-period observations)
6. Feature Engineering (Lt, Ls autoregressive features)
7. Stage 1 Logistic Regression (spatial CV baseline)

Author: Victor Collins Oppon
"""

import subprocess
import sys
from pathlib import Path
from datetime import datetime
from config import BASE_DIR

# Base directory
BASE_DIR = Path(str(BASE_DIR.parent.parent.parent))
SCRIPTS_DIR = BASE_DIR / 'Scripts' / 'district_pipeline'

# Pipeline steps in order (outputs go to district_level subfolders)
PIPELINE_STEPS = [
    {
        'name': '01. IPC Reference Preparation',
        'script': '01_prepare_ipc_reference.py',
        'description': 'Extract districts from geographic_unit_full_name',
        'output': 'data/district_level/ipc_reference.parquet'
    },
    {
        'name': '02. Articles Aggregation',
        'script': '02_aggregate_articles.py',
        'description': 'Match GDELT articles to IPC districts via GADM3/GADM2',
        'output': 'data/district_level/articles_aggregated.parquet',
        'requires_input': True  # Requires GDELT data
    },
    {
        'name': '03. Locations Aggregation',
        'script': '03_aggregate_locations.py',
        'description': 'Match GDELT locations to IPC districts via GADM3/GADM2',
        'output': 'data/district_level/locations_aggregated.parquet',
        'requires_input': True  # Requires GDELT data
    },
    {
        'name': '04. ML Dataset Creation',
        'script': '04_create_ml_dataset.py',
        'description': 'Merge articles and locations into unified dataset',
        'output': 'data/district_level/ml_dataset_complete.parquet'
    },
    {
        'name': '05. Deduplication',
        'script': '05_deduplicate.py',
        'description': 'Aggregate to unique district-period observations',
        'output': 'data/district_level/ml_dataset_deduplicated.parquet'
    },
    {
        'name': '06. Feature Engineering',
        'script': '06_stage1_feature_engineering.py',
        'description': 'Create Lt (temporal) and Ls (spatial) autoregressive features',
        'output': 'data/district_level/stage1_features.parquet'
    },
    {
        'name': '07. Stage 1 Logistic Regression',
        'script': '07_stage1_logistic_regression.py',
        'description': 'Run baseline model with spatial cross-validation',
        'output': 'results/district_level/stage1_baseline/'
    }
]


def run_script(script_path):
    """Run a Python script and return success status"""
    print(f"\n   Running: {script_path.name}")
    print(f"   " + "-" * 60)

    try:
        result = subprocess.run(
            [sys.executable, str(script_path)],
            capture_output=False,
            text=True,
            cwd=str(BASE_DIR)
        )
        return result.returncode == 0
    except Exception as e:
        print(f"   ERROR: {e}")
        return False


def check_output_exists(output_path):
    """Check if output file/directory exists"""
    full_path = BASE_DIR / output_path
    return full_path.exists()


def main():
    print("=" * 80)
    print("DISTRICT-LEVEL PIPELINE RUNNER")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print(f"Base directory: {BASE_DIR}")
    print()

    # Check which steps to run
    print("Pipeline Steps:")
    print("-" * 80)

    for i, step in enumerate(PIPELINE_STEPS, 1):
        script_path = SCRIPTS_DIR / step['script']
        output_exists = check_output_exists(step['output'])

        status = "[DONE]" if output_exists else "[TODO]"
        print(f"{i}. {step['name']} {status}")
        print(f"   Script: {step['script']}")
        print(f"   Output: {step['output']}")
        print()

    # Ask user which steps to run
    print("-" * 80)
    print("\nOptions:")
    print("  [A] Run ALL steps from beginning")
    print("  [C] Continue from first incomplete step")
    print("  [S] Select specific step to run")
    print("  [Q] Quit")

    choice = input("\nChoice: ").strip().upper()

    if choice == 'Q':
        print("Exiting...")
        return

    steps_to_run = []

    if choice == 'A':
        steps_to_run = list(range(len(PIPELINE_STEPS)))
    elif choice == 'C':
        # Find first incomplete step
        for i, step in enumerate(PIPELINE_STEPS):
            if not check_output_exists(step['output']):
                steps_to_run = list(range(i, len(PIPELINE_STEPS)))
                break
        if not steps_to_run:
            print("All steps already complete!")
            return
    elif choice == 'S':
        step_num = input("Enter step number (1-7): ").strip()
        try:
            idx = int(step_num) - 1
            if 0 <= idx < len(PIPELINE_STEPS):
                steps_to_run = [idx]
            else:
                print("Invalid step number!")
                return
        except ValueError:
            print("Invalid input!")
            return
    else:
        print("Invalid choice!")
        return

    # Run selected steps
    print("\n" + "=" * 80)
    print(f"Running {len(steps_to_run)} step(s)...")
    print("=" * 80)

    results = []

    for idx in steps_to_run:
        step = PIPELINE_STEPS[idx]
        script_path = SCRIPTS_DIR / step['script']

        print(f"\n{'='*80}")
        print(f"STEP {idx+1}: {step['name']}")
        print(f"{'='*80}")
        print(f"Description: {step['description']}")

        if not script_path.exists():
            print(f"ERROR: Script not found: {script_path}")
            results.append((step['name'], False, "Script not found"))
            continue

        start_time = datetime.now()
        success = run_script(script_path)
        elapsed = (datetime.now() - start_time).total_seconds()

        if success:
            results.append((step['name'], True, f"{elapsed:.1f}s"))
            print(f"\n   [OK] Completed in {elapsed:.1f}s")
        else:
            results.append((step['name'], False, "Failed"))
            print(f"\n   [FAILED] Step failed after {elapsed:.1f}s")

            # Ask to continue
            cont = input("\nContinue to next step? [y/N]: ").strip().lower()
            if cont != 'y':
                break

    # Summary
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)

    for name, success, info in results:
        status = "[OK]" if success else "[FAILED]"
        print(f"  {status} {name}: {info}")

    n_success = sum(1 for _, s, _ in results if s)
    n_failed = len(results) - n_success

    print(f"\nTotal: {n_success} succeeded, {n_failed} failed")
    print(f"End time: {datetime.now()}")


if __name__ == "__main__":
    main()
