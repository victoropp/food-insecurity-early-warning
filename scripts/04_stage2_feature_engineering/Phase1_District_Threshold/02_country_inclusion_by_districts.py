"""
Country Inclusion by Valid Districts
=====================================
Phase 1, Step 2: Determine country inclusion based on valid districts.

This script:
1. Loads the valid districts from Step 1
2. Counts valid districts per country
3. Determines which countries have sufficient valid districts
4. Outputs country inclusion summary

Author: Victor Collins Oppon
Date: December 2025
"""

import sys
from pathlib import Path

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent.parent))

import pandas as pd
import numpy as np
import json
from datetime import datetime

# Import from FINAL_PIPELINE config
from config import (
    BASE_DIR,
    RESULTS_DIR,
    STAGE2_FEATURES_DIR,
    DISTRICT_THRESHOLD_CONFIG
)

# =============================================================================
# CONFIGURATION
# =============================================================================

# Output directories (use STAGE2_FEATURES_DIR from config for consistency)
PHASE1_RESULTS = STAGE2_FEATURES_DIR / 'phase1_district_threshold'
PHASE1_RESULTS.mkdir(parents=True, exist_ok=True)

# Output files
OUTPUT_FILES = {
    'valid_districts': 'valid_districts.csv',
    'district_threshold_analysis': 'district_threshold_analysis.csv',
    'country_inclusion': 'country_inclusion.csv',
    'country_inclusion_summary': 'country_inclusion_summary.json',
    'phase1_summary': 'phase1_summary.json'
}

MIN_VALID_DISTRICTS = DISTRICT_THRESHOLD_CONFIG['min_valid_districts_per_country']

print("=" * 80)
print("PHASE 1: COUNTRY INCLUSION BY VALID DISTRICTS")
print("=" * 80)
print(f"\nMinimum valid districts per country: {MIN_VALID_DISTRICTS}")


def load_valid_districts():
    """Load valid districts from Step 1."""
    print("\n" + "-" * 40)
    print("Loading valid districts from Step 1...")

    input_path = PHASE1_RESULTS / OUTPUT_FILES['valid_districts']

    if not input_path.exists():
        raise FileNotFoundError(
            f"Valid districts file not found: {input_path}\n"
            "Please run 01_district_threshold_analysis.py first."
        )

    valid_df = pd.read_csv(input_path)
    print(f"   Loaded {len(valid_df):,} valid districts")

    return valid_df


def load_district_statistics():
    """Load full district statistics from Step 1."""
    input_path = PHASE1_RESULTS / OUTPUT_FILES['district_threshold_analysis']

    if input_path.exists():
        return pd.read_csv(input_path)
    return None


def count_districts_by_country(valid_df):
    """Count valid districts per country."""
    print("\n" + "-" * 40)
    print("Counting valid districts per country...")

    # Use ipc_country (canonical column from Phase 1)
    country_col = 'ipc_country'
    if country_col not in valid_df.columns:
        raise ValueError(f"Country column '{country_col}' not found in valid districts file. Available columns: {valid_df.columns.tolist()}")

    country_counts = valid_df.groupby(country_col).size().reset_index(name='n_valid_districts')
    # Keep column as ipc_country for consistency with downstream code
    country_counts = country_counts.sort_values('n_valid_districts', ascending=False)

    print(f"\n   Countries with valid districts: {len(country_counts)}")
    print(f"   Total valid districts: {country_counts['n_valid_districts'].sum():,}")

    return country_counts


def determine_country_inclusion(country_counts, district_stats=None):
    """
    Determine which countries meet inclusion criteria.

    A country is included if it has >= MIN_VALID_DISTRICTS valid districts.
    """
    print("\n" + "-" * 40)
    print(f"Determining country inclusion (threshold: {MIN_VALID_DISTRICTS} districts)...")

    # Flag countries meeting threshold
    country_counts['included'] = country_counts['n_valid_districts'] >= MIN_VALID_DISTRICTS

    included_countries = country_counts[country_counts['included']]['ipc_country'].tolist()
    excluded_countries = country_counts[~country_counts['included']]['ipc_country'].tolist()

    print(f"\n   Included countries: {len(included_countries)}")
    print(f"   Excluded countries: {len(excluded_countries)}")

    # Add additional statistics if available
    if district_stats is not None:
        # Total districts per country
        total_counts = district_stats.groupby('ipc_country').size().reset_index(name='n_total_districts')
        country_counts = country_counts.merge(total_counts, on='ipc_country', how='left')

        # Average articles per country
        avg_articles = district_stats.groupby('ipc_country')['mean_annual_articles'].mean().reset_index()
        avg_articles.columns = ['ipc_country', 'avg_annual_articles']
        country_counts = country_counts.merge(avg_articles, on='ipc_country', how='left')

        # Inclusion rate per country
        country_counts['district_inclusion_rate'] = (
            country_counts['n_valid_districts'] / country_counts['n_total_districts']
        )

    return country_counts, included_countries, excluded_countries


def generate_summary_report(country_counts, included_countries, excluded_countries):
    """Generate comprehensive summary report."""
    summary = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'min_valid_districts_per_country': MIN_VALID_DISTRICTS,
        },
        'results': {
            'total_countries_with_data': len(country_counts),
            'included_countries': len(included_countries),
            'excluded_countries': len(excluded_countries),
            'inclusion_rate': float(len(included_countries) / len(country_counts)),
        },
        'included_country_list': included_countries,
        'excluded_country_list': excluded_countries,
        'country_details': {},
    }

    # Add details for each country
    for _, row in country_counts.iterrows():
        country = row['ipc_country']
        summary['country_details'][country] = {
            'n_valid_districts': int(row['n_valid_districts']),
            'included': bool(row['included']),
        }

        if 'n_total_districts' in row:
            summary['country_details'][country]['n_total_districts'] = int(row['n_total_districts'])
            summary['country_details'][country]['district_inclusion_rate'] = float(
                row['district_inclusion_rate']
            ) if pd.notna(row['district_inclusion_rate']) else None

        if 'avg_annual_articles' in row:
            summary['country_details'][country]['avg_annual_articles'] = float(
                row['avg_annual_articles']
            ) if pd.notna(row['avg_annual_articles']) else None

    return summary


def print_country_summary(country_counts, included_countries, excluded_countries):
    """Print formatted summary of country inclusion."""
    print("\n" + "=" * 80)
    print("COUNTRY INCLUSION SUMMARY")
    print("=" * 80)

    print(f"\nINCLUDED COUNTRIES ({len(included_countries)}):")
    print("-" * 40)

    included_df = country_counts[country_counts['included']].sort_values(
        'n_valid_districts', ascending=False
    )

    for _, row in included_df.iterrows():
        country = row['ipc_country']
        n_valid = row['n_valid_districts']

        if 'n_total_districts' in row and pd.notna(row['n_total_districts']):
            n_total = int(row['n_total_districts'])
            rate = row['district_inclusion_rate'] * 100
            print(f"   {country:<30} {n_valid:>4} valid / {n_total:>4} total ({rate:>5.1f}%)")
        else:
            print(f"   {country:<30} {n_valid:>4} valid districts")

    if excluded_countries:
        print(f"\nEXCLUDED COUNTRIES ({len(excluded_countries)}):")
        print("-" * 40)

        excluded_df = country_counts[~country_counts['included']].sort_values(
            'n_valid_districts', ascending=False
        )

        for _, row in excluded_df.iterrows():
            country = row['ipc_country']
            n_valid = row['n_valid_districts']
            reason = f"Only {n_valid} valid districts (< {MIN_VALID_DISTRICTS} required)"
            print(f"   {country:<30} {reason}")


def ensure_directories():
    """Ensure output directories exist."""
    PHASE1_RESULTS.mkdir(parents=True, exist_ok=True)

def main():
    """Main execution function."""
    ensure_directories()

    # Step 1: Load valid districts
    valid_df = load_valid_districts()

    # Step 2: Load full district statistics (optional, for additional metrics)
    district_stats = load_district_statistics()

    # Step 3: Count districts by country
    country_counts = count_districts_by_country(valid_df)

    # Step 4: Determine country inclusion
    country_counts, included_countries, excluded_countries = determine_country_inclusion(
        country_counts, district_stats
    )

    # Step 5: Generate summary
    summary = generate_summary_report(country_counts, included_countries, excluded_countries)

    # ==========================================================================
    # SAVE OUTPUTS
    # ==========================================================================
    print("\n" + "-" * 40)
    print("Saving outputs...")

    # Save country inclusion matrix
    output_path = PHASE1_RESULTS / 'country_district_counts.csv'
    country_counts.to_csv(output_path, index=False)
    print(f"   Saved: {output_path}")

    # Save included countries list
    included_df = pd.DataFrame({
        'country': included_countries,
        'n_valid_districts': [
            country_counts[country_counts['ipc_country'] == c]['n_valid_districts'].values[0]
            for c in included_countries
        ]
    })
    output_path = PHASE1_RESULTS / 'included_countries.csv'
    included_df.to_csv(output_path, index=False)
    print(f"   Saved: {output_path}")

    # ==========================================================================
    # CRITICAL FIX: Filter valid_districts.csv to ONLY included countries
    # ==========================================================================
    print("\n" + "-" * 40)
    print("Filtering valid districts to included countries only...")

    # Filter to only districts from included countries
    country_col = 'ipc_country'
    valid_df_filtered = valid_df[valid_df[country_col].isin(included_countries)].copy()

    print(f"   Original valid districts (all countries): {len(valid_df):,}")
    print(f"   Filtered valid districts (included countries only): {len(valid_df_filtered):,}")
    print(f"   Removed {len(valid_df) - len(valid_df_filtered):,} districts from excluded countries")

    # Overwrite valid_districts.csv with filtered version
    output_path = PHASE1_RESULTS / OUTPUT_FILES['valid_districts']
    valid_df_filtered.to_csv(output_path, index=False)
    print(f"   Saved filtered valid_districts.csv: {output_path}")

    # Update valid_df for summary reporting
    valid_df = valid_df_filtered

    # Save full summary JSON
    output_path = PHASE1_RESULTS / OUTPUT_FILES['country_inclusion_summary']
    with open(output_path, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"   Saved: {output_path}")

    # Save phase 1 summary
    phase1_summary = {
        'phase': 'Phase 1: District Threshold',
        'timestamp': datetime.now().isoformat(),
        'status': 'complete',
        'config': {
            'min_articles_per_year': DISTRICT_THRESHOLD_CONFIG['min_articles_per_year'],
            'min_valid_districts_per_country': MIN_VALID_DISTRICTS,
        },
        'results': {
            'total_districts': len(valid_df) + (
                len(district_stats) - len(valid_df) if district_stats is not None else 0
            ),
            'valid_districts': len(valid_df),
            'total_countries': len(country_counts),
            'included_countries': len(included_countries),
        },
        'included_country_list': included_countries,
    }

    output_path = PHASE1_RESULTS / OUTPUT_FILES['phase1_summary']
    with open(output_path, 'w') as f:
        json.dump(phase1_summary, f, indent=2)
    print(f"   Saved: {output_path}")

    # ==========================================================================
    # PRINT SUMMARY
    # ==========================================================================
    print_country_summary(country_counts, included_countries, excluded_countries)

    print("\n" + "=" * 80)
    print("PHASE 1 COMPLETE")
    print("=" * 80)
    print(f"\n   Included countries: {len(included_countries)}")
    print(f"   Valid districts: {len(valid_df):,}")
    print(f"\n   Next step: Phase 2 - Feature Engineering")

    return included_countries, country_counts


if __name__ == '__main__':
    included_countries, country_counts = main()
