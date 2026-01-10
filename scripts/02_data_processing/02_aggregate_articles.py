"""
GDELT Articles Aggregation - DISTRICT LEVEL CORRECTED
Aggregates GDELT articles to IPC assessment periods at DISTRICT level.

KEY CORRECTIONS:
1. Uses geographic_unit_full_name as the unique geographic identifier
2. Matches via GADM3 → GADM2 → district name (NOT LHZ first)
3. Each IPC observation is (geographic_unit_full_name, period)

TEMPORAL AGGREGATION (CRITICAL):
4. Aggregates articles over 4-MONTH WINDOWS before each IPC assessment
   - IPC assessments occur in Feb, Jun, Oct
   - For Feb assessment: aggregate Nov, Dec, Jan, Feb articles
   - For Jun assessment: aggregate Mar, Apr, May, Jun articles
   - For Oct assessment: aggregate Jul, Aug, Sep, Oct articles

Author: Corrected for district-level analysis
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from dateutil.relativedelta import relativedelta
from collections import defaultdict
from rapidfuzz import fuzz  # 10-100x faster than fuzzywuzzy
import pyarrow.parquet as pq
import pyarrow as pa
import unicodedata
import gc
import sqlite3
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
from config import BASE_DIR

# 4-month aggregation window (articles from 3 months before to assessment month)
AGGREGATION_MONTHS = 4

# Paths
BASE_DIR = Path(str(BASE_DIR.parent.parent.parent))

# Input files (original locations)
ARTICLES_FILE = BASE_DIR / 'data' / 'african_gkg_articles.csv'
LOCATIONS_FILE = BASE_DIR / 'data' / 'african_gkg_locations_aligned.parquet'

# District pipeline I/O (district_level subfolder)
DISTRICT_DATA_DIR = BASE_DIR / 'data' / 'district_level'
IPC_REF_FILE = DISTRICT_DATA_DIR / 'ipc_reference.parquet'
OUTPUT_PARQUET = DISTRICT_DATA_DIR / 'articles_aggregated.parquet'
OUTPUT_CSV = DISTRICT_DATA_DIR / 'articles_aggregated.csv'

CHUNK_SIZE = 1000000  # Large chunks - SQLite handles memory now
FUZZY_THRESHOLD = 80  # Slightly lower for district name variations
NUM_WORKERS = max(1, multiprocessing.cpu_count() - 1)  # Use all but one CPU core
SQLITE_BATCH_SIZE = 50000  # Batch size for SQLite inserts (memory-efficient)

# Global cache for fuzzy matches (populated at startup)
FUZZY_MATCH_CACHE = {}

# Countries with ONLY national-level IPC data (very limited district info)
# CG (DRC) added due to poor GADM alignment (only 18.6% coverage)
COUNTRY_LEVEL_ONLY = {'AO', 'CG', 'CT', 'LT', 'MR', 'RW', 'TO'}


def normalize_text(text):
    """Normalize text for matching with accent removal."""
    if pd.isna(text):
        return ''
    text = str(text).lower().strip()
    # Remove accents (e.g., Kasaï -> kasai, Équateur -> equateur)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
    text = ' '.join(text.split())
    return text


def find_fuzzy_match(loc_name_norm, country_candidates, country_code):
    """
    Find best fuzzy match for location name among IPC district candidates.
    Uses pre-grouped country_candidates for O(1) country lookup.
    Returns: (ipc_info_list, match_score) or (None, 0)
    """
    if not loc_name_norm or country_code not in country_candidates:
        return None, 0

    best_match = None
    best_score = 0

    # Only iterate through candidates for this country (much faster!)
    for district_norm, ipc_list in country_candidates[country_code]:
        # Calculate fuzzy match score
        score = fuzz.ratio(loc_name_norm, district_norm)

        if score >= FUZZY_THRESHOLD and score > best_score:
            best_score = score
            best_match = ipc_list

    return best_match, best_score


def build_country_candidates(combined_lookup):
    """Pre-group candidates by country for fast fuzzy matching."""
    country_candidates = defaultdict(list)
    for (fips, district_norm), ipc_list in combined_lookup.items():
        country_candidates[fips].append((district_norm, ipc_list))
    return dict(country_candidates)


def precompute_fuzzy_matches(country_candidates, all_gadm_names_by_country):
    """
    Pre-compute ALL fuzzy matches at startup to avoid repeated computation.
    Returns: dict of {(country_code, gadm_norm): matched_ipc_list}
    """
    print("   Pre-computing fuzzy matches (one-time cost)...", flush=True)
    cache = {}
    total_computed = 0

    for country_code, gadm_names in all_gadm_names_by_country.items():
        if country_code not in country_candidates:
            continue

        candidates = country_candidates[country_code]
        for gadm_norm in gadm_names:
            if not gadm_norm:
                continue

            best_match = None
            best_score = 0

            for district_norm, ipc_list in candidates:
                score = fuzz.ratio(gadm_norm, district_norm)
                if score >= FUZZY_THRESHOLD and score > best_score:
                    best_score = score
                    best_match = ipc_list

            if best_match:
                cache[(country_code, gadm_norm)] = best_match
                total_computed += 1

    print(f"   Pre-computed {total_computed:,} fuzzy matches", flush=True)
    return cache


def get_fuzzy_match_cached(gadm_norm, country_code, fuzzy_cache, country_candidates):
    """Get fuzzy match from cache, or compute if not cached."""
    key = (country_code, gadm_norm)
    if key in fuzzy_cache:
        return fuzzy_cache[key], FUZZY_THRESHOLD
    # Fallback to computation (should be rare after pre-computation)
    return find_fuzzy_match(gadm_norm, country_candidates, country_code)


def build_word_lookup(ipc_ref):
    """
    Build word-based lookup from full_name_normalized at startup.

    This enables matching for livelihood zone countries (Zimbabwe, Burundi, Kenya, etc.)
    where IPC uses names like "agrofisheries binga matabeleland north zimbabwe"
    that contain embedded GADM admin names.

    Pre-building this lookup avoids O(gadm_names × ipc_entries) per batch.
    """
    print("   Building word-based lookup from full_name_normalized...", flush=True)
    word_lookup = defaultdict(list)

    for idx, row in ipc_ref.iterrows():
        if pd.notna(row['fips_code']) and pd.notna(row.get('full_name_normalized')):
            country = row['fips_code']
            full_name_norm = normalize_text(row['full_name_normalized'])
            words = full_name_norm.split()

            # Calculate 4-month aggregation window (3 months before + assessment month)
            ipc_period_end = pd.to_datetime(row['projection_end'])
            ipc_period_start = pd.to_datetime(row['projection_start'])
            # Aggregation window starts 3 months before the assessment month
            agg_window_start = ipc_period_start - relativedelta(months=AGGREGATION_MONTHS - 1)
            agg_window_end = ipc_period_end  # End at assessment month end

            # Build IPC info dict with 4-month aggregation window
            ipc_info = {
                'ipc_id': row['ipc_id'],
                'ipc_country': row['country'],
                'ipc_country_code': row['country_code'],
                'ipc_fips_code': row['fips_code'],
                'ipc_district': row['district'],
                'ipc_region': row['region'],
                'ipc_geographic_unit': row['geographic_unit_name'],
                'ipc_geographic_unit_full': row['geographic_unit_full_name'],
                'ipc_period_start': ipc_period_start,  # Original assessment start
                'ipc_period_end': ipc_period_end,  # Original assessment end
                'ipc_period_length_days': row['period_length_days'],
                # 4-month aggregation window for article filtering
                'agg_window_start': agg_window_start,
                'agg_window_end': agg_window_end,
                'ipc_value': row['ipc_value'],
                'ipc_description': row['ipc_description'],
                'ipc_binary_crisis': row['ipc_binary_crisis'],
                'ipc_is_allowing_assistance': row['is_allowing_for_assistance'],
                'ipc_fewsnet_region': row['fewsnet_region'],
                'ipc_geographic_group': row['geographic_group'],
                'ipc_scenario': row['scenario'],
                'ipc_classification_scale': row['classification_scale'],
                'ipc_reporting_date': row['reporting_date'],
            }

            for word in words:
                if len(word) > 2:  # Skip short words
                    key = (country, word)
                    word_lookup[key].append(ipc_info)

    print(f"   Word-based lookup: {len(word_lookup):,} keys", flush=True)
    return word_lookup


def cleanup_temp_files(temp_parquets, lookup_file=None):
    """Clean up temporary files on success or failure"""
    cleaned = 0
    for temp_file in temp_parquets:
        try:
            if temp_file.exists():
                temp_file.unlink()
                cleaned += 1
        except Exception as e:
            print(f"   Warning: Could not delete {temp_file}: {e}")

    if lookup_file and lookup_file.exists():
        try:
            lookup_file.unlink()
            cleaned += 1
        except Exception as e:
            print(f"   Warning: Could not delete {lookup_file}: {e}")

    return cleaned


def process_batch(args):
    """
    Process a single batch of locations in parallel.
    Workers read their own batch from file to avoid memory duplication.
    Returns: (batch_num, results_list, match_stats)
    """
    batch_num, locations_path, ipc_ref_path = args

    # Each worker builds its own lookup to avoid passing large dicts
    ipc_ref = pd.read_parquet(ipc_ref_path)

    # Build IPC lookup in worker
    ipc_lookup = defaultdict(list)
    for idx, row in ipc_ref.iterrows():
        if pd.notna(row['fips_code']) and pd.notna(row['district_normalized']):
            ipc_period_end = pd.to_datetime(row['projection_end'])
            ipc_period_start = pd.to_datetime(row['projection_start'])
            agg_window_start = ipc_period_start - relativedelta(months=AGGREGATION_MONTHS - 1)
            agg_window_end = ipc_period_end

            key = (row['fips_code'], row['district_normalized'])
            ipc_lookup[key].append({
                'ipc_id': row['ipc_id'],
                'ipc_country': row['country'],
                'ipc_country_code': row['country_code'],
                'ipc_fips_code': row['fips_code'],
                'ipc_district': row['district'],
                'ipc_region': row['region'],
                'ipc_geographic_unit': row['geographic_unit_name'],
                'ipc_geographic_unit_full': row['geographic_unit_full_name'],
                'ipc_period_start': ipc_period_start,
                'ipc_period_end': ipc_period_end,
                'ipc_period_length_days': row['period_length_days'],
                'agg_window_start': agg_window_start,
                'agg_window_end': agg_window_end,
                'ipc_value': row['ipc_value'],
                'ipc_description': row['ipc_description'],
                'ipc_binary_crisis': row['ipc_binary_crisis'],
                'ipc_is_allowing_assistance': row['is_allowing_for_assistance'],
                'ipc_fewsnet_region': row['fewsnet_region'],
                'ipc_geographic_group': row['geographic_group'],
                'ipc_scenario': row['scenario'],
                'ipc_classification_scale': row['classification_scale'],
                'ipc_reporting_date': row['reporting_date'],
            })

    # Build word-based lookup from full_name_normalized
    word_lookup = defaultdict(list)
    for idx, row in ipc_ref.iterrows():
        if pd.notna(row['fips_code']) and pd.notna(row.get('full_name_normalized')):
            country = row['fips_code']
            full_name_norm = normalize_text(row['full_name_normalized'])
            words = full_name_norm.split()

            ipc_period_end = pd.to_datetime(row['projection_end'])
            ipc_period_start = pd.to_datetime(row['projection_start'])
            agg_window_start = ipc_period_start - relativedelta(months=AGGREGATION_MONTHS - 1)
            agg_window_end = ipc_period_end

            ipc_info = {
                'ipc_id': row['ipc_id'],
                'ipc_country': row['country'],
                'ipc_country_code': row['country_code'],
                'ipc_fips_code': row['fips_code'],
                'ipc_district': row['district'],
                'ipc_region': row['region'],
                'ipc_geographic_unit': row['geographic_unit_name'],
                'ipc_geographic_unit_full': row['geographic_unit_full_name'],
                'ipc_period_start': ipc_period_start,
                'ipc_period_end': ipc_period_end,
                'ipc_period_length_days': row['period_length_days'],
                'agg_window_start': agg_window_start,
                'agg_window_end': agg_window_end,
                'ipc_value': row['ipc_value'],
                'ipc_description': row['ipc_description'],
                'ipc_binary_crisis': row['ipc_binary_crisis'],
                'ipc_is_allowing_assistance': row['is_allowing_for_assistance'],
                'ipc_fewsnet_region': row['fewsnet_region'],
                'ipc_geographic_group': row['geographic_group'],
                'ipc_scenario': row['scenario'],
                'ipc_classification_scale': row['classification_scale'],
                'ipc_reporting_date': row['reporting_date'],
            }

            for word in words:
                if len(word) > 2:
                    key = (country, word)
                    word_lookup[key].append(ipc_info)

    combined_lookup = defaultdict(list)
    for k, v in word_lookup.items():
        combined_lookup[k].extend(v)
    for k, v in ipc_lookup.items():
        combined_lookup[k].extend(v)

    country_candidates = build_country_candidates(combined_lookup)

    # Read just this batch from the parquet file
    parquet_file = pq.ParquetFile(locations_path)
    batch_df = parquet_file.read_row_group(batch_num).to_pandas()

    results = []
    match_stats = {
        'GADM3_exact': 0, 'GADM3_fuzzy': 0,
        'GADM2_exact': 0, 'GADM2_fuzzy': 0,
        'GADM1_exact': 0, 'GADM1_fuzzy': 0,
        'Country_level': 0
    }

    locations_chunk = batch_df
    locations_chunk['date_extracted'] = pd.to_datetime(locations_chunk['date_extracted'])

    # Normalize geographic fields
    locations_chunk['gadm1_norm'] = locations_chunk['gadm1_name'].apply(normalize_text)
    locations_chunk['gadm2_norm'] = locations_chunk['gadm2_name'].apply(normalize_text)
    locations_chunk['gadm3_norm'] = locations_chunk['gadm3_name'].apply(normalize_text)

    # Group by country for efficiency
    for country_code, country_group in locations_chunk.groupby('african_country_code'):

        # PRIORITY 1: GADM3 Matching
        for gadm3_norm, gadm3_group in country_group.groupby('gadm3_norm'):
            if not gadm3_norm:
                continue

            key = (country_code, gadm3_norm)
            matched_ipc_list = None
            match_type = None

            if key in combined_lookup:
                matched_ipc_list = combined_lookup[key]
                match_type = 'GADM3_exact'
            else:
                matched_ipc_list, _ = find_fuzzy_match(gadm3_norm, country_candidates, country_code)
                if matched_ipc_list:
                    match_type = 'GADM3_fuzzy'

            if matched_ipc_list:
                for ipc_info in matched_ipc_list:
                    date_mask = (
                        (gadm3_group['date_extracted'] >= ipc_info['agg_window_start']) &
                        (gadm3_group['date_extracted'] <= ipc_info['agg_window_end'])
                    )
                    matched_ids = gadm3_group.loc[date_mask, 'GKGRECORDID'].unique()
                    n_matches = len(matched_ids)

                    if n_matches > 0:
                        batch_tuples = list(zip(
                            matched_ids,
                            [match_type] * n_matches,
                            [ipc_info['ipc_id']] * n_matches,
                            [ipc_info['ipc_country']] * n_matches,
                            [ipc_info['ipc_country_code']] * n_matches,
                            [ipc_info['ipc_fips_code']] * n_matches,
                            [ipc_info['ipc_district']] * n_matches,
                            [ipc_info['ipc_region']] * n_matches,
                            [ipc_info['ipc_geographic_unit']] * n_matches,
                            [ipc_info['ipc_geographic_unit_full']] * n_matches,
                            [str(ipc_info['ipc_period_start'])] * n_matches,
                            [str(ipc_info['ipc_period_end'])] * n_matches,
                            [ipc_info['ipc_period_length_days']] * n_matches,
                            [ipc_info['ipc_value']] * n_matches,
                            [ipc_info['ipc_description']] * n_matches,
                            [ipc_info['ipc_binary_crisis']] * n_matches,
                            [ipc_info['ipc_is_allowing_assistance']] * n_matches,
                            [ipc_info['ipc_fewsnet_region']] * n_matches,
                            [ipc_info['ipc_geographic_group']] * n_matches,
                            [ipc_info['ipc_scenario']] * n_matches,
                            [ipc_info['ipc_classification_scale']] * n_matches,
                            [str(ipc_info['ipc_reporting_date'])] * n_matches
                        ))
                        results.extend(batch_tuples)
                        match_stats[match_type] += n_matches

        # PRIORITY 2: GADM2 Matching
        for gadm2_norm, gadm2_group in country_group.groupby('gadm2_norm'):
            if not gadm2_norm:
                continue

            key = (country_code, gadm2_norm)
            matched_ipc_list = None
            match_type = None

            if key in combined_lookup:
                matched_ipc_list = combined_lookup[key]
                match_type = 'GADM2_exact'
            else:
                matched_ipc_list, _ = find_fuzzy_match(gadm2_norm, country_candidates, country_code)
                if matched_ipc_list:
                    match_type = 'GADM2_fuzzy'

            if matched_ipc_list:
                for ipc_info in matched_ipc_list:
                    date_mask = (
                        (gadm2_group['date_extracted'] >= ipc_info['agg_window_start']) &
                        (gadm2_group['date_extracted'] <= ipc_info['agg_window_end'])
                    )
                    matched_ids = gadm2_group.loc[date_mask, 'GKGRECORDID'].unique()
                    n_matches = len(matched_ids)

                    if n_matches > 0:
                        batch_tuples = list(zip(
                            matched_ids,
                            [match_type] * n_matches,
                            [ipc_info['ipc_id']] * n_matches,
                            [ipc_info['ipc_country']] * n_matches,
                            [ipc_info['ipc_country_code']] * n_matches,
                            [ipc_info['ipc_fips_code']] * n_matches,
                            [ipc_info['ipc_district']] * n_matches,
                            [ipc_info['ipc_region']] * n_matches,
                            [ipc_info['ipc_geographic_unit']] * n_matches,
                            [ipc_info['ipc_geographic_unit_full']] * n_matches,
                            [str(ipc_info['ipc_period_start'])] * n_matches,
                            [str(ipc_info['ipc_period_end'])] * n_matches,
                            [ipc_info['ipc_period_length_days']] * n_matches,
                            [ipc_info['ipc_value']] * n_matches,
                            [ipc_info['ipc_description']] * n_matches,
                            [ipc_info['ipc_binary_crisis']] * n_matches,
                            [ipc_info['ipc_is_allowing_assistance']] * n_matches,
                            [ipc_info['ipc_fewsnet_region']] * n_matches,
                            [ipc_info['ipc_geographic_group']] * n_matches,
                            [ipc_info['ipc_scenario']] * n_matches,
                            [ipc_info['ipc_classification_scale']] * n_matches,
                            [str(ipc_info['ipc_reporting_date'])] * n_matches
                        ))
                        results.extend(batch_tuples)
                        match_stats[match_type] += n_matches

        # PRIORITY 3: GADM1 Matching
        for gadm1_norm, gadm1_group in country_group.groupby('gadm1_norm'):
            if not gadm1_norm:
                continue

            key = (country_code, gadm1_norm)
            matched_ipc_list = None
            match_type = None

            if key in combined_lookup:
                matched_ipc_list = combined_lookup[key]
                match_type = 'GADM1_exact'
            else:
                matched_ipc_list, _ = find_fuzzy_match(gadm1_norm, country_candidates, country_code)
                if matched_ipc_list:
                    match_type = 'GADM1_fuzzy'

            if matched_ipc_list:
                for ipc_info in matched_ipc_list:
                    date_mask = (
                        (gadm1_group['date_extracted'] >= ipc_info['agg_window_start']) &
                        (gadm1_group['date_extracted'] <= ipc_info['agg_window_end'])
                    )
                    matched_ids = gadm1_group.loc[date_mask, 'GKGRECORDID'].unique()
                    n_matches = len(matched_ids)

                    if n_matches > 0:
                        batch_tuples = list(zip(
                            matched_ids,
                            [match_type] * n_matches,
                            [ipc_info['ipc_id']] * n_matches,
                            [ipc_info['ipc_country']] * n_matches,
                            [ipc_info['ipc_country_code']] * n_matches,
                            [ipc_info['ipc_fips_code']] * n_matches,
                            [ipc_info['ipc_district']] * n_matches,
                            [ipc_info['ipc_region']] * n_matches,
                            [ipc_info['ipc_geographic_unit']] * n_matches,
                            [ipc_info['ipc_geographic_unit_full']] * n_matches,
                            [str(ipc_info['ipc_period_start'])] * n_matches,
                            [str(ipc_info['ipc_period_end'])] * n_matches,
                            [ipc_info['ipc_period_length_days']] * n_matches,
                            [ipc_info['ipc_value']] * n_matches,
                            [ipc_info['ipc_description']] * n_matches,
                            [ipc_info['ipc_binary_crisis']] * n_matches,
                            [ipc_info['ipc_is_allowing_assistance']] * n_matches,
                            [ipc_info['ipc_fewsnet_region']] * n_matches,
                            [ipc_info['ipc_geographic_group']] * n_matches,
                            [ipc_info['ipc_scenario']] * n_matches,
                            [ipc_info['ipc_classification_scale']] * n_matches,
                            [str(ipc_info['ipc_reporting_date'])] * n_matches
                        ))
                        results.extend(batch_tuples)
                        match_stats[match_type] += n_matches

        # PRIORITY 4: Country-level matching
        if country_code in COUNTRY_LEVEL_ONLY:
            for (fips, district), ipc_list in combined_lookup.items():
                if fips == country_code:
                    for ipc_info in ipc_list:
                        date_mask = (
                            (country_group['date_extracted'] >= ipc_info['agg_window_start']) &
                            (country_group['date_extracted'] <= ipc_info['agg_window_end'])
                        )
                        matched_ids = country_group.loc[date_mask, 'GKGRECORDID'].unique()
                        n_matches = len(matched_ids)

                        if n_matches > 0:
                            batch_tuples = list(zip(
                                matched_ids,
                                ['Country_level'] * n_matches,
                                [ipc_info['ipc_id']] * n_matches,
                                [ipc_info['ipc_country']] * n_matches,
                                [ipc_info['ipc_country_code']] * n_matches,
                                [ipc_info['ipc_fips_code']] * n_matches,
                                [ipc_info['ipc_district']] * n_matches,
                                [ipc_info['ipc_region']] * n_matches,
                                [ipc_info['ipc_geographic_unit']] * n_matches,
                                [ipc_info['ipc_geographic_unit_full']] * n_matches,
                                [str(ipc_info['ipc_period_start'])] * n_matches,
                                [str(ipc_info['ipc_period_end'])] * n_matches,
                                [ipc_info['ipc_period_length_days']] * n_matches,
                                [ipc_info['ipc_value']] * n_matches,
                                [ipc_info['ipc_description']] * n_matches,
                                [ipc_info['ipc_binary_crisis']] * n_matches,
                                [ipc_info['ipc_is_allowing_assistance']] * n_matches,
                                [ipc_info['ipc_fewsnet_region']] * n_matches,
                                [ipc_info['ipc_geographic_group']] * n_matches,
                                [ipc_info['ipc_scenario']] * n_matches,
                                [ipc_info['ipc_classification_scale']] * n_matches,
                                [str(ipc_info['ipc_reporting_date'])] * n_matches
                            ))
                            results.extend(batch_tuples)
                            match_stats['Country_level'] += n_matches

    return batch_num, results, match_stats


def main():
    print("=" * 80)
    print("GDELT Articles - DISTRICT LEVEL Alignment")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print(f"Fuzzy matching threshold: {FUZZY_THRESHOLD}")

    # Load IPC reference (district-level)
    print("\n1. Loading IPC district reference...")
    ipc_ref = pd.read_parquet(IPC_REF_FILE)
    print(f"   Loaded {len(ipc_ref):,} IPC periods")
    print(f"   Unique districts: {ipc_ref['district'].nunique():,}")
    print(f"   Unique full_names: {ipc_ref['geographic_unit_full_name'].nunique():,}")

    # Build IPC lookup - KEY CHANGE: Index by district_normalized
    print("\n2. Building IPC district lookup dictionary...")
    ipc_lookup = defaultdict(list)

    for idx, row in ipc_ref.iterrows():
        if pd.notna(row['fips_code']) and pd.notna(row['district_normalized']):
            # Calculate 4-month aggregation window (3 months before + assessment month)
            ipc_period_end = pd.to_datetime(row['projection_end'])
            ipc_period_start = pd.to_datetime(row['projection_start'])
            # Aggregation window starts 3 months before the assessment month
            agg_window_start = ipc_period_start - relativedelta(months=AGGREGATION_MONTHS - 1)
            agg_window_end = ipc_period_end  # End at assessment month end

            # Primary key: (country_fips, district_normalized)
            key = (row['fips_code'], row['district_normalized'])
            ipc_lookup[key].append({
                'ipc_id': row['ipc_id'],
                'ipc_country': row['country'],
                'ipc_country_code': row['country_code'],
                'ipc_fips_code': row['fips_code'],
                'ipc_district': row['district'],  # Extracted district
                'ipc_region': row['region'],  # Extracted region
                'ipc_geographic_unit': row['geographic_unit_name'],  # Original (may be LHZ)
                'ipc_geographic_unit_full': row['geographic_unit_full_name'],  # FULL identifier
                'ipc_period_start': ipc_period_start,  # Original assessment start
                'ipc_period_end': ipc_period_end,  # Original assessment end
                'ipc_period_length_days': row['period_length_days'],
                # 4-month aggregation window for article filtering
                'agg_window_start': agg_window_start,
                'agg_window_end': agg_window_end,
                'ipc_value': row['ipc_value'],
                'ipc_description': row['ipc_description'],
                'ipc_binary_crisis': row['ipc_binary_crisis'],
                'ipc_is_allowing_assistance': row['is_allowing_for_assistance'],
                'ipc_fewsnet_region': row['fewsnet_region'],
                'ipc_geographic_group': row['geographic_group'],
                'ipc_scenario': row['scenario'],
                'ipc_classification_scale': row['classification_scale'],
                'ipc_reporting_date': row['reporting_date'],
            })

    print(f"   Created lookup with {len(ipc_lookup):,} unique (country, district) combinations", flush=True)

    # Build word-based lookup from full_name_normalized (pre-built for efficiency)
    word_lookup = build_word_lookup(ipc_ref)

    # Combine word_lookup with ipc_lookup (ipc_lookup takes precedence)
    combined_lookup = defaultdict(list)
    for k, v in word_lookup.items():
        combined_lookup[k].extend(v)
    for k, v in ipc_lookup.items():
        combined_lookup[k].extend(v)
    print(f"   Combined lookup: {len(combined_lookup):,} keys", flush=True)

    # Pre-group candidates by country for fast fuzzy matching
    country_candidates = build_country_candidates(combined_lookup)
    print(f"   Pre-grouped by country: {len(country_candidates)} countries", flush=True)

    # Load locations and create GKGRECORDID -> IPC mapping
    # KEY CHANGE: Match via GADM3 → GADM2 → district (NOT LHZ first)
    print("\n3. Processing locations to build GKGRECORDID -> IPC mapping...", flush=True)
    print("   Matching priority: GADM3 -> GADM2 -> District name (skipping LHZ)", flush=True)
    print(f"   Using {NUM_WORKERS} parallel workers", flush=True)

    parquet_file = pq.ParquetFile(LOCATIONS_FILE)

    # Use SQLite for streaming - eliminates memory issues
    # Use unique timestamped filename to avoid conflicts from old processes
    timestamp_str = datetime.now().strftime('%Y%m%d_%H%M%S')
    sqlite_file = DISTRICT_DATA_DIR / f'temp_gkg_ipc_lookup_{timestamp_str}.db'
    # Clean up old temp files (ignore errors if locked)
    for old_file in DISTRICT_DATA_DIR.glob('temp_gkg_ipc_lookup_*.db'):
        try:
            old_file.unlink()
        except:
            pass  # Ignore locked files

    conn = sqlite3.connect(str(sqlite_file))
    cursor = conn.cursor()

    # SQLite optimizations for bulk inserts
    cursor.execute("PRAGMA journal_mode=WAL")
    cursor.execute("PRAGMA synchronous=OFF")
    cursor.execute("PRAGMA cache_size=100000")
    cursor.execute("PRAGMA temp_store=MEMORY")
    conn.commit()

    # Create table for GKG-IPC mappings
    cursor.execute('''
        CREATE TABLE gkg_ipc (
            GKGRECORDID TEXT,
            match_level TEXT,
            ipc_id TEXT,
            ipc_country TEXT,
            ipc_country_code TEXT,
            ipc_fips_code TEXT,
            ipc_district TEXT,
            ipc_region TEXT,
            ipc_geographic_unit TEXT,
            ipc_geographic_unit_full TEXT,
            ipc_period_start TEXT,
            ipc_period_end TEXT,
            ipc_period_length_days REAL,
            ipc_value REAL,
            ipc_description TEXT,
            ipc_binary_crisis INTEGER,
            ipc_is_allowing_assistance INTEGER,
            ipc_fewsnet_region TEXT,
            ipc_geographic_group TEXT,
            ipc_scenario TEXT,
            ipc_classification_scale TEXT,
            ipc_reporting_date TEXT
        )
    ''')
    conn.commit()

    # Get number of row groups in parquet file
    num_row_groups = parquet_file.metadata.num_row_groups
    print(f"   Total row groups to process: {num_row_groups}", flush=True)

    # Use SEQUENTIAL processing to avoid memory issues with multiprocessing
    print(f"\n   Processing {num_row_groups} batches sequentially (SQLite streaming)...", flush=True)
    print("   (Sequential mode avoids memory duplication in subprocess pickling)", flush=True)

    # Pre-compute unique GADM names from locations file for fuzzy cache
    print("   Scanning locations for unique GADM names (for fuzzy cache)...", flush=True)
    all_gadm_names_by_country = defaultdict(set)
    for rg_idx in range(num_row_groups):
        sample_batch = parquet_file.read_row_group(rg_idx, columns=['african_country_code', 'gadm1_name', 'gadm2_name', 'gadm3_name']).to_pandas()
        for _, row in sample_batch.groupby('african_country_code').first().iterrows():
            pass  # Just to get unique countries
        for country_code in sample_batch['african_country_code'].unique():
            country_data = sample_batch[sample_batch['african_country_code'] == country_code]
            all_gadm_names_by_country[country_code].update(country_data['gadm1_name'].dropna().apply(normalize_text).unique())
            all_gadm_names_by_country[country_code].update(country_data['gadm2_name'].dropna().apply(normalize_text).unique())
            all_gadm_names_by_country[country_code].update(country_data['gadm3_name'].dropna().apply(normalize_text).unique())
        del sample_batch
        if (rg_idx + 1) % 20 == 0:
            print(f"      Scanned {rg_idx + 1}/{num_row_groups} row groups...", flush=True)
    print(f"   Found {sum(len(v) for v in all_gadm_names_by_country.values()):,} unique GADM names across {len(all_gadm_names_by_country)} countries", flush=True)

    # Pre-compute fuzzy matches
    fuzzy_cache = precompute_fuzzy_matches(country_candidates, all_gadm_names_by_country)
    del all_gadm_names_by_country  # Free memory
    gc.collect()

    total_match_stats = {
        'GADM3_exact': 0, 'GADM3_fuzzy': 0,
        'GADM2_exact': 0, 'GADM2_fuzzy': 0,
        'GADM1_exact': 0, 'GADM1_fuzzy': 0,
        'Country_level': 0
    }

    total_rows = 0
    start_time = datetime.now()

    # Process batches sequentially - write directly to SQLite
    for batch_num in range(num_row_groups):
        batch_start = datetime.now()

        # Read this batch from parquet
        batch_df = parquet_file.read_row_group(batch_num).to_pandas()
        batch_df['date_extracted'] = pd.to_datetime(batch_df['date_extracted'])

        # Normalize geographic fields
        batch_df['gadm1_norm'] = batch_df['gadm1_name'].apply(normalize_text)
        batch_df['gadm2_norm'] = batch_df['gadm2_name'].apply(normalize_text)
        batch_df['gadm3_norm'] = batch_df['gadm3_name'].apply(normalize_text)

        batch_matches = 0
        batch_stats = {'GADM3_exact': 0, 'GADM3_fuzzy': 0, 'GADM2_exact': 0, 'GADM2_fuzzy': 0,
                       'GADM1_exact': 0, 'GADM1_fuzzy': 0, 'Country_level': 0}

        # Buffer for batched SQLite inserts
        insert_buffer = []

        # Track matched GKGRECORDIDs to avoid redundant lower-priority matching
        matched_gkgids = set()

        # Process by country for efficiency
        for country_code, country_group in batch_df.groupby('african_country_code'):

            # PRIORITY 1: GADM3 Matching
            for gadm3_norm, gadm3_group in country_group.groupby('gadm3_norm'):
                if not gadm3_norm:
                    continue

                key = (country_code, gadm3_norm)
                matched_ipc_list = None
                match_type = None

                if key in combined_lookup:
                    matched_ipc_list = combined_lookup[key]
                    match_type = 'GADM3_exact'
                else:
                    # Use cached fuzzy match
                    matched_ipc_list, _ = get_fuzzy_match_cached(gadm3_norm, country_code, fuzzy_cache, country_candidates)
                    if matched_ipc_list:
                        match_type = 'GADM3_fuzzy'

                if matched_ipc_list:
                    for ipc_info in matched_ipc_list:
                        date_mask = (
                            (gadm3_group['date_extracted'] >= ipc_info['agg_window_start']) &
                            (gadm3_group['date_extracted'] <= ipc_info['agg_window_end'])
                        )
                        matched_rows = gadm3_group.loc[date_mask]
                        if len(matched_rows) > 0:
                            # VECTORIZED: Use .values arrays instead of iterrows()
                            gkgids = matched_rows['GKGRECORDID'].values
                            n_rows = len(gkgids)
                            tuples = list(zip(
                                gkgids,
                                [match_type] * n_rows,
                                [ipc_info['ipc_id']] * n_rows,
                                [ipc_info['ipc_country']] * n_rows,
                                [ipc_info['ipc_country_code']] * n_rows,
                                [ipc_info['ipc_fips_code']] * n_rows,
                                [ipc_info['ipc_district']] * n_rows,
                                [ipc_info['ipc_region']] * n_rows,
                                [ipc_info['ipc_geographic_unit']] * n_rows,
                                [ipc_info['ipc_geographic_unit_full']] * n_rows,
                                [str(ipc_info['ipc_period_start'])] * n_rows,
                                [str(ipc_info['ipc_period_end'])] * n_rows,
                                [ipc_info['ipc_period_length_days']] * n_rows,
                                [ipc_info['ipc_value']] * n_rows,
                                [ipc_info['ipc_description']] * n_rows,
                                [ipc_info['ipc_binary_crisis']] * n_rows,
                                [ipc_info['ipc_is_allowing_assistance']] * n_rows,
                                [ipc_info['ipc_fewsnet_region']] * n_rows,
                                [ipc_info['ipc_geographic_group']] * n_rows,
                                [ipc_info['ipc_scenario']] * n_rows,
                                [ipc_info['ipc_classification_scale']] * n_rows,
                                [str(ipc_info['ipc_reporting_date'])] * n_rows
                            ))
                            insert_buffer.extend(tuples)
                            batch_matches += n_rows
                            batch_stats[match_type] += n_rows
                            matched_gkgids.update(gkgids)

                            # Flush buffer when it gets large
                            if len(insert_buffer) >= SQLITE_BATCH_SIZE:
                                cursor.executemany('INSERT INTO gkg_ipc VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', insert_buffer)
                                insert_buffer.clear()

            # PRIORITY 2: GADM2 Matching (for unmatched)
            for gadm2_norm, gadm2_group in country_group.groupby('gadm2_norm'):
                if not gadm2_norm:
                    continue

                # Skip already matched GKGRECORDIDs
                gadm2_group = gadm2_group[~gadm2_group['GKGRECORDID'].isin(matched_gkgids)]
                if len(gadm2_group) == 0:
                    continue

                key = (country_code, gadm2_norm)
                matched_ipc_list = None
                match_type = None

                if key in combined_lookup:
                    matched_ipc_list = combined_lookup[key]
                    match_type = 'GADM2_exact'
                else:
                    # Use cached fuzzy match
                    matched_ipc_list, _ = get_fuzzy_match_cached(gadm2_norm, country_code, fuzzy_cache, country_candidates)
                    if matched_ipc_list:
                        match_type = 'GADM2_fuzzy'

                if matched_ipc_list:
                    for ipc_info in matched_ipc_list:
                        date_mask = (
                            (gadm2_group['date_extracted'] >= ipc_info['agg_window_start']) &
                            (gadm2_group['date_extracted'] <= ipc_info['agg_window_end'])
                        )
                        matched_rows = gadm2_group.loc[date_mask]
                        if len(matched_rows) > 0:
                            # VECTORIZED: Use .values arrays instead of iterrows()
                            gkgids = matched_rows['GKGRECORDID'].values
                            n_rows = len(gkgids)
                            tuples = list(zip(
                                gkgids,
                                [match_type] * n_rows,
                                [ipc_info['ipc_id']] * n_rows,
                                [ipc_info['ipc_country']] * n_rows,
                                [ipc_info['ipc_country_code']] * n_rows,
                                [ipc_info['ipc_fips_code']] * n_rows,
                                [ipc_info['ipc_district']] * n_rows,
                                [ipc_info['ipc_region']] * n_rows,
                                [ipc_info['ipc_geographic_unit']] * n_rows,
                                [ipc_info['ipc_geographic_unit_full']] * n_rows,
                                [str(ipc_info['ipc_period_start'])] * n_rows,
                                [str(ipc_info['ipc_period_end'])] * n_rows,
                                [ipc_info['ipc_period_length_days']] * n_rows,
                                [ipc_info['ipc_value']] * n_rows,
                                [ipc_info['ipc_description']] * n_rows,
                                [ipc_info['ipc_binary_crisis']] * n_rows,
                                [ipc_info['ipc_is_allowing_assistance']] * n_rows,
                                [ipc_info['ipc_fewsnet_region']] * n_rows,
                                [ipc_info['ipc_geographic_group']] * n_rows,
                                [ipc_info['ipc_scenario']] * n_rows,
                                [ipc_info['ipc_classification_scale']] * n_rows,
                                [str(ipc_info['ipc_reporting_date'])] * n_rows
                            ))
                            insert_buffer.extend(tuples)
                            batch_matches += n_rows
                            batch_stats[match_type] += n_rows
                            matched_gkgids.update(gkgids)

                            # Flush buffer when it gets large
                            if len(insert_buffer) >= SQLITE_BATCH_SIZE:
                                cursor.executemany('INSERT INTO gkg_ipc VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', insert_buffer)
                                insert_buffer.clear()

            # PRIORITY 3: GADM1 Matching
            for gadm1_norm, gadm1_group in country_group.groupby('gadm1_norm'):
                if not gadm1_norm:
                    continue

                # Skip already matched GKGRECORDIDs
                gadm1_group = gadm1_group[~gadm1_group['GKGRECORDID'].isin(matched_gkgids)]
                if len(gadm1_group) == 0:
                    continue

                key = (country_code, gadm1_norm)
                matched_ipc_list = None
                match_type = None

                if key in combined_lookup:
                    matched_ipc_list = combined_lookup[key]
                    match_type = 'GADM1_exact'
                else:
                    # Use cached fuzzy match
                    matched_ipc_list, _ = get_fuzzy_match_cached(gadm1_norm, country_code, fuzzy_cache, country_candidates)
                    if matched_ipc_list:
                        match_type = 'GADM1_fuzzy'

                if matched_ipc_list:
                    for ipc_info in matched_ipc_list:
                        date_mask = (
                            (gadm1_group['date_extracted'] >= ipc_info['agg_window_start']) &
                            (gadm1_group['date_extracted'] <= ipc_info['agg_window_end'])
                        )
                        matched_rows = gadm1_group.loc[date_mask]
                        if len(matched_rows) > 0:
                            # VECTORIZED: Use .values arrays instead of iterrows()
                            gkgids = matched_rows['GKGRECORDID'].values
                            n_rows = len(gkgids)
                            tuples = list(zip(
                                gkgids,
                                [match_type] * n_rows,
                                [ipc_info['ipc_id']] * n_rows,
                                [ipc_info['ipc_country']] * n_rows,
                                [ipc_info['ipc_country_code']] * n_rows,
                                [ipc_info['ipc_fips_code']] * n_rows,
                                [ipc_info['ipc_district']] * n_rows,
                                [ipc_info['ipc_region']] * n_rows,
                                [ipc_info['ipc_geographic_unit']] * n_rows,
                                [ipc_info['ipc_geographic_unit_full']] * n_rows,
                                [str(ipc_info['ipc_period_start'])] * n_rows,
                                [str(ipc_info['ipc_period_end'])] * n_rows,
                                [ipc_info['ipc_period_length_days']] * n_rows,
                                [ipc_info['ipc_value']] * n_rows,
                                [ipc_info['ipc_description']] * n_rows,
                                [ipc_info['ipc_binary_crisis']] * n_rows,
                                [ipc_info['ipc_is_allowing_assistance']] * n_rows,
                                [ipc_info['ipc_fewsnet_region']] * n_rows,
                                [ipc_info['ipc_geographic_group']] * n_rows,
                                [ipc_info['ipc_scenario']] * n_rows,
                                [ipc_info['ipc_classification_scale']] * n_rows,
                                [str(ipc_info['ipc_reporting_date'])] * n_rows
                            ))
                            insert_buffer.extend(tuples)
                            batch_matches += n_rows
                            batch_stats[match_type] += n_rows
                            matched_gkgids.update(gkgids)

                            # Flush buffer when it gets large
                            if len(insert_buffer) >= SQLITE_BATCH_SIZE:
                                cursor.executemany('INSERT INTO gkg_ipc VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', insert_buffer)
                                insert_buffer.clear()

            # PRIORITY 4: Country-level matching
            if country_code in COUNTRY_LEVEL_ONLY:
                # Skip already matched GKGRECORDIDs
                unmatched_group = country_group[~country_group['GKGRECORDID'].isin(matched_gkgids)]
                if len(unmatched_group) == 0:
                    continue

                for (fips, district), ipc_list in combined_lookup.items():
                    if fips == country_code:
                        for ipc_info in ipc_list:
                            date_mask = (
                                (unmatched_group['date_extracted'] >= ipc_info['agg_window_start']) &
                                (unmatched_group['date_extracted'] <= ipc_info['agg_window_end'])
                            )
                            matched_rows = unmatched_group.loc[date_mask]
                            if len(matched_rows) > 0:
                                # VECTORIZED: Use .values arrays instead of iterrows()
                                gkgids = matched_rows['GKGRECORDID'].values
                                n_rows = len(gkgids)
                                tuples = list(zip(
                                    gkgids,
                                    ['Country_level'] * n_rows,
                                    [ipc_info['ipc_id']] * n_rows,
                                    [ipc_info['ipc_country']] * n_rows,
                                    [ipc_info['ipc_country_code']] * n_rows,
                                    [ipc_info['ipc_fips_code']] * n_rows,
                                    [ipc_info['ipc_district']] * n_rows,
                                    [ipc_info['ipc_region']] * n_rows,
                                    [ipc_info['ipc_geographic_unit']] * n_rows,
                                    [ipc_info['ipc_geographic_unit_full']] * n_rows,
                                    [str(ipc_info['ipc_period_start'])] * n_rows,
                                    [str(ipc_info['ipc_period_end'])] * n_rows,
                                    [ipc_info['ipc_period_length_days']] * n_rows,
                                    [ipc_info['ipc_value']] * n_rows,
                                    [ipc_info['ipc_description']] * n_rows,
                                    [ipc_info['ipc_binary_crisis']] * n_rows,
                                    [ipc_info['ipc_is_allowing_assistance']] * n_rows,
                                    [ipc_info['ipc_fewsnet_region']] * n_rows,
                                    [ipc_info['ipc_geographic_group']] * n_rows,
                                    [ipc_info['ipc_scenario']] * n_rows,
                                    [ipc_info['ipc_classification_scale']] * n_rows,
                                    [str(ipc_info['ipc_reporting_date'])] * n_rows
                                ))
                                insert_buffer.extend(tuples)
                                batch_matches += n_rows
                                batch_stats['Country_level'] += n_rows

                                # Flush buffer when it gets large
                                if len(insert_buffer) >= SQLITE_BATCH_SIZE:
                                    cursor.executemany('INSERT INTO gkg_ipc VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', insert_buffer)
                                    insert_buffer.clear()

        # Flush any remaining buffer and commit after each batch
        if insert_buffer:
            cursor.executemany('INSERT INTO gkg_ipc VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)', insert_buffer)
            insert_buffer.clear()
        conn.commit()
        total_rows += batch_matches

        # Update aggregate stats
        for key, count in batch_stats.items():
            total_match_stats[key] += count

        # Progress reporting
        batch_time = (datetime.now() - batch_start).total_seconds()
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = (batch_num + 1) / elapsed if elapsed > 0 else 0
        eta = (num_row_groups - batch_num - 1) / rate if rate > 0 else 0
        print(f"      Batch {batch_num + 1}/{num_row_groups}: {batch_matches:,} matches ({batch_time:.1f}s) | Total: {total_rows:,} | ETA: {eta/60:.1f}min", flush=True)

        # Free memory
        del batch_df, matched_gkgids
        gc.collect()

    # Free fuzzy cache before deduplication
    del fuzzy_cache
    gc.collect()

    # SKIP index on large table - causes OOM on 181M+ rows
    # Index will be created on deduplicated table instead (much smaller)
    print("   Skipping index on main table (will index after dedup)...", flush=True)

    match_stats = total_match_stats
    gc.collect()

    # Print match statistics
    print(f"\n   Match statistics:")
    total_matches = sum(match_stats.values())
    for method, count in sorted(match_stats.items(), key=lambda x: x[1], reverse=True):
        pct = (count / total_matches * 100) if total_matches > 0 else 0
        print(f"     {method}: {count:,} ({pct:.1f}%)")

    # Check if we have data
    cursor.execute('SELECT COUNT(*) FROM gkg_ipc')
    total_rows = cursor.fetchone()[0]
    if total_rows == 0:
        print("\n   WARNING: No matched locations found!")
        conn.close()
        return

    # Deduplicate in SQLite
    print(f"\n4. Deduplicating {total_rows:,} records in SQLite...")

    # Checkpoint WAL to reduce memory footprint before heavy operation
    print("   Checkpointing WAL to free memory...", flush=True)
    cursor.execute('PRAGMA wal_checkpoint(TRUNCATE)')
    gc.collect()

    # Create deduplicated table (no index on source = slower but no OOM)
    print("   Creating deduplicated table (this may take a while without index)...", flush=True)
    cursor.execute('''
        CREATE TABLE gkg_ipc_dedup AS
        SELECT * FROM gkg_ipc
        GROUP BY GKGRECORDID, ipc_id
    ''')
    conn.commit()
    gc.collect()

    # Now create index on the SMALLER deduplicated table
    print("   Creating index on deduplicated table...", flush=True)
    cursor.execute('CREATE INDEX idx_dedup_gkgrecordid ON gkg_ipc_dedup(GKGRECORDID)')
    conn.commit()

    cursor.execute('SELECT COUNT(*) FROM gkg_ipc_dedup')
    dedup_rows = cursor.fetchone()[0]
    print(f"   Deduplicated: {total_rows:,} -> {dedup_rows:,} unique matches")

    # Load articles and aggregate
    print("\n5. Loading and aggregating articles...")
    articles_sample = pd.read_csv(ARTICLES_FILE, nrows=1)
    all_article_cols = articles_sample.columns.tolist()

    # Define numeric columns for aggregation
    numeric_cols = [col for col in all_article_cols if col not in [
        'GKGRECORDID', 'DATE', 'SourceCollectionIdentifier', 'SourceCommonName',
        'DocumentIdentifier', 'Counts', 'V2Counts', 'Themes', 'V2Themes',
        'Dates', 'Persons', 'V2Persons', 'Organizations', 'V2Organizations',
        'V2AllNames', 'Locations', 'V2Locations', 'ADM1', 'ADM2', 'ADM3',
        'V2Quotations', 'V2Amounts', 'V2RelatedImages', 'V2DateTimeFields',
        'V2Extras', 'V2ExtendedField', 'V2TranslationInfo', 'OutletType',
        'all_countries_mentioned', 'all_african_countries', 'date_extracted',
        'V2Tone'
    ]]

    aggregated_data = []

    for chunk_num, articles_chunk in enumerate(pd.read_csv(ARTICLES_FILE, chunksize=CHUNK_SIZE)):
        print(f"   Articles chunk {chunk_num + 1}: {len(articles_chunk):,} articles...", flush=True)

        article_ids = list(articles_chunk['GKGRECORDID'].unique())

        # Query SQLite for matching GKGRECORDIDs
        placeholders = ','.join(['?'] * len(article_ids))
        query = f'SELECT * FROM gkg_ipc_dedup WHERE GKGRECORDID IN ({placeholders})'
        gkg_lookup_chunk = pd.read_sql_query(query, conn, params=article_ids)

        merged = articles_chunk.merge(gkg_lookup_chunk, on='GKGRECORDID', how='inner')
        print(f"      Matched: {len(merged):,} article-period pairs")

        del articles_chunk, gkg_lookup_chunk, article_ids

        if len(merged) == 0:
            continue

        # Group by IPC period (using full_name as identifier)
        group_cols = [
            'ipc_id', 'ipc_country', 'ipc_country_code', 'ipc_fips_code',
            'ipc_district', 'ipc_region',
            'ipc_geographic_unit', 'ipc_geographic_unit_full',
            'ipc_period_start', 'ipc_period_end', 'ipc_period_length_days',
            'ipc_value', 'ipc_description', 'ipc_binary_crisis',
            'ipc_is_allowing_assistance', 'ipc_fewsnet_region',
            'ipc_geographic_group', 'ipc_scenario', 'ipc_classification_scale',
            'ipc_reporting_date', 'match_level'
        ]

        # Aggregate
        agg_dict = {col: 'sum' for col in numeric_cols if col in merged.columns}
        agg_dict['GKGRECORDID'] = 'count'
        if 'SourceCommonName' in merged.columns:
            agg_dict['SourceCommonName'] = 'nunique'

        agg_result = merged.groupby(group_cols).agg(agg_dict).reset_index()
        agg_result = agg_result.rename(columns={
            'GKGRECORDID': 'article_count',
            'SourceCommonName': 'unique_sources'
        })

        aggregated_data.append(agg_result)

        del merged, agg_result
        gc.collect()

    # Combine and final aggregation
    print("\n6. Combining and final aggregation...")
    if not aggregated_data:
        print("   WARNING: No data to aggregate!")
        return

    final_df = pd.concat(aggregated_data, ignore_index=True)
    del aggregated_data
    gc.collect()

    # Final aggregation by IPC period
    group_cols = [
        'ipc_id', 'ipc_country', 'ipc_country_code', 'ipc_fips_code',
        'ipc_district', 'ipc_region',
        'ipc_geographic_unit', 'ipc_geographic_unit_full',
        'ipc_period_start', 'ipc_period_end', 'ipc_period_length_days',
        'ipc_value', 'ipc_description', 'ipc_binary_crisis',
        'ipc_is_allowing_assistance', 'ipc_fewsnet_region',
        'ipc_geographic_group', 'ipc_scenario', 'ipc_classification_scale',
        'ipc_reporting_date'
    ]

    # Sum numeric columns, keep first match_level
    numeric_to_sum = [c for c in final_df.columns if c not in group_cols and c != 'match_level']
    agg_dict = {col: 'sum' for col in numeric_to_sum if col in final_df.columns}
    agg_dict['match_level'] = 'first'

    final_agg = final_df.groupby(group_cols).agg(agg_dict).reset_index()

    # Summary
    print("\n" + "=" * 80)
    print("Aggregation Summary - DISTRICT LEVEL")
    print("=" * 80)
    print(f"\nTotal IPC period-aggregations: {len(final_agg):,}")
    print(f"Unique districts: {final_agg['ipc_district'].nunique():,}")
    print(f"Unique geographic_unit_full: {final_agg['ipc_geographic_unit_full'].nunique():,}")
    print(f"Date range: {final_agg['ipc_period_start'].min()} to {final_agg['ipc_period_end'].max()}")
    print(f"Countries: {final_agg['ipc_country'].nunique()}")

    print(f"\nMatch level distribution:")
    print(final_agg['match_level'].value_counts())

    print(f"\nRecords by country:")
    print(final_agg['ipc_country'].value_counts().head(10))

    # Save
    print(f"\n7. Saving to {OUTPUT_PARQUET}...")
    final_agg.to_parquet(OUTPUT_PARQUET, index=False)
    print("   [OK] Parquet saved")

    print(f"\n8. Saving to {OUTPUT_CSV}...")
    final_agg.to_csv(OUTPUT_CSV, index=False)
    print("   [OK] CSV saved")

    # Cleanup SQLite database
    print("\n   Cleaning up temporary SQLite database...")
    conn.close()
    if sqlite_file.exists():
        sqlite_file.unlink()
        print("   Cleaned up SQLite database")

    print("\n" + "=" * 80)
    print("Articles Aggregation Complete - DISTRICT LEVEL")
    print("=" * 80)
    print(f"End time: {datetime.now()}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n\nERROR: {e}")
        print("Attempting to clean up temporary SQLite database...")
        # Try to clean up SQLite file
        sqlite_file = DISTRICT_DATA_DIR / 'temp_gkg_ipc_lookup.db'
        if sqlite_file.exists():
            try:
                sqlite_file.unlink()
                print("Cleaned up SQLite database after error")
            except Exception:
                print("Could not clean up SQLite database")
        raise
