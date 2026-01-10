"""
GDELT Locations Aggregation - DISTRICT LEVEL CORRECTED
Aggregates GDELT location mentions to IPC assessment periods at DISTRICT level.

KEY CORRECTIONS:
1. Uses geographic_unit_full_name as the unique geographic identifier
2. Matches via GADM3 → GADM2 → district name (NOT LHZ first)
3. Each IPC observation is (geographic_unit_full_name, period)

TEMPORAL AGGREGATION (CRITICAL) - ALIGNED WITH 02_aggregate_articles.py:
4. Aggregates locations over 4-MONTH WINDOWS before each IPC assessment
   - IPC assessments occur in Feb, Jun, Oct
   - For Feb assessment: aggregate Nov, Dec, Jan, Feb locations
   - For Jun assessment: aggregate Mar, Apr, May, Jun locations
   - For Oct assessment: aggregate Jul, Aug, Sep, Oct locations

Author: Victor Collins Oppon
"""

import sys
from pathlib import Path
import pandas as pd
import numpy as np

# Add parent directory to path for config import
sys.path.append(str(Path(__file__).parent.parent))

# Import from config
from config import (
    BASE_DIR,
    STAGE1_DATA_DIR,
    STAGE1_RESULTS_DIR,
    STAGE2_FEATURES_DIR,
    STAGE2_MODELS_DIR,
    FIGURES_DIR,
    STAGE1_CONFIG,
    GDELT_LOCATIONS_FILE
)
from datetime import datetime
from dateutil.relativedelta import relativedelta
from fuzzywuzzy import fuzz
from collections import defaultdict
import unicodedata
import gc

# 4-month aggregation window (locations from 3 months before to assessment month)
# ALIGNED WITH 02_aggregate_articles.py
AGGREGATION_MONTHS = 4

# Paths
# BASE_DIR imported from config

# Input files (original locations)
LOCATIONS_FILE = GDELT_LOCATIONS_FILE  # Self-contained: FINAL_PIPELINE/DATA/gdelt/

# District pipeline I/O (district_level subfolder)
DISTRICT_DATA_DIR = BASE_DIR / 'data' / 'district_level'
IPC_REF_FILE = DISTRICT_DATA_DIR / 'ipc_reference.parquet'
OUTPUT_PARQUET = DISTRICT_DATA_DIR / 'locations_aggregated.parquet'
OUTPUT_CSV = DISTRICT_DATA_DIR / 'locations_aggregated.csv'

CHUNK_SIZE = 1000000
FUZZY_THRESHOLD = 80  # Slightly lower for district name variations

# Countries with ONLY national-level IPC data (very limited district info)
# CG (DRC) added due to poor GADM alignment (only 18.6% coverage) - ALIGNED WITH 02_aggregate_articles.py
COUNTRY_LEVEL_ONLY = {'AO', 'CG', 'CT', 'LT', 'MR', 'RW', 'TO'}


def normalize_text(text):
    """Normalize text for matching with accent removal - ALIGNED WITH 02_aggregate_articles.py"""
    if pd.isna(text):
        return ''
    text = str(text).lower().strip()
    # Remove accents (e.g., Kasaï -> kasai, Équateur -> equateur)
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if not unicodedata.combining(c))
    text = ''.join(c if c.isalnum() or c.isspace() else ' ' for c in text)
    text = ' '.join(text.split())
    return text


def find_fuzzy_match(loc_name_norm, ipc_candidates, country_code):
    """
    Find best fuzzy match for location name among IPC district candidates.
    Returns: (ipc_info_list, match_score) or (None, 0)
    """
    if not loc_name_norm or not ipc_candidates:
        return None, 0

    best_match = None
    best_score = 0

    for (fips, district_norm), ipc_list in ipc_candidates.items():
        if fips != country_code:
            continue

        score = fuzz.ratio(loc_name_norm, district_norm)

        if score >= FUZZY_THRESHOLD and score > best_score:
            best_score = score
            best_match = ipc_list

    return best_match, best_score


def build_word_lookup(ipc_ref):
    """
    Build word-based lookup from full_name_normalized at startup.
    ALIGNED WITH 02_aggregate_articles.py

    This enables matching for livelihood zone countries (Zimbabwe, Burundi, Kenya, etc.)
    where IPC uses names like "agrofisheries binga matabeleland north zimbabwe"
    that contain embedded GADM admin names.
    """
    print("   Building word-based lookup from full_name_normalized...", flush=True)
    word_lookup = defaultdict(list)

    for idx, row in ipc_ref.iterrows():
        if pd.notna(row['fips_code']) and pd.notna(row.get('full_name_normalized')):
            country = row['fips_code']
            full_name_norm = normalize_text(row['full_name_normalized'])
            words = full_name_norm.split()

            # Calculate 4-month aggregation window (3 months before + assessment month)
            # ALIGNED WITH 02_aggregate_articles.py
            ipc_period_end = pd.to_datetime(row['projection_end'])
            ipc_period_start = pd.to_datetime(row['projection_start'])
            agg_window_start = ipc_period_start - relativedelta(months=AGGREGATION_MONTHS - 1)
            agg_window_end = ipc_period_end  # End at assessment month end

            # Build IPC info dict (same structure as main lookup)
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
                'agg_window_start': agg_window_start,  # 4-month window start
                'agg_window_end': agg_window_end,      # 4-month window end
                'ipc_period_length_days': row['period_length_days'],
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


def main():
    print("=" * 80)
    print("GDELT Locations - DISTRICT LEVEL Alignment")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print(f"Fuzzy matching threshold: {FUZZY_THRESHOLD}")

    # Load IPC reference
    print("\n1. Loading IPC district reference data...")
    ipc_ref = pd.read_parquet(IPC_REF_FILE)
    print(f"   Loaded {len(ipc_ref):,} IPC periods")
    print(f"   Unique districts: {ipc_ref['district'].nunique():,}")

    # Build lookup dictionary - KEY CHANGE: Index by district_normalized
    print("\n2. Building IPC district lookup dictionary...")
    ipc_lookup = defaultdict(list)

    for idx, row in ipc_ref.iterrows():
        if pd.notna(row['fips_code']) and pd.notna(row['district_normalized']):
            # Calculate 4-month aggregation window (3 months before + assessment month)
            # ALIGNED WITH 02_aggregate_articles.py
            ipc_period_end = pd.to_datetime(row['projection_end'])
            ipc_period_start = pd.to_datetime(row['projection_start'])
            agg_window_start = ipc_period_start - relativedelta(months=AGGREGATION_MONTHS - 1)
            agg_window_end = ipc_period_end  # End at assessment month end

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
                'agg_window_start': agg_window_start,  # 4-month window start
                'agg_window_end': agg_window_end,      # 4-month window end
                'ipc_period_length_days': row['period_length_days'],
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

    print(f"   Created lookup with {len(ipc_lookup):,} unique (country, district) combinations")

    # Build word-based lookup from full_name_normalized (ALIGNED WITH 02_aggregate_articles.py)
    word_lookup = build_word_lookup(ipc_ref)

    # Combine word_lookup with ipc_lookup (ipc_lookup takes precedence)
    combined_lookup = defaultdict(list)
    for k, v in word_lookup.items():
        combined_lookup[k].extend(v)
    for k, v in ipc_lookup.items():
        combined_lookup[k].extend(v)
    print(f"   Combined lookup: {len(combined_lookup):,} keys", flush=True)

    # Process locations in chunks
    # KEY CHANGE: Match via GADM3 → GADM2 → GADM1 (ALIGNED WITH 02_aggregate_articles.py)
    print("\n3. Processing locations with DISTRICT-level matching...", flush=True)
    print("   Priority: GADM3 -> GADM2 -> GADM1 (aligned with articles script)", flush=True)

    import pyarrow.parquet as pq
    parquet_file = pq.ParquetFile(LOCATIONS_FILE)

    all_aggregations = []
    total_processed = 0
    total_matched = 0
    match_stats = {
        'GADM3_exact': 0, 'GADM3_fuzzy': 0,
        'GADM2_exact': 0, 'GADM2_fuzzy': 0,
        'GADM1_exact': 0, 'GADM1_fuzzy': 0,  # Added GADM1 - ALIGNED WITH 02_aggregate_articles.py
        'Country_level': 0,
        'no_match': 0
    }

    for batch_num, batch in enumerate(parquet_file.iter_batches(batch_size=CHUNK_SIZE)):
        chunk_start = datetime.now()
        locations_chunk = batch.to_pandas()

        print(f"\n   Batch {batch_num + 1}: Processing {len(locations_chunk):,} locations...", flush=True)

        locations_chunk['date_extracted'] = pd.to_datetime(locations_chunk['date_extracted'])

        # Normalize geographic fields (including GADM1 - ALIGNED WITH 02_aggregate_articles.py)
        locations_chunk['gadm1_norm'] = locations_chunk['gadm1_name'].apply(normalize_text)
        locations_chunk['gadm2_norm'] = locations_chunk['gadm2_name'].apply(normalize_text)
        locations_chunk['gadm3_norm'] = locations_chunk['gadm3_name'].apply(normalize_text)

        matched_records = []
        matched_ipc_ids = set()

        # Group by country for efficiency
        for country_code, country_locs in locations_chunk.groupby('african_country_code'):

            # ================================================================
            # PRIORITY 1: GADM3 Matching (Exact → Fuzzy)
            # GADM3 corresponds to district/woreda level
            # ================================================================
            for gadm3_norm, gadm3_locs in country_locs.groupby('gadm3_norm'):
                if not gadm3_norm:
                    continue

                key = (country_code, gadm3_norm)
                matched_ipc_list = None
                match_type = None
                match_score = 0

                if key in combined_lookup:
                    matched_ipc_list = combined_lookup[key]
                    match_type = 'GADM3_exact'
                    match_score = 100
                else:
                    matched_ipc_list, match_score = find_fuzzy_match(gadm3_norm, combined_lookup, country_code)
                    if matched_ipc_list:
                        match_type = 'GADM3_fuzzy'

                if matched_ipc_list:
                    for ipc_info in matched_ipc_list:
                        # Use ipc_geographic_unit_full as unique key (not just ipc_id)
                        ipc_key = (ipc_info['ipc_id'], ipc_info['ipc_geographic_unit_full'])
                        if ipc_key in matched_ipc_ids:
                            continue

                        # Use 4-month aggregation window (ALIGNED WITH 02_aggregate_articles.py)
                        date_mask = (
                            (gadm3_locs['date_extracted'] >= ipc_info['agg_window_start']) &
                            (gadm3_locs['date_extracted'] <= ipc_info['agg_window_end'])
                        )
                        period_locs = gadm3_locs[date_mask]

                        if len(period_locs) > 0:
                            agg_data = {
                                'location_mention_count': len(period_locs),
                                'unique_location_names': period_locs['location_fullname'].nunique(),
                                'unique_cities': period_locs['city_name'].nunique() if 'city_name' in period_locs.columns else 0,
                                'unique_days': period_locs['date_extracted'].nunique(),
                                'avg_latitude': period_locs['latitude'].mean(),
                                'avg_longitude': period_locs['longitude'].mean(),
                                'latitude_std': period_locs['latitude'].std(),
                                'longitude_std': period_locs['longitude'].std(),
                                'primary_gadm2': period_locs['gadm2_name'].mode()[0] if not period_locs['gadm2_name'].mode().empty else None,
                                'primary_gadm3': period_locs['gadm3_name'].mode()[0] if not period_locs['gadm3_name'].mode().empty else None,
                                'match_level': match_type,
                                'match_score': match_score,
                                **ipc_info
                            }
                            matched_records.append(agg_data)
                            matched_ipc_ids.add(ipc_key)
                            match_stats[match_type] += 1

            # ================================================================
            # PRIORITY 2: GADM2 Matching (Exact → Fuzzy)
            # GADM2 is zone/province level - use as fallback
            # ================================================================
            for gadm2_norm, gadm2_locs in country_locs.groupby('gadm2_norm'):
                if not gadm2_norm:
                    continue

                key = (country_code, gadm2_norm)
                matched_ipc_list = None
                match_type = None
                match_score = 0

                if key in combined_lookup:
                    matched_ipc_list = combined_lookup[key]
                    match_type = 'GADM2_exact'
                    match_score = 100
                else:
                    matched_ipc_list, match_score = find_fuzzy_match(gadm2_norm, combined_lookup, country_code)
                    if matched_ipc_list:
                        match_type = 'GADM2_fuzzy'

                if matched_ipc_list:
                    for ipc_info in matched_ipc_list:
                        ipc_key = (ipc_info['ipc_id'], ipc_info['ipc_geographic_unit_full'])
                        if ipc_key in matched_ipc_ids:
                            continue

                        # Use 4-month aggregation window (ALIGNED WITH 02_aggregate_articles.py)
                        date_mask = (
                            (gadm2_locs['date_extracted'] >= ipc_info['agg_window_start']) &
                            (gadm2_locs['date_extracted'] <= ipc_info['agg_window_end'])
                        )
                        period_locs = gadm2_locs[date_mask]

                        if len(period_locs) > 0:
                            agg_data = {
                                'location_mention_count': len(period_locs),
                                'unique_location_names': period_locs['location_fullname'].nunique(),
                                'unique_cities': period_locs['city_name'].nunique() if 'city_name' in period_locs.columns else 0,
                                'unique_days': period_locs['date_extracted'].nunique(),
                                'avg_latitude': period_locs['latitude'].mean(),
                                'avg_longitude': period_locs['longitude'].mean(),
                                'latitude_std': period_locs['latitude'].std(),
                                'longitude_std': period_locs['longitude'].std(),
                                'primary_gadm2': period_locs['gadm2_name'].mode()[0] if not period_locs['gadm2_name'].mode().empty else None,
                                'primary_gadm3': period_locs['gadm3_name'].mode()[0] if not period_locs['gadm3_name'].mode().empty else None,
                                'match_level': match_type,
                                'match_score': match_score,
                                **ipc_info
                            }
                            matched_records.append(agg_data)
                            matched_ipc_ids.add(ipc_key)
                            match_stats[match_type] += 1

            # ================================================================
            # PRIORITY 3: GADM1 Matching (State/Region level)
            # Important for countries like Nigeria where IPC uses state-level
            # ALIGNED WITH 02_aggregate_articles.py
            # ================================================================
            for gadm1_norm, gadm1_locs in country_locs.groupby('gadm1_norm'):
                if not gadm1_norm:
                    continue

                key = (country_code, gadm1_norm)
                matched_ipc_list = None
                match_type = None
                match_score = 0

                if key in combined_lookup:
                    matched_ipc_list = combined_lookup[key]
                    match_type = 'GADM1_exact'
                    match_score = 100
                else:
                    matched_ipc_list, match_score = find_fuzzy_match(gadm1_norm, combined_lookup, country_code)
                    if matched_ipc_list:
                        match_type = 'GADM1_fuzzy'

                if matched_ipc_list:
                    for ipc_info in matched_ipc_list:
                        ipc_key = (ipc_info['ipc_id'], ipc_info['ipc_geographic_unit_full'])
                        if ipc_key in matched_ipc_ids:
                            continue

                        # Use 4-month aggregation window (ALIGNED WITH 02_aggregate_articles.py)
                        date_mask = (
                            (gadm1_locs['date_extracted'] >= ipc_info['agg_window_start']) &
                            (gadm1_locs['date_extracted'] <= ipc_info['agg_window_end'])
                        )
                        period_locs = gadm1_locs[date_mask]

                        if len(period_locs) > 0:
                            agg_data = {
                                'location_mention_count': len(period_locs),
                                'unique_location_names': period_locs['location_fullname'].nunique(),
                                'unique_cities': period_locs['city_name'].nunique() if 'city_name' in period_locs.columns else 0,
                                'unique_days': period_locs['date_extracted'].nunique(),
                                'avg_latitude': period_locs['latitude'].mean(),
                                'avg_longitude': period_locs['longitude'].mean(),
                                'latitude_std': period_locs['latitude'].std(),
                                'longitude_std': period_locs['longitude'].std(),
                                'primary_gadm2': period_locs['gadm2_name'].mode()[0] if not period_locs['gadm2_name'].mode().empty else None,
                                'primary_gadm3': period_locs['gadm3_name'].mode()[0] if not period_locs['gadm3_name'].mode().empty else None,
                                'match_level': match_type,
                                'match_score': match_score,
                                **ipc_info
                            }
                            matched_records.append(agg_data)
                            matched_ipc_ids.add(ipc_key)
                            match_stats[match_type] += 1

            # ================================================================
            # PRIORITY 4: Country-level matching (for countries with limited data)
            # ================================================================
            if country_code in COUNTRY_LEVEL_ONLY:
                for (fips, district), ipc_list in combined_lookup.items():
                    if fips == country_code:
                        for ipc_info in ipc_list:
                            ipc_key = (ipc_info['ipc_id'], ipc_info['ipc_geographic_unit_full'])
                            if ipc_key in matched_ipc_ids:
                                continue

                            # Use 4-month aggregation window (ALIGNED WITH 02_aggregate_articles.py)
                            date_mask = (
                                (country_locs['date_extracted'] >= ipc_info['agg_window_start']) &
                                (country_locs['date_extracted'] <= ipc_info['agg_window_end'])
                            )
                            period_locs = country_locs[date_mask]

                            if len(period_locs) > 0:
                                agg_data = {
                                    'location_mention_count': len(period_locs),
                                    'unique_location_names': period_locs['location_fullname'].nunique(),
                                    'unique_cities': period_locs['city_name'].nunique() if 'city_name' in period_locs.columns else 0,
                                    'unique_days': period_locs['date_extracted'].nunique(),
                                    'avg_latitude': period_locs['latitude'].mean(),
                                    'avg_longitude': period_locs['longitude'].mean(),
                                    'latitude_std': period_locs['latitude'].std(),
                                    'longitude_std': period_locs['longitude'].std(),
                                    'primary_gadm2': period_locs['gadm2_name'].mode()[0] if not period_locs['gadm2_name'].mode().empty else None,
                                    'primary_gadm3': period_locs['gadm3_name'].mode()[0] if not period_locs['gadm3_name'].mode().empty else None,
                                    'match_level': 'Country_level',
                                    'match_score': 100,
                                    **ipc_info
                                }
                                matched_records.append(agg_data)
                                matched_ipc_ids.add(ipc_key)
                                match_stats['Country_level'] += 1

        if matched_records:
            chunk_df = pd.DataFrame(matched_records)
            all_aggregations.append(chunk_df)
            total_matched += len(matched_records)

        total_processed += len(locations_chunk)
        chunk_time = (datetime.now() - chunk_start).total_seconds()

        print(f"      Matched: {len(matched_records):,} IPC period-aggregations", flush=True)
        print(f"      Time: {chunk_time:.1f}s", flush=True)

        del locations_chunk
        gc.collect()

    # Combine all aggregations
    print(f"\n4. Combining {len(all_aggregations)} chunks...")
    if all_aggregations:
        final_df = pd.concat(all_aggregations, ignore_index=True)

        print("\n" + "=" * 80)
        print("Aggregation Summary - DISTRICT LEVEL")
        print("=" * 80)
        print(f"\nTotal locations processed: {total_processed:,}")
        print(f"Total IPC period-aggregations: {len(final_df):,}")
        print(f"Unique districts: {final_df['ipc_district'].nunique():,}")
        print(f"Unique geographic_unit_full: {final_df['ipc_geographic_unit_full'].nunique():,}")
        print(f"Date range: {final_df['ipc_period_start'].min()} to {final_df['ipc_period_end'].max()}")
        print(f"Countries: {final_df['ipc_country'].nunique()}")

        print(f"\nGeographic Match Statistics:")
        for level, count in sorted(match_stats.items()):
            if count > 0:
                pct = (count / total_matched * 100) if total_matched > 0 else 0
                print(f"   {level}: {count:,} ({pct:.1f}%)")

        print(f"\nRecords by country:")
        print(final_df['ipc_country'].value_counts())

        # Save
        print(f"\n5. Saving to {OUTPUT_PARQUET}...")
        final_df.to_parquet(OUTPUT_PARQUET, index=False)
        print("   [OK] Parquet saved")

        print(f"\n6. Saving to {OUTPUT_CSV}...")
        final_df.to_csv(OUTPUT_CSV, index=False)
        print("   [OK] CSV saved")

        print("\n" + "=" * 80)
        print("Locations Aggregation Complete - DISTRICT LEVEL")
        print("=" * 80)
    else:
        print("\n   WARNING: No matched records found!")

    print(f"\nEnd time: {datetime.now()}")


if __name__ == "__main__":
    main()
