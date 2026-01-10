"""
Stage 1 Recovery - Part 2: Complete Aggregation from Merged Lookup
Memory-optimized version that processes articles in smaller chunks

The temp files have been merged into temp_articles_district_lookup.parquet
This script completes the aggregation with articles.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pyarrow.parquet as pq
import gc
import sys
from config import BASE_DIR

# Paths
BASE_DIR = Path(rstr(BASE_DIR.parent.parent.parent))
ARTICLES_FILE = BASE_DIR / 'data' / 'african_gkg_articles.csv'
DISTRICT_DATA_DIR = BASE_DIR / 'data' / 'district_level'
LOOKUP_FILE = DISTRICT_DATA_DIR / 'temp_articles_district_lookup.parquet'
OUTPUT_PARQUET = DISTRICT_DATA_DIR / 'articles_aggregated.parquet'
OUTPUT_CSV = DISTRICT_DATA_DIR / 'articles_aggregated.csv'

# Smaller chunk size for memory efficiency
CHUNK_SIZE = 100000  # Reduced from 250k


def main():
    print("=" * 80)
    print("STAGE 1 RECOVERY - Part 2: Complete Aggregation")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")

    if not LOOKUP_FILE.exists():
        print(f"ERROR: Lookup file not found: {LOOKUP_FILE}")
        print("Run 02a_recover_aggregation.py first to merge temp files")
        return

    # Load the full lookup into memory (it's ~60MB compressed)
    print("\n1. Loading merged lookup file...")
    gkg_lookup = pd.read_parquet(LOOKUP_FILE)
    print(f"   Loaded {len(gkg_lookup):,} GKGRECORDID -> IPC mappings")

    # Create a set of all GKGRECORDID for fast filtering
    gkg_ids_set = set(gkg_lookup['GKGRECORDID'].unique())
    print(f"   Unique GKGRECORDIDs: {len(gkg_ids_set):,}")

    # Get sample of articles to identify columns
    print("\n2. Analyzing article columns...")
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
        'all_countries_mentioned', 'all_african_countries', 'date_extracted', 'V2Tone'
    ]]
    print(f"   Numeric columns for aggregation: {len(numeric_cols)}")

    del articles_sample
    gc.collect()

    # Group columns for final aggregation
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

    # Process articles in chunks
    print("\n3. Processing articles in chunks...")
    aggregated_data = []
    total_matched = 0
    chunk_count = 0

    try:
        for articles_chunk in pd.read_csv(
            ARTICLES_FILE,
            chunksize=CHUNK_SIZE,
            low_memory=True,
            dtype={'GKGRECORDID': str}
        ):
            chunk_count += 1

            # Filter to only articles that have IPC matches
            relevant_ids = articles_chunk['GKGRECORDID'].isin(gkg_ids_set)
            articles_matched = articles_chunk[relevant_ids].copy()

            if len(articles_matched) == 0:
                if chunk_count % 10 == 0:
                    print(f"   Chunk {chunk_count}: 0 matches (skipping)")
                del articles_chunk, articles_matched
                gc.collect()
                continue

            # Get lookup data for these articles
            matched_ids = set(articles_matched['GKGRECORDID'].unique())
            lookup_chunk = gkg_lookup[gkg_lookup['GKGRECORDID'].isin(matched_ids)].copy()

            # Merge articles with lookup
            merged = articles_matched.merge(lookup_chunk, on='GKGRECORDID', how='inner')

            if chunk_count % 5 == 0:
                print(f"   Chunk {chunk_count}: {len(articles_matched):,} articles -> {len(merged):,} pairs", flush=True)

            total_matched += len(merged)

            del articles_chunk, articles_matched, lookup_chunk

            if len(merged) == 0:
                del merged
                gc.collect()
                continue

            # Aggregate this chunk
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

            # Periodically consolidate to prevent memory growth
            if len(aggregated_data) >= 50:
                print(f"   Consolidating {len(aggregated_data)} partial aggregations...")
                partial_df = pd.concat(aggregated_data, ignore_index=True)

                # Re-aggregate the partial results
                numeric_to_sum = [c for c in partial_df.columns if c not in group_cols]
                agg_dict = {col: 'sum' for col in numeric_to_sum if col in partial_df.columns}

                partial_agg = partial_df.groupby(group_cols).agg(agg_dict).reset_index()

                aggregated_data = [partial_agg]
                del partial_df
                gc.collect()

    except Exception as e:
        print(f"\n   ERROR during processing: {e}")
        print(f"   Processed {chunk_count} chunks, {total_matched:,} matched pairs")
        if aggregated_data:
            print("   Saving partial results...")
            partial_df = pd.concat(aggregated_data, ignore_index=True)
            partial_file = DISTRICT_DATA_DIR / 'articles_aggregated_partial.parquet'
            partial_df.to_parquet(partial_file, index=False)
            print(f"   Saved partial results to {partial_file}")
        raise

    print(f"\n   Processed {chunk_count} chunks")
    print(f"   Total matched article-period pairs: {total_matched:,}")

    # Final aggregation
    print("\n4. Final aggregation...")
    if not aggregated_data:
        print("   WARNING: No data to aggregate!")
        return

    final_df = pd.concat(aggregated_data, ignore_index=True)
    del aggregated_data
    gc.collect()

    print(f"   Combined partial aggregations: {len(final_df):,} rows")

    # Final groupby
    group_cols_final = [
        'ipc_id', 'ipc_country', 'ipc_country_code', 'ipc_fips_code',
        'ipc_district', 'ipc_region',
        'ipc_geographic_unit', 'ipc_geographic_unit_full',
        'ipc_period_start', 'ipc_period_end', 'ipc_period_length_days',
        'ipc_value', 'ipc_description', 'ipc_binary_crisis',
        'ipc_is_allowing_assistance', 'ipc_fewsnet_region',
        'ipc_geographic_group', 'ipc_scenario', 'ipc_classification_scale',
        'ipc_reporting_date'
    ]

    numeric_to_sum = [c for c in final_df.columns if c not in group_cols_final and c != 'match_level']
    agg_dict = {col: 'sum' for col in numeric_to_sum if col in final_df.columns}
    agg_dict['match_level'] = 'first'

    final_agg = final_df.groupby(group_cols_final).agg(agg_dict).reset_index()

    del final_df
    gc.collect()

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
    print(f"\n5. Saving to {OUTPUT_PARQUET}...")
    final_agg.to_parquet(OUTPUT_PARQUET, index=False)
    print("   [OK] Parquet saved")

    print(f"\n6. Saving to {OUTPUT_CSV}...")
    final_agg.to_csv(OUTPUT_CSV, index=False)
    print("   [OK] CSV saved")

    # Cleanup
    print("\n7. Cleaning up...")
    # Keep temp files for now in case of issues
    # LOOKUP_FILE.unlink()
    # for temp_file in DISTRICT_DATA_DIR.glob('temp_articles_district_batch_*.parquet'):
    #     temp_file.unlink()
    print("   Temp files preserved (delete manually if desired)")

    print("\n" + "=" * 80)
    print("Articles Aggregation Complete - DISTRICT LEVEL")
    print("=" * 80)
    print(f"End time: {datetime.now()}")


if __name__ == '__main__':
    main()
