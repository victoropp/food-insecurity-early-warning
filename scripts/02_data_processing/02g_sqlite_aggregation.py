"""
Stage 1 Recovery - SQLite-based Aggregation
Uses SQLite for indexed lookup, which is memory efficient and reliable

Strategy:
1. Load lookup into SQLite with index on GKGRECORDID
2. Process articles in chunks, query SQLite for matches
3. Aggregate incrementally
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import pyarrow.parquet as pq
import sqlite3
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
SQLITE_DB = DISTRICT_DATA_DIR / 'temp_lookup.db'

ARTICLE_CHUNK_SIZE = 50000


def create_sqlite_lookup():
    """Create SQLite database from lookup parquet file."""
    print("   Creating SQLite database...", flush=True)

    conn = sqlite3.connect(str(SQLITE_DB))
    cursor = conn.cursor()

    # Create table
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS lookup (
            GKGRECORDID TEXT,
            match_level TEXT,
            ipc_id INTEGER,
            ipc_country TEXT,
            ipc_country_code TEXT,
            ipc_fips_code TEXT,
            ipc_district TEXT,
            ipc_region TEXT,
            ipc_geographic_unit TEXT,
            ipc_geographic_unit_full TEXT,
            ipc_period_start TEXT,
            ipc_period_end TEXT,
            ipc_period_length_days INTEGER,
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

    # Load parquet in batches and insert
    pf = pq.ParquetFile(LOOKUP_FILE)
    total_inserted = 0

    for batch_num, batch in enumerate(pf.iter_batches(batch_size=500000)):
        df = batch.to_pandas()

        # Convert timestamps to strings
        for col in ['ipc_period_start', 'ipc_period_end', 'ipc_reporting_date']:
            if col in df.columns:
                df[col] = df[col].astype(str)

        # Insert into SQLite
        df.to_sql('lookup', conn, if_exists='append', index=False)
        total_inserted += len(df)

        if batch_num % 10 == 0:
            print(f"      Batch {batch_num}: {total_inserted:,} rows inserted", flush=True)

        del df, batch
        gc.collect()

    # Create index
    print("   Creating index...", flush=True)
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_gkgrecordid ON lookup(GKGRECORDID)')
    conn.commit()
    conn.close()

    print(f"   SQLite DB created: {total_inserted:,} rows", flush=True)
    return total_inserted


def query_lookup(conn, gkg_ids):
    """Query lookup table for given GKGRECORDIDs with batching for SQLite limits."""
    if not gkg_ids:
        return pd.DataFrame()

    # SQLite has a limit of ~999 variables per query
    BATCH_SIZE = 900
    gkg_ids_list = list(gkg_ids)
    results = []

    for i in range(0, len(gkg_ids_list), BATCH_SIZE):
        batch_ids = gkg_ids_list[i:i+BATCH_SIZE]
        placeholders = ','.join(['?' for _ in batch_ids])
        query = f"SELECT * FROM lookup WHERE GKGRECORDID IN ({placeholders})"

        batch_result = pd.read_sql_query(query, conn, params=batch_ids)
        if len(batch_result) > 0:
            results.append(batch_result)

    if not results:
        return pd.DataFrame()

    return pd.concat(results, ignore_index=True)


def main():
    print("=" * 80, flush=True)
    print("STAGE 1 RECOVERY - SQLite Aggregation", flush=True)
    print("=" * 80, flush=True)
    print(f"Start time: {datetime.now()}", flush=True)

    if not LOOKUP_FILE.exists():
        print(f"ERROR: Lookup file not found", flush=True)
        return

    # Step 1: Create SQLite database
    print("\n1. Building SQLite database from lookup...", flush=True)
    if SQLITE_DB.exists():
        print("   Removing existing DB...", flush=True)
        SQLITE_DB.unlink()

    total_lookup_rows = create_sqlite_lookup()

    # Step 2: Get article columns
    print("\n2. Analyzing article columns...", flush=True)
    articles_sample = pd.read_csv(ARTICLES_FILE, nrows=1)
    all_cols = articles_sample.columns.tolist()

    numeric_cols = [col for col in all_cols if col not in [
        'GKGRECORDID', 'DATE', 'SourceCollectionIdentifier', 'SourceCommonName',
        'DocumentIdentifier', 'Counts', 'V2Counts', 'Themes', 'V2Themes',
        'Dates', 'Persons', 'V2Persons', 'Organizations', 'V2Organizations',
        'V2AllNames', 'Locations', 'V2Locations', 'ADM1', 'ADM2', 'ADM3',
        'V2Quotations', 'V2Amounts', 'V2RelatedImages', 'V2DateTimeFields',
        'V2Extras', 'V2ExtendedField', 'V2TranslationInfo', 'OutletType',
        'all_countries_mentioned', 'all_african_countries', 'date_extracted', 'V2Tone'
    ]]
    print(f"   Numeric columns: {len(numeric_cols)}", flush=True)
    del articles_sample
    gc.collect()

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

    # Step 3: Process articles
    print("\n3. Processing articles...", flush=True)

    conn = sqlite3.connect(str(SQLITE_DB))

    # First, get all unique GKGRECORDIDs from lookup
    print("   Getting lookup GKGRECORDIDs...", flush=True)
    cursor = conn.cursor()
    cursor.execute("SELECT DISTINCT GKGRECORDID FROM lookup")
    lookup_ids = set(row[0] for row in cursor.fetchall())
    print(f"   Unique GKGRECORDIDs in lookup: {len(lookup_ids):,}", flush=True)

    aggregated_data = []
    total_matched = 0
    chunk_count = 0

    try:
        for articles_chunk in pd.read_csv(
            ARTICLES_FILE,
            chunksize=ARTICLE_CHUNK_SIZE,
            low_memory=True,
            dtype={'GKGRECORDID': str}
        ):
            chunk_count += 1

            # Filter to matching articles
            mask = articles_chunk['GKGRECORDID'].isin(lookup_ids)
            articles_matched = articles_chunk.loc[mask].copy()

            if len(articles_matched) == 0:
                del articles_chunk
                gc.collect()
                continue

            # Get matching IDs
            matched_ids = articles_matched['GKGRECORDID'].unique().tolist()

            # Query lookup
            lookup_chunk = query_lookup(conn, matched_ids)

            if len(lookup_chunk) == 0:
                del articles_chunk, articles_matched
                gc.collect()
                continue

            # Merge
            merged = articles_matched.merge(lookup_chunk, on='GKGRECORDID', how='inner')

            if chunk_count % 10 == 0:
                print(f"   Chunk {chunk_count}: {len(articles_matched):,} articles -> {len(merged):,} pairs", flush=True)

            total_matched += len(merged)

            del articles_chunk, articles_matched, lookup_chunk

            if len(merged) == 0:
                del merged
                gc.collect()
                continue

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

            # Consolidate periodically
            if len(aggregated_data) >= 50:
                print(f"   Consolidating {len(aggregated_data)} batches...", flush=True)
                partial_df = pd.concat(aggregated_data, ignore_index=True)
                numeric_to_sum = [c for c in partial_df.columns if c not in group_cols]
                agg_dict = {col: 'sum' for col in numeric_to_sum}
                partial_agg = partial_df.groupby(group_cols).agg(agg_dict).reset_index()
                aggregated_data = [partial_agg]
                del partial_df
                gc.collect()

    except Exception as e:
        print(f"\n   ERROR: {e}", flush=True)
        import traceback
        traceback.print_exc()
        if aggregated_data:
            partial_df = pd.concat(aggregated_data, ignore_index=True)
            partial_df.to_parquet(DISTRICT_DATA_DIR / 'articles_aggregated_partial.parquet')
            print(f"   Saved partial results", flush=True)
        raise
    finally:
        conn.close()

    print(f"\n   Total chunks: {chunk_count}", flush=True)
    print(f"   Total matched pairs: {total_matched:,}", flush=True)

    # Step 4: Final aggregation
    print("\n4. Final aggregation...", flush=True)
    if not aggregated_data:
        print("   No data!", flush=True)
        return

    final_df = pd.concat(aggregated_data, ignore_index=True)
    del aggregated_data
    gc.collect()

    group_cols_final = [c for c in group_cols if c != 'match_level']
    numeric_to_sum = [c for c in final_df.columns if c not in group_cols_final and c != 'match_level']
    agg_dict = {col: 'sum' for col in numeric_to_sum}
    agg_dict['match_level'] = 'first'

    final_agg = final_df.groupby(group_cols_final).agg(agg_dict).reset_index()
    del final_df
    gc.collect()

    # Summary
    print("\n" + "=" * 80, flush=True)
    print("Summary", flush=True)
    print("=" * 80, flush=True)
    print(f"Total records: {len(final_agg):,}", flush=True)
    print(f"Countries: {final_agg['ipc_country'].nunique()}", flush=True)
    print(f"Districts: {final_agg['ipc_district'].nunique():,}", flush=True)

    print(f"\nMatch levels:", flush=True)
    print(final_agg['match_level'].value_counts(), flush=True)

    # Save
    print("\n5. Saving...", flush=True)
    final_agg.to_parquet(OUTPUT_PARQUET, index=False)
    final_agg.to_csv(OUTPUT_CSV, index=False)
    print(f"   Saved: {OUTPUT_PARQUET}", flush=True)
    print(f"   Saved: {OUTPUT_CSV}", flush=True)

    # Cleanup
    print("\n6. Cleanup...", flush=True)
    if SQLITE_DB.exists():
        SQLITE_DB.unlink()
        print("   Removed SQLite DB", flush=True)

    print(f"\nDone: {datetime.now()}", flush=True)
    print("=" * 80, flush=True)


if __name__ == '__main__':
    main()
