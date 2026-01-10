"""
GDELT Articles Aggregation - RESUME FROM EXISTING SQLITE
=========================================================
Resumes article aggregation from existing SQLite database that contains
181.8M article-IPC matches. Bypasses the OOM issue by using batched deduplication.

MEMORY-SAFE APPROACH:
1. Skip location matching (already done - 181.8M matches in gkg_ipc table)
2. Batched deduplication: Process 5M rows at a time instead of all 181M
3. Streaming article aggregation with periodic consolidation

Designed for 8GB RAM systems.

Author: Memory-optimized resume script
"""

import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import gc
import sqlite3
import sys
from config import BASE_DIR

# Paths
BASE_DIR = Path(str(BASE_DIR.parent.parent.parent))
ARTICLES_FILE = BASE_DIR / 'data' / 'african_gkg_articles.csv'
DISTRICT_DATA_DIR = BASE_DIR / 'data' / 'district_level'
OUTPUT_PARQUET = DISTRICT_DATA_DIR / 'articles_aggregated.parquet'
OUTPUT_CSV = DISTRICT_DATA_DIR / 'articles_aggregated.csv'

# The existing SQLite database with 181.8M matches
SQLITE_FILE = DISTRICT_DATA_DIR / 'temp_gkg_ipc_lookup_20251204_180617.db'

# Memory-safe settings for 8GB RAM
DEDUP_BATCH_SIZE = 5_000_000  # 5M rows per dedup batch
ARTICLE_CHUNK_SIZE = 100_000  # 100K articles per chunk
CONSOLIDATE_EVERY = 50  # Consolidate aggregations every 50 chunks
SQL_BATCH_SIZE = 500  # Max SQL placeholders per query


def check_sqlite_integrity(conn):
    """Quick integrity check on SQLite database."""
    print("   Checking SQLite database integrity...", flush=True)
    cursor = conn.cursor()

    # Check if main table exists
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='gkg_ipc'")
    if not cursor.fetchone():
        print("   ERROR: gkg_ipc table not found!")
        return False, 0

    # Get row count
    cursor.execute("SELECT COUNT(*) FROM gkg_ipc")
    total_rows = cursor.fetchone()[0]
    print(f"   Found {total_rows:,} rows in gkg_ipc table", flush=True)

    # Quick integrity check (PRAGMA integrity_check is slow on large DBs)
    # Just check a sample instead
    cursor.execute("SELECT * FROM gkg_ipc LIMIT 5")
    sample = cursor.fetchall()
    if len(sample) == 0:
        print("   ERROR: Table appears empty or corrupted!")
        return False, 0

    print(f"   Sample row columns: {len(sample[0])}", flush=True)
    print("   [OK] SQLite database appears intact", flush=True)

    return True, total_rows


def batched_deduplication(conn, total_rows):
    """
    Memory-safe deduplication: Process in batches of DEDUP_BATCH_SIZE rows.

    Instead of: CREATE TABLE ... GROUP BY (loads 181M rows into memory)
    We use: INSERT ... GROUP BY for each 5M row batch, then final dedup
    """
    print(f"\n2. Batched deduplication of {total_rows:,} rows...", flush=True)
    print(f"   Batch size: {DEDUP_BATCH_SIZE:,} rows", flush=True)
    cursor = conn.cursor()

    # Check if dedup table already exists (from previous interrupted run)
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='gkg_ipc_dedup'")
    if cursor.fetchone():
        cursor.execute("SELECT COUNT(*) FROM gkg_ipc_dedup")
        existing_count = cursor.fetchone()[0]
        print(f"   Found existing gkg_ipc_dedup table with {existing_count:,} rows", flush=True)

        # Check if it has the index
        cursor.execute("SELECT name FROM sqlite_master WHERE type='index' AND name='idx_dedup_gkgrecordid'")
        if cursor.fetchone():
            print("   [OK] Using existing deduplicated table with index", flush=True)
            return existing_count
        else:
            print("   Creating index on existing table...", flush=True)
            cursor.execute('CREATE INDEX IF NOT EXISTS idx_dedup_gkgrecordid ON gkg_ipc_dedup(GKGRECORDID)')
            conn.commit()
            return existing_count

    # Get min/max rowid for batch processing
    cursor.execute("SELECT MIN(rowid), MAX(rowid) FROM gkg_ipc")
    min_rowid, max_rowid = cursor.fetchone()
    print(f"   Rowid range: {min_rowid:,} to {max_rowid:,}", flush=True)

    # Create intermediate batch table (will hold partial dedup results)
    print("   Creating intermediate dedup table...", flush=True)
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS gkg_ipc_dedup_temp (
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

    # Process in batches
    num_batches = (max_rowid - min_rowid + DEDUP_BATCH_SIZE) // DEDUP_BATCH_SIZE
    print(f"   Processing {num_batches} batches...", flush=True)

    start_time = datetime.now()
    batch_start_rowid = min_rowid
    batch_num = 0

    while batch_start_rowid <= max_rowid:
        batch_end_rowid = batch_start_rowid + DEDUP_BATCH_SIZE - 1
        batch_num += 1

        # Insert deduplicated batch into temp table
        cursor.execute('''
            INSERT INTO gkg_ipc_dedup_temp
            SELECT
                GKGRECORDID, match_level, ipc_id, ipc_country, ipc_country_code,
                ipc_fips_code, ipc_district, ipc_region, ipc_geographic_unit,
                ipc_geographic_unit_full, ipc_period_start, ipc_period_end,
                ipc_period_length_days, ipc_value, ipc_description, ipc_binary_crisis,
                ipc_is_allowing_assistance, ipc_fewsnet_region, ipc_geographic_group,
                ipc_scenario, ipc_classification_scale, ipc_reporting_date
            FROM gkg_ipc
            WHERE rowid >= ? AND rowid <= ?
            GROUP BY GKGRECORDID, ipc_id
        ''', (batch_start_rowid, batch_end_rowid))

        inserted = cursor.rowcount
        conn.commit()
        gc.collect()

        # Progress
        elapsed = (datetime.now() - start_time).total_seconds()
        rate = batch_num / elapsed if elapsed > 0 else 0
        eta = (num_batches - batch_num) / rate if rate > 0 else 0
        print(f"      Batch {batch_num}/{num_batches}: rows {batch_start_rowid:,}-{batch_end_rowid:,} | +{inserted:,} dedup rows | ETA: {eta/60:.1f}min", flush=True)

        batch_start_rowid = batch_end_rowid + 1

    # Get count from temp table
    cursor.execute("SELECT COUNT(*) FROM gkg_ipc_dedup_temp")
    temp_count = cursor.fetchone()[0]
    print(f"\n   Intermediate dedup: {total_rows:,} -> {temp_count:,} rows", flush=True)

    # Final deduplication across batch boundaries
    print("   Final deduplication across batch boundaries...", flush=True)
    cursor.execute('''
        CREATE TABLE gkg_ipc_dedup AS
        SELECT
            GKGRECORDID, match_level, ipc_id, ipc_country, ipc_country_code,
            ipc_fips_code, ipc_district, ipc_region, ipc_geographic_unit,
            ipc_geographic_unit_full, ipc_period_start, ipc_period_end,
            ipc_period_length_days, ipc_value, ipc_description, ipc_binary_crisis,
            ipc_is_allowing_assistance, ipc_fewsnet_region, ipc_geographic_group,
            ipc_scenario, ipc_classification_scale, ipc_reporting_date
        FROM gkg_ipc_dedup_temp
        GROUP BY GKGRECORDID, ipc_id
    ''')
    conn.commit()

    # Get final count
    cursor.execute("SELECT COUNT(*) FROM gkg_ipc_dedup")
    final_count = cursor.fetchone()[0]
    print(f"   Final dedup: {temp_count:,} -> {final_count:,} unique matches", flush=True)

    # Drop temp table to save space
    print("   Dropping intermediate table...", flush=True)
    cursor.execute("DROP TABLE gkg_ipc_dedup_temp")
    conn.commit()

    # Create index on deduplicated table
    print("   Creating index on deduplicated table...", flush=True)
    cursor.execute('CREATE INDEX idx_dedup_gkgrecordid ON gkg_ipc_dedup(GKGRECORDID)')
    conn.commit()

    print(f"   [OK] Deduplication complete: {total_rows:,} -> {final_count:,} rows", flush=True)
    return final_count


def stream_article_aggregation(conn):
    """
    Memory-safe article aggregation with periodic consolidation.

    Key difference from original: Consolidates every CONSOLIDATE_EVERY chunks
    to prevent memory accumulation.
    """
    print("\n3. Streaming article aggregation...", flush=True)
    print(f"   Chunk size: {ARTICLE_CHUNK_SIZE:,} articles", flush=True)
    print(f"   Consolidate every: {CONSOLIDATE_EVERY} chunks", flush=True)

    cursor = conn.cursor()

    # Get sample to determine numeric columns
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

    # Group columns for aggregation
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

    aggregated_data = []
    start_time = datetime.now()
    total_articles = 0
    total_matched = 0
    consolidation_count = 0

    for chunk_num, articles_chunk in enumerate(pd.read_csv(ARTICLES_FILE, chunksize=ARTICLE_CHUNK_SIZE)):
        chunk_start = datetime.now()
        total_articles += len(articles_chunk)

        # Get unique article IDs in this chunk
        article_ids = articles_chunk['GKGRECORDID'].unique().tolist()

        # Query SQLite in batches (SQLite has placeholder limits)
        all_lookup_rows = []
        for i in range(0, len(article_ids), SQL_BATCH_SIZE):
            batch_ids = article_ids[i:i + SQL_BATCH_SIZE]
            placeholders = ','.join(['?'] * len(batch_ids))
            query = f'SELECT * FROM gkg_ipc_dedup WHERE GKGRECORDID IN ({placeholders})'
            batch_df = pd.read_sql_query(query, conn, params=batch_ids)
            if len(batch_df) > 0:
                all_lookup_rows.append(batch_df)

        if not all_lookup_rows:
            # No matches in this chunk
            if (chunk_num + 1) % 10 == 0:
                print(f"      Chunk {chunk_num + 1}: {len(articles_chunk):,} articles, 0 matches", flush=True)
            del articles_chunk, article_ids
            gc.collect()
            continue

        gkg_lookup_chunk = pd.concat(all_lookup_rows, ignore_index=True)
        del all_lookup_rows

        # Merge articles with IPC info
        merged = articles_chunk.merge(gkg_lookup_chunk, on='GKGRECORDID', how='inner')
        total_matched += len(merged)

        del articles_chunk, gkg_lookup_chunk, article_ids

        if len(merged) == 0:
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

        # CRITICAL: Consolidate periodically to prevent memory accumulation
        if len(aggregated_data) >= CONSOLIDATE_EVERY:
            consolidation_count += 1
            print(f"      [Consolidation {consolidation_count}] Merging {len(aggregated_data)} partial aggregations...", flush=True)

            combined = pd.concat(aggregated_data, ignore_index=True)

            # Re-aggregate the combined data
            numeric_to_sum = [c for c in combined.columns if c not in group_cols and c != 'match_level']
            consolidate_agg_dict = {col: 'sum' for col in numeric_to_sum if col in combined.columns}
            consolidate_agg_dict['match_level'] = 'first'

            consolidated = combined.groupby([c for c in group_cols if c != 'match_level']).agg(consolidate_agg_dict).reset_index()

            aggregated_data = [consolidated]
            del combined, consolidated
            gc.collect()

        # Progress every 10 chunks
        if (chunk_num + 1) % 10 == 0:
            elapsed = (datetime.now() - start_time).total_seconds()
            rate = total_articles / elapsed if elapsed > 0 else 0
            # Estimate total articles from file size (~7.6M)
            est_total = 7_638_809
            eta = (est_total - total_articles) / rate if rate > 0 else 0
            print(f"      Chunk {chunk_num + 1}: {total_articles:,} articles processed, {total_matched:,} matches | ETA: {eta/60:.1f}min", flush=True)

    # Final combination
    print(f"\n   Combining {len(aggregated_data)} final aggregations...", flush=True)
    if not aggregated_data:
        print("   WARNING: No data to aggregate!")
        return None

    final_df = pd.concat(aggregated_data, ignore_index=True)
    del aggregated_data
    gc.collect()

    # Final aggregation (remove match_level from grouping for final)
    final_group_cols = [c for c in group_cols if c != 'match_level']
    numeric_to_sum = [c for c in final_df.columns if c not in group_cols]
    agg_dict = {col: 'sum' for col in numeric_to_sum if col in final_df.columns}
    agg_dict['match_level'] = 'first'

    final_agg = final_df.groupby(final_group_cols).agg(agg_dict).reset_index()
    del final_df
    gc.collect()

    print(f"   [OK] Aggregation complete: {len(final_agg):,} IPC period-aggregations", flush=True)
    return final_agg


def main():
    print("=" * 80)
    print("GDELT Articles Aggregation - RESUME FROM EXISTING SQLITE")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print(f"SQLite file: {SQLITE_FILE}")
    print(f"File size: {SQLITE_FILE.stat().st_size / (1024**3):.2f} GB")

    # Check if SQLite file exists
    if not SQLITE_FILE.exists():
        print(f"\nERROR: SQLite file not found at {SQLITE_FILE}")
        print("Run the original 02_aggregate_articles.py to create it.")
        sys.exit(1)

    # Connect to SQLite
    print("\n1. Connecting to existing SQLite database...")
    conn = sqlite3.connect(str(SQLITE_FILE))

    # Set pragmas for read performance
    cursor = conn.cursor()
    cursor.execute("PRAGMA cache_size=100000")
    cursor.execute("PRAGMA temp_store=MEMORY")
    conn.commit()

    # Check integrity
    is_valid, total_rows = check_sqlite_integrity(conn)
    if not is_valid:
        print("ERROR: SQLite database appears corrupted or incomplete!")
        conn.close()
        sys.exit(1)

    # Batched deduplication
    dedup_count = batched_deduplication(conn, total_rows)

    # Stream article aggregation
    final_agg = stream_article_aggregation(conn)

    if final_agg is None or len(final_agg) == 0:
        print("\nERROR: No aggregated data produced!")
        conn.close()
        sys.exit(1)

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

    # Save results
    print(f"\n4. Saving to {OUTPUT_PARQUET}...")
    final_agg.to_parquet(OUTPUT_PARQUET, index=False)
    print("   [OK] Parquet saved")

    print(f"\n5. Saving to {OUTPUT_CSV}...")
    final_agg.to_csv(OUTPUT_CSV, index=False)
    print("   [OK] CSV saved")

    # Close connection (don't delete the SQLite file - keep for potential re-runs)
    conn.close()
    print("\n   SQLite database preserved for potential re-runs")

    print("\n" + "=" * 80)
    print("Articles Aggregation Complete - RESUME MODE")
    print("=" * 80)
    print(f"End time: {datetime.now()}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress saved in SQLite database.")
        print("Re-run this script to resume from where it left off.")
        sys.exit(1)
    except Exception as e:
        print(f"\n\nERROR: {e}")
        import traceback
        traceback.print_exc()
        print("\nThe SQLite database is preserved. You can re-run this script to retry.")
        sys.exit(1)
