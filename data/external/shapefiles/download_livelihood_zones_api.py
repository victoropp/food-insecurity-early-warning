"""
Download FEWS NET Livelihood Zones for Africa using FDW API
FEWS NET Data Warehouse provides livelihood zone shapefiles via REST API.

API Documentation: https://fdw.fews.net/en/docs/api_reference/api_reference.html
API Endpoint: https://fdw.fews.net/api/feature.geojson?country_code={CODE}&unit_type=livelihood_zone
"""

import os
import requests
import json
import time
from tqdm import tqdm

# Output directory
OUTPUT_DIR = r"D:\GDELT_Africa_Extract\shapefiles\livelihood_zones"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# FEWS NET Data Warehouse API endpoint for geographic features
FDW_API_BASE = "https://fdw.fews.net/api/feature.geojson"

# African countries with FEWS NET coverage (ISO 2-letter codes)
# FEWS NET covers food-insecure regions, not all 54 African countries
FEWSNET_AFRICA_COUNTRIES = {
    # East Africa
    'ET': 'Ethiopia',
    'KE': 'Kenya',
    'SO': 'Somalia',
    'SS': 'South Sudan',
    'SD': 'Sudan',
    'UG': 'Uganda',
    'TZ': 'Tanzania',
    'RW': 'Rwanda',
    'BI': 'Burundi',
    'DJ': 'Djibouti',

    # Southern Africa
    'ZW': 'Zimbabwe',
    'ZM': 'Zambia',
    'MW': 'Malawi',
    'MZ': 'Mozambique',
    'AO': 'Angola',
    'MG': 'Madagascar',
    'LS': 'Lesotho',

    # West Africa
    'NG': 'Nigeria',
    'NE': 'Niger',
    'ML': 'Mali',
    'BF': 'Burkina Faso',
    'TD': 'Chad',
    'SN': 'Senegal',
    'MR': 'Mauritania',
    'GM': 'Gambia',
    'GN': 'Guinea',
    'SL': 'Sierra Leone',
    'LR': 'Liberia',
    'TG': 'Togo',
    'GH': 'Ghana',

    # Central Africa
    'CF': 'Central African Republic',
    'CD': 'Democratic Republic of Congo',
    'CM': 'Cameroon',
}

def download_livelihood_zones(country_code, country_name, output_dir):
    """Download livelihood zones for a country using FDW API"""
    # Try multiple date parameters to get the most recent data
    dates_to_try = ['2024-12-31', '2023-12-31', '2022-12-31', '2021-12-31', None]

    for as_of_date in dates_to_try:
        try:
            params = {
                'country_code': country_code,
                'unit_type': 'livelihood_zone',
            }
            if as_of_date:
                params['as_of_date'] = as_of_date

            response = requests.get(FDW_API_BASE, params=params, timeout=120)

            if response.status_code == 200:
                data = response.json()

                # Check if we got valid features
                if data.get('features') and len(data['features']) > 0:
                    # Save as GeoJSON
                    output_file = os.path.join(output_dir, f"LHZ_{country_code}.geojson")
                    with open(output_file, 'w', encoding='utf-8') as f:
                        json.dump(data, f)

                    return True, f"Downloaded ({len(data['features'])} zones)"

        except Exception as e:
            continue

    return False, "No data available"

def download_all_livelihood_zones():
    """Try to download all livelihood zones at once"""
    try:
        # Try to get all African livelihood zones without country filter
        response = requests.get(
            FDW_API_BASE,
            params={'unit_type': 'livelihood_zone'},
            timeout=300
        )

        if response.status_code == 200:
            data = response.json()
            if data.get('features') and len(data['features']) > 0:
                output_file = os.path.join(OUTPUT_DIR, "all_africa_livelihood_zones.geojson")
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(data, f)
                return True, len(data['features'])
    except Exception as e:
        pass

    return False, 0

def main():
    print("=" * 80)
    print("FEWS NET LIVELIHOOD ZONES DOWNLOADER (FDW API)")
    print(f"API Endpoint: {FDW_API_BASE}")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    # First try to download all at once
    print("\n--- Attempting Bulk Download ---")
    success, count = download_all_livelihood_zones()
    if success:
        print(f"  Downloaded all livelihood zones: {count} zones")
    else:
        print("  Bulk download not available, proceeding country-by-country")

    # Download by country
    print(f"\n--- Downloading by Country ({len(FEWSNET_AFRICA_COUNTRIES)} countries) ---")
    successful = []
    failed = []

    for code, name in tqdm(FEWSNET_AFRICA_COUNTRIES.items(), desc="Downloading"):
        success, message = download_livelihood_zones(code, name, OUTPUT_DIR)

        if success:
            successful.append((code, name, message))
        else:
            failed.append((code, name, message))

        # Rate limiting
        time.sleep(0.5)

    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"Successful: {len(successful)}/{len(FEWSNET_AFRICA_COUNTRIES)}")
    print(f"Failed: {len(failed)}/{len(FEWSNET_AFRICA_COUNTRIES)}")

    if successful:
        print("\nSuccessful downloads:")
        for code, name, msg in successful[:10]:
            print(f"  {name} ({code}): {msg}")
        if len(successful) > 10:
            print(f"  ... and {len(successful) - 10} more")

    if failed:
        print("\nFailed downloads (no FEWS NET coverage):")
        for code, name, msg in failed:
            print(f"  {name} ({code}): {msg}")

    # List output files
    print("\n--- Output Files ---")
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.geojson'):
            fpath = os.path.join(OUTPUT_DIR, f)
            size_mb = os.path.getsize(fpath) / (1024 * 1024)
            print(f"  {f}: {size_mb:.2f} MB")

if __name__ == "__main__":
    main()
