"""
Download FEWS NET Livelihood Zones for Africa
These zones are used by IPC for food security classification alignment.

FEWS NET provides livelihood zone shapefiles that define areas with similar
food security characteristics - essential for IPC alignment.
"""

import os
import requests
import zipfile
from io import BytesIO
import geopandas as gpd
from config import BASE_DIR

# Output directory
OUTPUT_DIR = BASE_DIR / "data" / "external" / "shapefiles" / "livelihood_zones"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# FEWS NET Livelihood Zones URLs
# These are available from FEWS NET Data Center
FEWS_NET_URLS = {
    # Africa-wide livelihood zones
    'africa_lhz': 'https://fews.net/sites/default/files/lhz/Africa_LHZ.zip',
}

# Individual country livelihood zones (FEWS NET provides these separately)
COUNTRY_LHZ_URLS = {
    'ETH': 'https://fews.net/sites/default/files/lhz/ET_LHZ_2018.zip',
    'KEN': 'https://fews.net/sites/default/files/lhz/KE_LHZ_2011.zip',
    'SOM': 'https://fews.net/sites/default/files/lhz/SO_LHZ_2015.zip',
    'SSD': 'https://fews.net/sites/default/files/lhz/SS_LHZ_2018.zip',
    'SDN': 'https://fews.net/sites/default/files/lhz/SD_LHZ_2015.zip',
    'UGA': 'https://fews.net/sites/default/files/lhz/UG_LHZ_2013.zip',
    'NGA': 'https://fews.net/sites/default/files/lhz/NG_LHZ_2018.zip',
    'NER': 'https://fews.net/sites/default/files/lhz/NE_LHZ_2011.zip',
    'MLI': 'https://fews.net/sites/default/files/lhz/ML_LHZ_2014.zip',
    'BFA': 'https://fews.net/sites/default/files/lhz/BF_LHZ_2014.zip',
    'TCD': 'https://fews.net/sites/default/files/lhz/TD_LHZ_2011.zip',
    'CAF': 'https://fews.net/sites/default/files/lhz/CF_LHZ_2016.zip',
    'COD': 'https://fews.net/sites/default/files/lhz/CD_LHZ_2015.zip',
    'MWI': 'https://fews.net/sites/default/files/lhz/MW_LHZ_2015.zip',
    'MOZ': 'https://fews.net/sites/default/files/lhz/MZ_LHZ_2014.zip',
    'ZWE': 'https://fews.net/sites/default/files/lhz/ZW_LHZ_2011.zip',
    'ZMB': 'https://fews.net/sites/default/files/lhz/ZM_LHZ_2014.zip',
    'TZA': 'https://fews.net/sites/default/files/lhz/TZ_LHZ_2013.zip',
    'RWA': 'https://fews.net/sites/default/files/lhz/RW_LHZ_2015.zip',
    'BDI': 'https://fews.net/sites/default/files/lhz/BI_LHZ_2014.zip',
}

def download_and_extract(url, output_dir, name):
    """Download and extract a zip file"""
    try:
        print(f"Downloading {name}...")
        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()

        # Extract
        extract_dir = os.path.join(output_dir, name)
        os.makedirs(extract_dir, exist_ok=True)

        with zipfile.ZipFile(BytesIO(response.content)) as zf:
            zf.extractall(extract_dir)

        print(f"  Downloaded and extracted to {extract_dir}")
        return True
    except Exception as e:
        print(f"  Failed: {e}")
        return False

def main():
    print("=" * 80)
    print("FEWS NET LIVELIHOOD ZONES DOWNLOADER")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    successful = 0
    failed = 0

    # Download Africa-wide livelihood zones
    print("\n--- Africa-Wide Livelihood Zones ---")
    for name, url in FEWS_NET_URLS.items():
        if download_and_extract(url, OUTPUT_DIR, name):
            successful += 1
        else:
            failed += 1

    # Download country-specific livelihood zones
    print("\n--- Country Livelihood Zones ---")
    for code, url in COUNTRY_LHZ_URLS.items():
        if download_and_extract(url, OUTPUT_DIR, f"LHZ_{code}"):
            successful += 1
        else:
            failed += 1

    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")

    # List downloaded files
    print("\nDownloaded livelihood zones:")
    for item in os.listdir(OUTPUT_DIR):
        item_path = os.path.join(OUTPUT_DIR, item)
        if os.path.isdir(item_path):
            files = os.listdir(item_path)
            shp_files = [f for f in files if f.endswith('.shp')]
            print(f"  {item}: {len(shp_files)} shapefiles")

if __name__ == "__main__":
    main()
