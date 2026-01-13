"""
Download GADM Administrative Boundaries for ALL 54 African Countries
GADM provides administrative boundaries at multiple levels:
- Level 0: Country boundaries
- Level 1: First-level admin (states/provinces/regions)
- Level 2: Second-level admin (districts/counties)
- Level 3+: Lower admin levels where available

This will enable alignment with IPC classifications which use admin boundaries.
"""

import os
import requests
import zipfile
from io import BytesIO
from tqdm import tqdm
import time

# Output directory
OUTPUT_DIR = r"D:\GDELT_Africa_Extract\shapefiles\gadm"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# All 54 African Countries with ISO 3166-1 alpha-3 codes (used by GADM)
AFRICAN_COUNTRIES = {
    # North Africa (6)
    'DZA': 'Algeria',
    'EGY': 'Egypt',
    'LBY': 'Libya',
    'MAR': 'Morocco',
    'TUN': 'Tunisia',
    'SDN': 'Sudan',

    # West Africa (16)
    'BEN': 'Benin',
    'BFA': 'Burkina Faso',
    'CPV': 'Cape Verde',
    'CIV': 'Cote d Ivoire',
    'GMB': 'Gambia',
    'GHA': 'Ghana',
    'GIN': 'Guinea',
    'GNB': 'Guinea-Bissau',
    'LBR': 'Liberia',
    'MLI': 'Mali',
    'MRT': 'Mauritania',
    'NER': 'Niger',
    'NGA': 'Nigeria',
    'SEN': 'Senegal',
    'SLE': 'Sierra Leone',
    'TGO': 'Togo',

    # East Africa (14)
    'BDI': 'Burundi',
    'DJI': 'Djibouti',
    'ERI': 'Eritrea',
    'ETH': 'Ethiopia',
    'KEN': 'Kenya',
    'MDG': 'Madagascar',
    'MWI': 'Malawi',
    'MOZ': 'Mozambique',
    'RWA': 'Rwanda',
    'SOM': 'Somalia',
    'SSD': 'South Sudan',
    'TZA': 'Tanzania',
    'UGA': 'Uganda',
    'ZWE': 'Zimbabwe',

    # Central Africa (9)
    'AGO': 'Angola',
    'CMR': 'Cameroon',
    'CAF': 'Central African Republic',
    'COG': 'Congo Brazzaville',
    'COD': 'Congo DRC',
    'GNQ': 'Equatorial Guinea',
    'GAB': 'Gabon',
    'STP': 'Sao Tome and Principe',
    'TCD': 'Chad',

    # Southern Africa (6)
    'BWA': 'Botswana',
    'LSO': 'Lesotho',
    'NAM': 'Namibia',
    'ZAF': 'South Africa',
    'SWZ': 'Eswatini',
    'ZMB': 'Zambia',

    # Island Nations (3)
    'COM': 'Comoros',
    'MUS': 'Mauritius',
    'SYC': 'Seychelles',
}

# GADM version 4.1 download URL pattern
# Format: shp (shapefile), gpkg (geopackage), or json (geojson)
GADM_URL = "https://geodata.ucdavis.edu/gadm/gadm4.1/shp/gadm41_{iso3}_shp.zip"

def download_gadm_shapefile(iso3, country_name, output_dir):
    """Download GADM shapefile for a single country"""
    url = GADM_URL.format(iso3=iso3)
    country_dir = os.path.join(output_dir, iso3)

    # Check if already downloaded
    if os.path.exists(country_dir) and len(os.listdir(country_dir)) > 0:
        return True, "Already exists"

    os.makedirs(country_dir, exist_ok=True)

    try:
        response = requests.get(url, timeout=120, stream=True)
        response.raise_for_status()

        # Extract zip file
        with zipfile.ZipFile(BytesIO(response.content)) as zf:
            zf.extractall(country_dir)

        return True, "Downloaded"
    except requests.exceptions.RequestException as e:
        return False, str(e)
    except zipfile.BadZipFile as e:
        return False, f"Bad zip file: {e}"

def main():
    print("=" * 80)
    print("GADM SHAPEFILE DOWNLOADER FOR AFRICA")
    print(f"Downloading administrative boundaries for {len(AFRICAN_COUNTRIES)} countries")
    print(f"Output directory: {OUTPUT_DIR}")
    print("=" * 80)

    successful = []
    failed = []

    for iso3, country_name in tqdm(AFRICAN_COUNTRIES.items(), desc="Downloading"):
        success, message = download_gadm_shapefile(iso3, country_name, OUTPUT_DIR)

        if success:
            successful.append((iso3, country_name, message))
        else:
            failed.append((iso3, country_name, message))
            print(f"\n  Failed: {country_name} ({iso3}): {message}")

        # Rate limiting - be nice to the server
        time.sleep(1)

    print("\n" + "=" * 80)
    print("DOWNLOAD SUMMARY")
    print("=" * 80)
    print(f"Successful: {len(successful)}/{len(AFRICAN_COUNTRIES)}")
    print(f"Failed: {len(failed)}/{len(AFRICAN_COUNTRIES)}")

    if failed:
        print("\nFailed countries:")
        for iso3, name, msg in failed:
            print(f"  - {name} ({iso3}): {msg}")

    # Verify coverage
    print(f"\nCoverage: {100 * len(successful) / len(AFRICAN_COUNTRIES):.1f}%")

    # List downloaded files
    print("\nDownloaded shapefiles:")
    for iso3, name, _ in successful[:5]:
        country_dir = os.path.join(OUTPUT_DIR, iso3)
        files = os.listdir(country_dir)
        shp_files = [f for f in files if f.endswith('.shp')]
        print(f"  {name}: {len(shp_files)} shapefiles")
    if len(successful) > 5:
        print(f"  ... and {len(successful) - 5} more countries")

if __name__ == "__main__":
    main()
