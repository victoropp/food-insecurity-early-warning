# Installation Guide

**Food Insecurity Early Warning System**
**Author**: Victor Collins Oppon, MSc Data Science, Middlesex University 2025

---

## System Requirements

- **Operating System**: Windows 10+, macOS 10.14+, or Linux (Ubuntu 20.04+)
- **Python**: 3.9, 3.10, or 3.11 (tested on Python 3.13)
- **RAM**: Minimum 8GB, **recommended 16GB** for full pipeline
- **Storage**: ~5GB free space (code + processed data)
- **Additional**: ~900MB for data archive (download separately from Zenodo)

---

## Step 1: Clone Repository

```bash
git clone https://github.com/yourusername/food-insecurity-early-warning.git
cd food-insecurity-early-warning
```

Or download ZIP and extract:
```bash
unzip food-insecurity-early-warning-main.zip
cd food-insecurity-early-warning-main
```

---

## Step 2: Create Python Environment

### Option A: Using venv (built-in)

```bash
# Create virtual environment
python -m venv env

# Activate environment
# On Windows:
env\Scripts\activate

# On macOS/Linux:
source env/bin/activate
```

### Option B: Using conda (recommended for geospatial packages)

```bash
# Create environment
conda create -n foodinsecurity python=3.11

# Activate environment
conda activate foodinsecurity
```

---

## Step 3: Install Dependencies

### Quick Install (pip)

```bash
pip install -r requirements.txt
```

### Detailed Install (if quick install fails)

**Core packages**:
```bash
pip install pandas>=2.0.0 numpy>=1.24.0 pyarrow>=12.0.0
pip install scikit-learn>=1.3.0 xgboost>=2.0.0
pip install matplotlib>=3.7.0 seaborn>=0.12.0
```

**Geospatial packages** (may require conda on Windows):
```bash
# If pip fails on Windows, use conda:
conda install -c conda-forge geopandas shapely fiona

# Otherwise:
pip install geopandas>=0.13.0 shapely>=2.0.0 fiona>=1.9.0
```

**Advanced features** (optional):
```bash
pip install hmmlearn>=0.3.0  # Hidden Markov Models
pip install pydmd>=0.4.0  # Dynamic Mode Decomposition
pip install statsmodels>=0.14.0  # Mixed-effects models
pip install shap>=0.42.0  # Model interpretability
```

**Jupyter notebooks** (optional):
```bash
pip install jupyter jupyterlab notebook
```

---

## Step 4: Download Data Archive

Data archive (~900MB) is hosted on Zenodo for reproducibility.

### Option A: Automated Download (recommended)

```bash
# Download script (to be created)
python scripts/download_data.py
```

### Option B: Manual Download

1. Visit Zenodo: https://doi.org/10.5281/zenodo.XXXXXXX
2. Download `data_archive.zip` (900 MB)
3. Extract to `data/` directory:

```bash
# On Windows (PowerShell):
Expand-Archive -Path data_archive.zip -DestinationPath data/

# On macOS/Linux:
unzip data_archive.zip -d data/
```

**Expected structure after extraction**:
```
data/
├── external/
│   ├── ipc/ipcFic_Africa_Current_Only.csv (24 MB)
│   ├── gdelt/african_gkg_locations_aligned.parquet (613 MB)
│   └── shapefiles/ (162 MB)
├── interim/
│   ├── stage1/ (41 MB)
│   └── stage2/ (30 MB)
└── processed/
```

---

## Step 5: Verify Installation

```bash
python -c "
import sys
print(f'Python: {sys.version}')

import pandas as pd
print(f'pandas: {pd.__version__}')

import geopandas as gpd
print(f'geopandas: {gpd.__version__}')

import xgboost as xgb
print(f'xgboost: {xgb.__version__}')

from config import BASE_DIR, INTERIM_DATA_DIR, RESULTS_DIR
print(f'Config: OK')
print(f'BASE_DIR: {BASE_DIR}')

from src.stratified_spatial_cv import create_stratified_spatial_folds
print(f'Custom modules: OK')

print('')
print('✓ ALL CHECKS PASSED - READY TO RUN!')
"
```

**Expected output**:
```
Python: 3.11.x
pandas: 2.x.x
geopandas: 0.13.x
xgboost: 2.x.x
Config: OK
BASE_DIR: C:\path\to\dissertation_submission
Custom modules: OK

✓ ALL CHECKS PASSED - READY TO RUN!
```

---

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'geopandas'"

**Solution** (Windows):
```bash
conda install -c conda-forge geopandas
```

**Solution** (macOS/Linux):
```bash
# Install GDAL dependencies first
sudo apt-get install gdal-bin libgdal-dev  # Ubuntu/Debian
brew install gdal  # macOS

# Then install geopandas
pip install geopandas
```

### Issue: "FileNotFoundError: data/external/ipc/..."

**Solution**: Ensure data archive was extracted to correct location:
```bash
# Check if IPC file exists
ls data/external/ipc/ipcFic_Africa_Current_Only.csv

# If missing, re-download and extract data archive
```

### Issue: "Cannot import hmmlearn"

**Solution** (optional dependency):
```bash
pip install hmmlearn

# If fails:
conda install -c conda-forge hmmlearn
```

### Issue: "Memory error during model training"

**Cause**: Insufficient RAM (less than 16GB)

**Solutions**:
- Close other applications
- Use Stage 1 baseline only (requires ~8GB)
- Reduce GridSearchCV parameter grid in scripts
- Run on machine with more RAM

### Issue: "GDAL/GEOS errors on Windows"

**Solution**: Use conda for all geospatial packages:
```bash
conda create -n foodinsecurity python=3.11
conda activate foodinsecurity
conda install -c conda-forge geopandas shapely fiona rtree
pip install -r requirements.txt  # Install remaining packages
```

### Issue: "Long path errors on Windows"

**Solution**: Enable long path support
1. Run as Administrator: `gpedit.msc`
2. Navigate to: Computer Configuration → Administrative Templates → System → Filesystem
3. Enable "Enable Win32 long paths"
4. Restart terminal

---

## Platform-Specific Notes

### Windows

- Use **Anaconda** for easier geospatial package installation
- Ensure long path support enabled (see troubleshooting above)
- Use PowerShell or Git Bash (not Command Prompt)
- Path separators: use forward slashes `/` in Python code

### macOS

- May need Xcode command line tools:
  ```bash
  xcode-select --install
  ```

- Install GDAL via Homebrew if geopandas fails:
  ```bash
  brew install gdal
  pip install geopandas
  ```

### Linux (Ubuntu/Debian)

- Install system dependencies first:
  ```bash
  sudo apt-get update
  sudo apt-get install python3-dev libgdal-dev libgeos-dev \
                       libspatialindex-dev libproj-dev
  ```

- Then install Python packages:
  ```bash
  pip install -r requirements.txt
  ```

---

## Testing Installation

### Quick Test (5 minutes)

Test that core functionality works:

```bash
# Test config imports
python -c "from config import *; print('Config OK')"

# Test data loading
python -c "
import pandas as pd
from config import IPC_FILE
if IPC_FILE.exists():
    df = pd.read_csv(IPC_FILE, nrows=5)
    print(f'IPC data: {len(df)} rows (sample)')
    print('Data OK')
else:
    print('WARNING: IPC data not found - download data archive')
"

# Test spatial CV module
python -c "
from src.stratified_spatial_cv import create_stratified_spatial_folds
print('Spatial CV module OK')
"
```

### Full Test (30 minutes)

Run a complete pipeline test:

```bash
# Test Stage 1 baseline (requires data archive)
cd scripts/03_stage1_baseline
python 07_stage1_logistic_regression.py --test-mode

# Test Stage 2 model (requires Stage 1 results)
cd ../05_stage2_model_training/xgboost_models
python 01_xgboost_basic_WITH_AR_FILTER_OPTIMIZED.py --test-mode
```

---

## Verification Checklist

After installation, verify:

- [ ] Python 3.9+ installed and activated in virtual environment
- [ ] All packages from requirements.txt installed successfully
- [ ] Data archive downloaded and extracted to `data/` directory
- [ ] IPC data file exists: `data/external/ipc/ipcFic_Africa_Current_Only.csv`
- [ ] GDELT data file exists: `data/external/gdelt/african_gkg_locations_aligned.parquet`
- [ ] Config imports work: `from config import BASE_DIR`
- [ ] Spatial CV imports work: `from src.stratified_spatial_cv import create_stratified_spatial_folds`
- [ ] Verification script runs successfully (Step 5)
- [ ] Can import geopandas, xgboost, sklearn without errors

---

## Next Steps

After successful installation:

1. **Read REPRODUCTION_GUIDE.md** for step-by-step pipeline execution
2. **Explore Jupyter notebooks** in `notebooks/` directory for interactive tutorials
3. **Run quick test**:
   ```bash
   python scripts/06_cascade_analysis/05_cascade_ensemble_optimized_production.py --dry-run
   ```
4. **Check dissertation PDF** in `dissertation/main.pdf` for methodology

---

## Getting Help

**Installation issues**:
- Check [GitHub Issues](https://github.com/yourusername/food-insecurity-early-warning/issues)
- Include output of verification script (Step 5)
- Specify OS, Python version, error message

**Contact**:
- Email: [your-email@example.com]
- GitHub: [@yourusername]

---

## Optional: Development Setup

For contributors or extended development:

```bash
# Install development dependencies
pip install pytest pytest-cov black flake8

# Run tests
pytest tests/

# Format code
black scripts/ src/

# Check style
flake8 scripts/ src/
```

---

*Installation typically takes 15-30 minutes including data download. Most issues relate to geospatial package dependencies - use conda if pip fails.*
