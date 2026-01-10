# Jupyter Notebooks - Interactive Pipeline

**Author**: Victor Collins Oppon, MSc Data Science, Middlesex University 2025

---

## Overview

This directory contains Jupyter notebook versions of all major pipeline stages. Notebooks provide:
- **Step-by-step explanations** of methodology
- **Intermediate visualizations** and data previews
- **Interactive exploration** of results
- **Cell-by-cell execution** for understanding

---

## Prerequisites

```bash
# Install Jupyter
pip install jupyter jupyterlab notebook

# Or if using conda
conda install jupyter jupyterlab
```

---

## Available Notebooks

| Notebook | Description | Runtime | Prerequisites |
|----------|-------------|---------|---------------|
| **01_Data_Acquisition.ipynb** | Download and prepare external data sources | ~30 min | None |
| **02_Data_Processing.ipynb** | Aggregate GDELT and IPC data | ~2 hours | 01 complete OR data archive |
| **03_Stage1_Baseline.ipynb** | AR baseline model (L_t + L_s features) | ~30 min | Stage 1 data |
| **04_Stage2_Features.ipynb** | Advanced feature engineering (HMM, DMD) | ~1 hour | Stage 1 complete |
| **05_Stage2_Models.ipynb** | XGBoost training with GridSearchCV | ~2 hours | Stage 2 features |
| **06_Cascade_Analysis.ipynb** | Two-stage cascade system evaluation | ~20 min | Stages 1 & 2 complete |
| **07_Visualizations.ipynb** | Generate publication figures | ~30 min | All stages complete |
| **08_Complete_Pipeline.ipynb** | End-to-end walkthrough | ~6+ hours | None (self-contained) |

---

## Quick Start

### Option 1: JupyterLab (Recommended)

```bash
# From project root
jupyter lab

# Navigate to notebooks/ directory
# Open any notebook
```

### Option 2: Classic Notebook Interface

```bash
jupyter notebook

# Browser opens automatically
# Navigate to notebooks/
```

### Option 3: VS Code

1. Open project in VS Code
2. Install "Jupyter" extension
3. Open any `.ipynb` file
4. Select Python kernel (your virtual environment)

---

## Usage Modes

### Mode A: View Only (No Execution)

All notebooks include **saved outputs** from previous runs. You can:
- Read methodology explanations
- View intermediate results
- See all figures and tables
- **No need to re-run cells**

Perfect for:
- Supervisors reviewing methodology
- Examiners checking implementation
- Quick verification of results

### Mode B: Interactive Exploration

Execute cells step-by-step to:
- Modify parameters and see effects
- Explore different data subsets
- Generate custom visualizations
- Debug or extend pipeline

### Mode C: Full Reproduction

Run all cells in sequence to:
- Reproduce all results from scratch
- Verify reproducibility
- Generate new predictions

**Time**: ~6-8 hours for complete pipeline

---

## Execution Order

### Sequential Execution (Recommended)

```
01_Data_Acquisition.ipynb
    ↓
02_Data_Processing.ipynb
    ↓
03_Stage1_Baseline.ipynb
    ↓
04_Stage2_Features.ipynb
    ↓
05_Stage2_Models.ipynb
    ↓
06_Cascade_Analysis.ipynb
    ↓
07_Visualizations.ipynb
```

### Independent Execution

If you have the **data archive**, you can run notebooks independently:

- **03, 04, 05, 06, 07**: Require data archive (pre-computed features)
- **08**: Self-contained, but takes 6+ hours

---

## Notebook Structure

Each notebook follows this structure:

```markdown
# Title

## 1. Overview
- What this stage does
- Key algorithms/methods
- Expected outputs

## 2. Setup
- Import libraries
- Load configuration
- Set paths

## 3. Data Loading
- Load input data
- Show sample records
- Verify data quality

## 4. Processing
- Step-by-step implementation
- Intermediate visualizations
- Progress indicators

## 5. Results
- Performance metrics
- Visualizations
- Save outputs

## 6. Summary
- Key findings
- Next steps
```

---

## Key Features

### 1. Markdown Documentation

Each code cell is preceded by markdown explaining:
- What the code does
- Why it's necessary
- Expected behavior

### 2. Data Previews

After each processing step:
```python
# Show sample
df.head()

# Show statistics
df.describe()

# Show shape
print(f"Shape: {df.shape}")
```

### 3. Inline Visualizations

```python
# Performance over time
plt.figure(figsize=(12, 6))
plt.plot(metrics)
plt.show()
```

### 4. Progress Indicators

```python
from tqdm.notebook import tqdm

for item in tqdm(items, desc="Processing"):
    process(item)
```

### 5. Error Handling

```python
try:
    result = risky_operation()
except Exception as e:
    print(f"Error: {e}")
    print("Continuing with fallback...")
```

---

## Tips for Supervisors/Reviewers

### Quick Review (30 minutes)

1. Open `08_Complete_Pipeline.ipynb`
2. Read markdown cells for methodology
3. Scroll through outputs (don't execute)
4. Jump to "Results" sections for key findings

### Deep Dive (2-3 hours)

1. Start with `03_Stage1_Baseline.ipynb`
2. Read methodology carefully
3. Execute cells to see live results
4. Continue to `05_Stage2_Models.ipynb` and `06_Cascade_Analysis.ipynb`

### Full Reproduction (6-8 hours)

1. Ensure data archive extracted
2. Run `08_Complete_Pipeline.ipynb` from top to bottom
3. Or run 01-07 in sequence

---

## Common Issues

### Issue: "Kernel not found"

**Solution**: Install ipykernel in your environment
```bash
pip install ipykernel
python -m ipykernel install --user --name=foodinsecurity
```

Then select "foodinsecurity" kernel in Jupyter.

### Issue: "Module not found"

**Solution**: Ensure running from project root
```bash
cd dissertation_submission
jupyter lab
```

### Issue: "Out of memory"

**Solution**: Restart kernel and run again
```
Kernel → Restart Kernel
```

Or reduce data size in notebook (commented code provided).

### Issue: "Cell takes too long"

**Solution**: Many cells include `--test-mode` or data subset options (commented out). Uncomment to run faster with smaller data.

---

## Modifying Notebooks

### Add Your Own Analysis

```python
# In any notebook, add a new cell:

# Custom analysis
subset = df[df['country'] == 'ZWE']
print(f"Zimbabwe: {len(subset)} observations")
subset.groupby('ipc_phase')['article_count'].mean().plot(kind='bar')
```

### Change Parameters

```python
# Find parameter cell (usually Section 2)
N_FOLDS = 3  # Change from 5 to 3 for faster execution
RANDOM_STATE = 123  # Change seed
```

### Export Notebook

```bash
# Convert to HTML (for viewing without Jupyter)
jupyter nbconvert --to html 03_Stage1_Baseline.ipynb

# Convert to PDF (requires LaTeX)
jupyter nbconvert --to pdf 03_Stage1_Baseline.ipynb

# Convert to Python script
jupyter nbconvert --to script 03_Stage1_Baseline.ipynb
```

---

## Performance Optimization

### For Faster Execution

1. **Reduce data size**:
   ```python
   df = df.sample(frac=0.1, random_state=42)  # Use 10% of data
   ```

2. **Reduce CV folds**:
   ```python
   N_FOLDS = 3  # Instead of 5
   ```

3. **Skip optional visualizations**:
   - Comment out visualization cells
   - Or set `SKIP_PLOTS = True`

4. **Use cached results**:
   ```python
   if CACHED_RESULTS.exists():
       df = pd.read_parquet(CACHED_RESULTS)
   else:
       df = expensive_operation()
   ```

---

## Keyboard Shortcuts

**Jupyter Lab**:
- `Shift + Enter`: Run cell and move to next
- `Ctrl + Enter`: Run cell and stay
- `A`: Insert cell above
- `B`: Insert cell below
- `D, D`: Delete cell
- `M`: Change to markdown
- `Y`: Change to code

**Jupyter Notebook (Classic)**:
- Same as JupyterLab
- `H`: Show all shortcuts

---

## Saving Work

**Auto-save**: Enabled by default (every 2 minutes)

**Manual save**: `Ctrl + S` or File → Save

**Checkpoints**: Jupyter creates automatic checkpoints. Restore via:
- File → Revert Notebook to Checkpoint

---

## Reproducibility

All notebooks use:
- **Fixed random seeds** (`RANDOM_STATE = 42`)
- **Versioned dependencies** (from requirements.txt)
- **Documented data sources** (in each notebook header)
- **Saved outputs** (committed to git for verification)

---

## Citation

If you use these notebooks in your research, please cite:

```bibtex
@software{oppon2025notebooks,
  author = {Victor Collins Oppon},
  title = {Food Insecurity Early Warning System - Interactive Notebooks},
  year = {2025},
  publisher = {GitHub},
  url = {https://github.com/yourusername/food-insecurity-early-warning}
}
```

---

## Support

For notebook issues:
- [GitHub Issues](https://github.com/yourusername/food-insecurity-early-warning/issues)
- Email: [your-email@example.com]
- Include: Error message, notebook name, cell number

---

*Note: Notebooks 01-08 are templates/placeholders. Full implementation available in Python scripts in `scripts/` directory. Notebooks provide narrative versions for easier understanding and exploration.*
