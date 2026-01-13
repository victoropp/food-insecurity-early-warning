# FEWSNET IPC Dataset - Final Selection

## Active Dataset for Analysis

**File:** `ipcFic_Africa_Current_Only.csv`

### Dataset Specifications:
- **Records:** 55,129
- **Countries:** 24 African countries
- **Time Period:** 2021-02-01 to 2024-11-01 (3 years, 9 months)
- **Scenario Type:** Current Situation only (projections removed)
- **IPC Versions:** IPC 3.1 (66.5%), IPC 3.0 (32.8%), IPC Highest Household (0.7%)
- **Geographic Units:** 1,996 unique sub-national units
- **Classified Records:** 53,594 (Phases 1-5)

### Countries Covered:
1. Angola
2. Burkina Faso
3. Burundi
4. Cameroon
5. Central African Republic
6. Chad
7. Democratic Republic of the Congo
8. Ethiopia
9. Kenya
10. Lesotho
11. Madagascar
12. Malawi
13. Mali
14. Mauritania
15. Mozambique
16. Niger
17. Nigeria
18. Rwanda
19. Somalia
20. South Sudan
21. Sudan
22. Togo
23. Uganda
24. Zimbabwe

### Phase Distribution (2021-2024):
- **Phase 1 (Minimal):** 21,760 records (40.60%)
- **Phase 2 (Stressed):** 17,536 records (32.72%)
- **Phase 3 (Crisis):** 11,969 records (22.33%)
- **Phase 4 (Emergency):** 2,322 records (4.33%)
- **Phase 5 (Famine):** 7 records (0.01%)

### Why This Dataset Was Selected:

This dataset was chosen after comprehensive analysis and comparison of multiple FEWSNET IPC sources:

1. **Most Comprehensive:** Contains 100% of records from alternative datasets PLUS 440 additional records
2. **Broader Coverage:** Includes 6 additional African countries (Angola, CAR, Lesotho, Mauritania, Rwanda, Togo)
3. **Most Current:** Data extends through November 2024 (vs October 2024 in alternatives)
4. **Clean and Standardized:**
   - Only African countries
   - Only Current Situation scenarios (no projections)
   - Standardized country naming (DRC)
   - No duplicate or redundant records

5. **Research-Ready:**
   - Optimal for 2021-2024 dissertation analysis period
   - Aligned with GDELT data temporal coverage
   - Sub-national geographic granularity for spatial analysis
   - Consistent IPC methodology (versions 3.0 and 3.1)

### Data Cleaning Applied:
1. ✅ Removed non-African countries (Afghanistan, Lebanon, Yemen)
2. ✅ Retained only "Current Situation" scenarios
3. ✅ Standardized DRC naming convention
4. ✅ Verified no missing critical fields (country, dates, geographic units)

### Archive Location:

Previous datasets archived in: `D:\GDELT_Africa_Extract\FEWSNET IPC\archive\`

**Archived files:**
1. `FEWSNET Food Insecurity Dataset.csv` (214,031 records, 2011-2024, 37 countries)
2. `ipcFic_data.csv` (537,114 records, includes projections)
3. `ipcFic_data_current_only.csv` (60,986 records, includes non-African)
4. `FEWSNET_2021_2024_only.csv` (67,223 records, includes non-African)
5. `FEWSNET_2021_2024_Africa_Current_Only.csv` (54,689 records, subset of active dataset)

### Next Steps:

This dataset is now ready for:
- Integration with GDELT event data
- Temporal analysis (2021-2024 trends)
- Spatial analysis (sub-national patterns)
- Phase transition modeling
- Early warning correlation studies
- Dissertation statistical analysis

---

**Date Prepared:** November 28, 2025
**Analysis Scripts:**
- `analyze_and_clean_datasets.py`
- `final_cleaned_comparison.py`
- `compare_filtered_datasets.py`

**Documentation:**
- `Cleaned_Africa_Datasets_Final_Comparison.txt`
- `FEWSNET_Dataset_Documentation_2021_2024.md`
