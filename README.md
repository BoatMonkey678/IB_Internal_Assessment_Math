# IB Math Analysis and Approaches Internal Assessment

This repository contains all auxiliary resources required for data processing and reproducing results for the IB Math Analysis and Approaches Internal Assessment.

## Included Assets

- Point data sets for vase profiles
    - Filtered scanned images `Vase_0*\scan_filtered.png`
    - Generated files `Vase_0*\_detected_points.csv` and plots `Vase_0*\_detected_points.pdf`
- 3D model of bowl 
    - File `3D_object.3mf`
- Python scripts for data extraction and analysis
    - File `trace_vases_profile.py` 
        - Detects points from scan images `Vase_0*\scan_filtered.png`
        - Stores data in `Vase_0*\_detected_points.csv`
    - File `fit_poly_to_vases.py` 
        - Performs fitting procedure on points detected by `trace_vases_profile.py` in `Vase_0*\_detected_points.csv`
        - Stores results in `Vase_0*\_fit_results.txt`
