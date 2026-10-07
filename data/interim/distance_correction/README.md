# Stage 3 Distance Correction Data

This directory stores the outputs of:

```text
codes/SimulationOptimization/0preprocess/PreDistanceCorrection.py
```

## Main output

```text
CityDistanceCorrection.pkl
```

The file contains city-level base correction factors and date/hour-specific time factors used by the Stage 4 simulation.

The preprocessing script also writes a human-readable sample:

```text
CityDistanceCorrection_sample.json
```

## Inputs used to generate the correction data

- city boundary polygons (`City.shp`);
- processed highway network (`ChinaHighway.gpkg` or an equivalent configured file);
- date-specific hourly traffic-volume files (`YYYYMMDD_update.xls`).

Although the current historical configuration reads `CityDistanceCorrection.pkl` from the parameter directory, it is a generated intermediate dataset and is documented here for a cleaner reproducibility-oriented repository structure.
