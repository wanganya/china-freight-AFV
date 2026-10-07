# City Boundary Data

This directory contains city boundary polygons required by Stage 3.

## Expected file

The current preprocessing code is configured to read:

```text
City.shp
```

with its associated Shapefile components (`.shx`, `.dbf`, `.prj`, and, where applicable, `.cpg`).

## Used by

```text
codes/SimulationOptimization/0preprocess/PreDistanceCorrection.py
```

The polygons are used together with the road network and hourly traffic data to construct city-level and time-dependent distance-correction factors.

A city-boundary sample was not included in the sample-file list supplied for the public repository. If the full boundary data cannot be distributed, keep this README and state that the file is available upon request or can be reconstructed from an appropriate administrative-boundary source.
