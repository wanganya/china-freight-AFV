# Hourly Highway Traffic Volume Data

This directory contains date-specific hourly highway flow observations.

## File naming convention

```text
YYYYMMDD_update.xls
```

Examples for the winter sample:

```text
20190114_update.xls
20190115_update.xls
20190116_update.xls
20190117_update.xls
20190118_update.xls
20190119_update.xls
20190120_update.xls
```

## Used by

- Stage 2: constructs four-hour freight-flow constraints and supports traffic assignment.
- Stage 3: constructs city-level time-dependent distance-correction factors.
- Stage 4: supplies dynamic traffic/congestion information to the simulation.

Stage 2 directly uses fields including:

- `路线简码`
- `小时`
- `中货车加权流量`
- `大货车加权流量`
- `特大货加权流量`
- `集装箱加权流量`
- `拖拉机加权流量`

Stage 4 uses route/hour/congestion information. See `MAINTAINER_NOTES.md` in the documentation bundle for a current column-name consistency issue in the preprocessing code.

## Sample-data placement

Place the seven winter demo files in:

```text
data/input/traffic_volume/hourly/
```

A full four-season run requires all dates specified in the seasonal configuration.
