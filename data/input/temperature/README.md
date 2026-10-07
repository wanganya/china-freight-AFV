# Seasonal Temperature Data

This directory contains seasonal ambient temperature data used by the vehicle energy-consumption model.

## Expected file

```text
ChinaTemperature4Season.txt
```

The manuscript describes seasonal ambient temperature as an environmental input that captures regional and seasonal differences affecting AFV energy use. The simulation applies separate temperature effects to BEVs and HFCVs.

The current code requires:

- `CityCode`
- `Season`
- `Temperature`

## Used by

```text
codes/SimulationOptimization/main.py
```

through the data loader and preprocessor.

## Sample-data placement

Place:

```text
ChinaTemperature4Season.txt
```

here:

```text
data/input/temperature/ChinaTemperature4Season.txt
```
