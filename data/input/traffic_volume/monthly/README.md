# Monthly Highway Flow Distribution

This directory contains the monthly traffic-distribution input used by Stage 2 to convert annual OD expansion coefficients into representative seasonal one-week demand.

## Expected file

```text
MonthDistribution.txt
```

The current code reads the file as a tab-separated UTF-8 table and uses:

- `Month`
- `MonthRatio`

## Used by

```text
codes/02DetailedFreightTripsGeneration.py
```

The sample files listed by the repository owner currently do not include `MonthDistribution.txt`; it should be added here for a complete Stage 2 demonstration.
