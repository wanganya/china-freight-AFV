# Stage 1 Expanded Freight OD Data

This directory stores the main output of:

```text
codes/01FreightTripODGeneration.py
```

## Main file

```text
EFVTrip2019CoorReviseCityDiffODExpandImproved.txt
```

Stage 1 starts from the OD-based freight vehicle survey and city-level freight volume statistics and adds an estimated `Coefficient` to the freight-trip records.

The current Stage 2 code now reads this exact filename:

```text
EFVTrip2019CoorReviseCityDiffODExpandImproved.txt
```

so the Stage 1 -> Stage 2 filename interface is aligned in the current code package.
