# Candidate Refueling Infrastructure Locations

This directory contains candidate locations at which the optimization model may configure alternative-fuel refueling infrastructure.

The national scenario in the thesis considers 50,702 candidate infrastructure locations corresponding to existing highway petrol stations and service areas. Candidate locations can be configured as charging stations (CSs), battery swap stations (BSSs), hydrogen refueling stations (HRSs), or combinations according to the optimization decision variables.

## Full-data filename used by the current configuration

```text
ChinaStationHighway.shp
```

## Public sample filename

```text
ChinaStationHighway_demo.shp
```

The Stage 4 preprocessor requires:

- `ID`
- `LinkRoad`
- `Lat`
- `Lon`
- `CityCode`

## Important Shapefile note

A Shapefile is not a single `.shp` file. Keep all files with the same basename together, normally including at least:

```text
ChinaStationHighway_demo.shp
ChinaStationHighway_demo.shx
ChinaStationHighway_demo.dbf
ChinaStationHighway_demo.prj
```

A `.cpg` file should also be included if used.

## Sample-data placement

Place all Shapefile components in:

```text
data/input/station_candidates/
```
