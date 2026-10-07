# Highway Network Data

This directory contains highway-network data used by Stages 2-4.

The manuscript states that the Chinese mainland highway network was extracted from OpenStreetMap (OSM) in July 2024 and contains more than six million road links, with information on road classification, connectivity, geometry, and other network attributes.

## Stage 2 network

The current Stage 2 code is configured to read:

```text
OSM_Highway_no2nd_Code.gpkg
```

This network is used by the QGIS-based route-generation and traffic-assignment procedure.

## Stage 3 and Stage 4 network

The current Stage 3/4 code is configured to read a processed network named:

```text
ChinaHighway.gpkg
```

A public sample can be named:

```text
ChinaHighway_demo.gpkg
```

and placed in this directory. Update the relevant path(s) before execution.

The Stage 4 preprocessor requires the following fields:

- `ID`
- `Code`
- `fclass`
- `Long_meter`
- `CityCode`

## Sample-data placement

Place:

```text
ChinaHighway_demo.gpkg
```

here:

```text
data/input/road_network/ChinaHighway_demo.gpkg
```

Do not assume that the demo GeoPackage can replace the Stage 2 assignment network unless it contains the fields and network topology required by `02DetailedFreightTripsGeneration.py`.
