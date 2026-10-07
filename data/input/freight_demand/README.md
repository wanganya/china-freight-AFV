# Freight Demand Input Data

This directory contains the two principal Stage 1 datasets used to generate intercity freight OD demand.

## Expected files

```text
EFVTrip2019CoorReviseCityDiffODCargo.txt
2019CityFreVoluaddCodeRevisedDiffODFinalFinal.txt
```

## 1. OD-based freight vehicle survey data

`EFVTrip2019CoorReviseCityDiffODCargo.txt`

The manuscript describes this source as the 2019 Special Survey on Road Freight Transport Volume conducted by China's Ministry of Transport. The study dataset contains 605,980 freight trips from 77,936 vehicles. Records include vehicle and trip attributes such as cargo/rated load and origin/destination information.

The Stage 1 code uses, among others, the following fields:

- `核定载质量`
- `OLon`
- `OLat`
- `DLon`
- `DLat`
- `OCityCode`
- `DCityCode`

## 2. City-level highway freight volume statistics

`2019CityFreVoluaddCodeRevisedDiffODFinalFinal.txt`

The manuscript describes the city-level freight-volume statistics as 2019 data from the China City Statistical Yearbook. They are used to constrain/expand OD demand at the city level.

The Stage 1 code uses:

- `City`
- `CityCode`
- `2019FreightVolumeRevisedDiffOD(t)`
- `InterCityOVolume`
- `InterCityDVolume`

## Used by

```text
codes/01FreightTripODGeneration.py
```

## Stage 1 output

The Stage 1 result is:

```text
EFVTrip2019CoorReviseCityDiffODExpandImproved.txt
```

The recommended location for that generated file is:

```text
data/interim/freight_od/
```
