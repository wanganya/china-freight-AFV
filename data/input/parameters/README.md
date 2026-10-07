# Simulation and Optimization Parameter Data

This directory contains the parameter tables used by the AFV micro-simulation and optimization model.

The manuscript describes these inputs as vehicle technical/cost data, facility technical/cost data, electricity and hydrogen prices, life-cycle GHG data, and other model parameters. The national scenario uses representative BEV and HFCV models, charging/battery-swap/hydrogen-refueling infrastructure options, regional energy prices, and GREET-based GHG inputs.

## Files

| File | Role in the code | Minimum fields explicitly required by the current preprocessor |
|---|---|---|
| `TypicalVehicleBEVHFCV.txt` | BEV/HFCV technical and cost parameters | `Type`, `CargoWeight`, `VehicleCost`, `Capacity`, `BatteryCost`, `LifeSpanVe`, `LifeSpanBat` |
| `ChargingStation.txt` | Charging-station capital cost and GHG parameters | `Type`, `Cost`, `GHG`, `LifeSpanCS` |
| `BatterySwapStation.txt` | Battery-swap station cost/GHG/service parameters | `Type`, `Cost`, `GHG`, `LifeSpanBSS`, `ServiceSpeed` |
| `HydrogenRefuelingStation.txt` | HRS cost/GHG/capacity/efficiency/service parameters | `Type`, `Cost`, `GHG`, `Capacity`, `Efficiency`, `LifeSpanHRS`, `ServiceSpeed` |
| `ChargingPost.txt` | Fast/slow charging-post parameters | `Type`, `Cost`, `Power`, `Efficiency`, `GHG`, `LifeSpanCP` |
| `ElectricityPriceSummer.txt` | Summer electricity price | `CityCode`, `Time`, `Cost` |
| `ElectricityPriceUnsummer.txt` | Non-summer electricity price | `CityCode`, `Time`, `Cost` |
| `HydrogenPriceCG.txt` | Hydrogen price for the coal-gasification baseline pathway | `CityCode`, `Time`, `Cost` |
| `Parameters.txt` | Other scalar simulation parameters | `Parameter`, `Value` |
| `FreeSpeedByCategory.txt` | Free-flow speed by road category | `Category`, `FreeSpeed` |
| `BEVHFCVGHG.txt` | Vehicle life-cycle GHG factors | `ProvinceCode`, `Type`, `GHGs` |
| `BatteryGHG.txt` | Battery life-cycle GHG factors | `ProvinceCode`, `Type`, `GHGs` |
| `ProvinceCityCode.txt` | City-to-province mapping | `ProvinceCode`, `CityCode` |

`ChinaTemperature4Season.txt` is intentionally documented under `/input/temperature` because the code treats it as an environmental dataset rather than a general parameter table.

`CityDistanceCorrection.pkl` is a generated Stage 3 product and is therefore documented under `/interim/distance_correction`, although the current local configuration stores/reads it from the historical parameter directory.

## Sample-data placement

Place the following files in this directory:

```text
BatteryGHG.txt
BatterySwapStation.txt
BEVHFCVGHG.txt
ChargingPost.txt
ChargingStation.txt
ElectricityPriceSummer.txt
ElectricityPriceUnsummer.txt
FreeSpeedByCategory.txt
HydrogenPriceCG.txt
HydrogenRefuelingStation.txt
Parameters.txt
ProvinceCityCode.txt
TypicalVehicleBEVHFCV.txt
```
