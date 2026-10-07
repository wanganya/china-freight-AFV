# china-freight-AFV
# Overview
This repository contains all codes of the paper - **Spatially Differentiated Electrification Strategies for China’s Intercity Freight Transport: Multi-scale Actionable Insights from Nationwide Micro-simulations**

Note that the full dataset can be requested through Email.

# Requirements and Installation
The whole analysis-related codes should run with a **Python** environment, regardless of operating systems theoretically. We successfully execute all the codes in both Windows (Win10, Win11) machines and a macOS (Sequoia 15.2) machine. More detailed info is as below:

# Prerequisites
It is highly recommended to install and use the following versions of python/packages to run the codes:

## 01FreightTripODGeneration.py
   - ``python``==3.9
   - ``numpy``==1.24.3
   - ``pandas``==2.0.3
   - ``psutil``==5.9.5
     
## 02DetailedFreightTripsGeneration.py
   - ``python``==3.9
   - ``qgis``==3.34.1
   - ``numpy``==1.26.4
   - ``pandas``==2.2.3
   - ``tqdm``==4.67.1
   - ``geopandas``==1.0.1
   - ``openpyxl``==3.1.5
     
## codes in the dir./SimulationOptimization/...
   - ``python``==3.9
   - ``numpy``>=1.20.0
   - ``pandas``>=1.3.0
   - ``matplotlib``>=3.4.0
   - ``seaborn``>=0.11.0
   - ``geopandas``>=0.10.0
   - ``PyYAML``>=6.0
   - ``psutil``>=5.9.0
   - ``openpyxl``>=3.0.9
   - ``pyarrow``>=6.0.0
   - ``shapely``>=1.8.0
   - ``fiona``>=1.8.20
   - ``rtree``>=1.0.0
     
# Installation
It is highly recommended to download AnaConda to create/manage Python environments. You can create a new Python environment and install required aforementioned packages via both the GUI or Command Line. Typically, the installation should be prompt (around 10-20 min from a "_clean_" machine to "_ready-to-use_" machine, but highly dependent on the Internet speed).

- via **Anaconda GUI**
  1. Open the Anaconda
  2. Find and click "_Environments_" at the left sidebar
  3. Click "_Create_" to create a new Python environment
  4. Select the created Python environment in the list, and then search and install all packages one by one.
     
- via **Command Line** (using **_Terminal_** for macOS machine and **_Anaconda Prompt_** for Windows machine, respectively)
  1. Create your new Python environment )
     ```
     conda create --name <input_your_environment_name> python=3.10.6
     ```
  2. Activate the new environment 
     ```
     conda activate <input_your_environment_name>
     ```
  3. Install all packages one by one 
     ```
     conda install <package_name>=<specific_version>
     ```

# Usage

## 1. Download the repository
Clone or download this repository to your local disk.
## 2. Prepare datasets
The complete datasets required for running the framework are available upon request, see [Overview](https://github.com/wanganya/china-freight-AFV/blob/main/README.md).

After downloading the datasets, organize the folders according to the paths defined in the scripts and configuration file.
The required dataset structure is:
``` text
Dataset/
│
├── ChinaTrip/
│   └── Output/
│       ├── EFVTrip2019CoorReviseCityDiffODCargo.txt
│       ├── 2019CityFreVoluaddCodeRevisedDiffODFinalFinal.txt
│       └── Trajectory/
│           ├── Trajectory_Winter_20190114_20190120_sort.txt
│           ├── Trajectory_Spring_20190418_20190424_sort.txt
│           ├── Trajectory_Summer_20190716_20190722_sort.txt
│           └── Trajectory_Autumn_20191106_20191112_sort.txt
│
├── Road/
│   └── OSMOutput/
│       ├── OSM_Highway_no2nd_Code.gpkg
│       ├── 001VolumeMonUpdate/
│       │   └── MonthDistribution.txt
│       └── 002VolumeDayUpdate/
│
├── ChinaStation/
│   └── Output/
│       └── ChinaStationHighway.shp
│
├── ChinaTem/
│   └── NOAA_NCEI/
│       └── Output/
│           └── ChinaTemperature4Season.txt
│
└── ChinaEFVSimulation/
    ├── ChinaEFVParameter/
    │   ├── TypicalVehicleBEVHFCV.txt
    │   ├── ChargingStation.txt
    │   ├── BatterySwapStation.txt
    │   ├── HydrogenRefuelingStation.txt
    │   ├── ChargingPost.txt
    │   ├── ElectricityPriceSummer.txt
    │   ├── ElectricityPriceUnsummer.txt
    │   ├── HydrogenPriceCG.txt
    │   ├── Parameters.txt
    │   ├── FreeSpeedByCategory.txt
    │   ├── BEVHFCVGHG.txt
    │   ├── BatteryGHG.txt
    │   └── ProvinceCityCode.txt
    │
    └── Output/
```
Before running the simulation framework, modify the corresponding paths
in:
``` text
./codes/SimulationOptimization/config/parameters.yaml
```
to match your local dataset locations.
## 3. Run the workflow sequentially

The framework consists of four sequential stages. Intermediate files
generated in each stage are used as inputs for the following stage.

All executable scripts are located in:

``` text
./codes/
```

### Step 1. Generate freight trip OD demand

Run:

``` bash
python ./codes/01FreightTripODGeneration.py
```

This script generates freight origin-destination (OD) demand by
processing truck OD records and city freight volume data.

The generated OD expansion file is used as input for the next stage.

### Step 2. Generate detailed freight trajectories

Run:

``` bash
python ./codes/02DetailedFreightTripsGeneration.py
```

This script converts OD demand into detailed freight trajectories by
considering road networks, traffic volume, and seasonal variations.

The generated trajectory files are stored in:

``` text
ChinaTrip/Output/Trajectory/
```

### Step 3. Distance correction preprocessing

Run:

``` bash
python ./codes/SimulationOptimization/0preprocess/PreDistanceCorrection.py
```

This script calculates distance correction coefficients required by the
simulation framework.

The generated correction file is used in the simulation optimization
stage.

### Step 4. Run simulation-based optimization framework

Run:

``` bash
python ./codes/SimulationOptimization/main.py
```

This stage performs:

-   freight vehicle operation simulation;
-   BEV and HFCV energy consumption simulation;
-   charging, battery swapping, and hydrogen refueling service
    simulation;
-   multi-objective optimization of vehicle fleet composition and
    infrastructure configuration.

The required input datasets are loaded according to:

``` text
./codes/SimulationOptimization/config/parameters.yaml
```


## 4. Outputs

The simulation and optimization results are saved in:

``` text
ChinaEFVSimulation/Output/
```

The output files include:

``` text
Output/
│
├── optimization_history.csv
├── 01ElectrifiedVehicle.txt
├── 02Station.txt
├── 04Objective.txt
├── 07DrivingTimeStatistics.txt
│
├── figures/
│
├── cache/
│
└── intermediate/
    └── summary_iter_*.json
```

The generated outputs can be used to reproduce the simulation results,
optimization results, and figures presented in the manuscript.


# Contact
- Leave questions in [Issues on GitHub](https://github.com/wanganya/china-freight-AFV/issues)
- Get in touch with the Corresponding Author: [Dr. Chengxiang Zhuge](mailto:chengxiang.zhuge@polyu.edu.hk) or visit our research group website: [The TIP](https://thetipteam.editorx.io/website) for more information

# License
This repository is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
