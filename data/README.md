# About the Data

This directory documents the datasets used by the China intercity freight alternative-fuel vehicle (AFV) demand, micro-simulation, and multi-objective optimization framework.

The complete workflow combines four main data groups:

1. freight-demand data used to generate nationwide intercity freight OD demand;
2. highway-network and highway-flow observations used to generate detailed freight trajectories;
3. city-boundary, road-network, and traffic-condition data used to construct distance-correction factors;
4. trajectory, infrastructure-candidate, environmental, vehicle, facility, energy-price, and GHG parameter data used by the AFV simulation and optimization model.

The manuscript models Chinese mainland intercity freight transport for 2019. The freight-demand component integrates an OD-based freight vehicle survey, city-level highway freight volume statistics, highway flow observations, and highway network data. The simulation and optimization component additionally uses vehicle, facility, temperature, energy-price, energy-mix/GHG, and administrative-code data.

## Public sample data and full data

Because several full-scale datasets are large and/or subject to data-access restrictions, the public repository can contain representative sample/demo files rather than the complete datasets. Full datasets may be provided upon reasonable request where licensing and data-use conditions permit.

The recommended repository-level structure is:

```text
data/
├── README.md
├── input/
│   ├── README.md
│   ├── freight_demand/
│   ├── road_network/
│   ├── traffic_volume/
│   │   ├── monthly/
│   │   └── hourly/
│   ├── station_candidates/
│   ├── city_boundary/
│   ├── temperature/
│   └── parameters/
├── interim/
│   ├── README.md
│   ├── freight_od/
│   ├── trajectory_generation/
│   └── distance_correction/
└── output/
    ├── README.md
    ├── figures/
    ├── cache/
    └── intermediate/
```

The current Python scripts still contain Windows absolute paths in several places. When using this GitHub-style `data/` structure, update the paths in:

- `codes/01FreightTripODGeneration.py`;
- `codes/02DetailedFreightTripsGeneration.py`;
- `codes/SimulationOptimization/0preprocess/PreDistanceCorrection.py`;
- `codes/SimulationOptimization/config/parameters.yaml`.

See the README files in each subdirectory for the expected files and their roles.
