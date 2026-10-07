# Description of the Input Data Folder

This folder contains the external and baseline input datasets used by the four-stage workflow.

## Structure

- `/freight_demand`: OD-based freight survey records and city-level freight volume statistics used in Stage 1.
- `/road_network`: highway network data used in trajectory generation, distance correction, and simulation.
- `/traffic_volume`: monthly and hourly highway flow observation data used in Stages 2-4.
- `/station_candidates`: candidate charging/battery-swap/hydrogen-refueling locations used by the optimization model.
- `/city_boundary`: city boundary data used to calculate city-level distance correction factors.
- `/temperature`: seasonal ambient temperature data used in energy-consumption simulation.
- `/parameters`: vehicle, facility, energy-price, GHG, free-speed, and administrative mapping parameter files.

The public repository may contain sample/demo versions of these files. The full-scale datasets may be provided upon request where permitted.
