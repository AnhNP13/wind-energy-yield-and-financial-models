# Wind energy yield (simulation) models
This brand folder contains the Python models and supporting Excel files used to simulate energy yield and uncertainty for a hypothetical 120 MW onshore wind farm.
The modelling framework includes a Monte Carlo engine that propagates uncertainties from wind measurements and flow‑model predictions through the energy‑yield calculation..

Four measurement‑strategy scenarios are implemented to compare the effects of different spatial sampling approaches on the probability distribution of annual energy production (AEP).

# Contents
## Python scripts
- WIndfarm_simulation_1LIDAR uncertainty.py
- WIndfarm_simulation_2LIDAR uncertainty.py
- WIndfarm_simulation_2UAV uncertainty.py
- WIndfarm_simulation_4UAV uncertainty.py
Each script runs a full Monte Carlo simulation for its respective measurement configuration and outputs the probability distribution of annual energy production.

## Input files
- ws_measure.xlsx (wind‑speed measurement data at farm)
- ws_hor.extrapolation.xlsx (estimated horizontal-extrapolation factors)
- ws_powercurve.xlsx – (V162-6.2 MW turbine power curve)

# Requirements
- Python 3.10+
- Libraries: numpy, pandas, scipy
