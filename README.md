# PV Optimization with PyGAD and pandapower

This project optimizes photovoltaic (PV) placement and sizing  on a power distribution network using a genetic algorithm (PyGAD) and simulates grid behavior with pandapower.

## Stucture 

- `main.py`: entry point
- `optimizer/`
  - `ga_runner.py` – GA configuration, baseline and WTGA implementations
  - `crossover.py` – Custom crossover operator for PV siting and sizing
  - `fitness.py` – Cost-based fitness function with stochastic scenarios
  - `power_model.py` – Simplified PV generation and demand model
- `requirements.txt`: reproducible environment

## To run
```bash
pip install -r requirements.txt
python main.py