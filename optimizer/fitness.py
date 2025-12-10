import numpy as np
from .power_model import simulate_pv_generation

# Configuration (you can tune these)
N_ZONES = 5

# Cost per kW installed at each zone (dummy example)
CAPEX_PER_KW = np.array([800, 850, 820, 900, 870], dtype=float)  # € per kW

# Import and export prices (dummy, per kWh)
IMPORT_PRICE = 0.25   # €/kWh
EXPORT_PRICE = 0.10   # €/kWh

# Regularization weight for number of zones used
LAMBDA_ZONES = 50.0   # €/zone, tune this

# Example: demand & weather scenarios
# You should build these from your data / AR(1) model, etc.
# For now, think of them as a list of dicts:
#   scenario["demand"]  -> np.array shape (N_ZONES,)
#   scenario["weather"] -> anything needed by simulate_pv_generation

DEMAND_WEATHER_SCENARIOS = []  # fill this in your setup code


def fitness_func(ga_instance, solution, solution_idx):
    """
    Fitness = - (capex + expected_energy_cost + penalties)

    solution structure:
        [applyPV[0..N_ZONES-1], capacity[0..N_ZONES-1]]
    """

    # Split chromosome
    apply_pv = np.array(solution[:N_ZONES], dtype=int)
    capacity = np.array(solution[N_ZONES:], dtype=float)

    # Enforce consistency: no capacity if no PV
    capacity = np.where(apply_pv == 1, capacity, 0.0)

    # ---- 1. CAPEX ----
    capex = np.sum(apply_pv * capacity * CAPEX_PER_KW)

    # ---- 2. Expected energy cost over scenarios ----
    if len(DEMAND_WEATHER_SCENARIOS) == 0:
        # Fallback: no scenarios defined → just penalize capex
        expected_energy_cost = 0.0
    else:
        total_energy_cost = 0.0

        for scen in DEMAND_WEATHER_SCENARIOS:
            demand = scen["demand"]            # shape (N_ZONES,)
            weather = scen["weather"]          # whatever your power model expects

            # Compute PV generation for this scenario
            # Implement this in power_model.py
            generation = simulate_pv_generation(capacity, weather)  # shape (N_ZONES,)

            # Net demand (positive → need import, negative → surplus)
            net = np.sum(demand - generation)

            energy_import = max(net, 0.0)
            energy_surplus = max(-net, 0.0)

            scenario_cost = IMPORT_PRICE * energy_import - EXPORT_PRICE * energy_surplus
            total_energy_cost += scenario_cost

        expected_energy_cost = total_energy_cost / len(DEMAND_WEATHER_SCENARIOS)

    # ---- 3. Penalty: number of zones with PV ----
    num_zones_used = np.sum(apply_pv)
    penalty_zones = LAMBDA_ZONES * num_zones_used

    # ---- 4. Total cost and fitness ----
    total_cost = capex + expected_energy_cost + penalty_zones

    fitness = -total_cost   # GA maximizes fitness
    return fitness
