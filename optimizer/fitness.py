import numpy as np
from .power_model import (simulate_pv_generation, build_demand_weather_scenarios, N_ZONES)

CAPEX_WEEKS = 20 * 52   # 20 year lifetime of the model

# Cost per kW installed at each zone (Change it for testing)
CAPEX_PER_KW = np.array([800, 850, 820, 900, 870], dtype=float)  # € per kW

# Import and export prices (per kWh)
IMPORT_PRICE = 0.25   # €/kWh
EXPORT_PRICE = 0.10   # €/kWh

MIN_CAPACITY_KW = 10.0  # Minimum PV size (kW) when apply_pv[i] == 1

# Pick a budget for force GA to decide where PV is most valuable (Here 200 kW)
TOTAL_PV_BUDGET_KW = 200.0
BUDGET_PENALTY = 1e5        # penalty € per kW over budget (*tune*)

HOURS = 168     # hours in a week 

# Regularization weight for number of zones used
LAMBDA_ZONES = 1   # €/zone, **Tune**

# Example: demand & weather scenarios
# You should build these from data / AR(1) model, etc.
# For now, think of them as a list of dicts:
#  scenario["demand"] : np.array shape (N_ZONES,)
#  scenario["weather"] : anything needed by simulate_pv_generation

DEMAND_WEATHER_SCENARIOS = build_demand_weather_scenarios(num_scenarios = 50)

def fitness_func(ga_instance, solution, solution_idx):
    """
    Fitness = - (capex + expected_energy_cost + penalties)

    solution structure:
        [applyPV[0..N_ZONES-1], capacity[0..N_ZONES-1]]
    """

    # Split chromosome
    apply_pv = np.array(solution[:N_ZONES], dtype=int)
    capacity = np.array(solution[N_ZONES:], dtype=float)

    # No capacity if no PV
    capacity = np.where(apply_pv == 1, capacity, 0.0)
    capacity = np.where(apply_pv == 1, np.maximum(capacity, MIN_CAPACITY_KW), 0.0)

    # How much the solution is trying to install
    total_capacity = np.sum(capacity)       

    # Much much the solution violates the global PV budget
    over_budget = max(0.0, total_capacity - TOTAL_PV_BUDGET_KW)

    penalty_budget = BUDGET_PENALTY * over_budget

    # ---- 1. CAPEX ----
    capex = np.sum(apply_pv * capacity * CAPEX_PER_KW)
    capex_weekly = capex / CAPEX_WEEKS

    # ---- 2. Expected energy cost over scenarios ----
    if len(DEMAND_WEATHER_SCENARIOS) == 0:
        # Just penalize capex - no scenarios
        expected_energy_cost = 0.0
    else:
        total_energy_cost = 0.0

        for scen in DEMAND_WEATHER_SCENARIOS:
            demand = scen["demand"]            # shape (N_ZONES,)
            weather = scen["weather"]          # whatever your power model expects

            # Compute PV generation for this scenario
            # Implement this in power_model.py
            generation = simulate_pv_generation(capacity, weather)  # shape (N_ZONES,)

            # Net demand by positive: need import, negative: surplus
            net = np.sum(demand - generation)

            # Energy optimization
            energy_import = max(net, 0.0)
            energy_surplus = max(-net, 0.0)

            scenario_cost = (IMPORT_PRICE * energy_import - EXPORT_PRICE * energy_surplus) * HOURS
            total_energy_cost += scenario_cost

        expected_energy_cost = total_energy_cost / len(DEMAND_WEATHER_SCENARIOS)

    # ---- 3. Penalty: number of zones with PV ----
    num_zones_used = np.sum(apply_pv)
    penalty_zones = LAMBDA_ZONES * num_zones_used

    # ---- 4. Total cost and fitness ----
    total_cost = capex_weekly + expected_energy_cost + penalty_zones + penalty_budget

    fitness = -total_cost   # GA maximizes fitness
    return fitness
