import numpy as np

# Must be consistent with fitness/ga_runner
N_ZONES = 5

# --- Basic PV/energy parameters ---

# Capacity factor base per zone (how good each zone is on average)
# e.g. 0.18 means 18% of installed kW turns into kWh in the period that modeled
BASE_CAPACITY_FACTOR = np.array([0.18, 0.16, 0.20, 0.17, 0.19], dtype=float)

# Std dev of capacity factor noise per scenario
CF_SIGMA = 0.04

# Base demand per zone (kW or kWh for our chosen time window)
BASE_DEMAND = np.array([25, 40, 30, 50, 35], dtype=float)

# Demand variability per scenario
DEMAND_SIGMA = 5.0


# --------------------- 1. PV generation model --------------------------

def simulate_pv_generation(capacity_kw: np.ndarray, weather: dict) -> np.ndarray:
    """
    Simple PV generation model.

    Parameters
    ----------
    capacity_kw : array, shape (N_ZONES,)
        Installed PV capacity per zone (kW).
    weather : dict
        Must contain key "cf_factor": array of length N_ZONES with
        capacity factor multipliers for this scenario.

    Returns
    -------
    generation : array, shape (N_ZONES,)
        Energy produced in the modeled period (same unit as demand, e.g. kWh).
    """
    capacity_kw = np.asarray(capacity_kw, dtype=float)

    # Scenario-specific capacity factor
    cf_factor = np.asarray(weather["cf_factor"], dtype=float)

    # Effective capacity factor per zone
    cf = BASE_CAPACITY_FACTOR * cf_factor

    # Clip to avoid crazy values
    cf = np.clip(cf, 0.0, 0.35)

    # Energy = capacity * capacity_factor (for whatever time horizon we model)
    generation = capacity_kw * cf
    return generation


# --------------- 2. Scenario generation (demand + weather) -------------------

def build_demand_weather_scenarios(num_scenarios: int, rng: np.random.Generator | None = None):
    """
    Build a list of demand+weather scenarios.

    Each scenario is a dict:
        {
            "demand":  np.array shape (N_ZONES,),
            "weather": {"cf_factor": np.array shape (N_ZONES,)}
        }

    We can later replace this with AR(1), real data, etc.
    """
    if rng is None:
        rng = np.random.default_rng()

    scenarios = []

    for _ in range(num_scenarios):
        # Demand with Gaussian noise
        demand = BASE_DEMAND + rng.normal(loc=0.0, scale=DEMAND_SIGMA, size=N_ZONES)
        demand = np.clip(demand, 0.0, None)

        # Capacity factor multipliers, around 1.0 (sunny vs cloudy)
        cf_factor = 1.0 + rng.normal(loc=0.0, scale=CF_SIGMA, size=N_ZONES)
        cf_factor = np.clip(cf_factor, 0.5, 1.5)

        scen = {
            "demand": demand,
            "weather": {"cf_factor": cf_factor}
        }
        scenarios.append(scen)

    return scenarios
