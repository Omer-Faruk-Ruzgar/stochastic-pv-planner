import numpy as np
import pygad

from .crossover import pv_mixed_crossover
from .fitness import fitness_func

# Global configuration


# We have 5 zones -> chromosome = [5 binary | 5 continuous]
N_ZONES = 5
NUM_GENES = 2 * N_ZONES

# >>> IMPORTANT <<<
# Fill this based on your simulation spec (max PV capacity per zone, in kW).
# For now I put dummy values – you MUST replace them with your own numbers.
MAX_CAPACITY_KW = np.array([50, 80, 150, 100, 60], dtype=float)

# Base mutation settings (WTGA: adaptive mutation + diversity)
BASE_MUTATION_PERCENT = 10        # % of genes mutated when diversity is ok
HIGH_MUTATION_PERCENT = 30        # % of genes mutated when diversity is low
DIVERSITY_THRESHOLD = 1e-3        # std of fitness; ***tune this by experiments****

# -------------- Gene space / types ----------------

def build_gene_space_and_types():
    """Define allowed values for each gene (PyGAD gene_space / gene_type)."""

    gene_space = []
    gene_type = []

    # First N_ZONES genes: binary applyPV
    for _ in range(N_ZONES):
        gene_space.append([0, 1])   # discrete 0/1
        gene_type.append(int)

    # Last N_ZONES genes: continuous capacities in [0, max_cap]
    for i in range(N_ZONES):
        gene_space.append({
            "low": 0.0,
            "high": float(MAX_CAPACITY_KW[i])
        })
        gene_type.append(float)

    return gene_space, gene_type

# -------------Adaptive mutation callback (WTGA flavour)----------------

def on_generation(ga_instance: pygad.GA):
    """
    Simple diversity-driven mutation tuning:
    - If fitness std is low -> increase mutation to inject diversity.
    - Else -> keep base mutation.

    This is a lightweight implementation of the 'adaptive mutation' idea **** CHANGE FOR VARIENCE ******
    """

    fitness_values = ga_instance.last_generation_fitness

    # Safety: if not enough data, do nothing
    if fitness_values is None or len(fitness_values) < 2:
        return

    diversity = np.std(fitness_values)

    if diversity < DIVERSITY_THRESHOLD:
        # population is too similar -> increase mutation
        ga_instance.mutation_percent_genes = HIGH_MUTATION_PERCENT
    else:
        # diversity is fine -> use base mutation rate
        ga_instance.mutation_percent_genes = BASE_MUTATION_PERCENT

    # Optional debug print ********* COMMENT IF YOU WANT ********
    print(
        f"Gen {ga_instance.generations_completed}: "
        f"best_f={np.max(fitness_values):.3f}, "
        f"std={diversity:.3e}, "
        f"mut%={ga_instance.mutation_percent_genes}"
    )

# --------------- GA factory (WTGA configuration) -----------------

def build_wtga():
    gene_space, gene_type = build_gene_space_and_types()

    ga = pygad.GA(
        # Core GA structure
        num_generations=150,          # tune (30–200), *** COMPARE LATER ***
        sol_per_pop=40,               # population size -> 30–50
        num_parents_mating=20,

        num_genes=NUM_GENES,
        gene_space=gene_space,
        gene_type=gene_type,

        # 
        # Fitness and custom crossover
        fitness_func=fitness_func,
        crossover_type=pv_mixed_crossover,

        # WTGA-style mutation:
        mutation_type="random",
        mutation_percent_genes=BASE_MUTATION_PERCENT,
        random_mutation_min_val=-1.0,
        random_mutation_max_val=1.0,

        # Selection & elitism
        parent_selection_type="sss",      # steady state selection
        keep_parents=4,                   # retain some parents directly
        keep_elitism=2,                   # top-2 elite kept each generation

        # Stop if saturated for a while
        stop_criteria=["saturate_30"],

        # Callback for adaptive mutation/diversity
        on_generation=on_generation,
    )

    return ga

# Optional: a simpler baseline GA (EGA) without tuning, for comparison
def build_baseline_ga():
    gene_space, gene_type = build_gene_space_and_types()

    ga = pygad.GA(
        num_generations=150,
        sol_per_pop=40,
        num_parents_mating=20,

        num_genes=NUM_GENES,
        gene_space=gene_space,
        gene_type=gene_type,

        fitness_func=fitness_func,
        crossover_type="single_point",   # standard PyGAD crossover
        mutation_type="random",
        mutation_percent_genes=10,

        parent_selection_type="rws",     # roulette wheel
        keep_parents=0,
        keep_elitism=0,
    )

    return ga

# ---------------------------
# Entry point
# ---------------------------

def run_ga(use_wtga: bool = True, plot: bool = True):
    """
    Run GA optimization.

    :param use_wtga: if True, use tuned configuration with custom crossover.
                     if False, use a simple baseline GA for comparison.
    """

    if use_wtga:
        print("Running Well-Tuned GA (WTGA) with custom crossover...")
        ga = build_wtga()
    else:
        print("Running baseline GA (EGA) with default operators...")
        ga = build_baseline_ga()

    ga.run()

    # Retrieve best solution
    solution, solution_fitness, solution_idx = ga.best_solution()
    print("\n=== BEST SOLUTION ===")
    print("Chromosome:", solution)
    print("Best fitness:", solution_fitness)

    best_apply_pv = solution[:N_ZONES].astype(int)
    best_capacity = solution[N_ZONES:]

    print("applyPV:", best_apply_pv)
    print("capacity:", best_capacity)

    if plot:
        try:
            ga.plot_fitness()
        except Exception as e:
            print("Could not plot fitness curve:", e)

if __name__ == "__main__":
    # By default run WTGA; switch to False for baseline.
    run_ga(use_wtga=True, plot=True)
