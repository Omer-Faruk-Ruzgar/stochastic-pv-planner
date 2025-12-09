import numpy as np

def pv_mixed_crossover(parents, offspring_size, ga_instance):
    """
    Custom crossover for PV placement problem

    Chromosome structure:
        [applyPV[0...n_zones-1], capacity[0...n_zones-1]]

    - applyPV: binary value that representing applying PV to a zone
    - capacity: float (kW), meaningful only if applyPV == 1

    Strategy:
        - Split zones into two halves
        - First half: primary parent = parent 1, secondary = parent 2
        - Second half: primary parent = parent 2, secondary = parent 1
        - Combine using the logic described by the user
    """

    # num_offspring: how mant children to produce
    # num_genes: length of each chromosome
    num_offspring, num_genes = offspring_size
    offspring = np.empty((num_offspring, num_genes), stype=float)

    # Infer nnumber of zones
    n_zones = num_genes // 2
    mid = n_zones // 2 # split for first/second half

    for k in range(num_offspring):
        # Select two parents
        p1 = parents[k % parents.shape[0]]
        p2 = parents[(k + 1) % parents.shape[0]]

        p1_bin = p1[:n_zones].astype(int)
        p1_cap = p1[n_zones:]
        p2_bin = p2[:n_zones].astype(int)
        p2_cap = p2[n_zones:]

        child_bin = np.zeros(n_zones, dtype = int)
        child_cap = np.zeros(n_zones, dtype = float)

        for i in range(n_zones):
            # who is primary / secondary for this zone
            if i < mid:
                primary_bin, primary_cap = p1_bin[i], p1_cap[i]
                secondary_bin, decondary_cap = p2_bin[i], p2_cap[i]
            else:
                primary_bin, primary_cap = p2_bin[i], p2_cap[i]
                secondary_bin, secondary_cap = p1_bin[i], p1_cap[i]

            # Case A - both 1: keep PV, avg capacity
            if primary_bin == 1 and secondary_bin == 1:
                child_bin[i] = 1
                child_cap[i] = 0.5 * (primary_cap + secondary_cap)

            # Case B - primary 1, secondary 0: keep primary config
            elif primary_bin == 1 and secondary_bin == 0:
                child_bin[i] = 1
                child_cap[i] = primary_cap
            
            # Case C - primary 0, secondary 1: half probability for keeping secondary capacity
            elif primary_bin == 0 and secondary_bin == 1:
                if np.random.rand() < 0.5:
                    child_bin[i] = 1
                    child_cap[i] = secondary_cap
                else:
                    child_bin[i] = 0
                    child_cap[i] = 0.0

            # Case D - both 0: 0.0 kw (no PV)
            if primary_bin == 0 and secondary_bin == 0:
                child_bin[i] = 0
                child_cap[i] = 0.0

        # Merge back into single chromosome
        offspring[k, :n_zones] = child_bin
        offspring[k, n_zones:] = child_cap

    return offspring


