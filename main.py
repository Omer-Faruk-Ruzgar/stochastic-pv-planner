import numpy as np
from optimizer.ga_runner import run_ga

def main():
    np.random.seed(46)

    print("\n--- Baseline ---")
    run_ga(use_wtga = False, plot = True)

    print("\n--- WTGA ---")
    run_ga(use_wtga = True, plot = True)

if __name__ == "__main__":
    main()