import sys
import os
import timeit
import numpy as np

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from newimp import (
    hydrogen_atom,  # Current implementation
    setup_hydrogen_problem
)

import time
import numpy as np

def test_hydrogen_levels():
    # Test configuration
    states = 10
    l_values = [0]  # Test s, p, d orbitals
    mesh = 1421
    xmax = np.log(500)

    # Results table header
    print("\n{:<5} {:<5} {:<15} {:<10} {:<10} {:<10} {:<10}".format(
        "n", "l", "Energy (Ha)", "Error", "Time (s)", "Iters", "Degeneracy"))
    print("-"*80)

    for n in range(2, states+1):
        for l in l_values:
            if l >= n:  # Skip invalid l values (l < n)
                continue

            # Setup and solve
            start_time = time.perf_counter()
            params = setup_hydrogen_problem(n=n, l=l, xmax=xmax, mesh=mesh)
            energy, iterations = hydrogen_atom(params)
            elapsed = time.perf_counter() - start_time

            # Analytical solution
            e_anal = -params['Z']**2 / (n**2)
            error = abs(energy - e_anal)

            # Print results (highlight degeneracy when l > 0)
            deg_flag = "✓" if l > 0 else ""
            print("{:<5} {:<5} {:<15.6f} {:<10.2e} {:<10.4f} {:<10} {:<10}".format(
                n, l, energy, error, elapsed, iterations, deg_flag))

# Run the test
test_hydrogen_levels()
