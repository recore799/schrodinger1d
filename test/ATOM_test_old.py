import sys
import os
import timeit
import numpy as np

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from numerov_debug import (
    # hydrogen_atom,  # Current implementation
    solve_sheq
)

def test_hydrogen_levels():
    Z = 1
    l = 0
    xmin = -8.0
    xmax = np.log(500.0)
    # mesh = 1260
    mesh = 1421
    # Logarithmic mesh in x
    x, dx = np.linspace(xmin, xmax, mesh+1, retstep=True)

    # Physical mesh in r
    r = np.exp(x) / Z

    # Precompute common terms
    r2 = r**2
    sqr = np.sqrt(r)

    vpot = - 2*Z/r

    y = np.zeros(mesh+1)

    states = 10
    time_sum = 0
    errors = []

    print("\n{:<5} {:<15} {:<15} {:<10} {:<10} {:<10}".format(
        "n", "Numerov Energy", "Analytic Energy", "Error", "Time (s)", "Iters"))
    print("-" * 80)

    for n in range(states):
        n_quantum = n + 1  # Principal quantum number
        start_time = timeit.default_timer()
        e_numerov, iterations = solve_sheq(n_quantum, l, Z, mesh, dx, r, sqr, r2, vpot, y)
        elapsed = timeit.default_timer() - start_time
        e_anal = -Z**2 / (2 * n_quantum**2)
        error = abs(e_numerov - e_anal)
        errors.append(error)

        print("{:<5} {:<15.6f} {:<15.6f} {:<10.2e} {:<10.4f} {:<10}".format(
            n_quantum, e_numerov, e_anal, error, elapsed, iterations))
        time_sum += elapsed

    avg_time = time_sum / states
    max_error = max(errors)
    avg_error = sum(errors) / states

    print("\nSummary:")
    print(f"Average time per state: {avg_time:.4f} sec")
    print(f"Maximum error: {max_error:.2e}")
    print(f"Average error: {avg_error:.2e}")

# Run the test
test_hydrogen_levels()
