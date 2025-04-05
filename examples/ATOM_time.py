import sys
import os
import timeit
import numpy as np

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from numerov import (
    hydrogen_atom,  # Current implementation
)


def test_hydrogen_levels():
    print("\nHydrogen Atom Energy Levels (n=1 to 10, l=0):")
    print("--------------------------------------------")
    print(" n | Computed E (a.u.) | Expected E (a.u.) | Error")
    print("---|-------------------|-------------------|-------")

    Z = 1
    l = 0
    xmin = -8.0
    xmax = np.log(100.0)
    mesh = 1260

    for n in range(1, 7):
        start_time = timeit.default_timer()
        e_computed = hydrogen_atom(n, l, Z, xmin, xmax, mesh)
        e_expected = -Z**2 / (2*n**2)
        error = abs(e_computed - e_expected)
        elapsed = timeit.default_timer() - start_time

        print(f"{n:2} | {e_computed:.8f}        | {e_expected:.8f}        | {error:.2e} | {elapsed:.4f} sec")

    print("\n------ Performance Summary ------")
    print(f"Mesh size: {mesh}, r_max: {np.exp(xmax):.1f} a.u.")

# Run the test
test_hydrogen_levels()
