#!/usr/bin/env python3
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from numerov_debug import do_mesh, solve_sheq

import numpy as np

def vpot(zeta, r):
    """Compute the Coulomb potential."""
    return -2 * zeta / r

def main():
    zeta = 1  # Nuclear charge
    zmesh = 1
    rmax = 100  # Maximum radial distance
    xmin = -8.0  # Logarithmic grid parameter
    dx = 0.01  # Grid spacing

    # Calculate mesh size
    mesh = int((np.log(zmesh * rmax) - xmin) / dx)
    mesh = max(mesh, 0)  # Ensure mesh is non-negative

    # Initialize arrays
    r = np.zeros(mesh + 1, dtype=float)  # Radial grid
    sqr = np.zeros(mesh + 1, dtype=float)  # Square root of r
    r2 = np.zeros(mesh + 1, dtype=float)  # r^2
    y = np.zeros(mesh + 1, dtype=float)  # Wavefunction array

    # Generate the logarithmic mesh
    do_mesh(mesh, zmesh, xmin, dx, r, sqr, r2)

    # Compute the potential
    vpot_arr = vpot(zeta, r)

    # Solve the Schrödinger equation for different n and l
    n_max = 6  # Calculate up to n=5 (4 excited states)
    l_max = 5  # Maximum angular momentum
    energies = np.zeros((n_max, l_max + 1))  # Store energies for each (n, l)

    for n in range(1, n_max + 1):  # n starts from 1
        for l in range(n):  # l < n
            y = np.zeros(mesh + 1, dtype=float)  # Fresh wavefunction array for each (n, l)
            energy = solve_sheq(n, l, zeta, mesh, dx, r, sqr, r2, vpot_arr, y)
            energies[n - 1, l] = energy  # Store energy

    # Print energies in a readable format
    print("Energy eigenvalues (n, l, E):")
    for n in range(1, n_max + 1):
        for l in range(n):
            print(f"n={n}, l={l}: E = {energies[n - 1, l]:.6f}")

    # Print energies grouped by n
    print("\nEnergies grouped by n:")
    for n in range(1, n_max + 1):
        print(f"n={n}: {energies[n - 1, :n]}")

if __name__ == "__main__":
    main()
