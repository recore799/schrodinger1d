#!/usr/bin/env python3
import sys
import os
import pickle

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from numerov_log import do_mesh, solve_sheq

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
    r = np.zeros(mesh + 1, dtype=float)    # Radial grid
    sqr = np.zeros(mesh + 1, dtype=float)  # Square root of r
    r2 = np.zeros(mesh + 1, dtype=float)   # r^2
    y = np.zeros(mesh + 1, dtype=float)    # Wavefunction array

    # Generate the logarithmic mesh
    do_mesh(mesh, zmesh, xmin, dx, rmax, r, sqr, r2)

    # Compute the potential on the mesh
    vpot_arr = vpot(zeta, r)

    # Solve the Schrödinger equation for different n and l
    n_max = 6  # Calculate states for n = 1,2,...,6
    l_max = 5  # Maximum angular momentum
    energies = np.zeros((n_max, l_max + 1))
    logs = {}  # To store log data for each (n,l) state

    for n in range(1, n_max + 1):  # n starts at 1
        for l in range(n):  # l < n
            # Reset the wavefunction array for each (n,l)
            y = np.zeros(mesh + 1, dtype=float)
            # Enable logging (set second parameter True) so that we record iteration details.
            energy, log_data = solve_sheq(n, l, zeta, mesh, dx, r, sqr, r2, vpot_arr, y, log_enabled=True)
            energies[n - 1, l] = energy
            logs[f"n{n}_l{l}"] = log_data

    # Print energies in a readable format
    print("Energy eigenvalues (n, l, E):")
    for n in range(1, n_max + 1):
        for l in range(n):
            print(f"n={n}, l={l}: E = {energies[n - 1, l]:.6f}")

    # (Optional) Save the log data to a file for later analysis/animation.
    with open("numerov_log.pkl", "wb") as f:
        pickle.dump(logs, f)
    print("Log data saved to numerov_log.pkl")

if __name__ == "__main__":
    main()
