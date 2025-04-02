#!/usr/bin/env python3
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src')))

from numerov import do_mesh, solve_sheq

import numpy as np
import matplotlib.pyplot as plt

def vpot(zeta,r):
    return -2 * zeta / r

def main():
    zeta = 1
    zmesh = 1
    rmax = 100
    xmin = -8.0
    dx = 0.01

    # Calculate mesh as integer to avoid float indices
    mesh = int((np.log(zmesh * rmax) - xmin) / dx)
    # Ensure mesh is at least 0 to prevent negative array sizes
    mesh = max(mesh, 0)

    # Initialize arrays with mesh+1 elements
    r = np.zeros(mesh + 1, dtype=float)
    sqr = np.zeros(mesh + 1, dtype=float)
    r2 = np.zeros(mesh + 1, dtype=float)
    y = np.zeros(mesh + 1, dtype=float)  # Wavefunction array

    # Generate the logarithmic mesh
    do_mesh(mesh, zmesh, xmin, dx, rmax, r, sqr, r2)


    # Compute the potential
    vpot_arr = vpot(zeta, r)

    # Solve the Schrödinger equation for n=1, l=0
    n = 1
    l = 0

    e = solve_sheq(n, l, zeta, mesh, dx, r, sqr, r2, vpot_arr, y)

    print(f"Energy eigenvalue: {e:.6f}")




if __name__ == "__main__":
    main()
