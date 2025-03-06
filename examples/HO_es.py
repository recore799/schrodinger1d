#!/usr/bin/env python3

import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from schrodinger1d import SchrodingerSolver
import matplotlib.pyplot as plt
import numpy as np

def harmonic_oscillator(x):
    return x**2

def main():

    integrator = SchrodingerSolver(V=harmonic_oscillator)

    E = np.zeros(5)
    psi = np.zeros((5, len(integrator.x)))


    for i in range(5):
        E[i], psi[i] = integrator.solve_state(energy_level=i, E_min=0, E_max=10)


    # Example: Print results

    print("Energies:", E)
    print("Wavefunctions shape:", psi.shape)





if __name__ == "__main__":
    main()
