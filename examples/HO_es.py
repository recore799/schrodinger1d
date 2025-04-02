#!/usr/bin/env python3

import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from numerov import numerov0
import numpy as np

def harmonic_oscillator(x):
    return x**2

def main():

    integrator = numerov0(V=harmonic_oscillator)

    E = np.zeros(10)
    # psi = np.zeros((5, len(integrator.x)))


    for i in range(10):
        E[i] = integrator.solve_state(energy_level=i, E_min=0, E_max=10)
        print(f"Energy for state {i}: {E[i]:.8f}")






if __name__ == "__main__":
    main()
