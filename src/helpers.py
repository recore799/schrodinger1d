import numpy as np
import matplotlib.pyplot as plt

import sys
import os

# Add src directory to path (not needed when files are in the same dir)
# sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from numerov import harmonic_oscillator

def harmonic_oscillator_wf(state):
    mesh = 500
    xmax = 10.0

    e, iterations, psi = harmonic_oscillator(nodes=state, mesh=mesh, xmax=xmax)

    # Init full wf
    x_full = np.linspace(-xmax, xmax, 2*mesh + 1)
    psi_full = np.zeros(2*mesh + 1)

    # Fill the positive half (x >= 0)
    psi_full[mesh:] = psi

    # Fill the negative half (x < 0) using symmetry
    if state % 2 == 0:
        # Even state: ψ(-x) = ψ(x)
        psi_full[:mesh] = np.flip(psi[1:])
    else:
        # Odd state: ψ(-x) = -ψ(x)
        psi_full[:mesh] = -np.flip(psi[1:])

    return x_full, psi_full, e
