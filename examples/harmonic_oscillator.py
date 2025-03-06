#!/usr/bin/env python3
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from schrodinger1d import SchrodingerSolver

import numpy as np
import matplotlib.pyplot as plt

def harmonic_oscillator(x):
    return x**2

def main():
    # Configure style
    plt.style.use("seaborn-v0_8-darkgrid")
    plt.rcParams.update({"font.size": 12, "figure.autolayout": True})

    integrator = SchrodingerSolver(V=harmonic_oscillator)

    energies = np.zeros(5)
    psi = np.zeros((5, len(integrator.x)))


    for i in range(5):
        energies[i], psi[i] = integrator.solve_state(energy_level=i, E_min=0, E_max=10)

    # Initialize figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Color palette for wavefunctions
    colors = plt.cm.viridis(np.linspace(0, 1, len(energies)))

    # Scale factor for wavefunctions
    scale_factor = 0.8  # Adjust this value to control the size of the wavefunctions

    # Plot eigenfunctions
    for i, E in enumerate(energies):
        ax.plot(integrator.x, scale_factor * (psi[i]) + E,  # Scale and offset vertically
                lw=2.5, color=colors[i],
                label=fr"$n = {i}$; $E_{i} = {E:.1f}$")

        # Fill under curve with transparency
        ax.fill_between(integrator.x, E, scale_factor * (psi[i]) + E,
                        color=colors[i], alpha=0.15)

    # Formatting
    ax.set_xlabel("Position ($x$)", fontsize=14)
    ax.set_ylabel("Energy → Wavefunction Amplitude", fontsize=14)
    ax.set_title("Harmonic Oscillator Eigenfunctions", fontsize=16, pad=20)

    # Add energy levels
    for E in energies:
        ax.axhline(E, color='gray', linestyle='--', lw=1, alpha=0.7)

    ax.legend(loc='upper right', frameon=True, fontsize=12)
    ax.spines[['top', 'right']].set_visible(False)

    # Save figure for README
    plt.savefig("examples/harmonic_oscillator_eigenfunctions.png",
                dpi=300, bbox_inches="tight")
    plt.close()

if __name__ == "__main__":
    main()
