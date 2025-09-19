import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

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

def plot_wavefunction(x_full, psi_full, state, e):
    """
    Plot the wavefunction with proper labels and styling.
    
    Parameters:
        x_full (np.ndarray): Full x-grid [-xmax, xmax]
        psi_full (np.ndarray): Wavefunction values
        state (int): Quantum number n
        e (float): Energy eigenvalue
    """
    plt.figure(figsize=(10, 5))
    plt.plot(x_full, psi_full, 
             linewidth=2, 
             label=f"n = {state}, E = {e:.4f}")
    
    # Styling
    plt.title(f"Harmonic Oscillator Wavefunction (n = {state})", fontsize=14)
    plt.xlabel("Position (x)", fontsize=12)
    plt.ylabel("ψ(x)", fontsize=12)
    plt.axhline(0, color='black', linewidth=0.5, linestyle='--')
    plt.axvline(0, color='black', linewidth=0.5, linestyle='--')
    plt.grid(alpha=0.3)
    plt.legend(fontsize=10, framealpha=1)
    plt.tight_layout()
    plt.show()

# Generate and plot n=2 state
x, psi, energy = harmonic_oscillator_wf(state=1)
plot_wavefunction(x, psi, state=2, e=energy)
