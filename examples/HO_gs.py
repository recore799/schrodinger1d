#!/usr/bin/env python3
import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from schrodinger1d import SchrodingerSolver
import matplotlib.pyplot as plt

def harmonic_oscillator(x):
    return x**2


def main():
    # Initialize the integrator
    integrator = SchrodingerSolver(V=harmonic_oscillator, xL=-5, xR=5, n=501, tol=1e-7)

    # Find the ground state (0 nodes)
    E0, psi0 = integrator.solve_state(energy_level=0, E_min=0, E_max=10)
    nodes0 = integrator.count_nodes(psi0)

    print(f"Ground state energy: {E0:.6f}")
    print(f"Ground state node count: {nodes0}")

    # Plot the ground state wavefunction
    plt.figure(figsize=(8, 4))
    plt.plot(integrator.x, psi0, label='Ground State (0 nodes)')
    plt.title('Ground State Wavefunction')
    plt.xlabel('x')
    plt.ylabel(r'$\psi(x)$')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    main()
