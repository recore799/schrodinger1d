import numpy as np
import matplotlib.pyplot as plt
import sys
import os

# Add src directory to path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from numerov import harmonic_oscillator

def harmonic_oscillator_wf(state,mesh,xmax):
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


# Parameters
states_to_plot = [0, 1, 2, 3, 4, 5]  # Quantum numbers (n)
mesh = 500
xmax = 10.0

plt.figure(figsize=(10, 7))

# Plot the harmonic potential (V(x) = 0.5*x²)
x = np.linspace(-xmax, xmax, 1000)
vpot = 0.5 * x**2
plt.plot(x, vpot, 'k-', linewidth=2, label="V(x) = ½x²")

# Plot each state's wavefunction offset by its energy
for n in states_to_plot:
    # Get wavefunction and energy
    x_full, psi, e = harmonic_oscillator_wf(state=n, mesh=mesh, xmax=xmax)
    
    # Offset ψ(x) by its energy Eₙ
    psi_offset = psi + e
    
    # Plot (with energy in legend)
    plt.plot(x_full, psi_offset, linewidth=1.5, label=f"n = {n}, E = {e:.3f}")

# Add energy levels (horizontal lines)
for n in states_to_plot:
    e = n + 0.5  # Theoretical energy (for comparison)
    plt.axhline(e, color='gray', linestyle=':', alpha=0.5)

# Styling
plt.title("Harmonic Oscillator Wavefunctions (Offset by Energy)", fontsize=14)
plt.xlabel("Position (x)", fontsize=12)
plt.ylabel("Energy (Eₙ) + ψₙ(x)", fontsize=12)
plt.ylim(-0.5, max(states_to_plot) + 1.5)  # Adjust y-axis to fit all states
plt.legend(fontsize=10)
plt.grid(alpha=0.3)
plt.tight_layout()

plt.savefig(harmonic.png, dpi=300, bbox_inches='tight')
print(f"Plot saved to {save_path}")


plt.show()
