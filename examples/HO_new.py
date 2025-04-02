import sys
import os

# Add the src directory to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

from numerov import numerov0, harmonic_oscillator
import numpy as np


energies = np.zeros(10)

print("-------------------- First Implementation ----------")
for i in range(10):
    energies[i] = harmonic_oscillator(nodes=i)

    print(f"Energy for state {i}: {energies[i]:.8f}")

# print("\n")
# print("-------------------- Second Implementation ----------")
# def vho(x):
#     return x**2

# integrator = numerov0(V=vho)

# E = np.zeros(10)


# for i in range(10):
#     E[i] = integrator.solve_state(energy_level=i, E_min=0, E_max=10)
#     print(f"Energy for state {i}: {E[i]:.8f}")
